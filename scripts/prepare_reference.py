#!/usr/bin/env python3
"""
Prepare reference audio and transcript for dots-tts voice cloning.

Supports:
- Local WAV/MP3/M4A files
- YouTube URLs with optional start/end time trimming
- Background noise removal via demucs
- Automatic transcription via faster-whisper

Usage:
    # From local file
    python scripts/prepare_reference.py --input voice.wav --output-dir ./references

    # From YouTube with time range
    python scripts/prepare_reference.py --input "https://youtube.com/watch?v=xxx" \
        --start 00:30 --end 00:45 --output-dir ./references

    # Specify language for transcription
    python scripts/prepare_reference.py --input voice.wav --language en --output-dir ./references
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def find_yt_dlp():
    """Find yt-dlp binary, preferring Homebrew installation."""
    brew_path = "/opt/homebrew/bin/yt-dlp"
    if os.path.exists(brew_path):
        return brew_path
    if shutil.which("yt-dlp"):
        return "yt-dlp"
    return None


def parse_timestamp(ts: str) -> float:
    """Parse timestamp string to seconds. Supports HH:MM:SS, MM:SS, or seconds."""
    if ts is None:
        return None

    ts = ts.strip()
    if not ts:
        return None

    # Try pure seconds
    try:
        return float(ts)
    except ValueError:
        pass

    # Try MM:SS or HH:MM:SS
    parts = ts.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])

    raise ValueError(f"Invalid timestamp format: {ts}")


def is_youtube_url(url: str) -> bool:
    """Check if the input is a YouTube URL."""
    youtube_patterns = [
        r"(https?://)?(www\.)?youtube\.com/watch",
        r"(https?://)?(www\.)?youtu\.be/",
        r"(https?://)?(www\.)?youtube\.com/shorts/",
    ]
    return any(re.match(p, url) for p in youtube_patterns)


def download_youtube_audio(url: str, output_path: str, start: float = None, end: float = None):
    """Download audio from YouTube, optionally trimming to start/end times."""
    yt_dlp = find_yt_dlp()
    if not yt_dlp:
        raise RuntimeError(
            "yt-dlp not found. Install with: brew install yt-dlp (or pip install yt-dlp)"
        )

    print(f"Downloading audio from YouTube: {url}")

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_output = os.path.join(tmpdir, "download.%(ext)s")

        cmd = [
            yt_dlp,
            "-x",
            "--audio-format", "wav",
            "-o", temp_output,
            url,
        ]

        # Add time range if specified (yt-dlp supports this via ffmpeg postprocessor)
        if start is not None or end is not None:
            # yt-dlp uses --download-sections for time ranges
            section = "*"
            if start is not None:
                section += f"{start}-"
            else:
                section += "0-"
            if end is not None:
                section += f"{end}"
            else:
                section += "inf"
            cmd.extend(["--download-sections", section])

        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        # Find the downloaded file
        downloaded = list(Path(tmpdir).glob("download.*"))
        if not downloaded:
            raise FileNotFoundError("Failed to download audio from YouTube")

        # Convert to WAV if needed and copy to output
        src = downloaded[0]
        if src.suffix.lower() != ".wav":
            subprocess.run([
                "ffmpeg", "-i", str(src),
                "-ar", "24000", "-ac", "1",
                output_path, "-y"
            ], check=True, capture_output=True)
        else:
            # Normalize to 24kHz mono
            subprocess.run([
                "ffmpeg", "-i", str(src),
                "-ar", "24000", "-ac", "1",
                output_path, "-y"
            ], check=True, capture_output=True)

    print(f"Downloaded and saved to: {output_path}")
    return output_path


def trim_audio(input_path: str, output_path: str, start: float = None, end: float = None):
    """Trim audio file to specified time range using ffmpeg."""
    cmd = ["ffmpeg", "-i", input_path]

    if start is not None:
        cmd.extend(["-ss", str(start)])
    if end is not None:
        if start is not None:
            cmd.extend(["-t", str(end - start)])
        else:
            cmd.extend(["-t", str(end)])

    cmd.extend(["-ar", "24000", "-ac", "1", output_path, "-y"])

    print(f"Trimming audio: {start}s to {end}s")
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path


def remove_background_noise(input_path: str, output_path: str, method: str = "demucs"):
    """Remove background noise/music from audio, keeping only vocals."""

    if method == "demucs":
        return remove_noise_demucs(input_path, output_path)
    elif method == "ffmpeg":
        return remove_noise_ffmpeg(input_path, output_path)
    else:
        raise ValueError(f"Unknown noise removal method: {method}")


def remove_noise_demucs(input_path: str, output_path: str):
    """Use demucs for vocal separation (best quality)."""
    try:
        import demucs.separate
    except ImportError:
        print("demucs not installed. Install with: pip install demucs")
        print("Falling back to ffmpeg noise reduction...")
        return remove_noise_ffmpeg(input_path, output_path)

    print("Separating vocals using demucs (this may take a minute)...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Run demucs separation
        cmd = [
            sys.executable, "-m", "demucs.separate",
            "-n", "htdemucs",  # Best model for vocals
            "--two-stems", "vocals",  # Only separate vocals
            "-o", tmpdir,
            input_path
        ]

        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        # Find the vocals output
        input_name = Path(input_path).stem
        vocals_path = Path(tmpdir) / "htdemucs" / input_name / "vocals.wav"

        if not vocals_path.exists():
            # Try alternative path structure
            for vp in Path(tmpdir).rglob("vocals.wav"):
                vocals_path = vp
                break

        if not vocals_path.exists():
            raise FileNotFoundError(f"Demucs did not produce vocals output. Check {tmpdir}")

        # Normalize to 24kHz mono
        subprocess.run([
            "ffmpeg", "-i", str(vocals_path),
            "-ar", "24000", "-ac", "1",
            output_path, "-y"
        ], check=True, capture_output=True)

    print(f"Vocals extracted to: {output_path}")
    return output_path


def remove_noise_ffmpeg(input_path: str, output_path: str):
    """Use ffmpeg filters for basic noise reduction (faster, lower quality)."""
    print("Applying ffmpeg noise reduction...")

    cmd = [
        "ffmpeg", "-i", input_path,
        "-af", "highpass=f=80,lowpass=f=8000,afftdn=nf=-20,agate=threshold=0.02:ratio=3",
        "-ar", "24000", "-ac", "1",
        output_path, "-y"
    ]

    subprocess.run(cmd, check=True, capture_output=True)
    print(f"Noise-reduced audio saved to: {output_path}")
    return output_path


def transcribe_audio(audio_path: str, language: str = None) -> str:
    """Transcribe audio using faster-whisper."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise ImportError(
            "faster-whisper not installed. Install with: pip install faster-whisper"
        )

    print("Transcribing audio with faster-whisper...")

    # Use base model for speed, or medium for better accuracy
    model = WhisperModel("base", device="cpu", compute_type="int8")

    segments, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=5,
        vad_filter=True,
    )

    # Combine all segments
    transcript = " ".join(segment.text.strip() for segment in segments)

    detected_lang = info.language
    print(f"Detected language: {detected_lang}")
    print(f"Transcript: {transcript}")

    return transcript, detected_lang


def get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        audio_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def main():
    parser = argparse.ArgumentParser(
        description="Prepare reference audio and transcript for dots-tts voice cloning."
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Input audio file (WAV/MP3/M4A) or YouTube URL"
    )
    parser.add_argument(
        "--output-dir", "-o", default="./references",
        help="Output directory for reference files (default: ./references)"
    )
    parser.add_argument(
        "--name", "-n", default="reference",
        help="Name prefix for output files (default: reference)"
    )
    parser.add_argument(
        "--start", "-s", default=None,
        help="Start time for trimming (e.g., '00:30', '1:23', '90')"
    )
    parser.add_argument(
        "--end", "-e", default=None,
        help="End time for trimming (e.g., '00:45', '1:45', '105')"
    )
    parser.add_argument(
        "--language", "-l", default=None,
        help="Language code for transcription (e.g., 'en', 'zh', 'hi'). Auto-detect if not specified."
    )
    parser.add_argument(
        "--noise-removal", choices=["demucs", "ffmpeg", "none"], default="demucs",
        help="Noise removal method: demucs (best), ffmpeg (fast), none (default: demucs)"
    )
    parser.add_argument(
        "--max-duration", type=float, default=10.0,
        help="Maximum reference duration in seconds (default: 10.0)"
    )

    args = parser.parse_args()

    # Parse timestamps
    start_sec = parse_timestamp(args.start)
    end_sec = parse_timestamp(args.end)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Get the raw audio
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_audio = os.path.join(tmpdir, "raw.wav")

        if is_youtube_url(args.input):
            download_youtube_audio(args.input, raw_audio, start_sec, end_sec)
        else:
            # Local file - trim if needed
            if not os.path.exists(args.input):
                raise FileNotFoundError(f"Input file not found: {args.input}")

            if start_sec is not None or end_sec is not None:
                trim_audio(args.input, raw_audio, start_sec, end_sec)
            else:
                # Just normalize to 24kHz mono
                subprocess.run([
                    "ffmpeg", "-i", args.input,
                    "-ar", "24000", "-ac", "1",
                    raw_audio, "-y"
                ], check=True, capture_output=True)

        # Check duration and trim if too long
        duration = get_audio_duration(raw_audio)
        if duration > args.max_duration:
            print(f"Audio is {duration:.1f}s, trimming to {args.max_duration}s...")
            trimmed = os.path.join(tmpdir, "trimmed.wav")
            trim_audio(raw_audio, trimmed, 0, args.max_duration)
            raw_audio = trimmed

        # Step 2: Remove background noise
        if args.noise_removal != "none":
            clean_audio = os.path.join(tmpdir, "clean.wav")
            remove_background_noise(raw_audio, clean_audio, method=args.noise_removal)
        else:
            clean_audio = raw_audio

        # Step 3: Transcribe
        transcript, detected_lang = transcribe_audio(clean_audio, args.language)

        # Step 4: Save outputs
        output_audio = output_dir / f"{args.name}.wav"
        output_text = output_dir / f"{args.name}.txt"

        shutil.copy(clean_audio, output_audio)
        output_text.write_text(transcript, encoding="utf-8")

        # Also save metadata
        output_meta = output_dir / f"{args.name}_meta.txt"
        output_meta.write_text(
            f"source: {args.input}\n"
            f"language: {detected_lang}\n"
            f"duration: {get_audio_duration(str(output_audio)):.2f}s\n"
            f"transcript: {transcript}\n",
            encoding="utf-8"
        )

    print("\n" + "=" * 60)
    print("Reference preparation complete!")
    print("=" * 60)
    print(f"Audio:      {output_audio}")
    print(f"Transcript: {output_text}")
    print(f"Duration:   {get_audio_duration(str(output_audio)):.2f}s")
    print(f"Language:   {detected_lang}")
    print(f"\nTranscript text:\n  \"{transcript}\"")
    print("\nTo use with dots-tts:")
    print(f"  dots-tts --ref-audio {output_audio} --ref-text \"{transcript}\" \\")
    print(f"      --text \"Your text here\" --language {detected_lang.upper()}")


if __name__ == "__main__":
    main()
