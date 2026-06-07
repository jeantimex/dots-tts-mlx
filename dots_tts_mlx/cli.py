"""dots.tts-MLX voice-cloning CLI — pure-MLX multilingual TTS.

A continuous autoregressive flow-matching model: clean utterance onset, no discrete-codec
warm-up mumble. Writes ``{out-path}/{out-prefix}_000.wav`` @ 48 kHz.

Runtime code imports only mlx / numpy / soundfile + ``dots_tts_mlx`` and makes no
torch calls. (torch + transformers are present transitively via mlx-lm — same as
the wider MLX-audio ecosystem — but the inference math is pure MLX.)

Usage:
    dots-tts --text "..." --ref-audio ref.wav \
        --ref-text "transcript of ref.wav" --out-prefix dots_clone --language EN

    # equivalently, as a module:
    python -m dots_tts_mlx.cli --text "..." --ref-audio ref.wav --language EN
"""
from __future__ import annotations

import argparse

import mlx.core as mx

mx.set_memory_limit(int(45 * (1 << 30)))  # memory-ceiling safety guard — set BEFORE heavy alloc

import os  # noqa: E402
import subprocess  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from dots_tts_mlx.loader import from_pretrained  # noqa: E402


def _clean_onset(wav: np.ndarray, sr: int) -> np.ndarray:
    """Trim the fixed BigVGAN vocoder onset transient off the start of a dots.tts clip.

    dots.tts (via its BigVGAN vocoder) emits a deterministic ~50-150 ms low-level burst (a soft
    "hhh"/breath, byte-identical across clips) at sample 0, followed by a short silence gap before
    the real speech. It's a fixed ~-26 dBFS transient — on quiet dubs (peaks ~-20 dBFS) it's audible.

    Mirrors ``clean_tts_audio._speech_bounds`` (smoothed-RMS gate + sustained-run requirement) to find
    the *real* speech onset, then keeps ``[onset - lead_pad, end]``. Differences vs the MisoTTS cleaner:
    a tighter ``rel_db`` (12, not 16) and lighter smoothing (40 ms, not 150) so the threshold sits
    cleanly *between* the low transient and the louder speech body — the wide 16 dB / 150 ms gate would
    flag the transient itself as "speech" (it's within 16 dB of the body) and trim nothing. The lead-pad
    is kept small (30 ms) so a soft real onset consonant is never clipped; the transient + dead-air gap
    that precede the speech are both removed. A 10 ms fade-in suppresses the cut click.

    Pure numpy (no torch/MLX). Returns the trimmed-and-faded mono float32 array.
    """
    rel_db, hop_ms, smooth_ms, run_ms = 12.0, 10.0, 40.0, 60.0
    lead_pad_ms, fade_ms = 30.0, 10.0
    x = wav.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(1)
    hop = max(1, int(sr * hop_ms / 1000))
    n = max(1, (len(x) - hop) // hop)
    rms = np.array([np.sqrt((x[i * hop:i * hop + hop] ** 2).mean() + 1e-12) for i in range(n)])
    db = 20 * np.log10(rms + 1e-9)
    k = max(1, int(smooth_ms / hop_ms))
    sm = np.convolve(db, np.ones(k) / k, mode="same")     # light smooth — keep transient/speech distinct
    above = sm > (sm.max() - rel_db)
    run = max(1, int(run_ms / hop_ms))                    # require a sustained run (not the burst's edge)
    ok = np.convolve(above.astype(int), np.ones(run, dtype=int), mode="valid") == run
    starts = np.where(ok)[0]
    if len(starts) == 0:                                  # nothing looked like speech -> leave untouched
        return x
    s0 = int(starts[0] * hop)
    i0 = max(0, s0 - int(lead_pad_ms / 1000 * sr))        # small lead-pad: never clip the first phoneme
    y = x[i0:].copy()
    nf = min(int(sr * fade_ms / 1000), len(y) // 2)       # anti-click fade-in at the cut
    if nf > 0:
        y[:nf] *= np.linspace(0.0, 1.0, nf, dtype=y.dtype)
    return y


def build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    ap = argparse.ArgumentParser(description="dots.tts-MLX voice-cloning TTS")
    ap.add_argument("--model", default="weights/dots_tts_mlx")
    ap.add_argument("--text", default=None)
    ap.add_argument("--ref-audio", default=None)
    ap.add_argument(
        "--ref-text",
        default=None,
        help="reference transcript (prompt text). Omit -> x-vector-only clone.",
    )
    ap.add_argument("--out-path", default="outputs/dots_tts")
    ap.add_argument("--out-prefix", default="dots_clone")
    ap.add_argument(
        "--language",
        default=None,
        help="uppercase ISO code (e.g. EN/DE/ES/FR/HI); default None = no tag.",
    )
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument("--guidance-scale", type=float, default=1.2)
    ap.add_argument("--speaker-scale", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-generate-length", type=int, default=500)
    ap.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="playback tempo on the output wav via pitch-preserving ffmpeg atempo "
        "(<1 = slower, >1 = faster; 0.9 ~= 10%% slower). dots.tts is a self-pacing AR "
        "model with no native rate control, so this is a post-hoc time-stretch.",
    )
    ap.add_argument(
        "--trim-onset",
        dest="trim_onset",
        action="store_true",
        default=True,
        help="trim the fixed BigVGAN vocoder onset transient (a soft ~50-150 ms 'hhh'/breath at "
        "sample 0) via an energy gate + 10 ms fade-in. ON by default; applied BEFORE --speed.",
    )
    ap.add_argument(
        "--no-trim-onset",
        dest="trim_onset",
        action="store_false",
        help="disable onset-transient trimming (keep the raw vocoder output verbatim).",
    )
    ap.add_argument(
        "--enroll",
        action="store_true",
        help="enrollment mode: compute a reusable SpeakerProfile from --ref-audio/--ref-text.",
    )
    ap.add_argument(
        "--profile-out",
        default=None,
        help="(enroll mode) directory to write the .dtprofile bundle to.",
    )
    ap.add_argument(
        "--profile",
        default=None,
        help="(use mode) a .dtprofile to clone from — replaces --ref-audio/--ref-text.",
    )
    return ap


def main() -> int:
    ap = build_parser()
    args = ap.parse_args()

    if args.enroll:
        if not args.ref_audio or not args.ref_text or not args.profile_out:
            ap.error("--enroll requires --ref-audio, --ref-text, and --profile-out")
        model = from_pretrained(args.model, dtype=mx.bfloat16).model
        prof = model.enroll(args.ref_audio, args.ref_text, speaker_scale=args.speaker_scale)
        prof.save(args.profile_out)
        print(
            f"[dots] enrolled -> {args.profile_out}  "
            f"({prof.prompt_patch_count} prompt patches, scale {prof.speaker_scale})  "
            f"MLX peak {mx.get_peak_memory() / (1 << 30):.2f}GB",
            flush=True,
        )
        return 0

    if not args.text:
        ap.error("--text is required for generation")

    model = from_pretrained(args.model, dtype=mx.bfloat16).model

    if args.profile:
        from dots_tts_mlx.profile import SpeakerProfile

        prof = SpeakerProfile.load(args.profile)
        out = model.generate(
            text=args.text,
            profile=prof,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            language=args.language,
            seed=args.seed,
            max_generate_length=args.max_generate_length,
        )
    else:
        if not args.ref_audio:
            ap.error("--ref-audio is required (or pass --profile)")
        out = model.generate(
            text=args.text,
            prompt_audio=args.ref_audio,
            prompt_text=args.ref_text,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            speaker_scale=args.speaker_scale,
            language=args.language,
            seed=args.seed,
            max_generate_length=args.max_generate_length,
        )

    wav = np.asarray(out["audio"].astype(mx.float32)).ravel()
    sr = int(out["sample_rate"])

    if args.trim_onset:
        # Strip the fixed vocoder onset transient on the CLEAN array, BEFORE --speed atempo,
        # so we time-stretch clean speech (not cleaned-after-noise / noise-then-stretched).
        n0 = wav.shape[-1]
        wav = _clean_onset(wav, sr)
        trimmed_ms = (n0 - wav.shape[-1]) / sr * 1000
        print(f"[dots] --trim-onset: removed {trimmed_ms:.0f} ms leading transient", flush=True)

    os.makedirs(args.out_path, exist_ok=True)
    out_file = os.path.join(args.out_path, f"{args.out_prefix}_000.wav")
    sf.write(out_file, wav, sr)

    if abs(args.speed - 1.0) > 1e-3:
        # Post-hoc pitch-preserving time-stretch — dots.tts has no native speech-rate knob
        # (self-pacing AR model). atempo accepts [0.5, 2.0] per filter; chain for extremes.
        factors, t = [], float(args.speed)
        while t > 2.0:
            factors.append(2.0)
            t /= 2.0
        while t < 0.5:
            factors.append(0.5)
            t /= 0.5
        factors.append(t)
        chain = ",".join(f"atempo={f:.6f}" for f in factors)
        tmp = out_file + ".tmp.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", out_file, "-filter:a", chain, tmp],
            check=True,
        )
        os.replace(tmp, out_file)
        wav = np.asarray(sf.read(out_file)[0]).ravel()
        print(f"[dots] applied --speed {args.speed} (atempo {chain})", flush=True)

    dur = wav.shape[-1] / sr
    print(
        f"[dots] -> {out_file}  {dur:.2f}s @ {sr}Hz  "
        f"{out.get('num_patches', '?')} patches  "
        f"MLX peak {mx.get_peak_memory() / (1 << 30):.2f}GB",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
