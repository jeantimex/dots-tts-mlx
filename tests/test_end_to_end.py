"""End-to-end BEHAVIORAL gate for the dots.tts MLX AR runtime (Task 10).

Sample-exact parity is structurally unreachable on the shippable tf32 runtime (T9:
the euler ODE amplifies the per-step DiT bmm precision floor, so the waveform does
not sample-align with the torch golden). Per-stage gates (T2-T9) already prove every
component numerically; T10 gates END TO END on BEHAVIOR:

  * the waveform is finite, non-silent, and ~the right duration;
  * MLX-Whisper transcribes it with normalized-WER <= 0.20 vs the target text
    (a few-word slip is OK; gross garble fails);
  * the cloned voice resembles the reference (CAM++ cosine >= 0.30).

HEAVY (loads the full model + runs the AR decode + Whisper once). Selectable via
``-m slow``; skips if weights/fixtures are absent.
"""

import json
import pathlib

import numpy as np
import pytest

import mlx.core as mx

mx.set_memory_limit(int(45 * (1 << 30)))

W = pathlib.Path("weights/dots_tts_mlx")
CFG = pathlib.Path("tests/fixtures/dots_tts/golden_config.json")
WHISPER = pathlib.Path("weights/whisper-large-v3-mlx")
OUT_DIR = pathlib.Path("outputs/dots_tts/e2e")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not (W.exists() and CFG.exists()), reason="weights or golden config absent"
    ),
]


def _normalize(text: str) -> list[str]:
    keep = "".join(c.lower() if (c.isalnum() or c.isspace()) else " " for c in text)
    return keep.split()


def _wer(ref: str, hyp: str) -> float:
    r = _normalize(ref)
    h = _normalize(hyp)
    if not r:
        return 0.0 if not h else 1.0
    # Levenshtein on word tokens.
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=int)
    d[:, 0] = np.arange(len(r) + 1)
    d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
    return d[len(r), len(h)] / len(r)


def _campplus_xvec_16k(wav16k: np.ndarray) -> np.ndarray:
    from dots_tts_mlx.loader import load_speaker
    from dots_tts_mlx.speaker import kaldi_fbank

    speaker = load_speaker(str(W / "speaker.safetensors"), dtype=mx.float32)
    fb = kaldi_fbank(wav16k.astype(np.float32), sample_rate=16000)
    xv = speaker(mx.array(np.asarray(fb), dtype=mx.float32)[None])
    return np.asarray(xv.astype(mx.float32)).ravel()


def test_end_to_end_behavioral():
    from dots_tts_mlx.model import DotsTtsModel, _load_wav_mono, _resample

    cfg = json.loads(CFG.read_text())
    ref_audio = cfg["ref_audio"]
    ref_transcript = cfg["ref_transcript"]
    text = cfg["text"]

    model = DotsTtsModel.from_pretrained(W, dtype=mx.float32)
    out = model.generate(
        text=text,
        prompt_audio=ref_audio,
        prompt_text=ref_transcript,
        num_steps=int(cfg["num_steps"]),
        guidance_scale=float(cfg["guidance_scale"]),
        speaker_scale=float(cfg["speaker_scale"]),
        seed=int(cfg["seed"]),
    )
    wav = out["audio"]
    sr = out["sample_rate"]
    assert sr == 48000
    wav_np = np.asarray(wav.astype(mx.float32)).ravel()

    # --- save for inspection. ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "e2e_en.wav"
    import wave as _wave

    pcm = np.clip(wav_np, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype("<i2")
    with _wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())

    # --- basic sanity. ---
    assert np.all(np.isfinite(wav_np)), "non-finite samples in output"
    rms = float(np.sqrt(np.mean(wav_np**2)))
    assert rms > 0.01, f"output is silent (rms={rms:.4f})"
    duration = wav_np.shape[-1] / sr
    assert 1.0 <= duration <= 6.0, f"duration {duration:.2f}s outside [1.0, 6.0]"

    # --- intelligibility via MLX-Whisper. ---
    import mlx_whisper

    wav16k = _resample(wav_np, sr, 16000)
    result = mlx_whisper.transcribe(
        wav16k.astype(np.float32),
        path_or_hf_repo=str(WHISPER),
        language="en",
    )
    hyp = result["text"].strip()
    wer = _wer("the quick brown fox jumps over the lazy dog", hyp)
    print(f"\n[E2E] duration={duration:.2f}s rms={rms:.4f} patches={out['num_patches']}")
    print(f"[E2E] whisper: {hyp!r}")
    print(f"[E2E] WER={wer:.3f}")
    assert wer <= 0.20, f"WER {wer:.3f} > 0.20 (transcript={hyp!r})"

    # --- speaker similarity vs the reference. ---
    ref_raw, ref_sr = _load_wav_mono(ref_audio)
    ref16k = _resample(ref_raw, ref_sr, 16000)
    xv_out = _campplus_xvec_16k(wav16k)
    xv_ref = _campplus_xvec_16k(ref16k)
    sim = float(
        xv_out @ xv_ref / (np.linalg.norm(xv_out) * np.linalg.norm(xv_ref) + 1e-9)
    )
    print(f"[E2E] speaker-SIM={sim:.3f}")
    assert sim >= 0.30, f"speaker similarity {sim:.3f} < 0.30"
