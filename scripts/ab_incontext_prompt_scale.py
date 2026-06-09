"""Manual in-context prompt-scale A/B (FIXED vs OLD) for dots-tts-mlx.

The fix on branch ``fix/prompt-embedding-scale`` feeds the in-context (``prompt_text``)
patch encoder DENORMALIZED prompt latents (was: normalized — the bug). The numerical
gate already proves correctness. This harness produces FIXED-vs-OLD audio for the user
to hear + an x-vector (speaker-embedding) cosine-to-reference for the results doc.

It is NOT a committed test gate. Outputs land in ``outputs/dots_tts/prompt_scale_ab/``.

OLD (buggy) behavior is reproduced WITHOUT reverting the fix, by monkeypatching
``DotsTtsModel._prefill`` so the prompt latents it recomputes ``patch_emb`` from are
re-normalized — exactly the pre-fix scale of the patch_emb scattered into the LLM.

Metric: x-vector cosine via package deps only (no mlx_whisper dependency):
``load_speaker`` (CAM++) over ``kaldi_fbank`` of the 16 kHz-resampled waveform.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf

mx.set_memory_limit(int(45 * (1 << 30)))

from dots_tts_mlx.loader import load_speaker  # noqa: E402
from dots_tts_mlx.model import (  # noqa: E402
    _SPEAKER_FBANK_SAMPLE_RATE,
    DotsTtsModel,
    _load_wav_mono,
    _resample,
)
from dots_tts_mlx.speaker import kaldi_fbank  # noqa: E402

WEIGHTS = "weights/dots_tts_mlx"
REF = "/Users/shraey/.superset/worktrees/longcat/mlx/outputs/xdub_len/voice_short6s.wav"
REF_TEXT = (
    "I used to live in the cloud, now I'm running on your Mac, avatar video, dubbing."
)
TEXT = (
    "Thank you for trying this out. I think you will be surprised by how natural "
    "it sounds."
)
OUT_DIR = Path("outputs/dots_tts/prompt_scale_ab")

GEN_KW = dict(
    text=TEXT,
    prompt_audio=REF,
    prompt_text=REF_TEXT,
    num_steps=10,
    guidance_scale=1.2,
    speaker_scale=1.5,
    language="EN",
    seed=42,
)


def _write_wav(out: dict, path: Path) -> tuple[float, float]:
    """Write the generate() output wav; return (duration_s, std)."""
    wav = np.asarray(out["audio"].astype(mx.float32)).ravel()
    sr = int(out["sample_rate"])
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav, sr)
    return float(wav.shape[-1] / sr), float(np.std(wav))


def _xvector(speaker, wav: np.ndarray, sr: int) -> np.ndarray:
    """CAM++ x-vector for a mono waveform: resample->16k, fbank, forward, L2-normalize."""
    wav16k = _resample(wav.astype(np.float32), sr, _SPEAKER_FBANK_SAMPLE_RATE)
    max_len = _SPEAKER_FBANK_SAMPLE_RATE * 10
    if wav16k.shape[-1] > max_len:
        wav16k = wav16k[:max_len]
    fbank = kaldi_fbank(wav16k, sample_rate=_SPEAKER_FBANK_SAMPLE_RATE)  # [T, 80]
    fbank_mx = mx.array(np.asarray(fbank), dtype=mx.float32)[None]  # [1, T, 80]
    xvec = speaker(fbank_mx)  # [1, 512]
    mx.eval(xvec)
    v = np.asarray(xvec.astype(mx.float32)).ravel()
    n = float(np.linalg.norm(v)) + 1e-9
    return v / n


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[load] model <- {WEIGHTS} (bf16)")
    model = DotsTtsModel.from_pretrained(WEIGHTS, dtype=mx.bfloat16)

    # --- FIXED run (the un-patched, on-branch behavior). ---
    print("[gen] FIXED (denormalized prompt latents -> patch_emb)")
    mx.random.seed(GEN_KW["seed"])
    fixed_out = model.generate(**GEN_KW)
    fixed_path = OUT_DIR / "fixed_incontext.wav"
    fixed_s, fixed_std = _write_wav(fixed_out, fixed_path)
    print(f"      -> {fixed_path}  ({fixed_s:.2f}s, std={fixed_std:.4f})")

    # drain barrier between runs.
    mx.synchronize()
    mx.clear_cache()

    # --- OLD run: monkeypatch _prefill to re-normalize prompt latents (the bug). ---
    print("[gen] OLD (re-normalized prompt latents -> patch_emb)")
    _orig_prefill = DotsTtsModel._prefill

    def _old_prefill(self, *args, prompt_denorm_latents=None, **kw):
        pdl = (
            self.io.normalize(prompt_denorm_latents)
            if prompt_denorm_latents is not None
            else None
        )
        return _orig_prefill(self, *args, prompt_denorm_latents=pdl, **kw)

    DotsTtsModel._prefill = _old_prefill
    try:
        mx.random.seed(GEN_KW["seed"])
        old_out = model.generate(**GEN_KW)
    finally:
        DotsTtsModel._prefill = _orig_prefill  # restore the fix
    old_path = OUT_DIR / "old_incontext.wav"
    old_s, old_std = _write_wav(old_out, old_path)
    print(f"      -> {old_path}  ({old_s:.2f}s, std={old_std:.4f})")

    mx.synchronize()
    mx.clear_cache()

    # --- x-vector cosine similarity to the reference. ---
    print("[xvec] loading CAM++ speaker encoder")
    speaker = load_speaker(str(Path(WEIGHTS) / "speaker.safetensors"), dtype=mx.float32)

    ref_raw, ref_sr = _load_wav_mono(REF)
    fixed_raw, fixed_sr = _load_wav_mono(str(fixed_path))
    old_raw, old_sr = _load_wav_mono(str(old_path))

    ref_x = _xvector(speaker, ref_raw, ref_sr)
    fixed_x = _xvector(speaker, fixed_raw, fixed_sr)
    old_x = _xvector(speaker, old_raw, old_sr)

    cos_old = _cos(ref_x, old_x)
    cos_fixed = _cos(ref_x, fixed_x)

    summary = {
        "ref": REF,
        "ref_text": REF_TEXT,
        "text": TEXT,
        "gen_kwargs": {k: v for k, v in GEN_KW.items() if k not in ("text",)},
        "old": {
            "wav": str(old_path),
            "audio_s": round(old_s, 4),
            "std": round(old_std, 6),
            "xvec_cos_to_ref": round(cos_old, 6),
        },
        "fixed": {
            "wav": str(fixed_path),
            "audio_s": round(fixed_s, 4),
            "std": round(fixed_std, 6),
            "xvec_cos_to_ref": round(cos_fixed, 6),
        },
    }
    json_path = OUT_DIR / "ab.json"
    json_path.write_text(json.dumps(summary, indent=2))

    # --- 2-row table. ---
    print()
    print("=" * 60)
    print("  in-context prompt-scale A/B  (REF vs generated)")
    print("=" * 60)
    print(f"  {'run':<7}{'audio_s':>10}{'std':>12}{'xvec_cos_to_ref':>20}")
    print("  " + "-" * 47)
    print(f"  {'OLD':<7}{old_s:>10.2f}{old_std:>12.4f}{cos_old:>20.4f}")
    print(f"  {'FIXED':<7}{fixed_s:>10.2f}{fixed_std:>12.4f}{cos_fixed:>20.4f}")
    print("=" * 60)
    print(f"  ref x-vector self-cos = {_cos(ref_x, ref_x):.4f} (sanity, ~1.0)")
    print(f"  summary -> {json_path}")
    print(f"  fixed wav -> {fixed_path}")
    print(f"  old   wav -> {old_path}")


if __name__ == "__main__":
    main()
