"""MeanFlow GPU acceptance harness: bf16 ear A/B + per-clip speed table.

Renders mf-NFE4 vs soar-10-step across EN/DE/ES/HI, measures wall-clock speedup
(DiT forward count drops from ~10x2=20 CFG steps to 4x1=4 no-CFG steps; end-to-end
speedup is typically ~2x since LLM/VAE costs are shared).

Usage::

    cd ~/dots-tts-mlx-meanflow
    uv run python scripts/gate_meanflow_acceptance.py

Wavs land in outputs/dots_tts/meanflow_ab/ (gitignored, NEVER /tmp).
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf

# Memory guard MUST come before any heavy allocation (project hard ceiling 45 GB).
mx.set_memory_limit(int(45 * (1 << 30)))

from dots_tts_mlx.model import DotsTtsModel  # noqa: E402

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DEFAULT_REF = (
    "/Users/shraey/.superset/worktrees/longcat/mlx/outputs/xdub_len/voice_short6s.wav"
)
_DEFAULT_REF_TEXT = (
    "I used to live in the cloud, now I'm running on your Mac, avatar video, dubbing."
)
_DEFAULT_MF = "weights/dots_tts_mlx_mf_bf16"
_DEFAULT_SOAR = "weights/dots_tts_mlx_bf16"
_DEFAULT_OUT = "outputs/dots_tts/meanflow_ab"

# ---------------------------------------------------------------------------
# Test sentences
# ---------------------------------------------------------------------------
TEST_SENTENCES: dict[str, str] = {
    "EN": "I used to live in the cloud, but now I run entirely on your Mac, no internet required.",
    "DE": "Früher lebte ich in der Cloud, aber jetzt laufe ich komplett auf deinem Mac, ganz ohne Internet.",
    "ES": "Antes vivía en la nube, pero ahora funciono completamente en tu Mac, sin necesidad de internet.",
    "HI": "मैं पहले क्लाउड में रहता था, लेकिन अब मैं पूरी तरह आपके मैक पर चलता हूँ।",
}

# Language code mapping for the model's language= arg.
LANG_CODES: dict[str, str] = {
    "EN": "en",
    "DE": "de",
    "ES": "es",
    "HI": "hi",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MeanFlow acceptance: A/B timing + ear wavs")
    p.add_argument("--ref-audio", default=_DEFAULT_REF)
    p.add_argument("--ref-text", default=_DEFAULT_REF_TEXT)
    p.add_argument("--mf-model", default=_DEFAULT_MF)
    p.add_argument("--soar-model", default=_DEFAULT_SOAR)
    p.add_argument("--out", default=_DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def timed_generate(
    model: DotsTtsModel,
    text: str,
    *,
    ref_audio: str,
    ref_text: str,
    language: str,
    seed: int,
    num_steps: int | None,
    guidance_scale: float,
) -> tuple[dict, float]:
    """Run generate and return (result, elapsed_seconds).

    mx.synchronize() is called after generate returns to ensure all GPU work
    is complete before stopping the timer.
    """
    t0 = time.perf_counter()
    result = model.generate(
        text,
        prompt_audio=ref_audio,
        prompt_text=ref_text,
        language=language,
        seed=seed,
        num_steps=num_steps,
        guidance_scale=guidance_scale,
    )
    # generate() calls mx.eval(wav) internally, but synchronize to be safe.
    mx.synchronize()
    elapsed = time.perf_counter() - t0
    return result, elapsed


def cooldown() -> None:
    """Between renders: free cache + GC + thermal cooldown sleep.

    NOTE: back-to-back GPU renders without a cooldown can confound speedup
    numbers via thermal throttling — especially on M-series chips where sustained
    load causes ANE/GPU clock reduction within ~60s. 3s is a light guard, not
    a full thermal reset; treat consecutive numbers as indicative, not absolute.
    """
    mx.synchronize()
    mx.clear_cache()
    gc.collect()
    time.sleep(3)


def write_wav(path: Path, audio: mx.array, sample_rate: int) -> None:
    arr = np.asarray(audio[0].astype(mx.float32))
    sf.write(str(path), arr, sample_rate, subtype="PCM_16")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MeanFlow GPU acceptance — bf16 ear A/B + speed table")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load both models once.
    # ------------------------------------------------------------------
    print(f"\n[1/2] Loading mf model from {args.mf_model} …")
    mf: DotsTtsModel = DotsTtsModel.from_pretrained(args.mf_model, dtype=mx.bfloat16)
    print(f"  mf.mode = {mf.mode!r}")
    assert mf.mode == "meanflow", f"Expected mf.mode=='meanflow', got {mf.mode!r}"

    print(f"\n[2/2] Loading soar model from {args.soar_model} …")
    soar: DotsTtsModel = DotsTtsModel.from_pretrained(args.soar_model, dtype=mx.bfloat16)
    print(f"  soar.mode = {soar.mode!r}")
    assert soar.mode == "flow_matching", (
        f"Expected soar.mode=='flow_matching', got {soar.mode!r}"
    )

    # ------------------------------------------------------------------
    # Warm-up: one untimed throwaway render per model so the FIRST timed
    # render doesn't pay mx.compile / Metal shader compilation cost.
    # ------------------------------------------------------------------
    print("\n--- Warm-up (untimed) ---")
    _warmup_text = "Warm up."
    print("  warming mf …")
    mf.generate(
        _warmup_text,
        prompt_audio=args.ref_audio,
        prompt_text=args.ref_text,
        language="en",
        seed=args.seed,
        num_steps=None,  # NFE 4 for meanflow
    )
    mx.synchronize()
    mx.clear_cache()
    gc.collect()

    print("  warming soar …")
    soar.generate(
        _warmup_text,
        prompt_audio=args.ref_audio,
        prompt_text=args.ref_text,
        language="en",
        seed=args.seed,
        num_steps=10,
        guidance_scale=1.2,
    )
    mx.synchronize()
    mx.clear_cache()
    gc.collect()
    print("  warm-up complete.")

    # ------------------------------------------------------------------
    # Main A/B loop — render each language, mf then soar, with a cooldown
    # between EVERY render.
    # ------------------------------------------------------------------
    rows: list[dict] = []
    written_wavs: list[Path] = []

    for lang, text in TEST_SENTENCES.items():
        lang_code = LANG_CODES[lang]
        print(f"\n--- {lang} ---")
        print(f"  text: {text[:60]}{'…' if len(text) > 60 else ''}")

        # mf render (NFE 4 — num_steps=None resolves to 4 in meanflow mode)
        print("  mf (NFE 4) …")
        mf_result, mf_secs = timed_generate(
            mf,
            text,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            language=lang_code,
            seed=args.seed,
            num_steps=None,
            guidance_scale=1.2,  # ignored in meanflow mode but accepted
        )
        mf_wav_path = out_dir / f"{lang}_mf.wav"
        write_wav(mf_wav_path, mf_result["audio"], mf_result["sample_rate"])
        written_wavs.append(mf_wav_path)
        print(f"  mf done: {mf_secs:.2f}s, patches={mf_result['num_patches']}")

        cooldown()

        # soar render (10 steps, CFG 1.2)
        print("  soar (10 steps, CFG 1.2) …")
        soar_result, soar_secs = timed_generate(
            soar,
            text,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            language=lang_code,
            seed=args.seed,
            num_steps=10,
            guidance_scale=1.2,
        )
        soar_wav_path = out_dir / f"{lang}_soar.wav"
        write_wav(soar_wav_path, soar_result["audio"], soar_result["sample_rate"])
        written_wavs.append(soar_wav_path)
        print(f"  soar done: {soar_secs:.2f}s, patches={soar_result['num_patches']}")

        cooldown()

        rows.append(
            {
                "lang": lang,
                "mf_s": mf_secs,
                "soar_s": soar_secs,
                "speedup": soar_secs / mf_secs,
                "mf_patches": mf_result["num_patches"],
                "soar_patches": soar_result["num_patches"],
            }
        )

    # ------------------------------------------------------------------
    # Summary table.
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(
        f"{'lang':<6} {'mf_NFE4_s':>10} {'soar_10_s':>10} {'speedup':>9} "
        f"{'mf_patches':>11} {'soar_patches':>13}"
    )
    print("-" * 70)
    for r in rows:
        print(
            f"{r['lang']:<6} {r['mf_s']:>10.2f} {r['soar_s']:>10.2f} "
            f"{r['speedup']:>9.2f}x {r['mf_patches']:>11} {r['soar_patches']:>13}"
        )
    print("=" * 70)

    peak_gb = mx.get_peak_memory() / (1 << 30)
    print(f"\nMLX peak memory: {peak_gb:.2f} GB")

    print("\nWritten wavs (for human ear check):")
    for p in written_wavs:
        print(f"  {p.resolve()}")

    print("\nDone.")


if __name__ == "__main__":
    main()
