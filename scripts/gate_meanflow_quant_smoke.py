"""MeanFlow quant-ladder smoke: load + render test for int8 and int4 mf variants.

Verifies:
- model.mode == "meanflow" for both variants
- model.flow_solver.dit.duration_embedder is not None
- Renders EN/DE/ES/HI sentences (NFE4, seed=42) for each variant
- Reports MLX peak memory per variant

Usage::

    cd ~/dots-tts-mlx-meanflow
    uv run python scripts/gate_meanflow_quant_smoke.py

Wavs land in outputs/dots_tts/meanflow_ab/ (gitignored, NEVER /tmp).
"""

from __future__ import annotations

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
# Config
# ---------------------------------------------------------------------------
_REF = "/Users/shraey/.superset/worktrees/longcat/mlx/outputs/xdub_len/voice_short6s.wav"
_REF_TEXT = "I used to live in the cloud, now I'm running on your Mac, avatar video, dubbing."
_OUT_DIR = Path("outputs/dots_tts/meanflow_ab")
_SEED = 42

VARIANTS: dict[str, str] = {
    "int8": "weights/dots_tts_mlx_mf_int8",
    "int4": "weights/dots_tts_mlx_mf_int4",
}

TEST_SENTENCES: dict[str, str] = {
    "EN": "I used to live in the cloud, but now I run entirely on your Mac, no internet required.",
    "DE": "Früher lebte ich in der Cloud, aber jetzt laufe ich komplett auf deinem Mac, ganz ohne Internet.",
    "ES": "Antes vivía en la nube, pero ahora funciono completamente en tu Mac, sin necesidad de internet.",
    "HI": "मैं पहले क्लाउड में रहता था, लेकिन अब मैं पूरी तरह आपके मैक पर चलता हूँ।",
}

LANG_CODES: dict[str, str] = {
    "EN": "en",
    "DE": "de",
    "ES": "es",
    "HI": "hi",
}


def write_wav(path: Path, audio: mx.array, sample_rate: int) -> None:
    arr = np.asarray(audio[0].astype(mx.float32))
    sf.write(str(path), arr, sample_rate, subtype="PCM_16")


def drain() -> None:
    mx.synchronize()
    mx.clear_cache()
    gc.collect()
    time.sleep(3)


def main() -> None:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MeanFlow quant-ladder smoke: int8 + int4 load + render")
    print("=" * 70)

    written: list[Path] = []
    summary: list[dict] = []

    for variant_name, model_dir in VARIANTS.items():
        print(f"\n{'=' * 60}")
        print(f"Variant: {variant_name}  ({model_dir})")
        print(f"{'=' * 60}")

        # Reset peak memory counter before loading this variant.
        mx.reset_peak_memory()

        print(f"  Loading model from {model_dir} …")
        model: DotsTtsModel = DotsTtsModel.from_pretrained(model_dir, dtype=mx.bfloat16)

        # Assertions: correct mode + duration_embedder present.
        assert model.mode == "meanflow", (
            f"FAIL: expected mode='meanflow', got {model.mode!r}"
        )
        assert model.flow_solver.dit.duration_embedder is not None, (
            "FAIL: flow_solver.dit.duration_embedder is None — "
            "DiT did NOT load the duration_embedder despite the quantized LLM"
        )
        print(f"  mode={model.mode!r}  duration_embedder=present  [OK]")

        variant_wavs: list[Path] = []

        for lang, text in TEST_SENTENCES.items():
            lang_code = LANG_CODES[lang]
            out_path = _OUT_DIR / f"{lang}_mf_{variant_name}.wav"
            print(f"  Rendering {lang} → {out_path.name} …")

            result = model.generate(
                text,
                prompt_audio=_REF,
                prompt_text=_REF_TEXT,
                language=lang_code,
                seed=_SEED,
                num_steps=None,  # → NFE 4 in meanflow mode
            )
            mx.synchronize()

            write_wav(out_path, result["audio"], result["sample_rate"])
            variant_wavs.append(out_path)
            written.append(out_path)
            print(
                f"    done: patches={result['num_patches']}  "
                f"samples={result['audio'].shape[-1]}"
            )
            drain()

        peak_gb = mx.get_peak_memory() / (1 << 30)
        print(f"\n  MLX peak memory ({variant_name}): {peak_gb:.2f} GB")
        summary.append(
            {"variant": variant_name, "peak_gb": peak_gb, "wavs": variant_wavs}
        )

        # Unload model and drain before next variant.
        del model
        drain()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    for row in summary:
        print(f"  {row['variant']:8s}  peak={row['peak_gb']:.2f} GB")

    print("\nWritten wavs:")
    for p in written:
        print(f"  {p.resolve()}")

    print("\nDone.")


if __name__ == "__main__":
    main()
