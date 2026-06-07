"""Validates the CONVERTED dots.tts MLX output.

The conversion itself runs offline under the torch *oracle* venv (the repo mlx
venv has no torch), so this test only inspects the produced artifacts and skips
when they are absent.
"""

import pathlib

import mlx.core as mx
import pytest

OUT = pathlib.Path("weights/dots_tts_mlx")
pytestmark = pytest.mark.skipif(
    not (OUT / "core.safetensors").exists(),
    reason="converted weights absent (run convert.py first)",
)


def test_converted_weights_valid():
    for name in ("core", "vocoder", "speaker"):
        w = mx.load(str(OUT / f"{name}.safetensors"))
        assert w, f"{name} empty"
        for k, v in w.items():
            assert mx.all(mx.isfinite(v.astype(mx.float32))).item(), f"non-finite in {name}:{k}"
        assert not any(k.endswith(("weight_g", "weight_v")) for k in w), (
            f"unfolded weight_norm in {name}"
        )


def test_latent_stats_present():
    stats = mx.load(str(OUT / "latent_stats.npz"))
    assert set(stats) >= {"mean", "var"} and tuple(stats["mean"].shape) == (128,)
