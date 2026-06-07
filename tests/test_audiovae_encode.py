import pathlib

import numpy as np
import pytest

import mlx.core as mx

F = pathlib.Path("tests/fixtures/dots_tts/vae_encode.npz")
W = pathlib.Path("weights/dots_tts_mlx/vocoder.safetensors")
pytestmark = pytest.mark.skipif(
    not (F.exists() and W.exists()), reason="fixture/weights absent"
)


def _maxabs(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def test_encode_and_io():
    from dots_tts_mlx.io_helper import IOHelper
    from dots_tts_mlx.loader import load_audiovae

    d = np.load(F)
    vae = load_audiovae(str(W), dtype=mx.float32, with_encoder=True)
    lat = vae.encode(mx.array(d["wav"]).reshape(1, 1, -1))  # [1, 256, T]
    err = _maxabs(lat, d["lat"])
    assert err <= 2e-2, f"encode latent maxabs {err:.4f}"

    io = IOHelper("weights/dots_tts_mlx/latent_stats.npz")
    z = io.sample_from_latent(mx.array(d["lat"]), noise=mx.array(d["noise"]))
    assert _maxabs(z, d["z"]) <= 1e-4
    assert _maxabs(io.normalize(mx.array(d["z"])), d["zn"]) <= 1e-4
