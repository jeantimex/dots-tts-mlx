import pathlib

import numpy as np
import pytest

import mlx.core as mx

F = pathlib.Path("tests/fixtures/dots_tts/vocoder_decode.npz")
W = pathlib.Path("weights/dots_tts_mlx/vocoder.safetensors")
pytestmark = pytest.mark.skipif(
    not (F.exists() and W.exists()), reason="fixture/weights absent"
)


def _psnr(a, b):
    mse = float(np.mean((a - b) ** 2)) + 1e-12
    peak = float(np.max(np.abs(b))) + 1e-9
    return 10 * np.log10(peak * peak / mse)


def test_decode_matches_torch_psnr():
    from dots_tts_mlx.loader import load_audiovae  # builds AudioVAE, fp32

    d = np.load(F)
    vae = load_audiovae(str(W), dtype=mx.float32)
    wav = np.asarray(
        vae.decode(mx.array(d["latent"])).astype(mx.float32)
    ).reshape(d["wav"].shape)
    psnr = _psnr(wav, d["wav"])
    assert psnr >= 45.0, f"decode PSNR {psnr:.1f} dB < 45"
