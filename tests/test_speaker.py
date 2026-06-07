import pathlib

import numpy as np
import pytest

import mlx.core as mx

FB = pathlib.Path("tests/fixtures/dots_tts/fbank.npz")
XV = pathlib.Path("tests/fixtures/dots_tts/xvector.npz")
W = pathlib.Path("weights/dots_tts_mlx/speaker.safetensors")
pytestmark = pytest.mark.skipif(
    not (FB.exists() and XV.exists() and W.exists()), reason="fixture/weights absent"
)


def _cos(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def test_fbank_matches_torch():
    from dots_tts_mlx.speaker import kaldi_fbank

    d = np.load(FB)
    fb = np.asarray(kaldi_fbank(d["wav16k"], sample_rate=16000))
    err = np.max(np.abs(fb - d["fbank"]))
    assert err <= 1e-3, f"fbank maxabs {err:.4f}"


def test_xvector_cosine():
    from dots_tts_mlx.loader import load_speaker
    from dots_tts_mlx.speaker import CAMPPlus  # noqa: F401  (ensures module imports)

    d = np.load(XV)
    model = load_speaker(str(W), dtype=mx.float32)
    # Inject the TORCH fbank so the trunk is gated independently of the numpy fbank.
    xv = np.asarray(model(mx.array(d["fbank"])[None]).astype(mx.float32))
    c = _cos(xv, d["xvec"])
    assert c >= 0.9995, f"cosine {c:.5f}"
