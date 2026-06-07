import pathlib

import mlx.core as mx
import numpy as np
import pytest

F = pathlib.Path("tests/fixtures/dots_tts/semantic_encoder.npz")
W = pathlib.Path("weights/dots_tts_mlx/core.safetensors")
pytestmark = pytest.mark.skipif(not (F.exists() and W.exists()), reason="absent")


def _cos(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def test_semantic_encoder_parity():
    from dots_tts_mlx.loader import load_semantic_encoder

    d = np.load(F)
    enc = load_semantic_encoder(str(W), dtype=mx.float32)
    emb = np.asarray(enc(mx.array(d["x"])).astype(mx.float32))
    ma = float(np.max(np.abs(emb - d["emb"])))
    c = _cos(emb, d["emb"])
    assert c >= 0.9999 and ma <= 2e-2, f"semantic cosine {c:.5f} maxabs {ma:.4f}"
