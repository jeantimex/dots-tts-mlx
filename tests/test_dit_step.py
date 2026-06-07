import pathlib

import numpy as np
import pytest

import mlx.core as mx

F = pathlib.Path("tests/fixtures/dots_tts/dit_step.npz")
W = pathlib.Path("weights/dots_tts_mlx/core.safetensors")
pytestmark = pytest.mark.skipif(
    not (F.exists() and W.exists()), reason="fixture or weights absent"
)


def _cos(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def test_dit_velocity_parity():
    from dots_tts_mlx.loader import load_dit

    d = np.load(F)
    dit = load_dit(str(W), dtype=mx.float32)
    vt = np.asarray(
        dit(
            mx.array(d["x"]),
            mx.array(d["t"]),
            attn_mask=mx.array(d["attn_mask"]),
            pos_ids=mx.array(d["pos_ids"]),
            g_cond=mx.array(d["g_cond"]),
        ).astype(mx.float32)
    )
    ma = float(np.max(np.abs(vt - d["vt"])))
    c = _cos(vt, d["vt"])
    assert c >= 0.9999 and ma <= 2e-2, f"DiT cosine {c:.5f} maxabs {ma:.4f}"
