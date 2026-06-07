import pathlib

import numpy as np
import pytest

import mlx.core as mx

F = pathlib.Path("tests/fixtures/dots_tts/attention.npz")
pytestmark = pytest.mark.skipif(not F.exists(), reason="fixture absent")


def _maxabs(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def _cos(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _weights(d):
    """Map the npz ``w_*`` keys back to the MLX state-dict scheme (dotted)."""
    return {k: mx.array(d[k]) for k in d.files if k.startswith("w_")}


def test_rotary_exact():
    from dots_tts_mlx.layers import RotaryEmbedding, apply_rotary_pos_emb

    d = np.load(F)
    rope = RotaryEmbedding(64, 10000.0)
    emb = rope(mx.array(d["rope_pos_ids"]))
    out = apply_rotary_pos_emb(emb, mx.array(d["rope_t"]))
    err = _maxabs(out, d["rope_t_rot"])
    assert err <= 1e-5, f"rotary maxabs {err:.2e}"


def test_attention_parity():
    from dots_tts_mlx.layers import MultiHeadAttention

    d = np.load(F)
    attn = MultiHeadAttention(
        1024,
        16,
        qkv_bias=False,
        qk_norm=True,
        norm_layer="RMSNorm",
        rotary_bias=True,
        rotary_theta=10000.0,
    )
    attn.load_weights(_weights(d))

    out_hp = attn(
        mx.array(d["q"]),
        mask=mx.array(d["mask"]),
        pos_ids=mx.array(d["pos_ids"]),
        hp=True,
    )
    err_hp = _maxabs(out_hp, d["out"])
    assert err_hp <= 1e-3, f"attn hp maxabs {err_hp:.2e}"

    out_fast = attn(
        mx.array(d["q"]),
        mask=mx.array(d["mask"]),
        pos_ids=mx.array(d["pos_ids"]),
        hp=False,
    )
    cos_fast = _cos(out_fast, d["out"])
    assert cos_fast >= 0.9999, f"attn fast cosine {cos_fast:.5f}"


def test_attention_parity_offset_pos_ids():
    """Offset pos_ids (the windowed/continuous-gen positions DiT will use)."""
    from dots_tts_mlx.layers import MultiHeadAttention

    d = np.load(F)
    attn = MultiHeadAttention(
        1024,
        16,
        qkv_bias=False,
        qk_norm=True,
        norm_layer="RMSNorm",
        rotary_bias=True,
        rotary_theta=10000.0,
    )
    attn.load_weights(_weights(d))

    out_hp = attn(
        mx.array(d["q"]),
        mask=mx.array(d["mask"]),
        pos_ids=mx.array(d["pos_ids_b"]),
        hp=True,
    )
    err_hp = _maxabs(out_hp, d["out_b"])
    assert err_hp <= 1e-3, f"attn(offset) hp maxabs {err_hp:.2e}"

    out_fast = attn(
        mx.array(d["q"]),
        mask=mx.array(d["mask"]),
        pos_ids=mx.array(d["pos_ids_b"]),
        hp=False,
    )
    cos_fast = _cos(out_fast, d["out_b"])
    assert cos_fast >= 0.9999, f"attn(offset) fast cosine {cos_fast:.5f}"
