import pathlib

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.cache import KVCache

from dots_tts_mlx.layers import Conv1d, Linear, Mlp, MultiHeadAttention, RMSNorm, silu
from dots_tts_mlx.semantic_encoder import (
    SuperviseEncoder,
    TransformerEncoderLayer,
    VAESemanticEncoder,
)


def _rand(shape, seed, scale=0.1):
    rng = np.random.default_rng(seed)
    return mx.array((rng.standard_normal(shape) * scale).astype(np.float32))


def _tiny_encoder(*, in_dim=4, hidden=8, heads=2, ffn=16, layers=2, patch_size=4, seed=0):
    """A small but structurally-faithful VAESemanticEncoder with random weights.

    No real weights / no GPU needed; lets the streaming-vs-full equivalence run on CPU-ish
    tiny tensors. ds_proj is the causal stride-2 k=2 conv (left_padding=1); the encoder is
    a causal SuperviseEncoder with NO qk-norm / NO rotary (matches the real patch encoder).
    """
    odr = patch_size // 2
    ds_proj = Conv1d(_rand((in_dim, 2, in_dim), seed), _rand((in_dim,), seed + 1),
                     causal=True, stride=2)
    in_proj = Linear(_rand((hidden, in_dim), seed + 2), _rand((hidden,), seed + 3))
    out_proj = Linear(_rand((hidden, hidden * odr), seed + 4), _rand((hidden,), seed + 5))
    enc_layers = []
    for i in range(layers):
        s = seed + 10 + i * 20
        attn = MultiHeadAttention(hidden, heads, qkv_bias=False, qk_norm=False, rotary_bias=False)
        attn.load_weights({
            "w_q_proj_weight": _rand((hidden, hidden), s),
            "w_k_proj_weight": _rand((hidden, hidden), s + 1),
            "w_v_proj_weight": _rand((hidden, hidden), s + 2),
            "w_o_proj_weight": _rand((hidden, hidden), s + 3),
            "w_o_proj_bias": _rand((hidden,), s + 4),
        })
        attn_norm = RMSNorm(hidden, weight=_rand((hidden,), s + 5) + 1.0)
        ffn_norm = RMSNorm(hidden, weight=_rand((hidden,), s + 6) + 1.0)
        ffn_dim = ffn
        layer_ffn = Mlp(Linear(_rand((ffn_dim, hidden), s + 7), _rand((ffn_dim,), s + 8)),
                        Linear(_rand((hidden, ffn_dim), s + 9), _rand((hidden,), s + 10)), silu)
        enc_layers.append(TransformerEncoderLayer(attn_norm, attn, ffn_norm, layer_ffn))
    encoder = SuperviseEncoder(enc_layers, causal=True)
    return VAESemanticEncoder(ds_proj, in_proj, encoder, out_proj,
                              out_ds_rate=odr, patch_size=patch_size)


def test_init_decode_state_shapes():
    enc = _tiny_encoder(in_dim=4, hidden=8, layers=3, seed=1)
    state = enc.init_decode_state(dtype=mx.float32)
    # conv_tail = [1, left_padding=1, in_dim], zeros.
    assert state.conv_tail.shape == (1, 1, 4)
    assert float(mx.abs(state.conv_tail).max()) == 0.0
    # one KVCache per encoder layer, all empty (offset 0).
    assert len(state.layer_caches) == 3
    assert all(c.offset == 0 for c in state.layer_caches)


def test_downsample_step_equals_full_conv():
    enc = _tiny_encoder(in_dim=4, patch_size=4, seed=7)
    p = enc.patch_size
    # Two patches of denormalized latents, fed as one full stream vs streamed per-patch.
    x = _rand((1, 2 * p, 4), seed=99, scale=1.0)
    full = enc._downsample(x)  # [1, 2*odr, 4] full causal conv (zero left-pad)

    tail = mx.zeros((1, enc.ds_proj.left_padding, 4), dtype=mx.float32)
    outs = []
    for k in range(2):
        patch = x[:, k * p:(k + 1) * p, :]
        down, tail = enc._downsample_step(patch, tail)
        outs.append(down)
    streamed = mx.concatenate(outs, axis=1)  # [1, 2*odr, 4]

    assert streamed.shape == full.shape
    maxabs = float(mx.abs(streamed - full).max())
    assert maxabs <= 1e-5, f"streaming conv diverged: max|Δ|={maxabs:.2e}"
    # tail after the last patch == that patch's last left_padding frames.
    last = x[:, -enc.ds_proj.left_padding:, :]
    assert float(mx.abs(tail - last).max()) == 0.0


def test_mha_step_matches_full_causal():
    # One attention layer; compare streamed (2 tokens at a time) vs full causal __call__.
    hidden, heads, T = 8, 2, 6
    attn = MultiHeadAttention(hidden, heads, qkv_bias=False, qk_norm=False, rotary_bias=False)
    attn.load_weights({
        "w_q_proj_weight": _rand((hidden, hidden), 1),
        "w_k_proj_weight": _rand((hidden, hidden), 2),
        "w_v_proj_weight": _rand((hidden, hidden), 3),
        "w_o_proj_weight": _rand((hidden, hidden), 4),
        "w_o_proj_bias": _rand((hidden,), 5),
    })
    x = _rand((1, T, hidden), seed=42, scale=1.0)

    # Full causal reference (hp=True -> fp32, so any diff is logic not tf32 noise).
    full_mask = mx.tril(mx.ones((T, T), dtype=mx.bool_))[None]  # [1, T, T]
    ref = attn(x, mask=full_mask, hp=True)

    # Streamed: 2 tokens per step into a KVCache, block mask [n, t_past+n].
    cache = KVCache()
    outs = []
    for k in range(0, T, 2):
        blk = x[:, k:k + 2, :]
        n = blk.shape[1]
        t_past = cache.offset
        mask = mx.concatenate(
            [mx.ones((n, t_past), dtype=mx.bool_), mx.tril(mx.ones((n, n), dtype=mx.bool_))],
            axis=1,
        )
        outs.append(attn.step(blk, cache, mask, hp=True))
    streamed = mx.concatenate(outs, axis=1)

    maxabs = float(mx.abs(streamed - ref).max())
    assert maxabs <= 1e-4, f"streaming attention diverged: max|Δ|={maxabs:.2e}"


def test_supervise_encoder_decode_step_matches_call():
    enc = _tiny_encoder(in_dim=4, hidden=8, heads=2, layers=3, seed=11)
    sup = enc.encoder
    hidden, T = 8, 6
    x = _rand((1, T, hidden), seed=21, scale=1.0)

    ref = sup(x, hp=True)  # full causal forward

    caches = [KVCache() for _ in sup.layers]
    outs = []
    for k in range(0, T, 2):
        outs.append(sup.decode_step(x[:, k:k + 2, :], caches, hp=True))
    streamed = mx.concatenate(outs, axis=1)

    maxabs = float(mx.abs(streamed - ref).max())
    assert maxabs <= 1e-4, f"encoder decode_step diverged: max|Δ|={maxabs:.2e}"


def test_prefill_decode_patch_equals_recompute_full():
    enc = _tiny_encoder(in_dim=4, hidden=8, heads=2, layers=3, patch_size=4, seed=33)
    p = enc.patch_size
    n_prompt, n_gen = 3, 5
    # Full denormalized history: prompt patches followed by generated patches.
    hist = _rand((1, (n_prompt + n_gen) * p, 4), seed=55, scale=1.0)
    prompt = hist[:, : n_prompt * p, :]

    # Recompute-full oracle: encoder over the WHOLE history, last token per patch.
    full = enc(hist, hp=True)  # [1, n_prompt+n_gen, out_dim]

    # Streaming: prefill the prompt, then decode each generated patch.
    state = enc.init_decode_state(dtype=mx.float32)
    enc.prefill(prompt, state, hp=True)
    for k in range(n_gen):
        start = (n_prompt + k) * p
        patch = hist[:, start:start + p, :]
        tok = enc.decode_patch(patch, state, hp=True)  # [1, 1, out_dim]
        ref_tok = full[:, n_prompt + k:n_prompt + k + 1, :]
        maxabs = float(mx.abs(tok - ref_tok).max())
        assert maxabs <= 2e-4, f"patch {k} diverged: max|Δ|={maxabs:.2e}"


def test_decode_patch_no_prompt():
    # No-prompt path: empty cache + zero conv_tail; first patch must match __call__.
    enc = _tiny_encoder(in_dim=4, hidden=8, layers=2, patch_size=4, seed=66)
    p = enc.patch_size
    hist = _rand((1, 2 * p, 4), seed=77, scale=1.0)
    full = enc(hist, hp=True)
    state = enc.init_decode_state(dtype=mx.float32)
    for k in range(2):
        tok = enc.decode_patch(hist[:, k * p:(k + 1) * p, :], state, hp=True)
        maxabs = float(mx.abs(tok - full[:, k:k + 1, :]).max())
        assert maxabs <= 2e-4, f"no-prompt patch {k} diverged: max|Δ|={maxabs:.2e}"


_CORE = pathlib.Path("weights/dots_tts_mlx/core.safetensors")


@pytest.mark.skipif(not _CORE.exists(), reason="core weights absent")
def test_streaming_equals_recompute_real_weights():
    """Decisive parity gate: real 24-layer patch encoder, fp32, streaming == recompute-full.

    ~40 patches (where O(n^3) would visibly diverge). fp32 (hp=True) so any gap is a logic
    bug, not tf32 noise. Tolerance matches the semantic-encoder parity gate
    (tests/test_semantic_encoder.py: cosine >= 0.9999, maxabs <= 2e-2).
    """
    from dots_tts_mlx.loader import load_semantic_encoder

    mx.set_memory_limit(int(45 * (1 << 30)))
    enc = load_semantic_encoder(str(_CORE), dtype=mx.float32)
    p = enc.patch_size
    n_prompt, n_gen = 4, 36
    rng = np.random.default_rng(2024)
    hist = mx.array(rng.standard_normal((1, (n_prompt + n_gen) * p, 128)).astype(np.float32))
    prompt = hist[:, : n_prompt * p, :]

    full = enc(hist, hp=True)
    mx.eval(full)

    state = enc.init_decode_state(dtype=mx.float32)
    enc.prefill(prompt, state, hp=True)
    worst = 0.0
    for k in range(n_gen):
        start = (n_prompt + k) * p
        tok = enc.decode_patch(hist[:, start:start + p, :], state, hp=True)
        mx.eval(tok)
        ref = full[:, n_prompt + k:n_prompt + k + 1, :]
        worst = max(worst, float(mx.abs(tok - ref).max()))
    print(f"\n[8a] real-weights streaming vs recompute-full: worst max|Δ| over {n_gen} patches = {worst:.5f}")
    assert worst <= 2e-2, f"real-weights streaming diverged: max|Δ|={worst:.4f}"


@pytest.mark.skipif(not _CORE.exists(), reason="core weights absent")
def test_streaming_equals_recompute_real_weights_bf16():
    """Runtime-path parity: real encoder in bf16 (hp=False), streaming == recompute per token.

    The model's _reencode_and_step runs the encoder in bf16 (hp=False). 8a proves fp32
    bit-exact; this proves the bf16 RUNTIME path matches per-token (cos >= 0.9999), so any
    end-to-end waveform divergence is AR sampling chaos (a sub-percent per-token reduction-
    order difference compounding over the autoregressive loop), NOT a streaming bug.
    """
    from dots_tts_mlx.loader import load_semantic_encoder

    mx.set_memory_limit(int(45 * (1 << 30)))
    enc = load_semantic_encoder(str(_CORE), dtype=mx.bfloat16)
    p = enc.patch_size
    n = 40
    rng = np.random.default_rng(2025)
    hist = mx.array(rng.standard_normal((1, n * p, 128)).astype(np.float32)).astype(mx.bfloat16)

    full = enc(hist)  # bf16, hp=False (the model's recompute path)
    mx.eval(full)

    state = enc.init_decode_state(dtype=mx.bfloat16)
    worst_cos = 1.0
    worst_abs = 0.0
    for k in range(n):
        tok = enc.decode_patch(hist[:, k * p:(k + 1) * p, :], state)  # bf16, hp=False
        mx.eval(tok)
        ref = full[:, k:k + 1, :]
        a = np.asarray(tok.astype(mx.float32)).ravel()
        b = np.asarray(ref.astype(mx.float32)).ravel()
        worst_cos = min(worst_cos, float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)))
        worst_abs = max(worst_abs, float(np.max(np.abs(a - b))))
    print(f"\n[8a-bf16] runtime bf16 streaming vs recompute: min per-token cos={worst_cos:.5f} "
          f"max|Δ|={worst_abs:.5f}")
    assert worst_cos >= 0.9999, f"bf16 per-token cos {worst_cos:.5f} < 0.9999 (streaming bug, not chaos)"
