import mlx.core as mx
import numpy as np
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
