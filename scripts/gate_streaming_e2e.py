"""Streaming patch-decode: speed decomposition + end-to-end sanity (real weights).

Parity is proven at the MODULE level by tests/test_streaming_decode.py
(test_streaming_equals_recompute_real_weights[_bf16]: fp32 bit-exact, bf16 per-token
cos >= 0.9999). This script measures WHERE long-gen time actually goes and confirms the
streaming path generates valid audio end-to-end.

Key finding (recorded in docs/research/88): streaming makes the patch encoder O(n) (flat
per-patch) vs recompute-full's O(n^2)-total, BUT the encoder is only a few percent of total
generate time — the dominant cost is the FM denoise over the growing fm_seq. So end-to-end
speedup is ~1x; the encoder was not the bottleneck.

NOTE: in bf16 the AR sampler is chaotic — a sub-percent per-token reduction-order difference
(batched recompute vs incremental streaming) compounds over the autoregressive loop into a
different-but-valid utterance. So streaming vs recompute waveforms DIVERGE in bf16 by design;
this is NOT a parity failure (see the fp32 bit-exact module gate). We therefore do not assert
waveform equality here — only sane audio + the speed decomposition.

Run (ONE GPU job at a time, SIGTERM-only):
    cd ~/dots-tts-mlx-streaming && uv run python scripts/gate_streaming_e2e.py
"""
from __future__ import annotations

import time

import mlx.core as mx
import numpy as np

mx.set_memory_limit(int(45 * (1 << 30)))

from dots_tts_mlx.loader import from_pretrained, load_semantic_encoder  # noqa: E402

MODEL = "weights/dots_tts_mlx"
CORE = "weights/dots_tts_mlx/core.safetensors"

LONG_TEXT = (
    "The quick brown fox jumps over the lazy dog while the morning sun rises slowly over "
    "the quiet valley, and a gentle breeze carries the scent of pine across the open field "
    "as travelers begin their long journey toward the distant mountains far to the north, "
    "where rivers wind through ancient forests and the air grows cold and thin near the peaks."
)


def _encoder_decomposition():
    print("=== encoder-only cost: recompute-full (O(n^2) total) vs streaming (O(n)) ===")
    enc = load_semantic_encoder(CORE, dtype=mx.bfloat16)
    p = enc.patch_size
    for n in [50, 100, 200, 300]:
        rng = np.random.default_rng(1)
        patches = [mx.array(rng.standard_normal((1, p, 128)).astype(np.float32)).astype(mx.bfloat16)
                   for _ in range(n)]
        t0 = time.time()
        hist = []
        for k in range(n):
            hist.append(patches[k])
            _ = enc(mx.concatenate(hist, axis=1))[:, -1:, :]
            mx.eval(_)
        t_rec = time.time() - t0
        state = enc.init_decode_state(dtype=mx.bfloat16)
        t0 = time.time()
        for k in range(n):
            mx.eval(enc.decode_patch(patches[k], state))
        t_str = time.time() - t0
        print(f"  n={n:4d}  recompute={t_rec:6.2f}s ({t_rec/n*1000:5.1f}ms/patch)  "
              f"streaming={t_str:6.2f}s ({t_str/n*1000:4.1f}ms/patch flat)  "
              f"encoder-speedup={t_rec/max(t_str,1e-6):4.1f}x")


def _run(model, text, *, streaming):
    t0 = time.time()
    out = model.generate(text=text, num_steps=10, guidance_scale=1.2, seed=42,
                         max_generate_length=500, streaming_decode=streaming)
    mx.eval(out["audio"])
    wav = np.asarray(out["audio"][0].astype(mx.float32))
    return wav, int(out["num_patches"]), time.time() - t0


def main() -> int:
    _encoder_decomposition()

    print("\n=== full generate A/B: streaming vs recompute (speed + sanity) ===")
    model = from_pretrained(MODEL, dtype=mx.bfloat16).model
    wav_s, n_s, t_s = _run(model, LONG_TEXT, streaming=True)
    wav_r, n_r, t_r = _run(model, LONG_TEXT, streaming=False)
    print(f"  streaming:  {n_s} patches  {t_s:.1f}s  ({t_s/max(n_s,1)*1000:.0f}ms/patch)  "
          f"audio std={wav_s.std():.3f} len={wav_s.size}")
    print(f"  recompute:  {n_r} patches  {t_r:.1f}s  ({t_r/max(n_r,1)*1000:.0f}ms/patch)  "
          f"audio std={wav_r.std():.3f} len={wav_r.size}")
    print(f"  end-to-end speedup={t_r/max(t_s,1e-6):.2f}x  "
          f"(encoder is a small fraction of per-patch cost; FM denoise dominates)")

    # Sanity: streaming must produce valid, finite, non-trivial audio (NOT waveform equality —
    # bf16 AR divergence is expected; module-level parity is gated in the pytest suite).
    assert np.isfinite(wav_s).all(), "streaming audio has NaN/Inf"
    assert 0.001 < wav_s.std() < 10.0, f"streaming audio std {wav_s.std():.4f} out of sane range"
    assert n_s > 20, f"streaming produced too few patches ({n_s})"
    print("\nSTREAMING E2E SANITY PASSED (valid audio; module-level parity gated in pytest)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
