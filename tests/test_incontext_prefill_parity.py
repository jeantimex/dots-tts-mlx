import pathlib

import mlx.core as mx
import numpy as np
import pytest

mx.set_memory_limit(int(45 * (1 << 30)))

W = pathlib.Path("weights/dots_tts_mlx")
FIX = pathlib.Path("tests/fixtures/dots_tts/incontext_prefill.npz")
REF = "/Users/shraey/.superset/worktrees/longcat/mlx/outputs/xdub_len/voice_short6s.wav"
REF_TEXT = "I used to live in the cloud, now I'm running on your Mac, avatar video, dubbing."

# Same-framework (MLX vs MLX) tolerance — tight.
TOL = 1e-3
# Cross-framework (MLX vs upstream PyTorch) fp32 max-abs over the deep conv+transformer
# patch encoder: measured worst 1.4e-3 (mean 6e-5, median 5e-5, p99 2.7e-4; <0.01% of
# ~56k elements exceed 1e-3) — accumulation-order noise, not a scale error. The bug signal
# (normalized input) is ~0.64, ~128x this bound, so the gate keeps full discriminating power.
ORACLE_TOL = 5e-3


@pytest.mark.skipif(not (W / "core.safetensors").exists() or not FIX.exists(),
                    reason="needs real weights + the upstream in-context fixture")
def test_patch_encoder_matches_oracle_on_denorm():
    """Invariant: the MLX patch encoder reproduces upstream's embeddings GIVEN the correct
    (denormalized) input. Confirms NORMALIZED input (the bug) is materially different.
    Passes pre- and post-fix (cross-framework fp32; see ORACLE_TOL)."""
    from dots_tts_mlx.model import DotsTtsModel

    d = np.load(FIX, allow_pickle=True)
    model = DotsTtsModel.from_pretrained(W, dtype=mx.float32)
    denorm = mx.array(d["prompt_denorm_latents"].astype(np.float32))
    emb = np.asarray(model.patch_encoder(denorm).astype(mx.float32))
    ref = d["prompt_patch_emb"].astype(np.float32)
    worst = float(np.max(np.abs(emb - ref)))
    print(f"\n[oracle] patch_encoder(denorm) worst max|delta| = {worst:.6f} (tol {ORACLE_TOL})")
    assert emb.shape == ref.shape, (emb.shape, ref.shape)
    assert worst < ORACLE_TOL, f"encoder diverged from upstream on denorm input: {worst:.5f}"
    wrong = np.asarray(model.patch_encoder(model.io.normalize(denorm)).astype(mx.float32))
    assert float(np.max(np.abs(wrong - ref))) > 10 * ORACLE_TOL, "normalized input should diverge"


@pytest.mark.skipif(not (W / "core.safetensors").exists(),
                    reason="needs real weights (enroll runs the model)")
def test_enroll_caches_denorm_patch_emb():
    """BUG GATE: the cached patch_emb must equal patch_encoder applied to the profile's
    stored DENORMALIZED latents (same framework). Pre-fix FAILS (~0.81, enroll used
    normalized); post-fix PASSES (< TOL)."""
    from dots_tts_mlx.model import DotsTtsModel

    model = DotsTtsModel.from_pretrained(W, dtype=mx.float32)
    prof = model.enroll(REF, REF_TEXT)
    S = int(prof.prompt_patch_count)
    denorm = prof.prompt_denorm_latents[:, : S * model.patch_size]
    recomputed = np.asarray(model.patch_encoder(denorm).astype(mx.float32))
    cached = np.asarray(prof.patch_emb.astype(mx.float32))
    worst = float(np.max(np.abs(recomputed - cached)))
    print(f"\n[enroll] cached patch_emb vs patch_encoder(denorm) worst max|delta| = {worst:.6f}")
    assert worst < TOL, (
        f"cached patch_emb is NOT from denormalized latents (max|delta|={worst:.5f}) -- the bug"
    )
