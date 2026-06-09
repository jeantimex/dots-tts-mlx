"""MLX MeanFlow parity gate: ``FlowSolver.meanflow_sample`` vs the torch oracle.

Pure-MLX (NO torch). Run under the MLX runtime venv:

    cd ~/dots-tts-mlx-meanflow && uv run python scripts/gate_meanflow_parity.py

Loads the oracle fixture (``tests/fixtures/dit/meanflow_oracle.npz``, dumped by
``gen_meanflow_oracle_fixture.py``) + the converted MeanFlow MLX weights
(``weights/dots_tts_mlx_mf/core.safetensors``), runs the NFE=4 MeanFlow sampler on the
IDENTICAL inputs (the upstream boolean attn_mask is fed straight to the MLX DiT — both
frameworks read ``True == attend`` and convert to a 0/-inf additive bias the same way),
and asserts ``cosine >= 0.999 and max-abs <= 0.6`` vs the upstream ``z_out``.

Mirrors the FM gate (``tests/test_flow_solver.py``): same cosine-dominant gate, same
"feed upstream's mask verbatim" convention. Exits 0 on PASS, 1 on FAIL.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import mlx.core as mx

# Memory guard BEFORE any heavy allocation (project hard ceiling 45 GB).
mx.set_memory_limit(int(45 * (1 << 30)))

from dots_tts_mlx.dit import FlowSolver  # noqa: E402
from dots_tts_mlx.loader import load_coordinate_proj, load_dit  # noqa: E402

FIX = Path("tests/fixtures/dit/meanflow_oracle.npz")
W = Path("weights/dots_tts_mlx_mf/core.safetensors")


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main() -> int:
    if not FIX.exists():
        print(f"FAIL: fixture missing: {FIX} (run gen_meanflow_oracle_fixture.py)")
        return 1
    if not W.exists():
        print(f"FAIL: weights missing: {W}")
        return 1

    d = np.load(FIX)
    patch_size = int(d["patch_size"])
    latent_dim = int(d["latent_dim"])
    num_steps = int(d["num_steps"])

    dit = load_dit(str(W), dtype=mx.float32)
    coord = load_coordinate_proj(str(W), dtype=mx.float32)
    solver = FlowSolver(dit, coord, latent_dim=latent_dim)

    # Feed upstream's exact arrays straight in. The bool attn_mask is consumed
    # identically by the MLX MultiHeadAttention (True -> attend, else -inf bias).
    z = solver.meanflow_sample(
        input_sequence=mx.array(d["input_sequence"]),
        attn_mask=mx.array(d["attn_mask"]),
        pos_ids=mx.array(d["pos_ids"]),
        g_cond=mx.array(d["g_cond"]),
        num_steps=num_steps,
        patch_size=patch_size,
        noise=mx.array(d["noise"]),
    )
    out = np.asarray(z.astype(mx.float32))
    ref = d["z_out"]

    cosine = _cos(out, ref)
    maxabs = float(np.max(np.abs(out - ref)))
    print(f"meanflow parity: cosine={cosine:.6f} max-abs={maxabs:.4f}")
    print(
        f"(L={d['input_sequence'].shape[1]} fm_seq_len={int(d['fm_seq_len'])} "
        f"patch_size={patch_size} latent_dim={latent_dim} nfe={num_steps})"
    )

    if cosine >= 0.999 and maxabs <= 0.6:
        print("PASS")
        return 0
    print("FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
