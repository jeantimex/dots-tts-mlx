"""Dump the upstream (torch) MeanFlow NFE=4 oracle fixture for the MLX parity gate.

Run under the torch oracle venv (NOT the MLX runtime):

    ~/.venvs/dots-oracle/bin/python scripts/gen_meanflow_oracle_fixture.py

It loads the raw upstream MeanFlow checkpoint via ``DotsTtsModel.from_pretrained``
(confirming ``core.mode == "meanflow"``), builds a fixed, seeded parity input
(``input_sequence``, ``g_cond``, ``noise``) plus the EXACT attn_mask / pos_ids that
upstream's single-patch FM decode uses for that sequence length (replicated from
``DotsTtsModel._build_fm_attn_mask`` / ``_build_fm_pos_ids`` — a deterministic boolean
mask, ``True == attend``, identical to the MLX-side builder), then runs the upstream
NFE=4 MeanFlow loop (``_meanflow_step_fm`` semantics, but with OUR injected noise
instead of a fresh ``randn``) and saves everything to
``tests/fixtures/dit/meanflow_oracle.npz`` (fp32).

torch is imported ONLY here; the MLX gate (``gate_meanflow_parity.py``) never touches it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Upstream source on path BEFORE importing dots_tts.
sys.path.insert(0, str(Path.home() / "dots.tts" / "src"))

import torch  # noqa: E402

from dots_tts.models.dots_tts.model import DotsTtsModel  # noqa: E402

CKPT = Path("weights/dots_tts_src/dots.tts-mf")
OUT = Path("tests/fixtures/dit/meanflow_oracle.npz")

# Parity input geometry. fm_seq_len = conditioning/history tokens; the latent block
# is one patch of ``patch_size`` slots appended at the tail. Total L = fm_seq_len + p.
FM_SEQ_LEN = 20
NFE = 4
SEED = 0


def build_fm_attn_mask(fm_seq_len: int, patch_size: int, hidden_patch_size: int) -> torch.Tensor:
    """Replicate upstream ``DotsTtsModel._build_fm_attn_mask`` (single-patch decode).

    Boolean ``[1, L, L]`` mask, ``True == attend``. L = fm_seq_len + patch_size,
    latent_start = fm_seq_len (no history-bucket padding here). Mirrors model.py.
    """
    total = fm_seq_len + patch_size
    latent_start = total - patch_size  # == fm_seq_len
    mask = torch.zeros((1, total, total), dtype=torch.bool)
    block_start = fm_seq_len - hidden_patch_size
    if block_start > 0:
        causal = torch.ones((block_start, block_start), dtype=torch.bool).triu(1).logical_not()
        mask[:, :block_start, :block_start] = causal
    mask[:, block_start:fm_seq_len, :fm_seq_len] = True
    mask[:, block_start:fm_seq_len, latent_start:] = True
    mask[:, latent_start:, :fm_seq_len] = True
    mask[:, latent_start:, latent_start:] = True
    return mask


def build_fm_pos_ids(fm_seq_len: int, patch_size: int) -> torch.Tensor:
    """Replicate upstream ``DotsTtsModel._build_fm_pos_ids`` -> float ``[1, L]``."""
    total = fm_seq_len + patch_size
    latent_start = total - patch_size
    pos = torch.zeros((1, total), dtype=torch.float32)
    pos[:, :fm_seq_len] = torch.arange(fm_seq_len, dtype=torch.float32)
    pos[:, latent_start:] = torch.arange(
        fm_seq_len, fm_seq_len + patch_size, dtype=torch.float32
    )
    return pos


def main() -> int:
    if not CKPT.exists():
        raise FileNotFoundError(f"MeanFlow checkpoint not found: {CKPT}")

    model = DotsTtsModel.from_pretrained(CKPT)
    core = model.core
    assert core.mode == "meanflow", f"expected meanflow core, got mode={core.mode!r}"

    patch_size = int(core.latent_patch_size)
    hidden_patch_size = int(core.hidden_patch_size)
    latent_dim = int(core.latent_dim)
    hidden_size = int(core.velocity_field_predictor.input_layer.in_features)

    total_len = FM_SEQ_LEN + patch_size

    gen = torch.Generator().manual_seed(SEED)
    input_sequence = torch.randn(
        (1, total_len, hidden_size), generator=gen, dtype=torch.float32
    )
    g_cond = torch.randn((1, hidden_size), generator=gen, dtype=torch.float32)
    noise = torch.randn(
        (1, patch_size, latent_dim), generator=gen, dtype=torch.float32
    )

    attn_mask = build_fm_attn_mask(FM_SEQ_LEN, patch_size, hidden_patch_size)
    pos_ids = build_fm_pos_ids(FM_SEQ_LEN, patch_size)

    # NFE=4 MeanFlow loop, mirroring core._meanflow_step_fm but with OUR injected
    # noise (do NOT let it draw a fresh randn). Uniform schedule on [0, 1].
    dtype = input_sequence.dtype
    z = noise.clone()
    times = torch.linspace(0.0, 1.0, NFE + 1, dtype=dtype)
    with torch.no_grad():
        for step in range(NFE):
            t = times[step].expand(1)
            dt = (times[step + 1] - times[step]).expand(1)
            z = core.meanflow_solver_step(
                z,
                t=t,
                dt=dt,
                input_sequence=input_sequence,
                attn_mask=attn_mask,
                pos_ids=pos_ids,
                patch_size=patch_size,
                g_cond=g_cond,
            ).clone()
    z_out = z

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT,
        input_sequence=input_sequence.numpy().astype(np.float32),
        g_cond=g_cond.numpy().astype(np.float32),
        noise=noise.numpy().astype(np.float32),
        attn_mask=attn_mask.numpy(),  # bool [1, L, L]
        pos_ids=pos_ids.numpy().astype(np.float32),
        z_out=z_out.numpy().astype(np.float32),
        num_steps=np.int64(NFE),
        patch_size=np.int64(patch_size),
        latent_dim=np.int64(latent_dim),
        fm_seq_len=np.int64(FM_SEQ_LEN),
    )
    print(
        f"wrote {OUT}: L={total_len} fm_seq_len={FM_SEQ_LEN} patch_size={patch_size} "
        f"latent_dim={latent_dim} hidden={hidden_size} nfe={NFE} mode={core.mode}"
    )
    print(f"z_out stats: mean={float(z_out.mean()):.5f} std={float(z_out.std()):.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
