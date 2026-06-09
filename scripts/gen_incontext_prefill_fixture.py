"""Upstream (PyTorch) oracle fixture for in-context prompt embeddings.

This script runs ONLY under the oracle env (~/.venvs/dots-oracle), which has
torch 2.8.0 + the upstream ``dots_tts`` package (from ~/dots.tts/src). It is the
ONLY torch file in this package; ``dots_tts_mlx`` must never import torch.

It loads the UPSTREAM DotsTts model on a FIXED reference audio + transcript with
a FIXED seed, runs the prompt-conditioning + patch-encoder prefill path exactly
as upstream generation does, and dumps the ground-truth arrays that later MLX
parity tasks compare against.

Run:
    cd ~/dots-tts-mlx-prompt-scale
    ~/.venvs/dots-oracle/bin/python scripts/gen_incontext_prefill_fixture.py

Dumped arrays (np.savez -> tests/fixtures/dots_tts/incontext_prefill.npz):
    prompt_denorm_latents : float32 [1, S*patch_size, 128]
        The DENORMALIZED sampled prompt latents that are the patch-encoder
        INPUT. Upstream: ``io_helper.sample_from_latent(...)`` (denorm), then
        truncated by one patch (``[:, :-patch_size]``). This is exactly the
        ``prompt_latents`` argument passed to ``_prefill_prompt_latents``.
        NOTE the in-context bug under test: the MLX in-context path feeds the
        patch encoder NORMALIZED latents; upstream (this fixture) feeds these
        DENORMALIZED ones.
    prompt_patch_emb : float32 [1, S, 1536]
        The patch-encoder OUTPUT -- what ``patch_encoder.prefill`` returns and
        gets scattered into the LLM. ``1536`` == ``core.llm_hidden_size``.
    prompt_text : str
        The fixed reference transcript (REF_TEXT below).
    prompt_patch_count : int
        S, the number of output patches in ``prompt_patch_emb``.

Dimension note for the verifier: this checkpoint has ``patch_size=4`` and
``in_ds_rate=2`` -> ``out_ds_rate = patch_size // in_ds_rate = 2``. The patch
encoder emits ``(L // patch_size) * out_ds_rate`` intermediate tokens, which
``_project_embeddings`` then groups by ``out_ds_rate`` -> output count
``S = L // patch_size``. So ``prompt_denorm_latents.shape[1] == S * patch_size``
== S * 4 == 148 for this checkpoint (S = 37), NOT S * 16 and NOT S * 2.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

# Defensive: ensure the upstream package is importable even if the env's
# site-packages does not already expose it.
_UPSTREAM_SRC = Path("~/dots.tts/src").expanduser()
if _UPSTREAM_SRC.is_dir() and str(_UPSTREAM_SRC) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM_SRC))

from dots_tts.models.dots_tts.model import _GenerateState  # noqa: E402
from dots_tts.runtime import DotsTtsRuntime  # noqa: E402

CHECKPOINT = (
    "/Users/shraey/.superset/worktrees/longcat/mlx/weights/"
    "dots_tts_src/dots.tts-soar"
)
REF = (
    "/Users/shraey/.superset/worktrees/longcat/mlx/outputs/"
    "xdub_len/voice_short6s.wav"
)
REF_TEXT = (
    "I used to live in the cloud, now I'm running on your Mac, "
    "avatar video, dubbing."
)
SPEAKER_SCALE = 1.5
OUT_PATH = Path("tests/fixtures/dots_tts/incontext_prefill.npz")


def main() -> None:
    torch.manual_seed(0)

    runtime = DotsTtsRuntime.from_pretrained(
        CHECKPOINT,
        precision="float32",
        optimize=False,
    )
    model = runtime.model
    model.eval()

    # Load the prompt audio exactly as upstream generation does (librosa load +
    # trim + resample to the model sample rate).
    prompt_audio = runtime._load_prompt_audio(REF)

    # Re-seed immediately before the stochastic sampling so the result is
    # independent of any RNG consumed during model load. sample_from_latent uses
    # torch.randn_like, so this is the determinism-critical seed.
    torch.manual_seed(0)
    with torch.no_grad():
        conditioning = model._prepare_prompt_conditioning(
            prompt_audio,
            use_prompt_prefill=True,
            speaker_scale=SPEAKER_SCALE,
        )

        prompt_latents = conditioning.prompt_latents  # DENORM, patch-encoder input
        if prompt_latents is None:
            raise RuntimeError(
                "prompt_latents is None -- prompt conditioning did not run the "
                "prefill path. Check use_prompt_prefill / prompt_audio."
            )

        state = _GenerateState()
        prompt_patch_emb = model._prefill_prompt_latents(
            prompt_latents,
            state=state,
        )

    if prompt_patch_emb is None:
        raise RuntimeError("_prefill_prompt_latents returned None.")

    prompt_denorm_latents = prompt_latents.detach().to(torch.float32).cpu().numpy()
    prompt_patch_emb = prompt_patch_emb.detach().to(torch.float32).cpu().numpy()
    prompt_patch_count = int(prompt_patch_emb.shape[1])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_PATH,
        prompt_denorm_latents=prompt_denorm_latents,
        prompt_patch_emb=prompt_patch_emb,
        prompt_text=np.array(REF_TEXT),
        prompt_patch_count=np.array(prompt_patch_count, dtype=np.int64),
    )

    print(
        f"WROTE {OUT_PATH.resolve()} S={prompt_patch_count} "
        f"emb={prompt_patch_emb.shape} denorm={prompt_denorm_latents.shape}"
    )


if __name__ == "__main__":
    main()
