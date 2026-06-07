"""HF -> MLX weight converter for dots.tts (OFFLINE build tool; runs under the torch oracle venv).

This is the ONLY module in the package allowed to import torch. It is a one-time
build step, NOT part of the MLX runtime — never import it from runtime code.

What it does:
  1. Loads the three source safetensors (``model`` / ``vocoder`` / ``speaker_encoder``).
  2. Folds vocoder weight_norm (the *only* file with it): for each ``*.weight_v`` it finds
     the matching ``*.weight_g`` and emits ``w = g * v / ||v||`` (norm over all dims except
     dim 0, PyTorch's ``dim=0`` default), then drops the ``_g``/``_v`` pair. Expect 80 pairs.
  3. Passes speaker BN buffers (running_mean/var/weight/bias/num_batches_tracked) through unchanged.
  4. Keeps un-prefixed (submodule-level) key names; does NOT transpose conv weights
     (the MLX loader transposes to MLX layout at load time). Stores raw torch layout, fp32.
  5. Writes ``{core,vocoder,speaker}.safetensors`` as fp32 numpy (framework-agnostic;
     ``mx.load`` reads them). Alias-free filter tensors are kept fp32 as-is.
  6. Extracts ``latent_stats.pt`` -> ``latent_stats.npz`` (mean/var, fp32).
  7. Copies tokenizer files into ``<out>/tokenizer/``.

Faithful fp32 master weights (the oracle runs fp32) -> tight downstream parity. The MLX
loader casts to bf16 for the runtime; parity tests load fp32.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file as np_save_file
from safetensors.torch import load_file as torch_load_file

# Source -> output safetensors filename map (un-prefixed keys preserved).
_FILE_MAP = {
    "model.safetensors": "core.safetensors",
    "vocoder.safetensors": "vocoder.safetensors",
    "speaker_encoder.safetensors": "speaker.safetensors",
}

_TOKENIZER_FILES = (
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
)

# Config JSONs copied alongside the weights so the MLX loader's
# ``ModelConfig.from_checkpoint(out)`` resolves against the converted dir.
_CONFIG_FILES = ("config.json", "llm_config.json")


def _to_fp32_numpy(t: torch.Tensor) -> np.ndarray:
    """Cast a torch tensor to a contiguous fp32 numpy array (CPU)."""
    return t.detach().to(torch.float32).contiguous().cpu().numpy()


def _fold_weight_norm(state: dict[str, torch.Tensor]) -> tuple[dict[str, np.ndarray], int]:
    """Fold PyTorch weight_norm (dim=0) and return (fp32 numpy state, n_pairs_folded).

    For each ``<prefix>.weight_v`` (shape ``[out, in, k]``) with matching
    ``<prefix>.weight_g`` (shape ``[out, 1, 1]``):
        norm = v.flatten(1).norm(dim=1).view(-1, 1, 1)   # L2 over all dims except 0
        weight = g * v / norm
    Emit it as ``<prefix>.weight`` and drop the ``_g``/``_v`` pair. All other tensors
    pass through unchanged. Output is fp32 numpy.
    """
    out: dict[str, np.ndarray] = {}
    n_folded = 0
    handled: set[str] = set()

    for key, v in state.items():
        if key.endswith(".weight_v"):
            prefix = key[: -len(".weight_v")]
            g_key = prefix + ".weight_g"
            if g_key not in state:
                raise KeyError(f"weight_v without matching weight_g: {key}")
            g = state[g_key].to(torch.float32)
            vf = v.to(torch.float32)
            # L2 norm over all dims except dim 0 (PyTorch weight_norm dim=0 default).
            norm = vf.reshape(vf.shape[0], -1).norm(dim=1)
            # Reshape norm to broadcast against v: [out, 1, 1, ...].
            norm = norm.reshape([vf.shape[0]] + [1] * (vf.ndim - 1))
            weight = g * vf / norm
            out[prefix + ".weight"] = _to_fp32_numpy(weight)
            handled.add(key)
            handled.add(g_key)
            n_folded += 1

    for key, t in state.items():
        if key in handled or key.endswith(".weight_g"):
            continue
        out[key] = _to_fp32_numpy(t)

    return out, n_folded


def _passthrough_fp32(state: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    """Cast every tensor to fp32 numpy, preserving keys (no folding, no transpose)."""
    return {k: _to_fp32_numpy(v) for k, v in state.items()}


def convert_checkpoint(src: Path, out: Path) -> dict[str, object]:
    """Convert the dots.tts checkpoint at ``src`` into MLX-ready artifacts under ``out``.

    Returns a summary dict (file sizes, weight_norm pairs folded) for reporting.
    """
    src = Path(src)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {}

    # --- safetensors (loaded/converted/freed one file at a time to bound memory) ---
    for src_name, out_name in _FILE_MAP.items():
        src_path = src / src_name
        if not src_path.exists():
            raise FileNotFoundError(f"missing source weight file: {src_path}")
        state = torch_load_file(str(src_path))

        if src_name == "vocoder.safetensors":
            np_state, n_folded = _fold_weight_norm(state)
            summary["weight_norm_pairs_folded"] = n_folded
        else:
            np_state = _passthrough_fp32(state)

        del state  # free torch tensors before the next (large) file

        out_path = out / out_name
        np_save_file(np_state, str(out_path))
        del np_state
        summary[out_name] = out_path.stat().st_size

    # --- latent_stats.pt -> latent_stats.npz ---
    stats = torch.load(str(src / "latent_stats.pt"), weights_only=False)
    mean = stats["mean"]
    var = stats["var"]
    mean = np.asarray(mean.detach().cpu().numpy() if torch.is_tensor(mean) else mean, dtype=np.float32)
    var = np.asarray(var.detach().cpu().numpy() if torch.is_tensor(var) else var, dtype=np.float32)
    stats_path = out / "latent_stats.npz"
    np.savez(str(stats_path), mean=mean, var=var)
    summary["latent_stats.npz"] = stats_path.stat().st_size

    # --- config JSONs (so the loader can resolve ModelConfig from the converted dir) ---
    copied_cfg = []
    for name in _CONFIG_FILES:
        sp = src / name
        if not sp.exists():
            raise FileNotFoundError(f"missing config file: {sp}")
        shutil.copy2(sp, out / name)
        copied_cfg.append(name)
    summary["config_files"] = copied_cfg

    # --- tokenizer files ---
    tok_dir = out / "tokenizer"
    tok_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in _TOKENIZER_FILES:
        sp = src / name
        if sp.exists():
            shutil.copy2(sp, tok_dir / name)
            copied.append(name)
    summary["tokenizer_files"] = copied

    return summary


def _main() -> None:
    ap = argparse.ArgumentParser(description="Convert dots.tts checkpoint -> MLX fp32 artifacts.")
    ap.add_argument("--src", type=Path, default=Path("weights/dots_tts_src/dots.tts-soar"))
    ap.add_argument("--out", type=Path, default=Path("weights/dots_tts_mlx"))
    args = ap.parse_args()

    summary = convert_checkpoint(args.src, args.out)
    print("dots.tts convert complete:")
    for k, v in summary.items():
        if isinstance(v, int):
            print(f"  {k}: {v / (1 << 20):.1f} MiB")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    _main()
