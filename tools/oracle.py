#!/usr/bin/env python3
"""dots.tts PyTorch parity oracle (DEV-BOX ONLY).

This script MAY import torch / dots_tts. It is used to enumerate checkpoint
weight keys and (in later tasks) dump reference fixtures that the pure-MLX
runtime is gated against. It is never part of the shipped MLX runtime.

Requires the ``[oracle]`` extra (torch / transformers / torchdiffeq /
safetensors / librosa / torchaudio) AND the upstream ``dots_tts`` package, which
is NOT on PyPI — install it separately from the source repo:
    pip install -e /path/to/dots.tts
    # or: pip install "git+https://github.com/rednote-hilab/dots.tts"

Run from the repo root, e.g.:
    python tools/oracle.py keys --src weights/dots_tts_src/dots.tts-soar

Subcommands:
    keys   Enumerate weight keys from all three safetensors files into a
           manifest, plus a "resolved" block answering config-vs-code
           ambiguities that later MLX-port tasks depend on (AUTHORITATIVE).

Later tasks will add dump-* subcommands (e.g. dump-encoder, dump-dit,
dump-vocoder) to this same file.
"""

from __future__ import annotations

import argparse
import json
import re
from functools import lru_cache
from pathlib import Path

from safetensors import safe_open

# The three safetensors files shipped in the rednote-hilab/dots.tts-soar repo.
SAFETENSORS_FILES = {
    "model": "model.safetensors",
    "vocoder": "vocoder.safetensors",
    "speaker": "speaker_encoder.safetensors",
}


def add_common_args(p: argparse.ArgumentParser) -> None:
    """Add the shared ``--src`` checkpoint-dir argument to a subparser."""
    p.add_argument(
        "--src",
        required=True,
        help="path to the dots.tts-soar checkpoint directory",
    )


@lru_cache(maxsize=2)
def _load_audiovae(src: str):
    """Instantiate the torch ``AudioVAE`` from the checkpoint and load decode weights.

    Memoized so repeated dump subcommands reuse the same eval()-mode, weight_norm-
    folded model. Imports torch + dots_tts lazily (oracle venv only).
    """
    import torch
    from safetensors.torch import load_file as torch_load_file

    from dots_tts.modules.vocoder.bigvgan import AudioVAE
    from dots_tts.modules.vocoder.config import AudioVAEConfig

    src_path = Path(src)
    with open(src_path / "config.json") as f:
        cfg = json.load(f)
    vae_cfg = AudioVAEConfig(**cfg["vocoder"])

    audiovae = AudioVAE(vae_cfg).eval()
    # The shipped checkpoint's DECODER weights are already weight_norm-folded
    # (plain ``.weight``); only the encoder retains weight_g/weight_v. Fold the
    # decoder's weight_norm first so its module params match the checkpoint, then
    # load non-strictly (encoder _g/_v keys are irrelevant to the decode path).
    audiovae.remove_weight_norm()
    state = torch_load_file(str(src_path / "vocoder.safetensors"))
    missing, unexpected = audiovae.load_state_dict(state, strict=False)
    decode_prefixes = ("post_proj.", "dec_mi_layer.", "decoder.")
    decode_missing = [k for k in missing if k.startswith(decode_prefixes)]
    if decode_missing:
        raise RuntimeError(f"decode weights missing from checkpoint: {decode_missing}")
    for param in audiovae.parameters():
        param.requires_grad = False
    torch.set_grad_enabled(False)
    return audiovae


@lru_cache(maxsize=2)
def _load_campplus(src: str):
    """Instantiate the torch ``CAMPPlus`` speaker encoder + load its weights.

    Loads ``speaker_encoder.safetensors`` (keys carry a ``model.`` prefix from the
    parent ``SpeakerXVectorFeatures`` module) into a bare ``CAMPPlus``, in eval mode
    with grads off. Memoized so repeated dumps reuse the same model.
    """
    import torch
    from safetensors.torch import load_file as torch_load_file

    from dots_tts.modules.speaker.campplus import CAMPPlus
    from dots_tts.modules.speaker.fbank import _SPEAKER_FBANK_N_MELS

    model = CAMPPlus(feat_dim=_SPEAKER_FBANK_N_MELS, embedding_size=512).eval()
    state = torch_load_file(str(Path(src) / "speaker_encoder.safetensors"))
    # The shipped keys are prefixed ``model.`` (the wrapping SpeakerXVectorFeatures
    # holds the CAMPPlus as ``self.model``) and include a ``resample.kernel`` buffer
    # that belongs to the wrapper, not CAMPPlus. Strip the prefix and drop wrapper-
    # only keys before loading into the bare CAMPPlus.
    inner = {
        k[len("model.") :]: v
        for k, v in state.items()
        if k.startswith("model.")
    }
    model.load_state_dict(inner, strict=True)
    for param in model.parameters():
        param.requires_grad = False
    torch.set_grad_enabled(False)
    return model


def _load_keys(src: Path) -> dict[str, dict[str, list[int]]]:
    """Return {section: {key: shape}} for the three checkpoint safetensors."""
    out: dict[str, dict[str, list[int]]] = {}
    for section, filename in SAFETENSORS_FILES.items():
        path = src / filename
        if not path.exists():
            raise FileNotFoundError(f"missing checkpoint file: {path}")
        section_keys: dict[str, list[int]] = {}
        with safe_open(str(path), framework="pt") as f:
            for key in f.keys():
                section_keys[key] = list(f.get_slice(key).get_shape())
        out[section] = section_keys
    return out


def _has(keys: list[str], *substrings: str) -> list[str]:
    """Keys containing ALL of the given substrings (prefix-agnostic).

    The raw checkpoint keys have no ``core.`` prefix (e.g. the DiT key is
    ``velocity_field_predictor.blocks.0.attn.q_norm.weight``), so we match on
    substrings rather than exact prefixed paths.
    """
    return [k for k in keys if all(s in k for s in substrings)]


def _resolve(manifest: dict[str, dict[str, list[int]]]) -> dict[str, object]:
    """Answer the config-vs-code ambiguities the MLX-port tasks depend on.

    These are AUTHORITATIVE: they are read from the actual checkpoint, not
    inferred from config. Substring matching is used because the raw keys
    carry no ``core.`` prefix.
    """
    mkeys = list(manifest["model"].keys())
    vkeys = list(manifest["vocoder"].keys())

    # DiT attention q/k RMSNorm presence (expected True).
    dit_qk_norm = _has(
        mkeys, "velocity_field_predictor.blocks.0.attn.q_norm"
    )
    dit_qk_norm_present = bool(dit_qk_norm)

    # Patch-encoder attention q/k norm presence (expected False).
    encoder_qk_norm_present = bool(
        _has(mkeys, "patch_encoder.encoder.layers.0.attn.q_norm")
    )

    # Distinct patch-encoder layer indices (expected 24).
    enc_layer_idx: set[int] = set()
    for k in mkeys:
        m = re.search(r"patch_encoder\.encoder\.layers\.(\d+)\.", k)
        if m:
            enc_layer_idx.add(int(m.group(1)))
    encoder_num_layers = len(enc_layer_idx)

    # adaLN affine: a plain norm1.weight means affine LayerNorm; its absence
    # (DiT uses adaLN_modulation instead) means affine-free adaLN (expected
    # False).
    dit_norm1_affine_present = bool(
        _has(mkeys, "velocity_field_predictor.blocks.0.norm1.weight")
    )

    # Norm type from q_norm suffixes: weight-only => RMSNorm, weight+bias =>
    # LayerNorm.
    qn_suffixes = sorted({k.split(".")[-1] for k in dit_qk_norm})
    if qn_suffixes == ["weight"] or qn_suffixes == ["gamma"]:
        dit_norm_type = "RMSNorm"
    elif "bias" in qn_suffixes and "weight" in qn_suffixes:
        dit_norm_type = "LayerNorm"
    elif qn_suffixes:
        dit_norm_type = f"unknown(suffixes={qn_suffixes})"
    else:
        dit_norm_type = "absent"

    # Alias-free (BigVGAN-style) lowpass filter buffers are baked into the
    # checkpoint (expected True) => load them directly, no Kaiser recompute.
    activation_post_filter = _has(vkeys, "activation_post", "filter")
    activations_downsample_filter = _has(
        vkeys, "activations", "downsample", "filter"
    )
    alias_free_filters_present = bool(activation_post_filter) and bool(
        activations_downsample_filter
    )

    # weight_norm residue: raw checkpoint stores weight_g / weight_v
    # (expected True) => must be folded at convert time (Task 1).
    weight_g = [k for k in (mkeys + vkeys) if k.endswith("weight_g")]
    weight_v = [k for k in (mkeys + vkeys) if k.endswith("weight_v")]
    weight_norm_residue = bool(weight_g) and bool(weight_v)

    return {
        "dit_qk_norm_present": dit_qk_norm_present,
        "encoder_qk_norm_present": encoder_qk_norm_present,
        "encoder_num_layers": encoder_num_layers,
        "dit_norm1_affine_present": dit_norm1_affine_present,
        "dit_norm_type": dit_norm_type,
        "alias_free_filters_present": alias_free_filters_present,
        "weight_norm_residue": weight_norm_residue,
        # Supporting evidence (not part of the 7 answers, but useful context).
        "_evidence": {
            "dit_qk_norm_keys": dit_qk_norm,
            "dit_q_norm_suffixes": qn_suffixes,
            "encoder_layer_index_max": max(enc_layer_idx) if enc_layer_idx else None,
            "activation_post_filter_keys": activation_post_filter,
            "activations_downsample_filter_count": len(
                activations_downsample_filter
            ),
            "weight_g_count": len(weight_g),
            "weight_v_count": len(weight_v),
        },
    }


def cmd_keys(args: argparse.Namespace) -> int:
    src = Path(args.src)
    manifest = _load_keys(src)
    resolved = _resolve(manifest)

    out = {**manifest, "resolved": resolved}
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    counts = {k: len(v) for k, v in manifest.items()}
    print(f"wrote {dest} (key counts: {counts})")
    print("resolved:")
    for k, v in resolved.items():
        if k == "_evidence":
            continue
        print(f"  {k}: {v}")
    return 0


def cmd_dump_vocoder_decode(args: argparse.Namespace) -> int:
    """Dump a reference (latent -> 48 kHz waveform) decode fixture for MLX parity."""
    import numpy as np
    import torch

    audiovae = _load_audiovae(str(Path(args.src)))

    torch.manual_seed(0)
    z = torch.randn(1, audiovae.h.latent_dim, args.frames, dtype=torch.float32)
    wav = audiovae.inference_from_latents(z, do_sample=False)  # fp32

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(dest),
        latent=z.numpy(),
        wav=wav.detach().numpy(),
    )
    print(f"wrote {dest} (latent={tuple(z.shape)}, wav={tuple(wav.shape)})")
    return 0


def _load_ref_wav_48k(wav_path: str) -> "np.ndarray":  # noqa: F821
    """Load a reference wav, resample to 48 kHz mono fp32 (torchaudio kaiser).

    Mirrors the runtime's prompt-audio path (``high_quality_resample``). Returns a
    1-D numpy fp32 array at 48 kHz.
    """
    import librosa
    import numpy as np
    import torch
    from dots_tts.utils.audio import high_quality_resample

    audio, sr = librosa.load(wav_path, sr=None, mono=True)
    t = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)  # [1, S]
    t = high_quality_resample(t, orig_sr=sr, target_sr=48000)  # [1, S']
    return t.squeeze(0).contiguous().float().numpy()


def cmd_dump_vae_encode(args: argparse.Namespace) -> int:
    """Dump a reference encode + io_helper parity fixture (48 kHz wav -> latent)."""
    import numpy as np
    import torch

    from dots_tts.models.dots_tts.core import IOHelper

    audiovae = _load_audiovae(str(Path(args.src)))

    wav = _load_ref_wav_48k(args.wav)  # 1-D fp32 @ 48 kHz
    # Trim to ~max_seconds for a manageable fixture / runtime.
    max_samples = int(args.max_seconds * 48000)
    if wav.shape[0] > max_samples:
        wav = wav[:max_samples]

    wav_t = torch.from_numpy(wav).float()  # [S]

    torch.manual_seed(0)
    lat = audiovae.extract_latents(wav_t[None, None], do_sample=False)  # [1, 256, T]
    latent_dim = audiovae.h.latent_dim
    mean, log_std = lat[:, :latent_dim], lat[:, latent_dim:]
    noise = torch.randn_like(mean)  # capture the exact draw for injectable parity
    z = (mean + noise * torch.exp(log_std)).transpose(1, 2)  # [1, T, 128]

    io_helper = IOHelper(str(Path(args.src) / "latent_stats.pt"))
    zn = io_helper.normalize(z)  # [1, T, 128]

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(dest),
        wav=wav.astype(np.float32),
        lat=lat.detach().numpy(),
        noise=noise.detach().numpy(),
        z=z.detach().numpy(),
        zn=zn.detach().numpy(),
    )
    print(
        f"wrote {dest} (wav={tuple(wav.shape)}, lat={tuple(lat.shape)}, "
        f"z={tuple(z.shape)}, zn={tuple(zn.shape)})"
    )
    return 0


def cmd_dump_attention(args: argparse.Namespace) -> int:
    """Dump a backbone MultiHeadAttention + rotary parity fixture for the MLX port.

    Builds ONE upstream ``MultiHeadAttention`` with the DiT settings (hidden=1024,
    heads=16, qkv_bias=False, qk_norm=True, norm_layer="RMSNorm", rotary_bias=True,
    rotary_theta=10000), seeded + fp32 + eval. Runs it for two pos_ids variants
    (contiguous ``arange`` and an offset set that exercises the non-zero positions
    DiT will see) under a 3-D causal bool mask. Also dumps a rotary-only fixture
    (``RotaryEmbedding`` + ``apply_rotary_pos_emb`` on a random ``[1,16,7,64]`` head
    tensor) so the half-half layout can be gated in isolation.

    Weight tensors are saved with a ``w_`` prefix and dots replaced by underscores
    (npz-key-safe): ``q_proj.weight`` -> ``w_q_proj_weight``, etc. The MLX
    ``MultiHeadAttention.load_weights`` reverses this scheme.
    """
    import numpy as np
    import torch

    from dots_tts.modules.backbone.layers import (
        MultiHeadAttention,
        RotaryEmbedding,
        apply_rotary_pos_emb,
    )

    torch.manual_seed(0)
    attn = (
        MultiHeadAttention(
            1024,
            16,
            qkv_bias=False,
            qk_norm=True,
            norm_layer="RMSNorm",
            rotary_bias=True,
            rotary_theta=10000,
        )
        .float()
        .eval()
    )

    L = 7
    q = torch.randn(1, L, 1024)
    # Causal 3-D bool mask [1, L, L]: True = attend (lower triangular incl. diag).
    mask = torch.tril(torch.ones(L, L, dtype=torch.bool))[None]

    # pos_ids variant A: contiguous arange.
    pos_a = torch.arange(L)[None].float()
    # pos_ids variant B: offset positions (the kind DiT uses for windowed/cont. gen).
    pos_b = torch.tensor([[0, 1, 2, 3, 10, 11, 12]], dtype=torch.float32)

    with torch.no_grad():
        out_a = attn(q, mask=mask, pos_ids=pos_a)
        out_b = attn(q, mask=mask, pos_ids=pos_b)

    # Rotary-only fixture (gates the half-half layout in isolation).
    rope = RotaryEmbedding(64, 10000)
    emb = rope(pos_a)  # [1, L, 64]
    t = torch.randn(1, 16, L, 64)
    with torch.no_grad():
        t_rot = apply_rotary_pos_emb(emb, t)

    saved: dict[str, "np.ndarray"] = {}  # noqa: F821
    for k, v in attn.state_dict().items():
        saved["w_" + k.replace(".", "_")] = v.detach().numpy()
    saved.update(
        q=q.detach().numpy(),
        mask=mask.numpy(),
        pos_ids=pos_a.numpy(),
        out=out_a.detach().numpy(),
        pos_ids_b=pos_b.numpy(),
        out_b=out_b.detach().numpy(),
        rope_pos_ids=pos_a.numpy(),
        rope_t=t.detach().numpy(),
        rope_t_rot=t_rot.detach().numpy(),
    )

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(dest), **saved)
    print(
        f"wrote {dest} (out={tuple(out_a.shape)}, "
        f"t_rot={tuple(t_rot.shape)}, weights={len(attn.state_dict())})"
    )
    return 0


@lru_cache(maxsize=2)
def _load_dit(src: str):
    """Build the torch flow-matching ``DiT`` + load ``velocity_field_predictor.*``.

    Instantiates the DiT from the checkpoint's ``DiT`` config block (layers=18,
    heads=16, hidden=1024, ffn=4096, modulation=True, qk_norm=True/RMSNorm,
    rotary_bias=True, rotary_theta=10000), then loads the acoustic-head weights
    out of ``model.safetensors`` by stripping the ``velocity_field_predictor.``
    prefix (the 244 head keys map 1:1 onto the DiT state_dict). fp32 + eval +
    grads off. Memoized so repeated dumps reuse the same model.
    """
    import torch
    from safetensors.torch import load_file as torch_load_file

    from dots_tts.models.dots_tts.config import _DiTConfig
    from dots_tts.modules.backbone.dit import DiT

    src_path = Path(src)
    with open(src_path / "config.json") as f:
        cfg = json.load(f)
    dit_cfg = _DiTConfig(**cfg["DiT"])

    dit = DiT(
        in_dim=cfg["DiT"]["hidden_size"],
        out_dim=cfg["latent_dim"],
        transformer_config=dit_cfg,
        mode="flow_matching",
    )
    dit = dit.float().eval()

    state = torch_load_file(str(src_path / "model.safetensors"))
    prefix = "velocity_field_predictor."
    head = {
        k[len(prefix) :]: v.float()
        for k, v in state.items()
        if k.startswith(prefix)
    }
    missing, unexpected = dit.load_state_dict(head, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"DiT load mismatch: missing={missing} unexpected={unexpected}"
        )
    for param in dit.parameters():
        param.requires_grad = False
    torch.set_grad_enabled(False)
    return dit


def cmd_dump_dit_step(args: argparse.Namespace) -> int:
    """Dump a single flow-matching DiT velocity-prediction parity fixture.

    Synthetic but valid inputs (the DiT only needs MLX to match torch on the
    SAME inputs; the real mask/pos_ids construction is gated in a later task):
    seed 0, ``L=64``, ``x=randn(1,L,1024)``, ``t=rand(1)`` (in [0,1], raw — no
    1000x scaling), ``g_cond=randn(1,1024)``, a 3-D causal bool ``attn_mask =
    tril(ones(1,L,L))``, ``pos_ids = arange(L)[None].float()``. Runs one velocity
    prediction ``vt = dit(x, timesteps=t, attn_mask=attn_mask, pos_ids=pos_ids,
    g_cond=g_cond)`` -> ``[1, L, 128]``.
    """
    import numpy as np
    import torch

    dit = _load_dit(str(Path(args.src)))

    torch.manual_seed(0)
    L = args.length
    x = torch.randn(1, L, 1024)
    t = torch.rand(1)  # raw timestep in [0, 1]
    g_cond = torch.randn(1, 1024)
    attn_mask = torch.tril(torch.ones(L, L, dtype=torch.bool))[None]  # [1, L, L]
    pos_ids = torch.arange(L)[None].float()  # [1, L]

    with torch.no_grad():
        vt = dit(
            x,
            timesteps=t,
            attn_mask=attn_mask,
            pos_ids=pos_ids,
            g_cond=g_cond,
        )  # [1, L, 128]

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(dest),
        x=x.detach().numpy(),
        t=t.detach().numpy(),
        g_cond=g_cond.detach().numpy(),
        attn_mask=attn_mask.numpy(),
        pos_ids=pos_ids.numpy(),
        vt=vt.detach().numpy(),
    )
    print(f"wrote {dest} (x={tuple(x.shape)}, vt={tuple(vt.shape)})")
    return 0


@lru_cache(maxsize=2)
def _load_fm_harness(src: str):
    """Build a minimal harness exposing the upstream FM solver methods.

    The flow-matching euler + CFG logic lives in ``DotsTtsCore.fm_solver_step`` /
    ``_flow_matching_step_fm``, which only touch ``self.coordinate_proj``,
    ``self.velocity_field_predictor`` and ``self.latent_dim``. Building the full
    ``DotsTtsCore`` would require a tokenizer + the whole model, so instead we make a
    tiny ``nn.Module`` carrying exactly those three attributes and call the upstream
    methods unbound on it. The ``coordinate_proj`` (``Linear(128, 1024)``) is loaded
    from ``model.safetensors`` (key ``coordinate_proj.{weight,bias}`` — NOT under the
    ``velocity_field_predictor.`` prefix). fp32 + eval + grads off; memoized.
    """
    import types

    import torch
    import torch.nn as nn
    from safetensors.torch import load_file as torch_load_file

    from dots_tts.models.dots_tts.core import DotsTtsCore

    dit = _load_dit(src)

    src_path = Path(src)
    with open(src_path / "config.json") as f:
        cfg = json.load(f)
    latent_dim = cfg["latent_dim"]
    fm_hidden = cfg["DiT"]["hidden_size"]

    coordinate_proj = nn.Linear(latent_dim, fm_hidden).float().eval()
    state = torch_load_file(str(src_path / "model.safetensors"))
    coord_sub = {
        k[len("coordinate_proj.") :]: v.float()
        for k, v in state.items()
        if k.startswith("coordinate_proj.")
    }
    missing, unexpected = coordinate_proj.load_state_dict(coord_sub, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"coordinate_proj load mismatch: missing={missing} unexpected={unexpected}"
        )
    for param in coordinate_proj.parameters():
        param.requires_grad = False

    harness = nn.Module()
    harness.coordinate_proj = coordinate_proj
    harness.velocity_field_predictor = dit
    harness.latent_dim = latent_dim
    harness.mode = "flow_matching"
    # Bind the real upstream solver methods so ``_flow_matching_step_fm``'s default
    # ``self.fm_solver_step`` resolves on this harness.
    harness.fm_solver_step = types.MethodType(DotsTtsCore.fm_solver_step, harness)
    harness.eval()
    torch.set_grad_enabled(False)
    return harness


def cmd_dump_flow_solver(args: argparse.Namespace) -> int:
    """Dump a flow-matching euler-ODE + CFG denoise parity fixture (T9).

    Synthetic but shape-valid inputs (the solver math is agnostic to whether the
    sequence is "real" — the real AR sequence/mask construction is the T10 gate):
    seed 0, ``L=64``, ``patch_size=4``, ``latent_dim=128``,
    ``input_sequence = randn(1, L, 1024)``, ``cfg_sequence = randn(1, L, 1024)``
    (the null/history-only branch), ``attn_mask = tril(ones(1, L, L)).bool()``,
    ``pos_ids = arange(L)[None].float()``, ``g_cond = randn(1, 1024)``,
    ``guidance_scale = 1.2``, ``num_steps = 10``, ``noise = randn(1, 4, 128)``.

    Runs the euler loop MANUALLY by calling ``core.fm_solver_step`` with INJECTED
    ``noise`` (so the initial coordinate is explicit + dumped) — left-endpoint RHS
    eval, ``z = z + (1/N) * v(k/N, z)`` for ``k`` in ``0..N-1``. Also sanity-checks
    this manual euler against the upstream torchdiffeq path (``_flow_matching_step_fm``
    with its internal ``torch.randn`` monkeypatched to return the same ``noise``) and
    reports the delta. Dumps every input + ``denoised = z`` (the final, at t=1).
    """
    import numpy as np
    import torch

    from dots_tts.models.dots_tts.core import DotsTtsCore

    harness = _load_fm_harness(str(Path(args.src)))

    torch.manual_seed(0)
    L = args.length
    patch_size = args.patch_size
    latent_dim = harness.latent_dim
    hidden_size = harness.velocity_field_predictor.input_layer.in_features  # 1024
    num_steps = args.num_steps
    guidance_scale = args.guidance_scale

    input_sequence = torch.randn(1, L, hidden_size)
    cfg_sequence = torch.randn(1, L, hidden_size)
    attn_mask = torch.tril(torch.ones(L, L, dtype=torch.bool))[None]  # [1, L, L]
    pos_ids = torch.arange(L)[None].float()  # [1, L]
    g_cond = torch.randn(1, hidden_size)
    noise = torch.randn(1, patch_size, latent_dim)  # the injected initial coordinate

    gs_tensor = input_sequence.new_tensor(float(guidance_scale))

    def _rhs(t_scalar, z):
        t = input_sequence.new_tensor([t_scalar])
        return DotsTtsCore.fm_solver_step(
            harness,
            t,
            z,
            input_sequence=input_sequence,
            cfg_sequence=cfg_sequence,
            attn_mask=attn_mask,
            pos_ids=pos_ids,
            hidden_size=1,  # = self.hidden_patch_size; lands in DiT **kwargs, unused
            patch_size=patch_size,
            g_cond=g_cond,
            guidance_scale=gs_tensor,
        )

    # Manual euler (left-endpoint), dt = 1/N, t_k = k/N, injected noise.
    z = noise.clone()
    dt = 1.0 / num_steps
    for k in range(num_steps):
        z = z + dt * _rhs(k * dt, z)
    denoised = z

    # Sanity-check vs the upstream torchdiffeq path with the SAME injected noise.
    orig_randn = torch.randn
    target_shape = (1, patch_size, latent_dim)

    def _fake_randn(*size, **kw):
        # ``_flow_matching_step_fm`` calls ``torch.randn((1, patch_size, latent_dim),
        # ...)`` — a single tuple positional. Normalize both calling conventions.
        shape = tuple(size[0]) if len(size) == 1 and isinstance(size[0], tuple) else size
        if shape == target_shape:
            return noise.clone()
        return orig_randn(*size, **kw)

    torch.randn = _fake_randn
    try:
        denoised_td = DotsTtsCore._flow_matching_step_fm(
            harness,
            input_sequence=input_sequence,
            cfg_sequence=cfg_sequence,
            attn_mask=attn_mask,
            pos_ids=pos_ids,
            hidden_size=1,
            patch_size=patch_size,
            g_cond=g_cond,
            ode_method="euler",
            num_steps=num_steps,
            guidance_scale=float(guidance_scale),
        )
    finally:
        torch.randn = orig_randn

    delta = float((denoised - denoised_td).abs().max())
    print(f"manual-euler vs torchdiffeq max-abs delta: {delta:.3e}")

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(dest),
        input_sequence=input_sequence.detach().numpy(),
        cfg_sequence=cfg_sequence.detach().numpy(),
        attn_mask=attn_mask.numpy(),
        pos_ids=pos_ids.numpy(),
        g_cond=g_cond.detach().numpy(),
        guidance_scale=np.float32(guidance_scale),
        num_steps=np.int64(num_steps),
        patch_size=np.int64(patch_size),
        noise=noise.detach().numpy(),
        denoised=denoised.detach().numpy(),
        denoised_torchdiffeq=denoised_td.detach().numpy(),
    )
    print(
        f"wrote {dest} (noise={tuple(noise.shape)}, "
        f"denoised={tuple(denoised.shape)}, td_delta={delta:.3e})"
    )
    return 0


@lru_cache(maxsize=2)
def _load_patch_encoder(src: str):
    """Build the torch ``VAESemanticEncoder`` + load ``patch_encoder.*`` weights.

    Instantiates the encoder from the checkpoint config (in_dim=latent_dim=128,
    out_dim=PatchEncoder.hidden_size? no — out_dim = llm_hidden_size = 1536; the
    runtime wires it from the LLM, so we pass 1536 explicitly to match the trained
    ``out_proj`` shape ``[1536, 2048]``). The encoder's ``PatchEncoder`` config
    block says ``qk_norm=True``/``rotary_bias=True`` but ``SuperviseEncoder`` does
    NOT forward those into the layers, so the trained checkpoint has no q_norm /
    rotary params (verified against the key manifest). fp32 + eval + grads off.
    """
    import torch
    from safetensors.torch import load_file as torch_load_file

    from dots_tts.models.dots_tts.config import ModelConfig
    from dots_tts.modules.backbone.semantic_encoder import VAESemanticEncoder

    src_path = Path(src)
    with open(src_path / "config.json") as f:
        cfg = json.load(f)
    model_config = ModelConfig(**cfg)
    # out_dim is the LLM hidden size (1536); the trained out_proj is [1536, 2048]
    # (in-features 2048 = PatchEncoder.hidden_size(1024) * out_ds_rate(2)).
    encoder = (
        VAESemanticEncoder(
            in_dim=model_config.latent_dim,
            out_dim=1536,
            config=model_config,
        )
        .float()
        .eval()
    )

    state = torch_load_file(str(src_path / "model.safetensors"))
    prefix = "patch_encoder."
    sub = {
        k[len(prefix) :]: v.float()
        for k, v in state.items()
        if k.startswith(prefix)
    }
    missing, unexpected = encoder.load_state_dict(sub, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"patch_encoder load mismatch: missing={missing} unexpected={unexpected}"
        )
    for param in encoder.parameters():
        param.requires_grad = False
    torch.set_grad_enabled(False)
    return encoder


def cmd_dump_semantic(args: argparse.Namespace) -> int:
    """Dump a VAE semantic-encoder parity fixture ([1,T,128] latent -> [1,T/4,1536]).

    Uses the NON-streaming ``forward`` (the recompute-full reference): seed 0,
    ``x = randn(1, 16, 128)`` (16 latent frames -> 4 output tokens after the
    total time downsample of patch_size=4). The streaming KV-cache path
    (``decode_patch`` / ``prefill``) is a T10 concern and is numerically identical
    for this causal model.
    """
    import numpy as np
    import torch

    encoder = _load_patch_encoder(str(Path(args.src)))

    torch.manual_seed(0)
    x = torch.randn(1, args.frames, 128, dtype=torch.float32)
    with torch.no_grad():
        emb = encoder(x)  # [1, frames/4, 1536]

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(dest),
        x=x.detach().numpy(),
        emb=emb.detach().numpy(),
    )
    print(f"wrote {dest} (x={tuple(x.shape)}, emb={tuple(emb.shape)})")
    return 0


def cmd_dump_speaker(args: argparse.Namespace) -> int:
    """Dump CAM++ speaker-encoder parity fixtures (16 kHz wav -> fbank -> x-vector).

    Loads a reference wav, resamples to 16 kHz with the *default* torchaudio Resample
    (lowpass_filter_width=6, rolloff=0.99, sinc_interp_hann) — the gate works at 16 kHz
    so the numpy fbank front-end is compared against torch Kaldi on the *same* 16 kHz
    audio (24k->16k resampling is a runtime concern handled later, not here).

    Writes two fixtures:
      fbank.npz   wav16k [S] + fbank [T, 80] (post-CMN)
      xvector.npz fbank  [T, 80] + xvec [1, 512]
    """
    import librosa
    import numpy as np
    import torch
    import torchaudio

    from dots_tts.modules.speaker.fbank import extract_speaker_fbank

    model = _load_campplus(str(Path(args.src)))

    audio, sr = librosa.load(args.wav, sr=None, mono=True)
    wav = torch.from_numpy(np.asarray(audio, dtype=np.float32))  # [S], in [-1, 1]
    # Trim to at most --max-seconds (at the source rate) for a manageable fixture.
    max_src = int(args.max_seconds * sr)
    if wav.shape[0] > max_src:
        wav = wav[:max_src]

    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        wav16k = resampler(wav)  # defaults: lpf_width=6, rolloff=0.99, hann
    else:
        wav16k = wav
    wav16k = wav16k.contiguous().float()  # [S16], in [-1, 1]

    fbank = extract_speaker_fbank(wav16k, sample_rate=16000)  # [T, 80], post-CMN
    xvec = model(fbank[None])  # [1, 512]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fb_path = out_dir / "fbank.npz"
    xv_path = out_dir / "xvector.npz"
    np.savez(str(fb_path), wav16k=wav16k.numpy(), fbank=fbank.numpy())
    np.savez(str(xv_path), fbank=fbank.numpy(), xvec=xvec.detach().numpy())
    print(
        f"wrote {fb_path} (wav16k={tuple(wav16k.shape)}, fbank={tuple(fbank.shape)})"
    )
    print(f"wrote {xv_path} (fbank={tuple(fbank.shape)}, xvec={tuple(xvec.shape)})")
    return 0


@lru_cache(maxsize=2)
def _load_llm(src: str):
    """Build the torch ``Qwen2ForCausalLM`` trunk + load ``llm.*`` and ``eos_proj.*``.

    We don't need the full ``DotsTtsCore`` (which pulls in the VAE/tokenizer/DiT);
    the LLM trunk + the tiny ``eos_proj`` head reproduce ``step_llm`` exactly. The
    Qwen2.5 config (vocab 151672, hidden 1536, 28 layers, 2 kv-heads, rope_theta
    1e6, rms_eps 1e-6, tied embeddings) is read verbatim from the saved
    ``llm_config.json`` — the same object the trained checkpoint was built with,
    so the ``llm.*`` keys load 1:1. fp32 + eval + grads off; memoized.

    Returns ``(llm, eos_proj)`` torch modules.
    """
    import torch
    from safetensors.torch import load_file as torch_load_file
    from transformers import Qwen2Config, Qwen2ForCausalLM

    src_path = Path(src)
    llm_config_path = Path("weights/dots_tts_mlx/llm_config.json")
    llm_config = Qwen2Config.from_dict(json.loads(llm_config_path.read_text()))
    llm = Qwen2ForCausalLM._from_config(llm_config, dtype=torch.float32).float().eval()

    eos_proj = torch.nn.Sequential(
        torch.nn.Linear(1536, 1536),
        torch.nn.SiLU(),
        torch.nn.Linear(1536, 2),
    ).float().eval()

    state = torch_load_file(str(src_path / "model.safetensors"))
    llm_sub = {
        k[len("llm.") :]: v.float() for k, v in state.items() if k.startswith("llm.")
    }
    # lm_head is tied to embed_tokens (not separately stored); allow the missing key.
    missing, unexpected = llm.load_state_dict(llm_sub, strict=False)
    missing = [m for m in missing if m != "lm_head.weight"]
    if missing or unexpected:
        raise RuntimeError(
            f"llm load mismatch: missing={missing} unexpected={unexpected}"
        )

    eos_sub = {
        k[len("eos_proj.") :]: v.float()
        for k, v in state.items()
        if k.startswith("eos_proj.")
    }
    missing, unexpected = eos_proj.load_state_dict(eos_sub, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"eos_proj load mismatch: missing={missing} unexpected={unexpected}"
        )

    for param in llm.parameters():
        param.requires_grad = False
    for param in eos_proj.parameters():
        param.requires_grad = False
    torch.set_grad_enabled(False)
    return llm, eos_proj


def _step_llm(llm, inputs_embeds=None, input_ids=None):
    """Reproduce ``DotsTtsCore.step_llm`` on the standalone trunk.

    Returns ``(inputs_embeds, hidden)`` where ``hidden = outputs.hidden_states[-1]``
    (HF Qwen2's final-RMSNorm output). The full ``step_llm`` also returns logits +
    past_key_values; the MLX runtime only consumes the hidden + a fresh KV cache.
    """
    if inputs_embeds is None:
        inputs_embeds = llm.get_input_embeddings()(input_ids)
    outputs = llm(
        inputs_embeds=inputs_embeds,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )
    return inputs_embeds, outputs.hidden_states[-1]


def cmd_dump_llm(args: argparse.Namespace) -> int:
    """Dump a Qwen2.5 LLM-trunk + eos-head parity fixture for ``step_llm``.

    Two ``step_llm`` cases (seed 0): token ids ``randint(0,151672,(1,12))`` ->
    ``h_ids[1,12,1536]`` (the ids path embeds via ``get_input_embeddings``); raw
    ``embeds = randn(1,5,1536)`` -> ``h_emb[1,5,1536]`` (the inputs_embeds path).
    ``hidden = outputs.hidden_states[-1]`` (post-final-RMSNorm). Then the eos head
    ``eos_proj(h_ids) -> [1,12,2]`` (the decode uses ``softmax(...)[...,1] > 0.8``).
    """
    import numpy as np
    import torch

    llm, eos_proj = _load_llm(str(Path(args.src)))

    torch.manual_seed(0)
    ids = torch.randint(0, 151672, (1, 12))
    _, h_ids = _step_llm(llm, input_ids=ids)  # [1, 12, 1536]
    embeds = torch.randn(1, 5, 1536)
    _, h_emb = _step_llm(llm, inputs_embeds=embeds)  # [1, 5, 1536]
    eos_logits = eos_proj(h_ids)  # [1, 12, 2]

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(dest),
        ids=ids.numpy(),
        h_ids=h_ids.detach().numpy(),
        embeds=embeds.detach().numpy(),
        h_emb=h_emb.detach().numpy(),
        eos_logits=eos_logits.detach().numpy(),
    )
    print(
        f"wrote {dest} (h_ids={tuple(h_ids.shape)}, h_emb={tuple(h_emb.shape)}, "
        f"eos_logits={tuple(eos_logits.shape)})"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_keys = sub.add_parser(
        "keys", help="enumerate weight keys + resolved config block"
    )
    add_common_args(p_keys)
    p_keys.add_argument(
        "--out",
        default="tests/fixtures/dots_tts/keys_manifest.json",
        help="output manifest path",
    )
    p_keys.set_defaults(func=cmd_keys)

    p_dec = sub.add_parser(
        "dump-vocoder-decode",
        help="dump a latent->waveform decode parity fixture",
    )
    add_common_args(p_dec)
    p_dec.add_argument(
        "--frames",
        type=int,
        default=40,
        help="number of latent frames (T) to decode",
    )
    p_dec.add_argument(
        "--out",
        default="tests/fixtures/dots_tts/vocoder_decode.npz",
        help="output fixture path",
    )
    p_dec.set_defaults(func=cmd_dump_vocoder_decode)

    p_enc = sub.add_parser(
        "dump-vae-encode",
        help="dump a waveform->latent encode + io_helper parity fixture",
    )
    add_common_args(p_enc)
    p_enc.add_argument(
        "--wav",
        required=True,
        help="reference wav (resampled to 48 kHz mono fp32)",
    )
    p_enc.add_argument(
        "--max-seconds",
        type=float,
        default=3.0,
        help="trim the reference wav to at most this many seconds",
    )
    p_enc.add_argument(
        "--out",
        default="tests/fixtures/dots_tts/vae_encode.npz",
        help="output fixture path",
    )
    p_enc.set_defaults(func=cmd_dump_vae_encode)

    p_attn = sub.add_parser(
        "dump-attention",
        help="dump a backbone MultiHeadAttention + rotary parity fixture",
    )
    p_attn.add_argument(
        "--out",
        default="tests/fixtures/dots_tts/attention.npz",
        help="output fixture path",
    )
    p_attn.set_defaults(func=cmd_dump_attention)

    p_dit = sub.add_parser(
        "dump-dit-step",
        help="dump a single flow-matching DiT velocity-prediction fixture",
    )
    add_common_args(p_dit)
    p_dit.add_argument(
        "--length",
        type=int,
        default=64,
        help="number of latent positions (L)",
    )
    p_dit.add_argument(
        "--out",
        default="tests/fixtures/dots_tts/dit_step.npz",
        help="output fixture path",
    )
    p_dit.set_defaults(func=cmd_dump_dit_step)

    p_fm = sub.add_parser(
        "dump-flow-solver",
        help="dump a flow-matching euler + CFG denoise (injected-noise) fixture",
    )
    add_common_args(p_fm)
    p_fm.add_argument("--length", type=int, default=64, help="sequence length (L)")
    p_fm.add_argument(
        "--patch-size", type=int, default=4, help="latent patch size (denoised slots)"
    )
    p_fm.add_argument(
        "--num-steps", type=int, default=10, help="euler ODE step count"
    )
    p_fm.add_argument(
        "--guidance-scale", type=float, default=1.2, help="CFG guidance scale"
    )
    p_fm.add_argument(
        "--out",
        default="tests/fixtures/dots_tts/flow_solver.npz",
        help="output fixture path",
    )
    p_fm.set_defaults(func=cmd_dump_flow_solver)

    p_sem = sub.add_parser(
        "dump-semantic",
        help="dump a VAE semantic-encoder ([1,T,128]->[1,T/4,1536]) parity fixture",
    )
    add_common_args(p_sem)
    p_sem.add_argument(
        "--frames",
        type=int,
        default=16,
        help="number of input latent frames (T); output tokens = T/4",
    )
    p_sem.add_argument(
        "--out",
        default="tests/fixtures/dots_tts/semantic_encoder.npz",
        help="output fixture path",
    )
    p_sem.set_defaults(func=cmd_dump_semantic)

    p_spk = sub.add_parser(
        "dump-speaker",
        help="dump CAM++ speaker fbank + x-vector parity fixtures",
    )
    add_common_args(p_spk)
    p_spk.add_argument(
        "--wav",
        required=True,
        help="reference wav (resampled to 16 kHz mono fp32 for the gate)",
    )
    p_spk.add_argument(
        "--max-seconds",
        type=float,
        default=6.0,
        help="trim the reference wav to at most this many seconds",
    )
    p_spk.add_argument(
        "--out-dir",
        default="tests/fixtures/dots_tts",
        help="directory for fbank.npz + xvector.npz fixtures",
    )
    p_spk.set_defaults(func=cmd_dump_speaker)

    p_llm = sub.add_parser(
        "dump-llm",
        help="dump Qwen2.5 LLM-trunk hidden + eos-head parity fixture",
    )
    add_common_args(p_llm)
    p_llm.add_argument(
        "--out",
        default="tests/fixtures/dots_tts/llm_step.npz",
        help="output fixture path",
    )
    p_llm.set_defaults(func=cmd_dump_llm)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
