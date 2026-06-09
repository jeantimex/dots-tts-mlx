"""CPU unit tests for the MeanFlow few-step decode path (no weights)."""
from __future__ import annotations

import json

import mlx.core as mx

from dots_tts_mlx.config import MeanFlowConfig, ModelConfig
from dots_tts_mlx.dit import DiT, FinalLayer, TimestepEmbedder
from dots_tts_mlx.layers import Linear


def _write_min_checkpoint(tmp_path, *, meanflow: dict | None):
    """Write a minimal config.json + llm_config.json so ModelConfig.from_checkpoint loads."""
    cfg = {
        "latent_dim": 128, "patch_size": 4,
        "DiT": {"num_layers": 18, "num_heads": 16, "hidden_size": 1024},
        "PatchEncoder": {}, "vocoder": {"sample_rate": 48000},
    }
    if meanflow is not None:
        cfg["meanflow"] = meanflow
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    (tmp_path / "llm_config.json").write_text(json.dumps({"hidden_size": 1536}))
    return tmp_path


def test_meanflow_config_absent_is_flow_matching(tmp_path):
    p = _write_min_checkpoint(tmp_path, meanflow=None)
    cfg = ModelConfig.from_checkpoint(p)
    assert cfg.meanflow is None
    assert cfg.mode == "flow_matching"


def test_meanflow_config_present_enabled(tmp_path):
    p = _write_min_checkpoint(tmp_path, meanflow={"enabled": True, "use_duration_embedding": True})
    cfg = ModelConfig.from_checkpoint(p)
    assert isinstance(cfg.meanflow, MeanFlowConfig)
    assert cfg.meanflow.enabled is True
    assert cfg.meanflow.use_duration_embedding is True
    assert cfg.mode == "meanflow"


def test_meanflow_config_present_disabled_is_flow_matching(tmp_path):
    p = _write_min_checkpoint(tmp_path, meanflow={"enabled": False})
    cfg = ModelConfig.from_checkpoint(p)
    assert cfg.meanflow is not None
    assert cfg.meanflow.enabled is False
    assert cfg.mode == "flow_matching"


def _tiny_dit(*, with_duration: bool):
    """A minimal 0-block DiT at hidden=8, in/out=4 with deterministic weights."""
    H, D = 8, 4

    def lin(out_f, in_f, fill):
        w = mx.full((out_f, in_f), fill, dtype=mx.float32)
        b = mx.zeros((out_f,), dtype=mx.float32)
        return Linear(w, b)

    def t_embed():
        # TimestepEmbedder: mlp 256->H->H
        return TimestepEmbedder(lin(H, 256, 0.001), lin(H, H, 0.01))

    input_layer = lin(H, D, 0.1)
    output_layer = FinalLayer(lin(2 * H, H, 0.05), lin(D, H, 0.05), hidden_size=H)
    kwargs = dict(
        input_layer=input_layer, time_embedder=t_embed(), blocks=[], output_layer=output_layer
    )
    if with_duration:
        kwargs["duration_embedder"] = t_embed()
    return DiT(**kwargs)


def test_dit_duration_changes_conditioning():
    """With a duration_embedder, passing duration must change the output."""
    dit = _tiny_dit(with_duration=True)
    x = mx.ones((1, 4, 4), dtype=mx.float32)
    t = mx.array([0.5], dtype=mx.float32)
    out_no_dur = dit(x, t)                       # duration omitted
    out_dur = dit(x, t, duration=mx.array([0.25], dtype=mx.float32))
    assert float(mx.max(mx.abs(out_dur - out_no_dur))) > 1e-6


def test_dit_without_duration_embedder_ignores_duration():
    """A flow-matching DiT (no duration_embedder) is unaffected by a duration arg."""
    dit = _tiny_dit(with_duration=False)
    x = mx.ones((1, 4, 4), dtype=mx.float32)
    t = mx.array([0.5], dtype=mx.float32)
    out_a = dit(x, t)
    out_b = dit(x, t, duration=mx.array([0.25], dtype=mx.float32))
    assert float(mx.max(mx.abs(out_a - out_b))) == 0.0
