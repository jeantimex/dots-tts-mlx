"""CPU unit tests for the MeanFlow few-step decode path (no weights)."""
from __future__ import annotations

import json

from dots_tts_mlx.config import MeanFlowConfig, ModelConfig


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
