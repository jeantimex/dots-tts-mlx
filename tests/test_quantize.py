# tests/test_quantize.py
import json

import mlx.core as mx  # noqa: F401  (import guards the MLX env for later tests)


def _write_min_checkpoint(d, *, quantization=None):
    """Minimal config.json + llm_config.json that ModelConfig.from_checkpoint accepts."""
    cfg = {}
    if quantization is not None:
        cfg["quantization"] = quantization
    (d / "config.json").write_text(json.dumps(cfg))
    (d / "llm_config.json").write_text(json.dumps({}))  # LLMConfig.from_dict uses defaults


def test_quantization_block_parsed(tmp_path):
    from dots_tts_mlx.config import ModelConfig

    _write_min_checkpoint(tmp_path, quantization={"bits": 4})
    cfg = ModelConfig.from_checkpoint(tmp_path)
    assert cfg.quantization is not None
    assert cfg.quantization.bits == 4
    assert cfg.quantization.group_size == 64          # default
    assert cfg.quantization.components == ["llm"]      # default


def test_quantization_absent_is_none(tmp_path):
    from dots_tts_mlx.config import ModelConfig

    _write_min_checkpoint(tmp_path, quantization=None)
    cfg = ModelConfig.from_checkpoint(tmp_path)
    assert cfg.quantization is None
