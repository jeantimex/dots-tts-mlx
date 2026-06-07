# tests/test_quantize.py
import json

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten


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


def _tiny_llm_config():
    return {
        "model_type": "qwen2",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "vocab_size": 128,
        "max_position_embeddings": 512,
        "rope_theta": 1000000.0,
        "tie_word_embeddings": True,
    }


def _build_tiny_quantized_core(tmp_path, *, quantized: bool):
    """Write a tiny core.safetensors (llm.* + eos_proj.*) and llm_config.json.

    quantized=True → run nn.quantize so llm.* carries packed weight + scales + biases.
    quantized=False → plain float llm.* (no scales).
    """
    import mlx.nn as nn
    from mlx_lm.models.qwen2 import Model as Qwen2Model
    from mlx_lm.models.qwen2 import ModelArgs as Qwen2Args

    cfg = _tiny_llm_config()
    args = Qwen2Args(
        model_type="qwen2", hidden_size=cfg["hidden_size"],
        num_hidden_layers=cfg["num_hidden_layers"], intermediate_size=cfg["intermediate_size"],
        num_attention_heads=cfg["num_attention_heads"], rms_norm_eps=cfg["rms_norm_eps"],
        vocab_size=cfg["vocab_size"], num_key_value_heads=cfg["num_key_value_heads"],
        max_position_embeddings=cfg["max_position_embeddings"], rope_theta=cfg["rope_theta"],
        tie_word_embeddings=True,
    )
    m = Qwen2Model(args)
    if quantized:
        nn.quantize(m, group_size=64, bits=4)
    core = {f"llm.{k}": v for k, v in tree_flatten(m.parameters())}
    H = cfg["hidden_size"]
    core["eos_proj.0.weight"] = mx.zeros((H, H))
    core["eos_proj.0.bias"] = mx.zeros((H,))
    core["eos_proj.2.weight"] = mx.zeros((2, H))
    core["eos_proj.2.bias"] = mx.zeros((2,))
    mx.save_safetensors(str(tmp_path / "core.safetensors"), core)
    (tmp_path / "llm_config.json").write_text(json.dumps(cfg))
    return str(tmp_path / "core.safetensors"), str(tmp_path / "llm_config.json")


def test_from_core_quantized_runs(tmp_path):
    from dots_tts_mlx.config import QuantizationConfig
    from dots_tts_mlx.llm import DotsLLM

    core, cfg = _build_tiny_quantized_core(tmp_path, quantized=True)
    q = QuantizationConfig(bits=4, group_size=64, components=["llm"])
    llm = DotsLLM.from_core(core, cfg, dtype=mx.float32, quantization=q)
    h, _ = llm.step(input_ids=mx.array([[1, 2, 3]]))
    assert h.shape == (1, 3, 64)
    assert bool(mx.isfinite(h).all())


def test_from_core_quant_block_without_scales_raises(tmp_path):
    from dots_tts_mlx.config import QuantizationConfig
    from dots_tts_mlx.llm import DotsLLM

    core, cfg = _build_tiny_quantized_core(tmp_path, quantized=False)  # plain floats, no scales
    q = QuantizationConfig(bits=4, group_size=64, components=["llm"])
    with pytest.raises(ValueError, match="quantiz"):
        DotsLLM.from_core(core, cfg, dtype=mx.float32, quantization=q)
