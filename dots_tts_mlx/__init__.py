"""dots.tts MLX port — pure-MLX runtime for ``rednote-hilab/dots.tts`` on Apple Silicon.

Architecture (un-prefixed, submodule-level checkpoint keys):
  * ``llm.*`` — Qwen2.5 backbone (semantic token LM).
  * ``velocity_field_predictor.*`` — flow-matching DiT (velocity field).
  * ``patch_encoder.*`` — causal semantic patch encoder.
  * ``hidden_proj`` / ``latent_proj`` / ``coordinate_proj`` / ``xvec_proj`` / ``eos_proj`` — projections.
  * vocoder (``vocoder.safetensors``) — AudioVAE encoder + BigVGAN-style decoder.
  * speaker (``speaker_encoder.safetensors``) — CAMPPlus x-vector extractor.

``config.py`` is framework-agnostic (importable from the torch oracle venv).
``convert.py`` is the offline build tool (torch). ``loader.py`` is the MLX runtime entry.
"""

from .config import (
    DiTConfig,
    EncoderConfig,
    LLMConfig,
    ModelConfig,
    VocoderConfig,
)

__all__ = [
    "DiTConfig",
    "EncoderConfig",
    "LLMConfig",
    "ModelConfig",
    "VocoderConfig",
]
