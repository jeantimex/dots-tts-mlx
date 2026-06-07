import mlx.core as mx
import numpy as np

from dots_tts_mlx.profile import SpeakerProfile


def _toy_profile() -> SpeakerProfile:
    rng = np.random.default_rng(0)

    def a(*s):
        return mx.array(rng.standard_normal(s).astype(np.float32)).astype(mx.bfloat16)

    return SpeakerProfile(
        g_cond=a(1, 1024),
        prompt_patches=a(1, 3, 4, 128),
        prompt_denorm_latents=a(1, 12, 128),
        patch_emb=a(1, 3, 1536),
        prompt_text="the reference transcript",
        prompt_patch_count=3,
        speaker_scale=1.5,
        latent_dim=128,
        patch_size=4,
        hop_size=1920,
        sample_rate=48000,
        dtype="bfloat16",
        compat_hash="deadbeef",
        schema_version=1,
    )


def test_save_load_roundtrip(tmp_path):
    p = _toy_profile()
    path = tmp_path / "alice.dtprofile"
    p.save(path)
    assert (path / "cond.safetensors").exists()
    assert (path / "profile.json").exists()

    q = SpeakerProfile.load(path)
    for name in ("g_cond", "prompt_patches", "prompt_denorm_latents", "patch_emb"):
        lhs = np.asarray(getattr(p, name).astype(mx.float32))
        rhs = np.asarray(getattr(q, name).astype(mx.float32))
        assert np.array_equal(lhs, rhs), name
    assert q.prompt_text == p.prompt_text
    assert q.prompt_patch_count == 3
    assert q.speaker_scale == 1.5
    assert q.dtype == "bfloat16"
    assert q.compat_hash == "deadbeef"
    assert q.schema_version == 1


def test_load_missing_files_raises(tmp_path):
    import pytest

    (tmp_path / "empty.dtprofile").mkdir()
    with pytest.raises(FileNotFoundError):
        SpeakerProfile.load(tmp_path / "empty.dtprofile")


# --- model-compat hash tests (Task 2) ---
import json as _json  # noqa: E402

import mlx.core as _mx  # noqa: E402
import numpy as _np  # noqa: E402

from dots_tts_mlx.profile import model_compat_hash  # noqa: E402


def _fake_model_dir(tmp_path, *, latent_mean=0.0, extra_core=None, with_quant=False):
    d = tmp_path / "model"
    d.mkdir(parents=True, exist_ok=True)
    cfg = {"latent_dim": 128, "patch_size": 4, "vocoder": {"sample_rate": 48000}}
    if with_quant:
        cfg["quantization"] = {"bits": 4, "group_size": 64, "components": ["llm"]}
    (d / "config.json").write_text(_json.dumps(cfg))
    _np.savez(d / "latent_stats.npz", mean=_np.full(128, latent_mean, _np.float32),
              std=_np.ones(128, _np.float32))
    core = {"velocity_field_predictor.w": _mx.zeros((4, 4)),
            "patch_encoder.w": _mx.zeros((4, 4)),
            "llm.model.layers.0.w": _mx.zeros((4, 4))}
    if extra_core:
        core.update(extra_core)
    _mx.save_safetensors(str(d / "core.safetensors"), core)
    _mx.save_safetensors(str(d / "speaker.safetensors"), {"s": _mx.zeros((2, 2))})
    _mx.save_safetensors(str(d / "vocoder.safetensors"), {"v": _mx.zeros((2, 2))})
    return d


def test_compat_hash_stable(tmp_path):
    d = _fake_model_dir(tmp_path)
    assert model_compat_hash(d) == model_compat_hash(d)


def test_compat_hash_ignores_quant_and_llm(tmp_path):
    base = _fake_model_dir(tmp_path / "a")
    quant = _fake_model_dir(tmp_path / "b", with_quant=True,
                            extra_core={"llm.model.layers.1.w": _mx.zeros((4, 4))})
    assert model_compat_hash(base) == model_compat_hash(quant)


def test_compat_hash_sensitive_to_latent_stats(tmp_path):
    a = _fake_model_dir(tmp_path / "a", latent_mean=0.0)
    b = _fake_model_dir(tmp_path / "b", latent_mean=1.0)
    assert model_compat_hash(a) != model_compat_hash(b)


def test_compat_hash_sensitive_to_nonllm_core(tmp_path):
    a = _fake_model_dir(tmp_path / "a")
    b = _fake_model_dir(tmp_path / "b", extra_core={"patch_encoder.w2": _mx.zeros((8, 8))})
    assert model_compat_hash(a) != model_compat_hash(b)
