import json

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
