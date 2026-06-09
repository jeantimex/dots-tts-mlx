import os
import pathlib

import mlx.core as mx
import numpy as np
import pytest

W = pathlib.Path(os.environ.get("DOTS_TTS_WEIGHTS", "weights/dots_tts_mlx"))

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not W.exists(), reason=f"weights absent at {W} (set $DOTS_TTS_WEIGHTS)"),
]

REF = pathlib.Path(os.environ.get("DOTS_TTS_REF", "reference.wav"))
REF_TEXT = os.environ.get("DOTS_TTS_REF_TEXT", "this is the reference transcript")


def _model():
    from dots_tts_mlx.model import DotsTtsModel

    return DotsTtsModel.from_pretrained(W, dtype=mx.bfloat16)


@pytest.mark.skipif(not REF.exists(), reason="reference wav absent (set $DOTS_TTS_REF)")
def test_prefill_patch_emb_passthrough():
    """_prefill(patch_emb=precomputed) == _prefill(patch_emb=None) when precomputed
    equals patch_encoder(flat). Proves the new parameter is a pure pass-through."""
    import math

    from dots_tts_mlx.model import _load_wav_mono, _resample, _trim_silence

    m = _model()
    raw, sr = _load_wav_mono(REF)
    a48 = _resample(_trim_silence(raw, top_db=30.0), sr, m.sample_rate)
    chunk = m.patch_size * m.hop_size
    n = a48.shape[-1]
    target = math.ceil(n / chunk) * chunk
    if target > n:
        a48 = np.pad(a48, (0, target - n))

    g, patches, denorm = m._prepare_prompt_conditioning(
        a48, use_prompt_prefill=True, speaker_scale=1.5
    )
    s = int(patches.shape[1])
    ids = m.tokenizer.build_generation_schedule(
        "A short target.", prompt_text=REF_TEXT, language=None, max_audio_tokens=500
    )
    sched = mx.array(ids, dtype=mx.int32)[None]
    aud = {m.tokenizer.audio_gen_span_id, m.tokenizer.audio_comp_span_id}
    spans = [i for i, t in enumerate(ids) if t in aud]

    def run(patch_emb):
        m._fm_chunks, m._fm_cfg_chunks, m._fm_seq_len = [], [], 0
        m._denorm_patch_history = [denorm[:, i * m.patch_size : (i + 1) * m.patch_size] for i in range(s)]
        cache = m.llm.make_cache()
        m._prefill(ids, sched, span_positions=spans, prompt_patches=patches,
                   prompt_denorm_latents=denorm, patch_emb=patch_emb, cache=cache)
        return m._last_prefill_hidden, mx.concatenate(m._fm_chunks, axis=1)

    # patch_emb is recomputed from DENORMALIZED latents (the in-context fix), so the
    # precomputed reference must use the same denorm input to match run(None).
    flat = denorm[:, : s * m.patch_size]
    precomputed = m.patch_encoder(flat).astype(m.dtype)
    h_none, fm_none = run(None)
    h_pre, fm_pre = run(precomputed)
    assert np.array_equal(np.asarray(h_none.astype(mx.float32)), np.asarray(h_pre.astype(mx.float32)))
    assert np.array_equal(np.asarray(fm_none.astype(mx.float32)), np.asarray(fm_pre.astype(mx.float32)))


@pytest.mark.skipif(not REF.exists(), reason="reference wav absent (set $DOTS_TTS_REF)")
def test_enroll_builds_profile(tmp_path):
    from dots_tts_mlx.profile import SpeakerProfile

    m = _model()
    prof = m.enroll(str(REF), REF_TEXT, speaker_scale=1.5)
    assert isinstance(prof, SpeakerProfile)
    s = prof.prompt_patch_count
    assert s > 0
    assert tuple(prof.g_cond.shape) == (1, 1024)
    assert tuple(prof.prompt_patches.shape) == (1, s, m.patch_size, m.latent_dim)
    assert tuple(prof.patch_emb.shape) == (1, s, 1536)
    assert prof.prompt_text == REF_TEXT
    assert prof.compat_hash == m._compat_hash
    out = tmp_path / "v.dtprofile"
    prof.save(out)
    SpeakerProfile.load(out).check_compat(m._compat_hash)


def test_enroll_requires_prompt_text():
    m = _model()
    with pytest.raises(ValueError, match="prompt_text"):
        m.enroll(str(REF), "")


@pytest.mark.skipif(not REF.exists(), reason="reference wav absent (set $DOTS_TTS_REF)")
def test_profile_generate_matches_one_shot(tmp_path):
    """enroll -> save -> load -> generate(profile) == one-shot generate(ref) (same seed)."""
    m = _model()
    text = "Hello from the enrolled voice."
    kw = dict(num_steps=6, guidance_scale=1.2, speaker_scale=1.5, language="EN", seed=42)

    one_shot = m.generate(text, prompt_audio=str(REF), prompt_text=REF_TEXT, **kw)

    # The reference encoding is stochastic (upstream samples the VAE posterior:
    # z = mean + randn*exp(log_std)). The one-shot above seeds 42 internally, so its
    # prompt sample is seed-42's first draw. To compare the CACHE against that exact
    # encoding, enroll must draw its prompt sample from the same RNG state — seed 42
    # here so the cached artifacts match. (In production a profile just captures one
    # fixed sample and is reused as-is; this seeding only makes the parity check exact.)
    mx.random.seed(42)
    prof = m.enroll(str(REF), REF_TEXT, speaker_scale=1.5)
    p = tmp_path / "v.dtprofile"
    prof.save(p)
    from dots_tts_mlx.profile import SpeakerProfile

    loaded = SpeakerProfile.load(p)
    via = m.generate(text, profile=loaded, num_steps=6, guidance_scale=1.2, language="EN", seed=42)

    a = np.asarray(one_shot["audio"].astype(mx.float32)).ravel()
    b = np.asarray(via["audio"].astype(mx.float32)).ravel()
    assert a.shape == b.shape, (a.shape, b.shape)
    assert np.max(np.abs(a)) > 0.01, "one-shot output is silence — parity test is meaningless"
    assert np.max(np.abs(a - b)) < 1e-3, float(np.max(np.abs(a - b)))


def test_profile_mutual_exclusion():
    m = _model()
    if not REF.exists():
        pytest.skip("reference wav absent")
    prof = m.enroll(str(REF), REF_TEXT)
    with pytest.raises(ValueError, match="mutually exclusive"):
        m.generate("x", profile=prof, prompt_audio=str(REF))
    with pytest.raises(ValueError, match="mutually exclusive"):
        m.generate("x", profile=prof, prompt_text="y")


@pytest.mark.skipif(not REF.exists(), reason="reference wav absent (set $DOTS_TTS_REF)")
def test_generate_trims_onset_by_default():
    """generate() applies the onset trim by default; trim_onset=False keeps raw; default==True."""
    m = _model()
    kw = dict(prompt_audio=str(REF), prompt_text=REF_TEXT, num_steps=6, language="EN", seed=42)
    raw = m.generate("Testing the onset trim path now.", trim_onset=False, **kw)
    trimmed = m.generate("Testing the onset trim path now.", trim_onset=True, **kw)
    default = m.generate("Testing the onset trim path now.", **kw)  # no trim_onset arg -> default
    n_raw = int(raw["audio"].shape[-1])
    n_trim = int(trimmed["audio"].shape[-1])
    n_def = int(default["audio"].shape[-1])
    assert n_trim <= n_raw, (n_trim, n_raw)   # trimming never lengthens
    assert n_def == n_trim                     # default behaviour is trim_onset=True
