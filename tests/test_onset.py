"""Unit tests for the BigVGAN onset-transient trimmer (no GPU, no model load)."""
import numpy as np

from dots_tts_mlx.model import _clean_onset


def test_clean_onset_trims_leading_transient():
    """A low-level lead burst + silence gap before loud speech is trimmed; body kept."""
    sr = 48000
    rng = np.random.default_rng(0)
    lead = (rng.standard_normal(int(0.10 * sr)) * 0.05).astype(np.float32)  # ~100 ms soft burst
    gap = np.zeros(int(0.10 * sr), np.float32)                              # 100 ms silence
    t = np.arange(int(1.0 * sr)) / sr
    body = (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)           # 1 s loud tone
    x = np.concatenate([lead, gap, body])

    y = _clean_onset(x, sr)

    assert len(y) < len(x), "expected leading transient + gap to be trimmed"
    assert (len(x) - len(y)) / sr > 0.12, "should remove most of the ~200 ms lead"
    assert np.max(np.abs(y)) > 0.3, "loud speech body must be preserved"


def test_clean_onset_uniform_signal_not_trimmed():
    """A uniformly-loud signal (speech 'starts' at sample 0) is left at full length."""
    sr = 48000
    x = np.full(int(0.5 * sr), 0.2, np.float32)
    y = _clean_onset(x, sr)
    assert len(y) == len(x)  # nothing to trim; only a 10 ms fade-in is applied


def test_clean_onset_returns_float32_mono():
    sr = 48000
    x = np.zeros((int(0.3 * sr), 2), np.float32)  # stereo -> should be downmixed
    y = _clean_onset(x, sr)
    assert y.dtype == np.float32
    assert y.ndim == 1
