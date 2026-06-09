import numpy as np

from dots_tts_mlx.chunking import chunk_health

SR = 48000


def _tone(seconds: float, std: float = 0.1) -> np.ndarray:
    n = int(seconds * SR)
    rng = np.random.default_rng(0)
    return (rng.standard_normal(n) * std).astype(np.float32)


def test_healthy_chunk_passes():
    assert chunk_health(_tone(2.5), "I think it sounds natural", SR) is True


def test_silent_chunk_fails():
    assert chunk_health(np.zeros(SR, dtype=np.float32), "hello there friend", SR) is False


def test_nonfinite_chunk_fails():
    a = _tone(2.0)
    a[10] = np.nan
    assert chunk_health(a, "hello there friend", SR) is False


def test_empty_audio_fails():
    assert chunk_health(np.zeros(0, dtype=np.float32), "hello", SR) is False


def test_truncated_chunk_fails():
    assert chunk_health(_tone(0.5), "one two three four five six seven eight", SR) is False


def test_empty_text_is_healthy():
    assert chunk_health(_tone(1.0), "", SR) is True


def test_cjk_uses_char_floor():
    assert chunk_health(_tone(0.4), "感谢你试用这个我想你会自然", SR) is False
    assert chunk_health(_tone(2.0), "感谢你试用这个我想你会自然", SR) is True
