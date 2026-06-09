import mlx.core as mx
import numpy as np

from dots_tts_mlx.model import DotsTtsModel


def _make_model():
    m = object.__new__(DotsTtsModel)
    m.sample_rate = 48000
    return m


def _audio(seconds, std=0.1):
    n = int(seconds * 48000)
    return mx.array((np.random.default_rng(0).standard_normal(n) * std)[None].astype(np.float32))


def _install(monkeypatch, outcomes, single_chunk="one two three four five"):
    monkeypatch.setattr("dots_tts_mlx.chunking.split_for_generation",
                        lambda text, **k: [single_chunk])
    calls = {"n": 0}

    def fake_generate(self, chunk, **kw):
        i = calls["n"]
        calls["n"] += 1
        secs, std = outcomes[min(i, len(outcomes) - 1)]
        return {"audio": _audio(secs, std), "num_patches": 10, "sample_rate": 48000}

    monkeypatch.setattr(DotsTtsModel, "generate", fake_generate, raising=True)
    return calls


def test_retries_until_healthy(monkeypatch):
    m = _make_model()
    calls = _install(monkeypatch, [(0.3, 0.1), (2.0, 0.1)])
    out = m.generate_long("one two three four five", max_retries=2)
    assert calls["n"] == 2
    assert out["retries"] == 1 and out["unrecovered"] == 0
    assert out["audio"].shape[-1] == int(2.0 * 48000)


def test_max_retries_cap_keeps_longest(monkeypatch):
    m = _make_model()
    calls = _install(monkeypatch, [(0.2, 0.1), (0.3, 0.1), (0.5, 0.1)])
    out = m.generate_long("one two three four five", max_retries=2)
    assert calls["n"] == 3 and out["unrecovered"] == 1
    assert out["audio"].shape[-1] == int(0.5 * 48000)


def test_retry_disabled_is_one_call(monkeypatch):
    m = _make_model()
    calls = _install(monkeypatch, [(0.3, 0.1)])
    out = m.generate_long("one two three four five", retry_degenerate=False)
    assert calls["n"] == 1 and out["retries"] == 0


def test_validator_required_in_addition(monkeypatch):
    m = _make_model()
    calls = _install(monkeypatch, [(2.0, 0.1), (2.0, 0.1)])
    seen = {"n": 0}

    def validator(audio, text, sr):
        seen["n"] += 1
        return seen["n"] > 1

    out = m.generate_long("one two three four five", validator=validator, max_retries=2)
    assert calls["n"] == 2 and out["retries"] == 1


def test_validator_raising_is_not_accepted(monkeypatch):
    m = _make_model()
    calls = _install(monkeypatch, [(2.0, 0.1), (2.0, 0.1)])

    def boom(audio, text, sr):
        raise RuntimeError("asr down")

    out = m.generate_long("one two three four five", validator=boom, max_retries=1)
    assert calls["n"] == 2 and out["unrecovered"] == 1
