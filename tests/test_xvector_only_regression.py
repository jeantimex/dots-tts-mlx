import pathlib

import mlx.core as mx
import numpy as np
import pytest

mx.set_memory_limit(int(45 * (1 << 30)))

W = pathlib.Path("weights/dots_tts_mlx")
REF = "/Users/shraey/.superset/worktrees/longcat/mlx/outputs/xdub_len/voice_short6s.wav"


@pytest.mark.skipif(not (W / "core.safetensors").exists(), reason="needs real weights")
def test_xvector_only_deterministic_and_finite():
    from dots_tts_mlx.model import DotsTtsModel

    model = DotsTtsModel.from_pretrained(W, dtype=mx.bfloat16)
    kw = dict(prompt_audio=REF, language="EN", num_steps=6, seed=42, max_generate_length=120)
    a = np.asarray(model.generate(text="Quick determinism check.", **kw)["audio"].astype(mx.float32)).ravel()
    mx.synchronize()
    mx.clear_cache()
    b = np.asarray(model.generate(text="Quick determinism check.", **kw)["audio"].astype(mx.float32)).ravel()
    assert a.shape == b.shape and float(np.max(np.abs(a - b))) == 0.0, "x-vector-only not deterministic"
    assert np.all(np.isfinite(a)) and float(np.std(a)) > 0.01, "x-vector-only degenerate"
