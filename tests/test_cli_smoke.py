"""CLI smoke test for ``dots_tts_mlx.cli``.

Runs the voice-cloning CLI end-to-end via subprocess (one full generation) on a reference
clip and a short text, then asserts the ``{prefix}_000.wav`` exists, is 48 kHz, > 0.5 s,
finite, and peaks <= 1.0. Marked ``slow`` (full AR decode + vocode); skips if the converted
weights or the reference clip are absent. Set ``DOTS_TTS_REF`` to point at your own
reference wav (default: ``tests/fixtures/ref.wav``).
"""
import os
import pathlib
import subprocess
import sys
import wave

import numpy as np
import pytest

W = pathlib.Path("weights/dots_tts_mlx")
REF = pathlib.Path(os.environ.get("DOTS_TTS_REF", "tests/fixtures/ref.wav"))
OUT_DIR = pathlib.Path("outputs/dots_tts/smoke")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not ((W / "core.safetensors").exists() and REF.exists()),
        reason="dots_tts weights or reference clip absent",
    ),
]


def _read_wav(path: pathlib.Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        sw = wf.getsampwidth()
        n = wf.getnframes()
        raw = wf.readframes(n)
    if sw == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 4:
        data = np.frombuffer(raw, dtype="<f4").astype(np.float32)
    else:
        raise ValueError(f"unexpected sample width {sw}")
    return data, sr


def test_cli_smoke(tmp_path):
    prefix = "smoke_clone"
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_wav = out_dir / f"{prefix}_000.wav"
    if out_wav.exists():
        out_wav.unlink()

    cmd = [
        sys.executable,
        "-m",
        "dots_tts_mlx.cli",
        "--text",
        "Hello, this is a quick test of the on-device voice.",
        "--ref-audio",
        str(REF),
        "--ref-text",
        "I used to live in the cloud, now I'm running on your Mac, avatar video, dubbing.",
        "--out-path",
        str(out_dir),
        "--out-prefix",
        prefix,
        "--language",
        "EN",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, f"CLI failed:\n{proc.stdout}\n{proc.stderr}"

    # the _000.wav output contract.
    assert out_wav.exists(), f"missing {out_wav}\n{proc.stdout}\n{proc.stderr}"

    data, sr = _read_wav(out_wav)
    assert sr == 48000, f"sample rate {sr} != 48000"
    duration = data.shape[-1] / sr
    assert duration > 0.5, f"clip too short: {duration:.2f}s"
    assert np.all(np.isfinite(data)), "non-finite samples"
    assert float(np.max(np.abs(data))) <= 1.0, "samples exceed [-1, 1]"
