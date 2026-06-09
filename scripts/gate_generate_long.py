"""Acceptance gate for generate_long: per-language full-coverage + timing (real weights).

Generation only (transcription/coverage is a separate monorepo-env step — mlx_whisper is
not in the package venv). Renders ~25-30s per language via generate_long, cloned from the
user's reference voice, and records timing + a results JSON.

Run (ONE GPU job at a time, SIGTERM-only):
    cd ~/dots-tts-mlx-generate-long && uv run python scripts/gate_generate_long.py
Then transcribe + check coverage from the monorepo env (see the printed instructions).
"""
from __future__ import annotations

import json
import os
import time

import mlx.core as mx
import numpy as np
import soundfile as sf

mx.set_memory_limit(int(45 * (1 << 30)))

from dots_tts_mlx.loader import from_pretrained  # noqa: E402

OUT = "outputs/dots_tts/gate_long"
os.makedirs(OUT, exist_ok=True)
REF = "/Users/shraey/.superset/worktrees/longcat/mlx/outputs/xdub_len/voice_short6s.wav"
REF_TEXT = "I used to live in the cloud, now I'm running on your Mac, avatar video, dubbing."

LANGS = [
    ("EN", "Good morning everyone, and thank you for joining today. I want to walk through how "
           "this system works and why we built it this way. We aimed for something fast, reliable, "
           "and easy to understand. By the end you will see where we are headed and how we get "
           "there together.", "together"),
    ("HI", "नमस्ते दोस्तों, आज के इस सत्र में आपका स्वागत है। मैं बताना चाहता हूँ कि यह तकनीक कैसे "
           "काम करती है और हमने इसे ऐसे क्यों बनाया। हम कुछ ऐसा चाहते थे जो तेज़, भरोसेमंद और समझने "
           "में आसान हो। अंत में आप देखेंगे कि हम किस दिशा में जा रहे हैं और हम मिलकर वहाँ कैसे "
           "पहुँचेंगे।", "पहुँचेंगे"),
    ("ZH", "大家早上好，感谢各位今天的参与。我想介绍一下这个系统是如何工作的。我们希望做出快速、可靠"
           "并且易于理解的东西。最后，你们会看到我们如何一起到达那里。", "那里"),
]


def main() -> int:
    model = from_pretrained("weights/dots_tts_mlx", dtype=mx.bfloat16).model
    sr = model.sample_rate
    results = {}
    for code, text, final in LANGS:
        t0 = time.time()
        out = model.generate_long(text=text, prompt_audio=REF, prompt_text=REF_TEXT,
                                  num_steps=10, guidance_scale=1.2, speaker_scale=1.5, seed=42,
                                  language=code, gap_ms=80)
        wav = np.asarray(out["audio"][0].astype(mx.float32))
        p = os.path.join(OUT, f"long_{code.lower()}.wav")
        sf.write(p, wav, sr)
        results[code] = {"chunks": out["num_chunks"], "patches": out["num_patches"],
                         "audio_s": round(wav.size / sr, 1), "wall_s": round(time.time() - t0, 1),
                         "final_word": final, "wav": p}
        print(f"[{code}] {out['num_chunks']} chunks {out['num_patches']}p "
              f"{wav.size / sr:.1f}s {results[code]['wall_s']}s wall", flush=True)
        mx.synchronize()
        mx.clear_cache()
    with open(os.path.join(OUT, "gate_long.json"), "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nNEXT (coverage, monorepo env): mlx_whisper transcribe each clip + assert the "
          "final_word appears. Timing is thermal-confounded (sequential); cold ~2.5x realtime.",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
