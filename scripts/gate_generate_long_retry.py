"""Manual GPU acceptance gate for generate_long's degenerate-chunk reseed-retry guard.

Proves, on REAL weights, three things end-to-end (the CPU unit tests already prove the
control flow with fakes — this proves the real-weights path + that the ASR-validator hook
is wired correctly):

  (A) Validator-accepts-good (control): a normal multi-sentence EN render with an
      ASR-coverage validator (whisper, coverage >= 0.6) is accepted first try
      -> retries=0, unrecovered=0.

  (B) Mechanism on real weights (deterministic): the SAME text with a stateful
      "first-attempt-reject" validator (False the first time each distinct chunk is
      seen, True after) forces exactly one retry per chunk -> retries == num_chunks,
      and the kept audio DIFFERS from the attempt-0 audio (a real reseed produced
      different audio that was then accepted). Independent of model luck.

  (C) Natural-degeneration probe (honest): the historically-degenerate case
      ("I think it sounds quite natural.", seed=42) with and without the ASR validator.
      Transcribe both and report HONESTLY whether seed 42 still hallucinates on this
      build. No faking — if the no-validator output is already faithful, we say so.

whisper lives ONLY in this script (the package stays pure-MLX). Run:

    cd ~/dots-tts-mlx-degen-guard && \
        uv run --with mlx_whisper python scripts/gate_generate_long_retry.py

ONE GPU job at a time; SIGTERM-only. Outputs -> outputs/dots_tts/degen_guard/ (NEVER /tmp).
"""
from __future__ import annotations

import json
import os
import re
import time

import mlx.core as mx
import numpy as np
import soundfile as sf

mx.set_memory_limit(int(45 * (1 << 30)))

import mlx_whisper  # noqa: E402

from dots_tts_mlx.loader import from_pretrained  # noqa: E402

OUT = "outputs/dots_tts/degen_guard"
os.makedirs(OUT, exist_ok=True)

REF = "/Users/shraey/.superset/worktrees/longcat/mlx/outputs/xdub_len/voice_short6s.wav"
REF_TEXT = "I used to live in the cloud, now I'm running on your Mac, avatar video, dubbing."
WHISPER = "weights/whisper-large-v3-mlx"

# A/B share one normal multi-sentence EN text (3 sentence-chunks).
AB_TEXT = (
    "Good morning everyone, and thank you for joining today. "
    "I want to walk through how this system works and why we built it this way. "
    "We aimed for something fast, reliable, and easy to understand."
)
# C: the historically-degenerate single-sentence case.
C_TEXT = "I think it sounds quite natural."

COVERAGE_THRESHOLD = 0.6
_WORD_RE = re.compile(r"[a-z0-9']+")


def _words(s: str) -> list[str]:
    return _WORD_RE.findall((s or "").lower())


def _transcribe(path: str) -> str:
    return mlx_whisper.transcribe(path, path_or_hf_repo=WHISPER, language="en")["text"].strip()


def _coverage(reference_text: str, transcript: str) -> float:
    """Word-coverage of `reference_text` found in `transcript` (set membership, deduped)."""
    ref = set(_words(reference_text))
    if not ref:
        return 1.0
    heard = set(_words(transcript))
    return len(ref & heard) / len(ref)


class AsrCoverageValidator:
    """validator(audio_np, chunk_text, sr) -> coverage(chunk_text in whisper(audio)) >= 0.6.

    Writes the candidate audio to a temp wav under OUT (NEVER /tmp), transcribes it, and
    returns the boolean. Records every call for honest reporting.
    """

    def __init__(self, tag: str, threshold: float = COVERAGE_THRESHOLD):
        self.tag = tag
        self.threshold = threshold
        self.calls: list[dict] = []
        self._n = 0

    def __call__(self, audio_np: np.ndarray, chunk_text: str, sr: int) -> bool:
        self._n += 1
        tmp = os.path.join(OUT, f"_val_{self.tag}_{self._n:03d}.wav")
        sf.write(tmp, np.asarray(audio_np, dtype=np.float32).ravel(), sr)
        transcript = _transcribe(tmp)
        cov = _coverage(chunk_text, transcript)
        ok = cov >= self.threshold
        self.calls.append(
            {"chunk": chunk_text, "transcript": transcript, "coverage": round(cov, 3), "ok": ok}
        )
        print(f"  [val:{self.tag}] cov={cov:.2f} ok={ok}  '{chunk_text}' -> '{transcript}'",
              flush=True)
        return ok


class FirstAttemptRejectValidator:
    """Stateful: False the FIRST time each distinct chunk is seen, True afterwards.

    Forces every chunk to retry EXACTLY once regardless of model luck, deterministically
    exercising the real-weights reseed-retry path + the validator hook. Health passing is
    a precondition (the guard ANDs chunk_health with the validator), so on a normal render
    each chunk: attempt 0 -> health ok, validator False (reject) -> retry -> attempt 1 ->
    health ok, validator True (accept).
    """

    def __init__(self):
        self.seen: set[str] = set()
        self.calls: list[dict] = []

    def __call__(self, audio_np: np.ndarray, chunk_text: str, sr: int) -> bool:
        first = chunk_text not in self.seen
        self.seen.add(chunk_text)
        accept = not first
        self.calls.append({"chunk": chunk_text, "first_seen": first, "accepted": accept})
        return accept


def _drain():
    mx.synchronize()
    mx.clear_cache()


def main() -> int:
    model = from_pretrained("weights/dots_tts_mlx", dtype=mx.bfloat16).model
    sr = model.sample_rate
    summary: dict = {"ref": REF, "ref_text": REF_TEXT, "coverage_threshold": COVERAGE_THRESHOLD}

    common = dict(prompt_audio=REF, prompt_text=REF_TEXT, num_steps=10,
                  guidance_scale=1.2, speaker_scale=1.5, language="EN", seed=42, gap_ms=80)

    # ----- (A) validator-accepts-good control -----------------------------------------
    print("\n=== (A) control: healthy multi-sentence render + ASR-coverage validator ===",
          flush=True)
    t0 = time.time()
    val_a = AsrCoverageValidator("A")
    out_a = model.generate_long(text=AB_TEXT, retry_degenerate=True, max_retries=2,
                                validator=val_a, **common)
    wav_a = np.asarray(out_a["audio"][0].astype(mx.float32))
    path_a = os.path.join(OUT, "A_control.wav")
    sf.write(path_a, wav_a, sr)
    a_pass = out_a["retries"] == 0 and out_a["unrecovered"] == 0
    summary["A_control"] = {
        "num_chunks": out_a["num_chunks"], "retries": out_a["retries"],
        "unrecovered": out_a["unrecovered"], "audio_s": round(wav_a.size / sr, 2),
        "wall_s": round(time.time() - t0, 1), "wav": path_a,
        "validator_calls": val_a.calls, "expect": "retries=0 unrecovered=0", "pass": a_pass,
    }
    print(f"(A) chunks={out_a['num_chunks']} retries={out_a['retries']} "
          f"unrecovered={out_a['unrecovered']} -> pass={a_pass}", flush=True)
    _drain()

    # ----- (B) mechanism: forced one-retry-per-chunk + attempt-0 differs ---------------
    print("\n=== (B) mechanism: first-attempt-reject validator forces one retry/chunk ===",
          flush=True)
    t0 = time.time()
    # Baseline = attempt-0 audio under the same seed (retry disabled, no validator).
    out_b0 = model.generate_long(text=AB_TEXT, retry_degenerate=False, **common)
    wav_b0 = np.asarray(out_b0["audio"][0].astype(mx.float32))
    sf.write(os.path.join(OUT, "B_attempt0.wav"), wav_b0, sr)
    _drain()
    # Forced-retry: every chunk rejected once, then accepted -> retries == num_chunks.
    rej = FirstAttemptRejectValidator()
    out_b = model.generate_long(text=AB_TEXT, retry_degenerate=True, max_retries=2,
                                validator=rej, **common)
    wav_b = np.asarray(out_b["audio"][0].astype(mx.float32))
    sf.write(os.path.join(OUT, "B_retried.wav"), wav_b, sr)
    n_chunks = out_b["num_chunks"]
    # attempt-0 vs kept-after-retry audio must differ (a real reseed produced new audio).
    # np.array_equal is False if shapes/lengths differ OR any sample differs.
    differ = not np.array_equal(wav_b0, wav_b)
    b_retries_ok = out_b["retries"] == n_chunks
    b_pass = b_retries_ok and differ and out_b["unrecovered"] == 0
    summary["B_mechanism"] = {
        "num_chunks": n_chunks, "retries": out_b["retries"], "unrecovered": out_b["unrecovered"],
        "expect_retries_eq_num_chunks": b_retries_ok,
        "attempt0_samples": int(wav_b0.size), "retried_samples": int(wav_b.size),
        "attempt0_vs_retried_differ": bool(differ),
        "reject_validator_calls": rej.calls, "wall_s": round(time.time() - t0, 1), "pass": b_pass,
    }
    print(f"(B) chunks={n_chunks} retries={out_b['retries']} (==chunks: {b_retries_ok}) "
          f"unrecovered={out_b['unrecovered']} attempt0!=retried: {differ} -> pass={b_pass}",
          flush=True)
    _drain()

    # ----- (C) natural-degeneration probe (honest) ------------------------------------
    print("\n=== (C) natural-degeneration probe: 'I think it sounds quite natural.' seed=42 ===",
          flush=True)
    c_common = dict(prompt_audio=REF, prompt_text=REF_TEXT, num_steps=10, guidance_scale=1.2,
                    speaker_scale=1.5, language="EN", seed=42, gap_ms=80)
    # (C1) WITHOUT a validator -> what the fixed build actually produces.
    t0 = time.time()
    out_c_nov = model.generate_long(text=C_TEXT, retry_degenerate=True, max_retries=3,
                                    validator=None, **c_common)
    wav_c_nov = np.asarray(out_c_nov["audio"][0].astype(mx.float32))
    path_c_nov = os.path.join(OUT, "C_no_validator.wav")
    sf.write(path_c_nov, wav_c_nov, sr)
    _drain()
    tx_c_nov = _transcribe(path_c_nov)
    cov_c_nov = _coverage(C_TEXT, tx_c_nov)
    # (C2) WITH the ASR validator -> recover if it hallucinated.
    val_c = AsrCoverageValidator("C")
    out_c_val = model.generate_long(text=C_TEXT, retry_degenerate=True, max_retries=3,
                                    validator=val_c, **c_common)
    wav_c_val = np.asarray(out_c_val["audio"][0].astype(mx.float32))
    path_c_val = os.path.join(OUT, "C_with_validator.wav")
    sf.write(path_c_val, wav_c_val, sr)
    _drain()
    tx_c_val = _transcribe(path_c_val)
    cov_c_val = _coverage(C_TEXT, tx_c_val)

    no_val_faithful = cov_c_nov >= COVERAGE_THRESHOLD
    if no_val_faithful:
        verdict = ("natural degeneration NOT reproduced on the fixed build "
                   "(no-validator output already faithful) — guard is a latent safety net.")
    elif cov_c_val >= COVERAGE_THRESHOLD:
        verdict = ("HEADLINE: seed 42 hallucinated WITHOUT the validator and was RECOVERED "
                   "with it (no-validator faithless, validator-gated faithful).")
    else:
        verdict = ("seed 42 degenerated and the validator did NOT recover within max_retries "
                   "(both outputs below the coverage threshold).")
    summary["C_natural_probe"] = {
        "text": C_TEXT, "seed": 42, "max_retries": 3,
        "no_validator": {"transcript": tx_c_nov, "coverage": round(cov_c_nov, 3),
                         "faithful": no_val_faithful, "retries": out_c_nov["retries"],
                         "unrecovered": out_c_nov["unrecovered"], "wav": path_c_nov},
        "with_validator": {"transcript": tx_c_val, "coverage": round(cov_c_val, 3),
                           "faithful": cov_c_val >= COVERAGE_THRESHOLD,
                           "retries": out_c_val["retries"],
                           "unrecovered": out_c_val["unrecovered"], "wav": path_c_val,
                           "validator_calls": val_c.calls},
        "wall_s": round(time.time() - t0, 1), "verdict": verdict,
    }
    print(f"(C) no-validator: cov={cov_c_nov:.2f} '{tx_c_nov}'", flush=True)
    print(f"(C) with-validator: cov={cov_c_val:.2f} retries={out_c_val['retries']} "
          f"'{tx_c_val}'", flush=True)
    print(f"(C) VERDICT: {verdict}", flush=True)

    summary["overall_pass"] = bool(a_pass and b_pass)
    with open(os.path.join(OUT, "gate_retry.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== SUMMARY ===", flush=True)
    print(f"(A) retries={out_a['retries']} unrecovered={out_a['unrecovered']} pass={a_pass}",
          flush=True)
    print(f"(B) retries={out_b['retries']}==num_chunks({n_chunks}) "
          f"attempt0!=retried={differ} pass={b_pass}", flush=True)
    print(f"(C) {verdict}", flush=True)
    print(f"overall A&B pass: {summary['overall_pass']}  (JSON: {OUT}/gate_retry.json)",
          flush=True)
    return 0 if summary["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
