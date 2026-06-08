"""Measurement-only: per-phase memory peak of the dots.tts ref-text conditioning path.

Mirrors DotsTtsModel._prepare_prompt_conditioning + _prefill phase-by-phase to show
which phases (x-vector / AudioVAE-encode / patch-encode vs LLM-prefill / decode) drive
the ~10 GB ref-text peak. The artifact-cache feature removes the first three; this
script quantifies the remaining floor. Not part of the runtime.
"""
from __future__ import annotations

import argparse
import math
import os
import resource

import mlx.core as mx

mx.set_memory_limit(int(45 * (1 << 30)))  # ceiling guard — before heavy alloc

import numpy as np  # noqa: E402

from dots_tts_mlx.loader import from_pretrained  # noqa: E402
from dots_tts_mlx.model import _load_wav_mono, _resample, _trim_silence  # noqa: E402


def footprint_gb() -> float:
    # ru_maxrss is bytes on macOS.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 30)


def phase(label: str, fn):
    mx.synchronize()
    mx.reset_peak_memory()
    out = fn()
    if isinstance(out, mx.array):
        mx.eval(out)
    else:
        for o in out if isinstance(out, (list, tuple)) else []:
            if isinstance(o, mx.array):
                mx.eval(o)
    mx.synchronize()
    peak = mx.get_peak_memory() / (1 << 30)
    print(f"[probe] {label:<24} mlx_peak={peak:6.2f}GB  footprint={footprint_gb():6.2f}GB", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("DOTS_TTS_WEIGHTS", "weights/dots_tts_mlx"))
    ap.add_argument("--ref-audio", default=os.environ.get("DOTS_TTS_REF", "reference.wav"))
    ap.add_argument("--ref-text", default="this is the reference transcript")
    ap.add_argument("--text", default="This is a short target sentence for the probe.")
    ap.add_argument("--speaker-scale", type=float, default=1.5)
    args = ap.parse_args()

    m = from_pretrained(args.model, dtype=mx.bfloat16).model
    mx.synchronize()
    print(f"[probe] model loaded             footprint={footprint_gb():6.2f}GB", flush=True)

    raw, sr = _load_wav_mono(args.ref_audio)
    raw = _trim_silence(raw, top_db=30.0)
    a48 = _resample(raw, sr, m.sample_rate)
    chunk = m.patch_size * m.hop_size
    n = a48.shape[-1]
    target = math.ceil(n / chunk) * chunk
    if target > n:
        a48 = np.pad(a48, (0, target - n))
    print(f"[probe] ref seconds = {a48.shape[-1] / m.sample_rate:.2f}s", flush=True)

    # Phase 1: CAM++ x-vector -> g_cond
    def _xvec():
        xv = m._xvector_from_audio48k(a48)
        return m.xvec_proj((xv * args.speaker_scale).astype(m.dtype)).astype(m.dtype)
    phase("x-vector -> g_cond", _xvec)

    # Phase 2: AudioVAE encode -> normalized prompt latents
    def _vae():
        wav = mx.array(a48, dtype=mx.float32)[None, None]
        latent = m.vae.encode(wav)
        sampled = m.io.sample_from_latent(latent)[:, : -m.patch_size]
        return m.io.normalize(sampled)
    normalized = phase("AudioVAE encode", _vae)
    s = normalized.shape[1] // m.patch_size
    flat = normalized[:, : s * m.patch_size]

    # Phase 3: patch-encoder over the prompt latents
    patch_emb = phase("patch-encode (prompt)", lambda: m.patch_encoder(flat).astype(m.dtype))

    # Phase 4: LLM prefill over schedule[:prefill_end] (token embeds + scattered patch_emb)
    ids = m.tokenizer.build_generation_schedule(
        args.text, prompt_text=args.ref_text, language=None, max_audio_tokens=500
    )
    schedule = mx.array(ids, dtype=mx.int32)[None]
    aud = {m.tokenizer.audio_gen_span_id, m.tokenizer.audio_comp_span_id}
    spans = [i for i, t in enumerate(ids) if t in aud]
    prefill_end = spans[s]  # s prompt spans, then first target span

    def _prefill():
        head = schedule[:, :prefill_end]
        emb = m.llm._model.model.embed_tokens(head).astype(m.dtype)
        idx = mx.array(spans[:s], dtype=mx.int32)
        emb[:, idx, :] = patch_emb[:, :s, :]
        cache = m.llm.make_cache()
        h, _ = m.llm.step(inputs_embeds=emb, cache=cache)
        return h
    phase("LLM prefill", _prefill)

    print("\n[probe] INTERPRETATION:")
    print("  artifact-cache eliminates {x-vector, AudioVAE encode, patch-encode}.")
    print("  remaining per-call floor ~= max(LLM prefill, decode) + resident weights.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
