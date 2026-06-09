"""Exhaustive validation matrix for the in-context prompt-embedding fix.

Serial sweep (ONE GPU job) over: language x text-length x chunking x reference-mode x
reference-length x seed. Per cell it records wall time, peak MLX memory, audio duration,
RMS std (silence/finiteness), x-vector cosine to the reference, and (second pass) a
Whisper transcript + word coverage (adherence). Drain barrier between cells (Metal
stability); RESUMABLE — each cell's wav + a JSONL row are written as it finishes, and a
rerun skips cells whose wav already exists.

Run (mlx_whisper pulled in ephemerally; no package dep change):
    cd ~/dots-tts-mlx-prompt-scale
    uv run --with mlx_whisper python scripts/eval_matrix_prompt_scale.py

Outputs -> monorepo outputs/dots_tts/eval_matrix/ (wavs + results.jsonl + summary.json).
"""
from __future__ import annotations

import gc
import json
import os
import re
import time
import unicodedata

import mlx.core as mx
import numpy as np
import soundfile as sf

mx.set_memory_limit(int(45 * (1 << 30)))

from dots_tts_mlx.loader import from_pretrained  # noqa: E402

BASE = "/Users/shraey/.superset/worktrees/longcat/mlx"
OUT = f"{BASE}/outputs/dots_tts/eval_matrix"
os.makedirs(OUT, exist_ok=True)
JSONL = f"{OUT}/results.jsonl"
WEIGHTS = "weights/dots_tts_mlx"
WHISPER = "weights/whisper-large-v3-mlx"

REFS = {
    "short": (f"{BASE}/outputs/xdub_len/voice_short6s.wav",
              "I used to live in the cloud, now I'm running on your Mac, avatar video, dubbing."),
    "long": (f"{BASE}/outputs/xdub_len/voice_long14s.wav",
             "I used to live in the cloud, now I'm running on your Mac. Avatar video, dubbing, "
             "lip sync, and voiceovers fully on device. Coming soon to your Mac. I used to live "
             "in the cloud, now I'm"),
}

# (text, whisper_lang_code) by language + length.
TEXTS = {
    ("EN", "short"): ("Thank you for trying this out.", "en"),
    ("EN", "med"): ("Thank you for trying this out. I think you will be surprised by how "
                    "natural it sounds. Tell me what you think.", "en"),
    ("EN", "long"): ("Let me walk you through the whole idea. A few years ago this would have "
                     "been unthinkable on a laptop. Today the voice you hear is generated "
                     "entirely on this Mac, with no internet at all. It is fast, it is private, "
                     "and it runs offline. I hope it sounds natural to you.", "en"),
    ("ES", "med"): ("Gracias por probar esto. Creo que te sorprenderá lo natural que suena.",
                    "es"),
    ("DE", "med"): ("Danke, dass du das ausprobierst. Ich glaube, du wirst überrascht sein, "
                    "wie natürlich es klingt.", "de"),
    ("FR", "med"): ("Merci d'essayer ceci. Je pense que vous serez surpris de voir à quel "
                    "point cela semble naturel.", "fr"),
    ("ZH", "med"): ("感谢你试用这个。我想你会惊讶于它听起来有多自然。", "zh"),
    ("HI", "med"): ("इसे आज़माने के लिए धन्यवाद। मुझे लगता है कि आप यह सुनकर हैरान होंगे कि यह "
                    "कितना स्वाभाविक लगता है।", "hi"),
}

COMMON = dict(num_steps=10, guidance_scale=1.2, speaker_scale=1.5)


def build_cells() -> list[dict]:
    cells: dict[tuple, dict] = {}

    def add(lang, length, refmode, reflen, chunk, seed):
        key = (lang, length, refmode, reflen, chunk, seed)
        cid = f"{lang}_{length}_{refmode}_{reflen}_{chunk}_s{seed}"
        cells[key] = dict(id=cid, lang=lang, length=length, refmode=refmode,
                          reflen=reflen, chunk=chunk, seed=seed)

    # A) LANGUAGE x CHUNK — in-context short ref, med text, seed 42.
    for lang in ["EN", "ES", "DE", "FR", "ZH", "HI"]:
        for chunk in ["single", "long"]:
            add(lang, "med", "incontext", "short", chunk, 42)
    # B) REF-MODE x REF-LEN x CHUNK — EN med, seed 42.
    for refmode in ["xvector", "incontext"]:
        for reflen in ["short", "long"]:
            for chunk in ["single", "long"]:
                add("EN", "med", refmode, reflen, chunk, 42)
    # C) LENGTH x CHUNK x SEED — EN in-context short.
    for length in ["short", "med", "long"]:
        for chunk in ["single", "long"]:
            for seed in [0, 42, 7]:
                add("EN", length, "incontext", "short", chunk, seed)
    # D) SEED ROBUSTNESS — EN + HI in-context short, single, med, many seeds.
    for lang in ["EN", "HI"]:
        for seed in [0, 1, 2, 3, 42]:
            add(lang, "med", "incontext", "short", "single", seed)

    return list(cells.values())


def _xvec(model, wav48k: np.ndarray) -> np.ndarray:
    v = np.asarray(model._xvector_from_audio48k(wav48k).astype(mx.float32)).ravel()
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _norm_words(s: str) -> list[str]:
    return re.findall(r"[\w']+", s.lower(), flags=re.UNICODE)


def _is_cjk_heavy(s: str) -> bool:
    chars = [c for c in s if not c.isspace()]
    if not chars:
        return False
    cjk = sum(1 for c in chars if "㐀" <= c <= "鿿" or "぀" <= c <= "ヿ")
    return cjk / len(chars) > 0.3


def _coverage(inp: str, hyp: str) -> float:
    # CJK / no-space scripts: word tokenization is meaningless -> character-set recall.
    if _is_cjk_heavy(inp):
        ic = [c for c in inp if not c.isspace() and not unicodedata.category(c).startswith("P")]
        hc = {c for c in hyp if not c.isspace()}
        return round(sum(1 for c in ic if c in hc) / max(len(ic), 1), 3)
    iw = _norm_words(inp)
    if not iw:
        return 1.0
    hw = set(_norm_words(hyp))
    return round(sum(1 for w in iw if w in hw) / len(iw), 3)


def main() -> int:
    cells = build_cells()
    done = set()
    if os.path.exists(JSONL):
        with open(JSONL) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except Exception:  # noqa: BLE001
                    pass
    todo = [c for c in cells if c["id"] not in done or
            not os.path.exists(f"{OUT}/{c['id']}.wav")]
    print(f"[matrix] {len(cells)} cells total, {len(done)} already done, {len(todo)} to run",
          flush=True)

    model = from_pretrained(WEIGHTS, dtype=mx.bfloat16).model
    sr = model.sample_rate

    # cache reference x-vectors (load each ref at 48k once).
    from dots_tts_mlx.model import _load_prompt_audio48k
    ref_wav = {k: _load_prompt_audio48k(p, sr) for k, (p, _t) in REFS.items()}
    ref_xv = {k: _xvec(model, w) for k, w in ref_wav.items()}

    for i, c in enumerate(todo):
        text, _wl = TEXTS[(c["lang"], c["length"])]
        ref_path, ref_text = REFS[c["reflen"]]
        kw = dict(text=text, prompt_audio=ref_path, language=c["lang"], seed=c["seed"], **COMMON)
        if c["refmode"] == "incontext":
            kw["prompt_text"] = ref_text
        row = dict(c, audio_s=None, wall_s=None, peak_gb=None, std=None,
                   xvec_cos=None, chunks=None, error=None)
        try:
            mx.reset_peak_memory()
            t0 = time.time()
            if c["chunk"] == "long":
                out = model.generate_long(gap_ms=80, **kw)
                row["chunks"] = int(out.get("num_chunks", 1))
            else:
                out = model.generate(**kw)
                row["chunks"] = 1
            mx.eval(out["audio"])
            row["wall_s"] = round(time.time() - t0, 1)
            row["peak_gb"] = round(mx.get_peak_memory() / (1 << 30), 2)
            w = np.asarray(out["audio"][0].astype(mx.float32))
            sf.write(f"{OUT}/{c['id']}.wav", w, sr)
            row["audio_s"] = round(w.size / sr, 2)
            row["std"] = round(float(np.std(w)), 4)
            row["finite"] = bool(np.all(np.isfinite(w)))
            row["xvec_cos"] = round(float(np.dot(_xvec(model, w), ref_xv[c["reflen"]])), 4)
        except Exception as e:  # noqa: BLE001
            row["error"] = f"{type(e).__name__}: {e}"
        with open(JSONL, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[{i+1}/{len(todo)}] {c['id']}: {row.get('audio_s')}s "
              f"{row.get('wall_s')}s {row.get('peak_gb')}GB cos={row.get('xvec_cos')} "
              f"chunks={row.get('chunks')} err={row.get('error')}", flush=True)
        mx.synchronize()
        mx.clear_cache()
        gc.collect()

    # free the TTS model before whisper.
    del model
    mx.synchronize()
    mx.clear_cache()
    gc.collect()

    # transcription pass (idempotent): fill transcript + coverage for any row missing it.
    rows = []
    with open(JSONL) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:  # noqa: BLE001
                pass
    # keep the latest row per id.
    latest = {r["id"]: r for r in rows}
    import mlx_whisper
    print("\n[matrix] transcription pass", flush=True)
    for r in latest.values():
        if r.get("error") or r.get("transcript") is not None:
            continue
        wav = f"{OUT}/{r['id']}.wav"
        if not os.path.exists(wav):
            continue
        text, wl = TEXTS[(r["lang"], r["length"])]
        try:
            tr = mlx_whisper.transcribe(wav, path_or_hf_repo=WHISPER, language=wl)["text"].strip()
        except Exception as e:  # noqa: BLE001
            tr = f"<whisper error: {e}>"
        r["transcript"] = unicodedata.normalize("NFC", tr)
        r["coverage"] = _coverage(text, tr)
        print(f"  {r['id']}: cov={r['coverage']} :: {tr[:80]}", flush=True)

    summary = sorted(latest.values(), key=lambda r: r["id"])
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # flags
    flags = []
    for r in summary:
        if r.get("error"):
            flags.append((r["id"], f"ERROR {r['error']}"))
        elif r.get("std") is not None and r["std"] < 0.01:
            flags.append((r["id"], f"SILENT std={r['std']}"))
        elif r.get("finite") is False:
            flags.append((r["id"], "NON-FINITE"))
        elif r.get("coverage") is not None and r["coverage"] < 0.6:
            flags.append((r["id"], f"LOW-COVERAGE {r['coverage']}"))
    print("\n=== FLAGS ===", flush=True)
    if flags:
        for cid, why in flags:
            print(f"  [FLAG] {cid}: {why}", flush=True)
    else:
        print("  none — all cells produced finite, non-silent, well-covered audio", flush=True)
    print(f"\n[matrix] wrote {OUT}/summary.json ({len(summary)} cells, {len(flags)} flags)",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
