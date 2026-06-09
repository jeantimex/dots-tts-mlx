"""Long-form (~30s) chunked benchmark: soar-10 vs meanflow-NFE4, via generate_long.

For each (language, conditioning) cell, render the SAME ~30s script with both models
through generate_long (sentence chunking — the real long-text path) and report wall
time, audio duration, realtime factor, chunk count, retries, and the soar->mf speedup.
Conditioning: "ref" (prompt_audio + prompt_text) and "xvec" (prompt_audio only, x-vector
clone). A 35s drain+sleep cooldown after EVERY render to avoid thermal throttling.

Pure-MLX. Run serially (one GPU job). Writes wavs to outputs/dots_tts/meanflow_long/.
"""
from __future__ import annotations

import argparse
import gc
import time

import mlx.core as mx

mx.set_memory_limit(int(45 * (1 << 30)))

import os  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from dots_tts_mlx.model import DotsTtsModel  # noqa: E402

_REF = "/Users/shraey/.superset/worktrees/longcat/mlx/outputs/xdub_len/voice_short6s.wav"
_REF_TEXT = "I used to live in the cloud, now I'm running on your Mac, avatar video, dubbing."

# ~30s scripts (multi-sentence, so generate_long chunks them).
_TEXTS = {
    "EN": (
        "I used to live in the cloud, running on a giant server farm somewhere far away. "
        "Now I run entirely on your Mac, with no internet connection required at all. "
        "I can create talking avatars, dub videos into other languages, and clone a voice "
        "from just a few seconds of audio. Everything happens locally, so your data never "
        "leaves your device. It is faster than you would expect, and honestly, it keeps "
        "getting better every single week. You can run it on a laptop, on a plane, or "
        "anywhere with no signal at all. And because nothing is ever sent to a server, it "
        "stays completely private, which is exactly how it should be."
    ),
    "HI": (
        "मैं पहले क्लाउड में रहता था, किसी दूर के बहुत बड़े सर्वर पर चलता था। "
        "लेकिन अब मैं पूरी तरह आपके मैक पर चलता हूँ, और मुझे इंटरनेट की कोई ज़रूरत नहीं है। "
        "मैं बात करते हुए अवतार बना सकता हूँ, वीडियो को दूसरी भाषाओं में डब कर सकता हूँ, "
        "और कुछ ही सेकंड की आवाज़ से किसी की नकल बना सकता हूँ। "
        "सब कुछ आपके डिवाइस पर ही होता है, इसलिए आपका डेटा कभी बाहर नहीं जाता। "
        "यह आपकी सोच से कहीं ज़्यादा तेज़ है, और हर हफ़्ते बेहतर होता जा रहा है। "
        "आप इसे लैपटॉप पर, हवाई जहाज़ में, या कहीं भी बिना सिग्नल के चला सकते हैं। "
        "और चूँकि कुछ भी सर्वर पर नहीं भेजा जाता, यह पूरी तरह निजी रहता है, जैसा कि होना चाहिए।"
    ),
    "ZH": (
        "我以前住在云端，在很远的地方一个巨大的服务器上运行。"
        "但是现在，我完全在你的 Mac 上运行，根本不需要连接互联网。"
        "我可以制作会说话的虚拟形象，把视频配音成其他语言，还能用几秒钟的声音克隆出一个人的嗓音。"
        "所有的处理都在本地完成，所以你的数据永远不会离开你的设备。"
        "它比你想象的要快得多，而且老实说，它每一周都在变得更好。"
        "你可以在笔记本电脑上、在飞机上，或者在任何没有信号的地方运行它。"
        "而且因为没有任何数据被发送到服务器，它会完全保持私密，这正是它应该有的样子。"
    ),
}


def _cooldown(seconds: float) -> None:
    mx.synchronize()
    mx.clear_cache()
    gc.collect()
    time.sleep(seconds)


def _render(model, text, lang, cond, *, ref, ref_text, num_steps, seed):
    """One generate_long render; returns (audio_np, sr, info)."""
    kw = {"language": lang, "seed": seed, "num_steps": num_steps, "gap_ms": 80}
    if cond == "ref":
        kw["prompt_audio"] = ref
        kw["prompt_text"] = ref_text
    else:  # xvec — x-vector-only clone (no prompt transcript)
        kw["prompt_audio"] = ref
        kw["prompt_text"] = None
    t0 = time.perf_counter()
    out = model.generate_long(text, **kw)
    mx.eval(out["audio"])
    mx.synchronize()
    gen_s = time.perf_counter() - t0
    audio = np.asarray(out["audio"][0].astype(mx.float32)).ravel()
    sr = int(out["sample_rate"])
    info = {
        "gen_s": gen_s,
        "audio_s": audio.shape[-1] / sr,
        "chunks": int(out.get("num_chunks", -1)),
        "patches": int(out.get("num_patches", -1)),
        "retries": int(out.get("retries", 0)),
        "unrecovered": int(out.get("unrecovered", 0)),
    }
    return audio, sr, info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--soar-model", default="weights/dots_tts_mlx_bf16")
    ap.add_argument("--mf-model", default="weights/dots_tts_mlx_mf_bf16")
    ap.add_argument("--ref-audio", default=_REF)
    ap.add_argument("--ref-text", default=_REF_TEXT)
    ap.add_argument("--out", default="outputs/dots_tts/meanflow_long")
    ap.add_argument("--cooldown", type=float, default=35.0)
    ap.add_argument(
        "--configs",
        default="EN:ref,EN:xvec,HI:ref,HI:xvec,ZH:ref,ZH:xvec",
        help="comma-sep lang:cond cells (cond in {ref,xvec})",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    ref, ref_text = args.ref_audio, args.ref_text
    os.makedirs(args.out, exist_ok=True)
    cells = [tuple(c.split(":")) for c in args.configs.split(",")]

    print(f"[long] loading soar={args.soar_model} + mf={args.mf_model}", flush=True)
    soar = DotsTtsModel.from_pretrained(args.soar_model, dtype=mx.bfloat16)
    mf = DotsTtsModel.from_pretrained(args.mf_model, dtype=mx.bfloat16)
    assert soar.mode == "flow_matching", soar.mode
    assert mf.mode == "meanflow", mf.mode
    print(f"[long] soar.mode={soar.mode}  mf.mode={mf.mode}", flush=True)

    # warmup each model once (untimed) so the first timed render isn't paying compile.
    print("[long] warmup...", flush=True)
    for m, ns in ((soar, 10), (mf, None)):
        m.generate("Hello there, this is a short warmup.", prompt_audio=_REF,
                   prompt_text=_REF_TEXT, language="EN", seed=args.seed, num_steps=ns)
    _cooldown(args.cooldown)

    rows = []
    for lang, cond in cells:
        for label, model, ns in (("soar", soar, 10), ("mf", mf, None)):
            tag = f"{lang}_{cond}_{label}"
            print(f"[long] rendering {tag} ...", flush=True)
            audio, sr, info = _render(
                model, _TEXTS[lang], lang, cond, ref=ref, ref_text=ref_text,
                num_steps=ns, seed=args.seed,
            )
            path = os.path.join(args.out, f"{tag}.wav")
            sf.write(path, audio, sr)
            rows.append({"lang": lang, "cond": cond, "model": label, "path": path, **info})
            rtf = info["gen_s"] / max(info["audio_s"], 1e-6)
            print(
                f"[long]   {tag}: gen={info['gen_s']:.1f}s audio={info['audio_s']:.1f}s "
                f"RTF={rtf:.2f} chunks={info['chunks']} patches={info['patches']} "
                f"retries={info['retries']} -> {path}  peak={mx.get_peak_memory()/(1<<30):.1f}GB",
                flush=True,
            )
            _cooldown(args.cooldown)

    # paired table
    print("\n================ LONG-FORM RESULTS (soar-10 vs mf-NFE4, chunked) ================")
    hdr = f"{'cell':14s} {'soar_s':>8s} {'mf_s':>8s} {'speedup':>8s} {'audio_s(soar/mf)':>18s} {'chunks':>8s}"
    print(hdr)
    by = {}
    for r in rows:
        by[(r["lang"], r["cond"], r["model"])] = r
    for lang, cond in cells:
        s = by.get((lang, cond, "soar"))
        m = by.get((lang, cond, "mf"))
        if not s or not m:
            continue
        spd = s["gen_s"] / max(m["gen_s"], 1e-6)
        print(
            f"{lang+'/'+cond:14s} {s['gen_s']:8.1f} {m['gen_s']:8.1f} {spd:7.2f}x "
            f"{s['audio_s']:8.1f}/{m['audio_s']:<8.1f} {s['chunks']}/{m['chunks']:<5}"
        )
    print("\nwavs:")
    for r in rows:
        print(f"  {os.path.abspath(r['path'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
