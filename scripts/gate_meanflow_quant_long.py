"""Long-form (~30s) MeanFlow quant comparison: mf-int8 vs mf-int4, ref + x-vector.

Renders the SAME ~30s scripts + seed as gate_meanflow_long.py (bf16) so int8/int4 are
directly A/B-able against the bf16 long-form clips. MeanFlow only (NFE-4), chunked
(generate_long), EN + ZH, reference + x-vector-only. 35s cooldown after every render.
Pure-MLX, one GPU job. Writes to outputs/dots_tts/meanflow_quant_long/.
"""
from __future__ import annotations

import gc
import os
import time

import mlx.core as mx

mx.set_memory_limit(int(45 * (1 << 30)))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from dots_tts_mlx.model import DotsTtsModel  # noqa: E402

_REF = "/Users/shraey/.superset/worktrees/longcat/mlx/outputs/xdub_len/voice_short6s.wav"
_REF_TEXT = "I used to live in the cloud, now I'm running on your Mac, avatar video, dubbing."
_OUT = "outputs/dots_tts/meanflow_quant_long"
_COOLDOWN = 35.0
_SEED = 42

_VARIANTS = {"int8": "weights/dots_tts_mlx_mf_int8", "int4": "weights/dots_tts_mlx_mf_int4"}
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
_LANGS = ["EN", "ZH"]
_CONDS = ["ref", "xvec"]


def _cooldown():
    mx.synchronize()
    mx.clear_cache()
    gc.collect()
    time.sleep(_COOLDOWN)


def main() -> int:
    os.makedirs(_OUT, exist_ok=True)
    rows = []
    for variant, model_dir in _VARIANTS.items():
        print(f"[qlong] loading {variant} = {model_dir}", flush=True)
        model = DotsTtsModel.from_pretrained(model_dir, dtype=mx.bfloat16)
        assert model.mode == "meanflow", f"{variant}: {model.mode}"
        # warmup (untimed)
        model.generate("Hello there, a short warmup.", prompt_audio=_REF,
                        prompt_text=_REF_TEXT, language="EN", seed=_SEED, num_steps=None)
        _cooldown()
        for lang in _LANGS:
            for cond in _CONDS:
                kw = {"language": lang, "seed": _SEED, "num_steps": None, "gap_ms": 80,
                      "prompt_audio": _REF}
                kw["prompt_text"] = _REF_TEXT if cond == "ref" else None
                tag = f"{lang}_{cond}_{variant}"
                print(f"[qlong] rendering {tag} ...", flush=True)
                t0 = time.perf_counter()
                out = model.generate_long(_TEXTS[lang], **kw)
                mx.eval(out["audio"])
                mx.synchronize()
                gen_s = time.perf_counter() - t0
                audio = np.asarray(out["audio"][0].astype(mx.float32)).ravel()
                sr = int(out["sample_rate"])
                path = os.path.join(_OUT, f"{tag}.wav")
                sf.write(path, audio, sr)
                info = {
                    "variant": variant, "lang": lang, "cond": cond, "path": path,
                    "gen_s": gen_s, "audio_s": audio.shape[-1] / sr,
                    "chunks": int(out.get("num_chunks", -1)),
                    "patches": int(out.get("num_patches", -1)),
                    "retries": int(out.get("retries", 0)),
                }
                rows.append(info)
                print(
                    f"[qlong]   {tag}: gen={gen_s:.1f}s audio={info['audio_s']:.1f}s "
                    f"RTF={gen_s/max(info['audio_s'],1e-6):.2f} chunks={info['chunks']} "
                    f"patches={info['patches']} retries={info['retries']} "
                    f"peak={mx.get_peak_memory()/(1<<30):.1f}GB -> {path}",
                    flush=True,
                )
                _cooldown()
        del model
        mx.clear_cache()
        gc.collect()

    print("\n========== MEANFLOW QUANT LONG-FORM (int8 vs int4, ref + x-vec) ==========")
    print(f"{'cell':18s} {'gen_s':>7s} {'audio_s':>8s} {'RTF':>6s} {'chunks':>7s} {'retries':>8s}")
    for r in rows:
        print(f"{r['lang']+'/'+r['cond']+'/'+r['variant']:18s} {r['gen_s']:7.1f} "
              f"{r['audio_s']:8.1f} {r['gen_s']/max(r['audio_s'],1e-6):6.2f} "
              f"{r['chunks']:7d} {r['retries']:8d}")
    print("\nwavs:")
    for r in rows:
        print(f"  {os.path.abspath(r['path'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
