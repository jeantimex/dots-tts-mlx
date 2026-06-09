"""Phase-decomposed memory probe for one dots.tts render (mf vs soar, any quant).

Answers "where does the render peak go?" by reading phys_footprint (proc_pid_rusage
RUSAGE_INFO_V2 ri_phys_footprint — the Apple-blessed metric, not ps RSS) + the MLX
allocator peak at each phase: baseline -> after from_pretrained (resident weights) ->
after generate (render high-water). Reset the MLX peak right before generate so the
render peak is isolated from the load spike. One render, one process (clean footprint).

Run ONE config per process:
  uv run python scripts/meanflow_mem_probe.py --model weights/dots_tts_mlx_mf_int4 --label mf_int4 --steps auto
  uv run python scripts/meanflow_mem_probe.py --model weights/dots_tts_mlx_int4   --label soar_int4 --steps 10
"""
from __future__ import annotations

import argparse
import ctypes
import os

import mlx.core as mx

mx.set_memory_limit(int(45 * (1 << 30)))

from dots_tts_mlx.model import DotsTtsModel  # noqa: E402

_REF = "/Users/shraey/.superset/worktrees/longcat/mlx/outputs/xdub_len/voice_short6s.wav"
_REF_TEXT = "I used to live in the cloud, now I'm running on your Mac, avatar video, dubbing."
_TEXT = "I used to live in the cloud, but now I run entirely on your Mac, no internet required."


class _RUsageV2(ctypes.Structure):
    _fields_ = [
        ("ri_uuid", ctypes.c_uint8 * 16),
        ("ri_user_time", ctypes.c_uint64),
        ("ri_system_time", ctypes.c_uint64),
        ("ri_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_interrupt_wkups", ctypes.c_uint64),
        ("ri_pageins", ctypes.c_uint64),
        ("ri_wired_size", ctypes.c_uint64),
        ("ri_resident_size", ctypes.c_uint64),
        ("ri_phys_footprint", ctypes.c_uint64),
        ("ri_proc_start_abstime", ctypes.c_uint64),
        ("ri_proc_exit_abstime", ctypes.c_uint64),
        ("ri_child_user_time", ctypes.c_uint64),
        ("ri_child_system_time", ctypes.c_uint64),
        ("ri_child_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_child_interrupt_wkups", ctypes.c_uint64),
        ("ri_child_pageins", ctypes.c_uint64),
        ("ri_child_elapsed_abstime", ctypes.c_uint64),
        ("ri_diskio_bytesread", ctypes.c_uint64),
        ("ri_diskio_byteswritten", ctypes.c_uint64),
    ]


_libc = ctypes.CDLL("libSystem.dylib", use_errno=True)


def _footprint_gb() -> float:
    info = _RUsageV2()
    rc = _libc.proc_pid_rusage(ctypes.c_int(os.getpid()), ctypes.c_int(2), ctypes.byref(info))
    return info.ri_phys_footprint / (1 << 30) if rc == 0 else -1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--steps", default="auto", help='"auto" -> None (per-mode), or an int')
    args = ap.parse_args()
    steps = None if args.steps == "auto" else int(args.steps)

    base_fp = _footprint_gb()

    model = DotsTtsModel.from_pretrained(args.model, dtype=mx.bfloat16)
    mx.synchronize()
    load_fp = _footprint_gb()
    load_active = mx.get_active_memory() / (1 << 30)
    has_dur = getattr(model.flow_solver.dit, "duration_embedder", None) is not None

    # isolate the render high-water from the load spike
    mx.reset_peak_memory()
    out = model.generate(
        _TEXT, prompt_audio=_REF, prompt_text=_REF_TEXT, language="EN", seed=42, num_steps=steps
    )
    mx.eval(out["audio"])
    mx.synchronize()
    render_peak_mx = mx.get_peak_memory() / (1 << 30)
    end_active = mx.get_active_memory() / (1 << 30)
    end_cache = mx.get_cache_memory() / (1 << 30)
    render_fp = _footprint_gb()
    # is the high footprint MLX's releasable cache pool, or real working set?
    mx.clear_cache()
    mx.synchronize()
    post_clear_fp = _footprint_gb()
    post_clear_active = mx.get_active_memory() / (1 << 30)

    print(
        f"[{args.label}] mode={model.mode} dur_embedder={has_dur} patches={out['num_patches']} "
        f"steps={steps if steps is not None else 'auto'}\n"
        f"    baseline_fp={base_fp:5.2f}GB  load_fp={load_fp:5.2f}GB  load_active_mx={load_active:5.2f}GB\n"
        f"    render_peak_mx={render_peak_mx:5.2f}GB (alloc high-water DURING generate)\n"
        f"    end: active_mx={end_active:5.2f}GB  cache_mx={end_cache:5.2f}GB  phys_fp={render_fp:5.2f}GB\n"
        f"    after clear_cache: active_mx={post_clear_active:5.2f}GB  phys_fp={post_clear_fp:5.2f}GB",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
