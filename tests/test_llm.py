import pathlib

import numpy as np
import pytest

import mlx.core as mx

F = pathlib.Path("tests/fixtures/dots_tts/llm_step.npz")
W = pathlib.Path("weights/dots_tts_mlx/core.safetensors")
pytestmark = pytest.mark.skipif(
    not (F.exists() and W.exists()), reason="absent"
)


def _cos(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def test_llm_hidden_and_eos():
    from dots_tts_mlx.llm import DotsLLM

    d = np.load(F)
    llm = DotsLLM.from_core(
        str(W), "weights/dots_tts_mlx/llm_config.json", dtype=mx.float32
    )
    h_ids = np.asarray(llm.step(input_ids=mx.array(d["ids"]))[0].astype(mx.float32))
    assert _cos(h_ids, d["h_ids"]) >= 0.9999
    assert np.max(np.abs(h_ids - d["h_ids"])) <= 5e-2
    h_emb = np.asarray(
        llm.step(inputs_embeds=mx.array(d["embeds"]))[0].astype(mx.float32)
    )
    assert _cos(h_emb, d["h_emb"]) >= 0.9999
    assert np.max(np.abs(h_emb - d["h_emb"])) <= 5e-2
    eos = np.asarray(llm.eos_logits(mx.array(d["h_ids"])).astype(mx.float32))
    # The eos head produces LARGE logits (range ~[-10, 16.5]) because of the SiLU
    # on the 1536-d hidden, so the absolute max-abs (~5e-2) is pure tf32 matmul
    # rounding on big activations (rel-error ~0.5%, cosine = 1.0), NOT a structural
    # bug. Gate on cosine per the task's alternative ("or cosine>=0.999"). The
    # decode-relevant quantity softmax(eos)[...,1] matches the oracle to ~1e-11.
    assert _cos(eos, d["eos_logits"]) >= 0.999


def test_kv_cache_incremental_equals_full():
    from dots_tts_mlx.llm import DotsLLM

    d = np.load(F)
    ids = mx.array(d["ids"])
    llm = DotsLLM.from_core(
        str(W), "weights/dots_tts_mlx/llm_config.json", dtype=mx.float32
    )
    cache = llm.make_cache()
    hs = [
        llm.step(input_ids=ids[:, i : i + 1], cache=cache)[0]
        for i in range(ids.shape[1])
    ]
    inc = mx.concatenate(hs, axis=1)
    full = llm.step(input_ids=ids)[0]
    assert _cos(np.asarray(inc), np.asarray(full)) >= 0.9999


def test_kv_cache_chunked_prefill_equals_full():
    """T8/T10 regression: prefill L>1 then a L>1 chunk decode on the SAME cache.

    The AR orchestration (T10) prefills the schedule head in one multi-token forward,
    then consumes text runs as multi-token chunks against the same incremental cache.
    Assert that a chunked (prefill + chunk) forward equals a single full forward.
    """
    from dots_tts_mlx.llm import DotsLLM

    d = np.load(F)
    ids = mx.array(d["ids"])
    n = ids.shape[1]
    split = n // 2
    llm = DotsLLM.from_core(
        str(W), "weights/dots_tts_mlx/llm_config.json", dtype=mx.float32
    )
    cache = llm.make_cache()
    h0 = llm.step(input_ids=ids[:, :split], cache=cache)[0]  # prefill, L>1
    h1 = llm.step(input_ids=ids[:, split:], cache=cache)[0]  # chunk decode, L>1
    chunked = mx.concatenate([h0, h1], axis=1)
    full = llm.step(input_ids=ids)[0]
    assert _cos(np.asarray(chunked), np.asarray(full)) >= 0.9999


def test_kv_cache_alternating_ids_and_embeds():
    """T8/T10 regression: alternating step(input_ids=) -> step(inputs_embeds=).

    The decode loop alternates token-id steps (text runs) with re-encoded embedding
    steps (the patch-encoder feedback) on ONE cache. Assert this matches feeding the
    same effective embedding sequence in a single full forward.
    """
    from dots_tts_mlx.llm import DotsLLM

    llm = DotsLLM.from_core(
        str(W), "weights/dots_tts_mlx/llm_config.json", dtype=mx.float32
    )
    d = np.load(F)
    ids = mx.array(d["ids"])[:, :4]
    embed_tokens = llm._model.model.embed_tokens
    e0 = embed_tokens(ids[:, :2])  # first two as embeddings
    e1 = embed_tokens(ids[:, 2:4])  # next two as embeddings

    cache = llm.make_cache()
    # Alternate: ids path, then embeds path, on the same cache.
    ha = llm.step(input_ids=ids[:, :2], cache=cache)[0]
    hb = llm.step(inputs_embeds=e1, cache=cache)[0]
    alt = mx.concatenate([ha, hb], axis=1)

    full = llm.step(inputs_embeds=mx.concatenate([e0, e1], axis=1))[0]
    assert _cos(np.asarray(alt), np.asarray(full)) >= 0.9999
