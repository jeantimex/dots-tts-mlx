import pathlib

import pytest

TK = pathlib.Path("weights/dots_tts_mlx/tokenizer/tokenizer.json")
pytestmark = pytest.mark.skipif(not TK.exists(), reason="tokenizer absent")


def _tok():
    from dots_tts_mlx.text import DotsTokenizer

    return DotsTokenizer.from_pretrained("weights/dots_tts_mlx/tokenizer")


def test_special_token_ids():
    tk = _tok()
    # Verified against the converted tokenizer (Task 10).
    assert tk.audio_gen_start_id == 151668
    assert tk.audio_gen_span_id == 151669
    assert tk.audio_gen_end_id == 151670
    assert tk.text_cond_end_id == 151671
    assert tk.audio_comp_span_id == 151666


def test_schedule_structure():
    tk = _tok()
    n = 32
    sched = tk.build_generation_schedule(
        "The quick brown fox jumps over the lazy dog.",
        prompt_text="hello world",
        max_audio_tokens=n,
    )
    # exactly n audio_gen_span placeholders.
    assert sum(1 for t in sched if t == tk.audio_gen_span_id) == n
    # exactly one audio_gen_start, immediately before the first span run.
    assert sched.count(tk.audio_gen_start_id) == 1
    start_idx = sched.index(tk.audio_gen_start_id)
    # everything after the start is a contiguous run of spans.
    assert sched[start_idx + 1 :] == [tk.audio_gen_span_id] * n
    # literals "[文本]" precede the text; the audio prefix precedes the start token.
    lit_text = tk.encode("[文本]")
    assert sched[: len(lit_text)] == lit_text
    audio_prefix = tk.encode("[文本对应语音]")
    assert sched[start_idx - len(audio_prefix) : start_idx] == audio_prefix


def test_schedule_no_prompt_text():
    tk = _tok()
    sched = tk.build_generation_schedule("hello", max_audio_tokens=5)
    assert sum(1 for t in sched if t == tk.audio_gen_span_id) == 5
    assert tk.audio_gen_start_id in sched
