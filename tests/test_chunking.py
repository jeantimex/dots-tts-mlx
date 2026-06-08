from dots_tts_mlx.chunking import resolve_max_chars, split_for_generation


def test_sentence_split_latin():
    t = "Hello there. How are you? I am fine!"
    assert split_for_generation(t, max_chars=240, language="EN") == [
        "Hello there.", "How are you?", "I am fine!"
    ]


def test_sentence_split_devanagari_danda():
    t = "नमस्ते दोस्तों। आप कैसे हैं। धन्यवाद।"
    out = split_for_generation(t, max_chars=160, language="HI")
    assert len(out) == 3
    assert out[-1] == "धन्यवाद।"


def test_sentence_split_cjk():
    t = "大家好。今天怎么样？谢谢！"
    out = split_for_generation(t, max_chars=120, language="ZH")
    assert len(out) == 3


def test_clause_fallback_when_sentence_too_long():
    t = "alpha beta, gamma delta, epsilon zeta, eta theta."
    out = split_for_generation(t, max_chars=20, language="EN")
    assert all(len(c) <= 20 for c in out)
    assert len(out) >= 2


def test_word_fallback_never_breaks_mid_word():
    t = "the quick brown fox jumps over the lazy dog again and again"
    out = split_for_generation(t, max_chars=18, language="EN")
    assert all(len(c) <= 18 for c in out)
    in_words = set(t.split())
    for c in out:
        for w in c.split():
            assert w in in_words


def test_no_punctuation_runon_is_word_split():
    t = "one two three four five six seven eight nine ten eleven twelve"
    out = split_for_generation(t, max_chars=15, language="EN")
    assert all(len(c) <= 15 for c in out)
    assert "".join(out).replace(" ", "") == t.replace(" ", "")


def test_cjk_no_space_grapheme_split_caps_length():
    t = "中" * 50
    out = split_for_generation(t, max_chars=10, language="ZH")
    assert all(len(c) <= 10 for c in out)
    assert "".join(out) == t


def test_devanagari_grapheme_safe_keeps_combining_marks():
    import unicodedata
    t = "नमस्ते" * 20  # no spaces, no danda
    out = split_for_generation(t, max_chars=8, language="HI")
    assert "".join(out) == t
    for c in out:
        assert not unicodedata.category(c[0]).startswith("M"), "chunk starts mid-grapheme"


def test_resolve_max_chars_script_aware():
    assert resolve_max_chars("hello", language="EN") == 240
    assert resolve_max_chars("大家好", language="ZH") == 120
    assert resolve_max_chars("नमस्ते", language="HI") == 160
    assert resolve_max_chars("大家好") == 120
    assert resolve_max_chars("नमस्ते") == 160
    assert resolve_max_chars("hello world") == 240


def test_empty_text_returns_empty():
    assert split_for_generation("   ", max_chars=240, language="EN") == []
