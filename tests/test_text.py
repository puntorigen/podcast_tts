from podcast_tts.text import (
    normalize_text,
    prepare_text_for_conversion,
    remove_brackets,
    spell_numbers,
)


def test_spell_numbers_english():
    assert "three" in spell_numbers("I have 3 cats", "en")


def test_spell_numbers_spanish():
    assert "tres" in spell_numbers("Tengo 3 gatos", "es")


def test_spell_numbers_ignores_bracketed():
    # numbers inside tags must not be converted
    assert "[uv_break]" in spell_numbers("hi [uv_break]", "en")


def test_normalize_with_tags_keeps_reserved():
    out = normalize_text("Great [laugh] stuff", "en", use_tags=True)
    assert "[laugh]" in out


def test_normalize_without_tags_strips_brackets():
    out = normalize_text("Hola [uv_break] mundo", "es", use_tags=False)
    assert "[" not in out and "uv_break" not in out
    assert "Hola" in out and "mundo" in out


def test_remove_brackets_preserves_reserved():
    out = remove_brackets("keep [uv_break] drop [stage direction]")
    assert "[uv_break]" in out
    assert "stage direction" not in out


def test_prepare_text_returns_batches():
    text = (
        "Hello world. This is a longer sentence used for chunking here. "
        "Short. Another reasonably long sentence follows right now."
    )
    batches = prepare_text_for_conversion(text, "en", use_tags=True, merge_size=2)
    assert isinstance(batches, list)
    assert all(isinstance(batch, list) for batch in batches)
    assert all(len(batch) <= 2 for batch in batches)


def test_prepare_text_plain_has_no_break_tags():
    batches = prepare_text_for_conversion("Uno. Dos. Tres.", "es", use_tags=False)
    joined = " ".join(line for batch in batches for line in batch)
    assert "[uv_break]" not in joined
