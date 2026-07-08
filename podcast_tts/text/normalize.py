"""Text normalization shared across engines.

Two normalization styles are supported:

* ``use_tags=True``  -> ChatTTS style. Preserves prosody tags such as
  ``[uv_break]`` and ``[laugh]`` and converts stray punctuation into breaks.
* ``use_tags=False`` -> plain style for engines (Chatterbox, Kokoro) that read
  punctuation naturally and would otherwise speak ``[uv_break]`` out loud.

Number-to-words is language aware (English and Spanish) via ``num2words``.
"""

from __future__ import annotations

import re

import regex

# Prosody tags understood by ChatTTS. These survive bracket cleanup.
RESERVED_TAGS: frozenset[str] = frozenset(
    {
        "uv_break",
        "laugh",
        "laugh_1",
        "laugh_2",
        "laugh_3",
        "laugh_4",
        "laugh_5",
        "lbreak",
        "break",
    }
)


def _num2words_lang(language: str) -> str:
    """Map a language code to a ``num2words`` language, defaulting to English."""
    code = (language or "en").split("-")[0].lower()
    return "es" if code == "es" else "en"


def spell_numbers(text: str, language: str = "en") -> str:
    """Replace bare integers with their spelled-out form in ``language``.

    Numbers already inside square-bracket tags are left untouched. If
    ``num2words`` is unavailable the original text is returned unchanged.
    """
    try:
        from num2words import num2words
    except ImportError:  # pragma: no cover - optional at runtime
        return text

    lang = _num2words_lang(language)

    def _replace(match: re.Match[str]) -> str:
        return " " + num2words(int(match.group(0)), lang=lang) + " "

    # Variable-length look-behind requires the `regex` module.
    return regex.sub(r"(?<!\[.*)\b\d+\b(?!.*\])", _replace, text)


def remove_brackets(text: str, reserved: frozenset[str] = RESERVED_TAGS) -> str:
    """Remove bracketed content except reserved prosody tags."""
    pattern = r"\[(?!\b(?:" + "|".join(reserved) + r")\b)(.*?)\]"
    text = re.sub(pattern, "", text)
    return re.sub(r"\s+", " ", text).strip()


def strip_all_tags(text: str) -> str:
    """Remove every ``[...]`` bracket (used for engines without prosody tags)."""
    text = re.sub(r"\[[^\]]*\]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_text(text: str, language: str = "en", use_tags: bool = True) -> str:
    """Normalize ``text`` for synthesis.

    Args:
        text: Raw input text.
        language: Language code (``"en"`` or ``"es"``) used for number spelling.
        use_tags: When ``True`` keep/insert ChatTTS prosody tags; when ``False``
            strip all bracket tags and keep punctuation as-is.
    """
    result = spell_numbers(text, language)

    if use_tags:
        # Convert stray punctuation into a break, but avoid stacking breaks near
        # an existing one.
        proximity_source = result

        def _replace_invalid(match: re.Match[str]) -> str:
            window = proximity_source[max(0, match.start() - 10) : match.end() + 10]
            return " [uv_break]" if "[uv_break]" not in window else " "

        result = regex.sub(r'[!":]', _replace_invalid, result)
        result = regex.sub(r"-", " ", result)
        result = re.sub(r"\[uv_break\](?:\s*\[uv_break\])+", "[uv_break]", result)
        result = re.sub(r"\]\s*\[", "][", result)
    else:
        result = strip_all_tags(result)

    return re.sub(r"\s+", " ", result).strip()
