"""Split long text into synthesis-friendly chunks."""

from __future__ import annotations

import re

from .normalize import normalize_text


def split_by_punctuation(text: str, max_chunk_length: int = 150) -> list[str]:
    """Split ``text`` on sentence punctuation, keeping the delimiter.

    ``max_chunk_length`` is advisory; sentences are never split mid-word.
    """
    punctuation_marks = ".!?;"
    result: list[str] = []
    start = 0
    for match in re.finditer(rf"[{re.escape(punctuation_marks)}]", text):
        end = match.end()
        piece = text[start:end].strip()
        if piece:
            result.append(piece)
        start = end
    if start < len(text):
        tail = text[start:].strip()
        if tail:
            result.append(tail)
    return result


def prepare_text_for_conversion(
    text: str,
    language: str = "en",
    use_tags: bool = True,
    min_line_length: int = 30,
    merge_size: int = 3,
    max_chunk_length: int = 150,
) -> list[list[str]]:
    """Normalize and chunk ``text`` into batches ready for TTS.

    Short sentences are merged so each line is at least ``min_line_length``
    characters, then lines are grouped into batches of ``merge_size``.

    Args:
        text: Raw input text.
        language: Language code used for number spelling.
        use_tags: Whether to join short lines with a ``[uv_break]`` tag
            (ChatTTS) or a plain space (other engines).
        min_line_length: Minimum length before short lines are merged.
        merge_size: Number of lines grouped into each batch.
        max_chunk_length: Advisory maximum characters per sentence.

    Returns:
        A list of batches, where each batch is a list of text lines.
    """
    separator = "[uv_break]" if use_tags else ""
    text = normalize_text(text, language, use_tags)
    split_chunks = split_by_punctuation(text, max_chunk_length)

    def _join(left: str, right: str) -> str:
        return f"{left} {separator} {right}" if separator else f"{left} {right}"

    retext: list[str] = []
    short_text = ""
    for chunk in split_chunks:
        if len(chunk) < min_line_length:
            short_text += f"{chunk} {separator} " if separator else f"{chunk} "
            if len(short_text) >= min_line_length:
                retext.append(short_text.strip())
                short_text = ""
        else:
            if short_text:
                chunk = _join(short_text.strip(), chunk)
                short_text = ""
            retext.append(chunk)

    if short_text:
        retext.append(short_text.strip())

    return [retext[i : i + merge_size] for i in range(0, len(retext), merge_size)]
