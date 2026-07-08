"""Backward-compatibility shim.

Historically everything lived in ``podcast_tts/podcast_tts.py``. The code now
lives in focused modules, but this module keeps old imports working:

    from podcast_tts.podcast_tts import PodcastTTS, normalize_text
"""

from __future__ import annotations

from .core import PodcastTTS
from .text.chunk import prepare_text_for_conversion
from .text.normalize import RESERVED_TAGS, normalize_text, remove_brackets

__all__ = [
    "PodcastTTS",
    "normalize_text",
    "remove_brackets",
    "prepare_text_for_conversion",
    "RESERVED_TAGS",
]
