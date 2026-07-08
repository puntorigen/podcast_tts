"""Pluggable TTS engines and a small factory to build them by name."""

from __future__ import annotations

from typing import Any

from .base import TTSEngine, VoiceProfile, pick_device

_ALIASES = {
    "chattts": "chattts",
    "chat_tts": "chattts",
    "chat-tts": "chattts",
    "chatterbox": "chatterbox",
    "chatterbox_multilingual": "chatterbox",
    "kokoro": "kokoro",
}

AVAILABLE_ENGINES = ("chattts", "chatterbox", "kokoro")


def get_engine(name: str, **kwargs: Any) -> TTSEngine:
    """Instantiate a TTS engine by name.

    Args:
        name: One of ``"chattts"``, ``"chatterbox"`` or ``"kokoro"`` (aliases
            accepted).
        **kwargs: Forwarded to the engine constructor (``voices_dir``,
            ``default_voices_dir``, ``speed``, ``language``, ``device``, ...).
    """
    key = _ALIASES.get(name.lower().strip())
    if key is None:
        raise ValueError(
            f"Unknown engine '{name}'. Choose one of: {', '.join(AVAILABLE_ENGINES)}."
        )

    if key == "chattts":
        from .chattts_engine import ChatTTSEngine

        return ChatTTSEngine(**kwargs)
    if key == "chatterbox":
        from .chatterbox_engine import ChatterboxEngine

        return ChatterboxEngine(**kwargs)
    from .kokoro_engine import KokoroEngine

    return KokoroEngine(**kwargs)


__all__ = [
    "TTSEngine",
    "VoiceProfile",
    "pick_device",
    "get_engine",
    "AVAILABLE_ENGINES",
]
