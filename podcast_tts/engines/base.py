"""The pluggable TTS engine contract.

Every backend (ChatTTS, Chatterbox, Kokoro) implements :class:`TTSEngine`.
The core :class:`~podcast_tts.core.PodcastTTS` orchestrator only talks to this
interface, so engines can be swapped without touching mixing or dialog logic.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class VoiceProfile:
    """An opaque voice handle.

    ``data`` is interpreted by the owning engine (a ChatTTS embedding string, a
    path to a reference clip for Chatterbox, or a Kokoro voicepack id/tensor).
    """

    name: str
    data: Any = None
    engine: str = ""


def pick_device(preferred: str | None = None) -> str:
    """Choose the best available torch device."""
    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TTSEngine(ABC):
    """Base class for text-to-speech backends."""

    name: str = "base"
    sample_rate: int = 24000
    #: ChatTTS-style prosody tags such as ``[uv_break]``/``[laugh]``.
    uses_prosody_tags: bool = False
    default_language: str = "en"
    supported_languages: frozenset[str] = frozenset({"en"})

    def __init__(
        self,
        voices_dir: str,
        default_voices_dir: str,
        speed: int = 5,
        language: str = "en",
        device: str | None = None,
        **_: Any,
    ) -> None:
        self.voices_dir = voices_dir
        self.default_voices_dir = default_voices_dir
        self.speed = speed
        self.language = language
        self.device = pick_device(device)
        os.makedirs(self.voices_dir, exist_ok=True)
        os.makedirs(self.default_voices_dir, exist_ok=True)

    def supports_language(self, language: str) -> bool:
        base = (language or "en").split("-")[0].lower()
        return base in {lang.split("-")[0].lower() for lang in self.supported_languages}

    def is_raw_profile(self, value: str) -> bool:
        """Whether ``value`` is an inline voice payload rather than a name."""
        return False

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        voice: VoiceProfile,
        language: str = "en",
        emotion: float | None = None,
    ) -> torch.Tensor:
        """Synthesize one text chunk into a mono ``[1, N]`` waveform."""

    @abstractmethod
    async def create_voice(self, name: str) -> VoiceProfile:
        """Create and persist a new voice profile named ``name``."""

    @abstractmethod
    async def load_voice(self, name: str) -> VoiceProfile:
        """Load a voice profile, creating one if it does not exist yet."""
