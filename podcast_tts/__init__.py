"""podcast_tts - multi-speaker podcasts and dialogues with pluggable TTS engines."""

from .core import PodcastTTS
from .engines import AVAILABLE_ENGINES, TTSEngine, VoiceProfile, get_engine

__version__ = "0.1.1"

__all__ = [
    "PodcastTTS",
    "TTSEngine",
    "VoiceProfile",
    "get_engine",
    "AVAILABLE_ENGINES",
    "__version__",
]
