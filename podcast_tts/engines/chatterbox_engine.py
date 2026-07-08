"""Chatterbox Multilingual V3 backend (Resemble AI, MIT).

23 languages including Spanish, zero-shot voice cloning from a short reference
clip, and an emotion/intensity dial (``exaggeration``). Heavier than ChatTTS
and much faster on a GPU.

Voice cloning: drop a clean 10-30s reference clip named ``<voice>.wav`` in your
``voices/`` directory (or call :meth:`register_reference`). Without a reference
the model's built-in default voice is used.
"""

from __future__ import annotations

import asyncio
import os
import shutil

import torch

from .base import TTSEngine, VoiceProfile

_REFERENCE_EXTS = (".wav", ".mp3", ".flac", ".ogg")


class ChatterboxEngine(TTSEngine):
    name = "chatterbox"
    sample_rate = 24000
    uses_prosody_tags = False
    default_language = "en"
    supported_languages = frozenset(
        {
            "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it",
            "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh",
        }
    )

    def __init__(self, *args, exaggeration: float = 0.5, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        try:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Chatterbox is not installed. Install it with:\n"
                "    pip install 'podcast_tts[chatterbox]'"
            ) from exc

        self.default_exaggeration = exaggeration
        self.model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
        self.sample_rate = int(getattr(self.model, "sr", 24000))

    def _lang_id(self, language: str) -> str:
        return (language or self.default_language).split("-")[0].lower()

    async def synthesize(
        self,
        text: str,
        voice: VoiceProfile,
        language: str = "en",
        emotion: float | None = None,
    ) -> torch.Tensor:
        kwargs: dict = {"language_id": self._lang_id(language)}
        if voice and voice.data:
            kwargs["audio_prompt_path"] = voice.data
        kwargs["exaggeration"] = (
            emotion if emotion is not None else self.default_exaggeration
        )

        wav = await asyncio.to_thread(self.model.generate, text, **kwargs)
        return self._to_mono_2d(wav)

    @staticmethod
    def _to_mono_2d(wav) -> torch.Tensor:
        tensor = wav if isinstance(wav, torch.Tensor) else torch.as_tensor(wav)
        tensor = tensor.detach().to(torch.float32).cpu()
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.size(0) > 1:  # collapse any accidental multi-channel output
            tensor = tensor.mean(dim=0, keepdim=True)
        return tensor

    def _find_reference(self, name: str) -> str | None:
        for base_dir in (self.voices_dir, self.default_voices_dir):
            for ext in _REFERENCE_EXTS:
                path = os.path.join(base_dir, f"{name}{ext}")
                if os.path.exists(path):
                    return path
        return None

    def register_reference(self, name: str, reference_audio_path: str) -> VoiceProfile:
        """Copy a reference clip into ``voices/`` so ``name`` clones that voice."""
        if not os.path.exists(reference_audio_path):
            raise FileNotFoundError(reference_audio_path)
        ext = os.path.splitext(reference_audio_path)[1] or ".wav"
        dest = os.path.join(self.voices_dir, f"{name}{ext}")
        shutil.copyfile(reference_audio_path, dest)
        return VoiceProfile(name=name, data=dest, engine=self.name)

    async def create_voice(self, name: str) -> VoiceProfile:
        reference = self._find_reference(name)
        if reference is None:
            print(
                f"No reference clip for '{name}'. Using Chatterbox's default voice. "
                f"To clone a voice, add '{name}.wav' to '{self.voices_dir}'."
            )
        return VoiceProfile(name=name, data=reference, engine=self.name)

    async def load_voice(self, name: str) -> VoiceProfile:
        return await self.create_voice(name)
