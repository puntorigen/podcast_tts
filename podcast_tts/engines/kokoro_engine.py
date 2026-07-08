"""Kokoro backend (optional).

Tiny (82M), fast, CPU-friendly and permissively licensed (Apache-2.0). Ships
54 preset voices across 8 languages, including 3 Spanish voices
(``ef_dora``, ``em_alex``, ``em_santa``). No cloning, but you can blend
existing voices into new ones with :meth:`blend_voices`.

Requires the ``espeak-ng`` system package (``brew install espeak-ng`` /
``apt-get install espeak-ng``).
"""

from __future__ import annotations

import asyncio
import os
import re

import torch

from .base import TTSEngine, VoiceProfile

# language code -> Kokoro single-letter lang_code
_LANG_CODES = {
    "en": "a",
    "en-us": "a",
    "en-gb": "b",
    "es": "e",
    "fr": "f",
    "hi": "h",
    "it": "i",
    "pt": "p",
    "pt-br": "p",
    "ja": "j",
    "zh": "z",
}

# sensible default voice per Kokoro lang_code
_DEFAULT_VOICE = {
    "a": "af_heart",
    "b": "bf_emma",
    "e": "ef_dora",
    "f": "ff_siwis",
    "h": "hf_alpha",
    "i": "if_sara",
    "p": "pf_dora",
    "j": "jf_alpha",
    "z": "zf_xiaobei",
}

_VOICE_ID_RE = re.compile(r"^[abefhijpz][fm]_")


class KokoroEngine(TTSEngine):
    name = "kokoro"
    sample_rate = 24000
    uses_prosody_tags = False
    default_language = "en"
    supported_languages = frozenset(_LANG_CODES)

    def __init__(self, *args, speed_ratio: float = 1.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        try:
            from kokoro import KPipeline
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Kokoro is not installed. Install it with:\n"
                "    pip install 'podcast_tts[kokoro]'\n"
                "and the espeak-ng system package."
            ) from exc

        self._KPipeline = KPipeline
        self.speed_ratio = speed_ratio
        self._pipelines: dict[str, object] = {}

    def _lang_code(self, language: str) -> str:
        return _LANG_CODES.get((language or "en").lower(), "a")

    def _pipeline(self, language: str):
        code = self._lang_code(language)
        if code not in self._pipelines:
            self._pipelines[code] = self._KPipeline(lang_code=code)
        return self._pipelines[code]

    def _resolve_voice_id(self, voice: VoiceProfile, language: str):
        if voice is not None and voice.data is not None:
            return voice.data  # a voice id string or a blended tensor
        return _DEFAULT_VOICE[self._lang_code(language)]

    async def synthesize(
        self,
        text: str,
        voice: VoiceProfile,
        language: str = "en",
        emotion: float | None = None,
    ) -> torch.Tensor:
        pipeline = self._pipeline(language)
        voice_id = self._resolve_voice_id(voice, language)

        def _run() -> torch.Tensor:
            pieces = [
                self._to_tensor(audio)
                for _, _, audio in pipeline(text, voice=voice_id, speed=self.speed_ratio)
            ]
            if not pieces:
                return torch.zeros((1, 1))
            return torch.cat(pieces, dim=-1).unsqueeze(0)

        return await asyncio.to_thread(_run)

    @staticmethod
    def _to_tensor(audio) -> torch.Tensor:
        tensor = audio if isinstance(audio, torch.Tensor) else torch.as_tensor(audio)
        tensor = tensor.detach().to(torch.float32).cpu()
        if tensor.dim() > 1:
            tensor = tensor.reshape(-1)
        return tensor

    def blend_voices(self, name: str, weights: dict[str, float]) -> VoiceProfile:
        """Create a new voice by weighted-averaging existing voicepacks.

        Args:
            name: Name for the blended voice (saved as ``<name>.pt``).
            weights: Mapping of Kokoro voice id -> weight (auto-normalized).
        """
        if not weights:
            raise ValueError("Provide at least one voice to blend.")
        pipeline = self._pipeline("en")
        total = sum(weights.values())
        blended: torch.Tensor | None = None
        for voice_id, weight in weights.items():
            pack = pipeline.load_voice(voice_id)
            pack = pack if isinstance(pack, torch.Tensor) else torch.as_tensor(pack)
            contribution = pack * (weight / total)
            blended = contribution if blended is None else blended + contribution
        path = os.path.join(self.voices_dir, f"{name}.pt")
        torch.save(blended, path)
        return VoiceProfile(name=name, data=blended, engine=self.name)

    async def create_voice(self, name: str) -> VoiceProfile:
        # Kokoro cannot invent a voice; fall back to a preset.
        if _VOICE_ID_RE.match(name):
            return VoiceProfile(name=name, data=name, engine=self.name)
        print(
            f"Kokoro has no random voices. Using a preset for '{name}'. "
            f"Pass a Kokoro voice id (e.g. 'ef_dora') or blend one with blend_voices()."
        )
        return VoiceProfile(name=name, data=None, engine=self.name)

    async def load_voice(self, name: str) -> VoiceProfile:
        blended_path = os.path.join(self.voices_dir, f"{name}.pt")
        if os.path.exists(blended_path):
            tensor = await asyncio.to_thread(torch.load, blended_path)
            return VoiceProfile(name=name, data=tensor, engine=self.name)
        if _VOICE_ID_RE.match(name):
            return VoiceProfile(name=name, data=name, engine=self.name)
        return await self.create_voice(name)
