"""ChatTTS backend (default).

English/Chinese, expressive prosody tags (``[uv_break]``, ``[laugh]``) and
unlimited random speaker generation. Speaker profiles are stored as ``.txt``
embedding files, matching earlier releases of this library.
"""

from __future__ import annotations

import asyncio
import os

import torch

from .base import TTSEngine, VoiceProfile


class ChatTTSEngine(TTSEngine):
    name = "chattts"
    sample_rate = 24000
    uses_prosody_tags = True
    default_language = "en"
    supported_languages = frozenset({"en", "zh"})

    def __init__(self, *args, compile: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        try:
            import ChatTTS  # noqa: N814
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "ChatTTS is not installed. Install it with:\n"
                "    pip install 'podcast_tts[chattts]'"
            ) from exc

        self._ChatTTS = ChatTTS
        self.chat = ChatTTS.Chat()
        self.chat.load(compile=compile)

    def is_raw_profile(self, value: str) -> bool:
        # ChatTTS speaker embeddings are long encoded strings; names are short.
        return isinstance(value, str) and len(value) > 200

    def _lang_code(self, language: str) -> str:
        return "zh" if (language or "en").split("-")[0].lower() == "zh" else "en"

    async def synthesize(
        self,
        text: str,
        voice: VoiceProfile,
        language: str = "en",
        emotion: float | None = None,
    ) -> torch.Tensor:
        params_infer_code = self._ChatTTS.Chat.InferCodeParams(
            prompt=f"[speed_{self.speed}]",
            spk_emb=voice.data,
            temperature=0.15,
            top_P=0.75,
            top_K=20,
        )
        params_refine_text = self._ChatTTS.Chat.RefineTextParams(
            prompt="[oral_2][laugh_0][break_6]",
            temperature=0.12,
            max_new_token=500,
        )
        wavs = await asyncio.to_thread(
            self.chat.infer,
            text,
            lang=self._lang_code(language),
            skip_refine_text=False,
            params_refine_text=params_refine_text,
            params_infer_code=params_infer_code,
            do_text_normalization=False,
            do_homophone_replacement=True,
        )
        return torch.tensor(wavs[0]).unsqueeze(0)

    def _profile_path(self, name: str, base_dir: str) -> str:
        return os.path.join(base_dir, f"{name}.txt")

    async def create_voice(self, name: str) -> VoiceProfile:
        if not name:
            raise ValueError("Speaker name cannot be empty.")
        embedding = self.chat.speaker.sample_random()
        path = self._profile_path(name, self.voices_dir)
        await asyncio.to_thread(self._write, path, embedding)
        return VoiceProfile(name=name, data=embedding, engine=self.name)

    async def load_voice(self, name: str) -> VoiceProfile:
        for base_dir in (self.voices_dir, self.default_voices_dir):
            path = self._profile_path(name, base_dir)
            if os.path.exists(path):
                embedding = await asyncio.to_thread(self._read, path)
                return VoiceProfile(name=name, data=embedding, engine=self.name)
        print(f"Speaker '{name}' not found. Creating a new ChatTTS profile.")
        return await self.create_voice(name)

    @staticmethod
    def _write(path: str, data: str) -> None:
        with open(path, "w") as handle:
            handle.write(data)

    @staticmethod
    def _read(path: str) -> str:
        with open(path) as handle:
            return handle.read()
