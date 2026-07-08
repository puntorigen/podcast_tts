"""The high-level :class:`PodcastTTS` orchestrator.

This ties together a pluggable TTS engine, language-aware text processing,
audio mixing and subtitle export. The public API (``generate_tts``,
``generate_dialog``, ``generate_podcast``, ``create_speaker``,
``load_speaker``) stays backward compatible with earlier releases.
"""

from __future__ import annotations

import os

import torch

from .audio import (
    adjust_channel,
    build_music_bed,
    download_and_cache,
    load_waveform,
    normalize_volume,
    resample,
    save_waveform,
    to_stereo,
)
from .engines import VoiceProfile, get_engine
from .subtitles import Segment, write_subtitles
from .text import normalize_text, prepare_text_for_conversion

_VALID_EXTS = (".wav", ".mp3")
DialogEntry = dict  # {"Speaker": ["text", "channel", {"language":..,"emotion":..}]}


class PodcastTTS:
    """Generate TTS audio for podcasts and dialogues.

    Args:
        speed: Playback speed passed to engines that support it (ChatTTS).
        engine: Backend to use: ``"chattts"`` (default), ``"chatterbox"`` or
            ``"kokoro"``.
        language: Default language code (``"en"`` or ``"es"``) for lines that
            don't specify one.
        device: Force a torch device (``"cuda"``, ``"mps"``, ``"cpu"``);
            autodetected when omitted.
        voices_dir: Where user voice profiles live (default: ``./voices``).
        **engine_kwargs: Extra options forwarded to the engine (e.g.
            ``exaggeration`` for Chatterbox, ``compile`` for ChatTTS).
    """

    def __init__(
        self,
        speed: int = 5,
        engine: str = "chattts",
        language: str = "en",
        device: str | None = None,
        voices_dir: str | None = None,
        default_voices_dir: str | None = None,
        **engine_kwargs,
    ) -> None:
        self.speed = speed
        self.language = language

        self.voices_dir = voices_dir or os.path.join(os.getcwd(), "voices")
        self.default_voices_dir = default_voices_dir or os.path.join(
            os.path.dirname(__file__), "default_voices"
        )
        self.cache_dir = os.path.join(os.getcwd(), "cache")
        for directory in (self.voices_dir, self.default_voices_dir, self.cache_dir):
            os.makedirs(directory, exist_ok=True)

        self.engine = get_engine(
            engine,
            voices_dir=self.voices_dir,
            default_voices_dir=self.default_voices_dir,
            speed=speed,
            language=language,
            device=device,
            **engine_kwargs,
        )
        self.sampling_rate = self.engine.sample_rate

    # ------------------------------------------------------------------ #
    # Voice management
    # ------------------------------------------------------------------ #
    async def create_speaker(self, speaker_name: str):
        """Create and persist a new voice profile; returns its raw payload."""
        profile = await self.engine.create_voice(speaker_name)
        return profile.data

    async def load_speaker(self, speaker_name: str):
        """Load a voice profile (creating one if needed); returns its payload."""
        profile = await self.engine.load_voice(speaker_name)
        return profile.data

    def clone_voice(self, name: str, reference_audio_path: str) -> None:
        """Register a reference clip so ``name`` clones that voice.

        Only supported by engines with voice cloning (Chatterbox).
        """
        register = getattr(self.engine, "register_reference", None)
        if register is None:
            raise NotImplementedError(
                f"The '{self.engine.name}' engine does not support voice cloning."
            )
        register(name, reference_audio_path)

    async def _resolve_voice(self, speaker) -> VoiceProfile:
        if isinstance(speaker, VoiceProfile):
            return speaker
        if isinstance(speaker, str) and self.engine.is_raw_profile(speaker):
            return VoiceProfile(name="inline", data=speaker, engine=self.engine.name)
        return await self.engine.load_voice(speaker)

    # ------------------------------------------------------------------ #
    # Synthesis
    # ------------------------------------------------------------------ #
    def _warn_language(self, language: str) -> None:
        if not self.engine.supports_language(language):
            print(
                f"Warning: engine '{self.engine.name}' may not support language "
                f"'{language}'. Supported: {sorted(self.engine.supported_languages)}."
            )

    async def _synthesize_utterance(
        self,
        text: str,
        voice: VoiceProfile,
        language: str,
        emotion: float | None,
        channel: str,
    ) -> torch.Tensor:
        """Return a stereo ``[2, N]`` waveform for a single speaker turn."""
        use_tags = self.engine.uses_prosody_tags
        batches = prepare_text_for_conversion(text, language, use_tags)

        generated: list[torch.Tensor] = []
        total = len(batches)
        for index, batch in enumerate(batches):
            chunk_text = "".join(batch).replace(".", ". ")
            normalized = normalize_text(chunk_text, language, use_tags)
            if not normalized:
                continue
            print(f"Generating audio for chunk {index + 1}/{total}: {normalized}")
            mono = await self.engine.synthesize(normalized, voice, language, emotion)
            generated.append(adjust_channel(mono, channel))

        if not generated:
            return torch.zeros((2, 0))
        return torch.cat(generated, dim=1)

    async def generate_tts(
        self,
        text: str,
        speaker,
        filename: str = "generated_tts.wav",
        channel: str = "both",
        language: str | None = None,
        emotion: float | None = None,
    ) -> str:
        """Synthesize ``text`` in one voice and save it to ``filename``.

        Args:
            text: Text to synthesize.
            speaker: Voice name, a :class:`VoiceProfile`, or (ChatTTS) a raw
                embedding string.
            filename: Output path ending in ``.wav`` or ``.mp3``.
            channel: ``"left"``, ``"right"`` or ``"both"``.
            language: Language code; falls back to the instance default.
            emotion: Optional intensity (Chatterbox ``exaggeration``).
        """
        if not text:
            raise ValueError("Text cannot be empty.")
        if not speaker:
            raise ValueError("Speaker cannot be empty.")
        if not filename.endswith(_VALID_EXTS):
            raise ValueError("Filename must have a valid extension: '.mp3' or '.wav'.")

        language = language or self.language
        self._warn_language(language)
        voice = await self._resolve_voice(speaker)
        waveform = await self._synthesize_utterance(text, voice, language, emotion, channel)
        save_waveform(filename, waveform, self.sampling_rate)
        print(f"Audio saved to {filename}")
        return filename

    # ------------------------------------------------------------------ #
    # Dialog
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_entry(entry: DialogEntry):
        """Parse one dialog entry into its components.

        Accepts the classic ``{"Speaker": ["text", "channel"]}`` form plus an
        optional trailing options dict:
        ``{"Speaker": ["text", "right", {"language": "es", "emotion": 0.7}]}``.
        """
        if len(entry) != 1:
            raise ValueError("Each entry must contain exactly one speaker.")
        speaker_name, content = next(iter(entry.items()))
        if not isinstance(content, list) or len(content) < 1:
            raise ValueError(f"Invalid entry: {content!r}")

        text = content[0]
        channel = "both"
        language = None
        emotion = None
        for extra in content[1:]:
            if isinstance(extra, str):
                channel = extra
            elif isinstance(extra, dict):
                channel = extra.get("channel", channel)
                language = extra.get("language", language)
                emotion = extra.get("emotion", emotion)
        if channel not in {"left", "right", "both"}:
            raise ValueError(f"Invalid channel '{channel}'.")
        return speaker_name, text, channel, language, emotion

    async def _render_dialog(
        self,
        texts: list[DialogEntry],
        pause_duration: float,
        normalize: bool,
        language: str | None,
    ) -> tuple[torch.Tensor, int, list[Segment]]:
        if not texts:
            raise ValueError("The texts array cannot be empty.")
        if pause_duration < 0:
            raise ValueError("Pause duration must be non-negative.")

        base_language = language or self.language
        sr = self.sampling_rate
        pause_samples = int(pause_duration * sr)

        generated: list[torch.Tensor] = []
        segments: list[Segment] = []
        cursor = 0.0
        last_index = len(texts) - 1

        for index, entry in enumerate(texts):
            speaker_name, text, channel, entry_lang, emotion = self._parse_entry(entry)
            entry_language = entry_lang or base_language
            self._warn_language(entry_language)
            voice = await self._resolve_voice(speaker_name)
            utterance = await self._synthesize_utterance(
                text, voice, entry_language, emotion, channel
            )
            if normalize:
                utterance = normalize_volume(utterance)

            duration = utterance.size(1) / sr
            segments.append(
                Segment(speaker_name, text, cursor, cursor + duration, channel)
            )
            cursor += duration
            generated.append(utterance)

            if index != last_index and pause_samples > 0:
                generated.append(torch.zeros((2, pause_samples)))
                cursor += pause_duration

        merged = torch.cat(generated, dim=1)
        return merged, sr, segments

    async def generate_dialog(
        self,
        texts: list[DialogEntry],
        filename: str = "generated_dialog.wav",
        pause_duration: float = 0.5,
        normalize: bool = True,
        subtitles: bool = False,
        subtitle_format: str = "srt",
        language: str | None = None,
    ) -> str:
        """Render a multi-speaker dialog to a single audio file.

        Args:
            texts: Ordered list of ``{"Speaker": ["text", "channel", opts]}``.
            filename: Output path ending in ``.wav`` or ``.mp3``.
            pause_duration: Silence between turns, in seconds.
            normalize: RMS-normalize each turn's volume.
            subtitles: Also write a subtitle file next to the audio.
            subtitle_format: ``"srt"`` or ``"vtt"``.
            language: Default language for lines without an explicit one.
        """
        if not filename.endswith(_VALID_EXTS):
            raise ValueError("Filename must have a valid extension: '.mp3' or '.wav'.")

        waveform, sr, segments = await self._render_dialog(
            texts, pause_duration, normalize, language
        )
        save_waveform(filename, waveform, sr)
        if subtitles:
            write_subtitles(segments, filename, subtitle_format)
        return filename

    async def generate_podcast(
        self,
        texts: list[DialogEntry],
        music: list,
        filename: str = "podcast.wav",
        pause_duration: float = 0.5,
        normalize: bool = True,
        subtitles: bool = False,
        subtitle_format: str = "srt",
        language: str | None = None,
    ) -> str:
        """Render a dialog and mix it over background music.

        Args:
            texts: Dialog entries (see :meth:`generate_dialog`).
            music: ``[file_or_url, full_volume_duration, fade_duration,
                target_volume]``.
            filename: Output path ending in ``.wav`` or ``.mp3``.
            pause_duration: Silence between turns, in seconds.
            normalize: RMS-normalize each turn's volume.
            subtitles: Also write a subtitle file (timed to the music intro).
            subtitle_format: ``"srt"`` or ``"vtt"``.
            language: Default language for lines without an explicit one.
        """
        if not music or len(music) != 4:
            raise ValueError(
                "Music must be [file, full_volume_duration, fade_duration, target_volume]."
            )
        if not filename.endswith(_VALID_EXTS):
            raise ValueError("Filename must have a .wav or .mp3 extension.")

        music_source, full_volume_duration, fade_duration, target_volume = music
        if isinstance(music_source, str) and music_source.startswith("http"):
            music_file = download_and_cache(music_source, self.cache_dir)
        else:
            music_file = music_source

        dialog_waveform, sr, segments = await self._render_dialog(
            texts, pause_duration, normalize, language
        )

        music_waveform, music_sr = load_waveform(music_file)
        music_waveform = resample(music_waveform, music_sr, sr)
        music_waveform = to_stereo(music_waveform)
        dialog_waveform = to_stereo(dialog_waveform)

        final_audio, dialog_offset = build_music_bed(
            dialog_waveform,
            music_waveform,
            sr,
            full_volume_duration,
            fade_duration,
            target_volume,
        )
        save_waveform(filename, final_audio, sr)

        if subtitles:
            offset = dialog_offset / sr
            shifted = [
                Segment(s.speaker, s.text, s.start + offset, s.end + offset, s.channel)
                for s in segments
            ]
            write_subtitles(shifted, filename, subtitle_format)

        return filename
