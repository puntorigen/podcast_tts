import pytest
import torch

import podcast_tts.core as core
from podcast_tts.audio import save_waveform
from podcast_tts.core import PodcastTTS
from podcast_tts.engines.base import TTSEngine, VoiceProfile


class FakeEngine(TTSEngine):
    """A model-free engine returning fixed-length silence-ish audio."""

    name = "fake"
    sample_rate = 8000
    uses_prosody_tags = False
    supported_languages = frozenset({"en", "es"})

    async def synthesize(self, text, voice, language="en", emotion=None):
        return torch.ones((1, self.sample_rate // 2)) * 0.3  # 0.5s

    async def create_voice(self, name):
        return VoiceProfile(name=name, data=name, engine=self.name)

    async def load_voice(self, name):
        return VoiceProfile(name=name, data=name, engine=self.name)


@pytest.fixture
def tts(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(core, "get_engine", lambda name, **kw: FakeEngine(**kw))
    return PodcastTTS(engine="fake", voices_dir=str(tmp_path / "voices"))


async def test_generate_tts_creates_file(tts, tmp_path):
    out = tmp_path / "hello.wav"
    result = await tts.generate_tts("Hello there world", "narrator", filename=str(out))
    assert result == str(out)
    assert out.exists()


async def test_generate_tts_rejects_bad_extension(tts):
    with pytest.raises(ValueError):
        await tts.generate_tts("Hi", "narrator", filename="bad.ogg")


async def test_generate_dialog_with_subtitles(tts, tmp_path):
    out = tmp_path / "dialog.wav"
    texts = [
        {"Host": ["Welcome to the show today"]},
        {"Guest": ["Hola, gracias por la invitacion", "left", {"language": "es"}]},
    ]
    await tts.generate_dialog(texts, filename=str(out), subtitles=True, subtitle_format="srt")
    assert out.exists()
    srt = tmp_path / "dialog.srt"
    assert srt.exists()
    content = srt.read_text(encoding="utf-8")
    assert "Host:" in content and "Guest:" in content


async def test_generate_podcast_mixes_music(tts, tmp_path):
    music = tmp_path / "music.wav"
    save_waveform(str(music), torch.ones((2, 8000 * 20)) * 0.2, 8000)
    out = tmp_path / "podcast.wav"
    texts = [{"Host": ["Line one here friends"]}, {"Host": ["Line two here friends"]}]
    await tts.generate_podcast(
        texts, music=[str(music), 5, 2, 0.3], filename=str(out), subtitles=True
    )
    assert out.exists()
    assert (tmp_path / "podcast.srt").exists()


def test_parse_entry_classic():
    speaker, text, channel, language, emotion = PodcastTTS._parse_entry({"A": ["hi", "left"]})
    assert (speaker, text, channel) == ("A", "hi", "left")
    assert language is None and emotion is None


def test_parse_entry_extended():
    speaker, text, channel, language, emotion = PodcastTTS._parse_entry(
        {"B": ["hola", "right", {"language": "es", "emotion": 0.8}]}
    )
    assert channel == "right"
    assert language == "es"
    assert emotion == 0.8


def test_parse_entry_invalid_channel():
    with pytest.raises(ValueError):
        PodcastTTS._parse_entry({"A": ["hi", "center"]})
