from podcast_tts.subtitles import (
    Segment,
    subtitle_path,
    to_srt,
    to_vtt,
    write_subtitles,
)


def _segments():
    return [
        Segment("Host", "Hello", 0.0, 1.5, "both"),
        Segment("Guest", "Hola", 2.0, 3.25, "left"),
    ]


def test_srt_format():
    srt = to_srt(_segments())
    assert "00:00:00,000 --> 00:00:01,500" in srt
    assert "Host: Hello" in srt
    assert srt.strip().startswith("1")


def test_vtt_format():
    vtt = to_vtt(_segments())
    assert vtt.startswith("WEBVTT")
    assert "00:00:02.000 --> 00:00:03.250" in vtt


def test_subtitle_path_swaps_extension():
    assert subtitle_path("out/show.mp3", "srt") == "out/show.srt"
    assert subtitle_path("show.wav", "vtt") == "show.vtt"


def test_write_subtitles(tmp_path):
    audio = tmp_path / "show.wav"
    path = write_subtitles(_segments(), str(audio), "srt")
    assert path.endswith(".srt")
    assert "Host: Hello" in open(path, encoding="utf-8").read()
