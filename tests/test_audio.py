import pytest
import torch

from podcast_tts.audio import (
    adjust_channel,
    build_music_bed,
    normalize_volume,
    to_stereo,
)


def test_adjust_channel_left_is_silent_on_right():
    stereo = adjust_channel(torch.ones((1, 100)), "left")
    assert stereo.shape == (2, 100)
    assert stereo[1].abs().sum().item() == 0.0


def test_adjust_channel_both_matches():
    stereo = adjust_channel(torch.ones((1, 50)), "both")
    assert torch.equal(stereo[0], stereo[1])


def test_adjust_channel_invalid():
    with pytest.raises(ValueError):
        adjust_channel(torch.ones((1, 10)), "center")


def test_normalize_volume_targets_rms():
    wav = torch.ones((2, 1000)) * 0.02
    out = normalize_volume(wav, target_rms=0.1)
    rms = torch.sqrt(torch.mean(out**2)).item()
    assert abs(rms - 0.1) < 1e-3


def test_to_stereo():
    assert to_stereo(torch.ones((1, 10))).shape == (2, 10)
    assert to_stereo(torch.ones(10)).shape == (2, 10)


def test_build_music_bed_places_dialog_at_offset():
    sr = 1000
    dialog = torch.ones((2, sr))  # 1 second of dialog
    music = torch.ones((2, sr * 40))
    final, offset = build_music_bed(
        dialog, music, sample_rate=sr, full_volume_duration=5, fade_duration=2, target_volume=0.3
    )
    assert final.shape[0] == 2
    # dialog starts after fade_in + full_volume region
    assert offset == int(2 * sr) + int(5 * sr)
    # audio at the dialog offset should exceed the quiet music floor
    assert final[:, offset : offset + sr].abs().mean().item() > 0.3
