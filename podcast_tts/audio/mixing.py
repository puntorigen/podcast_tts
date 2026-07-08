"""Waveform mixing: channels, volume and the podcast music bed."""

from __future__ import annotations

import torch
import torchaudio

Channel = str  # "left" | "right" | "both"


def adjust_channel(waveform: torch.Tensor, channel: Channel) -> torch.Tensor:
    """Return a stereo ``[2, N]`` waveform panned to ``channel``.

    ``waveform`` is expected to be mono ``[1, N]``.
    """
    if channel not in {"left", "right", "both"}:
        raise ValueError("Channel must be 'left', 'right', or 'both'.")
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    silence = torch.zeros_like(waveform)
    left = waveform if channel in {"left", "both"} else silence
    right = waveform if channel in {"right", "both"} else silence
    return torch.cat([left, right], dim=0)


def normalize_volume(waveform: torch.Tensor, target_rms: float = 0.1) -> torch.Tensor:
    """Scale ``waveform`` to a target RMS amplitude."""
    current_rms = torch.sqrt(torch.mean(waveform**2))
    if current_rms > 0:
        waveform = waveform * (target_rms / current_rms)
    return waveform


def to_stereo(waveform: torch.Tensor) -> torch.Tensor:
    """Duplicate a mono channel so the result has shape ``[2, N]``."""
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.size(0) == 1:
        waveform = torch.cat([waveform, waveform], dim=0)
    return waveform


def resample(waveform: torch.Tensor, orig_freq: int, new_freq: int) -> torch.Tensor:
    """Resample ``waveform`` if the sample rates differ."""
    if orig_freq == new_freq:
        return waveform
    resampler = torchaudio.transforms.Resample(orig_freq=orig_freq, new_freq=new_freq)
    return resampler(waveform)


def build_music_bed(
    dialog_waveform: torch.Tensor,
    music_waveform: torch.Tensor,
    sample_rate: int,
    full_volume_duration: float,
    fade_duration: float,
    target_volume: float,
) -> tuple[torch.Tensor, int]:
    """Mix stereo dialog over a background music bed.

    The music plays at full volume, fades down under the dialog, then fades
    back up and out after it. ``dialog_waveform`` and ``music_waveform`` must
    already be stereo and share ``sample_rate``.

    Returns:
        A ``(final_audio, dialog_offset_samples)`` tuple, where
        ``dialog_offset_samples`` is where the dialog begins (useful for
        aligning subtitles).
    """
    dialog_waveform = to_stereo(dialog_waveform)
    music_waveform = to_stereo(music_waveform)

    fade_samples = int(fade_duration * sample_rate)
    full_volume_samples = int(full_volume_duration * sample_rate)
    post_dialog_full_volume_samples = int(10 * sample_rate)
    dialog_samples = dialog_waveform.size(1)

    total_needed_length = (
        fade_samples
        + full_volume_samples
        + dialog_samples
        + fade_samples
        + post_dialog_full_volume_samples
        + fade_samples
    )

    while music_waveform.size(1) < total_needed_length:
        music_waveform = torch.cat([music_waveform, music_waveform], dim=1)
    music_waveform = music_waveform[:, :total_needed_length]

    adjusted = torch.zeros_like(music_waveform)

    # 1. Fade in from silence to full volume.
    fade_in = torch.linspace(0, 1, fade_samples)
    adjusted[:, :fade_samples] = music_waveform[:, :fade_samples] * fade_in

    # 2. Hold full volume before the dialog begins.
    hold_end = fade_samples + full_volume_samples - fade_samples
    adjusted[:, fade_samples:hold_end] = music_waveform[:, fade_samples:hold_end]

    # 3. Fade down to the dialog (target) volume.
    fade_to_target = torch.linspace(1, target_volume, fade_samples)
    dialog_offset = fade_samples + full_volume_samples
    adjusted[:, dialog_offset - fade_samples : dialog_offset] = (
        music_waveform[:, dialog_offset - fade_samples : dialog_offset] * fade_to_target
    )

    # 4. Keep music quiet under the dialog.
    adjusted[:, dialog_offset : dialog_offset + dialog_samples] = (
        music_waveform[:, dialog_offset : dialog_offset + dialog_samples] * target_volume
    )

    # 5. Fade music back up as the dialog ends.
    fade_up_start = dialog_offset + dialog_samples - fade_samples
    fade_up = torch.linspace(target_volume, 1, fade_samples)
    if fade_up_start < adjusted.size(1):
        adjusted[:, fade_up_start : dialog_offset + dialog_samples] = (
            music_waveform[:, fade_up_start : dialog_offset + dialog_samples] * fade_up
        )

    # 6. Hold full volume after the dialog.
    end_dialog = dialog_offset + dialog_samples
    adjusted[:, end_dialog : end_dialog + full_volume_samples] = music_waveform[
        :, end_dialog : end_dialog + full_volume_samples
    ]

    # 7. Final fade out.
    fade_out_start = end_dialog + full_volume_samples
    fade_out = torch.linspace(1, 0, fade_samples)
    if fade_out_start < adjusted.size(1):
        adjusted[:, fade_out_start : fade_out_start + fade_samples] = (
            music_waveform[:, fade_out_start : fade_out_start + fade_samples] * fade_out
        )

    total_music_length = min(fade_out_start + fade_samples, adjusted.size(1))
    adjusted = adjusted[:, :total_music_length]

    total_length = max(dialog_waveform.size(1) + dialog_offset, adjusted.size(1))
    final_audio = torch.zeros((2, total_length))
    final_audio[:, dialog_offset : dialog_offset + dialog_waveform.size(1)] += dialog_waveform
    final_audio[:, : adjusted.size(1)] += adjusted[:, :total_length]

    return final_audio, dialog_offset
