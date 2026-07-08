"""Subtitle and transcript export (SRT / VTT).

Timings come from the measured duration of each rendered dialog line, so the
captions line up exactly with the audio without any speech recognition step.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Segment:
    """A single spoken line with its position on the timeline (in seconds)."""

    speaker: str
    text: str
    start: float
    end: float
    channel: str = "both"


def _format_timestamp(seconds: float, decimal: str) -> str:
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis == 1000:  # rounding overflow
        secs += 1
        millis = 0
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{decimal}{millis:03d}"


def to_srt(segments: list[Segment]) -> str:
    """Render segments as SRT subtitle text."""
    blocks = []
    for index, seg in enumerate(segments, start=1):
        start = _format_timestamp(seg.start, ",")
        end = _format_timestamp(seg.end, ",")
        blocks.append(f"{index}\n{start} --> {end}\n{seg.speaker}: {seg.text}\n")
    return "\n".join(blocks)


def to_vtt(segments: list[Segment]) -> str:
    """Render segments as WebVTT subtitle text."""
    lines = ["WEBVTT", ""]
    for seg in segments:
        start = _format_timestamp(seg.start, ".")
        end = _format_timestamp(seg.end, ".")
        lines.append(f"{start} --> {end}")
        lines.append(f"{seg.speaker}: {seg.text}")
        lines.append("")
    return "\n".join(lines)


def subtitle_path(audio_filename: str, fmt: str) -> str:
    """Derive a subtitle path from an audio filename (swap the extension)."""
    base, _ = os.path.splitext(audio_filename)
    return f"{base}.{fmt}"


def write_subtitles(segments: list[Segment], audio_filename: str, fmt: str = "srt") -> str:
    """Write ``segments`` next to ``audio_filename`` as ``srt`` or ``vtt``."""
    fmt = fmt.lower()
    if fmt not in {"srt", "vtt"}:
        raise ValueError("Subtitle format must be 'srt' or 'vtt'.")
    content = to_srt(segments) if fmt == "srt" else to_vtt(segments)
    path = subtitle_path(audio_filename, fmt)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)
    return path
