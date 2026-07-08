"""Loading, saving and caching audio files."""

from __future__ import annotations

import os
from hashlib import md5

import requests
import torch
import torchaudio

_VALID_EXTS = (".wav", ".mp3")


def _ensure_2d(waveform: torch.Tensor) -> torch.Tensor:
    return waveform.unsqueeze(0) if waveform.dim() == 1 else waveform


def save_waveform(filename: str, waveform: torch.Tensor, sample_rate: int) -> str:
    """Save ``waveform`` to ``filename`` as WAV or MP3.

    MP3 output requires an ``ffmpeg``/``sox`` backend for torchaudio. A clear
    error is raised if the backend is missing.
    """
    if not filename.endswith(_VALID_EXTS):
        raise ValueError("Filename must end with '.wav' or '.mp3'.")

    waveform = _ensure_2d(waveform.detach().cpu())
    fmt = "wav" if filename.endswith(".wav") else "mp3"

    try:
        torchaudio.save(filename, waveform, sample_rate, format=fmt)
    except TypeError as exc:
        # Older torchaudio builds expect a 1-D tensor for mono.
        if "unsqueeze" in str(exc):
            torchaudio.save(filename, waveform.squeeze(0), sample_rate, format=fmt)
        else:
            raise
    except Exception as exc:  # pragma: no cover - backend dependent
        if fmt == "mp3":
            raise RuntimeError(
                "Saving MP3 failed. Install ffmpeg (e.g. `brew install ffmpeg`) "
                "or use a '.wav' filename."
            ) from exc
        raise
    return filename


def load_waveform(path: str) -> tuple[torch.Tensor, int]:
    """Load an audio file into a ``(waveform, sample_rate)`` pair."""
    return torchaudio.load(path)


def download_and_cache(url: str, cache_dir: str) -> str:
    """Download ``url`` into ``cache_dir`` (keyed by URL hash) and return the path."""
    os.makedirs(cache_dir, exist_ok=True)
    ext = ".mp3"
    for candidate in (".mp3", ".wav", ".flac", ".ogg"):
        if url.lower().split("?")[0].endswith(candidate):
            ext = candidate
            break

    file_hash = md5(url.encode("utf-8")).hexdigest()
    cached_path = os.path.join(cache_dir, f"{file_hash}{ext}")
    if os.path.exists(cached_path):
        return cached_path

    response = requests.get(url, stream=True, timeout=60)
    if response.status_code != 200:
        raise ValueError(
            f"Failed to download file from URL: {url} (status {response.status_code})"
        )
    with open(cached_path, "wb") as handle:
        for chunk in response.iter_content(chunk_size=8192):
            handle.write(chunk)
    return cached_path
