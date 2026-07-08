"""Loading, saving and caching audio files.

Audio I/O goes through :mod:`soundfile` (libsndfile) rather than
``torchaudio.save``/``torchaudio.load``. Recent torchaudio releases route file
I/O through the optional ``torchcodec`` package, which is not always installed;
libsndfile handles WAV/FLAC/OGG (and MP3, when built with MPEG support)
directly and keeps the dependency footprint small.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from hashlib import md5

import numpy as np
import requests
import soundfile as sf
import torch

_VALID_EXTS = (".wav", ".mp3")


def _to_soundfile_array(waveform: torch.Tensor) -> np.ndarray:
    """Convert a ``(channels, frames)`` torch tensor to soundfile layout.

    soundfile expects ``(frames,)`` for mono or ``(frames, channels)`` for
    multi-channel audio, whereas torch uses a channel-first convention.
    """
    array = waveform.detach().cpu().float()
    if array.dim() == 1:
        return array.numpy()
    array = array.transpose(0, 1).contiguous().numpy()  # (frames, channels)
    return array[:, 0] if array.shape[1] == 1 else array


def _write_mp3(filename: str, data: np.ndarray, sample_rate: int) -> None:
    """Write MP3 via libsndfile, falling back to an ffmpeg transcode."""
    try:
        sf.write(filename, data, sample_rate, format="MP3")
        return
    except Exception:
        pass  # libsndfile build without MPEG encode support

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "Saving MP3 needs libsndfile with MPEG support or an ffmpeg binary. "
            "Install ffmpeg (e.g. `brew install ffmpeg`) or use a '.wav' filename."
        )

    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    try:
        sf.write(tmp_wav, data, sample_rate, subtype="PCM_16")
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_wav, filename],
            check=True,
            capture_output=True,
        )
    finally:
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)


def save_waveform(filename: str, waveform: torch.Tensor, sample_rate: int) -> str:
    """Save ``waveform`` to ``filename`` as WAV or MP3.

    MP3 output requires libsndfile built with MPEG support or an ``ffmpeg``
    binary on ``PATH``; a clear error is raised if neither is available.
    """
    if not filename.endswith(_VALID_EXTS):
        raise ValueError("Filename must end with '.wav' or '.mp3'.")

    data = _to_soundfile_array(waveform)
    if filename.endswith(".wav"):
        sf.write(filename, data, sample_rate, subtype="PCM_16")
    else:
        _write_mp3(filename, data, sample_rate)
    return filename


def load_waveform(path: str) -> tuple[torch.Tensor, int]:
    """Load an audio file into a ``(waveform, sample_rate)`` pair.

    Returns a channel-first tensor (``(channels, frames)``) to match the
    convention used across the package.
    """
    data, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    waveform = torch.from_numpy(data.T.copy())  # (channels, frames)
    return waveform, sample_rate


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
