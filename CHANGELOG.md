# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-07-08

### Fixed

- Audio I/O now uses `soundfile` (libsndfile) instead of `torchaudio.save`/
  `torchaudio.load`. Newer `torchaudio` routes file I/O through the optional
  `torchcodec` package, which is often not installed and broke WAV read/write
  (both in CI and at runtime). MP3 output falls back from libsndfile's MPEG
  encoder to an `ffmpeg` transcode.

## [0.1.0] - 2026-07-08

> Superseded by 0.1.1 — this release requires `torchcodec` for WAV read/write
> on recent `torchaudio` builds. Use 0.1.1 or newer.

### Added

- Pluggable TTS engine architecture with three backends:
  - **ChatTTS** — the original English engine, with `[uv_break]`/`[laugh]` prosody tags.
  - **Chatterbox Multilingual** — 23 languages, zero-shot voice cloning, emotion control.
  - **Kokoro** — fast, CPU-friendly, multilingual preset voices with blending.
- **Spanish support** (plus 20+ additional languages via the multilingual engines).
- Subtitle generation in **SRT** and **WebVTT** formats with per-segment timing.
- Command-line interface `podcast-tts` with `say` and `dialog` subcommands
  (also runnable via `python -m podcast_tts`).
- Optional Gradio web demo via `podcast-tts-demo`.
- Repository branding (logo + banner) and a rewritten, low-friction README.
- Test suite (`pytest`), CI (GitHub Actions on Python 3.10–3.12), and a
  tag-driven PyPI release workflow.

### Changed

- Refactored the monolithic module into a modular package: `text/`, `audio/`,
  `engines/`, and `subtitles`.
- Modernized packaging to `pyproject.toml` (PEP 621) with optional-dependency
  extras (`chattts`, `chatterbox`, `kokoro`, `all`, `demo`, `dev`) so you only
  install the engine you need.
- Number-to-words normalization now uses `num2words` (multilingual) instead of
  the English-only `inflect`.

### Compatibility

- The public API (`generate_tts`, `generate_dialog`, `generate_podcast`) and
  the classic dialogue entry format remain backward compatible.

## [0.0.1]

### Added

- Initial release: ChatTTS-based multi-speaker dialogue and podcast generation
  with background-music mixing and WAV/MP3 output.

[0.1.1]: https://github.com/puntorigen/podcast_tts/releases/tag/v0.1.1
[0.1.0]: https://github.com/puntorigen/podcast_tts/releases/tag/v0.1.0
[0.0.1]: https://pypi.org/project/podcast-tts/0.0.1/
