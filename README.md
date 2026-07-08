<p align="center">
  <img src="https://raw.githubusercontent.com/puntorigen/podcast_tts/main/assets/banner.png" alt="podcast_tts" width="820" />
</p>

<p align="center">
  <a href="https://pypi.org/project/podcast_tts/"><img src="https://img.shields.io/pypi/v/podcast_tts.svg?color=2dd4bf" alt="PyPI version"></a>
  <a href="https://pypi.org/project/podcast_tts/"><img src="https://img.shields.io/pypi/pyversions/podcast_tts.svg" alt="Python versions"></a>
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT">
  <a href="https://github.com/puntorigen/podcast_tts/actions/workflows/ci.yml"><img src="https://github.com/puntorigen/podcast_tts/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
</p>

Turn a script into a natural, multi-speaker **podcast or dialogue** - with background music, stereo panning, and subtitles - in a few lines of Python.

- **Multi-speaker dialogues** with per-line left/right/both channel control.
- **English and Spanish** (plus more), thanks to pluggable TTS engines.
- **Background music** with automatic fade-in/out and ducking under speech.
- **Voice cloning & emotion** (via the Chatterbox engine) or **unlimited random voices** (via ChatTTS).
- **Subtitles** (`.srt` / `.vtt`) generated automatically, perfectly timed to the audio.
- **WAV or MP3** output, plus a simple `podcast-tts` command-line tool.

### Listen to an example

https://github.com/user-attachments/assets/baf6aa80-2d8f-4a2c-8159-efa9d9596693

---

## Install

```bash
# 1. System audio tools (pick your OS)
brew install ffmpeg           # macOS  (Linux: apt-get install ffmpeg)

# 2. The library + the engine you want (see "Which engine?" below)
pip install "podcast_tts[chattts]"      # English, unlimited random voices (default)
pip install "podcast_tts[chatterbox]"   # English + Spanish, voice cloning, emotion
pip install "podcast_tts[kokoro]"       # Fast & light, English + Spanish presets
pip install "podcast_tts[all]"          # Everything
```

> Kokoro also needs `espeak-ng` (`brew install espeak-ng` / `apt-get install espeak-ng`).

## Which engine?

Pick one based on what matters most to you:

| Engine        | Languages         | Voices                          | Emotion | Speed        | Best for |
|---------------|-------------------|---------------------------------|---------|--------------|----------|
| **chattts** (default) | English, Chinese | Unlimited random + your saved profiles | `[laugh]`, breaks | Medium | English podcasts, spinning up many distinct voices |
| **chatterbox** | 23 langs incl. **Spanish** | Clone any voice from ~10s audio | Yes (dial) | Slower (GPU recommended) | Spanish, cloning a real host, expressive delivery |
| **kokoro**     | 8 langs incl. **Spanish** | 54 presets + blends | No | **Fastest** (CPU-friendly) | Quick, clean narration; low-resource machines |

You choose the engine when you create `PodcastTTS(engine=...)`.

---

## Quickstart

```python
import asyncio
from podcast_tts import PodcastTTS

async def main():
    tts = PodcastTTS(engine="chattts")            # English default
    await tts.generate_tts(
        text="Hello! Welcome to our podcast.",
        speaker="male1",                           # a premade voice
        filename="hello.wav",
    )

asyncio.run(main())
```

## A two-person dialogue

```python
import asyncio
from podcast_tts import PodcastTTS

async def main():
    tts = PodcastTTS(engine="chattts")
    dialogue = [
        {"male1":   ["Welcome to the show!", "left"]},
        {"female2": ["Thanks for having me. [laugh]", "right"]},
        {"male1":   ["Today we talk about open source.", "left"]},
    ]
    await tts.generate_dialog(dialogue, filename="dialogue.mp3", subtitles=True)
    # -> dialogue.mp3 + dialogue.srt

asyncio.run(main())
```

## Spanish

Use an engine that speaks Spanish (`chatterbox` or `kokoro`) and set the language.

```python
import asyncio
from podcast_tts import PodcastTTS

async def main():
    tts = PodcastTTS(engine="kokoro", language="es")
    await tts.generate_tts(
        text="Hola, bienvenidos al pódcast. Hoy hablamos de inteligencia artificial.",
        speaker="ef_dora",                         # a Spanish Kokoro voice
        filename="hola.wav",
    )

asyncio.run(main())
```

You can even **mix languages in one dialogue** by setting the language per line:

```python
dialogue = [
    {"Host":  ["Welcome! Today we go bilingual."]},
    {"Guest": ["Hola, gracias por la invitación.", "left", {"language": "es"}]},
]
await tts.generate_dialog(dialogue, filename="bilingual.mp3")
```

## Podcast with background music

`music = [file_or_url, full_volume_seconds, fade_seconds, volume_under_speech]`

```python
await tts.generate_podcast(
    texts=dialogue,
    music=["intro.mp3", 10, 3, 0.3],   # or a https:// URL (downloaded & cached)
    filename="episode.mp3",
    subtitles=True,
)
```

The music plays at full volume, fades down under the dialogue, then fades back up and out.

## Clone a voice (Chatterbox)

Drop a clean 10-30s clip in your `voices/` folder named after the speaker, or register it in code:

```python
tts = PodcastTTS(engine="chatterbox", language="es")
tts.clone_voice("Ana", "samples/ana_reference.wav")   # now "Ana" sounds like the clip

await tts.generate_tts(
    "Hola, soy Ana y este es mi pódcast.",
    speaker="Ana",
    filename="ana.wav",
    emotion=0.7,          # 0.0 calm ... 1.0 dramatic
)
```

## Command line

```bash
podcast-tts say "Hello there" --speaker male1 -o hello.wav
podcast-tts dialog script.json -o show.mp3 --subtitles srt
podcast-tts dialog script.json -o show.mp3 --engine kokoro --language es \
    --music intro.mp3 10 3 0.3
```

`script.json` is just the dialogue list:

```json
[
  {"male1": ["Welcome to the show!", "both"]},
  {"female2": ["Hola a todos.", "left", {"language": "es"}]}
]
```

## Web demo

Prefer clicking to coding? Launch a small local web UI:

```bash
pip install "podcast_tts[demo,chattts]"   # the demo + one engine
podcast-tts-demo                            # opens http://127.0.0.1:7860
```

Two tabs: synthesize a single line, or paste a dialogue script and render a full
podcast (with optional background music and a downloadable subtitle file).

---

## Voices

- **ChatTTS** ships three ready-to-use profiles: `male1`, `male2`, `female2`. Any new
  name you use is generated once and saved to `voices/<name>.txt` so it stays consistent.
- **Chatterbox** uses reference clips: put `voices/<name>.wav` (or call `clone_voice`).
  Without a reference it uses its default voice.
- **Kokoro** uses preset ids (e.g. `af_heart`, `ef_dora`, `em_alex`). Blend new ones:

  ```python
  tts.engine.blend_voices("myvoice", {"ef_dora": 0.6, "em_alex": 0.4})
  ```

## Dialogue entry format

Each turn is a one-key dict: `{"SpeakerName": [text, channel?, options?]}`

- `text` (str, required)
- `channel` (str, optional): `"left"`, `"right"`, or `"both"` (default)
- `options` (dict, optional): `{"language": "es", "emotion": 0.7}`

## API at a glance

```python
tts = PodcastTTS(engine="chattts", language="en", speed=5, device=None)

await tts.generate_tts(text, speaker, filename="out.wav", channel="both",
                       language=None, emotion=None)
await tts.generate_dialog(texts, filename="dialog.wav", pause_duration=0.5,
                          normalize=True, subtitles=False, subtitle_format="srt",
                          language=None)
await tts.generate_podcast(texts, music, filename="podcast.wav", pause_duration=0.5,
                           normalize=True, subtitles=False, subtitle_format="srt",
                           language=None)
```

## Upgrading from 0.0.x

The old API still works: `from podcast_tts import PodcastTTS`, plus `generate_tts`,
`generate_dialog`, and `generate_podcast` keep the same required arguments. New in 0.1.0:
the `engine`/`language`/`emotion` options, Spanish support, subtitles, and the CLI. The
default engine remains ChatTTS, so existing scripts behave as before.

## Development

```bash
pip install -e ".[dev]"
ruff check podcast_tts tests
pytest -q
```

### Releasing to PyPI

Releases are automated by [`.github/workflows/release.yml`](.github/workflows/release.yml):
push a version tag and CI builds and publishes the package.

```bash
# 1. Bump the version in pyproject.toml (e.g. 0.1.0 -> 0.1.1)
# 2. Tag and push (the tag must match the pyproject version):
git tag v0.1.1
git push origin v0.1.1
```

The workflow checks the tag matches the version, builds the sdist/wheel, and
uploads with `skip-existing` (so re-runs never clobber an existing release).

**Authentication:** the publish step uses a `PYPI_API_TOKEN` repository secret
(Settings → Secrets and variables → Actions). Create a PyPI API token scoped to
this project and store it there. To switch to
[trusted publishing](https://docs.pypi.org/trusted-publishers/) instead, drop
the `password:` line from the publish step, add `permissions: id-token: write`,
and register the publisher on PyPI.

## Contributing

Issues and pull requests are welcome on
[GitHub](https://github.com/puntorigen/podcast_tts).

## License

MIT - see [LICENSE](LICENSE). Note the underlying engines have their own model
licenses (ChatTTS, Chatterbox, Kokoro); review them for commercial use.
