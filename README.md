# Podcast TTS

`podcast_tts` is a Python package for generating high-quality Text-to-Speech (TTS) audio for podcasts and dialogues. It supports multiple speakers, channel-specific playback (left, right, both), and normalization of audio volume.

## Features

- Text-to-Speech generation using `ChatTTS`.
- Supports multiple speakers with speaker profiles.
- Channel-specific playback for left, right, or both audio channels.
- Normalizes audio volume for consistent playback.
- Handles text cleaning, splitting, and formatting.

## Installation

```bash
pip install podcast_tts
```

## Usage

### Generating Audio for a Single Speaker

```python 
import asyncio
from podcast_tts.podcast_tts import PodcastTTS

async def main():
    tts = PodcastTTS(speed=5)
    await tts.generate_wav("Hello, welcome to our podcast!", "Speaker1", "output.wav")

if __name__ == "__main__":
    asyncio.run(main())
``` 

### Generating a Podcast Dialogue

```python 
import asyncio
from podcast_tts.podcast_tts import PodcastTTS

async def main():
    tts = PodcastTTS(speed=5)
    texts = [
        {"Host": ["Welcome to the podcast, where we talk about AI trends.", "left"]},
        {"Guest": ["Thanks for having me! AI is such an exciting field.", "right"]},
        {"Host": ["Let's dive into the latest developments."]},  # Defaults to both channels
    ]
    await tts.generate_dialog_wav(texts, "podcast_dialog.wav")

if __name__ == "__main__":
    asyncio.run(main())
``` 
