# Podcast TTS

`podcast_tts` is a Python library for generating podcasts and dialogues using text-to-speech (TTS). It supports multiple speakers, background music, and precise audio mixing for professional-quality results.

## Features

- **Multi-Speaker Support**: Generate dialogues with distinct speaker profiles.
- **Premade Voices**: Use premade speaker profiles (male1, male2, female2) included with the library or create custom profiles.
- **Dynamic Speaker Generation**: Automatically generates new speaker profiles if the specified speaker does not exist, saving the profiles in the `voices` subfolder for future use.
- **Consistent Role Assignment**: Ensures consistency by assigning and reusing speaker profiles based on the speaker name.
- **Channel-Specific Playback**: Allows audio to be played on the left, right, or both channels for spatial separation.
- **Text Normalization**: Automatically normalize text, handle contractions, and format special cases.
- **Background Music Integration**: Add background music with fade-in/out and volume control.
- **MP3 and URL Support**: Use local MP3/WAV files or download music from a URL with caching.
- **Output Formats**: Save generated audio as WAV or MP3 files.


## Installation

```bash
pip install podcast_tts
```

## Usage

### Generating Audio for a Single Speaker

```python 
import asyncio
from podcast_tts import PodcastTTS

async def main():
    tts = PodcastTTS(speed=5)
    await tts.generate_wav(
        text="Hello! Welcome to our podcast.",
        speaker="male1",
        filename="output_audio.wav",
        channel="both"
    )

if __name__ == "__main__":
    asyncio.run(main())
``` 

### Example: Generating a Podcast with Music

The generate_podcast method combines dialogue and background music for a seamless podcast production.

```python 
import asyncio
from podcast_tts import PodcastTTS

async def main():
    tts = PodcastTTS(speed=5)

    # Define speakers and text
    texts = [
        {"male1": ["Welcome to the podcast!", "both"]},
        {"female2": ["Today, we discuss AI advancements.", "left"]},
        {"male2": ["Don't miss our exciting updates.", "right"]},
    ]

    # Define background music (local file or URL)
    music_config = ["https://example.com/background_music.mp3", 10, 3, 0.3]

    # Generate the podcast
    output_file = await tts.generate_podcast(
        texts=texts,
        music=music_config,
        filename="podcast_with_music.mp3",
        pause_duration=0.5,
        normalize=True
    )

    print(f"Podcast saved to: {output_file}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Music Configuration:

- [file/url, full_volume_duration, fade_duration, target_volume]
    - **file/url**: Path to a local MP3/WAV file or a URL to download.
    - **full_volume_duration**: Time (seconds) at full volume before dialogue starts and after ends.
    - **fade_duration**: Time (seconds) for fade-in/out effects.
    - **target_volum**e: Volume level (0.0 to 1.0) during dialogue playback.

## Premade Voices

PodcastTTS includes the following premade speaker profiles:

- male1
- male2
- female2

These profiles are included in the package's **default_voices** directory and can be used without additional setup.


## Dynamic Speaker Generation

When a speaker profile is specified but does not exist, the library will automatically generate a new speaker profile and save it in the voices subfolder. This ensures consistent voice roles across different turns in a dialogue.
For example:

```python
texts = [
    {"Narrator": ["Welcome to this exciting episode.", "left"]},
    {"Expert": ["Today, we'll explore AI's impact on healthcare.", "right"]},
]
# If "Narrator" or "Expert" profiles do not exist, they will be generated dynamically.
```

The profiles are saved in the script's voices directory and reused automatically if the same speaker is used in the future for consistency.

## Loading Existing Speaker Profiles

You can load any speaker profile by specifying its filename (without the .txt extension). Profiles are stored in the voices subfolder, so you don't need to specify the path explicitly.

```python
# Assuming a speaker profile "Host.txt" exists in the voices subfolder
await tts.generate_wav("This is a test for an existing speaker.", "Host", "existing_speaker.wav")
```

## Additional Notes

- The library uses ChatTTS for high-quality TTS generation.
- Text is automatically cleaned and split into manageable chunks, making it easy to generate audio for long scripts or conversations.
- The generated audio files are saved in WAV format, with support for channel-specific playback.

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests on the GitHub repository.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
