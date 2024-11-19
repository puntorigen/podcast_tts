# Podcast TTS

`podcast_tts` is a Python package for generating high-quality Text-to-Speech (TTS) audio for podcasts and dialogues. It supports multiple speakers, channel-specific playback (left, right, both), and normalization of audio volume.

## Features

- **Dynamic Speaker Generation**: Automatically generates new speaker profiles if the specified speaker does not exist, saving the profiles in the `voices` subfolder for future use.
- **Consistent Role Assignment**: Ensures consistency by assigning and reusing speaker profiles based on the speaker name.
- **Load Custom Speaker Profiles**: Supports loading any speaker profile simply by specifying its name.
- **Channel-Specific Playback**: Allows audio to be played on the left, right, or both channels for spatial separation.
- **Normalized Audio Volume**: Normalizes the audio for consistent playback volume.
- **Text Cleaning and Splitting**: Automatically cleans and splits input text into manageable chunks for TTS generation.


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

The profiles are saved in the voices directory and reused automatically if the same speaker is used in the future.


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
