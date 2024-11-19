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
