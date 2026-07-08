"""Clone a real voice and speak Spanish + English with it (Chatterbox engine).

Install:  pip install "podcast_tts[chatterbox]"   (a GPU is recommended)

Provide a clean 10-30s reference clip of the voice you want to clone.
"""

import asyncio

from podcast_tts import PodcastTTS

REFERENCE_CLIP = "samples/host_reference.wav"  # <- your voice sample


async def main():
    tts = PodcastTTS(engine="chatterbox", language="es")

    # Register the reference clip under the name "Host".
    tts.clone_voice("Host", REFERENCE_CLIP)

    dialogue = [
        {"Host": ["Hola, soy el anfitrión y esta es mi voz clonada.", "both", {"emotion": 0.6}]},
        {"Host": ["And here is the same cloned voice speaking English.", "both", {"language": "en"}]},
    ]

    await tts.generate_dialog(dialogue, filename="cloned-voice.mp3", subtitles=True)
    print("Done: cloned-voice.mp3 + cloned-voice.srt")


if __name__ == "__main__":
    asyncio.run(main())
