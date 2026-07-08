"""Spanish dialogue using the Kokoro engine (fast, CPU-friendly).

Install:  pip install "podcast_tts[kokoro]"  +  espeak-ng
"""

import asyncio

from podcast_tts import PodcastTTS


async def main():
    tts = PodcastTTS(engine="kokoro", language="es")

    dialogo = [
        {"ef_dora": ["Hola a todos y bienvenidos al pódcast."]},
        {"em_alex": ["Gracias, Dora. Hoy hablamos de inteligencia artificial.", "left"]},
        {"ef_dora": ["Empecemos con las 3 tendencias más importantes.", "right"]},
    ]

    await tts.generate_dialog(dialogo, filename="ejemplo-espanol.mp3", subtitles=True)
    print("Listo: ejemplo-espanol.mp3 + ejemplo-espanol.srt")


if __name__ == "__main__":
    asyncio.run(main())
