import asyncio
from podcast_tts.podcast_tts import PodcastTTS

async def main():
    tts = PodcastTTS(speed=5)

    # Define a realistic podcast dialog
    texts = [
        {"Host": ["Welcome to 'Tech Talks Unplugged,' the podcast where we explore the intersection of technology and creativity."]},
        {"Host": ["Today, we’re discussing open-source tools and their impact on the developer community. [uv_break] Joining me are two amazing guests—Sarah, a software engineer and open-source advocate, and James, a seasoned podcaster and tech enthusiast."]},
        {"Sarah": ["Thanks for having me! Open-source tools like 'podcast_tts' are changing the game for podcasters and developers alike.", "right"]},
        {"James": ["Absolutely. [laugh] I’ve been podcasting for years, and having a tailored TTS solution for podcasts is a lifesaver. It's so much more than a standard TTS.", "left"]},
        {"Host": ["Let’s break that down. James, what makes a podcast-specific TTS different from a regular one?"]},
        {"James": ["Great question. [uv_break] For starters, podcasts often have speakers close to the microphone, and this affects how the voice should sound—more natural, intimate, and less robotic.", "left"]},
        {"Sarah": ["Exactly. And let’s not forget about roles. Assigning turns and even controlling the channels—left, right, or both—makes a huge difference in the listening experience.", "right"]},
        {"Host": ["Sarah, can you share how developers benefit from tools like this?"]},
        {"Sarah": ["Of course. [laugh] Tools like 'podcast_tts' simplify complex workflows. Developers can focus on content instead of worrying about creating realistic voiceovers manually.", "right"]},
        {"James": ["And it’s versatile! [uv_break] Whether it’s assigning roles dynamically or generating entirely new voices, this tool has it all.", "left"]},
        {"Host": ["Well, that’s all the time we have for today. Thanks, Sarah and James, for sharing your insights on this exciting tool!"]},
        {"Sarah": ["Thanks for having me. This was fun! [laugh]", "right"]},
        {"James": ["Same here! [uv_break] Keep innovating, everyone.", "left"]},
    ]

    # Generate the dialog as a single audio file
    await tts.generate_dialog_wav(texts, "example-dialog.wav")

if __name__ == "__main__":
    asyncio.run(main())
