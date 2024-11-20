import asyncio
from podcast_tts.podcast_tts import PodcastTTS

async def main():
    tts = PodcastTTS(speed=5)

    texts = [
        {"male2": ["Welcome to 'Tech Talks Unplugged,' the podcast where we explore the intersection of technology and creativity."]},
        {"male2": ["Today, we’re discussing open-source tools and their impact on the developer community. [uv_break] Joining me are two amazing guests Sarah, a software engineer and open-source advocate, and James, a seasoned podcaster and tech enthusiast."]},
        {"female2": ["Thanks for having me! Open-source tools like 'podcast T T S' are changing the game for podcasters and developers alike.", "right"]},
        {"male1": ["Absolutely. [laugh] I’ve been podcasting for years, and having a tailored T T S solution for podcasts is a lifesaver. It's so much more than a standard T T S.", "left"]},
        {"male2": ["Let’s break that down. James, what makes a podcast-specific T T S different from a regular one?"]},
        {"male1": ["Great question. [uv_break] For starters, podcasts often have speakers close to the microphone, and this affects how the voice should sound—more natural, intimate, and less robotic.", "left"]},
        {"female2": ["Exactly. And let’s not forget about roles. Assigning turns and even controlling the channels—left, right, or both—makes a huge difference in the listening experience.", "right"]},
        {"male2": ["Sarah, can you share how developers benefit from tools like this?"]},
        {"female2": ["Of course. [laugh] Tools like 'podcast T T S' simplify complex workflows. Developers can focus on content instead of worrying about creating realistic voiceovers manually.", "right"]},
        {"male1": ["And it’s versatile! [uv_break] Whether it’s assigning roles dynamically or generating entirely new voices, this tool has it all.", "left"]},
        {"male2": ["Well, that’s all the time we have for today. Thanks, Sarah and James, for sharing your insights on this exciting tool!"]},
        {"female2": ["Thanks for having me. This was fun! [laugh]", "right"]},
        {"male1": ["Same here! [uv_break] Keep innovating, everyone.", "left"]},
    ]

    # Generate the dialog as a single audio file
    await tts.generate_podcast(
        texts, 
        music=["music1.mp3", 10, 3, 0.3], 
        filename="podcast-example.mp3"
    )

if __name__ == "__main__":
    asyncio.run(main())
