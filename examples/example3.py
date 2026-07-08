import asyncio

from podcast_tts import PodcastTTS


async def main():
    tts = PodcastTTS(speed=5)

    texts = [
       {
            "male1": [
                "Welcome everyone to another episode of Project Documentation Insights. I'm your host, John Smith, and today we're diving into the realm of project documentation with a focus on some exciting tools and frameworks that make a developer's life easier.",
                'both'
            ]
        },
        {
            "male1": [
                'Joining me today, we have two experts: Emily Brown, a software developer with a knack for creating user-friendly applications, and Michael Johnson, a backend developer specializing in scalable server-side solutions. Welcome to the show, Emily and Michael!',
                'both'
            ]
        },
        {
            "female2": [
                'Thanks for having me, John! Excited to be here and to talk about some of these amazing tools.',
                'left'
            ]
        },
        {
            "male2": [
                'Great to be here, John. Looking forward to the discussion!',
                'right'
            ]
        },
        {
            "male1": [
                "Let's kick off with some insights on some web development frameworks like Next.js, which has been making waves for its efficiency and features. Emily, could you share your thoughts on Next.js from your perspective as a frontend developer?",
                'both'
            ]
        },
        {
            "female2": [
                'Absolutely, John! Next.js is incredible for frontend development, especially with features like automatic page optimization and server-side rendering. It allows for a seamless transition from development to production, making life so much easier for developers who want to ensure their apps are fast and performant.',
                'left'
            ]
        },
        {
            "male1": [
                'And I saw in the documentation that it also provides built-in support for Google Fonts optimization. Have you used this feature?',
                'both'
            ]
        },
        {
            "female2": [
                "Yes, the `next/font` module! It's a fantastic feature. Fonts can be a huge pain point, slowing down your site if not handled right. With Next.js automatically optimizing fonts, it takes away the headache and lets me focus more on the user experience.",
                'left'
            ]
        },
        {
            "male1": [
                'Michael, shifting gears to backend development, Nest.js is another framework that stands out in terms of scalability with server-side applications. What’s your experience with it?',
                'both'
            ]
        },
        {
            "male2": [
                "Nest.js is a real gem for backend developers, John. It’s built on Node.js and comes with a lot of out-of-the-box functionality, including a strong dependency injection system. It's structured to create highly scalable applications, which is essential for maintaining efficiency as apps grow.",
                'right'
            ]
        },
        {
            "female2": [
                "And let's not forget it's really well-documented. Clear documentation makes a big difference, especially when you're tackling complex backend challenges.",
                'left'
            ]
        },
        {
            "male1": [
                'Documentation is indeed key. Speaking of which, I read about Pregame, an accessibility tool that helps with WCAG compliance right within your IDE. It sounds like a huge help for ensuring products are user-inclusive from the start.',
                'both'
            ]
        },
        {
            "female2": [
                "Exactly! Pregame provides real-time accessibility suggestions which make it much easier to address these issues actively instead of as a last-minute fix. It helps developers adhere to standards right from the coding phase, ensuring we're building something everyone can use.",
                'left'
            ]
        },
        {
            "male2": [
                'I agree, tools like Pregame can fundamentally change the accessibility landscape. Ensuring compliance at the development stage saves time and resources, ultimately delivering a better product.',
                'right'
            ]
        },
        {
            "male1": [
                "To wrap things up, it's clear that these tools and frameworks - Next.js, Nest.js, and Pregame - really streamline the development process, making it efficient and inclusive. Thank you, Emily and Michael, for sharing your amazing insights today.",
                'both'
            ]
        },
        {
            "female2": [ 
                'Thanks, John! It was great to dive into these topics.', 'left' ]
        },
        {
            "male2": [
                'Thank you, John. Always a pleasure to discuss tools that make our coding lives easier.',
                'right'
            ]
        },
        {
            "male1": [
                'And thank you to all our listeners. Tune in next time for more insights into the fascinating world of software development. Until then, keep documenting!',
                'both'
            ]
        }
    ]

    # Generate the podcast (dialog + background music) as a single audio file
    await tts.generate_podcast(
        texts, 
        music=["music1.mp3", 10, 3, 0.3], 
        filename="example-podcast3.mp3"
    )

if __name__ == "__main__":
    asyncio.run(main())
