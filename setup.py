from setuptools import setup, find_packages

setup(
    name="podcast_tts",
    version="0.0.1",
    description="Generate high-quality TTS audio for podcasts and dialogues.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Pablo Schaffner",
    author_email="pablo@puntorigen.com",
    url="https://github.com/puntorigen/podcast_tts",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchaudio",
        "ChatTTS",
        "inflect",
        "pydub",
        "regex"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
)
