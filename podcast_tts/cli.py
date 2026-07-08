"""Command-line interface: ``podcast-tts``.

Examples:
    podcast-tts say "Hello there" --speaker male1 -o hello.wav
    podcast-tts dialog script.json -o show.mp3 --subtitles srt
    podcast-tts dialog script.json -o show.mp3 --music intro.mp3 10 3 0.3

``script.json`` is a list of dialog entries, e.g.:
    [
      {"male1": ["Welcome to the show!", "both"]},
      {"female2": ["Hola a todos.", "left", {"language": "es"}]}
    ]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from . import __version__
from .core import PodcastTTS


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--engine", default="chattts", help="chattts | chatterbox | kokoro")
    parser.add_argument("--language", default="en", help="Default language code (en, es, ...)")
    parser.add_argument("--speed", type=int, default=5, help="Playback speed (ChatTTS)")
    parser.add_argument("--device", default=None, help="Force device: cuda | mps | cpu")


_EPILOG = """\
examples:
  podcast-tts say "Hello there" --speaker male1 -o hello.wav
  podcast-tts dialog script.json -o show.mp3 --subtitles srt
  podcast-tts dialog script.json -o show.mp3 --music intro.mp3 10 3 0.3
"""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="podcast-tts",
        description="Generate multi-speaker podcasts and dialogues (EN/ES) from the terminal.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"podcast_tts {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    say = sub.add_parser("say", help="Synthesize a single line of text.")
    _add_common(say)
    say.add_argument("text", help="Text to synthesize.")
    say.add_argument("--speaker", required=True, help="Voice name or id.")
    say.add_argument("--channel", default="both", choices=["left", "right", "both"])
    say.add_argument("--emotion", type=float, default=None, help="Intensity (Chatterbox).")
    say.add_argument("-o", "--output", default="output.wav")

    dialog = sub.add_parser("dialog", help="Render a dialog script (JSON) to audio.")
    _add_common(dialog)
    dialog.add_argument("script", help="Path to a JSON dialog script.")
    dialog.add_argument("-o", "--output", default="dialog.wav")
    dialog.add_argument("--pause", type=float, default=0.5, help="Pause between turns (s).")
    dialog.add_argument("--no-normalize", action="store_true", help="Disable volume normalize.")
    dialog.add_argument("--subtitles", choices=["srt", "vtt"], default=None)
    dialog.add_argument(
        "--music",
        nargs=4,
        metavar=("FILE", "FULL_VOL", "FADE", "TARGET_VOL"),
        default=None,
        help="Background music: file/url, full-volume secs, fade secs, target volume.",
    )
    return parser


async def _run(args: argparse.Namespace) -> str:
    tts = PodcastTTS(
        speed=args.speed,
        engine=args.engine,
        language=args.language,
        device=args.device,
    )

    if args.command == "say":
        return await tts.generate_tts(
            text=args.text,
            speaker=args.speaker,
            filename=args.output,
            channel=args.channel,
            emotion=args.emotion,
        )

    with open(args.script, encoding="utf-8") as handle:
        texts = json.load(handle)

    kwargs = dict(
        texts=texts,
        filename=args.output,
        pause_duration=args.pause,
        normalize=not args.no_normalize,
        subtitles=args.subtitles is not None,
        subtitle_format=args.subtitles or "srt",
    )
    if args.music:
        file, full_vol, fade, target = args.music
        music = [file, float(full_vol), float(fade), float(target)]
        return await tts.generate_podcast(music=music, **kwargs)
    return await tts.generate_dialog(**kwargs)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        output = asyncio.run(_run(args))
    except (ValueError, FileNotFoundError, NotImplementedError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Done: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
