"""A tiny Gradio web demo for podcast_tts.

Run it with:
    pip install "podcast_tts[demo]" "podcast_tts[chattts]"   # + any engine
    podcast-tts-demo

Two tabs: synthesize a single line, or render a full multi-speaker dialogue
(optionally over background music) with downloadable subtitles.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile

from . import __version__
from .core import PodcastTTS
from .engines import AVAILABLE_ENGINES

_EXAMPLE_SCRIPT = json.dumps(
    [
        {"male1": ["Welcome to the show!", "left"]},
        {"female2": ["Thanks for having me. [laugh]", "right"]},
        {"male1": ["Today we go bilingual."]},
        {"female2": ["Hola, gracias por la invitacion.", "right", {"language": "es"}]},
    ],
    indent=2,
)


def _run(coro):
    return asyncio.run(coro)


def _synthesize_line(engine, language, speaker, channel, emotion, text):
    if not text.strip():
        raise ValueError("Please enter some text.")
    tts = PodcastTTS(engine=engine, language=language)
    out = os.path.join(tempfile.mkdtemp(prefix="podcast_tts_"), "line.wav")
    _run(
        tts.generate_tts(
            text=text,
            speaker=speaker or "narrator",
            filename=out,
            channel=channel,
            emotion=emotion,
        )
    )
    return out


def _synthesize_dialog(
    engine, language, script_text, pause, subtitles, music_file, full_vol, fade, target_vol
):
    texts = json.loads(script_text)
    tts = PodcastTTS(engine=engine, language=language)
    out = os.path.join(tempfile.mkdtemp(prefix="podcast_tts_"), "dialog.wav")
    kwargs = dict(
        texts=texts,
        filename=out,
        pause_duration=float(pause),
        subtitles=bool(subtitles),
        subtitle_format="srt",
    )
    if music_file:
        _run(
            tts.generate_podcast(
                music=[music_file, float(full_vol), float(fade), float(target_vol)],
                **kwargs,
            )
        )
    else:
        _run(tts.generate_dialog(**kwargs))

    srt = f"{os.path.splitext(out)[0]}.srt"
    return out, (srt if subtitles and os.path.exists(srt) else None)


def build_ui():
    """Build and return the Gradio Blocks app."""
    import gradio as gr

    with gr.Blocks(title="podcast_tts") as demo:
        gr.Markdown(
            f"# podcast_tts &nbsp;`v{__version__}`\n"
            "Generate multi-speaker podcasts and dialogues (English + Spanish)."
        )

        with gr.Tab("Single line"):
            with gr.Row():
                engine1 = gr.Dropdown(list(AVAILABLE_ENGINES), value="chattts", label="Engine")
                language1 = gr.Textbox("en", label="Language (en, es, ...)")
            with gr.Row():
                speaker1 = gr.Textbox("male1", label="Speaker (name / voice id)")
                channel1 = gr.Radio(["left", "right", "both"], value="both", label="Channel")
                emotion1 = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Emotion (Chatterbox)")
            text1 = gr.Textbox(label="Text", lines=3, value="Hello! Welcome to our podcast.")
            go1 = gr.Button("Generate", variant="primary")
            audio1 = gr.Audio(label="Output", type="filepath")
            go1.click(
                _synthesize_line,
                [engine1, language1, speaker1, channel1, emotion1, text1],
                audio1,
            )

        with gr.Tab("Dialogue / podcast"):
            with gr.Row():
                engine2 = gr.Dropdown(list(AVAILABLE_ENGINES), value="chattts", label="Engine")
                language2 = gr.Textbox("en", label="Default language")
            script2 = gr.Code(value=_EXAMPLE_SCRIPT, language="json", label="Dialogue script")
            with gr.Row():
                pause2 = gr.Slider(0.0, 2.0, value=0.5, step=0.1, label="Pause between turns (s)")
                subtitles2 = gr.Checkbox(value=True, label="Export subtitles (.srt)")
            with gr.Accordion("Background music (optional)", open=False):
                music2 = gr.Textbox(label="Music file path or URL", value="")
                with gr.Row():
                    full_vol2 = gr.Number(value=10, label="Full-volume secs")
                    fade2 = gr.Number(value=3, label="Fade secs")
                    target2 = gr.Number(value=0.3, label="Volume under speech")
            go2 = gr.Button("Generate", variant="primary")
            audio2 = gr.Audio(label="Output", type="filepath")
            subs2 = gr.File(label="Subtitles")
            go2.click(
                _synthesize_dialog,
                [engine2, language2, script2, pause2, subtitles2, music2, full_vol2, fade2, target2],
                [audio2, subs2],
            )

    return demo


def main() -> int:
    try:
        import gradio  # noqa: F401
    except ImportError:
        print(
            "Gradio is not installed. Install the demo extra with:\n"
            "    pip install 'podcast_tts[demo]'"
        )
        return 1
    build_ui().launch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
