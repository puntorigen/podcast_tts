import ChatTTS, torch, torchaudio
import os, json, asyncio, inflect, re
from pydub import AudioSegment

# Initialize inflect engine
p = inflect.engine()

RESERVED_TAGS = {"uv_break", "laugh", "lbreak", "break"}

def remove_brackets(text):
    """
    Removes unwanted brackets but preserves reserved tags.

    Args:
        text (str): The input text to clean.

    Returns:
        str: The cleaned text with reserved tags preserved.
    """
    text = re.sub(r'\[(?!\b(?:' + '|'.join(RESERVED_TAGS) + r')\b)(.*?)\]', '', text)  # Remove unwanted brackets
    return re.sub(r'\s+', ' ', text).strip()  # Normalize spaces

def normalize_text(input_string):
    """
    Converts numbers to words and formats text for better readability,
    while preserving reserved tags.

    Args:
        input_string (str): The text to normalize.

    Returns:
        str: The normalized text.
    """
    def replacer(match):
        number = match.group(0)
        return p.number_to_words(int(number))

    # Preserve reserved tags while normalizing other parts
    def preserve_tags(match):
        tag = match.group(0)
        return tag if tag[1:-1] in RESERVED_TAGS else replacer(match)

    result = re.sub(r'\d+', replacer, input_string)
    result = re.sub(r'(\w)-(\w)', r'\1 \2', result)
    return result

def prepare_text_for_conversion(
    text: str,
    min_line_length: int = 30,
    merge_size: int = 3,
    break_tag: str = "[uv_break]",
    max_chunk_length: int = 200
) -> list:
    """
    Prepares the input text for text-to-speech conversion by cleaning, splitting, and formatting.

    Args:
        text (str): The input text to be processed.
        min_line_length (int): Minimum line length before merging short lines.
        merge_size (int): Number of lines to include in each chunk for batch processing.
        break_tag (str): Tag to insert between concatenated short lines.
        max_chunk_length (int): Maximum allowed length for a chunk.

    Returns:
        list: A list of processed text chunks ready for conversion.
    """
    def clean_text(text):
        # Remove unwanted characters while preserving reserved tags
        text = re.sub(r"[^\w\s.,!?;:'\"-\[\]]", "", text)
        return re.sub(r"\s+", " ", text).strip()

    def split_by_punctuation(text, max_chunk_length):
        punctuation_marks = ".!?;"
        result, start = [], 0
        for match in re.finditer(r"[{}]".format(re.escape(punctuation_marks)), text):
            end = match.end()
            if end - start > max_chunk_length:
                result.extend([text[i:i + max_chunk_length] for i in range(start, end, max_chunk_length)])
                start = end
            else:
                result.append(text[start:end].strip())
                start = end
        if start < len(text):
            result.append(text[start:].strip())
        return result

    cleaned_text = clean_text(text)
    split_chunks = split_by_punctuation(cleaned_text, max_chunk_length)

    retext, short_text = [], ""
    for line in split_chunks:
        if len(line) < min_line_length:
            short_text += f"{line} {break_tag} "
            if len(short_text) > min_line_length:
                retext.append(short_text.strip())
                short_text = ""
        else:
            retext.append(short_text + line)
            short_text = ""

    if len(short_text) > min_line_length or not retext:
        retext.append(short_text.strip())
    elif short_text:
        retext[-1] += f" {break_tag} {short_text.strip()}"

    return [retext[i:i + merge_size] for i in range(0, len(retext), merge_size)]

class PodcastTTS:
    """
    A helper class for generating audio (TTS) using ChatTTS for podcasts and dialogues.

    Args:
        speed (int): The playback speed for generated audio.
    """
    def __init__(self, speed: int = 5):
        chat = ChatTTS.Chat()
        chat.load(compile=True)
        self.chat = chat
        self.speed = speed
        self.sampling_rate = 24000
        self.voices_dir = os.path.join(os.getcwd(), "voices")
        os.makedirs(self.voices_dir, exist_ok=True)

    async def create_speaker(self, speaker_name: str) -> str:
        """
        Creates a new speaker profile.

        Args:
            speaker_name (str): The name of the speaker.

        Returns:
            str: The speaker profile data.
        """
        if not speaker_name:
            raise ValueError("Speaker name cannot be empty.")
        voice = self.chat.speaker.sample_random()
        file_path = os.path.join(self.voices_dir, f"{speaker_name}.txt")
        await asyncio.to_thread(self._write_to_file, file_path, voice)
        return voice

    def _write_to_file(self, file_path: str, data: str):
        """
        Writes data to a file.

        Args:
            file_path (str): Path to the file.
            data (str): Data to write.
        """
        with open(file_path, "w") as f:
            f.write(data)

    async def load_speaker(self, speaker_name: str) -> str:
        """
        Loads a speaker profile.

        Args:
            speaker_name (str): The name of the speaker.

        Returns:
            str: The speaker profile data.
        """
        file_path = os.path.join(self.voices_dir, f"{speaker_name}.txt")
        if not os.path.exists(file_path):
            print(f"Speaker '{speaker_name}' not found. Creating new profile.")
            return await self.create_speaker(speaker_name)
        return await asyncio.to_thread(self._read_from_file, file_path)

    def _read_from_file(self, file_path: str) -> str:
        """
        Reads data from a file.

        Args:
            file_path (str): Path to the file.

        Returns:
            str: The data read from the file.
        """
        with open(file_path, "r") as f:
            return f.read()

    async def generate_wav(self, text: str, speaker: str, filename: str = "generated_tts.wav", channel: str = "both"):
        """
        Generates a WAV file from text using a specified speaker profile, with channel control.

        Args:
            text (str): The input text to synthesize.
            speaker (str): The speaker profile as a plain string.
            filename (str): The output WAV file name (default: "generated_tts.wav").
            channel (str): The audio channel ("left", "right", or "both").

        Raises:
            ValueError: If the text is empty, speaker data is invalid, or the channel is invalid.
        """
        if not text:
            raise ValueError("Text cannot be empty.")
        if not speaker:
            raise ValueError("Speaker data cannot be empty.")
        if channel not in {"left", "right", "both"}:
            raise ValueError("Channel must be 'left', 'right', or 'both'.")

        # Prepare text chunks using prepare_text_for_conversion
        text_chunks = prepare_text_for_conversion(text)

        # Prepare inference parameters
        params_infer_code = ChatTTS.Chat.InferCodeParams(
            prompt=f"[speed_{self.speed}]",
            spk_emb=speaker,
            temperature=0.15,
            top_P=0.75,
            top_K=20,
        )
        params_refine_text = ChatTTS.Chat.RefineTextParams(
            prompt='[oral_0][laugh_3][break_4]',
            temperature=0.12,
        )

        # Generate audio for each chunk and collect the waveforms
        generated_wavs = []
        for i, chunk in enumerate(text_chunks):
            chunk_text = " ".join(chunk)  # Combine lines in the chunk
            normalized = normalize_text(chunk_text)
            print(f"Generating audio for chunk {i + 1}/{len(text_chunks)}: {normalized}")

            # Generate audio waveform for the chunk
            wavs = await asyncio.to_thread(
                self.chat.infer,
                normalize_text(chunk_text),
                lang="en",
                skip_refine_text=False,
                params_refine_text=params_refine_text,
                params_infer_code=params_infer_code,
                do_text_normalization=False,
                do_homophone_replacement=True,
            )
            waveform = torch.tensor(wavs[0]).unsqueeze(0)

            # Adjust the channel
            waveform = self._adjust_channel(waveform, channel)
            generated_wavs.append(waveform)

        # Merge all generated waveforms
        if len(generated_wavs) > 1:
            merged_waveform = torch.cat(generated_wavs, dim=1)
        else:
            merged_waveform = generated_wavs[0]

        # Save the merged waveform as a WAV file
        try:
            await asyncio.to_thread(
                torchaudio.save,
                filename,
                merged_waveform,
                self.sampling_rate,
            )
        except TypeError as e:
            # Fallback for older versions of torchaudio
            if "unsqueeze" in str(e):
                await asyncio.to_thread(
                    torchaudio.save,
                    filename,
                    merged_waveform.squeeze(0),
                    self.sampling_rate,
                )
            else:
                raise RuntimeError(f"Failed to save WAV file: {e}")

        print(f"Audio saved to {filename}")

    def _adjust_channel(self, waveform: torch.Tensor, channel: str) -> torch.Tensor:
        """
        Adjusts the audio waveform to play on the specified channel.

        Args:
            waveform (torch.Tensor): The input audio waveform (shape: [1, num_samples]).
            channel (str): The target channel ("left", "right", or "both").

        Returns:
            torch.Tensor: The adjusted waveform with shape [2, num_samples].
        """
        num_samples = waveform.size(1)
        left = waveform if channel in {"left", "both"} else torch.zeros_like(waveform)
        right = waveform if channel in {"right", "both"} else torch.zeros_like(waveform)
        return torch.cat([left, right], dim=0)

    async def generate_dialog_wav(
        self, 
        texts: list[dict], 
        filename: str = "generated_dialog.wav", 
        pause_duration: float = 0.5, 
        normalize: bool = True
    ):
        """
        Generates a single WAV file with a sequence of audio clips from dialog texts.

        Args:
            texts (list[dict]): An array of objects where each key is a speaker and value is an array of strings.
                                Example: {"Speaker1": ["Hello", "left"]}.
            filename (str): The name of the output WAV file.
            pause_duration (float): Duration of the pause between roles in seconds.
            normalize (bool): Whether to normalize the volume of each audio segment

        Returns:
            str: The filename of the merged WAV file.
        """
        if not texts:
            raise ValueError("The texts array cannot be empty.")
        if pause_duration < 0:
            raise ValueError("Pause duration must be non-negative.")
        
        generated_wavs = []
        for entry in texts:
            if len(entry) != 1:
                raise ValueError("Each entry in texts must contain exactly one speaker and their value as a list.")
            
            speaker_name, content = next(iter(entry.items()))
            if not isinstance(content, list) or len(content) < 1:
                raise ValueError(f"Invalid entry: {content}")
            
            text, channel = content[0], content[1] if len(content) > 1 else "both"
            if channel not in {"left", "right", "both"}:
                raise ValueError(f"Invalid channel '{channel}'.")
            
            speaker_profile = await self.load_speaker(speaker_name)
            temp_filename = f"{speaker_name}_temp.wav"
            await self.generate_wav(text, speaker_profile, temp_filename, channel)

            waveform, sample_rate = torchaudio.load(temp_filename)
            if normalize:
                waveform = self._normalize_volume(waveform)

            generated_wavs.append(waveform)
            os.remove(temp_filename)

            if entry != texts[-1]:
                silence = torch.zeros((waveform.size(0), int(pause_duration * self.sampling_rate)))
                generated_wavs.append(silence)

        merged_waveform = torch.cat(generated_wavs, dim=1)
        await asyncio.to_thread(torchaudio.save, filename, merged_waveform, self.sampling_rate)
        return filename

    def _normalize_volume(self, waveform: torch.Tensor, target_rms: float = 0.1) -> torch.Tensor:
        """
        Normalizes the waveform to a target RMS amplitude.

        Args:
            waveform (torch.Tensor): The input audio waveform.
            target_rms (float): The target RMS amplitude.

        Returns:
            torch.Tensor: The normalized waveform.
        """
        current_rms = torch.sqrt(torch.mean(waveform ** 2))
        if current_rms > 0:
            waveform *= target_rms / current_rms
        return waveform

    async def generate_podcast(
        self,
        texts: list[dict],
        music: list,
        filename: str = "podcast.wav",
        pause_duration: float = 0.5,
        normalize: bool = True,
    ):
        """
        Generates a podcast WAV file by combining TTS-generated dialog and background music.

        Args:
            texts (list[dict]): An array of objects where each key is a speaker and value is an array.
                                Example: {"Speaker1": ["Hello", "left"]}.
            music (list): A list where:
                          - music[0] is the path to the music file (MP3 or WAV).
                          - music[1] is the duration (in seconds) the music plays at full volume before dialog starts.
                          - music[2] is the fade duration in seconds (for fading out from full to sustained volume).
                          - music[3] is the sustained volume level (0.0 to 1.0).
            filename (str): The name of the output podcast file.
            pause_duration (float): Duration of the pause between roles in seconds.
            normalize (bool): Whether to normalize the volume of the TTS segments.

        Returns:
            str: The filename of the generated podcast.
        """
        if not music or len(music) != 4:
            raise ValueError("The 'music' argument must be a list of [music_path, full_volume_duration, fade_duration, volume].")

        music_path, full_volume_duration, fade_duration, music_volume = music
        if not os.path.exists(music_path):
            raise FileNotFoundError(f"Music file '{music_path}' not found.")
        if not (0.0 <= music_volume <= 1.0):
            raise ValueError("Music volume must be between 0.0 and 1.0.")
        if fade_duration < 0 or full_volume_duration < 0:
            raise ValueError("Fade duration and full-volume duration must be non-negative.")

        # Step 1: Generate the dialog WAV using `generate_dialog_wav`
        dialog_filename = "dialog_temp.wav"
        await self.generate_dialog_wav(
            texts, filename=dialog_filename, pause_duration=pause_duration, normalize=normalize
        )

        # Step 2: Load the generated dialog audio and the music
        dialog_audio = AudioSegment.from_file(dialog_filename)
        music_audio = AudioSegment.from_file(music_path)

        # Step 3: Handle music fade-in, full volume, and fade-out to sustained volume
        fade_in_milliseconds = int(fade_duration * 1000)
        full_volume_milliseconds = int(full_volume_duration * 1000)
        fade_to_sustained_milliseconds = fade_in_milliseconds  # Use the same fade duration for drop to sustained volume
        fade_out_milliseconds = fade_in_milliseconds  # Same duration for the final fade-out

        # Apply fade-in
        music_audio = music_audio.fade_in(fade_in_milliseconds)

        # Split the music into three parts:
        # 1. Initial full-volume part
        # 2. Fade-out to sustained volume part
        # 3. Remaining music at sustained volume
        music_start = music_audio[:full_volume_milliseconds]
        music_to_fade = music_audio[full_volume_milliseconds: full_volume_milliseconds + fade_to_sustained_milliseconds]
        music_remainder = music_audio[full_volume_milliseconds + fade_to_sustained_milliseconds:]

        # Fade-out to sustained volume
        music_to_fade = music_to_fade.fade_out(fade_to_sustained_milliseconds).apply_gain(-(1 - music_volume) * 40)

        # Adjust sustained volume
        music_remainder = music_remainder - (1 - music_volume) * 40

        # Combine the parts back together
        music_audio = music_start + music_to_fade + music_remainder

        # Append silence to dialog for the initial full-volume duration
        silent_gap = AudioSegment.silent(duration=full_volume_milliseconds)
        dialog_audio = silent_gap + dialog_audio

        # Apply fade-out to the music's last part
        if len(music_audio) > len(dialog_audio):
            fade_out_start = len(dialog_audio) - fade_out_milliseconds
            if fade_out_start > 0:
                music_audio = (
                    music_audio[:fade_out_start]
                    + music_audio[fade_out_start:].fade_out(fade_out_milliseconds)
                )

        # Step 4: Loop the music if it's shorter than the dialog
        if len(music_audio) < len(dialog_audio):
            music_audio = music_audio * ((len(dialog_audio) // len(music_audio)) + 1)

        # Trim the music to the dialog duration
        music_audio = music_audio[: len(dialog_audio)]

        # Step 5: Overlay the dialog onto the music
        podcast_audio = music_audio.overlay(dialog_audio)

        # Step 6: Export the final podcast audio
        podcast_audio.export(filename, format="wav")

        # Clean up temporary files
        os.remove(dialog_filename)

        print(f"Podcast saved to {filename}")
        return filename
