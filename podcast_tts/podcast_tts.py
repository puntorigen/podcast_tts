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
        self, texts: list[dict], music: list, filename: str = "podcast.wav", pause_duration: float = 0.5, normalize: bool = True
    ):
        """
        Generates a podcast audio file with background music.

        Args:
            texts (list[dict]): Dialog text data, with speakers and optional channels.
            music (list): Background music configuration [file, full_volume_duration, fade_duration, target_volume].
                        Example: ["background_music.mp3", 10, 3, 0.3].
            filename (str): Output podcast filename.
            pause_duration (float): Duration of pause between dialog segments.
            normalize (bool): Normalize the dialog audio volume.

        Returns:
            str: The filename of the generated podcast.
        """
        if not music or len(music) != 4:
            raise ValueError("Music argument must be a list of [file, full_volume_duration, fade_duration, target_volume].")

        music_file, full_volume_duration, fade_duration, target_volume = music

        # Generate the dialog audio
        dialog_audio_file = "dialog_temp.wav"
        await self.generate_dialog_wav(texts, dialog_audio_file, pause_duration, normalize)

        # Load dialog audio and music
        dialog_waveform, dialog_sample_rate = torchaudio.load(dialog_audio_file)
        music_waveform, music_sample_rate = torchaudio.load(music_file)

        # Resample music to match dialog sample rate if needed
        if music_sample_rate != dialog_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=music_sample_rate, new_freq=dialog_sample_rate)
            music_waveform = resampler(music_waveform)

        # Make both waveforms stereo if needed
        if dialog_waveform.size(0) == 1:
            dialog_waveform = torch.cat([dialog_waveform, dialog_waveform], dim=0)
        if music_waveform.size(0) == 1:
            music_waveform = torch.cat([music_waveform, music_waveform], dim=0)

        dialog_length = dialog_waveform.size(1) / dialog_sample_rate
        fade_samples = int(fade_duration * dialog_sample_rate)
        full_volume_samples = int(full_volume_duration * dialog_sample_rate)

        # Prepare music with fade in, fade out, and volume adjustments
        music_length = music_waveform.size(1)
        adjusted_music = torch.zeros_like(music_waveform)

        # 1. Fade in from 0% to 100% for the first fade_duration
        fade_in = torch.linspace(0, 1, fade_samples)
        adjusted_music[:, :fade_samples] = music_waveform[:, :fade_samples] * fade_in

        # 2. Maintain full volume for the remaining full_volume_duration
        adjusted_music[:, fade_samples:fade_samples + full_volume_samples - fade_samples] = music_waveform[
            :, fade_samples:fade_samples + full_volume_samples - fade_samples
        ]

        # 3. Fade out to target_volume during the next fade_duration (while dialog starts)
        fade_to_target = torch.linspace(1, target_volume, fade_samples)
        start_dialog = fade_samples + full_volume_samples
        adjusted_music[:, start_dialog - fade_samples:start_dialog] = (
            music_waveform[:, start_dialog - fade_samples:start_dialog] * fade_to_target
        )

        # 4. Continue at target_volume during dialog playback
        dialog_samples = int(dialog_length * dialog_sample_rate)
        adjusted_music[:, start_dialog:start_dialog + dialog_samples] = (
            music_waveform[:, start_dialog:start_dialog + dialog_samples] * target_volume
        )

        # 5. Fade up to 100% 3 seconds before dialog ends
        fade_up_start = start_dialog + dialog_samples - fade_samples
        fade_up = torch.linspace(target_volume, 1, fade_samples)
        adjusted_music[:, fade_up_start:start_dialog + dialog_samples] = (
            music_waveform[:, fade_up_start:start_dialog + dialog_samples] * fade_up
        )

        # 6. Maintain 100% volume for full_volume_duration after dialog
        end_dialog = start_dialog + dialog_samples
        adjusted_music[:, end_dialog:end_dialog + full_volume_samples] = music_waveform[
            :, end_dialog:end_dialog + full_volume_samples
        ]

        # 7. Fade out from 100% to 0% over fade_duration
        fade_out_start = end_dialog + full_volume_samples
        fade_out = torch.linspace(1, 0, fade_samples)
        adjusted_music[:, fade_out_start:fade_out_start + fade_samples] = (
            music_waveform[:, fade_out_start:fade_out_start + fade_samples] * fade_out
        )

        # Trim the music to end after the fade-out
        total_music_length = fade_out_start + fade_samples
        adjusted_music = adjusted_music[:, :total_music_length]

        # Mix dialog audio with music
        total_length = max(dialog_waveform.size(1), adjusted_music.size(1))
        final_audio = torch.zeros((2, total_length))
        final_audio[:, :dialog_waveform.size(1)] += dialog_waveform
        final_audio[:, :adjusted_music.size(1)] += adjusted_music[:, :total_length]

        # Save the combined audio to the output file
        await asyncio.to_thread(torchaudio.save, filename, final_audio, dialog_sample_rate)

        # Clean up temporary files
        os.remove(dialog_audio_file)

        return filename

