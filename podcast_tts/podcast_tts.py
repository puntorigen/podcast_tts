import ChatTTS, torch, torchaudio
import os, json, asyncio, inflect, re, regex
import requests
from hashlib import md5
from pydub import AudioSegment

# Initialize inflect engine
p = inflect.engine()

RESERVED_TAGS = {
    "uv_break", 
    "laugh", "laugh_1", "laugh_2", "laugh_3", "laugh_4", "laugh_5",  
    "lbreak", "break"
}

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
    Converts numbers to words, preserves contractions and reserved tags, replaces invalid characters with [uv_break],
    ensures hyphens are replaced with spaces, and prevents consecutive [uv_break] instances.

    Args:
        input_string (str): The text to normalize.

    Returns:
        str: The normalized text.
    """
    # Convert numbers to words, excluding numbers inside square brackets
    def replacer(match):
        number = match.group(0)
        return " " + p.number_to_words(int(number)) + " "

    # Match numbers not inside square brackets
    result = regex.sub(r'(?<!\[.*)\b\d+\b(?!.*\])', replacer, input_string)

    # Preserve contractions and possessives
    result = regex.sub(r"(?<=\b\w)'(?=\w\b)", "'", result)  # Preserve contractions
    result = regex.sub(r"(?<=\b\w)'(?=s\b)", "'", result)  # Preserve possessives

    # Replace invalid characters with [uv_break] if there's no nearby [uv_break]
    def replace_invalid(match):
        char = match.group(0)
        # If there's already a nearby [uv_break], replace with a space
        return " [uv_break]" if "[uv_break]" not in input_string[max(0, match.start() - 10): match.end() + 10] else " "

    result = regex.sub(r'[!":]', replace_invalid, result)

    # Replace hyphens with spaces
    result = regex.sub(r'-', ' ', result)

    # Consolidate repeated [uv_break]
    result = re.sub(r'\[uv_break\](?:\s*\[uv_break\])+', '[uv_break]', result)

    # Join separated `] [` into `][`
    result = re.sub(r'\]\s*\[', '][', result)

    # Fix spacing and strip extra whitespace
    result = re.sub(r'\s+', ' ', result).strip()

    return result

def prepare_text_for_conversion(
    text: str,
    min_line_length: int = 30,
    merge_size: int = 3,
    break_tag: str = "[uv_break]",
    max_chunk_length: int = 150
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
    # Step 1: Normalize the text first
    text = normalize_text(text)

    def split_by_punctuation(text, max_chunk_length):
        """
        Splits text by punctuation while ensuring chunks do not exceed max_chunk_length.
        """
        punctuation_marks = ".!?;"
        result, start = [], 0
        for match in re.finditer(r"[{}]".format(re.escape(punctuation_marks)), text):
            end = match.end()
            if end - start > max_chunk_length:
                result.append(text[start:end].strip())
                start = end
            else:
                result.append(text[start:end].strip())
                start = end
        if start < len(text):
            result.append(text[start:].strip())
        return result

    # Split the text into chunks by punctuation
    split_chunks = split_by_punctuation(text, max_chunk_length)

    # Combine smaller chunks into larger ones while respecting min_line_length
    retext, short_text = [], ""
    for chunk in split_chunks:
        if len(chunk) < min_line_length:
            short_text += f"{chunk} {break_tag} "
            if len(short_text) >= min_line_length:
                retext.append(short_text.strip())
                short_text = ""
        else:
            if short_text:
                chunk = f"{short_text.strip()} {break_tag} {chunk}"
                short_text = ""
            retext.append(chunk)

    # If there's leftover short text, append it
    if short_text:
        retext.append(short_text.strip())

    # Group chunks into batches of merge_size
    final_chunks = [
        retext[i:i + merge_size] for i in range(0, len(retext), merge_size)
    ]

    return final_chunks


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

        # User-defined voices directory
        self.voices_dir = os.path.join(os.getcwd(), "voices")
        os.makedirs(self.voices_dir, exist_ok=True)

        # Default voices directory inside the package
        self.default_voices_dir = os.path.join(os.path.dirname(__file__), "default_voices")
        os.makedirs(self.default_voices_dir, exist_ok=True)

        # Cache directory for downloaded files
        self.cache_dir = os.path.join(os.getcwd(), "cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    def _download_and_cache_file(self, url: str) -> str:
        """
        Downloads a file from a URL and caches it locally.

        Args:
            url (str): The URL to download.

        Returns:
            str: The path to the cached file.
        """
        # Generate a unique filename based on the URL
        file_hash = md5(url.encode('utf-8')).hexdigest()
        cached_file_path = os.path.join(self.cache_dir, f"{file_hash}.mp3")

        # Check if the file is already cached
        if os.path.exists(cached_file_path):
            print(f"Using cached file for URL: {url}")
            return cached_file_path

        # Download the file
        print(f"Downloading music from URL: {url}")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(cached_file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
            print(f"File cached at: {cached_file_path}")
        else:
            raise ValueError(f"Failed to download file from URL: {url} (status code: {response.status_code})")

        return cached_file_path
    
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
        Loads a speaker profile. First searches in the user-defined voices directory,
        then in the package-level default voices directory.

        Args:
            speaker_name (str): The name of the speaker.

        Returns:
            str: The speaker profile data.
        """
        # Check in the user-defined voices directory
        user_file_path = os.path.join(self.voices_dir, f"{speaker_name}.txt")
        if os.path.exists(user_file_path):
            return await asyncio.to_thread(self._read_from_file, user_file_path)

        # Check in the package-level default voices directory
        default_file_path = os.path.join(self.default_voices_dir, f"{speaker_name}.txt")
        if os.path.exists(default_file_path):
            return await asyncio.to_thread(self._read_from_file, default_file_path)

        # If not found in either location, create a new speaker profile
        print(f"Speaker '{speaker_name}' not found. Creating new profile.")
        return await self.create_speaker(speaker_name)

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
            prompt='[oral_2][laugh_0][break_6]',
            temperature=0.12,
            max_new_token=500
        )

        # Generate audio for each chunk and collect the waveforms
        generated_wavs = []
        for i, chunk in enumerate(text_chunks):
            chunk_text = "".join(chunk)  # Combine lines in the chunk
            chunk_text = chunk_text.replace(".", ". ")  # Add space after periods
            normalized = normalize_text(chunk_text)
            print(f"Generating audio for chunk {i + 1}/{len(text_chunks)}: {normalized}")

            # Generate audio waveform for the chunk
            wavs = await asyncio.to_thread(
                self.chat.infer,
                normalized,
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
            filename (str): Output podcast filename (must end with .wav or .mp3).
            pause_duration (float): Duration of pause between dialog segments.
            normalize (bool): Normalize the dialog audio volume.

        Returns:
            str: The filename of the generated podcast.
        """
        if not music or len(music) != 4:
            raise ValueError("Music argument must be a list of [file, full_volume_duration, fade_duration, target_volume].")

        if not filename.endswith((".wav", ".mp3")):
            raise ValueError("Filename must have a .wav or .mp3 extension.")

        music_file_or_url, full_volume_duration, fade_duration, target_volume = music

        # Check if the music source is a URL or a local file
        if music_file_or_url.startswith("http"):
            music_file = self._download_and_cache_file(music_file_or_url)
        else:
            music_file = music_file_or_url
            
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

        # Ensure music is stereo
        if music_waveform.size(0) == 1:
            music_waveform = torch.cat([music_waveform, music_waveform], dim=0)

        # Repeat the music if it's shorter than the required duration
        # dialog_samples = dialog_waveform.size(1)
        # Calculate total needed length dynamically
        fade_samples = int(fade_duration * dialog_sample_rate)  # Fade duration in samples
        full_volume_samples = int(full_volume_duration * dialog_sample_rate)  # Full volume duration in samples
        post_dialog_full_volume_samples = int(10 * dialog_sample_rate)  # 10 seconds at 100% volume
        dialog_samples = dialog_waveform.size(1)  # Number of samples in the dialog

        total_needed_length = (
            fade_samples  # Initial fade-in
            + full_volume_samples  # Full volume before dialog
            + dialog_samples  # Length of dialog
            + fade_samples  # Fade-up to 100% after dialog
            + post_dialog_full_volume_samples  # Full volume for 10 seconds
            + fade_samples  # Final fade-out
        )
        #total_needed_length = dialog_samples + int((full_volume_duration + fade_duration) * dialog_sample_rate)
        while music_waveform.size(1) < total_needed_length:
            music_waveform = torch.cat([music_waveform, music_waveform], dim=1)

        # Trim the music to the exact required length
        music_waveform = music_waveform[:, :total_needed_length]

        # Ensure dialog is stereo
        if dialog_waveform.size(0) == 1:
            dialog_waveform = torch.cat([dialog_waveform, dialog_waveform], dim=0)

        #fade_samples = int(fade_duration * dialog_sample_rate)
        #full_volume_samples = int(full_volume_duration * dialog_sample_rate)

        # Prepare music with fade in, fade out, and volume adjustments
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
        fade_out_end = start_dialog
        adjusted_music[:, fade_out_end - fade_samples:fade_out_end] = (
            music_waveform[:, fade_out_end - fade_samples:fade_out_end] * fade_to_target
        )

        # 4. Start the dialog and keep music at target_volume
        adjusted_music[:, fade_out_end:fade_out_end + dialog_samples] = (
            music_waveform[:, fade_out_end:fade_out_end + dialog_samples] * target_volume
        )

        # 5. Fade up to 100% 3 seconds before dialog ends
        fade_up_start = fade_out_end + dialog_samples - fade_samples
        fade_up = torch.linspace(target_volume, 1, fade_samples)
        if fade_up_start < adjusted_music.size(1):  # Ensure fade-up range is valid
            adjusted_music[:, fade_up_start:fade_out_end + dialog_samples] = (
                music_waveform[:, fade_up_start:fade_out_end + dialog_samples] * fade_up
            )

        # 6. Maintain 100% volume for full_volume_duration after dialog
        end_dialog = fade_out_end + dialog_samples
        adjusted_music[:, end_dialog:end_dialog + full_volume_samples] = music_waveform[
            :, end_dialog:end_dialog + full_volume_samples
        ]

        # 7. Fade out from 100% to 0% over fade_duration
        fade_out_start = end_dialog + full_volume_samples
        fade_out = torch.linspace(1, 0, fade_samples)
        if fade_out_start < adjusted_music.size(1):  # Ensure fade-out range is valid
            adjusted_music[:, fade_out_start:fade_out_start + fade_samples] = (
                music_waveform[:, fade_out_start:fade_out_start + fade_samples] * fade_out
            )

        # Trim the music to stop after fade-out
        total_music_length = min(fade_out_start + fade_samples, adjusted_music.size(1))
        adjusted_music = adjusted_music[:, :total_music_length]

        # Mix dialog audio with music
        total_length = max(dialog_waveform.size(1) + fade_out_end, adjusted_music.size(1))
        final_audio = torch.zeros((2, total_length))

        # Place the dialog after the music's full-volume phase
        final_audio[:, fade_out_end:fade_out_end + dialog_waveform.size(1)] += dialog_waveform
        final_audio[:, :adjusted_music.size(1)] += adjusted_music[:, :total_length]

        # Save the combined audio to the output file
        if filename.endswith(".wav"):
            await asyncio.to_thread(torchaudio.save, filename, final_audio, dialog_sample_rate, format="wav")
        elif filename.endswith(".mp3"):
            await asyncio.to_thread(torchaudio.save, filename, final_audio, dialog_sample_rate, format="mp3")

        # Clean up temporary files
        os.remove(dialog_audio_file)

        return filename
