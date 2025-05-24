# src/speech_recognizer.py

import sounddevice as sd
import numpy as np
import whisper
import tempfile
import os
from scipy.io.wavfile import write

# 1. Load Whisper model once at import
_model = whisper.load_model("small")  # you can choose 'tiny', 'base', 'small', etc.

def record_audio(duration=5, fs=16000):
    """
    Records `duration` seconds of audio from the default microphone.
    Returns a NumPy array of shape (n_samples,).
    """
    print(f"⏺ Recording {duration}s of audio…")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # wait until recording is finished
    return audio.flatten(), fs

def transcribe_audio(audio: np.ndarray, fs: int) -> str:
    """
    Saves the audio buffer to a temporary WAV file, runs Whisper, and returns the transcript.
    """
    # write to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        write(tmp.name, fs, audio)
        tmp_path = tmp.name

    # run transcription
    result = _model.transcribe(tmp_path)
    # clean up
    os.remove(tmp_path)
    return result["text"].strip()

def listen_and_transcribe(duration=5) -> str:
    """
    Convenience function: record then transcribe in one go.
    """
    audio, fs = record_audio(duration)
    return transcribe_audio(audio, fs)
