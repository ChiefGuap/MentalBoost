# scripts/test_stt.py

from src.speech_recognizer import listen_and_transcribe

if __name__ == "__main__":
    text = listen_and_transcribe(duration=4)
    print("You said:", text)
