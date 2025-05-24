# crew_ai_agents_functional_metadata.py
# Functional-style pipeline using direct Agent.run calls (avoiding Crew.run_agent) with metadata preserved

from crewai import Agent, Crew
import cv2
from deepface import DeepFace
import openai
import sounddevice as sd
import numpy as np
import queue
import threading
import time

# --- Audio capture setup ---
audio_queue = queue.Queue()
AUDIO_SAMPLE_RATE = 16000  # Hz
AUDIO_CHUNK_DURATION = 2   # seconds
AUDIO_CHUNK_SAMPLES = AUDIO_SAMPLE_RATE * AUDIO_CHUNK_DURATION

def audio_callback(indata, frames, time_info, status):
    """Callback to collect audio data chunks."""
    if status:
        print(f"Audio status: {status}")
    audio_queue.put(indata.copy())

# 1. Define task functions

def detect_emotion(frame, backend="ssd"):
    """Analyze a video frame (NumPy array) and return dominant emotion."""
    analysis = DeepFace.analyze(
        img=frame,
        actions=["emotion"],
        detector_backend=backend,
        enforce_detection=False
    )
    if isinstance(analysis, list):
        analysis = analysis[0]
    return {"emotion": analysis.get("dominant_emotion", "neutral")}


def transcribe_speech(audio_data, model="whisper-1"):
    """Transcribe an audio chunk (NumPy array) to text."""
    import io
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, audio_data, AUDIO_SAMPLE_RATE, format='WAV')
    buf.seek(0)
    transcript = openai.Audio.transcribe(
        model=model,
        file=buf
    )["text"]
    return {"transcript": transcript}


def compose_prompt(context):
    """Fuse emotion and transcript into an LLM prompt."""
    emotion = context.get("emotion", "neutral")
    transcript = context.get("transcript", "")
    prompt = (
        f"The user said: '{transcript}'\n"
        f"Detected emotion: {emotion}.\n"
        "Respond empathetically as a therapist and suggest coping strategies."
    )
    return {"llm_prompt": prompt}


def generate_response(prompt_packet, model="claude-v1"):
    """Call Claude (via OpenAI API) to generate a response."""
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt_packet["llm_prompt"]}]
    )
    return {"raw_reply": response.choices[0].message.content}


def postprocess_and_dispatch(packet):
    """Adjust tone if needed and dispatch the reply to UI or console."""
    emotion = packet.get("emotion", "neutral")
    raw = packet.get("raw_reply", "")
    if emotion in ["sad", "angry"]:
        raw = "[Supportive Tone] " + raw
    print(f"Therapist: {raw}\n")
    return {}

# 2. Instantiate Agents with explicit metadata
emotion_agent = Agent(
    name="EmotionAgent",
    task=detect_emotion,
    role="vision",
    goal="Detect the user’s predominant facial emotion in each frame.",
    backstory="You are a skilled emotion recognition model trained for real-time analysis.",
    verbose=True
)

speech_agent = Agent(
    name="SpeechAgent",
    task=transcribe_speech,
    role="audio",
    goal="Transcribe live user speech to text.",
    backstory="You are a fast, accurate speech recognizer specialized in conversational audio.",
    verbose=True
)

prompt_agent = Agent(
    name="PromptAgent",
    task=compose_prompt,
    role="logic",
    goal="Create a contextual prompt from emotion and transcript.",
    backstory="You excel at crafting clear, effective prompts for an LLM.",
    verbose=True
)

llm_agent = Agent(
    name="LLMAgent",
    task=generate_response,
    role="llm",
    goal="Generate an empathetic therapist reply via Claude.",
    backstory="You are a compassionate AI therapist powered by Claude.",
    verbose=False
)

response_agent = Agent(
    name="ResponseAgent",
    task=postprocess_and_dispatch,
    role="output",
    goal="Deliver the therapist’s response to the user.",
    backstory="You ensure the user receives a supportive, well-formatted reply.",
    verbose=True
)

# 3. Run loop using direct agent.run() calls

def main_loop():
    # Start live audio capture in background
    threading.Thread(
        target=sd.InputStream,
        kwargs={
            'samplerate': AUDIO_SAMPLE_RATE,
            'channels': 1,
            'callback': audio_callback,
            'blocksize': AUDIO_CHUNK_SAMPLES
        },
        daemon=True
    ).start()

    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print("❌ Could not open camera")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 1) Emotion detection
            e_out = emotion_agent.run(frame)

            # 2) Speech transcription (non-blocking)
            s_out = {}
            if not audio_queue.empty():
                audio_chunks = []
                while not audio_queue.empty():
                    audio_chunks.append(audio_queue.get())
                audio_data = np.concatenate(audio_chunks, axis=0)
                try:
                    s_out = speech_agent.run(audio_data)
                except Exception as e:
                    print(f"⚠️ SpeechAgent error: {e}")

            # 3) Prompt composition
            context = {**e_out, **s_out}
            p_out = prompt_agent.run(context)

            # 4) LLM response
            l_out = llm_agent.run(p_out)

            # 5) Final dispatch
            final_packet = {**context, **l_out}
            response_agent.run(final_packet)

            time.sleep(0.1)
    finally:
        cap.release()

if __name__ == "__main__":
    main_loop()
