# scripts/run_assistant.py

import cv2
from concurrent.futures import ThreadPoolExecutor
import threading

# Optional: face-cascade for ROI cropping
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

from src.emotion_detector import detect_emotion
from src.speech_recognizer import listen_and_transcribe
from src.assistant import get_bot_response

# Shared state for emotion
emotion = ""
emotion_lock = threading.Lock()

# Thread pool for background tasks
executor = ThreadPoolExecutor(max_workers=2)


def emotion_worker(frame_roi):
    """
    Background task: detects emotion on the provided ROI and updates shared state.
    """
    global emotion
    try:
        label = detect_emotion(frame_roi)
        with emotion_lock:
            emotion = label
    except Exception as e:
        print("Emotion worker error:", e)


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print("Could not open camera")
        return

    print("🤖 Assistant is running. Press Ctrl+C to exit.")

    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Lost camera feed")
                break

            frame_idx += 1

            # Downscale for faster detection
            small = cv2.resize(frame, (640, 480))

            # Submit emotion detection every 10 frames
            if frame_idx % 10 == 0:
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    roi = small[y : y + h, x : x + w]
                else:
                    roi = small

                # Run detection in background
                executor.submit(emotion_worker, roi)

            # Read the latest detected emotion safely
            with emotion_lock:
                display_emotion = emotion

            # Non-blocking UI update
            cv2.putText(
                frame,
                display_emotion,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            # Show frame
            cv2.imshow("Assistant View", frame)

            # Trigger speech-to-text and agent only when user presses 'r' (example)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                # Record & transcribe speech
                transcript = listen_and_transcribe(duration=3)
                response = get_bot_response(transcript, display_emotion)
                print(f"\n[User ({display_emotion})]: {transcript}")
                print(f"[Bot]: {response}\n")
            elif key == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n Exiting…")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
