# scripts/run_webcam.py

import cv2
from src.emotion_detector import detect_emotion

def main():
    # match your working test script:
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

    if not cap.isOpened():
        print("Could not open camera")
        return

    emotion = ""           # initialize before loop
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        print("Frame OK:", ret)   # debug: should print True continuously
        if not ret:
            break

        frame_idx += 1

        # Only analyze every 3rd frame to keep things snappy
        if frame_idx % 3 == 0:
            # safely catch any analysis errors
            try:
                emotion = detect_emotion(frame)
            except Exception as e:
                print("Emotion detect error:", e)

        # overlay the last-known emotion
        cv2.putText(frame, emotion, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("Emotion Detector", frame)

        # quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
