# src/emotion_detector.py
from deepface import DeepFace

def detect_emotion(frame, backend="ssd"):
    # Run analysis
    analysis = DeepFace.analyze(
        frame,
        actions=["emotion"],
        detector_backend=backend,
        enforce_detection=False
    )
    # If multiple faces are returned, grab the first result
    if isinstance(analysis, list):
        analysis = analysis[0]

    # Return the dominant emotion string
    return analysis.get("dominant_emotion", "")
