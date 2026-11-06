from emotion_detector import detect_emotion
from transition_detector import analyze_transition


def analyze_message(text, history):

    result = detect_emotion(text,history)

    current_emotion = result["emotion"]

    previous_emotion = None

    if history:
        previous_emotion = history[-1]["emotion"]

    transition = analyze_transition(
        previous_emotion,
        current_emotion
    )

    return {
        "text": text,
        "emotion": current_emotion,
        "confidence": result["confidence"],
        "transition": transition
    }