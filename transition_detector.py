EMOTION_VALENCE = {
    "joy": 1.0,
    "surprise": 0.3,
    "neutral": 0.0,
    "sadness": -0.7,
    "fear": -0.6,
    "anger": -0.9,
    "disgust": -0.8
}


def analyze_transition(previous, current):

    if previous is None:
        return {
            "changed": False,
            "significant": False,
            "intensity": 0
        }

    if previous == current:
        return {
            "changed": False,
            "significant": False,
            "intensity": 0
        }

    previous_value = EMOTION_VALENCE.get(previous.lower() if isinstance(previous, str) else previous, 0)
    current_value = EMOTION_VALENCE.get(current.lower() if isinstance(current, str) else current, 0)

    intensity = abs(current_value - previous_value)

    return {
        "changed": True,
        "significant": intensity >= 0.7,
        "intensity": round(intensity, 2),
        "from": previous,
        "to": current
    }