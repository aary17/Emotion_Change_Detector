from transformers import pipeline


MODEL_PATH = "./models/meld-emotion-model"


emotion_model = pipeline(
    "text-classification",
    model=MODEL_PATH,
    tokenizer=MODEL_PATH,
    top_k=None
)


def detect_emotion(text, history=None):

    if history is None:
        history = []

    previous = history[-2:]

    if previous:
        context = " ".join(
            [
                f"Previous: {item['text']}"
                for item in previous
            ]
        )

        model_input = (
            context +
            " Current: " +
            text
        )

    else:
        model_input = "Current: " + text


    results = emotion_model(model_input)[0]

    best = max(
        results,
        key=lambda x: x["score"]
    )


    return {
        "emotion": best["label"].capitalize(),
        "confidence": round(
            best["score"],
            4
        )
    }