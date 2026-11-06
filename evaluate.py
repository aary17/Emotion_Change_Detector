import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from transformers import pipeline


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "./models/meld-emotion-model"
TEST_FILE = "data/MELD/test_sent_emo.csv"

EMOTIONS = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "neutral",
    "sadness",
    "surprise"
]


# ============================================================
# LOAD TEST DATA
# ============================================================

print("Loading MELD test dataset...")

df = pd.read_csv(TEST_FILE)

print(f"Total test samples: {len(df)}")


# ============================================================
# BUILD CONVERSATION CONTEXT
# ============================================================

def build_context(df, context_size=2):

    df = df.sort_values(
        ["Dialogue_ID", "Utterance_ID"]
    ).reset_index(drop=True)

    texts = []

    dialogue_history = {}

    for _, row in df.iterrows():

        dialogue_id = row["Dialogue_ID"]
        current_text = str(row["Utterance"])

        if dialogue_id not in dialogue_history:
            dialogue_history[dialogue_id] = []

        previous = dialogue_history[dialogue_id][-context_size:]

        if previous:

            context = " ".join(
                [
                    f"Previous: {text}"
                    for text in previous
                ]
            )

            combined_text = (
                context +
                " Current: " +
                current_text
            )

        else:

            combined_text = (
                "Current: " +
                current_text
            )

        texts.append(combined_text)

        dialogue_history[dialogue_id].append(
            current_text
        )

    df["text"] = texts

    return df


print("\nBuilding conversation context...")

df = build_context(
    df,
    context_size=2
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading fine-tuned model...")

emotion_model = pipeline(
    "text-classification",
    model=MODEL_PATH,
    tokenizer=MODEL_PATH,
    top_k=None
)

print("Model loaded successfully.")


# ============================================================
# RUN PREDICTIONS
# ============================================================

y_true = []
y_pred = []

print("\nRunning predictions...\n")

for i, row in df.iterrows():

    text = row["text"]

    actual = (
        str(row["Emotion"])
        .strip()
        .lower()
    )

    results = emotion_model(text)[0]

    best = max(
        results,
        key=lambda x: x["score"]
    )

    predicted = (
        best["label"]
        .strip()
        .lower()
    )

    y_true.append(actual)
    y_pred.append(predicted)

    if (i + 1) % 100 == 0:

        print(
            f"Processed "
            f"{i + 1}/{len(df)}"
        )


# ============================================================
# OVERALL METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision, recall, f1, _ = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=EMOTIONS,
        average="macro",
        zero_division=0
    )
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)

print("CONTEXT-AWARE MELD EMOTION CLASSIFICATION RESULTS")

print("=" * 60)

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"Macro-F1 : {f1:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        labels=EMOTIONS,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=EMOTIONS
)

print("\nConfusion Matrix:")

print(cm)


# ============================================================
# PLOT CONFUSION MATRIX
# ============================================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=EMOTIONS
)

disp.plot(
    xticks_rotation=45,
    cmap="Blues"
)

plt.title(
    "MELD Context-Aware Emotion Classification"
)

plt.tight_layout()

plt.savefig(
    "emotion_confusion_matrix_context.png",
    dpi=300
)

plt.show()

print(
    "\nConfusion matrix saved as "
    "emotion_confusion_matrix_context.png"
)