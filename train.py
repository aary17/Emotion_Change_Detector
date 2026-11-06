import os

import pandas as pd
import numpy as np
import torch

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "distilroberta-base"

TRAIN_FILE = "data/MELD/train_sent_emo.csv"
VAL_FILE = "data/MELD/dev_sent_emo.csv"

OUTPUT_DIR = "./models/meld-emotion-model"

EMOTIONS = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "neutral",
    "sadness",
    "surprise"
]

label2id = {
    label: i
    for i, label in enumerate(EMOTIONS)
}

id2label = {
    i: label
    for i, label in enumerate(EMOTIONS)
}


# ============================================================
# LOAD DATA
# ============================================================

print("Loading MELD dataset...")

train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VAL_FILE)

print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")


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

train_df = build_context(
    train_df,
    context_size=2
)

val_df = build_context(
    val_df,
    context_size=2
)


# ============================================================
# PREPARE LABELS
# ============================================================

train_df["Emotion"] = (
    train_df["Emotion"]
    .str.lower()
    .str.strip()
)

val_df["Emotion"] = (
    val_df["Emotion"]
    .str.lower()
    .str.strip()
)

train_df["label"] = train_df["Emotion"].map(label2id)
val_df["label"] = val_df["Emotion"].map(label2id)


# Remove invalid labels

train_df = train_df.dropna(
    subset=["label"]
)

val_df = val_df.dropna(
    subset=["label"]
)

train_df["label"] = train_df["label"].astype(int)
val_df["label"] = val_df["label"].astype(int)


# ============================================================
# SHOW CLASS DISTRIBUTION
# ============================================================

print("\nTraining class distribution:")

print(
    train_df["Emotion"]
    .value_counts()
)


# ============================================================
# CONVERT TO HUGGINGFACE DATASET
# ============================================================

train_dataset = Dataset.from_pandas(
    train_df[["text", "label"]],
    preserve_index=False
)

val_dataset = Dataset.from_pandas(
    val_df[["text", "label"]],
    preserve_index=False
)


# ============================================================
# TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


def tokenize(batch):

    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=192
    )


print("Tokenizing dataset...")

train_dataset = train_dataset.map(
    tokenize,
    batched=True
)

val_dataset = val_dataset.map(
    tokenize,
    batched=True
)


# Remove original text column

train_dataset = train_dataset.remove_columns(
    ["text"]
)

val_dataset = val_dataset.remove_columns(
    ["text"]
)


# ============================================================
# MODEL
# ============================================================

print("\nLoading DistilRoBERTa...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(EMOTIONS),
    label2id=label2id,
    id2label=id2label
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

print("\nCalculating class weights...")

class_counts = (
    train_df["label"]
    .value_counts()
    .sort_index()
)

total_samples = len(train_df)
num_classes = len(EMOTIONS)

class_weights = (
    total_samples /
    (num_classes * class_counts)
)

class_weights = class_weights.reindex(
    range(num_classes)
)

class_weights = torch.tensor(
    class_weights.values,
    dtype=torch.float
)

print("\nClass weights:")

for i, emotion in enumerate(EMOTIONS):

    print(
        f"{emotion:10s}: "
        f"{class_weights[i]:.4f}"
    )


# ============================================================
# METRICS
# ============================================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=-1
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="macro",
            zero_division=0
        )
    )

    return {
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1
    }


# ============================================================
# CUSTOM TRAINER
# ============================================================

class WeightedTrainer(Trainer):

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None
    ):

        labels = inputs.pop("labels")

        outputs = model(**inputs)

        logits = outputs.logits

        loss_function = torch.nn.CrossEntropyLoss(
            weight=class_weights.to(
                logits.device
            )
        )

        loss = loss_function(
            logits,
            labels
        )

        return (
            (loss, outputs)
            if return_outputs
            else loss
        )


# ============================================================
# TRAINING SETTINGS
# ============================================================

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    num_train_epochs=4,

    learning_rate=2e-5,

    per_device_train_batch_size=16,

    per_device_eval_batch_size=16,

    weight_decay=0.01,

    eval_strategy="epoch",

    save_strategy="epoch",

    load_best_model_at_end=True,

    metric_for_best_model="macro_f1",

    greater_is_better=True,

    logging_steps=100,

    report_to="none",

    save_total_limit=1
)


# ============================================================
# TRAINER
# ============================================================

trainer = WeightedTrainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=val_dataset,

    processing_class=tokenizer,

    compute_metrics=compute_metrics
)


# ============================================================
# TRAIN
# ============================================================

print("\nStarting training...")

print("=" * 60)

trainer.train()


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\nValidation results:")

print("=" * 60)

results = trainer.evaluate()

for key, value in results.items():

    if isinstance(value, float):

        print(
            f"{key}: {value:.4f}"
        )


# ============================================================
# SAVE MODEL
# ============================================================

print("\nSaving final model...")

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

trainer.save_model(
    OUTPUT_DIR
)

tokenizer.save_pretrained(
    OUTPUT_DIR
)

print("\nTraining complete!")

print(
    f"Model saved to: {OUTPUT_DIR}"
)