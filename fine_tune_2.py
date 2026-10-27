from pathlib import Path
import random
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)

# ---------------- CONFIGURATION ----------------

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "aggregated_phoneme_results.csv"
if not CSV_PATH.exists():
    CSV_PATH = BASE_DIR / "test_evaluation_reloaded.csv"

OUTPUT_DIR = BASE_DIR / "byt5_phoneme_advanced"

INPUT_COLUMN = "canonical_phonemes"
TARGET_COLUMN = "phonemes_normalized"
CONTEXT_COLUMN = "word"

MODEL_NAME = "google/byt5-small"
MAX_INPUT_LENGTH = 96
MAX_TARGET_LENGTH = 64

BATCH_SIZE = 16
LEARNING_RATE = 3e-4
EPOCHS = 18
RANDOM_SEED = 42
BEAM_WIDTH = 4

# Multiplier to boost the weight of mismatch sequences in training
MISMATCH_OVERSAMPLE_FACTOR = 6

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------- DATA AUGMENTATION & PREPROCESSING ----------------

def augment_phonemes(phoneme_str, p=0.05):
    """Applies minor random phonological perturbation (subst/insert/delete) for data augmentation."""
    tokens = phoneme_str.split()
    if len(tokens) <= 2:
        return phoneme_str
    
    augmented = []
    for token in tokens:
        if random.random() < p:
            op = random.choice(['drop', 'dup'])
            if op == 'drop':
                continue
            elif op == 'dup':
                augmented.extend([token, token])
                continue
        augmented.append(token)
    return " ".join(augmented) if augmented else phoneme_str


print("Loading dataset...")
df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
df.columns = df.columns.str.strip()

if TARGET_COLUMN not in df.columns and "actual_phonemes_normalized" in df.columns:
    TARGET_COLUMN = "actual_phonemes_normalized"
if INPUT_COLUMN not in df.columns and "canonical_phonemes" in df.columns:
    INPUT_COLUMN = "canonical_phonemes"
if CONTEXT_COLUMN not in df.columns and "for_word" in df.columns:
    CONTEXT_COLUMN = "for_word"

df = df[[INPUT_COLUMN, TARGET_COLUMN, CONTEXT_COLUMN]].copy()

df["word_context"] = df[CONTEXT_COLUMN].fillna("").astype(str).str.strip().str.lower()
df["source_raw"] = df[INPUT_COLUMN].fillna("").astype(str).str.strip()
df["target_raw"] = df[TARGET_COLUMN].fillna("").astype(str).str.strip()

df = df[
    (df["word_context"] != "") & (df["word_context"] != "nan") &
    (df["source_raw"] != "") & (df["source_raw"] != "nan") &
    (df["target_raw"] != "") & (df["target_raw"] != "nan")
].copy()

df = df.drop_duplicates(subset=["word_context"]).reset_index(drop=True)

print(f"Usable unique word sequence pairs: {len(df)}")


# ---------------- FORMAT PROMPTS ----------------

df["input_text"] = df.apply(
    lambda row: f"word: {row['word_context']} | canonical: {row['source_raw']}",
    axis=1
)
df["target_text"] = df["target_raw"]


# ---------------- DATA SPLIT & OVERSAMPLING ----------------

train_df, temp_df = train_test_split(df, test_size=0.20, random_state=RANDOM_SEED, shuffle=True)
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=RANDOM_SEED, shuffle=True)

# Curriculum Learning: Sort training data by target sequence length (short to long) after split
train_df["seq_len"] = train_df["target_text"].apply(lambda x: len(x.split()))
train_df = train_df.sort_values("seq_len").drop(columns=["seq_len"]).reset_index(drop=True)

# Apply data augmentation ONLY to the training split
random.seed(RANDOM_SEED)
train_df["input_text"] = train_df.apply(
    lambda row: f"word: {row['word_context']} | canonical: {augment_phonemes(row['source_raw'])}",
    axis=1
)

# --- NEW: MISMATCH OVERSAMPLING TO BALANCE LEARNING ---
train_df["is_mismatch"] = train_df["source_raw"] != train_df["target_raw"]
matches_train = train_df[~train_df["is_mismatch"]]
mismatches_train = train_df[train_df["is_mismatch"]]

print(f"Training split before oversampling -> Matches: {len(matches_train)}, Mismatches: {len(mismatches_train)}")

if len(mismatches_train) > 0:
    mismatches_augmented = pd.concat([mismatches_train] * MISMATCH_OVERSAMPLE_FACTOR, ignore_index=True)
    train_df = pd.concat([matches_train, mismatches_augmented], ignore_index=True)
    train_df = train_df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

print(f"Training split after oversampling: {len(train_df)} samples")

def create_hf_dataset(subset_df):
    cols = ["input_text", "target_text"]
    return Dataset.from_dict({
        "input_text": subset_df["input_text"].tolist(),
        "target_text": subset_df["target_text"].tolist()
    })

train_dataset = create_hf_dataset(train_df)
val_dataset = create_hf_dataset(val_df)
test_dataset = create_hf_dataset(test_df)

print(f"Train split: {len(train_df)} | Val split: {len(val_df)} | Test split: {len(test_df)}")


# ---------------- TOKENIZATION ----------------

print(f"Loading tokenizer for {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["input_text"],
        max_length=MAX_INPUT_LENGTH,
        truncation=True
    )
    labels = tokenizer(
        text_target=examples["target_text"],
        max_length=MAX_TARGET_LENGTH,
        truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("Tokenizing datasets...")
tokenized_train = train_dataset.map(preprocess_function, batched=True, remove_columns=["input_text", "target_text"])
tokenized_val = val_dataset.map(preprocess_function, batched=True, remove_columns=["input_text", "target_text"])
tokenized_test = test_dataset.map(preprocess_function, batched=True, remove_columns=["input_text", "target_text"])


# ---------------- LOAD MODEL & TRAINER ----------------

print(f"Loading pre-trained model: {MODEL_NAME}...")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

training_args = Seq2SeqTrainingArguments(
    output_dir=str(OUTPUT_DIR),
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=EPOCHS,
    predict_with_generate=True,
    generation_max_length=MAX_TARGET_LENGTH,
    generation_num_beams=BEAM_WIDTH,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    seed=RANDOM_SEED,
    logging_steps=20,
    lr_scheduler_type="cosine",
    warmup_steps=250
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
)


# ---------------- TRAIN MODEL ----------------

print("\n========== STARTING ADVANCED BYT5 TRAINING ==========")
trainer.train()

best_model_path = OUTPUT_DIR / "best_byt5_advanced_model"
trainer.save_model(str(best_model_path))
tokenizer.save_pretrained(str(best_model_path))
print(f"Model saved to: {best_model_path}")


# ---------------- EVALUATION & CSV EXPORT ----------------

print("\nRunning evaluation on test set...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_model = AutoModelForSeq2SeqLM.from_pretrained(best_model_path).to(device)
best_model.eval()

def predict_phonemes(input_text_str):
    inputs = tokenizer(
        input_text_str,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH
    ).to(device)
    
    with torch.no_grad():
        outputs = best_model.generate(
            **inputs,
            max_length=MAX_TARGET_LENGTH,
            num_beams=BEAM_WIDTH
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, ref_token in enumerate(reference, start=1):
        current = [i]
        for j, hyp_token in enumerate(hypothesis, start=1):
            cost = 0 if ref_token == hyp_token else 1
            current.append(
                min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + cost
                )
            )
        previous = current
    return previous[-1]


def normalize_sequence(sequence):
    if isinstance(sequence, str):
        return sequence.strip().split()
    return list(sequence)


results = []

total_errors = 0
total_reference_phonemes = 0
exact_matches = 0

actual_matches = 0
correct_matches = 0
actual_mismatches = 0
correct_mismatches = 0

for _, row in test_df.iterrows():
    word_ctx = row["word_context"]
    source_raw = row["source_raw"]
    target_raw = row["target_raw"]
    
    input_text = f"word: {word_ctx} | canonical: {source_raw}"
    
    canonical = normalize_sequence(source_raw)
    actual = normalize_sequence(target_raw)
    
    pred_str = predict_phonemes(input_text)
    predicted = normalize_sequence(pred_str)

    errors = edit_distance(actual, predicted)
    is_exact = predicted == actual

    reference_is_match = actual == canonical
    prediction_is_match = predicted == canonical

    total_errors += errors
    total_reference_phonemes += len(actual)

    if is_exact:
        exact_matches += 1

    if reference_is_match:
        actual_matches += 1
        if is_exact:
            correct_matches += 1
    else:
        actual_mismatches += 1
        if is_exact:
            correct_mismatches += 1

    results.append({
        "for_word": word_ctx,
        "canonical_phonemes": " ".join(canonical),
        "actual_phonemes_normalized": " ".join(actual),
        "predicted_phonemes_normalized": " ".join(predicted),
        "phoneme_edit_distance": errors,
        "exact_match": is_exact,
        "actual_is_match": reference_is_match,
        "predicted_is_match": prediction_is_match
    })

results_df = pd.DataFrame(results)

output_dir = Path("phoneme_evaluation")
output_dir.mkdir(parents=True, exist_ok=True)
results_csv_path = output_dir / "test_predictions_byt5.csv"

results_df.to_csv(
    results_csv_path,
    index=False,
    encoding="utf-8-sig"
)

# Overall evaluation metrics
per = (
    total_errors / total_reference_phonemes
    if total_reference_phonemes > 0
    else 0.0
)

exact_accuracy = (
    exact_matches / len(results_df)
    if len(results_df) > 0
    else 0.0
)

match_recall = (
    correct_matches / actual_matches
    if actual_matches > 0
    else 0.0
)

mismatch_recall = (
    correct_mismatches / actual_mismatches
    if actual_mismatches > 0
    else 0.0
)

tp = sum(
    (not row["actual_is_match"]) and
    (not row["predicted_is_match"])
    for row in results
)

fp = sum(
    row["actual_is_match"] and
    (not row["predicted_is_match"])
    for row in results
)

fn = sum(
    (not row["actual_is_match"]) and
    row["predicted_is_match"]
    for row in results
)

tn = sum(
    row["actual_is_match"] and
    row["predicted_is_match"]
    for row in results
)

precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
detection_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

f1 = (
    (2 * precision * detection_recall) /
    (precision + detection_recall)
    if (precision + detection_recall) > 0
    else 0.0
)

# Print results
print("\n========== BYT5 TEST RESULTS ==========")
print(f"Test samples: {len(results_df)}")
print(f"Exact-match accuracy: {exact_accuracy:.2%}")
print(f"Phoneme error rate (PER): {per:.2%}")

print("\n--- Exact Reconstruction ---")
print(f"Actual mismatches: {actual_mismatches}")
print(f"Mismatches reconstructed correctly: {correct_mismatches}")
print(f"Mismatch reconstruction recall: {mismatch_recall:.2%}")

print("\n--- Mismatch Detection ---")
print(f"Mismatch detection precision: {precision:.2%}")
print(f"Mismatch detection recall: {detection_recall:.2%}")
print(f"Mismatch detection F1-score: {f1:.2%}")

print(f"\nPredictions saved to: {results_csv_path}")