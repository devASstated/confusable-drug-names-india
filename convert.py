from pathlib import Path
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------- CONFIGURATION ----------------

BASE_DIR = Path(__file__).resolve().parent

# Path to your new dataset CSV file (update this filename to your target dataset)
NEW_DATASET_PATH = BASE_DIR / "phones_compare.csv" 

# Path pointing precisely to your stored model from the training script
MODEL_DIR = BASE_DIR / "byt5_phoneme_advanced" / "best_byt5_advanced_model"
OUTPUT_CSV_PATH = BASE_DIR / "normalized_inference_output.csv"

# Column names in your new dataset (adjust if your column headers differ)
CONTEXT_COLUMN = "brand_root"              # Word context column
INPUT_COLUMN = "phones_g2p_arpa"   # Canonical Arpabet phonemes column

MAX_INPUT_LENGTH = 96
MAX_TARGET_LENGTH = 64
BEAM_WIDTH = 4
BATCH_SIZE = 32

# ---------------- LOAD MODEL & TOKENIZER ----------------

print(f"Loading stored model and tokenizer from: {MODEL_DIR}")
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model = AutoModelForSeq2SeqLM.from_pretrained(str(MODEL_DIR))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print(f"Model loaded successfully on device: {device}")


# ---------------- LOAD & PREPARE DATASET ----------------

print(f"Loading dataset from: {NEW_DATASET_PATH}")
df = pd.read_csv(NEW_DATASET_PATH, encoding="utf-8-sig")
df.columns = df.columns.str.strip()

if CONTEXT_COLUMN not in df.columns:
    raise ValueError(f"Context column '{CONTEXT_COLUMN}' not found in columns: {list(df.columns)}")
if INPUT_COLUMN not in df.columns:
    raise ValueError(f"Input column '{INPUT_COLUMN}' not found in columns: {list(df.columns)}")

df["word_context"] = df[CONTEXT_COLUMN].fillna("").astype(str).str.strip().str.lower()
df["source_raw"] = df[INPUT_COLUMN].fillna("").astype(str).str.strip()

# Format prompts identically to your training pipeline
df["input_text"] = df.apply(
    lambda row: f"word: {row['word_context']} | canonical: {row['source_raw']}",
    axis=1
)


# ---------------- BATCHED INFERENCE ----------------

print(f"Running inference on {len(df)} rows using batch size {BATCH_SIZE}...")
all_predictions = []

for i in range(0, len(df), BATCH_SIZE):
    batch_texts = df["input_text"].iloc[i : i + BATCH_SIZE].tolist()
    
    inputs = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        generated_tokens = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=MAX_TARGET_LENGTH,
            num_beams=BEAM_WIDTH
        )
    
    decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    all_predictions.extend([pred.strip() for pred in decoded_preds])
    
    print(f"Processed {min(i + BATCH_SIZE, len(df))}/{len(df)} rows...", end="\r")

print("\nInference complete!")


# ---------------- EXPORT RESULTS & SUMMARY ----------------

df["predicted_normalized_phonemes"] = all_predictions
df["n_g2p"] = df["predicted_normalized_phonemes"].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)

# Save output predictions to CSV
df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
print(f"Normalized predictions successfully saved to: {OUTPUT_CSV_PATH}")

# Compute unique phoneme token counts
unique_input_phonemes = set(token for seq in df["source_raw"].dropna() for token in seq.split())
unique_output_phonemes = set(token for seq in df["predicted_normalized_phonemes"].dropna() for token in seq.split())

print(f"\n========== PHONEME VOCABULARY SUMMARY ==========")
print(f"Total rows processed: {len(df)}")
print(f"Number of distinct unique input (canonical) phonemes: {len(unique_input_phonemes)}")
print(f"Input phonemes found: {sorted(list(unique_input_phonemes))}")
print(f"Number of distinct unique output (predicted normalized) phonemes: {len(unique_output_phonemes)}")
print(f"Output phonemes found: {sorted(list(unique_output_phonemes))}")