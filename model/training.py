from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
import torch

print("=" * 50)
print("Device: CPU (no GPU detected)")
print("=" * 50)

# ── 1. Load dataset ──────────────────────────────────
print("\n[1/5] Loading dataset...")
dataset = load_dataset("asset", "simplification")

# ASSET only has validation + test — split validation into train/val ourselves
split = dataset["validation"].train_test_split(test_size=0.2, seed=42)
dataset["train"]      = split["train"]   # ~1600 examples
dataset["validation"] = split["test"]    #  ~400 examples

print(f"Train samples   : {len(dataset['train'])}")
print(f"Val samples     : {len(dataset['validation'])}")

# Reduce for CPU speed
dataset["train"]      = dataset["train"].select(range(1500))
dataset["validation"] = dataset["validation"].select(range(300))

print(f"Using for training  : {len(dataset['train'])}")
print(f"Using for validation: {len(dataset['validation'])}")
print(f"Sample original : {dataset['train'][0]['original']}")
print(f"Sample simple   : {dataset['train'][0]['simplifications'][0]}")

# ── 2. Tokenizer ─────────────────────────────────────
print("\n[2/5] Loading tokenizer...")
MODEL_CHECKPOINT = "facebook/bart-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

def preprocess(examples):
    inputs  = ["simplify: " + s for s in examples["original"]]
    targets = [s[0] for s in examples["simplifications"]]

    model_inputs = tokenizer(
        inputs,
        max_length=512,
        truncation=True,
        padding="max_length"
    )
    labels = tokenizer(
        targets,
        max_length=128,
        truncation=True,
        padding="max_length"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("[3/5] Tokenizing dataset...")
tokenized = dataset.map(preprocess, batched=True)
print("Tokenization done.")

# ── 3. Model ─────────────────────────────────────────
print("\n[4/5] Loading BART-base model...")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)

# ── 4. Training arguments ────────────────────────────
args = Seq2SeqTrainingArguments(
    output_dir             = "./bart-simplification",
    eval_strategy          = "epoch",
    save_strategy          = "epoch",
    learning_rate          = 5e-5,
    per_device_train_batch_size = 2,
    per_device_eval_batch_size  = 2,
    num_train_epochs       = 5,
    weight_decay           = 0.01,
    predict_with_generate  = True,
    fp16                   = False,
    logging_steps          = 25,
    load_best_model_at_end = True,
    save_total_limit       = 1,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model             = model,
    args              = args,
    train_dataset     = tokenized["train"],
    eval_dataset      = tokenized["validation"],
    processing_class  = tokenizer,
    data_collator     = data_collator,
)

# ── 5. Train ─────────────────────────────────────────
print("\n[5/5] Starting training... (this will take ~45 mins on CPU)")
print("You'll see loss values every 25 steps — they should go down over time.\n")
trainer.train()

print("\nSaving model to ./bart-simplification-final ...")
trainer.save_model("./bart-simplification-final")
tokenizer.save_pretrained("./bart-simplification-final")
print("Model saved successfully.")