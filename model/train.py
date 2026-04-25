from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    Trainer
)

# ── Load dataset ──────────────────────────────────────────────────────────────
dataset = load_dataset("bogdancazan/wikilarge-text-simplification")

# ── Clean dataset ─────────────────────────────────────────────────────────────
def is_valid(example):
    normal = example["Normal"].strip()
    simple = example["Simple"].strip()

    if len(normal) < 10 or len(simple) < 10:
        return False
    if normal.lower() == simple.lower():
        return False
    ratio = len(simple.split()) / len(normal.split())
    if ratio < 0.2:
        return False
    if len(normal.split()) > 200 or len(simple.split()) > 200:
        return False

    return True

dataset = dataset.filter(is_valid)

# ── Use small subset so CPU can finish in reasonable time ─────────────────────
train_size = min(15000, len(dataset["train"]))
val_size = min(500, len(dataset["validation"]))

dataset["train"] = dataset["train"].shuffle(seed=42).select(range(train_size))
dataset["validation"] = dataset["validation"].shuffle(seed=42).select(range(val_size))

print("Dataset ready:", dataset)

# ── Load model — t5-small for local CPU training ──────────────────────────────
model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

# ── Tokenize ──────────────────────────────────────────────────────────────────
def tokenize_text(examples):
    inputs = ["simplify: " + text for text in examples["Normal"]]

    model_inputs = tokenizer(
        inputs,
        truncation=True,
        max_length=128,
    )

    labels = tokenizer(
        examples["Simple"],
        truncation=True,
        max_length=128,
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

dataset = dataset.map(tokenize_text, batched=True, remove_columns=["Normal", "Simple"])
print("Tokenized:", dataset)

# ── Training arguments — tuned for CPU ───────────────────────────────────────
training_args = TrainingArguments(
    output_dir="./simplifier-results",
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=False,
    logging_steps=50,
    report_to="none",
    optim="adafactor",
    use_cpu=True
)

# ── Trainer ───────────────────────────────────────────────────────────────────
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    padding=True,
    label_pad_token_id=-100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator,
    
)

# ── Train ─────────────────────────────────────────────────────────────────────
print("Starting training...")
trainer.train()

# ── Save ──────────────────────────────────────────────────────────────────────
save_path = "./final_simplifier_model"
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)
print(f"Saved to {save_path}")

# ── Quick test ────────────────────────────────────────────────────────────────
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

test_tokenizer = AutoTokenizer.from_pretrained(save_path)
test_model     = AutoModelForSeq2SeqLM.from_pretrained(save_path)

test_text = "simplify: Machine learning enables systems to learn from data without being explicitly programmed."
inputs  = test_tokenizer(test_text, return_tensors="pt", truncation=True, max_length=128)
outputs = test_model.generate(
    inputs["input_ids"],
    max_length=80,
    num_beams=6,
    length_penalty=0.8,
    early_stopping=True
)
result  = test_tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Test output:", result)