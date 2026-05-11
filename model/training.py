import re
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

# 1.PREPROCESSING 

_NOISE_PUNCT_RE = re.compile(r'[\"#$%&\*+/<=>@\[\\\]^_`{|}~]')
_WHITESPACE_RE  = re.compile(r'\s{2,}')
_URL_RE         = re.compile(r'http\S+')
_CITATION_RE    = re.compile(r'\[\d+\]')      
_WIKI_TMPL_RE   = re.compile(r'\{\{.*?\}\}')   

MIN_WORD_COUNT  = 5    
MAX_WORD_COUNT  = 100  
MAX_SIMP_RATIO  = 1.5  


def clean_english(text: str) -> str:

    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = _URL_RE.sub('', text)
    text = _CITATION_RE.sub('', text)
    text = _WIKI_TMPL_RE.sub('', text)
    text = _NOISE_PUNCT_RE.sub('', text)
    text = re.sub(r'\s([,.!?;:])', r'\1', text)  
    text = _WHITESPACE_RE.sub(' ', text)
    return text.strip()


def is_valid_sample(original: str, simplified: str) -> bool:
    if not original or not simplified:
        return False
    orig_len = len(original.split())
    simp_len = len(simplified.split())
    if orig_len < MIN_WORD_COUNT or simp_len < MIN_WORD_COUNT: return False
    if orig_len > MAX_WORD_COUNT or simp_len > MAX_WORD_COUNT: return False
    if original == simplified:
        return False
    #  simplified should not be longer than the original
    if (simp_len / orig_len) > MAX_SIMP_RATIO:
        return False
    return True


# 2.DATASET-SPECIFIC PREPROCESSING (ASSET format)

def expand_dataset(batch):
    originals = []
    simplifieds = []

    for original, refs in zip(batch["original"], batch["simplifications"]):
        original_clean = clean_english(original)

        for ref in refs:
            simplified_clean = clean_english(ref)

            if is_valid_sample(original_clean, simplified_clean):
                originals.append(original_clean)
                simplifieds.append(simplified_clean)

    return {
        "original_clean": originals,
        "simplified_clean": simplifieds
    }

# 3.TOKENIZATION

MODEL_CHECKPOINT  = "facebook/bart-base"
MAX_INPUT_LENGTH  = 128   
MAX_TARGET_LENGTH = 64    


def build_tokenize_fn(tokenizer):
    def tokenize(examples):
        inputs  = ["simplify: " + s for s in examples["original_clean"]]
        targets = examples["simplified_clean"]

        model_inputs = tokenizer(
            inputs,
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            padding=False,   
        )

        labels = tokenizer(
            text_target=targets,
            max_length=MAX_TARGET_LENGTH,
            truncation=True,
            padding=False,
        )

        model_inputs["labels"] = [
            [(t if t != tokenizer.pad_token_id else -100) for t in ids]
            for ids in labels["input_ids"]
        ]
        return model_inputs
    return tokenize

# 4. MAIN PIPELINE
def main():
    print("=" * 60)
    print("BART Text Simplification — ASSET Dataset")
    print("=" * 60)

    # ── 1. Load dataset ──────────────────────────────────────────────────────
    print("\n[1/6] Loading ASSET dataset...")
    dataset = load_dataset("asset", "simplification")

    split = dataset["validation"].train_test_split(test_size=0.2, seed=42)
    dataset["train"]      = split["train"]
    dataset["validation"] = split["test"]

    print(f"  Raw train : {len(dataset['train'])} samples")
    print(f"  Raw val   : {len(dataset['validation'])} samples")

    # ── 2. Text-level preprocessing ──────────────────────────────────────────
    print("\n[2/6] Applying English preprocessing...")
    dataset = dataset.map(expand_dataset,batched=True,remove_columns=dataset["train"].column_names)

    for split_name in ("train", "validation"):
        seen = set()
        def dedup(example, seen=seen):
            key = (example["original_clean"], example["simplified_clean"])
            if key in seen: return False
            seen.add(key);  return True
        dataset[split_name] = dataset[split_name].filter(dedup)

   
    print(f"  Clean train : {len(dataset['train'])} samples")
    print(f"  Clean val   : {len(dataset['validation'])} samples")
    print(f"\n  Sample original  : {dataset['train'][0]['original_clean']}")
    print(f"  Sample simplified: {dataset['train'][0]['simplified_clean']}")

    # ── 3. Tokenizer ─────────────────────────────────────────────────────────
    print("\n[3/6] Loading tokenizer...")
    tokenizer   = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    tokenize_fn = build_tokenize_fn(tokenizer)

    remove_cols = [
        c for c in dataset["train"].column_names
        if c not in ("input_ids", "attention_mask", "labels")
    ]

    print("[4/6] Tokenizing dataset...")
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=remove_cols, desc="Tokenizing")
    print("  Tokenization done.")

    # ── 4. Model ─────────────────────────────────────────────────────────────
    print("\n[5/6] Loading BART-base model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)

    # ── 5. Training arguments ────────────────────────────────────────────────
    args = Seq2SeqTrainingArguments(
        output_dir                  = "./bart-simplification",
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        learning_rate               = 3e-5,
        per_device_train_batch_size = 4,
        per_device_eval_batch_size  = 4,
        num_train_epochs            = 3,
        weight_decay                = 0.01,
        predict_with_generate       = True,
        generation_max_length       = MAX_TARGET_LENGTH,
        fp16                        = False,
        logging_steps               = 25,
        load_best_model_at_end      = True,
        save_total_limit            = 1,
        report_to                   = "none",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=None,
    )

    trainer = Seq2SeqTrainer(
        model            = model,
        args             = args,
        train_dataset    = tokenized["train"],
        eval_dataset     = tokenized["validation"],
        processing_class = tokenizer,
        data_collator    = data_collator,
    )

    # ── 6. Train ─────────────────────────────────────────────────────────────
    print("\n[6/6] Starting training... ")
    trainer.train()

    print("\nSaving model to ./bart-simplification-final ...")
    trainer.save_model("./bart-simplification-final")
    tokenizer.save_pretrained("./bart-simplification-final")
    print("Done.")


if __name__ == "__main__":
    main()