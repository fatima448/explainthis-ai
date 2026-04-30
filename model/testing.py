import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading fine-tuned model...")
MODEL_PATH = "./bart-simplification-final"

model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model.eval()
print("Model loaded.\n")


def simplify(text: str) -> str:
    # lowercase the input to match training preprocessing
    text = text.lower().strip()

    inputs = tokenizer(
        "simplify: " + text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )

    with torch.no_grad():
        output = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=64,
            num_beams=4,
            no_repeat_ngram_size=3,
            length_penalty=0.8,
            early_stopping=True,
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)


# ── Test sentences ────────────────────────────────────────────────────────────
# Group 1: simple sentences — model should mostly leave these alone
simple_sentences = [
    "The cat is a small carnivorous mammal that has been domesticated by humans for thousands of years.",
    "The Amazon River is the largest river in the world by the volume of water it carries.",
]

# Group 2: complex sentences — model should actually simplify these
complex_sentences = [
    "Albert Einstein was a physicist who developed the theory of relativity, which is one of the two pillars of modern physics.",
    "The mitochondria is a membrane-bound organelle found in the cytoplasm of eukaryotic cells that generates most of the cell's supply of adenosine triphosphate.",
    "Photosynthesis is a process used by plants in which energy from sunlight is used to convert carbon dioxide and water into glucose and oxygen.",
    "The legislative branch of government is responsible for enacting laws and confirming or rejecting presidential nominations.",
]

print("=" * 65)
print("GROUP 1 — Already simple (expect little or no change)")
print("=" * 65)
for i, text in enumerate(simple_sentences, 1):
    result = simplify(text)
    changed = "✅ changed" if result.replace(" ", "") != text.lower().replace(" ", "") else "➡️  kept"
    print(f"[{i}] Original  : {text}")
    print(f"[{i}] Simplified: {result}  {changed}")
    print()

print("=" * 65)
print("GROUP 2 — Complex sentences (expect real simplification)")
print("=" * 65)
for i, text in enumerate(complex_sentences, 1):
    result = simplify(text)
    changed = "✅ changed" if result.replace(" ", "") != text.lower().replace(" ", "") else "➡️  kept"
    print(f"[{i}] Original  : {text}")
    print(f"[{i}] Simplified: {result}  {changed}")
    print()