from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

print("Loading your fine-tuned model...")
model     = AutoModelForSeq2SeqLM.from_pretrained("./bart-simplification-final")
tokenizer = AutoTokenizer.from_pretrained("./bart-simplification-final")
model.eval()
print("Model loaded.\n")

test_texts = [
    "The mitochondria is a membrane-bound organelle found in the cytoplasm of eukaryotic cells that generates most of the cell's supply of adenosine triphosphate.",
    "Quantum entanglement is a physical phenomenon that occurs when a group of particles interact in a way such that the quantum state of each particle cannot be described independently.",
    "The legislative branch of government is responsible for enacting laws and confirming or rejecting presidential nominations.",
    "Photosynthesis is a complex biochemical reaction that occurs in the chloroplasts of plant cells and is essential for life on Earth."
]

print("=" * 60)
for i, text in enumerate(test_texts, 1):
    inputs = tokenizer(
        "simplify: " + text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )
    with torch.no_grad():
        output = model.generate(
            inputs["input_ids"],
            max_length=128,
            num_beams=4,
            early_stopping=True
        )
    simplified = tokenizer.decode(output[0], skip_special_tokens=True)

    print(f"[{i}] Original  : {text[:75]}...")
    print(f"[{i}] Simplified: {simplified}")
    print()