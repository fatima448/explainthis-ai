from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

save_path = "./bart-simplification-final"
tokenizer = AutoTokenizer.from_pretrained(save_path)
model     = AutoModelForSeq2SeqLM.from_pretrained(save_path)

def simplify(text):
    prefixed = "simplify: " + text

    inputs = tokenizer(
        prefixed,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )
    outputs = model.generate(
       inputs["input_ids"],
       attention_mask=inputs["attention_mask"],
       max_length=64,
       num_beams=4,
       repetition_penalty=1.2,
       length_penalty=1.0,
       early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

test_texts = [
    "The mitochondria are membrane-bound organelles found in the cytoplasm of eukaryotic cells that generate most of the cell's supply of adenosine triphosphate, used as a source of chemical energy.",
    "Photosynthesis is a complex biochemical process whereby chlorophyll-containing organisms utilize radiant solar energy to synthesize organic compounds from atmospheric carbon dioxide and water molecules.",
    "Globalization refers to the increasing interconnectedness of economies, cultures, and populations brought about by cross-border trade, technology, and the flow of information.",
    "The implementation of artificial neural networks involves the utilization of computational architectures inspired by biological neural systems to process and classify complex datasets.",
]

for text in test_texts:
    print(f"Input : {text}")
    print(f"Output: {simplify(text)}")
    print()