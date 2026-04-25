from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

save_path = "./final_simplifier_model"

tokenizer = AutoTokenizer.from_pretrained(save_path)
model     = AutoModelForSeq2SeqLM.from_pretrained(save_path)

test_texts = [
    "simplify: The mitochondria are membrane-bound organelles found in the cytoplasm of eukaryotic cells that generate most of the cell's supply of adenosine triphosphate, used as a source of chemical energy.",

    "simplify: Photosynthesis is a complex biochemical process whereby chlorophyll-containing organisms utilize radiant solar energy to synthesize organic compounds from atmospheric carbon dioxide and water molecules.",

    "simplify: Globalization refers to the increasing interconnectedness of economies, cultures, and populations brought about by cross-border trade, technology, and the flow of information.",

    "simplify: The implementation of artificial neural networks involves the utilization of computational architectures inspired by biological neural systems to process and classify complex datasets.",
]

for text in test_texts:
    inputs  = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    outputs = model.generate(
        inputs["input_ids"],
        max_length=80,
        num_beams=6,
        length_penalty=0.8,
        early_stopping=True
    )
    result  = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"Input : {text}")
    print(f"Output: {result}")
    print()