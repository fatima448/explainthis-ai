import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading model...")
MODEL_PATH = "./bart-wikilarge-final"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
model.eval()
print("Model loaded.\n")


# ── Preprocessing — must match training ──────────────────────────────────────
_NOISE_PUNCT_RE = re.compile(r'[\"#$%&\*+/<=>@\[\\\]^_`{|}~]')
_WHITESPACE_RE  = re.compile(r'\s{2,}')
_URL_RE         = re.compile(r'http\S+')
_CITATION_RE    = re.compile(r'\[\d+\]')
_WIKI_TMPL_RE   = re.compile(r'\{\{.*?\}\}')

def clean(text: str) -> str:
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


# ── Sentence splitter ─────────────────────────────────────────────────────────
def split_sentences(text: str) -> list:
    # protect abbreviations before splitting
    text = re.sub(r'\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|approx|dept)\.',
                  r'\1<PERIOD>', text)
    # split on punctuation followed by space + capital letter
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$', text)
    sentences = [s.replace('<PERIOD>', '.').strip() for s in sentences]
    return [s for s in sentences if s]


# ── Inference — single sentence ───────────────────────────────────────────────
def simplify_sentence(sentence: str) -> str:
    cleaned = clean(sentence)
    if not cleaned or len(cleaned.split()) < 4:
        return cleaned

    inputs = tokenizer(
        "simplify: " + cleaned,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=64,
            num_beams=4,
            no_repeat_ngram_size=3,
            length_penalty=0.8,
            early_stopping=True,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# ── Inference — full paragraph ────────────────────────────────────────────────
def simplify_paragraph(text: str) -> str:
    """
    Split BEFORE cleaning so the splitter can see capital letters,
    then clean and simplify each sentence individually,
    then rejoin into a paragraph.
    """
    sentences = split_sentences(text)
    results = []
    for s in sentences:
        result = simplify_sentence(s)
        if result:
            results.append(result)
    return " ".join(results)


# ── Test paragraph ────────────────────────────────────────────────────────────
paragraph = """In an era saturated with technological convenience, productivity has paradoxically become both more attainable and more elusive. While digital tools promise efficiency, they often cultivate a subtle illusion of progress rather than genuine achievement. Notifications, task managers, and constant connectivity fragment attention, reducing the depth of cognitive engagement required for meaningful work. This phenomenon can be understood through the lens of "pseudo-productivity," where individuals equate activity with accomplishment. Answering emails, organizing files, or switching between multiple applications may create a sense of busyness, yet these actions frequently lack substantive impact. Consequently, individuals may end their day feeling exhausted but unfulfilled, having invested energy without producing tangible value. Moreover, the pervasive culture of immediacy discourages sustained focus. The human mind, inherently limited in its attentional capacity, struggles to adapt to the relentless influx of information. Over time, this leads to cognitive fatigue, diminished creativity, and a reduced ability to engage in complex problem-solving. To counteract this trend, it is essential to redefine productivity not as the quantity of tasks completed, but as the quality of outcomes achieved. This requires deliberate effort: minimizing distractions, prioritizing deep work, and cultivating an environment conducive to concentration. Only by reclaiming control over attention can individuals transcend the illusion of productivity and engage in work that is both meaningful and impactful."""

print("=" * 65)
print("ORIGINAL")
print("=" * 65)
print(paragraph)

print("\n" + "=" * 65)
print("SIMPLIFIED — sentence by sentence")
print("=" * 65)

sentences = split_sentences(paragraph)
for i, s in enumerate(sentences, 1):
    result = simplify_sentence(s)
    changed = "✅" if result.replace(" ", "") != s.lower().replace(" ", "") else "➡️ "
    print(f"\n[{i}] {changed} Original  : {s}")
    print(f"[{i}]    Simplified: {result}")

print("\n" + "=" * 65)
print("FULL SIMPLIFIED PARAGRAPH")
print("=" * 65)
print(simplify_paragraph(paragraph))

# word count comparison
orig_wc = len(paragraph.split())
simp_wc = len(simplify_paragraph(paragraph).split())
print(f"\n  Original  : {orig_wc} words")
print(f"  Simplified: {simp_wc} words")
print(f"  Reduction : {round((orig_wc - simp_wc) / orig_wc * 100)}%")