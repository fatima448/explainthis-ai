import os
import re
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

load_dotenv()

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    r"C:\Users\fatoo\explain-this-ai\model\bart-simplification-final"  # fallback
)

try:
    print(f"Loading model from: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()  
    print(f"Model loaded on {device}")
except Exception as e:
    print(f"Failed to load model: {e}")
    raise RuntimeError(f"Model loading failed: {e}")



class ExplainRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=5000)

class ExplainResponse(BaseModel):
    simplified_text: str
    original_word_count: int
    simplified_word_count: int



app = FastAPI()

origins = [os.getenv("FRONTEND_URL", "http://localhost:5173")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


_NOISE_PUNCT_RE = re.compile(r'[\"#$%&\*+/<=>@\[\\\]^_`{|}~]')
_WHITESPACE_RE  = re.compile(r'\s{2,}')
_URL_RE         = re.compile(r'http\S+')
_CITATION_RE    = re.compile(r'\[\d+\]')
_WIKI_TMPL_RE   = re.compile(r'\{\{.*?\}\}')


def preprocess_text(text: str) -> str:
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


def count_words(text: str) -> int:
    return len(text.split())



def simplify_sentence(sentence: str) -> str:
    sentence = sentence.strip()
    if not sentence:
        return ""

    prefixed = "simplify: " + sentence

    inputs = tokenizer(
        prefixed,
        return_tensors="pt",
        truncation=True,
        max_length=128,              
    ).to(device)

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


def split_sentences(text: str) -> list[str]:

    text = re.sub(r'(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|approx|dept)\.',
                  r'<PERIOD>', text)
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$', text)
    
    sentences = [s.replace('<PERIOD>', '.').strip() for s in sentences]
    return [s for s in sentences if s]  

@app.get("/health")
async def health():
    return {"status": "ok", "model_path": MODEL_PATH, "device": device}


@app.post("/explain", response_model=ExplainResponse)
async def explain(request: ExplainRequest):
    try:
        orig_wc = count_words(request.text)
        sentences = split_sentences(request.text)

        simplified_sentences = []
        for sentence in sentences:
            clean = preprocess_text(sentence)  
            if len(clean.split()) < 4:         
                simplified_sentences.append(clean)
                continue
            result = simplify_sentence(clean)
            if result:
                simplified_sentences.append(result)

        simplified = " ".join(simplified_sentences)
        simp_wc    = count_words(simplified)

        return ExplainResponse(
            simplified_text=simplified,
            original_word_count=orig_wc,
            simplified_word_count=simp_wc,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simplification failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)