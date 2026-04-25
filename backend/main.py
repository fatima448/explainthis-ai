import os
import re
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch



MODEL_PATH = "eilamc14/t5-large-text-simplification"


try:
    model_name = os.getenv("MODEL_PATH", "eilamc14/t5-large-text-simplification")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"✅ Model loaded from: {model_name}")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    raise RuntimeError(f"Model loading failed: {e}")
 

class ExplainRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=5000)

class ExplainResponse(BaseModel):
    simplified_text: str
    original_word_count: int
    simplified_word_count: int


app = FastAPI()

def preprocess_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)          # collapse multiple spaces/newlines
    text = text.encode("utf-8", errors="ignore").decode("utf-8")
    return text
 
 
def count_words(text: str) -> int:
    return len(text.split())
 

def simplify_text(text: str) -> str:
    prefix = "summarize and simplify:"
    full_input = prefix + text

    inputs = tokenizer(
        full_input,
        return_tensors="pt",
        truncation=True,
    ).to(model.device)
 
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=150,
        min_length=30,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8,
        repetition_penalty=3.0,
    
    )
 
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
 
origins = [
    os.getenv("FRONTEND_URL", "http://localhost:5173"),
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "ok"}
 
@app.post("/explain", response_model=ExplainResponse)
async def explain(request: ExplainRequest):
    try:
        clean_text = preprocess_text(request.text)
        original_wc = count_words(clean_text)
 
        simplified = simplify_text(clean_text)
 
        simplified_wc = count_words(simplified)
 
        return ExplainResponse(
            simplified_text=simplified,
            original_word_count=original_wc,
            simplified_word_count=simplified_wc,
       )  
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simplification failed: {str(e)}")
 

if __name__ == "__main__":
    load_dotenv()
    uvicorn.run(app, host="0.0.0.0", port=8000)