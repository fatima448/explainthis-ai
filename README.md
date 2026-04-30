# Explain-This-AI 🧠✨

An AI-powered web application that simplifies complex English text using Natural Language Processing (NLP).

---

## 🚀 Overview

Explain-This-AI is designed to make difficult content easier to understand.
It takes complex sentences and rewrites them in a simpler, clearer form using a fine-tuned transformer model.

This project combines:

* AI / NLP (Text Simplification)
* Backend development (FastAPI )
* Frontend web interface
* Model training and evaluation pipeline

---

## 🧩 Features

* ✨ Simplify complex English text
* 🤖 Custom-trained BART model for text simplification
* 🔁 Training and testing pipeline included
* 🌐 Full-stack web application (Frontend + Backend)

---

## 🏗️ Project Structure

```
Explain-This-AI/
│
├── backend/        # API logic (FastAPI / Flask)
├── frontend/       # React 
├── model/          # Training & testing scripts (no large models included)
├── scripts/        # Helper scripts (Preprocessing scripts)
├── .gitignore
├── README.md
```

---

## 🧠 Model Details

* Model: BART (fine-tuned for text simplification)
* Dataset: ASSET dataset (multi-reference simplification)
* Training:

  * Supports multiple references per sentence
  * Improved generalization with expanded dataset
* Evaluation:

  * Testing script included for performance analysis

---

## ⚙️ Installation

### 1. Clone the repository

```
git clone https://github.com/your-username/explain-this-ai.git
cd explain-this-ai
```

### 2. Create virtual environment

```
python -m venv venv
```

### 3. Activate environment

* Windows:

```
venv\Scripts\activate
```

* Mac/Linux:

```
source venv/bin/activate
```

### 4. Install dependencies

```
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### Start Backend

```
cd backend
python app.py
```

### Open Frontend

Open the `frontend` files in your browser.

---

## 🏋️ Training the Model

Run:

```
python model/training.py
```

---

## 🧪 Testing the Model

Run:

```
python model/testing.py
```

---

## ⚠️ Note About Models

Trained models and checkpoints are **not included** in this repository due to size limitations.

You can:

* Train the model yourself using the provided scripts

---

## 📌 Future Improvements

* 🔥 Improve Arabic text simplification
* ⚡ Optimize model performance
* 📊 Add evaluation metrics (BLEU, ROUGE)


