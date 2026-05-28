# ExplainThis AI Project Documentation

## Project Overview

ExplainThis AI is a web application that simplifies complex English text into text that is easier to understand. The project uses a React frontend, a FastAPI backend, and a fine-tuned AI text simplification model.

The application is useful for students, English learners, and anyone who reads academic, technical, or formal English text.

## Project Idea

The idea of ExplainThis AI is to help users understand difficult English text faster. The user writes or pastes text into the web application, clicks the Simplify button, and receives a clearer version.

## Problem Statement

Many English texts are difficult because they contain long sentences, complex vocabulary, or formal academic language. Ready-made text simplification models did not always produce useful simplifications during testing. Some models returned almost the same text as the input.

Because of this, the project moved from only testing ready-made models to fine-tuning models using text simplification datasets.

## Objectives

- Build a simple web application for English text simplification.
- Design and implement a frontend using React.
- Develop a backend using FastAPI.
- Connect the frontend with the backend API.
- Train and test text simplification models.
- Compare ASSET and WikiLarge models.
- Integrate the final AI model into the backend.

## Frontend

The frontend was built using React. React was new to the team, so it was learned during the implementation of the project.

### Frontend Features

- Simple and clear design.
- Text box for entering text.
- Button to simplify the text.
- Organized display for the simplified result.
- Dark mode and light mode support.
- Responsive design for mobile, tablet, laptop, and large screens.
- Copy and clear actions.
- Loading and error feedback.

## Backend

The backend was built using FastAPI.

### Backend Responsibilities

- Receive text from the frontend.
- Send the text to the AI model.
- Receive the simplified text from the model.
- Return the simplified text to the frontend.

The backend acts as the connection between the user interface and the AI model.

## AI Model Development

### Ready-Made Model Testing

At the beginning, ready-made models from Hugging Face were tested. The problem was that some models returned the same text without real simplification.

Example problem:

```text
Input = Output
```

For this reason, the team decided to use fine-tuning with custom training data.

### ASSET Dataset Training

The first dataset used was ASSET. ASSET is one of the well-known datasets for text simplification.

ASSET provides 10 human-written simplifications for each original sentence.

#### First ASSET Attempt

- Used a small amount of data.
- Used only one simplification from the 10 available simplifications.

Result: the model performance was weak and the model barely changed the input text.

#### Improved ASSET Training

The training was improved by:

- Increasing the amount of data.
- Using all 10 simplifications for each sentence.

Result: the model improved significantly and produced better simplifications.

### WikiLarge Dataset Training

After improving the ASSET model, the team also trained a model using the WikiLarge dataset.

Training details:

- Used BART-base.
- Trained on 15,000 examples.
- Used 500 validation examples.
- Monitored training loss and validation loss.
- Used early stopping to stop training when improvement slowed down.

Result: the WikiLarge model gave good results and reduced text length by about 21%.

## Model Comparison

After training, the ASSET and WikiLarge models were compared using the same paragraph.

Comparison results:

- ASSET preserved the meaning better.
- WikiLarge shortened the text more.
- WikiLarge sometimes removed important information.
- ASSET gave safer and more accurate simplification.

Final decision: the ASSET model was selected for the final application because it gave the best balance between simplifying the text and keeping the meaning.

## Integration

After choosing the final model:

- The model was saved.
- It was connected to the FastAPI backend.
- The frontend was connected to the backend API.
- The application became able to receive user text and return a simplified version directly.

## Technologies Used

- React
- CSS
- FastAPI
- Python
- Hugging Face Transformers
- BART-base
- ASSET Dataset
- WikiLarge Dataset

## Project Structure

```text
explainthis-ai/
  backend/
    main.py
    requirements.txt
  frontend/
    src/
      App.jsx
      App.css
      api.js
      main.jsx
      components/
        Navbar.jsx
      assets/
        logo.png
  internship_submission/
    presentation/
      index.html
      style.css
      script.js
    speaker_notes.md
    project_documentation.md
    task_file.md
```

## How the System Works

1. The user opens the frontend.
2. The user writes or pastes complex English text.
3. The frontend sends the text to the FastAPI backend.
4. The backend passes the text to the final ASSET model.
5. The model generates a simpler version.
6. The backend returns the result.
7. The frontend displays the simplified text.

## Installation and Setup

### Backend

```bash
cd backend
pip install -r requirements.txt
python main.py
```

Backend URL:

```text
http://localhost:8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend URL:

```text
http://localhost:5173
```

## Manual Testing

| Test Case | Expected Result | Status |
| --- | --- | --- |
| Open frontend page | Page loads successfully | Completed |
| Enter English text | User can type or paste text | Completed |
| Click Simplify | Text is sent to backend | Completed |
| Backend returns result | Simplified text appears | Completed |
| Click clear button | Input and output are removed | Completed |
| Click copy button | Simplified output is copied | Completed |
| Test model performance | Output quality is reviewed | Completed |
| Compare ASSET and WikiLarge | Final model is selected | Completed |

## Challenges

- Learning React from scratch.
- Choosing the suitable AI model.
- Training models and improving results.
- Improving preprocessing.
- Comparing different model outputs.
- Integrating the model with the full web application.

## What We Learned

- Building a modern frontend with React.
- Creating an API with FastAPI.
- Fine-tuning AI models.
- Testing and comparing model performance.
- Integrating AI into a complete web application.

## Conclusion

ExplainThis AI is a complete web application that uses AI to simplify difficult English text. The project started with ready-made model testing, then moved to fine-tuning using ASSET and WikiLarge. After comparison, the ASSET model was selected because it preserved meaning better while still simplifying the text.
