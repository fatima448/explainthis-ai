# Internship Project Task File

Project name: ExplainThis AI

Team members: Fatima and Sayf

Submission deadline: 31 May 2026

Presentation date: 2 June 2026

## Project Summary

ExplainThis AI is a web application that simplifies complex English text using artificial intelligence. The user writes or pastes difficult text, clicks Simplify, and receives a clearer version that is easier to understand.

## Assigned Tasks

| Task | Owner | Status |
| --- | --- | --- |
| Fatima and Sayf testing | Fatima and Sayf | Completed |
| Design and implement frontend | Fatima and Sayf | Completed |
| Develop backend using FastAPI | Fatima and Sayf | Completed |
| Connect frontend with backend API | Fatima and Sayf | Completed |
| Train initial simplification model | Fatima and Sayf | Completed |
| Improve preprocessing | Fatima and Sayf | Completed |
| Train text simplification model | Fatima and Sayf | Completed |
| Test model performance | Fatima and Sayf | Completed |
| Integrate AI model into backend | Fatima and Sayf | Completed |
| Train WikiLarge text simplification model | Fatima and Sayf | Completed |
| Compare ASSET and WikiLarge models | Fatima and Sayf | Completed |

## Final Deliverables

| Deliverable | Status | Notes |
| --- | --- | --- |
| Final presentation slides | Completed | HTML/CSS/JavaScript presentation created in `internship_submission/presentation/` |
| Speaker notes | Completed | Reading script created in `internship_submission/speaker_notes.md` |
| Project documentation | Completed | Documentation created in `internship_submission/project_documentation.md` |
| Task file | Completed | This file includes the latest tasks and edits |
| Frontend project | Completed | React frontend exists in `frontend/` |
| Backend project | Completed | FastAPI backend exists in `backend/` |

## Completed Work Details

### Frontend

- Used React for the frontend.
- Learned React while building the project.
- Designed a simple and clear user interface.
- Added text input for the user.
- Added Simplify action.
- Added output area for simplified text.
- Added dark mode and light mode.
- Made the design responsive for mobile, tablet, laptop, and large screens.

### Backend

- Used FastAPI for the backend.
- Created the API endpoint for simplification.
- Received text from the frontend.
- Sent text to the AI model.
- Returned simplified text to the frontend.
- Connected frontend and backend together.

### AI Model Work

- Tested ready-made Hugging Face models.
- Found that some models returned the same text without real simplification.
- Trained an initial simplification model.
- Improved preprocessing.
- Trained using the ASSET dataset.
- Improved ASSET training by using more data and all 10 simplifications per sentence.
- Trained a WikiLarge text simplification model using BART-base.
- Used 15,000 training examples and 500 validation examples for WikiLarge.
- Monitored training loss and validation loss.
- Used early stopping.
- Tested model performance.
- Compared ASSET and WikiLarge models.
- Chose ASSET for the final app because it preserved meaning better.
- Integrated the final AI model into the backend.

## Model Comparison Summary

| Model | Strength | Weakness | Final Decision |
| --- | --- | --- | --- |
| ASSET | Preserved meaning better and gave safer simplification | Sometimes simplified less than WikiLarge | Selected for final app |
| WikiLarge | Shortened text more and reduced length by about 21% | Sometimes removed important information | Not selected for final app |

## Manual Test Summary

| Test | Result |
| --- | --- |
| Frontend page opens | Passed |
| User can enter English text | Passed |
| Simplify button sends request | Passed |
| Backend receives text | Passed |
| AI model returns simplified text | Passed |
| Simplified output displays | Passed |
| Clear button works | Passed |
| Copy button works | Passed |
| Dark/light mode works | Passed |
| Responsive design checked | Passed |
| ASSET model tested | Passed |
| WikiLarge model tested | Passed |
| ASSET and WikiLarge compared | Passed |

## Challenges

- Learning React from scratch.
- Choosing the right model.
- Training models and improving the results.
- Improving preprocessing.
- Testing model performance.
- Comparing ASSET and WikiLarge.
- Integrating the AI model into the complete application.

## What We Learned

- How to build a frontend with React.
- How to create a backend API with FastAPI.
- How to fine-tune text simplification models.
- How to evaluate model results.
- How to compare different AI models.
- How to integrate an AI model into a web application.

## Latest Edits Included

- Updated presentation to include the full project story.
- Updated speaker notes with a clear explanation script.
- Updated project documentation with frontend, backend, training, comparison, and integration details.
- Updated this task file with all assigned tasks.
- Added ASSET and WikiLarge comparison summary.

## Final Checklist

- Presentation completed
- Speaker notes completed
- Documentation completed
- Task file completed
- Assigned task list included
- Latest model work included
- Files checked and ready for review

Prepared by: Fatima and Sayf
