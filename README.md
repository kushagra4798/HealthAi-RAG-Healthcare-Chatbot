# HealthAi — Healthcare Related Chatbot Using RAG

HealthAi is a Python prototype for preliminary healthcare specialty routing. It combines semantic retrieval over a structured specialty knowledge base with locally run language models and deterministic rules for emergency-related phrases, selected red-flag symptoms, and hospital FAQs.

> **Important:** This is an educational prototype, not a medical device or diagnostic tool. It has not been clinically validated and must not be used to make medical decisions. For emergencies, contact local emergency services or seek urgent medical care.

## Features

- Retrieval-Augmented Generation (RAG)-style specialty retrieval using embeddings and cosine similarity
- Local model inference through Ollama
- Streamlit chat interface and command-line interface
- Rule-based handling for selected emergency phrases and red-flag symptom patterns
- Confidence-based follow-up questions and a fallback response
- FAQ shortcuts based on the included knowledge base

## Technology stack

- Python
- Streamlit
- Ollama
- Llama 3
- Nomic Embed Text
- NumPy

## Project structure

```text
ChatBot/
├── app.py
├── chatbot_core.py
├── knowledge_base.py
├── main.py
├── ChatBot.bat
├── requirements.txt
├── .gitignore
└── README.md
```

## Prerequisites

1. Install Python (use a version supported by the installed versions of Streamlit, NumPy, and the Ollama Python package).
2. Install [Ollama](https://ollama.com/) and make sure its local service is running.
3. Download the required models:

```bash
ollama pull llama3
ollama pull nomic-embed-text
```

## Setup

Open a terminal in this project directory and create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bat
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the application

Streamlit web interface:

```bash
python -m streamlit run app.py
```

Command-line interface:

```bash
python main.py
```

On Windows, you can also run `ChatBot.bat`.

The first startup builds an in-memory embedding index, so startup time depends on your machine and the local Ollama service.

## Privacy and security

- The project is designed to call models through a local Ollama service rather than a third-party cloud AI API.
- This alone does not make the application production-secure. The web interface has no built-in authentication or TLS configuration in this prototype.
- Do not enter real patient information into a public demo.
- Do not deploy this application on a shared or public network without appropriate authentication, transport security, access controls, and clinical/privacy review.
- Before making this repository public, verify permission to publish all hospital information, doctor names, contact details, and other third-party data in `knowledge_base.py`. Replace them with clearly labelled fictional demo data if permission is not available.

## Known limitations

- The knowledge base and vector index are held in memory.
- The current prototype has no persistent database or audit log.
- Emergency and red-flag detection uses predefined rules and can miss phrasing, languages, or symptoms not covered by those rules.
- Model output can be incorrect or misleading; confidence scores are retrieval similarity indicators, not clinically validated probabilities.
- Automated test coverage and clinical validation are not included.
- The application should not be considered production-ready.

## Suggested future improvements

- Add automated tests for emergency handling, FAQs, red-flag rules, and fallback paths.
- Add input limits, robust exception handling, and strict output validation.
- Separate current-query retrieval from conversation history.
- Add clinician-reviewed rules and a persistent, privacy-reviewed audit process where appropriate.
- Improve language coverage and evaluate the system with a documented test set.

## Acknowledgement

Project: **Healthcare Related Chatbot Using RAG**  
Developed under the guidance of Mr. Navpreet Singh, Computer Centre, IIT Kanpur, as stated in the project certificate.

## License

No license is included yet. Add a license only after confirming that you have the right to license every included part of the project and its data.
