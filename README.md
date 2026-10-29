# RFP Intelligence Platform

RAG search engine and multi-agent system for extracting structured data from bid / RFP documents.

## Setup
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
ollama pull qwen3:8b
```

## Run
```bash
python main.py --bid ./data/Bid1
python main.py --ask "What is the submission deadline for Bid1 after all addendums?"
```

_Architecture, design decisions and evaluation results: to be added._
