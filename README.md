# Nyaya Case Insight

An end-to-end Indian legal AI prototype for retrieval-augmented legal assistance. The system separates legal questions into source-aware lanes: official reference law, case-law retrieval, and uploaded-document question answering. It is designed as a research and portfolio project, not as production legal advice.

## Project Summary

Large language models can produce fluent legal answers, but legal assistance requires grounding in the correct source. In Indian law, a useful answer may need a statute, rule, constitutional provision, precedent, or a user-uploaded document. This project addresses that problem through a statute-first Retrieval-Augmented Generation (RAG) architecture with separate retrieval paths for reference law, case law, and uploaded documents.

The system combines a FastAPI backend, Streamlit frontend, FAISS vector search, SQLite/FTS metadata search, legal-domain embeddings, and a NyayaAnumana-related classifier for judgment-prediction support.

## Key Features

- Statute-first routing for rule, remedy, procedure, definition, and timeline questions.
- Reference-law retrieval over statutes, rules, constitutional provisions, and official legal texts.
- Case-law retrieval over Indian judgment records for precedent lookup and similar-case search.
- Uploaded-document Q/A for user-provided legal documents.
- Judgment prediction component using NyayaAnumana-related classifier resources.
- Structured answer format with source, reasoning, next step, and caution.
- Benchmark-oriented evaluation on ILDC, ILSIC-Lay, and IL-PCSR.
- Docker Compose deployment with separate FastAPI and Streamlit services.
- Cloudflare Tunnel support for temporary live demos.

## Architecture

```text
User Query
   |
   v
Streamlit Frontend
   |
   v
FastAPI Backend
   |
   v
Query Router
   |
   +--> Reference-Law Lane
   |       statutes, rules, constitutional provisions, official legal texts
   |
   +--> Case-Law Lane
   |       judgment retrieval, similar cases, case review, prediction support
   |
   +--> Uploaded-Document Lane
           session-level document Q/A
   |
   v
Evidence Pack
   |
   v
Answer Generation + Source/Caution Layer
```

More details are available in [docs/architecture.md](docs/architecture.md).

## Data Sources

The local system uses:

- Reference-law corpus: 13,652 indexed legal records from statutes, rules, constitutional provisions, codes, and official legal texts.
- Case-law corpus: 370,294 case-level records derived from NyayaAnumana and INLegalLLaMA-related resources.
- QA/text chunk store: 4,308,636 indexed text chunks in the local environment.
- Uploaded-document store: session-level user document processing.

Large datasets, vector indexes, SQLite stores, and model artifacts are not included in this repository because of size and licensing constraints.

## Evaluation Results

| Task | Dataset | Main Result |
|---|---:|---:|
| Judgment prediction | ILDC | Accuracy 61.24%; Macro F1 61.15% |
| Layperson statute identification | ILSIC-Lay | Micro F1 21.36%; Macro F1 20.50%; MRR 0.3213 |
| Statute retrieval | IL-PCSR | Recall@10 0.1846; MRR 0.2263; MAP 0.0971 |
| Precedent retrieval | IL-PCSR | Recall@10 0.3327; MRR 0.2860; MAP 0.1797 |

The results show moderate judgment-prediction performance and modest retrieval performance. The main contribution is the end-to-end source-routed RAG design and evaluation workflow, not state-of-the-art benchmark performance.

More details are available in [docs/evaluation.md](docs/evaluation.md).

## Local Setup

Create a `.env` file from `.env.example` and ensure local artifacts are available in `artifacts/`.

```powershell
cd C:\Project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the backend:

```powershell
python app.py
```

Run the frontend in a second terminal:

```powershell
streamlit run streamlit_app.py
```

Open:

```text
http://127.0.0.1:8501
```

## Docker Deployment

The project includes a Docker Compose setup with two services:

- `nyaya-api`: FastAPI backend on port `8000`.
- `nyaya-web`: Streamlit frontend on port `8501`.

Start the system:

```powershell
cd C:\Project
docker compose up -d --build
```

Open:

```text
http://127.0.0.1:8501
```

Check backend health:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Stop:

```powershell
docker compose down
```

More details are available in [docs/deployment.md](docs/deployment.md).

## Cloudflare Tunnel Demo

After Docker is running:

```powershell
cloudflared tunnel --url http://127.0.0.1:8501
```

Cloudflare returns a temporary public URL. The link remains active only while the local machine, Docker containers, and tunnel process are running.

## Repository Contents

```text
src/                  Core legal AI services, retrieval, routing, API
scripts/              Build and utility scripts
streamlit_app.py      Streamlit frontend
app.py                FastAPI launcher
Dockerfile            Container image definition
docker-compose.yml    Two-service local deployment
docs/                 Architecture, evaluation, deployment notes
demo_questions.md     Curated demo questions
requirements.txt      Python dependencies
.env.example          Safe environment variable template
```

## What Is Not Included

The following are intentionally excluded from GitHub:

- `.env`
- `.venv/`
- `artifacts/`
- FAISS indexes
- SQLite databases
- downloaded datasets
- model weights
- uploaded user documents
- Cloudflare credentials

## Responsible Use

This system provides legal information and research support. It does not provide legal advice and should not be used as a substitute for a qualified legal professional. Retrieved sources should be checked against official and current legal materials before reliance.

## Tech Stack

- Python
- FastAPI
- Streamlit
- FAISS
- SQLite / FTS5
- Sentence Transformers
- InLegal-SBERT
- Transformers / PyTorch
- Docker Compose
- Cloudflare Tunnel

