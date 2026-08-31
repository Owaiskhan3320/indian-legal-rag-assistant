# Indian Legal Research & Retrieval System

Indian Legal Research & Retrieval System is an Indian legal Retrieval-Augmented Generation (RAG) project. It answers legal-information questions by routing them to the most suitable source: reference law, case law, or a document uploaded by the user.

> This is a learning and portfolio project. It provides legal information, not legal advice.

## What It Does

- Answers questions using retrieved legal sources instead of relying only on an LLM.
- Uses official legal materials for questions about sections, articles, rules, rights, and procedures.
- Uses Indian judgments for similar-case and precedent questions.
- Supports question answering over a document uploaded during the current session.
- Shows the sources used for an answer and supports follow-up questions.

## Why Source Routing Matters

Different legal questions need different evidence. A question about a statutory time limit should be grounded in legislation, while a request for similar cases should search judgments. Keeping these sources separate makes retrieval easier to understand and reduces answers based on the wrong type of document.

## How It Works

```text
User question
    |
    v
Streamlit interface
    |
    v
FastAPI backend
    |
    v
Query router
    |
    +-- Reference law
    +-- Case law
    +-- Uploaded document
    |
    v
Answer with sources
```

More detail is available in [docs/architecture.md](docs/architecture.md).

## Current Scope

| Feature | Status |
|---|---|
| Source-routed legal Q&A | Implemented |
| Reference-law retrieval | Available with local or demo data |
| Case-law retrieval | Available with local or demo data |
| Uploaded-document Q&A | Implemented for the current session |
| Follow-up questions and source display | Implemented |
| Judgment prediction | Experimental backend research component; not shown in the current UI |
| Legal advice | Not supported |

The repository still contains an experimental judgment-prediction endpoint used during research and evaluation. It is not part of the current Streamlit workflow and its output must not be treated as a legal conclusion.

## Quick Start

These commands use PowerShell on Windows.

```powershell
cd C:\Project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Review `.env` and configure the local OpenAI-compatible LLM endpoint and model. The example configuration uses `http://127.0.0.1:1234/v1`, which can be provided by a local model server such as LM Studio.

Start the backend:

```powershell
python app.py
```

Start the frontend in a second terminal:

```powershell
cd C:\Project
.\.venv\Scripts\Activate.ps1
streamlit run streamlit_app.py
```

Open `http://127.0.0.1:8501` in a browser. The FastAPI launcher is the root [app.py](app.py), so the equivalent Uvicorn command is `python -m uvicorn app:app`.

## Demo Mode

The full legal datasets and indexes are not included in GitHub. A small sample corpus is provided so the main workflow can still be tested.

Build the demo store:

```powershell
python scripts/build_demo_store.py
```

Then set demo mode before starting both the backend and frontend:

```powershell
$env:DEMO_MODE="true"
python app.py
```

Use the same `DEMO_MODE` command in the frontend terminal before running Streamlit. Suggested prompts are listed in [demo_questions.md](demo_questions.md).

## Tests

Install the development dependencies and run the same focused checks used by CI:

```powershell
pip install -r requirements-dev.txt
python -m pytest -v
python -m ruff check app.py src tests scripts --select E9,F63,F7,F82
python -m py_compile streamlit_app.py
```

## Docker

For a local container run:

```powershell
docker compose up -d --build
```

To build the sample store with Docker:

```powershell
$env:DEMO_MODE="true"
docker compose --profile demo run --rm nyaya-demo-builder
docker compose up -d --build
```

See [docs/deployment.md](docs/deployment.md) for health checks, environment settings, and demo deployment notes.

## Data

- **Case law:** judgment data and classifier resources related to NyayaAnumana and INLegalLLaMA are used for research and evaluation. They are not redistributed in this repository.
- **Reference law:** the full local setup uses official statutes, rules, constitutional provisions, and other legal texts. These source files and generated indexes are not redistributed.
- **Public demo:** `sample_data/` contains a small set of records for checking the application after cloning it. It is not the full research corpus.
- **Uploaded documents:** files uploaded by a user are handled as session-level material and are not added to the permanent corpus.

Users reproducing the full setup should obtain legal texts from authoritative sources, check their usage terms, and rebuild the indexes locally.

## Evaluation

The components were evaluated separately because prediction, statute identification, and retrieval measure different behaviour.

| Task | Dataset | Main Result |
|---|---:|---:|
| Judgment prediction | ILDC | Accuracy 61.24%; Macro F1 61.15% |
| Layperson statute identification | ILSIC-Lay | Micro F1 21.36%; Macro F1 20.50%; MRR 0.3213 |
| Statute retrieval | IL-PCSR | Recall@10 0.1846; MRR 0.2263; MAP 0.0971 |
| Precedent retrieval | IL-PCSR | Recall@10 0.3327; MRR 0.2860; MAP 0.1797 |

These are modest results and are reported as measured. They help show where the system works and where retrieval still needs improvement. See [docs/evaluation.md](docs/evaluation.md) for the evaluation setup and interpretation.

## Limitations

- Answer quality depends on the coverage and freshness of the indexed corpus.
- Retrieval and generated answers can still be incomplete or incorrect.
- The public demo corpus is intentionally small and cannot answer broad legal questions reliably.
- The system does not automatically guarantee that a law or judgment is current.
- A qualified legal professional and current official sources should be consulted before taking action.

## Project Structure

```text
app.py                 FastAPI launcher
streamlit_app.py       Streamlit interface
src/                   Routing, retrieval, services, and API code
scripts/               Data-building and evaluation scripts
tests/                 Focused API and routing tests
sample_data/           Small public demo corpus
docs/                  Architecture, evaluation, and deployment notes
demo_questions.md      Example questions for the demo
```

Large runtime files such as `.env`, model weights, FAISS indexes, SQLite databases, downloaded datasets, and uploaded documents are intentionally excluded from Git.

## References

- Shubham Kumar Nigam et al. [NyayaAnumana and INLegalLLaMA](https://aclanthology.org/2025.coling-main.738/), COLING 2025.
- Vijit Malik et al. [ILDC for CJPE](https://aclanthology.org/2021.acl-long.313/), ACL-IJCNLP 2021.
- Shounak Paul et al. [ILSIC](https://aclanthology.org/2026.findings-eacl.354/), Findings of EACL 2026.
- Shounak Paul et al. [IL-PCSR](https://aclanthology.org/2025.emnlp-main.738/), EMNLP 2025.
- Government of India, [India Code](https://www.indiacode.nic.in/).

## License and Responsible Use

The source code is released under the MIT License. External datasets, legal texts, and models remain governed by their original licences and terms.

This project is for legal information, research, and education. It must not be used as the sole basis for legal action, and retrieved authorities should be checked against current official materials.
