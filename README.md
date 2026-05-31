# Nyaya Case Insight

Nyaya Case Insight is an end-to-end Indian legal RAG project that routes legal questions to the right kind of source before generating an answer. The system separates official reference law, case-law retrieval, and uploaded-document question answering instead of treating all legal text as one mixed corpus.

The project is built as a portfolio-grade legal AI engineering system: it includes retrieval, classification, structured answer generation, benchmark evaluation, Docker deployment, and Cloudflare Tunnel support for live demos. It is a legal information and research-assistance prototype, not a legal-advice product.

## Why This Project Exists

Legal questions are source-sensitive. A user asking about a time limit, remedy, definition, constitutional right, or procedure usually needs the correct statute, rule, article, or official legal text first. A user asking for similar cases or precedent support needs judgments. A user asking about an uploaded notice, order, contract, or judgment needs document-specific analysis.

General-purpose LLM answers often blur these boundaries. They may sound fluent while relying on the wrong source type. This project addresses that problem through a source-routed Retrieval-Augmented Generation architecture.

## What The System Does

- Routes legal questions to reference law, case law, uploaded documents, or a hybrid path.
- Retrieves official legal provisions for statute-first questions.
- Retrieves Indian judgments for similar-case search, precedent support, and case explanation.
- Supports uploaded-document Q/A within the current session.
- Includes a judgment-prediction component based on NyayaAnumana-related classifier resources.
- Produces structured answers with source labels, reasoning, next steps, and caution.
- Reports benchmark results on judgment prediction, statute identification, statute retrieval, and precedent retrieval.
- Runs locally with Docker Compose and can be exposed through Cloudflare Tunnel for demos.

## System Architecture

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
   |       judgments, similar cases, precedent support, case explanation
   |
   +--> Uploaded-Document Lane
           session-level document Q/A
   |
   v
Evidence Pack
   |
   v
Answer Generation + Citation/Caution Layer
```

The routing layer is the main design choice. It prevents the system from answering a statutory question using unrelated cases, or a document-specific question using general legal material only.

Detailed architecture notes are in [docs/architecture.md](docs/architecture.md).

## Main Workflows

### Case Q/A

Used for legal questions, statute-first answers, precedent support, and follow-up questions. The system decides whether the query should use official law, case law, uploaded document context, or a hybrid source path.

### Case Review

Used for structured case intake. The user provides facts, forum, role, relief sought, evidence, and opponent argument. The system retrieves similar cases and provides an outcome-oriented review with caution.

### Uploaded Document Q/A

Used when the user uploads a legal document and asks questions about that document. The uploaded file is treated as session-level material and is not added to the permanent corpus.

## Data And Source Acknowledgement

### Case-Law Data

The case-law side of this project uses judgment data and classifier resources related to **NyayaAnumana and INLegalLLaMA**. NyayaAnumana is introduced by Nigam et al. as a large-scale Indian legal judgment prediction dataset covering multiple Indian court and tribunal sources. In this project, those resources are used in two ways:

- As a case-law corpus for retrieval, similar-case search, case explanation, and precedent-style support.
- As the basis for the judgment-prediction component used as a triage signal.

The judgment-prediction output is not treated as a legal conclusion. It is only a supporting signal inside the broader legal assistance workflow.

### Reference-Law Data

The reference-law lane uses locally prepared official legal materials such as statutes, rules, constitutional provisions, codes, and other official legal texts. These files are used to support statute-first retrieval for questions about legal rules, remedies, procedures, timelines, definitions, and constitutional provisions.

The repository does not redistribute the local reference-law files or indexed artifacts. Users who want to reproduce the full setup should obtain official legal texts from authoritative sources such as India Code and relevant government/legal department sources, then rebuild the local reference-law index.

### Uploaded Documents

Uploaded documents are handled at session level. They are used only for the current interaction and are not part of the permanent case-law or reference-law corpus.

## Evaluation

The project was evaluated as a research artefact across separate benchmark lanes. These tasks measure different behaviours and should not be collapsed into one score.

| Task | Dataset | Main Result |
|---|---:|---:|
| Judgment prediction | ILDC | Accuracy 61.24%; Macro F1 61.15% |
| Layperson statute identification | ILSIC-Lay | Micro F1 21.36%; Macro F1 20.50%; MRR 0.3213 |
| Statute retrieval | IL-PCSR | Recall@10 0.1846; MRR 0.2263; MAP 0.0971 |
| Precedent retrieval | IL-PCSR | Recall@10 0.3327; MRR 0.2860; MAP 0.1797 |

The results are intentionally reported conservatively. The system shows moderate judgment-prediction performance and modest retrieval performance. The strongest contribution is not leaderboard performance; it is the complete source-routed architecture, benchmark mapping, and deployable legal RAG workflow.

Detailed evaluation notes are in [docs/evaluation.md](docs/evaluation.md).

## Technology Stack

- Python
- FastAPI
- Streamlit
- FAISS
- SQLite and SQLite FTS5
- Sentence Transformers
- `bhavyagiri/InLegal-Sbert`
- Transformers and PyTorch
- Docker Compose
- Cloudflare Tunnel

## Local Development

Create a local environment:

```powershell
cd C:\Project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy environment settings:

```powershell
copy .env.example .env
```

Run the backend:

```powershell
python app.py
```

Run the frontend in another terminal:

```powershell
streamlit run streamlit_app.py
```

Open:

```text
http://127.0.0.1:8501
```

## Docker Deployment

The Docker setup runs the backend and frontend as separate services:

```text
nyaya-api  -> FastAPI backend on port 8000
nyaya-web  -> Streamlit frontend on port 8501
```

Start:

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

Detailed deployment notes are in [docs/deployment.md](docs/deployment.md).

## Cloudflare Tunnel Demo

After Docker is running:

```powershell
cloudflared tunnel --url http://127.0.0.1:8501
```

This creates a temporary public link for live demos. The link works only while the local machine, Docker containers, and Cloudflare Tunnel process are running.

## Repository Contents

```text
src/                  Core backend, routing, retrieval, answer, and API code
scripts/              Build and evaluation utilities
docs/                 Architecture, evaluation, and deployment notes
streamlit_app.py      Streamlit frontend
app.py                FastAPI launcher
requirements.txt      Python dependencies
Dockerfile            Container image definition
docker-compose.yml    Two-service Docker deployment
.env.example          Safe environment template
demo_questions.md     Curated demo prompts
```

## Files Not Included In This Repository

The repository intentionally excludes large and sensitive runtime assets:

- `.env`
- `.venv/`
- `artifacts/`
- FAISS indexes
- SQLite databases
- downloaded datasets
- model weights
- uploaded user documents
- Cloudflare credentials

This keeps the repository lightweight, safe to clone, and suitable for portfolio review.

## Responsible Use

The system provides legal information and research support. It does not provide legal advice, does not replace a lawyer, and should not be used as the sole basis for legal action. Retrieved authorities should be checked against official and current legal materials before reliance.

## References

- Shubham Kumar Nigam, Deepak Patnaik Balaramamahanthi, Shivam Mishra, Noel Shallum, Kripabandhu Ghosh, and Arnab Bhattacharya. 2025. [NyayaAnumana and INLegalLLaMA: The Largest Indian Legal Judgment Prediction Dataset and Specialized Language Model for Enhanced Decision Analysis](https://aclanthology.org/2025.coling-main.738/). Proceedings of COLING 2025.
- Vijit Malik, Rishabh Sanjay, Shubham Kumar Nigam, Kripabandhu Ghosh, Shouvik Kumar Guha, Arnab Bhattacharya, and Ashutosh Modi. 2021. [ILDC for CJPE: Indian Legal Documents Corpus for Court Judgment Prediction and Explanation](https://aclanthology.org/2021.acl-long.313/). ACL-IJCNLP 2021.
- Shounak Paul, Raghav Dogra, Pawan Goyal, and Saptarshi Ghosh. 2026. [ILSIC: Corpora for Identifying Indian Legal Statutes from Queries by Laymen](https://aclanthology.org/2026.findings-eacl.354/). Findings of EACL 2026.
- Shounak Paul, Dhananjay Ghumare, Pawan Goyal, Saptarshi Ghosh, and Ashutosh Modi. 2025. [IL-PCSR: Legal Corpus for Prior Case and Statute Retrieval](https://aclanthology.org/2025.emnlp-main.738/). EMNLP 2025.
- Government of India. [India Code: Digital Repository of Central and State Acts](https://www.indiacode.nic.in/).

## Portfolio Summary

End-to-end Indian legal RAG system with statute-first routing, case-law retrieval, uploaded-document Q/A, judgment prediction, benchmark evaluation, Docker Compose deployment, and Cloudflare Tunnel demo support.
