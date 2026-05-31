# Architecture

## Overview

The system is implemented as a source-routed RAG application for Indian legal assistance. A standard RAG system retrieves external evidence before producing an answer. This project adds a routing layer before retrieval so that legal questions are sent to the most appropriate evidence source.

## Main Components

```text
User
  |
  v
Streamlit UI
  |
  v
FastAPI Backend
  |
  v
Query Router
  |
  +--> Reference-Law Retriever
  +--> Case-Law Retriever
  +--> Uploaded-Document Retriever
  |
  v
Evidence Pack Builder
  |
  v
Answer Generator
  |
  v
Structured Answer + Sources + Caution
```

## Source Lanes

### Reference-Law Lane

Used for questions about rules, remedies, procedures, timelines, definitions, statutory rights, constitutional provisions, and direct legal provisions. This lane uses official legal materials such as statutes, rules, codes, constitutional text, and related reference-law records.

### Case-Law Lane

Used for precedent support, similar-case retrieval, case explanation, and outcome-pattern analysis. This lane searches judgment-level and passage-level case material.

### Uploaded-Document Lane

Used when a user uploads a document and asks questions about that document. The uploaded document is treated as session-level context and is not mixed into the permanent legal corpus.

## Retrieval Design

The retrieval layer combines:

- Dense semantic retrieval with legal-domain embeddings.
- FAISS vector search.
- SQLite metadata storage.
- SQLite FTS5 lexical search.
- Domain-aware routing and scoring.
- Exact provision handling for direct statute and constitutional questions.

## Why Source Routing Matters

Legal sources are not interchangeable. A statute provides the legal rule, a judgment applies or interprets the rule, and an uploaded document provides user-specific facts. Separating these sources reduces the risk of generating a fluent answer from the wrong legal authority.
