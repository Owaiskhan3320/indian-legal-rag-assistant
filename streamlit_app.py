from __future__ import annotations

import base64
import os
from pathlib import Path
import uuid

import httpx
import streamlit as st
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

st.set_page_config(
    page_title="Indian Legal Research & Retrieval System",
    page_icon="N",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        :root {
            --ink: #172033;
            --muted: #657184;
            --line: #d8dee8;
            --paper: #fffdf8;
            --accent: #a33a2b;
        }
        .stApp {
            background:
                linear-gradient(rgba(255, 253, 248, 0.94), rgba(247, 244, 237, 0.97)),
                repeating-linear-gradient(90deg, #ece7dd 0, #ece7dd 1px, transparent 1px, transparent 72px);
            color: var(--ink);
        }
        .block-container {
            max-width: 880px;
            padding-top: 2.4rem;
        }
        h1, h2, h3 {
            font-family: Georgia, "Times New Roman", serif;
            color: var(--ink);
        }
        [data-testid="stChatMessage"] {
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 0.35rem 0.55rem;
        }
        [data-testid="stSidebar"] {
            background: #f1eee7;
            border-right: 1px solid var(--line);
        }
        .source-note {
            color: var(--muted);
            font-size: 0.85rem;
            margin-top: -0.25rem;
        }
        .stButton > button {
            border-radius: 9px;
        }
        div[data-testid="stChatInput"] textarea {
            background: #ffffff;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_default_api_url() -> str:
    return os.getenv("STREAMLIT_API_URL", "http://127.0.0.1:8000").rstrip("/")


def error_detail(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            payload = exc.response.json()
            return str(payload.get("detail") or exc.response.text)
        except ValueError:
            return exc.response.text or str(exc)
    if isinstance(exc, httpx.TimeoutException):
        return "The backend took too long to respond. Try the question again."
    if isinstance(exc, httpx.RequestError):
        return f"Could not reach the backend: {exc}"
    return str(exc)


@st.cache_data(ttl=15, show_spinner=False)
def get_health(api_url: str) -> dict | None:
    try:
        with httpx.Client(timeout=httpx.Timeout(6.0, connect=3.0)) as client:
            response = client.get(f"{api_url}/health")
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError:
        return None


def ask_question(api_url: str, payload: dict) -> dict:
    timeout = httpx.Timeout(240.0, connect=10.0)
    with httpx.Client(timeout=timeout) as client:
        response = client.post(f"{api_url}/ask", json=payload)
        response.raise_for_status()
        return response.json()


def upload_document(api_url: str, session_id: str, uploaded_file) -> dict:
    payload = {
        "session_id": session_id,
        "filename": uploaded_file.name,
        "content_type": uploaded_file.type or "application/octet-stream",
        "file_base64": base64.b64encode(uploaded_file.getvalue()).decode("utf-8"),
    }
    timeout = httpx.Timeout(240.0, connect=10.0)
    with httpx.Client(timeout=timeout) as client:
        response = client.post(f"{api_url}/session-documents", json=payload)
        response.raise_for_status()
        return response.json()


def remove_document(api_url: str, session_id: str) -> None:
    with httpx.Client(timeout=httpx.Timeout(30.0, connect=5.0)) as client:
        response = client.delete(f"{api_url}/session-documents/{session_id}")
        response.raise_for_status()


def chat_history() -> list[dict[str, str]]:
    return [
        {"role": message["role"], "content": message["content"]}
        for message in st.session_state.messages[-8:]
    ]


def build_question_payload(question: str) -> dict:
    document_active = st.session_state.document is not None
    return {
        "session_id": st.session_state.session_id,
        "question": question.strip(),
        "scope": "corpus",
        "source_mode": "document_only" if document_active else "document_plus_case",
        "retrieval_profile": "fast",
        "scope_case_ids": [],
        "chat_history": chat_history(),
        "top_k": 1 if document_active else 3,
    }


def deduplicate_sources(items: list[dict], *, is_case: bool) -> list[dict]:
    unique: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        if is_case:
            key = (str(item.get("case_id") or ""), str(item.get("title") or ""))
        else:
            key = (str(item.get("title") or ""), str(item.get("section_ref") or ""))
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def source_heading(item: dict, *, is_case: bool) -> str:
    if is_case:
        return item.get("title") or item.get("case_id") or "Retrieved case"

    title = item.get("title") or "Reference material"
    section = item.get("section_ref")
    page_start = item.get("page_start")
    page_end = item.get("page_end")
    details = []
    if section:
        details.append(str(section))
    if page_start:
        page_text = f"page {page_start}"
        try:
            start = int(page_start)
            end = int(page_end) if page_end else start
            if start < end <= start + 5:
                page_text += f"-{end}"
        except (TypeError, ValueError):
            pass
        details.append(page_text)
    return f"{title} ({', '.join(details)})" if details else title


def source_citation(item: dict, *, is_case: bool) -> str:
    if is_case:
        return item.get("title") or item.get("case_id") or "Retrieved case"
    title = item.get("title") or "Reference material"
    section = item.get("section_ref")
    return f"{title} - {section}" if section else title


def render_evidence_item(item: dict, *, is_case: bool, index: int) -> None:
    st.markdown(f"**{index}. {source_heading(item, is_case=is_case)}**")
    if is_case:
        metadata = [item.get("court"), item.get("date"), item.get("case_id")]
        excerpt = item.get("excerpt") or item.get("summary")
    else:
        metadata = [item.get("authority_type"), item.get("domain")]
        excerpt = item.get("excerpt")
    metadata = [str(value) for value in metadata if value]
    if metadata:
        st.caption(" | ".join(metadata))
    st.write(excerpt or "No excerpt was returned.")


def render_sources(response: dict) -> None:
    references = deduplicate_sources(response.get("reference_materials") or [], is_case=False)
    cases = deduplicate_sources(response.get("supporting_cases") or [], is_case=True)
    if not references and not cases:
        return

    primary = references[0] if references else cases[0]
    st.caption(f"Source: {source_citation(primary, is_case=not bool(references))}")
    with st.expander("View retrieved evidence"):
        evidence = [(item, False) for item in references] + [(item, True) for item in cases]
        for index, (item, is_case) in enumerate(evidence, start=1):
            render_evidence_item(item, is_case=is_case, index=index)
            if index < len(evidence):
                st.divider()


def visible_advisories(response: dict) -> list[str]:
    if response.get("evidence_strength") not in {"mixed", "insufficient"} and response.get(
        "retrieval_confidence"
    ) != "low":
        return []

    hidden_prefixes = (
        "This answer relied on",
        "This answer is statute-first",
        "Official law materials were used",
    )
    visible: list[str] = []
    for advisory in response.get("advisories") or []:
        if advisory.startswith(hidden_prefixes) or advisory in visible:
            continue
        visible.append(advisory)
    return visible[:2]


def render_answer(response: dict) -> None:
    st.markdown(response.get("answer") or "The backend returned an empty answer.")
    render_sources(response)

    advisories = visible_advisories(response)
    if advisories:
        with st.expander("Limitations"):
            for advisory in advisories:
                st.markdown(f"- {advisory}")


def reset_conversation() -> None:
    st.session_state.messages = []


for key, default in {
    "session_id": uuid.uuid4().hex,
    "messages": [],
    "document": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


st.title("Indian Legal Research & Retrieval System")
st.caption(
    "Ask questions about Indian law and retrieved judgments, or attach one legal document for focused Q&A. "
    "Legal information only, not legal advice."
)

with st.sidebar:
    st.header("Workspace")
    api_url = st.text_input("Backend URL", value=get_default_api_url()).rstrip("/")

    if st.button("New conversation", use_container_width=True):
        reset_conversation()
        st.rerun()

    st.divider()
    st.subheader("Document")
    selected_file = st.file_uploader(
        "Attach a PDF, DOCX, TXT, or Markdown file",
        type=["pdf", "docx", "txt", "md"],
    )
    if st.button(
        "Attach document",
        disabled=selected_file is None,
        use_container_width=True,
    ):
        try:
            with st.spinner("Reading document..."):
                st.session_state.document = upload_document(
                    api_url,
                    st.session_state.session_id,
                    selected_file,
                )
            st.rerun()
        except Exception as exc:
            st.error(error_detail(exc))

    document = st.session_state.document
    if document:
        st.success(f"Attached: {document['filename']}")
        st.caption(
            f"{document.get('word_count', 0):,} words in "
            f"{document.get('chunk_count', 0)} searchable chunks."
        )
        st.caption("Questions now use only this document.")
        if st.button("Remove document", use_container_width=True):
            try:
                remove_document(api_url, st.session_state.session_id)
                st.session_state.document = None
                st.rerun()
            except Exception as exc:
                st.error(error_detail(exc))

    st.divider()
    if st.button("Refresh backend status", use_container_width=True):
        get_health.clear()


health = get_health(api_url)
if health is None:
    st.error(f"Backend unavailable at {api_url}. Start FastAPI and refresh the status.")
elif health.get("llm_ready") and (
    health.get("qa_retrieval_ready") or health.get("reference_law_ready")
):
    st.success("Backend, legal sources, and language model are ready.")
else:
    st.warning("Backend is connected, but one or more retrieval or model services are not ready.")

if not st.session_state.messages:
    st.info(
        "Try: What remedy is available when an RTI request is unanswered?\n\n"
        "Or attach a document and ask: Summarize the main issue in this document."
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and message.get("response"):
            render_answer(message["response"])
        else:
            st.markdown(message["content"])


question = st.chat_input(
    "Ask a legal question...",
    disabled=health is None,
)

if question:
    minimum_length = 4 if st.session_state.document else (8 if st.session_state.messages else 12)
    if len(question.strip()) < minimum_length:
        st.warning("Please ask a fuller legal question so retrieval has enough context.")
    else:
        payload = build_question_payload(question)
        st.session_state.messages.append({"role": "user", "content": question.strip()})
        with st.chat_message("user"):
            st.markdown(question.strip())

        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching legal sources and preparing the answer..."):
                    response = ask_question(api_url, payload)
                answer = response.get("answer") or "The backend returned an empty answer."
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "response": response,
                    }
                )
                render_answer(response)
            except Exception as exc:
                st.error(error_detail(exc))

st.caption("Research prototype for legal information. Verify important points from the cited source.")
