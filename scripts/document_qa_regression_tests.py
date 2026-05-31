from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Project\src")

from legal_ai.config import get_settings
from legal_ai.services.session_documents import SessionDocumentStore
import numpy as np


DOCUMENT_QA_TESTS = [
    {
        "question": "why was the passport impounded",
        "must_include": ["public interest"],
        "must_not_include": ["petitioner:", "respondent:"],
    },
    {
        "question": "who won the case",
        "must_include": ["partially succeeded", "did not order immediate return"],
        "must_not_include": ["supreme court of india", "petitioner:"],
    },
    {
        "question": "what punishment was given to passport authority",
        "must_include": ["does not mention any punishment"],
        "must_not_include": ["article 21", "public interest"],
    },
    {
        "question": "under which law was her passport impounded",
        "must_include": ["section 10(3)(c)", "passport act"],
        "must_not_include": ["supreme court of india", "partially succeeded"],
    },
    {
        "question": "did the court allow the passport to be returned immediately",
        "must_include": ["no.", "remain with the authorities"],
        "must_not_include": ["public interest", "petitioner:"],
    },
    {
        "question": "how is artical 21 related to this case",
        "follow_up_context": "which court was involved",
        "must_include": ["article 21"],
        "must_not_include": ["supreme court of india"],
    },
    {
        "question": "what was the judgment",
        "follow_up_context": "what was this case about",
        "must_include": ["held", "article 14"],
        "must_not_include": ["was a case about"],
    },
    {
        "question": "what",
        "follow_up_context": "how is artical 21 related to this case",
        "must_include": ["could not understand", "recognizable words"],
        "must_not_include": ["ratio decidendi", "article 21"],
    },
    {
        "question": "what is the ratio decidendi and what is the obiter dicta",
        "must_include": ["1. answer:", "2. answer:", "where found:", "reliability:"],
        "must_not_include": ["ratio decidendi is commonly defined", "obiter dicta is commonly defined"],
    },
    {
        "question": "explain this case in simple language",
        "must_include": ["passport", "unfair", "article 21"],
        "must_not_include": ["ratio decidendi is commonly defined", "before stating the ratio"],
    },
    {
        "question": "summarize this case in 5 lines",
        "must_include": ["1.", "2.", "3.", "4.", "5."],
        "must_not_include": ["ratio decidendi is commonly defined", "picture credits"],
    },
]


class DummyEncoder:
    def encode_texts(self, texts, **kwargs):
        return np.zeros((len(texts), 8), dtype=float)

    def encode_query(self, text):
        return np.zeros(8, dtype=float)


def run_document_qa_regressions(pdf_path: Path) -> int:
    store = SessionDocumentStore(get_settings())
    encoder = DummyEncoder()
    file_bytes = pdf_path.read_bytes()
    store.upsert(
        session_id="regression-session",
        filename=pdf_path.name,
        content_type="application/pdf",
        file_bytes=file_bytes,
        encoder=encoder,
    )

    failures: list[str] = []
    for spec in DOCUMENT_QA_TESTS:
        question = spec["question"]
        follow_up_context = spec.get("follow_up_context")
        answer = store.answer_question(
            session_id="regression-session",
            question=question,
            question_profile={
                "answer_style": "simple",
                "response_length": "short",
                "rewritten_question": question,
            },
            follow_up_context=follow_up_context,
            encoder=encoder,
        )
        answer_text = (answer or {}).get("text", "").lower()
        for needle in spec["must_include"]:
            if needle.lower() not in answer_text:
                failures.append(f"{question!r} missing required text: {needle!r}")
        for needle in spec["must_not_include"]:
            if needle.lower() in answer_text:
                failures.append(f"{question!r} unexpectedly included: {needle!r}")

        for bad_token in ["â€™", "â€œ", "â€", "Â·", "ï¬", "ï¬‚", "undeï", "afï"]:
            if bad_token.lower() in answer_text:
                failures.append(f"{question!r} still contains mojibake/ligature junk: {bad_token!r}")

        if question == "summarize this case in 5 lines":
            line_count = sum(
                1
                for line in (answer or {}).get("answer_body", "").splitlines()
                if line.strip().startswith(tuple(str(i) + "." for i in range(1, 6)))
            )
            if line_count < 5:
                failures.append(f"{question!r} did not produce 5 numbered summary lines")

    if failures:
        print("document qa regression failures:")
        for item in failures:
            print(f"- {item}")
        return 1

    print("document qa regression checks passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run regression checks for uploaded-document Q/A.")
    parser.add_argument("pdf_path", type=Path, help="Path to the legal PDF used for regression checks.")
    args = parser.parse_args()
    return run_document_qa_regressions(args.pdf_path)


if __name__ == "__main__":
    raise SystemExit(main())
