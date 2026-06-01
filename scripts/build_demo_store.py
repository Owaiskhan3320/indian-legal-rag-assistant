from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any


os.environ["DEMO_MODE"] = "true"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from legal_ai.config import get_settings  # noqa: E402
from legal_ai.logging_utils import configure_logging  # noqa: E402
from legal_ai.services.qa_retriever import LegalQARetriever  # noqa: E402
from legal_ai.services.reference_law import ReferenceLawRetriever  # noqa: E402
from legal_ai.services.retriever import SimilarCaseRetriever  # noqa: E402
from legal_ai.utils.text import normalize_whitespace, shorten_text, split_into_word_chunks  # noqa: E402


SAMPLE_DIR = PROJECT_ROOT / "sample_data"
DEMO_DIR = PROJECT_ROOT / "artifacts" / "demo"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            cleaned = line.strip()
            if not cleaned:
                continue
            try:
                records.append(json.loads(cleaned))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}") from exc
    return records


def _slug(value: str) -> str:
    cleaned = normalize_whitespace(value).lower()
    return "".join(char if char.isalnum() else "_" for char in cleaned).strip("_")


def _aliases_for_title(title: str) -> list[str]:
    lowered = title.lower()
    aliases = [lowered]
    if "constitution" in lowered:
        aliases.extend(["constitution", "constitution of india"])
    if "right to information" in lowered:
        aliases.extend(["rti", "rti act", "right to information act"])
    if "consumer protection" in lowered:
        aliases.extend(["consumer act", "consumer protection act"])
    if "ccs" in lowered:
        aliases.extend(["ccs cca rules", "disciplinary rules"])
    unique: list[str] = []
    for alias in aliases:
        cleaned = normalize_whitespace(alias).lower()
        if cleaned and cleaned not in unique:
            unique.append(cleaned)
    return unique


def _build_reference_records(raw_records: list[dict[str, Any]], settings) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row_id, item in enumerate(raw_records):
        title = normalize_whitespace(item["title"])
        section_ref = normalize_whitespace(item["section_ref"])
        section_title = normalize_whitespace(item.get("section_title") or "")
        text = normalize_whitespace(item["text"])
        aliases = _aliases_for_title(title)
        authority_type = normalize_whitespace(item.get("authority_type") or "act")
        domain = normalize_whitespace(item.get("domain") or "general")
        retrieval_text = normalize_whitespace(
            " ".join(
                part
                for part in [
                    title,
                    " ".join(aliases),
                    authority_type,
                    domain,
                    section_ref,
                    section_title,
                    text,
                ]
                if part
            )
        )
        doc_id = _slug(title)
        parent_id = f"{doc_id}:{_slug(section_ref)}"
        records.append(
            {
                "row_id": row_id,
                "doc_id": doc_id,
                "title": title,
                "title_norm": title.lower(),
                "aliases_text": " | ".join(aliases),
                "authority_type": authority_type,
                "domain": domain,
                "source_path": f"sample_data/reference_law_demo.jsonl#{item.get('id') or row_id}",
                "page_start": None,
                "page_end": None,
                "parent_id": parent_id,
                "section_ref": section_ref,
                "section_ref_norm": section_ref.lower(),
                "section_title": section_title or None,
                "child_ref": None,
                "child_ref_norm": "",
                "retrieval_text": retrieval_text,
                "preview_text": shorten_text(text, settings.qa_retrieval_preview_char_limit),
                "child_text": text,
                "parent_text": text,
            }
        )
    return records


def _build_case_records(raw_records: list[dict[str, Any]], settings) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in raw_records:
        parts = [
            item.get("title"),
            item.get("court"),
            item.get("case_type"),
            item.get("facts"),
            item.get("issue"),
            item.get("holding"),
            item.get("relief"),
            item.get("text"),
        ]
        full_text = normalize_whitespace(" ".join(str(part) for part in parts if part))
        retrieval_text = shorten_text(full_text, settings.retrieval_char_limit)
        preview_text = shorten_text(
            normalize_whitespace(
                " ".join(
                    str(part)
                    for part in [item.get("facts"), item.get("issue"), item.get("holding"), item.get("relief")]
                    if part
                )
            ),
            settings.retrieval_preview_char_limit,
        )
        records.append(
            {
                "case_id": normalize_whitespace(item["case_id"]),
                "label": item.get("label"),
                "title": normalize_whitespace(item.get("title")),
                "court": normalize_whitespace(item.get("court")),
                "case_type": normalize_whitespace(item.get("case_type")),
                "date": normalize_whitespace(item.get("date")),
                "retrieval_text": retrieval_text,
                "preview_text": preview_text,
                "full_text": full_text,
            }
        )
    return records


def _build_qa_records(case_records: list[dict[str, Any]], settings) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for case_record in case_records:
        chunks = split_into_word_chunks(
            case_record["full_text"],
            chunk_words=settings.qa_chunk_words,
            overlap_words=settings.qa_chunk_overlap_words,
            min_words=settings.qa_chunk_min_words,
        ) or [case_record["full_text"]]
        chunk_count = len(chunks)
        for chunk_order, chunk_text in enumerate(chunks):
            retrieval_context = normalize_whitespace(
                " ".join(
                    part
                    for part in [
                        case_record.get("title"),
                        case_record.get("court"),
                        case_record.get("case_type"),
                        chunk_text,
                    ]
                    if part
                )
            )
            records.append(
                {
                    "case_id": case_record["case_id"],
                    "label": case_record.get("label"),
                    "title": case_record.get("title"),
                    "court": case_record.get("court"),
                    "case_type": case_record.get("case_type"),
                    "date": case_record.get("date"),
                    "chunk_order": chunk_order,
                    "chunk_count": chunk_count,
                    "retrieval_text": retrieval_context,
                    "preview_text": shorten_text(chunk_text, settings.qa_retrieval_preview_char_limit),
                    "chunk_text": chunk_text,
                }
            )
    return records


def _assert_demo_paths(settings) -> None:
    demo_paths = [
        settings.retrieval_index_path,
        settings.retrieval_metadata_path,
        settings.qa_retrieval_index_path,
        settings.qa_retrieval_metadata_path,
        settings.qa_retrieval_embedding_store_path,
        settings.reference_law_index_path,
        settings.reference_law_metadata_path,
    ]
    unsafe = [path for path in demo_paths if "artifacts/demo/" not in path.replace("\\", "/")]
    if unsafe:
        raise RuntimeError(f"Refusing to build demo store outside artifacts/demo: {unsafe}")


def main() -> None:
    get_settings.cache_clear()
    settings = get_settings()
    configure_logging(settings.log_level)
    if not settings.demo_mode:
        raise RuntimeError("DEMO_MODE must be true to build the public sample store.")
    _assert_demo_paths(settings)

    if DEMO_DIR.exists():
        shutil.rmtree(DEMO_DIR)
    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    reference_raw = _load_jsonl(SAMPLE_DIR / "reference_law_demo.jsonl")
    case_raw = _load_jsonl(SAMPLE_DIR / "case_law_demo.jsonl")
    reference_records = _build_reference_records(reference_raw, settings)
    case_records = _build_case_records(case_raw, settings)
    qa_records = _build_qa_records(case_records, settings)

    reference_retriever = ReferenceLawRetriever(settings)
    reference_retriever.build(reference_records)

    case_retriever = SimilarCaseRetriever(settings)
    case_retriever.build(case_records)
    case_retriever.save()

    qa_retriever = LegalQARetriever(settings)
    qa_retriever.build(qa_records)

    manifest = {
        "mode": "demo",
        "description": "Small public sample store for reproducible portfolio/demo runs.",
        "reference_law_records": len(reference_records),
        "case_records": len(case_records),
        "qa_chunk_records": len(qa_records),
        "embedding_model": settings.shared_embedding_model_name,
        "note": "Demo case records are short sample summaries for pipeline verification, not redistributed benchmark data.",
    }
    (DEMO_DIR / "demo_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Built demo retrieval store")
    print(f"Reference-law records: {len(reference_records)}")
    print(f"Case records: {len(case_records)}")
    print(f"QA chunks: {len(qa_records)}")
    print(f"Artifacts: {DEMO_DIR}")


if __name__ == "__main__":
    main()
