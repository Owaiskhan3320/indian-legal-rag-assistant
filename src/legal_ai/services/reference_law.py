from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from legal_ai.config import Settings
from legal_ai.services.session_documents import SessionDocumentStore
from legal_ai.utils.text import (
    lexical_overlap_score,
    normalize_whitespace,
    search_terms,
    shorten_text,
    split_into_word_chunks,
)


LOGGER = logging.getLogger(__name__)

ARTICLE_RE = re.compile(r"\bArticle\s+\d+[A-Za-z0-9()/-]*", re.I)
SECTION_RE = re.compile(r"\bSection\s+\d+[A-Za-z0-9()/-]*", re.I)
RULE_RE = re.compile(r"\bRule\s+\d+[A-Za-z0-9()/-]*", re.I)
CLAUSE_BOUNDARY_RE = re.compile(
    r"(?=(?:\(\d+[A-Za-z]*\)|\([a-z]\)|Explanation\.?|Provided that|Proviso\.?))",
    re.I,
)

DOMAIN_HINTS: list[tuple[str, str]] = [
    ("consumer", "consumer"),
    ("rti", "information"),
    ("information", "information"),
    ("administrative", "service"),
    ("cca", "service"),
    ("conduct", "service"),
    ("pension", "service"),
    ("income tax", "tax"),
    ("gst", "tax"),
    ("excise", "tax"),
    ("motor", "motor_accident"),
    ("vehicle", "motor_accident"),
    ("constitution", "constitutional"),
    ("nyaya sanhita", "criminal"),
    ("suraksha sanhita", "criminal"),
    ("sakshya adhiniyam", "criminal"),
]

TITLE_ALIASES: dict[str, list[str]] = {
    "Right to Information Act, 2005": ["rti act", "right to information act", "rti"],
    "Right to Information Rules, 2019": ["rti rules", "right to information rules"],
    "Consumer Protection Act, 2019": ["consumer protection act", "consumer act"],
    "Consumer Protection E-Commerce Rules, 2020": [
        "e commerce rules",
        "e-commerce rules",
        "consumer e commerce rules",
    ],
    "Consumer Protection Commission and General Rules, 2020": [
        "consumer commission rules",
        "consumer disputes redressal commissions rules",
        "consumer general rules",
    ],
    "Administrative Tribunals Act, 1985": [
        "administrative tribunals act",
        "tribunals act",
        "cat act",
    ],
    "CCS (CCA) Rules": [
        "ccs cca rules",
        "classification control and appeal rules",
        "disciplinary rules",
    ],
    "CCS Conduct Rules": ["ccs conduct rules", "conduct rules"],
    "Central Goods and Services Tax Act, 2017": ["cgst act", "gst act", "central goods and services tax act"],
    "Central Excise Act, 1944": ["central excise act", "excise act"],
    "Motor Vehicles Act, 1988": ["motor vehicles act", "mv act"],
    "Central Motor Vehicles Rules, 1989": ["central motor vehicles rules", "cmvr"],
    "Income-tax Act, 1961": ["income tax act", "income-tax act", "ita"],
    "Constitution of India": ["constitution", "constitution of india"],
    "Bharatiya Nyaya Sanhita, 2023": ["bns", "bharatiya nyaya sanhita"],
    "Bharatiya Nagarik Suraksha Sanhita, 2023": ["bnss", "bharatiya nagarik suraksha sanhita"],
    "Bharatiya Sakshya Adhiniyam, 2023": ["bsa", "bharatiya sakshya adhiniyam"],
}


class ReferenceLawRetriever:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        LOGGER.info(
            "Loading reference-law embedding model=%s on device=%s",
            settings.shared_embedding_model_name,
            self.device,
        )
        self.model = SentenceTransformer(settings.shared_embedding_model_name, device=self.device)
        self.index: faiss.Index | None = None
        self.record_count = 0
        self._document_parser = SessionDocumentStore(settings)

    def encode_texts(
        self,
        texts: list[str],
        *,
        is_query: bool,
        batch_size: int | None = None,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        prepared = (
            [self._prepare_query_text(text) for text in texts]
            if is_query
            else [self._prepare_document_text(text) for text in texts]
        )
        if not prepared:
            dimension = self.model.get_sentence_embedding_dimension()
            return np.empty((0, dimension), dtype="float32")
        embeddings = self.model.encode(
            prepared,
            batch_size=batch_size or self.settings.retrieval_embedding_batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.ascontiguousarray(embeddings.astype("float32"))

    def encode_query(self, text: str) -> np.ndarray:
        matrix = self.encode_texts([normalize_whitespace(text)], is_query=True, batch_size=1)
        return matrix[0]

    def build_records_from_directory(self, source_dir: Path) -> list[dict[str, Any]]:
        pdf_paths = sorted(path for path in source_dir.rglob("*.pdf") if path.is_file())
        records: list[dict[str, Any]] = []
        for pdf_path in pdf_paths:
            records.extend(self._build_records_for_pdf(pdf_path))
        for row_id, record in enumerate(records):
            record["row_id"] = row_id
        return records

    def build(self, records: list[dict[str, Any]]) -> None:
        texts = [record["retrieval_text"] for record in records]
        embeddings = self.encode_texts(
            texts,
            is_query=False,
            batch_size=max(self.settings.retrieval_embedding_batch_size, 96),
            show_progress_bar=True,
        )
        if embeddings.shape[0] == 0:
            raise RuntimeError("No reference-law retrieval records were generated from the supplied PDFs.")
        index = faiss.IndexFlatIP(int(embeddings.shape[1]))
        index.add(embeddings)
        self.index = index
        self.record_count = len(records)
        metadata_path = self.settings.resolve_path(self.settings.reference_law_metadata_path)
        index_path = self.settings.resolve_path(self.settings.reference_law_index_path)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_metadata_database(metadata_path, records)
        faiss.write_index(index, str(index_path))

    def load(self) -> None:
        index_path = self.settings.resolve_path(self.settings.reference_law_index_path)
        metadata_path = self.settings.resolve_path(self.settings.reference_law_metadata_path)
        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                "Reference-law artifacts are missing. Build the reference-law store before using statute-aware Q/A."
            )
        self.index = faiss.read_index(str(index_path))
        self.record_count = self._count_rows(metadata_path)
        LOGGER.info(
            "Loaded reference-law retrieval store path=%s records=%s",
            metadata_path,
            self.record_count,
        )

    def search(
        self,
        query: str,
        *,
        top_k: int,
        question_profile: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if self.index is None:
            raise RuntimeError("Reference-law retrieval store is not loaded.")
        if self.record_count == 0:
            return []

        normalized_query = self._normalize_query(query)
        fetch_k = min(
            max(top_k * self.settings.reference_law_overfetch, 96),
            self.record_count,
        )
        query_vector = self.encode_query(normalized_query)
        dense_scores, dense_indices = self.index.search(query_vector.reshape(1, -1), fetch_k)
        dense_row_ids = [int(idx) for idx in dense_indices[0].tolist() if idx >= 0]
        dense_score_map = {
            int(idx): float(dense_scores[0][offset])
            for offset, idx in enumerate(dense_indices[0].tolist())
            if int(idx) >= 0
        }
        dense_rank_map = {row_id: rank for rank, row_id in enumerate(dense_row_ids, start=1)}

        exact_plan = self._build_exact_query_plan(
            query=normalized_query,
            question_profile=question_profile,
        )
        exact_row_ids = self._exact_search_row_ids(
            explicit_refs=exact_plan["refs"],
            title_aliases=exact_plan["title_aliases"],
            domain=exact_plan["domain"],
            limit=fetch_k,
        )
        exact_rank_map = {row_id: rank for rank, row_id in enumerate(exact_row_ids, start=1)}
        title_row_ids = self._title_candidate_row_ids(
            normalized_query,
            limit=min(fetch_k * 4, self.record_count),
        )

        lexical_row_ids = self._lexical_search_row_ids(normalized_query, limit=fetch_k)
        lexical_rank_map = {row_id: rank for rank, row_id in enumerate(lexical_row_ids, start=1)}

        row_ids = list(dict.fromkeys(exact_row_ids + title_row_ids + lexical_row_ids + dense_row_ids))
        rows = self._fetch_rows_by_row_ids(row_ids)
        if not rows:
            return []
        if title_row_ids and not exact_plan["refs"]:
            title_set = set(title_row_ids)
            constrained_rows = [row for row in rows if int(row["row_id"]) in title_set]
            if constrained_rows:
                rows = constrained_rows

        aggregated = self._aggregate_parent_hits(
            query=normalized_query,
            rows=rows,
            dense_score_map=dense_score_map,
            dense_rank_map=dense_rank_map,
            lexical_rank_map=lexical_rank_map,
            exact_rank_map=exact_rank_map,
            question_profile=question_profile,
        )
        return aggregated[:top_k]

    def build_context(self, hits: list[dict[str, Any]]) -> dict[str, Any]:
        if not hits:
            return {
                "used": False,
                "materials": [],
                "coverage_note": "No official law materials were added for this turn.",
                "context_text": "",
                "best_match_type": "none",
                "retrieval_confidence": "low",
            }
        blocks: list[str] = []
        materials: list[dict[str, Any]] = []
        for index, hit in enumerate(hits[: self.settings.reference_law_max_hits], start=1):
            materials.append(
                {
                    "title": hit["title"],
                    "section_ref": hit.get("section_ref"),
                    "authority_type": hit.get("authority_type"),
                    "domain": hit.get("domain"),
                    "page_start": hit.get("page_start"),
                    "page_end": hit.get("page_end"),
                    "retrieval_strategy": hit.get("retrieval_strategy") or "hybrid",
                    "retrieval_confidence": hit.get("retrieval_confidence") or "moderate",
                    "excerpt": hit.get("excerpt") or "",
                }
            )
            blocks.append(
                "\n".join(
                    [
                        f"Official law material {index}",
                        f"- Title: {hit['title']}",
                        f"- Section: {hit.get('section_ref') or hit.get('child_ref') or 'Not specified'}",
                        f"- Authority type: {hit.get('authority_type') or 'act'}",
                        f"- Domain: {hit.get('domain') or 'general'}",
                        f"- Retrieval strategy: {hit.get('retrieval_strategy') or 'hybrid'}",
                        f"- Passage: {shorten_text(hit.get('excerpt') or '', 380)}",
                    ]
                )
            )
        return {
            "used": True,
            "materials": materials,
            "coverage_note": f"Prepared {len(materials)} official law material blocks from the reference-law lane.",
            "context_text": "\n\n".join(blocks),
            "best_match_type": hits[0].get("match_type") or "semantic",
            "retrieval_confidence": hits[0].get("retrieval_confidence") or "moderate",
        }

    def _prepare_query_text(self, text: str) -> str:
        cleaned = normalize_whitespace(text)
        instruction = normalize_whitespace(self.settings.shared_embedding_query_prefix)
        return f"{instruction} {cleaned}".strip() if instruction else cleaned

    def _prepare_document_text(self, text: str) -> str:
        cleaned = normalize_whitespace(text)
        instruction = normalize_whitespace(self.settings.shared_embedding_passage_prefix)
        return f"{instruction} {cleaned}".strip() if instruction else cleaned

    def _build_records_for_pdf(self, pdf_path: Path) -> list[dict[str, Any]]:
        extracted = self._document_parser._extract_document_parts(  # noqa: SLF001
            filename=pdf_path.name,
            content_type="application/pdf",
            file_bytes=pdf_path.read_bytes(),
        )
        cleaned_text = self._document_parser._clean_document_text(extracted["clean_text"])  # noqa: SLF001
        pages = [
            {
                "page_number": int(page.get("page_number") or 1),
                "text": self._document_parser._clean_document_text(page.get("text") or ""),  # noqa: SLF001
            }
            for page in extracted.get("pages") or []
        ]
        title = self._canonical_title(pdf_path.name)
        authority_type = self._infer_authority_type(title)
        domain = self._infer_domain(title)
        doc_id = hashlib.sha1(str(pdf_path).encode("utf-8")).hexdigest()[:16]
        aliases = self._aliases_for_title(title)
        parents = self._extract_parent_sections(cleaned_text, authority_type=authority_type)
        if not parents:
            parents = self._fallback_parent_sections(cleaned_text)

        records: list[dict[str, Any]] = []
        row_id = 0
        for parent_index, parent in enumerate(parents, start=1):
            parent_text = normalize_whitespace(parent.get("text") or "")
            if len(parent_text) < 40:
                continue
            page_start, page_end = self._estimate_page_range(
                pages=pages,
                section_ref=parent.get("section_ref") or "",
                parent_text=parent_text,
            )
            children = self._split_parent_into_children(parent_text)
            for child_index, child in enumerate(children, start=1):
                child_text = normalize_whitespace(child.get("text") or "")
                if len(child_text) < 30:
                    continue
                child_ref = normalize_whitespace(child.get("child_ref") or "")
                section_ref = normalize_whitespace(parent.get("section_ref") or "")
                section_title = normalize_whitespace(parent.get("section_title") or "")
                retrieval_text = normalize_whitespace(
                    " ".join(
                        part
                        for part in [
                            title,
                            authority_type,
                            domain,
                            " ".join(aliases),
                            section_ref,
                            section_title,
                            child_ref,
                            child_text,
                        ]
                        if part
                    )
                )
                records.append(
                    {
                        "row_id": row_id,
                        "doc_id": doc_id,
                        "title": title,
                        "title_norm": title.lower(),
                        "aliases_text": " | ".join(aliases),
                        "authority_type": authority_type,
                        "domain": domain,
                        "source_path": str(pdf_path),
                        "page_start": page_start,
                        "page_end": page_end,
                        "parent_id": f"{doc_id}:{parent_index}",
                        "section_ref": section_ref or None,
                        "section_ref_norm": section_ref.lower() if section_ref else "",
                        "section_title": section_title or None,
                        "child_ref": child_ref or None,
                        "child_ref_norm": child_ref.lower() if child_ref else "",
                        "retrieval_text": retrieval_text,
                        "preview_text": shorten_text(child_text, self.settings.qa_retrieval_preview_char_limit),
                        "child_text": child_text,
                        "parent_text": parent_text,
                    }
                )
                row_id += 1
        return records

    @staticmethod
    def _canonical_title(filename: str) -> str:
        cleaned = Path(filename).stem.replace("_", " ").replace("  ", " ").strip()
        title_map = {
            "29-05-27 Constitution English Final": "Constitution of India",
            "Administrative Tribunals Act, 1985": "Administrative Tribunals Act, 1985",
            "Bharatiya Nagarik Suraksha Sanhita 2023": "Bharatiya Nagarik Suraksha Sanhita, 2023",
            "Bharatiya Nyaya Sanhita 2023": "Bharatiya Nyaya Sanhita, 2023",
            "Bharatiya Sakshya Adhiniyam 2023": "Bharatiya Sakshya Adhiniyam, 2023",
            "CCS-CCA-Rules-FINAL": "CCS (CCA) Rules",
            "CCS Conduct Rules 1964 Updated 27Feb15 0": "CCS Conduct Rules",
            "Central Goods and Services Tax Act, 2017": "Central Goods and Services Tax Act, 2017",
            "Central Motor Vehicles Rules, 1989": "Central Motor Vehicles Rules, 1989",
            "Central Excise Act 1944": "Central Excise Act, 1944",
            "consumer protection act 2019": "Consumer Protection Act, 2019",
            "E commerce rules": "Consumer Protection E-Commerce Rules, 2020",
            "Income Tax Act 2025 as amended by FA Act 2026": "Income-tax Act, 1961",
            "Motor Vehicles Act, 1988": "Motor Vehicles Act, 1988",
            "RTI Act 2005 (updated as on 18-11-2025)": "Right to Information Act, 2005",
            "RTI Rules 2019": "Right to Information Rules, 2019",
            "The Consumer Protection (Consumer Disputes Redressal Commissions) Rules, 2020 & The Consumer Protection (General) Rules, 2020": (
                "Consumer Protection Commission and General Rules, 2020"
            ),
        }
        return title_map.get(cleaned, cleaned)

    @staticmethod
    def _infer_authority_type(title: str) -> str:
        lowered = title.lower()
        if "constitution" in lowered:
            return "constitution"
        if "rules" in lowered or "rule" in lowered:
            return "rules"
        if "guideline" in lowered:
            return "guideline"
        if "sanhita" in lowered or "adhiniyam" in lowered:
            return "code"
        return "act"

    @staticmethod
    def _infer_domain(title: str) -> str:
        lowered = title.lower()
        for marker, domain in DOMAIN_HINTS:
            if marker in lowered:
                return domain
        return "general"

    @staticmethod
    def _aliases_for_title(title: str) -> list[str]:
        aliases = TITLE_ALIASES.get(title, [])
        all_aliases = [title.lower(), *aliases]
        unique: list[str] = []
        for alias in all_aliases:
            cleaned = normalize_whitespace(alias).lower()
            if cleaned and cleaned not in unique:
                unique.append(cleaned)
        return unique

    def _extract_parent_sections(self, text: str, *, authority_type: str) -> list[dict[str, str]]:
        raw_text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
        if not normalize_whitespace(raw_text):
            return []
        lines = [line.rstrip() for line in raw_text.split("\n")]
        if authority_type == "constitution":
            heading_pattern = re.compile(r"^(?:Article\s+)?\d+[A-Za-z]?(?:\([^)]+\))*[.)]?\s+.+", re.I)
            ref_pattern = ARTICLE_RE
        elif authority_type == "rules":
            heading_pattern = re.compile(r"^(?:Rule\s+)?\d+[A-Za-z]?(?:\([^)]+\))*[.)]?\s+.+", re.I)
            ref_pattern = RULE_RE
        else:
            heading_pattern = re.compile(r"^(?:Section\s+)?\d+[A-Za-z]?(?:\([^)]+\))*[.)]?\s+.+", re.I)
            ref_pattern = SECTION_RE

        heading_indices = [index for index, line in enumerate(lines) if heading_pattern.match(line.strip())]
        if not heading_indices:
            return []
        sections: list[dict[str, str]] = []
        for position, start_index in enumerate(heading_indices):
            end_index = heading_indices[position + 1] if position + 1 < len(heading_indices) else len(lines)
            body_lines = [line.strip() for line in lines[start_index:end_index] if normalize_whitespace(line)]
            body = "\n".join(body_lines).strip()
            header = normalize_whitespace(body_lines[0]) if body_lines else ""
            ref_match = ref_pattern.search(header)
            if not ref_match:
                generic_ref = re.match(r"^\d+[A-Za-z]?(?:\([^)]+\))*", header)
                section_ref = normalize_whitespace(generic_ref.group(0)) if generic_ref else header
            else:
                section_ref = normalize_whitespace(ref_match.group(0))
            title_tail = header[len(section_ref) :].strip(" .:-") if section_ref and header.lower().startswith(section_ref.lower()) else header
            title = shorten_text(title_tail or header, 140)
            sections.append(
                {
                    "section_ref": section_ref,
                    "section_title": title,
                    "text": body,
                }
            )
        return sections

    def _fallback_parent_sections(self, text: str) -> list[dict[str, str]]:
        cleaned = normalize_whitespace(text)
        sections = split_into_word_chunks(
            cleaned,
            chunk_words=max(self.settings.retrieval_chunk_words, 260),
            overlap_words=max(self.settings.retrieval_chunk_overlap_words, 30),
            min_words=max(self.settings.retrieval_chunk_min_words, 80),
        )
        parents: list[dict[str, str]] = []
        for index, chunk in enumerate(sections, start=1):
            parents.append(
                {
                    "section_ref": f"Part {index}",
                    "section_title": shorten_text(chunk, 120),
                    "text": chunk,
                }
            )
        return parents

    def _split_parent_into_children(self, parent_text: str) -> list[dict[str, str | None]]:
        word_count = len(normalize_whitespace(parent_text).split())
        if word_count <= 320:
            return [{"child_ref": None, "text": parent_text}]

        parts = [normalize_whitespace(part) for part in CLAUSE_BOUNDARY_RE.split(parent_text) if normalize_whitespace(part)]
        children: list[dict[str, str | None]] = []
        if len(parts) >= 2 and word_count > 420:
            merged_parts: list[str] = []
            buffer = ""
            for part in parts:
                if not buffer:
                    buffer = part
                    continue
                if len(normalize_whitespace(buffer).split()) < 45:
                    buffer = f"{buffer} {part}"
                else:
                    merged_parts.append(buffer)
                    buffer = part
            if buffer:
                merged_parts.append(buffer)
            for index, part in enumerate(merged_parts, start=1):
                ref_match = re.match(r"^(\(\d+[A-Za-z]*\)|\([a-z]\)|Explanation\.?|Provided that|Proviso\.?)", part, flags=re.I)
                children.append(
                    {
                        "child_ref": normalize_whitespace(ref_match.group(1)) if ref_match else f"Clause {index}",
                        "text": part,
                    }
                )
            return children

        chunks = split_into_word_chunks(
            parent_text,
            chunk_words=max(int(self.settings.retrieval_chunk_words * 1.25), 260),
            overlap_words=max(int(self.settings.retrieval_chunk_overlap_words * 0.4), 16),
            min_words=max(int(self.settings.retrieval_chunk_min_words), 70),
        )
        if len(chunks) <= 1:
            return [{"child_ref": None, "text": parent_text}]
        for index, chunk in enumerate(chunks, start=1):
            children.append({"child_ref": f"Paragraph {index}", "text": chunk})
        return children

    @staticmethod
    def _estimate_page_range(
        *,
        pages: list[dict[str, Any]],
        section_ref: str,
        parent_text: str,
    ) -> tuple[int | None, int | None]:
        if not pages:
            return None, None
        ref_lower = normalize_whitespace(section_ref).lower()
        parent_seed = normalize_whitespace(parent_text)[:120].lower()
        matched_pages: list[int] = []
        for page in pages:
            page_text = normalize_whitespace(page.get("text") or "").lower()
            if not page_text:
                continue
            if ref_lower and ref_lower in page_text:
                matched_pages.append(int(page.get("page_number") or 1))
                continue
            if parent_seed and parent_seed[:80] and parent_seed[:80] in page_text:
                matched_pages.append(int(page.get("page_number") or 1))
        if matched_pages:
            return min(matched_pages), max(matched_pages)
        return int(pages[0].get("page_number") or 1), int(pages[0].get("page_number") or 1)

    def _write_metadata_database(self, metadata_path: Path, records: list[dict[str, Any]]) -> None:
        if metadata_path.exists():
            metadata_path.unlink()
        with sqlite3.connect(metadata_path) as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                CREATE TABLE reference_law_records (
                    row_id INTEGER PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    title_norm TEXT NOT NULL,
                    aliases_text TEXT,
                    authority_type TEXT,
                    domain TEXT,
                    source_path TEXT NOT NULL,
                    page_start INTEGER,
                    page_end INTEGER,
                    parent_id TEXT NOT NULL,
                    section_ref TEXT,
                    section_ref_norm TEXT,
                    section_title TEXT,
                    child_ref TEXT,
                    child_ref_norm TEXT,
                    retrieval_text TEXT NOT NULL,
                    preview_text TEXT NOT NULL,
                    child_text TEXT NOT NULL,
                    parent_text TEXT NOT NULL
                )
                """
            )
            cursor.executemany(
                """
                INSERT INTO reference_law_records (
                    row_id,
                    doc_id,
                    title,
                    title_norm,
                    aliases_text,
                    authority_type,
                    domain,
                    source_path,
                    page_start,
                    page_end,
                    parent_id,
                    section_ref,
                    section_ref_norm,
                    section_title,
                    child_ref,
                    child_ref_norm,
                    retrieval_text,
                    preview_text,
                    child_text,
                    parent_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        record["row_id"],
                        record["doc_id"],
                        record["title"],
                        record["title_norm"],
                        record.get("aliases_text"),
                        record.get("authority_type"),
                        record.get("domain"),
                        record.get("source_path"),
                        record.get("page_start"),
                        record.get("page_end"),
                        record["parent_id"],
                        record.get("section_ref"),
                        record.get("section_ref_norm"),
                        record.get("section_title"),
                        record.get("child_ref"),
                        record.get("child_ref_norm"),
                        record["retrieval_text"],
                        record["preview_text"],
                        record["child_text"],
                        record["parent_text"],
                    )
                    for record in records
                ],
            )
            try:
                cursor.execute(
                    """
                    CREATE VIRTUAL TABLE reference_law_records_fts
                    USING fts5(retrieval_text, content='reference_law_records', content_rowid='row_id')
                    """
                )
                cursor.execute(
                    """
                    INSERT INTO reference_law_records_fts(rowid, retrieval_text)
                    SELECT row_id, retrieval_text FROM reference_law_records
                    """
                )
            except sqlite3.OperationalError:
                LOGGER.warning("SQLite FTS5 is not available for reference-law retrieval.")
            connection.commit()

    @staticmethod
    def _count_rows(metadata_path: Path) -> int:
        with sqlite3.connect(metadata_path) as connection:
            row = connection.execute("SELECT COUNT(*) FROM reference_law_records").fetchone()
        return int(row[0]) if row else 0

    @staticmethod
    def _normalize_query(query: str) -> str:
        cleaned = normalize_whitespace(query)
        replacements = {
            "artical": "article",
            "artcle": "article",
            "secton": "section",
            "sectoin": "section",
        }
        lowered = cleaned.lower()
        for wrong, correct in replacements.items():
            lowered = lowered.replace(wrong, correct)
        return normalize_whitespace(lowered)

    def _build_exact_query_plan(
        self,
        *,
        query: str,
        question_profile: dict[str, Any] | None,
    ) -> dict[str, Any]:
        lowered_query = self._normalize_query(query)
        refs = list(self._extract_query_refs(lowered_query))
        title_aliases = list(self._query_aliases(lowered_query))
        domain = normalize_whitespace((question_profile or {}).get("domain") or "").lower()

        def add_ref(value: str) -> None:
            normalized = normalize_whitespace(value).lower()
            if normalized and normalized not in refs:
                refs.append(normalized)

        def add_alias(value: str) -> None:
            normalized = normalize_whitespace(value).lower()
            if normalized and normalized not in title_aliases:
                title_aliases.append(normalized)

        if "article 21a" in lowered_query or "right to education" in lowered_query:
            add_ref("article 21a")
            add_alias("constitution of india")
        if "article 300a" in lowered_query or "right to property" in lowered_query:
            add_ref("article 300a")
            add_alias("constitution of india")
        if "article 21" in lowered_query or "personal liberty" in lowered_query:
            add_ref("article 21")
            add_alias("constitution of india")

        if domain == "information" or "rti" in lowered_query:
            add_alias("right to information act")
            add_alias("rti act")
            if ("reply" in lowered_query or "pio" in lowered_query or "cpio" in lowered_query) and any(
                marker in lowered_query for marker in ("time", "days", "limit", "within")
            ):
                add_ref("section 7")
            if "first appeal" in lowered_query:
                add_ref("section 19")
            if "second appeal" in lowered_query:
                add_ref("section 19")
            if "personal information" in lowered_query:
                add_ref("section 8(1)(j)")
            if any(marker in lowered_query for marker in ("commercial confidence", "trade secret", "intellectual property", "8(1)(d)")):
                add_ref("section 8(1)(d)")
            if any(marker in lowered_query for marker in ("inspect records", "inspection of records", "certified copies", "inspection", "copies")):
                add_ref("section 2(j)")

        if domain == "consumer" or "consumer protection" in lowered_query:
            add_alias("consumer protection act")
            if any(marker in lowered_query for marker in ("remedy", "remedies", "consumer commission grant", "consumer commission can grant", "section 39", "refund", "replacement", "repair", "compensation")):
                add_ref("section 39")
            if "e commerce" in lowered_query or "e-commerce" in lowered_query:
                add_alias("consumer protection e-commerce rules")

        if domain == "service" or "ccs" in lowered_query:
            add_alias("ccs cca rules")
            add_alias("ccs (cca) rules")
            if "rule 14" in lowered_query or "major penalty" in lowered_query or "disciplinary proceedings" in lowered_query:
                add_ref("rule 14")
            if "rule 16" in lowered_query or "minor penalty" in lowered_query:
                add_ref("rule 16")

        return {
            "refs": refs,
            "title_aliases": title_aliases,
            "domain": domain,
        }

    @staticmethod
    def _extract_query_refs(query: str) -> list[str]:
        lowered_query = normalize_whitespace(query).lower()
        return [
            normalize_whitespace(match.group(0)).lower()
            for match in re.finditer(r"\b(?:section|article|rule)\s+\d+[A-Za-z0-9()/-]*", lowered_query, flags=re.I)
        ]

    def _exact_search_row_ids(
        self,
        *,
        explicit_refs: list[str],
        title_aliases: list[str],
        domain: str,
        limit: int,
    ) -> list[int]:
        metadata_path = self.settings.resolve_path(self.settings.reference_law_metadata_path)
        with sqlite3.connect(metadata_path) as connection:
            connection.row_factory = sqlite3.Row
            row_ids: list[int] = []
            for ref in explicit_refs:
                alt_ref = re.sub(r"^(section|article|rule)\s+", "", ref, flags=re.I)
                rows = connection.execute(
                    """
                    SELECT row_id, title_norm, aliases_text, domain
                    FROM reference_law_records
                    WHERE section_ref_norm IN (?, ?) OR child_ref_norm IN (?, ?)
                    LIMIT ?
                    """,
                    (ref, alt_ref, ref, alt_ref, limit),
                ).fetchall()
                preferred: list[int] = []
                fallback: list[int] = []
                for row in rows:
                    row_id = int(row["row_id"])
                    title_norm = normalize_whitespace(row["title_norm"] or "").lower()
                    alias_text = normalize_whitespace(row["aliases_text"] or "").lower()
                    row_domain = normalize_whitespace(row["domain"] or "").lower()
                    title_match = not title_aliases or any(
                        alias == title_norm or alias in alias_text or alias in title_norm
                        for alias in title_aliases
                    )
                    domain_match = not domain or not row_domain or row_domain == domain
                    target = preferred if title_match and domain_match else fallback
                    if row_id not in target and row_id not in row_ids:
                        target.append(row_id)
                for bucket in (preferred, fallback):
                    for row_id in bucket:
                        if row_id not in row_ids:
                            row_ids.append(row_id)
            return row_ids[:limit]

    def _title_candidate_row_ids(self, query: str, *, limit: int) -> list[int]:
        aliases = self._query_aliases(normalize_whitespace(query).lower())
        if not aliases:
            return []
        metadata_path = self.settings.resolve_path(self.settings.reference_law_metadata_path)
        with sqlite3.connect(metadata_path) as connection:
            connection.row_factory = sqlite3.Row
            row_ids: list[int] = []
            for alias in aliases:
                rows = connection.execute(
                    """
                    SELECT row_id
                    FROM reference_law_records
                    WHERE title_norm = ? OR aliases_text LIKE ?
                    LIMIT ?
                    """,
                    (alias, f"%{alias}%", limit),
                ).fetchall()
                for row in rows:
                    row_id = int(row["row_id"])
                    if row_id not in row_ids:
                        row_ids.append(row_id)
            return row_ids[:limit]

    def _lexical_search_row_ids(self, query: str, *, limit: int) -> list[int]:
        metadata_path = self.settings.resolve_path(self.settings.reference_law_metadata_path)
        terms = search_terms(query)
        if not terms:
            return []
        fts_query = " OR ".join(dict.fromkeys(terms[:8]))
        with sqlite3.connect(metadata_path) as connection:
            connection.row_factory = sqlite3.Row
            table_exists = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='reference_law_records_fts'"
            ).fetchone()
            if table_exists is None:
                return []
            rows = connection.execute(
                """
                SELECT rowid
                FROM reference_law_records_fts
                WHERE reference_law_records_fts MATCH ?
                LIMIT ?
                """,
                (fts_query, limit),
            ).fetchall()
        return [int(row["rowid"]) for row in rows if row["rowid"] is not None]

    def _fetch_rows_by_row_ids(self, row_ids: list[int]) -> list[dict[str, Any]]:
        if not row_ids:
            return []
        placeholders = ", ".join("?" for _ in row_ids)
        metadata_path = self.settings.resolve_path(self.settings.reference_law_metadata_path)
        with sqlite3.connect(metadata_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                f"""
                SELECT row_id, title, title_norm, aliases_text, authority_type, domain, source_path,
                       page_start, page_end, parent_id, section_ref, section_ref_norm, section_title,
                       child_ref, child_ref_norm, retrieval_text, preview_text, child_text, parent_text
                FROM reference_law_records
                WHERE row_id IN ({placeholders})
                """,
                tuple(row_ids),
            ).fetchall()
        payload = [dict(row) for row in rows]
        order = {row_id: index for index, row_id in enumerate(row_ids)}
        payload.sort(key=lambda row: order.get(int(row["row_id"]), 10**9))
        return payload

    def _aggregate_parent_hits(
        self,
        *,
        query: str,
        rows: list[dict[str, Any]],
        dense_score_map: dict[int, float],
        dense_rank_map: dict[int, int],
        lexical_rank_map: dict[int, int],
        exact_rank_map: dict[int, int],
        question_profile: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        grouped: dict[str, dict[str, Any]] = {}
        lowered_query = normalize_whitespace(query).lower()
        query_aliases = set(self._query_aliases(lowered_query))
        query_domain = normalize_whitespace((question_profile or {}).get("domain")).lower()
        exact_terms = [
            normalize_whitespace(str(item)).lower()
            for item in (question_profile or {}).get("exact_terms") or []
        ]
        remedy_terms = [
            normalize_whitespace(str(item)).lower()
            for item in (question_profile or {}).get("remedy_terms") or []
        ]
        task = normalize_whitespace((question_profile or {}).get("task") or "").lower()
        explicit_refs = set(self._extract_query_refs(lowered_query))
        for row in rows:
            row_id = int(row["row_id"])
            parent_id = str(row["parent_id"])
            dense_score = float(dense_score_map.get(row_id, 0.0))
            lexical_rrf = self._rrf_score(lexical_rank_map.get(row_id))
            dense_rrf = self._rrf_score(dense_rank_map.get(row_id))
            exact_rrf = self._rrf_score(exact_rank_map.get(row_id))
            row_text = row.get("retrieval_text") or ""
            row_title_norm = normalize_whitespace(row.get("title_norm") or "")
            row_aliases = normalize_whitespace(row.get("aliases_text") or "").lower()
            lexical_overlap = lexical_overlap_score(query, row_text)
            title_bonus = 0.0
            title_penalty = 0.0
            if query_aliases:
                if any(alias == row_title_norm or alias in row_aliases for alias in query_aliases):
                    title_bonus = 1.15
                else:
                    title_penalty = -0.65
            domain_bonus = 0.18 if query_domain and query_domain == normalize_whitespace(row.get("domain")).lower() else 0.0
            domain_penalty = -0.55 if query_domain and row.get("domain") and normalize_whitespace(row.get("domain")).lower() != query_domain else 0.0
            row_refs = {
                normalize_whitespace(row.get("section_ref_norm") or "").lower(),
                normalize_whitespace(row.get("child_ref_norm") or "").lower(),
            }
            row_refs = {value for value in row_refs if value}
            ref_bonus = 0.0
            ref_penalty = 0.0
            if explicit_refs:
                if row_id in exact_rank_map or explicit_refs.intersection(row_refs):
                    ref_bonus = 1.25
                elif task in {"exact_provision_lookup", "procedure_or_remedy"}:
                    ref_penalty = -0.95
                else:
                    ref_penalty = -0.35
            lowered_row_text = row_text.lower()
            exact_term_hits = [term for term in exact_terms if term and term in lowered_row_text]
            remedy_hits = [term for term in remedy_terms if term and term in lowered_row_text]
            exact_term_bonus = min(0.22 * len(exact_term_hits), 0.44)
            remedy_bonus = min(0.14 * len(remedy_hits), 0.28)
            keyword_bonus = self._keyword_bonus(lowered_query, row_text)
            score = (
                dense_score
                + dense_rrf
                + lexical_rrf
                + exact_rrf
                + lexical_overlap
                + title_bonus
                + title_penalty
                + domain_bonus
                + domain_penalty
                + ref_bonus
                + ref_penalty
                + exact_term_bonus
                + remedy_bonus
                + keyword_bonus
            )
            strategy_parts: list[str] = []
            if row_id in exact_rank_map:
                strategy_parts.append("exact")
            if row_id in lexical_rank_map:
                strategy_parts.append("fts")
            if row_id in dense_rank_map:
                strategy_parts.append("dense")
            entry = grouped.get(parent_id)
            if entry is None or score > float(entry["score"]):
                grouped[parent_id] = {
                    "title": row["title"],
                    "section_ref": row.get("section_ref"),
                    "section_title": row.get("section_title"),
                    "child_ref": row.get("child_ref"),
                    "authority_type": row.get("authority_type"),
                    "domain": row.get("domain"),
                    "page_start": row.get("page_start"),
                    "page_end": row.get("page_end"),
                    "score": score,
                    "match_type": "exact" if row_id in exact_rank_map else "related" if row_id in lexical_rank_map else "semantic",
                    "retrieval_strategy": " + ".join(strategy_parts) or "dense",
                    "excerpt": row.get("preview_text") or row.get("child_text") or "",
                    "parent_text": row.get("parent_text") or "",
                }
            else:
                entry["excerpt"] = shorten_text(
                    f"{entry['excerpt']} {row.get('preview_text') or ''}",
                    420,
                )
        hits: list[dict[str, Any]] = []
        for entry in grouped.values():
            score = float(entry["score"])
            entry["similarity"] = round(max(min(score, 1.0), 0.0), 4)
            if entry["match_type"] == "exact" or score >= 0.9:
                entry["retrieval_confidence"] = "high"
            elif score >= 0.45:
                entry["retrieval_confidence"] = "moderate"
            else:
                entry["retrieval_confidence"] = "low"
            hits.append(entry)
        hits.sort(key=lambda item: (float(item["score"]), item["retrieval_confidence"] == "high"), reverse=True)
        return hits

    def _query_aliases(self, lowered_query: str) -> list[str]:
        hits: list[str] = []
        for title, aliases in TITLE_ALIASES.items():
            for alias in [title.lower(), *aliases]:
                cleaned = normalize_whitespace(alias).lower()
                if cleaned and cleaned in lowered_query and cleaned not in hits:
                    hits.append(cleaned)
        return hits

    def _rrf_score(self, rank: int | None) -> float:
        if rank is None or rank <= 0:
            return 0.0
        return 1.0 / (self.settings.reference_law_rrf_k + rank)

    @staticmethod
    def _keyword_bonus(lowered_query: str, row_text: str) -> float:
        lowered_row = normalize_whitespace(row_text).lower()
        bonus = 0.0
        if "limitation" in lowered_query:
            limitation_markers = ("limitation", "time limit", "within thirty", "within thirty days", "within 30 days")
            bonus += 0.18 if any(marker in lowered_row for marker in limitation_markers) else -0.16
        if "disciplinary" in lowered_query:
            disciplinary_markers = ("disciplinary", "inquiry", "penalties", "proceedings")
            bonus += 0.16 if any(marker in lowered_row for marker in disciplinary_markers) else -0.12
        if any(marker in lowered_query for marker in ("remedy", "remedies", "relief")):
            remedy_markers = (
                "refund",
                "compensation",
                "remove the defect",
                "replace the goods",
                "return to the complainant the price",
                "return the price",
                "cease and desist",
                "withdraw the hazardous goods",
            )
            bonus += 0.18 if any(marker in lowered_row for marker in remedy_markers) else -0.12
        if any(marker in lowered_query for marker in ("commercial confidence", "trade secret", "intellectual property", "8(1)(d)")):
            commercial_markers = ("commercial confidence", "trade secret", "intellectual property", "larger public interest")
            bonus += 0.18 if any(marker in lowered_row for marker in commercial_markers) else -0.12
        if "rti" in lowered_query and "limitation" in lowered_query:
            bonus += 0.24 if all(any(marker in lowered_row for marker in group) for group in (("appeal",), ("thirty days", "30 days", "within thirty days", "within 30 days"))) else -0.16
        if any(marker in lowered_query for marker in ("article 21", "personal liberty")):
            liberty_markers = ("personal liberty", "article 21", "life and personal liberty")
            bonus += 0.18 if any(marker in lowered_row for marker in liberty_markers) else -0.12
        return bonus
