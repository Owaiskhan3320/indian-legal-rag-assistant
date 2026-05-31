from __future__ import annotations

from typing import Any

from legal_ai.config import Settings
from legal_ai.utils.text import normalize_whitespace, shorten_text


class RAGContextBuilder:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def build(
        self,
        *,
        question: str,
        similar_cases: list[dict[str, Any]],
        scope: str,
        source_mode: str = "document_plus_case",
        document_context: dict[str, Any] | None = None,
        evidence_pack: dict[str, Any] | None = None,
        law_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if evidence_pack:
            return self._build_from_evidence_pack(
                question=question,
                source_mode=source_mode,
                document_context=document_context,
                evidence_pack=evidence_pack,
                law_context=law_context,
            )

        selected_items: list[dict[str, Any]] = []
        used_case_ids: list[str] = []
        seen_keys: set[tuple[str, str]] = set()
        total_chars = 0

        for item in similar_cases:
            case_id = str(item.get("case_id") or "").strip()
            excerpt = normalize_whitespace(item.get("excerpt") or item.get("summary") or "")
            if not case_id or not excerpt:
                continue

            dedupe_key = (case_id, excerpt[:160])
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)

            clipped_excerpt = shorten_text(excerpt, self.settings.rag_context_excerpt_chars)
            block = "\n".join(
                [
                    f"Case ID: {case_id}",
                    f"Outcome: {item.get('label_name') or 'Unknown'}",
                    f"Court: {item.get('court') or 'Unknown'}",
                    f"Case type: {item.get('case_type') or 'Unknown'}",
                    f"Why retrieved: {item.get('retrieval_note') or 'Semantic and lexical similarity.'}",
                    f"Passage: {clipped_excerpt}",
                ]
            )

            if selected_items and total_chars + len(block) > self.settings.rag_context_max_chars:
                break

            selected_items.append(
                {
                    "case_id": case_id,
                    "title": item.get("title") or case_id,
                    "label_name": item.get("label_name") or "Unknown",
                    "court": item.get("court"),
                    "case_type": item.get("case_type"),
                    "excerpt": clipped_excerpt,
                    "context_block": block,
                }
            )
            total_chars += len(block)
            if case_id not in used_case_ids:
                used_case_ids.append(case_id)
            if len(selected_items) >= self.settings.rag_context_max_authorities:
                break

        document_note = normalize_whitespace((document_context or {}).get("coverage_note") or "")
        document_text = normalize_whitespace((document_context or {}).get("context_text") or "")
        law_note = normalize_whitespace((law_context or {}).get("coverage_note") or "")
        law_text = normalize_whitespace((law_context or {}).get("context_text") or "")
        if source_mode == "document_only":
            scope_label = "the uploaded document only"
            coverage_note = (
                "Answer from the uploaded document excerpts prepared for this turn."
                if document_text
                else "No uploaded document excerpts were available for this turn."
            )
        elif source_mode == "reference_law_only":
            scope_label = "official law materials only"
            coverage_note = (
                "Answer from official law materials only."
                if law_text
                else "No official law materials were available for this turn."
            )
        elif source_mode == "reference_law_plus_case":
            scope_label = "official law materials with case-law support"
            if law_text and selected_items:
                coverage_note = (
                    f"Answer from official law materials with {len(selected_items)} retrieved authorities as supporting case-law."
                )
            elif law_text:
                coverage_note = "Answer from official law materials only because no closely matching case-law authorities were required or retrieved."
            else:
                coverage_note = "No reliable official law material was prepared for this turn."
        elif source_mode == "document_plus_reference_law":
            scope_label = "the uploaded document and official law materials"
            if document_text and law_text:
                coverage_note = "Answer from the uploaded document plus official law materials."
            elif document_text:
                coverage_note = "Answer from the uploaded document because no official law materials were available."
            else:
                coverage_note = "No reliable uploaded-document or official law bundle was prepared for this turn."
        elif source_mode == "document_plus_reference_law_plus_case":
            scope_label = "the uploaded document, official law materials, and case-law support"
            if document_text and law_text and selected_items:
                coverage_note = (
                    f"Answer from the uploaded document and official law materials with {len(selected_items)} retrieved authorities as supporting case-law."
                )
            elif document_text and law_text:
                coverage_note = "Answer from the uploaded document plus official law materials."
            else:
                coverage_note = "No reliable combined evidence bundle was prepared for this turn."
        elif source_mode == "case_corpus_only":
            scope_label = (
                "the full judgment library"
                if scope == "corpus"
                else "the evidence already selected in this conversation"
            )
            coverage_note = (
                f"Answer from {scope_label} using {len(selected_items)} retrieved authorities."
                if selected_items
                else f"No reliable authority bundle was prepared from {scope_label}."
            )
        else:
            scope_label = (
                "the uploaded document and the full judgment library"
                if scope == "corpus"
                else "the uploaded document and the evidence already selected in this conversation"
            )
            if selected_items and document_text:
                coverage_note = (
                    f"Answer from {scope_label} using uploaded document excerpts and {len(selected_items)} retrieved authorities."
                )
            elif selected_items and law_text:
                coverage_note = (
                    f"Answer from {scope_label} using official law materials and {len(selected_items)} retrieved authorities."
                )
            elif selected_items:
                coverage_note = f"Answer from the judgment corpus using {len(selected_items)} retrieved authorities."
            elif law_text:
                coverage_note = "No reliable case-law authority bundle was prepared, so only official law materials were available."
            elif document_text:
                coverage_note = "No reliable case-law authority bundle was prepared, so only uploaded document excerpts were available."
            else:
                coverage_note = f"No reliable evidence bundle was prepared from {scope_label}."
        context_blocks: list[str] = []
        if document_text:
            context_blocks.append(f"Uploaded document context\n{document_text}")
        if law_text:
            context_blocks.append(f"Official law materials\n{law_text}")
        if selected_items:
            context_blocks.extend(item["context_block"] for item in selected_items)

        return {
            "question": question.strip(),
            "source_mode": source_mode,
            "scope_label": scope_label,
            "coverage_note": " ".join(
                part for part in [coverage_note, document_note, law_note] if part
            ),
            "used_case_ids": used_case_ids[:5],
            "document_used": bool(document_text),
            "document_filename": (document_context or {}).get("filename"),
            "law_used": bool(law_text),
            "reference_materials": list((law_context or {}).get("materials") or []),
            "items": selected_items,
            "context_text": "\n\n".join(context_blocks)
            or "No retrieved authorities were available.",
        }

    def _build_from_evidence_pack(
        self,
        *,
        question: str,
        source_mode: str,
        document_context: dict[str, Any] | None,
        evidence_pack: dict[str, Any],
        law_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        cards = list(evidence_pack.get("cards") or [])
        selected_items: list[dict[str, Any]] = []
        for card in cards[: self.settings.rag_context_max_authorities]:
            excerpt = shorten_text(card.get("excerpt") or "", self.settings.rag_context_excerpt_chars)
            selected_items.append(
                {
                    "case_id": card.get("case_id"),
                    "title": card.get("title") or card.get("case_id"),
                    "label_name": card.get("proposition") or "Unknown",
                    "court": card.get("court"),
                    "case_type": card.get("case_type"),
                    "excerpt": excerpt,
                    "context_block": "\n".join(
                        [
                            f"Case ID: {card.get('case_id')}",
                            f"Authority level: {card.get('authority_level') or 'Unknown'}",
                            f"Support type: {card.get('support_type') or 'Unknown'}",
                            f"Proposition: {card.get('proposition') or 'No proposition recorded'}",
                            f"Passage: {excerpt}",
                        ]
                    ),
                }
            )

        document_note = normalize_whitespace((document_context or {}).get("coverage_note") or "")
        document_text = normalize_whitespace((document_context or {}).get("context_text") or "")
        law_note = normalize_whitespace((law_context or {}).get("coverage_note") or "")
        law_text = normalize_whitespace((law_context or {}).get("context_text") or "")
        coverage_note = normalize_whitespace(evidence_pack.get("coverage_note") or "")
        context_blocks: list[str] = []
        if evidence_pack.get("context_text"):
            context_blocks.append(normalize_whitespace(evidence_pack["context_text"]))
        else:
            if document_text:
                context_blocks.append(f"Uploaded document context\n{document_text}")
            if law_text:
                context_blocks.append(f"Official law materials\n{law_text}")
        if not context_blocks and selected_items:
            context_blocks.extend(item["context_block"] for item in selected_items)

        return {
            "question": question.strip(),
            "source_mode": source_mode,
            "scope_label": evidence_pack.get("scope_label") or "the current evidence set",
            "coverage_note": " ".join(
                part for part in [coverage_note, document_note, law_note] if part
            ),
            "used_case_ids": list(evidence_pack.get("case_ids") or [])[:5],
            "document_used": bool(document_text),
            "document_filename": (document_context or {}).get("filename"),
            "law_used": bool(law_text),
            "reference_materials": list((law_context or {}).get("materials") or evidence_pack.get("reference_materials") or []),
            "items": selected_items,
            "context_text": "\n\n".join(block for block in context_blocks if block)
            or "No retrieved authorities were available.",
        }
