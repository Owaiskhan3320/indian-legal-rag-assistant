from __future__ import annotations

from typing import Any

from legal_ai.utils.domain import infer_candidate_domain
from legal_ai.utils.text import lexical_overlap_score, normalize_whitespace, overlapping_terms, shorten_text


SECTION_KEYWORDS = {
    "relief": ("award", "awarded", "compensation", "refund", "replacement", "directed", "relief"),
    "holding": ("held", "therefore", "we find", "it is ordered", "it is held"),
    "reasoning": ("because", "reason", "observed", "considered", "whether", "therefore"),
    "issue": ("issue", "question", "whether", "point for consideration"),
    "facts": ("facts", "purchased", "accident", "complainant", "assessee", "student"),
}


class EvidencePackBuilder:
    def __init__(self, max_cards: int = 4) -> None:
        self.max_cards = max_cards

    def build(
        self,
        *,
        question: str,
        question_profile: dict[str, Any],
        similar_cases: list[dict[str, Any]],
        source_mode: str,
        scope: str,
        document_context: dict[str, Any] | None = None,
        law_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        cards = [
            self._build_authority_card(
                question=question,
                question_profile=question_profile,
                item=item,
            )
            for item in similar_cases
        ]
        cards.sort(
            key=lambda item: (item["rank_score"], item["similarity"], item["authority_weight"]),
            reverse=True,
        )
        card_limit = self.max_cards
        if str(question_profile.get("retrieval_profile") or "fast") == "fast":
            card_limit = min(card_limit, 2)
        cards = cards[:card_limit]
        statute_support_available = bool((law_context or {}).get("used"))
        claim_map = self._build_claim_evidence_map(
            question_profile=question_profile,
            cards=cards,
            statute_support_available=statute_support_available,
        )
        direct_support_count = len([card for card in cards if card["support_type"] == "direct"])
        analogical_support_count = len([card for card in cards if card["support_type"] == "analogical"])
        coverage_note_parts = []
        if source_mode == "document_only":
            coverage_note_parts.append("Answer is restricted to the uploaded document.")
        elif source_mode == "reference_law_only":
            coverage_note_parts.append("Answer is grounded in official law materials first.")
        elif source_mode == "reference_law_plus_case":
            coverage_note_parts.append("Answer is grounded in official law materials with optional case-law support.")
        elif source_mode == "document_plus_reference_law":
            coverage_note_parts.append("Answer is grounded in the uploaded document plus official law materials.")
        elif source_mode == "document_plus_reference_law_plus_case":
            coverage_note_parts.append("Answer is grounded in the uploaded document, official law materials, and supporting case-law.")
        elif cards:
            coverage_note_parts.append(
                f"Prepared {len(cards)} authority cards from the case-law lane."
            )
        else:
            coverage_note_parts.append("No usable authority cards were prepared from the case-law lane.")
        if statute_support_available:
            coverage_note_parts.append((law_context or {}).get("coverage_note") or "Official law material support is available for this turn.")
        if question_profile.get("statute_sensitive") and not statute_support_available:
            coverage_note_parts.append(
                "No matching provision-level law material was retrieved for this turn, so doctrinal claims must stay limited to supported case-law or document context."
            )

        context_blocks = []
        document_text = normalize_whitespace((document_context or {}).get("context_text") or "")
        law_text = normalize_whitespace((law_context or {}).get("context_text") or "")
        if document_text and source_mode in {"document_only", "document_plus_case"}:
            context_blocks.append(f"Uploaded document context\n{document_text}")
        if law_text:
            context_blocks.append(f"Official law materials\n{law_text}")
        for index, card in enumerate(cards, start=1):
            context_blocks.append(
                "\n".join(
                    [
                        f"Authority card {index}",
                        f"- Case ID: {card['case_id']}",
                        f"- Authority level: {card['authority_level']}",
                        f"- Support type: {card['support_type']}",
                        f"- Section focus: {card['section_label']}",
                        f"- Proposition: {card['proposition']}",
                        f"- Why it matters: {card['why_it_matters']}",
                        f"- Passage: {card['excerpt']}",
                    ]
                )
            )
        if claim_map:
            claim_lines = ["Claim-evidence map"]
            for claim in claim_map:
                claim_lines.append(
                    f"- {claim['claim']} -> {claim['case_id']} ({claim['support_type']}, {claim['authority_level']})"
                )
            context_blocks.append("\n".join(claim_lines))

        return {
            "lane": question_profile.get("lane") or "case_law",
            "task": question_profile.get("task") or "general_research",
            "domain": question_profile.get("domain"),
            "cards": cards,
            "claim_evidence_map": claim_map,
            "case_ids": [card["case_id"] for card in cards],
            "direct_support_count": direct_support_count,
            "analogical_support_count": analogical_support_count,
            "statute_support_available": statute_support_available,
            "coverage_note": " ".join(coverage_note_parts),
            "context_text": "\n\n".join(context_blocks),
            "scope_label": (
                "the full judgment library"
                if scope == "corpus"
                else "the currently selected evidence set"
            ),
            "document_used": bool(document_text),
            "document_filename": (document_context or {}).get("filename"),
            "law_used": bool(law_text),
            "reference_materials": list((law_context or {}).get("materials") or []),
        }

    def _build_authority_card(
        self,
        *,
        question: str,
        question_profile: dict[str, Any],
        item: dict[str, Any],
    ) -> dict[str, Any]:
        excerpt = normalize_whitespace(item.get("excerpt") or item.get("summary") or "")
        section_label = str(item.get("section_label") or self._infer_section_label(excerpt))
        authority_level, authority_weight = self._infer_authority_level(
            case_id=str(item.get("case_id") or ""),
            court=item.get("court"),
        )
        support_type = str(
            item.get("support_type")
            or self._infer_support_type(
                question=question,
                question_profile=question_profile,
                excerpt=excerpt,
                case_id=str(item.get("case_id") or ""),
            )
        )
        proposition = str(
            item.get("proposition")
            or self._infer_proposition(question_profile=question_profile, excerpt=excerpt, section_label=section_label)
        )
        section_bonus = 0.08 if section_label in set(question_profile.get("preferred_sections") or []) else 0.0
        support_bonus = 0.12 if support_type == "direct" else 0.05 if support_type == "supportive" else 0.0
        rank_score = float(item.get("similarity") or 0.0) + authority_weight + section_bonus + support_bonus
        return {
            "case_id": str(item.get("case_id") or ""),
            "title": item.get("title") or item.get("case_id"),
            "court": item.get("court"),
            "case_type": item.get("case_type"),
            "date": item.get("date"),
            "authority_level": authority_level,
            "authority_weight": authority_weight,
            "support_type": support_type,
            "section_label": section_label,
            "proposition": proposition,
            "why_it_matters": self._why_it_matters(proposition, support_type, authority_level),
            "excerpt": shorten_text(excerpt, 320),
            "similarity": float(item.get("similarity") or 0.0),
            "rank_score": round(rank_score, 4),
        }

    @staticmethod
    def _infer_section_label(excerpt: str) -> str:
        lowered = normalize_whitespace(excerpt).lower()
        best_label = "reasoning"
        best_score = 0
        for label, keywords in SECTION_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in lowered)
            if score > best_score:
                best_label = label
                best_score = score
        return best_label

    @staticmethod
    def _infer_authority_level(*, case_id: str, court: str | None) -> tuple[str, float]:
        lowered_case_id = normalize_whitespace(case_id).lower()
        lowered_court = normalize_whitespace(court).lower()
        if "supremecourt" in lowered_case_id or "supreme court" in lowered_court:
            return "supreme_court", 0.18
        if "_hc_" in lowered_case_id or "high court" in lowered_court or lowered_case_id.endswith("_hc"):
            return "high_court", 0.11
        if any(token in lowered_case_id for token in ("tribunal", "commission", "consumer_disputes")):
            return "tribunal_or_forum", 0.05
        return "other", 0.0

    @staticmethod
    def _infer_support_type(
        *,
        question: str,
        question_profile: dict[str, Any],
        excerpt: str,
        case_id: str,
    ) -> str:
        referenced_case_ids = set(question_profile.get("referenced_case_ids") or [])
        if case_id and case_id in referenced_case_ids:
            return "direct"
        lexical = lexical_overlap_score(question, excerpt)
        matched_terms = overlapping_terms(question, excerpt, limit=5)
        if lexical >= 0.28 or len(matched_terms) >= 3:
            return "direct"
        if lexical >= 0.14 or len(matched_terms) >= 2:
            return "supportive"
        return "analogical"

    @staticmethod
    def _infer_proposition(
        *,
        question_profile: dict[str, Any],
        excerpt: str,
        section_label: str,
    ) -> str:
        domain = question_profile.get("domain")
        task = question_profile.get("task")
        lowered = normalize_whitespace(excerpt).lower()
        if domain == "consumer":
            if any(token in lowered for token in ("refund", "replacement", "repair")):
                return "consumer remedy discretion"
            if "deficiency" in lowered:
                return "deficiency in service analysis"
        if domain == "motor_accident":
            if any(token in lowered for token in ("disability", "earning", "future treatment", "compensation")):
                return "motor accident compensation factors"
        if domain == "education":
            if any(token in lowered for token in ("hearing", "notice", "unfair means", "exam")):
                return "procedural fairness in education disputes"
        if domain == "tax":
            if any(token in lowered for token in ("documents", "invoice", "bank", "ledger", "addition")):
                return "documentary sufficiency in tax disputes"
        if domain == "service":
            if any(token in lowered for token in ("suspension", "disciplinary", "departmental", "pension", "promotion")):
                return "service-law fairness and disciplinary review"
        if domain == "information":
            if any(token in lowered for token in ("information", "records", "disclosure", "cpio", "commission")):
                return "disclosure duties and access-to-records analysis"
        if task == "case_explanation":
            return f"{section_label} of the cited case"
        if task == "similarity_lookup":
            return "fact-pattern similarity support"
        return f"{section_label}-focused legal support"

    @staticmethod
    def _why_it_matters(proposition: str, support_type: str, authority_level: str) -> str:
        return (
            f"This passage offers {support_type} support for {proposition} "
            f"from a {authority_level.replace('_', ' ')} authority."
        )

    @staticmethod
    def _build_claim_evidence_map(
        *,
        question_profile: dict[str, Any],
        cards: list[dict[str, Any]],
        statute_support_available: bool,
    ) -> list[dict[str, Any]]:
        claim_map: list[dict[str, Any]] = []
        for card in cards[:3]:
            claim_map.append(
                {
                    "claim": card["proposition"],
                    "case_id": card["case_id"],
                    "support_type": card["support_type"],
                    "authority_level": card["authority_level"],
                }
            )
        if question_profile.get("statute_sensitive") and not statute_support_available:
            claim_map.insert(
                0,
                {
                    "claim": "statute-specific answer requires provision-level support",
                    "case_id": "case-law-only",
                    "support_type": "limitation",
                    "authority_level": "system_guardrail",
                },
            )
        return claim_map[:4]
