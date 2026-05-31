from __future__ import annotations

import re
from typing import Any

from legal_ai.utils.text import normalize_whitespace, shorten_text


CASE_ID_RE = re.compile(r"\b[A-Za-z]+(?:_[A-Za-z0-9]+){2,}\b")
STATUTE_RE = re.compile(r"\b(?:section|sec\.|article)\s+\d+[A-Za-z0-9()/-]*", re.I)
OVERCLAIM_REPLACEMENTS = (
    (r"\bnecessarily\b", "not necessarily"),
    (r"\balways\b", "often"),
    (r"\bmust\b", "generally should"),
    (r"\bexplicitly establishes\b", "may suggest"),
    (r"\bsettled rule\b", "possible line of case-law support"),
)


class AnswerAuditService:
    def audit(
        self,
        *,
        question: str,
        question_profile: dict[str, Any],
        evidence_pack: dict[str, Any],
        answer_text: str,
        source_mode: str,
    ) -> dict[str, Any]:
        cleaned = self._normalize_answer(answer_text)
        if not cleaned:
            return {
                "text": self._build_fallback(
                    question=question,
                    question_profile=question_profile,
                    evidence_pack=evidence_pack,
                    reason="No answer text was generated from the current evidence.",
                ),
                "status": "repaired",
                "advisories": ["The answer was rebuilt because the draft response was empty."],
                "flags": ["empty_answer"],
            }

        advisories: list[str] = []
        flags: list[str] = []
        supported_case_ids = set(evidence_pack.get("case_ids") or [])
        cited_case_ids = [match for match in CASE_ID_RE.findall(cleaned) if match]
        unsupported_case_ids = [case_id for case_id in cited_case_ids if case_id not in supported_case_ids]
        statute_refs = STATUTE_RE.findall(cleaned)

        if unsupported_case_ids:
            flags.append("unsupported_case_ids")
            advisories.append(
                "Some authority references were dropped because they were not supported by the current evidence pack."
            )

        statute_sensitive = bool(question_profile.get("statute_sensitive"))
        statute_support_available = bool(evidence_pack.get("statute_support_available"))
        if statute_refs and not statute_support_available:
            flags.append("unsupported_statute_reference")
            advisories.append(
                "This answer cannot safely rely on provision-level statutory wording because no matching law material was retrieved for this turn."
            )

        if statute_sensitive and not statute_support_available:
            advisories.append(
                "This answer is limited by the retrieved evidence because no matching provision-level law material was available for this turn."
            )

        direct_support_count = int(evidence_pack.get("direct_support_count") or 0)
        if direct_support_count == 0:
            softened = self._soften_overclaim_language(cleaned)
            if softened != cleaned:
                cleaned = softened
                flags.append("softened_overclaim")
                advisories.append(
                    "The wording was softened because the available support is mostly analogical rather than direct."
                )

        if unsupported_case_ids or (statute_sensitive and not statute_support_available and statute_refs):
            return {
                "text": self._build_fallback(
                    question=question,
                    question_profile=question_profile,
                    evidence_pack=evidence_pack,
                    reason="The available authorities did not support a stronger answer safely.",
                ),
                "status": "repaired",
                "advisories": advisories[:4],
                "flags": flags,
            }

        if advisories:
            cleaned = self._append_caution(cleaned, advisories[0])

        return {
            "text": cleaned,
            "status": "passed_with_notes" if advisories else "passed",
            "advisories": advisories[:4],
            "flags": flags,
        }

    @staticmethod
    def _normalize_answer(text: str) -> str:
        cleaned = (text or "").replace("\r\n", "\n").strip()
        normalized_lines: list[str] = []
        blank_pending = False
        for raw_line in cleaned.split("\n"):
            line = normalize_whitespace(raw_line)
            if not line:
                if normalized_lines and not blank_pending:
                    normalized_lines.append("")
                blank_pending = True
                continue
            normalized_lines.append(line)
            blank_pending = False
        return "\n".join(normalized_lines).strip()

    @staticmethod
    def _soften_overclaim_language(text: str) -> str:
        softened = text
        for pattern, replacement in OVERCLAIM_REPLACEMENTS:
            softened = re.sub(pattern, replacement, softened, flags=re.I)
        return softened

    @staticmethod
    def _append_caution(text: str, advisory: str) -> str:
        if not advisory:
            return text
        lowered = text.lower()
        if "limits" in lowered or "caution" in lowered:
            return text
        caution_line = f"Caution: {advisory}"
        if caution_line.lower() in text.lower():
            return text
        return f"{text}\n\n{caution_line}"

    def _build_fallback(
        self,
        *,
        question: str,
        question_profile: dict[str, Any],
        evidence_pack: dict[str, Any],
        reason: str,
    ) -> str:
        cards = list(evidence_pack.get("cards") or [])
        response_plan = str(question_profile.get("response_plan") or "direct_guidance")
        if not cards:
            if response_plan == "practical_steps":
                return (
                    "#### Bottom line\n"
                    f"- I cannot answer '{shorten_text(question, 120)}' reliably from the current evidence.\n\n"
                    "#### Best next step\n"
                    "- Narrow the issue, add the forum or remedy sought, or inspect the full judgment text directly."
                )
            return (
                "#### Answer\n"
                f"- I cannot answer '{shorten_text(question, 120)}' reliably from the current evidence.\n\n"
                "#### Why\n"
                f"- {reason}\n\n"
                "#### Best next step\n"
                "- Please narrow the issue, add more facts, or inspect the full judgment text directly."
            )

        lead_cards = cards[:2]
        lead_summary = " ".join(
            f"{card['case_id']} ({card['support_type']}, {card['authority_level']}) suggests {card['proposition']}."
            for card in lead_cards
        )
        if question_profile.get("statute_sensitive") and not evidence_pack.get("statute_support_available"):
            return (
                "#### Answer\n"
                f"- I cannot state a settled statutory rule for '{shorten_text(question, 120)}' from the current setup.\n\n"
                "#### Available case-law support\n"
                f"- No provision-level statute source was retrieved.\n"
                f"- The closest case-law support is: {lead_summary}\n\n"
                "#### Source used\n"
                + "\n".join(f"- {card['case_id']}." for card in lead_cards)
                + "\n\n#### Limits\n"
                + "- Treat this as analogous case-law guidance, not a statute-grounded doctrinal answer."
            )
        if response_plan == "practical_steps":
            return (
                "#### Bottom line\n"
                "- I cannot make the stronger practical recommendation safely from the current evidence.\n\n"
                "#### Why\n"
                f"- {reason}\n"
                f"- The strongest available authorities are: {lead_summary}\n\n"
                "#### Best next step\n"
                "- Treat this as a cautious starting point and verify the full judgments before acting."
            )
        if response_plan == "research_note":
            return (
                "#### Issue\n"
                f"- {shorten_text(question, 120)}\n\n"
                "#### Current position\n"
                f"- {reason}\n"
                f"- The strongest available authorities are: {lead_summary}\n\n"
                "#### Open points\n"
                "- Verify the full authorities before relying on this as a research-note conclusion."
            )
        if response_plan == "outcome_pattern":
            return (
                "#### Current pattern\n"
                f"- {reason}\n"
                f"- The strongest available authorities are: {lead_summary}\n\n"
                "#### Limits\n"
                "- Treat this as a tentative outcome pattern until the full accepted and rejected authorities are checked side by side."
            )
        return (
            "#### Answer\n"
            "- I cannot make the stronger claim safely from the current evidence.\n\n"
            "#### Why\n"
            f"- {reason}\n"
            f"- The strongest available authorities are: {lead_summary}\n\n"
            "#### Source used\n"
            + "\n".join(f"- {card['case_id']}." for card in lead_cards)
            + "\n\n#### Limits\n"
            + "- Treat this as a cautious case-law synthesis and verify the full authorities before relying on it."
        )
