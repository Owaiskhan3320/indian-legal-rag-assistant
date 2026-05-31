from __future__ import annotations

from collections import Counter
from typing import Any

from legal_ai.utils.text import normalize_whitespace, search_terms, shorten_text


ISSUE_KEYWORDS = [
    ("natural justice / hearing", ("hearing", "show cause", "notice", "opportunity", "audi")),
    ("procedural fairness", ("procedure", "irregularity", "violation", "due process")),
    ("unfair means / examination discipline", ("unfair means", "cheating", "result", "exam", "invigilator")),
    ("consumer defect / refund", ("refund", "defective", "seller", "marketplace", "warranty")),
    ("service termination / disciplinary action", ("termination", "dismissal", "charge sheet", "misconduct")),
    ("tax disallowance / demand", ("tax", "assessee", "addition", "disallowance", "assessment")),
    ("property possession / title", ("possession", "title", "land", "property", "mutation")),
    ("contract breach / recovery", ("contract", "agreement", "breach", "recovery", "payment")),
    ("bail / liberty", ("bail", "custody", "arrest", "liberty", "offence")),
    ("writ / administrative review", ("writ", "administrative", "authority", "order", "jurisdiction")),
]

DOMAIN_FACT_PATTERNS = {
    "tax": [
        ("Likely deciding issue: whether the purchases were genuine or merely accommodation entries.", ("genuine", "bogus", "purchase", "purchases", "supplier")),
        ("Likely pressure point: whether banking trail and ledger entries are enough without stronger third-party or goods-movement proof.", ("bank", "ledger", "invoice", "documents", "banking")),
        ("Tribunal focus often turns on documentary sufficiency versus mere suspicion by the assessing officer.", ("assessment", "addition", "unsatisfactory", "documents", "assessee")),
    ],
    "consumer": [
        ("Likely deciding issue: whether early defect reporting and prolonged non-resolution justify refund instead of only repair or replacement.", ("refund", "replacement", "repair", "warranty", "defect")),
        ("Likely pressure point: whether the seller and manufacturer kept shifting responsibility despite same-day or prompt complaint.", ("seller", "manufacturer", "service centre", "delay", "reported")),
    ],
    "motor_accident": [
        ("Likely deciding issue: functional disability, future medical needs, and actual loss of earning capacity.", ("disability", "amputation", "earning", "prosthetic", "future")),
        ("Likely pressure point: whether the evidence proves future expenses and long-term work impact, not just immediate treatment.", ("treatment", "medical", "future", "income", "certificate")),
    ],
    "education": [
        ("Likely deciding issue: whether the authority had enough evidence of unfair means and whether the student got a fair hearing.", ("unfair means", "exam", "result", "hearing", "invigilator")),
        ("Likely pressure point: whether the action was based on direct recovery/evidence or only suspicion and procedural shortcuts.", ("recovered", "notice", "show cause", "suspicion", "material")),
    ],
    "service": [
        ("Likely deciding issue: whether the suspension or disciplinary action followed fair procedure and a defensible factual basis.", ("suspension", "disciplinary", "charge", "inquiry", "service")),
        ("Likely pressure point: delay, mala fides, and whether the inquiry or punishment exceeded what the record supports.", ("delay", "retirement", "mala", "procedural", "harassment")),
    ],
    "information": [
        ("Likely deciding issue: whether the authority could lawfully deny copies, give only inspection, or withhold the requested records.", ("rti", "information", "disclosure", "records", "inspection")),
        ("Likely pressure point: whether the request seeks identifiable public records and whether any exemption was properly invoked.", ("public authority", "pio", "appeal", "reply", "copies")),
    ],
}

DOMAIN_EVIDENCE_GUIDANCE = {
    "tax": [
        "If available, add supplier confirmations, tax-registration details, transport or stock records, and proof that goods were actually received.",
        "Check whether the assessment relies on third-party statements and whether cross-examination was requested or denied.",
    ],
    "consumer": [
        "Keep a clean timeline of delivery, defect reporting, repair attempts, replacement promises, and the final refusal or delay.",
        "Preserve messages showing who controlled the remedy and whether the product remained unusable for an unreasonable period.",
    ],
    "motor_accident": [
        "If available, add disability certification, income proof, and future-treatment or prosthetic estimates to strengthen quantum.",
        "Separate physical injury evidence from functional disability and earning-capacity evidence.",
    ],
    "education": [
        "Preserve the show-cause notice, invigilator report, student reply, and proof of denied hearing or weak recovery evidence.",
        "Clarify exactly what material was recovered, from where, and whether similarly placed students were treated differently.",
    ],
    "service": [
        "Keep the suspension order, chargesheet, reply, and service-rule timeline in one chronology.",
        "If available, add proof of delay, bias, ignored documents, or denial of procedural safeguards.",
    ],
    "information": [
        "Keep the RTI application, PIO reply, first-appeal papers, and exact description of the records requested together.",
        "Check whether the authority relied on volume, exemption, or record-form objections and whether copies were specifically requested.",
    ],
}

DOMAIN_TRIAGE_STEPS = {
    "tax": [
        "Compare whether the strongest accepted authorities turned on supplier appearance, cross-examination, transport proof, or only banking trail and books.",
        "Use the similar cases to distinguish mere suspicion from evidence that transactions were actually sham.",
    ],
    "consumer": [
        "Check whether the strongest consumer authorities granted refund because of early defect reporting, repeated failure, or unreasonable delay in providing a usable remedy.",
        "Distinguish cases where warranty repair was treated as sufficient from cases where refund became justified.",
    ],
    "motor_accident": [
        "Check whether the leading authorities treat the injury as functional disability rather than only medical disability.",
        "Compare how the retrieved cases handled future treatment, prosthetic replacement, and loss of earning capacity.",
    ],
    "education": [
        "Check whether the strongest authorities turned on denied hearing, no direct recovery, weak evidence, or procedural irregularity.",
        "Compare whether the adverse action was remitted for fresh hearing or directly set aside.",
    ],
    "service": [
        "Compare whether the leading service authorities turned on delay in inquiry, lack of charges, disproportionate action, or procedural unfairness.",
        "Check whether the strongest cases granted interim relief, quashed the action, or only required a fresh inquiry.",
    ],
    "information": [
        "Compare whether the strongest authorities required disclosure of copies, allowed inspection only, or remanded the matter to the appellate authority.",
        "Check whether the authority's refusal rests on an actual RTI exemption or only on administrative inconvenience.",
    ],
}


class WorkspaceBuilder:
    def build_issue_outline(
        self,
        *,
        intake: dict[str, Any] | None = None,
        question: str | None = None,
        topic_query: str | None = None,
    ) -> list[str]:
        intake = intake or {}
        text_parts = [
            intake.get("case_type"),
            intake.get("forum"),
            intake.get("facts"),
            intake.get("relief_sought"),
            intake.get("evidence_summary"),
            intake.get("opponent_arguments"),
            question,
            topic_query,
        ]
        combined = normalize_whitespace(" ".join(part or "" for part in text_parts))
        lowered = combined.lower()
        domain = self._infer_workspace_domain(intake=intake, combined=combined)

        issues: list[str] = []
        case_type = normalize_whitespace(intake.get("case_type"))
        relief = normalize_whitespace(intake.get("relief_sought"))
        if case_type:
            issues.append(f"Core matter: {case_type.lower()}")
        if relief:
            issues.append(f"Requested relief: {shorten_text(relief, 72)}")

        for issue_line, markers in DOMAIN_FACT_PATTERNS.get(domain, []):
            if any(marker in lowered for marker in markers):
                issues.append(issue_line)

        for label, keywords in ISSUE_KEYWORDS:
            if any(keyword in lowered for keyword in keywords):
                if domain == "tax" and label in {"contract breach / recovery", "writ / administrative review", "natural justice / hearing"}:
                    continue
                if domain == "consumer" and label in {"contract breach / recovery", "writ / administrative review"}:
                    continue
                if domain == "motor_accident" and label in {"contract breach / recovery", "writ / administrative review"}:
                    continue
                if domain == "service" and label in {"contract breach / recovery", "writ / administrative review"}:
                    continue
                if domain == "information" and label in {"contract breach / recovery", "writ / administrative review"}:
                    continue
                issues.append(label)

        if not issues:
            token_terms = search_terms(combined)
            if token_terms:
                issues.append("Key factual/legal focus: " + ", ".join(token_terms[:5]))
            else:
                issues.append("Core legal issue needs fuller factual detail.")

        deduped: list[str] = []
        seen: set[str] = set()
        for item in issues:
            if item not in seen:
                deduped.append(item)
                seen.add(item)
        return deduped[:5]

    def build_evidence_gaps(
        self,
        *,
        intake: dict[str, Any] | None = None,
        question: str | None = None,
        similar_cases: list[dict[str, Any]] | None = None,
        strict_intake: bool = False,
    ) -> list[str]:
        intake = intake or {}
        gaps: list[str] = []
        if strict_intake:
            if not normalize_whitespace(intake.get("facts")):
                gaps.append("Add a tighter facts narrative so retrieval can match legal posture more precisely.")
            if not normalize_whitespace(intake.get("relief_sought")):
                gaps.append("State the exact remedy sought; relief often changes which precedents are most relevant.")
            if not normalize_whitespace(intake.get("evidence_summary")):
                gaps.append("Summarize the strongest documents or evidence; this often changes outcome direction.")
            if not normalize_whitespace(intake.get("opponent_arguments")):
                gaps.append("Add the opposing side's position so the system can surface counter-authorities more honestly.")
            if not normalize_whitespace(intake.get("user_role")):
                gaps.append("Clarify whether the user is filing or responding; side-aware interpretation depends on it.")
            if not normalize_whitespace(intake.get("forum")):
                gaps.append("Forum/court type is missing; this can reduce retrieval precision.")

        domain = self._infer_workspace_domain(
            intake=intake,
            combined=" ".join(
                part for part in [
                    intake.get("case_type"),
                    intake.get("facts"),
                    intake.get("relief_sought"),
                    intake.get("evidence_summary"),
                    intake.get("opponent_arguments"),
                    question,
                ] if part
            ),
        )
        gaps.extend(DOMAIN_EVIDENCE_GUIDANCE.get(domain, []))

        if question and len(normalize_whitespace(question)) < 28:
            gaps.append("Question is still broad; add facts, procedure, or remedy to narrow the authorities.")

        if similar_cases is not None and not similar_cases:
            gaps.append("No strong authorities were retrieved; broaden or restate the factual pattern.")

        return gaps[:5]

    def organize_authorities(
        self,
        similar_cases: list[dict[str, Any]],
        *,
        predicted_name: str | None = None,
    ) -> dict[str, Any]:
        if not similar_cases:
            return {
                "supporting": [],
                "conflicting": [],
                "mixed": [],
                "reference_label": predicted_name,
                "rationale": "No authorities were available to organize.",
            }

        reference_label = predicted_name or self._dominant_label(similar_cases)
        supporting: list[dict[str, Any]] = []
        conflicting: list[dict[str, Any]] = []
        mixed: list[dict[str, Any]] = []

        for item in similar_cases:
            label_name = item.get("label_name")
            fit_band = item.get("fit_band") or "moderate"
            if reference_label and label_name == reference_label and fit_band != "low":
                supporting.append(item)
            elif (
                reference_label in {"Accepted", "Rejected"}
                and label_name in {"Accepted", "Rejected"}
                and label_name != reference_label
            ):
                conflicting.append(item)
            elif fit_band == "low":
                conflicting.append(item)
            else:
                mixed.append(item)

        rationale = self._build_authority_rationale(
            reference_label=reference_label,
            supporting=supporting,
            conflicting=conflicting,
            mixed=mixed,
        )
        return {
            "supporting": supporting[:3],
            "conflicting": conflicting[:3],
            "mixed": mixed[:3],
            "reference_label": reference_label,
            "rationale": rationale,
        }

    def build_next_steps(
        self,
        *,
        workflow: str,
        evidence_gaps: list[str],
        authority_map: dict[str, Any],
        confidence_band: str | None = None,
        intake: dict[str, Any] | None = None,
        issue_outline: list[str] | None = None,
    ) -> list[str]:
        steps: list[str] = []
        if workflow == "triage":
            steps.append("Inspect the best-support and distinguishable authorities before relying on the triage result.")
        elif workflow == "research":
            steps.append("Open the strongest retrieved judgments and compare their factual posture before taking notes.")
        elif workflow == "ask":
            steps.append("Keep follow-up questions on the current evidence when you want a tighter, more grounded answer.")

        if confidence_band == "Low":
            steps.append("Because confidence is low, re-run with clearer facts, evidence, and relief before sharing the result.")
        if authority_map.get("conflicting"):
            steps.append("Review the conflicting authorities; they are the likeliest basis for challenge or distinction.")
        if evidence_gaps:
            steps.append("Fill the missing facts/evidence fields and run the workflow again for a more stable result.")
        if not authority_map.get("supporting") and authority_map.get("mixed"):
            steps.append("The result set is mixed; manually inspect the excerpts rather than relying on the generated summary.")

        domain = self._infer_workspace_domain(
            intake=intake or {},
            combined=" ".join((issue_outline or []) + evidence_gaps + [authority_map.get("rationale") or ""]),
        )
        steps.extend(DOMAIN_TRIAGE_STEPS.get(domain, []))

        deduped: list[str] = []
        for item in steps:
            if item not in deduped:
                deduped.append(item)
        return deduped[:4]

    def build_headline(
        self,
        *,
        workflow: str,
        intake: dict[str, Any] | None = None,
        question: str | None = None,
        topic_query: str | None = None,
    ) -> str:
        intake = intake or {}
        if workflow == "triage":
            case_type = normalize_whitespace(intake.get("case_type"))
            return f"{case_type} triage workspace" if case_type else "Case triage workspace"
        if workflow == "ask":
            return f"Current evidence for: {shorten_text(question or 'Legal question', 72)}"
        return f"Research workspace: {shorten_text(topic_query or 'Topic', 72)}"

    def build_research_snapshot(
        self,
        *,
        topic_query: str,
        authority_map: dict[str, Any],
        issue_outline: list[str],
        evidence_gaps: list[str],
    ) -> str:
        supporting = authority_map.get("supporting") or []
        conflicting = authority_map.get("conflicting") or []
        mixed = authority_map.get("mixed") or []

        leading_cases = supporting or mixed
        lines = [
            "#### Topic",
            f"- {normalize_whitespace(topic_query)}",
            "",
            "#### What the retrieval suggests",
            f"- Main issues surfaced: {', '.join(issue_outline[:3]) if issue_outline else 'No stable issue pattern yet.'}",
        ]
        if leading_cases:
            lines.append(
                "- Best starting authorities: "
                + ", ".join(f"`{item['case_id']}`" for item in leading_cases[:2])
                + "."
            )
        if conflicting:
            lines.append(
                "- Counter-authorities worth checking: "
                + ", ".join(f"`{item['case_id']}`" for item in conflicting[:2])
                + "."
            )
        if not supporting and mixed:
            lines.append(
                "- The leading authorities are mixed, so compare their facts manually before treating this as a settled position."
            )
        if evidence_gaps:
            lines.extend(["", "#### What would improve the result", f"- {evidence_gaps[0]}"])
        return "\n".join(lines)

    @staticmethod
    def _dominant_label(similar_cases: list[dict[str, Any]]) -> str | None:
        labels = [item.get("label_name") for item in similar_cases if item.get("label_name")]
        if not labels:
            return None
        counts = Counter(labels)
        return counts.most_common(1)[0][0]

    @staticmethod
    def _build_authority_rationale(
        *,
        reference_label: str | None,
        supporting: list[dict[str, Any]],
        conflicting: list[dict[str, Any]],
        mixed: list[dict[str, Any]],
    ) -> str:
        if reference_label:
            if supporting and not conflicting and not mixed:
                return (
                    f"All shortlisted authorities currently lean toward '{reference_label}', but the full judgments still need to be checked for factual distinctions."
                )
            return (
                f"Authorities are grouped relative to the leading direction '{reference_label}': "
                f"{len(supporting)} best support, {len(conflicting)} main risks or distinguishable, {len(mixed)} broader same-domain."
            )
        return (
            f"Authorities could not be anchored to one label direction: "
            f"{len(supporting)} best support, {len(conflicting)} main risks or distinguishable, {len(mixed)} broader same-domain."
        )

    @staticmethod
    def _infer_workspace_domain(
        *,
        intake: dict[str, Any],
        combined: str,
    ) -> str:
        case_type = normalize_whitespace((intake or {}).get("case_type")).lower()
        lowered = normalize_whitespace(combined).lower()
        if "tax" in case_type or any(token in lowered for token in ("assessee", "assessment", "addition", "bogus purchase", "income tax")):
            return "tax"
        if "consumer" in case_type or any(token in lowered for token in ("warranty", "refund", "replacement", "service centre", "defective")):
            return "consumer"
        if "motor accident" in case_type or any(token in lowered for token in ("amputation", "mact", "claimant", "prosthetic", "motor accident")):
            return "motor_accident"
        if "university" in case_type or any(token in lowered for token in ("unfair means", "exam", "result", "invigilator")):
            return "education"
        if "service" in case_type or any(token in lowered for token in ("suspension", "disciplinary", "chargesheet", "departmental inquiry", "promotion")):
            return "service"
        if "information" in case_type or any(token in lowered for token in ("rti", "information commission", "pio", "records", "inspection")):
            return "information"
        return ""
