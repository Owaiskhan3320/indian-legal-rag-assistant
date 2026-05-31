from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any

from legal_ai.utils.domain import (
    DATASET_ALIGNED_CASE_LAW_DOMAINS,
    SUPPORTED_PHASE_ONE_DOMAINS,
    build_retrieval_terms,
    extract_case_ids_from_text,
    infer_issue_subtypes,
    infer_query_domain,
)
from legal_ai.utils.text import normalize_whitespace


SIMILARITY_MARKERS = (
    "find similar",
    "similar cases",
    "similar judgments",
    "closest cases",
    "closest judgments",
    "nearest case",
)

CASE_EXPLANATION_MARKERS = (
    "explain this case",
    "explain this judgment",
    "explain the case",
    "summarize this judgment",
    "summarize the case",
    "facts of",
    "holding in",
    "ratio of",
)

STATUTE_MARKERS = (
    "section ",
    "article ",
    "rule ",
    "under the act",
    "under the code",
    "statute",
    "provision",
    "legal provision",
    "binding precedent",
    "ratio decidendi",
    "doctrinal basis",
)

COMPARATIVE_MARKERS = (
    "compare",
    "distinguish",
    "difference between",
    "factors usually",
    "how do courts usually",
    "when do courts",
    "what facts usually",
    "reasoning used",
    "what changes the outcome",
)

EXACT_PROVISION_PREFIXES = (
    "what does article",
    "what is article",
    "what does section",
    "what is section",
    "what does rule",
    "what is rule",
    "which article",
    "which section",
    "which rule",
)

LAW_ONLY_PATTERNS = (
    "what remedies",
    "what remedy",
    "time limit",
    "limitation period",
    "first appeal",
    "second appeal",
    "rti application",
    "not answered",
    "no response",
    "no reply",
    "within 30 days",
    "personal information",
    "personal data",
    "privacy law",
    "data protection",
    "without consent",
    "cctv",
    "surveillance",
    "commercial confidence",
    "trade secret",
    "intellectual property",
    "fiduciary",
    "certified copies",
    "inspect records",
    "inspection of records",
    "major penalty",
    "minor penalty",
    "difference between rule 14 and rule 16",
    "right to education",
    "right to property",
    "deprived of property",
    "authority of law",
    "personal liberty",
    "wrong product",
    "return window",
)

LAW_GUIDANCE_DOMAINS = {
    "consumer",
    "information",
    "service",
    "tax",
    "motor_accident",
    "constitutional",
    "criminal",
    "privacy",
}

DOCUMENT_FACT_MARKERS = (
    "who were the parties",
    "which court",
    "what were the facts",
    "what happened",
    "what was the issue",
    "what relief",
    "what amount",
    "what compensation",
    "page ",
)

DOCUMENT_REASONING_MARKERS = (
    "this judgment",
    "this document",
    "the court held",
    "why did the court",
    "how did the court",
    "explain the reasoning",
)

SIMPLE_STYLE_MARKERS = (
    "simplify",
    "simple language",
    "plain english",
    "plain language",
    "easy language",
    "easy words",
)

DETAILED_STYLE_MARKERS = (
    "in detail",
    "detailed explanation",
    "explain in detail",
    "deep analysis",
    "research note",
    "step by step",
    "compare carefully",
)

SHORT_STYLE_MARKERS = (
    "short answer",
    "brief answer",
    "briefly",
    "in short",
    "quick answer",
    "just answer shortly",
)

PRACTICAL_ADVICE_MARKERS = (
    "can i ",
    "should i ",
    "what should i do",
    "what can i do",
    "what can ",
    "what are my options",
    "what legal options",
    "how do i proceed",
    "how to proceed",
    "can i complain",
    "against whom",
    "collectively file",
    "next step",
    "next steps",
    "legal notice",
    "file a complaint",
    "file a case",
    "where should i file",
    "where can i complain",
    "what documents do i need",
    "what evidence should i keep",
)

RESEARCH_NOTE_MARKERS = (
    "research note",
    "memo",
    "memorandum",
    "brief note",
    "for my note",
    "for my brief",
    "lawyer preparing",
)

OUTCOME_PATTERN_MARKERS = (
    "accepted vs rejected",
    "accepted versus rejected",
    "rejected vs accepted",
    "rejected versus accepted",
    "successful vs unsuccessful",
    "successful versus unsuccessful",
    "unsuccessful vs successful",
    "unsuccessful versus successful",
    "allowed vs dismissed",
    "allowed versus dismissed",
    "dismissed vs allowed",
    "dismissed versus allowed",
    "outcome patterns",
    "patterns appear in accepted",
    "patterns appear in rejected",
)

LEGAL_ELEMENT_RULES: dict[str, tuple[str, ...]] = {
    "refund": ("refund", "repayment", "return price"),
    "replacement": ("replacement", "replace"),
    "repair": ("repair", "service center", "repairable"),
    "manufacturing_defect": ("manufacturing defect", "inherent defect", "defective product"),
    "deficiency_in_service": ("deficiency in service", "deficient service", "service deficiency"),
    "unfair_trade_practice": ("unfair trade practice", "misleading", "false representation"),
    "component_failure": ("component", "compressor", "part failed"),
    "functional_disability": ("functional disability", "loss of earning capacity", "earning capacity"),
    "physical_disability": ("physical disability", "permanent disability", "amputation"),
    "future_medical_costs": ("future treatment", "future medical", "recurring medical", "prosthetic", "attendant"),
    "natural_justice": ("natural justice", "hearing", "show cause", "opportunity"),
    "unfair_means": ("unfair means", "cheating", "exam result", "invigilator"),
    "documentary_sufficiency": ("unsatisfactory documents", "invoices", "bank statements", "ledger"),
}

TASK_SECTION_PREFERENCES = {
    "exact_provision_lookup": ["holding", "reasoning"],
    "procedure_or_remedy": ["holding", "reasoning", "relief"],
    "fact_pattern_guidance": ["reasoning", "holding", "relief"],
    "similarity_lookup": ["facts", "issue", "reasoning"],
    "case_explanation": ["facts", "issue", "reasoning", "relief"],
    "comparative_reasoning": ["reasoning", "holding", "relief"],
    "statute_question": ["issue", "reasoning", "holding", "relief"],
    "document_fact": ["facts", "relief"],
    "document_reasoning": ["reasoning", "holding"],
    "general_research": ["reasoning", "issue", "holding"],
}


@dataclass
class QueryProfile:
    workflow: str
    lane: str
    task: str
    domain: str | None
    domain_confidence: float
    complexity: str
    answer_style: str
    response_plan: str
    response_length: str
    statute_sensitive: bool
    direct_case_lookup: bool
    supported_case_law_domain: bool
    legal_elements: list[str] = field(default_factory=list)
    issue_subtypes: list[str] = field(default_factory=list)
    issue_tags: list[str] = field(default_factory=list)
    retrieval_terms: list[str] = field(default_factory=list)
    exact_terms: list[str] = field(default_factory=list)
    remedy_terms: list[str] = field(default_factory=list)
    preferred_sections: list[str] = field(default_factory=list)
    referenced_case_ids: list[str] = field(default_factory=list)
    route_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class QueryRouterService:
    def analyze(
        self,
        *,
        question: str,
        chat_history: list[dict[str, str]],
        session_has_document: bool,
        requested_source_mode: str,
        case_type_hint: str | None = None,
        forum_hint: str | None = None,
        context_note: str | None = None,
    ) -> QueryProfile:
        cleaned = normalize_whitespace(question)
        lowered = cleaned.lower()
        referenced_case_ids = extract_case_ids_from_text(cleaned)
        query_profile = infer_query_domain(cleaned, case_type_hint=case_type_hint, referenced_case_ids=referenced_case_ids)
        domain = query_profile.get("domain")
        domain_confidence = float(query_profile.get("confidence") or 0.0)

        direct_case_lookup = bool(referenced_case_ids)
        exact_provision_lookup = self._is_exact_provision_lookup(cleaned=cleaned, lowered=lowered)
        law_first_query = self._is_law_first_query(cleaned=cleaned, lowered=lowered, domain=str(domain) if domain else None)
        practical_guidance = any(marker in lowered for marker in PRACTICAL_ADVICE_MARKERS)
        statute_sensitive = exact_provision_lookup or law_first_query or any(marker in lowered for marker in STATUTE_MARKERS)
        complexity = self._infer_complexity(cleaned, lowered)
        answer_style = self._infer_answer_style(lowered)
        response_length = self._infer_response_length(
            cleaned=cleaned,
            lowered=lowered,
            answer_style=answer_style,
            complexity=complexity,
        )

        if requested_source_mode == "document_only" and session_has_document:
            task = self._document_task(lowered)
            lane = "document"
            workflow = "document_qa"
            route_reason = "document-only mode with active uploaded document"
        elif session_has_document and (
            any(marker in lowered for marker in DOCUMENT_FACT_MARKERS)
            or any(marker in lowered for marker in DOCUMENT_REASONING_MARKERS)
        ):
            task = self._document_task(lowered)
            lane = "document"
            workflow = "document_qa"
            route_reason = "question explicitly refers to uploaded-document facts or reasoning"
        elif any(marker in lowered for marker in SIMILARITY_MARKERS):
            task = "similarity_lookup"
            lane = "case_law"
            workflow = "case_qa"
            route_reason = "question asks for similar judgments"
        elif direct_case_lookup or any(marker in lowered for marker in CASE_EXPLANATION_MARKERS):
            task = "case_explanation"
            lane = "case_law"
            workflow = "case_qa"
            route_reason = "question focuses on a named case/judgment"
        elif exact_provision_lookup:
            task = "exact_provision_lookup"
            lane = "reference_law"
            workflow = "case_qa"
            route_reason = "question asks for an exact article, section, or rule"
        elif (law_first_query and practical_guidance) or (
            practical_guidance and str(domain) in LAW_GUIDANCE_DOMAINS
        ):
            task = "fact_pattern_guidance"
            lane = "statute_case_hybrid"
            workflow = "case_qa"
            route_reason = "question asks for practical guidance in a supported legal domain and should be grounded in law first"
        elif law_first_query or statute_sensitive:
            task = "procedure_or_remedy"
            lane = "reference_law"
            workflow = "case_qa"
            route_reason = "question asks for statutory procedure, remedy, right, or doctrinal grounding"
        elif any(marker in lowered for marker in COMPARATIVE_MARKERS):
            task = "comparative_reasoning"
            lane = "case_law"
            workflow = "case_qa"
            route_reason = "question asks for comparative legal reasoning"
        else:
            task = "general_research"
            lane = "case_law" if requested_source_mode != "document_only" else "document"
            workflow = "document_qa" if lane == "document" else "case_qa"
            route_reason = "default legal research path"

        legal_elements = self._extract_legal_elements(
            question=cleaned,
            case_type_hint=case_type_hint,
            forum_hint=forum_hint,
            context_note=context_note,
        )
        issue_subtypes = infer_issue_subtypes(
            " ".join(part for part in [cleaned, case_type_hint or "", context_note or ""] if part),
            domain=str(domain) if domain else None,
        )
        issue_tags = self._build_issue_tags(
            domain=domain,
            task=task,
            legal_elements=legal_elements,
            issue_subtypes=issue_subtypes,
        )
        retrieval_terms = build_retrieval_terms(
            question=cleaned,
            domain=str(domain) if domain else None,
            legal_elements=legal_elements,
            case_type_hint=case_type_hint,
            forum_hint=forum_hint,
        )
        exact_terms = self._extract_exact_terms(cleaned)
        remedy_terms = self._extract_remedy_terms(cleaned, legal_elements=legal_elements)
        preferred_sections = list(TASK_SECTION_PREFERENCES.get(task) or TASK_SECTION_PREFERENCES["general_research"])
        response_plan = self._infer_response_plan(
            lowered=lowered,
            task=task,
            answer_style=answer_style,
            response_length=response_length,
        )
        supported_case_law_domain = bool(
            direct_case_lookup
            or task in {"document_fact", "document_reasoning", "case_explanation", "similarity_lookup"}
            or not domain
            or domain in SUPPORTED_PHASE_ONE_DOMAINS
            or (task == "general_research" and domain in DATASET_ALIGNED_CASE_LAW_DOMAINS)
        )

        return QueryProfile(
            workflow=workflow,
            lane=lane,
            task=task,
            domain=str(domain) if domain else None,
            domain_confidence=domain_confidence,
            complexity=complexity,
            answer_style=answer_style,
            response_plan=response_plan,
            response_length=response_length,
            statute_sensitive=statute_sensitive,
            direct_case_lookup=direct_case_lookup,
            supported_case_law_domain=supported_case_law_domain,
            legal_elements=legal_elements,
            issue_subtypes=issue_subtypes,
            issue_tags=issue_tags,
            retrieval_terms=retrieval_terms,
            exact_terms=exact_terms,
            remedy_terms=remedy_terms,
            preferred_sections=preferred_sections,
            referenced_case_ids=referenced_case_ids[:5],
            route_reason=route_reason,
        )

    @staticmethod
    def _infer_complexity(cleaned: str, lowered: str) -> str:
        question_count = max(cleaned.count("?"), 0)
        if question_count >= 2 or any(
            marker in lowered for marker in ("compare", "distinguish", "in detail", "justify", "reconcile")
        ):
            return "deep"
        if len(re.findall(r"\b\w+\b", cleaned)) > 28:
            return "deep"
        return "fast"

    @staticmethod
    def _infer_answer_style(lowered: str) -> str:
        if any(marker in lowered for marker in SIMPLE_STYLE_MARKERS):
            return "simple"
        if any(marker in lowered for marker in SHORT_STYLE_MARKERS):
            return "short"
        if any(marker in lowered for marker in DETAILED_STYLE_MARKERS):
            return "detailed"
        return "structured"

    @staticmethod
    def _infer_response_length(
        *,
        cleaned: str,
        lowered: str,
        answer_style: str,
        complexity: str,
    ) -> str:
        if any(marker in lowered for marker in SHORT_STYLE_MARKERS):
            return "short"
        if answer_style == "simple":
            return "short"
        if answer_style == "detailed" or complexity == "deep":
            return "long"
        if len(re.findall(r"\b\w+\b", cleaned)) <= 16:
            return "short"
        return "medium"

    @staticmethod
    def _infer_response_plan(
        *,
        lowered: str,
        task: str,
        answer_style: str,
        response_length: str,
    ) -> str:
        if task == "similarity_lookup":
            return "similar_case_list"
        if task in {"case_explanation", "document_fact", "document_reasoning"}:
            return "case_brief_simple" if answer_style == "simple" else "case_brief"
        if any(marker in lowered for marker in OUTCOME_PATTERN_MARKERS):
            return "outcome_pattern"
        if any(marker in lowered for marker in RESEARCH_NOTE_MARKERS):
            return "research_note"
        if any(marker in lowered for marker in PRACTICAL_ADVICE_MARKERS):
            return "practical_steps"
        if task == "comparative_reasoning":
            return "comparative_analysis"
        if task == "exact_provision_lookup":
            return "exact_law_answer"
        if task in {"procedure_or_remedy", "statute_question"}:
            return "doctrinal_answer"
        if task == "fact_pattern_guidance":
            return "practical_steps"
        if answer_style == "simple":
            return "plain_guidance"
        if response_length == "short":
            return "direct_guidance"
        return "reasoned_analysis"

    @staticmethod
    def _document_task(lowered: str) -> str:
        if any(marker in lowered for marker in DOCUMENT_FACT_MARKERS):
            return "document_fact"
        return "document_reasoning"

    @staticmethod
    def _is_exact_provision_lookup(*, cleaned: str, lowered: str) -> bool:
        if any(lowered.startswith(prefix) for prefix in EXACT_PROVISION_PREFIXES):
            return True
        if re.search(r"\b(?:article|section|rule)\s+\d+[A-Za-z]?(?:\([^)]+\))*\b", lowered, flags=re.I):
            head = " ".join(lowered.split()[:8])
            if any(token in head for token in ("what", "which", "difference", "explain", "define")):
                return True
        return False

    @staticmethod
    def _is_law_first_query(*, cleaned: str, lowered: str, domain: str | None) -> bool:
        if any(pattern in lowered for pattern in LAW_ONLY_PATTERNS):
            return True
        if any(lowered.startswith(prefix) for prefix in ("what does", "what is", "which rule", "which section", "which article")):
            if domain in {"consumer", "information", "service", "tax", "motor_accident", "criminal", "constitutional", "privacy"}:
                return True
        if any(marker in lowered for marker in ("under the rti act", "under the consumer protection act", "under ccs cca rules", "under the constitution")):
            return True
        if (
            "article 21a" in lowered
            or "article 300a" in lowered
            or ("property" in lowered and "authority of law" in lowered)
            or "deprived of property" in lowered
        ):
            return True
        return False

    @staticmethod
    def _extract_exact_terms(question: str) -> list[str]:
        cleaned = normalize_whitespace(question)
        lowered = cleaned.lower()
        exact_terms: list[str] = []
        for quoted in re.findall(r'"([^"]{2,120})"', cleaned):
            normalized = normalize_whitespace(quoted)
            if normalized and normalized not in exact_terms:
                exact_terms.append(normalized)
        for match in re.findall(r"\b(?:section|article|rule)\s+\d+[A-Za-z]?(?:\(\d+[A-Za-z]?\))*\b", lowered, flags=re.I):
            normalized = normalize_whitespace(match)
            if normalized and normalized not in exact_terms:
                exact_terms.append(normalized)
        for phrase in (
            "consumer rights",
            "misleading advertisement",
            "consumer dispute",
            "natural justice",
            "bogus purchases",
            "non-genuine purchases",
            "departmental inquiry",
            "suspension",
            "refund",
            "replacement",
            "repair",
        ):
            if phrase in lowered and phrase not in exact_terms:
                exact_terms.append(phrase)
        return exact_terms[:8]

    @staticmethod
    def _extract_remedy_terms(question: str, *, legal_elements: list[str]) -> list[str]:
        lowered = normalize_whitespace(question).lower()
        remedy_terms: list[str] = []
        remedy_map = {
            "refund": ("refund", "repay", "return price"),
            "replacement": ("replacement", "replace"),
            "repair": ("repair",),
            "compensation": ("compensation", "damages", "award"),
            "quash": ("quash", "set aside", "revoke"),
            "deletion": ("delete addition", "deletion of addition", "delete the addition"),
            "disclosure": ("provide copies", "disclosure", "inspection", "information"),
        }
        for label, phrases in remedy_map.items():
            if any(phrase in lowered for phrase in phrases):
                remedy_terms.append(label)
        for element in legal_elements:
            if element in {"refund", "replacement", "repair"} and element not in remedy_terms:
                remedy_terms.append(element)
        return remedy_terms[:5]

    @staticmethod
    def _extract_legal_elements(
        *,
        question: str,
        case_type_hint: str | None,
        forum_hint: str | None,
        context_note: str | None,
    ) -> list[str]:
        combined = " ".join(
            part for part in [question, case_type_hint or "", forum_hint or "", context_note or ""] if part
        ).lower()
        elements: list[str] = []
        for label, phrases in LEGAL_ELEMENT_RULES.items():
            if any(phrase in combined for phrase in phrases):
                elements.append(label)
        if case_type_hint:
            elements.append(f"case_type:{normalize_whitespace(case_type_hint).lower()}")
        if forum_hint:
            elements.append(f"forum:{normalize_whitespace(forum_hint).lower()}")
        return elements[:8]

    @staticmethod
    def _build_issue_tags(
        *,
        domain: str | None,
        task: str,
        legal_elements: list[str],
        issue_subtypes: list[str],
    ) -> list[str]:
        issue_tags: list[str] = []
        if domain:
            issue_tags.append(f"domain:{domain}")
        issue_tags.append(f"task:{task}")
        for element in legal_elements[:5]:
            issue_tags.append(element)
        for subtype in issue_subtypes[:3]:
            issue_tags.append(f"subtype:{subtype}")
        return issue_tags[:8]
