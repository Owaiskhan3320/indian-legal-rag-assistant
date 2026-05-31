from __future__ import annotations

import logging
import re
from typing import Any

import httpx

from legal_ai.config import Settings
from legal_ai.utils.text import normalize_whitespace, shorten_text


LOGGER = logging.getLogger(__name__)

REVIEW_DOMAIN_GUIDANCE = {
    "tax": {
        "risk_focus": "supplier credibility, documentary sufficiency, goods-movement proof, third-party statements, and cross-examination issues",
        "verification_focus": "supplier confirmations, transport or stock evidence, books/banking trail, and whether the assessment rests only on suspicion",
        "avoid": "service-law suspension themes, consumer warranty language, or motor-accident quantum factors",
    },
    "consumer": {
        "risk_focus": "defect timing, repeated failed repairs, warranty delay, control over remedy, and whether the product stayed unusable",
        "verification_focus": "delivery timeline, complaint chronology, service-centre records, photos, and who actually refused refund or replacement",
        "avoid": "tax-documentary language, disciplinary-proceeding themes, or motor-accident compensation factors",
    },
    "motor_accident": {
        "risk_focus": "functional disability, future-treatment proof, income evidence, and causation for recurring expenses",
        "verification_focus": "disability certificates, prosthetic/future-treatment estimates, salary proof, and the difference between medical and functional disability",
        "avoid": "consumer warranty disputes, supplier-proof language, or RTI/disclosure themes",
    },
    "education": {
        "risk_focus": "quality of recovery evidence, hearing defects, procedural fairness, and proportionality of the academic penalty",
        "verification_focus": "show-cause notice, invigilator report, reply, proof of direct recovery, and whether similarly placed students were treated differently",
        "avoid": "tax-purchase themes, supplier-proof language, or service-retiral-benefit reasoning",
    },
    "service": {
        "risk_focus": "suspension basis, delay in inquiry, procedural safeguards, retaliatory motive, and retirement-related prejudice",
        "verification_focus": "suspension order, chargesheet, reply chronology, applicable service rules, and whether the relief sought is interim or final",
        "avoid": "supplier-proof or goods-movement themes, consumer defect language, or RTI copy-inspection disputes",
    },
    "information": {
        "risk_focus": "actual exemption grounds, adequacy of the PIO reply, inspection-versus-copies issues, and whether the records are identifiable",
        "verification_focus": "RTI application wording, PIO reply, first-appeal papers, and whether the refusal rests on a real exemption or only inconvenience",
        "avoid": "tax-addition language, consumer warranty disputes, or disciplinary-proceeding themes",
    },
}


class ExplanationService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.enabled = settings.llm_enabled

    def explain(
        self,
        case_text: str,
        prediction: dict,
        similar_cases: list[dict],
        *,
        intake: dict,
        favorability_label: str,
        favorability_reason: str,
        authority_map: dict[str, Any] | None = None,
        issue_outline: list[str] | None = None,
        evidence_gaps: list[str] | None = None,
    ) -> dict[str, str]:
        try:
            text = self._build_review_summary(
                case_text=case_text,
                prediction=prediction,
                similar_cases=similar_cases,
                intake=intake,
                favorability_label=favorability_label,
                favorability_reason=favorability_reason,
                authority_map=authority_map or {},
                issue_outline=issue_outline or [],
                evidence_gaps=evidence_gaps or [],
            )
            return {
                "text": self._normalize_review_summary(text),
                "source": "structured_review",
            }
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Structured review summary failed, using fallback. error=%s", exc)
        return {
            "text": self._fallback(
                case_text,
                prediction,
                similar_cases,
                intake=intake,
                favorability_label=favorability_label,
                favorability_reason=favorability_reason,
                authority_map=authority_map or {},
                issue_outline=issue_outline or [],
                evidence_gaps=evidence_gaps or [],
            ),
            "source": "fallback",
        }

    def _build_review_summary(
        self,
        *,
        case_text: str,
        prediction: dict[str, Any],
        similar_cases: list[dict[str, Any]],
        intake: dict[str, Any],
        favorability_label: str,
        favorability_reason: str,
        authority_map: dict[str, Any],
        issue_outline: list[str],
        evidence_gaps: list[str],
    ) -> str:
        predicted_name = str(prediction.get("predicted_name") or "Unclear")
        confidence_score = float(prediction.get("confidence_score") or 0.0)
        support_cases = list(authority_map.get("supporting") or [])
        risk_cases = list(authority_map.get("conflicting") or [])
        mixed_cases = list(authority_map.get("mixed") or [])
        review_domain = self._infer_review_domain(intake=intake, issue_outline=issue_outline)
        issue_bits = [item for item in issue_outline if item and not item.lower().startswith("core matter:")]
        bottom_line = self._build_bottom_line(
            predicted_name=predicted_name,
            favorability_label=favorability_label,
            confidence_score=confidence_score,
            review_domain=review_domain,
        )
        helps = self._build_helpful_factors(
            intake=intake,
            issue_outline=issue_bits,
            support_cases=support_cases,
        )
        risks = self._build_risk_factors(
            intake=intake,
            issue_outline=issue_bits,
            risk_cases=risk_cases or mixed_cases,
        )
        principle = self._build_simple_legal_principle(
            review_domain=review_domain,
            issue_outline=issue_bits,
        )
        case_blocks = self._build_review_case_blocks(
            support_cases=support_cases,
            risk_cases=risk_cases or mixed_cases,
        )
        practical_outcome = self._build_practical_outcome(
            review_domain=review_domain,
            predicted_name=predicted_name,
            confidence_score=confidence_score,
            favorability_label=favorability_label,
        )
        next_steps = self._build_next_steps_block(
            review_domain=review_domain,
            evidence_gaps=evidence_gaps,
            verification_focus=self._review_verification_focus(
                review_domain=review_domain,
                issue_outline=issue_bits,
                evidence_gaps=evidence_gaps,
                favorability_reason=favorability_reason,
            ),
        )
        confidence_line = self._build_confidence_line(
            confidence_score=confidence_score,
            similar_cases=similar_cases,
            risk_cases=risk_cases or mixed_cases,
        )

        sections = [
            f"**Bottom line**\n\n{bottom_line}",
            f"**What helps your case**\n\n{helps}",
            f"**What may weaken your case**\n\n{risks}",
            f"**Relevant legal principle**\n\n{principle}",
            f"**Supporting cases**\n\n{case_blocks}",
            f"**Practical outcome**\n\n{practical_outcome}",
            f"**What you should do next**\n\n{next_steps}",
            f"**Confidence**\n\n{confidence_line}",
        ]
        return "\n\n".join(sections)

    def answer_question(
        self,
        *,
        question: str,
        retrieval_query: str,
        similar_cases: list[dict],
        rag_context: dict[str, Any],
        evidence_pack: dict[str, Any] | None = None,
        question_profile: dict[str, Any] | None = None,
        detected_language: str,
        answer_language: str,
        filters: dict[str, Any],
        chat_history: list[dict[str, str]] | None = None,
        scope: str = "corpus",
        source_mode: str = "document_plus_case",
        retrieval_profile: str = "fast",
    ) -> dict[str, str]:
        if self.enabled:
            try:
                text = self._answer_with_llm(
                    question=question,
                    retrieval_query=retrieval_query,
                    similar_cases=similar_cases,
                    rag_context=rag_context,
                    evidence_pack=evidence_pack or {},
                    question_profile=question_profile or {},
                    detected_language=detected_language,
                    answer_language=answer_language,
                    filters=filters,
                    chat_history=chat_history or [],
                    scope=scope,
                    source_mode=source_mode,
                    retrieval_profile=retrieval_profile,
                )
                return {
                    "text": self._normalize_answer_text(text),
                    "source": "llm",
                }
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("LLM QA failed, using fallback. error=%s", exc)
        return {
            "text": self._answer_fallback(
                question=question,
                retrieval_query=retrieval_query,
                similar_cases=similar_cases,
                rag_context=rag_context,
                evidence_pack=evidence_pack or {},
                question_profile=question_profile or {},
                answer_language=answer_language,
                filters=filters,
                source_mode=source_mode,
            ),
            "source": "fallback",
        }

    def _explain_with_llm(
        self,
        case_text: str,
        prediction: dict,
        similar_cases: list[dict],
        *,
        intake: dict,
        favorability_label: str,
        favorability_reason: str,
        authority_map: dict[str, Any],
        issue_outline: list[str],
        evidence_gaps: list[str],
    ) -> str:
        return self._run_chat_completion(
            system_prompt=self._build_system_prompt(),
            user_prompt=self._build_user_prompt(
                case_text=case_text,
                prediction=prediction,
                similar_cases=similar_cases,
                intake=intake,
                favorability_label=favorability_label,
                favorability_reason=favorability_reason,
                authority_map=authority_map,
                issue_outline=issue_outline,
                evidence_gaps=evidence_gaps,
            ),
            max_tokens=self.settings.llm_triage_summary_tokens,
        )

    def _answer_with_llm(
        self,
        *,
        question: str,
        retrieval_query: str,
        similar_cases: list[dict],
        rag_context: dict[str, Any],
        evidence_pack: dict[str, Any],
        question_profile: dict[str, Any],
        detected_language: str,
        answer_language: str,
        filters: dict[str, Any],
        chat_history: list[dict[str, str]],
        scope: str,
        source_mode: str,
        retrieval_profile: str,
    ) -> str:
        return self._run_chat_completion(
            system_prompt=self._build_qa_system_prompt(
                answer_language=answer_language,
                scope=scope,
                source_mode=source_mode,
                retrieval_profile=retrieval_profile,
                question_profile=question_profile,
                evidence_pack=evidence_pack,
            ),
            user_prompt=self._build_qa_user_prompt(
                question=question,
                retrieval_query=retrieval_query,
                similar_cases=similar_cases,
                rag_context=rag_context,
                evidence_pack=evidence_pack,
                question_profile=question_profile,
                detected_language=detected_language,
                answer_language=answer_language,
                filters=filters,
                chat_history=chat_history,
                scope=scope,
                source_mode=source_mode,
                retrieval_profile=retrieval_profile,
            ),
            max_tokens=(
                self.settings.llm_deep_answer_tokens
                if retrieval_profile == "deep"
                else self.settings.llm_fast_answer_tokens
            ),
        )

    def _run_chat_completion(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float = 0.05,
    ) -> str:
        timeout = httpx.Timeout(
            connect=10.0,
            read=float(self.settings.llm_timeout_seconds),
            write=20.0,
            pool=20.0,
        )
        with httpx.Client(timeout=timeout) as client:
            model_name = self._resolve_model_name(client)
            response = client.post(
                self.settings.llm_base_url.rstrip("/") + "/chat/completions",
                headers=self._build_headers(),
                json={
                    "model": model_name,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                },
            )
            response.raise_for_status()
            payload = response.json()
            return payload["choices"][0]["message"]["content"].strip()

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        api_key = self.settings.llm_api_key.strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _resolve_model_name(self, client: httpx.Client) -> str:
        configured = self.settings.llm_model.strip()
        if configured.lower() != "auto":
            return configured

        response = client.get(
            self.settings.llm_base_url.rstrip("/") + "/models",
            headers=self._build_headers(),
        )
        response.raise_for_status()
        payload = response.json()
        models = payload.get("data") or []
        if not models:
            raise RuntimeError("No local LLM models are available from the configured server.")
        return str(models[0]["id"])

    def translate_legal_query(self, *, text: str, source_language: str) -> str:
        cleaned = normalize_whitespace(text)
        if not cleaned or source_language == "English" or not self.enabled:
            return cleaned

        try:
            translated = self._run_chat_completion(
                system_prompt=(
                    "You rewrite legal research questions into concise English retrieval queries. "
                    "Preserve legal meaning, factual nuance, procedure, forum, and remedy. "
                    "Do not answer the question. Return plain English only."
                ),
                user_prompt=(
                    f"Source language: {source_language}\n"
                    f"Question:\n{cleaned}\n\n"
                    "Return a single English retrieval query."
                ),
                max_tokens=self.settings.llm_translation_max_tokens,
                temperature=0.0,
            )
            return normalize_whitespace(translated).strip("\"'")
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("LLM query translation failed, using original query. error=%s", exc)
            return cleaned

    def probe(self) -> tuple[bool, str]:
        if not self.enabled:
            return False, "Local LLM is not configured."

        timeout = httpx.Timeout(connect=4.0, read=4.0, write=4.0, pool=4.0)
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.get(
                    self.settings.llm_base_url.rstrip("/") + "/models",
                    headers=self._build_headers(),
                )
                response.raise_for_status()
                models = response.json().get("data") or []
                if not models:
                    return False, "Local LLM server is reachable, but no model is exposed."
                model_ids = {str(item.get("id")) for item in models if item.get("id")}
                configured = self.settings.llm_model.strip()
                if configured.lower() != "auto" and configured not in model_ids:
                    return (
                        False,
                        f"Configured model '{configured}' is not exposed by the local LLM server.",
                    )
                active_model = configured if configured.lower() != "auto" else str(models[0]["id"])
                return True, f"Connected to local model '{active_model}'."
        except Exception as exc:  # pragma: no cover
            return False, f"Local LLM check failed: {exc}"

    @staticmethod
    def _build_system_prompt() -> str:
        return (
            "You are the explanation layer of a legal AI workbench. "
            "The classifier prediction is fixed. Do not change it. "
            "Explain only from the intake, prediction block, and retrieved cases. "
            "Do not invent facts, statutes, or holdings. "
            "If support is mixed or weak, say that clearly. "
            "Do not import risk factors from a different legal domain. "
            "For example, do not mention supplier proof in a service case, and do not mention suspension procedure in a tax case unless the evidence explicitly supports it. "
            "Return clean Markdown using exactly these level-4 headings:\n"
            "#### 1. Predicted Outcome\n"
            "#### 2. Why This Side Currently Has Support\n"
            "#### 3. Strongest Authorities For This Direction\n"
            "#### 4. Main Risks Or Distinguishing Facts\n"
            "#### 5. What To Verify Next\n"
            "#### 6. Confidence And Caution\n"
            "Under each heading, write 2 to 4 complete bullet points. "
            "Be concrete about documents, factual weaknesses, and what the other side may argue. "
            "Avoid vague filler like 'similar facts and reasoning' without saying what those facts are. "
            "Keep the answer detailed but still practical and readable. "
            "If the factual fit of the top authority is only broad or moderate, say that directly instead of overstating likely success."
        )

    @staticmethod
    def _build_qa_system_prompt(
        *,
        answer_language: str,
        scope: str,
        source_mode: str,
        retrieval_profile: str,
        question_profile: dict[str, Any],
        evidence_pack: dict[str, Any],
    ) -> str:
        statute_support = bool(evidence_pack.get("statute_support_available"))
        if source_mode == "document_only":
            scope_label = "the uploaded document excerpts prepared for this turn"
        elif source_mode == "reference_law_only":
            scope_label = "the official law materials prepared for this turn"
        elif source_mode == "reference_law_plus_case":
            scope_label = "the official law materials prepared for this turn, with optional case-law support"
        elif source_mode == "document_plus_reference_law":
            scope_label = "the uploaded document excerpts plus the official law materials prepared for this turn"
        elif source_mode == "document_plus_reference_law_plus_case":
            scope_label = "the uploaded document excerpts plus the official law materials prepared for this turn, with optional case-law support"
        elif source_mode == "case_corpus_only":
            if statute_support:
                scope_label = (
                    "the full case library plus any official law materials retrieved for this turn"
                    if scope == "corpus"
                    else "the currently selected evidence set plus any official law materials retrieved for this turn"
                )
            else:
                scope_label = (
                    "the full case library"
                    if scope == "corpus"
                    else "the currently selected evidence set"
                )
        else:
            if statute_support:
                scope_label = (
                    "the uploaded document excerpts plus official law materials and the full case library"
                    if scope == "corpus"
                    else "the uploaded document excerpts plus official law materials and the currently selected evidence set"
                )
            else:
                scope_label = (
                    "the uploaded document excerpts plus the full case library"
                    if scope == "corpus"
                    else "the uploaded document excerpts plus the currently selected evidence set"
                )
        answer_style = str(question_profile.get("answer_style") or "structured")
        style_rule = (
            "Keep the answer compact and decisive. Use one short opening answer and only the most useful support."
            if retrieval_profile == "fast"
            else "Give a fuller explanation in short sections when the evidence supports it."
        )
        route_rule = (
            f"Question task: {question_profile.get('task') or 'general_research'}. "
            f"Preferred lane: {question_profile.get('lane') or 'case_law'}. "
            f"Domain: {question_profile.get('domain') or 'unspecified'}. "
        )
        presentation_rule = {
            "simple": (
                "Use plain English and explain legal terms in everyday language. "
                "Prefer short sentences and avoid dense legal jargon."
            ),
            "short": (
                "Give a short, direct answer first. "
                "Only add the minimum explanation needed to keep it grounded and useful."
            ),
            "detailed": (
                "Give a careful research-note style explanation with slightly more detail, but keep each bullet focused."
            ),
        }.get(
            answer_style,
            "Keep the answer structured and practical rather than conversational or essay-like.",
        )
        statute_rule = (
            "No matching provision-level law material was retrieved for this answer, so do not claim a settled statutory rule unless the evidence bundle itself contains direct provision support. "
            if question_profile.get("statute_sensitive") and not evidence_pack.get("statute_support_available")
            else ""
        )
        statute_priority_rule = (
            "Official law materials are present for this turn. For section, article, rule, limitation, procedure, remedy, penalty, jurisdiction, or definition questions, answer from the official law materials first and use case-law only as secondary illustration. "
            if evidence_pack.get("statute_support_available")
            else ""
        )
        answer_shape_rule = ExplanationService._answer_shape_instruction(
            question_profile=question_profile,
            retrieval_profile=retrieval_profile,
        )
        return (
            "You are a grounded Indian legal RAG assistant. "
            "Answer only from the evidence bundle prepared for this turn and the visible chat history. "
            "Do not invent facts, statutes, holdings, or citations. "
            f"The answer must stay within {scope_label}. "
            "Write the answer strictly in English. "
            f"{route_rule}"
            "If support is mixed, say so plainly. "
            "If the evidence is weak, say what is missing. "
            "Prefer direct, user-friendly language over abstract legal jargon. "
            "Return only the final answer text for the user. "
            "Do not include XML or internal notes. "
            f"{style_rule} "
            f"{presentation_rule} "
            f"{answer_shape_rule} "
            "If the user asks about a specific case, explain that case's facts, outcome, and reasoning from the retrieved text. "
            "Mention case IDs inline when they directly support a point. "
            "Cite only authority cards or official law materials that are explicitly listed in the evidence bundle. "
            "Prefer direct-support authority cards over analogical ones, and higher-court authority over lower authority when support quality is otherwise similar. "
            "Separate direct support from analogy when the evidence is mixed. "
            f"{statute_rule}"
            f"{statute_priority_rule}"
            "Keep the answer practical, clear, and grounded. "
            "When the evidence contains an exact name, date, amount, page, or case detail, state it directly instead of hedging. "
            "Do not say a fact is not explicit if it is present in the evidence bundle. "
            "For uploaded-document questions, prefer the uploaded document over generic case-law background unless the user explicitly asks for comparison."
        )

    @staticmethod
    def _answer_shape_instruction(
        *,
        question_profile: dict[str, Any],
        retrieval_profile: str,
    ) -> str:
        plan = str(question_profile.get("response_plan") or "direct_guidance")
        response_length = str(question_profile.get("response_length") or "medium")

        if plan == "practical_steps":
            if response_length == "short":
                return (
                    "Structure the answer like a lawyer responding on a legal-advice platform: "
                    "start with a Bottom line, then 2 to 4 concise Next steps, then one short Limits note. "
                    "Do not force extra sections."
                )
            return (
                "Structure the answer like a practical client note: use short markdown headings for Bottom line, Why this matters, Next steps, Documents or evidence, and Limits. "
                "Lead with the answer, not background."
            )
        if plan in {"case_brief", "case_brief_simple"}:
            return (
                "Structure the answer as a case brief using short markdown headings for Facts, Issue, Decision, Reasoning, and Why it matters. "
                "If the user asked for simple language, keep each section very short and plain."
            )
        if plan == "research_note":
            return (
                "Structure the answer like a short legal research note using markdown headings for Issue, Answer, Key authorities, Analysis, and Open points. "
                "This is the only mode where memo-style structure is preferred."
            )
        if plan == "outcome_pattern":
            return (
                "Structure the answer around outcome patterns: use headings for Answer, What accepted cases tend to show, What rejected cases tend to show, Outcome-determinative differences, and Limits."
            )
        if plan == "comparative_analysis":
            return (
                "Structure the answer for comparison: use headings for Answer, Factors courts look at, What changes the outcome, Key authorities, and Limits."
            )
        if plan == "similar_case_list":
            return (
                "List the closest matches first, then briefly explain why each one is similar. "
                "Keep the structure retrieval-oriented, not memo-like."
            )
        if plan == "doctrinal_answer":
            return (
                "Lead with an Answer section, then explain the available case-law support, then state clearly what is still missing or uncertain. "
                "Do not overstate the law."
            )
        if plan == "exact_law_answer":
            return (
                "Structure the answer like a statute-first legal note: use short headings for Answer, Source used, Why it applies, Next step, and Caution when needed. "
                "For an exact article, section, or rule question, prioritize the official law materials and do not let case-law overshadow the direct provision."
            )
        if response_length == "short" or retrieval_profile == "fast":
            return (
                "If the question calls for a short answer, do not force four headings. "
                "A short answer may be one short conclusion plus 2 or 3 bullets of support."
            )
        return (
            "Use short markdown headings only when they help the reader. "
            "Match the structure to the query instead of forcing the same template every time."
        )

    def _build_user_prompt(
        self,
        *,
        case_text: str,
        prediction: dict[str, Any],
        similar_cases: list[dict[str, Any]],
        intake: dict[str, Any],
        favorability_label: str,
        favorability_reason: str,
        authority_map: dict[str, Any],
        issue_outline: list[str],
        evidence_gaps: list[str],
    ) -> str:
        review_domain = self._infer_review_domain(intake=intake, issue_outline=issue_outline)
        domain_guidance = REVIEW_DOMAIN_GUIDANCE.get(review_domain or "", {})
        sections = [
            "A. User Case Summary",
            self._format_intake_block(intake, case_text),
            "",
            "B. Prediction Block",
            self._format_prediction_block(prediction, favorability_label, favorability_reason),
            "",
            "C. Authority Map",
            self._format_review_authority_groups(authority_map, similar_cases),
            "",
            "D. Triage Signals",
            f"- Domain guardrail: {review_domain or 'unspecified domain'}.",
            f"- Main issues currently surfaced: {', '.join(issue_outline[:4]) if issue_outline else 'No stable issue pattern yet.'}",
            f"- Best support count: {len(authority_map.get('supporting') or [])}",
            f"- Distinguishable or cautionary authorities: {len(authority_map.get('conflicting') or []) + len(authority_map.get('mixed') or [])}",
            f"- Main missing or weak points: {', '.join(evidence_gaps[:2]) if evidence_gaps else 'No major missing point was auto-detected.'}",
            "",
            "E. Rules Block",
            "- Do not change the classifier prediction.",
            "- Explain only using the user case summary, prediction block, and retrieved evidence.",
            "- Do not invent facts or statutes.",
            "- Mention uncertainty when confidence is low, role alignment is unclear, or retrieved support is mixed.",
            "- Compare the current case against the retrieved cases by stating what is similar, what is different, and why that matters to the outcome.",
            "- In section 3, mention up to three retrieved case ids and explain exactly what each one contributes, including whether the factual fit is high, moderate, or only broad.",
            "- In section 4, focus only on the concrete pressure points supported by this domain, the issue outline, the evidence gaps, or the retrieved authority fit notes.",
            "- In section 5, give concrete verification steps that a lawyer or serious user would check next, using the domain-specific verification focus when available.",
            "- Keep every bullet complete; do not leave the sentence unfinished.",
            "- If a retrieved authority is only a broad domain match and not the same issue type, say so clearly.",
        ]
        if domain_guidance.get("risk_focus"):
            sections.append(f"- In section 4, focus on: {domain_guidance['risk_focus']}.")
        if domain_guidance.get("verification_focus"):
            sections.append(f"- In section 5, prioritize: {domain_guidance['verification_focus']}.")
        if domain_guidance.get("avoid"):
            sections.append(f"- Do not import unrelated themes such as {domain_guidance['avoid']}.")
        if not (authority_map.get("conflicting") or authority_map.get("mixed")):
            sections.append(
                "- If no strong counter-authority was surfaced, say that this is a retrieval limitation rather than proof that the other side has no answer."
            )
        return "\n".join(sections)

    def _build_qa_user_prompt(
        self,
        *,
        question: str,
        retrieval_query: str,
        similar_cases: list[dict[str, Any]],
        rag_context: dict[str, Any],
        evidence_pack: dict[str, Any],
        question_profile: dict[str, Any],
        detected_language: str,
        answer_language: str,
        filters: dict[str, Any],
        chat_history: list[dict[str, str]],
        scope: str,
        source_mode: str,
        retrieval_profile: str,
    ) -> str:
        if source_mode == "document_only":
            source_mode_label = "Uploaded document only"
            scope_label = "Uploaded document only"
        elif source_mode == "reference_law_only":
            source_mode_label = "Official law materials only"
            scope_label = "Official law materials only"
        elif source_mode == "reference_law_plus_case":
            source_mode_label = "Official law materials + case support"
            scope_label = "Official law materials with optional case support"
        elif source_mode == "document_plus_reference_law":
            source_mode_label = "Uploaded document + official law materials"
            scope_label = "Uploaded document + official law materials"
        elif source_mode == "document_plus_reference_law_plus_case":
            source_mode_label = "Uploaded document + official law materials + case support"
            scope_label = "Uploaded document + official law materials + case support"
        elif source_mode == "case_corpus_only":
            source_mode_label = "Case corpus only"
            scope_label = "Whole case library" if scope == "corpus" else "Current evidence only"
        else:
            source_mode_label = "Uploaded document + case corpus"
            scope_label = "Whole case library" if scope == "corpus" else "Current evidence only"
        filter_lines = [
            f"- Case type: {filters.get('case_type') or 'Not provided'}",
            f"- User role: {filters.get('user_role') or 'Not provided'}",
            f"- Forum: {filters.get('forum') or 'Not provided'}",
            f"- Source mode: {source_mode_label}",
            f"- Retrieval scope: {scope_label}",
            f"- Retrieval profile: {retrieval_profile}",
        ]
        history_lines = []
        for turn in chat_history[-self.settings.rag_history_turns :]:
            role = "User" if turn.get("role") == "user" else "Assistant"
            history_lines.append(f"- {role}: {shorten_text(turn.get('content') or '', 180)}")
        if not history_lines:
            history_lines.append("- No earlier turns in this chat.")
        claim_lines = [
            f"- {claim['claim']} -> {claim['case_id']} ({claim['support_type']}, {claim['authority_level']})"
            for claim in (evidence_pack.get("claim_evidence_map") or [])
        ] or ["- No explicit claim-evidence map was available."]
        law_lines = [
            f"- {item.get('title')} | {item.get('section_ref') or 'no section'} | {shorten_text(item.get('excerpt') or '', 180)}"
            for item in (evidence_pack.get("reference_materials") or [])[:3]
        ] or ["- No official law materials were included for this turn."]
        fast_cards = list((evidence_pack.get("cards") or []))[:2]
        if retrieval_profile == "fast":
            sections = [
                "A. Current User Question",
                f"- Question: {question}",
                f"- Retrieval query: {retrieval_query}",
                "",
                "B. Query Profile",
                f"- Task: {question_profile.get('task') or 'general_research'}",
                f"- Domain: {question_profile.get('domain') or 'unspecified'}",
                f"- Requested response plan: {question_profile.get('response_plan') or 'direct_guidance'}",
                f"- Requested answer length: {question_profile.get('response_length') or 'medium'}",
                "",
                "C. Official Law Materials",
                *law_lines,
                "",
                "D. Authority Cards",
            ]
            if fast_cards:
                for card in fast_cards:
                    sections.extend(
                        [
                            f"- {card['case_id']} ({card['support_type']}, {card['authority_level']})",
                            f"  - Proposition: {card['proposition']}",
                            f"  - Passage: {shorten_text(card['excerpt'], 180)}",
                        ]
                    )
            elif rag_context.get("context_text"):
                sections.append(f"- {shorten_text(rag_context.get('context_text') or '', 280)}")
            else:
                sections.append("- No strong authority cards were available.")
            sections.extend(
                [
                    "",
                    "E. Rules",
                    "- Answer only from the authority cards above, any official law materials shown in the evidence bundle, and any visible uploaded-document context.",
                    "- Keep the answer concise, complete, and user-facing.",
                    "- Do not leave the final sentence unfinished.",
                    "- If support is mixed or weak, say so plainly without using internal audit language.",
                ]
            )
            if source_mode == "document_only":
                sections.append("- This turn is document-only; do not rely on outside case-law.")
            elif source_mode in {"document_plus_case", "document_plus_reference_law", "document_plus_reference_law_plus_case"}:
                sections.append("- Treat uploaded document excerpts as facts and retrieved authorities or law materials as legal support.")
            return "\n".join(sections)
        sections = [
            "A. Current User Question",
            f"- Question: {question}",
            f"- Retrieval query: {retrieval_query}",
            "",
            "B. Earlier Chat Context",
            *history_lines,
            "",
            "C. Optional Filters",
            *filter_lines,
            "",
            "D. Query Profile",
            f"- Lane: {question_profile.get('lane') or 'case_law'}",
            f"- Task: {question_profile.get('task') or 'general_research'}",
            f"- Domain: {question_profile.get('domain') or 'unspecified'}",
            f"- Legal elements: {', '.join(question_profile.get('legal_elements') or []) or 'none extracted'}",
            "",
            "E. Evidence Bundle Prepared For This Turn",
            f"- {rag_context.get('coverage_note') or 'No coverage note available.'}",
            "",
            rag_context.get("context_text") or self._format_retrieval_block(similar_cases),
            "",
            "F. Official Law Materials",
            *law_lines,
            "",
            "G. Claim-Evidence Map",
            *claim_lines,
            "",
            "H. Presentation Instructions",
            f"- Requested answer style: {question_profile.get('answer_style') or 'structured'}.",
            f"- Requested response plan: {question_profile.get('response_plan') or 'direct_guidance'}.",
            f"- Requested answer length: {question_profile.get('response_length') or 'medium'}.",
            "- Follow the response plan instead of forcing a universal template.",
            "- Keep bullet points short and complete; do not leave a sentence unfinished.",
            "- If this is a rewrite/simplify follow-up, preserve the same legal position and evidence but change the form and clarity.",
            "",
            "I. Rules",
            "- Treat earlier chat turns only as conversational context, not as evidence.",
            "- Prefer simple language over legal jargon when possible.",
            "- If the retrieved authorities conflict, say that clearly.",
            "- If evidence is insufficient, say that directly.",
            "- Use the authority cards and any official law materials in the evidence bundle as the authoritative support set; do not cite anything outside them.",
        ]
        if retrieval_profile == "fast":
            sections.append("- Keep the answer concise and focused on the best-supported conclusion.")
        if source_mode == "document_only":
            sections.extend(
                [
                    "- Use only the uploaded document excerpts below.",
                    "- Do not treat the uploaded document as case-law authority; explain only what the file supports.",
                    "- If the uploaded document does not support a legal conclusion, say that clearly.",
                    "- If the file states an exact amount, party name, date, or calculation, quote that exact detail.",
                    "- If page numbers are shown in the evidence bundle, mention them when they help answer a factual question.",
                    "- For methodology, framework, or pipeline questions, preserve the judgment's own sequence instead of giving a generic legal summary.",
                    "- For comparison questions, separate each party's contention and then state the Court's response to each.",
                    "- For hypothetical questions, clearly mark the answer as an application of the judgment's logic, not as a direct holding.",
                ]
            )
        elif source_mode == "case_corpus_only":
            sections.extend(
                [
                    "- Use the retrieved authorities and any official law materials in the evidence bundle as legal support.",
                    "- Cite case ids directly when making a case-law point, and mention the Act/Rule/Article directly when the support comes from official law materials.",
                ]
            )
        else:
            sections.extend(
                [
                    "- Treat uploaded document excerpts as user facts, not as legal authority.",
                    "- Use the retrieved authorities and any official law materials as legal support.",
                    "- Cite case ids directly when making a case-law point, and mention the Act/Rule/Article directly when the support comes from official law materials.",
                    "- If the question is about the uploaded judgment itself, answer from the document facts first and use corpus material only as secondary context.",
                ]
            )
        return "\n".join(sections)

    @staticmethod
    def _format_intake_block(intake: dict[str, Any], case_text: str) -> str:
        fields = [
            ("Case type", intake.get("case_type")),
            ("Facts", intake.get("facts")),
            ("Relief sought", intake.get("relief_sought")),
            ("Evidence", intake.get("evidence_summary")),
            ("Opponent argument", intake.get("opponent_arguments")),
            ("User role", intake.get("user_role")),
            ("Forum", intake.get("forum")),
        ]
        if not intake.get("facts"):
            fields.append(("Model-ready summary", case_text))
        return "\n".join(
            [f"- {label}: {value or 'Not provided'}" for label, value in fields]
        )

    @staticmethod
    def _format_prediction_block(
        prediction: dict[str, Any],
        favorability_label: str,
        favorability_reason: str,
    ) -> str:
        probability_lines = [
            f"{label}: {value}%"
            for label, value in prediction.get("probabilities", {}).items()
        ]
        lines = [
            f"- Predicted label name: {prediction['predicted_name']}",
            f"- Confidence score: {prediction['confidence_score']}%",
            f"- Favorability label: {favorability_label}",
            f"- Class probabilities: {', '.join(probability_lines)}",
        ]
        return "\n".join(lines)

    def _format_retrieval_block(self, similar_cases: list[dict[str, Any]]) -> str:
        if not similar_cases:
            return "- No retrieved cases were available."

        entries: list[str] = []
        for idx, item in enumerate(similar_cases[: self.settings.llm_max_cases], start=1):
            summary = item.get("summary") or item.get("excerpt") or ""
            excerpt = item.get("excerpt") or "No excerpt available."
            entries.extend(
                [
                    f"Case {idx}",
                    f"- Case ID: {item['case_id']}",
                    f"- Outcome: {item.get('label_name') or 'Unknown'}",
                    f"- Similarity score: {item['similarity']}",
                    f"- Short summary: {shorten_text(summary, 90)}",
                    f"- Best excerpt: {shorten_text(excerpt, 120)}",
                ]
            )
        return "\n".join(entries)

    @staticmethod
    def _infer_review_domain(
        *,
        intake: dict[str, Any],
        issue_outline: list[str],
    ) -> str:
        case_type = normalize_whitespace((intake or {}).get("case_type")).lower()
        facts_blob = " ".join(
            part
            for part in [
                case_type,
                (intake or {}).get("facts") or "",
                (intake or {}).get("relief_sought") or "",
                (intake or {}).get("opponent_arguments") or "",
                " ".join(issue_outline or []),
            ]
            if part
        ).lower()
        if "tax" in case_type or any(token in facts_blob for token in ("assessee", "assessment", "bogus", "purchase", "income tax")):
            return "tax"
        if "consumer" in case_type or any(token in facts_blob for token in ("warranty", "refund", "replacement", "defect", "service centre", "service center")):
            return "consumer"
        if "motor accident" in case_type or any(token in facts_blob for token in ("amputation", "prosthetic", "claimant", "functional disability")):
            return "motor_accident"
        if "university" in case_type or any(token in facts_blob for token in ("unfair means", "invigilator", "result", "exam")):
            return "education"
        if "service" in case_type or any(token in facts_blob for token in ("suspension", "disciplinary", "chargesheet", "departmental")):
            return "service"
        if "information" in case_type or any(token in facts_blob for token in ("rti", "information", "pio", "inspection", "copies")):
            return "information"
        return ""

    @staticmethod
    def _format_review_authority_groups(
        authority_map: dict[str, Any],
        similar_cases: list[dict[str, Any]],
    ) -> str:
        grouped_lines: list[str] = []
        groups = [
            ("Best support", authority_map.get("supporting") or []),
            ("Main risks / distinguishable", authority_map.get("conflicting") or []),
            ("Broader same-domain", authority_map.get("mixed") or []),
        ]
        if not any(items for _label, items in groups):
            if not similar_cases:
                return "- No retrieved cases were available."
            lines: list[str] = []
            for idx, item in enumerate(similar_cases[:3], start=1):
                lines.extend(
                    [
                        f"Case {idx}",
                        f"- Case ID: {item.get('case_id')}",
                        f"- Outcome: {item.get('label_name') or 'Unknown'}",
                        f"- Fact fit: {item.get('fit_band') or 'unknown'}",
                        f"- Why it matters: {normalize_whitespace(item.get('fit_note') or item.get('retrieval_note') or 'No fit note was generated.')}",
                        f"- Best excerpt: {shorten_text(item.get('excerpt') or item.get('summary') or '', 160)}",
                    ]
                )
            return "\n".join(lines)
        for label, items in groups:
            grouped_lines.append(label)
            if not items:
                grouped_lines.append("- None surfaced in this group.")
                continue
            for item in items[:2]:
                grouped_lines.append(
                    f"- {item.get('case_id')} | outcome={item.get('label_name') or 'Unknown'} | fact fit={item.get('fit_band') or 'unknown'}"
                )
                grouped_lines.append(
                    f"  - Why it matters: {normalize_whitespace(item.get('fit_note') or 'No fit note was generated.')}"
                )
                grouped_lines.append(
                    f"  - Passage signal: {shorten_text(item.get('proposition') or item.get('excerpt') or item.get('summary') or '', 180)}"
                )
        return "\n".join(grouped_lines)

    @staticmethod
    def _build_bottom_line(
        *,
        predicted_name: str,
        favorability_label: str,
        confidence_score: float,
        review_domain: str,
    ) -> str:
        outcome_map = {
            "consumer": "The current record suggests a reasonable chance of relief, but the exact remedy may still depend on how the forum views delay, defect seriousness, and repair history.",
            "tax": "The current record suggests there is some support against the addition, but the outcome will depend heavily on the quality of documentary proof and whether the department has stronger contrary material.",
            "service": "The present material shows some support for challenging the action, but service cases often turn on procedure, timing, and whether the department can justify the order on record.",
            "information": "The present material shows some support for disclosure, but success will depend on whether the authority has a genuine exemption or only a weak administrative objection.",
            "motor_accident": "The present material suggests there is some support for stronger compensation, but the final amount will depend on proof of disability, future treatment, and income impact.",
        }
        base = outcome_map.get(
            review_domain,
            "The current material gives some support to your position, but the final outcome still depends on the exact record and how closely the authorities match your facts.",
        )
        direction = f"The review currently leans toward **{predicted_name}**, which is **{favorability_label.lower()}** for your side."
        caution = "Treat this as a first-pass legal view, not a final conclusion."
        if confidence_score >= 75:
            caution = "The signal is useful, but the full record still needs to be checked."
        return f"{direction} {base} {caution}"

    @staticmethod
    def _build_helpful_factors(
        *,
        intake: dict[str, Any],
        issue_outline: list[str],
        support_cases: list[dict[str, Any]],
    ) -> str:
        bullets: list[str] = []
        facts_blob = " ".join(
            normalize_whitespace(intake.get(key) or "").lower()
            for key in ("facts", "evidence_summary", "relief_sought")
        )
        if any(token in facts_blob for token in ("same day", "3 day", "3 days", "prompt", "immediately", "early")):
            bullets.append("- Early or prompt complaint reporting usually helps because it weakens the argument that the consumer accepted the defect.")
        if any(token in facts_blob for token in ("45 day", "45 days", "delay", "months", "service centre", "service center")):
            bullets.append("- A long unresolved repair or service-centre delay usually helps because forums often look at whether the product was made usable within a reasonable time.")
        if normalize_whitespace(intake.get("evidence_summary")):
            bullets.append("- Documentary proof such as invoices, emails, complaint records, and service documents strengthens credibility.")
        for line in issue_outline[:2]:
            bullets.append(f"- {line}")
        if support_cases:
            bullets.append("- The retrieved supporting authorities suggest that at least some similar disputes have resulted in relief for the consumer.")
        return "\n".join(bullets[:5]) or "- The current facts give some support, but the strengths are not yet sharply stated in the intake."

    @staticmethod
    def _build_risk_factors(
        *,
        intake: dict[str, Any],
        issue_outline: list[str],
        risk_cases: list[dict[str, Any]],
    ) -> str:
        bullets: list[str] = []
        opponent = normalize_whitespace(intake.get("opponent_arguments") or "")
        relief = normalize_whitespace(intake.get("relief_sought") or "")
        if opponent:
            bullets.append(f"- The opposite side is likely to rely on this point: {opponent}")
        else:
            bullets.append("- The other side may argue that repair was the correct first remedy and that refund was not yet justified.")
        if not relief:
            bullets.append("- The exact remedy is still not clearly stated, and that can change which precedents are most relevant.")
        for line in issue_outline[:2]:
            if "pressure point" in line.lower():
                bullets.append(f"- {line}")
        if risk_cases:
            bullets.append("- At least one retrieved authority points to a possible distinction or weaker outcome on similar facts.")
        return "\n".join(bullets[:5]) or "- No major risk factor is surfaced yet, but that may reflect retrieval limits rather than a truly risk-free case."

    @staticmethod
    def _build_simple_legal_principle(
        *,
        review_domain: str,
        issue_outline: list[str],
    ) -> str:
        principles = {
            "consumer": "In consumer matters, refund is usually considered when a defective product is not made properly usable within a reasonable time, especially if repair attempts drag on or the buyer keeps facing the same problem.",
            "tax": "In tax disputes, authorities usually look at whether the documents show a real transaction, not just whether the claim sounds plausible. Banking trail helps, but stronger supporting records often matter.",
            "service": "In service matters, courts and tribunals usually ask whether the action followed fair procedure and whether the department had a defensible factual basis for what it did.",
            "information": "In RTI disputes, the key question is usually whether the authority had a lawful reason to deny the records or whether it was simply avoiding disclosure.",
            "motor_accident": "In motor-accident compensation matters, the key question is not just the injury itself, but how it affects earning capacity, future treatment, and day-to-day functioning.",
        }
        if review_domain in principles:
            return principles[review_domain]
        if issue_outline:
            return "The likely rule here is that the forum will look at the actual facts, the remedy sought, and whether the evidence supports the legal claim being made."
        return "The likely rule here depends on how the forum reads the facts, the documents, and the remedy being requested."

    def _build_review_case_blocks(
        self,
        *,
        support_cases: list[dict[str, Any]],
        risk_cases: list[dict[str, Any]],
    ) -> str:
        blocks: list[str] = []
        for index, item in enumerate(support_cases[:2], start=1):
            blocks.append(self._format_review_case_block(item=item, heading=f"Case {index} (support)", helpful=True))
        if risk_cases:
            blocks.append(self._format_review_case_block(item=risk_cases[0], heading="Risk case", helpful=False))
        return "\n\n".join(blocks) if blocks else "No strong supporting or cautionary authority has been surfaced yet."

    def _format_review_case_block(
        self,
        *,
        item: dict[str, Any],
        heading: str,
        helpful: bool,
    ) -> str:
        case_id = str(item.get("case_id") or "Unnamed authority")
        excerpt = shorten_text(normalize_whitespace(item.get("excerpt") or item.get("summary") or ""), 180) or "No short retrieved passage is available."
        fit_note = self._humanize_case_note(item.get("fit_note") or item.get("retrieval_note") or "")
        outcome = str(item.get("label_name") or "Unknown")
        if helpful:
            return (
                f"**{heading}**\n"
                f"- Why it may be similar: {fit_note or 'The issue and remedy appear reasonably close on the current shortlist.'}\n"
                f"- What the forum decided: this authority is currently tagged `{outcome}` in the dataset, which suggests it ended in favor of the successful side there.\n"
                f"- How it may apply here: {case_id} may help you if the same delay, defect, or remedy pattern can be shown on your record.\n"
                f"- Useful factual signal: {excerpt}"
            )
        return (
            f"**{heading}**\n"
            f"- Why it matters: {fit_note or 'It may point to a weaker or more limited outcome on similar facts.'}\n"
            f"- What the forum decided: this authority is currently tagged `{outcome}` in the dataset.\n"
            f"- How it may affect you: if the other side can place your facts closer to this authority, relief may become narrower or harder to get.\n"
            f"- Useful factual signal: {excerpt}"
        )

    @staticmethod
    def _build_practical_outcome(
        *,
        review_domain: str,
        predicted_name: str,
        confidence_score: float,
        favorability_label: str,
    ) -> str:
        if review_domain == "consumer":
            return (
                "The most realistic outcomes are usually refund with some compensation, replacement, or repair-only relief. "
                f"On the current intake, the direction leans toward **{predicted_name}**, so a favorable outcome is plausible, but the exact relief may still change."
            )
        if review_domain == "information":
            return "A realistic outcome is either disclosure of records, disclosure with limits, inspection-only relief, or remand to the authority for a fresh decision."
        if review_domain == "service":
            return "A realistic outcome may be interim relief, fresh consideration, partial relief, or refusal if the department's record is stronger than it currently appears."
        return (
            f"The current result points to a **{favorability_label.lower()}** direction, but the actual order may still be narrower, broader, or partly conditional depending on the evidence."
        )

    @staticmethod
    def _build_next_steps_block(
        *,
        review_domain: str,
        evidence_gaps: list[str],
        verification_focus: str,
    ) -> str:
        bullets = [f"- {verification_focus}"]
        if evidence_gaps:
            bullets.append(f"- {evidence_gaps[0]}")
        if review_domain == "consumer":
            bullets.append("- Keep a clean timeline of purchase, complaint, service-centre handling, and refund refusal.")
            bullets.append("- Clearly state whether you are asking for refund, replacement, compensation, or all of them.")
        elif review_domain == "service":
            bullets.append("- Put the suspension/order timeline, reply, and supporting service documents in one chronology.")
        elif review_domain == "information":
            bullets.append("- Keep the RTI application, PIO reply, and appeal papers together so the refusal ground is clear.")
        return "\n".join(bullets[:4])

    @staticmethod
    def _build_confidence_line(
        *,
        confidence_score: float,
        similar_cases: list[dict[str, Any]],
        risk_cases: list[dict[str, Any]],
    ) -> str:
        strength = "Preliminary"
        if confidence_score >= 75:
            strength = "Moderate to strong"
        elif confidence_score >= 60:
            strength = "Moderate"
        reliability = "Use with caution" if risk_cases else "Reasonably usable for triage"
        authority_count = len(similar_cases)
        return (
            f"Legal strength: {strength}\n\n"
            f"Confidence: ~{round(confidence_score):d}%\n\n"
            f"Current reliability: {reliability}. The answer is based on {authority_count} shortlisted authorities, so the full judgments still need to be checked."
        )

    @staticmethod
    def _build_review_authority_paragraph(
        *,
        label: str,
        cases: list[dict[str, Any]],
        fallback: str,
    ) -> str:
        if not cases:
            return fallback

        sentences: list[str] = []
        for item in cases[:2]:
            case_id = str(item.get("case_id") or "Unnamed authority")
            fit_note = ExplanationService._humanize_case_note(
                item.get("fit_note") or item.get("retrieval_note") or ""
            )
            excerpt = normalize_whitespace(item.get("excerpt") or item.get("summary") or "")
            signal = shorten_text(excerpt, 140) if excerpt else "No short excerpt was available."
            if label == "support":
                sentences.append(
                    f"{case_id} may help because {fit_note or 'it appears close on the issue and remedy being discussed here'}. "
                    f"The retrieved passage suggests: {signal}"
                )
            else:
                sentences.append(
                    f"{case_id} is the main cautionary authority because {fit_note or 'it may point to a different outcome or a meaningful distinction'}. "
                    f"The retrieved passage suggests: {signal}"
                )
        return " ".join(sentences)

    @staticmethod
    def _humanize_case_note(note: str) -> str:
        cleaned = normalize_whitespace(note)
        if not cleaned:
            return ""
        lowered = cleaned.lower()
        replacements = {
            "subtype match:": "it matches the issue type:",
            "only broad domain match; issue subtype is not clearly aligned.": "it is from the same general legal area, but the factual issue is not perfectly aligned.",
            "different relief context": "the relief context appears different.",
            "same issue": "the issue appears close.",
            "same remedy": "the remedy appears close.",
            "same procedural posture": "the procedural posture appears close.",
            "distinguishable on evidence": "the case may turn on different evidence.",
        }
        for source, target in replacements.items():
            if source in lowered:
                cleaned = re.sub(re.escape(source), target, cleaned, flags=re.I)
        cleaned = re.sub(r"\balignment:\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s+\|\s+.*$", "", cleaned)
        return cleaned[:1].lower() + cleaned[1:] if cleaned else ""

    @staticmethod
    def _review_verification_focus(
        *,
        review_domain: str,
        issue_outline: list[str],
        evidence_gaps: list[str],
        favorability_reason: str,
    ) -> str:
        domain_guidance = REVIEW_DOMAIN_GUIDANCE.get(review_domain or "", {})
        focus_parts: list[str] = []
        if issue_outline:
            focus_parts.append(
                "First verify the issue drivers surfaced here: " + "; ".join(issue_outline[:2]) + "."
            )
        if domain_guidance.get("verification_focus"):
            focus_parts.append(
                "Then check " + domain_guidance["verification_focus"] + "."
            )
        if evidence_gaps:
            focus_parts.append(evidence_gaps[0])
        focus_parts.append(favorability_reason)
        return " ".join(part for part in focus_parts if part)

    @classmethod
    def _fallback(
        cls,
        case_text: str,
        prediction: dict,
        similar_cases: list[dict],
        *,
        intake: dict[str, Any],
        favorability_label: str,
        favorability_reason: str,
        authority_map: dict[str, Any] | None = None,
        issue_outline: list[str] | None = None,
        evidence_gaps: list[str] | None = None,
    ) -> str:
        facts = intake.get("facts") or "The intake describes the dispute in broad terms."
        relief = intake.get("relief_sought") or "Specific relief was not clearly stated."
        evidence = intake.get("evidence_summary") or "The dedicated evidence field was not populated."
        opponent = intake.get("opponent_arguments") or "The opposing side's position was not clearly stated."
        authority_map = authority_map or {}
        issue_outline = issue_outline or []
        evidence_gaps = evidence_gaps or []

        predicted_name = prediction["predicted_name"]
        confidence_score = prediction["confidence_score"]
        confidence_band = prediction["confidence_band"]

        top_cases = similar_cases[:3]
        similar_case_lines = []
        for item in top_cases:
            case_summary = shorten_text(item.get("summary") or item.get("excerpt") or "", 170)
            similar_case_lines.append(
                (
                    f"- `{item['case_id']}` ({item.get('label_name') or 'Unknown'}, similarity {item['similarity']}) "
                    f"supports the direction of the prediction because the retrieved text points to {case_summary}"
                )
            )

        if not similar_case_lines:
            similar_case_lines.append(
                "- No retrieved cases were available, so the system could not ground the explanation in historical matches."
            )

        similarity_lines = [
            f"- Similarity: the current intake describes {shorten_text(facts, 180)}",
            f"- Similarity: the request for relief is {shorten_text(relief, 140)}",
        ]
        if top_cases:
            similarity_lines.append(
                "- Similarity: the top retrieved matters also trend toward the same outcome label and contain overlapping factual cues in their excerpts."
            )
        if issue_outline:
            similarity_lines.append(
                "- Current issue outline: " + ", ".join(issue_outline[:3]) + "."
            )

        difference_lines = [
            f"- Difference: the current opposing argument is {shorten_text(opponent, 160)}",
        ]
        if not intake.get("evidence_summary"):
            difference_lines.append(
                "- Difference: the separate evidence-summary field is blank, so the explanation has to rely more heavily on the facts narrative than on a clean evidence list."
            )
        difference_lines.append(
            "- Difference: similarity search can find factually close matters, but it does not guarantee the same procedural posture, technical cause, or evidentiary strength."
        )
        if evidence_gaps:
            difference_lines.append(f"- Missing or weak point to verify next: {evidence_gaps[0]}")

        caution_lines = []
        if confidence_band == "Low":
            caution_lines.append(
                f"- Confidence is only {confidence_score:.2f}%, so the model sees support for '{predicted_name}' but not decisively."
            )
        else:
            caution_lines.append(
                f"- Confidence is {confidence_score:.2f}%, which is more supportive but still not a substitute for manual review."
            )
        caution_lines.append(
            "- Treat the retrieved cases as support for comparison, not as proof that the present matter will end the same way."
        )
        caution_lines.append(
            "- If the procedural history, evidence quality, or opposing defense materially differ from the retrieved cases, the outcome could shift."
        )

        return "\n\n".join(
            [
                "\n".join(
                    [
                        "#### 1. Predicted Outcome",
                        f"- The classifier predicts **{predicted_name}**.",
                        f"- For your side, the likely effect is **{favorability_label}**.",
                        f"- {favorability_reason}",
                    ]
                ),
                "\n".join(
                    [
                        "#### 2. Why This Side Currently Has Support",
                        f"- The factual pattern presented is: {shorten_text(facts, 220)}",
                        f"- The relief sought is: {shorten_text(relief, 180)}",
                        f"- The current evidence summary is: {shorten_text(evidence, 180)}",
                    ]
                ),
                "\n".join(
                    [
                        "#### 3. Strongest Authorities For This Direction",
                        *similar_case_lines,
                    ]
                ),
                "\n".join(
                    [
                        "#### 4. Main Risks Or Distinguishing Facts",
                        *similarity_lines,
                        *difference_lines,
                    ]
                ),
                "\n".join(
                    [
                        "#### 5. What To Verify Next",
                        "- Verify whether the strongest retrieved authorities turn on the same document set, procedural posture, and forum context as the current matter.",
                        "- Check whether the opposing side can undermine the present case through missing records, adverse statements, or factual distinctions.",
                        "- Review whether any supposedly supporting authority is only a broad domain match rather than the same issue subtype.",
                    ]
                ),
                "\n".join(
                    [
                        "#### 6. Confidence And Caution",
                        *caution_lines,
                    ]
                ),
            ]
        )

    @staticmethod
    def _answer_fallback(
        *,
        question: str,
        retrieval_query: str,
        similar_cases: list[dict[str, Any]],
        rag_context: dict[str, Any],
        evidence_pack: dict[str, Any],
        question_profile: dict[str, Any],
        answer_language: str,
        filters: dict[str, Any],
        source_mode: str,
    ) -> str:
        document_filename = rag_context.get("document_filename") or "uploaded document"
        document_used = bool(rag_context.get("document_used"))
        document_context = shorten_text(normalize_whitespace(rag_context.get("context_text") or ""), 220)
        response_plan = str(question_profile.get("response_plan") or "direct_guidance")
        if source_mode == "document_only":
            if not document_used:
                return (
                    "#### Bottom line\n"
                    "- I could not answer from the uploaded document alone.\n\n"
                    "#### Why\n"
                    "- No readable document excerpts were available for this turn.\n\n"
                    "#### Best next step\n"
                    "- Upload a readable document or switch back to case corpus mode."
                )
            return (
                "#### Bottom line\n"
                f"- Based only on the uploaded document ({document_filename}), this is the safest summary I can give.\n\n"
                "#### Why\n"
                f"- {document_context or 'The file excerpts were retrieved but too limited for a fuller summary.'}\n\n"
                "#### Limits\n"
                "- This mode did not consult the case-law corpus."
            )

        top_cases = similar_cases[:2]
        if not top_cases:
            if source_mode == "document_plus_case" and document_used:
                return (
                    "#### Bottom line\n"
                    "- No strong case-law authorities were retrieved, so this answer can only be tentative.\n\n"
                    "#### Why\n"
                    f"- The uploaded document ({document_filename}) supplied the strongest available support.\n"
                    f"- {document_context or 'The retrieved document excerpts were limited.'}\n\n"
                    "#### Limits\n"
                    "- No closely matching case-law authorities were available for this turn."
                )
            if response_plan == "practical_steps":
                return (
                    "#### Bottom line\n"
                    "- I do not have enough matching authority yet to give a safe practical recommendation.\n\n"
                    "#### Best next step\n"
                    "- Narrow the issue, add the forum or case type, or upload the relevant notice/order."
                )
            return (
                "#### Answer\n"
                "- I could not find strong supporting cases for this question from the current retrieval store.\n\n"
                "#### Why\n"
                "- The retrieved evidence was too weak or too mixed for a safe answer.\n\n"
                "#### Best next step\n"
                "- Try a more specific question or add case type, forum, or facts."
            )

        if question_profile.get("statute_sensitive") and not evidence_pack.get("statute_support_available"):
            lead_cards = evidence_pack.get("cards") or []
            if lead_cards:
                return (
                    "#### Answer\n"
                    "- I can only answer this at the case-law level, not as a settled statute answer.\n\n"
                    "#### Available case-law support\n"
                    + "\n".join(
                        f"- {card['case_id']} suggests {card['proposition']}."
                        for card in lead_cards[:2]
                    )
                    + "\n\n#### Limits\n"
                    + "- No provision-level statute source was retrieved."
                )
        answer_points = [
            f"- Based on the retrieved authorities, the question '{shorten_text(question, 120)}' is most strongly supported by the top matched cases below.",
        ]
        if filters.get("case_type") or filters.get("forum"):
            answer_points.append(
                f"- The search was constrained by context: case type={filters.get('case_type') or 'not specified'}, forum={filters.get('forum') or 'not specified'}."
            )
        authority_points = [
            f"- `{item['case_id']}` ({item.get('label_name') or 'Unknown'}, similarity {item['similarity']}) -> {shorten_text(item.get('excerpt') or item.get('summary') or '', 160)}"
            for item in top_cases
        ]
        caution = "- This is a retrieval-grounded summary, not a final legal conclusion."
        if len(top_cases) >= 2 and top_cases[0].get("label_name") != top_cases[1].get("label_name"):
            caution = "- The retrieved authorities are mixed in outcome, so treat the answer as tentative and inspect the full cases."
        if answer_language != "English":
            caution += " Multilingual generation was unavailable, so this fallback answer remains in English."
        if response_plan == "practical_steps":
            return "\n".join(
                [
                    "#### Bottom line",
                    *answer_points,
                    "",
                    "#### Why this answer is limited",
                    "- The answer is based only on the top retrieved authorities.",
                    *authority_points,
                    "",
                    "#### Best next step",
                    "- Use these authorities as a starting point, then verify the full judgments before acting.",
                    caution,
                ]
            )
        if response_plan in {"case_brief", "case_brief_simple"}:
            return "\n".join(
                [
                    "#### Answer",
                    *answer_points,
                    "",
                    "#### Closest authorities",
                    *authority_points,
                    "",
                    "#### Limits",
                    caution,
                ]
            )
        if response_plan == "research_note":
            return "\n".join(
                [
                    "#### Issue",
                    f"- {shorten_text(question, 120)}",
                    "",
                    "#### Answer",
                    *answer_points,
                    "",
                    "#### Key authorities",
                    *authority_points,
                    "",
                    "#### Open points",
                    caution,
                    f"- Retrieval query used: {shorten_text(retrieval_query, 100)}.",
                ]
            )
        if response_plan == "outcome_pattern":
            accepted = [item for item in similar_cases if item.get("label_name") == "Accepted"][:2]
            rejected = [item for item in similar_cases if item.get("label_name") == "Rejected"][:2]
            accepted_lines = [
                f"- `{item['case_id']}` -> {shorten_text(item.get('excerpt') or item.get('summary') or '', 150)}"
                for item in accepted
            ] or ["- Accepted-pattern authorities were not strong enough in this turn."]
            rejected_lines = [
                f"- `{item['case_id']}` -> {shorten_text(item.get('excerpt') or item.get('summary') or '', 150)}"
                for item in rejected
            ] or ["- Rejected-pattern authorities were not strong enough in this turn."]
            return "\n".join(
                [
                    "#### Answer",
                    *answer_points,
                    "",
                    "#### What accepted cases tend to show",
                    *accepted_lines,
                    "",
                    "#### What rejected cases tend to show",
                    *rejected_lines,
                    "",
                    "#### Limits",
                    caution,
                ]
            )
        return "\n".join(
            [
                "#### Answer",
                *answer_points,
                "",
                "#### Why",
                "- The answer is based only on the top retrieved authorities.",
                *authority_points,
                "",
                "#### Limits",
                caution,
                f"- Retrieval query used: {shorten_text(retrieval_query, 100)}.",
            ]
        )

    @staticmethod
    def _normalize_answer_text(text: str) -> str:
        cleaned = (text or "").replace("\r\n", "\n").strip()
        if not cleaned:
            return ""

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
        normalized = "\n".join(normalized_lines).strip()
        return ExplanationService._repair_truncated_answer(normalized)

    @staticmethod
    def _repair_truncated_answer(text: str) -> str:
        cleaned = (text or "").strip()
        if not cleaned:
            return ""

        if cleaned.endswith((".", "!", "?", "`")):
            return cleaned

        last_sentence_end = max(cleaned.rfind("."), cleaned.rfind("!"), cleaned.rfind("?"))
        if last_sentence_end >= max(int(len(cleaned) * 0.55), 0):
            return cleaned[: last_sentence_end + 1].rstrip()
        if "\n- " in cleaned:
            lines = [line.rstrip() for line in cleaned.split("\n") if line.strip()]
            if lines:
                last_line = lines[-1]
                if last_line.startswith("- ") and len(last_line.split()) >= 6:
                    lines[-1] = last_line.rstrip(",;:- ") + "."
                    return "\n".join(lines)
            trimmed = cleaned.rsplit("\n- ", 1)[0].rstrip()
            if trimmed:
                return trimmed
        if len(cleaned.split()) >= 8:
            return cleaned.rstrip(",;:- ") + "."
        return cleaned

    @staticmethod
    def _normalize_markdown_sections(text: str) -> str:
        cleaned = text.strip()
        if not cleaned:
            return cleaned

        replacements = [
            ("1. Predicted Outcome", "#### 1. Predicted Outcome"),
            ("2. Why This Side Currently Has Support", "#### 2. Why This Side Currently Has Support"),
            ("3. Strongest Authorities For This Direction", "#### 3. Strongest Authorities For This Direction"),
            (
                "4. Main Risks Or Distinguishing Facts",
                "#### 4. Main Risks Or Distinguishing Facts",
            ),
            ("5. What To Verify Next", "#### 5. What To Verify Next"),
            ("6. Confidence And Caution", "#### 6. Confidence And Caution"),
        ]
        for original, replacement in replacements:
            cleaned = re.sub(
                rf"(?m)^(?!####\s){re.escape(original)}\s*$",
                replacement,
                cleaned,
            )
        generic_headings = [
            ("Direct answer", "#### Answer"),
            ("Answer", "#### Answer"),
            ("Why these cases matter", "#### Why these cases matter"),
            ("Why", "#### Why these cases matter"),
            ("Authorities used", "#### Source used"),
            ("Authorities Mentioned", "#### Source used"),
            ("Supporting Authorities", "#### Source used"),
            ("Limits", "#### Limits"),
            ("Caution", "#### Limits"),
            ("Predicted Outcome", "#### Predicted Outcome"),
        ]
        for original, replacement in generic_headings:
            cleaned = re.sub(
                rf"(?m)^(?!####\s){re.escape(original)}\s*$",
                replacement,
                cleaned,
            )
        return cleaned

    @classmethod
    def _normalize_review_summary(cls, text: str) -> str:
        cleaned = (text or "").replace("\r\n", "\n").strip()
        if not cleaned:
            return ""

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
        normalized = "\n".join(normalized_lines).strip()
        normalized = cls._normalize_markdown_sections(normalized)
        return cls._repair_truncated_answer(normalized)
