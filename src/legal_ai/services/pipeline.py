from __future__ import annotations

import re
from typing import Any

from legal_ai.config import Settings
from legal_ai.schemas import (
    CaseDetailResponse,
    PredictionRequest,
    PredictionResponse,
    QuestionAnswerRequest,
    QuestionAnswerResponse,
    ReferenceMaterial,
    ResearchRequest,
    ResearchResponse,
    SimilarCase,
    WorkspaceSummary,
)
from legal_ai.services.classifier import LegalClassifier
from legal_ai.services.answer_audit import AnswerAuditService
from legal_ai.services.evidence_pack import EvidencePackBuilder
from legal_ai.services.explainer import ExplanationService
from legal_ai.services.labels import LABEL_ID_TO_NAME
from legal_ai.services.query_router import QueryRouterService
from legal_ai.services.qa_retriever import LegalQARetriever
from legal_ai.services.rag_context import RAGContextBuilder
from legal_ai.services.reference_law import ReferenceLawRetriever
from legal_ai.services.retriever import SimilarCaseRetriever
from legal_ai.services.session_documents import SessionDocumentStore
from legal_ai.services.workspace import WorkspaceBuilder
from legal_ai.utils.domain import infer_candidate_domain, infer_issue_subtypes, infer_query_domain
from legal_ai.utils.text import normalize_whitespace, search_terms, shorten_text

FILING_SIDE_MARKERS = {
    "petitioner",
    "appellant",
    "complainant",
    "applicant",
    "claimant",
    "plaintiff",
    "assessee",
    "filing party",
}

OPPOSING_SIDE_MARKERS = {
    "respondent",
    "opposite party",
    "defendant",
    "department",
    "revenue",
    "responding party",
}

LAW_FIRST_TASKS = {"exact_provision_lookup", "procedure_or_remedy"}
HYBRID_LAW_TASKS = {"fact_pattern_guidance"}


class LegalAIPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.classifier = LegalClassifier(settings)
        self.retriever = SimilarCaseRetriever(settings)
        self.qa_retriever = LegalQARetriever(settings)
        self.reference_law_retriever = ReferenceLawRetriever(settings)
        self.explainer = ExplanationService(settings)
        self.rag_context_builder = RAGContextBuilder(settings)
        self.query_router = QueryRouterService()
        self.evidence_pack_builder = EvidencePackBuilder(
            max_cards=settings.rag_context_max_authorities + 1
        )
        self.answer_auditor = AnswerAuditService()
        self.workspace_builder = WorkspaceBuilder()
        self.session_documents = SessionDocumentStore(settings)
        self._follow_up_rewrite_cache: dict[tuple[str, str], str] = {}
        self._document_answer_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
        self.retrieval_ready = False
        self.retrieval_status_message = ""
        self.qa_retrieval_ready = False
        self.qa_retrieval_status_message = ""
        self.reference_law_ready = False
        self.reference_law_status_message = ""
        try:
            self.retriever.load()
            self.retrieval_ready = True
            self.retrieval_status_message = (
                f"Case-level retrieval store loaded in the shared {settings.shared_embedding_model_name} embedding space."
            )
        except (FileNotFoundError, RuntimeError) as exc:
            self.retrieval_ready = False
            self.retrieval_status_message = str(exc)
        self.retrieval_record_count = self.retriever.record_count
        try:
            self.qa_retriever.load()
            self.qa_retrieval_ready = True
            self.qa_retrieval_status_message = (
                f"QA chunk retrieval store loaded in the shared {settings.shared_embedding_model_name} embedding space."
            )
        except (FileNotFoundError, RuntimeError) as exc:
            self.qa_retrieval_ready = False
            self.qa_retrieval_status_message = str(exc)
        self.qa_retrieval_record_count = self.qa_retriever.record_count
        try:
            self.reference_law_retriever.load()
            self.reference_law_ready = True
            self.reference_law_status_message = (
                f"Reference-law store loaded in the shared {settings.shared_embedding_model_name} embedding space."
            )
        except (FileNotFoundError, RuntimeError) as exc:
            self.reference_law_ready = False
            self.reference_law_status_message = str(exc)
        self.reference_law_record_count = self.reference_law_retriever.record_count

    def predict(self, payload: PredictionRequest) -> PredictionResponse:
        intake = self._build_intake(payload)
        intake = self._augment_intake_with_uploaded_document(
            intake=intake,
            session_id=payload.session_id,
        )
        if len(normalize_whitespace(intake.get("input_summary"))) < 30:
            raise ValueError(
                "Provide structured case details or upload a readable document before running case review."
            )
        issue_outline = self.workspace_builder.build_issue_outline(intake=intake)
        prediction = self.classifier.predict(intake["input_summary"])
        self._ensure_retrieval_ready("prediction")
        review_query_profile = self._build_case_review_query_profile(intake)
        review_retrieval_query = self._prepare_retrieval_query(
            text=intake["input_summary"],
            raw_question=intake["input_summary"],
            detected_language="English",
            question_profile=review_query_profile,
        )

        retrieval_pool_size = max(payload.top_k + 4, min(payload.top_k * 4, 10))
        similar_case_pool = self.retriever.search(
            review_retrieval_query,
            top_k=retrieval_pool_size,
            metadata_filters=self._metadata_filters(
                intake,
                query_text=review_retrieval_query,
                domain_override=review_query_profile.get("domain"),
            ),
            query_profile=review_query_profile,
        )
        similar_case_pool = self._postfilter_similar_cases(
            similar_cases=similar_case_pool,
            query=review_retrieval_query,
            query_profile=review_query_profile,
            metadata_filters=self._metadata_filters(
                intake,
                query_text=review_retrieval_query,
                domain_override=review_query_profile.get("domain"),
            ),
            top_k=retrieval_pool_size,
        )
        similar_cases = self._select_case_review_authorities(
            similar_cases=similar_case_pool,
            predicted_name=prediction["predicted_name"],
            top_k=payload.top_k,
        )
        authority_map = self.workspace_builder.organize_authorities(
            similar_cases,
            predicted_name=prediction["predicted_name"],
        )
        prediction_posture, prediction_posture_reason = self._derive_prediction_posture(
            prediction=prediction,
            authority_map=authority_map,
        )
        favorability_label, favorability_reason = self._derive_favorability(
            prediction["predicted_name"],
            intake.get("user_role"),
        )
        evidence_gaps = self.workspace_builder.build_evidence_gaps(
            intake=intake,
            similar_cases=similar_cases,
            strict_intake=True,
        )
        workspace = self._build_workspace(
            workflow="triage",
            intake=intake,
            issue_outline=issue_outline,
            evidence_gaps=evidence_gaps,
            authority_map=authority_map,
            confidence_band=prediction["confidence_band"],
        )
        advisories = self._build_advisories(
            prediction=prediction,
            similar_cases=similar_cases,
            favorability_label=favorability_label,
            evidence_gaps=evidence_gaps,
            authority_map=authority_map,
        )
        review_evidence_assessment = self._assess_review_evidence(
            prediction=prediction,
            similar_cases=similar_cases,
            authority_map=authority_map,
        )

        explanation = ""
        explanation_source = "disabled"
        if payload.include_explanation:
            explanation_result = self.explainer.explain(
                case_text=intake["input_summary"],
                prediction=prediction,
                similar_cases=similar_cases,
                intake=intake,
                favorability_label=favorability_label,
                favorability_reason=favorability_reason,
                authority_map=authority_map,
                issue_outline=issue_outline,
                evidence_gaps=evidence_gaps,
            )
            explanation = explanation_result["text"]
            explanation_source = explanation_result["source"]

        return PredictionResponse(
            input_summary=intake["input_summary"],
            case_type=intake.get("case_type"),
            user_role=intake.get("user_role"),
            forum=intake.get("forum"),
            predicted_label=prediction["predicted_label"],
            predicted_name=prediction["predicted_name"],
            prediction_posture=prediction_posture,
            prediction_posture_reason=prediction_posture_reason,
            favorability_label=favorability_label,
            favorability_reason=favorability_reason,
            confidence_score=prediction["confidence_score"],
            confidence_band=prediction["confidence_band"],
            retrieval_confidence=review_evidence_assessment["retrieval_confidence"],
            evidence_strength=review_evidence_assessment["evidence_strength"],
            answer_confidence=review_evidence_assessment["answer_confidence"],
            probabilities=prediction["probabilities"],
            chunk_count=prediction["chunk_count"],
            advisories=advisories,
            similar_cases=self._as_similar_cases(similar_cases),
            explanation=explanation,
            explanation_source=explanation_source,
            workspace=workspace,
        )

    def answer_question(self, payload: QuestionAnswerRequest) -> QuestionAnswerResponse:
        detected_language = "English"
        answer_language = "English"
        source_mode = self._resolve_source_mode(payload.source_mode)
        chat_history = [item.model_dump() for item in payload.chat_history]
        subquestions = self._split_compound_question(payload.question)
        if len(subquestions) > 1:
            turn_results = [
                self._run_single_qa_turn(
                    payload=payload,
                    question=subquestion,
                    detected_language=detected_language,
                    answer_language=answer_language,
                    chat_history=chat_history,
                    source_mode=source_mode,
                )
                for subquestion in subquestions[:3]
            ]
            combined_cases = self._merge_similar_cases(
                [result["similar_cases"] for result in turn_results]
            )
            combined_authority_map = self.workspace_builder.organize_authorities(combined_cases)
            payload_dict = payload.model_dump()
            issue_outline = self.workspace_builder.build_issue_outline(
                intake=payload_dict,
                question=payload.question,
            )
            evidence_gaps = self.workspace_builder.build_evidence_gaps(
                intake=payload_dict,
                question=payload.question,
                similar_cases=combined_cases,
                strict_intake=False,
            )
            combined_scope_case_ids: list[str] = []
            for result in turn_results:
                for case_id in result["scope_case_ids"]:
                    if case_id not in combined_scope_case_ids:
                        combined_scope_case_ids.append(case_id)
            workspace = self._build_workspace(
                workflow="ask",
                intake=payload_dict,
                question=payload.question,
                issue_outline=issue_outline,
                evidence_gaps=evidence_gaps,
                authority_map=combined_authority_map,
                scope_case_ids=combined_scope_case_ids[:5],
            )
            combined_answer_parts = []
            combined_sources: list[str] = []
            combined_advisories: list[str] = []
            combined_reference_materials: list[dict[str, Any]] = []
            answer_source = "llm"
            retrieval_queries: list[str] = []
            combined_source_modes: list[str] = []
            for index, result in enumerate(turn_results, start=1):
                answer_source = (
                    answer_source
                    if result["answer_result"]["source"] == "llm"
                    else result["answer_result"]["source"]
                )
                retrieval_queries.append(result["retrieval_query"])
                combined_source_modes.append(result["source_mode"])
                body = self._compose_qa_answer(
                    answer_text=result["answer_result"]["text"],
                    authority_map=result["authority_map"],
                    rag_context=result["rag_context"],
                    scope=payload.scope,
                    source_mode=result["source_mode"],
                    include_sources=False,
                )
                combined_answer_parts.append(f"{index}. {body}")
                for case_id in result["scope_case_ids"]:
                    if case_id not in combined_sources:
                        combined_sources.append(case_id)
                for material in result.get("reference_materials") or []:
                    key = (
                        material.get("title"),
                        material.get("section_ref"),
                    )
                    if not any(
                        existing.get("title") == key[0] and existing.get("section_ref") == key[1]
                        for existing in combined_reference_materials
                    ):
                        combined_reference_materials.append(material)
                for advisory in result["advisories"]:
                    if advisory not in combined_advisories:
                        combined_advisories.append(advisory)
            answer_text = "\n\n".join(combined_answer_parts)
            if combined_sources:
                answer_text += f"\n\nSources: {', '.join(combined_sources[:5])}"
            if combined_reference_materials:
                law_sources = [
                    self._format_reference_material_label(item)
                    for item in combined_reference_materials[:3]
                ]
                answer_text += f"\n\nLaw sources: {', '.join(law_sources)}"
            response_source_mode = (
                combined_source_modes[0]
                if combined_source_modes and len(set(combined_source_modes)) == 1
                else source_mode
            )
            combined_rewritten = [result.get("rewritten_question") for result in turn_results if result.get("rewritten_question")]
            combined_evidence = [result.get("evidence_assessment") or {} for result in turn_results]
            retrieval_confidence = "low"
            evidence_strength = "insufficient"
            answer_confidence = "low"
            if combined_evidence:
                if any(item.get("retrieval_confidence") == "high" for item in combined_evidence):
                    retrieval_confidence = "moderate"
                if all(item.get("retrieval_confidence") == "high" for item in combined_evidence):
                    retrieval_confidence = "high"
                if any(item.get("evidence_strength") == "mixed" for item in combined_evidence):
                    evidence_strength = "mixed"
                elif all(item.get("evidence_strength") == "supported" for item in combined_evidence):
                    evidence_strength = "supported"
                if any(item.get("answer_confidence") == "moderate" for item in combined_evidence):
                    answer_confidence = "moderate"
                if all(item.get("answer_confidence") == "high" for item in combined_evidence):
                    answer_confidence = "high"
            follow_up_suggestions = self._build_follow_up_suggestions(
                authority_map=combined_authority_map,
                scope=payload.scope,
                question=payload.question,
                source_mode=response_source_mode,
                evidence_assessment={
                    "retrieval_confidence": retrieval_confidence,
                    "evidence_strength": evidence_strength,
                    "answer_confidence": answer_confidence,
                },
            )
            return QuestionAnswerResponse(
                question=payload.question.strip(),
                retrieval_query=" || ".join(retrieval_queries),
                rewritten_question=" || ".join(combined_rewritten) if combined_rewritten else None,
                detected_language=detected_language,
                answer_language=answer_language,
                scope=payload.scope,
                source_mode=response_source_mode,
                answer=answer_text,
                answer_source=answer_source,
                retrieval_confidence=retrieval_confidence,
                evidence_strength=evidence_strength,
                answer_confidence=answer_confidence,
                advisories=combined_advisories[:5],
                follow_up_suggestions=follow_up_suggestions,
                supporting_cases=self._as_similar_cases(combined_cases),
                reference_materials=self._as_reference_materials(combined_reference_materials),
                workspace=workspace,
            )

        turn_result = self._run_single_qa_turn(
            payload=payload,
            question=payload.question,
            detected_language=detected_language,
            answer_language=answer_language,
            chat_history=chat_history,
            source_mode=source_mode,
        )
        answer_text = self._compose_qa_answer(
            answer_text=turn_result["answer_result"]["text"],
            authority_map=turn_result["authority_map"],
            rag_context=turn_result["rag_context"],
            scope=payload.scope,
            source_mode=turn_result["source_mode"],
        )
        follow_up_suggestions = self._build_follow_up_suggestions(
            authority_map=turn_result["authority_map"],
            scope=payload.scope,
            question=payload.question,
            source_mode=turn_result["source_mode"],
            evidence_assessment=turn_result.get("evidence_assessment") or {},
        )
        return QuestionAnswerResponse(
            question=payload.question.strip(),
            retrieval_query=turn_result["retrieval_query"],
            rewritten_question=turn_result.get("rewritten_question"),
            detected_language=detected_language,
            answer_language=answer_language,
            scope=payload.scope,
            source_mode=turn_result["source_mode"],
            answer=answer_text,
            answer_source=turn_result["answer_result"]["source"],
            retrieval_confidence=(turn_result.get("evidence_assessment") or {}).get("retrieval_confidence"),
            evidence_strength=(turn_result.get("evidence_assessment") or {}).get("evidence_strength"),
            answer_confidence=(turn_result.get("evidence_assessment") or {}).get("answer_confidence"),
            advisories=turn_result["advisories"],
            follow_up_suggestions=follow_up_suggestions,
            supporting_cases=self._as_similar_cases(turn_result["similar_cases"]),
            reference_materials=self._as_reference_materials(turn_result.get("reference_materials") or []),
            workspace=turn_result["workspace"],
        )

    def _run_single_qa_turn(
        self,
        *,
        payload: QuestionAnswerRequest,
        question: str,
        detected_language: str,
        answer_language: str,
        chat_history: list[dict[str, str]],
        source_mode: str,
    ) -> dict[str, Any]:
        payload_dict = payload.model_dump()
        payload_dict["question"] = question
        referenced_case_ids = self._resolve_referenced_case_ids(question, chat_history)
        follow_up_context = self._derive_follow_up_context(question, chat_history)
        topic_drift = self._detect_topic_drift(
            question=question,
            follow_up_context=follow_up_context,
            chat_history=chat_history,
        )
        rewritten_question = self._rewrite_follow_up_question(
            question=question,
            follow_up_context=follow_up_context,
            chat_history=chat_history,
        )
        if topic_drift:
            follow_up_context = None
        question_profile = self.query_router.analyze(
            question=question,
            chat_history=chat_history,
            session_has_document=self.session_documents.has_document(payload.session_id),
            requested_source_mode=source_mode,
            case_type_hint=payload.case_type,
            forum_hint=payload.forum,
            context_note=payload.context_note,
        )
        question_profile_dict = question_profile.to_dict()
        question_profile_dict["rewritten_question"] = rewritten_question
        question_profile_dict["topic_drift"] = topic_drift
        effective_source_mode = self._resolve_effective_source_mode(
            requested_source_mode=source_mode,
            question_route=question_profile.task,
            session_id=payload.session_id,
        )
        source_plan = self._build_source_plan(
            effective_source_mode=effective_source_mode,
            question_profile=question_profile_dict,
            session_id=payload.session_id,
        )
        retrieval_profile = self._resolve_retrieval_profile(
            value=payload.retrieval_profile,
            source_mode=effective_source_mode,
            recommended_profile=question_profile.complexity,
        )
        question_profile_dict["retrieval_profile"] = retrieval_profile
        base_retrieval_query = self._build_question_query(
            payload,
            chat_history=chat_history,
            follow_up_context=follow_up_context,
            referenced_case_ids=referenced_case_ids,
            question_override=rewritten_question,
            question_profile=question_profile_dict,
        )
        retrieval_query = self._prepare_retrieval_query(
            text=base_retrieval_query,
            raw_question=rewritten_question,
            question_profile=question_profile_dict,
            detected_language=detected_language,
        )
        reference_law_query = self._build_reference_law_query(
            question=question,
            rewritten_question=rewritten_question,
            question_profile=question_profile_dict,
        )
        use_case_corpus = bool(source_plan["use_case_corpus"])
        use_uploaded_document = bool(source_plan["use_uploaded_document"])
        use_reference_law = bool(source_plan["use_reference_law"]) and self._should_use_reference_law(
            source_mode=effective_source_mode,
            question=question,
            retrieval_query=retrieval_query,
            question_profile=question_profile_dict,
        )
        document_cache_key = (
            normalize_whitespace(payload.session_id or ""),
            normalize_whitespace(question).lower(),
            f"{question_profile_dict.get('answer_style')}|{question_profile_dict.get('response_length')}",
        )
        document_direct_answer = (
            self._document_answer_cache.get(document_cache_key)
            or self.session_documents.answer_question(
                session_id=payload.session_id,
                question=question,
                question_profile=question_profile_dict,
                follow_up_context=follow_up_context,
                encoder=self.qa_retriever,
            )
            if use_uploaded_document
            and (
                effective_source_mode == "document_only"
                or question_profile.task in {"document_fact", "document_reasoning"}
            )
            else None
        )
        if document_direct_answer:
            self._document_answer_cache[document_cache_key] = dict(document_direct_answer)
        if effective_source_mode == "document_only" and document_direct_answer:
            payload_dict = payload.model_dump()
            payload_dict["question"] = question
            issue_outline = self.workspace_builder.build_issue_outline(
                intake=payload_dict,
                question=question,
            )
            evidence_gaps = self.workspace_builder.build_evidence_gaps(
                intake=payload_dict,
                question=question,
                similar_cases=[],
                strict_intake=False,
            )
            authority_map = self.workspace_builder.organize_authorities([])
            workspace = self._build_workspace(
                workflow="ask",
                intake=payload_dict,
                question=question,
                issue_outline=issue_outline,
                evidence_gaps=evidence_gaps,
                authority_map=authority_map,
                scope_case_ids=[],
            )
            document_info = self.session_documents.get_document_info(payload.session_id) or {}
            document_context = {
                "used": True,
                "document_id": document_info.get("document_id"),
                "filename": document_info.get("filename"),
                "coverage_note": "Fast document-only answer extracted directly from the uploaded file.",
                "context_text": "",
            }
            advisories = []
            if str(document_direct_answer.get("confidence") or "").lower() != "high":
                advisories.append(
                    "This is a fast document extract. Open the fuller reasoning only if you need a deeper reading."
                )
            return {
                "retrieval_query": retrieval_query,
                "similar_cases": [],
                "rag_context": document_context,
                "evidence_pack": {"cards": [], "case_ids": [], "direct_support_count": 0},
                "authority_map": authority_map,
                "workspace": workspace,
                "answer_result": {
                    "text": document_direct_answer["text"],
                    "source": "document_extract",
                },
                "advisories": advisories,
                "scope_case_ids": [],
                "reference_materials": [],
                "source_mode": effective_source_mode,
                "retrieval_profile": retrieval_profile,
                "rewritten_question": rewritten_question if rewritten_question != normalize_whitespace(question) else None,
                "evidence_assessment": {
                    "retrieval_confidence": "high",
                    "evidence_strength": "supported",
                    "answer_confidence": "high" if str(document_direct_answer.get("confidence") or "").lower() == "high" else "moderate",
                },
            }
        if use_case_corpus or use_uploaded_document:
            self._ensure_qa_retrieval_ready("question answering")
        scope_case_ids = list(dict.fromkeys(payload.scope_case_ids))[:5] if use_case_corpus else []
        search_case_ids: list[str] | None = None
        if use_case_corpus and referenced_case_ids:
            search_case_ids = referenced_case_ids[:5]
        elif use_case_corpus and payload.scope == "current_result":
            search_case_ids = scope_case_ids
        elif use_case_corpus and not topic_drift and self._should_scope_follow_up(question, chat_history):
            search_case_ids = self._extract_recent_case_ids(chat_history)[:5] or None
        metadata_filters = self._metadata_filters(
            payload_dict,
            query_text=retrieval_query,
            domain_override=question_profile.domain,
        )
        if (
            use_case_corpus
            and question_profile.domain
            and not question_profile.supported_case_law_domain
            and not question_profile.direct_case_lookup
        ):
            similar_cases = []
            shortlisted_case_ids = []
        elif use_case_corpus:
            similar_cases, shortlisted_case_ids = self._retrieve_case_corpus_authorities(
                retrieval_query=retrieval_query,
                top_k=payload.top_k,
                metadata_filters=metadata_filters,
                preferred_case_ids=search_case_ids,
                retrieval_profile=retrieval_profile,
                query_profile=question_profile_dict,
            )
        else:
            similar_cases = []
            shortlisted_case_ids = []
        similar_cases = self._postfilter_similar_cases(
            similar_cases=similar_cases,
            query=retrieval_query,
            query_profile=question_profile_dict,
            metadata_filters=metadata_filters,
            top_k=payload.top_k,
        )
        document_hits = (
            self.session_documents.search(
                session_id=payload.session_id,
                query=(rewritten_question if question_profile.task in {"document_fact", "document_reasoning"} else retrieval_query),
                top_k=min(max(payload.top_k, 3), 4),
                encoder=self.qa_retriever,
            )
            if use_uploaded_document
            else []
        )
        document_context = (
            self.session_documents.build_context(
                session_id=payload.session_id,
                hits=document_hits,
            )
            if use_uploaded_document
            else self._empty_document_context()
        )
        reference_law_hits = (
            self.reference_law_retriever.search(
                reference_law_query,
                top_k=max(min(payload.top_k, self.settings.reference_law_max_hits), 2),
                question_profile=question_profile_dict,
            )
            if use_reference_law and self.reference_law_ready
            else []
        )
        law_context = (
            self.reference_law_retriever.build_context(reference_law_hits)
            if reference_law_hits
            else self._empty_law_context()
        )
        actual_source_mode = self._derive_response_source_mode(
            requested_source_mode=effective_source_mode,
            document_used=bool(document_context.get("used")),
            law_used=bool(law_context.get("used")),
            case_used=bool(similar_cases),
        )
        evidence_pack = self.evidence_pack_builder.build(
            question=question,
            question_profile=question_profile_dict,
            similar_cases=similar_cases,
            source_mode=actual_source_mode,
            scope=payload.scope,
            document_context=document_context,
            law_context=law_context,
        )
        rag_context = self.rag_context_builder.build(
            question=question,
            similar_cases=similar_cases,
            scope=payload.scope,
            source_mode=actual_source_mode,
            document_context=document_context,
            evidence_pack=evidence_pack,
            law_context=law_context,
        )
        authority_map = self.workspace_builder.organize_authorities(similar_cases)
        issue_outline = self.workspace_builder.build_issue_outline(
            intake=payload_dict,
            question=question,
        )
        evidence_gaps = self.workspace_builder.build_evidence_gaps(
            intake=payload_dict,
            question=question,
            similar_cases=similar_cases,
            strict_intake=False,
        )
        workspace = self._build_workspace(
            workflow="ask",
            intake=payload_dict,
            question=question,
            issue_outline=issue_outline,
            evidence_gaps=evidence_gaps,
            authority_map=authority_map,
            scope_case_ids=shortlisted_case_ids or rag_context["used_case_ids"],
        )
        evidence_assessment = self._assess_qa_evidence(
            question=question,
            similar_cases=similar_cases,
            case_type_hint=payload.case_type,
            referenced_case_ids=referenced_case_ids,
            source_mode=actual_source_mode,
            document_context=document_context,
            law_context=law_context,
            session_id=payload.session_id,
            question_profile=question_profile_dict,
            evidence_pack=evidence_pack,
        )
        evidence_status = str(evidence_assessment.get("status") or "ok")
        evidence_reason = evidence_assessment.get("reason")
        if document_direct_answer and effective_source_mode == "document_only":
            answer_result = {
                "text": document_direct_answer["text"],
                "source": "document_extract",
            }
        elif self._should_use_reference_law_direct_answer(
            question_profile=question_profile_dict,
            law_context=law_context,
            evidence_assessment=evidence_assessment,
        ):
            answer_result = {
                "text": self._build_reference_law_direct_answer(
                    question=question,
                    question_profile=question_profile_dict,
                    law_context=law_context,
                    similar_cases=similar_cases,
                ),
                "source": "reference_law_builder",
            }
        elif evidence_status == "poor":
            answer_result = {
                "text": self._build_cautious_qa_answer(
                    question=question,
                    reason=evidence_reason,
                    referenced_case_ids=referenced_case_ids,
                    similar_cases=similar_cases,
                    source_mode=actual_source_mode,
                    question_profile=question_profile_dict,
                ),
                "source": "guardrail",
            }
        else:
            answer_chat_history = (
                chat_history
                if (not topic_drift and self._should_use_answer_history(question, chat_history=chat_history))
                else []
            )
            if question_profile.task == "similarity_lookup" and similar_cases:
                answer_result = {
                    "text": self._build_similarity_lookup_answer(
                        question=question,
                        similar_cases=similar_cases,
                        evidence_pack=evidence_pack,
                    ),
                    "source": "retrieval_summary",
                }
            elif self.settings.demo_mode and question_profile.task == "case_explanation" and similar_cases:
                answer_result = {
                    "text": self._build_demo_case_explanation_answer(similar_cases[0]),
                    "source": "demo_case_summary",
                }
            else:
                llm_answer_result = self.explainer.answer_question(
                    question=question,
                    retrieval_query=retrieval_query,
                    similar_cases=similar_cases,
                    rag_context=rag_context,
                    evidence_pack=evidence_pack,
                    question_profile=question_profile_dict,
                    detected_language=detected_language,
                    answer_language=answer_language,
                    filters=payload_dict,
                    chat_history=answer_chat_history,
                    scope=payload.scope,
                    source_mode=actual_source_mode,
                    retrieval_profile=retrieval_profile,
                )
                answer_result = llm_answer_result

        audited_answer = self.answer_auditor.audit(
            question=question,
            question_profile=question_profile_dict,
            evidence_pack=evidence_pack,
            answer_text=answer_result["text"],
            source_mode=actual_source_mode,
        )
        answer_result = {
            "text": audited_answer["text"],
            "source": answer_result["source"],
        }

        advisories = self._build_qa_advisories(
            similar_cases=similar_cases,
            scope=payload.scope,
            answer_source=answer_result["source"],
            evidence_gaps=evidence_gaps,
            authority_map=authority_map,
            document_used=bool(document_context.get("used")),
            source_mode=actual_source_mode,
            evidence_assessment=evidence_assessment,
        )
        for advisory in audited_answer.get("advisories") or []:
            if advisory not in advisories:
                advisories.insert(0, advisory)
        if evidence_status == "weak" and evidence_reason and evidence_reason not in advisories:
            advisories.insert(0, evidence_reason)
        return {
            "retrieval_query": retrieval_query,
            "similar_cases": similar_cases,
            "rag_context": rag_context,
            "evidence_pack": evidence_pack,
            "authority_map": authority_map,
            "workspace": workspace,
            "answer_result": answer_result,
            "advisories": advisories[:5],
            "scope_case_ids": ((shortlisted_case_ids or rag_context["used_case_ids"])[:5] if use_case_corpus else []),
            "reference_materials": list(law_context.get("materials") or []),
            "source_mode": actual_source_mode,
            "retrieval_profile": retrieval_profile,
            "rewritten_question": rewritten_question if rewritten_question != normalize_whitespace(question) else None,
            "evidence_assessment": evidence_assessment,
        }

    def research(self, payload: ResearchRequest) -> ResearchResponse:
        self._ensure_qa_retrieval_ready("research")
        base_retrieval_query = self._build_research_query(payload)
        research_profile = self.query_router.analyze(
            question=payload.topic_query,
            chat_history=[],
            session_has_document=False,
            requested_source_mode="case_corpus_only",
            case_type_hint=payload.case_type,
            forum_hint=payload.forum,
        ).to_dict()
        research_profile["workflow"] = "research"
        research_profile["task"] = "general_research"
        research_profile["retrieval_profile"] = "deep"
        retrieval_query = self._prepare_retrieval_query(
            text=base_retrieval_query,
            raw_question=payload.topic_query,
            detected_language="English",
            question_profile=research_profile,
        )
        similar_cases, _shortlisted_case_ids = self._retrieve_case_corpus_authorities(
            retrieval_query=retrieval_query,
            top_k=payload.top_k,
            metadata_filters=self._metadata_filters(
                payload.model_dump(),
                query_text=retrieval_query,
                domain_override=research_profile.get("domain"),
            ),
            retrieval_profile="deep",
            query_profile=research_profile,
        )
        authority_map = self.workspace_builder.organize_authorities(similar_cases)
        issue_outline = self.workspace_builder.build_issue_outline(
            intake=payload.model_dump(),
            topic_query=payload.topic_query,
        )
        evidence_gaps = self.workspace_builder.build_evidence_gaps(
            intake=payload.model_dump(),
            question=payload.topic_query,
            similar_cases=similar_cases,
            strict_intake=False,
        )
        workspace = self._build_workspace(
            workflow="research",
            intake=payload.model_dump(),
            topic_query=payload.topic_query,
            issue_outline=issue_outline,
            evidence_gaps=evidence_gaps,
            authority_map=authority_map,
        )
        research_snapshot = self.workspace_builder.build_research_snapshot(
            topic_query=payload.topic_query,
            authority_map=authority_map,
            issue_outline=issue_outline,
            evidence_gaps=evidence_gaps,
        )
        advisories = self._build_research_advisories(
            similar_cases=similar_cases,
            authority_map=authority_map,
            evidence_gaps=evidence_gaps,
        )
        return ResearchResponse(
            topic_query=payload.topic_query.strip(),
            retrieval_query=retrieval_query,
            research_snapshot=research_snapshot,
            research_snapshot_source="workspace",
            advisories=advisories,
            retrieved_cases=self._as_similar_cases(similar_cases),
            workspace=workspace,
        )

    def get_case_detail(self, case_id: str) -> CaseDetailResponse | None:
        if not self.retrieval_ready:
            raise RuntimeError(
                "Retrieval index is not ready. Run build_retrieval_store.py before opening evidence cases."
            )
        payload = self.retriever.get_case_detail(case_id)
        if payload is None:
            return None
        label = payload.get("label")
        return CaseDetailResponse(
            case_id=payload["case_id"],
            label=label,
            label_name=LABEL_ID_TO_NAME.get(label) if label is not None else None,
            title=payload.get("title"),
            court=payload.get("court"),
            date=payload.get("date"),
            word_count=payload["word_count"],
            full_text=payload["full_text"],
        )

    def store_session_document(
        self,
        *,
        session_id: str,
        filename: str,
        content_type: str,
        file_bytes: bytes,
    ) -> dict[str, Any]:
        return self.session_documents.upsert(
            session_id=session_id,
            filename=filename,
            content_type=content_type,
            file_bytes=file_bytes,
            encoder=self.qa_retriever,
        )

    def clear_session_document(self, session_id: str) -> bool:
        return self.session_documents.clear(session_id)

    @staticmethod
    def _build_intake(payload: PredictionRequest) -> dict[str, str]:
        case_type = normalize_whitespace(payload.case_type)
        user_role = normalize_whitespace(payload.user_role)
        forum = normalize_whitespace(payload.forum)
        facts = normalize_whitespace(payload.facts)
        relief_sought = normalize_whitespace(payload.relief_sought)
        evidence_summary = normalize_whitespace(payload.evidence_summary)
        opponent_arguments = normalize_whitespace(payload.opponent_arguments)
        raw_case_text = normalize_whitespace(payload.case_text)

        intake = {
            "input_summary": "",
            "case_type": case_type,
            "user_role": user_role,
            "forum": forum,
            "facts": facts,
            "relief_sought": relief_sought,
            "evidence_summary": evidence_summary,
            "opponent_arguments": opponent_arguments,
            "case_text": raw_case_text,
        }
        intake["input_summary"] = LegalAIPipeline._compose_intake_summary(intake)
        return intake

    @staticmethod
    def _compose_intake_summary(intake: dict[str, str | None]) -> str:
        sections: list[str] = []
        case_type = normalize_whitespace(intake.get("case_type"))
        user_role = normalize_whitespace(intake.get("user_role"))
        forum = normalize_whitespace(intake.get("forum"))
        facts = normalize_whitespace(intake.get("facts"))
        relief_sought = normalize_whitespace(intake.get("relief_sought"))
        evidence_summary = normalize_whitespace(intake.get("evidence_summary"))
        opponent_arguments = normalize_whitespace(intake.get("opponent_arguments"))
        raw_case_text = normalize_whitespace(intake.get("case_text"))

        if case_type:
            sections.append(f"Case type: {case_type}.")
        if user_role:
            sections.append(f"User role: {user_role}.")
        if forum:
            sections.append(f"Forum or court type: {forum}.")
        if facts:
            sections.append(f"Facts: {facts}")
        if relief_sought:
            sections.append(f"Relief sought: {relief_sought}")
        if evidence_summary:
            sections.append(f"Important evidence and documents: {evidence_summary}")
        if opponent_arguments:
            sections.append(f"Opponent's main argument: {opponent_arguments}")
        if raw_case_text:
            sections.append(f"Additional case narrative: {raw_case_text}")
        return normalize_whitespace(" ".join(sections))

    def _augment_intake_with_uploaded_document(
        self,
        *,
        intake: dict[str, str],
        session_id: str | None,
    ) -> dict[str, str]:
        if not self.session_documents.has_document(session_id):
            return intake

        document_query = intake.get("input_summary") or "facts dispute relief evidence documents"
        document_hits = self.session_documents.search(
            session_id=session_id,
            query=document_query,
            top_k=3,
            encoder=self.qa_retriever,
        )
        if not document_hits:
            return intake

        document_excerpt = " ".join(
            hit.get("excerpt") or ""
            for hit in document_hits[:3]
            if hit.get("excerpt")
        ).strip()
        if not document_excerpt:
            return intake

        augmented = dict(intake)
        current_case_text = normalize_whitespace(augmented.get("case_text"))
        uploaded_note = f"Uploaded document context: {document_excerpt}"
        augmented["case_text"] = normalize_whitespace(
            " ".join(part for part in [current_case_text, uploaded_note] if part)
        )
        augmented["input_summary"] = self._compose_intake_summary(augmented)
        return augmented

    def _ensure_retrieval_ready(self, task_name: str) -> None:
        if not self.retrieval_ready:
            detail = f" {self.retrieval_status_message}" if self.retrieval_status_message else ""
            raise RuntimeError(
                f"Retrieval index is not ready. Run build_retrieval_store.py before using {task_name}.{detail}"
            )

    def _ensure_qa_retrieval_ready(self, task_name: str) -> None:
        if not self.qa_retrieval_ready:
            detail = (
                f" {self.qa_retrieval_status_message}"
                if self.qa_retrieval_status_message
                else ""
            )
            raise RuntimeError(
                f"QA retrieval index is not ready. Run build_retrieval_store.py before using {task_name}.{detail}"
            )

    @staticmethod
    def _build_question_query(
        payload: QuestionAnswerRequest,
        *,
        chat_history: list[dict[str, str]] | None = None,
        follow_up_context: str | None = None,
        referenced_case_ids: list[str] | None = None,
        question_override: str | None = None,
        question_profile: dict[str, Any] | None = None,
    ) -> str:
        question = normalize_whitespace(question_override or payload.question)
        case_type = normalize_whitespace(payload.case_type)
        user_role = normalize_whitespace(payload.user_role)
        forum = normalize_whitespace(payload.forum)
        context_note = normalize_whitespace(payload.context_note)
        recent_case_ids = LegalAIPipeline._extract_recent_case_ids(chat_history or [])
        question_word_count = len(re.findall(r"\b\w+\b", question))
        is_generic_follow_up = bool(
            follow_up_context
            and (
                question_word_count <= 5
                or LegalAIPipeline._is_style_only_follow_up(question)
                or question.lower().startswith(
                    (
                        "explain in detail",
                        "explain this",
                        "explain it",
                        "summarize this",
                        "tell me more",
                    )
                )
            )
        )
        if is_generic_follow_up:
            sections = [f"Legal question: {follow_up_context}"]
            sections.append(f"Follow-up request: {question}")
        else:
            sections = [f"Legal question: {question}"]
        if follow_up_context:
            sections.append(f"Conversation context: {follow_up_context}")
        if referenced_case_ids:
            sections.append(
                "Referenced case ids: " + ", ".join(referenced_case_ids[:3]) + "."
            )
        elif follow_up_context and recent_case_ids:
            sections.append(
                "Recent cited case ids: " + ", ".join(recent_case_ids[:3]) + "."
            )
        if case_type:
            sections.append(f"Case type context: {case_type}.")
        if user_role:
            sections.append(f"User role context: {user_role}.")
        if forum:
            sections.append(f"Forum context: {forum}.")
        if context_note:
            sections.append(f"Relevant factual context: {context_note}")
        profile = question_profile or {}
        if profile.get("domain"):
            sections.append(f"Detected legal domain: {profile['domain']}.")
        if profile.get("legal_elements"):
            sections.append(
                "Legal elements: "
                + ", ".join(str(item).replace("_", " ") for item in profile["legal_elements"][:6])
                + "."
            )
        if profile.get("task"):
            sections.append(f"Task focus: {profile['task']}.")
        return normalize_whitespace(" ".join(sections))

    @staticmethod
    def _build_research_query(payload: ResearchRequest) -> str:
        sections = [f"Research topic: {normalize_whitespace(payload.topic_query)}"]
        if normalize_whitespace(payload.case_type):
            sections.append(f"Case type context: {normalize_whitespace(payload.case_type)}.")
        if normalize_whitespace(payload.forum):
            sections.append(f"Forum context: {normalize_whitespace(payload.forum)}.")
        if normalize_whitespace(payload.user_role):
            sections.append(f"Role context: {normalize_whitespace(payload.user_role)}.")
        return normalize_whitespace(" ".join(sections))

    @staticmethod
    def _build_reference_law_query(
        *,
        question: str,
        rewritten_question: str,
        question_profile: dict[str, Any],
    ) -> str:
        base_question = normalize_whitespace(question or rewritten_question).replace("artical", "article")
        lowered = base_question.lower()
        domain = normalize_whitespace(question_profile.get("domain")).lower()
        expansions: list[str] = []
        task = str(question_profile.get("task") or "")
        if "article 21a" in lowered or "right to education" in lowered:
            expansions.append("constitution of india article 21a free and compulsory education children six to fourteen years")
        elif (
            "article 300a" in lowered
            or "right to property" in lowered
            or ("property" in lowered and "authority of law" in lowered)
            or "deprived of property" in lowered
        ):
            expansions.append("constitution of india article 300a no person shall be deprived of property save by authority of law")
        elif "article 21" in lowered or "personal liberty" in lowered:
            expansions.append("constitution of india article 21 personal liberty procedure established by law")
        elif "article" in lowered:
            expansions.append("constitution of india fundamental rights article")
        elif domain == "consumer":
            if "section 39" in lowered or task in LAW_FIRST_TASKS:
                expansions.append("consumer protection act 2019 section 39 remove defects replace goods refund price compensation punitive damages")
            if any(marker in lowered for marker in ("remedy", "remedies", "refund", "replacement", "repair", "compensation", "relief")):
                expansions.append("consumer protection act refund replacement repair compensation findings district commission")
            if any(marker in lowered for marker in ("defect", "defective", "service centre", "service center", "warranty", "repeated repair", "failed repair", "unrepaired")):
                expansions.append("consumer protection act defect deficiency in service repeated repair refund replacement compensation warranty")
            if any(marker in lowered for marker in ("wrong product", "return window", "return expired", "delivered wrong", "online marketplace", "e-commerce", "e commerce")):
                expansions.append("consumer protection act 2019 section 39 refund replacement deficiency in service unfair trade practice e commerce")
            if "e commerce" in lowered or "e-commerce" in lowered:
                expansions.append("consumer protection e commerce rules marketplace seller platform")
        elif domain == "privacy":
            if any(marker in lowered for marker in ("job portal", "third party", "third-party", "personal data", "without consent", "data protection")):
                expansions.append("digital personal data protection act personal data consent data fiduciary data principal grievance")
            if any(marker in lowered for marker in ("cctv", "surveillance", "recording", "records my family", "house entrance")):
                expansions.append("privacy surveillance cctv recording private act bharatiya nyaya sanhita voyeurism information technology act")
            else:
                expansions.append("privacy law personal data consent grievance data protection")
        elif domain == "information":
            if any(marker in lowered for marker in ("first appeal", "second appeal")):
                expansions.append("right to information act section 19 first appeal second appeal thirty days ninety days sufficient cause")
            if any(
                marker in lowered
                for marker in ("reply", "response", "respond", "answered", "not answered", "no response", "pio", "cpio")
            ) and any(marker in lowered for marker in ("time", "days", "limit", "within", "30")):
                expansions.append("right to information act section 7 public information officer thirty days forty eight hours life or liberty")
            if any(marker in lowered for marker in ("not answered", "no response", "no reply", "not received", "within 30 days")):
                expansions.append("right to information act section 19 first appeal no response public information officer")
            if any(marker in lowered for marker in ("personal information", "8(1)(j)")):
                expansions.append("right to information act section 8(1)(j) personal information public activity public interest unwarranted invasion of privacy")
            if any(marker in lowered for marker in ("commercial confidence", "trade secret", "intellectual property", "8(1)(d)")):
                expansions.append("right to information act section 8(1)(d) commercial confidence trade secret intellectual property larger public interest")
            if any(marker in lowered for marker in ("inspection", "inspect records", "certified copies", "2(j)")):
                expansions.append("right to information act section 2(j) inspection of records notes extracts certified copies")
            if any(marker in lowered for marker in ("private company", "private body", "private entity")):
                expansions.append("right to information act public authority private body information accessible through public authority section 2(f)")
            if "limitation" in lowered or "appeal" in lowered or "time" in lowered:
                expansions.append("rti act first appeal second appeal within thirty days information officer")
            else:
                expansions.append("rti act public information officer disclosure exemption appeal")
        elif domain == "service" or "ccs" in lowered:
            if "rule 14" in lowered or "major penalty" in lowered:
                expansions.append("ccs cca rules rule 14 procedure for imposing major penalties disciplinary inquiry")
            if "rule 16" in lowered or "minor penalty" in lowered:
                expansions.append("ccs cca rules rule 16 procedure for imposing minor penalties")
            if any(marker in lowered for marker in ("disciplinary", "penalty", "inquiry", "suspension", "charge sheet")):
                expansions.append("ccs cca rules disciplinary proceedings inquiry penalties suspension")
            if any(marker in lowered for marker in ("difference between rule 14 and rule 16", "rule 14 and rule 16")):
                expansions.append("ccs cca rules difference between rule 14 major penalty and rule 16 minor penalty")
            else:
                expansions.append("administrative tribunals act ccs conduct rules service matter")
        elif domain == "tax":
            if "gst" in lowered:
                expansions.append("cgst act appeal input tax credit penalty refund")
            elif "excise" in lowered:
                expansions.append("central excise act appeal duty penalty")
            else:
                expansions.append("income tax act appeal limitation penalty assessment")
        elif domain == "motor_accident":
            expansions.append("motor vehicles act compensation claims tribunal liability")
        elif domain == "criminal":
            expansions.append("bns bnss bsa punishment bail procedure evidence")
        parts = [base_question]
        if expansions:
            parts.extend(expansions[:2])
        return normalize_whitespace(" ".join(parts))

    @staticmethod
    def _split_compound_question(question: str) -> list[str]:
        cleaned = (question or "").strip()
        if not cleaned:
            return []

        normalized = re.sub(r"(?<!\n)(\s+\d+\s*[\.\)])", r"\n\1", cleaned)
        numbered_matches = re.findall(
            r"(?:^|\n)\s*\d+\s*[\.\)]\s*(.*?)(?=(?:\n\s*\d+\s*[\.\)])|$)",
            normalized,
            flags=re.S,
        )
        numbered_parts = [
            normalize_whitespace(part).rstrip("?!.") + "?"
            for part in numbered_matches
            if len(normalize_whitespace(part)) >= 12
        ]
        if len(numbered_parts) >= 2:
            return numbered_parts[:3]

        if cleaned.count("?") >= 2:
            sentence_parts = [
                normalize_whitespace(part).rstrip("?!.") + "?"
                for part in re.split(r"\?\s*", cleaned)
                if len(normalize_whitespace(part)) >= 12
            ]
            if len(sentence_parts) >= 2:
                return sentence_parts[:3]

        return [cleaned]

    @staticmethod
    def _history_without_current(
        question: str,
        chat_history: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        cleaned_question = normalize_whitespace(question)
        history = list(chat_history or [])
        if (
            history
            and history[-1].get("role") == "user"
            and normalize_whitespace(history[-1].get("content") or "") == cleaned_question
        ):
            return history[:-1]
        return history

    @classmethod
    def _derive_follow_up_context(
        cls,
        question: str,
        chat_history: list[dict[str, str]],
    ) -> str | None:
        prior_history = cls._history_without_current(question, chat_history)
        if not prior_history:
            return None
        if not cls._is_context_dependent_follow_up(question):
            return None

        for turn in reversed(prior_history):
            if turn.get("role") != "user":
                continue
            content = normalize_whitespace(turn.get("content") or "")
            if len(content) >= 20:
                return content
        return None

    @classmethod
    def _rewrite_follow_up_question(
        cls,
        *,
        question: str,
        follow_up_context: str | None,
        chat_history: list[dict[str, str]],
    ) -> str:
        cleaned = normalize_whitespace(question)
        if not cleaned:
            return ""
        if not follow_up_context or not cls._is_context_dependent_follow_up(cleaned):
            return cleaned
        if cls._is_style_only_follow_up(cleaned):
            rewritten = normalize_whitespace(follow_up_context)
        elif cleaned.lower().startswith(("which one", "which authority", "which case", "closest")):
            rewritten = normalize_whitespace(f"{follow_up_context}. Clarify which retrieved authority is closest on facts.")
        elif cleaned.lower().startswith(("explain this", "explain it", "summarize this", "tell me more")):
            rewritten = normalize_whitespace(f"{follow_up_context}. {cleaned}")
        else:
            rewritten = normalize_whitespace(f"{follow_up_context}. Follow-up: {cleaned}")
        return rewritten

    @classmethod
    def _detect_topic_drift(
        cls,
        *,
        question: str,
        follow_up_context: str | None,
        chat_history: list[dict[str, str]],
    ) -> bool:
        cleaned = normalize_whitespace(question)
        if not cleaned or not chat_history:
            return False
        if cls._is_context_dependent_follow_up(cleaned):
            return False
        prior_question = cls._derive_follow_up_context(question, chat_history)
        if not prior_question:
            return False
        current_terms = {term for term in search_terms(cleaned) if len(term) >= 4}
        prior_terms = {term for term in search_terms(prior_question) if len(term) >= 4}
        overlap = len(current_terms & prior_terms)
        if overlap >= 2:
            return False
        current_profile = infer_query_domain(cleaned)
        prior_profile = infer_query_domain(follow_up_context or prior_question)
        current_domain = normalize_whitespace(current_profile.get("domain") or "").lower()
        prior_domain = normalize_whitespace(prior_profile.get("domain") or "").lower()
        if current_domain and prior_domain and current_domain != prior_domain:
            return True
        return overlap == 0 and bool(prior_terms)

    @staticmethod
    def _normalize_similarity_signature(text: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", " ", normalize_whitespace(text).lower())
        return " ".join(normalized.split()[:40]).strip()

    @classmethod
    def _postfilter_similar_cases(
        cls,
        *,
        similar_cases: list[dict[str, Any]],
        query: str,
        query_profile: dict[str, Any] | None,
        metadata_filters: dict[str, str] | None,
        top_k: int,
    ) -> list[dict[str, Any]]:
        if not similar_cases:
            return []
        profile = query_profile or {}
        exact_terms = [normalize_whitespace(term).lower() for term in (profile.get("exact_terms") or []) if normalize_whitespace(term)]
        remedy_terms = [normalize_whitespace(term).lower() for term in (profile.get("remedy_terms") or []) if normalize_whitespace(term)]
        forum_filter = normalize_whitespace((metadata_filters or {}).get("forum") or "").lower()
        seen_signatures: set[str] = set()
        ranked: list[dict[str, Any]] = []
        for item in similar_cases:
            working = dict(item)
            haystack = " ".join(
                normalize_whitespace(part).lower()
                for part in [
                    working.get("title") or "",
                    working.get("court") or "",
                    working.get("case_type") or "",
                    working.get("summary") or "",
                    working.get("excerpt") or "",
                    working.get("proposition") or "",
                    " ".join(str(value).replace("_", " ") for value in (working.get("issue_subtypes") or [])),
                ]
                if normalize_whitespace(part)
            )
            score = float(working.get("similarity") or working.get("base_similarity") or 0.0)
            warnings: list[str] = []
            if any(marker in haystack for marker in ("uploaded on", "downloaded on", "page ", "page 1 of", "page no")):
                score -= 0.035
                warnings.append("boilerplate-heavy")
            if str(working.get("case_id") or "").find("_1800_") >= 0 or str(working.get("date") or "").startswith("1800"):
                score -= 0.08
                warnings.append("suspicious-year")
            if exact_terms and any(term in haystack for term in exact_terms):
                score += 0.05
            if remedy_terms and not any(term in haystack for term in remedy_terms):
                score -= 0.035
            if forum_filter and forum_filter not in normalize_whitespace(working.get("court") or "").lower():
                score -= 0.03
            fit_band = str(working.get("fit_band") or "").lower()
            if fit_band == "high":
                score += 0.03
            elif fit_band == "low":
                score -= 0.08
            support_type = str(working.get("support_type") or "").lower()
            if support_type == "direct":
                score += 0.02
            elif support_type == "analogical":
                score -= 0.02
            signature = cls._normalize_similarity_signature(working.get("excerpt") or working.get("summary") or working.get("title") or "")
            if signature and signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            working["similarity"] = round(min(max(score, 0.0), 1.0), 4)
            working["retrieval_confidence"] = cls._similarity_band(
                score=score,
                fit_band=fit_band,
                support_type=support_type,
            )
            if warnings:
                existing_note = normalize_whitespace(working.get("retrieval_note") or "")
                warning_text = "Warnings: " + ", ".join(warnings)
                working["retrieval_note"] = f"{existing_note} | {warning_text}" if existing_note else warning_text
            ranked.append(working)
        ranked.sort(
            key=lambda item: (
                1 if str(item.get("retrieval_confidence") or "") == "high" else 0 if str(item.get("retrieval_confidence") or "") == "moderate" else -1,
                float(item.get("similarity") or 0.0),
            ),
            reverse=True,
        )
        return ranked[: max(top_k, 1)]

    @staticmethod
    def _similarity_band(*, score: float, fit_band: str, support_type: str) -> str:
        if score >= 0.62 and fit_band == "high" and support_type != "analogical":
            return "high"
        if score >= 0.34 and fit_band in {"high", "moderate"}:
            return "moderate"
        return "low"

    @staticmethod
    def _is_context_dependent_follow_up(question: str) -> bool:
        cleaned = normalize_whitespace(question).lower()
        if not cleaned:
            return False
        word_count = len(re.findall(r"\b\w+\b", cleaned))
        ambiguous_prefixes = (
            "explain",
            "what if",
            "why",
            "where can",
            "where do",
            "show",
            "find",
            "what are",
            "what facts",
            "what evidence",
            "how is",
            "how are",
            "compare",
            "tell me more",
            "summarize",
            "rewrite",
            "simplify",
            "answer like",
            "put this",
        )
        pronoun_markers = (" it ", " this ", " that ", " these ", " those ", " mine ")
        return (
            word_count <= 7
            or LegalAIPipeline._is_style_only_follow_up(cleaned)
            or cleaned.startswith(ambiguous_prefixes)
            or any(marker in f" {cleaned} " for marker in pronoun_markers)
        )

    @staticmethod
    def _is_style_only_follow_up(question: str) -> bool:
        lowered = normalize_whitespace(question).lower()
        return any(
            marker in lowered
            for marker in (
                "simple language",
                "plain english",
                "plain language",
                "simplify",
                "rewrite",
                "rephrase",
                "short answer",
                "brief answer",
                "answer like a lawyer",
                "research note",
                "in detail",
                "detailed explanation",
            )
        )

    @staticmethod
    def _normalize_case_reference(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", (value or "").lower())

    @classmethod
    def _extract_recent_case_ids(cls, chat_history: list[dict[str, str]]) -> list[str]:
        case_ids: list[str] = []
        pattern = re.compile(r"\b[A-Za-z]+(?:_[A-Za-z0-9]+){2,}\b")
        for turn in reversed(chat_history or []):
            content = turn.get("content") or ""
            for match in pattern.findall(content):
                if match not in case_ids:
                    case_ids.append(match)
        return case_ids

    @classmethod
    def _resolve_referenced_case_ids(
        cls,
        question: str,
        chat_history: list[dict[str, str]],
    ) -> list[str]:
        normalized_question = cls._normalize_case_reference(question)
        if not normalized_question:
            return []

        matches: list[str] = []
        explicit_pattern = re.compile(r"\b[A-Za-z]+(?:_[A-Za-z0-9]+){2,}\b")
        for match in explicit_pattern.findall(question):
            if match not in matches:
                matches.append(match)

        for case_id in cls._extract_recent_case_ids(chat_history):
            normalized_case_id = cls._normalize_case_reference(case_id)
            if normalized_case_id and normalized_case_id in normalized_question and case_id not in matches:
                matches.append(case_id)
        return matches[:5]

    @classmethod
    def _should_scope_follow_up(
        cls,
        question: str,
        chat_history: list[dict[str, str]],
    ) -> bool:
        cleaned = normalize_whitespace(question).lower()
        if not cleaned:
            return False
        if cls._resolve_referenced_case_ids(question, chat_history):
            return True
        if not cls._extract_recent_case_ids(chat_history):
            return False
        return cls._is_context_dependent_follow_up(question)

    @staticmethod
    def _merge_similar_cases(case_lists: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for case_list in case_lists:
            for item in case_list:
                case_id = item.get("case_id")
                if not case_id:
                    continue
                current = merged.get(case_id)
                if current is None or float(item.get("similarity") or 0.0) > float(current.get("similarity") or 0.0):
                    merged[case_id] = dict(item)
        return sorted(merged.values(), key=lambda item: float(item.get("similarity") or 0.0), reverse=True)[:5]

    @staticmethod
    def _assess_qa_evidence(
        *,
        question: str,
        similar_cases: list[dict[str, Any]],
        case_type_hint: str | None,
        referenced_case_ids: list[str],
        source_mode: str,
        document_context: dict[str, Any],
        law_context: dict[str, Any],
        session_id: str | None,
        question_profile: dict[str, Any] | None = None,
        evidence_pack: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        document_used = bool(document_context.get("used"))
        law_used = bool(law_context.get("used"))
        law_match_type = str(law_context.get("best_match_type") or "none")
        law_retrieval_confidence = str(law_context.get("retrieval_confidence") or "low")
        evidence_pack = evidence_pack or {}
        law_validation = LegalAIPipeline._validate_reference_law_support(
            question=question,
            question_profile=question_profile,
            law_context=law_context,
        )
        assessment = {
            "status": "ok",
            "reason": None,
            "retrieval_confidence": "moderate",
            "evidence_strength": "supported",
            "answer_confidence": "moderate",
            "law_used": law_used,
            "law_validation_ok": law_validation.get("matched"),
        }
        if source_mode == "document_only":
            if not session_id:
                assessment.update(
                    {
                        "status": "poor",
                        "reason": "Uploaded document only mode needs an active uploaded document session.",
                        "retrieval_confidence": "low",
                        "evidence_strength": "insufficient",
                        "answer_confidence": "low",
                    }
                )
                return assessment
            if not document_used:
                assessment.update(
                    {
                        "status": "poor",
                        "reason": "No uploaded document excerpts were available for this question.",
                        "retrieval_confidence": "low",
                        "evidence_strength": "insufficient",
                        "answer_confidence": "low",
                    }
                )
                return assessment
            assessment.update(
                {
                    "status": "ok",
                    "retrieval_confidence": "high",
                    "evidence_strength": "supported",
                    "answer_confidence": "high",
                    "law_used": False,
                }
            )
            return assessment

        task = str((question_profile or {}).get("task") or "")
        if task in LAW_FIRST_TASKS:
            if not law_used:
                target = (law_validation.get("target") or {}).get("label") or "the relevant provision"
                assessment.update(
                    {
                        "status": "poor",
                        "reason": f"No official law material was retrieved for {target}.",
                        "retrieval_confidence": "low",
                        "evidence_strength": "insufficient",
                        "answer_confidence": "low",
                    }
                )
                return assessment
            if law_validation.get("required") and not law_validation.get("matched"):
                target = (law_validation.get("target") or {}).get("label") or "the governing provision"
                assessment.update(
                    {
                        "status": "poor",
                        "reason": f"The retrieved official law materials did not reliably match {target}.",
                        "retrieval_confidence": "low",
                        "evidence_strength": "insufficient",
                        "answer_confidence": "low",
                    }
                )
                return assessment

        if not similar_cases:
            if law_used and not document_used:
                assessment.update(
                    {
                        "status": "ok" if law_match_type == "exact" or task in LAW_FIRST_TASKS else "weak",
                        "reason": None if law_match_type == "exact" or task in LAW_FIRST_TASKS else "The answer is grounded mainly in official law materials rather than similar judgments.",
                        "retrieval_confidence": "high" if law_retrieval_confidence == "high" else "moderate",
                        "evidence_strength": "supported" if law_match_type in {"exact", "related"} or task in LAW_FIRST_TASKS else "mixed",
                        "answer_confidence": "high" if law_match_type == "exact" or task in LAW_FIRST_TASKS else "moderate",
                    }
                )
                return assessment
            if law_used and document_used and source_mode == "document_plus_case":
                assessment.update(
                    {
                        "status": "weak",
                        "reason": "No matching case-law authorities were retrieved, so the answer relies on the uploaded document plus official law materials.",
                        "retrieval_confidence": "moderate",
                        "evidence_strength": "mixed" if law_match_type == "semantic" else "supported",
                        "answer_confidence": "moderate",
                    }
                )
                return assessment
            if source_mode == "document_plus_case" and document_used:
                assessment.update(
                    {
                        "status": "weak",
                        "reason": "No matching case-law authorities were retrieved, so the answer relies mainly on the uploaded document.",
                        "retrieval_confidence": "low",
                        "evidence_strength": "mixed",
                        "answer_confidence": "low",
                    }
                )
                return assessment
            assessment.update(
                {
                    "status": "poor",
                    "reason": "No relevant authorities were retrieved for this question.",
                    "retrieval_confidence": "low",
                    "evidence_strength": "insufficient",
                    "answer_confidence": "low",
                }
            )
            return assessment

        if (
            question_profile
            and not question_profile.get("supported_case_law_domain")
            and not question_profile.get("direct_case_lookup")
            and not law_used
        ):
            domain_name = question_profile.get("domain") or "this"
            assessment.update(
                {
                    "status": "poor",
                    "reason": f"The phase-1 case-law path is not reliable enough yet for {domain_name} questions without a statute or domain-specific store.",
                    "retrieval_confidence": "low",
                    "evidence_strength": "insufficient",
                    "answer_confidence": "low",
                }
            )
            return assessment

        if referenced_case_ids:
            retrieved_case_ids = [item.get("case_id") for item in similar_cases[:3]]
            if not any(case_id in retrieved_case_ids for case_id in referenced_case_ids[:2]):
                assessment.update(
                    {
                        "status": "poor",
                        "reason": f"The requested case id {referenced_case_ids[0]} was not reliably retrieved.",
                        "retrieval_confidence": "low",
                        "evidence_strength": "insufficient",
                        "answer_confidence": "low",
                    }
                )
                return assessment

        top_similarity = float(similar_cases[0].get("similarity") or 0.0)
        top_retrieval_confidence = str(similar_cases[0].get("retrieval_confidence") or "")
        if top_similarity < 0.12:
            if int(evidence_pack.get("analogical_support_count") or 0) >= 1:
                assessment.update(
                    {
                        "status": "weak",
                        "reason": "The retrieved authorities are only loosely matched, so the answer can give direction but not a firm rule.",
                        "retrieval_confidence": "low",
                        "evidence_strength": "mixed",
                        "answer_confidence": "low",
                    }
                )
                return assessment
            assessment.update(
                {
                    "status": "poor",
                    "reason": "The retrieved authorities are too weakly matched to answer reliably.",
                    "retrieval_confidence": "low",
                    "evidence_strength": "insufficient",
                    "answer_confidence": "low",
                }
            )
            return assessment

        query_profile = question_profile or infer_query_domain(question, case_type_hint=case_type_hint)
        query_domain = query_profile.get("domain")
        query_confidence = float(query_profile.get("domain_confidence") or query_profile.get("confidence") or 0.0)
        if not query_domain or query_confidence < 0.65:
            assessment["retrieval_confidence"] = "high" if top_retrieval_confidence == "high" else "moderate"
            assessment["evidence_strength"] = "supported" if int(evidence_pack.get("direct_support_count") or 0) >= 1 else "mixed"
            assessment["answer_confidence"] = "moderate"
            return assessment

        if query_profile.get("statute_sensitive") and not evidence_pack.get("statute_support_available"):
            assessment.update(
                {
                    "status": "weak",
                    "reason": "No provision-level statute source is available, so the answer can only rely on case-law support.",
                    "retrieval_confidence": "moderate",
                    "evidence_strength": "mixed",
                    "answer_confidence": "low",
                }
            )
            return assessment
        if query_profile.get("statute_sensitive") and evidence_pack.get("statute_support_available"):
            assessment["retrieval_confidence"] = "high" if law_retrieval_confidence == "high" else "moderate"
            if not similar_cases:
                assessment["evidence_strength"] = "supported"
                assessment["answer_confidence"] = "high" if law_match_type == "exact" else "moderate"
                return assessment

        matched = 0
        mismatched = 0
        for item in similar_cases[:3]:
            candidate_domain, candidate_confidence = infer_candidate_domain(
                item.get("case_id") or "",
                case_type=item.get("case_type"),
                title=item.get("title"),
                court=item.get("court"),
                text=item.get("excerpt") or item.get("summary"),
            )
            if not candidate_domain or candidate_confidence < 0.6:
                continue
            if candidate_domain == query_domain:
                matched += 1
            else:
                mismatched += 1

        direct_support_count = int(evidence_pack.get("direct_support_count") or 0)
        analogical_support_count = int(evidence_pack.get("analogical_support_count") or 0)
        if query_confidence >= 0.8 and matched == 0 and mismatched >= 2 and analogical_support_count == 0:
            assessment.update(
                {
                    "status": "poor",
                    "reason": f"The retrieved authorities look off-domain for a {query_domain} question.",
                    "retrieval_confidence": "low",
                    "evidence_strength": "insufficient",
                    "answer_confidence": "low",
                }
            )
            return assessment
        if query_confidence >= 0.68 and matched == 0 and mismatched >= 1 and direct_support_count == 0:
            assessment.update(
                {
                    "status": "weak",
                    "reason": f"The retrieved authorities do not line up cleanly with the expected {query_domain} domain.",
                    "retrieval_confidence": "low",
                    "evidence_strength": "mixed",
                    "answer_confidence": "low",
                }
            )
            return assessment

        moderate_count = len([item for item in similar_cases[:3] if str(item.get("fit_band") or "") == "moderate"])
        low_count = len([item for item in similar_cases[:3] if str(item.get("fit_band") or "") == "low"])
        retrieval_confidence = "high" if matched >= 2 and low_count == 0 and top_retrieval_confidence != "low" else "moderate"
        evidence_strength = "supported"
        if low_count >= 1 or analogical_support_count >= max(direct_support_count, 1):
            evidence_strength = "mixed"
        if direct_support_count == 0 and analogical_support_count == 0:
            evidence_strength = "insufficient"
        if law_used and evidence_strength == "insufficient":
            evidence_strength = "supported" if law_match_type in {"exact", "related"} else "mixed"
        answer_confidence = "high"
        if retrieval_confidence != "high" or evidence_strength != "supported" or moderate_count >= 2:
            answer_confidence = "moderate"
        if evidence_strength == "insufficient":
            answer_confidence = "low"
        if law_used and law_retrieval_confidence == "high" and answer_confidence == "low":
            answer_confidence = "moderate"
        assessment.update(
            {
                "status": "weak" if evidence_strength == "mixed" else "ok",
                "retrieval_confidence": retrieval_confidence,
                "evidence_strength": evidence_strength,
                "answer_confidence": answer_confidence,
            }
        )
        return assessment

    @staticmethod
    def _build_cautious_qa_answer(
        *,
        question: str,
        reason: str | None,
        referenced_case_ids: list[str],
        similar_cases: list[dict[str, Any]],
        source_mode: str,
        question_profile: dict[str, Any] | None = None,
    ) -> str:
        response_plan = str((question_profile or {}).get("response_plan") or "direct_guidance")
        response_length = str((question_profile or {}).get("response_length") or "medium")

        def _top_authority_lines() -> str:
            if not similar_cases:
                return "- None."
            lines = [
                f"- {item.get('case_id')}: "
                f"{normalize_whitespace(item.get('proposition') or item.get('excerpt') or item.get('summary') or 'Closest available support.')}"
                for item in similar_cases[:2]
                if item.get("case_id")
            ]
            return "\n".join(lines) if lines else "- None."

        def _top_tail() -> str:
            if not similar_cases:
                return ""
            top_ids = ", ".join(item.get("case_id") or "" for item in similar_cases[:2] if item.get("case_id"))
            return f" The closest retrieved authorities were {top_ids}." if top_ids else ""

        def _support_direction_line() -> str:
            if not similar_cases:
                return "- No usable authorities were retrieved."
            lead = similar_cases[0]
            proposition = normalize_whitespace(
                lead.get("proposition")
                or lead.get("excerpt")
                or lead.get("summary")
                or "the closest available case-law support"
            )
            return (
                f"- The closest retrieved authority `{lead.get('case_id')}` points toward {shorten_text(proposition, 180)}."
            )

        def _practical_guardrail() -> str:
            next_step = (
                "- Narrow the question to one issue, or upload the notice/order so the next step can be anchored in stronger authority."
            )
            if response_length == "short":
                return (
                    "#### Bottom line\n"
                    "- I cannot safely give a firm yes-or-no answer from the current authorities.\n\n"
                    "#### What I can say now\n"
                    f"- {reason or 'The retrieved case-law support is too weak or mixed.'}{_top_tail()}\n"
                    f"{_support_direction_line()}\n\n"
                    "#### Best next step\n"
                    f"{next_step}"
                )
            return (
                "#### Bottom line\n"
                "- I cannot safely give a firm practical recommendation from the current authorities alone.\n\n"
                "#### Why I am stopping short\n"
                f"- {reason or 'The retrieved case-law support is too weak or mixed.'}{_top_tail()}\n\n"
                "#### What the closest cases still suggest\n"
                f"{_top_authority_lines()}\n\n"
                "#### Best next step\n"
                f"{next_step}"
            )

        if source_mode == "document_only":
            return (
                "#### Bottom line\n"
                "- I could not answer this reliably from the uploaded document alone.\n\n"
                "#### Why\n"
                f"- {reason or 'The uploaded document excerpts were too limited for this question.'}\n\n"
                "#### Best next step\n"
                "- Upload a fuller document or switch to document + case corpus mode."
            )
        if referenced_case_ids:
            return (
                "#### Answer\n"
                f"- I could not answer this reliably from the retrieved material for {referenced_case_ids[0]}.\n\n"
                "#### What is missing\n"
                f"- {reason or 'The available passages are too weak or off-target.'}\n\n"
                "#### Best next step\n"
                "- Ask separately for the facts, outcome, or reasoning of that case, or open the full judgment."
            )
        if "phase-1 case-law path" in (reason or ""):
            return (
                "#### Bottom line\n"
                "- I should not answer this confidently from the current phase-1 case-law setup.\n\n"
                "#### Why\n"
                f"- {reason}\n"
                "- This corpus-aware mode is strongest for consumer, education, tax, motor-accident, service, and RTI-style information disputes.\n\n"
                "#### Best next step\n"
                "- For other domains, upload the exact judgment, cite a specific case, or narrow the issue further."
            )
        if response_plan == "practical_steps":
            return _practical_guardrail()
        if response_plan in {"case_brief", "case_brief_simple"}:
            return (
                "#### Answer\n"
                "- I could not reconstruct this cleanly from the currently retrieved passages.\n\n"
                "#### What is missing\n"
                f"- {reason or 'The available passages are too limited or off-target for a reliable case brief.'}\n\n"
                "#### Closest authorities\n"
                f"{_top_authority_lines()}\n\n"
                "#### Best next step\n"
                "- Ask separately for the facts, outcome, or reasoning, or open the full judgment text."
            )
        if response_plan == "research_note":
            return (
                "#### Issue\n"
                f"- {shorten_text(question, 140)}\n\n"
                "#### Current position\n"
                f"- {reason or 'The retrieved authorities do not support a reliable research-note conclusion yet.'}{_top_tail()}\n"
                f"{_support_direction_line()}\n\n"
                "#### What would improve the note\n"
                "- Narrow the issue, add the governing forum or provision, or cite a specific authority to anchor the note."
            )
        if response_plan == "outcome_pattern":
            return (
                "#### Current pattern\n"
                f"- {reason or 'The retrieved authorities are not strong enough yet for a reliable accepted-versus-rejected pattern statement.'}{_top_tail()}\n"
                f"{_support_direction_line()}\n\n"
                "#### Best next step\n"
                "- Narrow the pattern to one domain and one issue so the accepted and rejected authorities can be compared cleanly."
            )
        if response_plan == "comparative_analysis":
            return (
                "#### Answer\n"
                "- I cannot safely compare outcomes from the current authorities yet.\n\n"
                "#### Why\n"
                f"- {reason or 'The retrieved authorities are too weak or too mixed for a reliable comparison.'}{_top_tail()}\n"
                f"{_support_direction_line()}\n\n"
                "#### Best next step\n"
                "- Narrow the comparison to one issue, one forum, or one remedy question."
            )
        return (
            "#### Answer\n"
            "- I cannot make a firm statement from the current authorities alone.\n\n"
            "#### Why\n"
            f"- {reason or 'The evidence bundle is too weak or mixed for a confident answer.'}{_top_tail()}\n\n"
            "#### What the closest authorities still suggest\n"
            f"{_top_authority_lines()}\n\n"
            "#### Best next step\n"
            "- Narrow the issue, add more facts, or specify the case type."
        )

    def _prepare_retrieval_query(
        self,
        *,
        text: str,
        detected_language: str,
        raw_question: str | None = None,
        question_profile: dict[str, Any] | None = None,
    ) -> str:
        del detected_language
        profile = question_profile or {}
        raw = normalize_whitespace(raw_question or text)
        retrieval_terms = [
            normalize_whitespace(term)
            for term in (profile.get("retrieval_terms") or [])
            if normalize_whitespace(term)
        ]
        exact_terms = [
            normalize_whitespace(term)
            for term in (profile.get("exact_terms") or [])
            if normalize_whitespace(term)
        ]
        task = str(profile.get("task") or "general_research")
        direct_case_lookup = bool(profile.get("direct_case_lookup"))
        if direct_case_lookup or task in {"case_explanation", "document_fact", "document_reasoning"}:
            if not retrieval_terms and not exact_terms:
                return raw
            return normalize_whitespace(" ".join([raw] + exact_terms[:4] + retrieval_terms[:8]))

        focus_terms = self._build_retrieval_focus_terms(
            raw_question=raw,
            retrieval_terms=retrieval_terms,
            question_profile=profile,
        )
        if not focus_terms:
            return raw
        return normalize_whitespace(" ".join(focus_terms))

    @staticmethod
    def _build_retrieval_focus_terms(
        *,
        raw_question: str,
        retrieval_terms: list[str],
        question_profile: dict[str, Any],
    ) -> list[str]:
        terms: list[str] = []
        seen: set[str] = set()
        low_value_tokens = {
            "facts",
            "based",
            "deal",
            "deals",
            "disputes",
            "dispute",
            "involving",
            "issue",
            "issues",
            "matter",
            "matters",
            "over",
            "using",
            "about",
            "what",
            "how",
            "why",
            "usually",
            "compare",
            "versus",
            "within",
        }

        def add(term: str) -> None:
            normalized = normalize_whitespace(term).lower()
            if not normalized or normalized in seen or normalized in low_value_tokens:
                return
            seen.add(normalized)
            terms.append(normalized)

        domain = normalize_whitespace(question_profile.get("domain")).replace("_", " ")
        if domain:
            add(domain)
        for subtype in (question_profile.get("issue_subtypes") or [])[:4]:
            add(str(subtype).replace("_", " "))

        for term in (question_profile.get("exact_terms") or [])[:6]:
            add(str(term))

        for term in (question_profile.get("remedy_terms") or [])[:4]:
            add(str(term))

        for term in retrieval_terms[:12]:
            add(term)

        for token in search_terms(raw_question):
            if len(token) < 3:
                continue
            add(token)
            if len(terms) >= 16:
                break

        return terms[:20]

    @staticmethod
    def _should_run_domain_rescue(
        *,
        candidate_cases: list[dict[str, Any]],
        query_profile: dict[str, Any] | None,
        top_k: int,
    ) -> bool:
        profile = query_profile or {}
        domain = normalize_whitespace(profile.get("domain")).lower()
        confidence = float(profile.get("domain_confidence") or 0.0)
        if not domain or confidence < 0.72 or not candidate_cases:
            return False
        matched = 0
        for item in candidate_cases[: max(top_k + 1, 4)]:
            candidate_domain, candidate_confidence = infer_candidate_domain(
                item.get("case_id") or "",
                case_type=item.get("case_type"),
                title=item.get("title"),
                court=item.get("court"),
                text=item.get("excerpt") or item.get("summary"),
            )
            if candidate_domain == domain and candidate_confidence >= 0.6:
                matched += 1
        top_similarity = float(candidate_cases[0].get("similarity") or candidate_cases[0].get("base_similarity") or 0.0)
        return matched == 0 or top_similarity < 0.16

    @classmethod
    def _build_domain_rescue_query(
        cls,
        *,
        retrieval_query: str,
        query_profile: dict[str, Any] | None,
    ) -> str:
        profile = query_profile or {}
        focus_terms = cls._build_retrieval_focus_terms(
            raw_question=retrieval_query,
            retrieval_terms=list(profile.get("retrieval_terms") or []),
            question_profile=profile,
        )
        domain = normalize_whitespace(profile.get("domain")).replace("_", " ")
        if domain and domain not in focus_terms:
            focus_terms.insert(0, domain)
        return normalize_whitespace(" ".join(focus_terms[:12]) or retrieval_query)

    @staticmethod
    def _merge_ranked_case_lists(
        primary: list[dict[str, Any]],
        secondary: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for item in primary + secondary:
            case_id = item.get("case_id")
            if not case_id:
                continue
            current = merged.get(case_id)
            current_score = float(current.get("similarity") or current.get("base_similarity") or 0.0) if current else -1.0
            score = float(item.get("similarity") or item.get("base_similarity") or 0.0)
            if current is None or score > current_score:
                merged[case_id] = dict(item)
        return sorted(
            merged.values(),
            key=lambda item: float(item.get("similarity") or item.get("base_similarity") or 0.0),
            reverse=True,
        )

    def _build_case_review_query_profile(self, intake: dict[str, Any]) -> dict[str, Any]:
        profile = self.query_router.analyze(
            question=intake.get("input_summary") or "",
            chat_history=[],
            session_has_document=False,
            requested_source_mode="case_corpus_only",
            case_type_hint=intake.get("case_type"),
            forum_hint=intake.get("forum"),
            context_note=intake.get("opponent_arguments"),
        ).to_dict()
        combined = " ".join(
            part
            for part in [
                intake.get("input_summary") or "",
                intake.get("relief_sought") or "",
                intake.get("evidence_summary") or "",
                intake.get("opponent_arguments") or "",
            ]
            if part
        )
        issue_subtypes = infer_issue_subtypes(combined, domain=profile.get("domain"))
        if issue_subtypes:
            profile["issue_subtypes"] = issue_subtypes
            retrieval_terms = list(profile.get("retrieval_terms") or [])
            for subtype in issue_subtypes:
                readable = normalize_whitespace(str(subtype).replace("_", " "))
                if readable and readable not in retrieval_terms:
                    retrieval_terms.append(readable)
            profile["retrieval_terms"] = retrieval_terms[:16]
        profile["workflow"] = "triage"
        profile["task"] = "case_review"
        profile["retrieval_profile"] = "fast"
        return profile

    @staticmethod
    def _select_case_review_authorities(
        *,
        similar_cases: list[dict[str, Any]],
        predicted_name: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        if not similar_cases:
            return []
        if predicted_name == "Partially Accepted":
            return similar_cases[:top_k]
        ranked = sorted(
            similar_cases,
            key=lambda item: (
                1 if item.get("fit_band") == "high" else 0 if item.get("fit_band") == "moderate" else -1,
                float(item.get("similarity") or 0.0),
            ),
            reverse=True,
        )
        strong_support = [
            item
            for item in ranked
            if item.get("label_name") == predicted_name and item.get("fit_band") != "low"
        ]
        cautionary = [
            item
            for item in ranked
            if (
                item.get("label_name") in {"Accepted", "Rejected"}
                and item.get("label_name") != predicted_name
            )
            or item.get("fit_band") == "low"
        ]
        broader_support = [
            item
            for item in ranked
            if item not in strong_support and item not in cautionary
        ]

        selected: list[dict[str, Any]] = []
        for pool, limit in (
            (strong_support, max(top_k - 1, 1)),
            (cautionary, 1),
            (broader_support, top_k),
        ):
            count = 0
            for item in pool:
                case_id = item.get("case_id")
                if not case_id or any(existing.get("case_id") == case_id for existing in selected):
                    continue
                selected.append(item)
                count += 1
                if len(selected) >= top_k or count >= limit:
                    break
            if len(selected) >= top_k:
                break

        if len(selected) < top_k:
            for item in ranked:
                case_id = item.get("case_id")
                if not case_id or any(existing.get("case_id") == case_id for existing in selected):
                    continue
                selected.append(item)
                if len(selected) >= top_k:
                    break
        return selected[:top_k]

    @staticmethod
    def _assess_review_evidence(
        *,
        prediction: dict[str, Any],
        similar_cases: list[dict[str, Any]],
        authority_map: dict[str, Any],
    ) -> dict[str, str]:
        if not similar_cases:
            return {
                "retrieval_confidence": "low",
                "evidence_strength": "insufficient",
                "answer_confidence": "low",
            }
        top_case = similar_cases[0]
        top_similarity = float(top_case.get("similarity") or 0.0)
        top_fit = str(top_case.get("fit_band") or "")
        conflicting = list(authority_map.get("conflicting") or [])
        supporting = list(authority_map.get("supporting") or [])

        retrieval_confidence = "moderate"
        if top_fit == "high" and top_similarity >= 0.82:
            retrieval_confidence = "high"
        elif top_fit == "low" or top_similarity < 0.56:
            retrieval_confidence = "low"

        evidence_strength = "supported"
        if conflicting:
            evidence_strength = "mixed"
        if not supporting or (top_fit == "low" and not conflicting):
            evidence_strength = "insufficient"

        answer_confidence = "moderate"
        if (
            retrieval_confidence == "high"
            and evidence_strength == "supported"
            and float(prediction.get("confidence_score") or 0.0) >= 75.0
        ):
            answer_confidence = "high"
        elif evidence_strength == "insufficient" or retrieval_confidence == "low":
            answer_confidence = "low"

        return {
            "retrieval_confidence": retrieval_confidence,
            "evidence_strength": evidence_strength,
            "answer_confidence": answer_confidence,
        }

    def _retrieve_case_corpus_authorities(
        self,
        *,
        retrieval_query: str,
        top_k: int,
        metadata_filters: dict[str, str] | None,
        retrieval_profile: str,
        preferred_case_ids: list[str] | None = None,
        query_profile: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        self._ensure_retrieval_ready("case-law retrieval")

        scoped_case_ids = list(dict.fromkeys(preferred_case_ids or []))
        if scoped_case_ids:
            similar_cases = self.qa_retriever.search(
                retrieval_query,
                top_k=top_k,
                case_ids=scoped_case_ids,
                metadata_filters=metadata_filters,
                candidate_case_scores=None,
                query_profile=query_profile,
            )
            return similar_cases, scoped_case_ids[:5]

        if retrieval_profile == "fast":
            shortlist_size = max(top_k + 2, int(self.settings.qa_fast_case_shortlist_top_n))
        else:
            shortlist_size = max(top_k + 4, int(self.settings.qa_deep_case_shortlist_top_n))
        if str((query_profile or {}).get("task") or "") in {"comparative_reasoning", "general_research"}:
            shortlist_size = max(shortlist_size, top_k + 6)
        if str((query_profile or {}).get("response_plan") or "") in {"research_note", "outcome_pattern"}:
            shortlist_size = max(shortlist_size, top_k + 6)
        candidate_cases = self.retriever.search(
            retrieval_query,
            top_k=shortlist_size,
            metadata_filters=metadata_filters,
            refine_chunks=(retrieval_profile == "deep"),
            query_profile=query_profile,
        )
        if self._should_run_domain_rescue(
            candidate_cases=candidate_cases,
            query_profile=query_profile,
            top_k=top_k,
        ):
            rescue_query = self._build_domain_rescue_query(
                retrieval_query=retrieval_query,
                query_profile=query_profile,
            )
            rescue_cases = self.retriever.search(
                rescue_query,
                top_k=max(shortlist_size, top_k + 4),
                metadata_filters=metadata_filters,
                refine_chunks=(retrieval_profile == "deep"),
                query_profile=query_profile,
            )
            candidate_cases = self._merge_ranked_case_lists(candidate_cases, rescue_cases)[: max(shortlist_size, top_k + 4)]
        candidate_cases = self._postfilter_similar_cases(
            similar_cases=candidate_cases,
            query=retrieval_query,
            query_profile=query_profile,
            metadata_filters=metadata_filters,
            top_k=max(shortlist_size, top_k + 4),
        )
        candidate_case_ids = [
            item.get("case_id")
            for item in candidate_cases
            if item.get("case_id")
        ]
        if not candidate_case_ids:
            return [], []
        candidate_case_scores = {
            item["case_id"]: float(
                item.get("similarity")
                or item.get("base_similarity")
                or item.get("evidence_similarity")
                or 0.0
            )
            for item in candidate_cases
            if item.get("case_id")
        }
        if retrieval_profile == "fast":
            candidate_case_ids = candidate_case_ids[: min(len(candidate_case_ids), max(top_k + 1, 4))]
        else:
            candidate_case_ids = candidate_case_ids[: min(len(candidate_case_ids), max(top_k + 5, 8))]
        similar_cases = self.qa_retriever.search(
            retrieval_query,
            top_k=top_k,
            case_ids=candidate_case_ids,
            metadata_filters=metadata_filters,
            candidate_case_scores=candidate_case_scores,
            query_profile=query_profile,
        )
        return similar_cases, candidate_case_ids[:5]

    def _classify_question_route(
        self,
        *,
        question: str,
        chat_history: list[dict[str, str]],
        session_id: str | None,
        requested_source_mode: str,
    ) -> str:
        if requested_source_mode == "case_corpus_only":
            return "legal_compare"
        if not self.session_documents.has_document(session_id):
            return "legal_compare"

        lowered = normalize_whitespace(question).lower()
        legal_compare_markers = (
            "similar cases",
            "similar judgments",
            "other judgments",
            "other cases",
            "generally",
            "usually",
            "across indian judgments",
            "compare accepted",
            "compare rejected",
            "today with higher inflation",
            "if a similar case occurs today",
            "what do courts usually",
        )
        document_fact_markers = (
            "who were the parties",
            "parties involved",
            "which court",
            "bench",
            "decided on",
            "case number",
            "what were the key facts",
            "what happened",
            "primary legal issue",
            "how many prosthetic limbs",
            "number of prosthetic limbs",
            "final total compensation",
            "list all the components",
            "what factors did the court consider",
        )
        document_reasoning_markers = (
            "this case",
            "this judgment",
            "uploaded judgment",
            "uploaded document",
            "summarize the judgment",
            "summarize this judgment",
            "summarize the uploaded",
            "the appellant",
            "the claimant",
            "the court",
            "why did the court",
            "how did the court",
            "explain the principle",
            "restitutio in integrum",
            "government notification rates",
            "fixed universal guideline",
        )
        if any(marker in lowered for marker in legal_compare_markers):
            return "legal_compare"
        if any(marker in lowered for marker in document_fact_markers):
            return "document_fact"
        if any(marker in lowered for marker in document_reasoning_markers):
            return "document_reasoning"
        if self._is_context_dependent_follow_up(question) and chat_history:
            return "document_reasoning"
        return "legal_compare"

    def _resolve_effective_source_mode(
        self,
        *,
        requested_source_mode: str,
        question_route: str,
        session_id: str | None,
    ) -> str:
        if requested_source_mode in {"document_only", "case_corpus_only"}:
            return requested_source_mode
        if question_route in {"document_fact", "document_reasoning"} and self.session_documents.has_document(session_id):
            return "document_plus_case"
        return requested_source_mode

    def _build_source_plan(
        self,
        *,
        effective_source_mode: str,
        question_profile: dict[str, Any],
        session_id: str | None,
    ) -> dict[str, bool]:
        task = str(question_profile.get("task") or "")
        session_has_document = self.session_documents.has_document(session_id)
        if effective_source_mode == "document_only":
            return {
                "use_case_corpus": False,
                "use_uploaded_document": True,
                "use_reference_law": False,
            }
        if task in {"document_fact", "document_reasoning"} and session_has_document:
            return {
                "use_case_corpus": False,
                "use_uploaded_document": True,
                "use_reference_law": bool(question_profile.get("statute_sensitive")),
            }
        if task in LAW_FIRST_TASKS:
            return {
                "use_case_corpus": False,
                "use_uploaded_document": False,
                "use_reference_law": True,
            }
        if task in HYBRID_LAW_TASKS:
            return {
                "use_case_corpus": effective_source_mode in {"document_plus_case", "case_corpus_only"},
                "use_uploaded_document": session_has_document and effective_source_mode == "document_plus_case",
                "use_reference_law": True,
            }
        return {
            "use_case_corpus": effective_source_mode in {"document_plus_case", "case_corpus_only"},
            "use_uploaded_document": session_has_document and effective_source_mode in {"document_plus_case", "document_only"},
            "use_reference_law": bool(question_profile.get("statute_sensitive")),
        }

    @staticmethod
    def _derive_response_source_mode(
        *,
        requested_source_mode: str,
        document_used: bool,
        law_used: bool,
        case_used: bool,
    ) -> str:
        if document_used and law_used and case_used:
            return "document_plus_reference_law_plus_case"
        if document_used and law_used:
            return "document_plus_reference_law"
        if law_used and case_used:
            return "reference_law_plus_case"
        if law_used:
            return "reference_law_only"
        if document_used and case_used:
            return "document_plus_case"
        if document_used:
            return "document_only"
        if case_used:
            return "case_corpus_only"
        return requested_source_mode

    @staticmethod
    def _derive_favorability(predicted_name: str, user_role: str | None) -> tuple[str, str]:
        normalized_role = normalize_whitespace(user_role).lower()
        role_alignment = "unknown"
        if any(marker in normalized_role for marker in FILING_SIDE_MARKERS):
            role_alignment = "filer"
        elif any(marker in normalized_role for marker in OPPOSING_SIDE_MARKERS):
            role_alignment = "opponent"

        if predicted_name == "Partially Accepted":
            return (
                "Mixed / partially favorable",
                "The model expects a mixed or partial disposition, so the likely benefit to your side is not one-directional.",
            )

        if role_alignment == "filer":
            if predicted_name == "Accepted":
                return (
                    "Likely favorable",
                    "You appear to be the party seeking relief, so an accepted matter is usually favorable to your side.",
                )
            return (
                "Likely unfavorable",
                "You appear to be the party seeking relief, so a rejected matter is usually unfavorable to your side.",
            )

        if role_alignment == "opponent":
            if predicted_name == "Accepted":
                return (
                    "Likely unfavorable",
                    "You appear to be responding to the matter, so an accepted filing is usually unfavorable to your side.",
                )
            return (
                "Likely favorable",
                "You appear to be responding to the matter, so a rejected filing is usually favorable to your side.",
            )

        return (
            "Role-sensitive / unclear",
            "The court-disposition prediction is available, but the favorable-to-you interpretation needs a clearer user role.",
        )

    def _build_advisories(
        self,
        *,
        prediction: dict[str, Any],
        similar_cases: list[dict[str, Any]],
        favorability_label: str,
        evidence_gaps: list[str],
        authority_map: dict[str, Any],
    ) -> list[str]:
        advisories: list[str] = []
        if prediction["confidence_band"] == "Low":
            advisories.append(
                "Prediction confidence is low. Treat this as decision support and not as a final legal conclusion."
            )
        if prediction["predicted_name"] == "Partially Accepted":
            advisories.append(
                "The partial-acceptance class is relatively rare in this dataset, so manual review is especially important."
            )
        if favorability_label == "Role-sensitive / unclear":
            advisories.append(
                "The system can predict court disposition, but your side-specific interpretation stays uncertain until the user role is clarified."
            )
        if authority_map.get("conflicting"):
            advisories.append(
                "Retrieved authorities are not one-directional. Review the conflicting cases before relying on the result."
            )
        if evidence_gaps:
            advisories.append(evidence_gaps[0])
        if not similar_cases:
            advisories.append(
                "No strong similar cases were retrieved from the current store, so explanation quality may be limited."
            )
        return advisories[:5]

    def _derive_prediction_posture(
        self,
        *,
        prediction: dict[str, Any],
        authority_map: dict[str, Any],
    ) -> tuple[str, str]:
        confidence_score = float(prediction["confidence_score"])
        if confidence_score < float(self.settings.prediction_review_threshold):
            return (
                "Review required",
                "Confidence is below the review threshold, so this should be treated as a provisional triage signal only.",
            )
        if authority_map.get("conflicting"):
            return (
                "Use with caution",
                "The prediction has some model support, but the retrieved authorities are mixed and need manual review.",
            )
        return (
            "Usable triage signal",
            "Confidence and retrieved authorities are aligned enough for first-pass triage, subject to manual legal review.",
        )

    def _build_qa_advisories(
        self,
        *,
        similar_cases: list[dict[str, Any]],
        scope: str,
        answer_source: str,
        evidence_gaps: list[str],
        authority_map: dict[str, Any],
        document_used: bool,
        source_mode: str,
        evidence_assessment: dict[str, Any] | None = None,
    ) -> list[str]:
        advisories: list[str] = []
        evidence_strength = str((evidence_assessment or {}).get("evidence_strength") or "")
        retrieval_confidence = str((evidence_assessment or {}).get("retrieval_confidence") or "")
        law_used = bool((evidence_assessment or {}).get("law_used"))
        if source_mode == "document_only":
            advisories.append(
                "This answer was restricted to the uploaded document only; no case-law authorities were consulted."
            )
        elif source_mode == "reference_law_only":
            advisories.append(
                "This answer relied on official law materials only, because that was the safest primary source for the question."
            )
        elif source_mode == "reference_law_plus_case":
            advisories.append(
                "This answer is statute-first and uses case-law only as supporting illustration."
            )
        elif source_mode == "document_plus_reference_law":
            advisories.append(
                "This answer relied on the uploaded document plus official law materials without adding case-law authorities."
            )
        elif source_mode == "document_plus_reference_law_plus_case":
            advisories.append(
                "This answer combines the uploaded document, official law materials, and supporting case-law."
            )
        elif law_used and not similar_cases:
            advisories.append(
                "This answer relied mainly on official law materials because no closely matching judgments were needed or retrieved."
            )
        elif not similar_cases and document_used and source_mode == "document_plus_case":
            advisories.append(
                "No closely matching case-law authorities were retrieved, so this answer relies mainly on the uploaded document."
            )
        elif not similar_cases:
            advisories.append(
                "No closely matching authorities were retrieved. Try a more specific legal question or add factual context."
            )
        if scope == "current_result":
            advisories.append(
                "This answer was limited to the current workspace authorities instead of the full corpus."
            )
        if retrieval_confidence == "low":
            advisories.append(
                "The retrieved authorities are only a weak factual fit for this question, so verify the result cautiously."
            )
        if authority_map.get("conflicting"):
            advisories.append(
                "The top authorities point in mixed directions, so inspect both supporting and conflicting cases."
            )
        elif evidence_strength == "mixed":
            advisories.append(
                "The current evidence is only partially aligned, so the answer should be treated as a starting point rather than a firm conclusion."
            )
        if evidence_gaps:
            advisories.append(evidence_gaps[0])
        if document_used and source_mode == "document_plus_case":
            advisories.append(
                "An uploaded document was used as additional factual context for this answer, but the legal authorities still come from the judgment corpus."
            )
        elif law_used:
            advisories.append(
                "Official law materials were used as provision-level support for this answer."
            )
        if similar_cases and evidence_strength != "insufficient":
            advisories.append(
                "The answer is grounded in the retrieved passages, but the full judgment should still be checked before use."
            )
        elif document_used and source_mode == "document_only":
            advisories.append(
                "The answer is grounded in uploaded document excerpts only, so broader legal support was not checked against the corpus."
            )
        return advisories[:5]

    def _build_research_advisories(
        self,
        *,
        similar_cases: list[dict[str, Any]],
        authority_map: dict[str, Any],
        evidence_gaps: list[str],
    ) -> list[str]:
        advisories: list[str] = []
        if authority_map.get("conflicting"):
            advisories.append(
                "This topic has both supporting and conflicting authorities; compare facts manually before drawing a conclusion."
            )
        if evidence_gaps:
            advisories.append(evidence_gaps[0])
        if not similar_cases:
            advisories.append("No strong research authorities were retrieved for this topic yet.")
        else:
            advisories.append(
                "Use the research tab to shortlist authorities, then switch to Ask on current result for grounded follow-up."
            )
        return advisories[:4]

    def _build_workspace(
        self,
        *,
        workflow: str,
        authority_map: dict[str, Any],
        issue_outline: list[str],
        evidence_gaps: list[str],
        confidence_band: str | None = None,
        intake: dict[str, Any] | None = None,
        question: str | None = None,
        topic_query: str | None = None,
        scope_case_ids: list[str] | None = None,
    ) -> WorkspaceSummary:
        next_steps = self.workspace_builder.build_next_steps(
            workflow=workflow,
            evidence_gaps=evidence_gaps,
            authority_map=authority_map,
            confidence_band=confidence_band,
            intake=intake,
            issue_outline=issue_outline,
        )
        current_scope_case_ids = scope_case_ids or self._collect_scope_case_ids(authority_map)
        return WorkspaceSummary(
            workflow=workflow,
            headline=self.workspace_builder.build_headline(
                workflow=workflow,
                intake=intake,
                question=question,
                topic_query=topic_query,
            ),
            issue_outline=issue_outline,
            evidence_gaps=evidence_gaps,
            next_steps=next_steps,
            current_scope_case_ids=current_scope_case_ids,
            authority_rationale=authority_map.get("rationale") or "Authorities were grouped by direction.",
            supporting_authorities=self._as_similar_cases(authority_map.get("supporting") or []),
            conflicting_authorities=self._as_similar_cases(authority_map.get("conflicting") or []),
            mixed_authorities=self._as_similar_cases(authority_map.get("mixed") or []),
        )

    def _compose_qa_answer(
        self,
        *,
        answer_text: str,
        authority_map: dict[str, Any],
        rag_context: dict[str, Any],
        scope: str,
        source_mode: str,
        include_sources: bool = True,
    ) -> str:
        cleaned = self._normalize_qa_answer(answer_text)
        if not cleaned:
            return "I could not produce a grounded answer for this question from the current evidence."
        source_ids = list(dict.fromkeys(rag_context.get("used_case_ids") or []))
        law_materials = list(rag_context.get("reference_materials") or [])
        if not source_ids:
            source_ids = self._collect_scope_case_ids(authority_map)
        if include_sources and source_ids:
            suffix = f"\n\nSources: {', '.join(source_ids[:3])}"
            if law_materials:
                law_sources = [self._format_reference_material_label(item) for item in law_materials[:2]]
                suffix += f"\nLaw sources: {', '.join(law_sources)}"
            if source_mode in {"document_plus_reference_law", "document_plus_reference_law_plus_case"} and rag_context.get("document_used"):
                filename = rag_context.get("document_filename") or "uploaded document"
                suffix += f"\nDocument context: {filename}"
            return f"{cleaned}{suffix}"
        if include_sources and law_materials:
            law_sources = [self._format_reference_material_label(item) for item in law_materials[:3]]
            suffix = f"\n\nLaw sources: {', '.join(law_sources)}"
            if source_mode in {"document_plus_reference_law", "document_plus_reference_law_plus_case"} and rag_context.get("document_used"):
                filename = rag_context.get("document_filename") or "uploaded document"
                suffix += f"\nDocument context: {filename}"
            return f"{cleaned}{suffix}"
        if include_sources and source_mode == "document_only" and rag_context.get("document_used"):
            filename = rag_context.get("document_filename") or "uploaded document"
            return f"{cleaned}\n\nSource: {filename}"
        if include_sources and source_mode == "document_plus_case" and rag_context.get("document_used"):
            filename = rag_context.get("document_filename") or "uploaded document"
            return f"{cleaned}\n\nDocument context: {filename}"
        return cleaned

    @staticmethod
    def _empty_document_context() -> dict[str, Any]:
        return {
            "used": False,
            "document_id": None,
            "filename": None,
            "coverage_note": "No uploaded document context was added for this turn.",
            "context_text": "",
        }

    @staticmethod
    def _empty_law_context() -> dict[str, Any]:
        return {
            "used": False,
            "materials": [],
            "coverage_note": "No official law materials were added for this turn.",
            "context_text": "",
            "best_match_type": "none",
            "retrieval_confidence": "low",
        }

    @staticmethod
    def _normalize_ref_label(value: str | None) -> str:
        cleaned = normalize_whitespace(value or "").lower()
        cleaned = re.sub(r"^(section|article|rule)\s+", "", cleaned, flags=re.I)
        cleaned = cleaned.replace(" ", "")
        return cleaned

    @classmethod
    def _expected_reference_target(
        cls,
        *,
        question: str,
        question_profile: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        lowered = normalize_whitespace(question).lower().replace("artical", "article")
        domain = normalize_whitespace((question_profile or {}).get("domain") or "").lower()
        task = str((question_profile or {}).get("task") or "")

        def payload(
            *,
            target_id: str,
            title_aliases: list[str],
            section_ref: str,
            label: str,
            required: bool = True,
        ) -> dict[str, Any]:
            return {
                "id": target_id,
                "title_aliases": title_aliases,
                "section_ref": section_ref,
                "label": label,
                "required": required,
            }

        if "article 21a" in lowered or "right to education" in lowered:
            return payload(
                target_id="constitution_article_21a",
                title_aliases=["constitution of india", "constitution"],
                section_ref="article 21a",
                label="Article 21A of the Constitution of India",
            )
        if (
            "article 300a" in lowered
            or "right to property" in lowered
            or ("property" in lowered and "authority of law" in lowered)
            or "deprived of property" in lowered
        ):
            return payload(
                target_id="constitution_article_300a",
                title_aliases=["constitution of india", "constitution"],
                section_ref="article 300a",
                label="Article 300A of the Constitution of India",
            )
        if "article 21" in lowered or "personal liberty" in lowered:
            return payload(
                target_id="constitution_article_21",
                title_aliases=["constitution of india", "constitution"],
                section_ref="article 21",
                label="Article 21 of the Constitution of India",
            )
        if domain == "privacy":
            if any(marker in lowered for marker in ("job portal", "third party", "third-party", "personal data", "without consent", "data protection")):
                return payload(
                    target_id="privacy_data_protection",
                    title_aliases=["digital personal data protection act", "data protection"],
                    section_ref="",
                    label="Digital Personal Data Protection Act, 2023",
                    required=False,
                )
            if any(marker in lowered for marker in ("cctv", "surveillance", "recording", "records my family", "house entrance")):
                return payload(
                    target_id="privacy_surveillance",
                    title_aliases=[],
                    section_ref="",
                    label="privacy and surveillance-related legal material",
                    required=False,
                )
        if domain == "information" or "rti" in lowered:
            no_response = any(
                marker in lowered
                for marker in ("not answered", "no response", "no reply", "not received", "not replied", "within 30 days")
            )
            remedy_asked = any(marker in lowered for marker in ("what can", "what should", "remedy", "appeal", "do if", "can do"))
            if no_response and remedy_asked:
                return payload(
                    target_id="rti_section_19_first_appeal",
                    title_aliases=["right to information act", "rti act", "rti"],
                    section_ref="section 19",
                    label="Section 19 of the Right to Information Act, 2005",
                )
            if any(marker in lowered for marker in ("reply", "response", "respond", "answered", "pio", "cpio")) and any(
                marker in lowered for marker in ("time", "days", "limit", "within", "30")
            ):
                return payload(
                    target_id="rti_section_7",
                    title_aliases=["right to information act", "rti act", "rti"],
                    section_ref="section 7",
                    label="Section 7 of the Right to Information Act, 2005",
                )
            if "first appeal" in lowered:
                return payload(
                    target_id="rti_section_19_first_appeal",
                    title_aliases=["right to information act", "rti act", "rti"],
                    section_ref="section 19",
                    label="Section 19 of the Right to Information Act, 2005",
                )
            if "second appeal" in lowered:
                return payload(
                    target_id="rti_section_19_second_appeal",
                    title_aliases=["right to information act", "rti act", "rti"],
                    section_ref="section 19",
                    label="Section 19 of the Right to Information Act, 2005",
                )
            if "personal information" in lowered or "8(1)(j)" in lowered:
                return payload(
                    target_id="rti_section_8_1_j",
                    title_aliases=["right to information act", "rti act", "rti"],
                    section_ref="section 8(1)(j)",
                    label="Section 8(1)(j) of the Right to Information Act, 2005",
                )
            if any(marker in lowered for marker in ("inspection", "inspect records", "certified copies", "2(j)")):
                return payload(
                    target_id="rti_section_2_j",
                    title_aliases=["right to information act", "rti act", "rti"],
                    section_ref="section 2(j)",
                    label="Section 2(j) of the Right to Information Act, 2005",
                )
        if domain == "consumer" or "consumer protection" in lowered:
            if any(marker in lowered for marker in ("section 39", "remedy", "remedies", "refund", "replacement", "repair", "compensation")):
                return payload(
                    target_id="consumer_section_39",
                    title_aliases=["consumer protection act", "consumer act"],
                    section_ref="section 39",
                    label="Section 39 of the Consumer Protection Act, 2019",
                )
        if domain == "service" or "ccs" in lowered:
            if "rule 14" in lowered or "major penalty" in lowered or "disciplinary proceedings" in lowered:
                return payload(
                    target_id="ccs_rule_14",
                    title_aliases=["ccs cca rules", "disciplinary rules"],
                    section_ref="rule 14",
                    label="Rule 14 of the CCS (CCA) Rules",
                )
            if "rule 16" in lowered or "minor penalty" in lowered:
                return payload(
                    target_id="ccs_rule_16",
                    title_aliases=["ccs cca rules", "disciplinary rules"],
                    section_ref="rule 16",
                    label="Rule 16 of the CCS (CCA) Rules",
                )
            if "difference between rule 14 and rule 16" in lowered:
                return payload(
                    target_id="ccs_rule_14_vs_16",
                    title_aliases=["ccs cca rules", "disciplinary rules"],
                    section_ref="rule 14",
                    label="Rules 14 and 16 of the CCS (CCA) Rules",
                )
        if task in LAW_FIRST_TASKS and any(token in lowered for token in ("section ", "article ", "rule ")):
            refs = re.findall(r"\b(?:section|article|rule)\s+\d+[A-Za-z]?(?:\([^)]+\))*", lowered, flags=re.I)
            if refs:
                return payload(
                    target_id="generic_exact_provision",
                    title_aliases=[],
                    section_ref=refs[0],
                    label=refs[0],
                    required=False,
                )
        return None

    @classmethod
    def _validate_reference_law_support(
        cls,
        *,
        question: str,
        question_profile: dict[str, Any] | None,
        law_context: dict[str, Any],
    ) -> dict[str, Any]:
        target = cls._expected_reference_target(question=question, question_profile=question_profile)
        materials = list(law_context.get("materials") or [])
        if not target:
            return {"required": False, "matched": bool(materials), "target": None}
        target_ref = cls._normalize_ref_label(target.get("section_ref"))
        aliases = [normalize_whitespace(alias).lower() for alias in target.get("title_aliases") or []]
        for material in materials[:3]:
            title_norm = normalize_whitespace(material.get("title") or "").lower()
            ref_norm = cls._normalize_ref_label(material.get("section_ref"))
            title_match = not aliases or any(alias == title_norm or alias in title_norm for alias in aliases)
            ref_match = not target_ref or target_ref == ref_norm
            if title_match and ref_match:
                return {"required": bool(target.get("required")), "matched": True, "target": target, "material": material}
        return {"required": bool(target.get("required")), "matched": False, "target": target}

    @classmethod
    def _should_use_reference_law_direct_answer(
        cls,
        *,
        question_profile: dict[str, Any],
        law_context: dict[str, Any],
        evidence_assessment: dict[str, Any],
    ) -> bool:
        task = str(question_profile.get("task") or "")
        if task not in LAW_FIRST_TASKS:
            return False
        if not law_context.get("used"):
            return False
        return str(evidence_assessment.get("status") or "") != "poor"

    @classmethod
    def _build_reference_law_direct_answer(
        cls,
        *,
        question: str,
        question_profile: dict[str, Any],
        law_context: dict[str, Any],
        similar_cases: list[dict[str, Any]],
    ) -> str:
        materials = list(law_context.get("materials") or [])
        validation = cls._validate_reference_law_support(
            question=question,
            question_profile=question_profile,
            law_context=law_context,
        )
        target = validation.get("target") or {}
        target_id = str(target.get("id") or "")
        lead = materials[0] if materials else {}
        lead_label = cls._format_reference_material_label(lead) if lead else "official law material"
        similar_case_ids = [item.get("case_id") for item in similar_cases[:2] if item.get("case_id")]
        case_tail = (
            "\n\nRelated cases: " + ", ".join(similar_case_ids)
            if similar_case_ids
            else ""
        )

        templates = {
            "constitution_article_21": (
                "#### Answer\n"
                "- Article 21 protects life and personal liberty. No person can be deprived of life or personal liberty except according to procedure established by law.\n\n"
                "#### Source used\n"
                f"- {lead_label}.\n\n"
                "#### Why it applies\n"
                "- The question asks which constitutional provision protects personal liberty, and Article 21 is the direct constitutional source.\n\n"
                "#### Caution\n"
                "- This is legal information, not legal advice."
            ),
            "constitution_article_21a": (
                "#### Answer\n"
                "- Article 21A expressly provides free and compulsory education for children between 6 and 14 years of age.\n\n"
                "#### Source used\n"
                f"- {lead_label}.\n\n"
                "#### Why it applies\n"
                "- The question concerns the right to education, and Article 21A is the direct constitutional provision.\n\n"
                "#### Caution\n"
                "- This is legal information, not legal advice."
            ),
            "constitution_article_300a": (
                "#### Answer\n"
                "- Article 300A protects property by stating that no person shall be deprived of property save by authority of law.\n\n"
                "#### Source used\n"
                f"- {lead_label}.\n\n"
                "#### Why it applies\n"
                "- The question asks whether property can be taken without legal authority. Article 300A directly answers that point.\n\n"
                "#### Caution\n"
                "- Article 300A is a constitutional protection, but it is not a fundamental right under Part III."
            ),
            "rti_section_7": (
                "#### Answer\n"
                "- Under the RTI Act, the Public Information Officer must ordinarily respond within 30 days. Where the information concerns life or liberty, the response is required within 48 hours.\n\n"
                "#### Source used\n"
                f"- {lead_label}.\n\n"
                "#### Why it applies\n"
                "- The question asks about the RTI response timeline, and Section 7 is the main timeline provision.\n\n"
                "#### Next step\n"
                "- If no response is received after the statutory period, check the first appeal route under Section 19."
            ),
            "rti_section_19_first_appeal": (
                "#### Answer\n"
                "- A first appeal under Section 19 is filed before the First Appellate Authority within 30 days from the expiry of the response period or from receipt of the PIO decision. Delay may be condoned for sufficient cause.\n\n"
                "#### Source used\n"
                f"- {lead_label}.\n\n"
                "#### Why it applies\n"
                "- The question asks what remedy is available after no RTI response, and Section 19 provides the appeal framework.\n\n"
                "#### Next step\n"
                "- File the first appeal with the RTI application, filing proof, and details of non-response."
            ),
            "rti_section_19_second_appeal": (
                "#### Answer\n"
                "- A second appeal under Section 19 is filed before the Information Commission within 90 days from the date on which the decision should have been made or was actually received, subject to condonation for sufficient cause.\n\n"
                "#### Source used\n"
                f"- {lead_label}.\n\n"
                "#### Why it applies\n"
                "- The question concerns the second appeal stage after the first appeal decision or failure to decide.\n\n"
                "#### Next step\n"
                "- Prepare the first appeal record, PIO reply if any, and the first appellate order or proof of non-decision."
            ),
            "rti_section_8_1_j": (
                "#### Answer\n"
                "- Section 8(1)(j) deals with personal information. In the current statutory text, information which relates to personal information is exempted from disclosure under this clause.\n\n"
                "#### Source used\n"
                f"- {lead_label}.\n\n"
                "#### Why it applies\n"
                "- The question asks whether a public authority can refuse information as personal information, and Section 8(1)(j) is the direct exemption provision.\n\n"
                "#### Caution\n"
                "- Apply this cautiously because RTI disclosure questions can also involve other RTI provisions, later notifications, and facts about whether the requested material is truly personal information."
            ),
            "rti_section_2_j": (
                "#### Answer\n"
                "- Yes. Section 2(j) includes inspection of work, documents, and records, and the right to take notes, extracts, and certified copies.\n\n"
                "#### Source used\n"
                f"- {lead_label}.\n\n"
                "#### Why it applies\n"
                "- The question asks about inspection or certified copies under RTI, and Section 2(j) defines the right to information.\n\n"
                "#### Caution\n"
                "- Access may still be subject to valid exemptions under the RTI Act."
            ),
            "consumer_section_39": (
                "#### Answer\n"
                "- Section 39 allows the Consumer Commission to order relief such as removal of defects, replacement of goods, refund of price, compensation, discontinuance of unfair trade practices, withdrawal of hazardous goods, corrective advertisement, and costs, depending on the case.\n\n"
                "#### Source used\n"
                f"- {lead_label}.\n\n"
                "#### Why it applies\n"
                "- The question asks about a refund or remedy for a defective product, and Section 39 lists the reliefs a Consumer Commission may order.\n\n"
                "#### Next step\n"
                "- Keep the invoice, complaint records, delivery proof, defect photos, and seller replies before filing or escalating the complaint."
            ),
            "privacy_data_protection": (
                "#### Answer\n"
                "- This looks like a privacy or data-protection issue, not an ordinary consumer-case or RTI-case question.\n\n"
                "#### Source used\n"
                f"- Target source family: {target.get('label') or 'privacy/data-protection law'}.\n"
                f"- Closest retrieved material: {lead_label}.\n\n"
                "#### Why it applies\n"
                "- The facts involve personal documents being shared with third parties without consent.\n\n"
                "#### Next step\n"
                "- Preserve supporting records, privacy-policy text, consent records, emails, and the job portal complaint history before escalating.\n\n"
                "#### Caution\n"
                "- Verify the current data-protection and IT-law position before relying on this. The system should not answer this from consumer judgments alone."
            ),
            "privacy_surveillance": (
                "#### Answer\n"
                "- This looks like a privacy, surveillance, or civil-neighbour dispute issue, not a consumer-dispute question.\n\n"
                "#### Source used\n"
                f"- Target source family: {target.get('label') or 'privacy/surveillance law'}.\n"
                f"- Closest retrieved material: {lead_label}.\n\n"
                "#### Why it applies\n"
                "- The facts involve continuous recording of a home entrance and family members.\n\n"
                "#### Next step\n"
                "- Preserve photos of the camera angle, any recordings, written objections, and police diary/complaint details before seeking a formal remedy.\n\n"
                "#### Caution\n"
                "- The correct route may depend on local facts and current privacy, criminal, municipal, or civil-injunction law. Do not rely on unrelated consumer cases for this."
            ),
            "ccs_rule_14": (
                "#### Answer\n"
                "- Rule 14 governs the procedure for imposing major penalties under the CCS (CCA) Rules.\n\n"
                "#### Source used\n"
                f"- {lead_label}.\n\n"
                "#### Why it applies\n"
                "- The question concerns disciplinary proceedings or major penalty procedure in service matters.\n\n"
                "#### Caution\n"
                "- The exact procedure depends on the applicable service rules and the documents in the disciplinary record."
            ),
            "ccs_rule_16": (
                "#### Answer\n"
                "- Rule 16 governs the procedure for imposing minor penalties under the CCS (CCA) Rules.\n\n"
                "#### Source used\n"
                f"- {lead_label}.\n\n"
                "#### Why it applies\n"
                "- The question concerns minor penalty procedure in service disciplinary matters.\n\n"
                "#### Caution\n"
                "- The exact procedure depends on the service rules applicable to the employee."
            ),
            "ccs_rule_14_vs_16": (
                "#### Answer\n"
                "- Rule 14 deals with major-penalty proceedings, while Rule 16 deals with minor-penalty proceedings under the CCS (CCA) Rules.\n\n"
                "#### Source used\n"
                f"- {lead_label} and the companion CCS (CCA) rule provisions.\n\n"
                "#### Why it applies\n"
                "- The question asks the difference between major and minor penalty procedures.\n\n"
                "#### Caution\n"
                "- Check the exact charge, proposed penalty, and applicable service rules before relying on the classification."
            ),
        }
        if target_id in templates:
            return templates[target_id] + case_tail

        excerpt = normalize_whitespace(str(lead.get("excerpt") or ""))
        if not excerpt:
            return (
                "#### Answer\n"
                "- I could not extract a reliable statute-first answer from the current official law materials.\n\n"
                "#### Limits\n"
                "- Please narrow the question to a specific article, section, rule, timeline, or remedy."
            )
        return (
            "#### Answer\n"
            f"- The closest official-law source retrieved for this question is {lead_label}.\n\n"
            "#### Source used\n"
            f"- {shorten_text(excerpt, 360)}\n\n"
            "#### Why it applies\n"
            "- The system selected this because it was the strongest match in the reference-law lane.\n\n"
            "#### Caution\n"
            "- Treat this as a starting point unless the retrieved Act, Article, Section, or Rule exactly matches the legal issue."
            + case_tail
        )

    @staticmethod
    def _should_use_reference_law(
        *,
        source_mode: str,
        question: str,
        retrieval_query: str,
        question_profile: dict[str, Any],
    ) -> bool:
        if source_mode == "document_only":
            return False
        if str(question_profile.get("task") or "") in LAW_FIRST_TASKS | HYBRID_LAW_TASKS:
            return True
        if question_profile.get("statute_sensitive"):
            return True
        task = str(question_profile.get("task") or "")
        domain = str(question_profile.get("domain") or "")
        if task in {"general_research", "legal_compare"} and domain in {
            "consumer",
            "information",
            "service",
            "tax",
            "motor_accident",
            "criminal",
            "privacy",
        }:
            return True
        lowered = normalize_whitespace(f"{question} {retrieval_query}").lower()
        law_markers = (
            "section ",
            "article ",
            "rule ",
            "act",
            "statute",
            "provision",
            "procedure",
            "limitation",
            "appeal",
            "jurisdiction",
            "remedy",
            "rights",
            "penalty",
            "compensation",
        )
        return any(marker in lowered for marker in law_markers)

    @staticmethod
    def _format_reference_material_label(item: dict[str, Any]) -> str:
        title = normalize_whitespace(item.get("title") or "official law material")
        section_ref = normalize_whitespace(item.get("section_ref") or "")
        if section_ref:
            return f"{title} - {section_ref}"
        return title

    @staticmethod
    def _as_reference_materials(materials: list[dict[str, Any]]) -> list[ReferenceMaterial]:
        payload: list[ReferenceMaterial] = []
        seen: set[tuple[str, str]] = set()
        for item in materials:
            title = normalize_whitespace(item.get("title") or "")
            section_ref = normalize_whitespace(item.get("section_ref") or "")
            if not title:
                continue
            key = (title, section_ref)
            if key in seen:
                continue
            seen.add(key)
            payload.append(
                ReferenceMaterial(
                    title=title,
                    section_ref=section_ref or None,
                    authority_type=item.get("authority_type"),
                    domain=item.get("domain"),
                    page_start=item.get("page_start"),
                    page_end=item.get("page_end"),
                    retrieval_strategy=item.get("retrieval_strategy") or "hybrid",
                    retrieval_confidence=item.get("retrieval_confidence"),
                    excerpt=item.get("excerpt") or "",
                )
            )
        return payload

    @staticmethod
    def _resolve_source_mode(value: str | None) -> str:
        normalized = normalize_whitespace(value or "").lower().replace("-", "_").replace(" ", "_")
        if normalized in {"document_only", "case_corpus_only", "document_plus_case"}:
            return normalized
        return "document_plus_case"

    @staticmethod
    def _resolve_retrieval_profile(
        *,
        value: str | None,
        source_mode: str,
        recommended_profile: str | None = None,
    ) -> str:
        if source_mode == "document_only":
            return "fast"
        normalized = normalize_whitespace(value or "").lower()
        if normalized == "deep":
            return "deep"
        if normalized == "fast":
            return "fast"
        if normalize_whitespace(recommended_profile or "").lower() == "deep":
            return "deep"
        return "fast"

    @staticmethod
    def _normalize_qa_answer(answer_text: str) -> str:
        cleaned = (answer_text or "").replace("\r\n", "\n").strip()
        if not cleaned:
            return ""

        normalized_lines: list[str] = []
        blank_pending = False
        for raw_line in cleaned.split("\n"):
            line = raw_line.strip()
            if not line:
                if normalized_lines and not blank_pending:
                    normalized_lines.append("")
                blank_pending = True
                continue
            normalized_lines.append(line)
            blank_pending = False
        return "\n".join(normalized_lines).strip()

    @staticmethod
    def _join_answer_lines(lines: list[str]) -> str:
        paragraphs: list[str] = []
        current: list[str] = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                if current:
                    paragraphs.append(normalize_whitespace(" ".join(current)))
                    current = []
                continue
            current.append(stripped)

        if current:
            paragraphs.append(normalize_whitespace(" ".join(current)))

        return "\n\n".join(paragraphs)

    @staticmethod
    def _build_follow_up_suggestions(
        *,
        authority_map: dict[str, Any],
        scope: str,
        question: str,
        source_mode: str,
        evidence_assessment: dict[str, Any] | None = None,
    ) -> list[str]:
        suggestions: list[str] = []
        evidence_strength = str((evidence_assessment or {}).get("evidence_strength") or "")
        if source_mode == "document_only":
            suggestions.append("Point me to the exact section, clause, or paragraph for this answer.")
            suggestions.append("Explain this in simpler language.")
            if any(token in normalize_whitespace(question).lower() for token in ("section", "act", "rule", "article")):
                suggestions.append("Give the exact wording and then paraphrase it.")
            else:
                suggestions.append("Summarize the same point in two short paragraphs.")
            return suggestions[:3]
        if source_mode in {"reference_law_only", "document_plus_reference_law"}:
            suggestions.append("Show the exact provision wording and then paraphrase it.")
            suggestions.append("Explain this in simpler language.")
            suggestions.append("Now show related cases only if they add something beyond the statute.")
            return suggestions[:3]
        if source_mode in {"reference_law_plus_case", "document_plus_reference_law_plus_case"}:
            suggestions.append("Which part comes from the statute, and which part comes from the cases?")
            suggestions.append("Explain this in simpler language.")
            suggestions.append("Show only the most relevant related cases.")
            return suggestions[:3]
        if authority_map.get("supporting"):
            suggestions.append("Which of these authorities is strongest, and why?")
        if authority_map.get("conflicting"):
            suggestions.append("What is the main counter-argument from the conflicting authorities?")
        elif evidence_strength == "mixed":
            suggestions.append("What facts would most likely weaken this position?")
        if scope == "corpus":
            suggestions.append("Now answer the next question only from the current evidence.")
        else:
            suggestions.append("Expand this question back to the full case library.")
        suggestions.append("Explain this in simpler language.")
        return suggestions[:3]

    @staticmethod
    def _should_use_answer_history(question: str, *, chat_history: list[dict[str, str]]) -> bool:
        if not chat_history:
            return False
        lowered = normalize_whitespace(question).lower()
        if LegalAIPipeline._is_style_only_follow_up(lowered):
            return True
        if any(marker in lowered for marker in ("this case", "this answer", "the above", "that case", "those cases")):
            return True
        if len(re.findall(r"\b\w+\b", lowered)) <= 6:
            return True
        return False

    @staticmethod
    def _is_similarity_lookup_question(question: str) -> bool:
        lowered = normalize_whitespace(question).lower()
        return any(
            marker in lowered
            for marker in (
                "find similar",
                "similar judgments",
                "similar cases",
                "closest cases",
                "closest judgments",
            )
        )

    @staticmethod
    def _build_similarity_lookup_answer(
        *,
        question: str,
        similar_cases: list[dict[str, Any]],
        evidence_pack: dict[str, Any] | None = None,
    ) -> str:
        if not similar_cases:
            return (
                "#### Answer\n"
                "- I could not find closely matching judgments for this query.\n\n"
                "#### Next step\n"
                "- Add the forum, case type, legal issue, or key facts to improve precedent retrieval."
            )
        lines = [
            "#### Answer",
            "- I found the following judgments as the closest available matches.",
            "",
            "#### Closest cases",
        ]
        cards = list((evidence_pack or {}).get("cards") or [])
        if cards:
            for card in cards[:3]:
                lines.append(
                    f"- {card['case_id']}: {card['proposition']} "
                    f"({card['support_type']}, {str(card['authority_level']).replace('_', ' ')})."
                )
            lines.extend(
                [
                    "",
                    "#### Why they match",
                    "- The matches are based on similar issue terms, factual patterns, and retrieved case-law passages.",
                    "",
                    "#### Limits",
                    "- Similarity is a retrieval signal, not a final legal conclusion. Verify the full judgments before relying on them.",
                ]
            )
            return "\n".join(lines)
        for item in similar_cases[:3]:
            case_id = item.get("case_id") or "Unknown case"
            label = item.get("label_name") or "Unknown outcome"
            excerpt = normalize_whitespace(item.get("excerpt") or item.get("summary") or "")
            short_excerpt = excerpt[:180].rstrip()
            if short_excerpt and len(excerpt) > 180:
                short_excerpt += "..."
            lines.append(f"- {case_id} ({label}): {short_excerpt or 'Relevant matching passage retrieved.'}")
        lines.extend(
            [
                "",
                "#### Why they match",
                "- The matches are based on the current case-law retrieval score and available passages.",
                "",
                "#### Limits",
                "- Similarity is a retrieval signal rather than a final doctrinal conclusion. Read the full judgments before citing them.",
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def _build_demo_case_explanation_answer(case: dict[str, Any]) -> str:
        case_id = case.get("case_id") or "demo case"
        title = case.get("title") or case_id
        court = case.get("court") or "demo forum"
        outcome = case.get("label_name") or "demo outcome"
        excerpt = normalize_whitespace(case.get("summary") or case.get("excerpt") or "")
        if not excerpt:
            excerpt = "The selected demo case was retrieved from the sample case-law store."
        return "\n".join(
            [
                "#### Answer",
                f"- `{case_id}` is a sample case-law record titled **{title}** from {court}.",
                "",
                "#### Core point",
                f"- {excerpt}",
                "",
                "#### Outcome signal",
                f"- The sample label for this record is **{outcome}**.",
                "",
                "#### How to use it",
                "- Treat this as a demo retrieval explanation. In the full system, the retrieved judgment should be opened and verified before being cited.",
            ]
        )

    @staticmethod
    def _authority_bullets(items: list[dict[str, Any]]) -> list[str]:
        return [
            f"- `{item['case_id']}` ({item.get('label_name') or 'Unknown'}) -> {item.get('excerpt') or item.get('summary') or 'No excerpt available.'}"
            for item in items[:3]
        ]

    @staticmethod
    def _collect_scope_case_ids(authority_map: dict[str, Any]) -> list[str]:
        case_ids: list[str] = []
        for group in ("supporting", "conflicting", "mixed"):
            for item in authority_map.get(group) or []:
                case_id = item.get("case_id")
                if case_id and case_id not in case_ids:
                    case_ids.append(case_id)
        return case_ids[:5]

    @staticmethod
    def _metadata_filters(
        source: dict[str, Any],
        query_text: str | None = None,
        domain_override: str | None = None,
    ) -> dict[str, str]:
        case_type = normalize_whitespace(source.get("case_type"))
        forum = normalize_whitespace(source.get("forum"))
        query_profile = (
            {"domain": normalize_whitespace(domain_override)}
            if normalize_whitespace(domain_override)
            else infer_query_domain(
                query_text or "",
                case_type_hint=case_type,
            )
        )
        return {
            "case_type": case_type,
            "forum": forum,
            "domain": str(query_profile.get("domain") or ""),
        }

    @staticmethod
    def _as_similar_cases(items: list[dict[str, Any]]) -> list[SimilarCase]:
        return [SimilarCase(**item) for item in items]
