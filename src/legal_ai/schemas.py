from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class SimilarCase(BaseModel):
    case_id: str
    similarity: float
    base_similarity: float
    evidence_similarity: float
    label: Optional[int] = None
    label_name: Optional[str] = None
    title: Optional[str] = None
    court: Optional[str] = None
    case_type: Optional[str] = None
    date: Optional[str] = None
    matched_chunk_index: Optional[int] = None
    matched_chunk_count: int = 0
    retrieval_strategy: str
    retrieval_note: Optional[str] = None
    summary: Optional[str] = None
    excerpt: str
    section_label: Optional[str] = None
    support_type: Optional[str] = None
    authority_level: Optional[str] = None
    proposition: Optional[str] = None
    fit_band: Optional[str] = None
    fit_note: Optional[str] = None
    retrieval_confidence: Optional[str] = None
    issue_subtypes: List[str] = Field(default_factory=list)


class ReferenceMaterial(BaseModel):
    title: str
    section_ref: Optional[str] = None
    authority_type: Optional[str] = None
    domain: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    retrieval_strategy: str
    retrieval_confidence: Optional[str] = None
    excerpt: str


class WorkspaceSummary(BaseModel):
    workflow: str
    headline: str
    issue_outline: List[str] = Field(default_factory=list)
    evidence_gaps: List[str] = Field(default_factory=list)
    next_steps: List[str] = Field(default_factory=list)
    current_scope_case_ids: List[str] = Field(default_factory=list)
    authority_rationale: str
    supporting_authorities: List[SimilarCase] = Field(default_factory=list)
    conflicting_authorities: List[SimilarCase] = Field(default_factory=list)
    mixed_authorities: List[SimilarCase] = Field(default_factory=list)


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class CaseDetailResponse(BaseModel):
    case_id: str
    label: Optional[int] = None
    label_name: Optional[str] = None
    title: Optional[str] = None
    court: Optional[str] = None
    date: Optional[str] = None
    word_count: int
    full_text: str


class PredictionRequest(BaseModel):
    session_id: Optional[str] = None
    case_text: Optional[str] = None
    case_type: Optional[str] = None
    user_role: Optional[str] = None
    forum: Optional[str] = None
    facts: Optional[str] = None
    relief_sought: Optional[str] = None
    evidence_summary: Optional[str] = None
    opponent_arguments: Optional[str] = None
    top_k: int = Field(default=3, ge=1, le=5)
    include_explanation: bool = True

    @model_validator(mode="after")
    def validate_input(self) -> "PredictionRequest":
        combined = " ".join(
            [
                self.case_text or "",
                self.facts or "",
                self.relief_sought or "",
                self.evidence_summary or "",
                self.opponent_arguments or "",
            ]
        ).strip()
        if len(combined) < 30 and not (self.session_id or "").strip():
            raise ValueError(
                "Provide either a detailed case narrative, enough structured fields, or an uploaded document session."
            )
        return self


class PredictionResponse(BaseModel):
    input_summary: str
    case_type: Optional[str] = None
    user_role: Optional[str] = None
    forum: Optional[str] = None
    predicted_label: int
    predicted_name: str
    prediction_posture: str
    prediction_posture_reason: str
    favorability_label: str
    favorability_reason: str
    confidence_score: float
    confidence_band: str
    retrieval_confidence: Optional[str] = None
    evidence_strength: Optional[str] = None
    answer_confidence: Optional[str] = None
    probabilities: dict[str, float]
    chunk_count: int
    advisories: List[str]
    similar_cases: List[SimilarCase]
    explanation: str
    explanation_source: str
    workspace: WorkspaceSummary


class QuestionAnswerRequest(BaseModel):
    session_id: Optional[str] = None
    question: str
    case_type: Optional[str] = None
    user_role: Optional[str] = None
    forum: Optional[str] = None
    context_note: Optional[str] = None
    scope: Literal["corpus", "current_result"] = "corpus"
    source_mode: Literal["document_plus_case", "document_only", "case_corpus_only"] = "document_plus_case"
    retrieval_profile: Literal["fast", "deep"] = "fast"
    scope_case_ids: List[str] = Field(default_factory=list, max_length=5)
    chat_history: List[ChatTurn] = Field(default_factory=list, max_length=8)
    top_k: int = Field(default=3, ge=1, le=5)

    @model_validator(mode="after")
    def validate_question(self) -> "QuestionAnswerRequest":
        cleaned_question = (self.question or "").strip()
        if self.source_mode == "document_only":
            min_length = 4
        else:
            min_length = 8 if self.chat_history else 12
        if len(cleaned_question) < min_length:
            raise ValueError("Provide a fuller legal question so retrieval has enough context.")
        if self.scope == "current_result" and not self.scope_case_ids:
            raise ValueError(
                "Current-result Q/A needs at least one retrieved case to scope the answer."
            )
        return self


class QuestionAnswerResponse(BaseModel):
    question: str
    retrieval_query: str
    rewritten_question: Optional[str] = None
    detected_language: str
    answer_language: str
    scope: str
    source_mode: str
    answer: str
    answer_source: str
    retrieval_confidence: Optional[str] = None
    evidence_strength: Optional[str] = None
    answer_confidence: Optional[str] = None
    advisories: List[str]
    follow_up_suggestions: List[str] = Field(default_factory=list)
    supporting_cases: List[SimilarCase]
    reference_materials: List[ReferenceMaterial] = Field(default_factory=list)
    workspace: WorkspaceSummary


class ResearchRequest(BaseModel):
    topic_query: str
    case_type: Optional[str] = None
    user_role: Optional[str] = None
    forum: Optional[str] = None
    top_k: int = Field(default=3, ge=1, le=5)

    @model_validator(mode="after")
    def validate_topic(self) -> "ResearchRequest":
        if len((self.topic_query or "").strip()) < 10:
            raise ValueError("Provide a fuller research topic or issue statement.")
        return self


class ResearchResponse(BaseModel):
    topic_query: str
    retrieval_query: str
    research_snapshot: str
    research_snapshot_source: str
    advisories: List[str]
    retrieved_cases: List[SimilarCase]
    workspace: WorkspaceSummary


class HealthResponse(BaseModel):
    status: str
    classifier_ready: bool
    retrieval_ready: bool
    retrieval_record_count: int
    qa_retrieval_ready: bool
    qa_retrieval_record_count: int
    reference_law_ready: bool = False
    reference_law_record_count: int = 0
    llm_ready: bool
    embedding_model_name: Optional[str] = None
    retrieval_status_message: Optional[str] = None
    qa_embedding_model_name: Optional[str] = None
    qa_retrieval_status_message: Optional[str] = None
    reference_law_status_message: Optional[str] = None
    embedding_space_status: Optional[str] = None
    llm_status_message: Optional[str] = None


class SessionDocumentUploadResponse(BaseModel):
    session_id: str
    document_id: str
    filename: str
    content_type: Optional[str] = None
    chunk_count: int
    word_count: int
    preview_text: str


class SessionDocumentUploadRequest(BaseModel):
    session_id: str
    filename: str
    content_type: Optional[str] = None
    file_base64: str


class SessionDocumentClearResponse(BaseModel):
    session_id: str
    removed: bool
