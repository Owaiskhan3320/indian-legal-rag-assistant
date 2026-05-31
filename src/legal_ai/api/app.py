from __future__ import annotations

import base64

from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import FastAPI, HTTPException

from legal_ai.config import get_settings
from legal_ai.schemas import (
    CaseDetailResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    QuestionAnswerRequest,
    QuestionAnswerResponse,
    ResearchRequest,
    ResearchResponse,
    SessionDocumentClearResponse,
    SessionDocumentUploadRequest,
    SessionDocumentUploadResponse,
)
from legal_ai.services.pipeline import LegalAIPipeline


@lru_cache(maxsize=1)
def get_pipeline() -> LegalAIPipeline:
    return LegalAIPipeline(get_settings())


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_pipeline()
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, lifespan=lifespan)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        pipeline = get_pipeline()
        llm_ready, llm_status_message = pipeline.explainer.probe()
        return HealthResponse(
            status="ok",
            classifier_ready=True,
            retrieval_ready=pipeline.retrieval_ready,
            retrieval_record_count=pipeline.retrieval_record_count,
            qa_retrieval_ready=pipeline.qa_retrieval_ready,
            qa_retrieval_record_count=pipeline.qa_retrieval_record_count,
            reference_law_ready=pipeline.reference_law_ready,
            reference_law_record_count=pipeline.reference_law_record_count,
            llm_ready=llm_ready,
            embedding_model_name=settings.shared_embedding_model_name,
            retrieval_status_message=pipeline.retrieval_status_message or None,
            qa_embedding_model_name=settings.shared_embedding_model_name,
            qa_retrieval_status_message=pipeline.qa_retrieval_status_message or None,
            reference_law_status_message=pipeline.reference_law_status_message or None,
            embedding_space_status=settings.embedding_space_status,
            llm_status_message=llm_status_message,
        )

    @app.post("/predict", response_model=PredictionResponse)
    def predict(payload: PredictionRequest) -> PredictionResponse:
        pipeline = get_pipeline()
        try:
            return pipeline.predict(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/ask", response_model=QuestionAnswerResponse)
    def ask(payload: QuestionAnswerRequest) -> QuestionAnswerResponse:
        pipeline = get_pipeline()
        try:
            return pipeline.answer_question(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/research", response_model=ResearchResponse)
    def research(payload: ResearchRequest) -> ResearchResponse:
        pipeline = get_pipeline()
        try:
            return pipeline.research(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/cases/{case_id}", response_model=CaseDetailResponse)
    def get_case(case_id: str) -> CaseDetailResponse:
        pipeline = get_pipeline()
        try:
            payload = pipeline.get_case_detail(case_id)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if payload is None:
            raise HTTPException(status_code=404, detail="Case not found in retrieval store.")
        return payload

    @app.post("/session-documents", response_model=SessionDocumentUploadResponse)
    def upload_session_document(payload: SessionDocumentUploadRequest) -> SessionDocumentUploadResponse:
        pipeline = get_pipeline()
        try:
            file_bytes = base64.b64decode(payload.file_base64.encode("utf-8"))
            payload = pipeline.store_session_document(
                session_id=payload.session_id,
                filename=payload.filename or "uploaded_document",
                content_type=payload.content_type or "application/octet-stream",
                file_bytes=file_bytes,
            )
            return SessionDocumentUploadResponse(**payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail="The uploaded document payload was invalid.") from exc

    @app.delete("/session-documents/{session_id}", response_model=SessionDocumentClearResponse)
    def clear_session_document(session_id: str) -> SessionDocumentClearResponse:
        pipeline = get_pipeline()
        removed = pipeline.clear_session_document(session_id)
        return SessionDocumentClearResponse(session_id=session_id, removed=removed)

    return app
