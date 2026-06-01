from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    app_name: str = "Legal AI Assistant"
    log_level: str = "INFO"
    demo_mode: bool = False

    classifier_repo_id: str = "L-NLProc/NyayaAnumana-Transformer-Models"
    classifier_subfolder: str = (
        "InLegalBert/ternary/"
        "InLegalBERT_CJPE_ext_SCI_HCs_Tribunal_Dailyorder_multi_wo_RoD_ternary"
    )
    embedding_model_name: str = "bhavyagiri/InLegal-Sbert"
    embedding_query_instruction: str = ""
    embedding_document_instruction: str = ""

    train_dataset_path: str = (
        r"B:\Dataset\CJPE_ext_SCI_HCs_tribunals_dailyorder_multi_wo_RoD_ternary.csv"
    )
    dev_dataset_path: str = (
        r"B:\Dataset\CJPE_ext_SCI_HCs_tribunals_dailyorder_dev_wo_RoD_ternary (1).csv"
    )
    test_dataset_path: str = (
        r"B:\Dataset\CJPE_ext_SCI_HCs_tribunals_dailyorder_test_wo_RoD_ternary.csv"
    )
    sample_dataset_path: str = r"B:\Dataset\sample_10k.csv"

    retrieval_index_path: str = "artifacts/case_index.faiss"
    retrieval_metadata_path: str = "artifacts/case_metadata.sqlite"
    qa_retrieval_index_path: str = "artifacts/qa_chunk_index.faiss"
    qa_retrieval_metadata_path: str = "artifacts/qa_chunk_metadata.sqlite"
    qa_retrieval_embedding_store_path: str = "artifacts/qa_chunk_embeddings.npy"
    reference_law_source_dir: str = r"C:\new corpus"
    reference_law_index_path: str = "artifacts/reference_law_index.faiss"
    reference_law_metadata_path: str = "artifacts/reference_law_metadata.sqlite"
    retrieval_build_manifest_path: str = "artifacts/retrieval_build_manifest.json"
    retrieval_build_state_path: str = "artifacts/retrieval_build_state.json"
    retrieval_build_work_dir: str = "artifacts/build_work"
    evaluation_output_path: str = "artifacts/classifier_predictions.csv"

    max_input_tokens: int = 512
    chunk_stride: int = 128
    inference_batch_size: int = 8
    retrieval_char_limit: int = 2200
    retrieval_preview_char_limit: int = 700
    retrieval_chunk_words: int = 220
    retrieval_chunk_overlap_words: int = 40
    retrieval_chunk_min_words: int = 60
    retrieval_default_top_k: int = 5
    retrieval_overfetch: int = 12
    retrieval_refine_top_n: int = 10
    retrieval_embedding_batch_size: int = 32
    retrieval_rrf_k: int = 50
    retrieval_match_terms_limit: int = 6
    prediction_review_threshold: float = 62.0

    qa_embedding_model_name: str = "bhavyagiri/InLegal-Sbert"
    qa_embedding_query_prefix: str = ""
    qa_embedding_passage_prefix: str = ""
    qa_retrieval_preview_char_limit: int = 320
    qa_chunk_words: int = 220
    qa_chunk_overlap_words: int = 40
    qa_chunk_min_words: int = 60
    qa_retrieval_overfetch: int = 10
    qa_retrieval_rrf_k: int = 50
    qa_case_shortlist_top_n: int = 12
    qa_fast_case_shortlist_top_n: int = 6
    qa_deep_case_shortlist_top_n: int = 16
    qa_runtime_passages_per_case: int = 3
    qa_runtime_passage_total_limit: int = 12
    qa_query_cache_size: int = 128
    reference_law_overfetch: int = 10
    reference_law_rrf_k: int = 50
    reference_law_max_hits: int = 5
    qa_reader_model_name: str = "google/flan-t5-base"
    qa_multilingual_reader_model_name: str = "google/mt5-small"
    qa_reader_max_context_chars: int = 2600
    qa_reader_max_new_tokens: int = 140
    qa_reader_num_beams: int = 2
    rag_context_max_authorities: int = 2
    rag_context_excerpt_chars: int = 220
    rag_context_max_chars: int = 1250
    rag_history_turns: int = 6

    llm_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("LLM_API_KEY", "OPENAI_API_KEY"),
    )
    llm_base_url: str = Field(
        default="http://127.0.0.1:1234/v1",
        validation_alias=AliasChoices("LLM_BASE_URL", "OPENAI_BASE_URL"),
    )
    llm_model: str = Field(
        default="auto",
        validation_alias=AliasChoices("LLM_MODEL", "OPENAI_MODEL"),
    )
    llm_timeout_seconds: int = Field(
        default=120,
        validation_alias=AliasChoices("LLM_TIMEOUT_SECONDS"),
    )
    llm_max_cases: int = Field(
        default=2,
        validation_alias=AliasChoices("LLM_MAX_CASES"),
    )
    llm_max_output_tokens: int = Field(
        default=140,
        validation_alias=AliasChoices("LLM_MAX_OUTPUT_TOKENS"),
    )
    llm_triage_summary_tokens: int = Field(
        default=320,
        validation_alias=AliasChoices("LLM_TRIAGE_SUMMARY_TOKENS"),
    )
    llm_max_answer_tokens: int = Field(
        default=140,
        validation_alias=AliasChoices("LLM_MAX_ANSWER_TOKENS"),
    )
    llm_fast_answer_tokens: int = Field(
        default=170,
        validation_alias=AliasChoices("LLM_FAST_ANSWER_TOKENS"),
    )
    llm_deep_answer_tokens: int = Field(
        default=260,
        validation_alias=AliasChoices("LLM_DEEP_ANSWER_TOKENS"),
    )
    llm_translation_max_tokens: int = Field(
        default=96,
        validation_alias=AliasChoices("LLM_TRANSLATION_MAX_TOKENS"),
    )

    streamlit_api_url: str = "http://127.0.0.1:8000"

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @model_validator(mode="after")
    def align_embedding_space(self) -> "Settings":
        model_name = (self.embedding_model_name or "").strip() or "bhavyagiri/InLegal-Sbert"
        query_prefix = (self.embedding_query_instruction or "").strip()
        passage_prefix = (self.embedding_document_instruction or "").strip()

        self.embedding_model_name = model_name
        self.embedding_query_instruction = query_prefix
        self.embedding_document_instruction = passage_prefix

        # Keep both retrieval subsystems in the same embedding space even if
        # older environment variables still define QA-specific values.
        self.qa_embedding_model_name = model_name
        self.qa_embedding_query_prefix = query_prefix
        self.qa_embedding_passage_prefix = passage_prefix

        if self.demo_mode:
            self.retrieval_index_path = "artifacts/demo/case_index.faiss"
            self.retrieval_metadata_path = "artifacts/demo/case_metadata.sqlite"
            self.qa_retrieval_index_path = "artifacts/demo/qa_chunk_index.faiss"
            self.qa_retrieval_metadata_path = "artifacts/demo/qa_chunk_metadata.sqlite"
            self.qa_retrieval_embedding_store_path = "artifacts/demo/qa_chunk_embeddings.npy"
            self.reference_law_index_path = "artifacts/demo/reference_law_index.faiss"
            self.reference_law_metadata_path = "artifacts/demo/reference_law_metadata.sqlite"
            self.retrieval_build_manifest_path = "artifacts/demo/retrieval_build_manifest.json"
            self.retrieval_build_state_path = "artifacts/demo/retrieval_build_state.json"
            self.retrieval_build_work_dir = "artifacts/demo/build_work"
        return self

    def resolve_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return PROJECT_ROOT / path

    @property
    def shared_embedding_model_name(self) -> str:
        return self.embedding_model_name

    @property
    def shared_embedding_query_prefix(self) -> str:
        return self.embedding_query_instruction

    @property
    def shared_embedding_passage_prefix(self) -> str:
        return self.embedding_document_instruction

    @property
    def embedding_space_status(self) -> str:
        return "Shared Indian legal embedding space active for case-level and QA retrieval."

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_base_url.strip() and self.llm_model.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
