from __future__ import annotations

import logging
import math
import sqlite3
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from legal_ai.config import Settings
from legal_ai.services.labels import LABEL_ID_TO_NAME
from legal_ai.utils.case_metadata import derive_case_metadata
from legal_ai.utils.domain import (
    apply_domain_rerank,
    candidate_domain_alignment,
    candidate_matches_domain,
    domain_filter_hints,
    issue_subtype_alignment,
)
from legal_ai.utils.text import (
    compact_text,
    lexical_overlap_score,
    normalize_whitespace,
    overlapping_terms,
    search_terms,
    shorten_text,
    split_into_word_chunks,
)


LOGGER = logging.getLogger(__name__)


class SimilarCaseRetriever:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        LOGGER.info(
            "Loading embedding model=%s on device=%s",
            settings.shared_embedding_model_name,
            self.device,
        )
        self.model = SentenceTransformer(settings.shared_embedding_model_name, device=self.device)
        self.index: faiss.Index | None = None
        self.metadata: list[dict] = []
        self.record_count = 0
        self._chunk_cache: dict[str, list[str]] = {}
        self._search_cache: dict[tuple[str, int, tuple[tuple[str, str], ...]], list[dict]] = {}

    def encode_texts(
        self,
        texts: list[str],
        *,
        is_query: bool,
        batch_size: int | None = None,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        prepared_texts = (
            [self._prepare_query_text(text) for text in texts]
            if is_query
            else [self._prepare_document_text(text) for text in texts]
        )
        if not prepared_texts:
            dimension = self.model.get_sentence_embedding_dimension()
            return np.empty((0, dimension), dtype="float32")

        embeddings = self.model.encode(
            prepared_texts,
            batch_size=batch_size or self.settings.retrieval_embedding_batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.ascontiguousarray(embeddings.astype("float32"))

    def encode_query(self, text: str) -> np.ndarray:
        query = self.encode_texts(
            [normalize_whitespace(text)],
            is_query=True,
            batch_size=1,
            show_progress_bar=False,
        )
        return query[0]

    def build(self, records: list[dict]) -> None:
        texts = [record["retrieval_text"] for record in records]
        embeddings = self.encode_texts(texts, is_query=False, show_progress_bar=True)
        if embeddings.shape[0] == 0:
            raise RuntimeError("No retrieval records were generated from the dataset.")

        index = faiss.IndexFlatIP(int(embeddings.shape[1]))
        index.add(embeddings)
        self.index = index
        self.metadata = records
        self.record_count = len(records)
        LOGGER.info(
            "Built FAISS retrieval index records=%s dimension=%s",
            self.record_count,
            embeddings.shape[1],
        )

    def save(self) -> None:
        if self.index is None:
            raise RuntimeError("Retrieval index is not built yet.")

        index_path = self.settings.resolve_path(self.settings.retrieval_index_path)
        metadata_path = self.settings.resolve_path(self.settings.retrieval_metadata_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_path))
        self._write_metadata_database(metadata_path, self.metadata)
        LOGGER.info("Saved FAISS index=%s records=%s", index_path, self.record_count)

    def load(self) -> None:
        index_path = self.settings.resolve_path(self.settings.retrieval_index_path)
        metadata_path = self.settings.resolve_path(self.settings.retrieval_metadata_path)
        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                "Retrieval artifacts are missing. Build the index before starting the API/UI."
            )

        self.index = faiss.read_index(str(index_path))
        self._validate_loaded_artifacts(metadata_path)
        self.record_count = self._count_rows(metadata_path)
        LOGGER.info(
            "Loaded FAISS retrieval index path=%s records=%s",
            index_path,
            self.record_count,
        )

    def search(
        self,
        query: str,
        top_k: int,
        metadata_filters: dict[str, str] | None = None,
        *,
        refine_chunks: bool = True,
        query_profile: dict | None = None,
    ) -> list[dict]:
        if self.index is None:
            raise RuntimeError("Retrieval index is not loaded.")

        cache_key = self._search_cache_key(
            query,
            top_k,
            metadata_filters,
            refine_chunks=refine_chunks,
            query_profile=query_profile,
        )
        cached = self._search_cache.get(cache_key)
        if cached is not None:
            return [dict(item) for item in cached]

        query_vector = self.encode_query(query)
        fetch_k = min(
            max(top_k * self.settings.retrieval_overfetch, self.settings.retrieval_refine_top_n),
            self.record_count,
        )
        if fetch_k == 0:
            return []

        scores, indices = self.index.search(query_vector.reshape(1, -1), fetch_k)
        scores = scores[0].tolist()
        dense_row_ids = [int(idx) for idx in indices[0].tolist() if idx >= 0]
        row_scores = {
            int(idx): float(scores[offset])
            for offset, idx in enumerate(indices[0].tolist())
            if int(idx) >= 0
        }
        dense_rank_map = {row_id: rank for rank, row_id in enumerate(dense_row_ids, start=1)}
        lexical_row_ids = self._lexical_search_row_ids(
            query,
            limit=fetch_k,
            metadata_filters=metadata_filters,
        )
        lexical_rank_map = {row_id: rank for rank, row_id in enumerate(lexical_row_ids, start=1)}
        row_ids = list(dict.fromkeys(dense_row_ids + lexical_row_ids))

        candidates = self._load_candidates(
            row_ids,
            row_scores,
            dense_rank_map=dense_rank_map,
            lexical_rank_map=lexical_rank_map,
            query=query,
            metadata_filters=metadata_filters,
            query_profile=query_profile,
        )
        refined = (
            self._refine_candidates(query_vector, candidates, query=query)
            if refine_chunks
            else self._build_case_level_results(candidates)
        )

        results: list[dict] = []
        for item in refined[:top_k]:
            results.append(
                {
                    "case_id": item["case_id"],
                    "similarity": round(item["similarity"], 4),
                    "base_similarity": round(item["base_similarity"], 4),
                    "evidence_similarity": round(item["evidence_similarity"], 4),
                    "label": item.get("label"),
                    "label_name": LABEL_ID_TO_NAME.get(item.get("label"))
                    if item.get("label") is not None
                    else None,
                    "title": item.get("title"),
                    "court": item.get("court"),
                    "case_type": item.get("case_type"),
                    "date": item.get("date"),
                    "matched_chunk_index": item.get("matched_chunk_index"),
                    "matched_chunk_count": item.get("matched_chunk_count", 0),
                    "retrieval_strategy": item["retrieval_strategy"],
                    "retrieval_note": item.get("retrieval_note"),
                    "summary": item.get("summary"),
                    "excerpt": item["excerpt"],
                    "fit_band": item.get("fit_band"),
                    "fit_note": item.get("fit_note"),
                    "issue_subtypes": item.get("issue_subtypes") or [],
                }
            )
        self._remember_search_result(cache_key, results)
        return results

    def get_case_detail(self, case_id: str) -> dict | None:
        metadata_path = self.settings.resolve_path(self.settings.retrieval_metadata_path)
        query = (
            "SELECT case_id, label, title, court, date, full_text "
            "FROM retrieval_records WHERE case_id = ? LIMIT 1"
        )
        with sqlite3.connect(metadata_path) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(query, (case_id,)).fetchone()
        if row is None:
            return None

        payload = dict(row)
        full_text = payload.get("full_text") or ""
        payload["word_count"] = len(full_text.split())
        return payload

    def _prepare_query_text(self, text: str) -> str:
        cleaned = normalize_whitespace(text)
        instruction = normalize_whitespace(self.settings.shared_embedding_query_prefix)
        if instruction:
            return f"{instruction} {cleaned}"
        return cleaned

    def _prepare_document_text(self, text: str) -> str:
        cleaned = normalize_whitespace(text)
        instruction = normalize_whitespace(self.settings.shared_embedding_passage_prefix)
        if instruction:
            return f"{instruction} {cleaned}"
        return cleaned

    def _refine_candidates(
        self,
        query_vector: np.ndarray,
        candidates: list[dict],
        *,
        query: str,
    ) -> list[dict]:
        refine_limit = min(self.settings.retrieval_refine_top_n, len(candidates))
        chunk_texts: list[str] = []
        chunk_owners: list[tuple[int, int, int]] = []
        chunk_counts: dict[int, int] = {}

        for candidate_idx, candidate in enumerate(candidates[:refine_limit]):
            chunks = self._get_case_chunks(candidate["case_id"], candidate["retrieval_text"])
            selected_chunks = self._prefilter_case_chunks(query, chunks)
            chunk_counts[candidate_idx] = len(selected_chunks)
            for chunk_idx, chunk_text in selected_chunks:
                chunk_texts.append(chunk_text)
                chunk_owners.append((candidate_idx, chunk_idx, len(chunk_text.split())))

        best_chunk_by_candidate: dict[int, dict] = {}
        if chunk_texts:
            chunk_embeddings = self.encode_texts(chunk_texts, is_query=False, show_progress_bar=False)
            chunk_scores = np.asarray(chunk_embeddings @ query_vector, dtype="float32")
            for offset, score in enumerate(chunk_scores.tolist()):
                candidate_idx, chunk_idx, chunk_word_count = chunk_owners[offset]
                current = best_chunk_by_candidate.get(candidate_idx)
                if current is None or score > current["score"]:
                    best_chunk_by_candidate[candidate_idx] = {
                        "score": float(score),
                        "chunk_index": chunk_idx,
                        "chunk_words": chunk_word_count,
                        "excerpt": shorten_text(
                            chunk_texts[offset],
                            self.settings.retrieval_preview_char_limit,
                        ),
                    }

        refined_results: list[dict] = []
        for candidate_idx, candidate in enumerate(candidates):
            best_chunk = best_chunk_by_candidate.get(candidate_idx)
            evidence_similarity = (
                best_chunk["score"] if best_chunk is not None else candidate["base_similarity"]
            )
            retrieval_strategy = (
                f"{candidate['retrieval_source']}+chunk"
                if best_chunk is not None
                else candidate["retrieval_source"]
            )
            blended_score = self._blend_scores(candidate["base_similarity"], evidence_similarity)
            refined_results.append(
                {
                    "case_id": candidate["case_id"],
                    "label": candidate.get("label"),
                    "title": candidate.get("title"),
                    "court": candidate.get("court"),
                    "case_type": candidate.get("case_type"),
                    "date": candidate.get("date"),
                    "base_similarity": candidate["base_similarity"],
                    "evidence_similarity": evidence_similarity,
                    "similarity": blended_score,
                    "matched_chunk_index": best_chunk["chunk_index"] if best_chunk else None,
                    "matched_chunk_count": chunk_counts.get(candidate_idx, 0),
                    "retrieval_strategy": retrieval_strategy,
                    "retrieval_note": candidate.get("retrieval_note"),
                    "summary": candidate["preview_text"],
                    "excerpt": best_chunk["excerpt"] if best_chunk else candidate["preview_text"],
                    "fit_band": candidate.get("fit_band"),
                    "fit_note": candidate.get("fit_note"),
                    "issue_subtypes": candidate.get("issue_subtypes"),
                }
            )

        refined_results.sort(key=lambda item: item["similarity"], reverse=True)
        return refined_results

    @staticmethod
    def _blend_scores(base_similarity: float, evidence_similarity: float) -> float:
        return (0.46 * base_similarity) + (0.54 * evidence_similarity)

    def _get_case_chunks(self, case_id: str, retrieval_text: str) -> list[str]:
        cached = self._chunk_cache.get(case_id)
        if cached is not None:
            return cached

        source_text = self._fetch_case_full_text(case_id) or retrieval_text
        chunks = split_into_word_chunks(
            source_text,
            chunk_words=self.settings.retrieval_chunk_words,
            overlap_words=self.settings.retrieval_chunk_overlap_words,
            min_words=self.settings.retrieval_chunk_min_words,
        )
        if len(self._chunk_cache) >= 512:
            oldest_key = next(iter(self._chunk_cache))
            self._chunk_cache.pop(oldest_key, None)
        self._chunk_cache[case_id] = chunks
        return chunks

    def _prefilter_case_chunks(self, query: str, chunks: list[str]) -> list[tuple[int, str]]:
        if not chunks:
            return []
        limit = max(int(self.settings.qa_runtime_passages_per_case), 1)
        scored: list[tuple[float, int, str]] = []
        for chunk_idx, chunk_text in enumerate(chunks):
            lexical = lexical_overlap_score(query, chunk_text)
            matched = len(overlapping_terms(query, chunk_text, limit=self.settings.retrieval_match_terms_limit))
            score = lexical + (0.04 * matched)
            scored.append((score, chunk_idx, chunk_text))
        scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        selected = [(chunk_idx, chunk_text) for score, chunk_idx, chunk_text in scored if score > 0][:limit]
        if selected:
            return selected
        return [(chunk_idx, chunk_text) for chunk_idx, chunk_text in enumerate(chunks[:limit])]

    def _search_cache_key(
        self,
        query: str,
        top_k: int,
        metadata_filters: dict[str, str] | None,
        *,
        refine_chunks: bool,
        query_profile: dict | None = None,
    ) -> tuple[str, int, tuple[tuple[str, str], ...]]:
        normalized_query = normalize_whitespace(query).lower()
        normalized_filters = tuple(
            sorted(
                (
                    str(key),
                    normalize_whitespace(value).lower(),
                )
                for key, value in (metadata_filters or {}).items()
                if normalize_whitespace(value)
            )
        )
        profile_signature = ""
        if query_profile:
            elements = ",".join(str(item) for item in (query_profile.get("legal_elements") or [])[:6])
            subtypes = ",".join(str(item) for item in (query_profile.get("issue_subtypes") or [])[:4])
            profile_signature = (
                f"|domain={normalize_whitespace(query_profile.get('domain') or '').lower()}"
                f"|task={normalize_whitespace(query_profile.get('task') or '').lower()}"
                f"|elements={elements}"
                f"|subtypes={subtypes}"
            )
        return (
            f"{normalized_query}|refine={int(refine_chunks)}{profile_signature}",
            int(top_k),
            normalized_filters,
        )

    def _remember_search_result(
        self,
        cache_key: tuple[str, int, tuple[tuple[str, str], ...]],
        results: list[dict],
    ) -> None:
        if len(self._search_cache) >= 128:
            oldest_key = next(iter(self._search_cache))
            self._search_cache.pop(oldest_key, None)
        self._search_cache[cache_key] = [dict(item) for item in results]

    @staticmethod
    def _build_case_level_results(candidates: list[dict]) -> list[dict]:
        results: list[dict] = []
        for candidate in candidates:
            results.append(
                {
                    "case_id": candidate["case_id"],
                    "label": candidate.get("label"),
                    "title": candidate.get("title"),
                    "court": candidate.get("court"),
                    "case_type": candidate.get("case_type"),
                    "date": candidate.get("date"),
                    "base_similarity": candidate["base_similarity"],
                    "evidence_similarity": candidate["base_similarity"],
                    "similarity": candidate["base_similarity"],
                    "matched_chunk_index": None,
                    "matched_chunk_count": 0,
                    "retrieval_strategy": candidate["retrieval_source"],
                    "retrieval_note": candidate.get("retrieval_note"),
                    "summary": candidate["preview_text"],
                    "excerpt": candidate["preview_text"],
                    "fit_band": candidate.get("fit_band"),
                    "fit_note": candidate.get("fit_note"),
                    "issue_subtypes": candidate.get("issue_subtypes"),
                }
            )
        return results

    def _fetch_case_full_text(self, case_id: str) -> str:
        metadata_path = self.settings.resolve_path(self.settings.retrieval_metadata_path)
        query = "SELECT full_text FROM retrieval_records WHERE case_id = ? LIMIT 1"
        with sqlite3.connect(metadata_path) as connection:
            row = connection.execute(query, (case_id,)).fetchone()
        if not row:
            return ""
        return normalize_whitespace(row[0] or "")

    def _load_candidates(
        self,
        row_ids: list[int],
        row_scores: dict[int, float],
        *,
        dense_rank_map: dict[int, int],
        lexical_rank_map: dict[int, int],
        query: str,
        metadata_filters: dict[str, str] | None,
        query_profile: dict | None = None,
    ) -> list[dict]:
        rows_by_id = self._fetch_rows(row_ids)
        candidates: list[dict] = []
        case_type_hint = normalize_whitespace((metadata_filters or {}).get("case_type"))
        domain_filter = normalize_whitespace((metadata_filters or {}).get("domain")).lower()
        domain_confidence = float((query_profile or {}).get("domain_confidence") or 0.0)
        legal_elements = list((query_profile or {}).get("legal_elements") or [])
        issue_subtypes = list((query_profile or {}).get("issue_subtypes") or [])
        exact_terms = [normalize_whitespace(str(item)).lower() for item in (query_profile or {}).get("exact_terms") or []]
        remedy_terms = [normalize_whitespace(str(item)).lower() for item in (query_profile or {}).get("remedy_terms") or []]
        referenced_case_ids = {
            normalize_whitespace(str(item)).lower()
            for item in (query_profile or {}).get("referenced_case_ids") or []
        }
        direct_case_lookup = bool((query_profile or {}).get("direct_case_lookup"))
        is_triage = str((query_profile or {}).get("workflow") or "").lower() == "triage"
        lowered_query = normalize_whitespace(query).lower()
        for row_id in row_ids:
            row = rows_by_id.get(int(row_id))
            if row is None:
                continue
            if not self._matches_metadata_filters(row, metadata_filters):
                continue
            dense_rank = dense_rank_map.get(int(row_id))
            lexical_rank = lexical_rank_map.get(int(row_id))
            hybrid_base = self._rrf_score(dense_rank, lexical_rank)
            lexical_score = lexical_overlap_score(query, row["retrieval_text"])
            matched_terms = overlapping_terms(
                query,
                row["retrieval_text"],
                limit=self.settings.retrieval_match_terms_limit,
            )
            retrieval_source = self._retrieval_source_label(dense_rank, lexical_rank)
            preview_text = str(row.get("preview_text") or row.get("retrieval_text") or "")
            lowered_preview = preview_text.lower()
            title_text = normalize_whitespace(row.get("title") or "").lower()
            case_id_text = normalize_whitespace(row.get("case_id") or "").lower()
            exact_term_hits = [term for term in exact_terms if term and term in lowered_preview]
            remedy_hits = [term for term in remedy_terms if term and term in lowered_preview]
            retrieval_note = self._build_retrieval_note(
                retrieval_source=retrieval_source,
                matched_terms=matched_terms,
                dense_score=float(row_scores.get(int(row_id), 0.0)),
                lexical_score=lexical_score,
            )
            adjusted_score, domain_note = apply_domain_rerank(
                base_score=hybrid_base,
                query=query,
                case_id=row["case_id"],
                case_type=row.get("case_type"),
                title=row.get("title"),
                court=row.get("court"),
                text=row.get("preview_text") or row.get("retrieval_text"),
                case_type_hint=case_type_hint,
            )
            if domain_note:
                retrieval_note = f"{retrieval_note} | {domain_note}"
            alignment_score, alignment_note = candidate_domain_alignment(
                domain=domain_filter,
                case_id=row["case_id"],
                case_type=row.get("case_type"),
                title=row.get("title"),
                court=row.get("court"),
                text=row.get("preview_text") or row.get("retrieval_text"),
                legal_elements=legal_elements,
            )
            subtype_score, subtype_note, matched_subtypes, candidate_subtypes = issue_subtype_alignment(
                domain=domain_filter,
                issue_subtypes=issue_subtypes,
                case_id=row["case_id"],
                case_type=row.get("case_type"),
                title=row.get("title"),
                court=row.get("court"),
                text=row.get("preview_text") or row.get("retrieval_text"),
            )
            if (
                domain_filter
                and not direct_case_lookup
                and domain_confidence >= 0.72
                and alignment_score < 0.12
            ):
                continue
            adjusted_score = min(adjusted_score + min(alignment_score * 0.18, 0.18), 1.0)
            if issue_subtypes:
                if matched_subtypes:
                    adjusted_score = min(adjusted_score + min(subtype_score * 0.22, 0.22), 1.0)
                elif is_triage and candidate_subtypes:
                    adjusted_score = max(adjusted_score - 0.24, 0.0)
                elif is_triage and subtype_score < 0.1:
                    adjusted_score = max(adjusted_score - 0.12, 0.0)
            if lexical_score >= 0.22:
                adjusted_score = min(adjusted_score + min(lexical_score * 0.18, 0.16), 1.0)
            elif lexical_rank is None and dense_rank is not None and domain_filter:
                adjusted_score = max(adjusted_score - 0.05, 0.0)
            if exact_term_hits:
                adjusted_score = min(adjusted_score + min(0.08 * len(exact_term_hits), 0.2), 1.0)
                retrieval_note = f"{retrieval_note} | Exact legal terms: {', '.join(exact_term_hits[:3])}"
            elif exact_terms and any(len(term.split()) >= 2 for term in exact_terms):
                adjusted_score = max(adjusted_score - 0.05, 0.0)
            if remedy_hits:
                adjusted_score = min(adjusted_score + min(0.05 * len(remedy_hits), 0.12), 1.0)
                retrieval_note = f"{retrieval_note} | Remedy match: {', '.join(remedy_hits[:2])}"
            direct_reference_hit = case_id_text in referenced_case_ids or (title_text and title_text in lowered_query)
            if direct_reference_hit:
                adjusted_score = min(adjusted_score + 0.18, 1.0)
                retrieval_note = f"{retrieval_note} | Direct case reference match"
            elif direct_case_lookup:
                adjusted_score = max(adjusted_score - 0.12, 0.0)
            boilerplate_markers = sum(
                1
                for marker in ("uploaded on", "downloaded on", "page ", "https://", "www.")
                if marker in lowered_preview
            )
            if boilerplate_markers >= 2:
                adjusted_score = max(adjusted_score - 0.08, 0.0)
                retrieval_note = f"{retrieval_note} | Boilerplate-heavy preview"
            if "_1800_" in str(row.get("case_id") or "").lower() or str(row.get("date") or "").startswith("1800"):
                adjusted_score = max(adjusted_score - 0.05, 0.0)
                retrieval_note = f"{retrieval_note} | Placeholder year penalty"
            if alignment_note:
                retrieval_note = f"{retrieval_note} | Alignment: {alignment_note}"
            if subtype_note:
                retrieval_note = f"{retrieval_note} | {subtype_note}"
            exact_score = 1.0 if exact_term_hits else 0.0
            remedy_score = min(len(remedy_hits) / 2.0, 1.0)
            fit_score = min(
                (0.28 * alignment_score)
                + (0.42 * subtype_score)
                + (0.18 * lexical_score)
                + (0.08 * exact_score)
                + (0.04 * remedy_score),
                1.0,
            )
            if issue_subtypes and candidate_subtypes and not matched_subtypes:
                fit_score = min(fit_score, 0.22 if is_triage else 0.28)
            fit_band = self._fit_band(fit_score)
            fit_note = self._fit_note(
                fit_band=fit_band,
                subtype_note=subtype_note,
                alignment_note=alignment_note,
                expected_subtypes=issue_subtypes,
                candidate_subtypes=candidate_subtypes,
            )
            candidates.append(
                {
                    "row_id": int(row_id),
                    "case_id": row["case_id"],
                    "label": row["label"],
                    "title": row["title"],
                    "court": row["court"],
                    "case_type": row.get("case_type"),
                    "date": row["date"],
                    "retrieval_text": row["retrieval_text"],
                    "preview_text": row["preview_text"],
                    "base_similarity": adjusted_score,
                    "dense_similarity": float(row_scores.get(int(row_id), 0.0)),
                    "lexical_similarity": lexical_score,
                    "retrieval_source": retrieval_source,
                    "retrieval_note": retrieval_note,
                    "fit_band": fit_band,
                    "fit_note": fit_note,
                    "issue_subtypes": candidate_subtypes or matched_subtypes or issue_subtypes[:2],
                }
            )
        candidates.sort(key=lambda item: item["base_similarity"], reverse=True)
        return candidates

    def _lexical_search_row_ids(
        self,
        query: str,
        *,
        limit: int,
        metadata_filters: dict[str, str] | None,
    ) -> list[int]:
        metadata_path = self.settings.resolve_path(self.settings.retrieval_metadata_path)
        lexical_query = self._fts_query(query)
        if not lexical_query:
            return []

        with sqlite3.connect(metadata_path) as connection:
            if not self._fts_table_exists(connection, "retrieval_records_fts"):
                return []
            connection.row_factory = sqlite3.Row
            filters_sql: list[str] = []
            params: list[str | int] = [lexical_query]
            case_type_filter = normalize_whitespace((metadata_filters or {}).get("case_type")).lower()
            if case_type_filter:
                filters_sql.append("LOWER(COALESCE(r.case_type, '')) LIKE ?")
                params.append(f"%{case_type_filter}%")
            domain_filter = normalize_whitespace((metadata_filters or {}).get("domain")).lower()
            if domain_filter:
                hints = domain_filter_hints(domain_filter)
                domain_clauses: list[str] = []
                for hint in hints["case_type"]:
                    domain_clauses.append("LOWER(COALESCE(r.case_type, '')) LIKE ?")
                    params.append(f"%{hint}%")
                for hint in hints["case_id"]:
                    domain_clauses.append("LOWER(COALESCE(r.case_id, '')) LIKE ?")
                    params.append(f"%{hint}%")
                for hint in hints["text"]:
                    domain_clauses.append("LOWER(COALESCE(r.retrieval_text, '')) LIKE ?")
                    params.append(f"%{hint.lower()}%")
                if domain_clauses:
                    filters_sql.append("(" + " OR ".join(domain_clauses) + ")")
            forum_filter = normalize_whitespace((metadata_filters or {}).get("forum")).lower()
            if forum_filter:
                filters_sql.append("LOWER(COALESCE(r.court, '')) LIKE ?")
                params.append(f"%{forum_filter}%")
            where_tail = f" AND {' AND '.join(filters_sql)}" if filters_sql else ""
            query_sql = (
                "SELECT f.rowid FROM retrieval_records_fts f "
                "JOIN retrieval_records r ON r.row_id = f.rowid "
                "WHERE retrieval_records_fts MATCH ?"
                f"{where_tail} "
                "ORDER BY bm25(retrieval_records_fts) LIMIT ?"
            )
            params.append(limit)
            rows = connection.execute(query_sql, params).fetchall()
        return [int(row["rowid"]) for row in rows]

    def _fetch_rows(self, row_ids: list[int]) -> dict[int, dict]:
        if not row_ids:
            return {}

        metadata_path = self.settings.resolve_path(self.settings.retrieval_metadata_path)
        placeholders = ",".join("?" for _ in row_ids)
        query = (
            "SELECT row_id, case_id, label, title, court, case_type, date, retrieval_text, preview_text "
            f"FROM retrieval_records WHERE row_id IN ({placeholders})"
        )
        with sqlite3.connect(metadata_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(query, row_ids).fetchall()
        return {int(row["row_id"]): dict(row) for row in rows}

    @staticmethod
    def _matches_metadata_filters(row: dict, metadata_filters: dict[str, str] | None) -> bool:
        if not metadata_filters:
            return True
        case_type_filter = normalize_whitespace(metadata_filters.get("case_type") or "").lower()
        if case_type_filter:
            row_case_type = normalize_whitespace(row.get("case_type") or "").lower()
            if row_case_type and case_type_filter not in row_case_type:
                return False
        domain_filter = normalize_whitespace(metadata_filters.get("domain") or "").lower()
        if domain_filter and not candidate_matches_domain(
            domain=domain_filter,
            case_id=str(row.get("case_id") or ""),
            case_type=row.get("case_type"),
            title=row.get("title"),
            court=row.get("court"),
            text=row.get("preview_text") or row.get("retrieval_text"),
        ):
            return False
        forum_filter = normalize_whitespace(metadata_filters.get("forum") or "").lower()
        if forum_filter:
            row_court = normalize_whitespace(row.get("court") or "").lower()
            if row_court and forum_filter not in row_court:
                return False
        return True

    @staticmethod
    def _fts_table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
        row = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        ).fetchone()
        return row is not None

    @staticmethod
    def _fts_query(query: str) -> str:
        terms = []
        for token in search_terms(query):
            if len(token) >= 2 and token not in terms:
                terms.append(token)
            if len(terms) >= 8:
                break
        return " OR ".join(f'"{term}"' for term in terms)

    def _rrf_score(self, dense_rank: int | None, lexical_rank: int | None) -> float:
        k = max(int(self.settings.retrieval_rrf_k), 1)
        raw = 0.0
        if dense_rank is not None:
            raw += 1.0 / (k + dense_rank)
        if lexical_rank is not None:
            raw += 1.0 / (k + lexical_rank)
        max_possible = 2.0 / (k + 1)
        if max_possible <= 0:
            return 0.0
        return min(raw / max_possible, 1.0)

    @staticmethod
    def _retrieval_source_label(dense_rank: int | None, lexical_rank: int | None) -> str:
        if dense_rank is not None and lexical_rank is not None:
            return "dense+fts"
        if lexical_rank is not None:
            return "fts"
        return "dense"

    @staticmethod
    def _fit_band(score: float) -> str:
        if score >= 0.58:
            return "high"
        if score >= 0.3:
            return "moderate"
        return "low"

    @staticmethod
    def _fit_note(
        *,
        fit_band: str,
        subtype_note: str | None,
        alignment_note: str | None,
        expected_subtypes: list[str] | None = None,
        candidate_subtypes: list[str] | None = None,
    ) -> str:
        if subtype_note and subtype_note.startswith("Subtype drift:"):
            return subtype_note
        if fit_band == "low" and expected_subtypes and candidate_subtypes:
            expected = ", ".join(item.replace("_", " ") for item in expected_subtypes[:2])
            seen = ", ".join(item.replace("_", " ") for item in candidate_subtypes[:2])
            return f"Same broad domain, but this authority is about {seen} rather than {expected}."
        if fit_band == "high":
            return subtype_note or alignment_note or "High factual fit within the same issue family."
        if fit_band == "moderate":
            return subtype_note or alignment_note or "Broadly aligned but still needs factual verification."
        return subtype_note or "Same broad domain, but the issue subtype match is weak."

    @staticmethod
    def _build_retrieval_note(
        *,
        retrieval_source: str,
        matched_terms: list[str],
        dense_score: float,
        lexical_score: float,
    ) -> str:
        note_parts = [f"Source: {retrieval_source}"]
        if matched_terms:
            note_parts.append("Matched terms: " + ", ".join(matched_terms))
        if retrieval_source != "fts":
            note_parts.append(f"Semantic score: {dense_score:.3f}")
        if retrieval_source != "dense":
            note_parts.append(f"Lexical overlap: {lexical_score:.3f}")
        return " | ".join(note_parts)

    @staticmethod
    def _normalize_label_value(raw_label) -> int | None:
        if raw_label is None:
            return None
        if isinstance(raw_label, float) and math.isnan(raw_label):
            return None
        try:
            return int(raw_label)
        except (TypeError, ValueError):
            return None

    def _validate_loaded_artifacts(self, metadata_path: Path) -> None:
        if self.index is None:
            raise RuntimeError("Retrieval index is not loaded.")

        expected_dimension = int(self.model.get_sentence_embedding_dimension())
        if int(self.index.d) != expected_dimension:
            raise RuntimeError(
                "Retrieval index dimension does not match the configured embedding model. "
                "Rebuild the retrieval store with the current legal embedding model."
            )

        metadata = self._read_metadata_meta(metadata_path)
        if not metadata:
            raise RuntimeError(
                "Retrieval metadata is missing build information. "
                "Rebuild the retrieval store so the legal embedding configuration is recorded."
            )

        expected_model = normalize_whitespace(self.settings.shared_embedding_model_name)
        built_model = normalize_whitespace(metadata.get("embedding_model_name"))
        if built_model and built_model != expected_model:
            raise RuntimeError(
                "Retrieval artifacts were built with a different embedding model "
                f"({built_model}). Rebuild the retrieval store for {expected_model}."
            )

    @staticmethod
    def _read_metadata_meta(metadata_path: Path) -> dict[str, str]:
        with sqlite3.connect(metadata_path) as connection:
            table_exists = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='retrieval_meta'"
            ).fetchone()
            if table_exists is None:
                return {}
            rows = connection.execute("SELECT key, value FROM retrieval_meta").fetchall()
        return {str(key): str(value) for key, value in rows}

    def _write_metadata_database(self, metadata_path: Path, records: list[dict]) -> None:
        if metadata_path.exists():
            metadata_path.unlink()

        with sqlite3.connect(metadata_path) as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                CREATE TABLE retrieval_records (
                    row_id INTEGER PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    label INTEGER,
                    title TEXT,
                    court TEXT,
                    case_type TEXT,
                    date TEXT,
                    retrieval_text TEXT NOT NULL,
                    preview_text TEXT NOT NULL,
                    full_text TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE retrieval_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            fts_enabled = True
            try:
                cursor.execute(
                    """
                    CREATE VIRTUAL TABLE retrieval_records_fts
                    USING fts5(retrieval_text, content='retrieval_records', content_rowid='row_id')
                    """
                )
            except sqlite3.OperationalError:
                fts_enabled = False
            cursor.executemany(
                """
                INSERT INTO retrieval_records (
                    row_id,
                    case_id,
                    label,
                    title,
                    court,
                    case_type,
                    date,
                    retrieval_text,
                    preview_text,
                    full_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        idx,
                        record["case_id"],
                        self._normalize_label_value(record.get("label")),
                        record.get("title"),
                        record.get("court"),
                        record.get("case_type"),
                        record.get("date"),
                        record["retrieval_text"],
                        record["preview_text"],
                        record["full_text"],
                    )
                    for idx, record in enumerate(records)
                ],
            )
            if fts_enabled:
                cursor.execute(
                    """
                    INSERT INTO retrieval_records_fts(rowid, retrieval_text)
                    SELECT row_id, retrieval_text FROM retrieval_records
                    """
                )
            cursor.executemany(
                "INSERT INTO retrieval_meta (key, value) VALUES (?, ?)",
                [
                    ("embedding_model_name", normalize_whitespace(self.settings.shared_embedding_model_name)),
                    ("embedding_query_instruction", normalize_whitespace(self.settings.shared_embedding_query_prefix)),
                    ("embedding_document_instruction", normalize_whitespace(self.settings.shared_embedding_passage_prefix)),
                    ("embedding_dimension", str(int(self.model.get_sentence_embedding_dimension()))),
                    ("fts5_enabled", "1" if fts_enabled else "0"),
                ],
            )
            cursor.execute(
                "CREATE INDEX idx_retrieval_records_case_id ON retrieval_records(case_id)"
            )
            connection.commit()

    @staticmethod
    def _count_rows(metadata_path: Path) -> int:
        with sqlite3.connect(metadata_path) as connection:
            row = connection.execute("SELECT COUNT(*) FROM retrieval_records").fetchone()
        return int(row[0]) if row is not None else 0

    @staticmethod
    def build_records(df, char_limit: int, preview_char_limit: int) -> list[dict]:
        records = []
        for row in df.to_dict(orient="records"):
            full_text = normalize_whitespace(row.get("case_text") or row.get("text") or "")
            source_text = row.get("facts") or row.get("summary") or full_text
            cleaned = normalize_whitespace(source_text)
            derived_meta = derive_case_metadata(str(row.get("case_id") or row.get("filename") or ""))
            title = normalize_whitespace(row.get("title")) or derived_meta.get("title")
            court = normalize_whitespace(row.get("court")) or derived_meta.get("court")
            case_type = normalize_whitespace(row.get("case_type")) or derived_meta.get("case_type")
            retrieval_context = " ".join(
                part
                for part in [
                    title,
                    court,
                    case_type,
                    cleaned,
                ]
                if normalize_whitespace(part)
            )
            retrieval_text = compact_text(retrieval_context or cleaned or full_text, char_limit)
            preview_text = shorten_text(cleaned or full_text, preview_char_limit)
            stored_full_text = full_text or cleaned
            if not retrieval_text or not stored_full_text:
                continue
            records.append(
                {
                    "case_id": row["case_id"],
                    "label": SimilarCaseRetriever._normalize_label_value(row.get("label")),
                    "title": title or None,
                    "court": court or None,
                    "case_type": case_type or None,
                    "date": row.get("date") or derived_meta.get("year"),
                    "retrieval_text": retrieval_text,
                    "preview_text": preview_text,
                    "full_text": stored_full_text,
                }
            )
        return records
