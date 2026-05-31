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
    extract_case_ids_from_text,
)
from legal_ai.utils.text import (
    lexical_overlap_score,
    normalize_whitespace,
    overlapping_terms,
    search_terms,
    shorten_text,
    split_into_word_chunks,
)


LOGGER = logging.getLogger(__name__)


class LegalQARetriever:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        LOGGER.info(
            "Loading QA embedding model=%s on device=%s",
            settings.shared_embedding_model_name,
            self.device,
        )
        self.model = SentenceTransformer(settings.shared_embedding_model_name, device=self.device)
        self.index: faiss.Index | None = None
        self.embedding_store: np.ndarray | np.memmap | None = None
        self.record_count = 0
        self._case_chunk_embedding_cache: dict[str, dict[str, object]] = {}
        self._query_embedding_cache: dict[str, np.ndarray] = {}

    def encode_texts(
        self,
        texts: list[str],
        *,
        is_query: bool,
        batch_size: int | None = None,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        prepared = (
            [self._prepare_query_text(text) for text in texts]
            if is_query
            else [self._prepare_passage_text(text) for text in texts]
        )
        if not prepared:
            dimension = self.model.get_sentence_embedding_dimension()
            return np.empty((0, dimension), dtype="float32")

        embeddings = self.model.encode(
            prepared,
            batch_size=batch_size or self.settings.retrieval_embedding_batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.ascontiguousarray(embeddings.astype("float32"))

    def encode_query(self, text: str) -> np.ndarray:
        normalized = normalize_whitespace(text)
        cached = self._query_embedding_cache.get(normalized)
        if cached is not None:
            return np.asarray(cached, dtype="float32")
        query = self.encode_texts([normalized], is_query=True, batch_size=1, show_progress_bar=False)
        vector = np.asarray(query[0], dtype="float32")
        if len(self._query_embedding_cache) >= max(int(self.settings.qa_query_cache_size), 1):
            oldest_key = next(iter(self._query_embedding_cache))
            self._query_embedding_cache.pop(oldest_key, None)
        self._query_embedding_cache[normalized] = vector
        return vector

    def build(self, records: list[dict]) -> None:
        texts = [record["retrieval_text"] for record in records]
        embeddings = self.encode_texts(texts, is_query=False, show_progress_bar=True)
        if embeddings.shape[0] == 0:
            raise RuntimeError("No QA chunk records were generated from the dataset.")

        index = faiss.IndexFlatIP(int(embeddings.shape[1]))
        index.add(embeddings)
        self.index = index
        self.record_count = len(records)
        index_path = self.settings.resolve_path(self.settings.qa_retrieval_index_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        self._save_embedding_store(embeddings)
        self._write_metadata_database(
            self.settings.resolve_path(self.settings.qa_retrieval_metadata_path),
            records,
        )
        faiss.write_index(self.index, str(index_path))
        LOGGER.info("Built QA retrieval index chunks=%s dimension=%s", self.record_count, embeddings.shape[1])

    def load(self) -> None:
        index_path = self.settings.resolve_path(self.settings.qa_retrieval_index_path)
        metadata_path = self.settings.resolve_path(self.settings.qa_retrieval_metadata_path)
        embedding_store_path = self.settings.resolve_path(
            self.settings.qa_retrieval_embedding_store_path
        )
        if not metadata_path.exists():
            raise FileNotFoundError(
                "QA retrieval metadata is missing. Rebuild the retrieval store before using Ask."
            )

        if index_path.exists():
            self.index = faiss.read_index(str(index_path))
        else:
            self.index = None
        if embedding_store_path.exists():
            self.embedding_store = np.load(embedding_store_path, mmap_mode="r")
        else:
            self.embedding_store = None
        self._validate_loaded_artifacts(metadata_path)
        self.record_count = self._count_rows(metadata_path)
        LOGGER.info(
            "Loaded QA retrieval store path=%s chunks=%s global_index=%s embedding_store=%s",
            metadata_path,
            self.record_count,
            "present" if self.index is not None else "metadata-only",
            "present" if self.embedding_store is not None else "absent",
        )

    def search(
        self,
        query: str,
        *,
        top_k: int,
        case_ids: list[str] | None = None,
        metadata_filters: dict[str, str] | None = None,
        candidate_case_scores: dict[str, float] | None = None,
        query_profile: dict | None = None,
    ) -> list[dict]:
        if case_ids:
            return self._search_within_cases(
                query,
                case_ids=case_ids,
                top_k=top_k,
                metadata_filters=metadata_filters,
                candidate_case_scores=candidate_case_scores,
                query_profile=query_profile,
            )
        if self.index is None:
            raise RuntimeError(
                "Global QA chunk retrieval is not available. Use case-first hierarchical retrieval."
            )

        query_vector = self.encode_query(query)
        fetch_k = min(
            max(top_k * self.settings.qa_retrieval_overfetch, top_k),
            self.record_count,
        )
        if fetch_k == 0:
            return []

        scores, indices = self.index.search(query_vector.reshape(1, -1), fetch_k)
        dense_row_ids = [int(idx) for idx in indices[0].tolist() if idx >= 0]
        row_scores = {
            int(idx): float(scores[0][offset])
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
        chunk_rows = self._fetch_rows_by_row_ids(row_ids)
        filtered_rows = [
            row for row in chunk_rows if self._matches_metadata_filters(row, metadata_filters)
        ]
        return self._aggregate_best_chunks(
            filtered_rows,
            row_scores,
            query=query,
            top_k=top_k,
            dense_rank_map=dense_rank_map,
            lexical_rank_map=lexical_rank_map,
            query_profile=query_profile,
        )

    def _search_within_cases(
        self,
        query: str,
        *,
        case_ids: list[str],
        top_k: int,
        metadata_filters: dict[str, str] | None,
        candidate_case_scores: dict[str, float] | None,
        query_profile: dict | None,
    ) -> list[dict]:
        chunk_rows = self._fetch_rows_by_case_ids(case_ids)
        chunk_rows = [
            row for row in chunk_rows if self._matches_metadata_filters(row, metadata_filters)
        ]
        if not chunk_rows:
            return []

        chunk_rows = self._prefilter_case_rows(
            query,
            chunk_rows,
            top_k=top_k,
            query_profile=query_profile,
        )
        if not chunk_rows:
            return []

        query_vector = self.encode_query(query)
        chunk_embeddings = self._get_chunk_embeddings_for_rows(chunk_rows)
        scores = np.asarray(chunk_embeddings @ query_vector, dtype="float32")
        row_scores = {
            int(row["row_id"]): float(score)
            for row, score in zip(chunk_rows, scores.tolist(), strict=False)
        }
        dense_rank_map = {
            int(row["row_id"]): rank
            for rank, row in enumerate(
                sorted(chunk_rows, key=lambda item: row_scores.get(int(item["row_id"]), -1.0), reverse=True),
                start=1,
            )
        }
        lexical_rank_map = self._lexical_rank_map_within_rows(query, chunk_rows)
        return self._aggregate_best_chunks(
            chunk_rows,
            row_scores,
            query=query,
            top_k=top_k,
            dense_rank_map=dense_rank_map,
            lexical_rank_map=lexical_rank_map,
            candidate_case_scores=candidate_case_scores,
            query_profile=query_profile,
        )

    def _prefilter_case_rows(
        self,
        query: str,
        chunk_rows: list[dict],
        *,
        top_k: int,
        query_profile: dict | None = None,
    ) -> list[dict]:
        if not chunk_rows:
            return []

        rows_by_case: dict[str, list[dict]] = {}
        for row in chunk_rows:
            rows_by_case.setdefault(str(row["case_id"]), []).append(row)

        retrieval_profile = str((query_profile or {}).get("retrieval_profile") or "fast")
        per_case_limit = max(int(self.settings.qa_runtime_passages_per_case), 1)
        total_limit = max(int(self.settings.qa_runtime_passage_total_limit), top_k * per_case_limit)
        if retrieval_profile == "fast":
            per_case_limit = min(per_case_limit, 2)
            total_limit = min(total_limit, max(top_k * per_case_limit, 6))
        domain = normalize_whitespace((query_profile or {}).get("domain")).lower()
        domain_confidence = float((query_profile or {}).get("domain_confidence") or 0.0)
        legal_elements = list((query_profile or {}).get("legal_elements") or [])
        exact_terms = [normalize_whitespace(str(item)).lower() for item in (query_profile or {}).get("exact_terms") or []]
        remedy_terms = [normalize_whitespace(str(item)).lower() for item in (query_profile or {}).get("remedy_terms") or []]

        selected: list[tuple[float, int, dict]] = []
        for rows in rows_by_case.values():
            local_scores: list[tuple[float, int, dict]] = []
            for row in rows:
                row_text = row.get("chunk_text") or row.get("preview_text") or ""
                lowered_row = row_text.lower()
                alignment_score, _ = candidate_domain_alignment(
                    domain=domain,
                    case_id=str(row.get("case_id") or ""),
                    case_type=row.get("case_type"),
                    title=row.get("title"),
                    court=row.get("court"),
                    text=row_text,
                    legal_elements=legal_elements,
                )
                if domain and domain_confidence >= 0.72 and alignment_score < 0.1:
                    continue
                lexical = lexical_overlap_score(query, row_text)
                matched = len(
                    overlapping_terms(
                        query,
                        row_text,
                        limit=self.settings.retrieval_match_terms_limit,
                    )
                )
                section_label = self._infer_section_label(row_text)
                exact_hits = sum(1 for term in exact_terms if term and term in lowered_row)
                remedy_hits = sum(1 for term in remedy_terms if term and term in lowered_row)
                score = lexical + (0.05 * matched) + self._section_preference_boost(
                    section_label=section_label,
                    query_profile=query_profile,
                ) + self._element_overlap_boost(
                    row_text=row_text,
                    query_profile=query_profile,
                ) + min(alignment_score * 0.22, 0.22) + min(exact_hits * 0.07, 0.18) + min(remedy_hits * 0.05, 0.1)
                local_scores.append((score, int(row.get("chunk_order") or 0), row))
            if not local_scores:
                continue
            local_scores.sort(key=lambda item: (item[0], -item[1]), reverse=True)
            positive_rows = [entry for entry in local_scores if entry[0] > 0][:per_case_limit]
            if positive_rows:
                selected.extend(positive_rows)
            else:
                selected.extend(local_scores[:1])

        selected.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        reduced_rows = [row for _, _, row in selected[:total_limit]]
        if reduced_rows:
            return reduced_rows
        return chunk_rows[: min(len(chunk_rows), total_limit)]

    def _aggregate_best_chunks(
        self,
        chunk_rows: list[dict],
        row_scores: dict[int, float],
        *,
        query: str,
        top_k: int,
        dense_rank_map: dict[int, int],
        lexical_rank_map: dict[int, int],
        candidate_case_scores: dict[str, float] | None = None,
        query_profile: dict | None = None,
    ) -> list[dict]:
        best_by_case: dict[str, dict] = {}
        referenced_case_ids = extract_case_ids_from_text(query)
        domain = normalize_whitespace((query_profile or {}).get("domain")).lower()
        domain_confidence = float((query_profile or {}).get("domain_confidence") or 0.0)
        legal_elements = list((query_profile or {}).get("legal_elements") or [])
        exact_terms = [normalize_whitespace(str(item)).lower() for item in (query_profile or {}).get("exact_terms") or []]
        remedy_terms = [normalize_whitespace(str(item)).lower() for item in (query_profile or {}).get("remedy_terms") or []]
        for row in chunk_rows:
            dense_score = float(row_scores.get(int(row["row_id"]), -1.0))
            row_text = row.get("chunk_text") or row.get("preview_text") or ""
            lowered_row_text = str(row_text).lower()
            lexical_score = lexical_overlap_score(query, row_text)
            dense_rank = dense_rank_map.get(int(row["row_id"]))
            lexical_rank = lexical_rank_map.get(int(row["row_id"]))
            score = self._rrf_score(dense_rank, lexical_rank)
            matched_terms = overlapping_terms(
                query,
                row_text,
                limit=self.settings.retrieval_match_terms_limit,
            )
            retrieval_source = self._retrieval_source_label(dense_rank, lexical_rank)
            case_id = row["case_id"]
            case_score = float((candidate_case_scores or {}).get(case_id, 0.0))
            section_label = self._infer_section_label(row_text)
            support_type = self._infer_support_type(
                query=query,
                row_text=row_text,
                case_id=case_id,
                query_profile=query_profile,
            )
            proposition = self._infer_proposition(
                row_text=row_text,
                section_label=section_label,
                query_profile=query_profile,
            )
            adjusted_score, domain_note = apply_domain_rerank(
                base_score=score,
                query=query,
                case_id=case_id,
                case_type=row.get("case_type"),
                title=row.get("title"),
                court=row.get("court"),
                text=row_text,
                referenced_case_ids=referenced_case_ids,
            )
            exact_term_hits = [term for term in exact_terms if term and term in lowered_row_text]
            remedy_hits = [term for term in remedy_terms if term and term in lowered_row_text]
            retrieval_note = self._build_retrieval_note(
                retrieval_source=retrieval_source,
                matched_terms=matched_terms,
                dense_score=dense_score,
                lexical_score=lexical_score,
            )
            if domain_note:
                retrieval_note = f"{retrieval_note} | {domain_note}"
            alignment_score, alignment_note = candidate_domain_alignment(
                domain=domain,
                case_id=case_id,
                case_type=row.get("case_type"),
                title=row.get("title"),
                court=row.get("court"),
                text=row_text,
                legal_elements=legal_elements,
            )
            if domain and domain_confidence >= 0.72 and alignment_score < 0.1:
                continue
            adjusted_score = min(adjusted_score + min(alignment_score * 0.22, 0.22), 1.0)
            if lexical_score >= 0.2:
                adjusted_score = min(adjusted_score + min(lexical_score * 0.16, 0.14), 1.0)
            if exact_term_hits:
                adjusted_score = min(adjusted_score + min(0.09 * len(exact_term_hits), 0.22), 1.0)
                retrieval_note = f"{retrieval_note} | Exact legal terms: {', '.join(exact_term_hits[:3])}"
            elif exact_terms and any(len(term.split()) >= 2 for term in exact_terms):
                adjusted_score = max(adjusted_score - 0.04, 0.0)
            if remedy_hits:
                adjusted_score = min(adjusted_score + min(0.06 * len(remedy_hits), 0.14), 1.0)
                retrieval_note = f"{retrieval_note} | Remedy match: {', '.join(remedy_hits[:2])}"
            boilerplate_markers = sum(
                1
                for marker in ("uploaded on", "downloaded on", "page ", "https://", "www.")
                if marker in lowered_row_text
            )
            if boilerplate_markers >= 2:
                adjusted_score = max(adjusted_score - 0.08, 0.0)
                retrieval_note = f"{retrieval_note} | Boilerplate-heavy chunk"
            if alignment_note:
                retrieval_note = f"{retrieval_note} | Alignment: {alignment_note}"
            if candidate_case_scores is not None:
                adjusted_score = self._blend_case_and_chunk_scores(
                    case_score=case_score,
                    chunk_score=adjusted_score,
                )
                retrieval_note = f"{retrieval_note} | Case shortlist score: {case_score:.3f}"
            current = best_by_case.get(case_id)
            if current is not None and adjusted_score <= current["similarity"]:
                continue
            best_by_case[case_id] = {
                "case_id": case_id,
                "similarity": round(adjusted_score, 4),
                "base_similarity": round(case_score if candidate_case_scores is not None else dense_score, 4),
                "evidence_similarity": round(score, 4),
                "label": row.get("label"),
                "label_name": LABEL_ID_TO_NAME.get(row.get("label"))
                if row.get("label") is not None
                else None,
                "title": row.get("title"),
                "court": row.get("court"),
                "case_type": row.get("case_type"),
                "date": row.get("date"),
                "matched_chunk_index": row.get("chunk_order"),
                "matched_chunk_count": row.get("chunk_count") or 0,
                "retrieval_strategy": retrieval_source,
                "retrieval_note": retrieval_note,
                "summary": row.get("preview_text"),
                "excerpt": row.get("preview_text") or shorten_text(row.get("chunk_text") or "", 280),
                "section_label": section_label,
                "support_type": support_type,
                "authority_level": self._infer_authority_level(case_id=case_id, court=row.get("court")),
                "proposition": proposition,
            }

        results = sorted(best_by_case.values(), key=lambda item: item["similarity"], reverse=True)
        return results[:top_k]

    def _get_chunk_embeddings_for_rows(self, chunk_rows: list[dict]) -> np.ndarray:
        if self.embedding_store is not None:
            row_ids = np.asarray([int(row["row_id"]) for row in chunk_rows], dtype=np.int64)
            vectors = np.asarray(self.embedding_store[row_ids], dtype="float32")
            return np.ascontiguousarray(vectors)

        rows_by_case: dict[str, list[dict]] = {}
        for row in chunk_rows:
            rows_by_case.setdefault(str(row["case_id"]), []).append(row)

        embeddings_by_case: dict[str, np.ndarray] = {}
        missing_case_order: list[str] = []
        missing_texts: list[str] = []
        missing_sizes: list[int] = []

        for case_id, rows in rows_by_case.items():
            signature = self._case_cache_signature(rows)
            cached = self._case_chunk_embedding_cache.get(case_id)
            if cached and cached.get("signature") == signature:
                embeddings_by_case[case_id] = np.asarray(cached["embeddings"], dtype="float32")
                continue
            missing_case_order.append(case_id)
            missing_sizes.append(len(rows))
            missing_texts.extend(row["chunk_text"] for row in rows)

        if missing_texts:
            encoded = self.encode_texts(
                missing_texts,
                is_query=False,
                show_progress_bar=False,
            )
            offset = 0
            for case_id, size in zip(missing_case_order, missing_sizes, strict=False):
                case_embeddings = np.ascontiguousarray(encoded[offset : offset + size], dtype="float32")
                offset += size
                embeddings_by_case[case_id] = case_embeddings
                self._store_case_chunk_embeddings(case_id, rows_by_case[case_id], case_embeddings)

        row_index_by_case: dict[str, int] = {}
        vectors: list[np.ndarray] = []
        for row in chunk_rows:
            case_id = str(row["case_id"])
            position = row_index_by_case.get(case_id, 0)
            vectors.append(embeddings_by_case[case_id][position])
            row_index_by_case[case_id] = position + 1
        return np.ascontiguousarray(np.vstack(vectors).astype("float32"))

    def _store_case_chunk_embeddings(
        self,
        case_id: str,
        rows: list[dict],
        embeddings: np.ndarray,
    ) -> None:
        if len(self._case_chunk_embedding_cache) >= 192:
            oldest_key = next(iter(self._case_chunk_embedding_cache))
            self._case_chunk_embedding_cache.pop(oldest_key, None)
        self._case_chunk_embedding_cache[case_id] = {
            "signature": self._case_cache_signature(rows),
            "embeddings": np.ascontiguousarray(embeddings.astype("float32")),
        }

    @staticmethod
    def _case_cache_signature(rows: list[dict]) -> tuple[int, int, int]:
        first_row = int(rows[0]["row_id"])
        last_row = int(rows[-1]["row_id"])
        return (len(rows), first_row, last_row)

    @staticmethod
    def _lexical_rank_map_within_rows(query: str, chunk_rows: list[dict]) -> dict[int, int]:
        lexical_scores = {
            int(row["row_id"]): lexical_overlap_score(query, row.get("chunk_text") or row.get("preview_text") or "")
            for row in chunk_rows
        }
        ordered = sorted(
            lexical_scores,
            key=lambda row_id: lexical_scores[row_id],
            reverse=True,
        )
        return {
            row_id: rank
            for rank, row_id in enumerate(ordered, start=1)
            if lexical_scores[row_id] > 0
        }

    @staticmethod
    def _blend_case_and_chunk_scores(*, case_score: float, chunk_score: float) -> float:
        return (0.42 * case_score) + (0.58 * chunk_score)

    def _save_embedding_store(self, embeddings: np.ndarray) -> None:
        store_path = self.settings.resolve_path(self.settings.qa_retrieval_embedding_store_path)
        store_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(store_path, np.ascontiguousarray(embeddings.astype("float32")), allow_pickle=False)

    def _fetch_rows_by_row_ids(self, row_ids: list[int]) -> list[dict]:
        if not row_ids:
            return []
        metadata_path = self.settings.resolve_path(self.settings.qa_retrieval_metadata_path)
        placeholders = ",".join("?" for _ in row_ids)
        query = (
            "SELECT row_id, case_id, label, title, court, case_type, date, chunk_order, chunk_count, "
            "preview_text, chunk_text FROM qa_chunk_records "
            f"WHERE row_id IN ({placeholders})"
        )
        with sqlite3.connect(metadata_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(query, row_ids).fetchall()
        row_map = {int(row["row_id"]): dict(row) for row in rows}
        return [row_map[row_id] for row_id in row_ids if row_id in row_map]

    def _fetch_rows_by_case_ids(self, case_ids: list[str]) -> list[dict]:
        if not case_ids:
            return []
        metadata_path = self.settings.resolve_path(self.settings.qa_retrieval_metadata_path)
        placeholders = ",".join("?" for _ in case_ids)
        query = (
            "SELECT row_id, case_id, label, title, court, case_type, date, chunk_order, chunk_count, "
            "preview_text, chunk_text FROM qa_chunk_records "
            f"WHERE case_id IN ({placeholders}) ORDER BY case_id, chunk_order"
        )
        with sqlite3.connect(metadata_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(query, case_ids).fetchall()
        return [dict(row) for row in rows]

    def _lexical_search_row_ids(
        self,
        query: str,
        *,
        limit: int,
        metadata_filters: dict[str, str] | None,
    ) -> list[int]:
        metadata_path = self.settings.resolve_path(self.settings.qa_retrieval_metadata_path)
        lexical_query = self._fts_query(query)
        if not lexical_query:
            return []

        with sqlite3.connect(metadata_path) as connection:
            if not self._fts_table_exists(connection, "qa_chunk_records_fts"):
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
                "SELECT f.rowid FROM qa_chunk_records_fts f "
                "JOIN qa_chunk_records r ON r.row_id = f.rowid "
                "WHERE qa_chunk_records_fts MATCH ?"
                f"{where_tail} "
                "ORDER BY bm25(qa_chunk_records_fts) LIMIT ?"
            )
            params.append(limit)
            rows = connection.execute(query_sql, params).fetchall()
        return [int(row["rowid"]) for row in rows]

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
            text=row.get("chunk_text") or row.get("preview_text"),
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
        k = max(int(self.settings.qa_retrieval_rrf_k), 1)
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
    def _infer_section_label(text: str) -> str:
        lowered = normalize_whitespace(text).lower()
        if any(token in lowered for token in ("award", "awarded", "relief", "refund", "replacement", "compensation", "directed")):
            return "relief"
        if any(token in lowered for token in ("held", "therefore", "we find", "it is ordered")):
            return "holding"
        if any(token in lowered for token in ("question", "issue", "whether", "point for consideration")):
            return "issue"
        if any(token in lowered for token in ("purchased", "accident", "complainant", "assessee", "student", "facts")):
            return "facts"
        return "reasoning"

    @staticmethod
    def _section_preference_boost(*, section_label: str, query_profile: dict | None) -> float:
        preferred_sections = set((query_profile or {}).get("preferred_sections") or [])
        if section_label in preferred_sections:
            return 0.12
        return 0.0

    @staticmethod
    def _element_overlap_boost(*, row_text: str, query_profile: dict | None) -> float:
        legal_elements = [str(item).replace("_", " ") for item in (query_profile or {}).get("legal_elements") or []]
        lowered = normalize_whitespace(row_text).lower()
        matched = sum(1 for element in legal_elements if element and element in lowered)
        return min(matched * 0.03, 0.12)

    @staticmethod
    def _infer_support_type(
        *,
        query: str,
        row_text: str,
        case_id: str,
        query_profile: dict | None,
    ) -> str:
        referenced_case_ids = set((query_profile or {}).get("referenced_case_ids") or [])
        if case_id in referenced_case_ids:
            return "direct"
        lexical = lexical_overlap_score(query, row_text)
        matched_terms = overlapping_terms(query, row_text, limit=5)
        if lexical >= 0.28 or len(matched_terms) >= 3:
            return "direct"
        if lexical >= 0.14 or len(matched_terms) >= 2:
            return "supportive"
        return "analogical"

    @staticmethod
    def _infer_proposition(*, row_text: str, section_label: str, query_profile: dict | None) -> str:
        lowered = normalize_whitespace(row_text).lower()
        domain = (query_profile or {}).get("domain")
        task = (query_profile or {}).get("task")
        if domain == "consumer":
            if any(token in lowered for token in ("refund", "replacement", "repair")):
                return "consumer remedy discretion"
            if "deficiency" in lowered:
                return "deficiency in service reasoning"
        if domain == "motor_accident":
            if any(token in lowered for token in ("disability", "compensation", "future treatment", "earning")):
                return "motor accident compensation factors"
        if domain == "education":
            if any(token in lowered for token in ("hearing", "notice", "unfair means", "exam")):
                return "procedural fairness in education disputes"
        if domain == "tax":
            if any(token in lowered for token in ("documents", "invoice", "bank", "ledger", "addition")):
                return "documentary sufficiency in tax disputes"
        if task == "similarity_lookup":
            return "fact-pattern similarity support"
        if task == "case_explanation":
            return f"{section_label} of the requested case"
        return f"{section_label}-focused legal support"

    @staticmethod
    def _infer_authority_level(*, case_id: str, court: str | None) -> str:
        lowered_case_id = normalize_whitespace(case_id).lower()
        lowered_court = normalize_whitespace(court).lower()
        if "supremecourt" in lowered_case_id or "supreme court" in lowered_court:
            return "supreme_court"
        if "_hc_" in lowered_case_id or "high court" in lowered_court or lowered_case_id.endswith("_hc"):
            return "high_court"
        if any(token in lowered_case_id for token in ("tribunal", "commission", "consumer_disputes")):
            return "tribunal_or_forum"
        return "other"

    def _prepare_query_text(self, text: str) -> str:
        prefix = normalize_whitespace(self.settings.shared_embedding_query_prefix)
        cleaned = normalize_whitespace(text)
        return f"{prefix} {cleaned}".strip()

    def _prepare_passage_text(self, text: str) -> str:
        prefix = normalize_whitespace(self.settings.shared_embedding_passage_prefix)
        cleaned = normalize_whitespace(text)
        return f"{prefix} {cleaned}".strip()

    def _validate_loaded_artifacts(self, metadata_path: Path) -> None:
        expected_dimension = int(self.model.get_sentence_embedding_dimension())
        if self.index is not None and int(self.index.d) != expected_dimension:
            raise RuntimeError(
                "QA retrieval index dimension does not match the configured legal embedding model. "
                "Rebuild the retrieval store before using Ask."
            )
        if self.embedding_store is not None:
            if len(self.embedding_store.shape) != 2 or int(self.embedding_store.shape[1]) != expected_dimension:
                raise RuntimeError(
                    "QA chunk embedding store dimension does not match the configured legal embedding model. "
                    "Rebuild the retrieval store before using Ask."
                )

        metadata = self._read_metadata_meta(metadata_path)
        if not metadata:
            raise RuntimeError(
                "QA retrieval metadata is missing build information. "
                "Rebuild the retrieval store before using Ask."
            )
        built_model = normalize_whitespace(metadata.get("qa_embedding_model_name"))
        expected_model = normalize_whitespace(self.settings.shared_embedding_model_name)
        if built_model != expected_model:
            raise RuntimeError(
                "QA retrieval artifacts were built with a different embedding model "
                f"({built_model}). Rebuild the retrieval store for {expected_model}."
            )
        expected_count = metadata.get("record_count")
        if self.embedding_store is not None and expected_count:
            if int(self.embedding_store.shape[0]) != int(expected_count):
                raise RuntimeError(
                    "QA chunk embedding store row count does not match QA metadata. "
                    "Rebuild the retrieval store before using Ask."
                )
        if self.index is None and self.embedding_store is None:
            LOGGER.warning(
                "QA retrieval artifacts do not include a global chunk index or disk-backed chunk embeddings; "
                "case-scoped retrieval will fall back to on-the-fly encoding."
            )

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

    @staticmethod
    def _read_metadata_meta(metadata_path: Path) -> dict[str, str]:
        with sqlite3.connect(metadata_path) as connection:
            table_exists = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='qa_retrieval_meta'"
            ).fetchone()
            if table_exists is None:
                return {}
            rows = connection.execute("SELECT key, value FROM qa_retrieval_meta").fetchall()
        return {str(key): str(value) for key, value in rows}

    def _write_metadata_database(self, metadata_path: Path, records: list[dict]) -> None:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        if metadata_path.exists():
            metadata_path.unlink()

        with sqlite3.connect(metadata_path) as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                CREATE TABLE qa_chunk_records (
                    row_id INTEGER PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    label INTEGER,
                    title TEXT,
                    court TEXT,
                    case_type TEXT,
                    date TEXT,
                    chunk_order INTEGER NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    retrieval_text TEXT NOT NULL,
                    preview_text TEXT NOT NULL,
                    chunk_text TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE qa_retrieval_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            fts_enabled = True
            try:
                cursor.execute(
                    """
                    CREATE VIRTUAL TABLE qa_chunk_records_fts
                    USING fts5(retrieval_text, content='qa_chunk_records', content_rowid='row_id')
                    """
                )
            except sqlite3.OperationalError:
                fts_enabled = False
            cursor.executemany(
                """
                INSERT INTO qa_chunk_records (
                    row_id, case_id, label, title, court, date, chunk_order, chunk_count,
                    retrieval_text, preview_text, chunk_text, case_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        idx,
                        record["case_id"],
                        self._normalize_label_value(record.get("label")),
                        record.get("title"),
                        record.get("court"),
                        record.get("date"),
                        record["chunk_order"],
                        record["chunk_count"],
                        record["retrieval_text"],
                        record["preview_text"],
                        record["chunk_text"],
                        record.get("case_type"),
                    )
                    for idx, record in enumerate(records)
                ],
            )
            if fts_enabled:
                cursor.execute(
                    """
                    INSERT INTO qa_chunk_records_fts(rowid, retrieval_text)
                    SELECT row_id, retrieval_text FROM qa_chunk_records
                    """
                )
            cursor.executemany(
                "INSERT INTO qa_retrieval_meta (key, value) VALUES (?, ?)",
                [
                    ("qa_embedding_model_name", normalize_whitespace(self.settings.shared_embedding_model_name)),
                    ("qa_embedding_query_prefix", normalize_whitespace(self.settings.shared_embedding_query_prefix)),
                    ("qa_embedding_passage_prefix", normalize_whitespace(self.settings.shared_embedding_passage_prefix)),
                    ("qa_embedding_dimension", str(int(self.model.get_sentence_embedding_dimension()))),
                    (
                        "qa_embedding_store_path",
                        str(self.settings.resolve_path(self.settings.qa_retrieval_embedding_store_path)),
                    ),
                    ("fts5_enabled", "1" if fts_enabled else "0"),
                ],
            )
            cursor.execute("CREATE INDEX idx_qa_chunk_records_case_id ON qa_chunk_records(case_id)")
            connection.commit()

    @staticmethod
    def _count_rows(metadata_path: Path) -> int:
        with sqlite3.connect(metadata_path) as connection:
            row = connection.execute("SELECT COUNT(*) FROM qa_chunk_records").fetchone()
        return int(row[0]) if row is not None else 0

    @staticmethod
    def build_records(df, settings: Settings) -> list[dict]:
        records: list[dict] = []
        for row in df.to_dict(orient="records"):
            full_text = normalize_whitespace(row.get("case_text") or row.get("text") or "")
            if not full_text:
                continue
            derived_meta = derive_case_metadata(str(row.get("case_id") or row.get("filename") or ""))
            chunks = split_into_word_chunks(
                full_text,
                chunk_words=settings.qa_chunk_words,
                overlap_words=settings.qa_chunk_overlap_words,
                min_words=settings.qa_chunk_min_words,
            )
            if not chunks:
                chunks = [full_text]

            chunk_count = len(chunks)
            title = normalize_whitespace(row.get("title")) or derived_meta.get("title")
            court = normalize_whitespace(row.get("court")) or derived_meta.get("court")
            case_type = normalize_whitespace(row.get("case_type")) or derived_meta.get("case_type")
            for chunk_order, chunk_text in enumerate(chunks):
                retrieval_context = " ".join(
                    part for part in [title, court, case_type, chunk_text] if part
                )
                records.append(
                    {
                        "case_id": str(row["case_id"]),
                        "label": LegalQARetriever._normalize_label_value(row.get("label")),
                        "title": title or None,
                        "court": court or None,
                        "case_type": case_type or None,
                        "date": row.get("date") or derived_meta.get("year"),
                        "chunk_order": chunk_order,
                        "chunk_count": chunk_count,
                        "retrieval_text": retrieval_context,
                        "preview_text": shorten_text(
                            chunk_text,
                            settings.qa_retrieval_preview_char_limit,
                        ),
                        "chunk_text": chunk_text,
                    }
                )
        return records
