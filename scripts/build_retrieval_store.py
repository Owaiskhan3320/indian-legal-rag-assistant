from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from pathlib import Path
import sys
from typing import Any

import faiss
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from legal_ai.config import get_settings  # noqa: E402
from legal_ai.logging_utils import configure_logging  # noqa: E402
from legal_ai.services.qa_retriever import LegalQARetriever  # noqa: E402
from legal_ai.services.retriever import SimilarCaseRetriever  # noqa: E402
from legal_ai.utils.data import iter_dataset_chunks  # noqa: E402
from legal_ai.utils.text import normalize_whitespace  # noqa: E402


STATE_VERSION = 1


def _atomic_save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(path)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("wb") as handle:
        np.save(handle, np.ascontiguousarray(array.astype("float32")), allow_pickle=False)
    temp_path.replace(path)


def _chunk_shard_path(directory: Path, prefix: str, chunk_id: int) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{prefix}_{chunk_id:06d}.npy"


def _fresh_cleanup(settings) -> None:
    for relative_path in [
        settings.retrieval_index_path,
        settings.retrieval_metadata_path,
        settings.qa_retrieval_index_path,
        settings.qa_retrieval_metadata_path,
        settings.qa_retrieval_embedding_store_path,
        settings.retrieval_build_manifest_path,
        settings.retrieval_build_state_path,
    ]:
        target = settings.resolve_path(relative_path)
        if target.exists():
            target.unlink()
    work_dir = settings.resolve_path(settings.retrieval_build_work_dir)
    if work_dir.exists():
        shutil.rmtree(work_dir)


def _build_signature(settings, *, input_path: str, max_rows: int | None, chunksize: int, build_global_qa_index: bool) -> dict[str, Any]:
    return {
        "state_version": STATE_VERSION,
        "input_path": str(input_path),
        "max_rows": max_rows,
        "chunksize": int(chunksize),
        "build_global_qa_index": bool(build_global_qa_index),
        "embedding_model": settings.shared_embedding_model_name,
        "embedding_query_prefix": settings.shared_embedding_query_prefix,
        "embedding_passage_prefix": settings.shared_embedding_passage_prefix,
        "retrieval_chunk_words": int(settings.retrieval_chunk_words),
        "retrieval_chunk_overlap_words": int(settings.retrieval_chunk_overlap_words),
        "retrieval_chunk_min_words": int(settings.retrieval_chunk_min_words),
        "qa_chunk_words": int(settings.qa_chunk_words),
        "qa_chunk_overlap_words": int(settings.qa_chunk_overlap_words),
        "qa_chunk_min_words": int(settings.qa_chunk_min_words),
        "qa_case_shortlist_top_n": int(settings.qa_case_shortlist_top_n),
    }


def _initial_state(signature: dict[str, Any]) -> dict[str, Any]:
    return {
        **signature,
        "phase1_complete": False,
        "phase2_complete": False,
        "case_index_finalized": False,
        "qa_store_finalized": False,
        "total_case_records": 0,
        "total_qa_records": 0,
        "case_progress_chunks": 0,
        "qa_metadata_progress_chunks": 0,
        "qa_embedding_progress_chunks": 0,
    }


def _ensure_state(settings, *, signature: dict[str, Any], fresh: bool) -> dict[str, Any]:
    state_path = settings.resolve_path(settings.retrieval_build_state_path)
    if fresh:
        _fresh_cleanup(settings)
        state = _initial_state(signature)
        _atomic_save_json(state_path, state)
        return state

    existing = _load_json(state_path)
    if existing is None:
        state = _initial_state(signature)
        _atomic_save_json(state_path, state)
        return state

    for key, value in signature.items():
        if existing.get(key) != value:
            raise RuntimeError(
                "Existing build checkpoint is incompatible with the current build settings. "
                "Use --fresh to start over."
            )
    return existing


def _save_state(settings, state: dict[str, Any]) -> None:
    _atomic_save_json(settings.resolve_path(settings.retrieval_build_state_path), state)


class CaseMetadataWriter:
    def __init__(self, metadata_path: Path, settings, *, resume: bool) -> None:
        self.metadata_path = metadata_path
        self.settings = settings
        self.resume = resume
        self.connection: sqlite3.Connection | None = None
        self.cursor: sqlite3.Cursor | None = None
        self.row_id = 0
        self.fts_enabled = True
        self.completed_chunks = 0

    def __enter__(self) -> "CaseMetadataWriter":
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        if self.resume and self.metadata_path.exists():
            self.connection = sqlite3.connect(self.metadata_path)
            self.cursor = self.connection.cursor()
            self._load_existing_state()
            return self

        if self.metadata_path.exists():
            self.metadata_path.unlink()
        self.connection = sqlite3.connect(self.metadata_path)
        self.cursor = self.connection.cursor()
        self.cursor.execute(
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
        self.cursor.execute(
            """
            CREATE TABLE retrieval_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        self.cursor.execute(
            """
            CREATE TABLE retrieval_build_progress (
                chunk_id INTEGER PRIMARY KEY,
                record_count INTEGER NOT NULL
            )
            """
        )
        try:
            self.cursor.execute(
                """
                CREATE VIRTUAL TABLE retrieval_records_fts
                USING fts5(retrieval_text, content='retrieval_records', content_rowid='row_id')
                """
            )
        except sqlite3.OperationalError:
            self.fts_enabled = False
        self.connection.commit()
        return self

    def _load_existing_state(self) -> None:
        assert self.connection is not None
        row = self.connection.execute("SELECT COALESCE(MAX(row_id), -1) FROM retrieval_records").fetchone()
        self.row_id = int(row[0]) + 1 if row is not None else 0
        progress_row = self.connection.execute(
            "SELECT COALESCE(MAX(chunk_id), 0) FROM retrieval_build_progress"
        ).fetchone()
        self.completed_chunks = int(progress_row[0]) if progress_row is not None else 0
        table_exists = self.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='retrieval_records_fts'"
        ).fetchone()
        self.fts_enabled = table_exists is not None

    def is_chunk_complete(self, chunk_id: int) -> bool:
        assert self.connection is not None
        row = self.connection.execute(
            "SELECT 1 FROM retrieval_build_progress WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        return row is not None

    def append(self, *, records: list[dict], chunk_id: int) -> int:
        assert self.cursor is not None
        rows = []
        fts_rows = []
        for offset, record in enumerate(records):
            row_id = self.row_id + offset
            rows.append(
                (
                    row_id,
                    record["case_id"],
                    record.get("label"),
                    record.get("title"),
                    record.get("court"),
                    record.get("case_type"),
                    record.get("date"),
                    record["retrieval_text"],
                    record["preview_text"],
                    record["full_text"],
                )
            )
            if self.fts_enabled:
                fts_rows.append((row_id, record["retrieval_text"]))
        if rows:
            self.cursor.executemany(
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
                rows,
            )
            if self.fts_enabled:
                self.cursor.executemany(
                    "INSERT INTO retrieval_records_fts(rowid, retrieval_text) VALUES (?, ?)",
                    fts_rows,
                )
            self.row_id += len(records)
        self.cursor.execute(
            "INSERT INTO retrieval_build_progress (chunk_id, record_count) VALUES (?, ?)",
            (chunk_id, len(records)),
        )
        self.connection.commit()
        self.completed_chunks = max(self.completed_chunks, chunk_id)
        return len(records)

    def finalize(self, embedding_dimension: int) -> None:
        assert self.cursor is not None
        self.cursor.executemany(
            "INSERT OR REPLACE INTO retrieval_meta (key, value) VALUES (?, ?)",
            [
                ("embedding_model_name", normalize_whitespace(self.settings.shared_embedding_model_name)),
                ("embedding_query_instruction", normalize_whitespace(self.settings.shared_embedding_query_prefix)),
                ("embedding_document_instruction", normalize_whitespace(self.settings.shared_embedding_passage_prefix)),
                ("embedding_dimension", str(int(embedding_dimension))),
                ("fts5_enabled", "1" if self.fts_enabled else "0"),
                ("record_count", str(int(self.row_id))),
                ("retrieval_char_limit", str(int(self.settings.retrieval_char_limit))),
                ("retrieval_preview_char_limit", str(int(self.settings.retrieval_preview_char_limit))),
            ],
        )
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_retrieval_records_case_id ON retrieval_records(case_id)")
        self.connection.commit()

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.connection is not None:
            self.connection.close()


class QAChunkMetadataWriter:
    def __init__(self, metadata_path: Path, settings, *, resume: bool) -> None:
        self.metadata_path = metadata_path
        self.settings = settings
        self.resume = resume
        self.connection: sqlite3.Connection | None = None
        self.cursor: sqlite3.Cursor | None = None
        self.row_id = 0
        self.fts_enabled = True
        self.completed_chunks = 0

    def __enter__(self) -> "QAChunkMetadataWriter":
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        if self.resume and self.metadata_path.exists():
            self.connection = sqlite3.connect(self.metadata_path)
            self.cursor = self.connection.cursor()
            self._load_existing_state()
            return self

        if self.metadata_path.exists():
            self.metadata_path.unlink()
        self.connection = sqlite3.connect(self.metadata_path)
        self.cursor = self.connection.cursor()
        self.cursor.execute(
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
        self.cursor.execute(
            """
            CREATE TABLE qa_retrieval_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        self.cursor.execute(
            """
            CREATE TABLE qa_build_progress (
                chunk_id INTEGER PRIMARY KEY,
                record_count INTEGER NOT NULL
            )
            """
        )
        try:
            self.cursor.execute(
                """
                CREATE VIRTUAL TABLE qa_chunk_records_fts
                USING fts5(retrieval_text, content='qa_chunk_records', content_rowid='row_id')
                """
            )
        except sqlite3.OperationalError:
            self.fts_enabled = False
        self.connection.commit()
        return self

    def _load_existing_state(self) -> None:
        assert self.connection is not None
        row = self.connection.execute("SELECT COALESCE(MAX(row_id), -1) FROM qa_chunk_records").fetchone()
        self.row_id = int(row[0]) + 1 if row is not None else 0
        progress_row = self.connection.execute(
            "SELECT COALESCE(MAX(chunk_id), 0) FROM qa_build_progress"
        ).fetchone()
        self.completed_chunks = int(progress_row[0]) if progress_row is not None else 0
        table_exists = self.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='qa_chunk_records_fts'"
        ).fetchone()
        self.fts_enabled = table_exists is not None

    def is_chunk_complete(self, chunk_id: int) -> bool:
        assert self.connection is not None
        row = self.connection.execute(
            "SELECT 1 FROM qa_build_progress WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        return row is not None

    def append(self, *, records: list[dict], chunk_id: int) -> int:
        assert self.cursor is not None
        rows = []
        fts_rows = []
        for offset, record in enumerate(records):
            row_id = self.row_id + offset
            rows.append(
                (
                    row_id,
                    record["case_id"],
                    record.get("label"),
                    record.get("title"),
                    record.get("court"),
                    record.get("case_type"),
                    record.get("date"),
                    record["chunk_order"],
                    record["chunk_count"],
                    record["retrieval_text"],
                    record["preview_text"],
                    record["chunk_text"],
                )
            )
            if self.fts_enabled:
                fts_rows.append((row_id, record["retrieval_text"]))
        if rows:
            self.cursor.executemany(
                """
                INSERT INTO qa_chunk_records (
                    row_id,
                    case_id,
                    label,
                    title,
                    court,
                    case_type,
                    date,
                    chunk_order,
                    chunk_count,
                    retrieval_text,
                    preview_text,
                    chunk_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            if self.fts_enabled:
                self.cursor.executemany(
                    "INSERT INTO qa_chunk_records_fts(rowid, retrieval_text) VALUES (?, ?)",
                    fts_rows,
                )
            self.row_id += len(records)
        self.cursor.execute(
            "INSERT INTO qa_build_progress (chunk_id, record_count) VALUES (?, ?)",
            (chunk_id, len(records)),
        )
        self.connection.commit()
        self.completed_chunks = max(self.completed_chunks, chunk_id)
        return len(records)

    def finalize(
        self,
        *,
        embedding_dimension: int,
        build_global_qa_index: bool,
    ) -> None:
        assert self.cursor is not None
        self.cursor.executemany(
            "INSERT OR REPLACE INTO qa_retrieval_meta (key, value) VALUES (?, ?)",
            [
                ("qa_embedding_model_name", normalize_whitespace(self.settings.shared_embedding_model_name)),
                ("qa_embedding_query_prefix", normalize_whitespace(self.settings.shared_embedding_query_prefix)),
                ("qa_embedding_passage_prefix", normalize_whitespace(self.settings.shared_embedding_passage_prefix)),
                ("qa_embedding_dimension", str(int(embedding_dimension))),
                (
                    "qa_embedding_store_path",
                    str(self.settings.resolve_path(self.settings.qa_retrieval_embedding_store_path)),
                ),
                ("fts5_enabled", "1" if self.fts_enabled else "0"),
                ("record_count", str(int(self.row_id))),
                ("global_qa_index_present", "1" if build_global_qa_index else "0"),
                ("qa_chunk_words", str(int(self.settings.qa_chunk_words))),
                ("qa_chunk_overlap_words", str(int(self.settings.qa_chunk_overlap_words))),
                ("qa_chunk_min_words", str(int(self.settings.qa_chunk_min_words))),
            ],
        )
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_qa_chunk_records_case_id ON qa_chunk_records(case_id)")
        self.connection.commit()

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.connection is not None:
            self.connection.close()


def _iter_records(df, retriever: SimilarCaseRetriever, qa_retriever: LegalQARetriever, settings):
    case_records = retriever.build_records(
        df,
        char_limit=settings.retrieval_char_limit,
        preview_char_limit=settings.retrieval_preview_char_limit,
    )
    qa_records = qa_retriever.build_records(df, settings)
    return case_records, qa_records


def _run_pass1(
    *,
    settings,
    state: dict[str, Any],
    input_path: str,
    max_rows: int | None,
    chunksize: int,
    retriever: SimilarCaseRetriever,
    qa_retriever: LegalQARetriever,
    build_global_qa_index: bool,
) -> dict[str, Any]:
    if state.get("phase1_complete"):
        return state

    work_dir = settings.resolve_path(settings.retrieval_build_work_dir)
    case_shard_dir = work_dir / "case_embedding_shards"
    case_dimension = int(retriever.model.get_sentence_embedding_dimension())

    with (
        CaseMetadataWriter(
            settings.resolve_path(settings.retrieval_metadata_path),
            settings,
            resume=True,
        ) as case_writer,
        QAChunkMetadataWriter(
            settings.resolve_path(settings.qa_retrieval_metadata_path),
            settings,
            resume=True,
        ) as qa_writer,
    ):
        state["case_progress_chunks"] = case_writer.completed_chunks
        state["qa_metadata_progress_chunks"] = qa_writer.completed_chunks
        state["total_case_records"] = case_writer.row_id
        state["total_qa_records"] = qa_writer.row_id
        _save_state(settings, state)

        for chunk_number, df in enumerate(
            iter_dataset_chunks(input_path, chunksize=chunksize, max_rows=max_rows),
            start=1,
        ):
            case_done = case_writer.is_chunk_complete(chunk_number)
            qa_done = qa_writer.is_chunk_complete(chunk_number)
            case_shard_path = _chunk_shard_path(case_shard_dir, "case", chunk_number)

            if case_done and not case_shard_path.exists():
                raise RuntimeError(
                    f"Resume checkpoint is inconsistent: case shard missing for completed chunk {chunk_number}."
                )
            if case_done and qa_done:
                continue

            case_records: list[dict] = []
            qa_records: list[dict] = []
            if not (case_done and qa_done):
                case_records, qa_records = _iter_records(df, retriever, qa_retriever, settings)

            if not case_done:
                if case_records:
                    case_embeddings = retriever.encode_texts(
                        [record["retrieval_text"] for record in case_records],
                        is_query=False,
                        show_progress_bar=False,
                    )
                    _atomic_save_npy(case_shard_path, case_embeddings)
                else:
                    if case_shard_path.exists():
                        case_shard_path.unlink()
                case_writer.append(records=case_records, chunk_id=chunk_number)
                state["case_progress_chunks"] = case_writer.completed_chunks
                state["total_case_records"] = case_writer.row_id

            if not qa_done:
                qa_writer.append(records=qa_records, chunk_id=chunk_number)
                state["qa_metadata_progress_chunks"] = qa_writer.completed_chunks
                state["total_qa_records"] = qa_writer.row_id

            _save_state(settings, state)
            print(
                f"[pass 1] chunk={chunk_number} cases={state['total_case_records']} qa_chunks={state['total_qa_records']}",
                flush=True,
            )

        case_writer.finalize(embedding_dimension=case_dimension)
        qa_writer.finalize(
            embedding_dimension=int(qa_retriever.model.get_sentence_embedding_dimension()),
            build_global_qa_index=build_global_qa_index,
        )

    state["phase1_complete"] = True
    _save_state(settings, state)
    return state


def _run_pass2(
    *,
    settings,
    state: dict[str, Any],
    input_path: str,
    max_rows: int | None,
    chunksize: int,
    qa_retriever: LegalQARetriever,
) -> dict[str, Any]:
    if state.get("phase2_complete"):
        return state

    work_dir = settings.resolve_path(settings.retrieval_build_work_dir)
    qa_shard_dir = work_dir / "qa_embedding_shards"

    for chunk_number, df in enumerate(
        iter_dataset_chunks(input_path, chunksize=chunksize, max_rows=max_rows),
        start=1,
    ):
        if chunk_number <= int(state.get("qa_embedding_progress_chunks", 0)):
            shard_path = _chunk_shard_path(qa_shard_dir, "qa", chunk_number)
            if not shard_path.exists():
                raise RuntimeError(
                    f"Resume checkpoint is inconsistent: QA embedding shard missing for completed chunk {chunk_number}."
                )
            continue

        qa_records = qa_retriever.build_records(df, settings)
        embeddings = (
            qa_retriever.encode_texts(
                [record["retrieval_text"] for record in qa_records],
                is_query=False,
                show_progress_bar=False,
            )
            if qa_records
            else np.empty(
                (0, int(qa_retriever.model.get_sentence_embedding_dimension())),
                dtype="float32",
            )
        )
        shard_path = _chunk_shard_path(qa_shard_dir, "qa", chunk_number)
        _atomic_save_npy(shard_path, embeddings)
        state["qa_embedding_progress_chunks"] = chunk_number
        _save_state(settings, state)
        print(
            f"[pass 2] chunk={chunk_number} qa_embeddings_chunks_done={state['qa_embedding_progress_chunks']}",
            flush=True,
        )

    state["phase2_complete"] = True
    _save_state(settings, state)
    return state


def _finalize_case_index(*, settings, state: dict[str, Any], retriever: SimilarCaseRetriever) -> dict[str, Any]:
    if state.get("case_index_finalized"):
        return state

    work_dir = settings.resolve_path(settings.retrieval_build_work_dir)
    case_shard_dir = work_dir / "case_embedding_shards"
    case_index_path = settings.resolve_path(settings.retrieval_index_path)
    case_index_path.parent.mkdir(parents=True, exist_ok=True)

    dimension = int(retriever.model.get_sentence_embedding_dimension())
    index = faiss.IndexFlatIP(dimension)
    total_rows = 0
    for chunk_id in range(1, int(state.get("case_progress_chunks", 0)) + 1):
        shard_path = _chunk_shard_path(case_shard_dir, "case", chunk_id)
        if not shard_path.exists():
            raise RuntimeError(f"Missing case embedding shard: {shard_path}")
        shard = np.load(shard_path, mmap_mode="r")
        if shard.shape[0]:
            index.add(np.asarray(shard, dtype="float32"))
            total_rows += int(shard.shape[0])
    if total_rows != int(state.get("total_case_records", 0)):
        raise RuntimeError(
            "Case index finalization row count mismatch. "
            f"Expected {state.get('total_case_records', 0)}, wrote {total_rows}."
        )
    faiss.write_index(index, str(case_index_path))
    state["case_index_finalized"] = True
    _save_state(settings, state)
    return state


def _finalize_qa_store(
    *,
    settings,
    state: dict[str, Any],
    qa_retriever: LegalQARetriever,
    build_global_qa_index: bool,
) -> dict[str, Any]:
    if state.get("qa_store_finalized"):
        return state

    work_dir = settings.resolve_path(settings.retrieval_build_work_dir)
    qa_shard_dir = work_dir / "qa_embedding_shards"
    store_path = settings.resolve_path(settings.qa_retrieval_embedding_store_path)
    index_path = settings.resolve_path(settings.qa_retrieval_index_path)

    if store_path.exists():
        store_path.unlink()
    if not build_global_qa_index and index_path.exists():
        index_path.unlink()

    dimension = int(qa_retriever.model.get_sentence_embedding_dimension())
    total_qa_records = int(state.get("total_qa_records", 0))
    embedding_store = np.lib.format.open_memmap(
        store_path,
        mode="w+",
        dtype="float32",
        shape=(total_qa_records, dimension),
    )
    qa_index = faiss.IndexFlatIP(dimension) if build_global_qa_index else None
    row_offset = 0

    for chunk_id in range(1, int(state.get("qa_embedding_progress_chunks", 0)) + 1):
        shard_path = _chunk_shard_path(qa_shard_dir, "qa", chunk_id)
        if not shard_path.exists():
            raise RuntimeError(f"Missing QA embedding shard: {shard_path}")
        shard = np.load(shard_path, mmap_mode="r")
        next_offset = row_offset + int(shard.shape[0])
        if shard.shape[0]:
            embedding_store[row_offset:next_offset] = shard
            if qa_index is not None:
                qa_index.add(np.asarray(shard, dtype="float32"))
        row_offset = next_offset

    embedding_store.flush()
    if row_offset != total_qa_records:
        raise RuntimeError(
            "QA embedding store finalization row count mismatch. "
            f"Expected {total_qa_records}, wrote {row_offset}."
        )
    if qa_index is not None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(qa_index, str(index_path))

    state["qa_store_finalized"] = True
    _save_state(settings, state)
    return state


def _write_manifest(
    *,
    settings,
    state: dict[str, Any],
    build_global_qa_index: bool,
) -> Path:
    manifest_path = settings.resolve_path(settings.retrieval_build_manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_version": STATE_VERSION,
        "input_path": state["input_path"],
        "max_rows": state["max_rows"],
        "chunksize": state["chunksize"],
        "build_global_qa_index": bool(build_global_qa_index),
        "embedding_model": settings.shared_embedding_model_name,
        "embedding_query_prefix": settings.shared_embedding_query_prefix,
        "embedding_passage_prefix": settings.shared_embedding_passage_prefix,
        "case_record_count": int(state["total_case_records"]),
        "qa_chunk_count": int(state["total_qa_records"]),
        "case_index_path": str(settings.resolve_path(settings.retrieval_index_path)),
        "case_metadata_path": str(settings.resolve_path(settings.retrieval_metadata_path)),
        "qa_embedding_store_path": str(settings.resolve_path(settings.qa_retrieval_embedding_store_path)),
        "qa_metadata_path": str(settings.resolve_path(settings.qa_retrieval_metadata_path)),
        "qa_index_path": (
            str(settings.resolve_path(settings.qa_retrieval_index_path))
            if build_global_qa_index
            else None
        ),
        "retrieval_chunk_words": int(settings.retrieval_chunk_words),
        "retrieval_chunk_overlap_words": int(settings.retrieval_chunk_overlap_words),
        "retrieval_chunk_min_words": int(settings.retrieval_chunk_min_words),
        "qa_chunk_words": int(settings.qa_chunk_words),
        "qa_chunk_overlap_words": int(settings.qa_chunk_overlap_words),
        "qa_chunk_min_words": int(settings.qa_chunk_min_words),
        "qa_case_shortlist_top_n": int(settings.qa_case_shortlist_top_n),
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def _cleanup_workdir(settings) -> None:
    work_dir = settings.resolve_path(settings.retrieval_build_work_dir)
    if work_dir.exists():
        shutil.rmtree(work_dir)


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)

    parser = argparse.ArgumentParser(description="Build the case retrieval and QA evidence stores.")
    parser.add_argument("--input", default=settings.train_dataset_path)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--chunksize", type=int, default=1000)
    parser.add_argument("--build-global-qa-index", action="store_true")
    parser.add_argument("--fresh", action="store_true", help="Discard checkpoints and rebuild from scratch.")
    args = parser.parse_args()

    signature = _build_signature(
        settings,
        input_path=args.input,
        max_rows=args.max_rows,
        chunksize=args.chunksize,
        build_global_qa_index=args.build_global_qa_index,
    )
    state = _ensure_state(settings, signature=signature, fresh=args.fresh)

    retriever = SimilarCaseRetriever(settings)
    qa_retriever = LegalQARetriever(settings)

    state = _run_pass1(
        settings=settings,
        state=state,
        input_path=args.input,
        max_rows=args.max_rows,
        chunksize=args.chunksize,
        retriever=retriever,
        qa_retriever=qa_retriever,
        build_global_qa_index=args.build_global_qa_index,
    )
    state = _finalize_case_index(
        settings=settings,
        state=state,
        retriever=retriever,
    )
    state = _run_pass2(
        settings=settings,
        state=state,
        input_path=args.input,
        max_rows=args.max_rows,
        chunksize=args.chunksize,
        qa_retriever=qa_retriever,
    )
    state = _finalize_qa_store(
        settings=settings,
        state=state,
        qa_retriever=qa_retriever,
        build_global_qa_index=args.build_global_qa_index,
    )

    manifest_path = _write_manifest(
        settings=settings,
        state=state,
        build_global_qa_index=args.build_global_qa_index,
    )
    _cleanup_workdir(settings)

    print(f"Indexed {int(state['total_case_records'])} cases")
    print(f"Case FAISS index saved to: {settings.resolve_path(settings.retrieval_index_path)}")
    print(f"Case metadata database saved to: {settings.resolve_path(settings.retrieval_metadata_path)}")
    print(f"Indexed {int(state['total_qa_records'])} QA chunks")
    print(
        "QA embedding store saved to: "
        f"{settings.resolve_path(settings.qa_retrieval_embedding_store_path)}"
    )
    if args.build_global_qa_index:
        print(f"QA FAISS index saved to: {settings.resolve_path(settings.qa_retrieval_index_path)}")
    else:
        print("QA FAISS index skipped (case-first hierarchical runtime mode).")
    print(f"QA metadata database saved to: {settings.resolve_path(settings.qa_retrieval_metadata_path)}")
    print(f"Build manifest saved to: {manifest_path}")
    print(f"Build state saved to: {settings.resolve_path(settings.retrieval_build_state_path)}")
    print(f"Shared embedding space: {settings.shared_embedding_model_name}")


if __name__ == "__main__":
    main()
