from __future__ import annotations

from pathlib import Path

import pandas as pd


def _normalize_dataset_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename_map = {}
    if "filename" in df.columns and "case_id" not in df.columns:
        rename_map["filename"] = "case_id"
    if "text" in df.columns and "case_text" not in df.columns:
        rename_map["text"] = "case_text"

    if rename_map:
        df = df.rename(columns=rename_map)

    required = {"case_id", "case_text"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df["case_id"] = df["case_id"].astype(str)
    df["case_text"] = df["case_text"].fillna("").astype(str)
    if "label" in df.columns:
        numeric_labels = pd.to_numeric(df["label"], errors="coerce")
        df["label"] = numeric_labels.astype("Int64")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return df


def load_dataset(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_csv(file_path)
    return _normalize_dataset_frame(df)


def iter_dataset_chunks(
    path: str | Path,
    *,
    chunksize: int = 5000,
    max_rows: int | None = None,
):
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    rows_emitted = 0
    for chunk_df in pd.read_csv(file_path, chunksize=chunksize):
        normalized_chunk = _normalize_dataset_frame(chunk_df)
        if max_rows is not None:
            remaining = max_rows - rows_emitted
            if remaining <= 0:
                break
            if len(normalized_chunk) > remaining:
                normalized_chunk = normalized_chunk.iloc[:remaining].copy()
        if normalized_chunk.empty:
            continue
        rows_emitted += len(normalized_chunk)
        yield normalized_chunk
        if max_rows is not None and rows_emitted >= max_rows:
            break
