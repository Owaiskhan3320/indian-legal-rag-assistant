from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def normalize_text(value: str | None) -> str:
    return " ".join((value or "").strip().lower().split())


def signature_for_record(record: dict[str, Any]) -> str:
    answer = normalize_text(record.get("answer"))
    sources = "|".join(sorted(normalize_text(item) for item in (record.get("retrieved_case_ids") or [])))
    workflow = normalize_text(record.get("workflow"))
    question = normalize_text(record.get("question"))
    payload = f"{workflow}::{question}::{sources}::{answer}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return payload["records"]
    raise ValueError("Expected a JSON array, an object with a 'records' array, or JSONL.")


def hit_rate(record: dict[str, Any]) -> float | None:
    expected = {normalize_text(item) for item in (record.get("expected_case_ids") or []) if normalize_text(item)}
    observed = {normalize_text(item) for item in (record.get("retrieved_case_ids") or []) if normalize_text(item)}
    if not expected:
        return None
    return len(expected & observed) / max(len(expected), 1)


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_workflow: dict[str, list[dict[str, Any]]] = defaultdict(list)
    signatures = Counter()
    for record in records:
        workflow = normalize_text(record.get("workflow")) or "unknown"
        by_workflow[workflow].append(record)
        signatures[signature_for_record(record)] += 1

    summary: dict[str, Any] = {
        "record_count": len(records),
        "duplicate_signature_count": sum(1 for value in signatures.values() if value > 1),
        "workflows": {},
    }
    for workflow, items in sorted(by_workflow.items()):
        latency_values = [
            float(item.get("latency_ms"))
            for item in items
            if item.get("latency_ms") is not None
        ]
        confidence_values = [
            normalize_text(item.get("answer_confidence"))
            for item in items
            if normalize_text(item.get("answer_confidence"))
        ]
        evidence_values = [
            normalize_text(item.get("evidence_strength"))
            for item in items
            if normalize_text(item.get("evidence_strength"))
        ]
        hit_values = [value for value in (hit_rate(item) for item in items) if value is not None]
        abstain_count = sum(
            1
            for item in items
            if any(
                marker in normalize_text(item.get("answer"))
                for marker in (
                    "could not answer reliably",
                    "insufficient evidence",
                    "not enough evidence",
                )
            )
        )
        summary["workflows"][workflow] = {
            "count": len(items),
            "avg_latency_ms": round(sum(latency_values) / len(latency_values), 2) if latency_values else None,
            "median_latency_ms": round(sorted(latency_values)[len(latency_values) // 2], 2)
            if latency_values
            else None,
            "avg_source_hit_rate": round(sum(hit_values) / len(hit_values), 3) if hit_values else None,
            "abstain_rate": round(abstain_count / max(len(items), 1), 3),
            "confidence_distribution": dict(Counter(confidence_values)),
            "evidence_distribution": dict(Counter(evidence_values)),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="No-rebuild evaluation harness for Nyaya workflows."
    )
    parser.add_argument("input", type=Path, help="Path to a JSON or JSONL evaluation export.")
    args = parser.parse_args()

    records = load_records(args.input)
    result = summarize(records)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
