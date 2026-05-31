from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from legal_ai.config import get_settings  # noqa: E402
from legal_ai.logging_utils import configure_logging  # noqa: E402
from legal_ai.services.reference_law import ReferenceLawRetriever  # noqa: E402


def build_manifest(records: list[dict]) -> dict:
    by_title = Counter(record.get("title") for record in records)
    by_domain = Counter(record.get("domain") or "general" for record in records)
    by_authority = Counter(record.get("authority_type") or "act" for record in records)
    return {
        "record_count": len(records),
        "documents": dict(sorted(by_title.items())),
        "domains": dict(sorted(by_domain.items())),
        "authority_types": dict(sorted(by_authority.items())),
    }


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Build the reference-law retrieval store from official PDFs.")
    parser.add_argument(
        "--source-dir",
        default=None,
        help="Directory containing official law PDFs. Defaults to settings.reference_law_source_dir.",
    )
    args = parser.parse_args()

    configure_logging(settings.log_level)
    source_dir = Path(args.source_dir or settings.reference_law_source_dir)
    if not source_dir.exists():
        raise SystemExit(f"Reference-law source directory does not exist: {source_dir}")

    retriever = ReferenceLawRetriever(settings)
    records = retriever.build_records_from_directory(source_dir)
    if not records:
        raise SystemExit("No reference-law records were generated from the supplied PDFs.")

    retriever.build(records)

    manifest_path = settings.resolve_path("artifacts/reference_law_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(build_manifest(records), indent=2),
        encoding="utf-8",
    )
    print(
        f"Built reference-law store with {len(records)} child records from {source_dir}.\n"
        f"Index: {settings.resolve_path(settings.reference_law_index_path)}\n"
        f"Metadata: {settings.resolve_path(settings.reference_law_metadata_path)}\n"
        f"Manifest: {manifest_path}"
    )


if __name__ == "__main__":
    main()
