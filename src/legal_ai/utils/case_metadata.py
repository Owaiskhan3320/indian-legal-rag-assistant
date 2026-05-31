from __future__ import annotations

import re

from legal_ai.utils.text import normalize_whitespace


def derive_case_metadata(case_id: str | None) -> dict[str, str | None]:
    normalized = normalize_whitespace(case_id or "")
    if not normalized:
        return {
            "case_id": "",
            "source_family": None,
            "court": None,
            "case_type": None,
            "year": None,
            "title": None,
        }

    parts = [part for part in normalized.split("_") if part]
    year = next((part for part in parts if re.fullmatch(r"(19|20)\d{2}", part)), None)
    source_parts = parts
    if year and year in parts:
        year_index = parts.index(year)
        source_parts = parts[:year_index]
    elif len(parts) >= 2:
        source_parts = parts[:-2]

    source_family = "_".join(source_parts) or normalized
    source_label = normalize_whitespace(source_family.replace("_", " "))
    lowered_source = source_family.lower()
    token_set = {token.lower() for token in source_parts}

    court = None
    if "sc" in token_set or "supreme" in token_set:
        court = "Supreme Court"
    elif "hc" in token_set or "high" in token_set:
        court = "High Court"
    elif any(token in lowered_source for token in ["tribunal", "appellate", "commission", "forum"]):
        court = "Tribunal / Appellate Forum"

    case_type = None
    if any(token in lowered_source for token in ["consumer"]):
        case_type = "Consumer dispute"
    elif any(token in lowered_source for token in ["income_tax", "tax", "custom", "excise", "gst"]):
        case_type = "Tax dispute"
    elif any(token in lowered_source for token in ["bail", "criminal"]):
        case_type = "Bail / criminal matter"
    elif any(token in lowered_source for token in ["motor", "accident", "mact"]):
        case_type = "Motor accident compensation"
    elif any(token in lowered_source for token in ["service", "employment", "labour", "pension"]):
        case_type = "Service / employment dispute"
    elif any(token in lowered_source for token in ["university", "education", "exam", "school"]):
        case_type = "University / examination dispute"
    elif any(token in lowered_source for token in ["property", "land", "tenancy"]):
        case_type = "Property / land dispute"
    elif any(token in lowered_source for token in ["contract", "recovery", "commercial"]):
        case_type = "Contract / recovery matter"
    elif any(token in lowered_source for token in ["writ"]):
        case_type = "Writ petition"

    title = source_label
    if year:
        title = f"{source_label} ({year})"

    return {
        "case_id": normalized,
        "source_family": source_family or None,
        "court": court,
        "case_type": case_type,
        "year": year,
        "title": title or None,
    }
