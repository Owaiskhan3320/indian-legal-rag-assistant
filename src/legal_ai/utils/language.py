from __future__ import annotations

from legal_ai.utils.text import normalize_whitespace


SCRIPT_LANGUAGE_RANGES = [
    ("Hindi", 0x0900, 0x097F),  # Devanagari
    ("Bengali", 0x0980, 0x09FF),
    ("Punjabi", 0x0A00, 0x0A7F),  # Gurmukhi
    ("Gujarati", 0x0A80, 0x0AFF),
    ("Odia", 0x0B00, 0x0B7F),
    ("Tamil", 0x0B80, 0x0BFF),
    ("Telugu", 0x0C00, 0x0C7F),
    ("Kannada", 0x0C80, 0x0CFF),
    ("Malayalam", 0x0D00, 0x0D7F),
]

LANGUAGE_ALIASES = {
    "auto": "Auto",
    "same as question": "Auto",
    "same as query": "Auto",
    "english": "English",
    "en": "English",
    "hindi": "Hindi",
    "hi": "Hindi",
    "bengali": "Bengali",
    "bn": "Bengali",
    "punjabi": "Punjabi",
    "pa": "Punjabi",
    "gujarati": "Gujarati",
    "gu": "Gujarati",
    "odia": "Odia",
    "or": "Odia",
    "oriya": "Odia",
    "tamil": "Tamil",
    "ta": "Tamil",
    "telugu": "Telugu",
    "te": "Telugu",
    "kannada": "Kannada",
    "kn": "Kannada",
    "malayalam": "Malayalam",
    "ml": "Malayalam",
}


def detect_language_label(text: str) -> str:
    cleaned = normalize_whitespace(text)
    if not cleaned:
        return "English"

    script_counts: dict[str, int] = {}
    latin_count = 0
    for char in cleaned:
        codepoint = ord(char)
        if "A" <= char <= "Z" or "a" <= char <= "z":
            latin_count += 1
            continue
        for language, start, end in SCRIPT_LANGUAGE_RANGES:
            if start <= codepoint <= end:
                script_counts[language] = script_counts.get(language, 0) + 1
                break

    if not script_counts:
        return "English"

    top_language, top_count = max(script_counts.items(), key=lambda item: item[1])
    if latin_count >= top_count:
        return "English"
    return top_language


def normalize_answer_language(value: str | None, detected_language: str) -> str:
    cleaned = normalize_whitespace(value).lower()
    if not cleaned:
        return detected_language if detected_language != "English" else "English"

    mapped = LANGUAGE_ALIASES.get(cleaned)
    if mapped == "Auto" or mapped is None and cleaned == "auto":
        return detected_language if detected_language != "English" else "English"
    if mapped:
        return mapped
    return value.strip()
