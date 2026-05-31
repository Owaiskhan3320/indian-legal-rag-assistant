import re


WHITESPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[a-z0-9]+")
SEARCH_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "if",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "with",
}


def normalize_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub(" ", str(text or "")).strip()


def shorten_text(text: str, max_chars: int) -> str:
    cleaned = normalize_whitespace(text)
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def compact_text(text: str, max_chars: int) -> str:
    cleaned = normalize_whitespace(text)
    if len(cleaned) <= max_chars:
        return cleaned

    separator = " [...] "
    head_chars = max(int(max_chars * 0.72), 1)
    tail_chars = max(max_chars - head_chars - len(separator), 1)
    return (
        cleaned[:head_chars].rstrip()
        + separator
        + cleaned[-tail_chars:].lstrip()
    )


def split_into_word_chunks(
    text: str,
    chunk_words: int,
    overlap_words: int,
    min_words: int,
) -> list[str]:
    cleaned = normalize_whitespace(text)
    if not cleaned:
        return []

    words = cleaned.split(" ")
    if len(words) <= chunk_words:
        return [cleaned]

    step = max(chunk_words - overlap_words, 1)
    chunks: list[str] = []
    for start in range(0, len(words), step):
        end = min(start + chunk_words, len(words))
        chunk_words_list = words[start:end]
        if len(chunk_words_list) < min_words and chunks:
            break
        chunks.append(" ".join(chunk_words_list))
        if end >= len(words):
            break
    return chunks


def search_terms(text: str) -> list[str]:
    cleaned = normalize_whitespace(text).lower()
    return [token for token in TOKEN_RE.findall(cleaned) if token not in SEARCH_STOPWORDS]


def lexical_overlap_score(query_text: str, document_text: str) -> float:
    query_tokens = search_terms(query_text)
    document_tokens = search_terms(document_text)
    if not query_tokens or not document_tokens:
        return 0.0

    query_set = set(query_tokens)
    document_set = set(document_tokens)
    unigram_overlap = len(query_set & document_set) / max(len(query_set), 1)

    query_bigrams = set(zip(query_tokens, query_tokens[1:]))
    document_bigrams = set(zip(document_tokens, document_tokens[1:]))
    if query_bigrams:
        bigram_overlap = len(query_bigrams & document_bigrams) / len(query_bigrams)
    else:
        bigram_overlap = 0.0

    exact_phrase_bonus = 0.0
    normalized_query = normalize_whitespace(query_text).lower()
    normalized_document = normalize_whitespace(document_text).lower()
    if normalized_query and len(normalized_query) >= 24 and normalized_query in normalized_document:
        exact_phrase_bonus = 0.1

    return min((0.7 * unigram_overlap) + (0.2 * bigram_overlap) + exact_phrase_bonus, 1.0)


def overlapping_terms(query_text: str, document_text: str, limit: int = 4) -> list[str]:
    query_tokens = search_terms(query_text)
    document_tokens = set(search_terms(document_text))
    ordered: list[str] = []
    for token in query_tokens:
        if token in document_tokens and token not in ordered:
            ordered.append(token)
        if len(ordered) >= limit:
            break
    return ordered
