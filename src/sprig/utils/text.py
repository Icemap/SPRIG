import re
from typing import Iterable, List

_word_re = re.compile(r"[A-Za-z0-9]+")
_capital_seq_re = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b")
_entity_strip_re = re.compile(r"[^a-z0-9\s]+")


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in _word_re.finditer(text)]


def normalize_entity(text: str, mode: str = "none") -> str:
    if mode == "none":
        return text
    normalized = text.strip()
    if mode in {"lower", "simple"}:
        normalized = normalized.lower()
    if mode == "simple":
        normalized = _entity_strip_re.sub(" ", normalized)
        normalized = " ".join(normalized.split())
    return normalized


def extract_entities_regex(text: str, dedup: bool = True) -> List[str]:
    # Simple heuristic: sequences of capitalized words (1-4 tokens)
    ents = [m.group(0).strip() for m in _capital_seq_re.finditer(text)]
    if not dedup:
        return [e for e in ents if e]
    # Deduplicate but preserve order
    seen = set()
    ordered = []
    for ent in ents:
        if ent and ent not in seen:
            seen.add(ent)
            ordered.append(ent)
    return ordered


def join_sentences(sentences: Iterable[str]) -> str:
    return normalize_whitespace(" ".join(sentences))
