from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List

from sprig.utils.text import extract_entities_regex, normalize_entity

_SPACY_LABELS = {
    "PERSON",
    "ORG",
    "GPE",
    "LOC",
    "FAC",
    "NORP",
    "EVENT",
    "PRODUCT",
    "WORK_OF_ART",
    "LAW",
}


@lru_cache(maxsize=1)
def _nlp():
    import spacy

    return spacy.load("en_core_web_sm")


def extract_entities(
    text: str,
    mode: str = "regex",
    dedup: bool = True,
    normalize: str = "none",
) -> List[str]:
    if mode == "spacy":
        try:
            nlp = _nlp()
        except Exception:
            # fallback to regex if model is missing
            ents = extract_entities_regex(text, dedup=dedup)
            return [normalize_entity(e, normalize) for e in ents if e]
        doc = nlp(text)
        ents = [ent.text.strip() for ent in doc.ents if ent.label_ in _SPACY_LABELS]
        if normalize != "none":
            ents = [normalize_entity(e, normalize) for e in ents if e]
        if dedup:
            seen = set()
            ordered = []
            for ent in ents:
                if ent and ent not in seen:
                    seen.add(ent)
                    ordered.append(ent)
            return ordered
        return [e for e in ents if e]

    ents = extract_entities_regex(text, dedup=dedup)
    return [normalize_entity(e, normalize) for e in ents if e]
