from __future__ import annotations

import re
from typing import Dict, Iterable, List

from sprig.utils.text import normalize_entity

_paren_re = re.compile(r"\s*\([^)]*\)\s*$")


def _title_aliases(title: str) -> List[str]:
    aliases = [title]
    stripped = _paren_re.sub("", title).strip()
    if stripped and stripped != title:
        aliases.append(stripped)
    return aliases


def build_title_alias_map(titles: Iterable[str]) -> Dict[str, str]:
    """Build a lightweight alias map from document titles.

    Keys are normalized with `simple` mode. Ambiguous aliases are dropped.
    """
    first: Dict[str, str] = {}
    counts: Dict[str, int] = {}
    for title in titles:
        for alias in _title_aliases(title):
            key = normalize_entity(alias, mode="simple")
            if not key:
                continue
            counts[key] = counts.get(key, 0) + 1
            if key not in first:
                first[key] = title
    return {k: v for k, v in first.items() if counts.get(k, 0) == 1}


def apply_alias_map(
    entities: Iterable[str],
    alias_map: Dict[str, str] | None,
    normalize_mode: str = "none",
) -> List[str]:
    mapped: List[str] = []
    for ent in entities:
        if not ent:
            continue
        out = ent
        if alias_map:
            key = normalize_entity(ent, mode="simple")
            canonical = alias_map.get(key)
            if canonical:
                out = canonical
        if normalize_mode != "none":
            out = normalize_entity(out, mode=normalize_mode)
        if out:
            mapped.append(out)
    return mapped
