from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np

from sprig.retrieval.graph import GraphIndex


@dataclass
class DegreeStats:
    avg: float
    p95: float
    max: float


@dataclass
class GraphStats:
    nodes: int
    entities: int
    docs: int
    edges: int
    entity_degree: DegreeStats
    doc_degree: DegreeStats


def _degree_stats(values: np.ndarray) -> DegreeStats:
    if values.size == 0:
        return DegreeStats(avg=0.0, p95=0.0, max=0.0)
    return DegreeStats(
        avg=float(values.mean()),
        p95=float(np.percentile(values, 95)),
        max=float(values.max()),
    )


def compute_graph_stats(index: GraphIndex) -> Dict[str, object]:
    n_entities = len(index.entity_ids)
    n_docs = len(index.doc_ids)
    nodes = n_entities + n_docs
    transition = index.transition
    # Non-zero counts capture bipartite degrees (entities -> docs, docs -> entities).
    ent_deg = transition[:n_entities].getnnz(axis=1)
    doc_deg = transition[n_entities:].getnnz(axis=1)

    stats = GraphStats(
        nodes=nodes,
        entities=n_entities,
        docs=n_docs,
        edges=index.edge_count,
        entity_degree=_degree_stats(ent_deg),
        doc_degree=_degree_stats(doc_deg),
    )
    return {
        "nodes": stats.nodes,
        "entities": stats.entities,
        "docs": stats.docs,
        "edges": stats.edges,
        "entity_degree": asdict(stats.entity_degree),
        "doc_degree": asdict(stats.doc_degree),
    }
