from __future__ import annotations

from datetime import datetime
import hashlib
from pathlib import Path
from typing import Sequence


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def slug_methods(methods: Sequence[str]) -> str:
    return "+".join(methods)


def make_output_dir(
    dataset: str,
    split: str,
    max_samples: int | None,
    methods: Sequence[str],
    seed: int,
    graph_ner: str,
    graph_entity_normalize: str,
    graph_use_aliases: bool,
    graph_seed_docs_k: int,
    graph_seed_docs_k_dense: int | None,
    graph_seed_docs_k_bm25: int | None,
    graph_seed_docs_k_rrf: int | None,
    graph_fallback_k: int | None,
    ppr_alpha: float,
    ppr_max_iter: int,
    ppr_mode: str,
    ppr_tol: float | None,
    ppr_push_eps: float,
    ppr_push_max_steps: int,
    graph_min_df: int,
    graph_max_df_ratio: float,
    graph_hub_penalty: float,
    graph_hub_top_ratio: float,
    graph_entity_edge_cap: int | None,
    graph_norm: str,
    graph_seed_entity_df_power: float,
    graph_seed_weighting: str,
    graph_seed_temp: float,
    graph_seed_mix_mode: str,
    graph_seed_mix_alpha: float | None,
    term_min_df: int,
    term_max_df_ratio: float,
    term_norm: str,
    rrf_k: int,
    hnsw_m: int,
    hnsw_ef_construction: int,
    hnsw_ef_search: int,
    dense_no_hnsw: bool,
    tag: str | None = None,
    ts: str | None = None,
) -> Path:
    ts = ts or timestamp()
    n = "full" if max_samples is None else f"n{max_samples}"
    methods_slug = slug_methods(methods)
    hnsw = "nohnsw" if dense_no_hnsw else f"hnsw_m{hnsw_m}_efc{hnsw_ef_construction}_efs{hnsw_ef_search}"
    tag_part = f"_tag-{tag}" if tag else ""
    tol_tag = "tol" if ppr_tol is not None else "notol"
    kd = graph_seed_docs_k_dense or graph_seed_docs_k
    kb = graph_seed_docs_k_bm25 or graph_seed_docs_k
    kr = graph_seed_docs_k_rrf or graph_seed_docs_k
    kf = graph_fallback_k or 1
    ga = 1 if graph_use_aliases else 0
    gec = graph_entity_edge_cap or 0
    gsa = "none" if graph_seed_mix_alpha is None else f"{graph_seed_mix_alpha}"
    name = (
        f"{dataset}_{split}_{n}_m-{methods_slug}_s{seed}_"
        f"ner-{graph_ner}_norm-{graph_entity_normalize}_ga{ga}_k{graph_seed_docs_k}_"
        f"kd{kd}_kb{kb}_kr{kr}_kf{kf}_"
        f"a{ppr_alpha}_it{ppr_max_iter}_pm{ppr_mode}_{tol_tag}_"
        f"peps{ppr_push_eps}_pmax{ppr_push_max_steps}_"
        f"gdf{graph_min_df}_gmx{graph_max_df_ratio}_gh{graph_hub_penalty}_ghc{graph_hub_top_ratio}_"
        f"gec{gec}_gn{graph_norm}_"
        f"gseeddf{graph_seed_entity_df_power}_gsw{graph_seed_weighting}_gst{graph_seed_temp}_"
        f"gsm{graph_seed_mix_mode}_gsa{gsa}_"
        f"tdf{term_min_df}_tmx{term_max_df_ratio}_tn{term_norm}_"
        f"rrf{rrf_k}_{hnsw}{tag_part}_{ts}"
    )
    if len(name) > 200:
        digest = hashlib.md5(name.encode("utf-8")).hexdigest()[:8]
        name = f"{name[:180]}_h{digest}"
    return Path("outputs") / name
