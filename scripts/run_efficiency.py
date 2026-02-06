from __future__ import annotations

import argparse
from pathlib import Path

from sprig.experiment import run_experiment
from sprig.run_utils import make_output_dir, timestamp


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["hotpotqa", "2wikimultihopqa"], required=True)
    p.add_argument("--split", default="validation")
    p.add_argument("--sizes", type=int, nargs="+", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--methods", nargs="+", default=["bm25", "dense", "graph", "graph_dense"])
    p.add_argument("--graph-ner", default="spacy")
    p.add_argument("--graph-entity-normalize", default="none", choices=["none", "lower", "simple"])
    p.add_argument("--graph-use-aliases", action="store_true")
    p.add_argument("--graph-seed-docs-k", type=int, default=5)
    p.add_argument("--graph-seed-docs-k-dense", type=int, default=None)
    p.add_argument("--graph-seed-docs-k-bm25", type=int, default=None)
    p.add_argument("--graph-seed-docs-k-rrf", type=int, default=None)
    p.add_argument("--graph-fallback-k", type=int, default=1)
    p.add_argument("--ppr-alpha", type=float, default=0.15)
    p.add_argument("--ppr-max-iter", type=int, default=10)
    p.add_argument("--ppr-mode", default="power", choices=["power", "push"])
    p.add_argument("--ppr-tol", type=float, default=None)
    p.add_argument("--ppr-push-eps", type=float, default=1e-4)
    p.add_argument("--ppr-push-max-steps", type=int, default=200000)
    p.add_argument("--graph-min-df", type=int, default=1)
    p.add_argument("--graph-max-df-ratio", type=float, default=1.0)
    p.add_argument("--graph-hub-penalty", type=float, default=0.0)
    p.add_argument("--graph-hub-top-ratio", type=float, default=0.0)
    p.add_argument("--graph-entity-edge-cap", type=int, default=None)
    p.add_argument("--graph-norm", default="row", choices=["row", "sym", "none"])
    p.add_argument("--graph-seed-entity-df-power", type=float, default=0.0)
    p.add_argument("--graph-seed-weighting", default="raw", choices=["raw", "softmax", "rank"])
    p.add_argument("--graph-seed-temp", type=float, default=1.0)
    p.add_argument("--graph-seed-mix-mode", default="l1", choices=["l1", "fixed", "auto"])
    p.add_argument("--graph-seed-mix-alpha", type=float, default=None)
    p.add_argument("--term-min-df", type=int, default=5)
    p.add_argument("--term-max-df-ratio", type=float, default=0.2)
    p.add_argument("--term-norm", default="row", choices=["row", "sym", "none"])
    p.add_argument("--rrf-k", type=int, default=60)
    p.add_argument("--hnsw-m", type=int, default=32)
    p.add_argument("--hnsw-ef-construction", type=int, default=200)
    p.add_argument("--hnsw-ef-search", type=int, default=64)
    p.add_argument("--dense-no-hnsw", action="store_true")
    p.add_argument("--cache-root", default="outputs/_cache")
    p.add_argument("--tag", default="eff")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    for size in args.sizes:
        out_dir = make_output_dir(
            dataset=args.dataset,
            split=args.split,
            max_samples=size,
            methods=args.methods,
            seed=args.seed,
            graph_ner=args.graph_ner,
            graph_entity_normalize=args.graph_entity_normalize,
            graph_use_aliases=args.graph_use_aliases,
            graph_seed_docs_k=args.graph_seed_docs_k,
            graph_seed_docs_k_dense=args.graph_seed_docs_k_dense,
            graph_seed_docs_k_bm25=args.graph_seed_docs_k_bm25,
            graph_seed_docs_k_rrf=args.graph_seed_docs_k_rrf,
            graph_fallback_k=args.graph_fallback_k,
            ppr_alpha=args.ppr_alpha,
            ppr_max_iter=args.ppr_max_iter,
            ppr_mode=args.ppr_mode,
            ppr_tol=args.ppr_tol,
            ppr_push_eps=args.ppr_push_eps,
            ppr_push_max_steps=args.ppr_push_max_steps,
            graph_min_df=args.graph_min_df,
            graph_max_df_ratio=args.graph_max_df_ratio,
            graph_hub_penalty=args.graph_hub_penalty,
            graph_hub_top_ratio=args.graph_hub_top_ratio,
            graph_entity_edge_cap=args.graph_entity_edge_cap,
            graph_norm=args.graph_norm,
            graph_seed_entity_df_power=args.graph_seed_entity_df_power,
            graph_seed_weighting=args.graph_seed_weighting,
            graph_seed_temp=args.graph_seed_temp,
            graph_seed_mix_mode=args.graph_seed_mix_mode,
            graph_seed_mix_alpha=args.graph_seed_mix_alpha,
            term_min_df=args.term_min_df,
            term_max_df_ratio=args.term_max_df_ratio,
            term_norm=args.term_norm,
            rrf_k=args.rrf_k,
            hnsw_m=args.hnsw_m,
            hnsw_ef_construction=args.hnsw_ef_construction,
            hnsw_ef_search=args.hnsw_ef_search,
            dense_no_hnsw=args.dense_no_hnsw,
            tag=args.tag,
            ts=timestamp(),
        )
        run_experiment(
            output_dir=out_dir,
            dataset=args.dataset,
            split=args.split,
            max_samples=size,
            seed=args.seed,
            methods=args.methods,
            graph_ner=args.graph_ner,
            graph_entity_normalize=args.graph_entity_normalize,
            graph_use_aliases=args.graph_use_aliases,
            graph_seed_docs_k=args.graph_seed_docs_k,
            graph_seed_docs_k_dense=args.graph_seed_docs_k_dense,
            graph_seed_docs_k_bm25=args.graph_seed_docs_k_bm25,
            graph_seed_docs_k_rrf=args.graph_seed_docs_k_rrf,
            graph_fallback_k=args.graph_fallback_k,
            dense_use_hnsw=not args.dense_no_hnsw,
            ppr_alpha=args.ppr_alpha,
            ppr_max_iter=args.ppr_max_iter,
            ppr_mode=args.ppr_mode,
            ppr_tol=args.ppr_tol,
            ppr_push_eps=args.ppr_push_eps,
            ppr_push_max_steps=args.ppr_push_max_steps,
            graph_min_df=args.graph_min_df,
            graph_max_df_ratio=args.graph_max_df_ratio,
            graph_hub_penalty=args.graph_hub_penalty,
            graph_hub_top_ratio=args.graph_hub_top_ratio,
            graph_entity_edge_cap=args.graph_entity_edge_cap,
            graph_norm=args.graph_norm,
            graph_seed_entity_df_power=args.graph_seed_entity_df_power,
            graph_seed_weighting=args.graph_seed_weighting,
            graph_seed_temp=args.graph_seed_temp,
            graph_seed_mix_mode=args.graph_seed_mix_mode,
            graph_seed_mix_alpha=args.graph_seed_mix_alpha,
            term_min_df=args.term_min_df,
            term_max_df_ratio=args.term_max_df_ratio,
            term_norm=args.term_norm,
            rrf_k=args.rrf_k,
            hnsw_m=args.hnsw_m,
            hnsw_ef_construction=args.hnsw_ef_construction,
            hnsw_ef_search=args.hnsw_ef_search,
            cache_root=Path(args.cache_root),
            tag=args.tag,
        )


if __name__ == "__main__":
    main()
