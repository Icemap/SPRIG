from __future__ import annotations

import argparse
from pathlib import Path

from sprig.experiment import run_experiment
from sprig.run_utils import make_output_dir, timestamp
from sprig.visualize import plot_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Sprig experimental runner")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="run retrieval experiments")
    run_p.add_argument(
        "--dataset", default="hotpotqa", choices=["hotpotqa", "2wikimultihopqa"]
    )
    run_p.add_argument("--split", default="validation")
    run_p.add_argument("--max-samples", type=int, default=200)
    run_p.add_argument("--seed", type=int, default=42)
    run_p.add_argument("--methods", nargs="+", default=["bm25", "dense", "graph"])
    run_p.add_argument("--top-ks", nargs="+", type=int, default=[1, 3, 5, 10])
    run_p.add_argument(
        "--dense-model", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    run_p.add_argument("--dense-no-hnsw", action="store_true")
    run_p.add_argument("--hnsw-m", type=int, default=32)
    run_p.add_argument("--hnsw-ef-construction", type=int, default=200)
    run_p.add_argument("--hnsw-ef-search", type=int, default=64)
    run_p.add_argument("--graph-ner", default="spacy", choices=["spacy", "regex"])
    run_p.add_argument(
        "--graph-entity-normalize",
        default="none",
        choices=["none", "lower", "simple"],
    )
    run_p.add_argument("--graph-use-aliases", action="store_true")
    run_p.add_argument("--graph-seed-docs-k", type=int, default=5)
    run_p.add_argument("--graph-seed-docs-k-dense", type=int, default=None)
    run_p.add_argument("--graph-seed-docs-k-bm25", type=int, default=None)
    run_p.add_argument("--graph-seed-docs-k-rrf", type=int, default=None)
    run_p.add_argument("--graph-fallback-k", type=int, default=1)
    run_p.add_argument("--ppr-alpha", type=float, default=0.15)
    run_p.add_argument("--ppr-max-iter", type=int, default=10)
    run_p.add_argument("--ppr-mode", default="power", choices=["power", "push"])
    run_p.add_argument("--ppr-tol", type=float, default=None)
    run_p.add_argument("--ppr-push-eps", type=float, default=1e-4)
    run_p.add_argument("--ppr-push-max-steps", type=int, default=200000)
    run_p.add_argument("--graph-min-df", type=int, default=1)
    run_p.add_argument("--graph-max-df-ratio", type=float, default=1.0)
    run_p.add_argument("--graph-hub-penalty", type=float, default=0.0)
    run_p.add_argument("--graph-hub-top-ratio", type=float, default=0.0)
    run_p.add_argument("--graph-entity-edge-cap", type=int, default=None)
    run_p.add_argument("--graph-norm", default="row", choices=["row", "sym", "none"])
    run_p.add_argument("--graph-seed-entity-df-power", type=float, default=0.0)
    run_p.add_argument(
        "--graph-seed-weighting",
        default="raw",
        choices=["raw", "softmax", "rank"],
    )
    run_p.add_argument("--graph-seed-temp", type=float, default=1.0)
    run_p.add_argument(
        "--graph-seed-mix-mode",
        default="l1",
        choices=["l1", "fixed", "auto"],
    )
    run_p.add_argument("--graph-seed-mix-alpha", type=float, default=None)
    run_p.add_argument("--term-min-df", type=int, default=5)
    run_p.add_argument("--term-max-df-ratio", type=float, default=0.2)
    run_p.add_argument("--term-norm", default="row", choices=["row", "sym", "none"])
    run_p.add_argument("--rrf-k", type=int, default=60)
    run_p.add_argument("--rm3-fb-docs", type=int, default=10)
    run_p.add_argument("--rm3-fb-terms", type=int, default=20)
    run_p.add_argument("--rm3-orig-weight", type=float, default=0.5)
    run_p.add_argument("--bm25-2step-k1", type=int, default=10)
    run_p.add_argument("--bm25-2step-expand-per-doc", type=int, default=5)
    run_p.add_argument("--bm25-2step-max-terms", type=int, default=30)
    run_p.add_argument(
        "--bm25-2step-entity-mode",
        default="regex",
        choices=["regex", "spacy"],
    )
    run_p.add_argument(
        "--bm25-2step-entity-normalize",
        default="simple",
        choices=["none", "lower", "simple"],
    )
    run_p.add_argument("--bm25-2step-query-weight", type=float, default=1.0)
    run_p.add_argument("--rrf-ppr-fusion-weight", type=float, default=1.0)
    run_p.add_argument("--rerank-model", default="cross-encoder/ms-marco-TinyBERT-L-2-v2")
    run_p.add_argument("--rerank-candidates", type=int, default=100)
    run_p.add_argument("--rerank-batch-size", type=int, default=32)
    run_p.add_argument("--cache-root", default="outputs/_cache")
    run_p.add_argument("--tag", default=None)
    run_p.add_argument("--output", default=None)

    plot_p = sub.add_parser("plot", help="plot metrics.json into figures")
    plot_p.add_argument("--input", required=True)

    args = parser.parse_args()

    if args.cmd == "run":
        run_id = timestamp()
        out_dir = (
            Path(args.output)
            if args.output
            else make_output_dir(
                dataset=args.dataset,
                split=args.split,
                max_samples=args.max_samples,
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
                ts=run_id,
            )
        )
        run_experiment(
            output_dir=out_dir,
            dataset=args.dataset,
            split=args.split,
            max_samples=args.max_samples,
            seed=args.seed,
            methods=args.methods,
            top_ks=args.top_ks,
            dense_model=args.dense_model,
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
            rm3_fb_docs=args.rm3_fb_docs,
            rm3_fb_terms=args.rm3_fb_terms,
            rm3_orig_weight=args.rm3_orig_weight,
            bm25_2step_k1=args.bm25_2step_k1,
            bm25_2step_expand_per_doc=args.bm25_2step_expand_per_doc,
            bm25_2step_max_terms=args.bm25_2step_max_terms,
            bm25_2step_entity_mode=args.bm25_2step_entity_mode,
            bm25_2step_entity_normalize=args.bm25_2step_entity_normalize,
            bm25_2step_query_weight=args.bm25_2step_query_weight,
            rrf_ppr_fusion_weight=args.rrf_ppr_fusion_weight,
            rerank_model=args.rerank_model,
            rerank_candidates=args.rerank_candidates,
            rerank_batch_size=args.rerank_batch_size,
            hnsw_m=args.hnsw_m,
            hnsw_ef_construction=args.hnsw_ef_construction,
            hnsw_ef_search=args.hnsw_ef_search,
            cache_root=Path(args.cache_root),
            tag=args.tag,
        )
        plot_metrics(out_dir / "metrics.json")
    elif args.cmd == "plot":
        plot_metrics(Path(args.input))


if __name__ == "__main__":
    main()
