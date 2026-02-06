# SPRIG Reproduction Guide

This repository contains the experimental code and scripts for the SPRIG paper. The steps
below reproduce all tables/figures in the main text and appendix (efficiency, ablations,
significance tests, QA evaluation, etc.). CPU-only is assumed.

## 1. Environment & Install

### Hardware / OS
- Python >= 3.12
- CPU-only (paper uses 4 GB RAM budget)
- Linux/macOS

### Install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[vector,data,plot,qa,dev]"
python3 -m spacy download en_core_web_sm
```

## 2. Datasets

We use HuggingFace `datasets`:
- HotpotQA: `hotpot_qa` (config `distractor`)
- 2WikiMultiHopQA: `framolfese/2WikiMultihopQA`

Default cache is `~/.cache/huggingface`; override if needed:
```bash
export HF_HOME=/path/to/hf_cache
```

Paper validation sizes:
- HotpotQA: 7,405 queries / 66,581 docs
- 2WikiMultiHopQA: 10,000 queries / 45,902 docs

## 3. Main Results (Tables)

Set a common tag and sample sizes first:
```bash
export TAG=xxx
export HOTPOT_N=7405
export TWOWIKI_N=10000
```

Choose your output locations (set any paths you like):
```bash
export HOTPOT_LEX_DIR="<path>"
export HOTPOT_DENSE_DIR="<path>"
export HOTPOT_RRF_DIR="<path>"
export HOTPOT_GRAPH_DIR="<path>"
export TWOWIKI_LEX_DIR="<path>"
export TWOWIKI_DENSE_DIR="<path>"
export TWOWIKI_RRF_DIR="<path>"
export TWOWIKI_GRAPH_DIR="<path>"
```

### 3.1 HotpotQA (main)

**(a) Lexical baselines**
```bash
sprig run --dataset hotpotqa --split validation --max-samples $HOTPOT_N \
  --methods bm25 rm3 bm25_2step \
  --tag $TAG --output $HOTPOT_LEX_DIR
```

**(b) Dense baseline (bge-small)**
```bash
sprig run --dataset hotpotqa --split validation --max-samples $HOTPOT_N \
  --methods dense \
  --dense-model BAAI/bge-small-en-v1.5 \
  --tag $TAG --output $HOTPOT_DENSE_DIR
```

**(c) RRF / Rerank**
```bash
sprig run --dataset hotpotqa --split validation --max-samples $HOTPOT_N \
  --methods rrf rerank rrf_rerank \
  --dense-model BAAI/bge-small-en-v1.5 \
  --rrf-k 60 \
  --tag $TAG --output $HOTPOT_RRF_DIR
```

**(d) Graph family (GraphHybrid/GraphDense/GraphRRF, etc.)**
```bash
sprig run --dataset hotpotqa --split validation --max-samples $HOTPOT_N \
  --methods tfidf_graph graph graph_hybrid graph_dense graph_rrf rrf_ppr_fusion graph_bm25_fallback \
  --dense-model BAAI/bge-small-en-v1.5 \
  --graph-ner spacy \
  --graph-entity-normalize simple \
  --graph-seed-weighting rank \
  --graph-hub-penalty 0.5 \
  --graph-seed-entity-df-power 0.5 \
  --graph-seed-docs-k 5 \
  --graph-seed-docs-k-bm25 10 \
  --graph-seed-docs-k-rrf 10 \
  --graph-fallback-k 1 \
  --ppr-alpha 0.15 --ppr-max-iter 5 --ppr-mode push \
  --term-min-df 3 --term-max-df-ratio 0.1 --term-norm row \
  --hnsw-m 32 --hnsw-ef-construction 200 --hnsw-ef-search 64 \
  --tag $TAG --output $HOTPOT_GRAPH_DIR
```

### 3.2 2WikiMultiHopQA (main)

**(a) Lexical baselines**
```bash
sprig run --dataset 2wikimultihopqa --split validation --max-samples $TWOWIKI_N \
  --methods bm25 rm3 bm25_2step \
  --tag $TAG --output $TWOWIKI_LEX_DIR
```

**(b) Dense baseline (bge-small)**
```bash
sprig run --dataset 2wikimultihopqa --split validation --max-samples $TWOWIKI_N \
  --methods dense \
  --dense-model BAAI/bge-small-en-v1.5 \
  --tag $TAG --output $TWOWIKI_DENSE_DIR
```

**(c) RRF / Rerank**
```bash
sprig run --dataset 2wikimultihopqa --split validation --max-samples $TWOWIKI_N \
  --methods rrf rerank rrf_rerank \
  --dense-model BAAI/bge-small-en-v1.5 \
  --rrf-k 60 \
  --tag $TAG --output $TWOWIKI_RRF_DIR
```

**(d) Graph family**
```bash
sprig run --dataset 2wikimultihopqa --split validation --max-samples $TWOWIKI_N \
  --methods tfidf_graph graph graph_hybrid graph_dense graph_rrf rrf_ppr_fusion graph_bm25_fallback \
  --dense-model BAAI/bge-small-en-v1.5 \
  --graph-ner spacy \
  --graph-entity-normalize lower \
  --graph-seed-weighting rank \
  --graph-hub-penalty 0.5 \
  --graph-seed-entity-df-power 1.0 \
  --graph-seed-docs-k 3 \
  --graph-seed-docs-k-bm25 5 \
  --graph-seed-docs-k-rrf 5 \
  --graph-fallback-k 1 \
  --ppr-alpha 0.15 --ppr-max-iter 5 --ppr-mode power \
  --term-min-df 3 --term-max-df-ratio 0.1 --term-norm row \
  --hnsw-m 32 --hnsw-ef-construction 200 --hnsw-ef-search 64 \
  --tag $TAG --output $TWOWIKI_GRAPH_DIR
```

## 4. Auxiliary Analysis (required before exporting tables)

### 4.1 Merge runs (for QA / significance)
```bash
export HOTPOT_MERGED_DIR="<path>"
export TWOWIKI_MERGED_DIR="<path>"

python3 scripts/merge_runs.py --output $HOTPOT_MERGED_DIR \
  --runs $HOTPOT_LEX_DIR $HOTPOT_DENSE_DIR \
         $HOTPOT_RRF_DIR $HOTPOT_GRAPH_DIR

python3 scripts/merge_runs.py --output $TWOWIKI_MERGED_DIR \
  --runs $TWOWIKI_LEX_DIR $TWOWIKI_DENSE_DIR \
         $TWOWIKI_RRF_DIR $TWOWIKI_GRAPH_DIR
```

### 4.2 QA eval (Appendix QA table)
```bash
python3 scripts/run_qa_eval.py --run-dir $HOTPOT_MERGED_DIR \
  --dataset hotpotqa --split validation \
  --method bm25 rrf rerank graph graph_hybrid graph_dense \
  --top-k 5 --limit-queries 1000 --seed 42

python3 scripts/run_qa_eval.py --run-dir $TWOWIKI_MERGED_DIR \
  --dataset 2wikimultihopqa --split validation \
  --method bm25 rrf rerank graph graph_hybrid graph_dense \
  --top-k 5 --limit-queries 1000 --seed 42
```

Copy `qa_metrics.json` into the main run dirs (table export expects them there):
```bash
cp $HOTPOT_MERGED_DIR/qa_metrics.json $HOTPOT_GRAPH_DIR/qa_metrics.json
cp $TWOWIKI_MERGED_DIR/qa_metrics.json $TWOWIKI_GRAPH_DIR/qa_metrics.json
```

### 4.3 NER proxy (Appendix NER table)
```bash
python3 scripts/ner_proxy_eval.py --run-dir $HOTPOT_GRAPH_DIR \
  --method graph graph_hybrid --ner-mode spacy regex --entity-normalize simple

python3 scripts/ner_proxy_eval.py --run-dir $TWOWIKI_GRAPH_DIR \
  --method graph graph_hybrid --ner-mode spacy regex --entity-normalize lower
```

### 4.4 Hub pruning coverage (Appendix Hub table)
```bash
python3 scripts/hub_pruning_analysis.py --dataset hotpotqa --split validation --max-samples $HOTPOT_N \
  --graph-ner spacy --graph-entity-normalize simple \
  --graph-hub-top-ratio 0.01 \
  --output $HOTPOT_GRAPH_DIR/hub_pruning.json

python3 scripts/hub_pruning_analysis.py --dataset 2wikimultihopqa --split validation --max-samples $TWOWIKI_N \
  --graph-ner spacy --graph-entity-normalize lower \
  --graph-hub-top-ratio 0.01 \
  --output $TWOWIKI_GRAPH_DIR/hub_pruning.json
```

## 5. Export paper tables

Generate summary CSV + LaTeX tables:
```bash
python3 scripts/summarize_results.py
python3 scripts/export_paper_tables.py --tag $TAG --supp-tag $TAG --ann-tag $TAG --dense-tag $TAG
```

## 6. Significance tests (Appendix Significance tables)

```bash
export SIG_HOTPOT_JSON="<path>"
export SIG_TWOWIKI_JSON="<path>"
export SIG_TABLE_DIR="<path>"

python3 scripts/bootstrap_significance_all.py --run-dir $HOTPOT_MERGED_DIR \
  --baseline bm25 rrf --k 10 --iters 1000 \
  --output $SIG_HOTPOT_JSON

python3 scripts/bootstrap_significance_all.py --run-dir $TWOWIKI_MERGED_DIR \
  --baseline bm25 rrf --k 10 --iters 1000 \
  --output $SIG_TWOWIKI_JSON

python3 scripts/export_significance_tables.py \
  --input $SIG_HOTPOT_JSON \
  --output $SIG_TABLE_DIR/significance_hotpotqa_bm25.tex \
  --baseline bm25 --metric recall --top 10

python3 scripts/export_significance_tables.py \
  --input $SIG_HOTPOT_JSON \
  --output $SIG_TABLE_DIR/significance_hotpotqa_rrf.tex \
  --baseline rrf --metric recall --top 10

python3 scripts/export_significance_tables.py \
  --input $SIG_TWOWIKI_JSON \
  --output $SIG_TABLE_DIR/significance_2wiki_bm25.tex \
  --baseline bm25 --metric recall --top 10

python3 scripts/export_significance_tables.py \
  --input $SIG_TWOWIKI_JSON \
  --output $SIG_TABLE_DIR/significance_2wiki_rrf.tex \
  --baseline rrf --metric recall --top 10
```

## 7. Efficiency & Scalability (Figure + Appendix table)

```bash
python3 scripts/run_efficiency.py --dataset hotpotqa --split validation \
  --sizes 200 1000 3000 7405 \
  --methods bm25 dense graph graph_dense \
  --graph-ner spacy --graph-entity-normalize simple \
  --graph-seed-docs-k 5 --graph-seed-docs-k-bm25 10 \
  --graph-seed-weighting rank --graph-hub-penalty 0.5 \
  --graph-seed-entity-df-power 0.5 \
  --ppr-alpha 0.15 --ppr-max-iter 5 --ppr-mode push \
  --dense-model BAAI/bge-small-en-v1.5 \
  --tag eff2

python3 scripts/run_efficiency.py --dataset 2wikimultihopqa --split validation \
  --sizes 200 1000 3000 10000 \
  --methods bm25 dense graph graph_dense \
  --graph-ner spacy --graph-entity-normalize lower \
  --graph-seed-docs-k 3 --graph-seed-docs-k-bm25 5 \
  --graph-seed-weighting rank --graph-hub-penalty 0.5 \
  --graph-seed-entity-df-power 1.0 \
  --ppr-alpha 0.15 --ppr-max-iter 5 --ppr-mode power \
  --dense-model BAAI/bge-small-en-v1.5 \
  --tag eff2

python3 scripts/plot_efficiency.py
```

Per-doc table (choose any output path):
```bash
python3 scripts/efficiency_per_doc.py --summary-csv <path> --tag eff2 --output <path>
```

## 8. Ablations

### 8.1 Graph/PPR ablations (500-query subsets)
```bash
export ABLATION_HOTPOT_DIR="<path>"
export ABLATION_TWOWIKI_DIR="<path>"
export ABLATION_PLOT_DIR="<path>"

python3 scripts/run_ablation.py --dataset hotpotqa --split validation \
  --max-samples 2000 --limit-queries 500 \
  --seed-source dense --ner spacy regex \
  --seed-k 1 3 5 10 \
  --seed-weighting raw softmax rank \
  --alpha 0.1 0.15 0.2 --iter 5 10 20 \
  --ppr-mode power push \
  --output $ABLATION_HOTPOT_DIR

python3 scripts/run_ablation.py --dataset 2wikimultihopqa --split validation \
  --max-samples 2000 --limit-queries 500 \
  --seed-source dense --ner spacy regex \
  --seed-k 1 3 5 10 \
  --seed-weighting raw softmax rank \
  --alpha 0.1 0.15 0.2 --iter 5 10 20 \
  --ppr-mode power push \
  --output $ABLATION_TWOWIKI_DIR

python3 scripts/plot_ablation.py --inputs \
  $ABLATION_HOTPOT_DIR/ablation_results.csv \
  $ABLATION_TWOWIKI_DIR/ablation_results.csv \
  --output $ABLATION_PLOT_DIR
```

### 8.2 TF-IDF Term Graph ablation
```bash
export ABLATION_TERM_HOTPOT_DIR="<path>"
export ABLATION_TERM_TWOWIKI_DIR="<path>"

python3 scripts/run_term_graph_ablation.py --dataset hotpotqa --split validation \
  --max-samples 2000 --limit-queries 500 \
  --min-df 3 5 10 --max-df-ratio 0.1 0.2 0.3 \
  --output $ABLATION_TERM_HOTPOT_DIR

python3 scripts/run_term_graph_ablation.py --dataset 2wikimultihopqa --split validation \
  --max-samples 2000 --limit-queries 500 \
  --min-df 3 5 10 --max-df-ratio 0.1 0.2 0.3 \
  --output $ABLATION_TERM_TWOWIKI_DIR
```

### 8.3 Ablation Top-10 tables
`run_ablation.py` writes `ablation_top10.json`; convert to LaTeX:
```bash
python3 scripts/export_ablation_top10.py --input <path> --output <path>
```

## 9. Dense Seeding / ANN / Model Sensitivity

Choose output locations for the runs below:
```bash
export HOTPOT_GRAPHDENSE_2K_EXACT_DIR="<path>"
export HOTPOT_GRAPHDENSE_2K_ANN_DIR="<path>"
export TWOWIKI_GRAPHDENSE_2K_EXACT_DIR="<path>"
export TWOWIKI_GRAPHDENSE_2K_ANN_DIR="<path>"
export HOTPOT_GRAPHDENSE_HNSW_DIR="<path>"
export TWOWIKI_GRAPHDENSE_HNSW_DIR="<path>"
export HOTPOT_DENSE_SENS_DIR="<path>"
export TWOWIKI_DENSE_SENS_DIR="<path>"
```

### 9.1 GraphDense: Exact vs ANN (2k subset)
```bash
sprig run --dataset hotpotqa --split validation --max-samples 2000 \
  --methods graph_dense --dense-model BAAI/bge-small-en-v1.5 \
  --graph-ner spacy --graph-entity-normalize simple \
  --graph-seed-docs-k 5 \
  --graph-seed-weighting rank --graph-hub-penalty 0.5 --graph-seed-entity-df-power 0.5 \
  --ppr-alpha 0.15 --ppr-max-iter 5 --ppr-mode push \
  --dense-no-hnsw \
  --tag $TAG --output $HOTPOT_GRAPHDENSE_2K_EXACT_DIR

sprig run --dataset hotpotqa --split validation --max-samples 2000 \
  --methods graph_dense --dense-model BAAI/bge-small-en-v1.5 \
  --graph-ner spacy --graph-entity-normalize simple \
  --graph-seed-docs-k 5 \
  --graph-seed-weighting rank --graph-hub-penalty 0.5 --graph-seed-entity-df-power 0.5 \
  --ppr-alpha 0.15 --ppr-max-iter 5 --ppr-mode push \
  --hnsw-m 32 --hnsw-ef-construction 200 --hnsw-ef-search 64 \
  --tag $TAG --output $HOTPOT_GRAPHDENSE_2K_ANN_DIR

sprig run --dataset 2wikimultihopqa --split validation --max-samples 2000 \
  --methods graph_dense --dense-model BAAI/bge-small-en-v1.5 \
  --graph-ner spacy --graph-entity-normalize lower \
  --graph-seed-docs-k 3 \
  --graph-seed-weighting rank --graph-hub-penalty 0.5 --graph-seed-entity-df-power 1.0 \
  --ppr-alpha 0.15 --ppr-max-iter 5 --ppr-mode power \
  --dense-no-hnsw \
  --tag $TAG --output $TWOWIKI_GRAPHDENSE_2K_EXACT_DIR

sprig run --dataset 2wikimultihopqa --split validation --max-samples 2000 \
  --methods graph_dense --dense-model BAAI/bge-small-en-v1.5 \
  --graph-ner spacy --graph-entity-normalize lower \
  --graph-seed-docs-k 3 \
  --graph-seed-weighting rank --graph-hub-penalty 0.5 --graph-seed-entity-df-power 1.0 \
  --ppr-alpha 0.15 --ppr-max-iter 5 --ppr-mode power \
  --hnsw-m 32 --hnsw-ef-construction 200 --hnsw-ef-search 64 \
  --tag $TAG --output $TWOWIKI_GRAPHDENSE_2K_ANN_DIR
```

### 9.2 HNSW grid (2k subset)
```bash
for m in 16 32 64; do
  for efs in 32 64 128 256; do
    sprig run --dataset hotpotqa --split validation --max-samples 2000 \
      --methods graph_dense --dense-model BAAI/bge-small-en-v1.5 \
      --graph-ner spacy --graph-entity-normalize simple \
      --graph-seed-docs-k 5 \
      --graph-seed-weighting rank --graph-hub-penalty 0.5 --graph-seed-entity-df-power 0.5 \
      --ppr-alpha 0.15 --ppr-max-iter 5 --ppr-mode push \
      --hnsw-m $m --hnsw-ef-construction 200 --hnsw-ef-search $efs \
      --tag $TAG --output $HOTPOT_GRAPHDENSE_HNSW_DIR/m${m}_efs${efs}

    sprig run --dataset 2wikimultihopqa --split validation --max-samples 2000 \
      --methods graph_dense --dense-model BAAI/bge-small-en-v1.5 \
      --graph-ner spacy --graph-entity-normalize lower \
      --graph-seed-docs-k 3 \
      --graph-seed-weighting rank --graph-hub-penalty 0.5 --graph-seed-entity-df-power 1.0 \
      --ppr-alpha 0.15 --ppr-max-iter 5 --ppr-mode power \
      --hnsw-m $m --hnsw-ef-construction 200 --hnsw-ef-search $efs \
      --tag $TAG --output $TWOWIKI_GRAPHDENSE_HNSW_DIR/m${m}_efs${efs}
  done
done
```

### 9.3 Dense model sensitivity (2k subset)
```bash
for model in BAAI/bge-small-en-v1.5 sentence-transformers/all-MiniLM-L6-v2 intfloat/e5-small-v2; do
  sprig run --dataset hotpotqa --split validation --max-samples 2000 \
    --methods dense graph_dense --dense-model $model \
    --graph-ner spacy --graph-entity-normalize simple \
    --graph-seed-docs-k 5 \
    --graph-seed-weighting rank --graph-hub-penalty 0.5 --graph-seed-entity-df-power 0.5 \
    --ppr-alpha 0.15 --ppr-max-iter 5 --ppr-mode push \
    --tag $TAG --output $HOTPOT_DENSE_SENS_DIR/${model##*/}

  sprig run --dataset 2wikimultihopqa --split validation --max-samples 2000 \
    --methods dense graph_dense --dense-model $model \
    --graph-ner spacy --graph-entity-normalize lower \
    --graph-seed-docs-k 3 \
    --graph-seed-weighting rank --graph-hub-penalty 0.5 --graph-seed-entity-df-power 1.0 \
    --ppr-alpha 0.15 --ppr-max-iter 5 --ppr-mode power \
    --tag $TAG --output $TWOWIKI_DENSE_SENS_DIR/${model##*/}
done
```

## 10. SPRIG-EL/PRUNE/MIX Enhancements

GraphHybrid comparison on full and q1000:
- Base: no enhancements
- +EL: `--graph-use-aliases`
- +PRUNE: `--graph-hub-top-ratio 0.01`
- +MIX: `--graph-seed-mix-mode auto`
- +ALL: all three combined

Choose output locations for enhancement runs:
```bash
export HOTPOT_ENH_BASE_DIR="<path>"
export HOTPOT_ENH_ALL_DIR="<path>"
export TWOWIKI_ENH_BASE_DIR="<path>"
export TWOWIKI_ENH_ALL_DIR="<path>"
```

Example (HotpotQA, full):
```bash
sprig run --dataset hotpotqa --split validation --max-samples $HOTPOT_N \
  --methods graph_hybrid --dense-model BAAI/bge-small-en-v1.5 \
  --graph-ner spacy --graph-entity-normalize simple \
  --graph-seed-docs-k 5 --graph-seed-docs-k-bm25 10 \
  --graph-seed-weighting rank --graph-hub-penalty 0.5 --graph-seed-entity-df-power 0.5 \
  --ppr-alpha 0.15 --ppr-max-iter 5 --ppr-mode push \
  --tag $TAG --output $HOTPOT_ENH_BASE_DIR

sprig run --dataset hotpotqa --split validation --max-samples $HOTPOT_N \
  --methods graph_hybrid --dense-model BAAI/bge-small-en-v1.5 \
  --graph-ner spacy --graph-entity-normalize simple \
  --graph-seed-docs-k 5 --graph-seed-docs-k-bm25 10 \
  --graph-seed-weighting rank --graph-hub-penalty 0.5 --graph-seed-entity-df-power 0.5 \
  --graph-use-aliases --graph-hub-top-ratio 0.01 --graph-seed-mix-mode auto \
  --ppr-alpha 0.15 --ppr-max-iter 5 --ppr-mode push \
  --tag $TAG --output $HOTPOT_ENH_ALL_DIR
```

Other variants are produced by adding `--graph-use-aliases` / `--graph-hub-top-ratio 0.01`
/ `--graph-seed-mix-mode auto` onto the Base command; for 2Wiki use the parameters from
Section 3.2 (e.g., `--graph-entity-normalize lower`, `--graph-seed-docs-k 3`, `--graph-seed-docs-k-bm25 5`).
q1000 is the same with `--max-samples 1000`.

Summarize GraphHybrid R@10/QTime into a table:
```bash
python3 scripts/summarize_graph_enhancements.py \
  --hotpot-base <path> \
  --hotpot-all <path> \
  --twowiki-base <path> \
  --twowiki-all <path> \
  --output <path>
```

## 11. Robustness (w/o tune)

The command below removes 500 tuning queries from full validation:
```bash
python3 scripts/robustness_no_tune.py \
  --hotpot-run <path> --hotpot-n 7405 \
  --twowiki-run <path> --twowiki-n 10000 \
  --output <path>
```
