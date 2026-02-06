# SPRIG 复现指南

中文 | [English](./README.md)

本仓库包含 SPRIG 论文的实验代码与脚本。以下步骤从零开始复现主文与附录中的
所有表格/图（含效率、消融、显著性检验、QA 评估等）。默认使用 CPU-only 环境。

## 1. 环境与安装

### 硬件/系统
- Python >= 3.12
- CPU-only（论文使用 4GB 内存配置）
- Linux/macOS 均可

### 安装依赖
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[vector,data,plot,qa,dev]"
python3 -m spacy download en_core_web_sm
```

## 2. 数据集

使用 HuggingFace `datasets`：
- HotpotQA: `hotpot_qa`（配置 `distractor`）
- 2WikiMultiHopQA: `framolfese/2WikiMultihopQA`

默认缓存到 `~/.cache/huggingface`，可用环境变量指定：
```bash
export HF_HOME=/path/to/hf_cache
```

论文使用的数据规模（validation）：
- HotpotQA：7,405 queries / 66,581 docs
- 2WikiMultiHopQA：10,000 queries / 45,902 docs

## 3. 主结果复现（主表）

建议先设置统一的标签与样本规模：

```bash
export TAG=xxx
export HOTPOT_N=7405
export TWOWIKI_N=10000
```

自定义输出目录（路径随意）：
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

### 3.1 HotpotQA（主结果）

**(a) 词法基线**

```bash
sprig run --dataset hotpotqa --split validation --max-samples $HOTPOT_N \
  --methods bm25 rm3 bm25_2step \
  --tag $TAG --output $HOTPOT_LEX_DIR
```

**(b) Dense 基线（bge-small）**

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

**(d) Graph 系列（含 GraphHybrid/GraphDense/GraphRRF 等）**

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

### 3.2 2WikiMultiHopQA（主结果）

**(a) 词法基线**
```bash
sprig run --dataset 2wikimultihopqa --split validation --max-samples $TWOWIKI_N \
  --methods bm25 rm3 bm25_2step \
  --tag $TAG --output $TWOWIKI_LEX_DIR
```

**(b) Dense 基线（bge-small）**
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

**(d) Graph 系列**
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

## 4. 辅助分析（导出表格前必须生成）

### 4.1 合并运行（QA/显著性用）
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

### 4.2 QA 评估（Appendix QA 表）
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

将 `qa_metrics.json` 放到主 run 目录（供导表脚本读取）：
```bash
cp $HOTPOT_MERGED_DIR/qa_metrics.json $HOTPOT_GRAPH_DIR/qa_metrics.json
cp $TWOWIKI_MERGED_DIR/qa_metrics.json $TWOWIKI_GRAPH_DIR/qa_metrics.json
```

### 4.3 NER Proxy（Appendix NER 表）
```bash
python3 scripts/ner_proxy_eval.py --run-dir $HOTPOT_GRAPH_DIR \
  --method graph graph_hybrid --ner-mode spacy regex --entity-normalize simple

python3 scripts/ner_proxy_eval.py --run-dir $TWOWIKI_GRAPH_DIR \
  --method graph graph_hybrid --ner-mode spacy regex --entity-normalize lower
```

### 4.4 Hub Pruning 覆盖率（Appendix Hub 表）
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

## 5. 导出论文表格

生成汇总 CSV + LaTeX 表格：
```bash
python3 scripts/summarize_results.py
python3 scripts/export_paper_tables.py --tag $TAG --supp-tag $TAG --ann-tag $TAG --dense-tag $TAG
```

## 6. 显著性检验（Appendix Significance 表）

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

## 7. 效率与扩展性（Figure + Appendix 表）

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

生成 per-doc 表格（输出路径自行指定）。`SUMMARY_CSV` 指向 `scripts/summarize_results.py`
生成的 `all_results.csv`：
```bash
export SUMMARY_CSV="<path>"
export EFF_TABLE_PATH="<path>"

python3 - <<'PY'
import os
import pandas as pd
from pathlib import Path
df = pd.read_csv(os.environ["SUMMARY_CSV"])
df = df[df["tag"] == "eff2"]
rows = []
for dataset in sorted(df["dataset"].dropna().unique()):
    for method in ["bm25", "dense", "graph", "graph_dense"]:
        sub = df[(df["dataset"] == dataset) & (df["method"] == method)]
        if sub.empty:
            continue
        per_doc = (sub["index_time_sec"] / sub["docs"]) * 1000.0
        rows.append((dataset, method, per_doc.median(), per_doc.min(), per_doc.max()))
out = Path(os.environ["EFF_TABLE_PATH"])
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as f:
    f.write("\\\\begin{tabular}{llrr}\\n")
    f.write("Dataset & Method & Median ms/doc & Min--Max ms/doc \\\\\\\\ \\n")
    f.write("\\\\hline\\n")
    for dataset, method, med, lo, hi in rows:
        f.write(f"{dataset} & {method} & {med:.3f} & [{lo:.3f}, {hi:.3f}] \\\\\\\\ \\n")
    f.write("\\\\end{tabular}\\n")
print(f\"Wrote {out}\")
PY
```

## 8. 消融实验（Ablations）

### 8.1 Graph/PPR 消融（500-query 子集）
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

### 8.2 TF-IDF Term Graph 消融
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

### 8.3 Ablation Top-10 表格
`run_ablation.py` 会生成 `ablation_top10.json`，可用以下脚本转为 LaTeX：
```bash
export ABLATION_TEX_DIR="<path>"

python3 - <<'PY'
import os
import json
from pathlib import Path
def to_tex(path, out):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = ["\\\\begin{tabular}{lrrrr}", "Method & R@1 & R@5 & R@10 & MRR \\\\\\\\ ", "\\\\hline"]
    for r in data:
        rows.append(
            f"{r.get('method','graph_dense')} & {float(r.get('recall@1',0)):.3f} "
            f"& {float(r.get('recall@5',0)):.3f} & {float(r.get('recall@10',0)):.3f} "
            f"& {float(r.get('mrr',0)):.3f} \\\\\\\\ "
        )
    rows.append("\\\\end{tabular}")
    Path(out).write_text("\\n".join(rows), encoding="utf-8")
out_dir = Path(os.environ["ABLATION_TEX_DIR"])
out_dir.mkdir(parents=True, exist_ok=True)
to_tex(Path(os.environ["ABLATION_HOTPOT_DIR"]) / "ablation_top10.json", out_dir / "ablation_hotpotqa_top10.tex")
to_tex(Path(os.environ["ABLATION_TWOWIKI_DIR"]) / "ablation_top10.json", out_dir / "ablation_2wikimultihopqa_top10.tex")
PY
```

## 9. Dense Seeding / ANN / 模型敏感性

输出目录自定义：
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

### 9.1 GraphDense: Exact vs ANN（2k 子集）
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

### 9.2 HNSW 参数网格（2k 子集）
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

### 9.3 Dense 模型敏感性（2k 子集）
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

## 10. SPRIG-EL/PRUNE/MIX 增强表

对 GraphHybrid 在 full 与 q1000 子集上做如下对比：
- Base：无增强
- +EL：`--graph-use-aliases`
- +PRUNE：`--graph-hub-top-ratio 0.01`
- +MIX：`--graph-seed-mix-mode auto`
- +ALL：三者叠加

增强实验输出目录自定义：
```bash
export HOTPOT_ENH_BASE_DIR="<path>"
export HOTPOT_ENH_ALL_DIR="<path>"
export TWOWIKI_ENH_BASE_DIR="<path>"
export TWOWIKI_ENH_ALL_DIR="<path>"
export ENH_TABLE_PATH="<path>"
```

示例（HotpotQA, full）：
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

其余变体在 Base 命令上分别加入 `--graph-use-aliases` / `--graph-hub-top-ratio 0.01`
/ `--graph-seed-mix-mode auto` 即可；2Wiki 需将参数替换为 3.2 节对应设置
（`--graph-entity-normalize lower`、`--graph-seed-docs-k 3`、`--graph-seed-docs-k-bm25 5` 等）。
q1000 同理，`--max-samples 1000`。

将各 run 的 `metrics.json` 中 GraphHybrid 的 R@10/QTime 汇总为
自定义表格路径（q1000 同理）：
```bash
python3 - <<'PY'
import json
import os
from pathlib import Path
def row(run_dir, label):
    data = json.loads(Path(run_dir, "metrics.json").read_text(encoding="utf-8"))
    res = [r for r in data["results"] if r["method"] == "graph_hybrid"][0]
    return label, res["metrics"]["recall_at_k"]["10"], res["query_time_sec"]

rows_hotpot = [
    row(os.environ["HOTPOT_ENH_BASE_DIR"], "Base"),
    row(os.environ["HOTPOT_ENH_ALL_DIR"], "+ALL"),
]
rows_2wiki = [
    row(os.environ["TWOWIKI_ENH_BASE_DIR"], "Base"),
    row(os.environ["TWOWIKI_ENH_ALL_DIR"], "+ALL"),
]

out = Path(os.environ["ENH_TABLE_PATH"])
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as f:
    f.write("\\\\begin{tabular}{lrrrr}\\n")
    f.write("Variant & Hotpot R@10 & Hotpot QTime (s) & 2Wiki R@10 & 2Wiki QTime (s) \\\\\\\\ \\n")
    f.write("\\\\hline\\n")
    for (label, r10_h, qt_h), (_, r10_w, qt_w) in zip(rows_hotpot, rows_2wiki):
        f.write(f"{label} & {r10_h:.3f} & {qt_h:.1f} & {r10_w:.3f} & {qt_w:.1f} \\\\\\\\ \\n")
    f.write("\\\\end{tabular}\\n")
print(f\"Wrote {out}\")
PY
```

## 11. Robustness (w/o tune) 表

下述脚本会在 full validation 上剔除 500 条 tuning queries：
```bash
python3 - <<'PY'
import os
import json, random
from pathlib import Path
from sprig.data.hotpotqa import load_hotpotqa
from sprig.data.twowiki import load_twowiki
from sprig.eval import evaluate

def run(dataset, run_dir, max_samples, seed=42, tune_n=500):
    if dataset == "hotpotqa":
        docs, queries = load_hotpotqa(split="validation", max_samples=max_samples, seed=seed)
    else:
        docs, queries = load_twowiki(split="validation", max_samples=max_samples, seed=seed)
    qids = [q.qid for q in queries]
    random.seed(seed)
    tune = set(random.sample(qids, min(tune_n, len(qids))))
    gold = {q.qid: list(q.gold_titles) for q in queries if q.qid not in tune}
    retrieved = json.loads(Path(run_dir, "retrieved.json").read_text(encoding="utf-8"))
    metrics = {}
    for method in ["bm25", "dense", "rrf", "graph_hybrid", "graph_dense"]:
        m = evaluate(gold, retrieved.get(method, {}), ks=[10])
        metrics[method] = {"recall@10": m.recall_at_k[10], "mrr": m.mrr}
    return metrics

hotpot = run("hotpotqa", os.environ["HOTPOT_MERGED_DIR"], 7405)
twowiki = run("2wikimultihopqa", os.environ["TWOWIKI_MERGED_DIR"], 10000)
print("hotpot", hotpot)
print("2wiki", twowiki)
PY
```
