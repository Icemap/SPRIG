from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to ablation_top10.json")
    p.add_argument("--output", required=True, help="Output LaTeX table path")
    p.add_argument("--method-default", default="graph_dense")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))

    rows = [
        "\\begin{tabular}{lrrrr}",
        "Method & R@1 & R@5 & R@10 & MRR \\\\ ",
        "\\hline",
    ]
    for r in data:
        rows.append(
            f"{r.get('method', args.method_default)} & {float(r.get('recall@1', 0)):.3f} "
            f"& {float(r.get('recall@5', 0)):.3f} & {float(r.get('recall@10', 0)):.3f} "
            f"& {float(r.get('mrr', 0)):.3f} \\\\ "
        )
    rows.append("\\end{tabular}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rows), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
