from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--hotpot-base", required=True)
    p.add_argument("--hotpot-all", required=True)
    p.add_argument("--twowiki-base", required=True)
    p.add_argument("--twowiki-all", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--method", default="graph_hybrid")
    p.add_argument("--base-label", default="Base")
    p.add_argument("--all-label", default="+ALL")
    return p.parse_args()


def _load_metrics(run_dir: str, method: str) -> tuple[float, float]:
    data = json.loads(Path(run_dir, "metrics.json").read_text(encoding="utf-8"))
    results = [r for r in data.get("results", []) if r.get("method") == method]
    if not results:
        raise ValueError(f"Method {method} not found in {run_dir}")
    result = results[0]
    r10 = float(result["metrics"]["recall_at_k"]["10"])
    qtime = float(result["query_time_sec"])
    return r10, qtime


def main() -> None:
    args = parse_args()
    hotpot_base = _load_metrics(args.hotpot_base, args.method)
    hotpot_all = _load_metrics(args.hotpot_all, args.method)
    twowiki_base = _load_metrics(args.twowiki_base, args.method)
    twowiki_all = _load_metrics(args.twowiki_all, args.method)

    rows = [
        "\\begin{tabular}{lrrrr}",
        "Variant & Hotpot R@10 & Hotpot QTime (s) & 2Wiki R@10 & 2Wiki QTime (s) \\\\ ",
        "\\hline",
    ]
    rows.append(
        f"{args.base_label} & {hotpot_base[0]:.3f} & {hotpot_base[1]:.1f} "
        f"& {twowiki_base[0]:.3f} & {twowiki_base[1]:.1f} \\\\ "
    )
    rows.append(
        f"{args.all_label} & {hotpot_all[0]:.3f} & {hotpot_all[1]:.1f} "
        f"& {twowiki_all[0]:.3f} & {twowiki_all[1]:.1f} \\\\ "
    )
    rows.append("\\end{tabular}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rows), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
