from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--metric", default="recall")
    p.add_argument("--baseline", default="bm25")
    p.add_argument("--top", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    comps = [
        c
        for c in payload.get("comparisons", [])
        if c.get("metric") == args.metric and c.get("baseline") == args.baseline
    ]
    comps = sorted(comps, key=lambda c: c.get("mean_diff", 0.0), reverse=True)
    def _tex_escape(text: str) -> str:
        return text.replace("_", "\\_")

    rows = ["Method & $\\Delta$ & 95\\% CI & Wins/Ties/Losses \\\\", "\\hline"]
    for c in comps[: args.top]:
        ci = c.get("ci_95", [0.0, 0.0])
        method = _tex_escape(str(c["method"]))
        rows.append(
            f"{method} & {c['mean_diff']:.4f} & [{ci[0]:.4f}, {ci[1]:.4f}] & "
            f"{c['wins']}/{c['ties']}/{c['losses']} \\\\"
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lrrr}\n")
        for row in rows:
            f.write(row + "\n")
        f.write("\\end{tabular}\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
