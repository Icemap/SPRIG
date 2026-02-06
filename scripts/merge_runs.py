from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--methods", nargs="+", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    retrieved: Dict[str, Dict[str, List[str]]] = {}
    docs_written = False
    queries_written = False

    for run_path in args.runs:
        run_dir = Path(run_path)
        if not docs_written and (run_dir / "docs.jsonl").exists():
            (out_dir / "docs.jsonl").write_text(
                (run_dir / "docs.jsonl").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            docs_written = True
        if not queries_written and (run_dir / "queries.jsonl").exists():
            (out_dir / "queries.jsonl").write_text(
                (run_dir / "queries.jsonl").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            queries_written = True

        retrieved_path = run_dir / "retrieved.json"
        if not retrieved_path.exists():
            continue
        data = json.loads(retrieved_path.read_text(encoding="utf-8"))
        for method, entries in data.items():
            if args.methods and method not in args.methods:
                continue
            retrieved[method] = entries

    if not docs_written or not queries_written:
        raise SystemExit("Missing docs.jsonl or queries.jsonl from inputs.")

    (out_dir / "retrieved.json").write_text(
        json.dumps(retrieved, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Merged {len(retrieved)} methods into {out_dir}")


if __name__ == "__main__":
    main()
