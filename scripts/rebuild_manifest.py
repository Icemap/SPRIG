from __future__ import annotations

import json
from pathlib import Path

root = Path("outputs")
manifest_path = root / "manifest.jsonl"

entries = []
for metrics_path in sorted(root.glob("*/metrics.json")):
    run_dir = metrics_path.parent
    meta_path = run_dir / "run_meta.json"
    if meta_path.exists():
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        config = payload.get("config", {})
        results = payload.get("results", [])
        docs = results[0].get("docs") if results else None
        queries = results[0].get("queries") if results else None
        data = {**config, "docs": docs, "queries": queries, "output_dir": str(run_dir)}
    entries.append(data)

manifest_path.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries), encoding="utf-8")
print(f"Wrote {len(entries)} entries to {manifest_path}")
