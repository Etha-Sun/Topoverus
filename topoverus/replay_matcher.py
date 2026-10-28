"""Replay a frozen legacy matcher against v2 without importing research code."""

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path

from .checkpoint import file_hash
from .data import read_jsonl, write_jsonl
from .graph import dependency_score
from .matcher_v2 import VERSION


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--graphs", type=Path, required=True, help="Explicit frozen legacy graph input")
    parser.add_argument("--legacy-module", type=Path, required=True, help="Frozen v1 module copied into this replay run")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if not args.output.resolve().is_relative_to(root / "runs") or not args.legacy_module.resolve().is_relative_to(root / "runs"):
        raise ValueError("Baseline snapshot and replay outputs belong in release runs/")
    args.output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location("frozen_v1", args.legacy_module)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    graphs = {}
    for record in read_jsonl(args.graphs):
        key = record["unverified_rel_path"]
        if key in graphs:
            raise ValueError(f"Duplicate legacy graph: {key}")
        graphs[key] = record
    results = []
    for row in read_jsonl(args.inputs):
        old = graphs.get(row["source_id"])
        corrected = {**old, "status": "valid"} if old else None
        value = dependency_score(row["generated_code"], corrected)
        v1 = legacy.compute_dependency_reward(row["generated_code"], old) if old else 0.0
        results.append({"source_id": row["source_id"], "origin": row["origin"],
                        "legacy_safe_label": row.get("legacy_safe_label"),
                        "generated_code_sha256": row["generated_code_sha256"],
                        "v1_score": v1, "v2": value})
    write_jsonl(args.output / "replay.jsonl", results)
    counts = Counter(row["v2"]["status"] for row in results)
    comparable = [row for row in results if row["v2"]["status"] in ("empty", "nonempty")]
    summary = {"version": VERSION, "rows": len(results), "graph_statuses": dict(counts),
               "historical_safe_labels_reused_without_new_safety_certification": True,
               "v1_mean": sum(row["v1_score"] for row in comparable) / len(comparable),
               "v2_mean": sum(row["v2"]["score"] for row in comparable) / len(comparable),
               "score_increased": sum(row["v2"]["score"] > row["v1_score"] + 1e-9 for row in comparable),
               "score_decreased": sum(row["v2"]["score"] < row["v1_score"] - 1e-9 for row in comparable),
               "input_hashes": {str(path): file_hash(path) for path in (args.inputs, args.graphs, args.legacy_module)}}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
