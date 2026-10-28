"""AST comparison audit beyond lexical duplicate detection, using Lynette."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile

from .code import FENCE, final_code
from .data import digest, read_jsonl
from .tools import invoke


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--lynette", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train, heldout = read_jsonl(args.data / "rl_train.jsonl"), read_jsonl(args.data / "heldout.jsonl")
    rows = []
    for task in train + heldout:
        for field in ("unverified_code", "verified_code"):
            rows.append({"source_id": task["source_id"], "split": task["split"], "field": field,
                         "code": task[field]})
    for row in read_jsonl(args.data / "sft_train.jsonl"):
        original = next(FENCE.finditer(row["messages"][1]["content"])).group(2)
        rows.extend([{"source_id": row["meta"]["source_id"], "split": "train", "field": "sft_input", "code": original},
                     {"source_id": row["meta"]["source_id"], "split": "train", "field": "sft_target",
                      "code": final_code(row["messages"][-1]["content"])}])
    unique = {}
    for row in rows:
        unique.setdefault(digest(row["code"]), []).append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as directory:
        root = Path(directory).resolve()
        files = {}
        for key, group in unique.items():
            files[key] = root / f"{key}.rs"
            files[key].write_text(group[0]["code"])
        def parse(key):
            result = invoke([args.lynette, "parse", "--check", str(files[key])], 30, root)
            return key, result.returncode == 0
        with ThreadPoolExecutor(max_workers=8) as pool:
            parsed = dict(pool.map(parse, files))
        train_keys = {digest(row["code"]) for row in rows if row["split"] == "train" and parsed[digest(row["code"]) ]}
        heldout_keys = {digest(row["code"]) for row in rows if row["split"] == "heldout" and parsed[digest(row["code"]) ]}
        comparisons = [(a, b) for a in sorted(train_keys) for b in sorted(heldout_keys)]
        overlaps, unknown = [], []
        def compare(pair):
            a, b = pair
            result = invoke([args.lynette, "compare", "-t", "--spec", str(files[a]), str(files[b])], 30, root)
            return a, b, result
        with ThreadPoolExecutor(max_workers=8) as pool:
            for a, b, result in pool.map(compare, comparisons):
                if result.returncode == 0:
                    overlaps.append({"train": [{k: row[k] for k in ("source_id", "field")} for row in unique[a] if row["split"] == "train"],
                                     "heldout": [{k: row[k] for k in ("source_id", "field")} for row in unique[b] if row["split"] == "heldout"]})
                elif result.returncode != 1 or result.stdout.strip() != "Files are different":
                    unknown.append({"train_sha256": a, "heldout_sha256": b, "result": result.to_dict()})
        result = {"comparison": "Lynette target mode plus spec, whole executable program including main",
                  "parsed_unique_inputs": sum(parsed.values()), "unique_inputs": len(parsed),
                  "comparisons": len(comparisons), "overlaps": overlaps, "unknown_comparisons": unknown,
                  "unparsed": [{"sha256": key, "sources": [{k: row[k] for k in ("source_id", "split", "field")} for row in unique[key]]}
                               for key, value in parsed.items() if not value]}
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({key: len(value) if isinstance(value, list) else value for key, value in result.items()}, indent=2))


if __name__ == "__main__":
    main()
