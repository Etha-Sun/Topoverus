"""VERL dataset conversion preserving canonical joins and the shared prompt."""

import argparse
from pathlib import Path

from .data import read_jsonl


def main():
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for task in read_jsonl(args.input):
        rows.append({"data_source": "topoverus", "prompt": task["prompt"],
                     "ability": "verus_repair", "reward_model": {"style": "rule", "ground_truth": task["verified_code"]},
                     "extra_info": {"source_id": task["source_id"], "source_sha256": task["source_sha256"],
                                    "unverified_code": task["unverified_code"], "split": task["split"]}})
    pd.DataFrame(rows).to_parquet(args.output, index=False)
    print(f"Wrote {len(rows)} canonical tasks to {args.output}")


if __name__ == "__main__":
    main()
