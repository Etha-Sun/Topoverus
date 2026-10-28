"""Rebuild first-loop inductive dependency graphs with checked tool outcomes."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re

from .data import digest, read_jsonl
from .matcher_v2 import first_invariants
from .tools import safety, verify


def replace_first_block(code: str, removed: int) -> tuple[str, dict[int, tuple[int, int]]]:
    lines = code.splitlines()
    start = next(i for i, line in enumerate(lines) if re.match(r"^\s*invariant\s*(?://.*)?$", line))
    end = next(i for i in range(start + 1, len(lines))
               if re.match(r"^\s*decreases\b", lines[i]) or lines[i].strip() == "{")
    invariants = first_invariants(code)
    output, ranges = lines[:start + 1], {}
    kept = [i for i in range(len(invariants)) if i != removed]
    for index in kept:
        clause = invariants[index].rstrip().rstrip(",").splitlines()
        clause[-1] += ","
        first = len(output) + 1
        output.extend(clause)
        ranges[index] = (first, len(output))
    if not kept:
        output = lines[:start]  # No dangling empty invariant keyword.
    output.extend(lines[end:])
    return "\n".join(output), ranges


def failed_invariants(stderr: str, ranges: dict[int, tuple[int, int]]) -> list[int]:
    failures = set()
    text = re.sub(r"\x1b\[[0-9;]*m", "", stderr)
    for block in re.split(r"(?m)(?=^error:)", text):
        if not re.search(r"^error: invariant (?:not satisfied|not maintained)", block):
            continue
        location = re.search(r"-->[^\n]*:(\d+):\d+", block)
        if location is None:
            raise ValueError("Invariant error has no source location")
        line = int(location.group(1))
        failures.update(index for index, (first, last) in ranges.items() if first <= line <= last)
    return sorted(failures)


def build(task: dict, binary: str, lynette: str, work_dir: Path) -> dict:
    code = task["verified_code"]
    invariants = first_invariants(code)
    record = {"source_id": task["source_id"], "source_sha256": task["source_sha256"],
              "reference_sha256": digest(code), "builder_version": "inductive-v2.0",
              "invariants": invariants, "dependency_edges": [], "checks": [], "status": "valid"}
    if not invariants or len(invariants) > 10:
        record["status"] = "no_invariants" if not invariants else "unsupported_size"
        return record
    safe = safety(task["unverified_code"], code, lynette, work_dir)
    record["reference_safety"] = safe.to_dict()
    if safe.status != "safe":
        record["status"] = "reference_unsafe" if safe.status == "unsafe" else "reference_safety_unknown"
        return record
    baseline = verify(code, binary, work_dir, multiple_errors=100)
    record["baseline"] = baseline.to_dict()
    if baseline.status != "verified":
        record["status"] = "baseline_failed"
        return record
    if len(invariants) == 1:
        return record
    for removed in range(len(invariants)):
        try:
            candidate, ranges = replace_first_block(code, removed)
            checked = verify(candidate, binary, work_dir, multiple_errors=100)
            if checked.status not in ("verified", "verification_failed"):
                raise ValueError(f"Removal tool result: {checked.status}")
            failures = failed_invariants(checked.stderr, ranges)
            record["checks"].append({"removed": removed, "target_failures": failures,
                                      "verification": checked.to_dict()})
            record["dependency_edges"].extend([[removed, target] for target in failures])
        except ValueError as exc:
            record.update(status="inconclusive", failure=str(exc))
            # A partial graph is not evidence of absence of missing edges.
            record["dependency_edges"] = None
            break
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verus", required=True)
    parser.add_argument("--lynette", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    work_dir = args.output.parent / "tmp"
    work_dir.mkdir(exist_ok=True)
    tasks = read_jsonl(args.tasks)
    with args.output.open("x", encoding="utf-8") as stream, ThreadPoolExecutor(max_workers=args.workers) as pool:
        for number, record in enumerate(pool.map(lambda task: build(task, args.verus, args.lynette, work_dir), tasks), 1):
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            if number % 10 == 0:
                print(f"graphs {number}/{len(tasks)}", flush=True)


if __name__ == "__main__":
    main()
