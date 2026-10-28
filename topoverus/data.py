"""Import explicit inputs once; training has no dependency on the research tree."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

from .checkpoint import file_hash

from .code import FENCE, ctx_target, final_code, normalize_code
from .graph import source_id
from .trust import rust_tokens, trust_change
from .tools import ToolResult, invoke, safety, verify

CURRICULUM_SAFETY_POLICY = "canonical_contracts_and_actual_input_execution_v4"

SYSTEM = ("You are a Verus formal-verification expert. Repair only proof annotations. "
          "Preserve executable code, function contracts and specification functions. "
          "Respond with <think>...</think>, <counterexample> containing [State], "
          "[Execution Trace] and [Violation], <reasoning>...</reasoning>, then "
          "exactly one final Rust code fence. Do not use assume or admit.")


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def prompt(code: str, errors: str = "") -> list[dict]:
    # Source identity lives in metadata, not an accidental supervised label.
    user = f"Repair this Verus program.\n\n[Buggy Code]\n```rust\n{normalize_code(code)}\n```"
    if errors.strip():
        user += f"\n\n[Verifier Errors]\n{errors.strip()}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def import_task(row: dict, split: str) -> dict:
    meta = row["meta"]
    text = "\n".join(message["content"] for message in row["prompt_messages"])
    blocks = list(FENCE.finditer(text))
    if len(blocks) != 1:
        raise ValueError("Task prompt must contain exactly one source program")
    code = normalize_code(blocks[0].group(2))
    key = source_id(meta["rel_path"]) if meta.get("rel_path") else f"VeruSAGE/{meta['task_id']}"
    error_match = re.search(r"Verifier output:\n(.*?)\n\nUnverified code", text, re.DOTALL)
    errors = error_match.group(1) if error_match else ""
    return {"source_id": key, "split": split, "source_sha256": digest(code),
            "unverified_code": code, "verified_code": final_code(row["answer"]),
            "benchmark": meta.get("benchmark", key.split("/")[0]),
            "prompt": prompt(code, errors), "verifier_errors": errors}


def resolve_trajectory(row: dict, tasks: dict[str, dict]) -> tuple[dict, str, str]:
    user = next(message["content"] for message in row["messages"] if message["role"] == "user")
    assistant = next(message["content"] for message in row["messages"] if message["role"] == "assistant")
    benchmark = re.search(r"^Benchmark:\s*(.+)$", user, re.MULTILINE)
    problem = re.search(r"^Problem:\s*(.+)$", user, re.MULTILINE)
    buggy = re.search(r"\[Buggy Code\]\s*```(?:rust|verus)?\n(.*?)```", user, re.DOTALL)
    if not benchmark or not problem or not buggy:
        raise ValueError("Missing trajectory source labels or buggy code")
    bm, stem = benchmark.group(1).strip(), Path(problem.group(1).strip()).stem
    candidates = [task for key, task in tasks.items() if Path(key).stem == stem and (
        bm == key.split("/")[0] or bm == "/".join(key.split("/")[:2]))]
    target_code = final_code(assistant)
    if len(candidates) > 1:
        candidates = [task for task in candidates if normalize_code(task["verified_code"]) == target_code]
    if len(candidates) != 1:
        raise ValueError(f"Unresolved or ambiguous source identity: {bm}/{stem}")
    return candidates[0], normalize_code(buggy.group(1)), assistant


def family(task: dict) -> str | None:
    key = task["source_id"]
    if not key.startswith("Obfuscated_Verusbench/"):
        return key
    match = re.fullmatch(r"(cloverbench|diffy|mbpp)_(.+)_\d+", Path(key).stem)
    if not match:
        return None
    category = {"cloverbench": "CloverBench", "diffy": "Diffy", "mbpp": "MBPP"}[match[1]]
    return f"cleaned-verusbench/{category}/unverified/{match[2]}.rs"


def token_fingerprint(code: str) -> str:
    # Some historical reference files end in an unclosed documentation comment.
    # Ignore that tail for conservative duplicate detection only. Safety and
    # verification still reject it; never silently repair reference programs.
    return digest(" ".join(rust_tokens(code, allow_unclosed_comments=True)))


def curriculum_safety(original: str, target: str, source: str, binary: str,
                      work_dir: Path) -> ToolResult:
    """Pin the problem contracts, while allowing intermediate proof helpers to change."""
    violation = trust_change(original, target)
    if violation:
        return ToolResult("unsafe", stderr=violation)
    result = safety(source, target, binary, work_dir)
    if result.status != "safe":
        return result
    binary = str(Path(binary).resolve()) if "/" in binary else binary
    with tempfile.TemporaryDirectory(prefix="curriculum-", dir=work_dir) as directory:
        root = Path(directory).resolve()
        first, second = root / "input.rs", root / "target.rs"
        first.write_text(original, encoding="utf-8")
        second.write_text(target, encoding="utf-8")
        # Canonical safety above pins trust context. The intermediate input may
        # contain an untrusted assumption that the verified target removes.
        result = invoke([binary, "compare", "--requires", "--ensures", "--spec",
                         "--decreases", str(first), str(second)], 30, root)
        if result.status == "completed":
            if result.returncode == 0:
                result.status = "safe"
            elif result.returncode == 1 and result.stdout.strip() == "Files are different":
                result.status = "unsafe"
            else:
                result.status = "tool_error"
        return result


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=False)
    work_dir = args.output / "tmp"
    work_dir.mkdir()
    legacy_train = [import_task(row, "train") for row in read_jsonl(args.train)]
    heldout = [import_task(row, "heldout") for row in read_jsonl(args.heldout)]
    main = [import_task(row, "evaluation") for row in read_jsonl(args.evaluation)]
    tasks = {task["source_id"]: task for task in legacy_train + heldout}
    if len(tasks) != len(legacy_train) + len(heldout):
        raise ValueError("Duplicate or overlapping source IDs")
    audit = {"legacy_train_sources": len(legacy_train), "heldout_sources": len(heldout),
             "evaluation_sources": len(main), "rejections": [], "counterexample_validation": "structure_only",
             "lexical_overlap_is_conservative_not_parser_equivalence": True}
    excluded = set()
    for field in ("unverified_code", "verified_code"):
        for name, transform in (("normalized", normalize_code), ("lexical", token_fingerprint)):
            heldout_hashes = {digest(transform(task[field])) for task in heldout}
            overlap = [task["source_id"] for task in legacy_train
                       if digest(transform(task[field])) in heldout_hashes]
            audit[f"legacy_overlap_{field}_{name}"] = overlap
            excluded.update(overlap)
    train = [task for task in legacy_train if task["source_id"] not in excluded]
    for key in excluded:
        tasks[key]["split"] = "excluded_heldout_duplicate"
    manifest = {"schema_version": 1, "train": [task["source_id"] for task in train],
                "heldout": [task["source_id"] for task in heldout],
                "legacy_train": [task["source_id"] for task in legacy_train],
                "excluded_heldout_duplicates": sorted(excluded),
                "overlap_policy": "heldout_priority_normalized_and_lexical_input_or_target",
                "checkpoint_selection": "fixed_final_budget_no_heldout_selection"}
    audit.update(train_sources=len(train), excluded_train_sources=len(excluded))
    train_ids, heldout_ids = set(manifest["train"]), set(manifest["heldout"])
    lineage = Counter()
    for task in main:
        parent = family(task)
        if task["source_id"].startswith("Obfuscated_Verusbench/"):
            lineage["train_parent" if parent in train_ids else "heldout_parent" if parent in heldout_ids else "unknown_parent"] += 1
        task["source_family"] = parent
        task["source_family_status"] = "seen_train" if parent in train_ids else "heldout" if parent in heldout_ids else "external_or_unknown"
    audit["obfuscated_filename_lineage"] = dict(lineage)
    audit["filename_lineage_is_heuristic"] = True
    candidates = []
    trajectory_file_hash = digest(args.sft.read_text(encoding="utf-8"))
    for number, row in enumerate(read_jsonl(args.sft), 1):
        try:
            task, code, assistant = resolve_trajectory(row, tasks)
            if task["split"] != "train":
                raise ValueError(task["split"] + "_source")
            target = ctx_target(assistant)
            errors = next(message["content"] for message in row["messages"] if message["role"] == "user").split("[Verifier Errors]", 1)
            messages = prompt(code, errors[1] if len(errors) == 2 else "")
            messages.append({"role": "assistant", "content": target})
            candidates.append({"messages": messages, "meta": {
                "source_id": task["source_id"], "source_sha256": task["source_sha256"],
                "input_sha256": digest(code), "target_sha256": digest(final_code(target)),
                "split": "train", "trajectory_origin": {"input_file_sha256": trajectory_file_hash, "line": number},
                "counterexample_validation": "structure_only"}, "input_code": code})
        except ValueError as exc:
            audit["rejections"].append({"line": number, "reason": str(exc)})
    heldout_codes = {token_fingerprint(task["unverified_code"]) for task in heldout}
    heldout_targets = {token_fingerprint(task["verified_code"]) for task in heldout}
    clean_candidates = []
    for row in candidates:
        if token_fingerprint(row["input_code"]) in heldout_codes or token_fingerprint(
                final_code(row["messages"][-1]["content"])) in heldout_targets:
            audit["rejections"].append({"line": row["meta"]["trajectory_origin"]["line"],
                                      "reason": "heldout_input_or_target_duplicate"})
        else:
            clean_candidates.append(row)
    candidates = clean_candidates

    def validate(row):
        original, target = row["input_code"], final_code(row["messages"][-1]["content"])
        source = tasks[row["meta"]["source_id"]]["unverified_code"]
        safe = curriculum_safety(original, target, source, args.lynette, work_dir)
        checked = verify(target, args.verus, work_dir) if safe.status == "safe" else None
        return row, safe.to_dict(), checked.to_dict() if checked else {"status": "not_run"}

    accepted, validation = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row, safe, checked in pool.map(validate, candidates):
            validation.append({"origin": row["meta"]["trajectory_origin"], "source_id": row["meta"]["source_id"],
                               "safety_policy": CURRICULUM_SAFETY_POLICY,
                               "contract_anchor_sha256": row["meta"]["source_sha256"],
                               "execution_anchor_sha256": row["meta"]["input_sha256"],
                               "safety": safe, "verification": checked})
            if safe["status"] == "safe" and checked["status"] == "verified":
                row.pop("input_code")
                row["meta"]["validation"] = {"safety": "safe", "verification": "verified"}
                accepted.append(row)
            else:
                audit["rejections"].append({"line": row["meta"]["trajectory_origin"]["line"],
                    "reason": f"safety={safe['status']};verification={checked['status']}"})
            if len(validation) % 25 == 0:
                print(f"validated {len(validation)}/{len(candidates)}; accepted {len(accepted)}", flush=True)
    if not accepted:
        raise ValueError("No strictly validated SFT trajectories")
    # Reject duplicate trajectories rather than weighting copies implicitly.
    unique = {}
    for row in accepted:
        unique.setdefault(digest(json.dumps(row["messages"], sort_keys=True)), row)
    audit.update(sft_candidates=len(candidates), sft_accepted=len(unique),
                 sft_duplicate_trajectories=len(accepted) - len(unique),
                 sft_covered_train_sources=len({row["meta"]["source_id"] for row in unique.values()}))
    write_jsonl(args.output / "sft_train.jsonl", list(unique.values()))
    write_jsonl(args.output / "rl_train.jsonl", train)
    write_jsonl(args.output / "heldout.jsonl", heldout)
    write_jsonl(args.output / "main_eval.jsonl", main)
    write_jsonl(args.output / "target_validation.jsonl", validation)
    for name, obj in (("split.json", manifest), ("audit.json", audit)):
        (args.output / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    modules = [Path(__file__), Path(__file__).with_name("code.py"),
               Path(__file__).with_name("tools.py"), Path(__file__).with_name("trust.py")]
    provenance = {"schema_version": 1, "counterexample_validation": "structure_only",
                  "sft_safety_policy": CURRICULUM_SAFETY_POLICY,
                  "sft_generation": "inherited_legacy_trajectories; teacher-model provenance absent in source rows",
                  "code_sha256": {path.name: file_hash(path) for path in modules},
                  "input_sha256": {path.name: file_hash(path) for path in (args.train, args.heldout, args.evaluation, args.sft)},
                  "tools": {"verus_sha256": file_hash(Path(args.verus)), "lynette_sha256": file_hash(Path(args.lynette)),
                            "verus_version": subprocess.run([args.verus, "--version"], capture_output=True, text=True).stdout.strip()}}
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in args.output.glob("*.json*")}
    (args.output / "hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(json.dumps({key: value for key, value in audit.items() if key != "rejections"}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    for name in ("train", "heldout", "evaluation", "sft", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--verus", required=True)
    parser.add_argument("--lynette", required=True)
    parser.add_argument("--workers", type=int, default=4)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
