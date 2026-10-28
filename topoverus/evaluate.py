"""Verus evaluation with optional strict safety and replayable per-chain outputs."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import sys
import tarfile

from .checkpoint import file_hash, inspect, weight_hashes
from .code import final_code
from .data import prompt, read_jsonl
from .launch import source_files
from .runtime import temp_address
from .tools import SAFETY_POLICY, safety, verify


def pass_at_k(total: int, correct: int, k: int) -> float:
    if not 0 <= correct <= total or not 1 <= k <= total:
        raise ValueError("Invalid pass@k counts; need at least k observed chains")
    return 1.0 if total - correct < k else 1 - math.comb(total - correct, k) / math.comb(total, k)


def generation_trace(output, formatted_prompt: str, seed: int, tokenizer) -> dict:
    tokens = list(output.outputs[0].token_ids)
    closing = tokenizer.encode("</think>", add_special_tokens=False)
    position = next((i for i in range(len(tokens) - len(closing) + 1)
                     if closing and tokens[i:i + len(closing)] == closing), None)
    return {"formatted_prompt": formatted_prompt, "generation_seed": seed,
            "prompt_token_count": len(output.prompt_token_ids) if output.prompt_token_ids is not None else None,
            "generated_token_count": len(tokens), "thinking_token_count": position,
            "thinking_closed": position is not None,
            "token_count_source": "vllm_token_ids"}


def judge(response: str, task: dict, verus: str, lynette: str | None, work_dir: Path) -> dict:
    result = {"source_id": task["source_id"], "response": response,
              "safety_policy": SAFETY_POLICY if lynette else "verus_only",
              "verified_pass": False, "strict_safe_pass": False if lynette else None,
              "generated_code": "", "safety": {"status": "not_run" if lynette else "disabled"}}
    try:
        code = final_code(response)
    except ValueError as exc:
        result.update(extraction_error=str(exc), verification={"status": "not_run"})
        return result
    result["generated_code"] = code
    if lynette:
        result["safety"] = safety(task["unverified_code"], code, lynette, work_dir).to_dict()
    checked = verify(code, verus, work_dir)
    result.update(verification=checked.to_dict(), verified_pass=checked.status == "verified")
    if lynette:
        result["strict_safe_pass"] = result["safety"]["status"] == "safe" and result["verified_pass"]
    return result


def metrics(tasks: list[dict], records: list[dict], samples: int, iterations: int,
            ks: list[int], strict: bool = True) -> dict:
    names = ("verified_pass", "strict_safe_pass") if strict else ("verified_pass",)
    success = "strict_safe_pass" if strict else "verified_pass"
    task_index = {row["source_id"]: row for row in tasks}
    if len(task_index) != len(tasks):
        raise ValueError("Duplicate evaluation task IDs")
    chains = defaultdict(list)
    seen = set()
    for record in records:
        key = record["source_id"], record["sample_index"], record["iteration"]
        if key in seen or key[0] not in task_index or not 0 <= key[1] < samples or not 1 <= key[2] <= iterations:
            raise ValueError(f"Invalid/duplicate evaluation attempt: {key}")
        seen.add(key)
        chains[key[:2]].append(record)
    groups = defaultdict(list)
    for task in tasks:
        key = task["source_id"]
        outcomes = []
        for index in range(samples):
            attempts = sorted(chains[(key, index)], key=lambda row: row["iteration"])
            if not attempts or [row["iteration"] for row in attempts] != list(range(1, len(attempts) + 1)):
                raise ValueError(f"Missing chain or iteration: {key}/{index}")
            if len(attempts) < iterations and not attempts[-1][success]:
                raise ValueError(f"Unfinished unsuccessful chain: {key}/{index}")
            outcomes.append({name: any(row[name] for row in attempts)
                             for name in names})
        labels = ["overall", f"benchmark:{key.split('/')[0]}"]
        if key.startswith("Obfuscated_Verusbench/"):
            labels.append("obfuscated_family:" + task["source_family_status"])
        for label in labels:
            groups[label].append(outcomes)
    result = {"samples_per_task": samples, "iterations": iterations, "groups": {}}
    for label, task_outcomes in groups.items():
        values = {"tasks": len(task_outcomes)}
        for name in names:
            for k in ks:
                values[f"{name}:unbiased_pass@{k}"] = sum(
                    pass_at_k(samples, sum(row[name] for row in outcomes), k)
                    for outcomes in task_outcomes) / len(task_outcomes)
                values[f"{name}:first_{k}_success"] = sum(
                    any(row[name] for row in outcomes[:k])
                    for outcomes in task_outcomes) / len(task_outcomes)
        result["groups"][label] = values
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--checkpoint-hashes", type=Path)
    parser.add_argument("--responses", type=Path, help="Offline replay instead of model inference")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-prompt-tokens", type=int, default=8192)
    parser.add_argument("--max-response-tokens", type=int, default=12288)
    parser.add_argument("--verus", required=True)
    parser.add_argument("--lynette", help="Enable strict safety checks with this Lynette binary; default: Verus only")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if bool(args.model) == bool(args.responses):
        raise ValueError("Provide exactly one model or offline response input")
    if args.samples < max(args.ks) or args.iterations < 1:
        raise ValueError("Sample/iteration budget does not cover requested metrics")
    root = Path(__file__).resolve().parents[1]
    run = args.run.resolve()
    if not run.is_relative_to(root / "runs"):
        raise ValueError("Evaluation artifacts must stay under release runs/")
    tasks = read_jsonl(args.tasks)
    run.mkdir(parents=True, exist_ok=False)
    for name in ("outputs", "metrics", "tmp", "config"):
        (run / name).mkdir()
    os.environ.update(HF_HOME=str(root / ".cache/huggingface"), HF_HUB_OFFLINE="1",
                      XDG_CACHE_HOME=str(root / ".cache"), TMPDIR=str(temp_address(root)),
                      TRITON_CACHE_DIR=str(root / ".cache/triton"), TOKENIZERS_PARALLELISM="false",
                      TORCH_EXTENSIONS_DIR=str(root / ".cache/torch_extensions"),
                      VLLM_CACHE_ROOT=str(root / ".cache/vllm"))
    task_index = {task["source_id"]: task for task in tasks}
    records = []
    manifest = {"stage": "evaluation", "status": "running", "command": sys.argv,
                "safety_policy": SAFETY_POLICY if args.lynette else "verus_only",
                "model": str(args.model), "samples": args.samples, "iterations": args.iterations,
                "seed": args.seed, "tasks_sha256": hashlib.sha256(args.tasks.read_bytes()).hexdigest(),
                "inference_dtype": "bfloat16",
                "max_prompt_tokens": args.max_prompt_tokens, "max_response_tokens": args.max_response_tokens}
    with tarfile.open(run / "config/source.tar.gz", "w:gz") as archive:
        for path in source_files():
            archive.add(path, arcname=str(path.relative_to(root)))
    manifest["source_sha256"] = file_hash(run / "config/source.tar.gz")
    manifest["tools_sha256"] = {name: file_hash(Path(binary).resolve())
                                for name, binary in (("verus", args.verus), ("lynette", args.lynette))
                                if binary is not None}
    manifest["frameworks"] = {}
    for package in ("torch", "transformers", "vllm"):
        try:
            manifest["frameworks"][package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            pass
    if args.responses:
        manifest["responses_sha256"] = file_hash(args.responses)
    else:
        checkpoint = inspect(args.model)
        checkpoint["sha256"] = weight_hashes(args.model, args.checkpoint_hashes)
        manifest["checkpoint"] = checkpoint
        if args.checkpoint_hashes:
            manifest["checkpoint_hash_certificate_sha256"] = file_hash(args.checkpoint_hashes)
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (run / "outputs/attempts.jsonl").open("w", encoding="utf-8") as stream, ThreadPoolExecutor(max_workers=args.workers) as pool:
        if args.responses:
            batches = [read_jsonl(args.responses)]
        else:
            from vllm import LLM, SamplingParams
            engine = LLM(model=str(args.model), tensor_parallel_size=4,
                         dtype="bfloat16",
                         max_model_len=args.max_prompt_tokens + args.max_response_tokens,
                         gpu_memory_utilization=0.8, seed=args.seed)
            tokenizer = engine.get_tokenizer()
            pending = [{"source_id": task["source_id"], "sample_index": index}
                       for task in tasks for index in range(args.samples)]

            def generated_batches():
                nonlocal pending
                for iteration in range(1, args.iterations + 1):
                    prompts, parameters = [], []
                    for chain in pending:
                        task = task_index[chain["source_id"]]
                        messages = task["prompt"]
                        if chain.get("previous"):
                            previous = chain["previous"]
                            feedback = previous["verification"].get("stderr", "")[-4000:]
                            if args.lynette and previous["safety"]["status"] != "safe":
                                feedback += "\nSafety check failed. Preserve the original executable code and contracts."
                            messages = prompt(task["unverified_code"], feedback)
                            messages[-1]["content"] += "\nPrevious failed repair:\n```rust\n" + previous["generated_code"] + "\n```"
                        value = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        if len(tokenizer.encode(value, add_special_tokens=False)) > args.max_prompt_tokens:
                            raise ValueError(f"Evaluation prompt exceeds frozen {args.max_prompt_tokens}-token budget: {chain['source_id']}")
                        prompts.append(value)
                        identifier = f"{args.seed}/{chain['source_id']}/{chain['sample_index']}/{iteration}"
                        seed = int(hashlib.sha256(identifier.encode()).hexdigest()[:8], 16)
                        parameters.append(SamplingParams(n=1, temperature=0.7, top_p=0.9, max_tokens=args.max_response_tokens, seed=seed))
                    outputs = engine.generate(prompts, parameters)
                    if len(outputs) != len(pending):
                        raise ValueError("Inference omitted requested chains")
                    batch = [{"source_id": chain["source_id"], "sample_index": chain["sample_index"],
                              "iteration": iteration, "response": output.outputs[0].text,
                              **generation_trace(output, value, parameter.seed, tokenizer)}
                             for chain, output, value, parameter in zip(pending, outputs, prompts, parameters)]
                    yield batch
                    if not pending:
                        break
            batches = generated_batches()
        for batch in batches:
            def checked(row):
                result = judge(row["response"], task_index[row["source_id"]], args.verus, args.lynette, run / "tmp")
                result.update(sample_index=row["sample_index"], iteration=row["iteration"])
                result.update({key: row[key] for key in ["formatted_prompt", "generation_seed", "prompt_token_count",
                               "generated_token_count", "thinking_token_count", "thinking_closed", "token_count_source"]
                               if key in row})
                return result
            results = list(pool.map(checked, batch))
            for result in results:
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            records.extend(results)
            if args.model:
                pending = [{"source_id": row["source_id"], "sample_index": row["sample_index"], "previous": row}
                           for row in results if not row["strict_safe_pass" if args.lynette else "verified_pass"]]
    summary = metrics(tasks, records, args.samples, args.iterations, args.ks, strict=bool(args.lynette))
    summary["safety_policy"] = manifest["safety_policy"]
    (run / "metrics/summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    manifest["status"] = "evaluated"
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary["groups"]["overall"], indent=2))


if __name__ == "__main__":
    main()
