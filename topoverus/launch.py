"""Freeze commands/inputs, then run SFT or the single Dep-v2 RL arm."""

import argparse
from dataclasses import asdict
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from urllib.parse import unquote, urlparse

from .checkpoint import file_hash, inspect, weight_hashes
from .data import read_jsonl
from .graph import load_graphs
from .reward import RewardConfig
from .runtime import temp_address
from .tools import CHECKER_VERSION, SAFETY_POLICY

ROOT = Path(__file__).resolve().parents[1]


def source_files():
    files = [ROOT / "README.md", ROOT / "pyproject.toml"]
    for name in ("topoverus", "scripts", "configs", "tests", "third_party"):
        files.extend(path for path in (ROOT / name).rglob("*")
                     if path.is_file() and "__pycache__" not in path.parts)
    return sorted(files)


def validate_dep_config(config: dict) -> dict:
    from hydra import compose, initialize_config_dir
    import verl
    from verl.utils.config import omega_conf_to_dataclass

    overrides = [f"{key}={json.dumps(value, separators=(',', ':'))}"
                 for key, value in config.items() if key != "seed"]
    overrides.append(f"++data.seed={config['seed']}")
    with initialize_config_dir(config_dir=str(Path(verl.__file__).parent / "trainer/config"), version_base=None):
        resolved = compose(config_name="ppo_trainer", overrides=overrides)
    actor = omega_conf_to_dataclass(resolved.actor_rollout_ref.actor)
    rollout = omega_conf_to_dataclass(resolved.actor_rollout_ref.rollout)
    actor.validate(4, resolved.data.train_batch_size, resolved.actor_rollout_ref.model)
    return {"actor": type(actor).__name__, "rollout": type(rollout).__name__,
            "data_seed": config["seed"], "actor_minibatch_seed": actor.get("data_loader_seed", 1),
            "rollout_engine_seed": rollout.get("seed", 0),
            "rollout_worker_torch_seed": "gen_dp_rank + 1000"}


def audit_tokens(model: Path, sft: Path | None, train: Path | None, config: dict) -> dict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    if sft:
        lengths = []
        for row in read_jsonl(sft):
            full = tokenizer.apply_chat_template(row["messages"], tokenize=False)
            prefix = tokenizer.apply_chat_template(row["messages"][:-1], tokenize=False, add_generation_prompt=True)
            if not full.startswith(prefix):
                raise ValueError("SFT chat-template prompt slicing mismatch")
            lengths.append(len(tokenizer(full, add_special_tokens=False)["input_ids"]) + 1)
        limit = config["max_len"]
    else:
        import pandas as pd
        rows = pd.read_parquet(train).to_dict("records")
        lengths = [len(tokenizer.apply_chat_template(list(row["prompt"]), tokenize=True, add_generation_prompt=True)) for row in rows]
        limit = config["data.max_prompt_length"]
    if not lengths or max(lengths) > limit:
        raise ValueError(f"Empty dataset or token overflow: max={max(lengths, default=0)}, limit={limit}")
    return {"rows": len(lengths), "max_tokens": max(lengths), "limit": limit}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("sft", "dep"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--reward-config", type=Path, default=ROOT / "configs/reward.json")
    parser.add_argument("--verus")
    parser.add_argument("--lynette")
    parser.add_argument("--checkpoint-hashes", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    model, data, run = args.model.resolve(), args.data.resolve(), args.run.resolve()
    if not run.is_relative_to(ROOT / "runs") or not data.is_relative_to(ROOT / "data"):
        raise ValueError("Datasets and run outputs must live in release data/ and runs/")
    config_path = args.config or ROOT / "configs" / ("sft.json" if args.stage == "sft" else "dep_rl.json")
    config = json.loads(config_path.read_text())
    checkpoint = inspect(model)
    temporary = temp_address(ROOT)
    split = json.loads((data / "split.json").read_text())
    if set(split["train"]) & set(split["heldout"]):
        raise ValueError("Overlapping train/heldout identities")
    input_files = [config_path, data / "split.json", data / "audit.json", data / "provenance.json"]
    env = os.environ.copy()
    env.update(PYTHONNOUSERSITE="1", TOKENIZERS_PARALLELISM="false", HF_HOME=str(ROOT / ".cache/huggingface"),
               HF_DATASETS_CACHE=str(ROOT / ".cache/datasets"), HF_HUB_OFFLINE="1", WANDB_MODE="disabled",
               PYTHONPATH=str(ROOT),
               TMPDIR=str(temporary), RAY_TMPDIR=str(temporary), XDG_CACHE_HOME=str(ROOT / ".cache"),
               TORCH_EXTENSIONS_DIR=str(ROOT / ".cache/torch_extensions"), TRITON_CACHE_DIR=str(ROOT / ".cache/triton"))
    if args.stage == "sft":
        original_source = model / "topoverus_source.json"
        if not original_source.exists() or json.loads(original_source.read_text())["repository"] != "Qwen/QwQ-32B":
            raise ValueError("Clean SFT must initialize from documented original Qwen/QwQ-32B")
        input_files.extend([data / "sft_train.jsonl", original_source])
        command = [sys.executable, "-m", "deepspeed.launcher.runner", "--num_gpus", str(config["num_gpus"]),
                   "--module", "openrlhf.cli.train_sft", "--pretrain", str(model),
                   "--dataset", str(data / "sft_train.jsonl"), "--input_key", "messages", "--apply_chat_template",
                   "--bf16", "--gradient_checkpointing", "--adam_offload", "--attn_implementation", "flash_attention_2",
                   "--save_path", str(run / "checkpoints"), "--save_steps", "-1", "--eval_steps", "-1", "--logging_steps", "1"]
        for key, value in config.items():
            if key != "num_gpus":
                command.extend([f"--{key}", str(value)])
        env.update(TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="86400", TORCH_NCCL_ENABLE_MONITORING="0",
                   TORCH_NCCL_ASYNC_ERROR_HANDLING="0", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        token_audit = audit_tokens(model, data / "sft_train.jsonl", None, config)
    else:
        # vLLM's sleep-mode memory pool rejects expandable-segment allocators.
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        env.pop("PYTORCH_ALLOC_CONF", None)
        if not args.verus or not args.lynette:
            raise ValueError("Dep-v2 requires explicit Verus and Lynette binaries")
        checker = subprocess.run([args.lynette, "--topoverus-safety-version"],
                                 capture_output=True, text=True, timeout=30)
        if checker.returncode != 0 or checker.stdout.strip() != CHECKER_VERSION:
            raise ValueError(f"Dep-v2 requires safety checker {CHECKER_VERSION}")
        origin = model.parent / "manifest.json"
        initialization = json.loads(origin.read_text()) if origin.exists() else {}
        if initialization.get("stage") != "sft" or initialization.get("status") != "trained":
            raise ValueError("Dep-v2 must initialize from this release's clean SFT run")
        sft_hashes = [value for name, value in initialization["inputs_sha256"].items()
                      if Path(name).name == "sft_train.jsonl"]
        if sft_hashes != [file_hash(data / "sft_train.jsonl")]:
            raise ValueError("Dep-v2 dataset does not match clean SFT initialization data")
        reward = RewardConfig(**json.loads(args.reward_config.read_text()))
        if not reward.dependency_enabled:
            raise ValueError("Dep run requires enabled dependency reward")
        graphs = load_graphs(data / "graphs.jsonl")
        tasks = read_jsonl(data / "rl_train.jsonl")
        # VERL disables all saving when save_freq <= 0, including final saving.
        # Use the planned final step as cadence, not the disabled sentinel.
        config["trainer.save_freq"] = (len(tasks) // config["data.train_batch_size"]) * config["trainer.total_epochs"]
        if config["trainer.save_freq"] < 1:
            raise ValueError("RL dataset/budget produces no training updates")
        config_audit = validate_dep_config(config)
        if set(graphs) != set(split["train"]) or {row["source_id"] for row in tasks} != set(split["train"]):
            raise ValueError("Graph and training source identities do not match")
        if any(graphs[task["source_id"]]["source_sha256"] != task["source_sha256"] for task in tasks):
            raise ValueError("Graph/source hash mismatch")
        input_files.extend([data / "rl_train.parquet", data / "heldout.parquet", data / "graphs.jsonl", args.reward_config, origin])
        seed = config["seed"]
        command = [sys.executable, "-m", "verl.trainer.main_ppo"]
        for key, value in config.items():
            if key != "seed":
                command.append(f"{key}={json.dumps(value, separators=(',', ':'))}")
        command.extend([f"++data.seed={seed}",
                        f"data.train_files={json.dumps(str(data / 'rl_train.parquet'))}",
                        f"data.val_files={json.dumps(str(data / 'heldout.parquet'))}",
                        f"actor_rollout_ref.model.path={json.dumps(str(model))}",
                        f"trainer.default_local_dir={json.dumps(str(run / 'checkpoints'))}",
                        f"trainer.experiment_name={json.dumps(run.name)}",
                        f"custom_reward_function.path={json.dumps(str(ROOT / 'topoverus/verl_adapter.py'))}",
                        "custom_reward_function.name=compute_score"])
        env.update(VERUS_BIN=str(Path(args.verus).resolve()), LYNETTE_BIN=str(Path(args.lynette).resolve()),
                   TOPOVERUS_REWARD_CONFIG=str(run / "config/reward.json"), TOPOVERUS_GRAPH_PATH=str(data / "graphs.jsonl"),
                   TOPOVERUS_RUN_DIR=str(run), MKL_THREADING_LAYER="GNU", HYDRA_FULL_ERROR="1",
                   TORCH_NCCL_ENABLE_MONITORING="0", TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="3600")
        token_audit = audit_tokens(model, None, data / "rl_train.parquet", config)
    # SFT is BF16. VERL's actor masters/full HF checkpoint are FP32; reserve
    # twice the BF16 model size. Save HF directly, avoiding a separate merge.
    needed = checkpoint["weight_bytes"] * (1 if args.stage == "sft" else 2) + 20 * 1024**3
    free = shutil.disk_usage(ROOT).free
    if args.execute and free < needed:
        raise RuntimeError(f"Insufficient free space for {args.stage}: need {needed / 1024**3:.1f} GiB, free {free / 1024**3:.1f} GiB")
    if args.execute:
        (ROOT / "tmp").mkdir(exist_ok=True)
        # Dataset workers use AF_UNIX sockets beneath TMPDIR. Check this before
        # spending time loading the full 32B model.
        subprocess.run([sys.executable, "-c",
                        "import multiprocess; manager=multiprocess.Manager(); manager.shutdown()"],
                       env=env, cwd=ROOT, check=True, timeout=30)
        import torch
        if torch.cuda.device_count() != 4:
            raise RuntimeError(f"Expected four visible GPUs, found {torch.cuda.device_count()}")
    run.mkdir(parents=True, exist_ok=False)
    for name in ("config", "logs", "checkpoints", "outputs", "metrics", "tmp"):
        (run / name).mkdir()
    for name in ("huggingface", "datasets", "torch_extensions", "triton"):
        (ROOT / ".cache" / name).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, run / "config" / config_path.name)
    if args.stage == "dep":
        (run / "config/reward.json").write_text(json.dumps(asdict(reward), indent=2) + "\n")
        (run / "config/dep_rl.json").write_text(json.dumps(config, indent=2) + "\n")
    with tarfile.open(run / "config/source.tar.gz", "w:gz") as archive:
        for path in source_files():
            archive.add(path, arcname=str(path.relative_to(ROOT)))
    versions = {}
    for package in ("torch", "transformers", "openrlhf", "verl", "deepspeed", "vllm", "flash-attn"):
        try:
            versions[package] = {"version": metadata.version(package),
                                 "direct_url": metadata.distribution(package).read_text("direct_url.json")}
        except metadata.PackageNotFoundError:
            pass
    # A version string cannot identify a locally modified editable trainer.
    if args.stage == "dep" and versions.get("verl", {}).get("direct_url"):
        direct = json.loads(versions["verl"]["direct_url"])
        if direct.get("dir_info", {}).get("editable"):
            framework_root = Path(unquote(urlparse(direct["url"]).path))
            package_root = framework_root / "verl"
            if not package_root.is_dir():
                raise ValueError("Cannot snapshot editable VERL package")
            framework_archive = run / "config/verl-source.tar.gz"
            with tarfile.open(framework_archive, "w:gz") as archive:
                for source in sorted(package_root.rglob("*")):
                    if source.is_file() and (source.suffix in (".py", ".yaml", ".yml", ".json") or source.name == "version"):
                        archive.add(source, arcname=str(source.relative_to(framework_root)))
                for name in ("pyproject.toml", "setup.py", "LICENSE"):
                    source = framework_root / name
                    if source.is_file():
                        archive.add(source, arcname=name)
            versions["verl"]["source_snapshot_sha256"] = file_hash(framework_archive)
    manifest = {"stage": args.stage, "status": "prepared" if not args.execute else "running", "command": command,
                "safety_policy": SAFETY_POLICY,
                "cwd": str(ROOT), "seed": config["seed"], "config": config, "frameworks": versions,
                "source_sha256": file_hash(run / "config/source.tar.gz"),
                "inputs_sha256": {str(path): file_hash(path) for path in input_files},
                "initial_checkpoint": checkpoint, "token_audit": token_audit,
                "selection": "fixed_final_budget_no_heldout_selection", "free_bytes_before": free}
    if args.stage == "dep":
        manifest["config_audit"] = config_audit
        manifest["tools_sha256"] = {name: file_hash(Path(binary).resolve())
                                    for name, binary in (("verus", args.verus), ("lynette", args.lynette))}
    # Bind the actual initialization tensors, never only the model directory name.
    manifest["initial_checkpoint"]["sha256"] = weight_hashes(model, args.checkpoint_hashes)
    if args.checkpoint_hashes:
        manifest["checkpoint_hash_certificate_sha256"] = file_hash(args.checkpoint_hashes)
    path = run / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"command": command, "run": str(run), "token_audit": token_audit}, indent=2), flush=True)
    if not args.execute:
        return
    if args.stage == "sft":
        with (run / "logs/preflight.log").open("w") as stream:
            check = subprocess.run([sys.executable, "-m", "topoverus.sft_preflight", "--model", str(model),
                                    "--data", str(data / "sft_train.jsonl"), "--max-len", str(config["max_len"])],
                                   cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    else:
        with (run / "logs/preflight.log").open("w") as stream:
            check = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone",
                                    "--nproc_per_node", "4", "-m", "topoverus.rl_preflight",
                                    "--model", str(model), "--config", str(run / "config/dep_rl.json"),
                                    "--output", str(run / "rl-preflight.json")],
                                   cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if check.returncode != 0:
        manifest.update(status="preflight_failed", returncode=check.returncode)
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        raise SystemExit(check.returncode)
    with (run / "logs/train.log").open("w") as stream:
        proc = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    manifest["returncode"] = proc.returncode
    manifest["status"] = "trained" if proc.returncode == 0 else "failed"
    if proc.returncode == 0:
        # Let the framework save tokenizer files. Never overwrite a trained index.
        final_model = run / "checkpoints"
        if args.stage == "dep":
            final_model /= f"global_step_{config['trainer.save_freq']}/actor/huggingface"
        try:
            final_checkpoint = inspect(final_model)
        except (OSError, ValueError) as exc:
            manifest.update(status="checkpoint_invalid", checkpoint_error=str(exc))
            path.write_text(json.dumps(manifest, indent=2) + "\n")
            raise
        manifest["final_checkpoint"] = final_checkpoint
        manifest["final_checkpoint_sha256"] = {name: file_hash(final_model / name)
                                               for name in final_checkpoint["shards"]}
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
