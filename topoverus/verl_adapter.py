"""VERL custom reward hook: own code, explicit configuration, full traces."""

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path

from topoverus.graph import load_graphs
from topoverus.reward import RewardConfig, score_response


@lru_cache(maxsize=1)
def context():
    config_path = Path(os.environ["TOPOVERUS_REWARD_CONFIG"])
    config = RewardConfig(**json.loads(config_path.read_text()))
    graphs = load_graphs(Path(os.environ["TOPOVERUS_GRAPH_PATH"])) if config.dependency_enabled else {}
    output = Path(os.environ["TOPOVERUS_RUN_DIR"]).resolve()
    work_dir = output / "tmp"
    work_dir.mkdir(parents=True, exist_ok=True)
    traces = output / "outputs"
    traces.mkdir(exist_ok=True)
    # One worker-local header plus version/config in every rollout.
    (traces / f"reward-worker-{os.getpid()}.json").write_text(json.dumps({
        "reward_config": json.loads(config_path.read_text()), "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest()}, indent=2) + "\n")
    return config, graphs, output, work_dir


def compute_score(data_source, solution_str, ground_truth, extra_info=None,
                  reward_model_configuration=None):
    config, graphs, output, work_dir = context()
    if extra_info is None or not extra_info.get("source_id") or not extra_info.get("unverified_code"):
        raise ValueError("Reward input lacks canonical source identity or source code")
    key = extra_info["source_id"]
    record = graphs.get(key)
    if record is not None and record["source_sha256"] != extra_info["source_sha256"]:
        raise ValueError(f"Graph/source hash mismatch: {key}")
    result = score_response(solution_str, extra_info["unverified_code"], record, config,
                            os.environ["VERUS_BIN"], os.environ["LYNETTE_BIN"], work_dir)
    result.update(source_id=key, response=solution_str,
                  response_sha256=hashlib.sha256(solution_str.encode()).hexdigest())
    with (output / "outputs" / f"rollouts-{os.getpid()}.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False) + "\n")
    return {"score": result["score"], "r_verus": result["r_verus"], "r_dep": result["r_dep"],
            "r_format": result["r_format"], "r_gate": result["r_gate"],
            "strict_safe_pass": int(result["strict_safe_pass"])}
