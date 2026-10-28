"""Shared strict-safe scoring for RL and offline evaluation."""

from dataclasses import asdict, dataclass
from pathlib import Path
import re

from .code import final_code
from .graph import dependency_score
from .matcher_v2 import VERSION
from .tools import SAFETY_POLICY, safety, verify


@dataclass(frozen=True)
class RewardConfig:
    verus_mode: str = "fraction"
    dependency_enabled: bool = True
    matcher_version: str = VERSION
    matcher_threshold: float = 0.6
    verus_weight: float = 1.0
    dependency_weight: float = 1.0
    format_partial: float = 0.5
    format_none: float = 0.2
    timeout_sec: float = 30

    def __post_init__(self):
        if self.verus_mode not in ("binary", "fraction") or self.matcher_version != VERSION:
            raise ValueError("Unsupported reward configuration")
        if not 0 <= self.matcher_threshold <= 1:
            raise ValueError("Invalid matcher threshold")
        if any(value < 0 for value in (self.verus_weight, self.dependency_weight)):
            raise ValueError("Negative reward weight")
        if not 0 <= self.format_none <= self.format_partial <= 1 or self.timeout_sec <= 0:
            raise ValueError("Invalid format gate or tool timeout")


def format_score(response: str, config: RewardConfig) -> tuple[float, float]:
    sections = {tag: re.search(rf"<{tag}>(.*?)</{tag}>", response, re.DOTALL)
                for tag in ("think", "counterexample", "reasoning")}
    score = 0.1 * sum(bool(value) and len(value.group(1).strip()) >= 10
                      for value in sections.values())
    if sections["think"] is None and response.find("</think>") >= 10:
        score += 0.1
    ce = sections["counterexample"]
    if ce and "[State]" in ce.group(1) and "[Violation]" in ce.group(1) and (
            "[Execution Trace]" in ce.group(1) or "[Execution]" in ce.group(1)):
        score += 0.1
    count = sum(sections[tag] is not None for tag in ("counterexample", "reasoning"))
    gate = 1.0 if count == 2 else config.format_partial if count == 1 else config.format_none
    return score, gate


def score_response(response: str, original: str, record: dict | None,
                   config: RewardConfig, verus: str, lynette: str, work_dir: Path) -> dict:
    result = {"score": 0.0, "r_verus": 0.0, "r_dep": 0.0, "safety_policy": SAFETY_POLICY,
              "r_format": 0.0, "r_gate": 0.0, "strict_safe_pass": False,
              "generated_code": "", "safety": {"status": "not_run"},
              "verification": {"status": "not_run"},
              "graph": {"status": "disabled" if not config.dependency_enabled else "not_run"},
              "reward_config": asdict(config)}
    try:
        code = final_code(response)
    except ValueError as exc:
        result["extraction_error"] = str(exc)
        return result
    result["generated_code"] = code
    safe = safety(original, code, lynette, work_dir, config.timeout_sec)
    result["safety"] = safe.to_dict()
    if safe.status != "safe":
        return result
    verified = verify(code, verus, work_dir, config.timeout_sec)
    result["verification"] = verified.to_dict()
    if verified.status not in ("verified", "verification_failed"):
        return result
    result["strict_safe_pass"] = verified.status == "verified"
    result["r_verus"] = (float(result["strict_safe_pass"]) if config.verus_mode == "binary"
                          else verified.verified / (verified.verified + verified.errors))
    result["r_format"], result["r_gate"] = format_score(response, config)
    if config.dependency_enabled:
        result["graph"] = dependency_score(code, record, config.matcher_threshold)
        result["r_dep"] = result["graph"]["score"]
    result["score"] = (config.verus_weight * result["r_verus"] +
                       config.dependency_weight * result["r_dep"]) * result["r_gate"] + result["r_format"]
    return result
