"""Explicit verifier/safety outcomes. Unknown results never count as success."""

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

from .trust import contract_projection, function_names, trust_change

SAFETY_POLICY = "strict_safe_v4_trusted_ghost_context"
CHECKER_VERSION = "topoverus-trusted-ghost-context-v4"


@dataclass
class ToolResult:
    status: str
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    elapsed_sec: float = 0.0
    verified: int = 0
    errors: int = 0

    def to_dict(self):
        return asdict(self)


def invoke(command: list[str], timeout: float, cwd: Path) -> ToolResult:
    start = time.monotonic()
    env = os.environ.copy()
    env["PATH"] = str(Path.home() / ".cargo/bin") + os.pathsep + env.get("PATH", "")
    try:
        proc = subprocess.run(command, cwd=cwd, env=env, capture_output=True,
                              text=True, timeout=timeout)
        return ToolResult("completed", proc.returncode, proc.stdout, proc.stderr,
                          time.monotonic() - start)
    except subprocess.TimeoutExpired:
        return ToolResult("timeout", elapsed_sec=time.monotonic() - start)
    except OSError as exc:
        return ToolResult("tool_error", stderr=str(exc), elapsed_sec=time.monotonic() - start)


def parse_verification(result: ToolResult) -> ToolResult:
    if result.status != "completed":
        return result
    matches = re.findall(r"verification results::\s*(\d+) verified,\s*(\d+) errors?",
                         result.stdout + "\n" + result.stderr)
    if len(matches) != 1:
        result.status = "tool_error"
        return result
    result.verified, result.errors = map(int, matches[0])
    if result.returncode == 0 and result.verified > 0 and result.errors == 0:
        result.status = "verified"
    elif result.returncode == 1 and result.errors > 0:
        result.status = "verification_failed"
    else:
        result.status = "tool_error"
    return result


def verify(code: str, binary: str, work_dir: Path, timeout: float = 30,
           multiple_errors: int | None = None) -> ToolResult:
    binary = str(Path(binary).resolve()) if os.path.sep in binary else binary
    with tempfile.TemporaryDirectory(prefix="verus-", dir=work_dir) as directory:
        root = Path(directory).resolve()
        path = root / "candidate.rs"
        path.write_text(code, encoding="utf-8")
        options = ["--multiple-errors", str(multiple_errors)] if multiple_errors else []
        return parse_verification(invoke([binary, *options, str(path)], timeout, root))


def safety(original: str, candidate: str, binary: str, work_dir: Path,
           timeout: float = 30) -> ToolResult:
    violation = trust_change(original, candidate)
    if violation:
        return ToolResult("unsafe", stderr=violation)
    try:
        names = function_names(original)
        projected_original = contract_projection(original, names)
        projected_candidate = contract_projection(candidate, names)
    except ValueError as exc:
        return ToolResult("unsafe", stderr=str(exc))
    binary = str(Path(binary).resolve()) if os.path.sep in binary else binary
    with tempfile.TemporaryDirectory(prefix="safety-", dir=work_dir) as directory:
        root = Path(directory).resolve()
        version = invoke([binary, "--topoverus-safety-version"], timeout, root)
        if version.status != "completed":
            return version
        if version.returncode != 0 or version.stdout.strip() != CHECKER_VERSION:
            return ToolResult("tool_error", stderr=f"Safety checker requires {CHECKER_VERSION}")
        first, second = root / "original.rs", root / "candidate.rs"
        first.write_text(original, encoding="utf-8")
        second.write_text(candidate, encoding="utf-8")
        # Do not remove main(): executable changes must remain visible.
        # Target mode pins contracts/assumes; spec functions stay visible too.
        result = invoke([binary, "compare", "-t", "--requires", "--ensures",
                         "--spec", "--decreases", "--assumes", str(first), str(second)],
                        timeout, root)
        if result.status == "completed" and result.returncode == 0 and (
                projected_original is not None or projected_candidate is not None):
            first.write_text(projected_original or original, encoding="utf-8")
            second.write_text(projected_candidate or candidate, encoding="utf-8")
            result = invoke([binary, "compare", "-t", "--requires", "--ensures",
                             "--spec", "--decreases", "--assumes", str(first), str(second)],
                            timeout, root)
        if result.status == "completed":
            if result.returncode == 0:
                result.status = "safe"
            elif result.returncode == 1 and result.stdout.strip() == "Files are different":
                result.status = "unsafe"
            else:
                result.status = "tool_error"
        return result
