"""One final-answer extraction rule shared by data, reward and evaluation."""

import re

TAGS = ("think", "counterexample", "reasoning")
FENCE = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)


def normalize_code(code: str) -> str:
    return code.replace("\r\n", "\n").strip()


def final_code(response: str) -> str:
    cleaned = response
    # QwQ's generation prefix may already contain the opening <think> tag.
    if "</think>" in cleaned and "<think>" not in cleaned:
        cleaned = cleaned.split("</think>", 1)[1]
        # Some checkpoints repeat the closing delimiter in the final answer.
        # Normalize prose only; never change tags inside complete code fences.
        parts, start = [], 0
        for block in FENCE.finditer(cleaned):
            parts.extend([cleaned[start:block.start()].replace("</think>", ""),
                          block.group(0)])
            start = block.end()
        parts.append(cleaned[start:].replace("</think>", ""))
        cleaned = "".join(parts)
    for tag in TAGS:
        cleaned = re.sub(rf"<{tag}>.*?</{tag}>", "", cleaned, flags=re.DOTALL)
    # An unfinished reasoning section cannot be a final repair.
    if any(f"<{tag}>" in cleaned or f"</{tag}>" in cleaned for tag in TAGS):
        raise ValueError("Unclosed or unmatched reasoning tag")
    blocks = [m.group(2) for m in FENCE.finditer(cleaned)
              if m.group(1).strip().lower() in ("rust", "verus", "")]
    if not blocks:
        raise ValueError("No final Rust/Verus code fence")
    code = normalize_code(blocks[-1])
    if not code:
        raise ValueError("Empty final code fence")
    return code


def ctx_target(response: str) -> str:
    """Retain reasoning and exactly one final code block; never invent reasoning."""
    parts = []
    for tag in TAGS:
        matches = list(re.finditer(rf"<{tag}>(.*?)</{tag}>", response, re.DOTALL))
        if len(matches) != 1 or len(matches[0].group(1).strip()) < 10:
            raise ValueError(f"Missing, duplicate or empty {tag} section")
        # Reasoning can contain draft code. Remove it from SFT targets so that
        # every consumer observes the same single final fenced program.
        text = FENCE.sub("", matches[0].group(1)).strip()
        if len(text) < 10:
            raise ValueError(f"Empty {tag} after removing draft code")
        parts.append(f"<{tag}>\n{text}\n</{tag}>")
    counterexample = parts[1]
    if not ("[State]" in counterexample and "[Violation]" in counterexample
            and ("[Execution Trace]" in counterexample or "[Execution]" in counterexample)):
        raise ValueError("Counterexample sections incomplete")
    parts.append(f"```rust\n{final_code(response)}\n```")
    return "\n\n".join(parts)
