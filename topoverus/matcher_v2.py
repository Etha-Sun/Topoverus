"""Lexical one-to-one invariant matching, not semantic equivalence."""

import re

VERSION = "v2.0"
THRESHOLD = 0.6
TOKEN = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+(?:[A-Za-z_][A-Za-z_0-9]*)?|==|!=|<=|>=|&&|\|\||::|->|=>|[^\s]")


def normalized(text: str) -> str:
    text = re.sub(r"//[^\n]*", "", text)
    return " ".join(text.strip().rstrip(",").split())


def first_invariants(code: str) -> list[str]:
    """The first invariant block only, matching the dependency graph scope."""
    lines = code.splitlines()
    start = next((i for i, line in enumerate(lines)
                  if re.match(r"^\s*invariant\s*(?://.*)?$", line)), None)
    if start is None:
        return []
    clauses, pending = [], []
    for line in lines[start + 1:]:
        if re.match(r"^\s*decreases\b", line) or line.strip() == "{":
            break
        line = re.sub(r"//.*", "", line).strip()
        if not line:
            continue
        if pending and pending[-1].endswith(","):
            clauses.append("\n".join(pending))
            pending = []
        pending.append(line)
    if pending:
        clauses.append("\n".join(pending))
    return clauses


def similarity(first: str, second: str) -> float:
    a, b = set(TOKEN.findall(normalized(first))), set(TOKEN.findall(normalized(second)))
    return len(a & b) / len(a | b) if a or b else 0.0


def match(candidates: list[str], canonical: list[str], threshold: float = THRESHOLD) -> list[dict]:
    if not 0 <= threshold <= 1:
        raise ValueError("Matcher threshold must be in [0, 1]")
    assigned, used = {}, set()
    # Resolve all exact matches before fuzzy matching can consume references.
    for position, candidate in enumerate(candidates):
        exact = [i for i, invariant in enumerate(canonical)
                 if i not in used and normalized(candidate) == normalized(invariant)]
        if exact:
            index = exact[0]
            assigned[position] = {"position": position, "canonical": index,
                                  "kind": "exact", "similarity": 1.0}
            used.add(index)
    for position, candidate in enumerate(candidates):
        if position in assigned:
            continue
        available = [(similarity(candidate, invariant), i)
                     for i, invariant in enumerate(canonical) if i not in used]
        if available:
            score, index = max(available, key=lambda pair: (pair[0], -pair[1]))
            if score >= threshold:
                assigned[position] = {"position": position, "canonical": index,
                                      "kind": "fuzzy", "similarity": score}
                used.add(index)
    return [assigned[position] for position in sorted(assigned)]
