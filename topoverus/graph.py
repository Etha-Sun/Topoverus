"""Canonical graph joins and explicit missing/empty/inconclusive states."""

import json
from pathlib import Path

from .matcher_v2 import THRESHOLD, first_invariants, match


def source_id(path: str) -> str:
    value = Path(path)
    if value.is_absolute() or ".." in value.parts or "/unverified/" not in path:
        raise ValueError(f"Expected a canonical source-relative path: {path}")
    return value.as_posix()


def load_graphs(path: Path) -> dict[str, dict]:
    index = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            key = source_id(record["source_id"])
            if key in index:
                raise ValueError(f"Duplicate graph identity: {key}")
            invariants = record.get("invariants", [])
            edges = record.get("dependency_edges")
            if record["status"] == "valid":
                if not isinstance(edges, list) or not invariants:
                    raise ValueError(f"Invalid graph: {key}")
                for edge in edges:
                    if (len(edge) != 2 or any(type(node) is not int or
                            not 0 <= node < len(invariants) for node in edge)):
                        raise ValueError(f"Invalid graph edge: {key}")
            index[key] = record
    return index


def component_ids(count: int, edges: list[list[int]]) -> list[int]:
    # Graphs contain at most ten nodes. Transitive closure avoids a dependency.
    reach = [[i == j for j in range(count)] for i in range(count)]
    for a, b in edges:
        reach[a][b] = True
    for k in range(count):
        for i in range(count):
            for j in range(count):
                reach[i][j] |= reach[i][k] and reach[k][j]
    return [min(j for j in range(count) if reach[i][j] and reach[j][i])
            for i in range(count)]


def dependency_score(code: str, record: dict | None, threshold: float = THRESHOLD) -> dict:
    result = {"score": 0.0, "status": "missing", "coverage": 0.0,
              "edge_satisfaction": 0.0, "considered_edges": 0, "matches": []}
    if record is None:
        return result
    if record["status"] != "valid":
        result["status"] = record["status"]
        return result
    canonical, edges = record["invariants"], record["dependency_edges"]
    matches = match(first_invariants(code), canonical, threshold)
    result.update(matches=matches, coverage=len(matches) / len(canonical),
                  status="nonempty" if edges else "empty")
    components = component_ids(len(canonical), edges)
    condensed = {(components[a], components[b]) for a, b in edges
                 if components[a] != components[b]}
    positions = {}
    for item in matches:
        positions.setdefault(components[item["canonical"]], item["position"])
    considered = [(a, b) for a, b in condensed if a in positions and b in positions]
    # Preserve legacy coverage * SCC-order formula. An absent comparable edge
    # contributes 1, including known empty graphs; missing graphs contribute 0.
    satisfaction = (sum(positions[a] < positions[b] for a, b in considered) /
                    len(considered)) if considered else 1.0
    result.update(score=result["coverage"] * satisfaction,
                  edge_satisfaction=satisfaction, considered_edges=len(considered))
    return result
