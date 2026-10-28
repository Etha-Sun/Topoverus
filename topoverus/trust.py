"""Supplement Lynette: proof helpers cannot introduce new trusted assumptions."""

from collections import Counter
import re


def rust_tokens(code: str, allow_unclosed_comments: bool = False,
                _spans: list[tuple[int, int]] | None = None) -> list[str]:
    tokens, i = [], 0
    while i < len(code):
        if code[i].isspace():
            i += 1
        elif code.startswith("//", i):
            end = code.find("\n", i)
            i = len(code) if end < 0 else end
        elif code.startswith("/*", i):
            depth, i = 1, i + 2
            while i < len(code) and depth:
                if code.startswith("/*", i):
                    depth, i = depth + 1, i + 2
                elif code.startswith("*/", i):
                    depth, i = depth - 1, i + 2
                else:
                    i += 1
            if depth and not allow_unclosed_comments:
                raise ValueError("Unclosed block comment")
        else:
            raw = re.match(r'(?:br|r)(#*)"', code[i:])
            quoted = re.match(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])\'', code[i:], re.DOTALL)
            if raw:
                end_marker = '"' + raw.group(1)
                end = code.find(end_marker, i + raw.end())
                if end < 0:
                    raise ValueError("Unclosed raw string")
                end += len(end_marker)
                tokens.append(code[i:end])
                if _spans is not None:
                    _spans.append((i, end))
                i = end
            elif quoted:
                tokens.append(quoted.group())
                if _spans is not None:
                    _spans.append((i, i + quoted.end()))
                i += quoted.end()
            elif code[i] == '"':
                raise ValueError("Unclosed string")
            else:
                token = re.match(r"[A-Za-z_][A-Za-z_0-9]*|\d+|[^\s]", code[i:]).group()
                tokens.append(token)
                if _spans is not None:
                    _spans.append((i, i + len(token)))
                i += len(token)
    return tokens


def _trusted_body(tokens: list[str], start: int) -> tuple[int, int] | None:
    item_starts = {"#", "pub", "fn", "proof", "spec", "exec", "open", "closed",
                   "broadcast", "unsafe", "async", "const", "static", "struct",
                   "enum", "impl", "trait", "mod", "use", "type", "union", "axiom",
                   "verus", "macro_rules", "extern"}
    pairs = {"(": ")", "[": "]", "{": "}"}
    i = start + 1
    while i < len(tokens):
        token = tokens[i]
        if token == ";":
            return None
        if token in pairs:
            stack, end = [pairs[token]], i + 1
            while end < len(tokens) and stack:
                value = tokens[end]
                if value in pairs:
                    stack.append(pairs[value])
                elif value in pairs.values():
                    if value != stack.pop():
                        raise ValueError("Unbalanced trusted function declaration")
                end += 1
            if stack:
                raise ValueError("Unclosed trusted function declaration")
            following = tokens[end] if end < len(tokens) else None
            macro_follows = end + 1 < len(tokens) and tokens[end + 1] in ("!", ":")
            if token == "{" and (following is None or following in item_starts or following in ("}", ";") or macro_follows):
                return i, end - 1
            i = end
        else:
            i += 1
    raise ValueError("Trusted function has no body or declaration terminator")


def function_names(code: str) -> set[str]:
    tokens = rust_tokens(code)
    return {tokens[i + 1] for i, token in enumerate(tokens[:-1])
            if token == "fn" and re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", tokens[i + 1])}


def _proof_settings(code: str) -> str:
    """These verifier settings affect proof search, not trust or contracts."""
    allowed = {f"#[verifier::{value}]" for value in
               ("loop_isolation(false)", "loop_isolation(true)", "spinoff_prover")}
    allowed.update(f"#[verifier({value})]" for value in
                   ("loop_isolation(false)", "loop_isolation(true)", "spinoff_prover"))
    spans = []
    tokens = rust_tokens(code, _spans=spans)
    replacements = []
    for i, token in enumerate(tokens):
        if token != "#" or tokens[i + 1:i + 2] != ["["]:
            continue
        end = i + 2
        while end < len(tokens) and tokens[end] != "]":
            end += 1
        if "".join(tokens[i:end + 1]) in allowed:
            replacements.append((spans[i][0], spans[end][1]))
    for start, stop in reversed(replacements):
        code = code[:start] + " " + code[stop:]
    return code


def _function_start(tokens: list[str], index: int) -> int:
    depth = 0
    while index > 0:
        token = tokens[index - 1]
        if depth == 0 and token in (";", "{", "}"):
            break
        depth += (token in ("]", ")")) - (token in ("[", "("))
        index -= 1
    return index


def contract_projection(code: str, original_functions: set[str]) -> str:
    """Compare original function declarations independently of erased bodies.

    Lynette drops proof functions and empty deghosted functions. Comparison-only
    spec functions retain their signatures/contracts/attributes. Ordinary bodies
    use a common placeholder. Existing assumptions are also pinned per function.
    New proof helpers without trust constructs remain allowed. Never verify this
    synthetic representation with Verus.
    """
    code = _proof_settings(code)
    spans = []
    tokens = rust_tokens(code, _spans=spans)
    replacements, covered_until = {}, -1
    for i, token in enumerate(tokens[:-1]):
        if i <= covered_until or token != "fn" or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", tokens[i + 1]):
            continue
        mode_start, mode = i, "exec"
        if i > 0 and tokens[i - 1] in ("proof", "exec", "spec", "axiom"):
            mode_start, mode = i - 1, tokens[i - 1]
        elif i >= 4 and tokens[i - 4:i] in (["spec", "(", "checked", ")"], ["proof", "(", "axiom", ")"]):
            mode_start, mode = i - 4, tokens[i - 2]
        if mode in ("spec", "checked"):
            continue  # Normal Lynette --spec comparison already clones these.
        body = _trusted_body(tokens, i)
        start = _function_start(tokens, mode_start)
        body_code = code[spans[body[0]][0]:spans[body[1]][1]] if body else ""
        assumptions = trusted_constructs(body_code)
        header = code[spans[start][0]:spans[i][0]]
        if (tokens[i + 1] not in original_functions and mode == "proof" and body
                and not assumptions and not trusted_constructs(header)):
            replacements[spans[start][0]] = (spans[body[1]][1], "")
            covered_until = body[1]
            continue
        markers = " ".join(" ".join(value) + ";" for value, count in sorted(assumptions.items()) for _ in range(count))
        tag = f"#[topoverus_mode({mode})] #[topoverus_assumptions({markers})] "
        first = spans[mode_start][0]
        if start == mode_start:
            replacements[first] = (spans[i][0], tag + "spec ")
        else:
            prefix = spans[start][0]
            replacements[prefix] = (prefix, tag)
            replacements[first] = (spans[i][0], "spec ")
        if body:
            first, last = body
            replacements[spans[first][0]] = (spans[last][1], "{ unimplemented!() }")
            covered_until = last
    for start, (stop, replacement) in sorted(replacements.items(), reverse=True):
        code = code[:start] + replacement + code[stop:]
    # Removing a new helper must also remove its doc comments, otherwise Rust
    # attaches those comments to the next retained function. Comments do not
    # affect the proof contract. Preserve token adjacency and literal contents.
    spans = []
    rust_tokens(code, _spans=spans)
    return "".join(code[start:stop] + (" " if i + 1 < len(spans) and spans[i + 1][0] > stop else "")
                   for i, (start, stop) in enumerate(spans))


def trusted_constructs(code: str) -> Counter:
    tokens, result = rust_tokens(code), Counter()
    for i, token in enumerate(tokens):
        if token in ("assume", "admit") and i + 1 < len(tokens) and tokens[i + 1] == "(":
            depth, end = 0, i + 1
            while end < len(tokens):
                depth += (tokens[end] == "(") - (tokens[end] == ")")
                end += 1
                if depth == 0:
                    break
            if depth:
                raise ValueError("Unclosed trusted call")
            result[tuple(tokens[i:end])] += 1
        elif token == "assume_specification":
            # This construct changes the trusted API; release inputs do not use it.
            raise ValueError("assume_specification is unsupported by strict safety")
        elif token == "axiom" and tokens[i + 1:i + 2] == ["fn"]:
            result[("axiom", "fn")] += 1
        elif token == "proof" and tokens[i:i + 5] == ["proof", "(", "axiom", ")", "fn"]:
            result[("proof", "(", "axiom", ")", "fn")] += 1
        elif token == "#" and tokens[i + 1:i + 2] == ["["]:
            end = i + 2
            while end < len(tokens) and tokens[end] != "]":
                end += 1
            attribute = tokens[i:end + 1]
            if "verifier" in attribute and ("external_body" in attribute or "external" in attribute
                    or ("verify" in attribute and "false" in attribute)):
                result[tuple(attribute)] += 1
    return result


def trust_change(original: str, candidate: str) -> str | None:
    try:
        additions = trusted_constructs(candidate) - trusted_constructs(original)
    except ValueError as exc:
        return str(exc)
    return "New or modified trusted assumption/verification bypass" if additions else None
