import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from topoverus.code import ctx_target, final_code
from topoverus.graph import dependency_score, load_graphs, source_id
from topoverus.matcher_v2 import first_invariants, match, similarity
from topoverus.reward import RewardConfig, score_response
from topoverus.tools import CHECKER_VERSION, ToolResult, parse_verification, safety


class ExtractionTests(unittest.TestCase):
    def test_draft_is_not_final(self):
        text = "<think>```rust\nwrong\n```</think>\n```rust\nfirst\n```\n```rust\nlast\n```"
        self.assertEqual(final_code(text), "last")
        self.assertEqual(final_code("draft</think>\n```rust\nlast\n```"), "last")

    def test_no_final_answer(self):
        for text in ("<reasoning>```rust\ndraft\n```", "<think>```rust\ndraft\n```</think>"):
            with self.assertRaises(ValueError):
                final_code(text)

    def test_repeated_prefilled_thinking_close_preserves_final_code(self):
        for suffix in ("</think>", "</think></think>"):
            text = "```rust\ndraft\n```</think>\n```rust\nfn main() {}\n```" + suffix
            self.assertEqual(final_code(text), "fn main() {}")
        self.assertEqual(final_code("draft</think></think>```rust\nfinal\n```"), "final")

    def test_repeated_close_cannot_recover_draft_or_incomplete_final(self):
        for text in ("```rust\ndraft\n```</think></think>",
                     "draft</think>```rust\nincomplete</think>",
                     "<think>draft</think>```rust\nfinal\n```</think>",
                     "draft</think>```rust\nfn main() { /* </think> */ }\n```</think>"):
            with self.subTest(response=text), self.assertRaises(ValueError):
                final_code(text)

    def test_target_removes_embedded_drafts(self):
        text = ("<think>Explanation long enough ```rust\ndraft\n```</think>"
                "<counterexample>[State] x [Execution] x [Violation] x</counterexample>"
                "<reasoning>A complete repair explanation.</reasoning>```rust\nfinal\n```")
        target = ctx_target(text)
        self.assertEqual(target.count("```"), 2)
        self.assertEqual(final_code(target), "final")


class MatcherTests(unittest.TestCase):
    def test_mixed_exact_fuzzy_and_order(self):
        result = match(["b >= j && b <= n", "a >= 0"], ["a >= 0", "b >= i && b <= n"])
        self.assertEqual([m["canonical"] for m in result], [1, 0])
        self.assertEqual([m["kind"] for m in result], ["fuzzy", "exact"])

    def test_duplicate_one_to_one(self):
        self.assertEqual(len(match(["x >= 0", "x >= 0"], ["x >= 0"])), 1)

    def test_exact_reservation(self):
        result = match(["x >= j && x <= n", "x >= i && x <= n"], ["x >= i && x <= n", "x >= k && x <= n"])
        self.assertEqual([m["canonical"] for m in result], [1, 0])

    def test_threshold_and_tie(self):
        self.assertEqual(similarity("a b c", "a b d"), 0.5)
        self.assertEqual(match(["a b c"], ["a b d", "a b e"], 0.5)[0]["canonical"], 0)
        self.assertEqual(match(["a b c"], ["a b d"], 0.500001), [])

    def test_numbers_are_distinct(self):
        self.assertLess(similarity("x >= 0", "x >= 100"), 1)

    def test_multiline(self):
        code = "while true\n invariant\n  forall|i: int|\n   i >= 0 ==> p(i),\n  n >= 0,\n decreases n,\n {\n }"
        self.assertEqual(len(first_invariants(code)), 2)
        self.assertIn("p(i)", first_invariants(code)[0])


class GraphTests(unittest.TestCase):
    def test_empty_and_missing_differ(self):
        code = "invariant\n x >= 0,\n {"
        record = {"status": "valid", "invariants": ["x >= 0"], "dependency_edges": []}
        self.assertEqual(dependency_score(code, record)["score"], 1)
        self.assertEqual(dependency_score(code, record)["status"], "empty")
        self.assertEqual(dependency_score(code, None)["status"], "missing")

    def test_order_and_scc(self):
        record = {"status": "valid", "invariants": ["x >= 0", "y >= 0"],
                  "dependency_edges": [[0, 1]]}
        code = "invariant\n y >= 0,\n x >= 0,\n {"
        self.assertEqual(dependency_score(code, record)["score"], 0)
        record["dependency_edges"].append([1, 0])
        self.assertEqual(dependency_score(code, record)["score"], 1)

    def test_no_basename_identity(self):
        with self.assertRaises(ValueError):
            source_id("binary_search.rs")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                load_graphs(Path(directory) / "missing.jsonl")


class StrictToolsTests(unittest.TestCase):
    def test_return_code_required_for_success(self):
        text = "verification results:: 3 verified, 0 errors"
        self.assertEqual(parse_verification(ToolResult("completed", 0, text)).status, "verified")
        for rc in (1, 2, -9):
            self.assertEqual(parse_verification(ToolResult("completed", rc, text)).status, "tool_error")

    def test_unknown_safety_is_zero(self):
        with tempfile.TemporaryDirectory() as directory, patch("topoverus.reward.safety", return_value=ToolResult("timeout")), patch("topoverus.reward.verify") as verifier:
            result = score_response("```rust\ncode\n```", "source", None, RewardConfig(), "verus", "lynette", Path(directory))
            self.assertEqual(result["score"], 0)
            verifier.assert_not_called()

    def test_parser_error_is_unknown_not_unsafe(self):
        with tempfile.TemporaryDirectory() as directory, patch("topoverus.tools.invoke", side_effect=[
                ToolResult("completed", 0, CHECKER_VERSION), ToolResult("completed", 1, stderr="Parse error")]):
            self.assertEqual(safety("source", "candidate", "lynette", Path(directory)).status, "tool_error")

    def test_legacy_checker_cannot_certify_safe(self):
        with tempfile.TemporaryDirectory() as directory, patch("topoverus.tools.invoke", return_value=ToolResult("completed", 0, "lynette 0.0.0")) as invoke:
            result = safety("source", "source", "lynette", Path(directory))
            self.assertEqual(result.status, "tool_error")
            self.assertIn(CHECKER_VERSION, result.stderr)
            self.assertEqual(invoke.call_count, 1)

    def test_tools_share_final_code_and_weights_apply(self):
        with tempfile.TemporaryDirectory() as directory, patch("topoverus.reward.safety", return_value=ToolResult("safe")) as compare, patch("topoverus.reward.verify", return_value=ToolResult("verified", verified=1)) as verifier:
            text = "<think>```rust\ndraft\n```</think>```rust\nfinal\n```"
            result = score_response(text, "source", None, RewardConfig(dependency_enabled=False, verus_weight=2), "v", "l", Path(directory))
            self.assertEqual(compare.call_args.args[1], "final")
            self.assertEqual(verifier.call_args.args[0], "final")
            self.assertAlmostEqual(result["score"], 2 * result["r_gate"] + result["r_format"])


if __name__ == "__main__":
    unittest.main()
