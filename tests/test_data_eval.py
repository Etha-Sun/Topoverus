import os
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from topoverus.build_graphs import failed_invariants, replace_first_block
from topoverus.data import curriculum_safety, family, resolve_trajectory, token_fingerprint
from topoverus.evaluate import generation_trace, judge, main as evaluate, metrics, pass_at_k
from topoverus.tools import ToolResult


class IdentityTests(unittest.TestCase):
    def test_colliding_basename_needs_reference_disambiguation(self):
        task = {"source_id": "cleaned-verusbench/Misc/unverified/binary_search.rs", "verified_code": "first"}
        other = {"source_id": "cleaned-verusbench/CloverBench/unverified/binary_search.rs", "verified_code": "second"}
        tasks = {row["source_id"]: row for row in (task, other)}
        row = {"messages": [{"role": "user", "content": "Benchmark: cleaned-verusbench\nProblem: binary_search\n[Buggy Code]\n```rust\ninput\n```"}, {"role": "assistant", "content": "```rust\nsecond\n```"}]}
        self.assertEqual(resolve_trajectory(row, tasks)[0]["source_id"], other["source_id"])
        other["verified_code"] = "first"
        with self.assertRaises(ValueError):
            resolve_trajectory(row, tasks)

    def test_comment_only_duplicates_and_numeric_differences(self):
        self.assertEqual(token_fingerprint("x >= 0; // one"), token_fingerprint("x>=0; // two"))
        self.assertNotEqual(token_fingerprint("x>=0;"), token_fingerprint("x>=10;"))

    def test_obfuscated_lineage(self):
        self.assertEqual(family({"source_id": "Obfuscated_Verusbench/unverified/cloverbench_array_append_strong_0.rs"}), "cleaned-verusbench/CloverBench/unverified/array_append_strong.rs")


@unittest.skipUnless(os.environ.get("LYNETTE_BIN"), "Set LYNETTE_BIN for curriculum safety regressions")
class CurriculumSafetyTests(unittest.TestCase):
    source = 'use vstd::prelude::*; verus! { proof fn target() ensures true {} } fn main() {}'

    def check(self, original, target, source=None):
        with tempfile.TemporaryDirectory() as directory:
            return curriculum_safety(original, target, source or self.source,
                                     os.environ["LYNETTE_BIN"], Path(directory)).status

    def test_intermediate_untrusted_helper_can_be_removed(self):
        original = self.source.replace('proof fn target()', 'proof fn intermediate() ensures true {} proof fn target()')
        self.assertEqual(self.check(original, self.source), 'safe')

    def test_canonical_proof_contract_cannot_be_weakened(self):
        source = self.source.replace('ensures true', 'ensures false')
        self.assertEqual(self.check(source, self.source, source), 'unsafe')

    def test_actual_execution_and_new_trust_remain_checked(self):
        source = 'use vstd::prelude::*; verus! { fn f() -> u32 { 1 } } fn main() {}'
        self.assertEqual(self.check(source.replace('{ 1 }', '{ 2 }'), source, source), 'unsafe')
        self.assertEqual(self.check(self.source, self.source.replace('ensures true {}', 'ensures true { assume(false); }')), 'unsafe')

    def test_intermediate_assumption_can_be_removed_without_changing_canonical_trust(self):
        source = 'use vstd::prelude::*; verus! { fn target() ensures true {} } fn main() {}'
        original = source.replace('ensures true {}', 'ensures true { proof { assume(false); } }')
        self.assertEqual(self.check(original, source, source), 'safe')


class GraphBuilderTests(unittest.TestCase):
    def test_removal_tracks_source_locations_not_similar_names(self):
        code = "while true\n invariant\n x >= 0,\n x >= 10,\n {\n }"
        changed, ranges = replace_first_block(code, 0)
        self.assertNotIn("x >= 0", changed)
        self.assertEqual(ranges, {1: (3, 3)})
        error = "error: invariant not satisfied at end of loop body\n --> /tmp/candidate.rs:3:1\n x >= 10"
        self.assertEqual(failed_invariants(error, ranges), [1])

    def test_unknown_diagnostic_fails_closed(self):
        with self.assertRaises(ValueError):
            failed_invariants("error: invariant not maintained", {0: (2, 2)})


class EvaluationTests(unittest.TestCase):
    def test_verus_only_does_not_call_safety_or_claim_strict_success(self):
        task = {"source_id": "bench/task", "unverified_code": "original"}
        with patch("topoverus.evaluate.safety") as check, patch(
                "topoverus.evaluate.verify", return_value=ToolResult("verified")):
            result = judge("```rust\ncandidate\n```", task, "verus", None, Path("."))
        check.assert_not_called()
        self.assertTrue(result["verified_pass"])
        self.assertIsNone(result["strict_safe_pass"])
        self.assertEqual(result["safety"], {"status": "disabled"})
        self.assertEqual(result["safety_policy"], "verus_only")

    def test_strict_success_requires_safe_and_verified(self):
        task = {"source_id": "bench/task", "unverified_code": "original"}
        for status in ("safe", "unsafe", "tool_error"):
            with self.subTest(status=status), patch(
                    "topoverus.evaluate.safety", return_value=ToolResult(status)), patch(
                    "topoverus.evaluate.verify", return_value=ToolResult("verified")):
                result = judge("```rust\ncandidate\n```", task, "verus", "lynette", Path("."))
            self.assertTrue(result["verified_pass"])
            self.assertEqual(result["strict_safe_pass"], status == "safe")

    def test_verus_only_metrics_allow_verifier_success_to_stop_early(self):
        task = {"source_id": "bench/task"}
        row = {"source_id": task["source_id"], "sample_index": 0, "iteration": 1,
               "verified_pass": True, "strict_safe_pass": None}
        values = metrics([task], [row], 1, 10, [1], strict=False)["groups"]["overall"]
        self.assertEqual(values["verified_pass:unbiased_pass@1"], 1)
        self.assertFalse(any(key.startswith("strict_safe_pass:") for key in values))
        row["verified_pass"] = False
        with self.assertRaises(ValueError):
            metrics([task], [row], 1, 10, [1], strict=False)

    def test_default_cli_refines_only_verus_failures_without_safety_feedback(self):
        root = Path(__file__).resolve().parents[1]
        (root / "runs").mkdir(exist_ok=True)
        batches = []
        tokenizer = SimpleNamespace(encode=lambda text, **kwargs: [1],
                                    apply_chat_template=lambda messages, **kwargs: repr(messages))

        def generate(prompts, parameters):
            batches.append(prompts)
            return [SimpleNamespace(prompt_token_ids=[1], outputs=[SimpleNamespace(
                text="```rust\ncandidate\n```", token_ids=[1])]) for value in prompts]

        engine = SimpleNamespace(get_tokenizer=lambda: tokenizer, generate=generate)
        vllm = SimpleNamespace(LLM=lambda **kwargs: engine, SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs))
        with tempfile.TemporaryDirectory(dir=root / "runs") as directory:
            work = Path(directory)
            tasks = work / "tasks.jsonl"
            tasks.write_text(json.dumps({"source_id": "bench/task", "unverified_code": "original",
                                         "prompt": [{"role": "user", "content": "repair"}]}) + "\n")
            command = ["topoverus.evaluate", "--tasks", str(tasks), "--run", str(work / "eval"),
                       "--model", str(work / "model"), "--verus", sys.executable,
                       "--samples", "2", "--iterations", "2", "--ks", "1", "2", "--workers", "1"]
            outcomes = [ToolResult("verified"), ToolResult("verification_failed", stderr="proof failed"),
                        ToolResult("verified")]
            with patch.dict(os.environ), patch.dict(sys.modules, {"vllm": vllm}), patch.object(sys, "argv", command), patch(
                    "topoverus.evaluate.inspect", return_value={}), patch(
                    "topoverus.evaluate.weight_hashes", return_value={}), patch(
                    "topoverus.evaluate.safety") as check, patch(
                    "topoverus.evaluate.verify", side_effect=outcomes), patch("builtins.print"):
                evaluate()
            check.assert_not_called()
            manifest = json.loads((work / "eval/manifest.json").read_text())
            summary = json.loads((work / "eval/metrics/summary.json").read_text())
            self.assertEqual(manifest["safety_policy"], "verus_only")
            self.assertEqual(set(manifest["tools_sha256"]), {"verus"})
            self.assertEqual(summary["groups"]["overall"]["verified_pass:unbiased_pass@1"], 1)
        self.assertEqual([len(batch) for batch in batches], [2, 1])
        self.assertIn("proof failed", batches[1][0])
        self.assertNotIn("Safety check failed", batches[1][0])

    def test_generated_token_counts_use_actual_ids_and_first_thinking_close(self):
        tokenizer = SimpleNamespace(encode=lambda text, **kwargs: [7, 8])
        output = SimpleNamespace(prompt_token_ids=[1, 2], outputs=[SimpleNamespace(token_ids=[3, 4, 7, 8, 5, 7, 8])])
        result = generation_trace(output, 'prompt', 42, tokenizer)
        self.assertEqual(result['prompt_token_count'], 2)
        self.assertEqual(result['generated_token_count'], 7)
        self.assertEqual(result['thinking_token_count'], 2)
        self.assertEqual(result['generation_seed'], 42)
        output.outputs[0].token_ids = [3, 4, 5]
        self.assertIsNone(generation_trace(output, 'prompt', 42, tokenizer)['thinking_token_count'])

    def test_estimator_and_first_k_are_different(self):
        tasks = [{"source_id": "bench/unverified/task.rs"}]
        records = [{"source_id": tasks[0]["source_id"], "sample_index": i, "iteration": 1,
                    "verified_pass": i == 2, "strict_safe_pass": i == 2} for i in range(3)]
        result = metrics(tasks, records, 3, 1, [1, 3])["groups"]["overall"]
        self.assertAlmostEqual(result["strict_safe_pass:unbiased_pass@1"], 1/3)
        self.assertEqual(result["strict_safe_pass:first_1_success"], 0)
        self.assertEqual(result["strict_safe_pass:first_3_success"], 1)

    def test_missing_chain_is_not_silently_zero(self):
        with self.assertRaises(ValueError):
            metrics([{"source_id": "bench/task"}], [], 3, 1, [1, 3])
        with self.assertRaises(ValueError):
            pass_at_k(2, 1, 3)

    def test_successful_chain_can_stop_early(self):
        task = {"source_id": "bench/task"}
        row = {"source_id": task["source_id"], "sample_index": 0, "iteration": 1,
               "verified_pass": True, "strict_safe_pass": True}
        self.assertEqual(metrics([task], [row], 1, 10, [1])["groups"]["overall"]["strict_safe_pass:first_1_success"], 1)
        row["strict_safe_pass"] = False
        with self.assertRaises(ValueError):
            metrics([task], [row], 1, 10, [1])


if __name__ == "__main__":
    unittest.main()
