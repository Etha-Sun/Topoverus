import json
from numbers import Real
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from topoverus import verl_adapter


class RewardAdapterTests(unittest.TestCase):
    def test_trainer_metrics_are_numeric_and_trace_preserves_statuses(self):
        result = {"score": 2.4, "r_verus": 1.0, "r_dep": 1.0,
                  "r_format": 0.4, "r_gate": 1.0, "strict_safe_pass": True,
                  "safety": {"status": "safe"}, "graph": {"status": "empty"}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "outputs").mkdir()
            with patch.object(verl_adapter, "context", return_value=(None, {}, root, root)), \
                    patch.object(verl_adapter, "score_response", return_value=result), \
                    patch.dict("os.environ", {"VERUS_BIN": "verus", "LYNETTE_BIN": "lynette"}):
                metrics = verl_adapter.compute_score(
                    "test", "response", "target",
                    {"source_id": "test/1.rs", "source_sha256": "source", "unverified_code": "code"})
            self.assertTrue(all(isinstance(value, Real) for value in metrics.values()), metrics)
            self.assertEqual(metrics["score"], 2.4)
            trace = json.loads(next((root / "outputs").glob("rollouts-*.jsonl")).read_text())
            self.assertEqual(trace["safety"]["status"], "safe")
            self.assertEqual(trace["graph"]["status"], "empty")


if __name__ == "__main__":
    unittest.main()
