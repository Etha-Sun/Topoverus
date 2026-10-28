import json
import os
from pathlib import Path
import struct
import tempfile
import unittest

from topoverus.checkpoint import file_hash, inspect, weight_hashes


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.weights = self.root / "model.safetensors"
        header = json.dumps({"weight": {"dtype": "F32", "shape": [1],
                                         "data_offsets": [0, 4]}}).encode()
        self.weights.write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<f", 1.0))
        self.certificate = self.root / "certificate.json"
        stat = self.weights.stat()
        self.record = {"status": "validated", "directory": str(self.root),
                       "sha256": {self.weights.name: file_hash(self.weights)},
                       "shard_stats": {self.weights.name: {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}}}
        self.certificate.write_text(json.dumps(self.record))

    def tearDown(self):
        self.temporary.cleanup()

    def test_certificate_matches_real_weight_hash(self):
        self.assertEqual(weight_hashes(self.root), weight_hashes(self.root, self.certificate))

    def test_changed_or_wrong_checkpoint_certificate_is_rejected(self):
        stat = self.weights.stat()
        os.utime(self.weights, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
        with self.assertRaisesRegex(ValueError, "changed"):
            weight_hashes(self.root, self.certificate)
        self.record["directory"] = str(self.root / "different")
        self.certificate.write_text(json.dumps(self.record))
        with self.assertRaisesRegex(ValueError, "different"):
            weight_hashes(self.root, self.certificate)

    def test_truncated_weights_are_rejected_before_certificate_reuse(self):
        self.weights.write_bytes(self.weights.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "Truncated"):
            inspect(self.root)


if __name__ == "__main__":
    unittest.main()
