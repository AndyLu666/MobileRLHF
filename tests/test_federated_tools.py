import json
import tempfile
import unittest
from pathlib import Path

from scripts.federated.aggregate_lora_jsonl import report_weight


class FederatedWeightTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.adapter = Path(self.tmpdir.name) / "qwen_lora_adapter.jsonl"

    def write_report(self, **fields):
        self.adapter.with_name("report.json").write_text(
            json.dumps(fields), encoding="utf-8"
        )

    def test_auto_weight_uses_objective_specific_feedback_count(self):
        self.write_report(
            objective="kto",
            binary_feedback_rows=18,
            train_samples=18,
            feedback_pairs=9,
        )
        self.assertEqual(report_weight(self.adapter, None, "auto"), 18.0)

        self.write_report(
            objective="dpo",
            feedback_pairs=7,
            train_samples=14,
            binary_feedback_rows=14,
        )
        self.assertEqual(report_weight(self.adapter, None, "auto"), 7.0)

    def test_explicit_weight_or_key_overrides_auto_selection(self):
        self.write_report(objective="kto", binary_feedback_rows=18, clients=3)
        self.assertEqual(report_weight(self.adapter, None, "clients"), 3.0)
        self.assertEqual(report_weight(self.adapter, 2.5, "auto"), 2.5)


if __name__ == "__main__":
    unittest.main()
