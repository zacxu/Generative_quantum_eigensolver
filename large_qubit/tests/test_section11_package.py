"""Counterexamples for the new analysis artifact boundary, using disposable files."""
import json
from pathlib import Path
import tempfile
import unittest

from large_qubit.common import digest
from large_qubit.section11 import anchors, output_inventory
from large_qubit.section11_analysis import ANALYSIS, SUBMISSION, load_sealed
from large_qubit.section11_report import markdown_html


class AnalysisPackageTests(unittest.TestCase):
    def test_changed_sealed_payload_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "manifest.json"
            payload = {"requests": 16398}
            p.write_text(json.dumps({"payload": payload, "sha256": digest(payload)}))
            self.assertEqual(load_sealed(p), payload)
            p.write_text(json.dumps({"payload": {"requests": 16382}, "sha256": digest(payload)}))
            with self.assertRaisesRegex(ValueError, "Changed envelope"):
                load_sealed(p)

    def test_output_additions_and_mutations_change_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in (ANALYSIS, SUBMISSION): (root / name).mkdir(parents=True)
            p = root / ANALYSIS / "analysis.json"
            p.write_text("original")
            before = output_inventory(root)
            p.write_text("modified")
            self.assertNotEqual(before, output_inventory(root))
            p.write_text("original")
            (root / SUBMISSION / "extra.txt").write_text("unsealed addition")
            self.assertNotEqual(before, output_inventory(root))

    def test_anchor_collection_does_not_treat_path_mentions_as_hashes(self):
        a = {"guide": {"path": "mere-mention"}, "nested": [{"path": "data", "sha256": "hash"}]}
        self.assertEqual(list(anchors(a)), [{"path": "data", "sha256": "hash"}])

    def test_heading_does_not_swallow_the_following_paragraph(self):
        rendered = markdown_html("### Scope\nOnly one seed.\n\nSeparate paragraph.")
        self.assertIn("<h2>Scope</h2>", rendered)
        self.assertIn("<p>Only one seed.</p>", rendered)


if __name__ == "__main__":
    unittest.main()
