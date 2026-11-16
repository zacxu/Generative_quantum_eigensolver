"""Section 5 archives Section 4 while preserving its live scientific sources."""

import json
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from large_qubit.common import read_json, sha256_file
from large_qubit import registration
from large_qubit.tests import test_registration_versions as fixtures


_verify_section4 = registration.verify_section4_registration


class Section5RegistrationTests(unittest.TestCase):
    write = fixtures.RegistrationVersionTests.write

    def setUp(self):
        fixtures.RegistrationVersionTests.setUp(self)
        self.frozen_path = self.write("large_qubit/evaluator.py", "# frozen Section 4 scientific source")
        self.frozen_hash = sha256_file(self.frozen_path)
        self.gate = {"saved_certificate_and_payload_verification": {
            "all_passed": True, "correctness_passed": True, "reference_converged": True,
            "section_4_complete": True, "feasibility_gate": "go_local_cpu_under_declared_window"}}

        def live_gate(root):
            if sha256_file(self.frozen_path) != self.frozen_hash:
                raise ValueError("Frozen Section 4 scientific source changed")
            return self.gate
        context = patch.object(registration, "section4_manifest_binding", side_effect=live_gate)
        context.start()
        self.addCleanup(context.stop)
        bindings = registration.current_bindings(self.root)
        saved = {
            "schema_version": "large-qubit-section4-registration-v1",
            "registration_id": "sections_1_4_v1", "execution_ready": False,
            "bindings_sha256": bindings,
            "original_registration_lineage": registration.verify_original_registration(self.root),
            "section3_registration_lineage": registration.verify_section3_registration(self.root),
            "reuse_inventory": registration.build_reuse_inventory(self.root),
            "base_asset_metadata_verification": {"gpt2_xl": self.assets},
            "grid_amendment_verification": registration.verify_grid_amendment(self.root),
            "realized_physics_manifest": None, "section4_gate_manifest": self.gate,
        }
        self.section4_path = self.write(registration.SECTION4_REGISTRATION + "/manifest.json", json.dumps(saved))
        self.section4_anchor = sha256_file(self.section4_path)
        self.write(registration.SECTION4_REGISTRATION + "/manifest.sha256", self.section4_anchor)
        for relative in bindings:
            self.write(registration.SECTION4_ARCHIVE + "/" + relative, (self.root / relative).read_bytes())
        context = patch.object(registration, "SECTION4_MANIFEST_SHA256", self.section4_anchor)
        context.start()
        self.addCleanup(context.stop)
        context = patch.object(registration, "verify_section4_registration", _verify_section4)
        context.start()
        self.addCleanup(context.stop)

    def test_explicit_old_registration_authenticates_archive_after_active_source_changes(self):
        self.write("large_qubit/protocol.py", "# Section 5 current source")
        report = registration.verify_registration(self.root, registration.SECTION4_REGISTRATION)
        self.assertTrue(report["all_passed"])
        self.assertFalse(report["active_source_verified_as_original"])
        self.assertTrue(report["live_section4_scientific_sources_unchanged"])
        self.assertEqual(report["source_verification_mode"],
                         "archived_sections_1_4_sources_and_live_saved_evidence")

    def test_live_scientific_source_cannot_change_even_when_archive_is_preserved(self):
        self.write("large_qubit/evaluator.py", "# changed kernel")
        with self.assertRaisesRegex(ValueError, "Frozen Section 4 scientific source changed"):
            registration.verify_section4_registration(self.root)

    def test_tampered_archived_section4_source_is_rejected(self):
        self.write(registration.SECTION4_ARCHIVE + "/large_qubit/protocol.py", "# tampered")
        with self.assertRaisesRegex(ValueError, "Section 4 archived binding changed"):
            registration.verify_section4_registration(self.root)

    def test_rewritten_old_manifest_and_sidecar_do_not_change_anchor(self):
        saved = read_json(self.section4_path)
        saved["execution_ready"] = True
        self.section4_path.write_text(json.dumps(saved), encoding="utf-8")
        self.write(registration.SECTION4_REGISTRATION + "/manifest.sha256", sha256_file(self.section4_path))
        with self.assertRaisesRegex(ValueError, "immutable anchor"):
            registration.verify_section4_registration(self.root)

    def test_new_generation_preserves_all_earlier_registration_lineages(self):
        original_bytes = self.section4_path.read_bytes()
        registration.register(self.root)
        saved = read_json(self.root / registration.DEFAULT_REGISTRATION / "manifest.json")
        self.assertEqual(saved["registration_id"], registration.REGISTRATION_ID)
        self.assertEqual(saved["section4_registration_lineage"]["manifest_sha256"], self.section4_anchor)
        self.assertEqual(saved["original_registration_lineage"]["manifest_sha256"], self.original_digest)
        self.assertTrue(saved["section3_registration_lineage"]["all_passed"])
        self.assertEqual(original_bytes, self.section4_path.read_bytes())

    def test_absent_section5_gate_never_claims_completed_experiment(self):
        registration.register(self.root)
        saved = read_json(self.root / registration.DEFAULT_REGISTRATION / "manifest.json")
        self.assertIsNone(saved["section5_gate_manifest"])
        report = registration.verify_registration(self.root)
        self.assertFalse(report["section_5_saved_gate_verified"])
        self.assertFalse(report["section_5_complete"])
        self.assertFalse(report["execution_ready"])

    def test_scientific_no_go_stays_visible_after_artifact_authentication(self):
        gate = {"saved_certificate_and_payload_verification": {
            "all_passed": True, "correctness_passed": True,
            "section_5_complete": True, "feasibility_gate": "no_go"}}
        with patch.object(registration, "section5_manifest_binding", return_value=gate):
            registration.register(self.root)
            report = registration.verify_registration(self.root)
        self.assertTrue(report["section_5_saved_gate_verified"])
        self.assertEqual(report["section_5_feasibility_gate"], "no_go")
        self.assertFalse(report["execution_ready"])

    def test_section5_catalogue_changes_require_a_new_registration(self):
        registration.register(self.root)
        self.write("large_qubit/catalogues/co2_screened_v1.json", '{"changed": true}')
        with self.assertRaisesRegex(ValueError, "changed"):
            registration.verify_registration(self.root)

    def test_gate_sidecar_mismatch_blocks_registration_before_writes(self):
        self.write(registration.SECTION5_MANIFEST_PATH, "{}")
        self.write(registration.SECTION5_MANIFEST_PATH.removesuffix(".json") + ".sha256", "0" * 64)
        with self.assertRaisesRegex(ValueError, "Section 5 gate manifest sidecar mismatch"):
            registration.register(self.root)
        self.assertFalse((self.root / registration.DEFAULT_REGISTRATION).exists())

    def test_failed_gate_authentication_blocks_registration(self):
        gate = self.write(registration.SECTION5_MANIFEST_PATH, "{}")
        self.write(registration.SECTION5_MANIFEST_PATH.removesuffix(".json") + ".sha256", sha256_file(gate))
        module = SimpleNamespace(verify_section5=lambda *args, **kwargs: {"all_passed": False})
        with patch.dict(sys.modules, {"large_qubit.section5": module}):
            with self.assertRaisesRegex(ValueError, "payload authentication failed"):
                registration.section5_manifest_binding(self.root)


if __name__ == "__main__":
    unittest.main()
