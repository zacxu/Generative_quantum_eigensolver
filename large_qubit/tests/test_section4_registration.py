"""Section 4 preserves anchored Section 3 source and scientific evidence."""

import json
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from large_qubit.common import read_json, sha256_file
from large_qubit import registration
from large_qubit.tests import test_registration_versions as fixtures


_verify_section3 = registration.verify_section3_registration


class Section4RegistrationTests(unittest.TestCase):
    write = fixtures.RegistrationVersionTests.write

    def setUp(self):
        fixtures.RegistrationVersionTests.setUp(self)
        bindings = registration.current_bindings(self.root)
        saved = {
            "schema_version": "large-qubit-section3-registration-v1",
            "registration_id": "sections_1_3_v1", "execution_ready": False,
            "bindings_sha256": bindings,
            "original_registration_lineage": registration.verify_original_registration(self.root),
            "reuse_inventory": registration.build_reuse_inventory(self.root),
            "base_asset_metadata_verification": {"gpt2_xl": self.assets},
            "grid_amendment_verification": registration.verify_grid_amendment(self.root),
            "realized_physics_manifest": None,
        }
        self.section3_path = self.write(registration.SECTION3_REGISTRATION + "/manifest.json", json.dumps(saved))
        self.section3_anchor = sha256_file(self.section3_path)
        self.write(registration.SECTION3_REGISTRATION + "/manifest.sha256", self.section3_anchor)
        for relative in bindings:
            self.write(registration.SECTION3_ARCHIVE + "/" + relative, (self.root / relative).read_bytes())
        context = patch.object(registration, "SECTION3_MANIFEST_SHA256", self.section3_anchor)
        context.start()
        self.addCleanup(context.stop)
        context = patch.object(registration, "verify_section3_registration", _verify_section3)
        context.start()
        self.addCleanup(context.stop)

    def test_section3_explicit_verification_uses_its_archive_after_active_edits(self):
        self.write("large_qubit/protocol.py", "# Section 4 current code")
        report = registration.verify_registration(self.root, registration.SECTION3_REGISTRATION)
        self.assertTrue(report["all_passed"])
        self.assertFalse(report["active_source_verified_as_original"])
        self.assertEqual(report["source_verification_mode"],
                         "archived_sections_1_3_sources_and_live_saved_evidence")

    def test_tampered_section3_archive_fails(self):
        self.write(registration.SECTION3_ARCHIVE + "/large_qubit/protocol.py", "# tampered")
        with self.assertRaisesRegex(ValueError, "Section 3 archived binding changed"):
            registration.verify_section3_registration(self.root)

    def test_rewritten_section3_manifest_and_sidecar_cannot_change_anchor(self):
        saved = read_json(self.section3_path)
        saved["execution_ready"] = True
        self.section3_path.write_text(json.dumps(saved), encoding="utf-8")
        self.write(registration.SECTION3_REGISTRATION + "/manifest.sha256", sha256_file(self.section3_path))
        with self.assertRaisesRegex(ValueError, "immutable anchor"):
            registration.verify_section3_registration(self.root)

    def test_active_registration_binds_both_immutable_generations(self):
        section3_bytes = self.section3_path.read_bytes()
        registration.register(self.root)
        saved = read_json(self.root / registration.DEFAULT_REGISTRATION / "manifest.json")
        self.assertEqual(saved["registration_id"], registration.REGISTRATION_ID)
        self.assertEqual(saved["section3_registration_lineage"]["manifest_sha256"], self.section3_anchor)
        self.assertEqual(saved["original_registration_lineage"]["manifest_sha256"], self.original_digest)
        self.assertEqual(self.section3_path.read_bytes(), section3_bytes)

    def test_missing_section3_snapshot_blocks_active_registration(self):
        (self.root / registration.SECTION3_ARCHIVE / "large_qubit/protocol.py").unlink()
        with self.assertRaisesRegex(ValueError, "Section 3 archived binding is missing"):
            registration.register(self.root)
        self.assertFalse((self.root / registration.DEFAULT_REGISTRATION).exists())

    def test_gate_sidecar_failure_prevents_registration(self):
        self.write(registration.SECTION4_MANIFEST_PATH, "{}")
        self.write(registration.SECTION4_MANIFEST_PATH.removesuffix(".json") + ".sha256", "0" * 64)
        with self.assertRaisesRegex(ValueError, "Section 4 gate manifest sidecar mismatch"):
            registration.register(self.root)

    def test_absent_gate_is_recorded_without_claiming_scientific_readiness(self):
        registration.register(self.root)
        saved = read_json(self.root / registration.DEFAULT_REGISTRATION / "manifest.json")
        self.assertIsNone(saved["section4_gate_manifest"])
        report = registration.verify_registration(self.root)
        self.assertFalse(report["section_4_saved_gate_verified"])
        self.assertFalse(report["reference_and_circuit_evaluator_validated"])
        self.assertFalse(report["execution_ready"])

    def test_scientific_no_go_is_preserved_separately_from_correctness(self):
        gate = {"saved_certificate_and_payload_verification": {
            "all_passed": True, "correctness_passed": True, "reference_converged": True,
            "feasibility_gate": "no_go", "section_4_complete": True}}
        with patch.object(registration, "section4_manifest_binding", return_value=gate):
            registration.register(self.root)
            report = registration.verify_registration(self.root)
        self.assertTrue(report["reference_and_circuit_evaluator_validated"])
        self.assertTrue(report["section_4_complete"])
        self.assertEqual(report["section_4_feasibility_gate"], "no_go")
        self.assertFalse(report["execution_ready"])

    def test_authenticated_failed_correctness_is_not_evaluator_validation(self):
        gate = {"saved_certificate_and_payload_verification": {
            "all_passed": True, "correctness_passed": False, "reference_converged": False,
            "feasibility_gate": "no_go", "section_4_complete": False}}
        with patch.object(registration, "section4_manifest_binding", return_value=gate):
            registration.register(self.root)
            report = registration.verify_registration(self.root)
        self.assertTrue(report["section_4_saved_gate_verified"])
        self.assertFalse(report["reference_and_circuit_evaluator_validated"])
        self.assertFalse(report["section_4_complete"])

    def test_failed_gate_payload_authentication_cannot_be_registered(self):
        gate = self.write(registration.SECTION4_MANIFEST_PATH, "{}")
        self.write(registration.SECTION4_MANIFEST_PATH.removesuffix(".json") + ".sha256", sha256_file(gate))
        module = SimpleNamespace(verify_section4=lambda *args, **kwargs: {"all_passed": False})
        with patch.dict(sys.modules, {"large_qubit.section4": module}):
            with self.assertRaisesRegex(ValueError, "payload authentication failed"):
                registration.section4_manifest_binding(self.root)


if __name__ == "__main__":
    unittest.main()
