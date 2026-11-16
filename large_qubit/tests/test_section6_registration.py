"""Section 6 preserves anchored Section 5 source and both live science gates."""

import json
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from large_qubit.common import read_json, sha256_file
from large_qubit import registration
from large_qubit.tests import test_section5_registration as fixtures


_verify_section5 = registration.verify_section5_registration


class Section6RegistrationTests(unittest.TestCase):
    write = fixtures.Section5RegistrationTests.write

    def setUp(self):
        fixtures.Section5RegistrationTests.setUp(self)
        self.frozen5_path = self.write("large_qubit/catalogue.py", "# frozen Section 5 scientific source")
        self.frozen5_hash = sha256_file(self.frozen5_path)
        self.gate5 = {"saved_certificate_and_payload_verification": {
            "all_passed": True, "correctness_passed": True, "section_5_complete": True,
            "feasibility_gate": "go", "ordinary_requests": 909, "selected_length": 32}}

        def live_gate(root):
            if sha256_file(self.frozen5_path) != self.frozen5_hash:
                raise ValueError("Frozen Section 5 scientific source changed")
            return self.gate5
        context = patch.object(registration, "section5_manifest_binding", side_effect=live_gate)
        context.start()
        self.addCleanup(context.stop)
        bindings = registration.current_bindings(self.root)
        saved = {
            "schema_version": "large-qubit-section5-registration-v1",
            "registration_id": "sections_1_5_v1", "execution_ready": False,
            "bindings_sha256": bindings,
            "original_registration_lineage": registration.verify_original_registration(self.root),
            "section3_registration_lineage": registration.verify_section3_registration(self.root),
            "section4_registration_lineage": registration.verify_section4_registration(self.root),
            "reuse_inventory": registration.build_reuse_inventory(self.root),
            "base_asset_metadata_verification": {"gpt2_xl": self.assets},
            "grid_amendment_verification": registration.verify_grid_amendment(self.root),
            "realized_physics_manifest": None, "section4_gate_manifest": self.gate,
            "section5_gate_manifest": self.gate5,
        }
        self.section5_path = self.write(registration.SECTION5_REGISTRATION + "/manifest.json", json.dumps(saved))
        self.section5_anchor = sha256_file(self.section5_path)
        self.write(registration.SECTION5_REGISTRATION + "/manifest.sha256", self.section5_anchor)
        for relative in bindings:
            self.write(registration.SECTION5_ARCHIVE + "/" + relative, (self.root / relative).read_bytes())
        context = patch.object(registration, "SECTION5_MANIFEST_SHA256", self.section5_anchor)
        context.start()
        self.addCleanup(context.stop)
        context = patch.object(registration, "verify_section5_registration", _verify_section5)
        context.start()
        self.addCleanup(context.stop)

    def test_old_section5_verifies_archive_after_active_implementation_changes(self):
        self.write("large_qubit/protocol.py", "# Section 6 active source")
        report = registration.verify_registration(self.root, registration.SECTION5_REGISTRATION)
        self.assertTrue(report["all_passed"])
        self.assertFalse(report["active_source_verified_as_original"])
        self.assertTrue(report["live_section4_and_section5_scientific_sources_unchanged"])
        self.assertEqual(report["source_verification_mode"],
                         "archived_sections_1_5_sources_and_live_saved_evidence")

    def test_archived_section5_bytes_cannot_change(self):
        self.write(registration.SECTION5_ARCHIVE + "/large_qubit/protocol.py", "# changed")
        with self.assertRaisesRegex(ValueError, "Section 5 archived binding changed"):
            registration.verify_section5_registration(self.root)

    def test_missing_section5_archive_blocks_new_registration(self):
        (self.root / registration.SECTION5_ARCHIVE / "large_qubit/protocol.py").unlink()
        with self.assertRaisesRegex(ValueError, "Section 5 archived binding is missing"):
            registration.register(self.root)
        self.assertFalse((self.root / registration.DEFAULT_REGISTRATION).exists())

    def test_rewritten_manifest_and_sidecar_do_not_replace_section5_anchor(self):
        saved = read_json(self.section5_path)
        saved["execution_ready"] = True
        self.section5_path.write_text(json.dumps(saved), encoding="utf-8")
        self.write(registration.SECTION5_REGISTRATION + "/manifest.sha256", sha256_file(self.section5_path))
        with self.assertRaisesRegex(ValueError, "immutable anchor"):
            registration.verify_section5_registration(self.root)

    def test_section4_live_kernel_remains_frozen(self):
        self.write("large_qubit/evaluator.py", "# changed Section 4 kernel")
        with self.assertRaisesRegex(ValueError, "Frozen Section 4 scientific source changed"):
            registration.verify_section5_registration(self.root)

    def test_section5_live_kernel_remains_frozen(self):
        self.write("large_qubit/catalogue.py", "# changed Section 5 kernel")
        with self.assertRaisesRegex(ValueError, "Frozen Section 5 scientific source changed"):
            registration.verify_section5_registration(self.root)

    def test_active_registration_preserves_anchored_section5_lineage(self):
        old_bytes = self.section5_path.read_bytes()
        registration.register(self.root)
        saved = read_json(self.root / registration.DEFAULT_REGISTRATION / "manifest.json")
        self.assertEqual(saved["registration_id"], registration.REGISTRATION_ID)
        self.assertEqual(saved["schema_version"], "large-qubit-section7-registration-v2")
        self.assertEqual(saved["section5_registration_lineage"]["manifest_sha256"], self.section5_anchor)
        self.assertEqual(old_bytes, self.section5_path.read_bytes())

    def test_missing_section6_gate_does_not_claim_corpus_completion(self):
        registration.register(self.root)
        report = registration.verify_registration(self.root)
        self.assertFalse(report["section_6_saved_gate_verified"])
        self.assertFalse(report["section_6_complete"])
        self.assertIsNone(report["section_6_release_sha256"])
        self.assertIsNone(report["section_6_production_ordinary_requests"])
        self.assertFalse(report["execution_ready"])

    def test_optional_gate_no_go_is_visible_without_execution_readiness(self):
        gate = {"saved_certificate_and_payload_verification": {
            "all_passed": True, "section_6_complete": False, "feasibility_gate": "no_go",
            "release_sha256": None}}
        with patch.object(registration, "section6_v2_manifest_binding", return_value=gate):
            registration.register(self.root)
            report = registration.verify_registration(self.root)
        self.assertTrue(report["section_6_saved_gate_verified"])
        self.assertFalse(report["section_6_complete"])
        self.assertEqual(report["section_6_feasibility_gate"], "no_go")
        self.assertFalse(report["execution_ready"])

    def test_optional_gate_go_exposes_release_but_not_training_readiness(self):
        gate = {"saved_certificate_and_payload_verification": {
            "all_passed": True, "section_6_complete": True, "feasibility_gate": "go",
            "release_sha256": "a" * 64, "production_ordinary_requests": 8192}}
        with patch.object(registration, "section6_v2_manifest_binding", return_value=gate):
            registration.register(self.root)
            report = registration.verify_registration(self.root)
        self.assertTrue(report["section_6_complete"])
        self.assertEqual(report["section_6_release_sha256"], "a" * 64)
        self.assertEqual(report["section_6_production_ordinary_requests"], 8192)
        self.assertFalse(report["execution_ready"])

    def test_later_preparation_changes_require_a_new_registration(self):
        self.write(registration.SECTION6_PREPARATION_PATH, "{}")
        registration.register(self.root)
        self.write(registration.SECTION6_PREPARATION_PATH, '{"changed":true}')
        with self.assertRaisesRegex(ValueError, "changed"):
            registration.verify_registration(self.root)

    def test_new_release_is_bound_even_without_optional_scientific_gate(self):
        release = self.write(registration.SECTION6_RELEASE_MANIFEST_PATH, "{}")
        self.write(registration.SECTION6_RELEASE_MANIFEST_PATH.removesuffix(".json") + ".sha256", sha256_file(release))
        bindings = registration.current_bindings(self.root)
        self.assertIn(registration.SECTION6_RELEASE_MANIFEST_PATH, bindings)
        registration.register(self.root)
        self.write(registration.SECTION6_RELEASE_MANIFEST_PATH, '{"changed":true}')
        with self.assertRaisesRegex(ValueError, "changed"):
            registration.verify_registration(self.root)

    def test_optional_policy_document_cannot_be_added_after_registration(self):
        registration.register(self.root)
        self.write(registration.SECTION6_DOCUMENTS[0], "{}")
        with self.assertRaisesRegex(ValueError, "changed"):
            registration.verify_registration(self.root)

    def test_gate_sidecar_failure_prevents_any_new_registration(self):
        self.write(registration.SECTION6_MANIFEST_PATH, "{}")
        self.write(registration.SECTION6_MANIFEST_PATH.removesuffix(".json") + ".sha256", "0" * 64)
        with self.assertRaisesRegex(ValueError, "Section 6 gate manifest sidecar mismatch"):
            registration.register(self.root)
        self.assertFalse((self.root / registration.DEFAULT_REGISTRATION).exists())

    def test_section6_verifier_is_explicitly_read_only(self):
        gate = self.write(registration.SECTION6_MANIFEST_PATH, "{}")
        self.write(registration.SECTION6_MANIFEST_PATH.removesuffix(".json") + ".sha256", sha256_file(gate))
        calls = []

        def verify(root, **kwargs):
            calls.append(kwargs)
            return {"all_passed": True}
        module = SimpleNamespace(verify_section6=verify)
        with patch.dict(sys.modules, {"large_qubit.section6": module}):
            result = registration.section6_manifest_binding(self.root)
        self.assertFalse(result["scientific_work_rerun_by_registration"])
        self.assertEqual(calls, [{"section6_root": "large_qubit/validation/section6_v1",
                                  "numerical_integrity": False}])

    def test_failed_section6_payload_authentication_is_refused(self):
        gate = self.write(registration.SECTION6_MANIFEST_PATH, "{}")
        self.write(registration.SECTION6_MANIFEST_PATH.removesuffix(".json") + ".sha256", sha256_file(gate))
        module = SimpleNamespace(verify_section6=lambda *args, **kwargs: {"all_passed": False})
        with patch.dict(sys.modules, {"large_qubit.section6": module}):
            with self.assertRaisesRegex(ValueError, "payload authentication failed"):
                registration.section6_manifest_binding(self.root)


if __name__ == "__main__":
    unittest.main()
