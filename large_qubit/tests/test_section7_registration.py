"""Section7 implementation readiness preserves completed corpus science."""

import copy
import json
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from large_qubit import registration
from large_qubit.common import read_json, sha256_file
from large_qubit.tests import test_section6_v2_registration as fixtures


_verify_section6_v2 = registration.verify_section6_v2_registration
_section7_binding = registration.section7_implementation_binding


class Section7RegistrationTests(unittest.TestCase):
    write = fixtures.Section6V2RegistrationTests.write

    def setUp(self):
        fixtures.Section6V2RegistrationTests.setUp(self)
        self.frozen_v2 = self.write("large_qubit/acquisition_v2.py", "# frozen released-corpus sampler")
        frozen_hash = sha256_file(self.frozen_v2)
        path = self.write(registration.SECTION6_V2_MANIFEST_PATH, '{"decision":"go"}')
        self.v2_gate_hash = sha256_file(path)
        self.write(registration.SECTION6_V2_MANIFEST_PATH.removesuffix(".json") + ".sha256", self.v2_gate_hash)
        self.gate_v2 = {"path": registration.SECTION6_V2_MANIFEST_PATH, "manifest_sha256": self.v2_gate_hash,
                       "saved_certificate_and_payload_verification": {
                           "all_passed": True, "section_6_complete": True, "feasibility_gate": "go",
                           "production_ordinary_requests": 6552, "cumulative_ordinary_requests": 13090,
                           "release_sha256": "b" * 64}}

        def frozen_gate(root):
            if sha256_file(self.frozen_v2) != frozen_hash or sha256_file(path) != self.v2_gate_hash:
                raise ValueError("Frozen Section6 v2 source or released science changed")
            return self.gate_v2
        context = patch.object(registration, "section6_v2_manifest_binding", side_effect=frozen_gate)
        context.start(); self.addCleanup(context.stop)
        saved = registration.make_registration(self.root)
        saved.update(schema_version="large-qubit-section6-registration-v2", registration_id="sections_1_6_v2")
        saved.pop("section6_v2_registration_lineage")
        saved.pop("section7_implementation")
        self.v2_manifest = self.write(registration.SECTION6_V2_REGISTRATION + "/manifest.json", json.dumps(saved))
        self.v2_anchor = sha256_file(self.v2_manifest)
        self.write(registration.SECTION6_V2_REGISTRATION + "/manifest.sha256", self.v2_anchor)
        for relative in saved["bindings_sha256"]:
            self.write(registration.SECTION6_V2_ARCHIVE + "/" + relative, (self.root / relative).read_bytes())
        for name, value in (("SECTION6_V2_MANIFEST_SHA256", self.v2_anchor),
                            ("verify_section6_v2_registration", _verify_section6_v2),
                            ("section7_implementation_binding", _section7_binding)):
            context = patch.object(registration, name, value)
            context.start(); self.addCleanup(context.stop)
        self.section7_verification = {"all_passed": True, "implementation_ready": True,
            "production_training_executed": False, "production_gpu_preflight_executed": False,
            "production_optimizer_steps_executed": 0, "model_weights_loaded": False,
            "execution_ready": False, "token_caches": {"gpt2_xl": "paired", "qwen3_4b": "native"}}
        module = SimpleNamespace(verify_section7=lambda root: copy.deepcopy(self.section7_verification))
        context = patch.dict(sys.modules, {"large_qubit.section7": module})
        context.start(); self.addCleanup(context.stop)

    def ready(self):
        prep = self.write(registration.SECTION7_PREPARATION_PATH, '{"immutable":"prepared"}')
        tests_path = "large_qubit/evidence/sections_1_7_gpu_uuid_v1/unit_validation.json"
        tests = self.write(tests_path, json.dumps({"all_passed": True, "exit_code": 0, "tests_run": 18, "skipped": 0,
            "production_training_executed": False, "production_gpu_preflight_executed": False,
            "production_optimizer_steps_executed": 0, "model_weights_loaded": False, "server_connected": False}))
        value = {"schema_version": "section7-implementation-certificate-v1", "all_passed": True,
            "implementation_ready": True, "production_training_executed": False,
            "production_gpu_preflight_executed": False, "production_optimizer_steps_executed": 0,
            "model_weights_loaded": False, "server_connected": False, "execution_ready": False,
            "trained_adapters": 0,
            "unit_tests": {"all_passed": True, "tests_run": 18, "receipt": tests_path,
                           "skipped": 0, "sha256": sha256_file(tests)},
            "preparation": {"path": registration.SECTION7_PREPARATION_PATH,
                            "sha256": sha256_file(prep), "verification": self.section7_verification}}
        self.write(registration.SECTION7_IMPLEMENTATION_PATH, json.dumps(value))
        return value

    def test_old_v2_verification_uses_archive_after_training_code_changes(self):
        self.write("large_qubit/models.py", "# current training loader")
        self.write("large_qubit/training.py", "# new runtime")
        report = registration.verify_registration(self.root, registration.SECTION6_V2_REGISTRATION)
        self.assertTrue(report["all_passed"])
        self.assertFalse(report["active_source_verified_as_original"])
        self.assertTrue(report["section_6_complete"])
        self.assertEqual(report["section_6_cumulative_ordinary_requests"], 13090)

    def test_changed_archived_v2_bytes_are_rejected(self):
        self.write(registration.SECTION6_V2_ARCHIVE + "/large_qubit/protocol.py", "# changed")
        with self.assertRaisesRegex(ValueError, "archived binding changed"):
            registration.verify_section6_v2_registration(self.root)

    def test_missing_v2_archive_blocks_any_new_registration(self):
        (self.root / registration.SECTION6_V2_ARCHIVE / "large_qubit/protocol.py").unlink()
        with self.assertRaisesRegex(ValueError, "archived binding is missing"):
            registration.register(self.root)
        self.assertFalse((self.root / registration.DEFAULT_REGISTRATION).exists())

    def test_rewritten_old_v2_manifest_and_sidecar_do_not_change_anchor(self):
        self.write(registration.SECTION6_V2_REGISTRATION + "/manifest.json", "{}")
        self.write(registration.SECTION6_V2_REGISTRATION + "/manifest.sha256", sha256_file(self.v2_manifest))
        with self.assertRaisesRegex(ValueError, "immutable anchor"):
            registration.verify_section6_v2_registration(self.root)

    def test_live_scientific_v2_source_stays_frozen(self):
        self.write("large_qubit/acquisition_v2.py", "# changed sampler")
        with self.assertRaisesRegex(ValueError, "Frozen Section6 v2"):
            registration.verify_section6_v2_registration(self.root)

    def test_implementation_ready_never_claims_trained_adapters_or_gpu_validation(self):
        self.ready()
        old = self.v2_manifest.read_bytes()
        registration.register(self.root)
        saved = read_json(self.root / registration.DEFAULT_REGISTRATION / "manifest.json")
        self.assertEqual(saved["registration_id"], registration.REGISTRATION_ID)
        self.assertEqual(saved["section6_v2_registration_lineage"]["manifest_sha256"], self.v2_anchor)
        self.assertEqual(old, self.v2_manifest.read_bytes())
        report = registration.verify_registration(self.root)
        self.assertTrue(report["section_7_implementation_ready"])
        self.assertEqual(report["section_7_status"], "IMPLEMENTATION_READY")
        self.assertTrue(report["section_7_training_not_run"])
        self.assertFalse(report["section_7_training_complete"])
        self.assertFalse(report["section_7_production_gpu_preflight_verified"])
        self.assertEqual(report["section_7_trained_adapters"], 0)
        self.assertFalse(report["execution_ready"])

    def test_missing_section7_artifacts_do_not_claim_readiness(self):
        registration.register(self.root)
        report = registration.verify_registration(self.root)
        self.assertFalse(report["section_7_implementation_ready"])
        self.assertEqual(report["section_7_status"], "PENDING")

    def test_preparation_without_test_certificate_cannot_claim_ready(self):
        self.write(registration.SECTION7_PREPARATION_PATH, "{}")
        with self.assertRaisesRegex(ValueError, "local implementation certificate"):
            registration.section7_implementation_binding(self.root)

    def test_certificate_cannot_claim_production_work(self):
        for key, value in (("production_training_executed", True), ("production_gpu_preflight_executed", True),
                           ("production_optimizer_steps_executed", 1), ("model_weights_loaded", True),
                           ("server_connected", True), ("execution_ready", True), ("trained_adapters", 1)):
            with self.subTest(key=key):
                certificate = self.ready()
                certificate[key] = value
                self.write(registration.SECTION7_IMPLEMENTATION_PATH, json.dumps(certificate))
                with self.assertRaisesRegex(ValueError, "must not claim"):
                    registration.section7_implementation_binding(self.root)

    def test_real_preparation_verifier_cannot_hide_failed_cache_authentication(self):
        self.ready()
        self.section7_verification["all_passed"] = False
        with self.assertRaisesRegex(ValueError, "authentication failed"):
            registration.section7_implementation_binding(self.root)

    def test_certificate_requires_positive_passing_fixture_count(self):
        certificate = self.ready()
        certificate["unit_tests"]["tests_run"] = 0
        self.write(registration.SECTION7_IMPLEMENTATION_PATH, json.dumps(certificate))
        with self.assertRaisesRegex(ValueError, "fixture or preparation"):
            registration.section7_implementation_binding(self.root)

    def test_preparation_identity_must_match_test_certificate(self):
        self.ready()
        self.write(registration.SECTION7_PREPARATION_PATH, '{"different":"cache"}')
        with self.assertRaisesRegex(ValueError, "fixture or preparation"):
            registration.section7_implementation_binding(self.root)

    def test_local_test_receipt_cannot_change_after_certificate_creation(self):
        self.ready()
        self.write("large_qubit/evidence/sections_1_7_gpu_uuid_v1/unit_validation.json", "{}")
        with self.assertRaisesRegex(ValueError, "test receipt changed"):
            registration.section7_implementation_binding(self.root)

    def test_mutable_production_artifacts_do_not_change_source_registration(self):
        self.ready()
        registration.register(self.root)
        before = registration.current_bindings(self.root)
        self.write("large_qubit/preflights/section7_v1/device_probe.py", "# user-run artifact")
        self.write("large_qubit/training_runs/section7_v1/checkpoint/configuration.py", "# saved artifact")
        self.assertEqual(before, registration.current_bindings(self.root))
        self.assertTrue(registration.verify_registration(self.root)["all_passed"])

    def test_two_immutable_cache_manifests_are_explicitly_bound(self):
        for path in registration.SECTION7_CACHE_MANIFEST_PATHS:
            manifest = self.write(path, "{}")
            self.write(path.removesuffix(".json") + ".sha256", sha256_file(manifest))
        bindings = registration.current_bindings(self.root)
        for path in registration.SECTION7_CACHE_MANIFEST_PATHS:
            self.assertIn(path, bindings)
            self.assertIn(path.removesuffix(".json") + ".sha256", bindings)

    def test_new_training_runtime_source_is_bound(self):
        self.ready()
        self.write("large_qubit/training.py", "# runtime")
        registration.register(self.root)
        self.write("large_qubit/training.py", "# changed runtime")
        with self.assertRaisesRegex(ValueError, "changed"):
            registration.verify_registration(self.root)


if __name__ == "__main__":
    unittest.main()
