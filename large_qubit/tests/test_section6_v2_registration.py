"""An approved fresh campaign preserves v1 negative evidence and archived code."""

import copy
import json
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from large_qubit import registration
from large_qubit.common import read_json, sha256_file
from large_qubit.tests import test_section6_registration as fixtures


_verify_section6 = registration.verify_section6_registration
_verify_amendment = registration.verify_acquisition_amendment


class Section6V2RegistrationTests(unittest.TestCase):
    write = fixtures.Section6RegistrationTests.write

    def setUp(self):
        fixtures.Section6RegistrationTests.setUp(self)
        self.frozen6 = self.write("large_qubit/acquisition.py", "# v1 frozen sampler")
        frozen_hash = sha256_file(self.frozen6)
        gate_file = self.write(registration.SECTION6_MANIFEST_PATH, '{"decision":"no_go"}')
        self.gate_hash = sha256_file(gate_file)
        self.write(registration.SECTION6_MANIFEST_PATH.removesuffix(".json") + ".sha256", self.gate_hash)
        self.gate6 = {"path": registration.SECTION6_MANIFEST_PATH, "manifest_sha256": self.gate_hash,
                      "saved_certificate_and_payload_verification": {
                          "all_passed": True, "section_6_complete": False, "feasibility_gate": "no_go",
                          "release_sha256": None, "production_ordinary_requests": 6538}}

        def frozen_gate(root):
            if sha256_file(self.frozen6) != frozen_hash or sha256_file(gate_file) != self.gate_hash:
                raise ValueError("Frozen Section6 v1 source or negative evidence changed")
            return self.gate6
        context = patch.object(registration, "section6_manifest_binding", side_effect=frozen_gate)
        context.start(); self.addCleanup(context.stop)
        bindings = registration.current_bindings(self.root)
        saved = {
            "schema_version": "large-qubit-section6-registration-v1", "registration_id": "sections_1_6_v1",
            "execution_ready": False, "bindings_sha256": bindings,
            "original_registration_lineage": registration.verify_original_registration(self.root),
            "section3_registration_lineage": registration.verify_section3_registration(self.root),
            "section4_registration_lineage": registration.verify_section4_registration(self.root),
            "section5_registration_lineage": registration.verify_section5_registration(self.root),
            "reuse_inventory": registration.build_reuse_inventory(self.root),
            "base_asset_metadata_verification": {"gpt2_xl": self.assets},
            "grid_amendment_verification": registration.verify_grid_amendment(self.root),
            "realized_physics_manifest": None, "section4_gate_manifest": self.gate,
            "section5_gate_manifest": self.gate5, "section6_gate_manifest": self.gate6,
        }
        self.old_manifest = self.write(registration.SECTION6_REGISTRATION + "/manifest.json", json.dumps(saved))
        self.old_anchor = sha256_file(self.old_manifest)
        self.write(registration.SECTION6_REGISTRATION + "/manifest.sha256", self.old_anchor)
        for relative in bindings:
            self.write(registration.SECTION6_ARCHIVE + "/" + relative, (self.root / relative).read_bytes())
        for name, value in (("SECTION6_MANIFEST_SHA256", self.old_anchor),
                            ("verify_section6_registration", _verify_section6)):
            context = patch.object(registration, name, value)
            context.start(); self.addCleanup(context.stop)
        quality = {"fixed_weight_lambdas": [.25, .1], "minimum_weight_ESS_fraction": .05,
                   "maximum_single_row_weight": .05}
        self.write("large_qubit/section6_policy.json", json.dumps({"acquisition": {"quality_policy": quality}}))
        self.amendment = {
            "approved": True, "authorization": {"verbatim_user_answer": "Approve one fresh v2 CPU campaign"},
            "previous_attempt": {"registration_sha256": self.old_anchor, "ordinary_requests": 6538,
                                 "gate_path": registration.SECTION6_MANIFEST_PATH, "gate_sha256": self.gate_hash},
            "prospective_campaign": {"fresh_ordinary_request_ceiling": 8192,
                "maximum_cumulative_ordinary_requests": 14730, "maximum_wall_seconds": 86400,
                "mixture": {"guided": .4, "uniform": .2, "local": .4}},
            "unchanged_quality_policy": quality,
            "stopping_and_scope": {"authorized_fresh_campaigns": 1, "automatic_repeated_campaigns": False,
                                  "post_label_lambda_threshold_seed_or_mixture_retuning": False}}
        self.amendment_path = self.write(registration.ACQUISITION_AMENDMENT_PATH, json.dumps(self.amendment))
        self.amendment_anchor = sha256_file(self.amendment_path)
        for name, value in (("ACQUISITION_AMENDMENT_SHA256", self.amendment_anchor),
                            ("verify_acquisition_amendment", _verify_amendment)):
            context = patch.object(registration, name, value)
            context.start(); self.addCleanup(context.stop)

    def test_old_v1_snapshot_authenticates_archive_and_negative_evidence_after_v2_edits(self):
        self.write("large_qubit/protocol.py", "# active v2 protocol")
        self.write("large_qubit/acquisition_v2.py", "# new sampler")
        report = registration.verify_registration(self.root, registration.SECTION6_REGISTRATION)
        self.assertTrue(report["all_passed"])
        self.assertFalse(report["active_source_verified_as_original"])
        self.assertFalse(report["section_6_complete"])
        self.assertEqual(report["section_6_feasibility_gate"], "no_go")
        self.assertEqual(report["section_6_production_ordinary_requests"], 6538)

    def test_changed_v1_archive_is_refused(self):
        self.write(registration.SECTION6_ARCHIVE + "/large_qubit/protocol.py", "# changed")
        with self.assertRaisesRegex(ValueError, "archived binding changed"):
            registration.verify_section6_registration(self.root)

    def test_missing_archive_blocks_new_registration(self):
        (self.root / registration.SECTION6_ARCHIVE / "large_qubit/protocol.py").unlink()
        with self.assertRaisesRegex(ValueError, "archived binding is missing"):
            registration.register(self.root)
        self.assertFalse((self.root / registration.DEFAULT_REGISTRATION).exists())

    def test_rewriting_v1_manifest_and_sidecar_cannot_change_anchored_identity(self):
        self.write(registration.SECTION6_REGISTRATION + "/manifest.json", "{}")
        self.write(registration.SECTION6_REGISTRATION + "/manifest.sha256", sha256_file(self.old_manifest))
        with self.assertRaisesRegex(ValueError, "immutable anchor"):
            registration.verify_section6_registration(self.root)

    def test_live_v1_scientific_source_remains_frozen_even_with_valid_archive(self):
        self.write("large_qubit/acquisition.py", "# changed v1 sampler")
        with self.assertRaisesRegex(ValueError, "Frozen Section6 v1"):
            registration.verify_section6_registration(self.root)

    def test_active_v2_registration_keeps_v1_negative_lineage_and_authorization(self):
        original = self.old_manifest.read_bytes()
        registration.register(self.root)
        saved = read_json(self.root / registration.DEFAULT_REGISTRATION / "manifest.json")
        self.assertEqual(saved["registration_id"], registration.REGISTRATION_ID)
        self.assertEqual(saved["schema_version"], "large-qubit-section7-registration-v2")
        self.assertEqual(saved["section6_v1_registration_lineage"]["manifest_sha256"], self.old_anchor)
        self.assertEqual(saved["section6_v1_gate_manifest"], self.gate6)
        self.assertEqual(saved["section6_acquisition_amendment"]["maximum_cumulative_ordinary_requests"], 14730)
        self.assertEqual(original, self.old_manifest.read_bytes())
        report = registration.verify_registration(self.root)
        self.assertFalse(report["section_6_complete"])
        self.assertIsNone(report["section_6_cumulative_ordinary_requests"])

    def test_v2_gate_is_distinct_from_preserved_v1_no_go(self):
        gate = {"saved_certificate_and_payload_verification": {
            "all_passed": True, "section_6_complete": True, "feasibility_gate": "go",
            "release_sha256": "a" * 64, "production_ordinary_requests": 7000}}
        with patch.object(registration, "section6_v2_manifest_binding", return_value=gate):
            registration.register(self.root)
            report = registration.verify_registration(self.root)
        self.assertTrue(report["section_6_complete"])
        self.assertEqual(report["section_6_cumulative_ordinary_requests"], 13538)
        self.assertEqual(report["section6_v1_registration_lineage"]["section_6_feasibility_gate"], "no_go")
        self.assertFalse(report["execution_ready"])

    def test_v2_verifier_is_explicitly_read_only(self):
        gate = self.write(registration.SECTION6_V2_MANIFEST_PATH, "{}")
        self.write(registration.SECTION6_V2_MANIFEST_PATH.removesuffix(".json") + ".sha256", sha256_file(gate))
        calls = []
        module = SimpleNamespace(verify_section6=lambda root, **kwargs: calls.append(kwargs) or {"all_passed": True})
        with patch.dict(sys.modules, {"large_qubit.section6_v2": module}):
            report = registration.section6_v2_manifest_binding(self.root)
        self.assertEqual(calls, [{"section6_root": "large_qubit/validation/section6_v2", "numerical_integrity": False}])
        self.assertFalse(report["scientific_work_rerun_by_registration"])

    def test_v2_bad_sidecar_prevents_snapshot(self):
        self.write(registration.SECTION6_V2_MANIFEST_PATH, "{}")
        self.write(registration.SECTION6_V2_MANIFEST_PATH.removesuffix(".json") + ".sha256", "0" * 64)
        with self.assertRaisesRegex(ValueError, "v2 gate manifest sidecar mismatch"):
            registration.register(self.root)

    def test_failed_v2_payload_verification_is_refused(self):
        gate = self.write(registration.SECTION6_V2_MANIFEST_PATH, "{}")
        self.write(registration.SECTION6_V2_MANIFEST_PATH.removesuffix(".json") + ".sha256", sha256_file(gate))
        module = SimpleNamespace(verify_section6=lambda *args, **kwargs: {"all_passed": False})
        with patch.dict(sys.modules, {"large_qubit.section6_v2": module}):
            with self.assertRaisesRegex(ValueError, "payload authentication failed"):
                registration.section6_v2_manifest_binding(self.root)

    def test_v2_preparation_cannot_change_after_source_registration(self):
        self.write(registration.SECTION6_V2_PREPARATION_PATH, "{}")
        registration.register(self.root)
        self.write(registration.SECTION6_V2_PREPARATION_PATH, '{"different":true}')
        with self.assertRaisesRegex(ValueError, "changed"):
            registration.verify_registration(self.root)

    def test_mutable_worker_evidence_is_not_in_source_binding_inventory(self):
        before = registration.current_bindings(self.root)
        self.write("large_qubit/evidence/sections_1_6_v2/local_execution.json", "{}")
        self.write("large_qubit/evidence/section6_v2_runs/section6_v2/attempt_0001/worker.log", "ongoing")
        self.assertEqual(before, registration.current_bindings(self.root))

    def test_authorization_byte_changes_cannot_rewrite_user_scope(self):
        self.write(registration.ACQUISITION_AMENDMENT_PATH, "{}")
        with self.assertRaisesRegex(ValueError, "immutable anchor"):
            registration.verify_acquisition_amendment(self.root)

    def test_authorized_budget_does_not_waive_fixed_quality_thresholds(self):
        changed = copy.deepcopy(self.amendment)
        changed["unchanged_quality_policy"]["minimum_weight_ESS_fraction"] = .04
        self.write(registration.ACQUISITION_AMENDMENT_PATH, json.dumps(changed))
        with patch.object(registration, "ACQUISITION_AMENDMENT_SHA256", sha256_file(self.amendment_path)):
            with self.assertRaisesRegex(ValueError, "frozen scientific constraints"):
                registration.verify_acquisition_amendment(self.root)


if __name__ == "__main__":
    unittest.main()
