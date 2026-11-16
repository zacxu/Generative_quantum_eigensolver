"""Synthetic migration tests: no model packages, weights, or quantum calls."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from large_qubit import reuse


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


class ReuseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.make_fixture()
        for name, value in (("ORIGINAL_PROTOCOL_SHA256", self.protocol_hash),
                            ("EXPECTED_BINDING_COUNT", 3),
                            ("EXPECTED_CONFIGURATION_COUNT", 3)):
            context = patch.object(reuse, name, value)
            context.start()
            self.addCleanup(context.stop)

    def write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, (dict, list)):
            path.write_text(json.dumps(value), encoding="utf-8")
        else:
            path.write_text(value, encoding="utf-8")
        return sha256(path.read_bytes()).hexdigest()

    def make_fixture(self):
        for path in reuse.ARTIFACT_ROLES:
            self.write(path, {} if path.endswith(".json") else "historical fixture\n")
        models = {key: {"repo_id": "fixture/" + key, "revision": "a" * 40}
                  for key in reuse.MODEL_KEYS}
        lock_hash = self.write("phase6/model_lock.json", {"models": models})
        for key, model in models.items():
            self.write(f"phase6/cache/models/weight_manifests/{key}_{model['revision']}.json", {
                "schema": "phase6-weight-download-v1", "model_key": key,
                **model, "model_lock_sha256": lock_hash})
        self.write(reuse.PILOT_REVIEW, {
            "schema": "phase6-pilot-review-v1", "step12_complete": True,
            "step13_complete": True, "step14_review_complete": True,
            "phase6_empirical_correctness_gate": "passed"})
        self.write(reuse.PILOT_AUDIT, {"audit_result": "passed", "hash_references_checked": 1695})
        self.write(reuse.SAVED_VERIFICATION, {
            "all_artifact_authentication_passed": True,
            "saved_selection_matches_recomputed": True,
            "saved_convergence_matches_recomputed": True,
            "all_convergence_ready": False, "checkpoint_jobs": 348, "rows_authenticated": 696})
        selected, conditions = {}, {}
        for method in reuse.SELECTED_METHODS:
            key = "gpt2_xl" if method.startswith("gpt2") else "qwen3_4b"
            config = {"pipeline_seed": 11, "model_key": key,
                      "objective": "uniform" if "uniform" in method else "weighted"}
            identity = {"configuration": config, "other_binding": "fixture"}
            identity_hash = digest(identity)
            base = f"phase7/runs/dev_v1/{method}_c01"
            manifest_hash = self.write(base + "/pass_001/checkpoint_manifest.json", {
                "identity_sha256": identity_hash, "complete_pass": 1,
                "files_sha256": {"adapter/adapter_model.safetensors": "b" * 64}})
            binding_hash = self.write(base + "/run_binding.json", {
                "identity": identity, "identity_sha256": identity_hash})
            # Deliberately missing binary payload: a metadata migration must not
            # pretend it authenticated or initialized from this old adapter.
            selected[method] = {"method_id": method, "candidate_id": "c01", "checkpoint_pass": 1,
                "checkpoint": {"configuration": config, "complete_pass": 1,
                    "path": base + "/pass_001", "manifest_sha256": manifest_hash,
                    "run_binding_sha256": binding_hash, "identity_sha256": identity_hash}}
            path = f"phase7/configs/{method}_c01.json"
            self.write(path, config)
            conditions[method] = {"configurations": [{"config_path": path,
                "configuration": config, "configuration_sha256": digest(config)}]}
        self.write(reuse.ASSESSMENT, {
            "schema_version": "phase7-checkpoint-selection-v1", "selection_is_final_freeze": False,
            "pretrained_allowance_passes": 5, "scratch_allowance_passes": 20,
            "convergence": {"all_convergence_ready": False}, "selected": selected})
        bindings = {path: sha256((self.root / path).read_bytes()).hexdigest()
                    for path in ("phase2/benchmark_manifest.json", "phase6/model_lock.json",
                                 "phase5/release_manifest.json")}
        protocol = {"schema_version": "phase7-development-protocol-v1", "final_protocol_frozen": False,
                    "bindings_sha256": bindings, "training_conditions": conditions}
        self.protocol_hash = self.write(reuse.ORIGINAL_PROTOCOL, protocol)
        self.write("phase7/development_protocol.sha256", self.protocol_hash + "  development_protocol.json\n")

    def test_reuse_is_metadata_only_and_keeps_nonconvergence(self):
        before_imports = set(__import__("sys").modules)
        record = reuse.build_reuse_inventory(self.root)
        report = reuse.verify_reuse_inventory(record, self.root)
        self.assertTrue(report["all_passed"])
        self.assertFalse(report["raw_results_reauthenticated"])
        self.assertFalse(record["saved_audit_claims"]["phase7"]["all_convergence_ready"])
        self.assertTrue(record["migration_policy"]["fresh_co2_labels_required"])
        self.assertEqual(len(record["migration_policy"]["fresh_co2_adapter_fits_required"]), 3)
        self.assertFalse(record["audit_work"]["weight_payloads_hashed"])
        for checkpoint in record["historical_selected_configurations"]:
            self.assertFalse(checkpoint["usable_as_co2_initialization"])
        imported = set(__import__("sys").modules) - before_imports
        self.assertFalse(any(name == "torch" or name.startswith("torch.") for name in imported))

    def test_inventory_deterministic_and_portable(self):
        self.assertEqual(reuse.build_reuse_inventory(self.root), reuse.build_reuse_inventory(self.root))
        self.assertNotIn(str(self.root), json.dumps(reuse.build_reuse_inventory(self.root)))

    def test_changed_registered_source_refused(self):
        self.write("phase2/benchmark_manifest.json", {"changed": True})
        with self.assertRaisesRegex(reuse.ReuseError, "hash mismatch"):
            reuse.build_reuse_inventory(self.root)

    def test_new_source_inside_phase7_refused(self):
        self.write("phase7/extra/new_source.py", "# accidental source mutation\n")
        with self.assertRaisesRegex(reuse.ReuseError, "source namespace changed"):
            reuse.build_reuse_inventory(self.root)

    def test_protocol_and_sidecar_rewrite_cannot_reanchor(self):
        new_hash = self.write(reuse.ORIGINAL_PROTOCOL, {"replaced": True})
        self.write("phase7/development_protocol.sha256", new_hash)
        with self.assertRaisesRegex(reuse.ReuseError, "hash mismatch"):
            reuse.build_reuse_inventory(self.root)

    def test_missing_registered_configuration_refused(self):
        path = self.root / "phase7/configs/gpt2_xl_uniform_lora_c01.json"
        path.unlink()
        with self.assertRaisesRegex(reuse.ReuseError, "missing"):
            reuse.build_reuse_inventory(self.root)

    def test_configuration_content_change_refused(self):
        self.write("phase7/configs/gpt2_xl_uniform_lora_c01.json", {"pipeline_seed": 99})
        with self.assertRaisesRegex(reuse.ReuseError, "configuration content differs"):
            reuse.build_reuse_inventory(self.root)

    def test_changed_historical_report_refused_on_verify(self):
        record = reuse.build_reuse_inventory(self.root)
        self.write("phase5/data_study.md", "changed report")
        with self.assertRaisesRegex(reuse.ReuseError, "differs from current files"):
            reuse.verify_reuse_inventory(record, self.root)

    def test_role_laundering_or_omission_refused(self):
        record = reuse.build_reuse_inventory(self.root)
        for mutation in ("omit", "role", "co2_labels", "pretend_reaudit"):
            altered = deepcopy(record)
            if mutation == "omit":
                altered["historical_artifacts"].pop()
            elif mutation == "role":
                altered["historical_artifacts"][0]["role"] = "new_co2_test_result"
            elif mutation == "co2_labels":
                altered["migration_policy"]["fresh_co2_labels_required"] = False
            else:
                altered["audit_work"]["raw_trajectories_reauthenticated"] = True
            with self.subTest(mutation=mutation), self.assertRaises(reuse.ReuseError):
                reuse.verify_reuse_inventory(altered, self.root)

    def test_incomplete_or_missing_pilot_flags_refused(self):
        original = json.loads((self.root / reuse.PILOT_REVIEW).read_text())
        for field in ("step12_complete", "step13_complete", "step14_review_complete"):
            for value in (False, None, 1):
                altered = deepcopy(original)
                if value is None:
                    altered.pop(field)
                else:
                    altered[field] = value
                self.write(reuse.PILOT_REVIEW, altered)
                with self.subTest(field=field, value=value), self.assertRaisesRegex(
                        reuse.ReuseError, "Saved Phase 6 pilot reports"):
                    reuse.build_reuse_inventory(self.root)
        self.write(reuse.PILOT_REVIEW, original)

    def test_failed_or_missing_pilot_correctness_gate_refused(self):
        original = json.loads((self.root / reuse.PILOT_REVIEW).read_text())
        for value in ("failed", True, None):
            altered = deepcopy(original)
            if value is None:
                altered.pop("phase6_empirical_correctness_gate")
            else:
                altered["phase6_empirical_correctness_gate"] = value
            self.write(reuse.PILOT_REVIEW, altered)
            with self.subTest(value=value), self.assertRaisesRegex(
                    reuse.ReuseError, "Saved Phase 6 pilot reports"):
                reuse.build_reuse_inventory(self.root)
        self.write(reuse.PILOT_REVIEW, original)

    def test_failed_or_missing_pilot_integrity_audit_refused(self):
        original = json.loads((self.root / reuse.PILOT_AUDIT).read_text())
        for value in ("failed", True, None):
            altered = deepcopy(original)
            if value is None:
                altered.pop("audit_result")
            else:
                altered["audit_result"] = value
            self.write(reuse.PILOT_AUDIT, altered)
            with self.subTest(value=value), self.assertRaisesRegex(
                    reuse.ReuseError, "Saved Phase 6 pilot reports"):
                reuse.build_reuse_inventory(self.root)
        self.write(reuse.PILOT_AUDIT, original)

    def test_original_nonconvergence_cannot_be_promoted(self):
        path = self.root / reuse.ASSESSMENT
        assessment = json.loads(path.read_text())
        assessment["convergence"]["all_convergence_ready"] = True
        self.write(reuse.ASSESSMENT, assessment)
        with self.assertRaisesRegex(reuse.ReuseError, "nonconvergence"):
            reuse.build_reuse_inventory(self.root)

    def test_selected_checkpoint_must_match_condition_path(self):
        path = self.root / reuse.ASSESSMENT
        assessment = json.loads(path.read_text())
        assessment["selected"][reuse.SELECTED_METHODS[0]]["checkpoint"]["path"] = "../other"
        self.write(reuse.ASSESSMENT, assessment)
        with self.assertRaisesRegex(reuse.ReuseError, "path does not match"):
            reuse.build_reuse_inventory(self.root)

    def test_selected_checkpoint_manifest_hash_is_checked(self):
        self.write("phase7/runs/dev_v1/gpt2_xl_weighted_lora_c01/pass_001/checkpoint_manifest.json", {})
        with self.assertRaisesRegex(reuse.ReuseError, "hash mismatch"):
            reuse.build_reuse_inventory(self.root)

    def test_revision_manifest_swap_refused(self):
        path = "phase6/cache/models/weight_manifests/qwen3_4b_" + "a" * 40 + ".json"
        value = json.loads((self.root / path).read_text())
        value["model_key"] = "gpt2_xl"
        self.write(path, value)
        with self.assertRaisesRegex(reuse.ReuseError, "metadata differs"):
            reuse.build_reuse_inventory(self.root)

    def test_unsafe_paths_rejected_cross_platform(self):
        for path in ("../outside.json", "/tmp/outside.json", "C:/outside.json",
                     "a/../../outside.json", "a\\outside.json", "./a.json", "a//b.json", ""):
            with self.subTest(path=path), self.assertRaises(reuse.ReuseError):
                reuse._relative_path(self.root, path)

    def test_symlink_outside_workspace_refused(self):
        with tempfile.TemporaryDirectory() as other:
            outside = Path(other) / "outside.json"
            outside.write_text("{}")
            try:
                (self.root / "link.json").symlink_to(outside)
            except (OSError, NotImplementedError):
                self.skipTest("This platform does not permit symlink creation")
            with self.assertRaisesRegex(reuse.ReuseError, "escapes workspace"):
                reuse._bytes(self.root, "link.json")

    def test_duplicate_json_keys_and_nonfinite_rejected(self):
        for text in ('{"schema": "first", "schema": "second"}', '{"value": NaN}'):
            self.write("invalid.json", text)
            with self.subTest(text=text), self.assertRaises(reuse.ReuseError):
                reuse._json(self.root, "invalid.json")

    def test_accidental_large_payload_refused(self):
        self.write("large.bin", "x" * 100)
        with patch.object(reuse, "MAX_METADATA_BYTES", 10):
            with self.assertRaisesRegex(reuse.ReuseError, "refusing large payload"):
                reuse._bytes(self.root, "large.bin")


if __name__ == "__main__":
    unittest.main()
