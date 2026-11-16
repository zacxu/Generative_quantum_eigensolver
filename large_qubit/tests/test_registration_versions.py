"""Migration checks: immutable original archive and independently bound active code."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from large_qubit.common import read_json, sha256_file
from large_qubit.protocol import build_study
from large_qubit.definitions import build_definition
from large_qubit import registration


class RegistrationVersionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in registration.BOUND_DOCUMENTS:
            self.write(name, "original " + name)
        for name in registration.SECTION3_DOCUMENTS:
            self.write(name, "{}" if name.endswith(".json") else "Section 3 fixture")
        for name in registration.SECTION4_DOCUMENTS:
            self.write(name, "{}" if name.endswith(".json") else "Section 4 fixture")
        for name in registration.SECTION5_DOCUMENTS:
            self.write(name, "{}" if name.endswith(".json") else "Section 5 fixture")
        self.write("large_qubit/study.json", json.dumps(build_study()))
        self.write("large_qubit/protocol.py", "# original source\n")
        self.write(registration.PHYSICAL_DEFINITION_PATH, json.dumps(build_definition()))
        self.write("historical_report.json", '{"converged": false}')
        self.history_hash = sha256_file(self.root / "historical_report.json")
        self.assets = {"metadata_authenticated": True}
        bindings = {name: sha256_file(self.root / name)
                    for name in (*registration.BOUND_DOCUMENTS, "large_qubit/protocol.py")}
        original = {
            "registration_id": "sections_1_2_v1",
            "schema_version": "large-qubit-foundation-registration-v1",
            "execution_ready": False, "bindings_sha256": bindings,
            "reuse_inventory": {"historical_report_sha256": self.history_hash},
            "base_asset_metadata_verification": {"gpt2_xl": self.assets},
            "reusable_components": {},
        }
        self.original_path = self.root / registration.ORIGINAL_REGISTRATION / "manifest.json"
        self.write(self.original_path.relative_to(self.root).as_posix(), json.dumps(original))
        self.original_digest = sha256_file(self.original_path)
        self.write(registration.ORIGINAL_REGISTRATION + "/manifest.sha256", self.original_digest + "\n")
        for name in bindings:
            self.write(registration.ORIGINAL_ARCHIVE + "/" + name, (self.root / name).read_bytes())
        for name, value in (
            ("ORIGINAL_MANIFEST_SHA256", self.original_digest),
            ("REUSABLE_COMPONENTS", {}),
        ):
            context = patch.object(registration, name, value)
            context.start()
            self.addCleanup(context.stop)
        context = patch.object(registration, "inspect_assets", return_value=self.assets)
        context.start()
        self.addCleanup(context.stop)
        context = patch.object(registration, "verify_section3_registration",
                               return_value={"all_passed": True, "fixture_section3_lineage": True})
        context.start()
        self.addCleanup(context.stop)
        context = patch.object(registration, "verify_section4_registration",
                               return_value={"all_passed": True, "fixture_section4_lineage": True})
        context.start()
        self.addCleanup(context.stop)
        context = patch.object(registration, "verify_section5_registration",
                               return_value={"all_passed": True, "fixture_section5_lineage": True})
        context.start()
        self.addCleanup(context.stop)
        for name, value in (
            ("verify_section6_registration", {"all_passed": True, "fixture_section6_v1_lineage": True}),
            ("verify_acquisition_amendment", {"all_passed": True, "fixture_v2_authorization": True}),
            ("verify_section6_v2_registration", {"all_passed": True, "fixture_section6_v2_lineage": True}),
            ("section7_implementation_binding", None),
        ):
            context = patch.object(registration, name, return_value=value)
            context.start()
            self.addCleanup(context.stop)
        context = patch.object(registration, "verify_grid_amendment",
                               return_value={"all_passed": True, "fixture_amendment": True})
        context.start()
        self.addCleanup(context.stop)
        context = patch.object(registration, "build_reuse_inventory",
                               return_value=original["reuse_inventory"])
        context.start()
        self.addCleanup(context.stop)

        def verify_history(document, root):
            if document != original["reuse_inventory"] or sha256_file(Path(root) / "historical_report.json") != self.history_hash:
                raise ValueError("Historical evidence changed")
            return {"all_passed": True, "raw_results_reauthenticated": False}
        context = patch.object(registration, "verify_reuse_inventory", side_effect=verify_history)
        context.start()
        self.addCleanup(context.stop)

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value if isinstance(value, bytes) else value.encode("utf-8"))
        return path

    def test_original_verification_uses_archive_after_live_source_changes(self):
        self.write("large_qubit/protocol.py", "# new Section 3 implementation\n")
        self.write("research_plan_revised.md", "Section 3 complete")
        report = registration.verify_registration(self.root, registration.ORIGINAL_REGISTRATION)
        self.assertTrue(report["all_passed"])
        self.assertFalse(report["active_source_verified_as_original"])
        self.assertEqual(report["source_verification_mode"],
                         "archived_original_sources_and_live_historical_evidence")

    def test_archive_tampering_is_rejected(self):
        self.write(registration.ORIGINAL_ARCHIVE + "/large_qubit/protocol.py", "# tampered")
        with self.assertRaisesRegex(ValueError, "archived binding changed"):
            registration.verify_original_registration(self.root)

    def test_missing_original_archive_is_rejected(self):
        (self.root / registration.ORIGINAL_ARCHIVE / "large_qubit/protocol.py").unlink()
        with self.assertRaisesRegex(ValueError, "archived binding is missing"):
            registration.verify_original_registration(self.root)

    def test_rewritten_original_manifest_and_sidecar_do_not_change_anchor(self):
        original = read_json(self.original_path)
        original["execution_ready"] = True
        self.original_path.write_text(json.dumps(original), encoding="utf-8")
        self.write(registration.ORIGINAL_REGISTRATION + "/manifest.sha256",
                   sha256_file(self.original_path))
        with self.assertRaisesRegex(ValueError, "immutable anchor"):
            registration.verify_original_registration(self.root)

    def test_live_historical_report_still_must_match(self):
        self.write("historical_report.json", '{"converged": true}')
        with self.assertRaisesRegex(ValueError, "Historical evidence changed"):
            registration.verify_original_registration(self.root)

    def test_original_model_metadata_still_must_match(self):
        with patch.object(registration, "inspect_assets", return_value={"changed": True}):
            with self.assertRaisesRegex(ValueError, "model metadata changed"):
                registration.verify_original_registration(self.root)

    def test_active_bindings_exclude_archival_python_and_include_section3_inputs(self):
        self.write("large_qubit/physics.py", "# new physical definitions")
        self.write("large_qubit/archives/other/large_qubit/danger.py", "raise RuntimeError")
        bindings = registration.current_bindings(self.root)
        self.assertIn("large_qubit/physics.py", bindings)
        self.assertIn(registration.PHYSICAL_DEFINITION_PATH, bindings)
        self.assertFalse(any("/archives/" in path for path in bindings))

    def test_new_registration_binds_current_sources_and_original_lineage(self):
        original_bytes = self.original_path.read_bytes()
        self.write("large_qubit/protocol.py", "# new implementation")
        registration.register(self.root)
        saved = read_json(self.root / registration.DEFAULT_REGISTRATION / "manifest.json")
        self.assertEqual(saved["registration_id"], registration.REGISTRATION_ID)
        self.assertEqual(saved["original_registration_lineage"]["manifest_sha256"], self.original_digest)
        self.assertEqual(saved["bindings_sha256"]["large_qubit/protocol.py"],
                         sha256_file(self.root / "large_qubit/protocol.py"))
        self.assertTrue(registration.verify_registration(self.root)["all_passed"])
        self.assertEqual(original_bytes, self.original_path.read_bytes())

    def test_fabricated_numerical_physics_cannot_be_registered(self):
        definition = build_definition()
        definition["scientific_results_present"] = True
        self.write(registration.PHYSICAL_DEFINITION_PATH, json.dumps(definition))
        with self.assertRaisesRegex(ValueError, "Physical definition differs"):
            registration.register(self.root)
        self.assertFalse((self.root / registration.DEFAULT_REGISTRATION).exists())

    def test_new_sidecar_cannot_approve_changed_active_sources(self):
        registration.register(self.root)
        path = self.root / registration.DEFAULT_REGISTRATION / "manifest.json"
        self.write("large_qubit/protocol.py", "# unregistered amendment")
        self.write(registration.DEFAULT_REGISTRATION + "/manifest.sha256", sha256_file(path))
        with self.assertRaisesRegex(ValueError, "changed"):
            registration.verify_registration(self.root)

    def test_register_never_overwrites_either_generation(self):
        registration.register(self.root)
        for target in (registration.DEFAULT_REGISTRATION, registration.ORIGINAL_REGISTRATION):
            path = self.root / target / "manifest.json"
            before = path.read_bytes()
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, "already exists"):
                registration.register(self.root, target)
            self.assertEqual(before, path.read_bytes())

    def test_missing_lineage_refuses_new_registration_before_writing(self):
        (self.root / registration.ORIGINAL_ARCHIVE / "large_qubit/protocol.py").unlink()
        with self.assertRaisesRegex(ValueError, "archived binding is missing"):
            registration.register(self.root)
        self.assertFalse((self.root / registration.DEFAULT_REGISTRATION).exists())

    def test_later_physics_results_require_a_new_registration(self):
        registration.register(self.root)
        saved = read_json(self.root / registration.DEFAULT_REGISTRATION / "manifest.json")
        self.assertIsNone(saved["realized_physics_manifest"])
        physics = self.write(registration.PHYSICS_MANIFEST_PATH, '{"realization": "fixture"}')
        self.write(registration.PHYSICS_MANIFEST_PATH.removesuffix(".json") + ".sha256",
                   sha256_file(physics))
        with patch('large_qubit.physics.verify_physics', return_value={'all_passed': True}):
            with self.assertRaisesRegex(ValueError, "changed"):
                registration.verify_registration(self.root)

    def test_inconsistent_physics_manifest_is_refused_before_registration(self):
        self.write(registration.PHYSICS_MANIFEST_PATH, '{"realization": "fixture"}')
        self.write(registration.PHYSICS_MANIFEST_PATH.removesuffix(".json") + ".sha256", "0" * 64)
        with self.assertRaisesRegex(ValueError, "physics manifest sidecar mismatch"):
            registration.register(self.root)
        self.assertFalse((self.root / registration.DEFAULT_REGISTRATION).exists())

    def test_archive_namespace_redirect_is_rejected(self):
        real_resolve = Path.resolve
        archive = self.root / registration.ORIGINAL_ARCHIVE
        other = self.root / "historical_location"
        def resolve(path, *args, **kwargs):
            if path.is_relative_to(archive):
                return other / path.relative_to(archive)
            return real_resolve(path, *args, **kwargs)
        with patch.object(Path, "resolve", resolve), self.assertRaisesRegex(ValueError, "redirect"):
            registration.verify_original_registration(self.root)


if __name__ == "__main__":
    unittest.main()
