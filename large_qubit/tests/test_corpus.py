"""Synthetic release/weight/partition checks; no physics, models or training."""

import ast
from copy import deepcopy
from dataclasses import dataclass
import json
import math
from numbers import Real
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from large_qubit import corpus
from large_qubit.acquisition import quality_report
from large_qubit.baselines import ROW_FIELDS, retrieval_candidates
from large_qubit.common import canonical_bytes, digest, read_json, sha256_file
from large_qubit.tests._section5_fixtures import catalogue_fixture


WORKSPACE = Path(__file__).resolve().parents[2]


def eligible_rows(counts=(1026, 1027, 1028)):
    """Reserve all diagnostics first, then issue round-robin fitting requests."""
    catalogue = catalogue_fixture()
    rng, seen = random.Random(22), {alias: set() for alias in corpus.TRAINING}
    result, ordinal = [], 4
    def row(alias, role, index):
        nonlocal ordinal
        while True:
            ops = tuple(rng.randrange(513) for _ in range(32))
            if ops not in seen[alias]:
                seen[alias].add(ops)
                break
        value = {"row_id": f"{alias}_{ordinal:05d}", "geometry_id": alias, "role": role,
                 "ops": list(ops), "energy_hartree": -10. - .01 * (index % 10),
                 "catalogue_sha256": catalogue["sha256"], "request_ordinal": ordinal}
        ordinal += 1
        return value
    for index in range(128):
        for alias in corpus.TRAINING:
            result.append(row(alias, "diagnostic", index))
    for index in range(max(counts)):
        for alias, count in zip(corpus.TRAINING, counts):
            if index < count:
                result.append(row(alias, "fitting", index))
    return result


def old_weight_function():
    """Compile the preserved arithmetic alone, without importing Torch."""
    tree = ast.parse((WORKSPACE / "phase6/objective.py").read_text(encoding="utf-8"))
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                and node.name in ("GeometryWeights", "geometry_weights")]
    namespace = {"np": np, "math": math, "Real": Real, "dataclass": dataclass,
        "deepcopy": deepcopy, "IQR_FLOOR_HA": 1e-6, "ALLOWED_GEOMETRIES": {"LIH_T01"},
        "_require": lambda condition, message: None if condition else (_ for _ in ()).throw(ValueError(message))}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "preserved-objective-math", "exec"), namespace)
    return namespace["geometry_weights"]


class PrefixAndWeightsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalogue = catalogue_fixture()
        cls.rows = eligible_rows()

    def test_prefix_is_balanced_and_independent_of_energy_or_parent_acceptance(self):
        result = corpus.select_balanced_rows(self.rows[::-1], catalogue=self.catalogue, stop_reason="schedule")
        self.assertEqual(result["rows_per_geometry"], 1024)
        for alias, raw_count in zip(corpus.TRAINING, (1026, 1027, 1028)):
            original = [row for row in self.rows if row["geometry_id"] == alias and row["role"] == "fitting"]
            self.assertEqual(result["fitting_by_geometry"][alias], original[:1024])
            self.assertEqual(result["raw_fitting_by_geometry"][alias], original)
            self.assertEqual(len(result["raw_fitting_by_geometry"][alias]), raw_count)
        with self.assertRaises(ValueError):
            corpus.select_balanced_rows(self.rows, catalogue=self.catalogue, stop_reason="target")

    def test_full_prefix_is_preferred_and_quality_cannot_select_fallback(self):
        rows = eligible_rows((2050, 2049, 2048))
        result = corpus.select_balanced_rows(rows, catalogue=self.catalogue, stop_reason="budget")
        self.assertEqual(result["rows_per_geometry"], 2048)
        self.assertTrue(all(len(group) == 2048 for group in result["fitting_by_geometry"].values()))
        self.assertEqual([len(group) for group in result["raw_fitting_by_geometry"].values()], [2050, 2049, 2048])
        with self.assertRaises(ValueError):
            corpus.select_balanced_rows(eligible_rows((1024, 1024, 1023)), catalogue=self.catalogue, stop_reason="schedule")

    def test_diagnostic_overlap_timing_and_heldout_rows_fail(self):
        for mutation in ("overlap", "late_diagnostic", "heldout", "duplicate_ordinal", "bad_label"):
            rows = deepcopy(self.rows)
            if mutation == "overlap":
                rows[384]["ops"] = list(rows[0]["ops"])
            elif mutation == "late_diagnostic":
                rows[0]["request_ordinal"], rows[-1]["request_ordinal"] = rows[-1]["request_ordinal"], rows[0]["request_ordinal"]
            elif mutation == "heldout":
                rows[0]["geometry_id"] = "CO2_V01"
            elif mutation == "duplicate_ordinal":
                rows[-1]["request_ordinal"] = rows[-2]["request_ordinal"]
            else:
                rows[-1]["energy_hartree"] = True
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                corpus.select_balanced_rows(rows, catalogue=self.catalogue, stop_reason="schedule")

    def test_weights_match_preserved_phase6_arithmetic_without_Torch(self):
        rows = [deepcopy(row) for row in self.rows if row["role"] == "fitting" and row["geometry_id"] == "CO2_T01"][:128]
        old = old_weight_function()
        historical = [{"row_id": row["row_id"], "alias": "LIH_T01", "pipeline_seed": 22,
                       "dataset_partition": "fitting", "energy_ha": row["energy_hartree"]} for row in rows]
        for mode, scale in (("weighted", .25), ("weighted", .10), ("uniform", None)):
            result = corpus.geometry_weights(rows, mode=mode, lambda_scale=scale)
            expected = old(historical, mode=mode, lambda_scale=scale)
            np.testing.assert_array_equal(result.weights, expected.weights)
            np.testing.assert_array_equal(result.multipliers, expected.multipliers)
            for field in ("fitting_iqr_ha", "dispersion_ha", "tau_ha", "effective_sample_size", "maximum_weight"):
                self.assertEqual(result.report[field], expected.report[field])
            result.report["rows_filtered"] = 100
            self.assertEqual(result.report["rows_filtered"], 0)

    def test_weight_guards_floor_underflow_and_shift_invariance(self):
        rows = deepcopy([row for row in self.rows if row["role"] == "fitting" and row["geometry_id"] == "CO2_T01"][:8])
        for row in rows:
            row["energy_hartree"] = 7.
        equal = corpus.geometry_weights(rows, mode="weighted", lambda_scale=.25)
        self.assertEqual(equal.report["dispersion_ha"], 1e-6)
        self.assertEqual(equal.multipliers, (1.,) * 8)
        rows[0]["energy_hartree"] = -1000.
        concentrated = corpus.geometry_weights(rows, mode="weighted", lambda_scale=.10)
        self.assertGreater(concentrated.report["zero_weights_from_underflow"], 0)
        shifted = deepcopy(rows)
        for row in shifted:
            row["energy_hartree"] += 100.
        np.testing.assert_array_equal(concentrated.weights, corpus.geometry_weights(shifted, mode="weighted", lambda_scale=.10).weights)
        for key, value in (("role", "diagnostic"), ("geometry_id", "CO2_I01"),
                           ("row_id", rows[0]["row_id"]), ("energy_hartree", float("nan"))):
            bad = deepcopy(rows); bad[1][key] = value
            with self.assertRaises((ValueError, PermissionError)):
                corpus.geometry_weights(bad, mode="weighted", lambda_scale=.25)
        for scale in (None, True, 0., float("inf")):
            with self.assertRaises(ValueError):
                corpus.geometry_weights(rows, mode="weighted", lambda_scale=scale)
        with self.assertRaises(ValueError):
            corpus.geometry_weights(rows, mode="uniform", lambda_scale=.25)


class ReleaseFixture:
    def __init__(self, root):
        self.root = root
        self.catalogue = catalogue_fixture()
        self.rows = eligible_rows()
        selection = corpus.select_balanced_rows(self.rows, catalogue=self.catalogue, stop_reason="schedule")
        self.quality = quality_report(selection["fitting_by_geometry"], selection["diagnostic_by_geometry"],
                                      {alias: -9.9 for alias in corpus.TRAINING}, self.catalogue)
        if not self.quality["all_passed"]:
            raise AssertionError("Synthetic fixture must pass the declared quality thresholds")
        for source in (*corpus.SOURCE_PATHS, *corpus.SECTION5_SOURCES, "phase6/grammar.py"):
            self.write(source, "# synthetic bound source; never imported\n")
        parent_sources = {source: sha256_file(root / source) for source in corpus.SECTION5_SOURCES}
        self.write("large_qubit/catalogues/co2_screened_v1.json", self.catalogue)
        self.write("large_qubit/physical_definition.json", {"synthetic": "physical metadata"})
        self.write("large_qubit/physics/section3_v3/manifest.json", {
            "all_passed": True, "definition_sha256": digest({"synthetic": "physical metadata"}),
            "geometries": {alias: {"hamiltonian_fingerprint": value}
                           for alias, value in self.catalogue["training_hamiltonian_bindings"].items()}})
        self.write("large_qubit/validation/section5_v1/manifest.json", {
            "section_5_complete": True, "feasibility_gate": "go", "decision": {"selected_length": 32},
            "catalogue_sha256": self.catalogue["sha256"], "source_bindings": parent_sources})
        self.write("large_qubit/evidence/sections_1_5_v1/preparation.json", {
            "catalogue_sha256": self.catalogue["sha256"], "candidate_energy_requests": 0,
            "model_training_executed": False, "source_bindings": parent_sources,
            "reused_grammar_sha256": sha256_file(root / "phase6/grammar.py")})
        self.write("large_qubit/section5_policy.json", {"synthetic": "preserved parent policy"})
        self.write(corpus.POLICY_PATH, {"synthetic": "predeclared Section6 policy"})
        self.eligible_path = corpus.PRIVATE_ROOT + "/section6_fixture/eligible_rows.jsonl"
        self.write_rows(self.eligible_path, self.rows)
        self.raw = {self.eligible_path: sha256_file(root / self.eligible_path)}
        self.cost = {"logical_energy_requests": len(self.rows) + 3, "wall_seconds": 100.}

    def write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((canonical_bytes(value) + b"\n") if not isinstance(value, str) else value.encode())

    def write_rows(self, relative, values):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"".join(canonical_bytes(row) + b"\n" for row in values))

    def build(self, **overrides):
        args = {"eligible_rows_path": self.eligible_path, "catalogue": self.catalogue,
            "quality_report": self.quality, "stop_reason": "schedule", "acquisition_cost": self.cost,
            "raw_artifacts": self.raw, "verification_cost": {"hamiltonian_actions": 12, "wall_seconds": 2.}}
        args.update(overrides)
        return corpus.build_release(self.root, **args)


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.fixture = ReleaseFixture(self.root)

    def test_release_roundtrip_prefix_full_cost_and_identical_finalization_is_idempotent(self):
        saved = self.fixture.build()
        report = corpus.verify_release(self.root, expected_release_sha256=saved["manifest_sha256"])
        self.assertTrue(report["all_passed"])
        self.assertEqual((report["fitting_rows"], report["diagnostic_rows"]), (3072, 384))
        self.assertEqual(report["hamiltonian_actions"], 0)
        view = corpus.load_release(self.root, consumer_id="gpt2_xl_weighted_lora", expected_release_sha256=saved["manifest_sha256"])
        self.assertEqual(view.source_acquisition_cost, {"release_sha256": saved["manifest_sha256"],
            "logical_energy_requests": len(self.fixture.rows) + 3, "wall_seconds": 102.})
        self.assertEqual(view.separate_verification_cost, {"hamiltonian_actions": 12, "wall_seconds": 2.})
        self.assertEqual(len(view.fitting_rows("CO2_T01")), 1024)
        self.assertEqual(view.weights("CO2_T01").report["lambda_scale"], .25)
        with self.assertRaises(TypeError):
            view.fitting_rows("CO2_T01")[0]["role"] = "diagnostic"
        files = [self.root / corpus.RELEASE_ROOT / filename for filename in ("manifest.json", "manifest.sha256")]
        files += [self.root / corpus.CORPUS_ROOT / filename for filename in ("fitting.jsonl", "diagnostic.jsonl")]
        before = {path: (sha256_file(path), path.stat().st_mtime_ns) for path in files}
        self.assertEqual(self.fixture.build(), saved)
        self.assertEqual({path: (sha256_file(path), path.stat().st_mtime_ns) for path in files}, before)
        self.assertEqual(list((self.root / "large_qubit/releases").glob("*_incomplete_*")), [])

    def test_partial_payload_is_preserved_before_rebuilding_same_inputs(self):
        partial = self.root / corpus.CORPUS_ROOT
        partial.mkdir(parents=True)
        original = b'{"interrupted":"fitting-write'
        (partial / "fitting.jsonl").write_bytes(original)
        saved = self.fixture.build()
        preserved = list(partial.parent.glob("co2_corpus_v1_incomplete_*"))
        self.assertEqual(len(preserved), 1)
        self.assertTrue(preserved[0].resolve().is_relative_to((self.root / "large_qubit/corpora").resolve()))
        self.assertEqual((preserved[0] / "fitting.jsonl").read_bytes(), original)
        self.assertTrue(corpus.verify_release(self.root, expected_release_sha256=saved["manifest_sha256"])["all_passed"])

    def test_manifest_without_sidecar_preserves_both_directories_and_recovers(self):
        saved = self.fixture.build()
        release = self.root / corpus.RELEASE_ROOT
        payload = self.root / corpus.CORPUS_ROOT
        original_manifest = (release / "manifest.json").read_bytes()
        original_payloads = {name: (payload / name).read_bytes() for name in ("fitting.jsonl", "diagnostic.jsonl")}
        sidecar = release / "manifest.sha256"
        self.assertTrue(sidecar.resolve().is_relative_to(self.root.resolve()))
        sidecar.unlink()  # Simulate interruption immediately before sidecar creation.
        recovered = self.fixture.build()
        self.assertEqual(recovered["manifest_sha256"], saved["manifest_sha256"])
        old_release = list(release.parent.glob("co2_corpus_v1_incomplete_*"))
        old_payload = list(payload.parent.glob("co2_corpus_v1_incomplete_*"))
        self.assertEqual((len(old_release), len(old_payload)), (1, 1))
        self.assertEqual(old_release[0].name, old_payload[0].name)
        self.assertEqual((old_release[0] / "manifest.json").read_bytes(), original_manifest)
        self.assertFalse((old_release[0] / "manifest.sha256").exists())
        for name, data in original_payloads.items():
            self.assertEqual((old_payload[0] / name).read_bytes(), data)
        self.assertTrue(corpus.verify_release(self.root, expected_release_sha256=recovered["manifest_sha256"])["all_passed"])

    def test_existing_release_refuses_changed_frozen_cost_quality_and_raw_bindings(self):
        self.fixture.build()
        release = self.root / corpus.RELEASE_ROOT
        payload = self.root / corpus.CORPUS_ROOT
        paths = [release / "manifest.json", release / "manifest.sha256",
                 payload / "fitting.jsonl", payload / "diagnostic.jsonl"]
        original = {path: (sha256_file(path), path.stat().st_mtime_ns) for path in paths}
        different_quality = deepcopy(self.fixture.quality)
        different_quality["groups"]["CO2_T01"]["best_HF_improvement_ha"] += .001
        extra = corpus.PRIVATE_ROOT + "/section6_fixture/extra_frozen_record.json"
        self.fixture.write(extra, {"synthetic": "a different frozen input set"})
        changes = ({"acquisition_cost": {**self.fixture.cost, "wall_seconds": 101.}},
                   {"verification_cost": {"hamiltonian_actions": 12, "wall_seconds": 3.}},
                   {"quality_report": different_quality},
                   {"raw_artifacts": {**self.fixture.raw, extra: sha256_file(self.root / extra)}})
        for changed in changes:
            with self.subTest(changed=list(changed)), self.assertRaisesRegex(ValueError, "different frozen finalization inputs"):
                self.fixture.build(**changed)
            self.assertEqual({path: (sha256_file(path), path.stat().st_mtime_ns) for path in paths}, original)
        self.assertEqual(list(release.parent.glob("*_incomplete_*")), [])
        self.assertEqual(list(payload.parent.glob("*_incomplete_*")), [])

    def test_views_never_open_other_role_or_raw_label_files(self):
        saved = self.fixture.build()
        seen = []
        original = corpus._read_rows
        def tracked(path):
            seen.append(Path(path).name)
            return original(path)
        with patch("large_qubit.corpus._read_rows", side_effect=tracked):
            fit = corpus.load_release(self.root, consumer_id="qwen3_4b_weighted_lora", expected_release_sha256=saved["manifest_sha256"])
            self.assertEqual(seen, [])
            self.assertEqual(fit.weights("CO2_T02").report["lambda_scale"], .10)
            self.assertEqual(seen, ["fitting.jsonl"])
            with self.assertRaises(PermissionError):
                fit.diagnostic_rows("CO2_T02")
            diagnostic = corpus.load_release(self.root, consumer_id="qwen3_4b_weighted_lora", purpose="diagnostics",
                                             expected_release_sha256=saved["manifest_sha256"])
            self.assertEqual(len(diagnostic.diagnostic_rows("CO2_T02")), 128)
            self.assertEqual(seen[-1], "diagnostic.jsonl")
            for method in (diagnostic.fitting_rows, diagnostic.weights):
                with self.assertRaises(PermissionError):
                    method("CO2_T02")
            retrieval = corpus.load_release(self.root, consumer_id="training_circuit_retrieval",
                                            expected_release_sha256=saved["manifest_sha256"])
            rows = retrieval.retrieval_rows()
            self.assertTrue(all(set(row) == ROW_FIELDS and row["role"] == "fitting" for row in rows))
            result = retrieval_candidates("CO2_I01", rows, catalogue=self.fixture.catalogue, length=32,
                                          acquisition_cost=retrieval.source_acquisition_cost)
            self.assertEqual(result["source_geometry_id"], "CO2_T01")
            self.assertEqual(result["post_HF_candidate_count"], 1023)
            with self.assertRaises(PermissionError):
                retrieval.weights("CO2_T01")

    def test_uniform_control_and_unknown_consumers_cannot_change_roles(self):
        saved = self.fixture.build()
        view = corpus.load_release(self.root, consumer_id="gpt2_xl_uniform_lora", expected_release_sha256=saved["manifest_sha256"])
        self.assertEqual(view.weights("CO2_T03").multipliers, (1.,) * 1024)
        for consumer, purpose in (("gpt2_xl_frozen", None), ("training_circuit_retrieval", "fitting"),
                                  ("training_circuit_retrieval", "diagnostics"), ("gpt2_xl_uniform_lora", "retrieval")):
            with self.assertRaises(PermissionError):
                corpus.load_release(self.root, consumer_id=consumer, purpose=purpose,
                                    expected_release_sha256=saved["manifest_sha256"])

    def test_missing_or_changed_pin_code_payload_and_raw_are_refused(self):
        saved = self.fixture.build()
        with self.assertRaises(ValueError):
            corpus.load_release(self.root, consumer_id="gpt2_xl_uniform_lora", expected_release_sha256="0" * 64)
        path = self.root / self.fixture.eligible_path
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaises(ValueError):
            corpus.verify_release(self.root, expected_release_sha256=saved["manifest_sha256"])
        self.fixture.write_rows(self.fixture.eligible_path, self.fixture.rows)
        source = self.root / "large_qubit/acquisition.py"
        source.write_text("# changed\n", encoding="ascii")
        with self.assertRaises(ValueError):
            corpus.load_release(self.root, consumer_id="gpt2_xl_uniform_lora", expected_release_sha256=saved["manifest_sha256"])

    def test_self_consistent_payload_rehash_cannot_change_raw_prefix_labels(self):
        saved = self.fixture.build()
        manifest_path = self.root / saved["manifest_path"]
        manifest = read_json(manifest_path)
        fitting_path = self.root / manifest["files"]["fitting"]["path"]
        rows = corpus._read_rows(fitting_path)
        rows[0]["energy_hartree"] -= .5
        self.fixture.write_rows(manifest["files"]["fitting"]["path"], rows)
        manifest["files"]["fitting"].update(bytes=fitting_path.stat().st_size, sha256=sha256_file(fitting_path))
        self.fixture.write(saved["manifest_path"], manifest)
        changed_pin = sha256_file(manifest_path)
        manifest_path.with_suffix(".sha256").write_text(changed_pin + "\n", encoding="ascii")
        with self.assertRaisesRegex(ValueError, "prefix"):
            corpus.verify_release(self.root, expected_release_sha256=changed_pin)

    def test_full_available_prefix_cannot_be_replaced_with_passing_small_prefix(self):
        rows = eligible_rows((2048, 2048, 2048))
        self.fixture.write_rows(self.fixture.eligible_path, rows)
        self.fixture.raw[self.fixture.eligible_path] = sha256_file(self.root / self.fixture.eligible_path)
        cost = {"logical_energy_requests": len(rows) + 3, "wall_seconds": 200.}
        # The old 1024-prefix quality report cannot authorize a1024 release when
        # all2048 rows already exist, even if that smaller quality report passed.
        with self.assertRaisesRegex(ValueError, "exact balanced prefix"):
            self.fixture.build(acquisition_cost=cost)
        self.assertFalse((self.root / corpus.RELEASE_ROOT).exists())

    def test_failed_quality_cannot_release_or_trigger_smaller_prefix(self):
        bad = deepcopy(self.fixture.quality)
        bad["all_passed"] = False
        with self.assertRaises(ValueError):
            self.fixture.build(quality_report=bad)
        self.assertFalse((self.root / corpus.RELEASE_ROOT).exists())
        bad = deepcopy(self.fixture.quality); bad["rows_per_geometry"] = 2048
        with self.assertRaises(ValueError):
            self.fixture.build(quality_report=bad)
        # Frozen per-geometry quota exhaustion can leave other quotas unused;
        # the executor authenticates that frontier rather than inventing queries.
        saved = self.fixture.build(stop_reason="budget")
        self.assertLess(saved["manifest"]["acquisition_cost"]["logical_energy_requests"], 8192)


if __name__ == "__main__":
    unittest.main()
