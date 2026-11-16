"""Tiny synthetic evaluator timings; never a CO2 benchmark or scientific result."""

import io
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from large_qubit.benchmark import (ANGLES, _charged_requests, _measure_requests,
                                   _run_concurrency, _thread_environment,
                                   excitation_templates, representative_circuits)
from large_qubit.common import digest
from large_qubit.sector import SectorBasis


class _TinyEvaluator:
    norb, nalpha, nbeta = 2, 1, 1

    def __init__(self, fail=False):
        self.action_seconds = 0.0
        self.fail = fail

    def statistics(self):
        return {"hamiltonian_action_seconds": self.action_seconds}

    def energy_sector(self, state, basis):
        if self.fail:
            raise ArithmeticError("synthetic evaluator failure")
        started = time.perf_counter()
        result = float(np.vdot(state, np.array([-1.0, -.8, -.6, -.4]) * state).real)
        self.action_seconds += time.perf_counter() - started
        return result


def _tiny_loader(workspace, alias="CO2_T02", *, threads=1):
    return _TinyEvaluator(), {"synthetic_test_fixture": True, "alias": alias, "threads": threads}


def _tiny_plans():
    plans = []
    for phase, samples in (("warmup", 1), ("measured", 2)):
        for family in ("whole_qubit", "fermionic"):
            for length in (1, 2):
                for sample in range(samples):
                    gates = [{"wires": [0, 2], "theta": .1 + .05 * sample}] * length
                    plans.append({"phase": phase, "family": family, "length": length,
                                  "sample": sample, "gates": gates,
                                  "circuit_sha256": digest({"family": family, "gates": gates})})
    return plans


class BenchmarkTest(unittest.TestCase):
    def test_full_catalogue_and_fixed_representatives_are_legal_and_repeatable(self):
        templates = excitation_templates()
        self.assertEqual(len(templates), 2684)
        self.assertEqual(len(set(templates)), 2684)
        self.assertEqual(sum(len(wires) == 2 for wires in templates), 88)
        plans = representative_circuits()
        self.assertEqual(plans, representative_circuits())
        self.assertEqual(len(plans), 12)
        self.assertEqual(sum(p["phase"] == "warmup" for p in plans), 4)
        self.assertEqual(sum(p["phase"] == "measured" for p in plans), 8)
        for plan in plans:
            self.assertEqual(len(plan["gates"]), plan["length"])
            for gate in plan["gates"]:
                self.assertIn(tuple(gate["wires"]), templates)
                self.assertIn(gate["theta"], ANGLES)
                half = len(gate["wires"]) // 2
                self.assertTrue(all(w < 22 for w in gate["wires"][:half]))
                self.assertTrue(all(22 <= w < 30 for w in gate["wires"][half:]))

    def test_native_thread_environment_is_restored(self):
        before = dict(os.environ)
        with _thread_environment(1):
            self.assertEqual(os.environ["OPENBLAS_NUM_THREADS"], "1")
            self.assertEqual(os.environ["OMP_NUM_THREADS"], "1")
        self.assertEqual(dict(os.environ), before)

    def test_tiny_request_kernel_times_components_and_keeps_energies_private(self):
        handle = io.StringIO()
        rows = _measure_requests(_TinyEvaluator(), SectorBasis(2, 1, 1), _tiny_plans(), handle)
        records = [json.loads(line) for line in handle.getvalue().splitlines()]
        started = [r for r in records if r["event"] == "request_started"]
        completed = [r for r in records if r["event"] == "request_completed"]
        self.assertEqual([r["request_ordinal"] for r in started], list(range(1, 13)))
        self.assertEqual(len(completed), 12)
        self.assertTrue(all("energy_hartree" in r for r in completed))
        self.assertTrue(any(r["duplicate_of"] is not None for r in started))
        for row in rows:
            self.assertNotIn("energy_hartree", row)
            self.assertGreaterEqual(row["state_preparation_seconds"], 0)
            self.assertGreaterEqual(row["hamiltonian_action_seconds"], 0)
            self.assertGreaterEqual(row["logging_seconds"], 0)

    def test_failed_energy_call_keeps_an_issued_request_intent(self):
        handle = io.StringIO()
        with self.assertRaisesRegex(ArithmeticError, "synthetic"):
            _measure_requests(_TinyEvaluator(fail=True), SectorBasis(2, 1, 1), _tiny_plans(), handle)
        records = [json.loads(line) for line in handle.getvalue().splitlines()]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["event"], "request_started")

    def test_two_spawned_workers_really_pass_warmup_barrier_and_count_all_queries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = _run_concurrency(root, root / "concurrent", _tiny_plans(), workers=2, threads=1,
                                   cache_budget_bytes=1024, memory_budget_bytes=4 * 1024 ** 3,
                                   loader_spec=("large_qubit.tests.test_benchmark", "_tiny_loader"))
            self.assertEqual(run["workers"], 2)
            self.assertEqual(run["logical_requests"], 24)
            self.assertEqual(run["warmup_requests"], 8)
            self.assertEqual(run["measured_requests"], 16)
            self.assertEqual(_charged_requests(root / "concurrent"), 24)
            self.assertGreater(run["measured_wall_seconds"], 0)
            self.assertEqual(len(run["worker_reports"]), 2)
            self.assertNotIn("energy_hartree", json.dumps(run))


if __name__ == "__main__":
    unittest.main()
