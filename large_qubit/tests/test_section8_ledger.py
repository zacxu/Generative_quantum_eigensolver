"""Synthetic scalar callbacks only: no model, GPU or molecular evaluations."""

import copy
from contextlib import closing
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from large_qubit.common import digest
from large_qubit.section8_ledger import OrderedScorer, ScalarCache, TrajectoryLedger, verify_ledger


CACHE_ID = {"partition": "validation", "geometry_id": "fake_geometry", "hamiltonian": "h-sha",
            "orbitals": "o-sha", "catalogue": "c-sha", "backend": "fake-only", "precision": "float64"}
IDENTITY = {"method": "synthetic", "freeze": "fixed", "cache_identity_sha256": digest(CACHE_ID)}


class LedgerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / "ledger.sqlite"
        self.ledger = TrajectoryLedger(self.path, IDENTITY)
        self.cache = ScalarCache(self.root / "cache.sqlite")

    def scorer(self, evaluate=lambda query: float(query["x"]), **kwargs):
        return OrderedScorer(self.ledger, self.cache, CACHE_ID, evaluate, **kwargs)

    def test_hf_first_exactly_once_and_invalid_transition_never_commits(self):
        before = self.ledger.snapshot()["head_sha256"]
        for call_id, kind in (("candidate", "candidate"), ("wrong", "hf"), ("hf", "candidate")):
            with self.assertRaisesRegex(ValueError, "first logical request"):
                self.ledger.reserve_request(call_id, {"x": 0}, kind=kind)
            self.assertEqual(self.ledger.snapshot()["head_sha256"], before)
        with self.scorer() as scorer:
            self.assertEqual(scorer.score("hf", {"x": 0}, kind="hf"), 0)
            with self.assertRaises(ValueError):
                scorer.score("again", {"x": 0}, kind="hf")
        self.assertEqual(self.ledger.snapshot()["logical_requests"], 1)

    def test_defaults_and_hf_inclusive_budget_charge_duplicates_and_cache_hits(self):
        self.ledger = TrajectoryLedger(self.root / "short.sqlite", IDENTITY, request_budget=3)
        calls = []
        with self.scorer(lambda query: calls.append(query["x"]) or query["x"]) as scorer:
            scorer.score("hf", {"x": 0}, kind="hf")
            scorer.score("candidate_1", {"x": -1})
            scorer.score("candidate_2", {"x": -1})
            with self.assertRaisesRegex(ValueError, "budget exhausted"):
                scorer.score("candidate_3", {"x": -2})
        state = self.ledger.snapshot()
        self.assertEqual(calls, [0, -1])
        self.assertEqual((state["logical_requests"], state["cache_hits"], state["duplicate_requests"]), (3, 1, 1))
        self.assertEqual(state["incumbent"]["ordinal"], 2)
        self.assertEqual(TrajectoryLedger(self.path, IDENTITY).snapshot()["budgets"], {"requests": 128, "attempts": 256})

    def test_attempts_reserved_before_generation_count_invalid_and_replay_without_refill(self):
        self.ledger = TrajectoryLedger(self.root / "attempt.sqlite", IDENTITY, attempt_budget=2)
        first = self.ledger.reserve_attempt("generation_1", {"rng_frontier": 0})
        self.assertEqual(first["status"], "pending")
        first = self.ledger.finish_attempt(1, "invalid", valid=False, failure_reason="parse")
        self.assertEqual(first["output_sha256"], digest("invalid"))
        head = self.ledger.snapshot()["head_sha256"]
        self.assertEqual(self.ledger.reserve_attempt("generation_1", {"rng_frontier": 0}), first)
        self.assertEqual(head, self.ledger.snapshot()["head_sha256"])
        with self.assertRaisesRegex(ValueError, "input differs"):
            self.ledger.reserve_attempt("generation_1", {"rng_frontier": 1})
        self.ledger.reserve_attempt("generation_2")
        self.ledger.recover_pending("generation interrupted; consumed")
        with self.assertRaisesRegex(ValueError, "budget exhausted"):
            self.ledger.reserve_attempt("generation_3")
        self.assertEqual(self.ledger.snapshot()["completion_attempts"], 2)

    def test_candidate_can_only_reference_authenticated_valid_attempt(self):
        self.ledger.reserve_request("hf", {"x": 0}, kind="hf")
        self.ledger.complete_request(1, 0)
        self.ledger.reserve_attempt("a1")
        with self.assertRaisesRegex(ValueError, "valid attempt"):
            self.ledger.reserve_request("c1", {"x": 1}, attempt_ordinal=1)
        self.ledger.finish_attempt(1, {"circuit": [1]}, valid=True)
        self.assertEqual(self.ledger.reserve_request("c1", {"x": 1}, attempt_ordinal=1)["ordinal"], 2)

    def test_completed_vqe_call_ids_replay_in_order_without_new_work_or_charges(self):
        with self.scorer() as scorer:
            scorer.score("hf", {"x": 0}, kind="hf")
            scorer.score("vqe:0:coordinate:0:probe:0", {"x": -1}, kind="vqe")
        before = self.ledger.snapshot()
        with self.scorer(lambda query: self.fail("completed calls must never rescore")) as scorer:
            self.assertEqual(scorer.score("hf", {"x": 0}, kind="hf"), 0)
            self.assertEqual(scorer.score("vqe:0:coordinate:0:probe:0", {"x": -1}, kind="vqe"), -1)
            with self.assertRaisesRegex(ValueError, "identity differs"):
                scorer.score("vqe:0:coordinate:0:probe:0", {"x": -2}, kind="vqe")
        self.assertEqual(before, self.ledger.snapshot())

    def test_duplicate_index_and_call_replay_use_exact_canonical_query_bytes(self):
        with self.scorer() as scorer:
            scorer.score("hf", {"x": 0}, kind="hf")
            scorer.score("integer", {"x": 1})
            scorer.score("float", {"x": 1.0})
            scorer.score("same-integer", {"x": 1})
            with self.assertRaisesRegex(ValueError, "identity differs"):
                scorer.score("integer", {"x": 1.0})
        self.assertEqual(self.ledger.snapshot()["duplicate_requests"], 1)
        self.assertEqual(self.ledger.snapshot()["requests"][3]["duplicate_of"], 2)

    def test_crash_intent_is_preserved_abandoned_charged_and_cannot_silently_retry(self):
        def crash(query):
            self.assertEqual(self.ledger.snapshot()["logical_requests"], 1)
            raise KeyboardInterrupt("synthetic crash after durable admission")
        with self.scorer(crash) as scorer:
            with self.assertRaises(KeyboardInterrupt):
                scorer.score("hf", {"x": 0}, kind="hf")
        self.assertEqual(self.ledger.snapshot()["pending_requests"], 1)
        with self.assertRaisesRegex(ValueError, "pending requests"):
            self.scorer()
        state = self.ledger.recover_pending("prior process disappeared; new documented run required")
        self.assertEqual((state["logical_requests"], state["abandoned_requests"], state["pending_requests"]), (1, 1, 0))
        with self.assertRaisesRegex(ValueError, "new run"):
            self.scorer()
        with self.assertRaisesRegex(ValueError, "new run"):
            self.ledger.reserve_request("retry", {"x": 0})

    def test_abrupt_process_exit_keeps_committed_physical_intent_and_original_chain(self):
        original_head = self.ledger.snapshot()["head_sha256"]
        script = ("import json,os,sys; from large_qubit.section8_ledger import TrajectoryLedger; "
                  "x=TrajectoryLedger(sys.argv[1],json.loads(sys.argv[2])); "
                  "x.reserve_request('hf',{'x':0},kind='hf'); os._exit(23)")
        result = subprocess.run([sys.executable, "-B", "-c", script, str(self.path), json.dumps(IDENTITY)],
                                cwd=Path(__file__).resolve().parents[2], capture_output=True, check=False)
        self.assertEqual(result.returncode, 23, result.stderr.decode(errors="replace"))
        restored = TrajectoryLedger(self.path, IDENTITY, create=False)
        state = restored.snapshot()
        self.assertEqual((state["logical_requests"], state["pending_requests"]), (1, 1))
        self.assertNotEqual(original_head, state["head_sha256"])
        restored.recover_pending("synthetic exited process; no scoring retry")
        self.assertEqual(restored.snapshot()["abandoned_requests"], 1)
        with closing(sqlite3.connect(self.path)) as connection:
            self.assertEqual(connection.execute("SELECT sha256 FROM events WHERE ordinal=1").fetchone()[0], original_head)

    def test_interrupted_attempt_consumes_ordinal_but_does_not_block_later_attempt(self):
        self.ledger.reserve_attempt("old", {"rng_after_batch": 3})
        self.ledger.recover_pending("generation result lost")
        self.assertEqual(self.ledger.reserve_attempt("next", {"rng_after_batch": 4})["ordinal"], 2)
        self.assertEqual(self.ledger.snapshot()["attempts"][0]["status"], "abandoned")

    def test_failed_energy_is_charged_and_same_call_id_never_retried(self):
        with self.scorer(lambda query: (_ for _ in ()).throw(RuntimeError("synthetic evaluator failure"))) as scorer:
            with self.assertRaisesRegex(RuntimeError, "synthetic evaluator"):
                scorer.score("hf", {"x": 0}, kind="hf")
            with self.assertRaisesRegex(ValueError, "silently retried"):
                scorer.score("hf", {"x": 0}, kind="hf")
        state = self.ledger.snapshot()
        self.assertEqual((state["logical_requests"], state["failed_requests"]), (1, 1))
        self.assertIn("RuntimeError", state["requests"][0]["failure_reason"])

    def test_uncertain_result_after_storage_failure_remains_charged(self):
        with self.scorer() as scorer, patch.object(self.cache, "put", side_effect=OSError("disk quota")):
            with self.assertRaises(OSError):
                scorer.score("hf", {"x": 0}, kind="hf")
        self.assertEqual(self.ledger.snapshot()["failed_requests"], 1)
        self.assertEqual(self.cache.verify()["entries"], 0)

    def test_production_evaluation_report_and_cache_origin_preserve_actual_cost_once(self):
        report = {"energy_hartree": -2.0, "norm": 1.0, "populations": [11, 11],
                  "wall_seconds": 3.25, "evaluator_calls": 1,
                  "costmetadata": {"hamiltonian_actions": 1, "peak_bytes": 12345}}
        with self.scorer(lambda query: copy.deepcopy(report)) as scorer:
            first = scorer.submit("hf", {"x": 0}, kind="hf").result()
            second = scorer.submit("same-circuit", {"x": 0}).result()
        self.assertEqual(first["status"], "completed")
        self.assertEqual(first["evaluation"], report)
        self.assertEqual(second["evaluation"]["origin_evaluation"], report)
        self.assertEqual(second["evaluation"]["evaluator_calls"], 0)
        self.assertEqual(second["evaluation"]["wall_seconds"], 0)
        self.assertNotIn("costmetadata", second["evaluation"])
        self.assertEqual(second["cache_provenance"]["producer"]["request_ordinal"], 1)
        self.assertEqual(sum(x["evaluation"]["evaluator_calls"] for x in self.ledger.snapshot()["requests"]), 1)

    def test_failed_physics_exception_preserves_diagnostic_report(self):
        class BadPhysics(RuntimeError):
            report = {"norm": .5, "populations": [11, 11], "evaluator_calls": 1,
                      "wall_seconds": .2, "physical_checks_passed": False}
        def evaluate(query):
            raise BadPhysics("synthetic invalid norm")
        with self.scorer(evaluate) as scorer:
            with self.assertRaises(BadPhysics):
                scorer.score("hf", {"x": 0}, kind="hf")
        result = self.ledger.snapshot()["requests"][0]
        self.assertEqual(result["evaluation"], BadPhysics.report)
        self.assertEqual(result["status"], "failed")

    def test_transition_report_scalar_mismatch_is_not_committed(self):
        self.ledger.reserve_request("hf", {"x": 0}, kind="hf")
        before = self.ledger.snapshot()
        with self.assertRaisesRegex(ValueError, "report and scalar"):
            self.ledger.complete_request(1, -1, evaluation={"energy_hartree": -2})
        self.assertEqual(before, self.ledger.snapshot())

    def test_cache_payload_tampering_rejected_before_reuse(self):
        self.cache.put(CACHE_ID, {"x": 0}, 1)
        with closing(sqlite3.connect(self.root / "cache.sqlite")) as connection, connection:
            connection.execute("DROP TRIGGER no_update")
            raw = connection.execute("SELECT payload FROM events WHERE ordinal=2").fetchone()[0]
            value = json.loads(raw)
            value["energy_hartree"] = -100
            connection.execute("UPDATE events SET payload=? WHERE ordinal=2",
                               (json.dumps(value, sort_keys=True, separators=(",", ":")),))
        with self.assertRaisesRegex(ValueError, "hash differs"):
            self.cache.get(CACHE_ID, {"x": 0})

    def test_full_cache_identity_and_exact_query_isolation(self):
        self.cache.put(CACHE_ID, {"theta": 0.123}, -1)
        for key in CACHE_ID:
            other = copy.deepcopy(CACHE_ID)
            other[key] = "different"
            self.assertIsNone(self.cache.get(other, {"theta": 0.123}))
        extra = dict(CACHE_ID, source_sha256="changed")
        self.assertIsNone(self.cache.get(extra, {"theta": 0.123}))
        self.assertIsNone(self.cache.get(CACHE_ID, {"theta": 0.12300000001}))
        with self.assertRaisesRegex(ValueError, "Cache identity"):
            self.cache.get({"partition": "validation"}, {})
        with self.assertRaisesRegex(ValueError, "exact scorer cache identity"):
            OrderedScorer(self.ledger, self.cache, extra, lambda query: 0)

    def test_immutable_cache_and_pinned_final_digest(self):
        self.cache.put(CACHE_ID, {"x": 1}, 2, provenance={"producer": "fake"})
        before = self.cache.verify()
        self.cache.put(CACHE_ID, {"x": 1}, 2)
        self.assertEqual(before, self.cache.verify())
        with self.assertRaisesRegex(ValueError, "Immutable"):
            self.cache.put(CACHE_ID, {"x": 1}, 3)
        for value in (float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                self.cache.put(CACHE_ID, {"x": 2}, value)
        self.cache.verify(expected_sha256=before["sha256"])
        with self.assertRaisesRegex(ValueError, "pinned contents"):
            self.cache.verify(expected_sha256="0" * 64)

    def test_identity_budget_and_trusted_head_mismatches_rejected_read_only(self):
        state = self.ledger.snapshot()
        verify_ledger(self.path, IDENTITY, expected_head_sha256=state["head_sha256"])
        original = self.path.read_bytes()
        for kwargs in ({"identity": dict(IDENTITY, method="different")},
                       {"identity": IDENTITY, "request_budget": 64},
                       {"identity": IDENTITY, "expected_head_sha256": "f" * 64}):
            with self.assertRaises(ValueError):
                verify_ledger(self.path, **kwargs)
        self.assertEqual(original, self.path.read_bytes())
        reader = TrajectoryLedger(self.path, IDENTITY, create=False, read_only=True)
        with self.assertRaisesRegex(ValueError, "read-only"):
            reader.reserve_attempt("not-written")

    def test_hash_chain_rejects_sqlite_payload_tampering_and_tail_removal(self):
        self.ledger.reserve_attempt("a1")
        pinned = self.ledger.snapshot()["head_sha256"]
        with closing(sqlite3.connect(self.path)) as connection, connection:
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute("UPDATE events SET payload='{}' WHERE ordinal=2")
            connection.execute("DROP TRIGGER no_update")
            connection.execute("UPDATE events SET payload='{}' WHERE ordinal=2")
        with self.assertRaisesRegex(ValueError, "hash differs"):
            verify_ledger(self.path, IDENTITY, expected_head_sha256=pinned)

    def test_deleted_terminal_event_rejected_against_external_head(self):
        self.ledger.reserve_attempt("a1")
        pinned = self.ledger.snapshot()["head_sha256"]
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute("DROP TRIGGER no_delete")
            connection.execute("DELETE FROM events WHERE ordinal=2")
        with self.assertRaisesRegex(ValueError, "terminal head"):
            verify_ledger(self.path, IDENTITY, expected_head_sha256=pinned)

    def test_single_scorer_lease_blocks_second_worker_and_releases_after_close(self):
        first = self.scorer()
        try:
            with self.assertRaises(OSError):
                self.scorer()
        finally:
            first.close()
        with self.scorer() as second:
            self.assertEqual(second.score("hf", {"x": 0}, kind="hf"), 0)

    def test_single_cpu_worker_bounded_queue_and_deterministic_admission(self):
        entered, release, submitted = threading.Event(), threading.Event(), threading.Event()
        calls, active, maximum = [], 0, 0
        def evaluate(query):
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            calls.append(query["x"])
            if query["x"] == 0:
                entered.set()
                self.assertTrue(release.wait(5))
            active -= 1
            return query["x"]
        with self.scorer(evaluate, max_pending=2) as scorer:
            first = scorer.submit("hf", {"x": 0}, kind="hf", expected_ordinal=1)
            self.assertTrue(entered.wait(5))
            second = scorer.submit("c1", {"x": 1}, expected_ordinal=2)
            def submit_third():
                result = scorer.submit("c2", {"x": 2}, expected_ordinal=3)
                submitted.set()
                return result.result()
            with ThreadPoolExecutor(max_workers=1) as caller:
                third = caller.submit(submit_third)
                try:
                    self.assertFalse(submitted.wait(.05))
                    self.assertEqual(self.ledger.snapshot()["logical_requests"], 2)
                finally:
                    release.set()
                self.assertEqual(third.result(timeout=5)["request_ordinal"], 3)
            self.assertEqual((first.result()["ordinal"], second.result()["ordinal"]), (1, 2))
        self.assertEqual(calls, [0, 1, 2])
        self.assertEqual(maximum, 1)

    def test_wrong_deterministic_frontier_does_not_reserve_or_score(self):
        with self.scorer(lambda query: self.fail("no evaluation")) as scorer:
            with self.assertRaisesRegex(ValueError, "frontier"):
                scorer.submit("hf", {"x": 0}, kind="hf", expected_ordinal=2)
        self.assertEqual(self.ledger.snapshot()["logical_requests"], 0)

    def test_concurrent_transactions_preserve_every_attempt_index_without_overwrite(self):
        with ThreadPoolExecutor(max_workers=4) as executor:
            result = list(executor.map(lambda i: self.ledger.reserve_attempt(f"a{i}"), range(20)))
        self.assertEqual(sorted(x["ordinal"] for x in result), list(range(1, 21)))
        self.assertEqual(self.ledger.snapshot()["completion_attempts"], 20)


if __name__ == "__main__":
    unittest.main()
