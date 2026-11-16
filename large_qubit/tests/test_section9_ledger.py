"""Main-ledger fixtures use synthetic scalars only; no model/GPU/physics work."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime
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
from large_qubit.section9_ledger import OrderedScorer, ScalarCache, TrajectoryLedger, verify_ledger


CACHE = {"partition": "final_test", "geometry_id": "synthetic_alias", "hamiltonian": "h",
         "orbitals": "o", "catalogue": "c", "backend": "fake-only", "precision": "float64"}
IDENTITY = {"job_id": "synthetic_job", "queue_id": "single_cpu_queue", "seed": 22,
            "cache_identity": CACHE}


class MainLedgerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / "ledger.sqlite"
        self.ledger = TrajectoryLedger(self.path, IDENTITY)
        self.cache = ScalarCache(self.root / "cache.sqlite")
        self.addCleanup(self.ledger.close)
        self.addCleanup(self.cache.close)

    def scorer(self, evaluate=lambda q: q["x"], **kwargs):
        return OrderedScorer(self.ledger, self.cache, CACHE, evaluate, **kwargs)

    def test_main_defaults_utc_job_queue_cursors_and_hf_inclusion(self):
        with self.scorer() as scorer:
            scorer.score("hf", {"x": 0}, kind="hf")
            scorer.score("candidate", {"x": -1}, expected_ordinal=2)
        state = self.ledger.snapshot()
        self.assertEqual(state["budgets"], {"requests": 1024, "attempts": 2048})
        self.assertEqual((state["job_id"], state["queue_id"]), ("synthetic_job", "single_cpu_queue"))
        self.assertEqual(state["cursors"], {"event": 5, "attempt": 0, "request": 2})
        self.assertEqual(state["incumbent"]["energy_hartree"], -1)
        for request in state["requests"]:
            self.assertTrue(datetime.fromisoformat(request["reserved_utc"].replace("Z", "+00:00")).tzinfo)
            self.assertTrue(request["completed_utc"].endswith("Z"))
        self.assertNotIn("requests", self.ledger.frontier())

    def test_rejected_admission_is_atomic_and_budget_counts_hf_duplicates_cache_hits(self):
        self.ledger.close()
        self.ledger = TrajectoryLedger(self.root / "short.sqlite", IDENTITY, request_budget=3)
        self.addCleanup(self.ledger.close)
        before = self.ledger.frontier()["head_sha256"]
        with self.assertRaisesRegex(ValueError, "first logical"):
            self.ledger.reserve_request("wrong", {"x": 0})
        self.assertEqual(before, self.ledger.frontier()["head_sha256"])
        calls = []
        with self.scorer(lambda q: calls.append(q["x"]) or q["x"]) as scorer:
            scorer.score("hf", {"x": 0}, kind="hf")
            scorer.score("a", {"x": -1})
            scorer.score("b", {"x": -1})
            with self.assertRaisesRegex(ValueError, "budget exhausted"):
                scorer.score("c", {"x": -2})
        self.assertEqual(calls, [0, -1])
        state = self.ledger.frontier()
        self.assertEqual((state["logical_requests"], state["duplicate_requests"], state["cache_hits"]), (3, 1, 1))

    def test_explicit_attempt_seed_durable_invalid_and_abandoned_frontiers(self):
        with self.assertRaisesRegex(ValueError, "explicit nonnegative seed"):
            self.ledger.reserve_attempt("missing", {})
        a = self.ledger.reserve_attempt("a1", {"seed": 456, "batch_cursor": 0})
        self.assertEqual((a["ordinal"], a["status"]), (1, "pending"))
        self.ledger.finish_attempt(1, {"valid": False, "raw": "bad"}, valid=False)
        self.ledger.reserve_attempt("a2", {"seed": 789})
        self.ledger.recover_pending("lost generation output consumes attempt")
        self.assertEqual(self.ledger.reserve_attempt("a3", {"seed": 790})["ordinal"], 3)
        self.assertEqual(self.ledger.lookup_attempt("a2")["status"], "abandoned")
        with self.assertRaisesRegex(ValueError, "input differs"):
            self.ledger.reserve_attempt("a1", {"seed": 457})

    def test_generation_query_requires_completed_valid_attempt(self):
        with self.scorer() as scorer:
            scorer.score("hf", {"x": 0}, kind="hf")
        self.ledger.reserve_attempt("a", {"seed": 1})
        with self.assertRaisesRegex(ValueError, "completed valid"):
            self.ledger.reserve_request("q", {"x": -1}, attempt_ordinal=1)
        self.ledger.finish_attempt(1, {"ops": [1]}, valid=True)
        self.assertEqual(self.ledger.reserve_request("q", {"x": -1}, attempt_ordinal=1)["ordinal"], 2)

    def test_late_gpu_output_after_cpu_halt_keeps_cost_but_never_regains_admission(self):
        self.ledger.reserve_attempt("gpu_in_flight", {"seed": 123})
        self.ledger.halt("concurrent CPU scorer failed")
        before = self.ledger.frontier()
        output = {"valid": True, "ops": [1] * 32, "wall_seconds": 2.5,
                  "input_tokens": 100, "output_tokens": 40, "error_kind": None}
        record = self.ledger.finish_attempt(1, output, valid=True)
        self.assertEqual(record["status"], "abandoned")
        self.assertTrue(record["late_result_unusable"])
        self.assertEqual(record["output"], output)
        self.assertEqual(record["output_sha256"], digest(output))
        self.assertEqual(record["failure_reason"], "concurrent CPU scorer failed")
        self.assertTrue(record["late_completed_utc"].endswith("Z"))
        self.assertEqual(self.ledger.frontier()["completion_attempts"], before["completion_attempts"])
        self.assertEqual(self.ledger.frontier()["logical_requests"], before["logical_requests"])
        head = self.ledger.frontier()["head_sha256"]
        with self.assertRaisesRegex(ValueError, "terminal intent"):
            self.ledger.finish_attempt(1, output, valid=True)
        self.assertEqual(self.ledger.frontier()["head_sha256"], head)
        with self.assertRaisesRegex(ValueError, "Stopped trajectory"):
            self.ledger.reserve_request("cannot_admit", {"x": -1}, attempt_ordinal=1)
        checked = verify_ledger(self.path, IDENTITY, expected_head_sha256=head)
        self.assertEqual(checked["attempts"][0], record)

    def test_completed_vqe_replay_returns_saved_value_and_never_recharges(self):
        with self.scorer() as scorer:
            scorer.score("hf", {"x": 0}, kind="hf")
            scorer.score("vqe:restart:0:probe:0", {"x": -1}, kind="vqe")
        before = self.ledger.snapshot()
        with self.scorer(lambda q: self.fail("no replay evaluation")) as scorer:
            self.assertEqual(scorer.score("vqe:restart:0:probe:0", {"x": -1}, kind="vqe"), -1)
            with self.assertRaisesRegex(ValueError, "identity differs"):
                scorer.score("vqe:restart:0:probe:0", {"x": -1.0}, kind="vqe")
        self.assertEqual(before, self.ledger.snapshot())

    def test_failure_halts_queued_work_preserves_incumbent_and_all_charges(self):
        entered, release = threading.Event(), threading.Event()
        calls = []
        def evaluate(q):
            calls.append(q["x"])
            if q["x"] == 99:
                entered.set()
                self.assertTrue(release.wait(5))
                raise RuntimeError("synthetic physical failure")
            return q["x"]
        with self.scorer(evaluate, max_pending=2) as scorer:
            scorer.score("hf", {"x": 0}, kind="hf")
            scorer.score("good", {"x": -1})
            failed = scorer.submit("bad", {"x": 99})
            self.assertTrue(entered.wait(5))
            queued = scorer.submit("queued", {"x": 1000})
            release.set()
            with self.assertRaises(RuntimeError):
                failed.result(timeout=5)
            with self.assertRaises(ValueError):
                queued.result(timeout=5)
            with self.assertRaises(ValueError):
                scorer.score("refill", {"x": -100})
        state = self.ledger.snapshot()
        self.assertEqual(calls, [0, -1, 99])
        self.assertEqual((state["logical_requests"], state["completed_requests"], state["failed_requests"],
                          state["abandoned_requests"]), (4, 2, 1, 1))
        self.assertEqual(state["incumbent"]["energy_hartree"], -1)
        self.assertTrue(state["halted"])

    def test_full_report_cache_origin_and_failed_diagnostic_costs(self):
        report = {"energy_hartree": -1, "norm": 1, "populations": [11, 11], "wall_seconds": 2,
                  "evaluator_calls": 1, "costmetadata": {"h_actions": 1}}
        with self.scorer(lambda q: report) as scorer:
            first = scorer.submit("hf", {"x": 0}, kind="hf").result()
            second = scorer.submit("same", {"x": 0}).result()
        self.assertEqual(first["evaluation"], report)
        self.assertEqual(second["evaluation"]["origin_evaluation"], report)
        self.assertEqual(second["evaluation"]["evaluator_calls"], 0)
        self.assertNotIn("costmetadata", second["evaluation"])
        class BadPhysics(RuntimeError):
            report = {"evaluator_calls": 1, "wall_seconds": .5, "norm": .1}
        with self.scorer(lambda q: (_ for _ in ()).throw(BadPhysics("bad norm"))) as scorer:
            with self.assertRaises(BadPhysics):
                scorer.score("bad", {"x": 2})
        self.assertEqual(self.ledger.lookup_request("bad")["evaluation"], BadPhysics.report)

    def test_cache_partition_alias_and_all_physics_identity_fields_are_exact(self):
        self.cache.put(CACHE, {"theta": .1}, -1)
        for key in CACHE:
            other = dict(CACHE, **{key: "different"})
            if key == "partition":
                with self.assertRaises(ValueError):
                    self.cache.get(other, {"theta": .1})
            else:
                self.assertIsNone(self.cache.get(other, {"theta": .1}))
        self.assertIsNone(self.cache.get(CACHE, {"theta": .10000000001}))
        before = self.cache.verify()
        with self.assertRaisesRegex(ValueError, "Immutable"):
            self.cache.put(CACHE, {"theta": .1}, -2)
        self.assertEqual(before, self.cache.verify(expected_sha256=before["sha256"]))

    def test_external_sqlite_tamper_detected_by_live_index_and_final_verifier(self):
        self.ledger.reserve_attempt("a", {"seed": 1})
        head = self.ledger.frontier()["head_sha256"]
        with closing(sqlite3.connect(self.path)) as c, c:
            c.execute("DROP TRIGGER no_update")
            raw = c.execute("SELECT payload FROM events WHERE ordinal=2").fetchone()[0]
            value = json.loads(raw)
            value["input"]["seed"] = 2
            c.execute("UPDATE events SET payload=? WHERE ordinal=2", (json.dumps(value, sort_keys=True, separators=(",", ":")),))
        with self.assertRaisesRegex(ValueError, "hash differs"):
            self.ledger.frontier()
        with self.assertRaisesRegex(ValueError, "hash differs"):
            verify_ledger(self.path, IDENTITY, expected_head_sha256=head)

    def test_deleting_terminal_event_requires_external_head_pin_to_detect_truncation(self):
        self.ledger.reserve_attempt("a", {"seed": 1})
        head = self.ledger.frontier()["head_sha256"]
        with closing(sqlite3.connect(self.path)) as c, c:
            c.execute("DROP TRIGGER no_delete")
            c.execute("DELETE FROM events WHERE ordinal=2")
        with self.assertRaisesRegex(ValueError, "terminal head"):
            verify_ledger(self.path, IDENTITY, expected_head_sha256=head)

    def test_own_appends_increment_indexes_external_valid_writer_reauthenticates(self):
        with patch.object(self.ledger._journal, "make_index", wraps=self.ledger._journal.make_index) as replay:
            for i in range(20):
                self.ledger.reserve_attempt(f"a{i}", {"seed": i})
            self.assertEqual(replay.call_count, 0)
            other = TrajectoryLedger(self.path, IDENTITY, create=False)
            try:
                other.reserve_attempt("external", {"seed": 99})
            finally:
                other.close()
            self.assertEqual(self.ledger.frontier()["completion_attempts"], 21)
            self.assertEqual(replay.call_count, 1)
            self.ledger.snapshot(expected_head_sha256=self.ledger.frontier()["head_sha256"])
            self.assertEqual(replay.call_count, 2)

    def test_abrupt_child_death_keeps_intent_then_recovery_halts_without_evaluation(self):
        self.ledger.close()
        script = ("import json,os,sys;from large_qubit.section9_ledger import TrajectoryLedger;"
                  "x=TrajectoryLedger(sys.argv[1],json.loads(sys.argv[2]),create=False);"
                  "x.reserve_request('hf',{'x':0},kind='hf');os._exit(23)")
        result = subprocess.run([sys.executable, "-B", "-c", script, str(self.path), json.dumps(IDENTITY)],
                                cwd=Path(__file__).resolve().parents[2], capture_output=True, check=False)
        self.assertEqual(result.returncode, 23, result.stderr.decode(errors="replace"))
        self.ledger = TrajectoryLedger(self.path, IDENTITY, create=False)
        self.addCleanup(self.ledger.close)
        self.assertEqual(self.ledger.frontier()["pending_requests"], 1)
        self.ledger.recover_pending("lost process; uncertain request stays charged")
        self.assertEqual(self.ledger.frontier()["abandoned_requests"], 1)
        with self.assertRaisesRegex(ValueError, "Stopped trajectory"):
            self.scorer()

    def test_read_only_identity_budget_and_head_authentication_do_not_write(self):
        state = self.ledger.snapshot()
        before = self.path.read_bytes()
        for kwargs in ({"identity": dict(IDENTITY, job_id="other")}, {"identity": IDENTITY, "request_budget": 10},
                       {"identity": IDENTITY, "expected_head_sha256": "0" * 64}):
            with self.assertRaises(ValueError):
                verify_ledger(self.path, **kwargs)
        self.assertEqual(self.path.read_bytes(), before)
        checked = verify_ledger(self.path, IDENTITY, expected_head_sha256=state["head_sha256"])
        self.assertEqual(checked, state)

    def test_existing_empty_or_corrupt_ledger_and_cache_never_reinitialize(self):
        for kind in ("zero_bytes", "empty_schema", "unrelated_schema", "empty_events", "corrupt_bytes"):
            for create in (False, True):
                for cache in (False, True):
                    with self.subTest(kind=kind, create=create, cache=cache):
                        path = self.root / f"existing_{kind}_{create}_{cache}.sqlite"
                        if kind == "zero_bytes":
                            path.write_bytes(b"")
                        elif kind == "corrupt_bytes":
                            path.write_bytes(b"preserve corrupt sync artifact verbatim")
                        else:
                            with closing(sqlite3.connect(path)) as c, c:
                                if kind == "empty_schema":
                                    c.execute("PRAGMA user_version=1")
                                elif kind == "unrelated_schema":
                                    c.execute("CREATE TABLE unrelated (value TEXT)")
                                else:
                                    c.execute("CREATE TABLE events (ordinal INTEGER PRIMARY KEY, payload TEXT NOT NULL, "
                                              "previous_sha256 TEXT NOT NULL, sha256 TEXT NOT NULL)")
                        before = path.read_bytes()
                        names = set(self.root.iterdir())
                        with self.assertRaises((ValueError, sqlite3.DatabaseError)):
                            if cache:
                                ScalarCache(path, create=create)
                            else:
                                TrajectoryLedger(path, IDENTITY, create=create)
                        self.assertEqual(path.read_bytes(), before)
                        self.assertEqual(set(self.root.iterdir()), names)

    def test_queue_lease_frontier_and_finish_prevent_unregistered_new_work(self):
        scorer = self.scorer()
        try:
            with self.assertRaises(OSError):
                self.scorer()
            with self.assertRaisesRegex(ValueError, "frontier"):
                scorer.submit("hf", {"x": 0}, kind="hf", expected_ordinal=2)
            scorer.score("hf", {"x": 0}, kind="hf")
        finally:
            scorer.close()
        self.ledger.finish("budget or attempt cap reached")
        with self.assertRaisesRegex(ValueError, "Stopped trajectory"):
            self.ledger.reserve_request("extra", {"x": -1})

    def test_concurrent_intents_keep_unique_durable_indexes(self):
        with ThreadPoolExecutor(max_workers=4) as executor:
            rows = list(executor.map(lambda i: self.ledger.reserve_attempt(f"a{i}", {"seed": i}), range(32)))
        self.assertEqual(sorted(x["ordinal"] for x in rows), list(range(1, 33)))
        self.assertEqual(self.ledger.frontier()["completion_attempts"], 32)

    def test_disk_failure_after_durable_intent_stops_and_retains_cost(self):
        with self.scorer() as scorer, patch.object(self.cache, "put", side_effect=OSError("synthetic quota")):
            with self.assertRaises(OSError):
                scorer.score("hf", {"x": 0}, kind="hf")
            with self.assertRaises(ValueError):
                scorer.score("retry", {"x": -1})
        state = self.ledger.frontier()
        self.assertEqual((state["logical_requests"], state["failed_requests"]), (1, 1))
        self.assertTrue(state["halted"])


if __name__ == "__main__":
    unittest.main()
