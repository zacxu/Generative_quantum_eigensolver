"""Durable validation accounting; no model, chemistry or GPU imports.

SQLite commits the intent before work. Completed logical call IDs replay without a
new charge; uncertain physical requests must be abandoned and the run stopped.
Hash chains authenticate committed history against a caller-pinned final head.
"""

import json
import math
import os
from contextlib import contextmanager
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
import sqlite3
import threading

from .common import canonical_bytes, digest, require


SCHEMA = "section8-durable-ledger-v1"
CACHE_KEYS = ("partition", "geometry_id", "hamiltonian", "orbitals", "catalogue", "backend", "precision")
ZERO = "0" * 64


def _json(value):
    return json.loads(canonical_bytes(value))


def _name(value):
    require(isinstance(value, str) and bool(value.strip()), "A nonempty logical call ID is required")
    return value


def _energy(value):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value), "Energy must be a finite scalar")
    return float(value)


class _Journal:
    """A transactional append-only event chain, with no mutable frontier file."""

    def __init__(self, path, initial, *, create=True, read_only=False):
        self.path = Path(path).absolute()
        require(self.path.resolve() == self.path, "Ledger paths cannot redirect through links")
        self.initial = _json(initial)
        self.read_only = read_only
        if not self.path.exists():
            require(create and not read_only, "Ledger does not exist")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute("CREATE TABLE IF NOT EXISTS events (ordinal INTEGER PRIMARY KEY, "
                                   "payload TEXT NOT NULL, previous_sha256 TEXT NOT NULL, sha256 TEXT NOT NULL)")
                connection.execute("CREATE TRIGGER IF NOT EXISTS no_update BEFORE UPDATE ON events "
                                   "BEGIN SELECT RAISE(ABORT, 'immutable events'); END")
                connection.execute("CREATE TRIGGER IF NOT EXISTS no_delete BEFORE DELETE ON events "
                                   "BEGIN SELECT RAISE(ABORT, 'immutable events'); END")
                if connection.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0:
                    self._insert(connection, [], self.initial)
        self.read()

    @contextmanager
    def _connect(self):
        if self.read_only:
            connection = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, timeout=60)
        else:
            connection = sqlite3.connect(str(self.path), timeout=60)
            connection.execute("PRAGMA synchronous=FULL")
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _read(self, connection):
        rows = connection.execute("SELECT ordinal,payload,previous_sha256,sha256 FROM events ORDER BY ordinal").fetchall()
        require(bool(rows), "Ledger has no committed identity")
        result = []
        previous = ZERO
        for expected, (ordinal, raw, prior, saved_hash) in enumerate(rows, 1):
            payload = json.loads(raw)
            require(ordinal == expected and prior == previous
                    and canonical_bytes(payload).decode() == raw, "Ledger event ordering or canonical bytes differ")
            computed = digest({"ordinal": ordinal, "previous_sha256": prior, "payload": payload})
            require(saved_hash == computed, "Ledger event hash differs")
            result.append({"ordinal": ordinal, "sha256": saved_hash, "payload": payload})
            previous = saved_hash
        require(result[0]["payload"] == self.initial, "Ledger identity or budgets differ")
        return result

    @staticmethod
    def _insert(connection, events, payload):
        ordinal = len(events) + 1
        previous = events[-1]["sha256"] if events else ZERO
        payload = _json(payload)
        sha = digest({"ordinal": ordinal, "previous_sha256": previous, "payload": payload})
        connection.execute("INSERT INTO events VALUES (?,?,?,?)",
                           (ordinal, canonical_bytes(payload).decode(), previous, sha))

    def read(self):
        with self._connect() as connection:
            return self._read(connection)

    def change(self, build):
        require(not self.read_only, "Ledger is read-only")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            events = self._read(connection)
            payload = build(events)
            if payload is not None:
                self._insert(connection, events, payload)
            return self._read(connection)


class TrajectoryLedger:
    def __init__(self, path, identity, *, request_budget=128, attempt_budget=256,
                 create=True, read_only=False):
        require(isinstance(identity, dict) and bool(identity), "A full trajectory identity is required")
        require(type(request_budget) is int and request_budget >= 1
                and type(attempt_budget) is int and attempt_budget >= 0, "Invalid ledger budgets")
        self.identity = _json(identity)
        self.request_budget, self.attempt_budget = request_budget, attempt_budget
        self._journal = _Journal(path, {"event": "identity", "schema_version": SCHEMA,
                                       "identity": self.identity, "request_budget": request_budget,
                                       "attempt_budget": attempt_budget}, create=create, read_only=read_only)
        self.snapshot()

    def _reduce(self, events):
        attempts, requests = [], []
        seen_attempts, seen_requests, first_query = set(), set(), {}
        abandoned_physical = False
        for event in events[1:]:
            item = event["payload"]
            name = item.get("event")
            if name in ("attempt_reserved", "request_reserved"):
                records = attempts if name == "attempt_reserved" else requests
                seen = seen_attempts if name == "attempt_reserved" else seen_requests
                limit = self.attempt_budget if name == "attempt_reserved" else self.request_budget
                ordinal = item["ordinal"]
                require(type(ordinal) is int and ordinal == len(records) + 1 and ordinal <= limit,
                        "Nonconsecutive admission or budget exceeded")
                _name(item["call_id"])
                require(item["call_id"] not in seen, "Duplicate logical call ID")
                seen.add(item["call_id"])
                if name == "attempt_reserved":
                    require(item["input_sha256"] == digest(item["input"]), "Attempt input hash differs")
                    records.append({"ordinal": ordinal, "call_id": item["call_id"], "input": item["input"],
                                    "input_sha256": item["input_sha256"], "status": "pending", "output": None,
                                    "output_sha256": None, "valid": None, "failure_reason": None})
                else:
                    require(not abandoned_physical,
                            "An uncertain physical request requires a documented new run")
                    require((ordinal == 1 and item["kind"] == "hf" and item["call_id"] == "hf")
                            or (ordinal > 1 and item["kind"] != "hf" and item["call_id"] != "hf"),
                            "The first logical request must be HF exactly once")
                    require(isinstance(item["kind"], str) and bool(item["kind"]), "Request kind is required")
                    require(item["query_sha256"] == digest(item["query"]), "Request query hash differs")
                    attempt = item["attempt_ordinal"]
                    require(attempt is None or (type(attempt) is int and 1 <= attempt <= len(attempts)
                                               and attempts[attempt - 1]["status"] == "completed"
                                               and attempts[attempt - 1]["valid"] is True),
                            "Request does not refer to an authenticated valid attempt")
                    duplicate = first_query.get(item["query_sha256"])
                    require(item["duplicate_of"] == duplicate, "Duplicate admission identity differs")
                    first_query.setdefault(item["query_sha256"], ordinal)
                    records.append({"ordinal": ordinal, "request_ordinal": ordinal, "call_id": item["call_id"],
                                    "query": item["query"], "query_sha256": item["query_sha256"],
                                    "kind": item["kind"], "attempt_ordinal": attempt, "duplicate_of": duplicate,
                                    "status": "pending", "energy_hartree": None, "cache_hit": False,
                                    "evaluation": None, "cache_provenance": None, "failure_reason": None})
            elif name in ("attempt_finished", "request_completed", "request_failed"):
                records = attempts if name == "attempt_finished" else requests
                ordinal = item["ordinal"]
                require(type(ordinal) is int and 1 <= ordinal <= len(records)
                        and records[ordinal - 1]["status"] == "pending", "Missing or already terminal intent")
                record = records[ordinal - 1]
                if name == "attempt_finished":
                    require(type(item["valid"]) is bool and item["output_sha256"] == digest(item["output"]),
                            "Attempt output hash or validity differs")
                    record.update(status="completed", valid=item["valid"], output=item["output"],
                                  output_sha256=item["output_sha256"], failure_reason=item["failure_reason"])
                elif name == "request_completed":
                    require(type(item["cache_hit"]) is bool, "Cache-hit flag must be boolean")
                    energy = _energy(item["energy_hartree"])
                    evaluation = item["evaluation"]
                    require(isinstance(evaluation, dict)
                            and _energy(evaluation["energy_hartree"]) == energy,
                            "Evaluation report and scalar energy differ")
                    record.update(status="completed", energy_hartree=energy, cache_hit=item["cache_hit"],
                                  evaluation=evaluation, cache_provenance=item["cache_provenance"])
                else:
                    require(isinstance(item["failure_reason"], str) and bool(item["failure_reason"]),
                            "Physical failure needs a recorded reason")
                    record.update(status="failed", failure_reason=item["failure_reason"],
                                  evaluation=item["evaluation"])
                record["completion_sha256"] = event["sha256"]
            elif name == "pending_abandoned":
                require(isinstance(item["reason"], str) and bool(item["reason"]), "Recovery reason required")
                for key, records in (("attempts", attempts), ("requests", requests)):
                    pending = [x["ordinal"] for x in records if x["status"] == "pending"]
                    require(item[key] == pending, "Recovery does not cover exactly the pending intents")
                    for ordinal in pending:
                        records[ordinal - 1].update(status="abandoned", failure_reason=item["reason"])
                    if key == "requests" and pending:
                        abandoned_physical = True
            else:
                raise ValueError("Unknown ledger event")
        scored = [x for x in requests if x["status"] == "completed"]
        return {"schema_version": SCHEMA, "identity": _json(self.identity), "identity_sha256": digest(self.identity),
                "budgets": {"requests": self.request_budget, "attempts": self.attempt_budget},
                "head_sha256": events[-1]["sha256"], "events_count": len(events),
                "attempts": attempts, "requests": requests, "completion_attempts": len(attempts),
                "logical_requests": len(requests), "completed_requests": len(scored),
                "failed_requests": sum(x["status"] == "failed" for x in requests),
                "abandoned_requests": sum(x["status"] == "abandoned" for x in requests),
                "pending_requests": sum(x["status"] == "pending" for x in requests),
                "pending_attempts": sum(x["status"] == "pending" for x in attempts),
                "cache_hits": sum(x["cache_hit"] for x in requests),
                "duplicate_requests": sum(x["duplicate_of"] is not None for x in requests),
                "incumbent": min(scored, key=lambda x: (x["energy_hartree"], x["ordinal"])) if scored else None}

    def snapshot(self, *, expected_head_sha256=None):
        state = self._reduce(self._journal.read())
        require(expected_head_sha256 is None or expected_head_sha256 == state["head_sha256"],
                "Ledger does not match the pinned terminal head")
        return state

    def _change(self, builder):
        def build(events):
            payload = builder(self._reduce(events))
            if payload is not None:
                # Reject invalid transitions before committing any bytes.
                self._reduce(events + [{"payload": payload, "sha256": digest(payload)}])
            return payload
        return self._reduce(self._journal.change(build))

    def reserve_attempt(self, call_id, payload=None):
        call_id, payload = _name(call_id), _json(payload)
        def build(state):
            old = next((x for x in state["attempts"] if x["call_id"] == call_id), None)
            if old:
                require(old["input_sha256"] == digest(payload), "Replayed attempt input differs")
                return None
            require(state["completion_attempts"] < self.attempt_budget, "Completion-attempt budget exhausted")
            return {"event": "attempt_reserved", "ordinal": state["completion_attempts"] + 1,
                    "call_id": call_id, "input": payload, "input_sha256": digest(payload)}
        return next(x for x in self._change(build)["attempts"] if x["call_id"] == call_id)

    def finish_attempt(self, ordinal, output, *, valid, failure_reason=None):
        require(type(valid) is bool, "Validity must be boolean")
        payload = {"event": "attempt_finished", "ordinal": ordinal, "output": _json(output),
                   "output_sha256": digest(output), "valid": valid, "failure_reason": failure_reason}
        state = self._change(lambda state: payload)
        return state["attempts"][ordinal - 1]

    def lookup_request(self, call_id):
        return next((x for x in self.snapshot()["requests"] if x["call_id"] == call_id), None)

    def reserve_request(self, call_id, query, *, kind="candidate", attempt_ordinal=None, require_new=False):
        call_id, query = _name(call_id), _json(query)
        query_sha = digest(query)
        def build(state):
            old = next((x for x in state["requests"] if x["call_id"] == call_id), None)
            if old:
                require(not require_new, "Another scorer already admitted this logical call")
                require(old["query_sha256"] == query_sha and old["kind"] == kind
                        and old["attempt_ordinal"] == attempt_ordinal, "Replayed physical request identity differs")
                return None
            require(state["logical_requests"] < self.request_budget, "Logical-request budget exhausted")
            return {"event": "request_reserved", "ordinal": state["logical_requests"] + 1,
                    "call_id": call_id, "query": query, "query_sha256": query_sha, "kind": kind,
                    "attempt_ordinal": attempt_ordinal,
                    "duplicate_of": next((x["ordinal"] for x in state["requests"]
                                          if x["query_sha256"] == query_sha), None)}
        return next(x for x in self._change(build)["requests"] if x["call_id"] == call_id)

    def complete_request(self, ordinal, energy_hartree, *, cache_hit=False, evaluation=None, cache_provenance=None):
        evaluation = {"energy_hartree": _energy(energy_hartree)} if evaluation is None else _json(evaluation)
        payload = {"event": "request_completed", "ordinal": ordinal,
                   "energy_hartree": _energy(energy_hartree), "cache_hit": cache_hit,
                   "evaluation": evaluation, "cache_provenance": _json(cache_provenance)}
        return self._change(lambda state: payload)["requests"][ordinal - 1]

    def fail_request(self, ordinal, reason, *, evaluation=None):
        payload = {"event": "request_failed", "ordinal": ordinal, "failure_reason": str(reason),
                   "evaluation": _json(evaluation)}
        return self._change(lambda state: payload)["requests"][ordinal - 1]

    def recover_pending(self, reason="Interrupted work has an uncertain outcome; charged and abandoned"):
        def build(state):
            attempts = [x["ordinal"] for x in state["attempts"] if x["status"] == "pending"]
            requests = [x["ordinal"] for x in state["requests"] if x["status"] == "pending"]
            return {"event": "pending_abandoned", "reason": reason, "attempts": attempts,
                    "requests": requests} if attempts or requests else None
        return self._change(build)


def verify_ledger(path, identity, *, request_budget=128, attempt_budget=256, expected_head_sha256=None):
    return TrajectoryLedger(path, identity, request_budget=request_budget, attempt_budget=attempt_budget,
                            create=False, read_only=True).snapshot(expected_head_sha256=expected_head_sha256)


class ScalarCache:
    """Shared exact scalar values; every cache use still needs a ledger intent."""

    def __init__(self, path, *, create=True, read_only=False):
        self._journal = _Journal(path, {"event": "identity", "schema_version": "section8-scalar-cache-v1"},
                                 create=create, read_only=read_only)
        self.verify()

    @staticmethod
    def key(identity, query):
        require(isinstance(identity, dict) and all(k in identity and identity[k] is not None for k in CACHE_KEYS),
                "Cache identity needs partition, geometry, Hamiltonian, orbitals, catalogue, backend and precision")
        return digest({"identity": identity, "query": query})

    def _reduce(self, events):
        rows = {}
        for event in events[1:]:
            item = event["payload"]
            require(item.get("event") == "scalar_saved", "Unknown scalar-cache event")
            key = self.key(item["identity"], item["query"])
            require(key == item["key"] and key not in rows, "Scalar-cache key differs or is repeated")
            _energy(item["energy_hartree"])
            rows[key] = item
        return rows

    def get(self, identity, query):
        return self._reduce(self._journal.read()).get(self.key(identity, query))

    def put(self, identity, query, energy_hartree, *, provenance=None, evaluation=None):
        identity, query, energy_hartree = _json(identity), _json(query), _energy(energy_hartree)
        key = self.key(identity, query)
        def build(events):
            old = self._reduce(events).get(key)
            if old:
                require(old["energy_hartree"] == energy_hartree, "Immutable scalar-cache value differs")
                return None
            return {"event": "scalar_saved", "key": key, "identity": identity, "query": query,
                    "energy_hartree": energy_hartree, "provenance": _json(provenance),
                    "evaluation": _json(evaluation)}
        return self._reduce(self._journal.change(build))[key]

    def verify(self, *, expected_sha256=None):
        events = self._journal.read()
        rows = self._reduce(events)
        result = {"schema_version": "section8-scalar-cache-verification-v1", "entries": len(rows),
                  "head_sha256": events[-1]["sha256"], "sha256": digest(rows)}
        require(expected_sha256 is None or expected_sha256 == result["sha256"],
                "Scalar cache does not match its pinned contents")
        return result


class OrderedScorer:
    """One CPU scoring worker and a bounded, admission-ordered pending queue.

    The controller submits in deterministic order (or supplies expected_ordinal).
    Closing drains admitted work. Do not recover a ledger while its worker lives.
    """

    def __init__(self, ledger, cache, cache_identity, evaluate, *, max_pending=2):
        require(type(max_pending) is int and max_pending >= 1, "Invalid scorer queue bound")
        ScalarCache.key(cache_identity, None)
        require(ledger.identity.get("cache_identity_sha256") == digest(cache_identity)
                or ledger.identity.get("cache_identity") == cache_identity,
                "Trajectory identity must bind the exact scorer cache identity")
        self._lease = (ledger._journal.path.with_suffix(ledger._journal.path.suffix + ".scorer.lock")).open("a+b")
        try:
            self._lease.seek(0, 2)
            if self._lease.tell() == 0:
                self._lease.write(b"0")
                self._lease.flush()
            self._lease.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self._lease.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            require(ledger.snapshot()["pending_requests"] == 0, "Recover uncertain pending requests before scoring")
            require(ledger.snapshot()["abandoned_requests"] == 0, "Abandoned physical work requires a new run")
        except BaseException:
            self._lease.close()
            raise
        self.ledger, self.cache, self.cache_identity = ledger, cache, _json(cache_identity)
        self.evaluate = evaluate
        self._slots = threading.BoundedSemaphore(max_pending)
        self._admission = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="section8-cpu-scorer")
        self._closed = False

    def _evaluate(self, record):
        try:
            cached = self.cache.get(self.cache_identity, record["query"])
            if cached is None:
                result = self.evaluate(_json(record["query"]))
                evaluation = _json(result) if isinstance(result, dict) else {"energy_hartree": _energy(result)}
                energy = _energy(evaluation["energy_hartree"])
                self.cache.put(self.cache_identity, record["query"], energy,
                               provenance={"ledger_identity_sha256": digest(self.ledger.identity),
                                           "request_ordinal": record["ordinal"]}, evaluation=evaluation)
                cache_provenance = None
            else:
                energy = cached["energy_hartree"]
                evaluation = {"energy_hartree": energy, "cache_hit": True, "evaluator_calls": 0,
                              "wall_seconds": 0.0, "origin_evaluation": cached["evaluation"],
                              "origin_provenance": cached["provenance"]}
                cache_provenance = {"key": cached["key"], "producer": cached["provenance"]}
            return self.ledger.complete_request(record["ordinal"], energy, cache_hit=cached is not None,
                                                evaluation=evaluation, cache_provenance=cache_provenance)
        except Exception as error:
            # If failure recording also fails, the original durable intent stays
            # pending and recovery conservatively abandons it as charged work.
            current = self.ledger.lookup_request(record["call_id"])
            if current is not None and current["status"] == "pending":
                report = getattr(error, "report", None)
                try:
                    report = _json(report)
                except (ValueError, TypeError):
                    report = {"unserializable_report_repr": repr(report)}
                self.ledger.fail_request(record["ordinal"], f"{type(error).__name__}: {error}", evaluation=report)
            raise

    def submit(self, call_id, query, *, kind="candidate", attempt_ordinal=None, expected_ordinal=None):
        # Hold admission while waiting for capacity so callers cannot overtake.
        with self._admission:
            require(not self._closed, "Scorer is closed")
            old = self.ledger.lookup_request(call_id)
            if old is not None:
                record = self.ledger.reserve_request(call_id, query, kind=kind, attempt_ordinal=attempt_ordinal)
                require(record["status"] == "completed", "An unfinished/failed logical call cannot be silently retried")
                result = Future()
                result.set_result(record)
                return result
            if expected_ordinal is not None:
                require(expected_ordinal == self.ledger.snapshot()["logical_requests"] + 1,
                        "Admission ordinal differs from the deterministic controller frontier")
            self._slots.acquire()
            try:
                record = self.ledger.reserve_request(call_id, query, kind=kind, attempt_ordinal=attempt_ordinal,
                                                     require_new=True)
                future = self._pool.submit(self._evaluate, record)
            except BaseException:
                self._slots.release()
                raise
            future.add_done_callback(lambda unused: self._slots.release())
            return future

    def score(self, call_id, query, **kwargs):
        return self.submit(call_id, query, **kwargs).result()["energy_hartree"]

    def close(self):
        with self._admission:
            self._closed = True
        try:
            self._pool.shutdown(wait=True)
        finally:
            self._lease.close()

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()
