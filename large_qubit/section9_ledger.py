"""Incrementally indexed main-matrix accounting; no models or physics imports.

Each intent is a FULL-synchronous SQLite transaction before work. A persistent
connection/index avoids replay on our own append; an external SQLite commit
forces complete authentication. Open and final verification always replay the
entire hash chain. Final heads must be pinned by the controller's manifest.
"""

from concurrent.futures import Future
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import threading

from .common import canonical_bytes, digest, require
from .section8_ledger import OrderedScorer as _OldScorer, _energy, _json, _name, CACHE_KEYS, ZERO


SCHEMA = "section9-durable-ledger-v1"


def _utc():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _stamp(value):
    require(isinstance(value, str) and value.endswith("Z")
            and datetime.fromisoformat(value.replace("Z", "+00:00")).utcoffset().total_seconds() == 0,
            "Event UTC timestamp is invalid")


class _Store:
    """One verified in-memory index protected by SQLite and a process RLock."""

    def __init__(self, path, initial, make_index, *, create=True, read_only=False):
        self.path = Path(path).absolute()
        require(self.path.resolve() == self.path, "Ledger paths cannot redirect through links")
        existed = self.path.exists()
        require(existed or (create and not read_only), "Ledger does not exist")
        if not existed:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Own the new pathname exclusively. A crash here leaves an empty
            # preserved artifact, which future opens must refuse, never reset.
            with self.path.open("xb"):
                pass
        self.read_only, self.initial, self.make_index = read_only, _json(initial), make_index
        self.lock = threading.RLock()
        target = self.path.as_uri() + "?mode=ro" if read_only else str(self.path)
        self.connection = sqlite3.connect(target, uri=read_only, check_same_thread=False, timeout=60)
        self.closed = False
        try:
            if not read_only:
                self.connection.execute("PRAGMA synchronous=FULL")
            if not existed:
                self.connection.execute("BEGIN IMMEDIATE")
                self.connection.execute("CREATE TABLE events (ordinal INTEGER PRIMARY KEY, "
                                        "payload TEXT NOT NULL, previous_sha256 TEXT NOT NULL, sha256 TEXT NOT NULL)")
                self.connection.execute("CREATE TRIGGER no_update BEFORE UPDATE ON events "
                                        "BEGIN SELECT RAISE(ABORT, 'immutable events'); END")
                self.connection.execute("CREATE TRIGGER no_delete BEFORE DELETE ON events "
                                        "BEGIN SELECT RAISE(ABORT, 'immutable events'); END")
                payload = dict(self.initial, utc=_utc())
                self._insert(1, ZERO, payload)
                self.connection.commit()
            self.file_id = (self.path.stat().st_dev, self.path.stat().st_ino)
            self.version = None
            self.read(lambda index: None, full=True)
        except BaseException:
            self.connection.close()
            self.closed = True
            raise

    def _insert(self, ordinal, previous, payload):
        sha = digest({"ordinal": ordinal, "previous_sha256": previous, "payload": payload})
        self.connection.execute("INSERT INTO events VALUES (?,?,?,?)",
                                (ordinal, canonical_bytes(payload).decode(), previous, sha))
        return sha

    def _refresh(self, full=False):
        require(not self.closed and self.path.resolve() == self.path
                and (self.path.stat().st_dev, self.path.stat().st_ino) == self.file_id,
                "Ledger file was replaced or redirected")
        version = self.connection.execute("PRAGMA data_version").fetchone()[0]
        if full or version != self.version:
            rows = self.connection.execute("SELECT ordinal,payload,previous_sha256,sha256 "
                                           "FROM events ORDER BY ordinal").fetchall()
            require(bool(rows), "Missing committed ledger identity")
            index, previous = self.make_index(), ZERO
            for ordinal, (saved_ordinal, raw, prior, sha) in enumerate(rows, 1):
                payload = json.loads(raw)
                require(saved_ordinal == ordinal and previous == prior and canonical_bytes(payload).decode() == raw,
                        "Ledger ordering or canonical bytes differ")
                require(sha == digest({"ordinal": ordinal, "previous_sha256": prior, "payload": payload}),
                        "Ledger event hash differs")
                _stamp(payload.get("utc"))
                if ordinal == 1:
                    require({k: v for k, v in payload.items() if k != "utc"} == self.initial,
                            "Ledger identity or budgets differ")
                    index.created_utc = payload["utc"]
                else:
                    index.accept(payload, sha)
                previous = sha
            self.index, self.count, self.head = index, len(rows), previous
            self.last_utc, self.version = payload["utc"], version

    def _run(self, callback, *, write=False, full=False):
        with self.lock:
            require(not self.closed, "Ledger is closed")
            require(not write or not self.read_only, "Ledger is read-only")
            self.connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            try:
                self._refresh(full)
                if not write:
                    result = callback(self.index)
                    self.connection.commit()
                    return result
                payload = callback(self.index)
                if payload is None:
                    self.connection.commit()
                    return None
                payload = dict(_json(payload), utc=_utc())
                _stamp(payload["utc"])
                self.index.accept(payload, None, validate_only=True)
                sha = self._insert(self.count + 1, self.head, payload)
                self.connection.commit()
                try:
                    self.index.accept(payload, sha)
                except BaseException:
                    self.version = None  # Rebuild from the durable commit on next access.
                    raise
                self.count, self.head, self.last_utc = self.count + 1, sha, payload["utc"]
                # Own commits do not increment this connection's data_version.
                # Do not sample it after commit: that could swallow a racing
                # external write before the next operation authenticates it.
                return sha
            except BaseException:
                self.connection.rollback()
                raise

    def read(self, callback, *, full=False):
        return self._run(callback, full=full)

    def append(self, build):
        return self._run(build, write=True)

    def close(self):
        with self.lock:
            if not self.closed:
                self.connection.close()
                self.closed = True


class _Trajectory:
    def __init__(self, request_budget, attempt_budget):
        self.request_budget, self.attempt_budget = request_budget, attempt_budget
        self.attempts, self.requests, self.attempt_ids, self.request_ids = [], [], {}, {}
        self.first_query, self.pending_attempts, self.pending_requests = {}, set(), set()
        self.failed, self.abandoned, self.cache_hits, self.duplicates, self.completed = 0, 0, 0, 0, 0
        self.incumbent, self.halted, self.finished, self.stop_reason = None, False, False, None
        self.created_utc = None

    def accept(self, p, sha, validate_only=False):
        event, ordinal = p.get("event"), p.get("ordinal")
        if event in ("attempt_reserved", "request_reserved"):
            attempt = event == "attempt_reserved"
            rows, ids = (self.attempts, self.attempt_ids) if attempt else (self.requests, self.request_ids)
            limit = self.attempt_budget if attempt else self.request_budget
            _name(p["call_id"])
            require(not self.halted and not self.finished, "Stopped trajectory cannot admit new work")
            require(type(ordinal) is int and ordinal == len(rows) + 1 and ordinal <= limit,
                    "Nonconsecutive admission or budget exhausted")
            require(p["call_id"] not in ids, "Duplicate logical call ID")
            if attempt:
                require(isinstance(p["input"], dict) and type(p["input"].get("seed")) is int
                        and p["input"]["seed"] >= 0, "Each generation attempt needs an explicit nonnegative seed")
                require(digest(p["input"]) == p["input_sha256"], "Attempt input hash differs")
                record = {"ordinal": ordinal, "call_id": p["call_id"], "input": p["input"],
                          "input_sha256": p["input_sha256"], "status": "pending", "valid": None,
                          "output": None, "output_sha256": None, "failure_reason": None}
            else:
                require((ordinal == 1 and p["kind"] == "hf" and p["call_id"] == "hf")
                        or (ordinal > 1 and p["kind"] != "hf" and p["call_id"] != "hf"),
                        "The first logical request must be HF exactly once")
                require(isinstance(p["kind"], str) and bool(p["kind"]), "Request kind is required")
                require(digest(p["query"]) == p["query_sha256"], "Request query hash differs")
                source = p["attempt_ordinal"]
                require(source is None or (type(source) is int and 1 <= source <= len(self.attempts)
                        and self.attempts[source - 1]["status"] == "completed"
                        and self.attempts[source - 1]["valid"] is True), "Request needs a completed valid attempt")
                duplicate = self.first_query.get(p["query_sha256"])
                require(p["duplicate_of"] == duplicate, "Duplicate request identity differs")
                record = {"ordinal": ordinal, "request_ordinal": ordinal, "call_id": p["call_id"],
                          "query": p["query"], "query_sha256": p["query_sha256"], "kind": p["kind"],
                          "attempt_ordinal": source, "duplicate_of": duplicate, "status": "pending",
                          "energy_hartree": None, "evaluation": None, "cache_provenance": None,
                          "cache_hit": False, "failure_reason": None}
            if not validate_only:
                record.update(reserved_utc=p["utc"], completed_utc=None)
                rows.append(record)
                ids[p["call_id"]] = record
                (self.pending_attempts if attempt else self.pending_requests).add(ordinal)
                if not attempt:
                    self.first_query.setdefault(p["query_sha256"], ordinal)
                    self.duplicates += p["duplicate_of"] is not None
        elif event in ("attempt_finished", "attempt_late_result", "request_completed", "request_failed", "request_abandoned"):
            attempt = event in ("attempt_finished", "attempt_late_result")
            late = event == "attempt_late_result"
            rows = self.attempts if attempt else self.requests
            require(type(ordinal) is int and 1 <= ordinal <= len(rows), "Missing intent")
            record = rows[ordinal - 1]
            require((late and record["status"] == "abandoned" and record["output"] is None
                     and not record.get("late_result_unusable"))
                    or (not late and record["status"] == "pending"), "Missing or already terminal intent")
            if attempt:
                require(type(p["valid"]) is bool and digest(p["output"]) == p["output_sha256"],
                        "Attempt output or validity differs")
                update = {"status": "completed", "valid": p["valid"], "output": p["output"],
                          "output_sha256": p["output_sha256"], "failure_reason": p["failure_reason"]}
                if late:
                    update.update(status="abandoned", late_result_unusable=True,
                                  failure_reason=record["failure_reason"], late_failure_reason=p["failure_reason"])
            elif event == "request_completed":
                energy = _energy(p["energy_hartree"])
                require(type(p["cache_hit"]) is bool and isinstance(p["evaluation"], dict)
                        and _energy(p["evaluation"]["energy_hartree"]) == energy,
                        "Evaluation report or cache flag differs")
                update = {"status": "completed", "energy_hartree": energy, "evaluation": p["evaluation"],
                          "cache_hit": p["cache_hit"], "cache_provenance": p["cache_provenance"]}
            else:
                require(isinstance(p["failure_reason"], str) and bool(p["failure_reason"]), "Failure reason required")
                update = {"status": "failed" if event == "request_failed" else "abandoned",
                          "failure_reason": p["failure_reason"], "evaluation": p.get("evaluation")}
            if not validate_only:
                if late:
                    record.update(update, late_completion_sha256=sha, late_completed_utc=p["utc"])
                else:
                    record.update(update, completion_sha256=sha, completed_utc=p["utc"])
                    (self.pending_attempts if attempt else self.pending_requests).remove(ordinal)
                if not attempt:
                    if event == "request_completed":
                        self.completed += 1
                        self.cache_hits += p["cache_hit"]
                        if self.incumbent is None or (record["energy_hartree"], ordinal) < (
                                self.incumbent["energy_hartree"], self.incumbent["ordinal"]):
                            self.incumbent = record
                    else:
                        self.failed += event == "request_failed"
                        self.abandoned += event == "request_abandoned"
                        self.halted, self.stop_reason = True, p["failure_reason"]
        elif event in ("pending_abandoned", "halted"):
            require(isinstance(p["reason"], str) and bool(p["reason"]), "Recovery reason required")
            require(p["attempts"] == sorted(self.pending_attempts)
                    and p["requests"] == sorted(self.pending_requests), "Recovery frontier differs")
            if not validate_only:
                for ids, rows in ((self.pending_attempts, self.attempts), (self.pending_requests, self.requests)):
                    for i in ids:
                        rows[i - 1].update(status="abandoned", failure_reason=p["reason"], completed_utc=p["utc"],
                                           completion_sha256=sha)
                self.abandoned += len(self.pending_requests)
                if self.pending_requests or event == "halted":
                    self.halted, self.stop_reason = True, p["reason"]
                self.pending_attempts.clear()
                self.pending_requests.clear()
        elif event == "finished":
            require(not self.pending_attempts and not self.pending_requests and not self.finished
                    and isinstance(p["reason"], str) and bool(p["reason"]), "Cannot finish this frontier")
            if not validate_only:
                self.finished, self.stop_reason = True, p["reason"]
        else:
            raise ValueError("Unknown ledger event")


class TrajectoryLedger:
    def __init__(self, path, identity, *, request_budget=1024, attempt_budget=2048,
                 create=True, read_only=False):
        require(isinstance(identity, dict) and bool(identity), "A trajectory identity is required")
        require(type(request_budget) is int and request_budget >= 1
                and type(attempt_budget) is int and attempt_budget >= 0, "Invalid ledger budgets")
        require(all(isinstance(identity.get(k), str) and bool(identity[k]) for k in ("job_id", "queue_id")),
                "Trajectory identity must bind job_id and queue_id")
        self.identity = _json(identity)
        self.request_budget, self.attempt_budget = request_budget, attempt_budget
        self._journal = _Store(path, {"event": "identity", "schema_version": SCHEMA, "identity": self.identity,
            "request_budget": request_budget, "attempt_budget": attempt_budget},
            lambda: _Trajectory(request_budget, attempt_budget), create=create, read_only=read_only)

    def _summary(self, state, records):
        result = {"schema_version": SCHEMA, "identity": self.identity, "identity_sha256": digest(self.identity),
                  "budgets": {"requests": self.request_budget, "attempts": self.attempt_budget},
                  "head_sha256": self._journal.head, "events_count": self._journal.count,
                  "created_utc": state.created_utc, "last_event_utc": self._journal.last_utc,
                  "job_id": self.identity["job_id"], "queue_id": self.identity["queue_id"],
                  "cursors": {"event": self._journal.count, "attempt": len(state.attempts), "request": len(state.requests)},
                  "completion_attempts": len(state.attempts), "logical_requests": len(state.requests),
                  "completed_requests": state.completed, "failed_requests": state.failed,
                  "abandoned_requests": state.abandoned, "pending_requests": len(state.pending_requests),
                  "pending_attempts": len(state.pending_attempts), "cache_hits": state.cache_hits,
                  "duplicate_requests": state.duplicates, "incumbent": state.incumbent,
                  "halted": state.halted, "finished": state.finished, "stop_reason": state.stop_reason,
                  "status": "failed" if state.halted else "finished" if state.finished else "running"}
        if records:
            result.update(attempts=state.attempts, requests=state.requests)
        return _json(result)

    def snapshot(self, *, expected_head_sha256=None):
        result = self._journal.read(lambda state: self._summary(state, True), full=expected_head_sha256 is not None)
        require(expected_head_sha256 is None or result["head_sha256"] == expected_head_sha256,
                "Ledger does not match the pinned terminal head")
        return result

    def frontier(self):
        return self._journal.read(lambda state: self._summary(state, False))

    def lookup_request(self, call_id):
        return self._journal.read(lambda state: _json(state.request_ids.get(call_id)))

    def lookup_attempt(self, call_id):
        return self._journal.read(lambda state: _json(state.attempt_ids.get(call_id)))

    def reserve_attempt(self, call_id, payload=None):
        call_id, payload = _name(call_id), _json(payload)
        sha = digest(payload)
        def build(s):
            old = s.attempt_ids.get(call_id)
            if old:
                require(old["input_sha256"] == sha, "Replayed attempt input differs")
                return None
            return {"event": "attempt_reserved", "ordinal": len(s.attempts) + 1,
                    "call_id": call_id, "input": payload, "input_sha256": sha}
        self._journal.append(build)
        return self.lookup_attempt(call_id)

    def finish_attempt(self, ordinal, output, *, valid, failure_reason=None):
        output = _json(output)
        def build(s):
            require(type(ordinal) is int and 1 <= ordinal <= len(s.attempts), "Missing intent")
            late = s.attempts[ordinal - 1]["status"] == "abandoned"
            return {"event": "attempt_late_result" if late else "attempt_finished", "ordinal": ordinal,
                    "output": output, "output_sha256": digest(output), "valid": valid, "failure_reason": failure_reason}
        self._journal.append(build)
        return self._journal.read(lambda s: _json(s.attempts[ordinal - 1]))

    def reserve_request(self, call_id, query, *, kind="candidate", attempt_ordinal=None, require_new=False):
        call_id, query = _name(call_id), _json(query)
        sha = digest(query)
        def build(s):
            old = s.request_ids.get(call_id)
            if old:
                require(not require_new, "Another scorer already admitted this logical call")
                require(old["query_sha256"] == sha and old["kind"] == kind
                        and old["attempt_ordinal"] == attempt_ordinal, "Replayed physical request identity differs")
                return None
            return {"event": "request_reserved", "ordinal": len(s.requests) + 1, "call_id": call_id,
                    "query": query, "query_sha256": sha, "kind": kind, "attempt_ordinal": attempt_ordinal,
                    "duplicate_of": s.first_query.get(sha)}
        self._journal.append(build)
        return self.lookup_request(call_id)

    def complete_request(self, ordinal, energy_hartree, *, cache_hit=False, evaluation=None, cache_provenance=None):
        energy = _energy(energy_hartree)
        evaluation = {"energy_hartree": energy} if evaluation is None else _json(evaluation)
        self._journal.append(lambda s: {"event": "request_completed", "ordinal": ordinal,
            "energy_hartree": energy, "cache_hit": cache_hit, "evaluation": evaluation,
            "cache_provenance": _json(cache_provenance)})
        return self._journal.read(lambda s: _json(s.requests[ordinal - 1]))

    def fail_request(self, ordinal, reason, *, evaluation=None):
        return self._terminal_request(ordinal, reason, "request_failed", evaluation)

    def abandon_request(self, ordinal, reason):
        return self._terminal_request(ordinal, reason, "request_abandoned", None)

    def _terminal_request(self, ordinal, reason, event, evaluation):
        self._journal.append(lambda s: {"event": event, "ordinal": ordinal,
                                       "failure_reason": str(reason), "evaluation": _json(evaluation)})
        return self._journal.read(lambda s: _json(s.requests[ordinal - 1]))

    def recover_pending(self, reason="Interrupted work is uncertain; charged and abandoned"):
        return self._recover(reason, False)

    def halt(self, reason):
        return self._recover(reason, True)

    def _recover(self, reason, halt):
        def build(s):
            if not halt and not s.pending_attempts and not s.pending_requests:
                return None
            return {"event": "halted" if halt else "pending_abandoned", "reason": reason,
                    "attempts": sorted(s.pending_attempts), "requests": sorted(s.pending_requests)}
        self._journal.append(build)
        return self.snapshot()

    def finish(self, reason):
        self._journal.append(lambda s: {"event": "finished", "reason": reason})
        return self.snapshot()

    def close(self):
        self._journal.close()


def verify_ledger(path, identity, *, request_budget=1024, attempt_budget=2048, expected_head_sha256=None):
    ledger = TrajectoryLedger(path, identity, request_budget=request_budget, attempt_budget=attempt_budget,
                              create=False, read_only=True)
    try:
        return ledger.snapshot(expected_head_sha256=expected_head_sha256)
    finally:
        ledger.close()


class _Cache:
    def __init__(self):
        self.rows, self.created_utc = {}, None

    def accept(self, p, sha, validate_only=False):
        require(p.get("event") == "scalar_saved", "Unknown scalar-cache event")
        key = ScalarCache.key(p["identity"], p["query"])
        require(key == p["key"] and key not in self.rows, "Scalar-cache key differs or repeats")
        energy = _energy(p["energy_hartree"])
        require(p["evaluation"] is None or (isinstance(p["evaluation"], dict)
                and _energy(p["evaluation"]["energy_hartree"]) == energy), "Cache evaluation and energy differ")
        if not validate_only:
            self.rows[key] = p


class ScalarCache:
    def __init__(self, path, *, create=True, read_only=False):
        self._journal = _Store(path, {"event": "identity", "schema_version": "section9-scalar-cache-v1"},
                               _Cache, create=create, read_only=read_only)

    @staticmethod
    def key(identity, query):
        require(isinstance(identity, dict) and all(k in identity and identity[k] is not None for k in CACHE_KEYS)
                and identity["partition"] == "final_test", "Cache needs complete final_test physics identity")
        return digest({"identity": identity, "query": query})

    def get(self, identity, query):
        key = self.key(identity, query)
        return self._journal.read(lambda s: _json(s.rows.get(key)))

    def put(self, identity, query, energy_hartree, *, provenance=None, evaluation=None):
        identity, query, energy = _json(identity), _json(query), _energy(energy_hartree)
        key = self.key(identity, query)
        def build(s):
            old = s.rows.get(key)
            if old:
                require(old["energy_hartree"] == energy, "Immutable scalar-cache value differs")
                return None
            return {"event": "scalar_saved", "key": key, "identity": identity, "query": query,
                    "energy_hartree": energy, "provenance": _json(provenance), "evaluation": _json(evaluation)}
        self._journal.append(build)
        return self.get(identity, query)

    def verify(self, *, expected_sha256=None):
        result = self._journal.read(lambda s: {"schema_version": "section9-scalar-cache-verification-v1",
            "entries": len(s.rows), "head_sha256": self._journal.head, "sha256": digest(s.rows)}, full=True)
        require(expected_sha256 is None or expected_sha256 == result["sha256"], "Cache does not match pinned contents")
        return result

    def close(self):
        self._journal.close()


class OrderedScorer(_OldScorer):
    """Same one-worker API, with fail-closed queue draining and O(1) admission."""

    def __init__(self, ledger, cache, cache_identity, evaluate, *, max_pending=2):
        ScalarCache.key(cache_identity, None)
        require(not ledger.frontier()["halted"], "Stopped trajectory cannot resume scoring")
        self._stopped = threading.Event()
        super().__init__(ledger, cache, cache_identity, evaluate, max_pending=max_pending)

    def _evaluate(self, record):
        if self._stopped.is_set() or self.ledger.frontier()["halted"]:
            if self.ledger.lookup_request(record["call_id"])["status"] == "pending":
                self.ledger.abandon_request(record["ordinal"], "Earlier scorer failure; queued request charged without evaluation")
            raise ValueError("Trajectory stopped after an earlier failure")
        try:
            return super()._evaluate(record)
        except BaseException as error:
            self._stopped.set()
            try:
                self.ledger.halt(f"{type(error).__name__}: {error}")
            except Exception:
                # Disk failure cannot refund durable reservations; they remain
                # pending and a later recovery must conservatively abandon them.
                pass
            raise

    def submit(self, call_id, query, *, kind="candidate", attempt_ordinal=None, expected_ordinal=None):
        with self._admission:
            require(not self._closed, "Scorer is closed")
            old = self.ledger.lookup_request(call_id)
            if old is not None:
                record = self.ledger.reserve_request(call_id, query, kind=kind, attempt_ordinal=attempt_ordinal)
                require(record["status"] == "completed", "An unfinished/failed call cannot be silently retried")
                future = Future()
                future.set_result(record)
                return future
            require(not self._stopped.is_set(), "Trajectory stopped after scorer failure")
            if expected_ordinal is not None:
                require(expected_ordinal == self.ledger.frontier()["logical_requests"] + 1,
                        "Admission differs from the deterministic controller frontier")
            self._slots.acquire()
            try:
                record = self.ledger.reserve_request(call_id, query, kind=kind, attempt_ordinal=attempt_ordinal,
                                                     require_new=True)
                try:
                    future = self._pool.submit(self._evaluate, record)
                except BaseException:
                    self._stopped.set()
                    self.ledger.abandon_request(record["ordinal"], "Worker submission failed before evaluation")
                    raise
            except BaseException:
                self._slots.release()
                raise
            future.add_done_callback(lambda unused: self._slots.release())
            return future
