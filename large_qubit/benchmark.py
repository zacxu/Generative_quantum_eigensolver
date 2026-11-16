"""Measured CPU evaluator throughput with bounded, independent worker processes.

Importing this module loads no numerical backend. Representative legal circuits
are deterministic engineering workloads, not LLM generations or optimized VQE.
Every warm-up and measured energy request is charged in private append-only
logs. This module never reads a correlated reference or starts model training.
"""

from contextlib import contextmanager
import importlib
from itertools import combinations
import json
import math
import multiprocessing
import os
from pathlib import Path
import queue
import random
import sys
import time
import traceback

from .common import canonical_bytes, digest, require, sha256_file, workspace_path
from .protocol import BENCHMARK_ID, seed_for


ANGLES = (-.1, -.05, -.025, -.0125, .0125, .025, .05, .1)
FAMILIES = ("whole_qubit", "fermionic")
LENGTHS = (32, 64)
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")
MAX_RSS_BYTES = int(7.5 * 1024 ** 3)


def excitation_templates():
    """All full-space CO2 occupied-to-virtual spin-conserving templates."""
    occupied, virtual = range(22), range(22, 30)
    singles = [(i, a) for i in occupied for a in virtual if i % 2 == a % 2]
    doubles = [(i, j, a, b) for i, j in combinations(occupied, 2)
               for a, b in combinations(virtual, 2) if i % 2 + j % 2 == a % 2 + b % 2]
    result = tuple(singles + doubles)
    require(len(result) == 2684, "Representative workload catalogue count differs")
    return result


def representative_circuits(*, samples=2, warmups=1):
    """Return warm-up-first fixed recipes, shared across worker comparisons."""
    require(type(samples) is int and samples > 0 and type(warmups) is int and warmups >= 1,
            "Use positive measured samples and at least one warm-up per group")
    templates, result = excitation_templates(), []
    for phase, count in (("warmup", warmups), ("measured", samples)):
        for family in FAMILIES:
            for length in LENGTHS:
                for sample in range(count):
                    group = f"section4:CO2_T02:{family}:{length}:{phase}:{sample}"
                    seed = seed_for("generation", group=group)
                    rng = random.Random(seed)
                    gates = [{"wires": list(templates[rng.randrange(len(templates))]),
                              "theta": ANGLES[rng.randrange(len(ANGLES))]} for _ in range(length)]
                    result.append({"phase": phase, "family": family, "length": length,
                                   "sample": sample, "seed_group": group, "seed": seed,
                                   "gates": gates, "circuit_sha256": digest({"family": family, "gates": gates})})
    return result


@contextmanager
def _thread_environment(threads):
    previous = {name: os.environ.get(name) for name in THREAD_VARIABLES}
    try:
        for name in THREAD_VARIABLES:
            os.environ[name] = str(threads)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _rss_bytes(pid=None):
    path = Path("/proc/self/status" if pid is None else f"/proc/{pid}/status")
    try:
        lines = path.read_text(encoding="ascii").splitlines()
        return int(next(line.split()[1] for line in lines if line.startswith("VmRSS:"))) * 1024
    except (OSError, StopIteration, ValueError):
        return 0


def _peak_rss_bytes():
    if sys.platform.startswith("linux"):
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    return _rss_bytes()


def _total_memory_bytes():
    try:
        lines = Path("/proc/meminfo").read_text(encoding="ascii").splitlines()
        return int(next(line.split()[1] for line in lines if line.startswith("MemTotal:"))) * 1024
    except (OSError, StopIteration, ValueError) as error:
        raise RuntimeError("Run CPU throughput measurements in Linux/WSL with native /proc memory accounting") from error


def _append(handle, record):
    started = time.perf_counter()
    handle.write(canonical_bytes(record).decode("ascii") + "\n")
    handle.flush()
    return time.perf_counter() - started


def _measure_requests(evaluator, basis, plans, handle, *, first_ordinal=1, memory_budget_bytes=MAX_RSS_BYTES,
                      first_seen=None):
    """Internal measurable kernel; injectable tiny evaluators support unit tests."""
    public = []
    first_seen = {} if first_seen is None else first_seen
    for ordinal, plan in enumerate(plans, first_ordinal):
        started = time.perf_counter()
        state = basis.apply_circuit(basis.hf_state(), [(g["wires"], g["theta"]) for g in plan["gates"]],
                                    family=plan["family"])
        preparation_seconds = time.perf_counter() - started
        duplicate_of = first_seen.get(plan["circuit_sha256"])
        first_seen.setdefault(plan["circuit_sha256"], ordinal)
        identity = {key: plan[key] for key in ("phase", "family", "length", "sample", "circuit_sha256")}
        # Commit the intent before the energy call: exceptions and interrupted
        # calculations still consume this issued logical request.
        logging_seconds = _append(handle, {"event": "request_started", "request_ordinal": ordinal,
                                            **identity, "duplicate_of": duplicate_of,
                                            "energy_cache_hit": False, "gates": plan["gates"]})
        before = evaluator.statistics()["hamiltonian_action_seconds"]
        score_started = time.perf_counter()
        energy = evaluator.energy_sector(state, basis=basis)
        score_seconds = time.perf_counter() - score_started
        require(not isinstance(energy, bool) and math.isfinite(energy), "Nonfinite benchmark energy")
        action_seconds = evaluator.statistics()["hamiltonian_action_seconds"] - before
        cache = basis.cache_info()
        timing = {**identity, "request_ordinal": ordinal,
                  "state_preparation_seconds": preparation_seconds,
                  "score_seconds": score_seconds, "hamiltonian_action_seconds": action_seconds,
                  "gate_cache_bytes": cache["bytes"], "peak_rss_bytes": _peak_rss_bytes()}
        logging_seconds += _append(handle, {"event": "request_completed", **timing,
                                             "energy_hartree": float(energy), "cache_info": cache,
                                             "normalization": basis.diagnostics(state)})
        timing["logging_seconds"] = logging_seconds
        public.append(timing)
        require(timing["peak_rss_bytes"] <= memory_budget_bytes,
                "Worker RSS exceeds the declared memory allowance")
    return public


def _worker_entry(workspace, logfile, plans, threads, cache_budget_bytes, memory_budget_bytes,
                  event, mailbox, worker_id, loader_spec=("large_qubit.section4", "load_co2_evaluator")):
    """Spawn entry: environment limits are inherited before scientific imports."""
    try:
        with _thread_environment(threads), Path(logfile).open("x", encoding="ascii", newline="\n") as handle:
            initialized = time.perf_counter()
            module = importlib.import_module(loader_spec[0])
            evaluator, metadata = getattr(module, loader_spec[1])(workspace, alias="CO2_T02", threads=threads)
            from .sector import SectorBasis
            basis = SectorBasis(evaluator.norb, evaluator.nalpha, evaluator.nbeta,
                                cache_budget_bytes=cache_budget_bytes)
            initialization_seconds = time.perf_counter() - initialized
            _append(handle, {"event": "worker_initialized", "worker_id": worker_id,
                             "geometry": "CO2_T02", "metadata": metadata,
                             "threads": threads, "native_thread_environment": {
                                 name: os.environ[name] for name in THREAD_VARIABLES}})
            require(_peak_rss_bytes() <= memory_budget_bytes, "Initialization exceeded worker memory allowance")
            warmups = [p for p in plans if p["phase"] == "warmup"]
            measured = [p for p in plans if p["phase"] == "measured"]
            warmup_started = time.perf_counter()
            first_seen = {}
            warmup_rows = _measure_requests(evaluator, basis, warmups, handle,
                                            memory_budget_bytes=memory_budget_bytes, first_seen=first_seen)
            warmup_seconds = time.perf_counter() - warmup_started
            mailbox.put({"type": "ready", "worker_id": worker_id,
                         "initialization_seconds": initialization_seconds, "warmup_seconds": warmup_seconds})
            require(event.wait(timeout=900), "Controller did not release the measured-batch barrier")
            measured_started = time.perf_counter()
            measured_rows = _measure_requests(evaluator, basis, measured, handle,
                                              first_ordinal=len(warmups) + 1,
                                              memory_budget_bytes=memory_budget_bytes, first_seen=first_seen)
            measured_seconds = time.perf_counter() - measured_started
            report = {"type": "done", "worker_id": worker_id,
                      "initialization_seconds": initialization_seconds, "warmup_seconds": warmup_seconds,
                      "measured_wall_seconds": measured_seconds, "measured_requests": len(measured_rows),
                      "logical_requests": len(warmup_rows) + len(measured_rows),
                      "peak_rss_bytes": _peak_rss_bytes(), "cache_info": basis.cache_info(),
                      "timings": warmup_rows + measured_rows}
            _append(handle, {"event": "worker_finished", "logical_requests": report["logical_requests"]})
            mailbox.put(report)
    except BaseException as error:
        # All diagnostic details stay beside the raw energy ledger.
        failure = Path(logfile).with_suffix(".failure.json")
        try:
            with failure.open("x", encoding="utf-8") as handle:
                handle.write(canonical_bytes({"error_type": type(error).__name__, "message": str(error),
                                              "traceback": traceback.format_exc()}).decode("ascii") + "\n")
        finally:
            mailbox.put({"type": "failed", "worker_id": worker_id, "error_type": type(error).__name__})


def _charged_requests(folder):
    count = 0
    for path in Path(folder).glob("worker_*.jsonl"):
        for line in path.read_text(encoding="ascii").splitlines():
            if json.loads(line).get("event") == "request_started":
                count += 1
    return count


def _run_concurrency(workspace, folder, plans, *, workers, threads, cache_budget_bytes, memory_budget_bytes,
                     loader_spec=("large_qubit.section4", "load_co2_evaluator")):
    context = multiprocessing.get_context("spawn")
    event, mailbox = context.Event(), context.Queue()
    processes, ready, completed = [], {}, {}
    started = time.perf_counter()
    measured_started = None
    sampled_peak = 0
    folder.mkdir(exist_ok=False)
    try:
        with _thread_environment(threads):
            for worker_id in range(workers):
                process = context.Process(target=_worker_entry, args=(
                    str(workspace), str(folder / f"worker_{worker_id:02d}.jsonl"), plans, threads,
                    cache_budget_bytes, memory_budget_bytes // workers, event, mailbox, worker_id, loader_spec))
                process.start()
                processes.append(process)
        while len(completed) < workers:
            current_rss = _rss_bytes() + sum(_rss_bytes(p.pid) for p in processes if p.is_alive())
            sampled_peak = max(sampled_peak, current_rss)
            require(current_rss <= memory_budget_bytes, "Concurrent worker/controller RSS exceeded allowance")
            require(time.perf_counter() - started <= 1800, "Bounded evaluator timing exceeded 30-minute limit")
            try:
                message = mailbox.get(timeout=.1)
            except queue.Empty:
                require(all(p.is_alive() or p.exitcode == 0 for p in processes),
                        "Evaluator worker exited without a successful record")
                if all(not p.is_alive() for p in processes):
                    # Allow the multiprocessing feeder to deliver final reports.
                    try:
                        message = mailbox.get(timeout=1.0)
                    except queue.Empty as error:
                        raise RuntimeError("Worker ended without a complete timing report") from error
                else:
                    continue
            require(message["type"] != "failed", f"Evaluator worker failed: {message.get('error_type')}")
            worker_id = message["worker_id"]
            if message["type"] == "ready":
                require(worker_id not in ready, "Repeated worker-ready identity")
                ready[worker_id] = message
                if len(ready) == workers:
                    measured_started = time.perf_counter()
                    event.set()
            elif message["type"] == "done":
                require(worker_id in ready and worker_id not in completed, "Unordered/repeated worker completion")
                completed[worker_id] = message
            else:
                raise ValueError("Unknown evaluator worker event")
        measured_wall = time.perf_counter() - measured_started
        for process in processes:
            process.join(timeout=10)
            require(process.exitcode == 0, "Evaluator worker did not exit cleanly")
        records = [completed[i] for i in range(workers)]
        charged = _charged_requests(folder)
        require(charged == len(plans) * workers, "Private logical-request ledger differs from measured workload")
        # Sum of process high-water marks is a conservative upper bound even
        # when their individual maxima occurred at different instants.
        aggregate_peak = max(sampled_peak, _peak_rss_bytes() + sum(r["peak_rss_bytes"] for r in records))
        require(aggregate_peak <= memory_budget_bytes, "Conservative aggregate RSS bound exceeded allowance")
        measured_requests = sum(r["measured_requests"] for r in records)
        return {"workers": workers, "threads_per_worker": threads,
                "logical_requests": charged, "warmup_requests": charged - measured_requests,
                "measured_requests": measured_requests, "measured_wall_seconds": measured_wall,
                "sustained_requests_per_second": measured_requests / measured_wall,
                "total_wall_seconds_including_init_and_warmup": time.perf_counter() - started,
                "aggregate_peak_rss_bytes": aggregate_peak,
                "sampled_concurrent_peak_rss_bytes": sampled_peak,
                "memory_measurement": "sum_process_high_watermarks_and_sampled_controller_plus_workers_RSS",
                "worker_reports": [{k: v for k, v in r.items() if k != "type"} for r in records]}
    finally:
        event.set()
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
        mailbox.close()
        mailbox.join_thread()


def benchmark_evaluator(workspace, *, output_directory, threads=1, workers=(1, 2), samples=2,
                        cache_budget_bytes=64 * 1024 ** 2):
    """Execute bounded training-only timings; caller explicitly authorizes this work.

    Returns public timing/count/memory metadata only. Raw request energies and
    worker diagnostics are written exclusively beneath the supplied private
    scoring directory. Completed and failed attempts are never overwritten.
    """
    root = Path(workspace).resolve(strict=True)
    require(sys.platform.startswith("linux"), "Run scientific timing in Linux/WSL")
    require(type(threads) is int and threads > 0, "threads must be a positive integer")
    require(type(samples) is int and samples > 0, "samples must be a positive integer")
    require(type(cache_budget_bytes) is int and cache_budget_bytes >= 0, "Invalid gate cache byte budget")
    require(type(workers) in (list, tuple) and tuple(workers) == (1, 2), "Benchmark actual concurrency 1 then 2")
    destination = Path(output_directory)
    require(destination.is_absolute(), "Benchmark output_directory must be an explicit absolute private path")
    try:
        relative = destination.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("Benchmark output must remain inside the project") from error
    folder = workspace_path(root, relative, namespace=f"large_qubit/scoring_only/{BENCHMARK_ID}")
    require(not folder.exists(), "Benchmark output already exists; preserve it and select a new attempt")
    total_memory = _total_memory_bytes()
    memory_budget = min(MAX_RSS_BYTES, int(total_memory * .75))
    plans = representative_circuits(samples=samples)
    folder.mkdir(parents=True, exist_ok=False)
    runs = []
    try:
        for count in workers:
            runs.append(_run_concurrency(root, folder / f"concurrency_{count}", plans, workers=count,
                                         threads=threads, cache_budget_bytes=cache_budget_bytes,
                                         memory_budget_bytes=memory_budget))
        report = {"schema_version": "section4-evaluator-benchmark-v1", "benchmark_id": BENCHMARK_ID,
                  "geometry": "CO2_T02", "cpu_backend": True, "gpu_acceleration_claimed": False,
                  "circuits": "deterministic_representatives_not_LLM_generations_or_optimized_VQE",
                  "template_count": len(excitation_templates()), "angles": list(ANGLES),
                  "families": list(FAMILIES), "lengths": list(LENGTHS), "samples": samples,
                  "warmups_per_family_length_per_worker": 1,
                  "cache_budget_bytes_per_worker": cache_budget_bytes,
                  "logical_requests": sum(r["logical_requests"] for r in runs),
                  "warmup_requests": sum(r["warmup_requests"] for r in runs),
                  "measured_requests": sum(r["measured_requests"] for r in runs),
                  "all_warmups_duplicates_and_cache_hits_charged": True,
                  "scalar_energy_cache_used": False, "correctness_overhead_requests": 0,
                  "total_memory_bytes": total_memory, "effective_memory_budget_bytes": memory_budget,
                  "runs": runs, "reference_solving_included": False, "model_training_started": False,
                  "raw_logs_namespace": relative}
        report["private_raw_log_sha256"] = {
            p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(folder.rglob("worker_*.jsonl"))}
        with (folder / "benchmark_summary.json").open("x", encoding="ascii") as handle:
            handle.write(canonical_bytes(report).decode("ascii") + "\n")
        return report
    except Exception as error:
        charged = sum(_charged_requests(p) for p in folder.glob("concurrency_*"))
        with (folder / "benchmark_failure.json").open("x", encoding="ascii") as handle:
            handle.write(canonical_bytes({"all_passed": False, "error_type": type(error).__name__,
                                           "logical_requests_charged": charged, "completed_runs": runs}).decode("ascii") + "\n")
        raise
