"""Read-only resource accounting from the completed, registered CO2 experiment.

This module imports no chemistry, model, accelerator, or training package.  It
does not infer missing work as zero, multiply shared costs by the number of
methods, or equate overlapping component time with elapsed time.
"""

from pathlib import Path
import math
import csv

from .common import digest, read_json, require, sha256_file, workspace_path


MATRIX = "large_qubit/main/section9_recovery_v1/matrix_0001"


def _number(value):
    require(isinstance(value, (float, int)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0, "Invalid cost measurement")
    return value


def summarize_components(lanes):
    """Sum disjoint trajectory costs; session generation is a cross-check only."""
    trajectories = {}
    totals = {"generation_seconds": 0., "scoring_seconds": 0.,
              "hamiltonian_action_seconds": 0., "physical_hamiltonian_actions": 0,
              "input_tokens": 0, "output_tokens": 0, "generation_attempts": 0,
              "logical_requests": 0, "wall_is_lower_bound": False,
              "numerical_work_is_lower_bound": False}
    setup = {"model_load_seconds": 0., "adapter_load_seconds": 0.,
             "physics_load_seconds": 0., "lane_process_seconds": 0.,
             "vqe_algebraic_seconds": 0., "vqe_algebraic_fits_completed": 0,
             "wall_is_lower_bound": False}
    winners = {"physical_hamiltonian_actions": 0, "wall_seconds": 0.,
               "wall_is_lower_bound": False}
    for lane_id, lane in sorted(lanes.items()):
        session_generation = 0.
        lane_generation = 0.
        lane_model_cost_unknown = False
        for job, cost in sorted(lane["main_by_trajectory"].items()):
            require(job not in trajectories, "Trajectory charged in multiple lanes")
            g, s = cost["generation"], cost["search"]
            row = {"job_id": job, "lane": lane_id,
                   "logical_requests": _number(s["records"]),
                   "generation_attempts": _number(g["records"]),
                   "generation_seconds": _number(g["recorded_wall_seconds"]),
                   "scoring_seconds": _number(s["recorded_wall_seconds"]),
                   "physical_hamiltonian_actions": _number(s["evaluator_calls"]),
                   "hamiltonian_action_seconds": _number(s["hamiltonian_action_seconds"]),
                   "input_tokens": _number(g["input_tokens"]),
                   "output_tokens": _number(g["output_tokens"]),
                   "wall_is_lower_bound": bool(g["wall_is_lower_bound"] or s["wall_is_lower_bound"]),
                   "numerical_work_is_lower_bound": bool(g["numerical_work_is_lower_bound"] or s["numerical_work_is_lower_bound"]),
                   "component_seconds_are_not_elapsed_wall": True}
            trajectories[job] = row
            lane_generation += row["generation_seconds"]
            for key in totals:
                if isinstance(totals[key], bool):
                    totals[key] |= row[key]
                else:
                    totals[key] += row[key]
        for attempt in lane["execution_attempts"]:
            setup["lane_process_seconds"] += _number(attempt.get("wall_seconds", 0))
            setup["wall_is_lower_bound"] |= attempt["status"] != "finished"
            setup["vqe_algebraic_seconds"] += _number(attempt["recorded_algebraic_candidate_preparation_seconds"])
            setup["vqe_algebraic_fits_completed"] += _number(attempt["completed_algebraic_coordinate_fits_including_replay"])
            setup["wall_is_lower_bound"] |= attempt["partial_algebraic_work_may_be_unknown"]
            for session in attempt["inference_sessions"]:
                c = session.get("cost")
                setup["wall_is_lower_bound"] |= session["unknown_unflushed_model_cost"]
                lane_model_cost_unknown |= session["unknown_unflushed_model_cost"]
                if c:
                    session_generation += _number(c["generation_wall_seconds"])
                    setup["model_load_seconds"] += _number(c["model_load_seconds"])
                    setup["adapter_load_seconds"] += _number(c["adapter_load_seconds"])
            for load in attempt["physics_worker_loads"]:
                setup["wall_is_lower_bound"] |= load["unknown_unflushed_setup_cost"]
                if load.get("completed"):
                    setup["physics_load_seconds"] += _number(load["completed"]["wall_seconds"])
        # A session records generation and the trajectory ledger repeats it.
        # Only the latter is charged.  Small floating-point reduction differences
        # are permitted, while extra unflushed work is explicitly a lower bound.
        if not lane_model_cost_unknown:
            require(math.isclose(session_generation, lane_generation, rel_tol=1e-8, abs_tol=1e-5),
                    "Session and trajectory generation accounting disagree")
        for winner in lane["winner_integrity_overhead"].values():
            winners["wall_is_lower_bound"] |= winner.get("uncertain_cost", False)
            if winner.get("result"):
                winners["physical_hamiltonian_actions"] += _number(winner["result"]["evaluator_calls"])
                winners["wall_seconds"] += _number(winner["result"]["wall_seconds"])
    return {"trajectories": trajectories, "components": totals,
            "setup_and_algebra": setup, "winner_verification": winners}


def summarize_training(fits, preflights):
    """Committed exposures describe the final adapter; actual work includes replay."""
    rows = {}
    for name, fit in sorted(fits.items()):
        c = fit["cost_summary"]
        b = preflights[name]["benchmark"]
        preflight_seconds = (_number(b["model_load_seconds"])
                             + _number(b["benchmark_wall_seconds_including_warmup_and_failed_candidates"]))
        require(c["additional_energy_requests"] == b["additional_energy_requests"] == 0,
                "Training unexpectedly scored new circuits")
        rows[name] = {"committed_optimizer_steps": c["committed_exposure"]["optimizer_steps"],
                      "committed_examples": c["committed_exposure"]["examples_seen"],
                      "actual_work": c["actual_totals"],
                      "replayed_or_uncommitted_work": c["uncommitted_or_replayed_work"],
                      "training_seconds": _number(c["training_wall_seconds"]),
                      "model_load_seconds": _number(c["model_load_seconds"]),
                      "attributable_seconds": _number(c["attributable_wall_seconds"]),
                      "assigned_device_hours": _number(c["assigned_device_hours"]),
                      "wall_is_lower_bound": c["wall_is_lower_bound"],
                      "peak_memory_is_lower_bound": c["peak_memory_is_lower_bound"],
                      "peak_allocated_bytes": c["peak_allocated_bytes"],
                      "peak_reserved_bytes": c["peak_reserved_bytes"],
                      "attempt_count": c["attempt_count"],
                      "attempts": c["attempts"],
                      "preflight_seconds_including_load": preflight_seconds,
                      "preflight_actual_work": b["disposable_actual_work"],
                      "preflight_failed_candidates": sum(not x["success"] for x in b["candidates"])}
        require(math.isclose(rows[name]["training_seconds"] + rows[name]["model_load_seconds"],
                             rows[name]["attributable_seconds"], abs_tol=1e-6),
                "Training load time double counted or missing")
    return {"fits": rows, "additional_energy_requests": 0,
            "attributable_seconds": sum(r["attributable_seconds"] for r in rows.values()),
            "assigned_device_hours": sum(r["assigned_device_hours"] for r in rows.values()),
            "wall_is_lower_bound": any(r["wall_is_lower_bound"] for r in rows.values()),
            "preflight_seconds_including_load": sum(r["preflight_seconds_including_load"] for r in rows.values()),
            "scope": "Sum of distinct process/device costs, not elapsed time of concurrent fits"}


def request_budget(pilot, acquisition, validation, main):
    """Count each attempt once, using the prospectively amended ceilings."""
    require(main["cumulative"] == main["failed_attempt"] + main["recovery_attempt"],
            "Failed main attempt omitted or charged twice")
    require(acquisition["cumulative"] == acquisition["failed_v1"] + acquisition["fresh_v2"],
            "Acquisition attempt omitted or charged twice")
    categories = [
        {"category": "training_only_pilot", "actual": pilot["cumulative_requests"], "ceiling": pilot["ceiling"]},
        {"category": "acquisition_including_rejected_v1", "actual": acquisition["cumulative"], "ceiling": acquisition["ceiling"]},
        {"category": "validation", "actual": validation["actual"], "ceiling": validation["ceiling"]},
        {"category": "main_including_failed_attempt", "actual": main["cumulative"], "ceiling": main["maximum_cumulative"]},
    ]
    for row in categories:
        require(0 <= row["actual"] <= row["ceiling"], "Ordinary allowance exceeded")
        row["unused_allowance"] = row["ceiling"] - row["actual"]
    return {"categories": categories, "actual": sum(r["actual"] for r in categories),
            "amended_ceiling": sum(r["ceiling"] for r in categories),
            "original_plan_ceiling": 26624,
            "acquisition_attempts": acquisition, "main_attempts": main,
            "overhead_included": False,
            "definition": "Durable logical circuit-energy requests, including duplicates, cache hits and failures; references and verification separate"}


def extract_costs(root):
    """Extract immutable resource evidence. Full predecessor verification is caller-owned."""
    root = Path(root).resolve()
    sources = {}

    def load(relative):
        path = workspace_path(root, relative)
        value = read_json(path)
        sources[relative] = {"path": relative, "sha256": sha256_file(path), "bytes": path.stat().st_size}
        if isinstance(value, dict) and "payload" in value and "sha256" in value:
            require(digest(value["payload"]) == value["sha256"], f"Cost envelope changed: {relative}")
            return value["payload"]
        return value

    completion = load(f"{MATRIX}/completed.json")
    require(completion["matrix_execution_complete"] and completion["all_trajectories_successful"],
            "Main results are not complete")
    cost_path = completion["costs_path"]
    costs = load(cost_path)
    require(completion["inventory"][cost_path] == sources[cost_path]["sha256"], "Completion does not bind costs")
    p4 = load("large_qubit/evidence/sections_1_4_v1/local_execution.json")
    p5 = load("large_qubit/evidence/sections_1_5_v1/local_execution.json")
    a1 = load("large_qubit/evidence/sections_1_6_v1/local_execution.json")
    a2 = load("large_qubit/evidence/sections_1_6_v2/local_execution.json")
    b4 = load("large_qubit/scoring_only/gqe-co2-sto3g-30q-v1/section4/section4_v1/benchmark/benchmark_summary.json")
    failed = costs["preserved_failed_main_attempt"]
    original = load(failed["source_costs_path"])
    require(sources[failed["source_costs_path"]]["sha256"] == failed["source_costs_sha256"], "Failed costs changed")
    require(original["lanes"] == failed["lanes"], "Preserved failed lane costs differ")
    recovery = summarize_components(costs["lanes"])
    prior_main = summarize_components(failed["lanes"])
    require(recovery["components"]["logical_requests"] == costs["ordinary_request_accounting"]["recovery_attempt"], "Recovery requests differ")
    require(prior_main["components"]["logical_requests"] == costs["ordinary_request_accounting"]["failed_attempt"], "Failed requests differ")
    training = summarize_training(costs["three_training_fits_charged_once"]["fits"],
                                  costs["training_preflight_overhead_charged_once"]["fits"])
    for name in training["fits"]:
        fit = load(f"large_qubit/training_runs/section7_gpu_uuid_v1/{name}/completed.json")
        preflight = load(f"large_qubit/preflights/section7_gpu_uuid_v1/{name}.json")
        require(fit["cost_summary"] == costs["three_training_fits_charged_once"]["fits"][name]["cost_summary"], "Training snapshot differs")
        require(preflight["benchmark"] == costs["training_preflight_overhead_charged_once"]["fits"][name]["benchmark"], "Preflight snapshot differs")
    validation = costs["all_validation_attempts"]
    load("large_qubit/freezes/section8_v1/freeze.json")
    physics_preparation = {}
    for version, filename in [("section3_v1", "failure.json"),
                              ("section3_v2", "failure.json"),
                              ("section3_v3", "manifest.json")]:
        relative = f"large_qubit/physics/{version}/{filename}"
        document = load(relative)
        physics_preparation[version] = {"wall_seconds": document["wall_seconds"],
                                        "all_passed": document["all_passed"],
                                        "source": relative,
                                        "cost_category": "SCF_integral_preparation_not_ordinary_energy_requests"}
    saved_state_replays = {}
    for version in ("sections_1_5_v1", "sections_1_6_v1", "sections_1_6_v2"):
        relative = f"large_qubit/evidence/{version}/native_state_verification.json"
        document = load(relative)
        saved_state_replays[version] = {"source": relative,
                                       "recorded_wall_seconds": document.get("wall_seconds"),
                                       "verification": document["verification"],
                                       "accounting": "These reports repeat original charged counts for authentication; only additional actions and state replays are new overhead"}
    acquisition = {"failed_v1": a1["ordinary_training_only_acquisition"]["issued"],
                   "fresh_v2": a2["ordinary_training_only_acquisition"]["issued"],
                   "cumulative": a2["cumulative_ordinary_requests"],
                   "ceiling": a2["maximum_cumulative_ordinary_requests"]}
    require(acquisition["cumulative"] == costs["source_acquisition_charged_once"]["logical_energy_requests"], "Acquisition snapshot differs")
    ordinary = request_budget(p5["ordinary_training_only_pilot"], acquisition,
                               {"actual": validation["total_validation_logical_requests"], "ceiling": 1024},
                               costs["ordinary_request_accounting"])
    refs = {}
    for target, item in costs["target_references_charged_once"].items():
        result = item["result"]
        refs[target] = {"wall_seconds": result["wall_seconds"],
                        "hamiltonian_actions": result["report"]["hamiltonian_actions"],
                        "hamiltonian_action_seconds": result["report"]["hamiltonian_action_seconds"],
                        "ordinary_energy_requests": result["ordinary_energy_requests"],
                        "peak_rss_bytes": result["peak_rss_bytes"],
                        "reused_by_recovery": True}
    p4times = [t for run in b4["runs"] for worker in run["worker_reports"] for t in worker["timings"]]
    phase_costs = [
        {"phase": "section4_throughput", "ordinary_requests": p4["ordinary_training_only_pilot"]["logical_energy_requests"],
         "recorded_seconds": sum(r["total_wall_seconds_including_init_and_warmup"] for r in b4["runs"]),
         "time_scope": "sum of sequential benchmark run walls including initialization and warmup; workers overlap within each run",
         "scoring_component_seconds": sum(t["score_seconds"] for t in p4times)},
        {"phase": "section5_expressivity_vqe", "ordinary_requests": p5["ordinary_training_only_pilot"]["section5_requests"],
         "recorded_seconds": p5["scientific_wall_seconds"], "time_scope": "scientific pilot wall"},
        {"phase": "section6_rejected_v1", "ordinary_requests": acquisition["failed_v1"],
         "recorded_seconds": a1["recorded_active_wall_seconds"], "time_scope": "recorded active session wall; retained NO-GO"},
        {"phase": "section6_fresh_v2", "ordinary_requests": acquisition["fresh_v2"],
         "recorded_seconds": a2["recorded_active_wall_seconds"], "time_scope": "recorded active session wall"},
        {"phase": "section7_preflights", "ordinary_requests": 0,
         "recorded_seconds": training["preflight_seconds_including_load"], "time_scope": "sum of GPU process measurements including load and failed candidates"},
        {"phase": "section7_fits", "ordinary_requests": 0,
         "recorded_seconds": training["attributable_seconds"], "wall_is_lower_bound": training["wall_is_lower_bound"],
         "time_scope": "sum of assigned-device process costs, including interrupted/replayed work and loads; concurrent fits"},
    ]
    for panel, data in sorted(validation["panels"].items()):
        phase_costs.append({"phase": f"section8_{panel}", "ordinary_requests": data["logical_requests"],
                            "recorded_seconds": sum(a.get("wall_seconds", 0) for a in data["execution_attempts"]),
                            "time_scope": "panel process wall; generation and scoring overlap",
                            "wall_is_lower_bound": any(a["status"] != "finished" for a in data["execution_attempts"])})
    for label, detail, accounting in [("failed", failed, prior_main), ("recovery", costs, recovery)]:
        seconds = sum(a.get("outcome", {}).get("wall_seconds", 0) for a in detail["coordinator_attempts"])
        phase_costs.append({"phase": f"section9_{label}", "ordinary_requests": accounting["components"]["logical_requests"],
                            "recorded_seconds": seconds, "time_scope": "coordinator elapsed wall across concurrent lanes; includes setup and overhead",
                            "wall_is_lower_bound": any(a["status"] != "finished" for a in detail["coordinator_attempts"])})
    # Never add embedded acquisition/training/validation again for each main lane.
    return {"schema_version": "section11-resource-analysis-v1", "ordinary_requests": ordinary,
            "phase_costs": phase_costs, "training": training,
            "main_trajectories": recovery["trajectories"],
            "main_recovery": recovery, "main_failed_attempt": prior_main,
            "validation": validation,
            "overhead": {"section3_physical_preparation": physics_preparation,
                         "saved_state_replays": saved_state_replays,
                         "section4": p4["separately_recorded_overhead"],
                         "section5": p5["separate_overhead"],
                         "acquisition_v1": a1["separate_label_integrity_overhead"],
                         "acquisition_v2": a2["separate_label_integrity_overhead"],
                         "target_references": refs,
                         "target_reference_actions": sum(r["hamiltonian_actions"] for r in refs.values()),
                         "target_reference_seconds": sum(r["wall_seconds"] for r in refs.values()),
                         "main_winner_verification": recovery["winner_verification"],
                         "validation_reference_and_winner_costs": {p: {k: v for k, v in d.items() if "reference" in k or "winner" in k} for p, d in validation["panels"].items()}},
            "historical_scope": {"pooled_with_CO2": False,
                                 "scope": "LiH pilot/development retained under its original protocol; see separate historical analysis and original phase7 artifacts",
                                 "original_guide": "phase7/README.md"},
            "sources": list(sources.values()),
            "caveats": [
                "No dollar, electricity, carbon or GPU-utilization total is inferred from elapsed seconds.",
                "Process/device seconds and overlapping generation/scoring components must not be added as elapsed project wall time.",
                "Shared acquisition, adapters, preflights, validation and two target references are each counted once, not once per method or recovery attempt.",
                "Weighted GPT-2 and Qwen training retain unclosed process attempts: wall, peak memory and actual numerical work can be lower bounds despite complete final adapters.",
                "Failed main search wall is a recorded lower bound; its physical actions and logical requests remain charged.",
                "Historical development, implementation time and uninstrumented correctness overhead are not silently treated as zero.",
                "Target reference matvecs are overhead and cannot be exchanged for ordinary circuit-energy requests.",
                "Main winner replays use fresh basis/evaluator/cache bypass but do not establish independence of the Hamiltonian backend.",
            ]}


def write_resource_appendix(root, costs):
    """Write a transparent cost appendix and overhead CSV from extracted evidence."""
    root = Path(root)
    out = root / "large_qubit/submission/section11_v1/resource_budget.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    def overhead(stage, actions, seconds, scope, detail=""):
        rows.append({"stage": stage, "ordinary_energy_requests": 0,
                     "hamiltonian_actions": actions, "recorded_seconds": seconds,
                     "time_scope": scope, "detail": detail})

    h = costs["overhead"]
    for key, value in h["section3_physical_preparation"].items():
        overhead(key, None, value["wall_seconds"], "SCF/integral preparation wall",
                 "passed" if value["all_passed"] else "preserved failed preparation")
    s4 = h["section4"]
    probe = s4["initial_HF_correctness_probe"]
    overhead("section4_initial_HF_probe", 1, probe["initialization_seconds"] + probe["first_energy_call_seconds"],
             "initialization plus first energy call", "Nested backend initialization is already included")
    probe = s4["thread_scaling_action_probe"]
    overhead("section4_thread_scaling", probe["hamiltonian_actions"],
             sum(x["action_seconds"] for x in probe["measurements"]), "sum of native action components at 1/2/4 threads")
    correctness = s4["CO2_correctness"]["statistics"]
    overhead("section4_CO2_correctness", correctness["apply_h_calls"], correctness["hamiltonian_action_seconds"],
             "native Hamiltonian action components", "Not complete correctness wall time")
    reference = s4["measured_reference"]["statistics"]
    overhead("section4_training_reference", reference["hamiltonian_actions"], reference["wall_seconds"],
             "training-geometry reference solver wall")
    check = s4["final_saved_reference_integrity_check"]
    overhead("section4_saved_reference_check", check["hamiltonian_actions"], check["wall_seconds"], "saved-reference replay wall")
    for run in s4["historical_correctness_passes"]:
        for case, value in run["cases"].items():
            reference = value["recorded_reference_overhead"]
            overhead(f"section4_{run['label']}_{case}_reference", reference["hamiltonian_actions"], reference["wall_seconds"],
                     "tiny-system reference solver wall", "New-branch correctness overhead; not historical LiH scientific search")
        overhead(f"section4_{run['label']}_circuit_fixtures", None, None, "complete action counter and wall unavailable",
                 f"{run['fixed_circuit_fixtures']} fixtures; do not infer one action per fixture")
    overhead("section5_saved_winner_integrity", h["section5"]["saved_winner_integrity_hamiltonian_actions"], None,
             "wall unavailable", "Not admitted to search")
    for name in ("acquisition_v1", "acquisition_v2"):
        overhead(name + "_label_integrity", h[name]["hamiltonian_action_intents_charged"], None,
                 "included in recorded acquisition active-session wall; isolated wall unavailable")
    for name, value in h["saved_state_replays"].items():
        verified = value["verification"]
        states = verified.get("numerical_states_replayed", verified.get("winning_state_preparations_replayed", 0))
        overhead(name + "_normalization_replay", verified["additional_hamiltonian_actions"], value["recorded_wall_seconds"],
                 "read-only authentication plus state replay wall where recorded", f"{states} state preparations; source counters are not charged again")
    for panel, value in h["validation_reference_and_winner_costs"].items():
        overhead(panel + "_reference", value["reference_hamiltonian_actions"], value["recorded_reference_seconds"],
                 "included within validation panel elapsed wall")
        overhead(panel + "_winner_integrity", value["winner_verification_hamiltonian_actions"], value["recorded_winner_verification_seconds"],
                 "included within validation panel elapsed wall")
    for target, value in h["target_references"].items():
        overhead(target + "_reference", value["hamiltonian_actions"], value["wall_seconds"],
                 "included within initial failed-matrix coordinator wall", "Reused by recovery without another solve")
    value = h["main_winner_verification"]
    overhead("main_winner_integrity", value["physical_hamiltonian_actions"], value["wall_seconds"],
             "included within recovery coordinator/lane walls")
    value = costs["main_recovery"]["setup_and_algebra"]
    overhead("main_VQE_algebra", 0, value["vqe_algebraic_seconds"], "coordinate-fit algebra component within recovery wall",
             f"{value['vqe_algebraic_fits_completed']} complete coordinate fits; their physical probes are already ordinary requests")
    for group in ("main_recovery", "main_failed_attempt"):
        setup = costs[group]["setup_and_algebra"]
        for field in ("model_load_seconds", "adapter_load_seconds", "physics_load_seconds"):
            overhead(group + "_" + field, 0, setup[field], "setup component already included in coordinator/lane walls")
    csv_path = root / "large_qubit/analysis/section11_v1/tables/overhead.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def table(headers, values):
        return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"] +
                         ["| " + " | ".join(str(v) for v in row) + " |" for row in values])

    budget = costs["ordinary_requests"]
    lines = ["# Section 11 resource budget and attribution", "",
             "Generated from the sealed saved experiment by `large_qubit.section11_costs.write_resource_appendix`. No model, Hamiltonian evaluation or training is run.", "",
             "## Ordinary requests", "",
             table(["Category", "Actual", "Ceiling", "Unused"],
                   [[r["category"], r["actual"], r["ceiling"], r["unused_allowance"]] for r in budget["categories"]]), "",
             f"**Total: {budget['actual']:,} / {budget['amended_ceiling']:,} ordinary logical requests.** The original plan allowed {budget['original_plan_ceiling']:,}. The corpus amendment retained 6,538 rejected v1 requests and permitted a fresh 8,192-request campaign (cumulative corpus ceiling 14,730). Main recovery retained 16 failed requests and allowed a fresh 16,384 (cumulative main ceiling 16,400). Neither rejected work nor failures are removed from total expenditure.", "",
             "The actual recovery count is 16,382: each of two VQE trajectories ends with one unusable request of allowance. Validation uses 1,019: its six-query VQE updates leave five units unused. Unused allowance is not work performed. Every cache hit and duplicate retains its logical charge.", "",
             "## Measured phase times", "",
             table(["Phase", "Requests", "Recorded seconds", "Hours", "Timing meaning"],
                   [[r["phase"], r["ordinary_requests"], (">= " if r.get("wall_is_lower_bound") else "") + f"{r['recorded_seconds']:.3f}",
                     f"{r['recorded_seconds']/3600:.6f}", r["time_scope"]] for r in costs["phase_costs"]]), "",
             "These rows have different scopes and are **not an all-in elapsed-time sum**. The three fit measurements add GPU-assigned process time even where fits overlap. Matrix coordinator wall spans four concurrent lanes. Generation/scoring, reference, setup and winner-check components occur inside those walls. Human implementation time, scheduling gaps and unrecorded tails are not measured by this table.", "",
             "## Per-method attribution", "",
             "The following is a dependency/allocation rule for a standalone pipeline using each method. The accepted corpus has 6,552 fresh requests; the actual study's cumulative acquisition source cost is 13,090 requests and 35,958.277 recorded active seconds, including rejected v1. In the aggregate study this shared source is charged **once**, not four times. Reporting it next to each consumer is a standalone cost perspective, not additive experimental expenditure.", ""]
    attribution = []
    methods = sorted({job.split("__", 1)[0] for job in costs["main_trajectories"]})
    for method in methods:
        uses_corpus = method in costs["training"]["fits"] or method == "training_circuit_retrieval"
        fit = costs["training"]["fits"].get(method)
        marginal = [row for job, row in costs["main_trajectories"].items() if job.split("__", 1)[0] == method]
        attribution.append([method, "13,090 shared; counted once globally" if uses_corpus else "No corpus dependency",
                            (">= " if fit["wall_is_lower_bound"] else "") + f"{fit['attributable_seconds']:.3f}" if fit else "No new fit",
                            f"{fit['preflight_seconds_including_load']:.3f}" if fit else "No fitting preflight",
                            sum(row["logical_requests"] for row in marginal),
                            f"{sum(row['generation_seconds'] for row in marginal):.3f}",
                            f"{sum(row['scoring_seconds'] for row in marginal):.3f}"])
    lines += [table(["Method", "Source requests", "Own fit seconds", "Own preflight seconds", "Main requests", "Generation seconds", "Scoring seconds"], attribution), "",
              "The three adapted methods additionally bear their own measured fit and fitting-preflight cost; retrieval does not. Frozen models, uniform random and VQE have no new neural fit. Frozen models still load pretrained weights and perform inference; upstream pretraining expenditure is outside this study and must not be called zero. Marginal model-generation and physical-scoring columns exclude shared corpus, training, validation and references.", "",
              "The training-only representation/throughput/VQE pilots, correctness fixtures, common validation panel, references and shared protocol setup are global study overhead. Target model loads and adapter switches are recorded per inference session/lane; because a resident base is shared across methods in its lane, no arbitrary per-method split is invented. The 16 failed HF requests belong to the initial matrix attempt, one per method-target cell, and remain separately charged. An end-to-end comparison must show these global/standalone dependencies alongside marginal target cost.", "",
              "## Numerical and setup overhead", "",
              "All rows below have zero **ordinary circuit-energy requests**. An unknown action counter or elapsed time is shown as `not recorded`, not zero. Hamiltonian actions, reference iterations, state preparations and optimizer algebra are different units. Some overhead is already within the phase walls above, as explicitly noted; it must not be added twice.", "",
              table(["Stage", "H actions", "Seconds", "Timing scope / qualification"],
                    [[r["stage"], r["hamiltonian_actions"] if r["hamiltonian_actions"] is not None else "not recorded",
                      f"{r['recorded_seconds']:.6f}" if r["recorded_seconds"] is not None else "not recorded",
                      r["time_scope"] + ("; " + r["detail"] if r["detail"] else "")] for r in rows]), "",
              "The correctness fixture count is preserved without manufacturing a total Hamiltonian-action count. Additional CPU unit tests and implementation/authentication work lack a single comprehensive instrumented expenditure ledger. Therefore no complete all-history elapsed time, numerical-action total, dollar cost, energy consumption or carbon estimate is claimed.", "",
              "## Lower bounds and parallelism", "",
              "Weighted GPT-2 and Qwen have complete final five-pass adapters but unclosed process cost tails. Their fit wall and peak memory remain lower bounds; actual work includes 87 and 29 replayed/uncommitted completed optimizer steps respectively, in addition to the 1,920 steps represented by each final adapter. Initial failed-main request wall is also a lower bound, although all 16 requests and physical actions are known.", "",
              "Recovery's 7.052 coordinator hours overlap 18.492 model-generation component-hours and 8.378 scoring component-hours. Its 23.455 lane-process hours include CPU-only baseline periods and idle/waiting time; they are not measured GPU-busy hours. Hardware utilization, contention and inference scaling are not inferred from these aggregates.", "",
              "## Historical evidence", "",
              "Historical LiH acquisition, pilots and development are reported separately in [literature_and_history.md](literature_and_history.md) and the unchanged Phase 5-7 records. They are not added to the CO2 ordinary allowance or used as independent CO2 training replications.", "",
              "## Source provenance", "",
              "Every input file and SHA-256 appears in `../../analysis/section11_v1/costs.json` under `sources`. The source completion binds the master cost record; its embedded source/training/preflight data are checked against their own saved files. `../../analysis/section11_v1/tables/overhead.csv` exports the overhead table without display rounding. The full read-only predecessor verification remains a prerequisite of the Section 11 build.", ""]
    out.write_text("\n".join(lines), encoding="utf-8")
    return {"appendix": out.relative_to(root).as_posix(), "overhead_csv": csv_path.relative_to(root).as_posix(), "overhead_rows": len(rows)}
