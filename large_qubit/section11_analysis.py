"""Descriptive analysis of the sealed, fixed two-target experiment. No science runs."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from .common import digest, read_json, require, sha256_file

MATRIX = "large_qubit/main/section9_recovery_v1/matrix_0001"
COMPLETION_SHA = "5ac3686076766b9802716dbe575110351c5b710933c17aa67b9fc5a697f77bff"
ANALYSIS = "large_qubit/analysis/section11_v1"
SUBMISSION = "large_qubit/submission/section11_v1"
TARGETS = ("CO2_I01", "CO2_I02")
METHODS = ("gpt2_xl_frozen", "gpt2_xl_weighted_lora", "gpt2_xl_uniform_lora",
           "qwen3_4b_frozen", "qwen3_4b_weighted_lora", "uniform_random",
           "training_circuit_retrieval", "conventional_vqe")
LABELS = dict(zip(METHODS, ("GPT-2 XL frozen", "GPT-2 XL weighted", "GPT-2 XL uniform",
                          "Qwen3-4B frozen", "Qwen3-4B weighted", "Uniform random",
                          "Training-circuit retrieval", "Compact VQE")))
COLORS = ("#8b9cad", "#167c80", "#63bcb2", "#b3a6c9", "#724a92", "#a48561", "#d28426", "#d14450")
PAIRS = {"gpt_weighted_minus_frozen": ("gpt2_xl_weighted_lora", "gpt2_xl_frozen"),
         "gpt_weighted_minus_uniform": ("gpt2_xl_weighted_lora", "gpt2_xl_uniform_lora"),
         "qwen_weighted_minus_frozen": ("qwen3_4b_weighted_lora", "qwen3_4b_frozen"),
         "qwen_weighted_minus_gpt_weighted": ("qwen3_4b_weighted_lora", "gpt2_xl_weighted_lora")}


def load_sealed(path):
    record = read_json(path)
    require(set(record) >= {"payload", "sha256"} and digest(record["payload"]) == record["sha256"],
            f"Changed envelope: {path}")
    return record["payload"]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def summarize(matrix, fitting_sequences=()):
    """Recompute all endpoints/means; refuse missing or failed confirmatory cells."""
    expected = {f"{m}__{t}" for m in METHODS for t in TARGETS}
    require(set(matrix["jobs"]) == expected and set(matrix["resources"]) == expected,
            "The analysis requires all sixteen trajectories and resource records")
    require(not matrix["failed_jobs"] and not matrix["missing_jobs"] and matrix["all_endpoints_verified"],
            "Incomplete outcomes require a separately scoped analysis")
    endpoints, curves, counts, resources = [], [], [], []
    fitting_sequences = set(fitting_sequences)
    for method in METHODS:
        for target in TARGETS:
            key = f"{method}__{target}"
            job, resource = matrix["jobs"][key], matrix["resources"][key]
            require(job["status"] == "complete" and job["incumbent_verified"], f"Unverified endpoint: {key}")
            error = job["best_verified_energy_hartree"] - job["reference_energy_hartree"]
            require(math.isclose(error, job["best_verified_error_hartree"], rel_tol=0, abs_tol=1e-12), "Endpoint arithmetic differs")
            curve = job["budget_curve"]
            require([x["logical_requests"] for x in curve] == list(range(1, 1025)), "Incomplete budget curve")
            require(all(a["best_energy_hartree"] >= b["best_energy_hartree"] - 1e-10
                        for a, b in zip(curve, curve[1:])), "Incumbent worsened")
            require(math.isclose(curve[-1]["best_energy_hartree"], job["best_verified_energy_hartree"], rel_tol=0, abs_tol=1e-8),
                    "Curve and verified endpoint disagree")
            cnt = job["counts"]
            require(sum(bool(x["actual_request"]) for x in curve) == cnt["logical_requests"], "Carried budget miscount")
            ops = job["best_query"].get("ops")
            endpoints.append({"job_id": key, "method": method, "target": target, "status": job["status"],
                              "energy_hartree": job["best_verified_energy_hartree"],
                              "hf_hartree": job["hf_energy_hartree"], "reference_hartree": job["reference_energy_hartree"],
                              "error_mha": error * 1000, "hf_improvement_mha": job["hf_improvement_hartree"] * 1000,
                              "best_request": job["best_request"], "actual_requests": cnt["logical_requests"],
                              "carried_allowance": job["unused_allowance"],
                              "winner_exactly_in_fitting_corpus": tuple(ops) in fitting_sequences if ops is not None else None})
            counts.append({"job_id": key, "method": method, "target": target, **cnt,
                           "duplicate_fraction_all_requests": cnt["duplicate_requests"] / cnt["logical_requests"]})
            for x in curve:
                curves.append({"method": method, "target": target, **x,
                               "error_mha": (x["best_energy_hartree"] - job["reference_energy_hartree"]) * 1000})
            resources.append({"job_id": key, "method": method, "target": target, "native_family": resource["native_family"],
                              **{f"excluding_hf_{k}": resource["resources_excluding_hf"][k]
                                 for k in ("depth", "two_qubit_count", "two_qubit_depth", "gates")},
                              **{f"including_hf_{k}": resource["resources_including_hf"][k]
                                 for k in ("depth", "two_qubit_count", "two_qubit_depth", "gates")}})
    means = []
    for method in METHODS:
        selected = [x for x in endpoints if x["method"] == method]
        mean = sum(x["error_mha"] for x in selected) / 2
        require(math.isclose(mean, matrix["methods"][method]["mean_verified_error_hartree"] * 1000, rel_tol=0, abs_tol=1e-9),
                "Fixed-target mean differs")
        means.append({"method": method, "I01_error_mha": selected[0]["error_mha"],
                      "I02_error_mha": selected[1]["error_mha"], "mean_error_mha": mean})
    paired = []
    lookup = {(r["method"], r["target"]): r for r in endpoints}
    require(set(matrix["paired_comparisons"]) == set(PAIRS), "Paired comparisons incomplete")
    for key, (left, right) in PAIRS.items():
        pair = matrix["paired_comparisons"][key]
        require((pair["left_method"], pair["right_method"]) == (left, right), "Paired controls differ")
        diffs = [lookup[pair["left_method"], t]["error_mha"] - lookup[pair["right_method"], t]["error_mha"] for t in TARGETS]
        require(math.isclose(sum(diffs) / 2, pair["mean_difference_hartree"] * 1000, rel_tol=0, abs_tol=1e-9), "Paired arithmetic differs")
        paired.append({"comparison": key, "left_method": pair["left_method"], "right_method": pair["right_method"],
                       "I01_difference_mha": diffs[0], "I02_difference_mha": diffs[1], "mean_difference_mha": sum(diffs) / 2})
    return {"endpoint_rows": endpoints, "method_rows": means, "paired_rows": paired,
            "count_rows": counts, "resource_rows": resources, "curve_rows": curves}


def extract_analysis(root):
    root = Path(root)
    completion = root / MATRIX / "completed.json"
    require(sha256_file(completion) == COMPLETION_SHA, "A different main completion requires a new analysis version")
    cp = load_sealed(completion)
    report_path = f"{MATRIX}/matrix_report.json"
    require(sha256_file(root / report_path) == cp["inventory"][report_path], "Main report changed")
    report = load_sealed(root / report_path)
    fitting_path = "large_qubit/corpora/co2_corpus_v2/fitting.jsonl"
    release_path = root / "large_qubit/releases/co2_corpus_v2/manifest.json"
    require(sha256_file(release_path) == "323fe5fa6d4830339aa9b14dc546364fe5d24765745f08c421e3ca562e2d3e2a", "Fitting release changed")
    release = read_json(release_path)
    declared = release["files"]["fitting"]
    require(declared["path"] == fitting_path and sha256_file(root / fitting_path) == declared["sha256"]
            and (root / fitting_path).stat().st_size == declared["bytes"], "Fitting corpus bytes changed")
    # The original matrix closure authenticates the complete release; this step
    # additionally records the exact file consumed by this post-hoc membership check.
    fitting = [json.loads(x) for x in (root / fitting_path).read_text(encoding="utf-8").splitlines()]
    require(len(fitting) == 6144 and all(x["role"] == "fitting" for x in fitting), "Fitting population differs")
    result = {**report, **summarize(report, [tuple(x["ops"]) for x in fitting]),
              "schema_version": "section11-fixed-target-analysis-v1",
              "scope": "Descriptive one-seed, two-fixed-target case study; no new labels, training, selection or independent replications",
              "source": {"completion_path": f"{MATRIX}/completed.json", "completion_sha256": COMPLETION_SHA,
                         "report_path": report_path, "report_sha256": sha256_file(root / report_path),
                         "fitting_path": fitting_path, "fitting_sha256": sha256_file(root / fitting_path)},
              "posthoc_membership_scope": "Exact 32-ID sequence membership across all fitting geometries; does not establish semantic novelty or causal generalization",
              "figures": {name: f"{ANALYSIS}/figures/{name}.png" for name in
                          ("endpoints", "budget_curves", "paired_effects", "duplicates_resources")}}
    result.pop("curve_rows")  # exported losslessly as CSV; original curves remain under jobs.
    return result


def write_csv(path, rows):
    rows = list(rows)
    require(rows, "Cannot export empty table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def export_tables(root, analysis, costs):
    dest = Path(root) / ANALYSIS / "tables"
    for name in ("endpoint_rows", "method_rows", "paired_rows", "count_rows", "resource_rows"):
        write_csv(dest / (name + ".csv"), analysis[name])
    write_csv(dest / "budget_curves.csv", ({"method": j["method"], "target": j["geometry_id"], **r,
             "error_mha": 1000 * (r["best_energy_hartree"] - j["reference_energy_hartree"])}
            for j in analysis["jobs"].values() for r in j["budget_curve"]))
    write_csv(dest / "ordinary_budget.csv", costs["ordinary_requests"]["categories"])
    write_csv(dest / "marginal_costs.csv", costs["main_trajectories"].values())
    write_csv(dest / "training_costs.csv", ({"fit": fit, **{key: r[key] for key in
              ("committed_optimizer_steps", "committed_examples", "training_seconds", "model_load_seconds",
               "assigned_device_hours", "wall_is_lower_bound", "peak_memory_is_lower_bound",
               "peak_allocated_bytes", "peak_reserved_bytes", "attempt_count", "preflight_seconds_including_load")}}
              for fit, r in costs["training"]["fits"].items()))
    keys = ("phase", "ordinary_requests", "recorded_seconds", "time_scope", "wall_is_lower_bound")
    write_csv(dest / "phase_costs.csv", ({k: r.get(k, False if k == "wall_is_lower_bound" else None) for k in keys} for r in costs["phase_costs"]))


def make_figures(root, analysis):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 200, "pdf.fonttype": 42, "svg.fonttype": "none"})
    dest = Path(root) / ANALYSIS / "figures"
    dest.mkdir(parents=True, exist_ok=True)
    def save(fig, name):
        for ext in ("png", "pdf", "svg"):
            fig.savefig(dest / f"{name}.{ext}", bbox_inches="tight", facecolor="white")
        plt.close(fig)
    y = np.arange(8)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), sharey=True, layout="constrained")
    for ax, field, title in zip(axes, ("I01_error_mha", "I02_error_mha", "mean_error_mha"), ("I01 · R = 1.10 Å", "I02 · R = 1.30 Å", "Fixed two-target mean")):
        ax.barh(y, [x[field] for x in analysis["method_rows"]], color=COLORS, height=.7)
        ax.set_title(title, fontsize=11); ax.set_xlim(0, 285); ax.set_xlabel("Verified error (mHa)")
        ax.set_yticks(y, [LABELS[m] for m in METHODS]); ax.grid(axis="x", alpha=.18); ax.set_axisbelow(True)
        for i, x in enumerate(analysis["method_rows"]): ax.text(x[field] + 3, i, f'{x[field]:.2f}', va="center", fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle("All eight methods · same-Hamiltonian reference at zero · lower is better", fontsize=12)
    save(fig, "endpoints")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), layout="constrained")
    for ax, target, title in zip(axes, TARGETS, ("I01 · R = 1.10 Å", "I02 · R = 1.30 Å")):
        for method, color in zip(METHODS, COLORS):
            j = analysis["jobs"][f"{method}__{target}"]
            ax.step(range(1, 1025), [(x["best_energy_hartree"] - j["reference_energy_hartree"]) * 1000 for x in j["budget_curve"]],
                    where="post", color=color, label=LABELS[method], lw=1.6, linestyle="--" if method.endswith("frozen") else "-")
        hf = (j["hf_energy_hartree"] - j["reference_energy_hartree"]) * 1000
        ax.axhline(hf, color="black", lw=.8, ls=":", label="HF")
        ax.axhline(0, color="black", lw=.8, label="Classical reference")
        ax.set(xlim=(1, 1024), ylim=(-5, 275), xlabel="Charged logical requests (HF included)", ylabel="Best-so-far error (mHa)", title=title)
        ax.grid(alpha=.15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, fontsize=8)
    save(fig, "budget_curves")
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.9), layout="constrained")
    for ax, indices, title in ((axes[0], [0, 2], "Adaptation relative to frozen"), (axes[1], [1, 3], "Small paired differences (expanded scale)")):
        for i, index in enumerate(indices):
            r = analysis["paired_rows"][index]
            for shift, field, label, color in ((-.19, "I01_difference_mha", "I01", "#167c80"), (.19, "I02_difference_mha", "I02", "#724a92")):
                ax.barh(i + shift, r[field], height=.34, color=color, label=label if i == 0 else None)
        ax.set_yticks(range(2), [analysis["paired_rows"][i]["comparison"].replace("_minus_", " − ").replace("_", " ") for i in indices], fontsize=8)
        ax.axvline(0, color="black", lw=.8); ax.set_title(title, fontsize=11); ax.set_xlabel("Paired error difference (mHa; negative favors left)")
        ax.legend(fontsize=8); ax.grid(axis="x", alpha=.15)
    save(fig, "paired_effects")
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True, layout="constrained")
    for t, shift, color in zip(TARGETS, (-.18, .18), ("#167c80", "#724a92")):
        cnt = [next(r for r in analysis["count_rows"] if r["method"] == m and r["target"] == t) for m in METHODS]
        res = [next(r for r in analysis["resource_rows"] if r["method"] == m and r["target"] == t) for m in METHODS]
        for ax, values in zip(axes, ([r["duplicate_fraction_all_requests"] * 100 for r in cnt],
                                      [r["excluding_hf_two_qubit_count"] for r in res], [r["excluding_hf_depth"] for r in res])):
            ax.barh(y + shift, values, height=.34, color=color, label=t.replace("CO2_", ""))
            ax.set_yticks(y, [LABELS[m] for m in METHODS]); ax.grid(axis="x", alpha=.15); ax.set_axisbelow(True)
    for ax, title in zip(axes, ("Duplicates / all requests (%)", "Winner CX gates", "Winner logical depth")):
        ax.set_title(title, fontsize=11); ax.legend(fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle("Exact-query duplicates; circuit counts exclude HF and hardware routing", fontsize=12)
    save(fig, "duplicates_resources")
