"""Build a separate long-form research report without changing the sealed submission."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import html
import io
import json
import math
import os
from pathlib import Path
import re
import sqlite3

from .common import canonical_bytes, digest, read_json, require, sha256_file
from .section11_analysis import METHODS, LABELS, TARGETS, COLORS, write_json, write_csv

OUT = "large_qubit/reports/comprehensive_v1"
ANALYSIS = "large_qubit/analysis/section11_v1"
TITLE = "Energy-Weighted Adaptation of Pretrained Language Models for Molecular Circuit Search"
SUBTITLE = "A comprehensive study of the historical LiH programme and a fixed-budget 30-qubit CO₂ benchmark"


def ledger_distributions(root, analysis):
    """Post-hoc summaries of charged proposals; never new physical evaluations."""
    import numpy as np
    records, source_records, populations = [], [], {}
    base = root / "large_qubit/main/section9_recovery_v1/matrix_0001/lanes"
    for method in METHODS:
        for target in TARGETS:
            key = f"{method}__{target}"
            paths = list(base.glob(f"*/{key}/ledger.sqlite"))
            require(len(paths) == 1, "Trajectory ledger is missing or ambiguous")
            path = paths[0]
            conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
            try:
                rows = conn.execute("SELECT ordinal,payload,previous_sha256,sha256 FROM events ORDER BY ordinal").fetchall()
            finally:
                conn.close()
            previous = "0" * 64
            reserved, completed = {}, {}
            for expected, (ordinal, raw, prior, sha) in enumerate(rows, 1):
                payload = json.loads(raw)
                require(expected == ordinal and prior == previous and canonical_bytes(payload).decode() == raw,
                        "Ledger canonical ordering changed")
                require(sha == digest({"ordinal": ordinal, "previous_sha256": prior, "payload": payload}), "Ledger hash changed")
                previous = sha
                if payload["event"] == "request_reserved":
                    require(payload["ordinal"] not in reserved, "Duplicate reservation ordinal")
                    reserved[payload["ordinal"]] = payload
                if payload["event"] == "request_completed":
                    require(payload["ordinal"] not in completed, "Duplicate completion ordinal")
                    completed[payload["ordinal"]] = payload
            job = analysis["jobs"][key]
            require(previous == job["ledger_head"], "Ledger is not the authenticated final head")
            require(len(reserved) == len(completed) == job["counts"]["logical_requests"], "Request population incomplete")
            require(set(reserved) == set(completed) == set(range(1, len(reserved) + 1)), "Request ordinals are not contiguous")
            require([i for i in reserved if reserved[i]["kind"] == "hf"] == [1], "Exactly one initial HF must be excluded")
            proposals = [completed[i]["energy_hartree"] for i in sorted(reserved) if reserved[i]["kind"] != "hf"]
            require(len(proposals) == job["counts"]["logical_requests"] - 1, "Initial HF exclusion differs")
            e = np.array(proposals)
            require(np.isfinite(e).all(), "Non-finite proposal energy")
            populations[key] = e
            err = 1000 * (e - job["reference_energy_hartree"])
            records.append({"method": method, "target": target, "proposal_requests": len(e),
                "mean_error_mha": float(err.mean()), "median_error_mha": float(np.median(err)),
                "minimum_error_mha": float(err.min()), "maximum_error_mha": float(err.max()),
                "p10_error_mha": float(np.quantile(err, .1)), "p90_error_mha": float(np.quantile(err, .9)),
                "strictly_below_hf": int(np.sum(e < job["hf_energy_hartree"])),
                "below_hf_minus_5mha": int(np.sum(e < job["hf_energy_hartree"] - .005)),
                "unique_proposal_queries": len({reserved[i]["query_sha256"] for i in reserved if reserved[i]["kind"] != "hf"}),
                "duplicates_retained": True, "initial_hf_excluded": True,
                "scope": "post-hoc empirical charged-proposal distribution; not independent seed observations"})
            source_records.append({"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path), "ledger_head": previous})
    return {"schema_version": "comprehensive-descriptive-proposal-analysis-v1", "rows": records, "sources": source_records}, populations


def make_figures(root, a, c, extra, populations):
    os.environ.setdefault("MPLCONFIGDIR", str(root / "large_qubit/.analysis_cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
        "axes.spines.right": False, "savefig.dpi": 240, "pdf.fonttype": 42, "svg.fonttype": "none"})
    dest = root / OUT / "figures"
    dest.mkdir(parents=True, exist_ok=True)
    def save(fig, name):
        for suffix in ("png", "pdf", "svg"):
            fig.savefig(dest / f"{name}.{suffix}", bbox_inches="tight", facecolor="white")
        plt.close(fig)
    # All sixteen trajectories individually: stable axes and no overplotting.
    for target in TARGETS:
        fig, axes = plt.subplots(4, 2, figsize=(10, 10), sharex=True, sharey=True, layout="constrained")
        for ax, method, color in zip(axes.flat, METHODS, COLORS):
            job = a["jobs"][f"{method}__{target}"]
            errors = [(r["best_energy_hartree"] - job["reference_energy_hartree"]) * 1000 for r in job["budget_curve"]]
            ax.step(range(1, 1025), errors, where="post", color=color, lw=1.8)
            ax.axhline((job["hf_energy_hartree"] - job["reference_energy_hartree"]) * 1000, color=".3", ls=":", lw=.8)
            ax.axhline(0, color=".3", lw=.7)
            ax.set_title(LABELS[method], fontsize=10)
            ax.set(xlim=(1, 1024), ylim=(-5, 275)); ax.grid(alpha=.13)
            ax.text(.98, .89, f'Final: {errors[-1]:.3f} mHa', transform=ax.transAxes, ha="right", fontsize=8)
        fig.supxlabel("Charged logical requests, including initial HF")
        fig.supylabel("Best verified-incumbent error relative to sector reference (mHa)")
        fig.suptitle(target + " · all eight search trajectories\nDotted: HF; zero: same-Hamiltonian classical reference", fontsize=12)
        save(fig, f"trajectories_{target}")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), layout="constrained")
    for ax, target in zip(axes, TARGETS):
        for method, color in zip(METHODS[:5], COLORS[:5]):
            job = a["jobs"][f"{method}__{target}"]
            x = np.sort((populations[f"{method}__{target}"] - job["reference_energy_hartree"]) * 1000)
            ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", label=LABELS[method], color=color)
        ax.set(xlabel="Proposal reference error (mHa; logarithmic axis)", ylabel="Empirical cumulative fraction", title=target, xscale="log")
        ax.grid(alpha=.15)
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=3, loc="outside lower center", fontsize=8)
    save(fig, "proposal_distributions")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    y = np.arange(8)
    gen = [sum(c["main_trajectories"][f"{m}__{t}"]["generation_seconds"] for t in TARGETS)/3600 for m in METHODS]
    score = [sum(c["main_trajectories"][f"{m}__{t}"]["scoring_seconds"] for t in TARGETS)/3600 for m in METHODS]
    axes[0].barh(y, gen, color="#177e89"); axes[1].barh(y, score, color="#ba7c26")
    for ax, title in zip(axes, ("Model generation component hours", "Physical scoring component hours")):
        ax.set_yticks(y, [LABELS[m] for m in METHODS]); ax.invert_yaxis(); ax.set_title(title, fontsize=10); ax.grid(axis="x", alpha=.15)
    fig.suptitle("Marginal cost over both targets · overlapping components, not elapsed time", fontsize=11)
    save(fig, "marginal_components")


def md_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"] +
                     ["| " + " | ".join(str(x) for x in row) + " |" for row in rows])


def write_results(root, a, c, extra):
    means = md_table(["Method", "I01 error (mHa)", "I02 error (mHa)", "Fixed-target mean"],
        [[LABELS[r["method"]], *(f'{r[k]:.6f}' for k in ("I01_error_mha", "I02_error_mha", "mean_error_mha"))] for r in a["method_rows"]])
    paired = md_table(["Paired comparison", "I01 difference", "I02 difference", "Mean difference"],
        [[r["comparison"].replace("_", " "), *(f'{r[k]:.6f}' for k in ("I01_difference_mha", "I02_difference_mha", "mean_difference_mha"))] for r in a["paired_rows"]])
    timing = md_table(["Method", "Best request I01 / I02", "HF gain I01 / I02 (mHa)", "Exact fitting match I01 / I02"],
        [[LABELS[m], " / ".join(str(a["jobs"][f"{m}__{t}"]["best_request"]) for t in TARGETS),
          " / ".join(f'{a["jobs"][f"{m}__{t}"]["hf_improvement_hartree"]*1000:.3f}' for t in TARGETS),
          " / ".join("N/A" if (v := next(r for r in a["endpoint_rows"] if r["method"] == m and r["target"] == t)["winner_exactly_in_fitting_corpus"]) is None else ("Yes" if v else "No") for t in TARGETS)] for m in METHODS])
    proposal = md_table(["Method", "I01 mean / median (mHa)", "I02 mean / median (mHa)", "Below HF − 5 mHa, I01 / I02"],
        [[LABELS[m], *(" / ".join(f'{next(r for r in extra["rows"] if r["method"] == m and r["target"] == t)[k]:.3f}' for k in ("mean_error_mha", "median_error_mha")) for t in TARGETS),
          " / ".join(str(next(r for r in extra["rows"] if r["method"] == m and r["target"] == t)["below_hf_minus_5mha"]) for t in TARGETS)] for m in METHODS])
    budget = md_table(["Ordinary work", "Actual", "Amended ceiling", "Unused"],
        [[r["category"].replace("_", " "), f'{r["actual"]:,}', f'{r["ceiling"]:,}', f'{r["unused_allowance"]:,}'] for r in c["ordinary_requests"]["categories"]] + [["Total", "31,452", "33,178", "1,726"]])
    phase = md_table(["Phase / attempt", "Recorded hours", "Scope"],
        [[r["phase"].replace("_", " "), ("≥ " if r.get("wall_is_lower_bound") else "") + f'{r["recorded_seconds"]/3600:.6f}', r["time_scope"]] for r in c["phase_costs"]])
    training = md_table(["Fit", "Assigned hours", "Peak allocated GiB", "Peak reserved GiB", "Qualification"],
        [[LABELS[k], f'{r["assigned_device_hours"]:.6f}', f'{r["peak_allocated_bytes"]/2**30:.3f}', f'{r["peak_reserved_bytes"]/2**30:.3f}',
          "Interrupted-tail lower bounds" if r["wall_is_lower_bound"] else "Recorded process"] for k, r in c["training"]["fits"].items()])
    text = f"""# 8. Results of the completed main matrix

## 8.1 Verified endpoints and the two-target mean

All sixteen recovery trajectories completed successfully and have verified winning incumbents. Fourteen use 1,024 ordinary requests; the two VQE trajectories use 1,023 and explicitly carry their final incumbents to the common allowance. The report includes all methods and both targets. There is no outcome-dependent exclusion, replacement of an unfavourable target or best-seed selection. The primary table reports error relative to the same-Hamiltonian classical sector reference in milli-Hartree, where 1 mHa = 0.001 Ha. Lower values are better. [A:large_qubit/main/section9_recovery_v1/matrix_0001/completed.json] [A:large_qubit/analysis/section11_v1/analysis.json]

{means}

**Table 8.1. Primary verified endpoints.** The last column is the unweighted arithmetic mean of the two preceding fixed-target values. It is not an estimate of average performance over a molecular population. Absolute energies and complete winning-request/resource details are retained in Appendix B.

![Figure 8.1. Verified best-of-budget reference errors for every method and target. The mean is descriptive over these two targets; there are no uncertainty bars because independent training-seed replication was not performed.](large_qubit/analysis/section11_v1/figures/endpoints.png)

Adaptation produces the dominant improvement. At I01, weighted GPT-2 reduces error from 177.838 to 87.488 mHa; weighted Qwen reduces it from the HF error of 190.464 to 86.633 mHa. At I02, the corresponding reductions are 245.360 to 123.765 mHa and 260.708 to 123.765 mHa. Both learned weighted systems therefore find substantially better circuits than their frozen counterparts under the stated allowance.

The strongest non-neural controls change the interpretation. Compact VQE gives the best I01 endpoint, 85.178 mHa, while retrieval gives the best I02 endpoint, 123.582 mHa. Retrieval also beats weighted GPT-2 at both targets: by approximately 0.232 mHa and 0.183 mHa. Qwen weighted has a lower two-target mean than retrieval by only 0.220 mHa, with opposite per-target orderings. Its mean lead is numerically resolved, but its scientific reliability across seeds is unmeasured and its computational cost is greater.

Uniform random search performs poorly relative to the adapted methods and retrieval, but beats frozen Qwen on both targets and frozen GPT-2 at I02. Thus language pretraining by itself is not a reliable molecular search strategy in this representation. The frozen models remain necessary negative controls rather than unsuccessful candidates to omit from the presentation.

## 8.2 Paired adaptation and weighting comparisons

{paired}

**Table 8.2. Paired differences in mHa.** Each value is the left method's verified error minus the right method's error on the same target. A negative value favours the left method. The mean averages those two paired differences, without resampling generated circuits.

The GPT adaptation effect averages −105.973 mHa, whereas the matched weighting effect averages only −0.652 mHa. At I01, the weighted/uniform difference is −0.005991 mHa, or approximately −5.99 × 10⁻⁶ Ha. At I02 it is −1.298156 mHa. These values are above the declared numerical agreement tolerances, but numerical resolution should not be confused with statistical replication or practical importance. A reproducible arithmetic difference in this completed run does not establish that another training seed will preserve the ordering.

The compact validation panel adds context without becoming a third test target: uniform GPT-2 had a lower validation endpoint error than weighted GPT-2, 114.807 versus 115.640 mHa. That result was retained without changing the final checkpoint or temperature. It is consistent with a small, geometry-dependent weighting increment rather than a uniform dominance claim. The main table remains the prespecified two-target result, and validation is not pooled into its mean. [A:large_qubit/freezes/section8_v1/freeze.json]

![Figure 8.2. Large adaptation effects and much smaller paired weighting/backbone differences on separate horizontal scales. Negative differences favour the method on the left of each comparison.](large_qubit/analysis/section11_v1/figures/paired_effects.png)

Qwen weighted has a 0.854532 mHa advantage over weighted GPT-2 at I01 and an exact tie at I02, yielding a 0.427266 mHa mean difference. The I02 equality reflects the same winning operator string, not a coarse rounding artefact. A claim about architecture or model size would need controls beyond this comparison, because tokenizer, pretraining and adaptation interfaces also differ.

## 8.3 Search trajectories, early incumbents and late improvements

![Figure 8.3. All sixteen best-so-far trajectories, grouped by target. The horizontal axis counts charged logical requests, including HF, cache hits and duplicates. The VQE endpoint at 1,024 is a carried value. Individual method panels appear in Appendix A.](large_qubit/analysis/section11_v1/figures/budget_curves.png)

{timing}

**Table 8.3. Endpoint provenance.** The winning request is the first recorded request associated with the final incumbent. Exact fitting-string membership is a post-hoc audit over all accepted fitting geometries; it does not establish semantic novelty or causal memorization. HF and continuous VQE states are inapplicable to the discrete-string test.

Retrieval finds its final I01 incumbent at request 49 and its I02 incumbent at request 5. Weighted Qwen reaches its final incumbents at requests 387 and 224, while weighted GPT-2 reaches them at 885 and 614. This indicates that direct corpus reuse supplied strong candidates early in these two fixed orderings. It does not retrospectively authorize stopping retrieval at those requests: their status as final winners is known only after the full trajectory. The completed budgets and costs remain charged.

VQE improves until request 1,023 on both targets. Its final point therefore conveys a different search pattern from an early plateau, without proving that continued optimization would close the remaining reference gap. Frozen Qwen never improves on the initial HF incumbent. Random search and frozen GPT-2 find only modest HF improvements, and their best-so-far curves hide the poor energies of many other proposals. The next subsection examines that distinction using the full saved request populations.

## 8.4 Additional descriptive analysis of proposal quality

For this expanded report, all sixteen final SQLite event chains were read without modification. Every event's canonical bytes, predecessor hash and hash value were checked, and each final chain head was matched to the authenticated main report. The analysis collects completed proposal energies after excluding only the initial HF request. Duplicates and cached results remain in the empirical population, because they represent actual probability mass or actual optimizer work. The denominators are 1,023 proposals per method-target cell, except VQE's 1,022 post-HF requests. These summaries are additional post-hoc descriptions, not newly registered primary endpoints. [A:large_qubit/reports/comprehensive_v1/proposal_analysis.json]

{proposal}

**Table 8.4. Empirical proposal errors and useful-proposal counts.** Means and medians refer to each charged proposal's error, rather than the incumbent after that proposal. The 5 mHa HF-relative cutoff is a descriptive usefulness threshold, not chemical accuracy. The observations are not independent training seeds. VQE's large-angle fitting probes and retrieval's source-energy ordering are part of different search mechanisms, so their means are not a common-distribution performance estimand.

Both weighted models produce 1,023 of 1,023 proposals below HF − 5 mHa at each target. Uniform GPT-2 produces 940 such proposals at I01 and 932 at I02. GPT weighting reduces empirical mean proposal error from 113.411 to 94.408 mHa at I01 and from 158.885 to 133.188 mHa at I02. This is a considerably larger distributional shift than its best-of-budget increment. Weighted fitting can therefore concentrate probability on useful circuits even when a uniformly fitted generator also eventually samples a similarly good endpoint. The result answers a different question from which method finds the lowest minimum.

Frozen GPT-2 has median errors near 205.558 and 273.810 mHa, but means near 2,594.581 and 2,674.080 mHa. Its heavy high-energy tail explains that disparity. Frozen Qwen repeats one unique proposal string at each target; the proposal errors are approximately 44,619.429 and 43,685.111 mHa, so HF remains much better than every generated candidate. These high proposal energies do not contradict its primary endpoints, because the incumbent includes the initial HF state. All such values come from retained physical evaluations or authenticated cache reuse, rather than imputed penalties for invalid text.

![Figure 8.4. Empirical cumulative distributions of the five language-model proposal populations. The horizontal error axis is logarithmic to retain high-energy frozen-model tails while showing the adapted populations. Initial HF is excluded and duplicate proposals retain their multiplicity.](large_qubit/reports/comprehensive_v1/figures/proposal_distributions.png)

The observed improvement in proposal quality suggests a possible use when only a few proposals can be afforded, but this report does not select a new favourable stopping budget or claim a replicated low-budget advantage. A future study could preregister several budgets and independent seeds, then evaluate whether the weighting effect is robust across both typical proposals and minima. Here, the registered B = 1,024 endpoint remains primary.

## 8.5 Validity, duplicates, cache reuse and circuit resources

All 10,230 model completion attempts in recovery are valid and complete, with zero invalid or infrastructure-failed completions. This confirms the executed grammar and accounting path, not physical usefulness. Frozen Qwen demonstrates the distinction directly: valid syntax coexists with one repeated, high-energy proposal and no HF improvement. There are zero failed, abandoned or pending energy requests in recovery; the sixteen failed HF requests of the predecessor remain separately charged.

Within-trajectory duplicates and scalar-cache hits are different counts. A query may be new to a trajectory but already available from another method in the same target/lane partition. Conversely, both measures retain the logical charge. Across recovery, 16,382 requests comprise 9,882 physical Hamiltonian actions and 6,500 hits. The sums describe numerical reuse under the chosen lane execution order, not free additional search opportunities. They would not necessarily remain identical under a different cache-sharing schedule, even with unchanged logical proposals. [A:large_qubit/analysis/section11_v1/costs.json]

![Figure 8.5. Within-trajectory duplicate fractions and common logical resources of verified winners. Resource counts exclude HF preparation and routing. Frozen Qwen's zero-extra-gate winner is HF, rather than an accurate correlated state.](large_qubit/analysis/section11_v1/figures/duplicates_resources.png)

Common resources are derived from exact commuting Pauli-rotation decompositions into H, X, RX, RZ and CX gates. The convention assumes logical all-to-all connectivity, omits identity/zero rotations and does not perform cross-gate cancellation or device-specific optimization. Including HF adds 22 X gates and one initial layer. Circuit-family differences remain explicit: generated strings use whole-qubit excitations, while the compact VQE is fermionic. At I02 the shared weighted GPT/Qwen winner has 1,536 CX gates and depth 1,333; VQE has 1,792 CX gates and depth 1,853. Logical resource estimates are informative but cannot establish noisy-device execution time or advantage.

# 9. Resource accounting, implementation efficiency and reproducibility

## 9.1 Ordinary allowances, amendments and unsuccessful work

{budget}

**Table 9.1. Complete new-branch ordinary request accounting.** A logical circuit-energy request remains charged for a duplicate, cache hit, failed evaluation or retry according to the frozen ledger rules. Reference solving, independent correctness fixtures and winner verification are separately recorded overhead, not interchangeable units of search allowance. [A:large_qubit/analysis/section11_v1/costs.json]

The original plan allowed 26,624 ordinary requests. The fresh-corpus amendment retained the rejected campaign's 6,538 requests and allowed up to 8,192 additional requests, producing a cumulative acquisition ceiling of 14,730. The executed fresh campaign used 6,552 requests. Main recovery retained sixteen failed first-attempt requests and permitted a fresh maximum of 16,384, giving a main ceiling of 16,400. Actual main work is sixteen plus 16,382, or 16,398. The resulting aggregate ceiling is 33,178 and actual work is 31,452. The increase is disclosed through preserved amendments rather than being disguised as compliance with the original smaller allowance.

Validation's 1,019 requests and the main VQE counts below their nominal caps follow the complete six-query update rule. Unused allowance is not work performed. Manufacturing extra rows to make all table totals equal their ceilings would corrupt accounting. Likewise, removing an unsuccessful acquisition or infrastructure attempt would understate the research expenditure that produced the final successful artifacts.

## 9.2 Measured elapsed time versus accumulated component time

{phase}

**Table 9.2. Measured phase scopes.** The rows are not directly additive as total elapsed time. Some measure concurrent device processes, others an entire controller, and several include setup or verification components that are separately itemized. Missing process tails remain lower bounds. [A:large_qubit/submission/section11_v1/resource_budget.md]

The completed four-lane recovery took 25,387.444 seconds, or 7.052 hours of coordinator elapsed time. Generation accumulated 66,569.684 seconds (18.492 component-hours) and scoring accumulated 30,159.386 seconds (8.378 component-hours). Both kinds of work overlap across lanes, and generation/scoring can also overlap inside the pipeline. Adding their sums would not reproduce elapsed time. The 23.455 accumulated lane-process hours include waiting and CPU-only baseline periods; they must not be described as measured GPU-busy hours.

Recovery records 5,685,834 model input tokens and 816,409 output tokens. Token counts differ across tokenizers and do not equal numbers of circuits. Main setup includes 49.105 seconds of model loads, 13.947 seconds of adapter loads and 21.463 seconds of physics-worker loads, all already within the corresponding execution walls. The 340 algebraic VQE coordinate fits take 0.140457 recorded seconds in addition to their charged physical probes. None of these components is a reason to add another allowance of energy calls.

![Figure 9.1. Per-method model-generation and physical-scoring component hours, summed across the two targets. Components overlap in the executed schedule. Shared acquisition, fitting, validation and references are excluded from this marginal view.](large_qubit/reports/comprehensive_v1/figures/marginal_components.png)

## 9.3 Fitting costs, memory and interrupted execution

{training}

**Table 9.3. Recorded fitting cost and allocator peaks.** All final adapters represent the same 1,920 committed optimizer steps. GPT weighted and Qwen incurred replayed/uncommitted work after interrupted attempts; their unresolved process tails make time and peak-memory evidence lower bounds. Allocator allocated/reserved peaks are not complete host/process memory or exclusive-device utilization measurements.

The three fits consumed at least 7.390783 assigned-device hours, including attributable model loading and recorded replay. GPU preflight adds 590.432 seconds, including model load and disposable benchmark work. These costs are separate from the fixed committed exposure in the final adapters. Storage-quota interruptions and partial transfers can therefore coexist with correctly resumed, complete scientific exposure; neither condition justifies erasing the failed work or restarting from an untracked fresh adapter. [A:large_qubit/analysis/section11_v1/costs.json]

No monetary cost, electricity consumption, carbon estimate or comprehensive human implementation time is available from these measurements. Hardware sharing and CPU contention affect observed performance, but process timestamps alone do not quantify their causal contribution. The measured recovery rate of roughly 0.645 logical requests per elapsed second describes this particular four-lane job, including its cache mix and method mix; it is not a universal scaling factor for additional GPUs.

## 9.4 Shared offline costs and complete-method comparisons

The accepted fitting source costs 6,552 fresh acquisition requests; the actual source-acquisition history costs 13,090 requests including rejected v1 and 35,958.277 recorded active seconds. Aggregate research expenditure counts that shared source once. A standalone adapted generator or retrieval pipeline nevertheless depends on acquiring the source; it cannot claim a cost-free corpus merely because another method in the study reused the same bytes.

Each learned pipeline additionally depends on its own fitting and preflight costs. Retrieval does not require a neural fit. Frozen models have no new adaptation cost but still require pretrained weights, model loading and generation; upstream language-pretraining expenditure is outside the study, not zero. VQE uses the training-only pilot and initialization policy, with no dependence on the accepted production fitting release. Random search uses the frozen discrete representation. The representation pilot, numerical correctness, compact validation, references and protocol preparation are common research overhead. The standalone and aggregate views are distinct allocation questions, so a single unqualified “total hours per method” would conceal more than it explains.

The two target references consumed 122 Hamiltonian actions and 347.052 seconds, solved during the initial main attempt and reused by recovery. Sixteen fresh winner checks add 49.787 seconds within recovery. Earlier correctness fixtures, corpus label checks, physics-construction failures and training-only reference work are listed separately in Appendix C. Some historical checks have fixture counts without a complete action or timing counter; those measurements are unknown rather than silently zero.

## 9.5 Applied speedups and their actual limits

The completed system avoids several unnecessary costs: one physical corpus feeds all consumers, fitting tokens and grammar choices are cached, pretrained weights remain local, base models remain resident across compatible methods, and optimizer/RNG/frontier checkpoints permit continuation. Immutable integrals and determinant indexing are reused, transition-map caches are bounded, and scalar energy reuse does not refund logical requests. Durable ordered queues separate model production from CPU evaluation while retaining deterministic admission and complete accounting. [A:large_qubit/training_data.py] [A:large_qubit/section8_inference.py] [A:large_qubit/section9_ledger.py]

Training batches were measured and frozen, but final inference remained at batch size one with eager attention. Four lanes parallelized independent target/backbone work; they did not demonstrate larger inference batches or both targets sharing one resident instance. CPU evaluators did not automatically become GPU kernels when more H200s were assigned. These implementation facts explain both achieved efficiency and remaining limitations without claiming an unmeasured speedup.

## 9.6 Authentication and the reproducibility boundary

Final scientific records bind physical fingerprints, catalogue, model revisions, adapters, release, prompts, numerical settings, random streams and request ledgers. Resume validates payload hashes and the expected frontier rather than trusting a directory name or a mutable status message. The completed matrix has sixteen successful trajectories; the Section 11 package was reauthenticated before this expanded report was assembled, retaining its 156 source and 69 output identities. [A:large_qubit/registrations/section11_analysis_v1/manifest.json]

This document is a new explanatory artifact outside that sealed package. Its proposal-distribution analysis checks existing ledger chains and performs arithmetic over stored energies; it trains no model, generates no circuit and executes no Hamiltonian action. A provenance catalogue binds cited local evidence, and source Markdown plus rendering code permit regeneration. Reproducing the saved-result analysis is a local CPU task. Repeating the original scientific experiment would additionally require the registered numerical and model environments, retained payloads and researcher-controlled execution; a PDF alone is not a complete computational archive.
"""
    (root / OUT / "parts/04_results_costs.md").write_text(text, encoding="utf-8")


def write_appendices(root, a, c, extra):
    endpoints = md_table(["Method / target", "Verified energy (Ha)", "Error (mHa)", "HF gain (mHa)", "Requests"],
        [[LABELS[r["method"]] + " / " + r["target"].replace("CO2_", ""), f'{r["energy_hartree"]:.10f}', f'{r["error_mha"]:.6f}', f'{r["hf_improvement_mha"]:.6f}', r["actual_requests"]] for r in a["endpoint_rows"]])
    resources = md_table(["Method / target", "Native family", "CX", "Depth", "Two-qubit depth", "All gates incl. HF"],
        [[LABELS[r["method"]] + " / " + r["target"].replace("CO2_", ""), r["native_family"], r["excluding_hf_two_qubit_count"],
          r["excluding_hf_depth"], r["excluding_hf_two_qubit_depth"], r["including_hf_gates"]] for r in a["resource_rows"]])
    counts = md_table(["Method / target", "Completion attempts", "Duplicates", "Cache hits", "Unique proposals"],
        [[LABELS[r["method"]] + " / " + r["target"].replace("CO2_", ""), r["completion_attempts"], r["duplicate_requests"], r["cache_hits"],
          next(x["unique_proposal_queries"] for x in extra["rows"] if x["method"] == r["method"] and x["target"] == r["target"])] for r in a["count_rows"]])
    with (root / ANALYSIS / "tables/overhead.csv").open(encoding="utf-8", newline="") as f: overhead_rows = list(csv.DictReader(f))
    # Use the already-authenticated readable overhead table; preserve exact counter qualifications.
    budget_text = (root / "large_qubit/submission/section11_v1/resource_budget.md").read_text(encoding="utf-8")
    overhead = budget_text.split("## Numerical and setup overhead\n\n", 1)[1].split("## Lower bounds and parallelism", 1)[0]
    text = f"""# Appendix A. All sixteen individual trajectories

The following figures separate the eight methods at each target so that overlapping curves in the main overview cannot conceal an outcome. The same vertical range is used throughout. HF is dotted, the reference is zero and every step uses charged requests. Each VQE final point at budget 1,024 is carried from request 1,023. These curves contain no seed uncertainty estimate. [A:large_qubit/analysis/section11_v1/tables/budget_curves.csv]

![Figure A.1. CO2_I01: all eight best-so-far trajectories, with individual method panels. Final errors are labelled in mHa.](large_qubit/reports/comprehensive_v1/figures/trajectories_CO2_I01.png)

![Figure A.2. CO2_I02: all eight best-so-far trajectories under the same plotting conventions.](large_qubit/reports/comprehensive_v1/figures/trajectories_CO2_I02.png)

# Appendix B. Complete numerical endpoint and resource tables

## B.1 Absolute endpoint energies

{endpoints}

All energies include the same geometry-specific nuclear contribution. I01 uses HF −185.02644274960136 Ha and reference −185.21690687388843 Ha; I02 uses HF −185.0244454651018 Ha and reference −185.2851539240048 Ha. Display rounding does not alter the full-precision CSV/JSON record. [A:large_qubit/analysis/section11_v1/tables/endpoint_rows.csv]

## B.2 Logical resources of the verified winners

{resources}

CX and depth columns exclude HF preparation. The last column includes its 22 X gates. These are logical all-to-all counts under the frozen decomposition; there is no target-device routing, error model or cross-gate optimization. `hf_only` denotes the unchanged HF incumbent. [A:large_qubit/analysis/section11_v1/tables/resource_rows.csv]

## B.3 Request populations, duplicate queries and scalar-cache hits

{counts}

The initial HF request is included in duplicate/cache accounting but excluded from unique proposal strings and the proposal-distribution summaries. Completion-attempt counts apply only to model generators; baseline energy requests are not text completions. All sixteen final trajectories have zero failed, abandoned or pending requests. The preserved predecessor has sixteen failed requests, separately charged. [A:large_qubit/analysis/section11_v1/tables/count_rows.csv] [A:large_qubit/reports/comprehensive_v1/proposal_summary.csv]

# Appendix C. Numerical verification and setup overhead

{overhead}

The table reproduces the verified Section 11 accounting breakdown. Full numeric values and units are retained in the overhead CSV; the displayed seconds are rounded. State replays that authenticate already-scored labels are distinguished from new Hamiltonian actions. All predecessor failures and unknown counter scopes remain visible. [A:large_qubit/analysis/section11_v1/tables/overhead.csv] [A:large_qubit/submission/section11_v1/resource_budget.md]

# Appendix D. Reproducibility, protocol identity and artifact access

## D.1 What constitutes the preserved experiment

The core chain is the physical definition and realized integral records; the screened catalogue and training-only pilots; both acquisition attempts and the accepted v2 release; all three fitted adapters and numerical freezes; eight compact-validation records; the final scientific freeze; the failed main predecessor; and the successful sixteen-trajectory recovery. Source archives retain the implementation used at each registered stage. Later documentation cannot retroactively change the meaning of those records. [A:large_qubit/submission/section11_v1/artifact_index.json]

The main completion file has SHA-256 `5ac3686076766b9802716dbe575110351c5b710933c17aa67b9fc5a697f77bff`. The final Section 8 freeze is `0c63273820b12eacab392e1cd38cdd8d99d8943b5b20c0e8f61c09c7cf685396`. The accepted corpus release is `323fe5fa6d4830339aa9b14dc546364fe5d24765745f08c421e3ca562e2d3e2a`. Full source and output inventories are available in the corresponding manifests, rather than inferred from these three anchor hashes alone.

Historical LiH data, trained checkpoints and development outcomes remain under their original benchmark identity. A new calculation that happens to give a similar energy would not recover a missing original artifact or authenticate an old hash. Likewise, a matched filename is insufficient evidence of a successful synchronization: the retained manifests and complete payload bytes determine identity.

## D.2 Local saved-result verification

From the preserved project root, the CPU-only Section 11 verifier authenticates its source/output bindings and recomputes the derived tables from saved input records. The optional deep form additionally invokes the original matrix closure, following predecessors and ledgers. Neither command loads a model or evaluates a new circuit. The project-local Python environment contains the required analysis packages.

```bash
# Linux: verify the completed submission package.
./.venv/bin/python -B -m large_qubit.section11 verify
# Optional full predecessor closure; can take substantially longer.
./.venv/bin/python -B -m large_qubit.section11 verify --deep
```

```powershell
# Windows PowerShell: the same saved-result verification.
.\\.venv\\Scripts\\python.exe -B -m large_qubit.section11 verify
```

The unrestricted report renderer is separate from that seal. Its `analyze` action reads existing event chains and writes the additional descriptive summaries; `render` reads the manuscript parts and figures to produce this PDF. The new renderer does not introduce a training or scoring entry point. Its source, Markdown, proposal-summary CSV, figures and provenance accompany the PDF under `large_qubit/reports/comprehensive_v1/`.

```bash
./.venv/bin/python -B -m large_qubit.comprehensive_report analyze
./.venv/bin/python -B -m large_qubit.comprehensive_report render
./.venv/bin/python -B -m large_qubit.comprehensive_report verify
```

The recorded analysis environment is Python 3.12.0, NumPy 2.5.4, Matplotlib 3.11.2, PyMuPDF 1.28.2 and Pillow 13.0.0. These are the report environment, not a substitute for the training/chemistry environment locks. Scientific reruns additionally require their pinned numerical packages, cached base models, physics payloads and explicit execution workflow. [A:large_qubit/analysis/section11_v1/environment.json] [A:large_qubit/submission/section11_v1/REPRODUCE.md]

## D.3 Completed outputs and remaining scientific questions

All required CO₂ experiments in the revised matrix are complete. The report, cost tables, figures, editable presentation and recording script were prepared from that evidence. Actual presentation recording and course submission are separate human actions, not scientific results. Historical LiH convergence remains unresolved under its original criterion, and the optional 40-qubit extension was not performed. Additional seeds, a Qwen-uniform control, matched-family optimization, learned-generator warm starts and broader molecules are prospective studies rather than gaps silently filled by extra computation during reporting.

The following evidence catalogue uses portable project-relative paths. Full hashes and byte counts are in this report's provenance file. The existing ten-page report and submission package remain unchanged; this document expands explanation and saved-data analysis without rewriting their registrations.
"""
    (root / OUT / "parts/06_appendices.md").write_text(text, encoding="utf-8")


CSS = """
body { font-family: serif; font-size: 11pt; line-height: 1.35; color: #172533; margin: 0; }
p { margin: 0 0 9pt; text-align: justify; }
h1 { font-family: sans-serif; font-size: 17pt; line-height: 1.2; color: #13324a; margin: 22pt 0 12pt; page-break-after: avoid; }
h2 { font-family: sans-serif; font-size: 12pt; line-height: 1.2; color: #155f6d; margin: 15pt 0 7pt; page-break-after: avoid; }
h3 { font-family: sans-serif; font-size: 11pt; margin: 12pt 0 6pt; page-break-after: avoid; }
table { border-collapse: collapse; width: 100%; margin: 10pt 0 8pt; font-family: sans-serif; }
th { font-size: 8.1pt; text-align: left; }
td { font-size: 8.0pt; } th,td { padding: 4pt 3pt; border-bottom: .5pt solid #d4dce1; }
tr { page-break-inside: avoid; } .figure { page-break-inside: avoid; margin: 12pt 0; }
.caption { font-size: 9pt; line-height: 1.25; text-align: left; color: #344758; }
img { display: block; } .equation { font-size: 10pt; text-align: center; margin: 12pt 0; }
code { font-family: monospace; font-size: 8.5pt; } pre { font-size: 8pt; padding: 8pt; white-space: pre-wrap; }
a { color: #155f6d; text-decoration: none; } .evidence { font-size: 8pt; text-align: left; }
li { margin-bottom: 5pt; } .break { page-break-before: always; }
"""


def render_markdown(text, sources):
    """Render the controlled research-document subset without external services."""
    def inline(s):
        s = html.escape(s)
        def cite(match):
            path = html.unescape(match[1])
            if path not in sources: sources.append(path)
            number = sources.index(path) + 1
            return f'<a href="#data-{number}">[D{number}]</a>'
        s = re.sub(r"\[A:([^\]]+)\]", cite, s)
        s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
        s = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", s)
        s = re.sub(r"`([^`]+)`", lambda m: '<code>' + m[1].replace("/", "/&#8203;") + '</code>', s)
        s = re.sub(r"\[([^\]]+)\]\((https?://[^\s)]+)\)", r'<a href="\2">\1</a>', s)
        return s
    out, paragraph, table_lines, code_lines = [], [], [], None
    def flush():
        if paragraph:
            out.append("<p>" + inline(" ".join(paragraph)) + "</p>"); paragraph.clear()
        if table_lines:
            out.append("<table>")
            for i, row in enumerate(table_lines):
                if i == 1 and re.match(r"^[| :\-]+$", row): continue
                tag = "th" if i == 0 else "td"
                out.append("<tr>" + "".join(f"<{tag}>{inline(cell.strip())}</{tag}>" for cell in row.strip().strip("|").split("|")) + "</tr>")
            out.append("</table>"); table_lines.clear()
    for line in text.splitlines():
        if line.startswith("```"):
            flush()
            if code_lines is None: code_lines = []
            else:
                out.append("<pre>" + html.escape("\n".join(code_lines)) + "</pre>"); code_lines = None
            continue
        if code_lines is not None:
            code_lines.append(line); continue
        if not line.strip(): flush(); continue
        heading = re.match(r"^(#{1,3})\s+(.*)$", line)
        if heading:
            flush(); level = len(heading[1]); title = heading[2]
            anchor = "section-" + re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
            out.append(f'<h{level} id="{anchor}">{inline(title)}</h{level}>'); continue
        pic = re.fullmatch(r"!\[(.*?)\]\((.*?)\)", line.strip())
        if pic:
            flush(); out.append(f'<div class="figure"><img src="{html.escape(pic[2])}" width="480"><p class="caption">{inline(pic[1])}</p></div>'); continue
        if line.startswith("| "):
            if paragraph: flush()
            table_lines.append(line); continue
        if line.strip() == "---": flush(); continue
        if line.startswith("- "):
            flush(); out.append('<p>• ' + inline(line[2:]) + '</p>'); continue
        if line.startswith("> "):
            flush(); out.append('<p class="equation">' + inline(line[2:]) + '</p>'); continue
        if line.startswith("    "):
            flush(); out.append('<p class="equation">' + inline(line.strip()) + '</p>'); continue
        paragraph.append(line)
    flush()
    require(code_lines is None, "Unclosed code block")
    return "\n".join(out)


def paginate_body(root, full_html, heading_titles):
    """Keep figures, short tables and headings intact with explicit block layout.

    MuPDF Story can scale images into the remainder of a page and does not
    implement every CSS keep rule. Measure whole units against a fresh page
    before drawing them; split only long tables, repeating their headers.
    """
    import pymupdf
    pattern = re.compile(r'<div class="figure">.*?</div>|<table>.*?</table>|<h[1-3]\b.*?</h[1-3]>|<p\b.*?</p>|<pre>.*?</pre>', re.S)
    matches = list(pattern.finditer(full_html))
    require(not pattern.sub("", full_html).strip(), "Unrecognized manuscript layout block")
    blocks = []
    for match in matches:
        block = match[0]
        if block.startswith("<table>"):
            rows = re.findall(r"<tr>.*?</tr>", block, re.S)
            if len(rows) > 18:
                # Setup-overhead inventories are long; each continuation has
                # the same column headings and at most ten data rows.
                for start in range(1, len(rows), 10):
                    blocks.append("<table>" + rows[0] + "".join(rows[start:start+10]) + "</table>")
                continue
        blocks.append(block)
    units = []
    pending = []
    i = 0
    while i < len(blocks):
        block = blocks[i]
        if re.match(r"<h[1-3]\b", block):
            pending.append(block)
        else:
            pending.append(block)
            if block.startswith("<p"):
                while i + 1 < len(blocks) and blocks[i+1].startswith('<p class="equation">'):
                    pending.append(blocks[i+1]); i += 1
            if block.startswith("<table>") and i + 1 < len(blocks) and blocks[i+1].startswith("<p><b>Table "):
                pending.append(blocks[i+1]); i += 1
            units.append("\n".join(pending)); pending = []
        i += 1
    require(not pending, "Manuscript ends with an orphan heading")

    stream = io.BytesIO()
    writer = pymupdf.DocumentWriter(stream)
    page_rect = pymupdf.Rect(0, 0, 595.276, 841.89)
    content = pymupdf.Rect(49, 58, 546, 787)
    archive = pymupdf.Archive(str(root))
    page_number, cursor, dev = 0, content.y0, None
    link_positions, headings, layout = [], [], []
    for index, unit in enumerate(units):
        story = pymupdf.Story(unit, user_css=CSS, archive=archive)
        more, measured = story.place(content)
        require(not more, f"Layout unit {index} exceeds one page: {unit[:120]}")
        height = pymupdf.Rect(measured).height
        require(height <= content.height + .1, "Measured block exceeds content rectangle")
        if dev is None or cursor + height > content.y1 - .2:
            if dev is not None:
                writer.end_page()
            page_number += 1
            require(page_number < 120, "Unexpectedly long report")
            dev = writer.begin_page(page_rect)
            cursor = content.y0
        story.reset()
        more, filled = story.place(pymupdf.Rect(content.x0, cursor, content.x1, content.y1))
        require(not more, f"Measured layout did not fit unit {index}")
        def position(pos):
            pos.page_num = page_number
            link_positions.append(pos)
            if pos.heading and pos.open_close & 1:
                headings.append({"level": pos.heading, "text": heading_titles.get(pos.id, pos.text),
                                 "page": page_number, "id": pos.id})
        story.element_positions(position)
        story.draw(dev)
        layout.append({"unit": index, "page": page_number, "top": cursor,
                       "bottom": pymupdf.Rect(filled).y1, "has_figure": '<img ' in unit,
                       "has_table": '<table>' in unit})
        cursor = pymupdf.Rect(filled).y1
    if dev is not None:
        writer.end_page()
    writer.close()
    stream.seek(0)
    return pymupdf.Story.add_pdf_links(stream, link_positions), headings, layout


def render_pdf(root, manuscript, abstract):
    import pymupdf
    dest = root / OUT
    sources = []
    body_html = render_markdown(manuscript, sources)
    evidence = []
    for i, relative in enumerate(sources, 1):
        path = root / relative
        require(path.is_file(), f"Cited evidence is missing: {relative}")
        require(path.resolve().is_relative_to(root), "Evidence path escapes project")
        evidence.append({"id": f"D{i}", "path": relative, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    evidence_html = '<h1 id="data-sources">Appendix E. Local evidence catalogue</h1><p>The following identifiers link claims to preserved project-relative artifacts. Full SHA-256 values are in provenance.json. These records are the study evidence, distinct from the scientific literature references.</p>'
    for r in evidence:
        evidence_html += f'<p class="evidence" id="data-{r["id"][1:]}"><b>[{r["id"]}]</b> ' + html.escape(r["path"]).replace("/", "/&#8203;").replace("_", "_&#8203;") + '</p>'
    full_html = body_html + evidence_html
    (dest / "comprehensive_report.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>' + TITLE + '</title><style>' + CSS + '</style><body><h1>' + TITLE + '</h1><p>' + SUBTITLE + '</p><h2>Abstract</h2><p>' + html.escape(abstract) + '</p>' + full_html + '</body></html>', encoding="utf-8")
    # Project-root image paths make the HTML/PDF evidence resolution explicit.
    # Browser HTML is portable from its output folder after path rebasing.
    browser = (dest / "comprehensive_report.html").read_text(encoding="utf-8")
    browser = re.sub(r'src="((?:large_qubit|phase\d)/[^"]+)"', lambda m: 'src="../../../' + m[1] + '"', browser)
    (dest / "comprehensive_report.html").write_text(browser, encoding="utf-8")
    heading_titles = {"section-" + re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-"): title
                      for title in re.findall(r"^#{1,3}\s+(.+)$", manuscript, re.M)}
    heading_titles["data-sources"] = "Appendix E. Local evidence catalogue"
    body, positions, layout = paginate_body(root, full_html, heading_titles)
    document = pymupdf.open()
    cover = document.new_page(width=595.276, height=841.89)
    cover.draw_rect(pymupdf.Rect(49, 61, 546, 67), color=None, fill=(.08,.37,.43))
    cover_html = f'<p style="font-family:sans-serif;font-size:10pt;color:#155f6d">SC4055 · COMPREHENSIVE RESEARCH REPORT · 9 NOVEMBER 2026</p><h1 style="font-size:26pt">{TITLE}</h1><p style="font-size:14pt;color:#526579">{SUBTITLE}</p><h2>Abstract</h2><p>{html.escape(abstract)}</p><p style="font-size:9pt"><b>Keywords:</b> molecular electronic structure; variational quantum eigensolver; language models; LoRA; energy-weighted likelihood; fixed-budget evaluation; reproducibility.</p><p style="font-size:9pt"><b>Evidence boundary.</b> One pipeline seed and two fixed CO₂ interpolation targets. Historical H₂/LiH findings retain their original pilot/development scope. Prepared entirely from saved results; no new training or energy evaluation.</p>'
    spare, scale = cover.insert_htmlbox(pymupdf.Rect(49, 84, 546, 785), cover_html, css=CSS, scale_low=1)
    require(spare >= 0, "Cover overflow")
    toc_page = document.new_page(width=595.276, height=841.89)
    major = [p for p in positions if p["level"] == 1]
    toc_html = '<h1>Contents</h1><table>'
    for p in major:
        toc_html += '<tr><td>' + html.escape(p["text"]) + '</td><td style="text-align:right">' + str(p["page"] + 2) + '</td></tr>'
    toc_html += '</table><h2>How to read this report</h2><p>The main text explains the physical model, dataset, learned objective, controls, results and costs. Appendices retain individual trajectories, full numerical endpoints and the provenance needed to audit the claims. [1]–[10] identify primary literature; [D1] onward identify local saved evidence.</p><p>The original ten-page report and its authenticated submission package remain unchanged. This longer document adds methodological detail and explicitly labelled descriptive summaries of the same completed experiment.</p>'
    spare, scale = toc_page.insert_htmlbox(pymupdf.Rect(49, 64, 546, 785), toc_html, css=CSS, scale_low=1)
    require(spare >= 0, "Contents overflow")
    document.insert_pdf(body)
    toc = [[1, "Abstract", 1], [1, "Contents", 2]]
    for p in positions:
        toc.append([min(p["level"], 2), p["text"], p["page"] + 2])
    document.set_toc(toc)
    image_count = 0
    for i, page in enumerate(document):
        if i > 1:
            page.insert_text((49, 34), "ENERGY-WEIGHTED MOLECULAR CIRCUIT SEARCH", fontname="helv", fontsize=7.5, color=(.25,.35,.42))
            page.draw_line((49, 43), (546, 43), color=(.75,.81,.84), width=.5)
        page.insert_text((49, 815), "Comprehensive report · 9 November 2026", fontsize=7.5, color=(.3,.38,.44))
        page.insert_text((509, 815), f"{i+1} / {len(document)}", fontsize=7.5, color=(.3,.38,.44))
        require(len(page.get_text().strip()) > 100, f"Blank or near-empty page {i+1}")
        for entry in page.get_images():
            for r in page.get_image_rects(entry[0]):
                require(r.x0 >= 45 and r.x1 <= 550 and r.y0 >= 50 and r.y1 <= 791, f"Clipped figure on page {i+1}")
                image_count += 1
    expected_images = full_html.count("<img ")
    require(image_count == expected_images, f"Missing/duplicated figures: {image_count} versus {expected_images}")
    document.set_metadata({"title": TITLE, "subject": SUBTITLE, "author": "SC4055 research project", "keywords": "CO2, LiH, LoRA, quantum circuits, fixed-budget research"})
    document.save(dest / "comprehensive_report.pdf", garbage=4, deflate=True)
    previews = dest / "previews"; previews.mkdir(exist_ok=True)
    for i, page in enumerate(document):
        page.get_pixmap(matrix=pymupdf.Matrix(1,1), alpha=False).save(previews / f"page_{i+1:02d}.png")
    metrics = {"pages": len(document), "images": image_count, "bookmarks": len(toc), "source_citations": len(evidence),
               "word_count_markdown": len(manuscript.split()), "headings": positions,
               "layout_units": layout,
               "no_blank_pages": True, "all_images_embedded_and_inside_margins": True}
    write_json(dest / "provenance.json", {"sources": evidence, "existing_section11_manifest_sha256": sha256_file(root / "large_qubit/registrations/section11_analysis_v1/manifest.json"),
               "new_scientific_evaluations": 0, "new_training_executed": False, "analysis_scope": "Expanded write-up and post-hoc descriptive summaries; existing primary endpoints unchanged"})
    write_json(dest / "render_checks.json", metrics)
    document.close(); body.close()
    # This inventory belongs only to the expanded report, not to a scientific
    # registration or to the already sealed ten-page submission.
    report_files = [root / "large_qubit/comprehensive_report.py"]
    report_files.extend(sorted((dest / "parts").glob("*")))
    report_files.extend(sorted((dest / "figures").glob("*")))
    report_files.extend(dest / name for name in ("comprehensive_report.pdf", "comprehensive_report.md",
        "comprehensive_report.html", "proposal_analysis.json", "proposal_summary.csv", "provenance.json", "render_checks.json"))
    write_json(dest / "report_manifest.json", {"schema_version": "expanded-report-inventory-v1",
        "scope": "Report reproducibility inventory, not a new scientific registration",
        "files": {p.relative_to(root).as_posix(): {"sha256": sha256_file(p), "bytes": p.stat().st_size}
                  for p in report_files if p.is_file()}})
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("analyze", "render", "verify"))
    parser.add_argument("--root", default=".")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve(); dest = root / OUT; dest.mkdir(parents=True, exist_ok=True)
    if args.action == "analyze":
        a, c = read_json(root / ANALYSIS / "analysis.json"), read_json(root / ANALYSIS / "costs.json")
        extra, populations = ledger_distributions(root, a)
        write_json(dest / "proposal_analysis.json", extra)
        write_csv(dest / "proposal_summary.csv", extra["rows"])
        make_figures(root, a, c, extra, populations)
        print(json.dumps({"saved_trajectories_read": len(extra["rows"]), "new_energy_requests": 0}))
    elif args.action == "render":
        abstract = (dest / "parts/abstract.txt").read_text(encoding="utf-8").strip()
        parts = sorted((dest / "parts").glob("*.md"))
        manuscript = "\n\n".join(p.read_text(encoding="utf-8").strip() for p in parts)
        (dest / "comprehensive_report.md").write_text("# " + TITLE + "\n\n## Abstract\n\n" + abstract + "\n\n" + manuscript, encoding="utf-8")
        result = render_pdf(root, manuscript, abstract)
        print(json.dumps({k: v for k, v in result.items() if k not in ("headings", "layout_units")}, indent=2))
    else:
        from .section11 import verify
        result = verify(root)
        inventory = read_json(dest / "report_manifest.json")
        for relative, expected in inventory["files"].items():
            p = root / relative
            require(p.resolve().is_relative_to(root) and p.is_file(), f"Report file missing: {relative}")
            require(p.stat().st_size == expected["bytes"] and sha256_file(p) == expected["sha256"], f"Report file changed: {relative}")
        for record in read_json(dest / "provenance.json")["sources"]:
            p = root / record["path"]
            require(p.is_file() and p.stat().st_size == record["bytes"] and sha256_file(p) == record["sha256"],
                    f"Report evidence changed: {record['path']}")
        checks = read_json(dest / "render_checks.json")
        print(json.dumps({"original_submission_still_authenticated": result["all_passed"],
                          "expanded_report_files_authenticated": len(inventory["files"]),
                          "cited_evidence_authenticated": len(read_json(dest / "provenance.json")["sources"]),
                          "report": {k:v for k,v in checks.items() if k not in ("headings", "layout_units")}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
