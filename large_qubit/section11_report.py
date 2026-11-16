"""Ten-page report rendered from saved analysis, with editable Markdown/HTML."""
from __future__ import annotations
import html
import re
from pathlib import Path

from .common import require
from .section11_analysis import ANALYSIS, LABELS, METHODS, SUBMISSION, TARGETS


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"] +
                     ["| " + " | ".join(str(x) for x in row) + " |" for row in rows])


def pages(a, c):
    means = table(["Method", "I01 (mHa)", "I02 (mHa)", "Mean (mHa)"],
                  [[LABELS[r["method"]], *(f'{r[k]:.3f}' for k in ("I01_error_mha", "I02_error_mha", "mean_error_mha"))] for r in a["method_rows"]])
    paired = table(["Paired difference", "I01 (mHa)", "I02 (mHa)", "Mean (mHa)"],
                   [[r["comparison"].replace("_", " "), *(f'{r[k]:.6f}' for k in ("I01_difference_mha", "I02_difference_mha", "mean_difference_mha"))] for r in a["paired_rows"]])
    references = table(["Target", "R (Å)", "HF (Ha)", "Sector reference (Ha)"],
                       [[t, r, f'{a["jobs"][f"gpt2_xl_frozen__{t}"]["hf_energy_hartree"]:.10f}',
                         f'{a["jobs"][f"gpt2_xl_frozen__{t}"]["reference_energy_hartree"]:.10f}'] for t, r in zip(TARGETS, (1.10, 1.30))])
    budgets = table(["Ordinary category", "Actual", "Amended ceiling"],
                    [[r["category"].replace("_", " "), f'{r["actual"]:,}', f'{r["ceiling"]:,}'] for r in c["ordinary_requests"]["categories"]] + [["TOTAL", "31,452", "33,178"]])
    training = table(["Final fit", "Steps", "Assigned hours", "Cost qualification"],
                     [[LABELS[k], r["committed_optimizer_steps"], f'{r["assigned_device_hours"]:.3f}',
                       "Lower bound" if r["wall_is_lower_bound"] else "Recorded"] for k, r in c["training"]["fits"].items()])
    marginal_rows = []
    for m in METHODS:
        x = [c["main_trajectories"][f"{m}__{t}"] for t in TARGETS]
        marginal_rows.append([LABELS[m], f'{sum(r["generation_seconds"] for r in x)/3600:.3f}',
                              f'{sum(r["scoring_seconds"] for r in x)/3600:.3f}', sum(r["physical_hamiltonian_actions"] for r in x)])
    marginal = table(["Both-target marginal cost", "Generation h", "Scoring h", "H actions"], marginal_rows)
    resources = table(["Method", "Dup I01/I02", "CX I01/I02", "Depth I01/I02"],
        [[LABELS[m], "/".join(str(next(r for r in a["count_rows"] if r["method"] == m and r["target"] == t)["duplicate_requests"]) for t in TARGETS),
          *["/".join(str(next(r for r in a["resource_rows"] if r["method"] == m and r["target"] == t)[k]) for t in TARGETS)
            for k in ("excluding_hf_two_qubit_count", "excluding_hf_depth")]] for m in METHODS])
    exact_membership = [(LABELS[x["method"]], x["target"]) for x in a["endpoint_rows"] if x["winner_exactly_in_fitting_corpus"]]
    membership = ("The exact ordered 32-ID winner occurs in the fitting corpus for " +
                  "; ".join(f"{m} at {t}" for m, t in exact_membership) + ".") if exact_membership else "No discrete winning 32-ID string exactly matches a fitting string."
    return [
    ("Offline language-model proposals for molecular circuits", f"""
**SC4055 group project · Revised CO₂ study · 9 November 2026**

**Abstract.** Can an energy-weighted language model propose useful molecular circuits under a small, explicit energy-query budget? We study full STO-3G CO₂, encoded in 30 Jordan–Wigner qubits and simulated in a fixed electron-population sector. A single accepted corpus supplies three five-pass LoRA fits: GPT-2 XL with weighted and uniform likelihood, and Qwen3-4B with weighted likelihood. Eight methods are compared at two fixed interpolation geometries, using one pipeline seed and an allowance of 1,024 charged requests per trajectory. All sixteen final trajectories and winners are authenticated.

Adaptation reduces mean reference error by 105.973 mHa for GPT-2 and 120.388 mHa for Qwen. Weighted GPT-2 improves upon its matched uniform control by only 0.652 mHa averaged over the two targets. Qwen weighted has the lowest mean among generators, 105.199 mHa, but training-circuit retrieval remains competitive at 105.419 mHa. Compact VQE wins one target and retrieval wins the other. Substantial residual errors, duplicate generation and acquisition costs limit the practical claim. This is a descriptive case study, not a demonstration of quantum advantage or statistical reliability across seeds.

### Motivation and questions
Finding low-energy wavefunctions connects quantum algorithms to molecular electronic structure. VQE optimizes a parameterized circuit through energy feedback [3]. A generator can instead amortize information gathered at training geometries: propose discrete circuit strings at a new geometry without updating itself from target energies. This may shift work from target-time optimization into expensive offline acquisition and fitting. Both portions must therefore be reported.

The experiment asks three questions: does adaptation help these particular frozen language models; does energy weighting help beyond matched uniform fitting; and do learned proposals outperform non-neural baselines at equal target request allowance? The GPT pair addresses weighting with matched initialization, examples and optimizer exposure. A Qwen–GPT comparison additionally changes architecture, tokenizer, size and pretraining, so it does not isolate any one cause.

### Contribution and scope
The contribution is an auditable implementation and controlled comparison: physical identity checks, fixed training exposure, complete logical accounting, credible retrieval/VQE controls, recoverable execution and negative-result reporting. Language-model circuit generation and geometry transfer already have precedents [1,8–10]. No claim is made to invent LoRA, generative eigensolving or quantum circuit synthesis. All numerical results here come from stored evidence; analysis loads no model and issues no physical energy query.

**Reading guide.** Pages 2–4 specify physics, fitting and evaluation; pages 5–7 give every method and target; page 8 accounts for costs; page 9 separates historical LiH evidence; page 10 states limitations and reproduction requirements. The accompanying tables retain full numerical precision.
"""),
    ("Physical problem and numerical integrity", f"""
### Hamiltonian and state representation
We use linear, symmetric CO₂ in STO-3G: 22 electrons, 15 spatial orbitals and 30 spin orbitals/qubits. The electronic Hamiltonian contains one- and two-electron integrals plus nuclear repulsion. The circuit objective is E_g(s) = ⟨HF|U_s† H_g U_s|HF⟩. Orbital ordering, units, basis, geometry and integral fingerprints are frozen by the new benchmark; old LiH operator meanings are never reused. There is no frozen-core approximation or qubit tapering in this benchmark.

The simulator stores the sector with Nα = Nβ = 11, containing C(15,11)² = 1,863,225 determinants. One complex128 state occupies 29,811,600 bytes (28.43 MiB); this is much smaller than a dense 2³⁰ state. Hamiltonian action is matrix free. PySCF supplies electronic-structure and direct-spin FCI components [4]. Separate alpha/beta population preservation is checked, but it does not establish total-spin purity. “30 qubits” describes the encoding, not evidence that this instance is classically hard.

### Geometry split and classical context
Here R is the length of each C?O bond. The three fitting geometries are R = 1.00, 1.20 and 1.32 Å; compact validation uses 1.25 Å. Final interpolation targets are 1.10 and 1.30 Å. The last fitting geometry moved from 1.40 through 1.35 to 1.32 Å following preproduction HF stability failures; those receipts and costs remain preserved. Target outcomes did not choose the training geometry.

{references}

The reference is the numerically converged lowest energy in the same Hamiltonian and fixed population sector, obtained from two starts with residual checks. It is not an experimental energy or a complete-basis physical prediction. HF provides a common feasible incumbent at the first charged request. Signed reference error is E_best,verified − E_reference; HF improvement is E_HF − E_best,verified. All energies include the same nuclear contribution.

### What the correctness evidence establishes
Small H₂/LiH fixtures compare independent constructions, ordering and signs. CO₂ tests check Hamiltonian action, norm and electron-population conservation, reference convergence and resource bounds. Final winners are replayed with a fresh basis/evaluator and cache bypass, then checked against saved incumbents. All sixteen pass. This replay is independent of the search cache, but uses the same Hamiltonian backend; we do not call it an independent-backend replication.

References required 122 Hamiltonian actions and 347.052 recorded seconds for the two targets, reused once by recovery. Winner replay required sixteen actions and 49.787 seconds. These are separately recorded verification/reference overhead, not extra requests available to a method. Full fingerprints, residuals and initial correctness receipts are linked in the artifact index.
"""),
    ("One shared corpus and three fixed-budget fits", """
### Acquisition and the preserved rejection
A training-only pilot fixed the catalogue and chose the shortest viable tested sequence length, 32. The discrete representation has 513 operator IDs, built from 64 double-excitation patterns plus the identity and eight signed angles ±{0.0125, 0.025, 0.05, 0.1} radians. A candidate is a complete ordered list of 32 IDs. The same accepted fitting rows serve all three models.

The first production corpus failed its frozen concentration gate at T03, λ = 0.10: effective sample size was 4.55% of rows and the largest normalized weight was 5.79%, against 5% limits. That NO-GO remains a result. An approved fresh v2 campaign changed only the proposal mixture to 40% guided, 20% uniform and 40% local, keeping quality rules and λ choices fixed. V1 consumed 6,538 requests and v2 consumed 6,552. No rejected v1 labels entered the accepted confirmation release.

The release contains 6,144 fitting examples (2,048 per geometry) and 384 separate diagnostic examples. Diagnostics are inaccessible to the fitter. Shared acquisition is charged once globally, while its cost remains relevant when judging amortized methods. The release manifest, raw request ledgers, rejection evidence and amendment are preserved.

### Objective and matched control
For fitting row i in geometry g, let ℓᵢ be the sum of grammar-masked negative log probabilities of all answer tokens, including EOS. Prompts and padding contribute no loss; there is no answer-length divisor. With fitting-only energy minimum E_min,g and dispersion d_g = max(IQR_g, 10⁻⁶ Ha), the normalized weight is

**wᵢ = exp[−(Eᵢ − E_min,g)/(λ d_g)] / Σⱼ∈g exp[−(Eⱼ − E_min,g)/(λ d_g)].**

Exponential weighting has precedent in advantage-weighted regression [7], without implementing its full algorithm. The row multiplier is n_g wᵢ for weighted fitting and 1 for uniform fitting. Each effective batch uses L = Σᵢ multiplierᵢ ℓᵢ / B_actual; weights are applied exactly once, and the divisor is not the sum of weights. GPT weighted uses λ = 0.25; Qwen weighted uses λ = 0.10. The two GPT fits share initialization, row order and numerical settings, differing in objective weights.

LoRA trains only low-rank A/B updates [2]: rank 16, alpha 32 and dropout 0.05 across the registered native attention/MLP projections. AdamW uses learning rate 2×10⁻⁴, β = (0.9, 0.999), epsilon 10⁻⁸, zero weight decay, constant schedule and norm clipping at 1. Each fit makes five passes: 30,720 example presentations and 1,920 optimizer steps at effective batch 16. The final checkpoint is used; no energy-based checkpoint selection occurs.

### What is learned
GPT-2 XL and Qwen3-4B use pinned pretrained revisions [5,6], fresh adapters and BF16 execution on assigned H200s. Qwen thinking is disabled. This is offline weighted likelihood inspired by generative eigensolving, not a reproduction of GQE's circuit replay/logit-matching/GRPO algorithm [1]. Target energies do not update model weights or enter later generator prompts. The study cannot isolate the causal benefit of language pretraining without a matched randomly initialized control.
"""),
    ("Evaluation protocol, baselines and recovery", """
### Fixed matrix
There are eight methods at each of two targets: frozen GPT-2 XL, weighted GPT-2 XL, uniform GPT-2 XL, frozen Qwen3-4B, weighted Qwen3-4B, uniform random strings, training-circuit retrieval and compact VQE. One pipeline seed (22) is used throughout. A compact eight-trajectory validation panel precedes the final freeze. The main matrix uses the frozen temperature 0.7, grammar, prompts, circuit length, model revisions and final adapters. Validation outcomes do not select a new main hyperparameter sweep.

Each trajectory is allowed 1,024 logical energy requests, including its initial HF evaluation. A cache hit, duplicate, failed request or retry retains its registered charge. Models produce 1,023 complete proposals following HF, with valid full-answer grammar enforcement. Generation batch size remains one; training batching and four independent inference lanes are separate speedups. These facts matter when interpreting the measured time.

### Non-neural competitors and circuit families
Uniform random draws from the same discrete support. Retrieval reuses stored training circuits and evaluates them at the target; it directly tests whether the shared acquisition already supplies useful answers. Its source is fitting data, not target labels. These controls prevent an apparent neural benefit from being confused with the value of classical data collection.

Compact VQE uses a fixed 24-parameter fermionic excitation ansatz and a deterministic coordinate optimization rule with complete six-query coordinate updates. Two restarts use HF-zero and a training-derived vector, with caps of 511 and 512 requests. Each update uses five Fourier samples and a sixth checked candidate; sampled ±π and ±π/2 probes can lie outside the candidate bounds ±0.25, and all charged probes may become incumbents. The generator catalogue instead uses whole-qubit excitation gates. Operator signs and resulting gate families differ, so equal query allowance is not an identical-ansatz comparison. Both VQE trajectories finish with 1,023 charged requests; the unused final request cannot fund a whole update. The incumbent is carried to budget 1,024, explicitly marked in the curve and table rather than represented as a new evaluation.

### Preserved failure and final recovery
The first four-lane attempt failed its inherited RSS high-water guard on sixteen HF requests, before candidate generation. A separate recovery registration corrected the measurement to the Linux process's own VmHWM while preserving the 4 GiB cap. Recovery retained original definitions, seed, targets, budgets and references. The failed sixteen requests remain charged, taking the amended cumulative main ceiling to 16,400. Recovery completed 16,382 requests; cumulative main work is 16,398.

All sixteen recovery jobs are complete, with zero failed, abandoned or pending energy requests. All 10,230 model completion attempts are valid; there are zero invalid or infrastructure-failed completions in recovery. Those zeroes do not erase the sixteen failed requests of the first attempt. Full ledgers and both attempts remain linked.

### Estimand and interpretation
The primary endpoint is each verified best-of-budget reference error. The displayed mean is the unweighted average of the two fixed target endpoints. Paired differences use the same target before averaging. There are no seed-level confidence intervals, significance tests or circuit-bootstrap error bars: thousands of proposals from one fitted model do not create independent training replications. Curves show search progress, not independent observations of reliability.
"""),
    ("All sixteen verified endpoints", f"""
{means}

![Verified reference errors for every method and both targets](../../analysis/section11_v1/figures/endpoints.png)

**Table/figure 1.** Errors are in milli-Hartree (1 mHa = 0.001 Ha); lower is better. The reference is zero. Both target columns remain visible; the two-target mean cannot replace them. Tables in `analysis/section11_v1/tables/endpoint_rows.csv` additionally retain absolute energies, HF improvements, winning request positions and actual request counts.

Adaptation improves both backbones at both targets. However, weighting is a much smaller effect than adaptation. Retrieval beats weighted GPT-2 at both targets. Qwen weighted has a 0.220 mHa lower mean than retrieval, with a win at I01 and a loss at I02. This mixed result does not establish broad superiority of the learned generator.

The best I01 endpoint comes from compact VQE, 85.178 mHa above the reference. The best I02 endpoint comes from retrieval, 123.582 mHa above the reference. Weighted GPT-2 and Qwen find the same exact winning sequence at I02. These residual errors remain substantial; a completed and numerically validated run is not evidence of chemically accurate predictions.
"""),
    ("Energy versus requests and paired effects", f"""
![All sixteen best-so-far trajectories with HF and classical reference](../../analysis/section11_v1/figures/budget_curves.png)

**Figure 2.** Every step counts a logical request, including duplicates and shared-cache hits. Each target has eight trajectories; overlapping traces remain separate in the full 16,384-row curve CSV. The VQE value at request 1,024 is a carried incumbent, not an extra physical evaluation. The zero reference line and HF baseline share the same Hamiltonian.

{paired}

**Table 2.** Negative paired differences favor the left method. Adaptation reduces GPT error by 105.973 mHa on average and Qwen error by 120.388 mHa. GPT weighting reduces error by 0.005991 mHa at I01 and 1.298156 mHa at I02. The mean 0.652073 mHa benefit is descriptive and driven mainly by I02. It warrants neither a significance claim nor a prediction that another training seed will preserve the ordering.

The Qwen–GPT weighted mean difference is only 0.427266 mHa, with a tie at I02. Architecture and tokenizer differences confound causal interpretation. Best-of-budget performance can also hide concentrated proposal distributions: the following resource/duplicate panel therefore complements, rather than replaces, endpoint accuracy. No main outcome has been used to revise the frozen model, temperature, target set or request budget.
"""),
    ("Duplicates, circuit resources and reuse", f"""
{resources}

**Table 3.** Duplicate counts use all charged logical requests (1,024 except VQE's 1,023). They are exact physical-query repeats within a trajectory, not independent seeds. Cache hits also include identical cross-method queries in the same target partition; therefore hits and within-trajectory duplicates need not agree. Across recovery there are 6,500 cache hits and 9,882 physical Hamiltonian actions.

Frozen Qwen duplicates 1,022 queries at each target and retains HF as its winner. Syntactic validity alone therefore says little about useful diversity. Weighted training can increase low-energy probability while also increasing duplicates; best energy, diversity and numerical cost must be reported together. Every duplicate remains charged even when evaluation reuses a cached value.

### Common decomposition and qualifications
Resource counts use the same logical all-to-all gate vocabulary (H, X, RX, RZ and CX), with exact commuting Pauli-rotation decompositions. There is no hardware routing, noise model, connectivity constraint or cross-gate optimization. Native generator gates are whole-qubit; VQE gates are fermionic. Only identity/zero rotations are omitted. Counts exclude HF state preparation; the exported resource table also includes HF, which adds 22 X gates and one initial depth layer.

Frozen Qwen's zero CX/depth winner is HF, not a free accurate solution. Circuit depth depends on the ordered decomposition and available parallelism, so the CX count and depth are complementary. At I02, the identical GPT/Qwen weighted winner has 1,536 CX gates and depth 1,333; VQE has 1,792 CX gates and depth 1,853. These are logical estimates, not measured quantum-device runtimes or a resource advantage claim.

### Post-hoc exact-string membership
{membership}

This is a labelled descriptive check over the accepted fitting corpus, performed after the main run. Exact absence would not prove semantic novelty: different strings can prepare equivalent states. Exact presence would not alone prove memorization or causation. The main conclusion remains grounded in the predeclared retrieval comparison, not a selected novelty metric. Membership flags for every endpoint and the fitting-file fingerprint are exported.
"""),
    ("Resource budget: total work and marginal search", f"""
{budgets}

The original ordinary allowance was 26,624; approved fresh-corpus and recovery amendments increased it to 33,178. Actual ordinary work is 31,452. Rejected v1 acquisition and the failed main attempt are included exactly once. References, correctness fixtures, verification and model fitting are separate categories, not hidden allowances or interchangeable query units.

{training}

The three fits total at least 7.391 assigned-device hours, plus 590.432 seconds of separately measured preflight/load work. The weighted GPT and Qwen interrupted process tails are not fully timed: their costs and peaks remain lower bounds even though final adapters and committed exposure are complete. No dollar, electricity or carbon total can be inferred from these logs.

{marginal}

**Table 4.** Marginal costs cover both targets and exclude shared acquisition, fitting, preflight, validation and references. Recovery took 7.052 hours of coordinator elapsed time over four lanes. Generation totals 18.492 component-hours; scoring totals 8.378 component-hours. They overlap across lanes and within execution, and must not be added as project elapsed time. Aggregate research expenditure counts shared acquisition once; a standalone corpus-using method still bears its attributable acquisition cost. Complete phase timings, loads, retries, token totals and overhead appear in `costs.json`, the cost CSVs and `resource_budget.md`.

The observed rate of about 0.645 logical requests per elapsed second describes this completed four-lane workload, not a universal GPU scaling factor. CPU scoring, reference solving, model loads, cache overlap and other users' load affect it. Implementation time and untimed historical work are not silently assigned zero. The protected analysis window was used for saved-artifact CPU analysis; no further GPU work was necessary.
"""),
    ("Retained LiH evidence and critical discussion", """
### Separate historical scope
The earlier `gqe-local-llm-lih-v1.1` branch used a 12-qubit LiH problem and different operator semantics. Its evidence is retained as acquisition, pilot and development history. It is not pooled with CO₂ endpoints, and its multiple acquisition seeds do not become multiple independent CO₂ training replications.

Historical acquisition consumed 197,740 logical requests (197,675 executions and 65 hits), plus 1,984 independent full-space state/expectation checks and 496 PennyLane executions. The main LiH release has 122,880 fitting and 15,360 diagnostic rows across five seeds and six training geometries. Only 3 of 16,314 fitting rows below HF minus 5 mHa came from uniform sampling; 16,311 came from the local component. Classical acquisition is therefore central to the learned proposal distribution.

![Preserved LiH acquisition coverage, not held-out model performance](../../../phase5/coverage/figures/coverage_LIH_T02_main_mixed.png)

**Figure 3 (historical, unchanged).** LIH_T02 coverage and descriptive effective sample size from the original Phase 5 release. Mixed fitting and uniform diagnostics have different sampling mechanisms; their contrast is not an IID generalization test. Original vector figures are linked in the historical appendix.

The seed-11 pilot evaluates six training geometries with 32 attempts per condition. Weighted GPT's adapted mean energy relative to HF was −11.959 mHa, versus +6.424 mHa for uniform GPT; the below-HF counts were 187/192 and 102/192. Yet uniform diagnostic NLL was better (106.312 versus 110.632). This illustrates why likelihood on uniform strings is not a substitute for generated energy. It is not a held-out LiH result.

Five successful pilot fits consumed 62.825 recorded training-loop device-hours, with 1,980 pilot scoring requests. A preserved interrupted GPT fit adds 346 steps and 1,666.059 seconds. These exclude other setup/development work and are not a complete historical expenditure total. Phase 7 authenticated 348 checkpoint-validation jobs and 696 alias records, but its old convergence gate remains false. The revised fixed-budget CO₂ results do not retroactively satisfy it.

For CO₂, strong adaptation gains coexist with small weighting gains and competitive retrieval. The practical benefit may be compression or transfer of already useful acquired strings, not discovery beyond the training source. More independent training seeds and broader held-out geometry/molecule coverage would be needed to evaluate that hypothesis; they are future work, not missing evidence filled by this report.
"""),
    ("Limits, reproduction and conclusion", """
### Claims supported and not supported
This one-seed, two-target study supports paired descriptive comparisons of the specified systems. It does not support seed reliability, statistical significance, general cross-molecule transfer, causal benefits of pretraining, global VQE superiority/inferiority, hardware advantage or quantum advantage. The minimal basis, fixed population sector, unequal native circuit families, narrow interpolation split and large residual errors constrain interpretation. Winner replay checks cache-independent consistency, not independent Hamiltonian-backend replication. Some interrupted training costs are lower bounds.

The result is nevertheless informative: adapting both models greatly improves their frozen proposals, while the matched GPT weighting effect is small and retrieval/VQE remain credible competitors. The strongest conclusion is the value of a controlled and fully accounted evaluation, including failed work and negative outcomes, rather than a universal advantage of language-model circuit generation.

### Reproduction and deliverables
`REPRODUCE.md` gives CPU-only verification and analysis commands for the existing checkout on Linux and Windows. `artifact_index.json` links the benchmark, physical fingerprints, references, corpus release/ledgers, all three adapters, eight validation records, sixteen main records and their predecessors. The final Section 11 manifest authenticates the analysis sources, tables, figures and submission files. This is a directory of linked artifacts, not a self-contained replacement for the full research checkout. Repeating training or scientific evaluation would still require the registered Linux/GPU environments and separate researcher execution.

The ten-page report is supplied as editable Markdown/HTML and PDF. The editable 15-slide presentation, PDF, timed 15-minute script and saved-results demonstration are prepared for the researcher to record. No video has been fabricated and no course submission has been made. Figures and full-precision CSVs are available separately. Historical registrations, scientific artifacts and the hash-bound research plan are preserved; `large_qubit/section11.md` records the current handoff.

### References
[1] Nakaji et al. The generative quantum eigensolver (GQE) and its application for ground state search. arXiv:2401.09253v2 (2025). https://arxiv.org/html/2401.09253v2

[2] Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685v2 (2021). https://arxiv.org/abs/2106.09685v2

[3] Peruzzo et al. A variational eigenvalue solver on a quantum processor. Nature Communications 5, 4213 (2014). doi:10.1038/ncomms5213.

[4] Sun et al. Recent developments in the PySCF program package. J. Chem. Phys. 153, 024109 (2020). doi:10.1063/5.0006074.

[5] Radford et al. Language Models are Unsupervised Multitask Learners. OpenAI report (2019). [6] Yang et al. Qwen3 Technical Report. arXiv:2505.09388v1 (2025).

[7] Peng et al. Advantage-Weighted Regression. arXiv:1910.00177v3 (2019). [8] Choi. Generating Quantum Circuits with Finetuned Pretrained Large Language Models. MSc thesis, Toronto (2024).

[9] Liang et al. Unleashing the Potential of LLMs for Quantum Computing: A Study in Quantum Architecture Design. arXiv:2307.08191v1 (2023). [10] Zou et al. Generative flow-based warm start of the variational quantum eigensolver. npj Quantum Information 12, 5 (2026), doi:10.1038/s41534-025-01159-x. Primary links and complete metadata are in `references.bib` and `literature_and_history.md`.
"""),
    ]


def _inline(s):
    s = html.escape(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", s)


def markdown_html(text):
    """Small renderer for the explicit report subset, not a general MD parser."""
    out = []
    for block in text.strip().split("\n\n"):
        if block.startswith("### "):
            heading, _, rest = block[4:].partition("\n")
            out.append("<h2>" + _inline(heading) + "</h2>")
            if rest: out.append("<p>" + _inline(rest.replace("\n", " ")) + "</p>")
        elif block.startswith("!["):
            match = re.fullmatch(r"!\[(.*?)\]\((.*?)\)", block.strip())
            require(match, "Malformed report image")
            out.append(f'<img src="{html.escape(match[2])}" width="490"><p class="caption">{html.escape(match[1])}</p>')
        elif block.startswith("| "):
            lines = block.splitlines()
            out.append("<table>")
            for i, line in enumerate(lines):
                if i == 1: continue
                tag = "th" if i == 0 else "td"
                out.append("<tr>" + "".join(f"<{tag}>{_inline(x.strip())}</{tag}>" for x in line.strip("|").split("|")) + "</tr>")
            out.append("</table>")
        else:
            out.append("<p>" + _inline(block.replace("\n", " ")) + "</p>")
    return "\n".join(out)


CSS = """
body { font-family: sans-serif; font-size: 10.2pt; line-height: 1.28; color: #203549; }
h1 { font-size: 22pt; line-height: 1.12; color: #142b43; margin: 0 0 14pt; }
h2 { font-size: 12pt; color: #177e89; margin: 12pt 0 4pt; }
p { margin: 0 0 8pt; } table { border-collapse: collapse; width: 100%; margin: 8pt 0 10pt; }
th { background: #e5eff2; color: #142b43; font-size: 8.7pt; text-align: left; }
td { font-size: 8.5pt; } th,td { padding: 4pt; border-bottom: 0.5pt solid #cbd4db; }
code { font-family: monospace; font-size: 8.3pt; } .caption { font-size: 8pt; color:#526579; }
img { display: block; margin: 6pt 0; } a { color: #177e89; }
"""


def build_report(root, analysis, costs):
    import pymupdf
    root = Path(root)
    dest = root / SUBMISSION
    content = pages(analysis, costs)
    require(len(content) == 10, "Report must contain ten planned pages")
    md = "\n\n---\n\n".join(f"# {i}. {title}\n\n{body.strip()}" for i, (title, body) in enumerate(content, 1))
    (dest / "report.md").write_text(md + "\n", encoding="utf-8")
    html_pages = [f"<h1>{html.escape(title)}</h1>\n{markdown_html(body)}" for title, body in content]
    document = '<!doctype html><html lang="en"><meta charset="utf-8"><title>SC4055 CO2 study</title><style>' + CSS + '\nsection{max-width:760px;margin:40px auto;page-break-after:always;} @media print{section{margin:0}}</style><body>' + "".join(f"<section>{s}</section>" for s in html_pages) + "</body></html>"
    (dest / "report.html").write_text(document, encoding="utf-8")
    pdf = pymupdf.open()
    metrics = []
    for number, page_html in enumerate(html_pages, 1):
        page = pdf.new_page(width=595.276, height=841.89)
        page.draw_rect(pymupdf.Rect(42, 34, 553, 38), color=None, fill=(.09, .49, .54))
        # MuPDF's archive intentionally does not traverse parent components.
        # Keep portable relative links in HTML, but normalize them for PDF input.
        def image_source(match):
            image_path = (dest / match[1]).resolve()
            require(image_path.is_relative_to(root.resolve()) and image_path.is_file(), "Missing report image")
            return 'src="' + image_path.relative_to(root.resolve()).as_posix() + '"'
        page_html = re.sub(r'src="([^"]+)"', image_source, page_html)
        if number == 9:
            # Historical multi-panel thumbnail; the original full-size figure is linked.
            page_html = page_html.replace('width="490"', 'width="440"')
        spare, scale = page.insert_htmlbox(pymupdf.Rect(43, 54, 552, 793), page_html,
                                          css=CSS, archive=pymupdf.Archive(str(root)), scale_low=.80)
        require(spare >= 0, f"Report page {number} overflowed")
        require(scale >= .88, f"Report page {number} needs a content/layout revision (scale={scale:.3f})")
        page.insert_text((43, 818), "SC4055 | Fixed-budget CO2 study | One seed, two targets", fontsize=8, color=(.32, .4, .47))
        page.insert_text((515, 818), f"{number} / 10", fontsize=8)
        metrics.append({"page": number, "scale": scale, "spare_points": spare, "text_characters": len(page.get_text())})
        require(len(page.get_images()) == html_pages[number - 1].count('<img '), "Report image failed to embed")
        for info in page.get_images():
            for rectangle in page.get_image_rects(info[0]):
                require(rectangle.x0 >= 42 and rectangle.x1 <= 553 and rectangle.y1 <= 794,
                        f"Report page {number} image is clipped")
    pdf.set_metadata({"title": "Offline language-model proposals for molecular circuits", "subject": "SC4055 revised CO2 fixed-budget case study", "author": "SC4055 project"})
    pdf.save(dest / "report.pdf", garbage=4, deflate=True)
    preview = dest / "report_previews"
    preview.mkdir(exist_ok=True)
    for i in range(len(pdf)):
        pdf[i].get_pixmap(matrix=pymupdf.Matrix(1, 1), alpha=False).save(preview / f"page_{i+1:02d}.png")
    pdf.close()
    return metrics
