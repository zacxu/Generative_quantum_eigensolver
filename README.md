# Generative quantum eigensolver: CO2 study

Source code and upload index for the completed **30-qubit CO2/STO-3G study**: offline, energy-weighted LoRA adaptation of GPT-2 XL and Qwen3-4B to generate molecular quantum circuits. All reported energies were evaluated with classical simulation.

The experiment used one pipeline seed, an accepted corpus of **6,144 fitting circuits plus 384 separate diagnostics**, three adapter fits, and eight methods on two held-out bond lengths. All **16 main evaluation trajectories** completed. This is a molecular case study; the results do not establish quantum advantage or generalisation across molecules.

## What a fresh download can run

**GitHub plus the complete pinned Hugging Face dataset reproduces the saved numerical results.** GitHub alone contains the source and isolated tests. The original production commands require additional authenticated inputs from the preserved research workspace.

| Operation | Fresh-download status | Required files |
| --- | --- | --- |
| Source smoke tests | Verified: 19 passed | GitHub only; Python standard library |
| Saved result and resource reanalysis | Verified: all 16 trajectories match exactly | Complete Hugging Face dataset snapshot |
| Ledger chains and proposal distributions | Verified: all 16 distributions match exactly | Same dataset plus NumPy |
| Read the finished research reports | Available | Two PDFs in the dataset's `reports/` directory |
| Rebuild the original PDFs | Requires additional inputs | Original manuscript parts and historical figure/evidence inputs |
| Rerun original training, validation or main evaluation | Requires additional inputs | Original preparation records, registration/archive closure, token caches, checkpoint/run state and pinned pretrained bases |

The fresh-download audit on **16 November 2026** fetched all **191 GitHub files** and **538 dataset files**, checked their identities, and ran both reanalysis modes in a separate Python 3.12 environment. It validated the 6,144 fitting rows and 384 diagnostic rows, and reconstructed accounting for **31,452 ordinary requests**. No models were loaded and no training or new energy evaluations ran. Reanalysis checks the released data and results; it does not establish that a fresh stochastic experiment would produce identical outcomes.

## Where everything is uploaded

| Location | Uploaded contents | Version |
| --- | --- | --- |
| [GitHub: zacxu/Generative_quantum_eigensolver](https://github.com/zacxu/Generative_quantum_eigensolver) | Current research source, tests, launch scripts, small scientific definitions and dependency pins; this upload index | `main`; exact file identities in [source_manifest.json](source_manifest.json) |
| [Hugging Face: research release](https://huggingface.co/datasets/bruhrelax/gqe-co2-30q-research-v1) | Accepted corpus, diagnostic data, scientific inputs, raw evaluation ledgers, saved analyses, resource accounting, reports, original research plan and provenance; includes the frozen research source snapshot | [064e24614a9092cac48b75e735693391b2b30c29](https://huggingface.co/datasets/bruhrelax/gqe-co2-30q-research-v1/tree/064e24614a9092cac48b75e735693391b2b30c29) |
| [Hugging Face: GPT-2 XL uniform LoRA](https://huggingface.co/bruhrelax/gqe-co2-gpt2-xl-uniform-lora) | Final uniform-objective adapter, configuration, model card and provenance; 78.7 MB | [c4b6628e441f7c54a3ca45e2131baf14d1a5afd9](https://huggingface.co/bruhrelax/gqe-co2-gpt2-xl-uniform-lora/tree/c4b6628e441f7c54a3ca45e2131baf14d1a5afd9) |
| [Hugging Face: GPT-2 XL weighted LoRA](https://huggingface.co/bruhrelax/gqe-co2-gpt2-xl-weighted-lora) | Final energy-weighted adapter, configuration, model card and provenance; 78.7 MB | [4a3177335f3c01982c4fbd906f841a3bbed5b407](https://huggingface.co/bruhrelax/gqe-co2-gpt2-xl-weighted-lora/tree/4a3177335f3c01982c4fbd906f841a3bbed5b407) |
| [Hugging Face: Qwen3-4B weighted LoRA](https://huggingface.co/bruhrelax/gqe-co2-qwen3-4b-weighted-lora) | Final energy-weighted adapter, configuration, model card and provenance; 132.2 MB | [9da201ba59110700426b539a4dff6d3788c33037](https://huggingface.co/bruhrelax/gqe-co2-qwen3-4b-weighted-lora/tree/9da201ba59110700426b539a4dff6d3788c33037) |

As checked on **16 November 2026**, the GitHub repository is private and all four Hugging Face repositories are public. The Hugging Face release contains 567 verified files across the four repositories, approximately 464.3 MB in total. The research dataset repository accounts for 174.7 MB.

Read the [research reports](https://huggingface.co/datasets/bruhrelax/gqe-co2-30q-research-v1/tree/064e24614a9092cac48b75e735693391b2b30c29/reports), [dataset reproduction guide](https://huggingface.co/datasets/bruhrelax/gqe-co2-30q-research-v1/blob/064e24614a9092cac48b75e735693391b2b30c29/REPRODUCE.md) and the three model cards for the scientific results and usage details.

## Code layout

| Path | Purpose |
| --- | --- |
| `large_qubit/` | Physical definitions, circuit grammar, energy evaluator, shared-corpus acquisition, LoRA fits, validation, main evaluation and analysis |
| `large_qubit/tests/` | Unit tests and isolated protocol fixtures |
| `large_qubit/scripts/` | Original detached tmux launchers for the GPU-server workflow |
| `large_qubit/catalogues/co2_screened_v1.json` | Screened operation catalogue used by the study |
| `large_qubit/requirements-*.txt` | Separate recorded analysis, chemistry and model environments |
| `phase3/`, `phase6/`, `tools/` | Only the small existing modules reused by the current implementation and its tests |

The source is preserved byte for byte. The manifest records its checksums and the pinned Hugging Face release identities. `.gitignore` uses an explicit allowlist, so local data, weights, outputs, historical guides and coursework stay out of GitHub. New source directories or file types require an explicit allowlist change; refresh the source manifest when selected files change.

## Check the source locally

From the repository root, this standard-library smoke check runs **19 isolated tests**, without loading models or evaluating molecular energies:

```bash
python -B -S -m unittest large_qubit.tests.test_section11_analysis large_qubit.tests.test_section11_costs large_qubit.tests.test_section11_package
```

Install only the environment needed for a particular research operation. The chemistry environment is separate from the CUDA model environment; CUDA-specific PyTorch installation depends on the target machine. The complete test suite includes numerical and model tests and requires those additional dependencies.

## Reproduce the saved analysis

Download the **complete pinned dataset snapshot** into its own directory, separate from the GitHub source. The snapshot includes a self-contained verifier and reanalysis helper:

```bash
python -m pip install huggingface-hub==0.36.2
hf download bruhrelax/gqe-co2-30q-research-v1 --repo-type dataset --revision 064e24614a9092cac48b75e735693391b2b30c29 --local-dir .research_data/co2-release
python -I -S .research_data/co2-release/release_tools/release_runtime.py reanalyse
```

This verifies the curated checksums, reconstructs the saved result and resource tables, and checks separation of the fitting and diagnostic rows. It uses the standard library, requires no GPU, and starts no training or new energy evaluations. Downloading only selected files will fail the complete-inventory check.

For the additional ledger and proposal checks, use **Python 3.12 or newer** with the pinned NumPy version:

```bash
python -m pip install numpy==2.5.4
python -I .research_data/co2-release/release_tools/release_runtime.py reanalyse --proposals
```

The standard-library check works on Python 3.10 or newer; the pinned NumPy package requires Python 3.12 or newer. Python 3.12 was used for the verified download audit.

The curated release supports saved-result reanalysis. The original full-workspace authentication commands and production launchers also require the historical archives, registrations, checkpoint states and runtime inputs preserved in the original workspace; neither a source-only clone nor the combined curated GitHub/Hugging Face release supplies that complete authenticated workspace. In particular, Sections 7-9 verify their predecessor preparation records and archived source chains; the three exported adapters cannot substitute for resumable training checkpoints. Rebuilding the original PDFs additionally requires unpublished manuscript and historical figure inputs. Use the dataset's reproduction guide for the portable release, and read the uploaded PDFs for the finished reports.

## Using the trained adapters

The three model repositories contain **LoRA adapters**, including all parameters trained in this study. The frozen pretrained weights are obtained from their upstream owners:

| Adapters | Required base | Exact base revision |
| --- | --- | --- |
| Both GPT-2 XL adapters | [openai-community/gpt2-xl](https://huggingface.co/openai-community/gpt2-xl) | [15ea56dee5df4983c59b2538573817e1667135e2](https://huggingface.co/openai-community/gpt2-xl/tree/15ea56dee5df4983c59b2538573817e1667135e2) |
| Qwen3-4B weighted adapter | [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | [1cfa9a7208912126459214e8b04321603b3df60c](https://huggingface.co/Qwen/Qwen3-4B/tree/1cfa9a7208912126459214e8b04321603b3df60c) |

Each model card contains loading instructions. Circuit generation also needs the released prompts, catalogue and grammar constraints. Exact training continuation additionally requires the original optimizer and RNG state, which remain in the local research archive.

## Preserved locally

The full historical workspace, obsolete experiments, intermediate checkpoints, optimizer/RNG states, runtime caches, transfer logs and presentation coursework remain outside this GitHub release. Existing historical files and registrations have been retained. Hugging Face carries the selected scientific data, results and final adapters; upstream repositories supply the full base weights.
