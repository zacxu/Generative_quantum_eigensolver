"""CO2 definitions and authentication; SCF requires an explicit --execute flag.

Training is available separately through ``python -m large_qubit.section7``;
this command's Section7 action only previews that researcher-run workflow.
"""

import argparse
import json
from pathlib import Path
import sys

from .common import read_json
from .protocol import MODEL_REVISIONS, build_study, planned_jobs, validate_study
from .registration import DEFAULT_REGISTRATION


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="Project root (default: parent of this package)")
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("plan", help="Validate and display the declarative 16-job matrix; executes nothing")
    sub.add_parser("definitions", help="Validate and display the Section 3 physical specification")
    sub.add_parser("section7", help="Preview the three fixed-budget fits; executes no training")
    for name in ("verify", "status"):
        p = sub.add_parser(name, help="Authenticate registered foundation and historical reuse")
        p.add_argument("--registration", default=DEFAULT_REGISTRATION)
    p = sub.add_parser("register", help="One-time creation; refuses overwrite")
    p.add_argument("--output", default=DEFAULT_REGISTRATION)
    p = sub.add_parser("models", help="Verify cached assets without loading models or downloading")
    p.add_argument("--full-weights", action="store_true", help="Also hash all GPT2/Qwen tensor bytes")
    p = sub.add_parser("physics", help="Preview or explicitly run bounded Section 3 SCF/integral validation")
    p.add_argument("--execute", action="store_true", help="Run six geometry SCF/stability checks; no FCI/training")
    p.add_argument("--output", default="large_qubit/physics/section3_v3")
    p.add_argument("--threads", type=int, default=1)
    p = sub.add_parser("physics-verify", help="Authenticate saved Section 3 physical realizations")
    p.add_argument("--physics-root", default="large_qubit/physics/section3_v3")
    p.add_argument("--numerical-integrity", action="store_true", help="Also check saved arrays; no SCF rerun")
    p = sub.add_parser("section4", help="Preview or explicitly run the local CPU evaluator/reference gate")
    p.add_argument("--execute", action="store_true", help="Run bounded physics and timing; no model training")
    p.add_argument("--output", default="large_qubit/validation/section4_v1")
    p = sub.add_parser("section4-verify", help="Authenticate a saved Section 4 scientific gate")
    p.add_argument("--section4-root", default="large_qubit/validation/section4_v1")
    p.add_argument("--numerical-integrity", action="store_true", help="Recompute saved reference residual; no new eigensolve")
    p = sub.add_parser("section5-prepare", help="Preview or create the pre-pilot catalogue and tokenizer certificates")
    preparation_mode = p.add_mutually_exclusive_group()
    preparation_mode.add_argument("--execute", action="store_true", help="Use saved training integrals and cached tokenizers; no candidate energies")
    preparation_mode.add_argument("--verify", action="store_true", help="Recompute saved analytic screening and authenticate context certificates; no energy queries")
    p = sub.add_parser("section5", help="Preview or execute the bounded local CPU expressivity/compact VQE pilot")
    p.add_argument("--execute", action="store_true", help="Run declared training-geometry physics pilot; no model training")
    p.add_argument("--output", default="large_qubit/validation/section5_v1")
    p = sub.add_parser("section5-verify", help="Authenticate saved Section5 catalogue/context/pilot evidence")
    p.add_argument("--section5-root", default="large_qubit/validation/section5_v1")
    p.add_argument("--numerical-integrity", action="store_true", help="Reprepare saved winner states; no new energy queries")
    p = sub.add_parser("section6-prepare", help="Preview, freeze or authenticate the shared-corpus acquisition policy")
    preparation_mode = p.add_mutually_exclusive_group()
    preparation_mode.add_argument("--execute", action="store_true", help="Freeze source, policy and diagnostic reservations; no energies or training")
    preparation_mode.add_argument("--verify", action="store_true", help="Replay reserved metadata and authenticate the preparation; no energies")
    p = sub.add_parser("section6", help="Preview or explicitly acquire the shared training-only corpus on CPU")
    p.add_argument("--execute", action="store_true", help="Run the frozen budgeted acquisition; never model training")
    p.add_argument("--resume", action="store_true", help="Explicitly resume the same campaign and charged frontier")
    p.add_argument("--output", default="large_qubit/validation/section6_v1")
    p = sub.add_parser("section6-verify", help="Authenticate every acquisition record and the frozen shared release")
    p.add_argument("--section6-root", default="large_qubit/validation/section6_v1")
    p.add_argument("--numerical-integrity", action="store_true", help="Replay fixed saved-label states without new energy scoring")
    args = parser.parse_args(argv)
    try:
        if args.action == "section7":
            from .section7 import build_section7_policy
            result = {"mode": "preview_only", "policy": build_section7_policy(),
                      "production_training_executed": False, "production_gpu_preflight_executed": False}
        elif args.action == "section6-prepare":
            from .section6 import prepare_section6, verify_preparation, build_section6_policy
            if args.verify:
                _, catalogue, _, _, preparation = verify_preparation(args.root)
                result = {"all_passed": True, "catalogue_sha256": catalogue['sha256'],
                          "diagnostics_reserved": 384, "candidate_energy_requests": 0,
                          "source_bindings": preparation['source_bindings'], "model_training_executed": False}
            elif args.execute:
                result = prepare_section6(args.root)
            else:
                catalogue = read_json(args.root / 'large_qubit/catalogues/co2_screened_v1.json')
                result = {"mode": "preview_only", "policy": build_section6_policy(catalogue),
                          "candidate_energy_requests": 0, "model_training_executed": False}
        elif args.action == "section6":
            from .section6 import run_section6, build_section6_policy
            if args.execute:
                result = run_section6(args.root, args.output, resume=args.resume)
            else:
                catalogue = read_json(args.root / 'large_qubit/catalogues/co2_screened_v1.json')
                result = {"mode": "preview_only", "policy": build_section6_policy(catalogue),
                          "output": args.output, "resume_requested": args.resume,
                          "scientific_work_started": False, "model_training_executed": False}
        elif args.action == "section6-verify":
            from .section6 import verify_section6
            result = verify_section6(args.root, args.section6_root, numerical_integrity=args.numerical_integrity)
        elif args.action == "section5-prepare":
            from .section5 import prepare_section5, build_section5_policy, verify_preparation
            if args.verify:
                _, catalogue, _, _ = verify_preparation(args.root, numerical_integrity=True)
                result = {"all_passed": True, "catalogue_sha256": catalogue['sha256'],
                          "candidate_energy_requests": 0, "model_training_executed": False}
            else:
                result = (prepare_section5(args.root) if args.execute else
                      {"mode": "preview_only", "policy": build_section5_policy(),
                       "execution_requires": "--execute with the cached tokenizer environment",
                       "candidate_energy_requests": 0, "model_training_executed": False})
        elif args.action == "section5":
            from .section5 import run_section5, build_section5_policy
            result = (run_section5(args.root, args.output) if args.execute else
                      {"mode": "preview_only", "policy": build_section5_policy(), "output": args.output,
                       "execution_requires": "--execute in Linux/WSL chemistry environment",
                       "scientific_work_started": False, "model_training_executed": False})
        elif args.action == "section5-verify":
            from .section5 import verify_section5
            result = verify_section5(args.root, args.section5_root, numerical_integrity=args.numerical_integrity)
        elif args.action == "section4":
            from .section4 import build_section4_policy, run_section4
            result = (run_section4(args.root, args.output) if args.execute else
                      {"mode": "preview_only", "scientific_work_started": False,
                       "policy": build_section4_policy(), "output": args.output,
                       "execution_requires": "--execute in Linux/WSL chemistry environment"})
        elif args.action == "section4-verify":
            from .section4 import verify_section4
            result = verify_section4(args.root, args.section4_root,
                                     numerical_integrity=args.numerical_integrity)
        elif args.action == "definitions":
            from .registration import validate_physical_definition
            result = validate_physical_definition(args.root)
        elif args.action == "physics":
            if args.execute:
                from .physics import build_physics
                result = build_physics(args.root, args.output, threads=args.threads)
            else:
                result = {"mode": "preview_only", "scientific_calculations_started": False,
                          "output": args.output, "threads": args.threads,
                          "scope": "Section 3 SCF, stability, orbital alignment and integral consistency",
                          "execution_requires": "--execute in the isolated chemistry environment",
                          "training_or_correlated_reference": False}
        elif args.action == "physics-verify":
            from .physics import verify_physics
            result = verify_physics(args.root, args.physics_root,
                                    numerical_integrity=args.numerical_integrity)
        elif args.action == "plan":
            study = validate_study(read_json(args.root / "large_qubit/study.json"))
            result = {"mode": "plan_only", "execution_ready": False,
                      "benchmark_id": study["benchmark_id"],
                      "fits": list(study["training_conditions"]), "jobs": planned_jobs(study),
                      "budget": study["budget"], "pending": list(study["pending_bindings"])}
        elif args.action == "models":
            from .models import inspect_assets
            result = {"models": {key: inspect_assets(args.root, key, full_hash=args.full_weights)
                                 for key in MODEL_REVISIONS},
                      "models_loaded": 0, "downloads": 0}
        else:
            from .registration import register, status, verify_registration
            if args.action == "register":
                result = register(args.root, args.output)
            elif args.action == "verify":
                result = verify_registration(args.root, args.registration)
            else:
                result = status(args.root, args.registration)
        print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False))
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
