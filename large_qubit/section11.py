"""CPU-only analysis and submission preparation from the preserved experiment."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import zipfile

from .common import digest, read_json, require, sha256_file, workspace_path
from .section11_analysis import (ANALYSIS, SUBMISSION, COMPLETION_SHA, extract_analysis,
                                  export_tables, load_sealed, make_figures, write_json)
from .section11_costs import extract_costs, write_resource_appendix

REGISTRATION = "large_qubit/registrations/section11_analysis_v1/manifest.json"


def anchors(value):
    """Collect explicit file identities, never treat a mentioned path as a hash."""
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            yield value
        for item in value.values():
            yield from anchors(item)
    elif isinstance(value, list):
        for item in value:
            yield from anchors(item)


def source_inventory(root):
    root = Path(root)
    index = read_json(root / SUBMISSION / "artifact_index.json")
    history = read_json(root / SUBMISSION / "historical_summary.json")
    records = list(anchors(index)) + list(anchors(history))
    records += [{"path": p, "sha256": s} for p, s in history["source_files_sha256"].items()]
    current = {}
    for record in records:
        path = workspace_path(root, record["path"])
        require(path.is_file() and sha256_file(path) == record["sha256"], f"Input anchor changed: {record['path']}")
        require("bytes" not in record or path.stat().st_size == record["bytes"], f"Input size changed: {record['path']}")
        current[record["path"]] = {"sha256": record["sha256"], "bytes": path.stat().st_size}
    for pattern in ("large_qubit/section11*.py", "large_qubit/tests/test_section11*.py"):
        for path in sorted(root.glob(pattern)):
            current[path.relative_to(root).as_posix()] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    for relative in ("large_qubit/section11.md", "large_qubit/requirements-analysis.txt"):
        path = root / relative
        current[relative] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return current


def output_inventory(root):
    files = {}
    for directory in (ANALYSIS, SUBMISSION):
        for path in sorted((Path(root) / directory).rglob("*")):
            if path.is_file():
                require(not path.name.endswith((".tmp", ".filepart")), "Transient analysis output remains")
                files[path.relative_to(root).as_posix()] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return files


def check_deliverables(root, analysis, costs):
    import pymupdf
    root = Path(root)
    with pymupdf.open(root / SUBMISSION / "report.pdf") as doc:
        require(len(doc) == 10 and all(len(p.get_text()) > 500 for p in doc), "Report page count/content differs")
        require(all(len(doc[i].get_images()) >= 1 for i in (4, 5, 8)), "Report figure missing")
    with pymupdf.open(root / SUBMISSION / "presentation/presentation.pdf") as doc:
        require(len(doc) == 15 and all(p.get_text().strip() for p in doc), "Presentation page count/content differs")
    with zipfile.ZipFile(root / SUBMISSION / "presentation/presentation.pptx") as deck:
        require(deck.testzip() is None, "Corrupt presentation")
        require(len([p for p in deck.namelist() if p.startswith("ppt/slides/slide") and p.endswith(".xml")]) == 15,
                "Editable presentation slide count differs")
    slides = read_json(root / SUBMISSION / "presentation/slides.json")
    require(slides["total_seconds"] == 900 and len(slides["slides"]) == 15, "Recording schedule differs")
    require(analysis == extract_analysis(root), "Analysis no longer matches saved main results")
    require(costs == extract_costs(root), "Costs no longer match saved evidence")
    require(costs["ordinary_requests"]["actual"] == 31452, "Cumulative accounting differs")
    import csv
    with (root / ANALYSIS / "tables/budget_curves.csv").open(encoding="utf-8", newline="") as handle:
        curves = list(csv.DictReader(handle))
    require(len(curves) == 16384 and sum(x["actual_request"] == "True" for x in curves) == 16382,
            "Curve export omits or manufactures requests")
    return {"report_pages": 10, "presentation_slides": 15, "recording_script_seconds": 900,
            "trajectories": 16, "curve_rows": 16384, "ordinary_requests": 31452,
            "new_training_executed": False, "new_energy_requests": 0}


def verify(root, *, deep=False):
    root = Path(root).resolve()
    manifest = load_sealed(root / REGISTRATION)
    require(manifest["sources"] == source_inventory(root), "Section11 source inventory changed")
    require(manifest["outputs"] == output_inventory(root), "Section11 output inventory changed")
    analysis, costs = read_json(root / ANALYSIS / "analysis.json"), read_json(root / ANALYSIS / "costs.json")
    checks = check_deliverables(root, analysis, costs)
    closure = read_json(root / ANALYSIS / "matrix_verification.json")
    require(closure["result"]["all_passed"] and closure["result"]["completed_sha256"] == COMPLETION_SHA,
            "Saved verification is not bound to this completion")
    if deep:
        from .section9_recovery import verify_matrix
        require(verify_matrix(root) == closure["result"], "Original matrix verification differs")
    return {"all_passed": True, "analysis_complete": True, "submission_package_prepared": True,
            "presentation_recording_complete": False, "course_submission_complete": False,
            "scope": "CPU saved-artifact analysis; actual video recording and submission remain with researcher",
            "deep_experiment_verification_performed_now": deep,
            "source_files": len(manifest["sources"]), "output_files": len(manifest["outputs"]), **checks}


def build(root):
    root = Path(root).resolve()
    if (root / REGISTRATION).exists():
        return verify(root)
    print("Authenticating the original experiment (read-only; may take several minutes)...", flush=True)
    from .section9_recovery import verify_matrix
    closure = verify_matrix(root)
    return build_verified(root, closure)


def build_verified(root, closure):
    """Internal entry after the original verifier completes in the same work session."""
    root = Path(root).resolve()
    require(not (root / REGISTRATION).exists(), "An existing Section11 seal must be verified, not replaced")
    require(closure["all_passed"] and closure["all_trajectories_successful"] and closure["completed_sha256"] == COMPLETION_SHA,
            "Successful original matrix verification is required")
    source_inventory(root)  # authenticate upstream anchors before generating output
    # Keep Matplotlib's optional cache in the workspace, outside sealed results.
    os.environ.setdefault("MPLCONFIGDIR", str(root / "large_qubit/.analysis_cache/matplotlib"))
    analysis, costs = extract_analysis(root), extract_costs(root)
    write_json(root / ANALYSIS / "analysis.json", analysis)
    write_json(root / ANALYSIS / "costs.json", costs)
    write_json(root / ANALYSIS / "matrix_verification.json", {"verified_utc": datetime.now(timezone.utc).isoformat(),
                                                             "result": closure, "scope": "Original full read-only matrix closure; no new science"})
    export_tables(root, analysis, costs)
    write_resource_appendix(root, costs)
    make_figures(root, analysis)
    from .section11_report import build_report
    from .section11_presentation import build_presentation
    layout = build_report(root, analysis, costs)
    build_presentation(root, analysis, costs, root / SUBMISSION / "presentation")
    checks = check_deliverables(root, analysis, costs)
    write_json(root / ANALYSIS / "quality_checks.json", {**checks, "report_layout": layout,
        "all_report_figures_embedded": True, "numerical_analysis_recomputed_from_saved_inputs": True,
        "historical_artifacts_regenerated": False, "scientific_protocol_changed": False})
    write_json(root / ANALYSIS / "environment.json", {"python": sys.version, "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "matplotlib", "PyMuPDF", "Pillow")},
        "local_cpu_analysis_only": True, "gpu_required": False})
    payload = {"schema_version": "section11-analysis-registration-v1", "registration_kind": "post-experiment descriptive analysis and artifact seal; not a prospective experiment amendment",
        "created_utc": datetime.now(timezone.utc).isoformat(), "main_completion_sha256": COMPLETION_SHA,
        "sources": source_inventory(root), "outputs": output_inventory(root),
        "analysis_complete": True, "submission_package_prepared": True, "presentation_recording_complete": False,
        "course_submission_complete": False, "new_energy_requests": 0, "new_training_executed": False,
        "scope": "One pipeline seed, two fixed targets; all outcomes retained; historical LiH separate"}
    write_json(root / REGISTRATION, {"payload": payload, "sha256": digest(payload)})
    return verify(root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "verify", "status"))
    parser.add_argument("--root", default=".")
    parser.add_argument("--deep", action="store_true", help="Also run the original full experiment verifier")
    args = parser.parse_args(argv)
    try:
        if args.action == "build":
            result = build(args.root)
        elif args.action == "status" and not (Path(args.root) / REGISTRATION).exists():
            result = {"analysis_complete": False, "submission_package_prepared": False,
                      "presentation_recording_complete": False, "course_submission_complete": False}
        else:
            result = verify(args.root, deep=args.deep)
        print(json.dumps(result, indent=2))
        return 0
    except Exception as error:
        print("REFUSED: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
