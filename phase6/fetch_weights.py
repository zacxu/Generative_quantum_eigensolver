"""User-run pinned safetensors download. Default prints a plan; --execute downloads.

This command is never invoked with --execute by the Phase 6 implementation audit.
HF API: https://huggingface.co/docs/huggingface_hub/en/package_reference/file_download
"""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re

from phase6.model_registry import (DEFAULT_LOCK, MODEL_SPECS, ROOT, WORKSPACE,
                                   file_hash, validate_model_lock)
from phase6.models import DEFAULT_CACHE

_WEIGHT = re.compile(r"model(?:-\d{5}-of-\d{5})?\.safetensors")


def weight_files(record):
    source = WORKSPACE / record["files"]["source_api.json"]["path"]
    names = {item["rfilename"] for item in json.loads(source.read_text(encoding="utf-8"))["siblings"]}
    weights = sorted(name for name in names if _WEIGHT.fullmatch(name))
    if not weights or ("model.safetensors" in weights and len(weights) > 1):
        raise ValueError("Expected one safetensors checkpoint, either single-file or sharded.")
    allowed = {"config.json", "generation_config.json", "model.safetensors.index.json"}
    files = sorted(set(weights) | (names & allowed))
    if "config.json" not in files or (len(weights) > 1 and "model.safetensors.index.json" not in files):
        raise ValueError("Checkpoint config/shard index is incomplete.")
    return files


def _stream_hash(path):
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_download(model_key, *, cache_dir=DEFAULT_CACHE, lock_path=DEFAULT_LOCK):
    """Authenticate current cached bytes before a user loads checkpoint weights."""
    cache = Path(cache_dir).resolve()
    if not cache.is_relative_to(ROOT/"cache"):
        raise ValueError("Checkpoint cache must remain under phase6/cache.")
    record = validate_model_lock(lock_path)["models"][model_key]
    path = cache/"weight_manifests"/f"{model_key}_{record['revision']}.json"
    if not path.is_file() or not path.with_suffix(".sha256").is_file():
        raise FileNotFoundError(
            f"No completed download manifest for {model_key}. The download may have been interrupted. "
            f"Rerun phase6.fetch_weights --models {model_key} --execute first, then --verify. "
            "Existing cached checkpoint files are reused; do not delete the cache.")
    sidecar = path.with_suffix(".sha256").read_text(encoding="ascii").strip().split()
    if len(sidecar) != 2 or sidecar[1] != path.name or file_hash(path) != sidecar[0]:
        raise ValueError("Cached checkpoint manifest checksum mismatch.")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (manifest.get("schema") != "phase6-weight-download-v1"
            or manifest.get("model_key") != model_key
            or manifest.get("repo_id") != record["repo_id"]
            or manifest.get("revision") != record["revision"]
            or manifest.get("model_lock_sha256") != file_hash(lock_path)
            or set(manifest.get("files", {})) != set(weight_files(record))):
        raise ValueError("Cached checkpoint manifest differs from the pinned model/lock/files.")
    snapshot = cache/("models--"+record["repo_id"].replace("/", "--"))/"snapshots"/record["revision"]
    for name, descriptor in manifest["files"].items():
        current = WORKSPACE/descriptor["path"]
        if current.resolve() != (snapshot/name).resolve() or not current.resolve().is_relative_to(cache):
            raise ValueError("Cached checkpoint file escaped its expected snapshot/cache.")
        if current.stat().st_size != descriptor["bytes"] or _stream_hash(current) != descriptor["sha256"]:
            raise ValueError(f"Cached checkpoint bytes changed: {name}")
        if name in record["files"] and descriptor["sha256"] != record["files"][name]["sha256"]:
            raise ValueError("Cached checkpoint config differs from authenticated model metadata.")
    return {"manifest_sha256": file_hash(path), "revision": record["revision"],
            "model_lock_sha256": file_hash(lock_path),
            "files_sha256": {name: descriptor["sha256"] for name, descriptor in manifest["files"].items()}}


def download(model_keys, *, execute=False, cache_dir=DEFAULT_CACHE, lock_path=DEFAULT_LOCK):
    """No optimizer/model load; execution only downloads exact pinned safe files."""
    cache = Path(cache_dir).resolve()
    if not cache.is_relative_to(ROOT/"cache"):
        raise ValueError("Checkpoint cache must remain under phase6/cache.")
    lock = validate_model_lock(lock_path)
    if not model_keys or len(set(model_keys)) != len(model_keys) or any(key not in MODEL_SPECS for key in model_keys):
        raise ValueError("Select distinct registered model keys.")
    plan = []
    for key in model_keys:
        record = lock["models"][key]
        manifest_path = cache/"weight_manifests"/f"{key}_{record['revision']}.json"
        plan.append({"model_key": key, "repo_id": record["repo_id"], "revision": record["revision"],
            "files": weight_files(record), "cache_dir": str(cache),
            "manifest_path": str(manifest_path), "manifest_exists": manifest_path.exists()})
    if not execute:
        return {"mode": "plan_only", "model_lock_sha256": file_hash(lock_path), "downloads_executed": 0,
                "model_weights_loaded": False, "model_training_executed": False, "models": plan}
    if any(Path(row["manifest_path"]).exists() or Path(row["manifest_path"]).with_suffix(".sha256").exists()
           for row in plan):
        raise FileExistsError("A selected canonical weight manifest exists; refusing overwrite.")
    from huggingface_hub import snapshot_download
    from huggingface_hub.file_download import are_symlinks_supported
    manifests = []
    for row in plan:
        record = lock["models"][row["model_key"]]
        expected = cache/("models--"+row["repo_id"].replace("/", "--"))/"snapshots"/row["revision"]
        # Hub 0.36.x tentatively caches True while probing symlink support.
        # Prime the exact blob/snapshot common directory BEFORE its worker pool:
        # otherwise another worker can see that temporary True and raise Windows
        # error 1314. A completed False probe enables Hub's regular-file fallback.
        symlinks_supported = are_symlinks_supported(str(expected.parent.parent.resolve()))
        snapshot = Path(snapshot_download(repo_id=row["repo_id"], revision=row["revision"],
            cache_dir=str(cache), allow_patterns=row["files"],
            ignore_patterns=["*.py", "*.bin", "*.pt", "*.pth", "*.pkl", "*.pickle"],
            max_workers=4, local_files_only=False)).resolve()
        if snapshot != expected.resolve():
            raise ValueError("HF returned a snapshot outside the selected immutable checkpoint cache.")
        files = {}
        for name in row["files"]:
            path = snapshot/name
            if not path.is_file() or not path.resolve().is_relative_to(cache):
                raise ValueError("Missing or escaped cached weight/config file.")
            fingerprint = _stream_hash(path)
            if name in record["files"] and fingerprint != record["files"][name]["sha256"]:
                raise ValueError("Checkpoint config differs from authenticated pinned metadata.")
            files[name] = {"path": str(path.relative_to(WORKSPACE)).replace("\\", "/"),
                           "bytes": path.stat().st_size, "sha256": fingerprint}
        index = snapshot/"model.safetensors.index.json"
        if index.is_file():
            mapped = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
            if mapped != {name for name in row["files"] if _WEIGHT.fullmatch(name)}:
                raise ValueError("Safetensors shard index differs from the exact downloaded shard population.")
        manifest = {"schema": "phase6-weight-download-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "model_key": row["model_key"], "repo_id": row["repo_id"], "revision": row["revision"],
            "model_lock_sha256": file_hash(lock_path), "files": files,
            "source_sha256": file_hash(Path(__file__)), "requested_by": "explicit user --execute",
            "cache_symlinks_supported": symlinks_supported,
            "model_weights_loaded": False, "model_training_executed": False, "optimizer_steps": 0}
        path = Path(row["manifest_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
        with path.with_suffix(".sha256").open("x", encoding="ascii") as handle:
            handle.write(file_hash(path)+"  "+path.name+"\n")
        manifests.append(str(path))
    return {"mode": "download_executed_by_user", "downloads_executed": len(manifests),
            "manifests": manifests, "model_weights_loaded": False, "model_training_executed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True, choices=["all", *MODEL_SPECS])
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--execute", action="store_true", help="Explicitly download the planned checkpoint weights")
    parser.add_argument("--verify", action="store_true", help="Authenticate existing local weight manifests; no download")
    args = parser.parse_args()
    if "all" in args.models and len(args.models) != 1:
        parser.error("Use 'all' alone, or select explicit model keys.")
    selected = list(MODEL_SPECS) if args.models == ["all"] else args.models
    if args.verify and args.execute:
        parser.error("--verify and --execute are mutually exclusive.")
    try:
        result = ({"mode": "verify_only", "models": {key: verify_download(key, cache_dir=args.cache_dir, lock_path=args.lock)
                  for key in selected}} if args.verify else
                  download(selected, execute=args.execute, cache_dir=args.cache_dir, lock_path=args.lock))
    except FileNotFoundError as error:
        parser.exit(1, f"phase6.fetch_weights: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
