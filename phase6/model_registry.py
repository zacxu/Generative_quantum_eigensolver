"""Immutable checkpoint identities and small, authenticated HF metadata only.

The fetch command never requests checkpoint weights or repository Python code.
Official sources: https://huggingface.co/docs/huggingface_hub/en/package_reference/hf_api
and each pinned repository's README/config/tokenizer files recorded in the lock.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
DEFAULT_LOCK = ROOT / "model_lock.json"
DEFAULT_CACHE = ROOT / "cache/models"
MODEL_SPECS = {
    "qwen3_4b": {"repo_id": "Qwen/Qwen3-4B", "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
                  "model_type": "qwen3", "license": "apache-2.0", "interface": "chat",
                  "template_kwargs": {"enable_thinking": False}},
    "smollm3_3b": {"repo_id": "HuggingFaceTB/SmolLM3-3B", "revision": "a07cc9a04f16550a088caea529712d1d335b0ac1",
                    "model_type": "smollm3", "license": "apache-2.0", "interface": "chat",
                    "template_kwargs": {"enable_thinking": False}, "template_clock_date": "2026-10-18"},
    "phi4_mini": {"repo_id": "microsoft/Phi-4-mini-instruct", "revision": "cfbefacb99257ffa30c83adab238a50856ac3083",
                   "model_type": "phi3", "license": "mit", "interface": "chat", "template_kwargs": {}},
    "gpt2_xl": {"repo_id": "openai-community/gpt2-xl", "revision": "15ea56dee5df4983c59b2538573817e1667135e2",
                 "model_type": "gpt2", "license": "mit", "interface": "completion", "template_kwargs": {}},
}
ALLOWED_FILES = frozenset({"README.md", "LICENSE", "LICENSE.txt", "LICENSE.md", "NOTICE.md",
    "config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json",
    "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json", "chat_template.jinja"})


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def get_spec(model_key):
    if model_key not in MODEL_SPECS:
        raise ValueError(f"Unregistered model: {model_key}")
    return json.loads(json.dumps(MODEL_SPECS[model_key]))


def _request(url):
    for attempt in range(4):
        try:
            with urlopen(Request(url, headers={"User-Agent": "phase6-metadata-audit/1"}), timeout=40) as response:
                # No legitimate whitelisted metadata/tokenizer file needs more than 40 MB.
                payload = response.read(40_000_001)
            break
        except (OSError, TimeoutError):
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    if len(payload) > 40_000_000:
        raise ValueError("Metadata response exceeded 40 MB; no weight download is permitted.")
    return payload


def _fetch_one(model_key, cache_dir):
    spec = get_spec(model_key)
    api_url = f"https://huggingface.co/api/models/{spec['repo_id']}/revision/{spec['revision']}"
    raw = _request(api_url)
    info = json.loads(raw)
    if info.get("sha") != spec["revision"]:
        raise ValueError(f"Pinned API revision mismatch for {model_key}.")
    if info.get("cardData", {}).get("license") != spec["license"]:
        raise ValueError(f"Declared license mismatch for {model_key}.")
    names = sorted(item["rfilename"] for item in info["siblings"] if item["rfilename"] in ALLOWED_FILES)
    required = {"README.md", "config.json", "tokenizer_config.json", "tokenizer.json"}
    if not required.issubset(names):
        raise ValueError(f"Required small metadata missing for {model_key}.")
    directory = Path(cache_dir).resolve() / model_key / spec["revision"]
    directory.mkdir(parents=True, exist_ok=True)
    payloads = {"source_api.json": (raw, api_url)}
    for name in names:
        url = f"https://huggingface.co/{spec['repo_id']}/resolve/{spec['revision']}/{name}"
        payloads[name] = (_request(url), url)
    files = {}
    for name, (payload, url) in payloads.items():
        path = directory / name
        if path.exists() and path.read_bytes() != payload:
            raise ValueError(f"Refusing to replace changed pinned metadata: {path}")
        path.write_bytes(payload)
        files[name] = {"path": str(path.relative_to(WORKSPACE)).replace("\\", "/"),
                       "sha256": sha256(payload).hexdigest(), "bytes": len(payload), "source_url": url}
    config = json.loads(payloads["config.json"][0])
    tokconfig = json.loads(payloads["tokenizer_config.json"][0])
    if config.get("model_type") != spec["model_type"]:
        raise ValueError(f"Architecture mismatch for {model_key}.")
    template_name = "chat_template.jinja" if "chat_template.jinja" in payloads else "tokenizer_config.json"
    template = (payloads[template_name][0].decode() if template_name == "chat_template.jinja"
                else tokconfig.get("chat_template"))
    if spec["interface"] == "chat" and not isinstance(template, str):
        raise ValueError(f"No native chat template for {model_key}.")
    return {**spec, "model_key": model_key, "metadata_dir": str(directory.relative_to(WORKSPACE)).replace("\\", "/"),
        "source_api_url": api_url, "files": files, "config_model_type": config["model_type"],
        "config_context_limit": config.get("max_position_embeddings", config.get("n_positions")),
        "tokenizer_declared_context_limit": tokconfig.get("model_max_length"),
        "native_chat_template_sha256": sha256(template.encode()).hexdigest() if template else None,
        "template_source_file": template_name if template else None,
        "license_evidence": "LICENSE" if "LICENSE" in files else "README.md front-matter license",
        "trust_remote_code": False, "large_weights_downloaded": False}


def write_model_lock(lock_path=DEFAULT_LOCK, cache_dir=DEFAULT_CACHE):
    lock_path, cache_dir = Path(lock_path).resolve(), Path(cache_dir).resolve()
    if not cache_dir.is_relative_to(ROOT) or not lock_path.is_relative_to(ROOT):
        raise ValueError("Model metadata cache and lock must remain inside phase6.")
    if lock_path.exists() or lock_path.with_suffix(".sha256").exists():
        raise FileExistsError("Model lock or sidecar already exists; refusing overwrite.")
    with ThreadPoolExecutor(max_workers=4) as executor:
        records = list(executor.map(lambda key: _fetch_one(key, cache_dir), MODEL_SPECS))
    lock = {"schema": "phase6-model-lock-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "shared_context_limit": 1024, "native_implementations_only": True,
            "model_training_executed": False, "checkpoint_weights_downloaded": False,
            "models": {record["model_key"]: record for record in records}}
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(lock, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    lock_path.with_suffix(".sha256").write_text(file_hash(lock_path)+"  "+lock_path.name+"\n", encoding="ascii")
    return validate_model_lock(lock_path)


def validate_model_lock(lock_path=DEFAULT_LOCK):
    path = Path(lock_path).resolve()
    expected = path.with_suffix(".sha256").read_text(encoding="ascii").strip().split()
    if len(expected) != 2 or expected[1] != path.name or file_hash(path) != expected[0]:
        raise ValueError("Model lock checksum mismatch.")
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("schema") != "phase6-model-lock-v1" or set(lock.get("models", {})) != set(MODEL_SPECS):
        raise ValueError("Unexpected model registry panel/schema.")
    if lock.get("shared_context_limit") != 1024 or not lock.get("native_implementations_only"):
        raise ValueError("Shared context or native-implementation policy changed.")
    for key, spec in MODEL_SPECS.items():
        record = lock["models"][key]
        if any(record.get(field) != value for field, value in spec.items()):
            raise ValueError(f"Registry identity changed for {key}.")
        if not re.fullmatch(r"[0-9a-f]{40}", record["revision"]):
            raise ValueError("Immutable checkpoint commit required.")
        directory = (WORKSPACE / record["metadata_dir"]).resolve()
        if not directory.is_relative_to(ROOT) or record.get("trust_remote_code") is not False:
            raise ValueError("Unsafe metadata directory or remote code policy.")
        for name, descriptor in record["files"].items():
            if name not in ALLOWED_FILES | {"source_api.json"}:
                raise ValueError("Nonmetadata repository file in model lock.")
            metadata = (WORKSPACE / descriptor["path"]).resolve()
            if metadata.parent != directory or metadata.name != name:
                raise ValueError("Metadata path escaped its pinned directory.")
            if file_hash(metadata) != descriptor["sha256"] or metadata.stat().st_size != descriptor["bytes"]:
                raise ValueError(f"Pinned metadata changed: {metadata}")
        config = json.loads((directory/"config.json").read_text(encoding="utf-8"))
        if config.get("model_type") != spec["model_type"]:
            raise ValueError("Native architecture identity changed.")
    return lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--verify", action="store_true", help="Offline verification; no download.")
    args = parser.parse_args()
    lock = validate_model_lock(args.lock) if args.verify else write_model_lock(args.lock, args.cache_dir)
    print(json.dumps({"all_passed": True, "model_keys": list(lock["models"]),
                      "lock_sha256": file_hash(args.lock), "checkpoint_weights_downloaded": False}))


if __name__ == "__main__":
    main()
