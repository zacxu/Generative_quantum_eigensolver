"""Reuse only authenticated pretrained GPT-2 XL/Qwen3-4B assets, without downloads.

Inspection uses the standard library. Loading is an explicit opt-in, performs
full byte authentication first, and uses native local-only Transformers APIs.
LiH adapters are intentionally not an initialization option for the CO2 branch.
"""

from __future__ import annotations

from hashlib import sha256
from importlib import metadata
import json
from pathlib import Path, PurePosixPath
import re


CONTEXT_LIMIT = 1024
PINNED_LOCK_SHA256 = "fbe40edd99eadb8a8839b6d8d0cfee8d2aeb8f561612ce34c4997ed1a9b7d3f7"
MODEL_SPECS = {
    "gpt2_xl": {
        "repo_id": "openai-community/gpt2-xl",
        "revision": "15ea56dee5df4983c59b2538573817e1667135e2",
        "model_type": "gpt2", "interface": "completion", "template_kwargs": {},
        "weight_manifest_sha256": "a09a38dee647670ef9080fbfaca4c33fdf2276562ef61836c1be488534380967",
    },
    "qwen3_4b": {
        "repo_id": "Qwen/Qwen3-4B",
        "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        "model_type": "qwen3", "interface": "chat",
        "template_kwargs": {"enable_thinking": False},
        "weight_manifest_sha256": "d3bcbe8f51be499526562728f3f821fb6efc145bf191fe10bf21a61f0be63354",
    },
}
_METADATA_NAMES = frozenset({
    "source_api.json", "README.md", "LICENSE", "LICENSE.txt", "LICENSE.md",
    "NOTICE.md", "config.json", "generation_config.json", "tokenizer_config.json",
    "tokenizer.json", "vocab.json", "merges.txt", "special_tokens_map.json",
    "added_tokens.json", "chat_template.jinja",
})
_WEIGHT = re.compile(r"model(?:-\d{5}-of-\d{5})?\.safetensors")
_SHA = re.compile(r"[0-9a-f]{64}")
_VERSIONS = {
    "transformers": "4.57.6", "tokenizers": "0.22.2",
    "huggingface-hub": "0.36.2", "safetensors": "0.7.0",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _hash(path):
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object_pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(path):
    def reject(value):
        raise ValueError(f"Non-finite JSON value: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=_object_pairs, parse_constant=reject)


def _path(workspace, relative, allowed):
    _require(isinstance(relative, str) and "\\" not in relative,
             "Asset paths must be portable relative POSIX paths.")
    parsed = PurePosixPath(relative)
    _require(not parsed.is_absolute() and ":" not in relative
             and all(part not in (".", "..") for part in relative.split("/")),
             "Absolute or escaping asset path.")
    candidate = workspace.joinpath(*parsed.parts)
    _require(candidate.resolve().is_relative_to(allowed.resolve()),
             f"Asset path or symlink escaped its allowed directory: {relative}")
    return candidate


def _checked_json(path, expected_hash):
    _require(path.is_file(), f"Missing authenticated manifest: {path}")
    sidecar_path = path.with_suffix(".sha256")
    _require(sidecar_path.resolve().is_relative_to(path.parent.resolve()),
             "Manifest sidecar symlink escaped its directory.")
    _require(sidecar_path.is_file(), f"Missing manifest checksum: {sidecar_path}")
    sidecar = sidecar_path.read_text(encoding="ascii").strip().split()
    actual = _hash(path)
    _require(sidecar == [expected_hash, path.name] and actual == expected_hash,
             f"Pinned manifest checksum mismatch: {path.name}")
    return _json(path)


def _descriptor(path, descriptor, *, full_hash):
    _require(isinstance(descriptor, dict) and type(descriptor.get("bytes")) is int
             and descriptor["bytes"] > 0
             and isinstance(descriptor.get("sha256"), str)
             and _SHA.fullmatch(descriptor["sha256"]) is not None,
             f"Invalid asset descriptor: {path.name}")
    _require(path.is_file(), f"Missing cached asset: {path}")
    _require(path.stat().st_size == descriptor["bytes"],
             f"Cached asset size changed or file was truncated: {path.name}")
    if full_hash:
        _require(_hash(path) == descriptor["sha256"],
                 f"Cached asset checksum mismatch: {path.name}")


def inspect_assets(workspace, model_key, *, full_hash=False):
    """Authenticate metadata; inspect weight sizes, or fully hash weights if requested.

    A size-only inspection does not certify weight contents. Only the registered
    selected model is inspected: unavailable deferred families are irrelevant.
    Neither mode imports Torch/Transformers, loads a model, or accesses a network.
    """
    _require(model_key in MODEL_SPECS, "Select registered gpt2_xl or qwen3_4b base weights.")
    _require(type(full_hash) is bool, "full_hash must be a boolean.")
    workspace = Path(workspace).resolve()
    _require(workspace.is_dir(), "Workspace directory is missing.")
    cache = _path(workspace, "phase6/cache/models", workspace)
    spec = MODEL_SPECS[model_key]
    lock_path = _path(workspace, "phase6/model_lock.json", workspace / "phase6")
    lock = _checked_json(lock_path, PINNED_LOCK_SHA256)
    _require(lock.get("schema") == "phase6-model-lock-v1"
             and lock.get("shared_context_limit") == CONTEXT_LIMIT
             and lock.get("native_implementations_only") is True,
             "Model lock schema or native/context policy changed.")
    record = lock.get("models", {}).get(model_key, {})
    for field in ("repo_id", "revision", "model_type", "interface", "template_kwargs"):
        _require(record.get(field) == spec[field], f"Pinned model identity changed: {field}")
    _require(record.get("model_key") == model_key and record.get("trust_remote_code") is False,
             "Base-model key or remote-code policy changed.")
    metadata_relative = f"phase6/cache/models/{model_key}/{spec['revision']}"
    _require(record.get("metadata_dir") == metadata_relative, "Unexpected model metadata directory.")
    directory = _path(workspace, metadata_relative, cache)
    files = record.get("files", {})
    required = {"source_api.json", "config.json", "tokenizer_config.json", "tokenizer.json"}
    _require(isinstance(files, dict) and required <= set(files) <= _METADATA_NAMES,
             "Missing required or unsupported metadata file.")
    for name, descriptor in files.items():
        _require(descriptor.get("path") == f"{metadata_relative}/{name}",
                 f"Metadata descriptor escaped its expected path: {name}")
        current = _path(workspace, descriptor["path"], directory)
        _descriptor(current, descriptor, full_hash=True)
    config = _json(directory / "config.json")
    _require(config.get("model_type") == spec["model_type"] and not config.get("auto_map"),
             "Native architecture differs from its pinned implementation.")
    _require(config.get("max_position_embeddings", config.get("n_positions", 0)) >= CONTEXT_LIMIT,
             "Model does not support the shared 1024-token context.")
    tok_config = _json(directory / "tokenizer_config.json")
    template = ((directory / "chat_template.jinja").read_text(encoding="utf-8")
                if record.get("template_source_file") == "chat_template.jinja"
                else tok_config.get("chat_template"))
    if spec["interface"] == "chat":
        _require(isinstance(template, str) and bool(template)
                 and sha256(template.encode()).hexdigest() == record.get("native_chat_template_sha256"),
                 "Qwen native chat template checksum mismatch.")
    else:
        _require(not template and record.get("native_chat_template_sha256") is None,
                 "GPT-2 must use its explicit completion interface.")
    source = _json(directory / "source_api.json")
    _require(source.get("sha") == spec["revision"] and source.get("id") == spec["repo_id"],
             "Source API identity differs from the pinned revision/repository.")
    source_names = {row["rfilename"] for row in source["siblings"]}
    weight_names = {name for name in source_names if _WEIGHT.fullmatch(name)}
    _require(bool(weight_names) and not ("model.safetensors" in weight_names and len(weight_names) > 1),
             "Expected one safetensors checkpoint, single-file or sharded.")
    required_weights = weight_names | (source_names & {
        "config.json", "generation_config.json", "model.safetensors.index.json"})
    _require("config.json" in required_weights
             and (len(weight_names) == 1 or "model.safetensors.index.json" in required_weights),
             "Checkpoint configuration or shard index is incomplete.")
    manifest_relative = f"phase6/cache/models/weight_manifests/{model_key}_{spec['revision']}.json"
    manifest_path = _path(workspace, manifest_relative, cache)
    manifest = _checked_json(manifest_path, spec["weight_manifest_sha256"])
    _require(manifest.get("schema") == "phase6-weight-download-v1"
             and all(manifest.get(key) == record[key] for key in ("model_key", "repo_id", "revision"))
             and manifest.get("model_lock_sha256") == PINNED_LOCK_SHA256
             and set(manifest.get("files", {})) == required_weights,
             "Cached weight manifest differs from the pinned base model/lock/files.")
    snapshot_relative = ("phase6/cache/models/models--" + spec["repo_id"].replace("/", "--")
                         + "/snapshots/" + spec["revision"])
    snapshot = _path(workspace, snapshot_relative, cache)
    for name, descriptor in manifest["files"].items():
        _require(descriptor.get("path") == f"{snapshot_relative}/{name}",
                 f"Weight descriptor escaped its expected snapshot: {name}")
        current = _path(workspace, descriptor["path"], cache)
        _require(current.resolve() == (snapshot / name).resolve(), "Weight snapshot path mismatch.")
        _descriptor(current, descriptor, full_hash=full_hash or name not in weight_names)
        if name in files:
            _require(descriptor["sha256"] == files[name]["sha256"],
                     "Cached configuration differs from authenticated model metadata.")
    index_path = snapshot / "model.safetensors.index.json"
    if "model.safetensors.index.json" in required_weights:
        weight_map = _json(index_path).get("weight_map")
        _require(isinstance(weight_map, dict) and bool(weight_map)
                 and set(weight_map.values()) == weight_names,
                 "Safetensors index does not cover exactly the registered shards.")
    for folder in (directory, snapshot):
        for candidate in folder.iterdir():
            _require(not candidate.name.startswith("adapter_")
                     and not (candidate.suffix in {".bin", ".pt", ".pth", ".pkl", ".pickle"})
                     and not (candidate.suffix == ".safetensors" and candidate.name not in weight_names),
                     f"Unexpected adapter or checkpoint file in base asset directory: {candidate.name}")
    return {
        "schema": "large-qubit-base-assets-v1", "model_key": model_key,
        "repo_id": spec["repo_id"], "revision": spec["revision"],
        "model_type": spec["model_type"], "interface": spec["interface"],
        "template_kwargs": dict(spec["template_kwargs"]), "shared_context_limit": CONTEXT_LIMIT,
        "model_lock_sha256": PINNED_LOCK_SHA256,
        "weight_manifest_sha256": spec["weight_manifest_sha256"],
        "native_chat_template_sha256": record.get("native_chat_template_sha256"),
        "metadata_directory": metadata_relative, "snapshot_directory": snapshot_relative,
        "weight_bytes": sum(manifest["files"][name]["bytes"] for name in weight_names),
        "weight_files_sha256": {name: manifest["files"][name]["sha256"] for name in sorted(weight_names)},
        "metadata_authenticated": True, "weight_sizes_verified": True,
        "weight_contents_authenticated": full_hash, "local_files_only": True,
        "trust_remote_code": False, "initialization": "pretrained_base_only",
        "old_lih_adapters_reused": False, "model_weights_loaded": False,
        "downloads_executed": 0, "optimizer_steps_executed": 0,
    }


def authenticate_assets(workspace, model_key, *, full_hash=True):
    """Full byte authentication by default; explicit false is inspection only."""
    return inspect_assets(workspace, model_key, full_hash=full_hash)


def _require_versions(*, model=False):
    expected = {**_VERSIONS, **({"torch": "2.10.0"} if model else {})}
    actual = {name: metadata.version(name) for name in expected}
    for name, version in expected.items():
        _require(actual[name].split("+", 1)[0] == version,
                 f"Unvalidated package version {name}={actual[name]}; expected {version}.")
    return actual


def _load_binding(workspace, model_key, execute, local_files_only):
    _require(execute is True, "Model/tokenizer loading requires explicit execute=True.")
    _require(local_files_only is True, "Only authenticated local files may be loaded; network downloads are disabled.")
    return authenticate_assets(workspace, model_key, full_hash=True)


def load_tokenizer(workspace, model_key, *, execute=False, local_files_only=True):
    """Explicit native tokenizer load, after authenticating the selected asset set."""
    binding = _load_binding(workspace, model_key, execute, local_files_only)
    packages = _require_versions()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(Path(workspace).resolve() / binding["metadata_directory"]), use_fast=True,
        trust_remote_code=False, local_files_only=True)
    _require(tokenizer.is_fast and tokenizer.eos_token_id is not None,
             "Native fast tokenizer and an explicit EOS token are required.")
    if binding["interface"] == "chat":
        template = tokenizer.get_chat_template()
        _require(isinstance(template, str) and sha256(template.encode()).hexdigest()
                 == binding["native_chat_template_sha256"], "Loaded Qwen template changed.")
    else:
        _require(tokenizer.chat_template is None, "GPT-2 must use completion formatting.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "right"
    tokenizer.model_max_length = CONTEXT_LIMIT
    tokenizer._large_qubit_binding = {**binding, "packages": packages,
                                      "tokenizer_loaded": True}
    return tokenizer


def render_prompt(tokenizer, model_key, task, *, answer_token_reserve):
    """Render the registered interface and reserve a complete answer without truncation.

    The caller must compute answer_token_reserve from its complete answer grammar,
    including EOS. Grammar/catalogue construction belongs to the later phase.
    """
    _require(model_key in MODEL_SPECS, "Unknown model interface.")
    _require(isinstance(task, str) and bool(task), "A nonempty circuit task is required.")
    _require(type(answer_token_reserve) is int and answer_token_reserve > 0,
             "Reserve a positive complete-answer token count including EOS.")
    binding = getattr(tokenizer, "_large_qubit_binding", {})
    _require(binding.get("model_key") == model_key
             and binding.get("revision") == MODEL_SPECS[model_key]["revision"]
             and binding.get("weight_contents_authenticated") is True,
             "Use the authenticated tokenizer for the selected base model.")
    kwargs = dict(MODEL_SPECS[model_key]["template_kwargs"])
    if model_key == "gpt2_xl":
        text = task + "\nAnswer:\n"
        ids = list(tokenizer.encode(text, add_special_tokens=False))
    else:
        template = tokenizer.get_chat_template()
        _require(sha256(template.encode()).hexdigest() == binding["native_chat_template_sha256"],
                 "Tokenizer chat template changed after loading.")
        messages = [{"role": "user", "content": task}]
        text = tokenizer.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=True, **kwargs)
        ids = list(tokenizer.apply_chat_template(messages, tokenize=True,
            add_generation_prompt=True, **kwargs))
        _require(ids == list(tokenizer.encode(text, add_special_tokens=False)),
                 "Native rendered and tokenized prompts disagree.")
    _require(len(ids) + answer_token_reserve <= CONTEXT_LIMIT,
             "Prompt plus complete answer exceeds 1024 tokens; truncation is forbidden.")
    return {"text": text, "input_ids": ids, "prompt_tokens": len(ids),
            "answer_token_reserve": answer_token_reserve, "context_limit": CONTEXT_LIMIT,
            "template_kwargs": kwargs, "model_key": model_key}


def load_base_model(workspace, model_key, *, execute=False, device="cuda",
                    precision="bf16", local_files_only=True):
    """Explicit pretrained-only load. Fresh LoRA attachment is a later fit operation."""
    binding = _load_binding(workspace, model_key, execute, local_files_only)
    packages = _require_versions(model=True)
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    selected = torch.device(device)
    if selected.type == "cpu":
        _require(precision == "float32", "CPU model fixtures require explicit float32.")
        dtype = torch.float32
        runtime = {"device": "cpu", "precision": "float32", "fixture_only": True}
    else:
        _require(selected.type == "cuda" and precision == "bf16", "Production loading requires BF16 CUDA.")
        _require(torch.cuda.is_available(), "CUDA is unavailable.")
        index = selected.index if selected.index is not None else torch.cuda.current_device()
        _require(0 <= index < torch.cuda.device_count(), "Selected CUDA device is unavailable.")
        with torch.cuda.device(index):
            _require(torch.cuda.is_bf16_supported(including_emulation=False), "Native BF16 is unavailable.")
            properties = torch.cuda.get_device_properties(index)
        dtype = torch.bfloat16
        runtime = {"device": f"cuda:{index}", "precision": "bf16", "fixture_only": False,
                   "device_name": properties.name, "total_memory_bytes": properties.total_memory}
    config = AutoConfig.from_pretrained(str(Path(workspace).resolve() / binding["metadata_directory"]),
        trust_remote_code=False, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(Path(workspace).resolve() / binding["snapshot_directory"]), config=config,
        trust_remote_code=False, local_files_only=True, dtype=dtype,
        use_safetensors=True, attn_implementation="eager")
    _require(model.config.model_type == binding["model_type"] and not hasattr(model, "peft_config"),
             "Loaded model is not the selected native pretrained base.")
    model.to(torch.device(runtime["device"]))
    model.requires_grad_(False)
    model.eval()
    model._large_qubit_binding = {**binding, "model_weights_loaded": True,
                                  "packages": packages, "runtime": runtime,
                                  "attention_implementation": "eager"}
    return model
