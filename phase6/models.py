"""Pinned native model loading and inspected LoRA adapters; never runs training.

LoRA orientation/initialization source:
https://huggingface.co/docs/peft/en/package_reference/lora
Native model sources are the installed, version-recorded Transformers modules.
"""

from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import re

from phase6.model_registry import (DEFAULT_CACHE, DEFAULT_LOCK, MODEL_SPECS,
                                   WORKSPACE, canonical_hash, file_hash, get_spec,
                                   validate_model_lock)

LORA_DEFAULTS = {"rank": 16, "alpha": 32, "dropout": 0.05}
_PATTERNS = {
    "gpt2": r"transformer\.h\.\d+\.(?:attn\.(?:c_attn|c_proj)|mlp\.(?:c_fc|c_proj))",
    "qwen3": r"model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))",
    "smollm3": r"model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))",
    "phi3": r"model\.layers\.\d+\.(?:self_attn\.(?:qkv_proj|o_proj)|mlp\.(?:gate_up_proj|down_proj))",
}


def package_versions():
    return {name: importlib.metadata.version(name) for name in
            ("torch", "transformers", "peft", "accelerate", "safetensors")}


def check_runtime(device="cuda", precision="bf16", require_h200=False):
    """Explicit CPU fixture or BF16 CUDA policy; no silent precision fallback."""
    import torch
    selected = torch.device(device)
    if selected.type == "cpu":
        if precision not in ("float32", "fp32") or require_h200:
            raise ValueError("CPU fixtures require float32 and cannot certify H200 execution.")
        return {"device": "cpu", "precision": "float32", "cuda_available": torch.cuda.is_available(),
                "h200_verified": False, "fixture_only": True, "packages": package_versions()}
    if selected.type != "cuda" or precision not in ("bf16", "bfloat16"):
        raise ValueError("Production model loading requires explicitly selected BF16 CUDA.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; BF16 CUDA loading cannot proceed.")
    index = selected.index if selected.index is not None else torch.cuda.current_device()
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError("CUDA device index is unavailable.")
    with torch.cuda.device(index):
        if not torch.cuda.is_bf16_supported(including_emulation=False):
            raise RuntimeError("Selected CUDA device lacks native BF16 support.")
        props = torch.cuda.get_device_properties(index)
    h200 = "H200" in props.name.upper()
    if require_h200 and not h200:
        raise RuntimeError(f"H200 execution requested, actual device is {props.name}.")
    return {"device": f"cuda:{index}", "precision": "bfloat16", "cuda_available": True,
            "device_name": props.name, "compute_capability": [props.major, props.minor],
            "total_memory_bytes": props.total_memory, "h200_verified": h200, "fixture_only": False,
            "torch_cuda_version": torch.version.cuda, "packages": package_versions()}


def _metadata(model_key, lock_path):
    record = validate_model_lock(lock_path)["models"][model_key]
    return record, WORKSPACE / record["metadata_dir"]


def load_tokenizer(model_key, lock_path=DEFAULT_LOCK, cache_dir=DEFAULT_CACHE,
                   local_files_only=True):
    """Use authenticated local native tokenizer; downloads belong to registry CLI."""
    from transformers import AutoTokenizer
    record, directory = _metadata(model_key, lock_path)
    tokenizer = AutoTokenizer.from_pretrained(str(directory), use_fast=True, trust_remote_code=False,
                                             local_files_only=True, cache_dir=str(cache_dir))
    if not tokenizer.is_fast:
        raise ValueError("Verified byte-level grammar requires the native fast tokenizer.")
    if record["interface"] == "chat":
        template = tokenizer.get_chat_template()
        if sha256(template.encode()).hexdigest() != record["native_chat_template_sha256"]:
            raise ValueError("Loaded native chat template differs from authenticated metadata.")
    elif tokenizer.chat_template is not None:
        raise ValueError("GPT-2 must use the explicit completion interface.")
    if tokenizer.eos_token_id is None:
        raise ValueError("An explicit genuine EOS target is required.")
    # No vocabulary/embedding changes. Padding EOS does not imply masking real EOS labels.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "right"
    tokenizer.model_max_length = 1024
    tokenizer._phase6_binding = {"model_key": model_key, "revision": record["revision"],
        "model_lock_sha256": file_hash(lock_path), "shared_context_limit": 1024,
        "padding_side": "left", "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id, "trust_remote_code": False,
        "native_template_sha256": record["native_chat_template_sha256"]}
    return tokenizer


def resolve_lora_targets(model):
    """Resolve every registered attention/MLP projection, excluding embeddings/head."""
    import torch
    from transformers.pytorch_utils import Conv1D
    kind = model.config.model_type
    if kind not in _PATTERNS or hasattr(model, "peft_config"):
        raise ValueError("Resolve targets on an unwrapped registered native architecture.")
    pattern = re.compile(_PATTERNS[kind])
    target_type = Conv1D if kind == "gpt2" else torch.nn.Linear
    targets = []
    for name, module in model.named_modules():
        if pattern.fullmatch(name):
            if not isinstance(module, target_type):
                raise TypeError(f"Unexpected native projection type: {name}: {type(module)}")
            shape = list(module.weight.shape)
            incoming, outgoing = (shape if kind == "gpt2" else shape[::-1])
            targets.append({"name": name, "class": module.__class__.__name__, "weight_shape": shape,
                            "in_features": incoming, "out_features": outgoing,
                            "fan_in_fan_out": kind == "gpt2"})
    layer_count = model.config.n_layer if kind == "gpt2" else model.config.num_hidden_layers
    expected = layer_count * (4 if kind in ("gpt2", "phi3") else 7)
    if len(targets) != expected:
        raise ValueError(f"Expected {expected} native projection modules, resolved {len(targets)}.")
    # Catch architecture changes adding otherwise eligible block projections.
    eligible = {name for name, module in model.named_modules() if isinstance(module, target_type)
                and (name.startswith("transformer.h.") if kind == "gpt2" else name.startswith("model.layers."))}
    if eligible != {row["name"] for row in targets}:
        raise ValueError("Unrecognized block projection; review adapter coverage explicitly.")
    return sorted(targets, key=lambda row: row["name"])


def attach_adapters(model, rank=16, alpha=32, dropout=0.05):
    """Install zero-B LoRA; return eval-mode model with only adapters trainable."""
    from peft import LoraConfig, TaskType, get_peft_model
    if rank <= 0 or alpha <= 0 or not 0 <= dropout < 1:
        raise ValueError("Invalid LoRA hyperparameters.")
    targets = resolve_lora_targets(model)
    model.requires_grad_(False)
    config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=rank, lora_alpha=alpha,
        lora_dropout=dropout, target_modules=[row["name"] for row in targets],
        fan_in_fan_out=model.config.model_type == "gpt2", bias="none",
        modules_to_save=None, init_lora_weights=True, use_rslora=False, use_dora=False)
    wrapped = get_peft_model(model, config)
    wrapped.eval()
    if any(parameter.requires_grad and not (".lora_A." in name or ".lora_B." in name)
           for name, parameter in wrapped.named_parameters()):
        raise ValueError("Unexpected nonadapter trainable parameters.")
    expected = rank * sum(row["in_features"]+row["out_features"] for row in targets)
    actual = sum(parameter.numel() for parameter in wrapped.parameters() if parameter.requires_grad)
    if actual != expected:
        raise ValueError(f"Adapter parameter count mismatch: {actual} vs {expected}.")
    wrapped._phase6_adapter_manifest = {"model_type": wrapped.config.model_type,
        "rank": rank, "alpha": alpha, "dropout": dropout,
        "fan_in_fan_out": wrapped.config.model_type == "gpt2", "targets": targets,
        "target_count": len(targets), "trainable_parameters": actual,
        "total_parameters_with_adapters": sum(parameter.numel() for parameter in wrapped.parameters()),
        "bias": "none", "modules_to_save": None, "initialization": "PEFT default zero-B no-op",
        "trainable_dtype_parameter_counts": {
            dtype: sum(parameter.numel() for parameter in wrapped.parameters()
                       if parameter.requires_grad and str(parameter.dtype) == dtype)
            for dtype in sorted({str(parameter.dtype) for parameter in wrapped.parameters() if parameter.requires_grad})},
        "frozen_dtype_parameter_counts": {
            dtype: sum(parameter.numel() for parameter in wrapped.parameters()
                       if not parameter.requires_grad and str(parameter.dtype) == dtype)
            for dtype in sorted({str(parameter.dtype) for parameter in wrapped.parameters() if not parameter.requires_grad})},
        "optimizer_steps": 0, "packages": package_versions()}
    wrapped._phase6_adapter_manifest["manifest_sha256"] = canonical_hash(wrapped._phase6_adapter_manifest)
    return wrapped


def adapter_manifest(model):
    if not hasattr(model, "_phase6_adapter_manifest"):
        raise ValueError("Model was not configured by this inspected adapter path.")
    return json.loads(json.dumps(model._phase6_adapter_manifest))


def load_frozen_model(model_key, lock_path=DEFAULT_LOCK, cache_dir=DEFAULT_CACHE,
                      device="cuda", precision="bf16", local_files_only=True,
                      require_h200=False):
    """Explicit checkpoint-load API; not invoked by metadata/fixture validation."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    from phase6.fetch_weights import verify_download
    record, directory = _metadata(model_key, lock_path)
    runtime = check_runtime(device, precision, require_h200)
    weights = verify_download(model_key, cache_dir=cache_dir, lock_path=lock_path)
    config = AutoConfig.from_pretrained(str(directory), trust_remote_code=False, local_files_only=True)
    dtype = torch.float32 if runtime["precision"] == "float32" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(record["repo_id"], revision=record["revision"],
        config=config, trust_remote_code=False, local_files_only=local_files_only,
        cache_dir=str(cache_dir), dtype=dtype, use_safetensors=True,
        attn_implementation="eager")
    model.to(torch.device(runtime["device"]))
    model.requires_grad_(False)
    model.eval()
    resolve_lora_targets(model)
    model._phase6_model_binding = {"model_key": model_key, "repo_id": record["repo_id"],
        "revision": record["revision"], "model_lock_sha256": file_hash(lock_path),
        "condition": "frozen", "runtime": runtime, "trust_remote_code": False,
        "weight_download_manifest_sha256": weights["manifest_sha256"],
        "verified_weight_files_sha256": weights["files_sha256"],
        "attention_implementation": "eager"}
    return model


def load_adapted_model(model_key, lock_path=DEFAULT_LOCK, cache_dir=DEFAULT_CACHE,
                       device="cuda", precision="bf16", local_files_only=True,
                       require_h200=False, rank=16, alpha=32, dropout=0.05):
    base = load_frozen_model(model_key, lock_path, cache_dir, device, precision,
                             local_files_only, require_h200)
    binding = dict(base._phase6_model_binding)
    model = attach_adapters(base, rank, alpha, dropout)
    model._phase6_model_binding = {**binding, "condition": "untrained_lora_initialized"}
    return model


def inspect_architecture(model_key, lock_path=DEFAULT_LOCK):
    """Full checkpoint-config target/count inspection on meta tensors, no weights."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    record, directory = _metadata(model_key, lock_path)
    config = AutoConfig.from_pretrained(str(directory), trust_remote_code=False, local_files_only=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=False, attn_implementation="eager")
        wrapped = attach_adapters(model)
    report = adapter_manifest(wrapped)
    report.update({"model_key": model_key, "repo_id": record["repo_id"], "revision": record["revision"],
                   "model_lock_sha256": file_hash(lock_path), "inspection_device": "meta",
                   "checkpoint_weights_loaded": False, "optimizer_steps": 0})
    report["adapter_configuration_sha256"] = report.pop("manifest_sha256")
    report["manifest_sha256"] = canonical_hash(report)
    return report
