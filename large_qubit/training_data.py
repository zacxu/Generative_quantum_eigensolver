"""Authenticated fitting-only token caches and complete-answer training losses.

No model, optimizer, scorer or diagnostic payload is loaded here. GPT weighted
and uniform fits share one objective-independent token cache. Energy preference
enters only through the fixed fitting-view multipliers attached at load time.
"""

from collections import defaultdict
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
import random
from types import MappingProxyType

from . import corpus_v2 as corpus
from .common import canonical_bytes, digest, read_json, require, sha256_file, workspace_path
from .grammar import CircuitGrammar, format_answer
from .models import CONTEXT_LIMIT, MODEL_SPECS, inspect_assets
from .prompts import BOUNDARY_VERSION, PROMPT_VERSION, build_prompt, load_prompt_tokenizer
from .protocol import FITS, seed_for


RELEASE_SHA256 = "323fe5fa6d4830339aa9b14dc546364fe5d24765745f08c421e3ca562e2d3e2a"
CACHE_ROOT = "large_qubit/training_data/section7_v1"
ROWS_PER_GEOMETRY = 2048
PASSES = 5
EFFECTIVE_BATCH_SIZE = 16
SOURCE_PATHS = ("large_qubit/training_data.py", "large_qubit/corpus_v2.py",
                "large_qubit/prompts.py", "large_qubit/grammar.py", "large_qubit/protocol.py",
                "large_qubit/models.py", "large_qubit/catalogue.py", "large_qubit/common.py",
                "large_qubit/definitions.py", "large_qubit/endpoints.py", "phase6/grammar.py")


def _integer(value, name, *, minimum=0, maximum=None):
    require(type(value) is int and value >= minimum and (maximum is None or value <= maximum),
            f"Invalid {name}")
    return value


def _source_bindings(root):
    return {name: sha256_file(workspace_path(root, name)) for name in SOURCE_PATHS}


def _condition_model(condition):
    require(condition in FITS, "Only the three registered fitting consumers may load training data")
    return "qwen3_4b" if condition == "qwen3_4b_weighted_lora" else "gpt2_xl"


def _metadata_identity(root, model_key):
    binding = inspect_assets(root, model_key, full_hash=False)
    directory = workspace_path(root, binding["metadata_directory"])
    files = {p.name: sha256_file(p) for p in sorted(directory.iterdir()) if p.is_file()}
    return {"model_key": model_key, "revision": binding["revision"],
            "model_lock_sha256": binding["model_lock_sha256"],
            "metadata_directory": binding["metadata_directory"], "metadata_files_sha256": files,
            "native_chat_template_sha256": binding["native_chat_template_sha256"]}


def _bytelevel_decoder_spec(value):
    """Normalize pinned tokenizers0.22.2 ByteLevel serialization defaults.

    Older GPT metadata omits use_regex; the pinned native library materializes
    its True default. Explicit saved flags remain exact, including Qwen False.
    Unknown flags or nonboolean options fail rather than being discarded.
    """
    flags = ("add_prefix_space", "trim_offsets", "use_regex")
    require(type(value) is dict and value.get("type") == "ByteLevel"
            and set(value) <= {"type", *flags}
            and all(type(value[key]) is bool for key in flags if key in value),
            "Unsupported native ByteLevel decoder specification")
    return {"type": "ByteLevel", **{key: value.get(key, True) for key in flags}}


def _tokenizer_identity(root, model_key, tokenizer):
    identity = _metadata_identity(root, model_key)
    binding = getattr(tokenizer, "_large_qubit_binding", {})
    require(binding.get("model_key") == model_key and binding.get("revision") == identity["revision"]
            and binding.get("metadata_authenticated") is True, "Use the pinned metadata-authenticated tokenizer")
    raw = read_json(root / identity["metadata_directory"] / "tokenizer.json")
    vocabulary = dict(raw["model"]["vocab"])
    for token in raw.get("added_tokens", []):
        require(token["content"] not in vocabulary or vocabulary[token["content"]] == token["id"],
                "Tokenizer vocabulary has inconsistent added tokens")
        vocabulary[token["content"]] = token["id"]
    require(tokenizer.get_vocab() == vocabulary, "Loaded tokenizer vocabulary differs from pinned native metadata")
    config = read_json(root / identity["metadata_directory"] / "tokenizer_config.json")
    model_config = read_json(root / identity["metadata_directory"] / "config.json")
    eos_id = _integer(model_config.get("eos_token_id"), "pinned model EOS token")
    require(eos_id in vocabulary.values() and tokenizer.eos_token_id == eos_id
            and eos_id in tokenizer.all_special_ids,
            "Loaded genuine EOS differs from pinned model metadata")
    eos_records = [token for token in raw.get("added_tokens", []) if token["id"] == eos_id]
    require(len(eos_records) == 1 and eos_records[0].get("special") is True,
            "Pinned EOS must be explicitly registered as a native special token")
    declarations = [config.get("eos_token")]
    special_map = root / identity["metadata_directory"] / "special_tokens_map.json"
    if special_map.is_file(): declarations.append(read_json(special_map).get("eos_token"))
    for declaration in declarations:
        if declaration is not None:
            text = declaration.get("content") if type(declaration) is dict else declaration
            require(type(text) is str and vocabulary.get(text) == eos_id,
                    "Tokenizer EOS declaration disagrees with the pinned model EOS")
    configured_added = config.get("added_tokens_decoder", {}).get(str(eos_id))
    if configured_added is not None:
        require(configured_added.get("special") is True
                and vocabulary.get(configured_added.get("content")) == eos_id,
                "Tokenizer added-token EOS declaration is inconsistent")
    decoder = _bytelevel_decoder_spec(json.loads(tokenizer.backend_tokenizer.decoder.__getstate__()))
    require(decoder == _bytelevel_decoder_spec(raw["decoder"]),
            "Training grammar requires the pinned native ByteLevel decoder")
    require(type(tokenizer.pad_token_id) is int and tokenizer.pad_token_id in vocabulary.values(),
            "Registered vocabulary padding token required")
    return {**identity, "vocabulary_sha256": digest(vocabulary), "vocabulary_size": len(vocabulary),
            "maximum_token_id": max(vocabulary.values()), "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id, "decoder": decoder}


def _release_rows(root, model_key, expected_release_sha256):
    require(expected_release_sha256 == RELEASE_SHA256, "Section7 uses the exact approved v2 fitting release")
    consumer = "qwen3_4b_weighted_lora" if model_key == "qwen3_4b" else "gpt2_xl_weighted_lora"
    view = corpus.load_release(root, consumer_id=consumer, expected_release_sha256=expected_release_sha256)
    rows = []
    for alias in corpus.TRAINING:
        group = view.fitting_rows(alias)
        require(len(group) == ROWS_PER_GEOMETRY, "Exact complete released fitting geometry required")
        rows.extend(group)
    rows.sort(key=lambda row: row["request_ordinal"])
    require(len({row["row_id"] for row in rows}) == len(rows), "Repeated fitting row identity")
    return view, rows


def _write(path, data):
    import os
    with path.open("xb") as handle:
        handle.write(data); handle.flush(); os.fsync(handle.fileno())


def _grammar_from_fragments(graph, catalogue):
    """Recheck cached DFA paths from fragments; no native tokenizer is loaded."""
    class Decoder:
        def __getstate__(self): return canonical_bytes({"type": "ByteLevel"})
    class Fragments:
        is_fast = True
        def __init__(self):
            self.eos_token_id = graph["eos_token_id"]
            self.all_special_ids = [self.eos_token_id]
            self.backend_tokenizer = self
            self.decoder = Decoder()
        def get_vocab(self):
            return {**{str(k): int(k) for k in graph["token_pieces"]}, "<genuine_eos>": self.eos_token_id}
        def decode(self, ids, **kwargs):
            return ''.join(graph["token_pieces"].get(str(token), "<genuine_eos>") for token in ids)
        def decode_batch(self, sequences, **kwargs): return [self.decode(ids) for ids in sequences]
    require(str(graph["eos_token_id"]) not in graph["token_pieces"], "EOS cannot be a text fragment")
    return CircuitGrammar(Fragments(), catalogue, 32)


def _preserve_partial(root, folder):
    from uuid import uuid4
    boundary = workspace_path(root, CACHE_ROOT, namespace=CACHE_ROOT).resolve()
    original = workspace_path(root, folder.relative_to(root).as_posix(), namespace=CACHE_ROOT)
    target = original.with_name(original.name + "_incomplete_" + uuid4().hex)
    require(original.resolve().is_relative_to(boundary) and target.resolve().is_relative_to(boundary)
            and target.parent.resolve() == original.parent.resolve() and not target.exists(),
            "Partial cache preservation escaped its namespace")
    original.rename(target)


def prepare_training_data(workspace, *, model_key, tokenizer=None, execute=False,
                          cache_root=CACHE_ROOT, expected_release_sha256=RELEASE_SHA256):
    """Write one immutable model cache, or authenticate and reuse the same cache.

    Explicit execution may load a local tokenizer only. No model weights are
    loaded or downloaded. Objective-specific energy weights are never persisted.
    """
    require(execute is True, "Token-cache preparation requires explicit execute=True")
    require(model_key in MODEL_SPECS and cache_root == CACHE_ROOT, "Use the fixed Section7 model-cache namespace")
    root = Path(workspace).resolve(strict=True)
    if tokenizer is None:
        tokenizer = load_prompt_tokenizer(root, model_key, execute=True)
    identity = _tokenizer_identity(root, model_key, tokenizer)
    folder = workspace_path(root, f"{cache_root}/{model_key}", namespace=CACHE_ROOT)
    manifest_path = folder / "manifest.json"
    if manifest_path.is_file() and (folder / "manifest.sha256").is_file():
        fingerprint = sha256_file(manifest_path)
        verified = verify_training_cache(root, manifest_path.relative_to(root).as_posix(), fingerprint,
                                         expected_release_sha256=expected_release_sha256)
        manifest = read_json(manifest_path)
        require(manifest["tokenizer"] == identity and verified["model_key"] == model_key,
                "Existing cache uses different tokenizer inputs")
        return {"manifest_path": manifest_path.relative_to(root).as_posix(),
                "manifest_sha256": fingerprint, "manifest": manifest}
    view, rows = _release_rows(root, model_key, expected_release_sha256)
    catalogue = read_json(root / "large_qubit/catalogues/co2_screened_v1.json")
    grammar = CircuitGrammar(tokenizer, catalogue, 32)
    prompts = {alias: build_prompt(tokenizer, model_key, alias, catalogue, 32) for alias in corpus.TRAINING}
    allowed, lookup, records = [], {}, []
    for row in rows:
        answer = grammar.encode_answer(row["ops"])
        masks = grammar.teacher_forcing_masks(answer)
        groups = []
        for mask in masks:
            key = tuple(mask)
            if key not in lookup:
                lookup[key] = len(allowed); allowed.append(list(key))
            groups.append(lookup[key])
        require(len(prompts[row["geometry_id"]]["input_ids"]) + len(answer) <= CONTEXT_LIMIT,
                "Complete training example exceeds context; truncation is forbidden")
        records.append({"row_id": row["row_id"], "geometry_id": row["geometry_id"],
            "request_ordinal": row["request_ordinal"], "answer_ids": answer,
            "grammar_group_ids": groups,
            "canonical_answer_sha256": sha256(format_answer(row["ops"], catalogue=catalogue, length=32).encode()).hexdigest()})
    graph = {"allowed_token_sets": allowed, "token_pieces": {str(k): v for k, v in grammar.token_pieces.items()},
             "eos_token_id": grammar.eos_token_id, "length": 32, "max_operator_id": 512}
    payloads = {"prompts.json": canonical_bytes(prompts) + b"\n", "grammar.json": canonical_bytes(graph) + b"\n",
                "rows.jsonl": b"".join(canonical_bytes(row) + b"\n" for row in records)}
    files = {name: {"path": f"{cache_root}/{model_key}/{name}", "bytes": len(data),
                    "sha256": sha256(data).hexdigest()} for name, data in payloads.items()}
    manifest = {"schema_version": "co2-training-token-cache-v1", "model_key": model_key,
        "release_sha256": expected_release_sha256, "release_path": corpus.RELEASE_PATH,
        "catalogue_sha256": catalogue["sha256"], "length": 32, "context_limit": CONTEXT_LIMIT,
        "boundary": BOUNDARY_VERSION, "prompt_version": PROMPT_VERSION,
        "tokenizer": identity, "source_sha256": _source_bindings(root), "files": files,
        "rows": len(records), "rows_per_geometry": ROWS_PER_GEOMETRY,
        "ordered_row_ids_sha256": digest([r["row_id"] for r in records]),
        "objective_specific_data_cached": False, "diagnostic_payloads_read": False,
        "source_acquisition_cost": view.source_acquisition_cost,
        "separate_verification_cost": view.separate_verification_cost,
        "model_weights_loaded": False, "model_training_executed": False, "downloads_executed": 0}
    if folder.exists():
        _preserve_partial(root, folder)
    folder.mkdir(parents=True)
    for name, data in payloads.items(): _write(folder / name, data)
    _write(manifest_path, canonical_bytes(manifest) + b"\n")
    fingerprint = sha256_file(manifest_path)
    _write(folder / "manifest.sha256", (fingerprint + "\n").encode("ascii"))
    verify_training_cache(root, manifest_path.relative_to(root).as_posix(), fingerprint,
                          expected_release_sha256=expected_release_sha256)
    return {"manifest_path": manifest_path.relative_to(root).as_posix(),
            "manifest_sha256": fingerprint, "manifest": manifest}


def _read_cache(workspace, manifest_path, expected_manifest_sha256, expected_release_sha256):
    root = Path(workspace).resolve(strict=True)
    path = workspace_path(root, manifest_path, namespace=CACHE_ROOT)
    require(path.name == "manifest.json" and sha256_file(path) == expected_manifest_sha256
            and path.with_suffix(".sha256").read_text(encoding="ascii").strip() == expected_manifest_sha256,
            "Token cache differs from its trusted manifest pin")
    manifest = read_json(path); model_key = manifest["model_key"]
    require(set(manifest) == {"schema_version", "model_key", "release_sha256", "release_path", "catalogue_sha256",
        "length", "context_limit", "boundary", "prompt_version", "tokenizer", "source_sha256", "files",
        "rows", "rows_per_geometry", "ordered_row_ids_sha256", "objective_specific_data_cached",
        "diagnostic_payloads_read", "source_acquisition_cost", "separate_verification_cost",
        "model_weights_loaded", "model_training_executed", "downloads_executed"}, "Unexpected cache manifest fields")
    require(model_key in MODEL_SPECS and manifest_path == f"{CACHE_ROOT}/{model_key}/manifest.json"
            and manifest["schema_version"] == "co2-training-token-cache-v1"
            and manifest["release_sha256"] == expected_release_sha256 == RELEASE_SHA256
            and manifest["release_path"] == corpus.RELEASE_PATH
            and manifest["length"] == 32 and manifest["context_limit"] == CONTEXT_LIMIT
            and manifest["boundary"] == BOUNDARY_VERSION and manifest["prompt_version"] == PROMPT_VERSION
            and manifest["objective_specific_data_cached"] is False and manifest["diagnostic_payloads_read"] is False
            and manifest["model_weights_loaded"] is False and manifest["model_training_executed"] is False
            and manifest["downloads_executed"] == 0, "Training-cache scope differs")
    require(manifest["source_sha256"] == _source_bindings(root), "Token-cache source identity changed")
    identity = _metadata_identity(root, model_key)
    require(all(manifest["tokenizer"][key] == value for key, value in identity.items()),
            "Pinned tokenizer metadata changed")
    data = {}
    require(set(manifest["files"]) == {"prompts.json", "grammar.json", "rows.jsonl"}, "Incomplete token cache")
    for name, descriptor in manifest["files"].items():
        require(descriptor["path"] == f"{CACHE_ROOT}/{model_key}/{name}", "Cross-model cache routing")
        payload = workspace_path(root, descriptor["path"], namespace=f"{CACHE_ROOT}/{model_key}")
        require(payload.stat().st_size == descriptor["bytes"] and sha256_file(payload) == descriptor["sha256"],
                "Token-cache payload changed")
        if name == "rows.jsonl":
            content = payload.read_bytes(); require(content.endswith(b"\n"), "Partial cached row")
            from .common import _pairs
            data[name] = [json.loads(line, object_pairs_hook=_pairs) for line in content.splitlines()]
        else: data[name] = read_json(payload)
    view, source_rows = _release_rows(root, model_key, expected_release_sha256)
    require(manifest["source_acquisition_cost"] == view.source_acquisition_cost
            and manifest["separate_verification_cost"] == view.separate_verification_cost,
            "Cached data omitted shared offline costs")
    catalogue = read_json(root / "large_qubit/catalogues/co2_screened_v1.json")
    require(manifest["catalogue_sha256"] == catalogue["sha256"], "Cached catalogue changed")
    rows, prompts, graph = data["rows.jsonl"], data["prompts.json"], data["grammar.json"]
    require(set(graph) == {"allowed_token_sets", "token_pieces", "eos_token_id", "length", "max_operator_id"},
            "Unexpected cached grammar fields")
    require(len(rows) == manifest["rows"] == len(source_rows)
            and manifest["rows_per_geometry"] == ROWS_PER_GEOMETRY
            and digest([r["row_id"] for r in rows]) == manifest["ordered_row_ids_sha256"], "Cached row selection/order changed")
    require(set(prompts) == set(corpus.TRAINING) and graph["length"] == 32 and graph["max_operator_id"] == 512
            and graph["eos_token_id"] == manifest["tokenizer"]["eos_token_id"], "Cached grammar/prompt identity differs")
    allowed = graph["allowed_token_sets"]; require(type(allowed) is list and allowed, "Missing grammar sets")
    require(len({tuple(mask) for mask in allowed}) == len(allowed), "Grammar sets must be deduplicated")
    for mask in allowed:
        require(type(mask) is list and mask and len(set(mask)) == len(mask), "Invalid cached grammar set")
        for token in mask: _integer(token, "grammar token", maximum=manifest["tokenizer"]["maximum_token_id"])
    rebuilt_grammar = _grammar_from_fragments(graph, catalogue)
    for alias, prompt in prompts.items():
        ids = prompt["input_ids"]; require(type(ids) is list and ids, "Empty prompt")
        for token in ids: _integer(token, "prompt token", maximum=manifest["tokenizer"]["maximum_token_id"])
        cert = prompt["context_certificate"]
        require(cert["geometry_id"] == alias and cert["model_key"] == model_key
                and cert["catalogue_sha256"] == catalogue["sha256"] and cert["boundary"] == BOUNDARY_VERSION
                and cert["prompt_tokens"] == len(ids) and cert["total_upper_bound"] <= CONTEXT_LIMIT,
                "Cached prompt certificate differs")
    for row, source in zip(rows, source_rows):
        require(set(row) == {"row_id", "geometry_id", "request_ordinal", "answer_ids", "grammar_group_ids", "canonical_answer_sha256"}
                and all(row[key] == source[key] for key in ("row_id", "geometry_id", "request_ordinal")),
                "Cache imported another row or role")
        answer, groups = row["answer_ids"], row["grammar_group_ids"]
        require(type(answer) is list and len(answer) > 1 and type(groups) is list and len(answer) == len(groups)
                and answer[-1] == graph["eos_token_id"] and graph["eos_token_id"] not in answer[:-1],
                "Cached answer must include exactly one final genuine EOS")
        for token, group in zip(answer, groups):
            _integer(token, "answer token", maximum=manifest["tokenizer"]["maximum_token_id"])
            _integer(group, "grammar group", maximum=len(allowed)-1)
            require(token in allowed[group], "Cached answer target is outside its legal set")
        require([tuple(allowed[group]) for group in groups] == rebuilt_grammar.teacher_forcing_masks(answer),
                "Cached legal sets differ from the complete-answer DFA")
        text = ''.join(graph["token_pieces"][str(token)] for token in answer[:-1])
        require(text == format_answer(source["ops"], catalogue=catalogue, length=32)
                and sha256(text.encode()).hexdigest() == row["canonical_answer_sha256"],
                "Cached answer differs from released circuit")
        require(len(prompts[row["geometry_id"]]["input_ids"]) + len(answer) <= CONTEXT_LIMIT,
                "Cached complete example requires truncation")
    return root, manifest, rows, prompts, graph


def verify_training_cache(workspace, manifest_path, expected_manifest_sha256, *, expected_release_sha256=RELEASE_SHA256):
    _, manifest, rows, prompts, graph = _read_cache(workspace, manifest_path, expected_manifest_sha256, expected_release_sha256)
    return {"all_passed": True, "manifest_path": manifest_path, "manifest_sha256": expected_manifest_sha256,
        "model_key": manifest["model_key"], "release_sha256": expected_release_sha256,
        "rows": len(rows), "rows_per_geometry": ROWS_PER_GEOMETRY,
        "ordered_row_ids_sha256": manifest["ordered_row_ids_sha256"], "grammar_allowed_sets": len(graph["allowed_token_sets"]),
        "maximum_complete_example_tokens": max(len(prompts[r["geometry_id"]]["input_ids"])+len(r["answer_ids"]) for r in rows),
        "diagnostic_payloads_read": False, "model_weights_loaded": False, "model_training_executed": False,
        "tokenizer_loaded_by_verification": False, "hamiltonian_actions": 0}


class CachedTrainingData:
    """Verified cached examples plus condition-specific fixed fitting weights."""

    def __init__(self, condition_id, manifest, records, prompts, graph, multipliers, manifest_path, fingerprint):
        self.condition_id, self.model_key = condition_id, manifest["model_key"]
        self.cache_manifest_path, self.cache_manifest_sha256 = manifest_path, fingerprint
        self.release_sha256 = manifest["release_sha256"]
        self._records, self._prompts, self._graph = records, prompts, graph
        self._multipliers, self._device_grammar = tuple(multipliers), {}
        self._maximum_grammar_token_id = max(max(mask) for mask in graph["allowed_token_sets"])
        self._memory_stress_order = tuple(sorted(range(len(records)), key=lambda i: (
            -(len(prompts[records[i]["geometry_id"]]["input_ids"])+len(records[i]["answer_ids"])), i)))
        self.rows = tuple(MappingProxyType({key: row[key] for key in ("row_id", "geometry_id", "request_ordinal")}) for row in records)
        self.row_ids = tuple(row["row_id"] for row in records)
        self._summary = {"condition_id": condition_id, "model_key": self.model_key,
            "cache_manifest_path": manifest_path, "cache_manifest_sha256": fingerprint,
            "release_sha256": self.release_sha256, "rows": len(records), "rows_per_geometry": ROWS_PER_GEOMETRY,
            "ordered_row_ids_sha256": manifest["ordered_row_ids_sha256"],
            "loss_multipliers_sha256": digest(list(self._multipliers)), "passes": PASSES,
            "example_presentations": len(records)*PASSES, "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "optimizer_steps": math.ceil(len(records)/EFFECTIVE_BATCH_SIZE)*PASSES,
            "row_order_seed": seed_for("row_order"),
            "row_order_recipe": "Random(seed_for(row_order)+pass_index), shuffle each training geometry, then round-robin T01,T02,T03",
            "source_acquisition_cost": deepcopy(manifest["source_acquisition_cost"]),
            "separate_verification_cost": deepcopy(manifest["separate_verification_cost"]),
            "objective": "complete_answer_grammar_masked_sequence_NLL_sum; n_g*w_gi_once; divide by actual accumulation-window examples",
            "diagnostic_payloads_read": False, "model_training_executed": False}

    @property
    def summary(self): return deepcopy(self._summary)

    def __len__(self): return len(self._records)

    def row_order(self, pass_index):
        _integer(pass_index, "zero-based pass index", maximum=PASSES-1)
        rng = random.Random(seed_for("row_order") + pass_index)
        groups = [[i for i, row in enumerate(self._records) if row["geometry_id"] == alias] for alias in corpus.TRAINING]
        require(len({len(group) for group in groups}) == 1, "Geometry-balanced visits require equal groups")
        for group in groups: rng.shuffle(group)
        return tuple(index for triple in zip(*groups) for index in triple)

    def memory_stress_indices(self, count=16):
        """Longest actual complete examples; no truncation or energy ordering."""
        _integer(count, "memory stress example count", minimum=1, maximum=len(self))
        return self._memory_stress_order[:count]

    def collate(self, indices, device="cpu"):
        import torch
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        indices = tuple(indices); require(indices and len(set(indices)) == len(indices), "A nonempty unique row microbatch is required")
        for index in indices: _integer(index, "cached row index", maximum=len(self)-1)
        selected = [self._records[i] for i in indices]
        sequences = [self._prompts[r["geometry_id"]]["input_ids"]+r["answer_ids"] for r in selected]
        width = max(map(len, sequences)); require(width <= CONTEXT_LIMIT, "No training truncation")
        ids = torch.full((len(indices), width), self._graph["pad_token_id"], dtype=torch.long, device=device)
        attention = torch.zeros_like(ids); labels = torch.full_like(ids, -100)
        groups = defaultdict(list)
        for b, (row, sequence) in enumerate(zip(selected, sequences)):
            prompt_length = len(self._prompts[row["geometry_id"]]["input_ids"])
            ids[b, :len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
            attention[b, :len(sequence)] = 1
            labels[b, prompt_length:len(sequence)] = torch.tensor(row["answer_ids"], dtype=torch.long, device=device)
            for offset, (target, group) in enumerate(zip(row["answer_ids"], row["grammar_group_ids"])):
                groups[group].append((b, prompt_length+offset-1, target))
        key = str(torch.device(device))
        if key not in self._device_grammar:
            self._device_grammar[key] = tuple(torch.tensor(mask, dtype=torch.long, device=device) for mask in self._graph["allowed_token_sets"])
        loss_groups = []
        for group, positions in sorted(groups.items()):
            coordinates = torch.tensor(positions, dtype=torch.long, device=device)
            loss_groups.append((coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], self._device_grammar[key][group]))
        return {"input_ids": ids, "attention_mask": attention,
            "position_ids": (attention.cumsum(-1)-1).clamp_min(0).masked_fill(attention == 0, 0),
            "labels": labels, "loss_multipliers": torch.tensor([self._multipliers[i] for i in indices], dtype=torch.float32, device=device),
            "row_ids": tuple(r["row_id"] for r in selected), "answer_lengths": tuple(len(r["answer_ids"]) for r in selected),
            "loss_groups": tuple(loss_groups),
            "maximum_grammar_token_id": self._maximum_grammar_token_id,
            "cache_manifest_sha256": self.cache_manifest_sha256}

    def sequence_nll(self, logits, batch):
        return sequence_nll(logits, batch)


def sequence_nll(logits, batch):
    """Sparse grouped complete-answer NLL; no full-vocabulary mask or token mean."""
    import torch
    require(isinstance(logits, torch.Tensor) and logits.ndim == 3 and logits.is_floating_point()
            and tuple(logits.shape[:2]) == tuple(batch["labels"].shape)
            and batch["labels"].dtype == torch.long and logits.device == batch["labels"].device,
            "Causal logits and cached batch labels must match")
    require(logits.shape[0] > 0 and logits.shape[1] >= 2 and batch["loss_groups"]
            and logits.shape[-1] > batch["maximum_grammar_token_id"], "Model vocabulary is smaller than cached grammar")
    dtype = torch.float32 if logits.dtype in (torch.float16, torch.bfloat16) else logits.dtype
    losses = torch.zeros(logits.shape[0], dtype=dtype, device=logits.device)
    finite, target_count = [], 0
    for rows, positions, targets, allowed in batch["loss_groups"]:
        require(allowed.numel() > 0 and rows.numel() > 0, "Empty cached loss group")
        legal = logits[rows[:, None], positions[:, None], allowed[None, :]].to(dtype)
        target_values = logits[rows, positions, targets].to(dtype)
        finite.extend((torch.isfinite(legal).all(), torch.isfinite(target_values).all(),
                       (batch["labels"][rows, positions+1] == targets).all()))
        target_count += rows.numel()
        losses = losses.index_add(0, rows, torch.logsumexp(legal, dim=-1)-target_values)
    require(bool(torch.stack(finite).all()) and target_count == int((batch["labels"] != -100).sum()),
            "Nonfinite legal logits or cached target/causal-shift mismatch")
    return losses


def load_training_data(workspace, *, condition_id, manifest_path, expected_manifest_sha256,
                       expected_release_sha256=RELEASE_SHA256):
    model_key = _condition_model(condition_id)
    root, manifest, rows, prompts, graph = _read_cache(workspace, manifest_path, expected_manifest_sha256, expected_release_sha256)
    require(model_key == manifest["model_key"], "Condition cannot consume another model's token cache")
    view = corpus.load_release(root, consumer_id=condition_id, expected_release_sha256=expected_release_sha256)
    by_id = {}
    for alias in corpus.TRAINING:
        fixed = view.weights(alias)
        by_id.update(zip(fixed.row_ids, fixed.multipliers))
    require(set(by_id) == {row["row_id"] for row in rows}, "Fixed weights must cover exactly all cached fitting rows")
    graph = {**graph, "pad_token_id": manifest["tokenizer"]["pad_token_id"]}
    return CachedTrainingData(condition_id, manifest, rows, prompts, graph,
                              [by_id[row["row_id"]] for row in rows], manifest_path, expected_manifest_sha256)
