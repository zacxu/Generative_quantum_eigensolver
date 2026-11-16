"""Fast consumers of the two independently, deeply audited Section7 caches.

The cache pins below are an allowlist, not caller-provided trust. Their original
builder and deep validator remain unchanged in training_data.py. Routine reads
authenticate every cache byte, current frozen source and native metadata, and
the exact release manifest. They reuse the already proved grammar paths rather
than reconstructing the DFA or formatting 6,144 answers again. Section7's
preparation additionally binds the preserved native, three-consumer audit.

Weights are never accepted from a token cache. Loading a condition still opens
its protected fitting-only release view and computes the registered weights.
No tokenizer, model weights, diagnostic labels or physical evaluator is loaded.
"""

from hashlib import sha256
import json
from pathlib import Path

from . import training_data as prepared
from .common import _pairs, digest, require, workspace_path


RELEASE_SHA256 = prepared.RELEASE_SHA256
TOKEN_CACHE_SHA256 = {
    "gpt2_xl": "f249b19110fb88c4d8170e8709b5e43e03a3def972615318b6e68d6d5bff8ca7",
    "qwen3_4b": "753516ba33f211247b4ee0ed2405a2a08c94004dfc7ec16f496af38aee84d365",
}


def _json(content):
    def reject(value):
        raise ValueError(f"Non-finite JSON value: {value}")
    return json.loads(content, object_pairs_hook=_pairs, parse_constant=reject)


def _read_cache(workspace, manifest_path, expected_manifest_sha256, expected_release_sha256):
    root = Path(workspace).resolve(strict=True)
    paths = {f"{prepared.CACHE_ROOT}/{key}/manifest.json": key for key in TOKEN_CACHE_SHA256}
    require(type(manifest_path) is str and manifest_path in paths, "Use an audited model-cache path")
    key = paths[manifest_path]
    require(expected_manifest_sha256 == TOKEN_CACHE_SHA256[key], "Use the exact deeply audited cache pin")
    require(expected_release_sha256 == RELEASE_SHA256, "Use the exact approved fitting release")
    path = workspace_path(root, manifest_path, namespace=f"{prepared.CACHE_ROOT}/{key}")
    content = path.read_bytes()
    require(sha256(content).hexdigest() == expected_manifest_sha256
            and path.with_suffix(".sha256").read_text(encoding="ascii").strip() == expected_manifest_sha256,
            "Token cache differs from its trusted manifest pin")
    manifest = _json(content)
    require(set(manifest) == {"schema_version", "model_key", "release_sha256", "release_path", "catalogue_sha256",
        "length", "context_limit", "boundary", "prompt_version", "tokenizer", "source_sha256", "files",
        "rows", "rows_per_geometry", "ordered_row_ids_sha256", "objective_specific_data_cached",
        "diagnostic_payloads_read", "source_acquisition_cost", "separate_verification_cost",
        "model_weights_loaded", "model_training_executed", "downloads_executed"}, "Unexpected cache manifest fields")
    require(manifest["model_key"] == key and manifest["schema_version"] == "co2-training-token-cache-v1"
            and manifest["release_sha256"] == expected_release_sha256
            and manifest["release_path"] == prepared.corpus.RELEASE_PATH
            and manifest["length"] == 32 and manifest["context_limit"] == prepared.CONTEXT_LIMIT
            and manifest["boundary"] == prepared.BOUNDARY_VERSION and manifest["prompt_version"] == prepared.PROMPT_VERSION
            and manifest["objective_specific_data_cached"] is False and manifest["diagnostic_payloads_read"] is False
            and manifest["model_weights_loaded"] is False and manifest["model_training_executed"] is False
            and type(manifest["downloads_executed"]) is int and manifest["downloads_executed"] == 0,
            "Training-cache scope differs")
    require(manifest["source_sha256"] == prepared._source_bindings(root), "Token-cache source identity changed")
    identity = prepared._metadata_identity(root, key)
    require(all(manifest["tokenizer"][name] == value for name, value in identity.items()),
            "Pinned tokenizer metadata changed")
    release = workspace_path(root, manifest["release_path"], namespace="large_qubit/releases/co2_corpus_v2")
    require(sha256(release.read_bytes()).hexdigest() == expected_release_sha256
            and release.with_suffix(".sha256").read_text(encoding="ascii").strip() == expected_release_sha256,
            "Approved fitting release manifest changed")
    require(set(manifest["files"]) == {"prompts.json", "grammar.json", "rows.jsonl"}, "Incomplete token cache")
    payloads = {}
    for name, descriptor in manifest["files"].items():
        require(set(descriptor) == {"path", "bytes", "sha256"}
                and descriptor["path"] == f"{prepared.CACHE_ROOT}/{key}/{name}", "Cross-model cache routing")
        payload = workspace_path(root, descriptor["path"], namespace=f"{prepared.CACHE_ROOT}/{key}")
        raw = payload.read_bytes()
        require(type(descriptor["bytes"]) is int and len(raw) == descriptor["bytes"]
                and sha256(raw).hexdigest() == descriptor["sha256"], "Token-cache payload changed")
        if name == "rows.jsonl":
            require(raw.endswith(b"\n"), "Partial cached row")
            payloads[name] = [_json(line) for line in raw.splitlines()]
        else:
            payloads[name] = _json(raw)
    rows, prompts, graph = payloads["rows.jsonl"], payloads["prompts.json"], payloads["grammar.json"]
    require(len(rows) == manifest["rows"] == len(prepared.corpus.TRAINING)*prepared.ROWS_PER_GEOMETRY
            and manifest["rows_per_geometry"] == prepared.ROWS_PER_GEOMETRY
            and digest([row["row_id"] for row in rows]) == manifest["ordered_row_ids_sha256"],
            "Cached row selection/order changed")
    require(set(prompts) == set(prepared.corpus.TRAINING)
            and set(graph) == {"allowed_token_sets", "token_pieces", "eos_token_id", "length", "max_operator_id"}
            and graph["length"] == 32 and graph["max_operator_id"] == 512
            and graph["eos_token_id"] == manifest["tokenizer"]["eos_token_id"], "Cached grammar/prompt identity differs")
    # These are cheap runtime-shape checks, not a second semantic proof. The
    # immutable pins establish the answer/grammar correspondence already audited.
    require(graph["allowed_token_sets"] and all(graph["allowed_token_sets"]), "Missing cached grammar sets")
    require(len({row["row_id"] for row in rows}) == len(rows), "Repeated fitting row identity")
    for row in rows:
        require(set(row) == {"row_id", "geometry_id", "request_ordinal", "answer_ids", "grammar_group_ids", "canonical_answer_sha256"}
                and row["geometry_id"] in prompts and row["row_id"].startswith(row["geometry_id"] + "_v2_F")
                and len(row["answer_ids"]) == len(row["grammar_group_ids"])
                and row["answer_ids"][-1] == graph["eos_token_id"]
                and len(prompts[row["geometry_id"]]["input_ids"]) + len(row["answer_ids"]) <= prepared.CONTEXT_LIMIT,
                "Cached fitting example shape differs")
    return root, manifest, rows, prompts, graph


def verify_training_cache(workspace, manifest_path, expected_manifest_sha256, *, expected_release_sha256=RELEASE_SHA256):
    """Authenticate audited cache bytes without reading any release labels."""
    _, manifest, rows, prompts, graph = _read_cache(workspace, manifest_path, expected_manifest_sha256, expected_release_sha256)
    return {"all_passed": True, "manifest_path": manifest_path, "manifest_sha256": expected_manifest_sha256,
        "model_key": manifest["model_key"], "release_sha256": expected_release_sha256,
        "rows": len(rows), "rows_per_geometry": prepared.ROWS_PER_GEOMETRY,
        "ordered_row_ids_sha256": manifest["ordered_row_ids_sha256"], "grammar_allowed_sets": len(graph["allowed_token_sets"]),
        "maximum_complete_example_tokens": max(len(prompts[r["geometry_id"]]["input_ids"])+len(r["answer_ids"]) for r in rows),
        "semantic_validation": "preserved_deep_audit_of_exact_allowlisted_manifest_and_transitive_bytes",
        "diagnostic_payloads_read": False, "release_label_payloads_read": False,
        "model_weights_loaded": False, "model_training_executed": False,
        "tokenizer_loaded_by_verification": False, "hamiltonian_actions": 0}


def load_training_data(workspace, *, condition_id, manifest_path, expected_manifest_sha256,
                       expected_release_sha256=RELEASE_SHA256):
    """Reuse proved tokenizations and obtain weights through fitting-only guards."""
    key = prepared._condition_model(condition_id)
    root, manifest, rows, prompts, graph = _read_cache(workspace, manifest_path, expected_manifest_sha256, expected_release_sha256)
    require(key == manifest["model_key"], "Condition cannot consume another model's token cache")
    view = prepared.corpus.load_release(root, consumer_id=condition_id, expected_release_sha256=expected_release_sha256)
    by_id = {}
    for alias in prepared.corpus.TRAINING:
        fixed = view.weights(alias)
        require(not set(by_id).intersection(fixed.row_ids), "Fitting weights repeated a row identity")
        by_id.update(zip(fixed.row_ids, fixed.multipliers))
    require(set(by_id) == {row["row_id"] for row in rows}, "Fixed weights must cover exactly all cached fitting rows")
    require(view.source_acquisition_cost == manifest["source_acquisition_cost"]
            and view.separate_verification_cost == manifest["separate_verification_cost"],
            "Cached data omitted shared offline costs")
    graph = {**graph, "pad_token_id": manifest["tokenizer"]["pad_token_id"]}
    return prepared.CachedTrainingData(condition_id, manifest, rows, prompts, graph,
        [by_id[row["row_id"]] for row in rows], manifest_path, expected_manifest_sha256)
