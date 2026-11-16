"""Synthetic asset/loader tests; no real weights, model imports or downloads."""

from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

from large_qubit import models


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def descriptor(root, path):
    return {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
            "sha256": sha256(path.read_bytes()).hexdigest()}


class AssetFixture:
    def __init__(self, root, key="gpt2_xl"):
        self.root, self.key = root, key
        self.spec = models.MODEL_SPECS[key]
        self.metadata = root / "phase6/cache/models" / key / self.spec["revision"]
        self.snapshot = (root / "phase6/cache/models" /
                         ("models--" + self.spec["repo_id"].replace("/", "--")) /
                         "snapshots" / self.spec["revision"])
        self.metadata.mkdir(parents=True)
        self.snapshot.mkdir(parents=True)
        sharded = key == "qwen3_4b"
        weights = (["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
                   if sharded else ["model.safetensors"])
        names = weights + ["config.json", "generation_config.json"]
        if sharded:
            names.append("model.safetensors.index.json")
        template = "native Qwen fixture template" if sharded else None
        payloads = {
            "source_api.json": {"id": self.spec["repo_id"], "sha": self.spec["revision"],
                                "siblings": [{"rfilename": name} for name in names]},
            "config.json": {"model_type": self.spec["model_type"], "n_positions": 1024},
            "generation_config.json": {"eos_token_id": 9},
            "tokenizer_config.json": {"chat_template": template} if template else {},
            "tokenizer.json": {"model": "fixture"},
        }
        for name, payload in payloads.items():
            write_json(self.metadata / name, payload)
        self.record = {key: self.spec[key] for key in
                       ("repo_id", "revision", "model_type", "interface", "template_kwargs")}
        self.record.update({"model_key": key, "trust_remote_code": False,
            "metadata_dir": self.metadata.relative_to(root).as_posix(),
            "files": {name: descriptor(root, self.metadata / name) for name in payloads},
            "native_chat_template_sha256": sha256(template.encode()).hexdigest() if template else None,
            "template_source_file": "tokenizer_config.json" if template else None})
        self.lock = {"schema": "phase6-model-lock-v1", "shared_context_limit": 1024,
                     "native_implementations_only": True, "models": {key: self.record}}
        self.lock_path = root / "phase6/model_lock.json"
        self.lock_hash = self.save_lock()
        for name in names:
            path = self.snapshot / name
            if name in payloads:
                path.write_bytes((self.metadata / name).read_bytes())
            elif name == "model.safetensors.index.json":
                write_json(path, {"weight_map": {f"layer{i}": n for i, n in enumerate(weights)}})
            else:
                path.write_bytes(b"synthetic weight bytes")
        self.weight_names = weights
        self.manifest = {"schema": "phase6-weight-download-v1", "model_key": key,
                         "repo_id": self.spec["repo_id"], "revision": self.spec["revision"],
                         "model_lock_sha256": self.lock_hash,
                         "files": {name: descriptor(root, self.snapshot / name) for name in names}}
        self.manifest_path = (root / "phase6/cache/models/weight_manifests" /
                              f"{key}_{self.spec['revision']}.json")
        self.manifest_hash = self.save_manifest()

    def save_lock(self):
        write_json(self.lock_path, self.lock)
        fingerprint = sha256(self.lock_path.read_bytes()).hexdigest()
        self.lock_path.with_suffix(".sha256").write_text(f"{fingerprint}  {self.lock_path.name}\n")
        self.lock_hash = fingerprint
        return fingerprint

    def save_manifest(self):
        write_json(self.manifest_path, self.manifest)
        fingerprint = sha256(self.manifest_path.read_bytes()).hexdigest()
        self.manifest_path.with_suffix(".sha256").write_text(f"{fingerprint}  {self.manifest_path.name}\n")
        self.manifest_hash = fingerprint
        return fingerprint

    def trusted(self):
        return patch.multiple(models, PINNED_LOCK_SHA256=self.lock_hash,
            MODEL_SPECS={**models.MODEL_SPECS, self.key: {**self.spec,
                "weight_manifest_sha256": self.manifest_hash}})


class ModelAssetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def fixture(self, key="gpt2_xl"):
        return AssetFixture(self.root, key)

    def test_registered_subset_inspects_without_other_model_assets(self):
        fixture = self.fixture()
        with fixture.trusted():
            result = models.inspect_assets(self.root, fixture.key)
            full = models.authenticate_assets(self.root, fixture.key)
        self.assertFalse(result["weight_contents_authenticated"])
        self.assertTrue(full["weight_contents_authenticated"])
        self.assertEqual(full["initialization"], "pretrained_base_only")
        self.assertEqual(full["shared_context_limit"], 1024)
        self.assertFalse(full["old_lih_adapters_reused"])

    def test_sharded_qwen_native_thinking_disabled(self):
        fixture = self.fixture("qwen3_4b")
        with fixture.trusted():
            result = models.authenticate_assets(self.root, fixture.key)
        self.assertEqual(result["template_kwargs"], {"enable_thinking": False})
        self.assertEqual(len(result["weight_files_sha256"]), 2)

    def test_no_heavy_imports_during_authentication(self):
        fixture = self.fixture()
        import builtins
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in {"torch", "transformers", "huggingface_hub", "peft"}:
                raise AssertionError(f"Unexpected heavy import {name}")
            return original(name, *args, **kwargs)
        with fixture.trusted(), patch("builtins.__import__", guarded):
            models.authenticate_assets(self.root, fixture.key)

    def test_unknown_key_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "registered"):
            models.inspect_assets(self.root, "smollm3_3b")

    def test_mutated_lock_and_rewritten_sidecar_cannot_change_trust(self):
        fixture = self.fixture()
        with fixture.trusted():
            fixture.record["revision"] = "f" * 40
            fixture.save_lock()
            with self.assertRaisesRegex(ValueError, "Pinned manifest checksum"):
                models.inspect_assets(self.root, fixture.key)

    def test_wrong_revision_is_rejected_even_with_fixture_trust(self):
        fixture = self.fixture()
        fixture.record["revision"] = "f" * 40
        fixture.save_lock()
        with fixture.trusted(), self.assertRaisesRegex(ValueError, "identity changed: revision"):
            models.inspect_assets(self.root, fixture.key)

    def test_missing_and_truncated_weight_are_rejected(self):
        fixture = self.fixture()
        target = fixture.snapshot / fixture.weight_names[0]
        with fixture.trusted():
            target.write_bytes(b"short")
            with self.assertRaisesRegex(ValueError, "truncated"):
                models.inspect_assets(self.root, fixture.key)
            target.unlink()
            with self.assertRaisesRegex(ValueError, "Missing cached asset"):
                models.inspect_assets(self.root, fixture.key)

    def test_same_size_weight_corruption_needs_full_hash(self):
        fixture = self.fixture()
        target = fixture.snapshot / fixture.weight_names[0]
        target.write_bytes(b"x" * target.stat().st_size)
        with fixture.trusted():
            self.assertFalse(models.inspect_assets(self.root, fixture.key)["weight_contents_authenticated"])
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                models.authenticate_assets(self.root, fixture.key)

    def test_metadata_hash_is_checked_in_quick_mode(self):
        fixture = self.fixture()
        target = fixture.metadata / "tokenizer.json"
        target.write_bytes(b"x" * target.stat().st_size)
        with fixture.trusted(), self.assertRaisesRegex(ValueError, "checksum mismatch"):
            models.inspect_assets(self.root, fixture.key)

    def test_descriptor_traversal_is_rejected(self):
        fixture = self.fixture()
        fixture.record["files"]["tokenizer.json"]["path"] = "phase6/cache/models/../../tokenizer.json"
        fixture.save_lock()
        with fixture.trusted(), self.assertRaisesRegex(ValueError, "escaped its expected path"):
            models.inspect_assets(self.root, fixture.key)

    def test_legacy_adapter_path_is_rejected(self):
        fixture = self.fixture()
        fixture.manifest["files"][fixture.weight_names[0]]["path"] = "phase7/runs/lih/adapter_model.safetensors"
        fixture.save_manifest()
        with fixture.trusted(), self.assertRaisesRegex(ValueError, "expected snapshot"):
            models.inspect_assets(self.root, fixture.key)

    def test_added_adapter_cannot_trigger_automatic_adapter_loading(self):
        fixture = self.fixture()
        write_json(fixture.snapshot / "adapter_config.json", {"base_model_name_or_path": "remote"})
        with fixture.trusted(), self.assertRaisesRegex(ValueError, "Unexpected adapter"):
            models.inspect_assets(self.root, fixture.key)

    def test_missing_manifest_file_entry_is_rejected(self):
        fixture = self.fixture()
        fixture.manifest["files"].pop(fixture.weight_names[0])
        fixture.save_manifest()
        with fixture.trusted(), self.assertRaisesRegex(ValueError, "differs from the pinned"):
            models.inspect_assets(self.root, fixture.key)

    def test_wrong_shard_map_is_rejected(self):
        fixture = self.fixture("qwen3_4b")
        index = fixture.snapshot / "model.safetensors.index.json"
        write_json(index, {"weight_map": {"layer": "wrong.safetensors"}})
        fixture.manifest["files"][index.name] = descriptor(self.root, index)
        fixture.save_manifest()
        with fixture.trusted(), self.assertRaisesRegex(ValueError, "exactly the registered shards"):
            models.inspect_assets(self.root, fixture.key)

    def test_escaped_weight_symlink_is_rejected(self):
        fixture = self.fixture()
        with tempfile.TemporaryDirectory() as external:
            destination = Path(external) / "weight"
            destination.write_bytes(b"synthetic weight bytes")
            target = fixture.snapshot / fixture.weight_names[0]
            target.unlink()
            try:
                target.symlink_to(destination)
            except OSError as error:
                self.skipTest(f"Host cannot create fixture symlinks: {error}")
            with fixture.trusted(), self.assertRaisesRegex(ValueError, "symlink escaped"):
                models.inspect_assets(self.root, fixture.key)

    def test_resolved_path_escape_guard_on_hosts_without_symlink_privilege(self):
        inside = self.root / "linked" / "weights"
        outside = self.root.parent / (self.root.name + "_outside") / "weights"
        original = Path.resolve
        def redirected(path, *args, **kwargs):
            return outside if path == inside else original(path, *args, **kwargs)
        with patch.object(Path, "resolve", redirected), self.assertRaisesRegex(ValueError, "symlink escaped"):
            models._path(self.root, "linked/weights", self.root)

    def test_inspection_receipt_paths_are_portable(self):
        fixture = self.fixture()
        with fixture.trusted():
            result = models.inspect_assets(self.root, fixture.key)
        for key in ("metadata_directory", "snapshot_directory"):
            self.assertTrue(result[key].startswith("phase6/cache/models/"))
            self.assertNotIn("\\", result[key])
            self.assertNotIn(str(self.root), result[key])

    def test_loaders_refuse_without_explicit_execution_or_with_network(self):
        for loader in (models.load_tokenizer, models.load_base_model):
            with self.assertRaisesRegex(ValueError, "execute=True"):
                loader(self.root, "gpt2_xl")
            with self.assertRaisesRegex(ValueError, "downloads are disabled"):
                loader(self.root, "gpt2_xl", execute=True, local_files_only=False)

    def test_loader_full_auth_precedes_dependency_imports(self):
        fixture = self.fixture()
        target = fixture.snapshot / fixture.weight_names[0]
        target.write_bytes(b"x" * target.stat().st_size)
        with fixture.trusted(), patch.object(models, "_require_versions") as versions:
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                models.load_base_model(self.root, fixture.key, execute=True)
            versions.assert_not_called()

    def test_package_version_mismatch_is_not_silent(self):
        with patch.object(models.metadata, "version", return_value="999.0"):
            with self.assertRaisesRegex(ValueError, "Unvalidated package version"):
                models._require_versions()

    def test_tokenizer_loader_uses_local_path_and_records_context(self):
        fixture = self.fixture()
        tokenizer = MagicMock(is_fast=True, eos_token_id=9, pad_token_id=9, chat_template=None)
        auto = MagicMock()
        auto.from_pretrained.return_value = tokenizer
        with fixture.trusted(), patch.object(models, "_require_versions", return_value={}), \
                patch.dict(sys.modules, {"transformers": types.SimpleNamespace(AutoTokenizer=auto)}):
            loaded = models.load_tokenizer(self.root, fixture.key, execute=True)
        call = auto.from_pretrained.call_args
        self.assertEqual(call.args, (str(fixture.metadata),))
        self.assertIs(call.kwargs["local_files_only"], True)
        self.assertIs(call.kwargs["trust_remote_code"], False)
        self.assertEqual(loaded.model_max_length, 1024)
        self.assertTrue(loaded._large_qubit_binding["weight_contents_authenticated"])

    def test_model_loader_uses_snapshot_safetensors_and_frozen_eval(self):
        fixture = self.fixture()
        auto = MagicMock()
        model = auto.from_pretrained.return_value
        model.config.model_type = "gpt2"
        del model.peft_config
        config = MagicMock()
        fake_torch = types.SimpleNamespace(device=lambda value: types.SimpleNamespace(type="cpu"),
                                           float32="fp32")
        with fixture.trusted(), patch.object(models, "_require_versions", return_value={}), \
                patch.dict(sys.modules, {"torch": fake_torch,
                    "transformers": types.SimpleNamespace(AutoConfig=config, AutoModelForCausalLM=auto)}):
            loaded = models.load_base_model(self.root, fixture.key, execute=True,
                                            device="cpu", precision="float32")
        call = auto.from_pretrained.call_args
        self.assertEqual(call.args, (str(fixture.snapshot),))
        self.assertIs(call.kwargs["local_files_only"], True)
        self.assertIs(call.kwargs["trust_remote_code"], False)
        self.assertIs(call.kwargs["use_safetensors"], True)
        loaded.requires_grad_.assert_called_once_with(False)
        loaded.eval.assert_called_once()
        self.assertEqual(loaded._large_qubit_binding["optimizer_steps_executed"], 0)

    def test_qwen_prompt_explicitly_disables_thinking(self):
        fixture = self.fixture("qwen3_4b")
        with fixture.trusted():
            binding = models.authenticate_assets(self.root, fixture.key)
        tokenizer = MagicMock()
        tokenizer._large_qubit_binding = binding
        tokenizer.get_chat_template.return_value = "native Qwen fixture template"
        tokenizer.apply_chat_template.side_effect = ["rendered", [1, 2, 3]]
        tokenizer.encode.return_value = [1, 2, 3]
        result = models.render_prompt(tokenizer, fixture.key, "circuit task", answer_token_reserve=100)
        self.assertEqual(result["template_kwargs"], {"enable_thinking": False})
        for call in tokenizer.apply_chat_template.call_args_list:
            self.assertIs(call.kwargs["enable_thinking"], False)

    def test_complete_answer_overflow_fails_without_truncation(self):
        fixture = self.fixture()
        with fixture.trusted():
            binding = models.authenticate_assets(self.root, fixture.key)
        tokenizer = MagicMock()
        tokenizer._large_qubit_binding = binding
        tokenizer.encode.return_value = [1] * 925
        with self.assertRaisesRegex(ValueError, "truncation is forbidden"):
            models.render_prompt(tokenizer, fixture.key, "task", answer_token_reserve=100)
        result = models.render_prompt(tokenizer, fixture.key, "task", answer_token_reserve=99)
        self.assertEqual(result["prompt_tokens"] + result["answer_token_reserve"], 1024)


if __name__ == "__main__":
    unittest.main()
