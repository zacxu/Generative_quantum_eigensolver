"""Tiny token-cache/loss fixtures: no pretrained models, optimizers or energies."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from phase6.objective import masked_sequence_nll, accumulated_objective
from phase6.test_grammar import FakeByteLevel
from large_qubit import training_data as data
from large_qubit.common import canonical_bytes, digest, read_json, sha256_file
from large_qubit.tests._section5_fixtures import catalogue_fixture


class FittingView:
    def __init__(self, fixture, consumer): self.fixture, self.consumer = fixture, consumer
    def fitting_rows(self, alias): return tuple(self.fixture.rows[alias])
    def diagnostic_rows(self, alias): raise AssertionError("Diagnostic payload opened")
    def weights(self, alias):
        self.fixture.weight_calls.append(self.consumer)
        uniform = self.consumer == "gpt2_xl_uniform_lora"
        return data.corpus.geometry_weights(self.fitting_rows(alias), mode="uniform" if uniform else "weighted",
            lambda_scale=None if uniform else (.1 if self.consumer.startswith("qwen") else .25))
    @property
    def source_acquisition_cost(self):
        return {"release_sha256": data.RELEASE_SHA256, "logical_energy_requests": 13090, "wall_seconds": 200.}
    @property
    def separate_verification_cost(self): return {"hamiltonian_actions": 12, "wall_seconds": 2.}


class TokenCacheFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.catalogue = catalogue_fixture()
        path = self.root / "large_qubit/catalogues/co2_screened_v1.json"
        path.parent.mkdir(parents=True); path.write_bytes(canonical_bytes(self.catalogue))
        self.tokenizer = FakeByteLevel(); self.tokenizer.pieces[2000] = "PROMPT"
        self.weight_calls = []
        self.rows = {alias: [] for alias in data.corpus.TRAINING}
        for index in range(4):
            for geometry_index, alias in enumerate(data.corpus.TRAINING):
                self.rows[alias].append({"row_id": f"{alias}_v2_F{index+1:06d}", "geometry_id": alias,
                    "request_ordinal": 388 + index*3 + geometry_index, "role": "fitting",
                    "ops": [24 if index == 0 else index*111] * 32,
                    "energy_hartree": -10. - .01*index, "catalogue_sha256": self.catalogue["sha256"]})
        patches = [patch.object(data, "ROWS_PER_GEOMETRY", 4),
            patch.object(data, "_source_bindings", return_value={"synthetic_source": "f"*64}),
            patch.object(data, "_metadata_identity", side_effect=self.metadata),
            patch.object(data, "_tokenizer_identity", side_effect=self.tokenizer_identity),
            patch.object(data.corpus, "load_release", side_effect=lambda root, consumer_id, **kwargs: FittingView(self, consumer_id)),
            patch.object(data, "build_prompt", side_effect=self.prompt)]
        for context in patches:
            context.start(); self.addCleanup(context.stop)

    def metadata(self, root, model_key):
        return {"model_key": model_key, "revision": data.MODEL_SPECS[model_key]["revision"],
            "model_lock_sha256": "a"*64, "metadata_directory": "synthetic/local-tokenizer",
            "metadata_files_sha256": {"tokenizer.json": "b"*64}, "native_chat_template_sha256": None}

    def tokenizer_identity(self, root, model_key, tokenizer):
        return {**self.metadata(root, model_key), "vocabulary_sha256": digest(tokenizer.get_vocab()),
            "vocabulary_size": len(tokenizer.get_vocab()), "maximum_token_id": 2000,
            "eos_token_id": 1000, "pad_token_id": 1000, "decoder": {"type": "ByteLevel"}}

    def prompt(self, tokenizer, model_key, alias, catalogue, length):
        ids = [2000] * (data.corpus.TRAINING.index(alias)+1)
        return {"text": "PROMPT"*len(ids), "input_ids": ids, "prompt_tokens": len(ids), "answer_token_reserve": 138,
            "context_certificate": {"geometry_id": alias, "model_key": model_key,
                "catalogue_sha256": catalogue["sha256"], "boundary": data.BOUNDARY_VERSION,
                "prompt_tokens": len(ids), "total_upper_bound": len(ids)+138}}

    def prepare(self, model_key="gpt2_xl"):
        return data.prepare_training_data(self.root, model_key=model_key, tokenizer=self.tokenizer, execute=True)

    def load(self, saved, condition="gpt2_xl_weighted_lora"):
        return data.load_training_data(self.root, condition_id=condition, manifest_path=saved["manifest_path"],
                                       expected_manifest_sha256=saved["manifest_sha256"])

    def rehash_payload(self, saved, filename, mutate):
        path = self.root / data.CACHE_ROOT / saved["manifest"]["model_key"] / filename
        if filename.endswith("jsonl"):
            value = [json.loads(line) for line in path.read_bytes().splitlines()]
            mutate(value); content = b"".join(canonical_bytes(row)+b"\n" for row in value)
        else:
            value = read_json(path); mutate(value); content = canonical_bytes(value)+b"\n"
        path.write_bytes(content)
        manifest_path = self.root / saved["manifest_path"]
        manifest = read_json(manifest_path)
        manifest["files"][filename].update(bytes=len(content), sha256=sha256_file(path))
        manifest_path.write_bytes(canonical_bytes(manifest)+b"\n")
        pin = sha256_file(manifest_path); manifest_path.with_suffix(".sha256").write_text(pin+"\n")
        return pin


class CacheTests(TokenCacheFixture):
    def test_shared_GPT_cache_has_identical_rows_and_no_cached_labels_or_weights(self):
        saved = self.prepare(); self.assertEqual(self.weight_calls, [])
        weighted = self.load(saved); uniform = self.load(saved, "gpt2_xl_uniform_lora")
        self.assertEqual(weighted.row_ids, uniform.row_ids)
        self.assertEqual(weighted.cache_manifest_sha256, uniform.cache_manifest_sha256)
        self.assertEqual(len(weighted), 12)
        for epoch in range(5): self.assertEqual(weighted.row_order(epoch), uniform.row_order(epoch))
        batch = uniform.collate([0, 1, 2]); self.assertEqual(batch["loss_multipliers"].tolist(), [1., 1., 1.])
        for row in weighted.rows: self.assertNotIn("energy_hartree", row)
        contents = (self.root / data.CACHE_ROOT / "gpt2_xl" / "rows.jsonl").read_text()
        self.assertNotIn("energy", contents); self.assertNotIn("multiplier", contents)
        self.assertEqual(weighted.summary["source_acquisition_cost"]["logical_energy_requests"], 13090)
        self.assertEqual(weighted.summary["separate_verification_cost"]["hamiltonian_actions"], 12)
        summary = weighted.summary; summary["passes"] = 0
        self.assertEqual(weighted.summary["passes"], 5)

    def test_Qwen_has_separate_cache_and_model_cross_access_is_rejected(self):
        gpt = self.prepare(); qwen = self.prepare("qwen3_4b")
        self.assertNotEqual(gpt["manifest_path"], qwen["manifest_path"])
        self.assertNotEqual(gpt["manifest_sha256"], qwen["manifest_sha256"])
        qdata = self.load(qwen, "qwen3_4b_weighted_lora")
        self.assertEqual(qdata.row_ids, self.load(gpt).row_ids)
        self.assertEqual(qdata.row_order(0), self.load(gpt).row_order(0))
        with self.assertRaisesRegex(ValueError, "another model"):
            self.load(gpt, "qwen3_4b_weighted_lora")

    def test_balanced_five_pass_order_visits_every_row_once_per_pass(self):
        dataset = self.load(self.prepare()); all_visits = []
        for epoch in range(5):
            order = dataset.row_order(epoch)
            self.assertEqual(sorted(order), list(range(12)))
            self.assertEqual([dataset.rows[i]["geometry_id"] for i in order], list(data.corpus.TRAINING)*4)
            all_visits.extend(order)
        self.assertEqual(len(all_visits), 60)
        self.assertNotEqual(dataset.row_order(0), dataset.row_order(1))
        for epoch in (True, -1, 5):
            with self.assertRaises(ValueError): dataset.row_order(epoch)

    def test_memory_stress_uses_longest_complete_cached_sequences_with_stable_ties(self):
        dataset = self.load(self.prepare())
        lengths = [len(dataset._prompts[row['geometry_id']]['input_ids'])+len(row['answer_ids'])
                   for row in dataset._records]
        expected = tuple(sorted(range(len(dataset)), key=lambda index: (-lengths[index], index)))
        self.assertEqual(dataset.memory_stress_indices(4), expected[:4])
        self.assertEqual(dataset.memory_stress_indices(len(dataset)), expected)
        batch = dataset.collate(dataset.memory_stress_indices(4))
        self.assertEqual(int(batch['attention_mask'].sum()), sum(lengths[i] for i in expected[:4]))
        for count in (0, True, len(dataset)+1):
            with self.assertRaises(ValueError): dataset.memory_stress_indices(count)

    def test_complete_cache_reuse_is_idempotent_and_verification_loads_no_tokenizer(self):
        saved = self.prepare()
        folder = self.root / data.CACHE_ROOT / "gpt2_xl"
        before = {p.name: (sha256_file(p), p.stat().st_mtime_ns) for p in folder.iterdir()}
        with patch.object(data, "load_prompt_tokenizer", side_effect=AssertionError("Implicit tokenizer load")):
            verified = data.verify_training_cache(self.root, saved["manifest_path"], saved["manifest_sha256"])
            again = self.prepare()
        self.assertTrue(verified["all_passed"])
        self.assertFalse(verified["tokenizer_loaded_by_verification"])
        self.assertEqual(again, saved)
        self.assertEqual(before, {p.name: (sha256_file(p), p.stat().st_mtime_ns) for p in folder.iterdir()})

    def test_partial_cache_is_preserved_and_exact_inputs_rebuilt(self):
        folder = self.root / data.CACHE_ROOT / "gpt2_xl"; folder.mkdir(parents=True)
        (folder / "rows.jsonl").write_bytes(b'interrupted-cache')
        saved = self.prepare()
        preserved = list(folder.parent.glob("gpt2_xl_incomplete_*"))
        self.assertEqual(len(preserved), 1)
        self.assertEqual((preserved[0] / "rows.jsonl").read_bytes(), b'interrupted-cache')
        self.assertTrue(data.verify_training_cache(self.root, saved["manifest_path"], saved["manifest_sha256"])["all_passed"])

    def test_missing_execute_and_changed_pin_or_source_are_refused(self):
        with self.assertRaisesRegex(ValueError, "execute=True"):
            data.prepare_training_data(self.root, model_key="gpt2_xl", tokenizer=self.tokenizer)
        saved = self.prepare()
        with self.assertRaises(ValueError):
            data.verify_training_cache(self.root, saved["manifest_path"], "0"*64)
        with patch.object(data, "_source_bindings", return_value={"changed": "0"*64}):
            with self.assertRaisesRegex(ValueError, "source identity"):
                self.load(saved)

    def test_rehashed_answer_label_or_DFA_masks_cannot_change_supervision(self):
        saved = self.prepare()
        pin = self.rehash_payload(saved, "rows.jsonl", lambda rows: rows[0].update(geometry_id="CO2_V01"))
        with self.assertRaisesRegex(ValueError, "another row"):
            data.verify_training_cache(self.root, saved["manifest_path"], pin)

    def test_rehashed_legal_set_cannot_admit_illegal_tokens(self):
        saved = self.prepare()
        def mutate(graph): graph["allowed_token_sets"][0].append(2000)
        pin = self.rehash_payload(saved, "grammar.json", mutate)
        with self.assertRaisesRegex(ValueError, "DFA"):
            data.verify_training_cache(self.root, saved["manifest_path"], pin)


class LossTests(TokenCacheFixture):
    def setUp(self):
        super().setUp(); self.saved = self.prepare(); self.dataset = self.load(self.saved)

    def test_right_padding_retains_genuine_EOS_and_causal_prompt_boundary(self):
        batch = self.dataset.collate([0, 3])
        labels, attention, ids = batch["labels"], batch["attention_mask"], batch["input_ids"]
        self.assertGreater(int((attention[0] == 0).sum()), 0)
        for b in range(2):
            used = int(attention[b].sum()); start = used-batch["answer_lengths"][b]
            self.assertTrue(bool((labels[b, :start] == -100).all()))
            self.assertEqual(int(labels[b, used-1]), 1000)
            self.assertTrue(bool((labels[b, used:] == -100).all()))
            self.assertTrue(bool((ids[b, used:] == 1000).all()))
            self.assertTrue(bool((batch["position_ids"][b, :used] == torch.arange(used)).all()))
        for bad in ([], [True], [-1], [12], [0, 0]):
            with self.assertRaises(ValueError): self.dataset.collate(bad)

    def test_cached_group_loss_and_gradients_equal_preserved_sequence_objective(self):
        batch = self.dataset.collate([0, 3]); b, length = batch["input_ids"].shape
        logits = torch.randn(b, length, 2001, dtype=torch.float64, requires_grad=True)
        legal = [[tuple(self.dataset._graph["allowed_token_sets"][g]) for g in self.dataset._records[i]["grammar_group_ids"]] for i in (0, 3)]
        expected = masked_sequence_nll(logits, batch["labels"], legal)
        measured = self.dataset.sequence_nll(logits, batch)
        torch.testing.assert_close(measured, expected, rtol=1e-13, atol=1e-12)
        first = torch.autograd.grad(measured.sum(), logits, retain_graph=True)[0]
        second = torch.autograd.grad(expected.sum(), logits)[0]
        torch.testing.assert_close(first, second, rtol=1e-13, atol=1e-13)
        self.assertTrue(bool((first[:, -1] == 0).all()))
        self.assertTrue(bool((first[:, :, 1999] == 0).all()))
        self.assertGreater(float(first[0, 0].abs().sum()), 0.)  # Last prompt position predicts the first answer token.

    def test_actual_window_normalizer_applies_multiplier_once(self):
        indices = [0, 1, 3, 4]
        batch = self.dataset.collate(indices)
        logits = torch.randn(4, batch["input_ids"].shape[1], 2001, dtype=torch.float64)
        nll = self.dataset.sequence_nll(logits, batch)
        full = accumulated_objective(nll, batch["loss_multipliers"], window_example_count=4)
        parts = sum(accumulated_objective(nll[i:i+1], batch["loss_multipliers"][i:i+1], window_example_count=4) for i in range(4))
        torch.testing.assert_close(full, parts)
        self.assertNotEqual(float(full), float((nll*batch["loss_multipliers"]).sum()/batch["loss_multipliers"].sum()))

    def test_half_logits_accumulate_in_float32_and_invalid_legal_logits_fail(self):
        batch = self.dataset.collate([0])
        logits = torch.zeros(1, batch["input_ids"].shape[1], 2001, dtype=torch.bfloat16)
        self.assertEqual(self.dataset.sequence_nll(logits, batch).dtype, torch.float32)
        rows, positions, _, allowed = batch["loss_groups"][0]
        logits[rows[0], positions[0], allowed[0]] = float("nan")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            self.dataset.sequence_nll(logits, batch)

    def test_device_grammar_tensors_are_reused_and_labels_cannot_shift(self):
        batch = self.dataset.collate([0]); second = self.dataset.collate([3])
        common = {tuple(group[3].tolist()): group[3].data_ptr() for group in batch["loss_groups"]}
        self.assertTrue(any(common.get(tuple(group[3].tolist())) == group[3].data_ptr() for group in second["loss_groups"]))
        batch["labels"].fill_(-100)
        logits = torch.zeros(1, batch["input_ids"].shape[1], 2001)
        with self.assertRaisesRegex(ValueError, "causal-shift"):
            self.dataset.sequence_nll(logits, batch)


class NativeEOSMetadataTests(unittest.TestCase):
    """GPT-style metadata omits tokenizer_config EOS; model config is authoritative."""
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name); self.directory = self.root/'metadata'; self.directory.mkdir()
        self.tokenizer = FakeByteLevel()
        self.tokenizer._large_qubit_binding = {'model_key':'gpt2_xl',
            'revision':data.MODEL_SPECS['gpt2_xl']['revision'], 'metadata_authenticated':True}
        self.raw = {'model':{'vocab':self.tokenizer.get_vocab()}, 'decoder':{'type':'ByteLevel'},
            'added_tokens':[{'content':'v1000','id':1000,'special':True}]}
        self.config = {}
        self.model = {'eos_token_id':1000}
        self.identity = {'model_key':'gpt2_xl','revision':data.MODEL_SPECS['gpt2_xl']['revision'],
            'metadata_directory':'metadata','metadata_files_sha256':{},'model_lock_sha256':'a'*64,
            'native_chat_template_sha256':None}
        context = patch.object(data, '_metadata_identity', return_value=self.identity)
        context.start(); self.addCleanup(context.stop); self.save()

    def save(self):
        for name,value in (('tokenizer.json',self.raw),('tokenizer_config.json',self.config),('config.json',self.model)):
            (self.directory/name).write_bytes(canonical_bytes(value))

    def test_model_config_special_EOS_is_valid_when_tokenizer_config_omits_it(self):
        result = data._tokenizer_identity(self.root,'gpt2_xl',self.tokenizer)
        self.assertEqual(result['eos_token_id'],1000)
        self.assertFalse((self.directory/'special_tokens_map.json').exists())

    def test_present_tokenizer_or_added_token_declarations_must_agree(self):
        for declaration in ('v1001', {'content':'v1001'}):
            self.config = {'eos_token':declaration}; self.save()
            with self.assertRaisesRegex(ValueError,'EOS declaration'):
                data._tokenizer_identity(self.root,'gpt2_xl',self.tokenizer)
        self.config = {'eos_token':'v1000','added_tokens_decoder':{'1000':{'content':'v1000','special':False}}}; self.save()
        with self.assertRaisesRegex(ValueError,'added-token EOS'):
            data._tokenizer_identity(self.root,'gpt2_xl',self.tokenizer)

    def test_model_EOS_and_special_registration_cannot_be_weakened(self):
        self.model['eos_token_id'] = 1001; self.save()
        with self.assertRaisesRegex(ValueError,'model metadata'):
            data._tokenizer_identity(self.root,'gpt2_xl',self.tokenizer)
        self.model['eos_token_id'] = 1000; self.raw['added_tokens'][0]['special'] = False; self.save()
        with self.assertRaisesRegex(ValueError,'native special token'):
            data._tokenizer_identity(self.root,'gpt2_xl',self.tokenizer)

    def test_ByteLevel_omitted_defaults_match_but_explicit_flags_stay_exact(self):
        defaults = {'type':'ByteLevel','add_prefix_space':True,'trim_offsets':True,'use_regex':True}
        self.assertEqual(data._bytelevel_decoder_spec({'type':'ByteLevel','add_prefix_space':True,'trim_offsets':True}), defaults)
        self.assertEqual(data._bytelevel_decoder_spec(defaults), defaults)
        self.raw['decoder']['use_regex'] = False; self.save()
        with self.assertRaisesRegex(ValueError,'native ByteLevel decoder'):
            data._tokenizer_identity(self.root,'gpt2_xl',self.tokenizer)
        for bad in ({'type':'ByteLevel','unknown_flag':True}, {'type':'ByteLevel','use_regex':1}, {'type':'WordPiece'}):
            with self.subTest(decoder=bad), self.assertRaisesRegex(ValueError,'Unsupported'):
                data._bytelevel_decoder_spec(bad)


if __name__ == "__main__": unittest.main()
