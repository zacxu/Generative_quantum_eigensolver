"""Immutable fast-cache consumers: synthetic tokenizers only, no model loading."""

from copy import deepcopy
from contextlib import ExitStack
from unittest.mock import patch

from large_qubit import training_cache_access as fast
from large_qubit import training_data as data
from large_qubit.common import canonical_bytes, sha256_file
from large_qubit.tests.test_training_data import TokenCacheFixture


class FastCacheTests(TokenCacheFixture):
    def setUp(self):
        super().setUp()
        release = self.root / data.corpus.RELEASE_PATH
        release.parent.mkdir(parents=True)
        release.write_bytes(canonical_bytes({"synthetic_fitting_release": True})+b"\n")
        self.release_sha = sha256_file(release)
        release.with_suffix(".sha256").write_text(self.release_sha+"\n", encoding="ascii")
        for context in (patch.object(data, "RELEASE_SHA256", self.release_sha),
                        patch.object(fast, "RELEASE_SHA256", self.release_sha)):
            context.start(); self.addCleanup(context.stop)
        self.pins = {}
        context = patch.object(fast, "TOKEN_CACHE_SHA256", self.pins)
        context.start(); self.addCleanup(context.stop)

    def prepare(self, model_key="gpt2_xl"):
        saved = data.prepare_training_data(self.root, model_key=model_key, tokenizer=self.tokenizer,
            execute=True, expected_release_sha256=self.release_sha)
        self.pins[model_key] = saved["manifest_sha256"]
        return saved

    def verify_fast(self, saved, **changes):
        arguments = dict(manifest_path=saved["manifest_path"], expected_manifest_sha256=saved["manifest_sha256"],
                         expected_release_sha256=self.release_sha)
        arguments.update(changes)
        return fast.verify_training_cache(self.root, **arguments)

    def load_fast(self, saved, condition="gpt2_xl_weighted_lora"):
        return fast.load_training_data(self.root, condition_id=condition, manifest_path=saved["manifest_path"],
            expected_manifest_sha256=saved["manifest_sha256"], expected_release_sha256=self.release_sha)

    def forbid_semantic_reconstruction(self):
        stack = ExitStack()
        for owner, name in ((data, "_read_cache"), (data, "_grammar_from_fragments"),
                            (data, "format_answer"), (data, "load_prompt_tokenizer"),
                            (data.CircuitGrammar, "teacher_forcing_masks")):
            stack.enter_context(patch.object(owner, name, side_effect=AssertionError("Repeated deep semantic validation")))
        return stack

    def test_verification_authenticates_bytes_without_release_labels_or_semantic_replay(self):
        saved = self.prepare()
        deep = data.verify_training_cache(self.root, saved["manifest_path"], saved["manifest_sha256"],
                                          expected_release_sha256=self.release_sha)
        with self.forbid_semantic_reconstruction(), \
             patch.object(data.corpus, "load_release", side_effect=AssertionError("Opened release labels")):
            verified = self.verify_fast(saved)
        for key, value in deep.items(): self.assertEqual(verified[key], value)
        self.assertFalse(verified["release_label_payloads_read"])
        self.assertFalse(verified["diagnostic_payloads_read"])
        self.assertEqual(verified["hamiltonian_actions"], 0)

    def test_all_three_fits_preserve_frozen_rows_schedule_and_weight_math(self):
        gpt, qwen = self.prepare(), self.prepare("qwen3_4b")
        datasets = []
        for condition, saved in (("gpt2_xl_weighted_lora", gpt), ("gpt2_xl_uniform_lora", gpt),
                                 ("qwen3_4b_weighted_lora", qwen)):
            old = data.load_training_data(self.root, condition_id=condition, manifest_path=saved["manifest_path"],
                expected_manifest_sha256=saved["manifest_sha256"], expected_release_sha256=self.release_sha)
            with self.forbid_semantic_reconstruction(): current = self.load_fast(saved, condition)
            self.assertEqual(current.summary, old.summary)
            self.assertEqual(current._multipliers, old._multipliers)
            self.assertEqual(current.row_ids, old.row_ids)
            datasets.append(current)
            self.assertEqual(current.summary["source_acquisition_cost"]["logical_energy_requests"], 13090)
            self.assertEqual(current.summary["separate_verification_cost"]["hamiltonian_actions"], 12)
        self.assertEqual(datasets[0].cache_manifest_sha256, datasets[1].cache_manifest_sha256)
        self.assertNotEqual(datasets[0].cache_manifest_sha256, datasets[2].cache_manifest_sha256)
        for epoch in range(5):
            self.assertEqual(datasets[0].row_order(epoch), datasets[1].row_order(epoch))
            self.assertEqual(datasets[0].row_order(epoch), datasets[2].row_order(epoch))
        self.assertEqual(datasets[1]._multipliers, (1.,)*12)

    def test_payload_tamper_is_rejected_even_when_json_remains_valid(self):
        saved = self.prepare()
        path = self.root / data.CACHE_ROOT / "gpt2_xl" / "prompts.json"
        path.write_bytes(path.read_bytes().replace(b"PROMPT", b"MUTATE", 1))
        with self.assertRaisesRegex(ValueError, "payload changed"): self.verify_fast(saved)

    def test_rehashed_mutation_cannot_supply_a_new_trusted_pin(self):
        saved = self.prepare()
        new_pin = self.rehash_payload(saved, "rows.jsonl", lambda rows: rows[0].update(row_id="CO2_T01_v2_D000001"))
        self.assertNotEqual(new_pin, saved["manifest_sha256"])
        with self.assertRaisesRegex(ValueError, "audited cache pin"):
            self.verify_fast(saved, expected_manifest_sha256=new_pin)
        with self.assertRaisesRegex(ValueError, "trusted manifest pin"): self.verify_fast(saved)

    def test_manifest_sidecar_tamper_is_rejected(self):
        saved = self.prepare()
        (self.root / saved["manifest_path"]).with_suffix(".sha256").write_text("0"*64+"\n")
        with self.assertRaisesRegex(ValueError, "trusted manifest pin"): self.verify_fast(saved)

    def test_changed_frozen_source_and_tokenizer_metadata_are_rejected(self):
        saved = self.prepare()
        with patch.object(data, "_source_bindings", return_value={"changed_source": "0"*64}):
            with self.assertRaisesRegex(ValueError, "source identity"): self.verify_fast(saved)
        changed = deepcopy(self.metadata(self.root, "gpt2_xl"))
        changed["metadata_files_sha256"]["tokenizer.json"] = "0"*64
        with patch.object(data, "_metadata_identity", return_value=changed):
            with self.assertRaisesRegex(ValueError, "tokenizer metadata"): self.verify_fast(saved)

    def test_release_manifest_and_release_pin_are_authenticated_without_labels(self):
        saved = self.prepare()
        with self.assertRaisesRegex(ValueError, "approved fitting release"):
            self.verify_fast(saved, expected_release_sha256="0"*64)
        (self.root / data.corpus.RELEASE_PATH).write_bytes(b'{"changed":"release"}\n')
        with self.assertRaisesRegex(ValueError, "release manifest changed"): self.verify_fast(saved)

    def test_paths_cannot_route_to_another_model_or_leave_namespace(self):
        saved = self.prepare(); self.prepare("qwen3_4b")
        for path in (saved["manifest_path"].replace("gpt2_xl", "qwen3_4b"),
                     "../manifest.json", str(self.root / saved["manifest_path"]),
                     saved["manifest_path"].replace("/", "\\")):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.verify_fast(saved, manifest_path=path)
        with self.assertRaisesRegex(ValueError, "another model"):
            self.load_fast(saved, "qwen3_4b_weighted_lora")

    def test_consumer_roles_and_exact_weight_row_coverage_are_enforced(self):
        saved = self.prepare()
        with self.assertRaisesRegex(ValueError, "registered fitting consumers"):
            self.load_fast(saved, "training_circuit_retrieval")
        self.rows["CO2_T01"] = self.rows["CO2_T01"][:-1]
        with self.assertRaisesRegex(ValueError, "cover exactly"): self.load_fast(saved)

    def test_source_cost_and_separate_integrity_cost_cannot_be_hidden(self):
        saved = self.prepare()
        from large_qubit.tests.test_training_data import FittingView
        class WrongCost(FittingView):
            @property
            def source_acquisition_cost(self):
                return {**super().source_acquisition_cost, "logical_energy_requests": 6552}
        with patch.object(data.corpus, "load_release", side_effect=lambda root, consumer_id, **kwargs: WrongCost(self, consumer_id)):
            with self.assertRaisesRegex(ValueError, "offline costs"): self.load_fast(saved)
