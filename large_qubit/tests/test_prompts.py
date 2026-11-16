"""Factoring and boundary guards; native tokenizer certification is separate."""

from hashlib import sha256
import unittest
from unittest.mock import patch

from phase6.test_grammar import FakeByteLevel
from large_qubit.models import MODEL_SPECS
from large_qubit.prompts import (build_prompt, compact_task, factored_templates,
    load_prompt_tokenizer, reconstruct_templates)
from large_qubit.tests._section5_fixtures import catalogue_fixture


class PromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalogue = catalogue_fixture()

    def test_factored_wire_recipe_round_trip(self):
        encoded = factored_templates(self.catalogue)
        self.assertEqual(reconstruct_templates(encoded),
                         [tuple(row["wires"]) for row in self.catalogue["templates"]])
        for invalid in (encoded + ",00", encoded.replace(",", "", 1), encoded.replace("0", "Z", 1)):
            with self.assertRaises(ValueError):
                reconstruct_templates(invalid)

    def test_task_accepts_structural_alias_only_and_has_no_labels(self):
        for alias in ("CO2_T01", "CO2_T02", "CO2_T03", "CO2_V01", "CO2_I01", "CO2_I02"):
            task = compact_task(alias, self.catalogue, 64)
            self.assertIn(self.catalogue["sha256"], task)
            self.assertIn("64 integer IDs0..512", task)
            self.assertNotIn("reference", task.lower())
            self.assertNotIn("energy", task.lower())
            self.assertNotIn("coupling", task.lower())
            self.assertNotIn("score", task.lower())
        with self.assertRaises(ValueError):
            compact_task({"geometry_id": "CO2_T02", "energy": -123}, self.catalogue, 64)

    def test_native_boundaries_and_1024_guard(self):
        tokenizer = FakeByteLevel()
        tokenizer.chat_template = None
        tokenizer._large_qubit_binding = {"model_key": "gpt2_xl",
            "revision": MODEL_SPECS["gpt2_xl"]["revision"], "metadata_authenticated": True}
        # This additive fixture can encode the minimal synthetic task/wrapper.
        tokenizer.pieces.update({2000: "tiny", 2001: "\nAnswer:\n"})
        with patch("large_qubit.prompts.compact_task", return_value="tiny"):
            result = build_prompt(tokenizer, "gpt2_xl", "CO2_T02", self.catalogue, 64)
        self.assertEqual(result["answer_token_reserve"], 266)
        self.assertEqual(result["context_certificate"]["total_upper_bound"], 268)
        tokenizer.pieces[2002] = "X"
        with patch("large_qubit.prompts.compact_task", return_value="X" * 759):
            with self.assertRaisesRegex(ValueError, "truncation"):
                build_prompt(tokenizer, "gpt2_xl", "CO2_T02", self.catalogue, 64)
        tokenizer._large_qubit_binding["metadata_authenticated"] = False
        with self.assertRaisesRegex(ValueError, "authenticated"):
            build_prompt(tokenizer, "gpt2_xl", "CO2_T02", self.catalogue, 64)

    def test_no_implicit_tokenizer_load(self):
        with self.assertRaisesRegex(ValueError, "execute=True"):
            load_prompt_tokenizer(".", "gpt2_xl")


if __name__ == "__main__":
    unittest.main()
