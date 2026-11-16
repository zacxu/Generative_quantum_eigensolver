"""Exact CO2 text and token-language tests, with no model loads."""

import copy
import unittest

from phase6.test_grammar import FakeByteLevel
from large_qubit.grammar import CircuitGrammar, format_answer, parse_answer, grammar_spec
from large_qubit.tests._section5_fixtures import catalogue_fixture


class GrammarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalogue = catalogue_fixture()

    def test_round_trip_both_lengths_and_identity(self):
        for length in (32, 64):
            for ops in ([0] * length, [512] * length, [i % 513 for i in range(length)]):
                text = format_answer(ops, catalogue=self.catalogue, length=length)
                self.assertEqual(parse_answer(text, catalogue=self.catalogue, length=length), ops)
                grammar = CircuitGrammar(FakeByteLevel(), self.catalogue, length)
                ids = grammar.encode_answer(ops)
                self.assertEqual(grammar.validate_answer_ids(ids), ops)
                self.assertEqual(len(grammar.teacher_forcing_masks(ids)), len(ids))
                self.assertEqual(grammar.teacher_forcing_masks(ids)[-1], (grammar.eos_token_id,))

    def test_bound_covers_and_is_attained_by_character_fallback(self):
        for length, bound in ((32, 138), (64, 266)):
            tokenizer = FakeByteLevel()
            grammar = CircuitGrammar(tokenizer, self.catalogue, length)
            text = format_answer([512] * length, catalogue=self.catalogue, length=length)
            path = tokenizer.characters(text) + [tokenizer.eos_token_id]
            self.assertEqual(len(path), bound)
            self.assertEqual(grammar.max_completion_tokens, bound)
            self.assertEqual(grammar.validate_answer_ids(path), [512] * length)
            self.assertEqual(grammar_spec()["complete_answer_token_bound_including_EOS"][str(length)], bound)

    def test_rejects_noncanonical_invalid_truncated_and_HF_tokens(self):
        text = format_answer([1] * 32, catalogue=self.catalogue, length=32)
        bad = [text[:-1], text + "\n", text.replace("1,", "01,", 1),
               text.replace("1,", "513,", 1), text.replace("1,", "true,", 1),
               text.replace("1,", "1.0,", 1), text.replace("1,", '"HF",', 1),
               text.replace('"ops":', '"HF":'), text.replace('[1,', '[1, '),
               '{"ops":[],"ops":[]}', '{"ops":[],"hf":true}', "HF"]
        for value in bad:
            with self.subTest(value=value[:35]), self.assertRaises(ValueError):
                parse_answer(value, catalogue=self.catalogue, length=32)
        for length in (True, 16, 0, 32.0):
            with self.assertRaises(ValueError):
                format_answer([0] * 32, catalogue=self.catalogue, length=length)
        grammar = CircuitGrammar(FakeByteLevel(), self.catalogue, 32)
        tokens = grammar.encode_answer([1] * 32)
        for path in (tokens[:-1], tokens + [grammar.eos_token_id], [grammar.eos_token_id]):
            with self.assertRaises(ValueError):
                grammar.validate_answer_ids(path)

    def test_refuses_altered_catalogue(self):
        altered = copy.deepcopy(self.catalogue)
        altered["templates"][0]["wires"][0] = True
        with self.assertRaises(ValueError):
            format_answer([0] * 32, catalogue=altered, length=32)


if __name__ == "__main__":
    unittest.main()
