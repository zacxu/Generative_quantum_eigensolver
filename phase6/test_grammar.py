"""Tokenizer/grammar checks only: no model weights, optimizer, or training."""

import json
from pathlib import Path
import random
import unittest

from phase3.catalogue import build_catalogue, canonical_json
from phase6.grammar import CanonicalCircuitGrammar, HEADER
from phase6.prompts import (_definition, build_prompt, compact_condition,
                           factored_catalogue, fixed_strftime, reconstruct_catalogue)


class _Decoder:
    def __init__(self, kind="ByteLevel"):
        self.kind = kind

    def __getstate__(self):
        return json.dumps({"type": self.kind}).encode()


class _Backend:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.decoder = _Decoder()

    def decode_batch(self, sequences, skip_special_tokens=False):
        return [self.tokenizer.decode(sequence, skip_special_tokens=skip_special_tokens)
                for sequence in sequences]


class FakeByteLevel:
    """A small additive vocabulary with intentionally merged/duplicate pieces."""

    is_fast = True
    eos_token_id = 1000
    pad_token_id = 1000
    all_special_ids = [1000, 1001]

    def __init__(self):
        self.pieces = {i: char for i, char in enumerate(sorted(set(HEADER + ']}0123456789,')))}
        self.pieces.update({100: HEADER, 101: '24,', 102: ']}' , 103: '24',
                            104: '{"ops":[24,0]}', 105: '24', 106: ' ',
                            107: '01', 108: '', 109: 'é', 1000: '<eos>', 1001: '0'})
        self.backend_tokenizer = _Backend(self)

    def get_vocab(self):
        # Duplicate decoded strings are permitted and have different token IDs.
        return {f"v{token}": token for token in self.pieces}

    def encode(self, text, add_special_tokens=False):
        result = []
        while text:
            choices = [(len(piece), -token, token) for token, piece in self.pieces.items()
                       if token not in self.all_special_ids and piece and text.startswith(piece)]
            if not choices:
                raise ValueError("Unencodable fixture text")
            _, _, token = max(choices)
            result.append(token); text = text[len(self.pieces[token]):]
        return result

    def decode(self, tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return ''.join(self.pieces[token] for token in tokens
                       if not skip_special_tokens or token not in self.all_special_ids)

    def characters(self, text):
        return [next(token for token, piece in self.pieces.items()
                     if piece == char and token not in self.all_special_ids) for char in text]


def condition(molecule="LiH"):
    ne, nq, length = (2, 4, 4) if molecule == "H2" else (4, 12, 16)
    catalogue = build_catalogue(ne, nq)
    return {"molecule": molecule, "bond_length": "0.74" if molecule == "H2" else "1.2",
            "bond_length_unit": "angstrom", "basis": "STO-3G", "electrons": ne,
            "qubits": nq, "hf_occupations": [1] * ne + [0] * (nq - ne),
            "mapping": "jordan_wigner", "spin_ordering": "alpha even, beta odd",
            "catalogue_sha256": catalogue["sha256"], "operator_definition": _definition(ne, nq),
            "requested_operator_length": length}


class GrammarTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeByteLevel()
        self.grammar = CanonicalCircuitGrammar(self.tokenizer, 24, 2)

    def test_merged_and_character_paths_parse_same_circuit(self):
        merged = self.grammar.encode_answer([24, 0])
        split = self.tokenizer.characters('{"ops":[24,0]}') + [1000]
        self.assertNotEqual(merged, split)
        self.assertEqual(self.grammar.validate_answer_ids(merged), [24, 0])
        self.assertEqual(self.grammar.validate_answer_ids(split), [24, 0])
        self.assertEqual(merged, [104, 1000])

    def test_teacher_forcing_is_exact_generation_transition(self):
        answer = self.tokenizer.characters('{"ops":[24,0]}') + [1000]
        masks = self.grammar.teacher_forcing_masks(answer)
        state = self.grammar.initial_state
        for index, token in enumerate(answer):
            self.assertEqual(masks[index], self.grammar.allowed_from_state(state))
            self.assertEqual(masks[index], self.grammar.allowed_token_ids(answer[:index]))
            self.assertIn(token, masks[index])
            state = self.grammar.advance(state, token)
        self.assertEqual(self.grammar.allowed_from_state(state), ())

    def test_eos_only_after_complete_and_pad_eos_is_a_target(self):
        self.assertNotIn(1000, self.grammar.allowed_token_ids([]))
        after_json = self.grammar.allowed_token_ids([104])
        self.assertEqual(after_json, (1000,))
        self.assertEqual(self.grammar.teacher_forcing_masks([104, 1000])[-1], (1000,))
        for bad in ([104], [1000], [104, 1000, 1000], [104, 1001]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.grammar.validate_answer_ids(bad)

    def test_range_length_leading_zero_and_noncanonical_text_rejected(self):
        texts = ['{"ops":[25,0]}', '{"ops":[01,0]}', '{"ops":[0]}',
                 '{"ops":[0,0,0]}', '{"ops":[0,0]}0']
        for text in texts:
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.grammar.validate_answer_ids(self.tokenizer.characters(text) + [1000])
        for token in (106, 108, 109, 1001):
            self.assertNotIn(token, self.grammar.token_pieces)

    def test_integer_schema_rejects_bool_float_and_wrong_lengths(self):
        for ops in ([True, 0], [0.0, 0], [-1, 0], [25, 0], [0], "00"):
            with self.subTest(ops=ops), self.assertRaises(ValueError):
                self.grammar.encode_answer(ops)
        with self.assertRaises(ValueError):
            self.grammar.advance(self.grammar.initial_state, True)

    def test_compressed_state_forgets_completed_id_values(self):
        states = []
        for prefix in ('{"ops":[0,', '{"ops":[24,'):
            state = self.grammar.initial_state
            for token in self.tokenizer.characters(prefix):
                state = self.grammar.advance(state, token)
            states.append(state)
        self.assertEqual(states[0], states[1])
        self.assertEqual(self.grammar.allowed_from_state(states[0]),
                         self.grammar.allowed_from_state(states[1]))

    def test_duplicate_token_pieces_have_same_semantics(self):
        state = self.grammar.advance(self.grammar.initial_state, 100)
        self.assertIn(103, self.grammar.allowed_from_state(state))
        self.assertIn(105, self.grammar.allowed_from_state(state))
        self.assertEqual(self.grammar.advance(state, 103), self.grammar.advance(state, 105))

    def test_unsupported_decoder_missing_fallback_and_fake_eos_fail(self):
        self.tokenizer.backend_tokenizer.decoder.kind = "WordPiece"
        with self.assertRaises(ValueError):
            CanonicalCircuitGrammar(self.tokenizer, 24, 2)
        self.tokenizer = FakeByteLevel()
        del self.tokenizer.pieces[next(k for k, v in self.tokenizer.pieces.items() if v == '7')]
        with self.assertRaises(ValueError):
            CanonicalCircuitGrammar(self.tokenizer, 24, 2)
        with self.assertRaises(ValueError):
            CanonicalCircuitGrammar(FakeByteLevel(), 24, 2, eos_token_id=100)

    def test_character_bound_covers_every_split_token(self):
        for max_id, length, expected in ((24, 4, 22), (736, 16, 74)):
            grammar = CanonicalCircuitGrammar(self.tokenizer, max_id, length)
            text = grammar.canonical_answer([max_id] * length)
            ids = self.tokenizer.characters(text) + [1000]
            self.assertEqual(len(ids), expected)
            self.assertEqual(grammar.max_completion_tokens, expected)
            self.assertEqual(grammar.worst_case_completion_tokens, expected)
            grammar.validate_answer_ids(ids)


class PromptTests(unittest.TestCase):
    def test_factored_recipe_reconstructs_full_catalogue(self):
        for molecule, ne, nq in (("H2", 2, 4), ("LiH", 4, 12)):
            factors = factored_catalogue(condition(molecule))
            self.assertEqual(canonical_json(reconstruct_catalogue(factors)),
                             canonical_json(build_catalogue(ne, nq)))
            factors["id_rule"] = "0+8*t+k"
            with self.assertRaises(ValueError):
                reconstruct_catalogue(factors)

    def test_conditions_reject_energy_fields_and_changed_physics(self):
        for key, value in (("energy_ha", -7), ("basis", "other"),
                           ("catalogue_sha256", "wrong"), ("hf_occupations", [0] * 12)):
            payload = condition(); payload[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                compact_condition(payload)
        task = compact_condition(condition())
        self.assertIn("ID=1+8*t+k", task)
        self.assertIn("out_low=c*in_low-s*in_high", task)
        self.assertNotIn("energy_ha", task)

    def test_fixed_template_clock_is_locale_independent(self):
        self.assertEqual(fixed_strftime("%d %B %Y"), "18 October 2026")
        self.assertEqual(fixed_strftime("%Y-%m-%d %% %H:%M:%S %z"),
                         "2026-10-18 % 00:00:00 +0000")
        with self.assertRaises(ValueError):
            fixed_strftime("%c")


class ActualTokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not Path("phase6/model_lock.json").exists():
            raise unittest.SkipTest("Authenticated tokenizer-only cache not present")
        from phase6.models import load_tokenizer
        cls.tokenizers = {key: load_tokenizer(key) for key in
                          ("qwen3_4b", "smollm3_3b", "phi4_mini", "gpt2_xl")}

    def test_all_actual_tokenizers_support_every_id_and_same_masks(self):
        for key, tokenizer in self.tokenizers.items():
            for maximum, length in ((24, 4), (736, 16)):
                grammar = CanonicalCircuitGrammar(tokenizer, maximum, length)
                for operator_id in range(maximum + 1):
                    with self.subTest(model=key, maximum=maximum, operator_id=operator_id):
                        answer = grammar.encode_answer([operator_id] * length)
                        masks = grammar.teacher_forcing_masks(answer)
                        self.assertTrue(all(token in mask for token, mask in zip(answer, masks)))
                        self.assertEqual(masks[-1], (tokenizer.eos_token_id,))
                        self.assertLessEqual(len(answer), grammar.max_completion_tokens)

    def test_all_actual_native_wrappers_fit_and_fix_smol_clock(self):
        common = {}
        for molecule in ("H2", "LiH"):
            common[molecule] = set()
            for key, tokenizer in self.tokenizers.items():
                prompt = build_prompt(tokenizer, key, condition(molecule))
                reserve = 22 if molecule == "H2" else 74
                self.assertLessEqual(len(prompt["prompt_ids"]) + reserve, 1024)
                common[molecule].add(prompt["boundary_metadata"]["common_task_sha256"])
                self.assertEqual(prompt, build_prompt(tokenizer, key, condition(molecule)))
                if key == "smollm3_3b":
                    self.assertIn("18 October 2026", prompt["prompt_text"])
                if key == "gpt2_xl":
                    self.assertTrue(prompt["prompt_text"].endswith("\nAnswer:\n"))
            self.assertEqual(len(common[molecule]), 1)

    def test_random_legal_generation_terminates_and_roundtrips(self):
        rng = random.Random(90817)
        for key, tokenizer in self.tokenizers.items():
            grammar = CanonicalCircuitGrammar(tokenizer, 736, 16)
            for _ in range(12):
                state, answer = grammar.initial_state, []
                while state != grammar.terminated_state:
                    token = rng.choice(grammar.allowed_from_state(state))
                    answer.append(token); state = grammar.advance(state, token)
                    self.assertLessEqual(len(answer), 74)
                parsed = grammar.validate_answer_ids(answer)
                self.assertEqual(len(parsed), 16)
                self.assertEqual(grammar.teacher_forcing_masks(answer)[-1], (grammar.eos_token_id,))


if __name__ == "__main__":
    unittest.main(verbosity=2)
