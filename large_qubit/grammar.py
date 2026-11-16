"""CO2 answer contract over the preserved generic Phase 6 ByteLevel DFA.

The HF determinant is prepared before these operators. ID zero is Identity,
including when every answer position is zero; there is no separate HF token.
Prompt and answer are encoded separately and concatenated as token IDs.
"""

import json
from numbers import Integral

from phase6.grammar import CanonicalCircuitGrammar as _GenericGrammar

from .catalogue import validate_catalogue
from .common import require


LENGTHS = (32, 64)
ANSWER_VERSION = "co2-canonical-ops-json-v1"


def requested_length(length):
    require(type(length) is int and length in LENGTHS,
            "CO2 representation length must be exactly 32 or 64")
    return length


def operator_limit(catalogue):
    """Check the factored public interface, independently of screening labels."""
    validate_catalogue(catalogue)
    require(type(catalogue) is dict, "A public screened catalogue is required")
    templates = catalogue.get("templates")
    angles = catalogue.get("angles")
    require(type(templates) is list and len(templates) == 64,
            "The declared CO2 pool has exactly 64 screened templates")
    require(type(angles) is list and len(angles) == 8
            and all(type(x) in (int, float) for x in angles)
            and angles == [-.1, -.05, -.025, -.0125, .0125, .025, .05, .1],
            "The declared CO2 pool uses the registered eight angles")
    seen = set()
    for index, template in enumerate(templates):
        require(type(template) is dict and type(template.get("template_id")) is int
                and template["template_id"] == index,
                "Template IDs must be contiguous zero-based integers")
        wires = template.get("wires")
        kind = template.get("kind")
        require(kind in ("single", "double") and type(wires) is list
                and len(wires) == (2 if kind == "single" else 4)
                and all(type(w) is int and 0 <= w < 30 for w in wires)
                and len(set(wires)) == len(wires), "Invalid template wires")
        halves = len(wires) // 2
        require(all(w < 22 for w in wires[:halves])
                and all(w >= 22 for w in wires[halves:])
                and sorted(w % 2 for w in wires[:halves])
                == sorted(w % 2 for w in wires[halves:]),
                "Templates must preserve both spin populations from the HF partition")
        key = tuple(wires)
        require(key not in seen, "Duplicate screened template")
        seen.add(key)
    counts = catalogue.get("counts")
    require(type(counts) is dict and type(counts.get("operators")) is int
            and counts["operators"] == 513, "The pool must have 513 operators including Identity")
    if "max_operator_id" in catalogue:
        require(type(catalogue["max_operator_id"]) is int
                and catalogue["max_operator_id"] == 512, "Invalid maximum operator ID")
    return 512


def _checked_values(ops, *, maximum, length):
    """Inner sequence check after a caller has authenticated its catalogue once."""
    requested_length(length)
    require(type(ops) in (list, tuple) and len(ops) == length,
            "The answer must contain the exact requested operator count")
    require(all(isinstance(value, Integral) and not isinstance(value, bool)
                and 0 <= int(value) <= maximum for value in ops),
            "Operator IDs must be integers in the registered pool")
    return [int(value) for value in ops]


def checked_ops(ops, *, catalogue, length):
    return _checked_values(ops, maximum=operator_limit(catalogue), length=length)


def format_answer(ops, *, catalogue, length):
    return json.dumps({"ops": checked_ops(ops, catalogue=catalogue, length=length)},
                      separators=(",", ":"))


def parse_answer(text, *, catalogue, length):
    require(type(text) is str and text.isascii(), "Answer must be canonical ASCII JSON")
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "Duplicate answer key")
            value[key] = item
        return value
    try:
        value = json.loads(text, object_pairs_hook=pairs)
    except (ValueError, TypeError) as error:
        raise ValueError("Invalid circuit answer") from error
    require(type(value) is dict and set(value) == {"ops"},
            "Only the ops object is legal; HF is the implicit initial state")
    ops = checked_ops(value["ops"], catalogue=catalogue, length=length)
    require(format_answer(ops, catalogue=catalogue, length=length) == text,
            "Whitespace, leading zeros and alternative answer encodings are forbidden")
    return ops


class CircuitGrammar(_GenericGrammar):
    """The same sparse DFA drives training masks and constrained generation."""

    def __init__(self, tokenizer, catalogue, length, eos_token_id=None):
        maximum = operator_limit(catalogue)
        requested_length(length)
        super().__init__(tokenizer, maximum, length, eos_token_id=eos_token_id)
        self.catalogue_sha256 = catalogue["sha256"]
        self.answer_version = ANSWER_VERSION


def grammar_spec():
    return {
        "schema": ANSWER_VERSION,
        "canonical_answer": '{"ops":[decimal_integer_ids]}',
        "lengths": list(LENGTHS), "operator_ids": [0, 512],
        "identity_id": 0, "HF": "implicit_initial_determinant_not_an_answer_token",
        "boundary": "separate_prompt_and_answer_tokenization_then_ID_concatenation",
        "termination": "exact_JSON_then_one_registered_genuine_EOS",
        "reused_component": "phase6/grammar.py:CanonicalCircuitGrammar",
        "complete_answer_token_bound_including_EOS": {"32": 138, "64": 266},
        "bound_proof": "each_legal_nonEOS_ByteLevel_token_adds_at_least_one_canonical_ASCII_character",
        "training_and_generation": "identical_sparse_DFA_transitions",
        "padding": "outside_DFA_never_removes_genuine_supervised_EOS",
    }
