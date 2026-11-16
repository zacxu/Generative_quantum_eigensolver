"""Finite canonical circuit grammar shared by teacher forcing and generation.

This supports authenticated fast ByteLevel tokenizers, not arbitrary incremental
decoders. Multiple token paths can represent the same circuit text. The grammar
constrains token paths; it does not promise a Boltzmann distribution on circuits.
"""

from dataclasses import dataclass
import json
from numbers import Integral


HEADER = '{"ops":['
ALPHABET = frozenset(HEADER + ']}0123456789,')


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer, excluding booleans.")
    return int(value)


@dataclass(frozen=True)
class GrammarState:
    """Compressed DFA state; no circuit history or full prefix is retained."""

    phase: str = "header"
    index: int = 0
    digits: str = ""


class CanonicalCircuitGrammar:
    """Recognize exactly one no-whitespace JSON object and genuine EOS.

    EOS has no textual interpretation here: it is legal only after the closing
    brace. A finished batch's transport padding is outside this grammar. In
    particular, pad==EOS never removes the genuine supervised EOS position.
    """

    def __init__(self, tokenizer, max_operator_id, length, eos_token_id=None):
        self.tokenizer = tokenizer
        self.max_operator_id = _integer(max_operator_id, "max_operator_id")
        self.length = _integer(length, "length")
        if self.max_operator_id < 0 or self.length < 1:
            raise ValueError("A nonnegative maximum ID and positive length are required.")
        backend = getattr(tokenizer, "backend_tokenizer", None)
        if not getattr(tokenizer, "is_fast", False) or backend is None:
            raise ValueError("A verified fast ByteLevel tokenizer is required.")
        try:
            decoder = json.loads(backend.decoder.__getstate__())
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("Cannot verify the tokenizer decoder.") from error
        if not isinstance(decoder, dict) or decoder.get("type") != "ByteLevel":
            raise ValueError("Unsupported decoder: ASCII ByteLevel additivity is required.")
        self.decoder_spec = decoder
        vocabulary = sorted(set(tokenizer.get_vocab().values()))
        self.eos_token_id = _integer(
            tokenizer.eos_token_id if eos_token_id is None else eos_token_id,
            "eos_token_id",
        )
        specials = set(tokenizer.all_special_ids)
        if self.eos_token_id not in vocabulary or self.eos_token_id not in specials:
            raise ValueError("Genuine EOS must be a registered special vocabulary token.")
        self.initial_state = GrammarState()
        self.terminated_state = GrammarState("terminated")
        self._numbers = frozenset(map(str, range(self.max_operator_id + 1)))
        self._number_prefixes = frozenset(
            text[:i] for text in self._numbers for i in range(1, len(text) + 1)
        )
        # Each non-EOS legal token contributes at least one canonical ASCII
        # character. This bounds every legal path, including split-token paths.
        self.max_completion_tokens = (
            len(HEADER) + self.length * len(str(self.max_operator_id))
            + self.length - 1 + 2 + 1
        )
        self.worst_case_completion_tokens = self.max_completion_tokens
        decoded = backend.decode_batch([[token] for token in vocabulary],
                                       skip_special_tokens=False)
        self.token_pieces = {
            token: text for token, text in zip(vocabulary, decoded)
            if token not in specials and text and text.isascii()
            and set(text) <= ALPHABET
        }
        # Single-character fallback makes every remaining legal DFA prefix
        # completable. Reject an unsupported vocabulary rather than stranding
        # generation after a locally valid fragment.
        required = set(HEADER + ']},' + ''.join(self._numbers))
        if not required <= set(self.token_pieces.values()):
            raise ValueError("Tokenizer lacks canonical ASCII fallback paths.")
        self._trie = {}
        for token, piece in self.token_pieces.items():
            node = self._trie
            for character in piece:
                node = node.setdefault(character, {})
            node.setdefault(None, []).append(token)
        self._allowed_cache = {}
        self._transition_cache = {}

    def _character(self, state, character):
        if state.phase == "header":
            if character != HEADER[state.index]:
                return None
            if state.index + 1 == len(HEADER):
                return GrammarState("number", 0, "")
            return GrammarState("header", state.index + 1)
        if state.phase == "number":
            if character in "0123456789":
                digits = state.digits + character
                if digits in self._number_prefixes:
                    return GrammarState("number", state.index, digits)
                return None
            if state.digits not in self._numbers:
                return None
            if character == "," and state.index + 1 < self.length:
                return GrammarState("number", state.index + 1, "")
            if character == "]" and state.index + 1 == self.length:
                return GrammarState("brace")
            return None
        if state.phase == "brace" and character == "}":
            return GrammarState("complete")
        return None

    def allowed_from_state(self, state):
        """Return sparse legal token IDs before the next answer token."""
        if not isinstance(state, GrammarState):
            raise ValueError("An incremental GrammarState is required.")
        if state in self._allowed_cache:
            return self._allowed_cache[state]
        if state.phase == "complete":
            result = (self.eos_token_id,)
            self._transition_cache[(state, self.eos_token_id)] = self.terminated_state
        elif state.phase == "terminated":
            result = ()
        else:
            found = []
            stack = [(self._trie, state)]
            while stack:
                node, current = stack.pop()
                for token in node.get(None, ()):
                    found.append(token)
                    self._transition_cache[(state, token)] = current
                for character, child in node.items():
                    if character is not None:
                        following = self._character(current, character)
                        if following is not None:
                            stack.append((child, following))
            result = tuple(sorted(found))
            if not result:
                raise ValueError("Unrepresentable or invalid grammar state.")
        self._allowed_cache[state] = result
        return result

    def advance(self, state, token_id):
        token_id = _integer(token_id, "token_id")
        self.allowed_from_state(state)
        try:
            return self._transition_cache[(state, token_id)]
        except KeyError as error:
            raise ValueError(f"Token {token_id} is illegal after {state}.") from error

    def allowed_token_ids(self, answer_prefix_ids):
        state = self.initial_state
        for token in answer_prefix_ids:
            state = self.advance(state, token)
        return self.allowed_from_state(state)

    def canonical_answer(self, ops):
        if isinstance(ops, (str, bytes, dict)):
            raise ValueError("ops must be a sequence of integer operator IDs.")
        values = [_integer(value, "operator ID") for value in ops]
        if len(values) != self.length or any(
                not 0 <= value <= self.max_operator_id for value in values):
            raise ValueError("Operator count or ID lies outside the task grammar.")
        return json.dumps({"ops": values}, separators=(",", ":"))

    def encode_answer(self, ops):
        """Encode answer separately from prompt; append exactly one true EOS."""
        answer = self.canonical_answer(ops)
        tokens = list(self.tokenizer.encode(answer, add_special_tokens=False))
        tokens.append(self.eos_token_id)
        self.validate_answer_ids(tokens)
        return tokens

    def _walk_answer(self, answer_ids):
        tokens = tuple(_integer(token, "answer token") for token in answer_ids)
        state = self.initial_state
        masks = []
        for token in tokens:
            masks.append(self.allowed_from_state(state))
            state = self.advance(state, token)
        if state != self.terminated_state:
            raise ValueError("Answer must close the exact JSON object and end with EOS.")
        if len(tokens) > self.max_completion_tokens:
            raise ValueError("Canonical completion exceeded its character-based bound.")
        return tokens, masks

    def teacher_forcing_masks(self, answer_ids_including_eos):
        """Sparse masks aligned with all answer targets, including true EOS.

        The caller masks prompt/padding by position. These exact state transitions
        also drive generation; there is no unmasked training fallback.
        """
        _, masks = self._walk_answer(answer_ids_including_eos)
        return masks

    def validate_answer_ids(self, answer_ids_with_eos):
        tokens, _ = self._walk_answer(answer_ids_with_eos)
        text = self.tokenizer.decode(list(tokens[:-1]), skip_special_tokens=False,
                                     clean_up_tokenization_spaces=False)
        fragments = ''.join(self.token_pieces[token] for token in tokens[:-1])
        if text != fragments:
            raise ValueError("Tokenizer decoding violates verified fragment additivity.")
        try:
            ops = json.loads(text)["ops"]
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError("Invalid circuit answer.") from error
        if self.canonical_answer(ops) != text:
            raise ValueError("Noncanonical circuit answer.")
        return ops
