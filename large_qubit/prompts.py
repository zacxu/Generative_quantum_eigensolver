"""Energy-free factored CO2 prompts and exhaustive-path context certificates.

Only cached tokenizer files may be loaded here. No model weights are loaded or
hashed by the tokenizer-only loader. The context proof uses the complete DFA
character bound, covering every legal token path rather than sampled answers.
"""

from hashlib import sha256
from pathlib import Path

from .common import require
from .endpoints import model_geometry_payload
from .grammar import CircuitGrammar, format_answer, operator_limit, requested_length
from .models import CONTEXT_LIMIT, MODEL_SPECS, inspect_assets, _require_versions


WIRE_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRST"
PROMPT_VERSION = "co2-factored-64-template-prompt-v1"
BOUNDARY_VERSION = "separate-prompt-answer-tokenization-v1"


def factored_templates(catalogue):
    operator_limit(catalogue)
    return ",".join("".join(WIRE_ALPHABET[w] for w in template["wires"])
                    for template in catalogue["templates"])


def reconstruct_templates(text):
    """Decode the prompt's wire tuples, without accessing screening energies."""
    require(type(text) is str, "A factored template string is required")
    parts = text.split(",")
    require(len(parts) == 64 and all(len(p) in (2, 4) for p in parts),
            "Exactly 64 single/double templates are required")
    require(all(set(p) <= set(WIRE_ALPHABET) for p in parts), "Invalid wire symbol")
    return [tuple(WIRE_ALPHABET.index(char) for char in part) for part in parts]


def compact_task(alias, catalogue, length):
    """Accept an alias, not caller-supplied energy-bearing condition objects."""
    requested_length(length)
    templates = factored_templates(catalogue)
    data = model_geometry_payload(alias)
    return (
        f"CO2 linear O-C-O; both bonds R={data['bond_length_angstrom']:g} Angstrom; "
        "STO-3G; charge0; 30qubits,22electrons,Na=Nb=11. "
        "JW alpha=even,beta=odd; wire0=MSB. Initial HF: wires0..21=1,22..29=0.\n"
        "Apply whole-qubit excitation rotations left to right. Singles act on "
        "01/10; doubles on 0011/1100; other local states unchanged. "
        "In that ordered pair use [[c,-s],[s,c]],c=cos(theta/2),s=sin(theta/2).\n"
        "Wire alphabet 0123456789ABCDEFGHIJKLMNOPQRST means wires0..29. "
        "Each tuple has 2(single) or 4(double) wires; templates t=0..63 in order:\n"
        f"{templates}\n"
        "Angles k=0..7: [-.1,-.05,-.025,-.0125,.0125,.025,.05,.1]. "
        "ID0=Identity; ID=1+8*t+k. Repetition allowed. "
        f"Catalogue SHA256={catalogue['sha256']}.\n"
        f'Return exactly {{"ops":[IDs]}} with {length} integer IDs0..512, no spaces or explanation, then EOS.'
    )


def load_prompt_tokenizer(workspace, model_key, *, execute=False):
    """Explicit, local-only tokenizer load; authenticate metadata without weights."""
    require(execute is True, "Tokenizer loading requires explicit execute=True")
    binding = inspect_assets(workspace, model_key, full_hash=False)
    packages = _require_versions()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(Path(workspace).resolve() / binding["metadata_directory"]),
        use_fast=True, local_files_only=True, trust_remote_code=False)
    require(tokenizer.is_fast and tokenizer.eos_token_id is not None,
            "A native fast tokenizer and genuine EOS are required")
    if model_key == "qwen3_4b":
        require(sha256(tokenizer.get_chat_template().encode()).hexdigest()
                == binding["native_chat_template_sha256"], "Cached Qwen template changed")
    else:
        require(tokenizer.chat_template is None, "GPT2 uses the completion interface")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.model_max_length = CONTEXT_LIMIT
    tokenizer._large_qubit_binding = {**binding, "packages": packages,
        "tokenizer_loaded": True, "tokenizer_only_load": True}
    return tokenizer


def build_prompt(tokenizer, model_key, alias, catalogue, length):
    """Return native prompt IDs plus a rigorous complete-answer reserve; no truncation."""
    require(model_key in MODEL_SPECS, "Unknown registered model")
    binding = getattr(tokenizer, "_large_qubit_binding", {})
    require(binding.get("model_key") == model_key
            and binding.get("revision") == MODEL_SPECS[model_key]["revision"]
            and binding.get("metadata_authenticated") is True,
            "Use the metadata-authenticated registered tokenizer")
    grammar = CircuitGrammar(tokenizer, catalogue, length)
    task = compact_task(alias, catalogue, length)
    template_hash = None
    kwargs = dict(MODEL_SPECS[model_key]["template_kwargs"])
    if model_key == "gpt2_xl":
        require(getattr(tokenizer, "chat_template", None) is None,
                "GPT2 uses its native completion boundary")
        text = task + "\nAnswer:\n"
        ids = list(tokenizer.encode(text, add_special_tokens=False))
    else:
        template = tokenizer.get_chat_template()
        template_hash = sha256(template.encode()).hexdigest()
        require(template_hash == binding["native_chat_template_sha256"],
                "Native Qwen template changed after authentication")
        messages = [{"role": "user", "content": task}]
        text = tokenizer.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=True, **kwargs)
        ids = list(tokenizer.apply_chat_template(messages, tokenize=True,
            add_generation_prompt=True, **kwargs))
        require(ids == list(tokenizer.encode(text, add_special_tokens=False)),
                "Native chat rendering and encoding disagree")
    reserve = grammar.max_completion_tokens
    require(len(ids) + reserve <= CONTEXT_LIMIT,
            "Prompt plus complete worst-case answer exceeds 1024; truncation is forbidden")
    # Verify an actual legal longest-width character-token path, in addition to
    # the DFA proof that every other path is at most this long. This does not
    # assume that the tokenizer's preferred encoding is the longest path.
    longest_text = format_answer([512] * length, catalogue=catalogue, length=length)
    characters = {}
    for token, piece in grammar.token_pieces.items():
        if len(piece) == 1:
            characters.setdefault(piece, token)
    longest_ids = [characters[char] for char in longest_text] + [grammar.eos_token_id]
    require(len(longest_ids) == reserve, "The complete answer path bound changed")
    grammar.validate_answer_ids(longest_ids)
    certificate = {
        "schema": "co2-context-certificate-v1", "prompt_version": PROMPT_VERSION,
        "model_key": model_key, "model_revision": MODEL_SPECS[model_key]["revision"],
        "geometry_id": alias, "length": length, "catalogue_sha256": catalogue["sha256"],
        "context_limit": CONTEXT_LIMIT, "prompt_tokens": len(ids),
        "complete_answer_bound_including_EOS": reserve,
        "total_upper_bound": len(ids) + reserve,
        "all_legal_answer_token_paths_covered": True,
        "longest_character_fallback_path_verified": True,
        "proof": "nonempty_ASCII_ByteLevel_token_pieces_bound_tokens_by_maximum_answer_characters_plus_EOS",
        "prompt_text_sha256": sha256(text.encode()).hexdigest(),
        "common_task_sha256": sha256(task.encode()).hexdigest(),
        "native_chat_template_sha256": template_hash, "template_kwargs": kwargs,
        "model_lock_sha256": binding.get("model_lock_sha256"),
        "packages": dict(binding.get("packages", {})),
        "boundary": BOUNDARY_VERSION, "energies_or_examples_in_prompt": False,
        "model_weights_loaded": False, "downloads_executed": 0,
    }
    return {"text": text, "input_ids": ids, "prompt_tokens": len(ids),
            "answer_token_reserve": reserve, "context_certificate": certificate}
