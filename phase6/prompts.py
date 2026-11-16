"""Energy-free common physics conditions and native, deterministic wrappers."""

from datetime import date
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json

from phase3.catalogue import ANGLES, build_catalogue, canonical_json


FIXED_TEMPLATE_DATE = "2026-10-18"
BOUNDARY_VERSION = "separate-prompt-answer-tokenization-v1"
CONDITION_FIELDS = frozenset({
    "molecule", "bond_length", "bond_length_unit", "basis", "electrons", "qubits",
    "hf_occupations", "mapping", "spin_ordering", "catalogue_sha256",
    "operator_definition", "requested_operator_length",
})


def _digest(text):
    return sha256(text.encode("utf-8")).hexdigest()


def _definition(electrons, qubits):
    return {
        "occupied": list(range(electrons)), "virtual": list(range(electrons, qubits)),
        "single_rule": "occupied i, virtual a, equal spin parity; lexicographic (i,a)",
        "double_rule": "occupied i<j, virtual a<b, i%2+j%2=a%2+b%2; lexicographic (i,j,a,b)",
        "ordering": "identity ID0, then each single and each double in lexicographic order, each with ascending angles",
        "angles_radian": list(ANGLES),
        "gates": "PennyLane whole SingleExcitation/DoubleExcitation half-angle rotations, wires in listed order",
    }


def factored_catalogue(condition):
    """Validate a physical whitelist and expose the complete deterministic recipe.

    This only reconstructs catalogue metadata; it opens no molecular data,
    reference energies, corpus labels, or scoring files.
    """
    if not isinstance(condition, dict) or set(condition) != CONDITION_FIELDS:
        raise ValueError("Exactly the energy-free physical condition is required.")
    molecule = condition["molecule"]
    if molecule not in {"H2", "LiH"}:
        raise ValueError("Unknown physical system.")
    ne, nq, length = (2, 4, 4) if molecule == "H2" else (4, 12, 16)
    if (type(condition["electrons"]) is not int or type(condition["qubits"]) is not int
            or type(condition["requested_operator_length"]) is not int
            or condition["electrons"] != ne or condition["qubits"] != nq
            or condition["requested_operator_length"] != length):
        raise ValueError("System counts or frozen circuit length changed.")
    try:
        bond = Decimal(str(condition["bond_length"]))
    except (InvalidOperation, ValueError) as error:
        raise ValueError("A finite positive bond length is required.") from error
    if not bond.is_finite() or bond <= 0:
        raise ValueError("A finite positive bond length is required.")
    expected = {
        "bond_length_unit": "angstrom", "basis": "STO-3G",
        "hf_occupations": [1] * ne + [0] * (nq - ne),
        "mapping": "jordan_wigner", "spin_ordering": "alpha even, beta odd",
        "operator_definition": _definition(ne, nq),
    }
    # Canonical JSON also distinguishes booleans from occupation integers.
    for key, value in expected.items():
        if canonical_json(condition[key]) != canonical_json(value):
            raise ValueError(f"Physical condition changed: {key}.")
    catalogue = build_catalogue(ne, nq)
    if condition["catalogue_sha256"] != catalogue["sha256"]:
        raise ValueError("Condition catalogue fingerprint changed.")
    return {
        "schema": "whole-excitation-factors-v1", "electrons": ne, "qubits": nq,
        "occupied": [0, ne], "virtual": [ne, nq], "range_end_exclusive": True,
        "spin": "alpha even, beta odd", "single": "(i,a); i%2=a%2",
        "double": "(i,j,a,b); i<j,a<b; i%2+j%2=a%2+b%2",
        "tuples": "lexicographic singles followed by lexicographic doubles",
        "angles": list(ANGLES), "identity_id": 0,
        "id_rule": "1+8*t+k; t,k zero-based tuple,ascending-angle indices",
        "gate": "whole SingleExcitation/DoubleExcitation on listed wires",
        "rotation": "low=01/high=10 or low=0011/high=1100; out_low=c*low-s*high; out_high=s*low+c*high; c=cos(angle/2),s=sin(angle/2)",
        "execution": "prepare HF once; IDs left to right; wire 0 most significant",
        "sha256": catalogue["sha256"],
    }


def reconstruct_catalogue(factors):
    """Check every factored rule and reproduce the full saved Phase3 object."""
    if not isinstance(factors, dict):
        raise ValueError("A complete factored catalogue is required.")
    ne, nq = factors.get("electrons"), factors.get("qubits")
    catalogue = build_catalogue(ne, nq)
    molecule = "H2" if (ne, nq) == (2, 4) else "LiH" if (ne, nq) == (4, 12) else None
    if molecule is None:
        raise ValueError("Unsupported catalogue system.")
    condition = {
        "molecule": molecule, "bond_length": "1", "bond_length_unit": "angstrom",
        "basis": "STO-3G", "electrons": ne, "qubits": nq,
        "hf_occupations": [1] * ne + [0] * (nq - ne),
        "mapping": "jordan_wigner", "spin_ordering": "alpha even, beta odd",
        "catalogue_sha256": catalogue["sha256"], "operator_definition": _definition(ne, nq),
        "requested_operator_length": 4 if molecule == "H2" else 16,
    }
    if canonical_json(factors) != canonical_json(factored_catalogue(condition)):
        raise ValueError("Factored catalogue rules changed.")
    return catalogue


def compact_condition(condition):
    factors = factored_catalogue(condition)
    reconstructed = reconstruct_catalogue(factors)
    ne, nq = factors["electrons"], factors["qubits"]
    # All models receive exactly this same task text. The catalogue formula
    # replaces hundreds of repeated entries, not any physical information.
    return (
        "Propose a low-energy molecular circuit. Return only canonical JSON "
        '{"ops":[IDs]} with no whitespace or other text, then EOS.\n'
        f"Molecule={condition['molecule']}; bond={condition['bond_length']} angstrom; "
        f"basis=STO-3G; electrons={ne}; qubits={nq}; mapping=Jordan-Wigner; "
        f"HF={''.join(map(str, condition['hf_occupations']))}; "
        f"length={condition['requested_operator_length']}; IDs=0..{len(reconstructed['entries'])-1}.\n"
        f"Wires are zero-based, wire0 is most-significant; alpha even/beta odd. "
        f"Occupied=range(0,{ne}), virtual=range({ne},{nq}), end excluded. "
        "Enumerate singles (i,a), occupied i/virtual a, i%2=a%2; then doubles "
        "(i,j,a,b), occupied i<j/virtual a<b, i%2+j%2=a%2+b%2. "
        "Sort each tuple list lexicographically. Angles in radians ascending="
        "[-0.1,-0.05,-0.025,-0.0125,0.0125,0.025,0.05,0.1]. "
        "ID0=Identity; ID=1+8*t+k for zero-based tuple index t in singles then "
        "doubles and zero-based angle index k. Each ID is a whole PennyLane "
        "SingleExcitation or DoubleExcitation on its tuple wires. "
        "For local low=01/high=10 (single) or low=0011/high=1100 (double), "
        "out_low=c*in_low-s*in_high, out_high=s*in_low+c*in_high, "
        "c=cos(angle/2),s=sin(angle/2); other basis amplitudes unchanged. "
        "Prepare HF once and execute IDs left to right; repeats and Identity allowed.\n"
        f"Catalogue SHA256={reconstructed['sha256']}."
    )


def fixed_strftime(format_string):
    """Locale-independent replacement for a template's runtime clock helper."""
    fixed = date.fromisoformat(FIXED_TEMPLATE_DATE)
    replacements = {
        "d": f"{fixed.day:02d}", "m": f"{fixed.month:02d}", "Y": str(fixed.year),
        "B": "October", "b": "Oct", "A": "Sunday", "a": "Sun",
        "H": "00", "M": "00", "S": "00", "z": "+0000", "Z": "UTC", "%": "%",
    }
    result, index = [], 0
    while index < len(format_string):
        if format_string[index] != "%":
            result.append(format_string[index]); index += 1; continue
        index += 1
        if index == len(format_string) or format_string[index] not in replacements:
            raise ValueError("Unsupported deterministic template date format.")
        result.append(replacements[format_string[index]]); index += 1
    return ''.join(result)


def build_prompt(tokenizer, model_key, condition):
    """Render native wrappers and independently encode the prompt boundary.

    Answers are separately tokenized by grammar.encode_answer and concatenated
    as IDs. Retokenizing prompt+answer is deliberately not the model interface.
    """
    task = compact_condition(condition)
    if model_key not in {"qwen3_4b", "smollm3_3b", "phi4_mini", "gpt2_xl"}:
        raise ValueError("Unknown model interface.")
    kwargs = {}
    template = None
    if model_key == "gpt2_xl":
        text = task + "\nAnswer:\n"
        ids = list(tokenizer.encode(text, add_special_tokens=False))
        interface = "plain_completion"
    else:
        template = tokenizer.get_chat_template()
        if not isinstance(template, str) or not template:
            raise ValueError("A native chat template is required.")
        if model_key in {"qwen3_4b", "smollm3_3b"}:
            kwargs["enable_thinking"] = False
        if model_key == "smollm3_3b":
            kwargs["strftime_now"] = fixed_strftime
        messages = [{"role": "user", "content": task}]
        text = tokenizer.apply_chat_template(messages, tokenize=False,
                    add_generation_prompt=True, **kwargs)
        ids = list(tokenizer.apply_chat_template(messages, tokenize=True,
                    add_generation_prompt=True, **kwargs))
        if ids != list(tokenizer.encode(text, add_special_tokens=False)):
            raise ValueError("Native rendered prompt and tokenized prompt disagree.")
        interface = "native_chat"
    metadata = {
        "version": BOUNDARY_VERSION, "model_key": model_key, "interface": interface,
        "prompt_token_count": len(ids), "prompt_text_sha256": _digest(text),
        "common_task_sha256": _digest(task),
        "native_template_sha256": _digest(template) if template else None,
        "template_kwargs": {key: value for key, value in kwargs.items() if not callable(value)},
        "template_clock_date": FIXED_TEMPLATE_DATE if model_key == "smollm3_3b" else None,
        "answer_encoding": "separate add_special_tokens=False; append one genuine EOS; concatenate IDs",
        "energies_or_examples_in_prompt": False,
        "catalogue_sha256": condition["catalogue_sha256"],
    }
    return {"prompt_ids": ids, "prompt_text": text, "common_task": task,
            "boundary_metadata": metadata}
