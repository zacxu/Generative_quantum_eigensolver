"""Exact, unoptimized common-gate resources for authenticated circuit recipes.

This is algebra and counting only: no Hamiltonian, state, model or energy is
loaded. Fermionic excitation generators include their Jordan--Wigner strings;
whole-qubit excitations intentionally do not. Both are expanded into commuting
Pauli rotations and then {H, RX, RZ, CX}; HF preparation adds 22 X gates.
"""

from copy import deepcopy
from fractions import Fraction
import math

from .catalogue import validate_catalogue, validate_operator_ids
from .common import digest, require
from .definitions import geometry


SCHEMA = 'section9-exact-common-gate-resources-v1'
N_QUBITS = 30
_PAULI_PRODUCT = {
    ('X', 'Y'): (1j, 'Z'), ('Y', 'X'): (-1j, 'Z'),
    ('Y', 'Z'): (1j, 'X'), ('Z', 'Y'): (-1j, 'X'),
    ('Z', 'X'): (1j, 'Y'), ('X', 'Z'): (-1j, 'Y'),
}


def _multiply_words(left, right):
    phase, result = 1, []
    for a, b in zip(left, right):
        if a == 'I':
            result.append(b)
        elif b == 'I' or a == b:
            result.append(a if b == 'I' else 'I')
        else:
            factor, word = _PAULI_PRODUCT[a, b]
            phase *= factor
            result.append(word)
    return phase, ''.join(result)


def _multiply(left, right):
    result = {}
    for a, ca in left.items():
        for b, cb in right.items():
            phase, word = _multiply_words(a, b)
            result[word] = result.get(word, 0) + ca * cb * phase
    return {word: value for word, value in result.items() if value != 0}


def _ladder(wire, create, n_qubits, *, jordan_wigner):
    # a=(X+iY)/2, a^dagger=(X-iY)/2; preceding lower-numbered
    # wires carry Z because wire zero is the most significant occupation.
    prefix = 'Z' * wire if jordan_wigner else 'I' * wire
    suffix = 'I' * (n_qubits - wire - 1)
    return {prefix + 'X' + suffix: .5,
            prefix + 'Y' + suffix: -.5j if create else .5j}


def _commute(left, right):
    return sum(a != 'I' and b != 'I' and a != b for a, b in zip(left, right)) % 2 == 0


def _pauli_rotation(word, theta, coefficient):
    """exp(theta*i*coefficient*P/2), implemented without extra global phase."""
    support = [wire for wire, letter in enumerate(word) if letter != 'I']
    require(bool(support), 'An excitation generator cannot have an identity term')
    result, changes = [], []
    for wire in support:
        if word[wire] == 'X':
            result.append({'gate': 'h', 'wires': [wire]})
            changes.append({'gate': 'h', 'wires': [wire]})
        elif word[wire] == 'Y':
            result.append({'gate': 'rx', 'wires': [wire], 'angle_radians': math.pi / 2,
                           'angle_pi_fraction': [1, 2]})
            changes.append({'gate': 'rx', 'wires': [wire], 'angle_radians': -math.pi / 2,
                            'angle_pi_fraction': [-1, 2]})
    ladder = [{'gate': 'cx', 'wires': [a, b]} for a, b in zip(support, support[1:])]
    factor = Fraction.from_float(float(-coefficient))
    result.extend(ladder)
    result.append({'gate': 'rz', 'wires': [support[-1]],
                   'angle_radians': -theta * coefficient,
                   'angle_native_fraction': [factor.numerator, factor.denominator],
                   'native_angle_hex': float(theta).hex()})
    result.extend(reversed(ladder))
    result.extend(reversed(changes))
    return result


def excitation_decomposition(wires, theta, *, family, n_qubits=N_QUBITS):
    """Return an exact ideal-unitary decomposition, also usable by small fixtures.

    For both families U=exp(theta*(T-T^dagger)/2). Whole-qubit T maps
    01->10 (or 0011->1100) with no spectator signs. Fermionic T applies
    annihilators on the first half in listed order followed by creators on the
    second half in listed order; e.g. T=a_b^dagger a_a^dagger a_j a_i.
    Every nonzero Pauli term in T-T^dagger commutes, so this factorization
    has no Trotter approximation. Dyadic coefficients are represented exactly.
    """
    require(type(n_qubits) is int and 1 <= n_qubits <= 62, 'Invalid qubit count')
    require(family in ('whole_qubit', 'fermionic'), 'Unknown native excitation family')
    require(type(wires) in (list, tuple) and len(wires) in (2, 4)
            and all(type(w) is int and 0 <= w < n_qubits for w in wires)
            and len(set(wires)) == len(wires), 'Expected two or four distinct valid wires')
    require(type(theta) in (int, float) and math.isfinite(theta), 'Expected a finite excitation angle')
    theta, wires = float(theta), tuple(wires)
    half = len(wires) // 2
    operator = {'I' * n_qubits: 1.}
    for position, wire in enumerate(wires):
        create = position < half if family == 'whole_qubit' else position >= half
        factor = _ladder(wire, create, n_qubits, jordan_wigner=family == 'fermionic')
        operator = _multiply(factor, operator)
    generator = {word: coefficient - complex(coefficient).conjugate()
                 for word, coefficient in operator.items()
                 if coefficient - complex(coefficient).conjugate() != 0}
    require(generator and all(value.real == 0 for value in generator.values()),
            'Excitation generator must be nonzero and anti-Hermitian')
    words = sorted(generator)
    require(all(_commute(a, b) for i, a in enumerate(words) for b in words[i + 1:]),
            'Noncommuting generator terms cannot use this exact factorization')
    terms, sequence = [], []
    for word in words:
        coefficient = generator[word].imag
        fraction = Fraction.from_float(float(coefficient))
        terms.append({'pauli_word': word,
                      'generator_imaginary_coefficient': [fraction.numerator, fraction.denominator]})
        if theta != 0.:
            sequence.extend(_pauli_rotation(word, theta, coefficient))
    return {'family': family, 'wires': list(wires), 'angle_radians': theta,
            'angle_hex': theta.hex(), 'pauli_terms': terms,
            'pairwise_commuting': True, 'exact_zero_omitted': theta == 0.,
            'common_gate_sequence': sequence}


def _metrics(sequence, n_qubits=N_QUBITS):
    layers, entangling_layers = [0] * n_qubits, [0] * n_qubits
    counts = {name: 0 for name in ('h', 'rx', 'rz', 'cx', 'x')}
    for gate in sequence:
        name, wires = gate['gate'], gate['wires']
        counts[name] += 1
        layer = 1 + max(layers[w] for w in wires)
        entangling = int(len(wires) == 2) + max(entangling_layers[w] for w in wires)
        for wire in wires:
            layers[wire], entangling_layers[wire] = layer, entangling
    return {'gates': len(sequence), 'one_qubit_count': len(sequence) - counts['cx'],
            'two_qubit_count': counts['cx'], 'depth': max(layers),
            'two_qubit_depth': max(entangling_layers), 'gate_counts': counts}


def resource_report(query, catalogue):
    """Report the Section 9 winner recipe, without evaluating its energy.

    ``geometry_id`` is optional structural metadata and is bound when supplied.
    The catalogue remains the exact public screened 30-qubit CO2 catalogue.
    The explicit common sequence excludes HF; prepending hf_preparation_sequence
    gives the sequence counted by resources_including_hf.
    """
    validate_catalogue(catalogue)
    require(catalogue['qubits'] == N_QUBITS and catalogue['electrons'] == 22,
            'Resources require the registered 30-qubit, 22-electron CO2 catalogue')
    require(type(query) is dict and type(query.get('kind')) is str, 'A native circuit query is required')
    kind = query['kind']
    fields = {'hf': {'kind'}, 'discrete': {'kind', 'ops'}, 'vqe': {'kind', 'parameters'}}
    require(kind in fields and set(query) <= fields[kind] | {'geometry_id'}
            and fields[kind] <= set(query), 'Unexpected native circuit query fields')
    if 'geometry_id' in query:
        geometry(query['geometry_id'])  # Physical coordinates only, no energies.
    recipe, sequence = [], []
    if kind == 'discrete':
        require(type(query['ops']) is list, 'Discrete operator IDs must be a list')
        ids = validate_operator_ids(query['ops'], catalogue, length=32)
        for position, identifier in enumerate(ids):
            if identifier == 0:
                recipe.append({'position': position, 'operator_id': 0, 'exact_identity_omitted': True})
                continue
            template, angle = divmod(identifier - 1, 8)
            row = excitation_decomposition(catalogue['templates'][template]['wires'],
                catalogue['angles'][angle], family='whole_qubit')
            start = len(sequence)
            sequence.extend(row.pop('common_gate_sequence'))
            recipe.append({'position': position, 'operator_id': identifier, 'template_index': template,
                           **row, 'common_sequence_range': [start, len(sequence)]})
        family = 'whole_qubit'
    elif kind == 'vqe':
        parameters = query['parameters']
        require(type(parameters) is list and len(parameters) == 24
                and all(type(x) in (int, float) and math.isfinite(x) for x in parameters),
                'Conventional VQE requires exactly 24 finite parameters')
        for position, (template, theta) in enumerate(zip(catalogue['vqe_template_indices'], parameters)):
            row = excitation_decomposition(catalogue['templates'][template]['wires'], theta, family='fermionic')
            start = len(sequence)
            sequence.extend(row.pop('common_gate_sequence'))
            recipe.append({'position': position, 'template_index': template,
                           **row, 'common_sequence_range': [start, len(sequence)]})
        family = 'fermionic'
    else:
        family = 'hf_only'
    hf = [{'gate': 'x', 'wires': [wire]} for wire in range(22)]
    without, including = _metrics(sequence), _metrics(hf + sequence)
    return {'schema_version': SCHEMA, 'query': deepcopy(query), 'query_sha256': digest(query),
        'catalogue_sha256': catalogue['sha256'], 'qubits': N_QUBITS,
        'wire_order': 'wire_0_most_significant; alpha_even_beta_odd',
        'native_family': family, 'native_recipe': recipe,
        'common_gate_sequence': sequence, 'common_sequence_sha256': digest(sequence),
        'hf_preparation_sequence': hf, 'hf_preparation': 'from_all_zero; X_on_wires_0_through_21',
        'resources_excluding_hf': without, 'resources_including_hf': including,
        'depth': without['depth'], 'two_qubit_count': without['two_qubit_count'],
        'headline_resources_exclude_hf': True,
        'gate_conventions': {'rx': 'exp(-i*angle*X/2)', 'rz': 'exp(-i*angle*Z/2)',
            'cx': 'first_wire_control_second_wire_target', 'order': 'left_to_right',
            'variable_angle': 'native_angle_hex times angle_native_fraction exactly',
            'constant_angle': 'angle_pi_fraction defines exact multiple of pi'},
        'decomposition': 'exact_commuting_Pauli_rotations; no_Trotter_approximation',
        'optimization': 'none; only_ID0_and_exact_zero_angles_omitted; no_cross_gate_cancellation',
        'connectivity': 'all_to_all_logical_CX; no_routing_or_hardware_optimization',
        'depth_definition': 'ASAP_dependency_depth; each_common_gate_unit_duration; disjoint_wires_parallel',
        'two_qubit_depth_definition': 'same_dependency_graph_with_only_CX_unit_weight',
        'global_phase': 'no_extra_global_phase', 'hardware_advantage_claimed': False,
        'scientific_energy_requests': 0}

