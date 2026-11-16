"""Fresh CO2 excitation IDs and a declared training-integral screening rule.

Screening uses analytic HF matrix elements and diagonal-Fock gaps only.  It
does not query a candidate energy, a correlated reference, or held-out data.
The public catalogue excludes all scalar coupling and gap labels.
"""

from itertools import combinations
from numbers import Integral
import re

from .common import canonical_bytes, digest, require
from .protocol import BENCHMARK_ID


ANGLES = (-0.1, -0.05, -0.025, -0.0125, 0.0125, 0.025, 0.05, 0.1)
TRAINING_ALIASES = ('CO2_T01', 'CO2_T02', 'CO2_T03')
MAIN_TEMPLATE_COUNT = 64
VQE_TEMPLATE_COUNT = 24
COUPLING_FLOOR_HA = 1e-10
GAP_FLOOR_HA = 1e-8


def _integer(value, name):
    require(not isinstance(value, bool) and isinstance(value, Integral),
            f'{name} must be an integer, excluding booleans')
    return int(value)


def full_templates(electrons=22, qubits=30):
    """Canonical 88 singles then 2,596 doubles for the declared CO2 problem."""
    electrons, qubits = _integer(electrons, 'electrons'), _integer(qubits, 'qubits')
    require(2 <= qubits <= 62 and qubits % 2 == 0 and 0 < electrons < qubits
            and electrons % 2 == 0, 'Use a nonempty closed-shell occupied/virtual space')
    occupied, virtual = range(electrons), range(electrons, qubits)
    singles = sorted((i, a) for i in occupied for a in virtual if i % 2 == a % 2)
    doubles = sorted((i, j, a, b) for i, j in combinations(occupied, 2)
                     for a, b in combinations(virtual, 2)
                     if i % 2 + j % 2 == a % 2 + b % 2)
    result = []
    for kind, name, templates in (('single', 'SingleExcitation', singles),
                                   ('double', 'DoubleExcitation', doubles)):
        for wires in templates:
            result.append({'source_index': len(result), 'kind': kind, 'name': name,
                           'wires': list(wires)})
    return result


def screening_policy():
    return {
        'schema_version': 'co2-hf-coupling-screening-v1',
        'training_aliases': list(TRAINING_ALIASES),
        'main_template_count': MAIN_TEMPLATE_COUNT, 'vqe_template_count': VQE_TEMPLATE_COUNT,
        'coupling': 'signed <D_exc|H|HF> from full aligned h1/chemists ERI and explicit interleaved fermionic phase',
        'gap': 'sum diagonal closed-shell Fock energies on virtual wires minus occupied wires',
        'score': 'arithmetic mean across all three training geometries of abs(coupling)/positive_gap',
        'eligibility': 'maximum absolute coupling across training >= floor; every training gap > gap floor',
        'absolute_coupling_floor_ha': COUPLING_FLOOR_HA,
        'positive_gap_floor_ha': GAP_FLOOR_HA,
        'maximum_HF_single_coupling_ha': 1e-7,
        'mandatory_coverage': 'all 30 wires and all 15 spatial orbitals in both main and compact VQE template pools',
        'cover_selection': 'repeatedly maximize uncovered-wire count, then score, then smallest canonical source_index',
        'coverage_failure': 'stop before the pilot if any wire has no eligible template or covering set exceeds compact VQE count',
        'fill_selection': 'after shared covering set, fill main and VQE pools by descending score then source_index',
        'id_order': 'selected source_index ascending; within each template angles ascending; identity is ID 0',
        'vqe_initialization': 'fixed training mean of -2*fermionic_HF_derivative/gap, clipped to [-0.1,0.1]; HF zero is separate prescribed restart',
        'whole_qubit_HF_derivative': '-physical HF coupling, because HF is the high-bit-pattern member',
        'fermionic_HF_derivative': 'physical HF coupling times listed-order ladder phase of T; double T=a_b^dagger a_a^dagger a_j a_i',
        'logical_energy_requests': 0, 'correlated_reference_access': False,
        'held_out_access': False, 'candidate_energy_access': False,
        'coverage_claim_limit': 'wire participation is structural coverage, not demonstrated correlated dynamics or chemical accuracy',
    }


def _payload(templates, *, electrons, qubits, pool, vqe_sources=(), sources=None):
    templates = [dict(template, template_id=index) for index, template in enumerate(templates)]
    source_to_id = {template['source_index']: template['template_id'] for template in templates}
    covered = sorted({wire for template in templates for wire in template['wires']})
    return {
        'schema_version': 'large-qubit-excitation-catalogue-v1',
        'benchmark_id': BENCHMARK_ID, 'pool': pool,
        'electrons': electrons, 'qubits': qubits,
        'angles': list(ANGLES), 'identity_id': 0,
        'max_operator_id': len(templates) * len(ANGLES),
        'templates': templates,
        'vqe_template_indices': [source_to_id[source] for source in sorted(vqe_sources)],
        'counts': {'singles': sum(template['kind'] == 'single' for template in templates),
                   'doubles': sum(template['kind'] == 'double' for template in templates),
                   'templates': len(templates), 'operators': 1 + len(templates) * len(ANGLES)},
        'id_recipe': 'ID 0 is identity; ID>0: template=(ID-1)//8 and angle_index=(ID-1)%8',
        'execution_order': 'left_to_right',
        'spin_convention': 'alpha_even_beta_odd', 'bit_order': 'wire_0_most_significant',
        'whole_qubit_rotation': 'low=01 or0011, high=10 or1100; low_out=cos(phi/2)*low_in-sin(phi/2)*high_in; high_out=sin(phi/2)*low_in+cos(phi/2)*high_in',
        'covered_wires': covered, 'covered_spatial_orbitals': sorted({wire // 2 for wire in covered}),
        'training_hamiltonian_bindings': {} if sources is None else dict(sources),
        'screening_policy': screening_policy() if pool == 'screened_training_integrals' else None,
        'logical_energy_requests_for_screening': 0,
    }


def build_full_catalogue(electrons=22, qubits=30):
    payload = _payload(full_templates(electrons, qubits), electrons=electrons, qubits=qubits,
                       pool='full_occupied_virtual')
    return {**payload, 'sha256': digest(payload)}


def validate_catalogue(catalogue):
    require(isinstance(catalogue, dict), 'Catalogue must be a plain mapping')
    payload = {key: value for key, value in catalogue.items() if key != 'sha256'}
    require(catalogue.get('sha256') == digest(payload), 'Catalogue checksum mismatch')
    electrons, qubits = catalogue.get('electrons'), catalogue.get('qubits')
    full = full_templates(electrons, qubits)
    templates = catalogue.get('templates')
    require(isinstance(templates, list) and templates, 'Catalogue templates are missing')
    source_indices = []
    for index, actual in enumerate(templates):
        require(isinstance(actual, dict), 'Malformed catalogue template')
        source = _integer(actual.get('source_index'), 'source index')
        require(0 <= source < len(full), 'Template source index is outside the full family')
        require(canonical_bytes(actual) == canonical_bytes({**full[source], 'template_id': index}),
                'Template IDs, wires or operator semantics differ')
        source_indices.append(source)
    require(source_indices == sorted(set(source_indices)), 'Templates must use unique ascending canonical sources')
    pool = catalogue.get('pool')
    if pool == 'full_occupied_virtual':
        require(source_indices == list(range(len(full))), 'Full catalogue omitted a template')
        expected = build_full_catalogue(electrons, qubits)
    else:
        require(pool == 'screened_training_integrals' and (electrons, qubits) == (22, 30),
                'Unknown screened physical problem')
        require(len(templates) == MAIN_TEMPLATE_COUNT, 'Screened pool must contain exactly 64 templates')
        vqe = catalogue.get('vqe_template_indices')
        require(isinstance(vqe, list) and len(vqe) == VQE_TEMPLATE_COUNT
                and all(type(index) is int and 0 <= index < len(templates) for index in vqe)
                and vqe == sorted(set(vqe)), 'Compact VQE indices differ')
        vqe_wires = {wire for index in vqe for wire in templates[index]['wires']}
        require(vqe_wires == set(range(30)), 'Compact VQE pool leaves wires structurally fixed')
        sources = catalogue.get('training_hamiltonian_bindings')
        require(isinstance(sources, dict) and set(sources) == set(TRAINING_ALIASES)
                and all(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value)
                        for value in sources.values()), 'Training Hamiltonian bindings are missing')
        expected_payload = _payload([full[source] for source in source_indices],
                                    electrons=22, qubits=30, pool=pool,
                                    vqe_sources=[source_indices[index] for index in vqe], sources=sources)
        require(expected_payload['covered_wires'] == list(range(30)),
                'Main pool leaves wires structurally fixed')
        expected = {**expected_payload, 'sha256': digest(expected_payload)}
    require(canonical_bytes(catalogue) == canonical_bytes(expected),
            'Catalogue differs from its structural specification')
    return catalogue


def validate_operator_ids(ops, catalogue, *, length=None):
    validate_catalogue(catalogue)
    require(not isinstance(ops, (str, bytes, dict)), 'Operator IDs must be an integer sequence')
    try:
        result = tuple(_integer(value, 'operator ID') for value in ops)
    except TypeError as error:
        raise ValueError('Operator IDs must be an integer sequence') from error
    require(all(0 <= value <= catalogue['max_operator_id'] for value in result),
            'Operator ID is outside the fresh catalogue')
    if length is not None:
        require(type(length) is int and length > 0 and len(result) == length,
                'Circuit must contain the complete declared operator count')
    return result


def decode_operator(operator_id, catalogue):
    value = validate_operator_ids([operator_id], catalogue)[0]
    if value == 0:
        return {'id': 0, 'name': 'Identity', 'kind': 'identity', 'wires': [],
                'angle': 0.0, 'template_id': None}
    template_id, angle_index = divmod(value - 1, len(ANGLES))
    template = catalogue['templates'][template_id]
    return {'id': value, 'name': template['name'], 'kind': template['kind'],
            'wires': list(template['wires']), 'angle': ANGLES[angle_index],
            'template_id': template_id}


def encode_operator(template_id, angle_index, catalogue):
    validate_catalogue(catalogue)
    template_id, angle_index = _integer(template_id, 'template ID'), _integer(angle_index, 'angle index')
    require(0 <= template_id < len(catalogue['templates']) and 0 <= angle_index < len(ANGLES),
            'Template or angle index is outside the fresh catalogue')
    return 1 + template_id * len(ANGLES) + angle_index


def circuit_gates(ops, catalogue):
    values = validate_operator_ids(ops, catalogue)
    result = []
    for value in values:
        if value:
            template, angle = divmod(value - 1, len(ANGLES))
            result.append((list(catalogue['templates'][template]['wires']), ANGLES[angle]))
    return result


def random_circuit(rng, length, catalogue):
    require(type(length) is int and length > 0, 'Random circuit length must be positive')
    validate_catalogue(catalogue)
    return tuple(map(int, rng.integers(0, catalogue['counts']['operators'], size=length)))


def _phase(reference_occupied, ladder_order):
    occupied = set(reference_occupied)
    sign = 1
    for wire in ladder_order:
        sign *= -1 if sum(index < wire for index in occupied) % 2 else 1
        if wire in occupied:
            occupied.remove(wire)
        else:
            occupied.add(wire)
    return sign


def template_metrics(h1, eri, nalpha=11):
    """Exact signed HF couplings and a diagonal-Fock-gap importance proxy.

    This standalone routine permits small fixtures and does not assert SCF
    stationarity.  The screened CO2 builder checks that condition separately.
    """
    import numpy as np
    require(type(nalpha) is int and nalpha > 0, 'Closed-shell nalpha must be positive')
    require(not np.iscomplexobj(h1) and not np.iscomplexobj(eri), 'Screening integrals must be real')
    h1, eri = np.asarray(h1, dtype=np.float64), np.asarray(eri, dtype=np.float64)
    norb = len(h1)
    require(h1.shape == (norb, norb) and eri.shape == (norb,) * 4 and nalpha < norb,
            'Screening requires full spatial integrals and a nonempty virtual space')
    require(np.all(np.isfinite(h1)) and np.all(np.isfinite(eri)), 'Non-finite screening integrals')
    require(np.max(np.abs(h1 - h1.T)) <= 1e-10
            and np.max(np.abs(eri - eri.swapaxes(0, 1))) <= 1e-10
            and np.max(np.abs(eri - eri.swapaxes(2, 3))) <= 1e-10
            and np.max(np.abs(eri - eri.transpose(2, 3, 0, 1))) <= 1e-10,
            'Screening integrals lack real chemists-order symmetries')
    fock = h1 + 2 * np.einsum('pqii->pq', eri[:, :, :nalpha, :nalpha]) - np.einsum('piiq->pq', eri[:, :nalpha, :nalpha, :])
    diagonal, records = np.diag(fock), []
    for template in full_templates(2 * nalpha, 2 * norb):
        wires = template['wires']
        half = len(wires) // 2
        gap = float(sum(diagonal[wire // 2] for wire in wires[half:])
                    - sum(diagonal[wire // 2] for wire in wires[:half]))
        if half == 1:
            i, a = wires
            coupling = _phase(range(2 * nalpha), [i, a]) * fock[a // 2, i // 2]
        else:
            i, j, a, b = wires
            direct = (eri[a // 2, i // 2, b // 2, j // 2]
                      if (a % 2, b % 2) == (i % 2, j % 2) else 0.)
            exchange = (eri[a // 2, j // 2, b // 2, i // 2]
                        if (a % 2, b % 2) == (j % 2, i % 2) else 0.)
            coupling = _phase(range(2 * nalpha), [i, j, b, a]) * (direct - exchange)
        generator_phase = _phase(range(2 * nalpha), wires)
        records.append({**template, 'coupling': float(coupling), 'gap': gap,
                        'score': float(abs(coupling) / gap) if gap > GAP_FLOOR_HA else None,
                        'fermionic_phase': generator_phase,
                        'whole_qubit_hf_derivative': -float(coupling),
                        'fermionic_hf_derivative': float(generator_phase * coupling)})
    return records


def build_screened_catalogue(training_integrals, *, template_count=MAIN_TEMPLATE_COUNT,
                             vqe_template_count=VQE_TEMPLATE_COUNT):
    """Return ``(public_catalogue, private_screening_report)`` without energy calls."""
    import numpy as np
    require(template_count == MAIN_TEMPLATE_COUNT and type(template_count) is int
            and vqe_template_count == VQE_TEMPLATE_COUNT and type(vqe_template_count) is int,
            'Pool sizes are predeclared at 64 main and 24 compact VQE templates')
    require(isinstance(training_integrals, dict) and set(training_integrals) == set(TRAINING_ALIASES),
            'Screening requires all three training geometries and forbids held-out inputs')
    metrics, sources = {}, {}
    for alias in TRAINING_ALIASES:
        data = training_integrals[alias]
        require(isinstance(data, dict) and data.get('geometry_id') == alias
                and data.get('partition') == 'training', 'Screening geometry/partition mismatch')
        fingerprint = data.get('hamiltonian_fingerprint')
        require(isinstance(fingerprint, str) and re.fullmatch('[0-9a-f]{64}', fingerprint),
                'Screening requires an authenticated training Hamiltonian binding')
        require(np.shape(data.get('h1')) == (15, 15) and np.shape(data.get('eri')) == (15,) * 4,
                'CO2 screening requires all 15 spatial orbitals')
        metrics[alias] = template_metrics(data['h1'], data['eri'])
        sources[alias] = fingerprint
        require(max(abs(record['coupling']) for record in metrics[alias] if record['kind'] == 'single') <= 1e-7,
                'Input orbitals do not satisfy the declared closed-shell HF stationarity check')
        require(all(record['gap'] > GAP_FLOOR_HA for record in metrics[alias]),
                'A nonpositive/ill-conditioned diagonal-Fock excitation gap prevents screening')
    templates = full_templates()
    scores = [float(np.mean([metrics[alias][index]['score'] for alias in TRAINING_ALIASES]))
              for index in range(len(templates))]
    eligible = [index for index in range(len(templates))
                if max(abs(metrics[alias][index]['coupling']) for alias in TRAINING_ALIASES) >= COUPLING_FLOOR_HA]
    require(len(eligible) >= MAIN_TEMPLATE_COUNT, 'Too few physically resolvable templates for the main pool')
    ranking = sorted(eligible, key=lambda index: (-scores[index], index))
    covered, mandatory = set(), []
    while len(covered) < 30:
        candidates = [index for index in eligible if index not in mandatory]
        chosen = min(candidates, key=lambda index: (-len(set(templates[index]['wires']) - covered),
                                                    -scores[index], index))
        new = set(templates[chosen]['wires']) - covered
        require(new, 'An orbital wire has no physically resolvable training-only excitation')
        mandatory.append(chosen)
        covered.update(new)
        require(len(mandatory) <= VQE_TEMPLATE_COUNT, 'All-wire coverage exceeds the compact VQE template allowance')

    def selected(size):
        return sorted(mandatory + [index for index in ranking if index not in mandatory][:size - len(mandatory)])

    main, vqe = selected(MAIN_TEMPLATE_COUNT), selected(VQE_TEMPLATE_COUNT)
    payload = _payload([templates[index] for index in main], electrons=22, qubits=30,
                       pool='screened_training_integrals', vqe_sources=vqe, sources=sources)
    catalogue = {**payload, 'sha256': digest(payload)}
    validate_catalogue(catalogue)
    initial_angles = [float(np.clip(np.mean([
        -2 * metrics[alias][index]['fermionic_hf_derivative'] / metrics[alias][index]['gap']
        for alias in TRAINING_ALIASES]), -.1, .1)) for index in vqe]
    report = {
        'schema_version': 'co2-private-hf-coupling-screening-v1', 'benchmark_id': BENCHMARK_ID,
        'policy': screening_policy(), 'catalogue_sha256': catalogue['sha256'],
        'training_hamiltonian_bindings': sources, 'metrics_by_geometry': metrics,
        'aggregate_scores': scores, 'eligible_source_indices': eligible,
        'coverage_selection_order': mandatory, 'ranking_source_indices': ranking,
        'main_source_indices': main, 'vqe_source_indices': vqe,
        'vqe_initial_angles': initial_angles, 'logical_energy_requests': 0,
        'screening_scalar_labels_are_model_inputs': False,
        'no_held_out_or_reference_or_candidate_energy_access': True,
    }
    return catalogue, report


def factored_recipe(catalogue):
    validate_catalogue(catalogue)
    return {'id_formula': catalogue['id_recipe'], 'angles': list(ANGLES),
            'templates': [{'template_id': row['template_id'], 'wires': list(row['wires'])}
                          for row in catalogue['templates']],
            'identity_id': 0, 'max_operator_id': catalogue['max_operator_id'],
            'execution_order': 'left_to_right', 'whole_qubit_rotation': catalogue['whole_qubit_rotation']}
