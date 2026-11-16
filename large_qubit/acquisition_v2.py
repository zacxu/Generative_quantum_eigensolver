"""Fresh v2 acquisition with the frozen v1 math and a 40/20/40 mixture.

No evaluator or model is imported.  Every returned fitting proposal must be
charged by the caller, including duplicates and reserved-diagnostic collisions.
Every eligible novel scored row is retained, including unfavorable local moves.
"""

from copy import deepcopy
import math
import random

from .catalogue import (ANGLES, TRAINING_ALIASES, validate_catalogue,
                        validate_operator_ids)
from .common import digest, require
from .definitions import require_geometry_access
from .protocol import BENCHMARK_ID, seed_for


ACQUISITION_ID = 'co2-corpus-acquisition-v2'
FITTING_GOAL = 2048
FALLBACK_ROWS = 1024
DIAGNOSTIC_ROWS = 128
REQUEST_QUOTAS = {'CO2_T01': 2731, 'CO2_T02': 2731, 'CO2_T03': 2730}
COMPONENT_CYCLE = ('guided', 'uniform', 'local', 'guided', 'local',
                   'guided', 'uniform', 'local', 'guided', 'local')
ROW_FIELDS = {'row_id', 'geometry_id', 'role', 'ops', 'energy_hartree',
              'catalogue_sha256', 'request_ordinal'}


def acquisition_policy(catalogue, *, length=32):
    validate_catalogue(catalogue)
    require(catalogue['pool'] == 'screened_training_integrals'
            and catalogue['counts']['operators'] == 513, 'Use the frozen CO2 screened catalogue')
    require(type(length) is int and length in (32, 64), 'Use a complete frozen length32 or64')
    return {
        'schema_version': 'co2-corpus-acquisition-policy-v2', 'acquisition_id': ACQUISITION_ID,
        'benchmark_id': BENCHMARK_ID, 'pipeline_seed': 22,
        'training_aliases': list(TRAINING_ALIASES), 'catalogue_sha256': catalogue['sha256'],
        'row_id_convention': '<geometry_id>_v2_F<6-digit fitting index> or _v2_D<6-digit diagnostic index>',
        'length': length, 'fitting_goal_per_geometry': FITTING_GOAL,
        'fallback_rows_per_geometry': FALLBACK_ROWS, 'diagnostics_per_geometry': DIAGNOSTIC_ROWS,
        'maximum_logical_requests': 8192, 'geometry_request_quotas': dict(REQUEST_QUOTAS),
        'quota_includes': ['HF', 'diagnostics', 'all fitting proposals', 'duplicates', 'local work'],
        'request_order': 'three HF requests, all384 pre-reserved diagnostics, then balanced geometry-round-robin fitting',
        'diagnostic_reservation': 'metadata-only independent seeded128unique circuits per geometry, before the first HF request',
        'diagnostic_components': {'uniform': 64, 'guided': 64},
        'reservation_collision_policy': 'fail before any physical request; never silently replace a reservation',
        'diagnostic_labels_excluded_from': ['fitting', 'weighting', 'quality selection', 'parent selection', 'retrieval'],
        'fitting_component_cycle': list(COMPONENT_CYCLE),
        'fitting_mixture': {'guided': .4, 'uniform': .2, 'local': .4},
        'uniform': 'independent exact uniform IDs0..512 including identity at each position',
        'guided_template_selection': 'mandatory all-wire cover; half remaining slots highest analytic score, rest square-root-score weighted sampling without replacement',
        'guided_angle_recipe': 'global uniform[0.5,1.5] times per-gate choice[0.75,1,1.25] times -2*whole_qubit_HF_derivative/gap; nearest existing angle, lowest index ties; shuffle order',
        'local_parent': 'uniformly choose among best32 retained fitting rows by energy, circuit tuple, row_id',
        'local_mutations': 'uniform1..3 edits; each edit angle-neighbor50%, replacement-ID25%, swap-positions25%',
        'local_without_parent': 'guided proposal with explicit fallback metadata; still charged',
        'novelty': 'exact complete ID tuple within one geometry; no energy-ranked row filtering',
        'all_eligible_novel_rows_retained': True, 'poor_or_nonimproving_local_rows_retained': True,
        'diagnostic_collision_proposal_is_queried_and_charged_but_not_fitting_eligible': True,
        'duplicate_proposal_is_queried_and_charged_but_not_retained_twice': True,
        'full_release': 'first2048 eligible rows per geometry in acquisition order; stop on completeness, leave unused quota unused',
        'fallback': 'first1024 per geometry only if actual quota or predeclared schedule prevents full completeness, all3 have1024 and quality passes; never quality-failure rescue',
        'seed_derivation': 'protocol.seed_for(acquisition,group=acquisition_id:alias:role:ordinal); independent deterministic RNG per request',
        'scoring_source': 'fresh corpus requests only; no historical or pilot labels copied',
        'analytic_source': 'authenticated training-only Section5 HF coupling/gap screening; no correlated reference or held-out input',
        'quality_policy': quality_policy(),
        'model_training_executed': False, 'heldout_or_reference_access': False,
    }


def quality_policy():
    return {
        'scope': 'balanced released acquisition-order fitting prefix only',
        'minimum_useful_HF_improvement_ha': .005,
        'minimum_useful_fraction': .10, 'minimum_useful_rows_absolute': 64,
        'minimum_best_HF_improvement_ha': .02,
        'required_template_coverage': 64, 'required_angle_coverage': 8,
        'minimum_nonidentity_ID_coverage_fraction': .95, 'required_wire_coverage': 30,
        'hamming_pairs': 256, 'minimum_mean_normalized_hamming_distance': .25,
        'fixed_weight_lambdas': [.25, .10],
        'minimum_weight_ESS_fraction': .05, 'maximum_single_row_weight': .05,
        'weight_concentration_origin': 'old study5percent engineering safeguard; not a validated LLM sample requirement',
        'weight_statistics': 'corpus.geometry_weights on fitting rows only, exact fixed lambdas, no automatic lambda choice',
        'diagnostic_energies_used': False, 'references_used': False, 'heldout_used': False,
        'correlation_limit': 'string diversity and ESS do not count independent molecular observations or certify model generalization',
        'failure': 'retain negative evidence and stop before fitting; no energy-ranked subset, seed replacement or silent threshold amendment',
    }


def _ordinal(value, label):
    require(type(value) is int and value >= 1, f'{label} must be a positive integer')
    return value


def _finite(value, label):
    require(type(value) in (int, float), f'{label} must be a real scalar')
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    require(finite, f'{label} must be finite')
    return float(value)


def _ops(value, maximum, length):
    require(type(value) in (list, tuple) and len(value) == length,
            'Use the complete declared circuit ID sequence')
    require(all(type(operator) is int and 0 <= operator <= maximum for operator in value),
            'Circuit contains a noninteger or out-of-catalogue ID')
    return tuple(value)


def _v2_row_id(value, alias, role):
    prefix = f'{alias}_v2_{"F" if role == "fitting" else "D"}'
    require(type(value) is str and value.startswith(prefix), 'Rows must carry fresh v2 campaign provenance')
    suffix = value[len(prefix):]
    require(len(suffix) == 6 and all(char in '0123456789' for char in suffix) and int(suffix) >= 1,
            'Use the complete v2 row index')


def _fitting_rows(rows, alias, catalogue, length, reserved):
    require(type(rows) in (list, tuple), 'Fitting history must be an ordered row sequence')
    previous, identities, circuits, checked = 0, set(), set(), []
    for row in rows:
        require(type(row) is dict and set(row) == ROW_FIELDS, 'Fitting parent row fields differ')
        require(row['geometry_id'] == alias and row['role'] == 'fitting'
                and row['catalogue_sha256'] == catalogue['sha256'],
                'Parents accept same-geometry fitting rows only; never filter diagnostics or held-out data')
        require(type(row['row_id']) is str and row['row_id'] and row['row_id'] not in identities,
                'Fitting parent IDs must be unique nonempty text')
        _v2_row_id(row['row_id'], alias, 'fitting')
        ordinal = _ordinal(row['request_ordinal'], 'row request ordinal')
        require(ordinal > previous, 'Fitting parent history must retain acquisition order')
        circuit = _ops(row['ops'], catalogue['max_operator_id'], length)
        require(circuit not in circuits and circuit not in reserved,
                'Fitting history contains a duplicate or reserved diagnostic circuit')
        energy = _finite(row['energy_hartree'], 'fitting parent energy')
        checked.append((energy, circuit, row['row_id']))
        previous = ordinal
        identities.add(row['row_id']); circuits.add(circuit)
    return checked


class AcquisitionSampler:
    """Stateless request seeds plus fitting-only parent adaptation.

    Reservation generation and analytic proposals call no energy oracle.  The
    caller's ledger, not this sampler, determines whether a query is novel.
    """

    def __init__(self, alias, catalogue, screening, *, length=32):
        require_geometry_access(alias, 'acquisition')
        self.policy = acquisition_policy(catalogue, length=length)
        require(type(screening) is dict and screening.get('catalogue_sha256') == catalogue['sha256']
                and screening.get('training_hamiltonian_bindings') == catalogue['training_hamiltonian_bindings']
                and screening.get('no_held_out_or_reference_or_candidate_energy_access') is True
                and type(screening.get('metrics_by_geometry')) is dict
                and set(screening['metrics_by_geometry']) == set(TRAINING_ALIASES),
                'Analytic screening source is not the authenticated training-only preparation')
        self.alias, self.catalogue, self.length = alias, deepcopy(catalogue), length
        self.screening = deepcopy(screening)
        self.maximum = catalogue['max_operator_id']
        self._source_to_template = {row['source_index']: row['template_id'] for row in catalogue['templates']}
        self._metrics = self.screening['metrics_by_geometry'][alias]
        self._main = set(self.screening['main_source_indices'])
        self._cover = list(self.screening['coverage_selection_order'])
        self._ranking = [source for source in self.screening['ranking_source_indices'] if source in self._main]
        require(len(self._main) == 64 and len(self._cover) <= length
                and set(self._cover) <= self._main and set(self._ranking) == self._main,
                'Screening cover/ranking does not describe the complete shared64-template pool')
        require({wire for source in self._cover for wire in self._metrics[source]['wires']} == set(range(30)),
                'Mandatory guided templates must cover all30 wires')
        self._reservations = self._reserve()
        self._reserved = frozenset(tuple(row['ops']) for row in self._reservations)

    def _seed(self, role, ordinal):
        return seed_for('acquisition', group=f'{ACQUISITION_ID}:{self.alias}:{role}:{ordinal}')

    def _uniform(self, rng):
        return [rng.randrange(self.maximum + 1) for _ in range(self.length)], {'distribution': 'independent_uniform_IDs_including0'}

    def guided_sources(self, rng):
        """Shared all-wire cover, useful head, then analytic-score diversity."""
        remaining = [source for source in self._ranking if source not in self._cover]
        slots = self.length - len(self._cover)
        head_count = slots // 2
        selected = list(self._cover) + remaining[:head_count]
        available = remaining[head_count:]
        for _ in range(slots - head_count):
            weights = [math.sqrt(max(float(self.screening['aggregate_scores'][source]), 1e-30))
                       for source in available]
            threshold = rng.random() * sum(weights)
            cumulative, chosen = 0., len(available) - 1
            for index, weight in enumerate(weights):
                cumulative += weight
                if threshold < cumulative:
                    chosen = index
                    break
            selected.append(available.pop(chosen))
        require(len(selected) == self.length and len(set(selected)) == self.length,
                'Guided proposal must use distinct templates before shuffling')
        return selected

    def _guided(self, rng):
        sources = self.guided_sources(rng)
        scale = rng.uniform(.5, 1.5)
        factors, ops = [], []
        for source in sources:
            factor = rng.choice((.75, 1., 1.25))
            metric = self._metrics[source]
            target = -2 * scale * factor * metric['whole_qubit_hf_derivative'] / metric['gap']
            angle_index = min(range(8), key=lambda index: (abs(ANGLES[index] - target), index))
            ops.append(1 + 8 * self._source_to_template[source] + angle_index)
            factors.append(factor)
        order = list(range(self.length)); rng.shuffle(order)
        return [ops[index] for index in order], {'global_scale': scale,
            'source_indices_before_shuffle': sources, 'per_gate_factors_before_shuffle': factors,
            'shuffle_order': order}

    def _reserve(self):
        rows, seen = [], set()
        for ordinal in range(1, DIAGNOSTIC_ROWS + 1):
            seed = self._seed('diagnostic', ordinal)
            rng = random.Random(seed)
            component = 'uniform' if ordinal % 2 else 'guided'
            ops, recipe = self._uniform(rng) if component == 'uniform' else self._guided(rng)
            require(tuple(ops) not in seen, 'Diagnostic reservation collision: stop before any physical query')
            seen.add(tuple(ops))
            rows.append({'diagnostic_ordinal': ordinal, 'ops': ops, 'component': component,
                         'seed': seed, 'recipe': recipe})
        return rows

    def reserve_diagnostics(self):
        return deepcopy(self._reservations)

    @property
    def reserved_circuits(self):
        return tuple(tuple(row['ops']) for row in self._reservations)

    def propose(self, fitting_request_ordinal, fitting_rows):
        ordinal = _ordinal(fitting_request_ordinal, 'fitting request ordinal')
        require(ordinal <= REQUEST_QUOTAS[self.alias] - 1 - DIAGNOSTIC_ROWS,
                'Geometry fitting request stream exceeds its fixed HF-inclusive quota')
        parents = _fitting_rows(fitting_rows, self.alias, self.catalogue, self.length, self._reserved)
        require(len(parents) <= ordinal - 1, 'Fitting parent history cannot include future request outcomes')
        requested = COMPONENT_CYCLE[(ordinal - 1) % len(COMPONENT_CYCLE)]
        seed = self._seed('fitting', ordinal)
        rng, component, parent_id = random.Random(seed), requested, None
        if requested == 'uniform':
            ops, recipe = self._uniform(rng)
        elif requested == 'guided' or not parents:
            ops, recipe = self._guided(rng)
            if requested == 'local':
                component = 'guided_no_fitting_parent'
                recipe['local_fallback_reason'] = 'no eligible fitting parent exists'
        else:
            energy, parent, parent_id = rng.choice(sorted(parents)[:32])
            ops, edits = list(parent), []
            for _ in range(rng.randint(1, 3)):
                position, draw = rng.randrange(self.length), rng.randrange(4)
                if draw < 2:
                    old = ops[position]
                    if old == 0:
                        ops[position] = rng.randrange(1, self.maximum + 1)
                    else:
                        template, angle = divmod(old - 1, 8)
                        neighbors = [index for index in (angle - 1, angle + 1) if 0 <= index < 8]
                        ops[position] = 1 + template * 8 + rng.choice(neighbors)
                    edits.append({'kind': 'angle_neighbor', 'position': position, 'before': old, 'after': ops[position]})
                elif draw == 2:
                    old = ops[position]; ops[position] = rng.randrange(self.maximum + 1)
                    edits.append({'kind': 'replacement_ID', 'position': position, 'before': old, 'after': ops[position]})
                else:
                    other = rng.randrange(self.length)
                    ops[position], ops[other] = ops[other], ops[position]
                    edits.append({'kind': 'swap_positions', 'positions': [position, other]})
            recipe = {'parent_row_id': parent_id, 'parent_energy_hartree': energy,
                      'parent_ops': list(parent), 'mutations': edits,
                      'retain_if_novel_even_when_nonimproving': True}
        _ops(ops, self.maximum, self.length)
        return {'fitting_request_ordinal': ordinal, 'ops': ops, 'component': component,
                'requested_component': requested, 'seed': seed,
                'parent_row_id': parent_id, 'recipe': recipe}


def quality_report(fitting_by_alias, diagnostics_by_alias, hf_by_alias, catalogue, *, length=32):
    """Assess the balanced release prefix, never diagnostic label values.

    Diagnostics enter only role, count, uniqueness and disjointness checks.
    Exact weighting mathematics is shared with the authenticated corpus loader.
    """
    from .corpus_v2 import geometry_weights
    validate_catalogue(catalogue)
    policy = acquisition_policy(catalogue, length=length)
    for mapping in (fitting_by_alias, diagnostics_by_alias, hf_by_alias):
        require(type(mapping) is dict and set(mapping) == set(TRAINING_ALIASES),
                'Quality assessment requires exactly the three training geometries')
    sizes = {len(fitting_by_alias[alias]) for alias in TRAINING_ALIASES}
    require(len(sizes) == 1 and next(iter(sizes)) in (FALLBACK_ROWS, FITTING_GOAL),
            'Assess one balanced1024 or2048 acquisition-order fitting prefix')
    count, groups = next(iter(sizes)), {}
    for alias in TRAINING_ALIASES:
        require_geometry_access(alias, 'acquisition')
        hf = _finite(hf_by_alias[alias], 'fresh corpus HF energy')
        diagnostics = diagnostics_by_alias[alias]
        require(type(diagnostics) in (list, tuple) and len(diagnostics) == DIAGNOSTIC_ROWS,
                'Exactly128 reserved diagnostics are required per geometry')
        reserved, diagnostic_ids = set(), set()
        for row in diagnostics:
            require(type(row) is dict and set(row) == ROW_FIELDS and row['geometry_id'] == alias
                    and row['role'] == 'diagnostic' and row['catalogue_sha256'] == catalogue['sha256'],
                    'Diagnostic metadata identity differs')
            require(type(row['row_id']) is str and row['row_id'] and row['row_id'] not in diagnostic_ids,
                    'Diagnostic IDs must be unique')
            _v2_row_id(row['row_id'], alias, 'diagnostic')
            _ordinal(row['request_ordinal'], 'diagnostic request ordinal')
            circuit = _ops(row['ops'], catalogue['max_operator_id'], length)
            require(circuit not in reserved, 'Diagnostic reservations must be unique')
            reserved.add(circuit); diagnostic_ids.add(row['row_id'])
            # Deliberately do not inspect row['energy_hartree'] here.
        checked = _fitting_rows(fitting_by_alias[alias], alias, catalogue, length, reserved)
        require(not ({row['row_id'] for row in fitting_by_alias[alias]} & diagnostic_ids),
                'Fitting and diagnostic row IDs overlap')
        circuits = [row[1] for row in checked]
        improvements = [hf - row[0] for row in checked]
        identities = {operator for circuit in circuits for operator in circuit if operator != 0}
        templates = {(operator - 1) // 8 for operator in identities}
        angles = {(operator - 1) % 8 for operator in identities}
        wires = {wire for index in templates for wire in catalogue['templates'][index]['wires']}
        rng = random.Random(seed_for('acquisition', group=f'{ACQUISITION_ID}:{alias}:quality_hamming'))
        hamming = 0
        for _ in range(256):
            left, right = rng.sample(range(count), 2)
            hamming += sum(a != b for a, b in zip(circuits[left], circuits[right]))
        distance = hamming / (256 * length)
        useful = sum(improvement >= .005 for improvement in improvements)
        minimum_useful = max(64, math.ceil(.10 * count))
        weight_checks = {}
        for scale in (.25, .10):
            weights = geometry_weights(fitting_by_alias[alias], mode='weighted', lambda_scale=scale)
            report = weights.report
            weight_checks[str(scale)] = {'ess_fraction': report['ess_fraction'],
                'maximum_weight': report['maximum_weight'],
                'passed': report['ess_fraction'] >= .05 and report['maximum_weight'] <= .05}
        checks = {'useful_tail': useful >= minimum_useful, 'best_HF_improvement': max(improvements) >= .02,
            'template_coverage': len(templates) == 64, 'angle_coverage': len(angles) == 8,
            'nonidentity_ID_coverage': len(identities) / 512 >= .95, 'wire_coverage': len(wires) == 30,
            'circuit_diversity': distance >= .25,
            'fixed_weight_concentration': all(row['passed'] for row in weight_checks.values())}
        groups[alias] = {'passed': all(checks.values()), 'checks': checks, 'fitting_rows': count,
            'diagnostic_rows_reserved': DIAGNOSTIC_ROWS, 'fitting_row_order_sha256': digest([
                [row['request_ordinal'], row['row_id'], row['ops'], row['energy_hartree']]
                for row in fitting_by_alias[alias]]),
            'useful_rows': useful, 'minimum_useful_rows': minimum_useful,
            'best_HF_improvement_ha': max(improvements),
            'templates_seen': len(templates), 'angles_seen': len(angles),
            'nonidentity_IDs_seen': len(identities), 'wire_coverage': sorted(wires),
            'mean_normalized_hamming_distance': distance, 'weight_concentration': weight_checks,
            'diagnostic_energies_used_for_quality': False}
    return {'schema_version': 'co2-fitting-prefix-quality-v2', 'acquisition_id': ACQUISITION_ID,
        'benchmark_id': BENCHMARK_ID, 'all_passed': all(group['passed'] for group in groups.values()),
        'rows_per_geometry': count, 'catalogue_sha256': catalogue['sha256'],
        'policy_sha256': digest(policy), 'quality_policy': quality_policy(),
        'fresh_HF_baseline_binding_sha256': digest(hf_by_alias), 'groups': groups,
        'diagnostic_energies_used_for_quality': False, 'heldout_or_reference_access': False,
        'model_training_executed': False}


assess_release = quality_report
