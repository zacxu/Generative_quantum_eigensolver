"""Real spatial-orbital gauges and integral transforms for the CO2 definition.

No energy labels select an orbital gauge.  The training anchor fixes its gauge
from ordered AOs; other geometries use only AO overlaps and canonical SCF
orbital degeneracies.  Occupied and virtual spaces never mix.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from .common import require


DEGENERACY_TOL_HA = 1e-7
ORTHONORMAL_TOL = 1e-8
ASSIGNMENT_TIE_TOL = 1e-12
MIN_ALIGNMENT_SINGULAR_VALUE = 1e-6


def orbital_conventions():
    return {
        'anchor_geometry_id': 'CO2_T02',
        'anchor_distance_angstrom': 1.20,
        'real_spatial_orbitals': True,
        'ao_basis': 'STO-3G contracted real spherical Gaussian; PySCF cart=False',
        'ao_order': 'provider atom/shell/component order; exact ao_labels serialized',
        'atom_order': ['O(z=-R)', 'C(z=0)', 'O(z=+R)'],
        'mo_order': 'occupied first (11), virtual after (4); anchor energy ascending within each sector',
        'anchor_gauge': 'ordered AO projections into each energy-degenerate block, two-pass metric Gram-Schmidt; largest AO coefficient positive, lowest AO index on ties',
        'alignment': 'sector-wise maximum absolute cross-AO MO overlap assignment; lexicographic optimal ties; orthogonal Procrustes inside target energy-degenerate blocks; positive matched overlap for singletons',
        'degeneracy_tolerance_ha': DEGENERACY_TOL_HA,
        'degeneracy_grouping': 'ascending energy, each block total span <= tolerance; no transitive chaining',
        'assignment_tie_tolerance': ASSIGNMENT_TIE_TOL,
        'minimum_alignment_singular_value': MIN_ALIGNMENT_SINGULAR_VALUE,
        'orthonormality_tolerance': ORTHONORMAL_TOL,
        'spin_orbital_order': 'wire 2*p alpha, wire 2*p+1 beta',
        'jordan_wigner_bit_order': 'wire 0 is most significant displayed/basis-index bit',
        'hf_occupation_bits': '1' * 22 + '0' * 8,
        'hf_spatial_occupations': [2] * 11 + [0] * 4,
        'integral_convention': 'h[p,q]; eri[p,q,r,s]=(pq|rs) chemists ordering; real float64; all four AO indices transformed with final MO coefficients',
        'selection_information': 'coordinates, AO overlaps and SCF orbital degeneracies only; no correlated reference or candidate-circuit energies',
    }


def _real_array(value, name, ndim=None):
    value = np.asarray(value)
    require(not np.iscomplexobj(value), f'{name} must be real')
    value = np.asarray(value, dtype=np.float64)
    require(np.all(np.isfinite(value)), f'{name} must be finite')
    if ndim is not None:
        require(value.ndim == ndim, f'{name} must have {ndim} dimensions')
    return value


def _check_orbitals(coeff, energies, overlap, occupations):
    c = _real_array(coeff, 'MO coefficients', 2)
    e = _real_array(energies, 'MO energies', 1)
    s = _real_array(overlap, 'AO overlap', 2)
    occ = _real_array(occupations, 'MO occupations', 1)
    require(c.shape[1] == len(e) == len(occ), 'MO dimensions differ')
    require(s.shape == (c.shape[0], c.shape[0]), 'AO overlap dimensions differ')
    require(np.all((occ == 0) | (occ == 2)), 'Only real closed-shell RHF occupations are supported')
    require(np.allclose(s, s.T, atol=1e-12, rtol=0), 'AO overlap is not symmetric')
    require(np.linalg.eigvalsh(s)[0] > 1e-10, 'AO overlap is singular or indefinite')
    residual = np.max(np.abs(c.T @ s @ c - np.eye(c.shape[1])))
    require(residual <= ORTHONORMAL_TOL, 'MO coefficients are not S-orthonormal')
    return c, e, s, occ


def _energy_groups(indices, energies):
    ordered = sorted(indices, key=lambda i: (float(energies[i]), int(i)))
    groups = []
    for index in ordered:
        if not groups or energies[index] - energies[groups[-1][0]] > DEGENERACY_TOL_HA:
            groups.append([index])
        else:
            groups[-1].append(index)
    return groups


def _anchor_block(block, overlap):
    """Gauge-invariant ordered projections, including arbitrary degenerate rotations."""
    vectors, pivots = [], []
    projections = block.T @ overlap
    for ao in range(block.shape[0]):
        v = projections[:, ao].copy()
        # Reorthogonalization matters when an early AO is almost dependent.
        for _ in range(2):
            for q in vectors:
                v -= q * np.dot(q, v)
        norm = np.linalg.norm(v)
        if norm <= 1e-10:
            continue
        vectors.append(v / norm)
        pivots.append(ao)
        if len(vectors) == block.shape[1]:
            break
    require(len(vectors) == block.shape[1], 'Cannot resolve the AO-projected degenerate subspace')
    q = np.column_stack(vectors)
    result = block @ q
    for j in range(result.shape[1]):
        largest = np.max(np.abs(result[:, j]))
        # Numerical near-ties use the lower AO index under a fixed tolerance.
        pivot = int(np.flatnonzero(np.abs(result[:, j]) >= largest - 1e-12)[0])
        if result[pivot, j] < 0:
            result[:, j] *= -1
            q[:, j] *= -1
    return result, q, pivots


def canonicalize_anchor(coeff, mo_energy, overlap, occupations):
    """Return ``(C, U, metadata)`` with C=input_C@U and deterministic anchor gauge."""
    c, e, s, occ = _check_orbitals(coeff, mo_energy, overlap, occupations)
    n = c.shape[1]
    result, rotation = np.empty_like(c), np.zeros((n, n))
    blocks, order, offset = [], [], 0
    for population in (2, 0):
        for group in _energy_groups(np.flatnonzero(occ == population), e):
            aligned, q, pivots = _anchor_block(c[:, group], s)
            destination = list(range(offset, offset + len(group)))
            result[:, destination] = aligned
            rotation[np.ix_(group, destination)] = q
            blocks.append({'input_indices': list(map(int, group)),
                           'output_indices': destination, 'ao_projection_pivots': pivots,
                           'occupation': population})
            order.extend(group)
            offset += len(group)
    require(np.allclose(rotation.T @ rotation, np.eye(n), atol=1e-10, rtol=0),
            'Anchor rotation is not orthogonal')
    return result, rotation, {
        'method': 'ordered_AO_projected_anchor', 'anchor_geometry_id': 'CO2_T02',
        'blocks': blocks, 'input_energy_order': list(map(int, order)),
        'aligned_fock_diagonal_ha': np.diag(rotation.T @ np.diag(e) @ rotation).tolist(),
        'occupation_order': [2] * int(np.sum(occ == 2)) + [0] * int(np.sum(occ == 0)),
        'orthonormality_max_abs': float(np.max(np.abs(result.T @ s @ result - np.eye(n)))),
    }


def _lexicographic_assignment(scores):
    """Lexicographically first maximum-weight matching within a fixed tolerance."""
    scores = _real_array(scores, 'alignment scores', 2)
    require(scores.shape[0] == scores.shape[1], 'Alignment assignment must be square')
    remaining = list(range(len(scores)))
    chosen, spent = [], 0.0
    rows, cols = linear_sum_assignment(-scores)
    optimum = float(scores[rows, cols].sum())
    for row in range(len(scores)):
        for col in remaining:
            rest = [j for j in remaining if j != col]
            residual = scores[row + 1:, rest]
            r, k = linear_sum_assignment(-residual)
            total = spent + scores[row, col] + residual[r, k].sum()
            if total >= optimum - ASSIGNMENT_TIE_TOL:
                chosen.append(col)
                spent += scores[row, col]
                remaining.remove(col)
                break
        else:
            raise ValueError('Could not deterministically resolve orbital assignment')
    return chosen


def align_orbitals(anchor_coeff, coeff, cross_overlap, overlap, mo_energy,
                   occupations, anchor_occupations=None):
    """Overlap-only target gauge; return ``(C, U, metadata)`` with C=input_C@U.

    cross_overlap is <AO_anchor|AO_target>, including displacement of basis
    functions between geometries.  Reusing the target overlap here is incorrect.
    """
    c, e, s, occ = _check_orbitals(coeff, mo_energy, overlap, occupations)
    anchor = _real_array(anchor_coeff, 'anchor coefficients', 2)
    cross = _real_array(cross_overlap, 'cross-AO overlap', 2)
    require(anchor.shape[1] == c.shape[1], 'Anchor and target MO counts differ')
    require(cross.shape == (anchor.shape[0], c.shape[0]), 'Cross-overlap dimensions differ')
    n = c.shape[1]
    nocc = int(np.sum(occ == 2))
    aocc = _real_array(anchor_occupations if anchor_occupations is not None
                       else [2] * nocc + [0] * (n - nocc), 'anchor occupations', 1)
    require(aocc.shape == occ.shape and np.all((aocc == 0) | (aocc == 2)),
            'Invalid anchor occupations')
    require(np.sum(aocc == 2) == nocc, 'Anchor and target populations differ')
    require(np.array_equal(aocc, [2] * nocc + [0] * (n - nocc)),
            'Anchor must be occupied-first')
    overlaps = anchor.T @ cross @ c
    assignment = np.zeros(n, dtype=int)
    for population in (2, 0):
        dest = np.flatnonzero(aocc == population)
        source = np.flatnonzero(occ == population)
        order = _lexicographic_assignment(np.abs(overlaps[np.ix_(dest, source)]))
        assignment[dest] = source[order]
    rotation = np.eye(n)[:, assignment]
    result = c @ rotation
    blocks = []
    for population in (2, 0):
        for source in _energy_groups(np.flatnonzero(occ == population), e):
            dest = sorted(int(np.flatnonzero(assignment == j)[0]) for j in source)
            matrix = anchor[:, dest].T @ cross @ result[:, dest]
            u, singular, vt = np.linalg.svd(matrix)
            require(float(singular[-1]) >= MIN_ALIGNMENT_SINGULAR_VALUE,
                    'Ambiguous/disconnected orbital alignment; amend before production')
            q = vt.T @ u.T
            result[:, dest] = result[:, dest] @ q
            rotation[:, dest] = rotation[:, dest] @ q
            blocks.append({'input_indices': list(map(int, source)), 'output_indices': dest,
                           'occupation': population, 'overlap_singular_values': singular.tolist(),
                           'procrustes_rotation': q.tolist()})
    residual = float(np.max(np.abs(result.T @ s @ result - np.eye(n))))
    require(residual <= ORTHONORMAL_TOL, 'Aligned MO coefficients are not S-orthonormal')
    require(np.max(np.abs(rotation.T @ rotation - np.eye(n))) < 1e-10,
            'Alignment rotation is not orthogonal')
    require(np.all(np.abs(rotation[np.ix_(occ == 2, aocc == 0)]) < 1e-12)
            and np.all(np.abs(rotation[np.ix_(occ == 0, aocc == 2)]) < 1e-12),
            'Orbital alignment mixed occupied and virtual spaces')
    return result, rotation, {
        'method': 'cross_AO_overlap_sector_assignment_and_degenerate_Procrustes',
        'anchor_geometry_id': 'CO2_T02', 'input_assignment': assignment.tolist(),
        'blocks': blocks, 'occupation_order': aocc.astype(int).tolist(),
        'aligned_fock_diagonal_ha': np.diag(rotation.T @ np.diag(e) @ rotation).tolist(),
        'matched_overlap_diagonal': np.diag(anchor.T @ cross @ result).tolist(),
        'orthonormality_max_abs': residual,
        'occupied_virtual_mixing': False,
    }


def transform_integrals(h_ao, eri_ao, coeff):
    """Transform h and the complete chemists-order ERI with the same real C."""
    c = _real_array(coeff, 'MO coefficients', 2)
    h = _real_array(h_ao, 'AO one-electron integrals', 2)
    eri = _real_array(eri_ao, 'AO electron-repulsion integrals', 4)
    require(h.shape == (c.shape[0],) * 2 and eri.shape == (c.shape[0],) * 4,
            'AO integral dimensions differ from MO coefficients')
    return c.T @ h @ c, np.einsum('ap,bq,cr,ds,abcd->pqrs', c, c, c, c, eri, optimize=True)


def closed_shell_hf_energy(h, eri, nocc, nuclear_repulsion=0.0):
    """Independent RHF determinant contraction, including nuclear repulsion."""
    h = _real_array(h, 'MO one-electron integrals', 2)
    eri = _real_array(eri, 'MO electron-repulsion integrals', 4)
    require(h.shape[0] == h.shape[1] and eri.shape == h.shape * 2, 'MO integral dimensions differ')
    require(type(nocc) is int and 0 <= nocc <= len(h), 'Invalid occupied orbital count')
    require(np.isfinite(nuclear_repulsion), 'Nuclear repulsion must be finite')
    occupied = eri[:nocc, :nocc, :nocc, :nocc]
    return float(2 * np.trace(h[:nocc, :nocc]) + 2 * np.einsum('iijj->', occupied)
                 - np.einsum('ijji->', occupied) + nuclear_repulsion)
