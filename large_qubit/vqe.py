"""Budgeted compact fermionic excitation VQE, without model training.

An excitation generator has eigenvalues 0,+i/2,-i/2. A coordinate energy
therefore contains frequencies 0,1/2,1, rather than just a single sinusoid.
Five charged samples determine that curve; a sixth independently checks the
proposed minimum. The evaluator callback owns every physical request/ledger.
"""

import math

from .common import require


OFFSETS = (-math.pi, -math.pi / 2, 0., math.pi / 2, math.pi)


def _features(theta):
    import numpy as np
    return np.array([1., math.cos(theta), math.sin(theta),
                     math.cos(theta / 2), math.sin(theta / 2)])


def coordinate_minimum(energies, *, center=0., bounds=(-.25, .25)):
    """Find all stationary points of the fitted five-term coordinate curve.

    A tan(theta/4) substitution makes its derivative a quartic polynomial.
    Checking every real polynomial root and both endpoints avoids a grid-based
    missed minimum. The declared angle interval stays strictly within (-2pi,2pi).
    """
    import numpy as np
    require(len(energies) == 5 and all(type(x) in (int, float) and math.isfinite(x)
                                    for x in energies), 'Five finite coordinate energies required')
    low, high = bounds
    require(-2 * math.pi < low <= center <= high < 2 * math.pi, 'Invalid coordinate bounds/center')
    coefficients = np.linalg.solve(np.array([_features(center + offset) for offset in OFFSETS]), energies)
    a, b, c, d, e = coefficients
    # derivative = -b sin(theta)+c cos(theta)-d/2 sin(theta/2)+e/2 cos(theta/2)
    # Multiply by (1+t*t)**2, t=tan(theta/4).
    polynomial = np.array([c - e / 2, 4 * b - d, -6 * c,
                           -4 * b - d, c + e / 2])
    first = 0
    while first < 4 and abs(polynomial[first]) < 1e-15:
        first += 1
    roots = np.roots(polynomial[first:]) if first < 4 else []
    candidates = [float(low), float(high), float(center)]
    for root in roots:
        if abs(root.imag) <= 1e-9:
            theta = float(4 * math.atan(float(root.real)))
            if low <= theta <= high:
                candidates.append(theta)
    scored = [(float(np.dot(coefficients, _features(x))), abs(x - center), x) for x in candidates]
    value, _, theta = min(scored)
    return {'theta': theta, 'predicted_energy': value, 'coefficients': coefficients.tolist(),
            'stationary_points_checked': len(candidates) - 3}


def optimize_compact(evaluate, initial, *, max_requests, bounds=(-.25, .25),
                     interpolation_tolerance=1e-8, progress=None):
    """Optimize with an exact local request cap and deterministic coordinate order.

    ``evaluate(parameters, metadata)`` must charge duplicates and failed calls.
    A partial last sweep stops before exceeding the cap. Returned history keeps
    parameters, all samples, proposals and the incumbent; no convergence claim.
    """
    import numpy as np
    parameters = np.array(initial, dtype=float, copy=True)
    require(parameters.ndim == 1 and 1 <= len(parameters) <= 64
            and np.all(np.isfinite(parameters)), 'Use 1..64 finite compact VQE parameters')
    require(type(max_requests) is int and max_requests >= 1, 'Invalid VQE request allowance')
    require(np.all(parameters >= bounds[0]) and np.all(parameters <= bounds[1]), 'Initial angles exceed bounds')
    requests = 0

    def query(values, metadata):
        nonlocal requests
        require(requests < max_requests, 'VQE cannot exceed its request cap')
        requests += 1
        energy = float(evaluate(values.tolist(), metadata))
        require(math.isfinite(energy), 'Nonfinite VQE energy')
        return energy

    energy = query(parameters, {'stage': 'initial', 'local_request': 1})
    initial_energy = energy
    best_energy, best_parameters = energy, parameters.copy()
    history, sweep = [], 0
    while requests + 6 <= max_requests:
        for coordinate in range(len(parameters)):
            if requests + 6 > max_requests:
                break
            center = float(parameters[coordinate])
            samples = []
            for offset in OFFSETS:
                candidate = parameters.copy()
                candidate[coordinate] = center + offset
                samples.append(query(candidate, {'stage': 'coordinate_sample', 'sweep': sweep,
                    'coordinate': coordinate, 'offset': offset, 'local_request': requests + 1}))
                # Every actual sampled energy is a possible budgeted incumbent.
                if samples[-1] < best_energy:
                    best_energy, best_parameters = samples[-1], candidate.copy()
            minimum = coordinate_minimum(samples, center=center, bounds=bounds)
            candidate = parameters.copy()
            candidate[coordinate] = minimum['theta']
            measured = query(candidate, {'stage': 'coordinate_proposal', 'sweep': sweep,
                'coordinate': coordinate, 'local_request': requests + 1})
            mismatch = abs(measured - minimum['predicted_energy'])
            require(mismatch <= interpolation_tolerance,
                    'Excitation coordinate does not satisfy the five-term energy identity')
            if measured <= energy + 1e-10:
                parameters, energy = candidate, measured
            if measured < best_energy:
                best_energy, best_parameters = measured, candidate.copy()
            history.append({'sweep': sweep, 'coordinate': coordinate, 'sample_energies': samples,
                'center': center, 'proposal': minimum, 'measured_energy': measured,
                'interpolation_error': mismatch, 'accepted_energy': energy,
                'accepted_parameters': parameters.tolist(), 'best_energy': best_energy,
                'requests': requests})
            if progress is not None:
                progress({'coordinate': coordinate, 'sweep': sweep, 'requests': requests})
        sweep += 1
    return {'parameters': best_parameters.tolist(), 'energy': best_energy,
            'accepted_parameters': parameters.tolist(), 'accepted_energy': energy,
            'initial_energy': initial_energy, 'requests': requests, 'request_cap': max_requests,
            'unused_requests': max_requests - requests, 'completed_sweeps': len(history) // len(parameters),
            'history': history, 'optimizer': 'five_point_excitation_coordinate_minimization',
            'convergence_claimed': False, 'all_calls_charged_by_caller': True}
