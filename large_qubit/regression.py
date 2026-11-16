"""Read-only H2/LiH regressions against preserved independent Pauli fixtures.

Recover spatial integrals from the old Pauli operator's vacuum, one-electron
and opposite-spin two-electron blocks. This preserves the exact historical
Hamiltonian instead of rerunning chemistry with a different AO basis or gauge.
Absolute energies are returned only in the explicit private result.
"""

from pathlib import Path

import numpy as np
from scipy import sparse

from .common import digest, read_json, require, sha256_file, workspace_path


DEVELOPMENT_ALIASES = ("H2_DEV01", "LIH_T02")
ENERGY_TOLERANCE = 1e-8


def independent_pauli_matrix(operator):
    """Small independent computational-basis CSR; never used for 30 qubits."""
    definition = operator["physical_definition"]
    nqubits = definition["n_qubits"]
    require(type(nqubits) is int and 1 <= nqubits <= 12,
            "Independent Pauli regression is restricted to at most 12 wires")
    columns = np.arange(1 << nqubits, dtype=np.int64)
    row_parts, column_parts, values = [], [], []
    for term in operator["pauli_terms"]:
        word = term["word"]
        require(type(word) is str and len(word) == nqubits and set(word) <= set("IXYZ"),
                "Malformed historical dense Pauli word")
        coefficient = complex(term["real"], term["imag"])
        require(np.isfinite(coefficient), "Non-finite historical Pauli coefficient")
        phase = np.full(columns.size, coefficient, dtype=np.complex128)
        flip = 0
        for wire, letter in enumerate(word):
            mask = 1 << (nqubits - 1 - wire)
            occupied = (columns & mask) != 0
            if letter in "XY":
                flip ^= mask
            if letter == "Y":
                phase *= np.where(occupied, -1j, 1j)
            elif letter == "Z":
                phase *= np.where(occupied, -1, 1)
        row_parts.append(columns ^ flip)
        column_parts.append(columns)
        values.append(phase)
    require(bool(values), "Historical Pauli operator is empty")
    matrix = sparse.coo_matrix((np.concatenate(values),
                               (np.concatenate(row_parts), np.concatenate(column_parts))),
                              shape=(columns.size, columns.size)).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    delta = matrix - matrix.getH()
    require(delta.nnz == 0 or np.max(np.abs(delta.data)) <= 1e-10,
            "Historical Pauli operator is not Hermitian")
    return matrix


def extract_spatial_integrals(operator, matrix=None):
    """Recover h[p,q], (pq|rs), nuclear constant with explicit fermionic signs.

    An alpha/beta product state is a†(p,alpha)a†(r,beta)|vac>. Its
    computational-basis phase is negative when p>r. Opposite-spin two-electron
    matrix elements then give the chemists Coulomb tensor after subtracting
    one-electron and nuclear terms. This derivation is independent of PySCF.
    """
    matrix = independent_pauli_matrix(operator) if matrix is None else matrix
    n = operator["physical_definition"]["n_spatial_orbitals"]
    require(type(n) is int and 2 * n == operator["physical_definition"]["n_qubits"],
            "Historical spatial/spin orbital dimensions differ")
    alpha = np.array([1 << (2 * n - 1 - 2 * p) for p in range(n)])
    beta = np.array([1 << (2 * n - 2 - 2 * p) for p in range(n)])
    nuclear = complex(matrix[0, 0])
    h_alpha = matrix[alpha, :][:, alpha].toarray() - nuclear * np.eye(n)
    h_beta = matrix[beta, :][:, beta].toarray() - nuclear * np.eye(n)
    require(np.max(np.abs(h_alpha - h_beta)) <= 1e-9,
            "Historical operator is not spin-independent in the one-electron sector")
    indices = (alpha[:, None] | beta[None, :]).reshape(-1)
    phases = np.where(np.arange(n)[:, None] > np.arange(n)[None, :], -1.0, 1.0).reshape(-1)
    two = matrix[indices, :][:, indices].toarray() * phases[:, None] * phases[None, :]
    # Two block indices are (p,r) and (q,s), not (p,q) and (r,s).
    eri = two.reshape(n, n, n, n).transpose(0, 2, 1, 3)
    identity = np.eye(n)
    eri -= np.einsum("pq,rs->pqrs", h_alpha, identity)
    eri -= np.einsum("pq,rs->pqrs", identity, h_alpha)
    eri -= nuclear * np.einsum("pq,rs->pqrs", identity, identity)
    require(max(abs(nuclear.imag), np.max(abs(h_alpha.imag)), np.max(abs(eri.imag))) <= 1e-9,
            "Historical real spatial integral reconstruction has imaginary components")
    h1, eri = h_alpha.real, eri.real
    require(np.max(abs(h1 - h1.T)) <= 1e-9
            and np.max(abs(eri - eri.swapaxes(0, 1))) <= 1e-9
            and np.max(abs(eri - eri.transpose(2, 3, 0, 1))) <= 1e-9,
            "Recovered historical spatial integrals fail molecular symmetries")
    return h1, eri, float(nuclear.real)


def _independent_circuit(hf_bits, operations, catalogue):
    """Full small-register whole-qubit rotations, independent of SectorBasis."""
    nqubits = len(hf_bits)
    state = np.zeros(1 << nqubits, dtype=np.complex128)
    state[int("".join(map(str, hf_bits)), 2)] = 1.0
    all_indices = np.arange(state.size)
    for identifier in operations:
        entry = catalogue["entries"][identifier]
        if entry["name"] == "Identity":
            continue
        masks = [1 << (nqubits - 1 - wire) for wire in entry["wires"]]
        union = sum(masks)
        low_mask = sum(masks[len(masks) // 2:])
        low = all_indices[(all_indices & union) == low_mask]
        high = low ^ union
        low_values, high_values = state[low].copy(), state[high].copy()
        c, s = np.cos(entry["angle"] / 2), np.sin(entry["angle"] / 2)
        state[low] = c * low_values - s * high_values
        state[high] = s * low_values + c * high_values
    return state


def run_historical_regressions(workspace, *, aliases=DEVELOPMENT_ALIASES, threads=1, seed=22):
    """Compare new evaluator/gates/references with exact preserved development H.

    This function writes nothing. Persist ``private`` only under scoring_only;
    ``public`` contains source identities and numerical discrepancies, never
    absolute reference/HF/candidate energies or model-facing labels.
    """
    from .evaluator import IntegralEvaluator
    from .reference import solve_reference
    from .sector import SectorBasis

    root = Path(workspace).resolve(strict=True)
    require(type(aliases) in (tuple, list) and aliases and len(set(aliases)) == len(aliases)
            and set(aliases) <= set(DEVELOPMENT_ALIASES),
            "Only the fixed H2/LiH development regression aliases are permitted")
    sources = ("phase3/benchmark_amendment.json", "phase3/references_development.json",
               "phase3/validation_fixtures.json", "phase3/catalogues/h2.json", "phase3/catalogues/lih.json")
    source_hashes = {name: sha256_file(workspace_path(root, name)) for name in sources}
    amendment = read_json(root / sources[0])
    references = {r["alias"]: r for r in read_json(root / sources[1])["references"]}
    fixtures = read_json(root / sources[2])
    operators = {item["alias"]: item for item in amendment["operators"]}
    require(fixtures["selection_precedes_energy_evaluation"] is True,
            "Historical fixtures were not independently fixed before scoring")
    require(fixtures["amendment_sha256"] == source_hashes[sources[0]],
            "Historical fixtures do not bind the preserved operator amendment")
    public_cases, private_cases = {}, {}
    for alias in aliases:
        record = operators[alias]
        path = workspace_path(root, record["operator_path"])
        operator_hash = sha256_file(path)
        require(operator_hash == record["operator_file_sha256"], "Historical operator artifact changed")
        source_hashes[record["operator_path"]] = operator_hash
        operator = read_json(path)
        physical = operator["physical_definition"]
        reference = references[alias]
        require(reference["active_operator_file_sha256"] == operator_hash
                and reference["amendment_sha256"] == source_hashes[sources[0]],
                "Historical saved reference has a different Hamiltonian identity")
        n = physical["n_spatial_orbitals"]
        populations = (physical["n_alpha"], physical["n_beta"])
        basis = SectorBasis(n, *populations)
        full = independent_pauli_matrix(operator)
        h1, eri, nuclear = extract_spatial_integrals(operator, full)
        evaluator = IntegralEvaluator(h1, eri, nuclear, n, populations, threads=threads)
        real_path_check = evaluator.validate_real_path(seed=seed, samples=2, tolerance=1e-10)
        sector_indices = basis.sector_indices.astype(np.int64)
        small = full[sector_indices, :][:, sector_indices].toarray()
        rng = np.random.default_rng(seed)
        action_error = hermiticity_error = float_error = 0.0
        for _ in range(3):
            state = rng.normal(size=basis.dimension) + 1j * rng.normal(size=basis.dimension)
            state /= np.linalg.norm(state)
            acted = evaluator.apply_sector(state, basis)
            action_error = max(action_error, float(np.max(abs(acted - small @ state))))
            energy = evaluator.energy_sector(state, basis)
            hermiticity_error = max(hermiticity_error, abs(complex(np.vdot(state, acted)).imag))
            float_state = state.real.copy()
            float_state /= np.linalg.norm(float_state)
            # A real optimization must agree with the complex128 reference path.
            float_error = max(float_error, float(np.max(abs(
                evaluator.apply_h_real(basis.to_ci(float_state))
                - evaluator.apply_h(basis.to_ci(float_state.astype(np.complex128)))))))
            require(abs(energy - np.vdot(state, small @ state).real) <= ENERGY_TOLERANCE,
                    "Historical random-complex energy disagrees")
        species = "H2" if alias.startswith("H2") else "LiH"
        catalogue_path = f"phase3/catalogues/{species.lower()}.json"
        catalogue = read_json(root / catalogue_path)
        catalogue_body = {k: v for k, v in catalogue.items() if k != "sha256"}
        # Historical catalogue JSON used compact sorted JSON with default ASCII.
        require(digest(catalogue_body) == catalogue["sha256"]
                == fixtures[species]["catalogue_sha256"] == reference["catalogue_sha256"],
                "Historical catalogue or fixture identity changed")
        state_error = circuit_energy_error = norm_error = leakage = 0.0
        circuit_energies = []
        outside = np.ones(full.shape[0], dtype=bool)
        outside[basis.sector_indices] = False
        for fixture in fixtures[species]["fixtures"]:
            expected_full = _independent_circuit(physical["hf_bitstring"], fixture["ops"], catalogue)
            state = basis.hf_state()
            for identifier in fixture["ops"]:
                entry = catalogue["entries"][identifier]
                if entry["name"] != "Identity":
                    state = basis.apply_gate(state, entry["wires"], entry["angle"], family="whole_qubit")
            expected = expected_full[basis.sector_indices]
            state_error = max(state_error, float(np.max(abs(state - expected))))
            norm_error = max(norm_error, abs(float(np.vdot(state, state).real) - 1))
            leakage = max(leakage, float(np.vdot(expected_full[outside], expected_full[outside]).real))
            actual_energy = evaluator.energy_sector(state, basis)
            independent_energy = float(np.vdot(expected_full, full @ expected_full).real)
            circuit_energy_error = max(circuit_energy_error, abs(actual_energy - independent_energy))
            circuit_energies.append({"name": fixture["name"], "energy_ha": actual_energy})
        solved = solve_reference(evaluator, seed=seed)
        ref_state = basis.from_ci(solved["state"])
        independent_ground = float(np.linalg.eigvalsh(small)[0])
        reference_difference = abs(float(solved["energy"]) - independent_ground)
        saved_difference = abs(float(solved["energy"]) - reference["reference_energy_ha"])
        residual = float(np.linalg.norm(small @ ref_state - solved["energy"] * ref_state))
        hf = evaluator.energy_sector(basis.hf_state(), basis)
        hf_difference = abs(hf - reference["hartree_fock_energy_ha"])
        checks = {"complex_apply_h_max_abs_error_ha": action_error,
                  "complex_expectation_imaginary_max_abs_ha": hermiticity_error,
                  "real_vs_complex_apply_h_max_abs_error_ha": float_error,
                  "circuit_state_max_abs_error": state_error, "circuit_norm_max_abs_error": norm_error,
                  "circuit_sector_leakage": leakage, "circuit_energy_max_abs_error_ha": circuit_energy_error,
                  "reference_vs_independent_ground_abs_error_ha": reference_difference,
                  "reference_vs_saved_ground_abs_error_ha": saved_difference,
                  "independent_reference_residual_norm_ha": residual,
                  "hf_vs_saved_abs_error_ha": hf_difference}
        thresholds = {name: (1e-10 if name in (
            "circuit_state_max_abs_error", "circuit_norm_max_abs_error",
            "complex_expectation_imaginary_max_abs_ha") else ENERGY_TOLERANCE) for name in checks}
        passed = all(value <= thresholds[name] for name, value in checks.items())
        public_cases[alias] = {"all_passed": passed, "populations": list(populations),
                               "spatial_orbitals": n, "sector_dimension": basis.dimension,
                               "fixed_circuit_fixtures_checked": len(circuit_energies), "checks": checks,
                               "check_tolerances": thresholds,
                               "real_path_preflight": real_path_check,
                               "operator_file_sha256": operator_hash,
                               "integral_recovery": "exact_Pauli_vacuum_1e_opposite_spin_2e_blocks_no_new_SCF"}
        private_cases[alias] = {"reference_energy_ha": float(solved["energy"]),
                                "saved_reference_energy_ha": reference["reference_energy_ha"],
                                "independent_ground_energy_ha": independent_ground,
                                "hf_energy_ha": hf, "circuits": circuit_energies,
                                "reference_solver": solved["report"]}
    public = {"all_passed": all(x["all_passed"] for x in public_cases.values()),
              "scope": "historical_H2_LiH_development_correctness_not_CO2_results",
              "source_files_sha256": source_hashes, "cases": public_cases,
              "energy_and_residual_tolerance_ha": ENERGY_TOLERANCE,
              "historical_files_modified": False, "new_molecular_SCF_runs": 0}
    return {"public": public, "private": {"scope": "correctness_scoring_only", "cases": private_cases}}
