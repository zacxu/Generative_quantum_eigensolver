"""Structural fixtures only; no integral screening, energies or model files."""

from large_qubit.catalogue import VQE_TEMPLATE_COUNT, _payload, full_templates, validate_catalogue
from large_qubit.common import digest


def catalogue_fixture():
    full = full_templates()
    uncovered, cover = set(range(30)), []
    while uncovered:
        best = min(full, key=lambda row: (-len(set(row["wires"]) & uncovered), row["source_index"]))
        cover.append(best["source_index"])
        uncovered.difference_update(best["wires"])
    vqe = set(cover)
    for source in range(len(full)):
        if len(vqe) == VQE_TEMPLATE_COUNT:
            break
        vqe.add(source)
    selected = set(vqe)
    for source in range(len(full)):
        if len(selected) == 64:
            break
        selected.add(source)
    payload = _payload([full[source] for source in sorted(selected)],
        electrons=22, qubits=30, pool="screened_training_integrals",
        vqe_sources=sorted(vqe), sources={alias: str(i) * 64 for i, alias in enumerate(
            ("CO2_T01", "CO2_T02", "CO2_T03"), start=1)})
    return validate_catalogue({**payload, "sha256": digest(payload)})
