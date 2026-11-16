"""Fitting-only energy weights and complete-answer grammar likelihood.

No optimizer is constructed or stepped here. Weighted and uniform objectives use
the same row sampler; preference enters exactly once through n_g*w_gi.
"""

from copy import deepcopy
from dataclasses import dataclass
import math
from numbers import Real

import numpy as np
import torch


IQR_FLOOR_HA = 1e-6
ALLOWED_GEOMETRIES = {"H2_DEV01", *(f"LIH_T{i:02d}" for i in range(1, 7))}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class GeometryWeights:
    row_ids: tuple
    weights: tuple
    multipliers: tuple
    _report: dict

    @property
    def report(self):
        return deepcopy(self._report)


def geometry_weights(rows, *, mode, lambda_scale=None):
    """Normalize within one complete fitting geometry; never select lambda.

    Weighted mode requires a caller-supplied positive lambda. Uniform SFT uses
    w_i=1/n and multiplier one; it must not receive an energy temperature.
    Diagnostic, mixed-geometry, mixed-seed, duplicate-ID and nonfinite rows fail
    rather than being filtered. Underflowed tiny weights remain explicit zeros.
    """
    rows = list(rows)
    if not rows or any(row.get("dataset_partition") != "fitting" for row in rows):
        raise PermissionError("Weight statistics accept fitting rows only, without partition filtering")
    contexts = {(row.get("alias"), row.get("pipeline_seed")) for row in rows}
    _require(len(contexts) == 1, "Compute weights separately for each geometry and pipeline seed")
    alias, seed = next(iter(contexts))
    if alias not in ALLOWED_GEOMETRIES:
        raise PermissionError("Validation/test geometries cannot define training weights")
    _require(type(seed) is int and seed >= 0, "A declared integer pipeline seed is required")
    ids = tuple(row.get("row_id") for row in rows)
    _require(all(isinstance(value, str) and value for value in ids) and len(set(ids)) == len(ids),
             "Complete unique fitting row IDs are required")
    _require(all(isinstance(row.get("energy_ha"), Real) and not isinstance(row["energy_ha"], bool)
                 and math.isfinite(row["energy_ha"]) for row in rows), "Nonfinite/nonreal fitting label")
    _require(mode in {"weighted", "uniform"}, "Unknown objective mode")
    energies = np.asarray([row["energy_ha"] for row in rows], dtype=np.float64)
    q25, q75 = np.quantile(energies, [.25, .75], method="linear")
    minimum, iqr = float(np.min(energies)), float(q75-q25)
    dispersion = max(iqr, IQR_FLOOR_HA)
    n = len(rows)
    if mode == "weighted":
        _require(isinstance(lambda_scale, Real) and not isinstance(lambda_scale, bool)
                 and math.isfinite(lambda_scale) and lambda_scale > 0,
                 "Weighted training requires an explicit positive finite lambda_scale")
        tau = float(lambda_scale)*dispersion
        _require(math.isfinite(tau) and tau > 0, "Invalid/underflowed training temperature")
        with np.errstate(under="ignore"):
            unnormalized = np.exp(-(energies-minimum)/tau)
        normalizer = float(unnormalized.sum())
        weights = unnormalized/normalizer
    else:
        _require(lambda_scale is None, "Uniform SFT has no energy-weighting lambda")
        tau, normalizer = None, None
        weights = np.full(n, 1.0/n, dtype=np.float64)
    multipliers = n*weights
    ess = float(1.0/np.dot(weights, weights))
    report = {"alias": alias, "pipeline_seed": seed, "population": "fitting_only", "mode": mode,
              "rows": n, "lambda_scale": None if mode == "uniform" else float(lambda_scale),
              "fitting_minimum_energy_ha": minimum, "fitting_iqr_ha": iqr,
              "iqr_floor_ha": IQR_FLOOR_HA, "dispersion_ha": dispersion, "tau_ha": tau,
              "shifted_exponential_normalizer": normalizer, "effective_sample_size": ess,
              "ess_fraction": ess/n, "maximum_weight": float(np.max(weights)),
              "zero_weights_from_underflow": int(np.count_nonzero(weights == 0)),
              "ess_below_descriptive_5_percent": bool(ess < .05*n),
              "lambda_selected_automatically": False, "rows_filtered": 0,
              "sampling": "Uniform geometry visits and uniform shuffled rows; no weight-proportional sampling",
              "multiplier": "n_g*w_gi exactly once; no sum-of-weights minibatch divisor",
              "interpretation": "ESS is stored-row weight concentration, not independent information"}
    return GeometryWeights(ids, tuple(map(float, weights)), tuple(map(float, multipliers)), report)


def masked_sequence_nll(logits, labels, legal_token_ids):
    """Return one complete-answer NLL per example at grammar temperature one.

    logits[B,L,V] predicts target labels[b,t] at logits[b,t-1]. Labels use -100
    at prompt/right-padding positions. legal_token_ids[b] contains one tuple of
    legal IDs BEFORE each nonignored answer target, including genuine EOS. The
    sparse gather avoids allocating a B*L*V grammar mask. There is no answer-
    length average, token-average model loss, or decoding-temperature argument.
    """
    _require(isinstance(logits, torch.Tensor) and logits.ndim == 3 and logits.is_floating_point()
             and logits.shape[0] > 0 and logits.shape[1] >= 2 and logits.shape[2] > 0,
             "Expected floating causal logits [batch,sequence,vocabulary]")
    _require(isinstance(labels, torch.Tensor) and labels.ndim == 2
             and tuple(labels.shape) == tuple(logits.shape[:2]) and labels.dtype == torch.long,
             "Expected int64 labels matching causal logits")
    _require(labels.device == logits.device and len(legal_token_ids) == logits.shape[0],
             "Labels/grammar batch or device mismatch")
    # Half-precision log-normalizers are evaluated in float32. Float64 test and
    # numerical inputs retain their precision; the cast remains differentiable.
    values = logits.float() if logits.dtype in {torch.float16, torch.bfloat16} else logits
    losses = []
    for b in range(logits.shape[0]):
        positions = torch.nonzero(labels[b] != -100, as_tuple=False).flatten().tolist()
        _require(positions and positions[0] >= 1
                 and positions == list(range(positions[0], positions[-1]+1)),
                 "A contiguous complete answer after a nonempty prompt is required")
        _require(len(positions) == len(legal_token_ids[b]), "Grammar/answer target count mismatch")
        token_losses = []
        for t, allowed in zip(positions, legal_token_ids[b]):
            ids = tuple(allowed)
            _require(ids and len(set(ids)) == len(ids)
                     and all(type(value) is int and 0 <= value < logits.shape[-1] for value in ids),
                     "Invalid/duplicated grammar token ID")
            target = int(labels[b, t])
            _require(target in ids, "A supervised target is outside its legal grammar continuation")
            indices = torch.tensor(ids, dtype=torch.long, device=logits.device)
            legal = values[b, t-1].index_select(0, indices)
            _require(bool(torch.isfinite(legal).all()), "Nonfinite legal-token logit")
            token_losses.append(torch.logsumexp(legal, dim=0)-values[b, t-1, target])
        losses.append(torch.stack(token_losses).sum())
    return torch.stack(losses)


def accumulated_objective(sequence_nll, multipliers, *, window_example_count):
    """Microbatch contribution to a predeclared accumulation-window mean.

    Sum these contributions across all window microbatches before a future user
    optimizer step. Supply the ACTUAL full-window example count, including for a
    final partial window. Resume must preserve/reconstruct that normalizer. Never
    divide each microbatch by its random sum of energy weights. For DDP callers,
    normalizer and any world-size factor must account for DDP gradient averaging.
    """
    _require(isinstance(sequence_nll, torch.Tensor) and sequence_nll.ndim == 1
             and sequence_nll.numel() > 0 and sequence_nll.is_floating_point(),
             "Expected one complete-answer NLL per example")
    _require(type(window_example_count) is int and window_example_count >= len(sequence_nll),
             "Use the actual positive full-window example count")
    weights = torch.as_tensor(multipliers, dtype=sequence_nll.dtype, device=sequence_nll.device)
    _require(weights.shape == sequence_nll.shape and bool(torch.isfinite(weights).all())
             and bool((weights >= 0).all()) and bool(torch.isfinite(sequence_nll).all()),
             "Invalid fixed per-example loss multipliers or NLLs")
    return (sequence_nll*weights).sum()/window_example_count
