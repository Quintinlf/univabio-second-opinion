"""Distribution-shift stress test: withhold should rise, confident error should not.

A single in-distribution split cannot show the product. Progressively corrupt
the held-out matrix — Gaussian measurement noise, then a size-feature scale
shift that stands in for a different lab's resolution — and record whether
the model starts guessing or starts refusing.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from second_opinion import conformal
from second_opinion.data import size_feature_mask


Kind = Literal["noise", "scale"]


def evaluate_shift(
    model,
    X: np.ndarray,
    y: np.ndarray,
    *,
    levels: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0),
    kind: Kind = "noise",
    seed: int = 0,
) -> list[dict[str, float]]:
    """Corrupt X, score withhold and confident error at each level."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    rng = np.random.default_rng(seed)
    rows = []
    for level in levels:
        corrupted = _corrupt(model, X, level=level, kind=kind, rng=rng)
        rows.append({"level": float(level), "kind": kind, **_rates(model, corrupted, y)})
    return rows


def _corrupt(model, X, *, level: float, kind: Kind, rng) -> np.ndarray:
    if level == 0:
        return X.copy()
    if kind == "noise":
        scale = model.scaler.scale_
        noise = rng.normal(0.0, level, size=X.shape) * scale
        return X + noise
    mask = size_feature_mask(model.data.feature_names)
    out = X.copy()
    out[:, mask] *= 1.0 + level
    return out


def _rates(model, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    p = model.calibrated.predict_proba(X)
    z = model.scaler.transform(X)
    outlier = model.novelty.score(z) > model.novelty.threshold
    included = p >= (1.0 - model.qhat)
    sizes = included.sum(axis=1)
    abstain = sizes == 0
    ambiguous = sizes == 2
    decisive = sizes == 1
    true_in_set = included[np.arange(len(y)), y]
    confident_error = decisive & ~outlier & ~true_in_set
    withhold = outlier | abstain | ambiguous
    return {
        "n": float(len(y)),
        "novelty": float(outlier.mean()),
        "abstain": float(abstain.mean()),
        "ambiguous": float(ambiguous.mean()),
        "decisive": float(decisive.mean()),
        "withhold": float(withhold.mean()),
        "confident_error": float(confident_error.mean()),
        "coverage": float(conformal.coverage(p, y, model.qhat)),
    }
