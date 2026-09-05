"""Named starting points for the UI, built from real WDBC geometry.

WDBC has no cell-type labels. Naming a row "Typical Fibroblast" would invent
histology the dataset does not have. Presets are real samples (or the
documented impossible-geometry demo) described by what the numbers are.
"""

from __future__ import annotations

import numpy as np

from second_opinion.conformal import ABSTAIN, AMBIGUOUS, DECISIVE
from second_opinion.data import (
    BASE_FEATURES,
    BENIGN,
    MALIGNANT,
    expand_base_features,
    feature_reference,
    load,
)


def archetypal_presets(model) -> dict[str, np.ndarray]:
    """Full 30-feature vectors, one per named profile."""
    data = model.data or load()
    reference = feature_reference(data)
    X, y = data.X, data.y
    scaler = model.scaler

    median = np.array([reference[n]["median"] for n in data.feature_names])
    malignant = _typical_of_class(model, X, y, MALIGNANT, scaler)
    benign = _typical_of_class(model, X, y, BENIGN, scaler)
    borderline = _borderline_sample(model, X)
    implausible = _implausible_geometry(data.feature_names, reference)

    return {
        "Dataset median": median,
        "Typical malignant nucleus": malignant,
        "Typical benign nucleus": benign,
        "Borderline (withhold)": borderline,
        "Implausible geometry": implausible,
    }


def _typical_of_class(model, X, y, label, scaler) -> np.ndarray:
    mask = y == label
    idx = np.where(mask)[0]
    Z = scaler.transform(X[mask])
    mu = Z.mean(axis=0)
    order = np.argsort(((Z - mu) ** 2).sum(axis=1))
    for local in order:
        x = X[idx[local]]
        a = model.assess(x)
        if not a.novelty.is_outlier and a.prediction_set.verdict == DECISIVE:
            return x.copy()
    return X[idx[int(order[0])]].copy()


def _borderline_sample(model, X) -> np.ndarray:
    closest = None
    closest_gap = np.inf
    for x in X:
        a = model.assess(x)
        if a.novelty.is_outlier:
            continue
        if a.prediction_set.verdict in (ABSTAIN, AMBIGUOUS):
            return x.copy()
        gap = abs(a.probability_malignant - 0.5)
        if gap < closest_gap:
            closest_gap = gap
            closest = x
    if closest is not None:
        blended = _blend_until_withhold(model, closest)
        if blended is not None:
            return blended
        return closest.copy()
    return X[0].copy()


def _blend_until_withhold(model, seed: np.ndarray) -> np.ndarray | None:
    """Walk toward the opposite-class mean until conformal withholds."""
    data = model.data
    p = float(model.calibrated.predict_proba(seed.reshape(1, -1))[0, 1])
    target = BENIGN if p >= 0.5 else MALIGNANT
    other = data.X[data.y == target].mean(axis=0)
    for t in np.linspace(0.05, 0.95, 37):
        x = (1.0 - t) * seed + t * other
        a = model.assess(x)
        if not a.novelty.is_outlier and a.prediction_set.verdict != DECISIVE:
            return x
    return None


def _implausible_geometry(
    feature_names: list[str], reference: dict[str, dict[str, float]]
) -> np.ndarray:
    """README demo: area at its dataset max, radius and perimeter at median."""
    start = {n: reference[n]["median"] for n in feature_names}
    chosen = {n: start[n] for n in BASE_FEATURES}
    chosen["mean area"] = reference["mean area"]["max"]
    return expand_base_features(
        chosen, start=start, reference=reference, feature_names=feature_names
    )
