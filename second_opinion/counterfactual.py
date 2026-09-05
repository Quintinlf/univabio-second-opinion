"""What would have to change for the model to stop withholding.

Two withholds, two different answers. Mixing them is how a UI pretends
that "I don't know" is one problem.

  conformal  — the sample looks like training data, but the prediction
               set is empty or contains both labels. Move a measurement
               until the set is decisive.
  novelty    — the sample is not the kind of thing the model has seen.
               Name the measurements that look like typos; do not offer
               a diagnosis on the far side of that edit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from second_opinion.conformal import ABSTAIN, AMBIGUOUS, DECISIVE
from second_opinion.data import (
    BASE_FEATURES,
    base_feature_name,
    expand_base_features,
    feature_reference,
    slider_bounds,
)


@dataclass(frozen=True)
class FeatureDriver:
    feature: str
    value: float
    z_score: float
    mahalanobis_share: float


@dataclass
class Counterfactual:
    kind: str  # "conformal" | "novelty"
    explanation: str
    feature: str | None = None
    old_value: float | None = None
    new_value: float | None = None
    relative_change: float | None = None
    new_verdict: str | None = None
    drivers: list[FeatureDriver] = field(default_factory=list)


def suggest(model, x: np.ndarray) -> Counterfactual | None:
    """Route to the withhold that actually fired. None if the call is decisive."""
    x = np.asarray(x, dtype=float).reshape(-1)
    assessment = model.assess(x)
    if assessment.novelty.is_outlier:
        return novelty_counterfactual(model, x)
    if assessment.prediction_set.verdict in (ABSTAIN, AMBIGUOUS):
        return conformal_counterfactual(model, x)
    return None


def conformal_counterfactual(model, x: np.ndarray) -> Counterfactual | None:
    """Smallest one-slider move that makes the conformal set decisive."""
    x = np.asarray(x, dtype=float).reshape(-1)
    assessment = model.assess(x)
    if assessment.novelty.is_outlier:
        return None
    if assessment.prediction_set.verdict == DECISIVE:
        return None

    names = model.data.feature_names
    reference = feature_reference(model.data)
    start = dict(zip(names, x))
    chosen = {n: start[n] for n in BASE_FEATURES}

    best: tuple[float, str, float, str] | None = None
    for name in BASE_FEATURES:
        lo, hi = slider_bounds(name, reference)
        span = hi - lo
        if span <= 0:
            continue
        current = chosen[name]
        for candidate in np.linspace(lo, hi, 41):
            if abs(candidate - current) < 1e-12:
                continue
            trial = dict(chosen)
            trial[name] = float(candidate)
            vec = expand_base_features(
                trial, start=start, reference=reference, feature_names=names
            )
            trial_assessment = model.assess(vec)
            if trial_assessment.novelty.is_outlier:
                continue
            if trial_assessment.prediction_set.verdict != DECISIVE:
                continue
            score = abs(candidate - current) / span
            label = (
                "MALIGNANT"
                if trial_assessment.prediction_set.includes_malignant
                else "Benign"
            )
            if best is None or score < best[0]:
                best = (score, name, float(candidate), label)

    if best is None:
        return Counterfactual(
            kind="conformal",
            explanation=(
                "No single measurement change within the slider range moves "
                "this sample to a decisive call while staying in distribution. "
                "A human has to look."
            ),
        )

    _, feature, new_value, label = best
    old_value = float(chosen[feature])
    if abs(old_value) > 1e-12:
        relative = (new_value - old_value) / old_value
    else:
        relative = float("inf")
    direction = "drops" if new_value < old_value else "rises"
    pct = f"{abs(relative):.0%}" if np.isfinite(relative) else "a large amount"
    pct_conf = f"{1.0 - model.alpha:.0%}"
    return Counterfactual(
        kind="conformal",
        explanation=(
            f"If {feature} {direction} by {pct} "
            f"({old_value:.3g} → {new_value:.3g}), this sample moves to a "
            f"{pct_conf} confident {label} classification."
        ),
        feature=feature,
        old_value=old_value,
        new_value=new_value,
        relative_change=relative if np.isfinite(relative) else None,
        new_verdict=label,
    )


def novelty_counterfactual(model, x: np.ndarray) -> Counterfactual | None:
    """Name the measurements that make this look unlike training data."""
    x = np.asarray(x, dtype=float).reshape(-1)
    assessment = model.assess(x)
    if not assessment.novelty.is_outlier:
        return None

    scaler = model.scaler
    z = scaler.transform(x.reshape(1, -1))[0]
    shares, _label = _mahalanobis_shares(model.novelty, z)
    family = _aggregate_to_base(shares, model.data.feature_names)
    ranked = sorted(family.items(), key=lambda kv: abs(kv[1]), reverse=True)

    drivers = []
    start = dict(zip(model.data.feature_names, x))
    for name, share in ranked[:2]:
        if name not in start:
            continue
        value = float(start[name])
        idx = model.data.feature_names.index(name)
        z_score = float(z[idx]) if name in model.data.feature_names else 0.0
        # z is aligned with all 30 features; base feature index is the mean column.
        drivers.append(
            FeatureDriver(
                feature=name,
                value=value,
                z_score=z_score,
                mahalanobis_share=float(share),
            )
        )

    if not drivers:
        return Counterfactual(
            kind="novelty",
            explanation=assessment.novelty.explain(),
        )

    top = drivers[0]
    idx = model.data.feature_names.index(top.feature)
    top = FeatureDriver(
        feature=top.feature,
        value=top.value,
        z_score=float(z[idx]),
        mahalanobis_share=top.mahalanobis_share,
    )
    drivers[0] = top
    sigma = f"{abs(top.z_score):.1f}"
    return Counterfactual(
        kind="novelty",
        explanation=(
            f"{top.feature} sits {sigma}σ from the training centre "
            f"(value {top.value:.3g}). This does not look like the breast-"
            f"tissue measurements the model was calibrated on. Check the "
            f"decimal place or re-measure before reading any percentage. "
            f"This is not a diagnosis."
        ),
        feature=top.feature,
        old_value=top.value,
        drivers=drivers,
    )


def _mahalanobis_shares(detector, z: np.ndarray) -> tuple[np.ndarray, int]:
    """Per-feature contribution to the winning class-conditional quadratic form."""
    best_label = None
    best_dist = np.inf
    best_share = None
    z = np.asarray(z, dtype=float).reshape(-1)
    for label, cov in detector._class_covs.items():
        delta = z - cov.location_
        precision_delta = cov.precision_ @ delta
        dist = float(delta @ precision_delta)
        if dist < best_dist:
            best_dist = dist
            best_label = label
            best_share = delta * precision_delta
    return best_share, best_label


def _aggregate_to_base(
    shares: np.ndarray, feature_names: list[str]
) -> dict[str, float]:
    family: dict[str, float] = {name: 0.0 for name in BASE_FEATURES}
    for name, share in zip(feature_names, shares):
        base = base_feature_name(name)
        if base in family:
            family[base] += abs(float(share))
    return family
