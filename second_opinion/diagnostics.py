"""Numbers behind the charts, and what a good value for each one looks like.

A metric nobody can read is decoration. `summary()` printed AUC, Brier and
ECE as bare floats, which answers "what is it" and not "is that good" — so
every figure here ships with its direction, its ideal, and a plain sentence
saying what it measures.

Nothing in this module changes the model. It reshapes what the model already
computed into frames a chart can draw.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SPLIT_RESULTS = Path(__file__).resolve().parent / "split_results.json"


@dataclass(frozen=True)
class MetricGuide:
    """One performance number, plus what counts as good."""

    key: str
    label: str
    ideal: str
    better: str          # "lower" | "higher"
    plain: str           # one sentence, no jargon

    def verdict(self, value: float) -> str:
        """Plain-language read on an observed value."""
        good, ok = _THRESHOLDS[self.key]
        if self.better == "higher":
            band = "good" if value >= good else ("fair" if value >= ok else "poor")
        else:
            band = "good" if value <= good else ("fair" if value <= ok else "poor")
        return band


# good, fair cut-offs. Deliberately conventional rather than tuned to flatter
# this model: AUC bands follow the usual 0.9/0.8 reading, Brier and ECE are
# judged against the "well below the no-skill baseline" rule of thumb.
_THRESHOLDS = {
    "auc": (0.90, 0.80),
    "brier": (0.10, 0.20),
    "ece": (0.05, 0.10),
    "coverage": (0.0, 0.0),  # judged against its own target, not a constant
}

METRIC_GUIDES = {
    "auc": MetricGuide(
        key="auc", label="AUC", ideal="1.000 is perfect, 0.5 is a coin flip",
        better="higher",
        plain="How well the model ranks a malignant sample above a benign one. "
              "It says nothing about whether the percentages are honest.",
    ),
    "brier": MetricGuide(
        key="brier", label="Brier", ideal="0.000 is perfect, lower is better",
        better="lower",
        plain="Average squared error of the probability itself. Punishes a "
              "confident wrong answer far harder than a hedged one.",
    ),
    "ece": MetricGuide(
        key="ece", label="ECE", ideal="0.000 is perfect, lower is better",
        better="lower",
        plain="Expected calibration error: when the model says 70%, how far "
              "from 70% is the truth? This is the number this project rests on.",
    ),
}


def reliability_curve(model, n_bins: int = 8) -> list[dict]:
    """Observed frequency vs stated probability, on held-out data.

    A perfectly calibrated model lies on the diagonal: of the samples it
    called 70%, 70% really are malignant. This is ECE drawn instead of
    summarised.

    Bins hold an equal *number of samples*, not an equal width. This model's
    predictions pile up near 0 and 1, so equal-width deciles leave the middle
    bins holding one or two points each, and a single sample then swings the
    observed frequency from 0% to 100%. That zigzag is sampling noise being
    read as miscalibration. Equal-count bins put the same weight behind every
    point on the line; `n` is carried through so the chart can still show it.
    """
    raw = model.base.predict_proba(model.X_test)[:, 1]
    cal = model.calibrated.predict_proba(model.X_test)[:, 1]
    y = np.asarray(model.y_test, dtype=int)

    rows: list[dict] = []
    for name, p in (("before calibration", raw), ("after calibration", cal)):
        order = np.argsort(p)
        for chunk in np.array_split(order, n_bins):
            if chunk.size == 0:
                continue
            rows.append({
                "stated": float(p[chunk].mean()),
                "observed": float(y[chunk].mean()),
                "n": int(chunk.size),
                "series": name,
            })
    return rows


def isotonic_mapping(model) -> list[dict]:
    """What isotonic calibration actually does to each held-out score.

    x is the raw model score, y is the probability reported after
    calibration. The staircase is the isotonic fit; points on the diagonal
    were left alone. This is the answer to "what is the calibration *doing*".
    """
    raw = model.base.predict_proba(model.X_test)[:, 1]
    cal = model.calibrated.predict_proba(model.X_test)[:, 1]
    y = np.asarray(model.y_test, dtype=int)
    order = np.argsort(raw)
    return [
        {
            "raw": float(raw[i]),
            "calibrated": float(cal[i]),
            "truth": "malignant" if y[i] == 1 else "benign",
        }
        for i in order
    ]


def probability_context(model) -> list[dict]:
    """Every held-out sample's calibrated probability, with its true diagnosis.

    A bare "99.6% malignant" is a number with nothing to sit against. Plotted
    beside the held-out population it becomes a position: this sample sits
    inside the malignant cluster, far from where the benign ones live. That is
    what the percentage is actually claiming.
    """
    p = model.calibrated.predict_proba(model.X_test)[:, 1]
    y = np.asarray(model.y_test, dtype=int)
    return [
        {
            "probability": model.reportable_probability(float(pi)),
            "truth": "malignant" if yi == 1 else "benign",
        }
        for pi, yi in zip(p, y)
    ]


def split_distribution() -> dict | None:
    """The 40-split results, if `scripts/evaluate.py` has written them.

    The app quotes these as headline numbers, so it should be able to show
    the spread they came from rather than asking the reader to trust a mean.
    """
    if not SPLIT_RESULTS.exists():
        return None
    try:
        return json.loads(SPLIT_RESULTS.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


def coverage_rows(dist: dict) -> list[dict]:
    """Per-seed conformal coverage, for a histogram against the target."""
    return [
        {"seed": i, "coverage": c}
        for i, c in enumerate(dist.get("coverage", []))
    ]


def format_probability(p: float) -> str:
    """Never print 0% or 100% for a probability that is neither.

    Rounding 0.0004 to "0%" tells the reader the model is certain when it is
    not, which is the exact failure this project is about.
    """
    pct = p * 100.0
    if 0.0 < pct < 0.1:
        return "<0.1%"
    if 99.9 < pct < 100.0:
        return ">99.9%"
    if pct in (0.0, 100.0):
        return f"{pct:.0f}%"
    if pct < 1.0 or pct > 99.0:
        return f"{pct:.2f}%"
    return f"{pct:.1f}%"
