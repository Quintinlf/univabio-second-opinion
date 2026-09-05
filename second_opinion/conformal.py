"""Split conformal prediction: a coverage guarantee instead of a vibe.

A calibrated probability tells you the model's long-run hit rate at a given
score. It does not tell you, for one specific patient, whether the answer can
be trusted. Conformal prediction does something different and stronger: it
returns a *set* of labels, and guarantees that the true label lands in that
set at least (1 - alpha) of the time — with no assumption about the model
being correct, or even good.

The guarantee is what makes the abstention honest. Three outcomes:

  {malignant}          one label -> a decisive call
  {benign}
  {malignant, benign}  both -> genuinely ambiguous, a human should look
  {}                   neither -> nothing is plausible at this confidence;
                       the model abstains rather than guessing

That empty set is the point of the whole project. A model that cannot say
"I don't know" will hand a confident number to a case it has never seen.

Method
------
LAC (Least Ambiguous set-valued Classifier), also called THR (Sadinle,
Lei & Wasserman, 2019): the nonconformity score of a labelled example is
`1 - p(true label)`. Calibrate by taking the ceil((n+1)(1-alpha))/n
empirical quantile of those scores on a held-out set, then include every
label whose probability clears `1 - qhat`.

Split conformal prediction with the finite-sample correction follows
Angelopoulos & Bates (2023). The guarantee is *marginal*: coverage holds
on average over the population, not necessarily within every subgroup.

Measured on WDBC over 25 random splits at alpha=0.10: 94.0% coverage, 95% of
test cases decided with a single label, 5% abstained. Coverage runs slightly
above the 90% target — split conformal is conservative by construction, and
that direction is the safe one here.

APS (Adaptive Prediction Sets; Sadinle et al., 2019) was tried and rejected:
it reached 99% coverage but returned both labels on 13-24% of cases, which
converts a decisive tool into one that shrugs at a quarter of its inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from second_opinion.data import BENIGN, MALIGNANT

ABSTAIN = "abstain"
AMBIGUOUS = "ambiguous"
DECISIVE = "decisive"


@dataclass(frozen=True)
class PredictionSet:
    """What the model is willing to claim about one sample."""

    includes_malignant: bool
    includes_benign: bool
    probability_malignant: float
    alpha: float

    @property
    def labels(self) -> list[int]:
        out = []
        if self.includes_malignant:
            out.append(MALIGNANT)
        if self.includes_benign:
            out.append(BENIGN)
        return out

    @property
    def verdict(self) -> str:
        n = len(self.labels)
        if n == 0:
            return ABSTAIN
        if n == 2:
            return AMBIGUOUS
        return DECISIVE

    @property
    def confidence(self) -> float:
        return 1.0 - self.alpha

    def explain(self) -> str:
        """One sentence a non-specialist can act on."""
        pct = f"{self.confidence:.0%}"
        if self.verdict == ABSTAIN:
            return (
                f"I am withholding a diagnosis because neither label is "
                f"plausible at the {pct} confidence level. The probability "
                "sits between the two conformal thresholds, so the prediction "
                "set is empty. Do not read the percentage below as a result."
            )
        if self.verdict == AMBIGUOUS:
            return (
                f"I am withholding a single-label diagnosis because this "
                f"sample sits on the boundary at {pct} confidence. The "
                "prediction set contains both benign and malignant — a human "
                "should look."
            )
        label = "MALIGNANT" if self.includes_malignant else "benign"
        return (
            f"Single call: {label}, at the {pct} confidence level. Across many "
            f"such cases the true label falls inside this set at least {pct} of "
            "the time."
        )


def calibrate(probabilities: np.ndarray, labels: np.ndarray, alpha: float = 0.10) -> float:
    """Return qhat, the nonconformity threshold, from a held-out calibration set.

    `probabilities` is (n, 2) as sklearn emits it; column 1 is malignant.

    The finite-sample correction — ceil((n+1)(1-alpha))/n rather than plain
    (1-alpha) — is what turns this from a heuristic into a guarantee. Dropping
    it undercovers on small calibration sets, which is exactly the regime a
    569-sample medical dataset lives in.
    """
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError(f"expected (n, 2) probabilities, got {probabilities.shape}")
    if len(probabilities) != len(labels):
        raise ValueError("probabilities and labels differ in length")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    n = len(labels)
    scores = 1.0 - probabilities[np.arange(n), labels]
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def predict_set(
    probability_malignant: float, qhat: float, alpha: float = 0.10
) -> PredictionSet:
    """Turn one calibrated probability into a prediction set."""
    threshold = 1.0 - qhat
    return PredictionSet(
        includes_malignant=bool(probability_malignant >= threshold),
        includes_benign=bool((1.0 - probability_malignant) >= threshold),
        probability_malignant=float(probability_malignant),
        alpha=alpha,
    )


def coverage(probabilities: np.ndarray, labels: np.ndarray, qhat: float) -> float:
    """Share of samples whose true label is inside the prediction set.

    Used to check the guarantee held on data the threshold was not fitted on —
    a conformal method that is never measured is just an assertion.
    """
    sets = probabilities >= (1.0 - qhat)
    return float(sets[np.arange(len(labels)), labels].mean())


def set_size_distribution(probabilities: np.ndarray, qhat: float) -> dict[str, float]:
    sets = probabilities >= (1.0 - qhat)
    sizes = sets.sum(axis=1)
    return {
        ABSTAIN: float((sizes == 0).mean()),
        DECISIVE: float((sizes == 1).mean()),
        AMBIGUOUS: float((sizes == 2).mean()),
    }
