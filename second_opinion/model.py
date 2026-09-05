"""The model, and the three-way split that makes its claims checkable.

Most of this file is bookkeeping around one idea: a prediction is not one
number, it is three separate questions that fail in different ways.

  1. What is the probability?        -> calibrated classifier
  2. Can that probability be
     trusted for THIS sample?        -> conformal prediction set
  3. Is this sample even the kind
     of thing the model has seen?    -> Mahalanobis novelty detection

Any one of them alone is misleading. A calibrated model is well-calibrated
*on average* and still confidently wrong on an outlier. A conformal set is
guaranteed *in distribution* and silently voids on a distribution shift.
Novelty detection catches the shift but says nothing about the diagnosis.
Reporting all three is the product.

Calibration follows Guo et al. (2017): high accuracy does not imply
calibration. We report AUC alongside Brier and expected calibration error
(ECE) to make that trade visible. Isotonic regression (not Guo's temperature
scaling) is used because this is an additive tabular model, not a deep
network — but the point is the same: a model can rank well and still
misstate its probabilities.

The base classifier is plain logistic regression on standardised features.
A per-feature spline GAM was tried and measured worse on every metric --
AUC, Brier, ECE, conformal coverage and abstention rate, over 25 seeds (the
table is in gam.LinearAdditive). At n=569 the extra flexibility is not
supported by the data and it is calibration that pays, which is the one
property this project rests on. A GP would bend the same way and also lose
the exact attributions.

The data is split three ways, not two: train / calibration / test. The
calibration split does double duty (isotonic calibration and the conformal
quantile) and is never trained on; the test split is touched only when
reporting final numbers. Calibrating on the training set would produce a
model that looks perfectly calibrated and is not — the classic leak.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from second_opinion import conformal
from second_opinion.data import Dataset, load
from second_opinion.gam import LinearAdditive
from second_opinion.ood import NoveltyDetector, NoveltyVerdict

DEFAULT_ALPHA = 0.10
DEFAULT_SEED = 0
ECE_BINS = 10


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, n_bins: int = ECE_BINS
) -> float:
    """Expected calibration error (Guo et al., 2017) with uniform binning."""
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=int)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for low, high in zip(bin_edges[:-1], bin_edges[1:]):
        if high == 1.0:
            mask = (probabilities >= low) & (probabilities <= high)
        else:
            mask = (probabilities >= low) & (probabilities < high)
        if not mask.any():
            continue
        bin_acc = labels[mask].mean()
        bin_conf = probabilities[mask].mean()
        ece += mask.mean() * abs(bin_acc - bin_conf)
    return float(ece)


@dataclass
class Assessment:
    """Everything the model is willing to say about one sample."""

    probability_malignant: float
    prediction_set: conformal.PredictionSet
    novelty: NoveltyVerdict
    contributions: list[tuple[str, float]] = field(default_factory=list)

    @property
    def trustworthy(self) -> bool:
        """Whether the probability should be shown as a result at all.

        False when the sample is out of distribution, or when conformal
        abstained. Deliberately conservative: this gates the headline number
        in the UI, and the cost of hiding a good prediction is far lower than
        the cost of showing a confident wrong one.
        """
        return not self.novelty.is_outlier and self.prediction_set.verdict != conformal.ABSTAIN

    def headline(self) -> str:
        if self.novelty.is_outlier:
            return "Out of distribution — no diagnosis offered"
        if self.prediction_set.verdict == conformal.ABSTAIN:
            return "Abstained — no confident label"
        if self.prediction_set.verdict == conformal.AMBIGUOUS:
            return "Borderline — both labels plausible"
        label = "MALIGNANT" if self.prediction_set.includes_malignant else "Benign"
        return f"{label} ({self.probability_malignant:.0%} malignant)"


@dataclass
class Metrics:
    """Held-out performance. Every figure the README quotes comes from here."""

    n_train: int
    n_calibration: int
    n_test: int
    prevalence: float
    auc_uncalibrated: float
    auc_calibrated: float
    brier_uncalibrated: float
    brier_calibrated: float
    ece_uncalibrated: float
    ece_calibrated: float
    coverage: float
    target_coverage: float
    set_sizes: dict[str, float]
    novelty_flagged_test: float

    def summary(self) -> str:
        return (
            f"train {self.n_train} / calibration {self.n_calibration} / test {self.n_test}"
            f"  (malignant prevalence {self.prevalence:.1%})\n"
            f"AUC   {self.auc_uncalibrated:.3f} -> {self.auc_calibrated:.3f} after calibration\n"
            f"Brier {self.brier_uncalibrated:.4f} -> {self.brier_calibrated:.4f}\n"
            f"ECE   {self.ece_uncalibrated:.4f} -> {self.ece_calibrated:.4f}\n"
            f"conformal coverage {self.coverage:.1%} (target {self.target_coverage:.0%})\n"
            f"  decisive {self.set_sizes['decisive']:.1%}"
            f" · ambiguous {self.set_sizes['ambiguous']:.1%}"
            f" · abstain {self.set_sizes['abstain']:.1%}\n"
            f"novelty-flagged test samples {self.novelty_flagged_test:.1%}"
        )


class SecondOpinion:
    """Calibrated classifier + conformal sets + novelty detection."""

    def __init__(self, alpha: float = DEFAULT_ALPHA, seed: int = DEFAULT_SEED):
        self.alpha = alpha
        self.seed = seed
        self.data: Dataset | None = None
        self.scaler: StandardScaler | None = None
        self.base: LinearAdditive | None = None
        self.calibrated: CalibratedClassifierCV | None = None
        self.novelty: NoveltyDetector | None = None
        self.qhat: float | None = None
        self.metrics: Metrics | None = None
        self.X_test: np.ndarray | None = None
        self.y_test: np.ndarray | None = None

    # ── fitting ────────────────────────────────────────────────────────────────

    def fit(self, data: Dataset | None = None) -> "SecondOpinion":
        self.data = data or load()
        X, y = self.data.X, self.data.y

        X_fit, X_test, y_fit, y_test = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=self.seed
        )
        # 30% of the fitting half, not 50%: measured over 25 seeds, a larger
        # calibration split buys no extra coverage here and costs training data
        # the isotonic step then has to work around.
        X_train, X_cal, y_train, y_cal = train_test_split(
            X_fit, y_fit, test_size=0.30, stratify=y_fit, random_state=self.seed
        )

        self.scaler = StandardScaler().fit(X_train)
        self.base = LinearAdditive(random_state=self.seed).fit(X_train, y_train)

        # FrozenEstimator, not cv="prefit" — the latter was removed in
        # scikit-learn 1.8. Both mean "calibrate this already-fitted model
        # without refitting it".
        self.calibrated = CalibratedClassifierCV(
            FrozenEstimator(self.base), method="isotonic"
        ).fit(X_cal, y_cal)

        p_cal = self.calibrated.predict_proba(X_cal)
        self.qhat = conformal.calibrate(p_cal, y_cal, alpha=self.alpha)

        self.novelty = NoveltyDetector().fit(self.scaler.transform(X_train), y_train)
        self.X_test, self.y_test = X_test, y_test

        self.metrics = self._evaluate(X_train, X_cal, X_test, y_test)
        return self

    def _evaluate(self, X_train, X_cal, X_test, y_test) -> Metrics:
        raw = self.base.predict_proba(X_test)[:, 1]
        p_test = self.calibrated.predict_proba(X_test)
        flagged = (
            self.novelty.score(self.scaler.transform(X_test)) > self.novelty.threshold
        ).mean()

        return Metrics(
            n_train=len(X_train),
            n_calibration=len(X_cal),
            n_test=len(X_test),
            prevalence=float(self.data.y.mean()),
            auc_uncalibrated=float(roc_auc_score(y_test, raw)),
            auc_calibrated=float(roc_auc_score(y_test, p_test[:, 1])),
            brier_uncalibrated=float(brier_score_loss(y_test, raw)),
            brier_calibrated=float(brier_score_loss(y_test, p_test[:, 1])),
            ece_uncalibrated=expected_calibration_error(raw, y_test),
            ece_calibrated=expected_calibration_error(p_test[:, 1], y_test),
            coverage=conformal.coverage(p_test, y_test, self.qhat),
            target_coverage=1 - self.alpha,
            set_sizes=conformal.set_size_distribution(p_test, self.qhat),
            novelty_flagged_test=float(flagged),
        )

    # ── predicting ─────────────────────────────────────────────────────────────

    def assess(self, x: np.ndarray) -> Assessment:
        """Full three-part assessment of one sample."""
        if self.calibrated is None:
            raise RuntimeError("fit() must be called first")
        x = np.asarray(x, dtype=float).reshape(1, -1)
        if x.shape[1] != len(self.data.feature_names):
            raise ValueError(
                f"expected {len(self.data.feature_names)} features, got {x.shape[1]}"
            )

        p = float(self.calibrated.predict_proba(x)[0, 1])
        return Assessment(
            probability_malignant=p,
            prediction_set=conformal.predict_set(p, self.qhat, self.alpha),
            novelty=self.novelty.verdict(self.scaler.transform(x)),
            contributions=self.contributions(x),
        )

    def feature_contributions(self, x: np.ndarray) -> np.ndarray:
        """Additive term per feature. Plus intercept, these equal the logit."""
        x = np.asarray(x, dtype=float).reshape(1, -1)
        return self.base.feature_contributions(x)[0]

    def contributions(self, x: np.ndarray, top: int = 6) -> list[tuple[str, float]]:
        """Which features pushed this sample toward malignant, and how hard.

        For an additive GAM this is exact, not an approximation: the
        contribution of feature i is the spline value f_i(x_i), and those
        terms plus the intercept sum to the logit. A SHAP explainer over the
        same model would compute a slower number and invite the reader to
        believe it was model-agnostic magic.

        Reported from the UNCALIBRATED base model, because that is the thing
        with an additive expansion. Isotonic calibration is a monotone
        reshaping of the output and so cannot change the sign of these terms.
        """
        terms = list(zip(self.data.feature_names, self.feature_contributions(x)))
        terms.sort(key=lambda t: abs(t[1]), reverse=True)
        return terms[:top]
