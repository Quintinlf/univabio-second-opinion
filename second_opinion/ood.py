"""Is this sample anything like what the model was trained on?

Conformal prediction guarantees coverage over the *same distribution* the
calibration set came from. Feed it something from a different distribution —
a different scanner, a different lab's staining protocol, a typo in a field —
and the guarantee quietly stops applying. Nothing in the probability warns
you: a model will return 0.97 for an input that is physically impossible.

So the distance is measured directly. Following Lee et al. (2018), we fit
class-conditional Gaussians on the scaled training features and score each
sample by its minimum Mahalanobis distance to either class centroid. This is
the tabular analogue of Lee's Mahalanobis OOD detector (which uses deep layer
features); the score and threshold logic are the same.

Mahalanobis distance accounts for how features covary — radius, perimeter,
and area of a nucleus move together, and treating them as independent would
wildly understate the distance of a sample that breaks that correlation.

Measured on WDBC: with the threshold at the 99th percentile of training
distances, about 7% of held-out test samples are flagged (the test set is drawn
from the same distribution, so this is above the nominal 1% but still well
below the scrambled-input rate), while 100% of deliberately scrambled inputs
are flagged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.covariance import EmpiricalCovariance

from second_opinion.data import BENIGN, MALIGNANT


@dataclass(frozen=True)
class NoveltyVerdict:
    distance: float
    threshold: float

    @property
    def is_outlier(self) -> bool:
        return self.distance > self.threshold

    @property
    def ratio(self) -> float:
        """Distance as a multiple of the threshold. 1.0 sits exactly on it."""
        return self.distance / self.threshold if self.threshold else float("inf")

    def explain(self) -> str:
        if not self.is_outlier:
            return (
                "This sample sits inside the range of the training data, so the "
                "confidence level below is meaningful."
            )
        return (
            f"I am withholding a prediction because this sample is "
            f"{self.ratio:.1f}x beyond the distance that contains 99% of "
            "training data. This does not look like the breast-tissue "
            "measurements the model was calibrated on. The percentage below "
            "is extrapolation, not evidence."
        )


class NoveltyDetector:
    """Class-conditional Mahalanobis novelty detector (Lee et al., 2018)."""

    def __init__(self, percentile: float = 99.0):
        if not 0 < percentile < 100:
            raise ValueError(f"percentile must be in (0, 100), got {percentile}")
        self.percentile = percentile
        self._class_covs: dict[int, EmpiricalCovariance] = {}
        self._threshold: float | None = None

    def fit(
        self, X_scaled: np.ndarray, y: np.ndarray
    ) -> "NoveltyDetector":
        """Fit class-conditional Gaussians on scaled training features.

        Lee et al. fit separate covariances per class; the OOD score is the
        minimum Mahalanobis distance across classes. Fitting on raw features
        would let a unit change (millimetres to centimetres) move the threshold
        without anything else changing.
        """
        X_scaled = np.asarray(X_scaled, dtype=float)
        y = np.asarray(y, dtype=int)
        self._class_covs = {}
        for label in (BENIGN, MALIGNANT):
            mask = y == label
            if not mask.any():
                raise ValueError(f"no training samples for label {label}")
            self._class_covs[label] = EmpiricalCovariance().fit(X_scaled[mask])

        train_scores = self.score(X_scaled)
        self._threshold = float(np.percentile(train_scores, self.percentile))
        return self

    @property
    def threshold(self) -> float:
        if self._threshold is None:
            raise RuntimeError("NoveltyDetector.fit must be called first")
        return self._threshold

    def score(self, X_scaled: np.ndarray) -> np.ndarray:
        """Minimum class-conditional Mahalanobis distance per sample."""
        if not self._class_covs:
            raise RuntimeError("NoveltyDetector.fit must be called first")
        X_scaled = np.atleast_2d(np.asarray(X_scaled, dtype=float))
        distances = np.column_stack(
            [self._class_covs[label].mahalanobis(X_scaled) for label in (BENIGN, MALIGNANT)]
        )
        return distances.min(axis=1)

    def verdict(self, x_scaled: np.ndarray) -> NoveltyVerdict:
        return NoveltyVerdict(
            distance=float(self.score(x_scaled)[0]), threshold=self.threshold
        )
