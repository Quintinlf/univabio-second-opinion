"""The dataset, and where it came from.

Provenance is kept in code rather than in a README line, because every number
this project reports has to be traceable to a real, citable source. Nothing
here is synthetic and nothing is generated.

Source
------
Wisconsin Diagnostic Breast Cancer (WDBC). Street, Wolberg & Mangasarian
(1993), "Nuclear feature extraction for breast tumor diagnosis". Distributed
with scikit-learn as `load_breast_cancer`, originally from the UCI Machine
Learning Repository.

Upstream pipeline (Street et al., 1993)
---------------------------------------
This repo starts *after* image processing. The original system digitised
fine-needle-aspirate slides, segmented individual nuclei with active-contour
models ("snakes"), and computed ten geometric/textural measurements per
nucleus. The 30 columns here are the mean, standard error, and "worst"
(mean of the three largest) of those ten features across all nuclei in each
sample. Raw slide images are not bundled with WDBC — only these extracted
numbers — so any upstream denoising or segmentation is assumed already done.

569 fine-needle-aspirate samples, 30 features: radius, texture, perimeter,
area, smoothness, compactness, concavity, concave points, symmetry, fractal
dimension (each as mean, SE, and worst).

A note on the label direction
-----------------------------
scikit-learn ships this with `target == 0` meaning *malignant*, so a naive
`predict_proba(X)[:, 1]` is the probability of BENIGN. That is the wrong way
round for a screening tool: the positive class should be the thing you are
trying not to miss. This module flips it once, here, so that everything
downstream reads `1 == malignant` and a "high probability" means "concerning".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_breast_cancer

from second_opinion.references import REFERENCES

CITATION = REFERENCES["street1993"]

MALIGNANT = 1
BENIGN = 0
LABELS = {MALIGNANT: "malignant", BENIGN: "benign"}


@dataclass(frozen=True)
class Dataset:
    X: np.ndarray
    y: np.ndarray               # 1 = malignant, 0 = benign
    feature_names: list[str]
    citation: str = CITATION

    def __len__(self) -> int:
        return len(self.y)

    @property
    def prevalence(self) -> float:
        """Share of samples that are malignant.

        The base rate a model has to beat. Quoting accuracy without it is how
        a model that always says "benign" looks 63% correct.
        """
        return float(self.y.mean())


def load() -> Dataset:
    raw = load_breast_cancer()
    return Dataset(
        X=raw.data,
        # Flip: sklearn encodes malignant as 0. See module docstring.
        y=(raw.target == 0).astype(int),
        feature_names=list(raw.feature_names),
    )


def feature_reference(data: Dataset | None = None) -> dict[str, dict[str, float]]:
    """Per-feature min/median/max over the whole dataset.

    The UI needs real ranges to build sliders from. Hardcoding plausible-looking
    bounds would silently let a user dial in a physically impossible cell
    nucleus and get a confident answer back.
    """
    data = data or load()
    return {
        name: {
            "min": float(np.min(col)),
            "median": float(np.median(col)),
            "max": float(np.max(col)),
        }
        for name, col in zip(data.feature_names, data.X.T)
    }
