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

# The ten mean measurements the UI exposes. Error and worst variants are
# reconstructed from these so the sliders cannot invent a nucleus whose
# mean radius tripled while worst radius stayed put.
BASE_FEATURES = [
    "mean radius", "mean texture", "mean perimeter", "mean area",
    "mean smoothness", "mean compactness", "mean concavity",
    "mean concave points", "mean symmetry", "mean fractal dimension",
]

# Geometric size family: the features a different lab's resolution would scale.
SIZE_FEATURE_TOKENS = ("radius", "perimeter", "area")


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


def base_feature_name(name: str) -> str:
    """Map an error/worst column back to its mean measurement.

    sklearn's WDBC names are `mean radius`, `radius error`, `worst radius`
    — not `mean radius error`. Stripping ' error' alone leaves a token that
    is not a slider name.
    """
    if name.startswith("worst "):
        return "mean " + name[len("worst "):]
    if name.endswith(" error"):
        return "mean " + name[: -len(" error")]
    return name


def slider_bounds(
    name: str, reference: dict[str, dict[str, float]]
) -> tuple[float, float]:
    """Slider range: dataset min/max, padded so a user can trip novelty."""
    ref = reference[name]
    span = ref["max"] - ref["min"]
    return float(ref["min"] - span * 0.5), float(ref["max"] + span * 0.5)


def expand_base_features(
    chosen: dict[str, float],
    *,
    start: dict[str, float] | None = None,
    reference: dict[str, dict[str, float]] | None = None,
    feature_names: list[str] | None = None,
) -> np.ndarray:
    """Rebuild the 30-feature vector from the ten base sliders.

    Error and worst columns scale with how far their mean moved *from the
    starting sample's own mean* — not from the dataset median. Anchoring on
    the median double-counts a real sample's size: a malignant nucleus whose
    worst radius is already large would have it scaled up again by the ratio
    of its mean radius to the median, landing the vector off the training
    manifold and tripping the novelty detector on a textbook sample. With the
    sample's own value as the anchor, loading a preset unchanged reproduces it
    exactly (ratio 1.0) and moving a slider scales from where that sample was.
    """
    if feature_names is None or reference is None:
        data = load()
        feature_names = feature_names or data.feature_names
        reference = reference or feature_reference(data)
    if start is None:
        start = {n: reference[n]["median"] for n in feature_names}

    vector = []
    for name in feature_names:
        if name in chosen:
            vector.append(float(chosen[name]))
            continue
        base = base_feature_name(name)
        anchor = start.get(base) or reference[base]["median"]
        if base in chosen and anchor:
            ratio = chosen[base] / anchor
            vector.append(start.get(name, reference[name]["median"]) * ratio)
        else:
            vector.append(start.get(name, reference[name]["median"]))
    return np.asarray(vector, dtype=float)


def size_feature_mask(feature_names: list[str]) -> np.ndarray:
    """True for radius / perimeter / area columns (mean, error, and worst)."""
    return np.array(
        [any(token in name for token in SIZE_FEATURE_TOKENS) for name in feature_names]
    )
