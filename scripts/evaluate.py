"""Honest evaluation: many splits, not one.

A single train/test split of a 569-sample dataset is a noisy estimate of
anything. Conformal coverage on one split of this data ranges from about 86%
to 100% depending only on the random seed — so quoting the number from
whichever seed you happened to run is not a result, it is a coin flip you
reported as a finding.

Every figure in the README comes from this script. Run it to regenerate them:

    python scripts/evaluate.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from second_opinion import SecondOpinion  # noqa: E402
from second_opinion.data import CITATION, load  # noqa: E402
from second_opinion.references import REFERENCES_BIBLIOGRAPHY  # noqa: E402

N_SEEDS = 40


def main() -> int:
    data = load()
    print(f"Dataset: {len(data)} samples, {data.X.shape[1]} features, "
          f"{data.prevalence:.1%} malignant")
    print(f"Source:  {CITATION}\n")

    rows = []
    for seed in range(N_SEEDS):
        m = SecondOpinion(seed=seed).fit(data).metrics
        rows.append([
            m.auc_uncalibrated, m.auc_calibrated,
            m.brier_uncalibrated, m.brier_calibrated,
            m.ece_uncalibrated, m.ece_calibrated,
            m.coverage, m.set_sizes["decisive"], m.set_sizes["ambiguous"],
            m.set_sizes["abstain"], m.novelty_flagged_test,
        ])
    arr = np.array(rows)
    mean, sd = arr.mean(axis=0), arr.std(axis=0)
    names = [
        "AUC (uncalibrated)", "AUC (calibrated)",
        "Brier (uncalibrated)", "Brier (calibrated)",
        "ECE (uncalibrated)", "ECE (calibrated)",
        "Conformal coverage", "  decisive (1 label)", "  ambiguous (2 labels)",
        "  abstained (0 labels)", "Novelty-flagged (test)",
    ]
    print(f"Across {N_SEEDS} random splits, mean +/- sd:\n")
    for name, mu, s in zip(names, mean, sd):
        print(f"  {name:24} {mu:.4f} +/- {s:.4f}")

    print(f"\nTarget coverage was {1 - SecondOpinion().alpha:.0%}. "
          "Split conformal is conservative by construction, so running above "
          "target is expected and is the safe direction.")
    print(f"Coverage range over seeds: {arr[:, 6].min():.1%} to {arr[:, 6].max():.1%} "
          "— which is why no single split is quoted as the result.")

    # The novelty detector has to be shown to actually detect something, or it
    # is decoration. Scrambled inputs are physically impossible cell nuclei.
    model = SecondOpinion().fit(data)
    scaler = model.base.named_steps["standardscaler"]
    rng = np.random.default_rng(0)
    junk = data.X[rng.integers(0, len(data), 200)] * rng.uniform(2.5, 4.0, (200, data.X.shape[1]))
    flagged = (model.novelty.score(scaler.transform(junk)) > model.novelty.threshold).mean()
    print(f"\nNovelty check: {flagged:.0%} of 200 scrambled inputs flagged "
          f"(vs {mean[10]:.1%} of genuine held-out samples).")

    print("\nReferences:")
    for ref in REFERENCES_BIBLIOGRAPHY:
        print(f"  {ref}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
