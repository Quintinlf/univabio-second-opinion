# Second Opinion

**A diagnostic model that reports when it should not be trusted.**

UnivaBio 2026 — *Artificial Intelligence for Human Health*

---

## The problem

Ask almost any health-AI demo for a prediction and it will give you one. Feed
it a patient unlike anyone in its training data, a mistyped measurement, or a
scan from a different machine, and it will still give you one — with the same
confident percentage attached. The number carries no signal about whether the
model had any business producing it.

That failure mode is not hypothetical. It is the central reason clinical ML
struggles to deploy: a model validated at one hospital degrades at the next,
and nothing in its output says so.

## What this does

Second Opinion answers **three separate questions** about every sample, and
withholds the diagnosis when the last two say it should.

| | Question | Method | Fails when |
|---|---|---|---|
| 1 | What is the probability? | Isotonic-calibrated logistic regression | Always answers, even for nonsense |
| 2 | Is that probability trustworthy *for this sample*? | Split conformal prediction (LAC) | Silently voids under distribution shift |
| 3 | Is this sample even the kind of thing I've seen? | Mahalanobis novelty detection | Says nothing about the diagnosis |

Each is unreliable alone. Reporting all three — and gating the headline number
on the last two — is the product.

The model has four possible outputs, not two:

```
{malignant}            a decisive call
{benign}
{malignant, benign}    genuinely borderline — a human should look
{}                     abstain: nothing is plausible; do not read the percentage
```

**That empty set is the point of the project.** A model that cannot say
"I don't know" will hand a confident answer to a case it has never seen.

## Results

Wisconsin Diagnostic Breast Cancer, 569 samples, 30 features, 37.3% malignant.
Three-way split: train / calibration / test, where the calibration split is
never trained on and the test split is touched only for these numbers.

Across **40 random splits**, mean ± sd:

| Metric | Value |
|---|---|
| AUC (uncalibrated → calibrated) | 0.994 → 0.985 ± 0.012 |
| Brier (uncalibrated → calibrated) | 0.022 → 0.025 ± 0.011 |
| ECE (uncalibrated → calibrated) | 0.032 → 0.026 ± 0.013 |
| **Conformal coverage** (target 90%) | **93.5% ± 3.5%** |
| — decisive, one label | 95.1% |
| — abstained, no label | 5.0% |
| Novelty-flagged, genuine held-out samples | 6.8% |
| Novelty-flagged, deliberately scrambled inputs | **100%** |

Coverage on any *single* split ranges from 86% to 99%. The headline figures
are averaged over 40 splits rather than taken from whichever split flattered
the model — on a 569-sample dataset one split is a coin flip, not a finding.
Regenerate everything above with:

```bash
python scripts/evaluate.py
```

**On the calibration trade:** isotonic calibration is a monotone reshape of
the model's probability. Ranking matters less here than whether the probability
means what it says — the point of Guo et al. (2017): accuracy and calibration
are different quantities. Compare the uncalibrated and calibrated columns in
`evaluate.py`.

## Try it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Pick **Implausible geometry** in the cell-profile dropdown, or open
**Fine-tune the measurements** and push a slider past the dataset range, and
watch the diagnosis disappear. The two withholds are not the same sentence: a
conformal abstain names the measurement change that would make the set
decisive; a novelty withhold names the measurement that looks like a typo.

Every method is defined in plain English in the app's own glossary, including
what each one *cannot* tell you on its own.

A subtler demo: the implausible-geometry preset sets *mean area* to its
true dataset maximum while leaving *mean radius* and *mean perimeter* at
their medians. Every value is individually in range, but the combination is
geometrically impossible — a nucleus cannot have that area at that radius.
The novelty detector flags it because Mahalanobis distance accounts for how
features covary. A per-feature range check would pass it without comment.

## Layout

```
second_opinion/
  data.py           dataset + provenance; 10-slider → 30-feature rebuild
  diagnostics.py    chart data, metric guides, honest probability formatting
  model.py          the three-part assessment, and the train/cal/test split
  gam.py            the linear base model, and the spline GAM it was measured against
  conformal.py      split conformal prediction (LAC), coverage guarantee
  ood.py            class-conditional Mahalanobis novelty detection
  counterfactual.py computed next step for each withhold
  presets.py        real-sample cell profiles for the UI
  shift.py          noise / lab-shift stress test
  references.py     bibliography for all load-bearing methods
scripts/
  evaluate.py       40-split numbers + shift table (writes split_results.json)
  capture_cue_cards.py  screenshots the running app for the demo video
tests/              claim-level tests, one per claim made here or in the UI
app.py              Streamlit prototype
```

```bash
python -m pytest tests -q      # 32 claim-level tests
```

## Reading the numbers

Every metric in the app ships with its ideal value and a plain-language read
(`good` / `fair` / `poor`), because a number nobody can judge is decoration.

| Metric | Ideal | What it measures |
|---|---|---|
| AUC | 1.000 (0.5 = coin flip) | Ranking: does it put malignant above benign? Says nothing about the percentages. |
| Brier | 0.000, lower better | Squared error of the probability itself. Punishes confident-and-wrong hardest. |
| ECE | 0.000, lower better | When it says 70%, how far from 70% is the truth? **The number this project rests on.** |
| Conformal coverage | at or above the 90% target | How often the true label is inside the prediction set. |

Two things that are easy to misread, and are now explained in the UI:

- **A reported probability is never 0% or 100%.** Isotonic calibration returns
  the frequency observed in a calibration bin, so a pure bin returns exactly
  1.0. Printed as "100%" that claims certainty a 128-sample calibration split
  cannot buy, so the reported value is clipped to 1/(2n) off each end.
- **Confident error sitting flat at zero is the good outcome**, not a missing
  line. It counts predictions that were decisive, in-distribution, and wrong.

## Honest limitations

- **This is not a medical device.** It cannot diagnose anyone, has never been
  clinically validated, and must not inform a real medical decision.
- The dataset is 569 samples from one institution, collected in the early
  1990s. Nothing here has been shown to transfer anywhere else.
- It classifies **pre-extracted nuclear measurements**, not images. The hard
  clinical problem — getting from a slide to those 30 numbers — is assumed
  already solved (Street et al., 1993: digitised FNA → snake segmentation →
  ten nuclear features → mean/SE/worst). Raw slide images are not bundled
  with WDBC, so upstream denoising or segmentation is out of scope here.
- The base model is plain logistic regression on standardised features, not
  a Gaussian process and not a black-box booster. The contribution chart is
  the model's own arithmetic — coefficient x standardised value, summing
  exactly to the logit — not SHAP.
- **A per-feature spline GAM was tried and reverted.** Controlled A/B over 25
  seeds, identical pipeline: linear won on all eight metrics (ECE 0.031 vs
  0.111, Brier 0.022 vs 0.049, coverage 93.5% vs 91.4%). At n=569 the extra
  flexibility is not supported by the data and calibration pays for it —
  which is the one property this project rests on. The GAM is kept in
  `gam.py` as the record of what was tried; re-run `evaluate.py` to check
  the claim rather than trusting it.
- The conformal guarantee is *marginal*, not conditional (Angelopoulos & Bates,
  2023): it holds across the population, not necessarily within every subgroup.
  Subgroup-conditional coverage is the obvious next step and is not implemented.

## References

Each method traces to a published source. Full strings are in
`second_opinion/references.py`.

| Paper | Role here |
|---|---|
| Street, Wolberg & Mangasarian (1993) | WDBC dataset; upstream snake-segmentation feature extraction |
| Guo et al. (2017) | Why we report AUC *and* Brier/ECE — calibration ≠ accuracy |
| Hastie & Tibshirani (1986) | The additive-model formulation in `gam.py`, measured and not adopted |
| Lee et al. (2018) | Class-conditional Mahalanobis OOD score in `ood.py` |
| Sadinle, Lei & Wasserman (2019) | LAC/THR set-valued classifier in `conformal.py` |
| Angelopoulos & Bates (2023) | Split conformal prediction, finite-sample correction, marginal coverage |

## Data

Street, W.N., Wolberg, W.H., & Mangasarian, O.L. (1993). *Nuclear feature
extraction for breast tumor diagnosis.* IS&T/SPIE International Symposium on
Electronic Imaging, 1905, 861–870. UCI Machine Learning Repository (WDBC),
distributed with scikit-learn.
