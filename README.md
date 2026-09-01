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
| ECE (uncalibrated → calibrated) | see `evaluate.py` |
| **Conformal coverage** (target 90%) | **93.5% ± 3.5%** |
| — decisive, one label | 95.0% |
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

**On the calibration trade:** isotonic calibration costs ~0.009 AUC and buys a
better Brier score, lower ECE, and higher conformal coverage than Platt scaling
(measured: 93.5% vs 91.2% coverage, 4.9% vs 7.8% abstention). Ranking matters
less here than whether the probability means what it says — the point of Guo
et al. (2017): accuracy and calibration are different quantities.

## Try it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then **push a slider past the dataset range** and watch the diagnosis
disappear and be replaced by a refusal.

A subtler demo: set *mean area* to its true dataset maximum while leaving
*mean radius* and *mean perimeter* at their medians. Every value is
individually in range, but the combination is geometrically impossible — a
nucleus cannot have that area at that radius. The novelty detector flags it at
26× threshold, because Mahalanobis distance accounts for how features covary.
A per-feature range check would pass it without comment.

## Layout

```
second_opinion/
  data.py        dataset + provenance; flips sklearn's label direction
  model.py       the three-part assessment, and the train/cal/test split
  conformal.py   split conformal prediction (LAC), coverage guarantee
  ood.py         class-conditional Mahalanobis novelty detection
  references.py  bibliography for all load-bearing methods
scripts/
  evaluate.py    the 40-split evaluation behind every number above
tests/           18 tests, one per claim made here or in the UI
app.py           Streamlit prototype
```

```bash
python -m pytest tests -q
```

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
- Logistic regression was chosen over a stronger model deliberately: at
  n=569 a gradient-boosted ensemble buys little AUC and makes the feature
  attributions approximate instead of exact. The contribution chart in the UI
  is the model's actual arithmetic, not a post-hoc explanation of it.
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
| Lee et al. (2018) | Class-conditional Mahalanobis OOD score in `ood.py` |
| Sadinle, Lei & Wasserman (2019) | LAC/THR set-valued classifier in `conformal.py` |
| Angelopoulos & Bates (2023) | Split conformal prediction, finite-sample correction, marginal coverage |

## Data

Street, W.N., Wolberg, W.H., & Mangasarian, O.L. (1993). *Nuclear feature
extraction for breast tumor diagnosis.* IS&T/SPIE International Symposium on
Electronic Imaging, 1905, 861–870. UCI Machine Learning Repository (WDBC),
distributed with scikit-learn.
