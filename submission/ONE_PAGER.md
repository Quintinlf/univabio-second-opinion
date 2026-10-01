# Second Opinion
### A diagnostic model that reports when it should not be trusted
UnivaBio 2026 — *Artificial Intelligence for Human Health* · Research prototype, not a medical device

---

**The problem.** Ask almost any health-AI demo for a prediction and it will give you one. Feed it a
patient unlike anyone in its training data, a mistyped measurement, or a sample from a different lab,
and it still gives you one — with the same confident percentage attached. The number carries no signal
about whether the model had any business producing it. This is a central reason clinical ML struggles
to deploy: a model validated at one hospital degrades at the next, and nothing in its output says so.

**The approach.** Second Opinion answers three separate questions about every sample, and withholds
the diagnosis when the last two say it should.

| | Question | Method | Fails when |
|---|---|---|---|
| 1 | What is the probability? | Isotonic-calibrated logistic regression | Always answers, even for nonsense |
| 2 | Is that probability trustworthy *for this sample*? | Split conformal prediction (LAC) | Silently voids under distribution shift |
| 3 | Is this sample the kind of thing I've seen? | Class-conditional Mahalanobis novelty detection | Says nothing about the diagnosis |

Each is unreliable alone. Reporting all three — and gating the headline number on the last two — is the
product. The model therefore has **four** possible outputs, not two: `{malignant}`, `{benign}`,
`{malignant, benign}` (genuinely borderline — a human should look), and `{}` — **abstain: do not read
the percentage.** That empty set is the point of the project.

**Key features.**
- **Diagnosis that disappears.** When novelty or conformal analysis says no, the headline is a refusal,
  not a number. The probability stays visible, explicitly labelled as transparency rather than a result.
- **Two withholds, two different next steps — computed, not canned.** A conformal abstain names the
  smallest single measurement change that would make the call decisive ("if mean concave points drops
  37%, this becomes a 90%-confident benign"). A novelty withhold instead names the measurement that
  looks like a typo and refuses to speculate past it.
- **Catches impossible combinations, not just out-of-range values.** The demo preset sets mean area to
  its true dataset maximum while radius and perimeter stay at their medians: every value is individually
  in range, but the geometry is impossible. Mahalanobis distance flags it because it accounts for how
  features covary; a per-feature bounds check would pass it silently. The base classifier is **100%
  confident** on this input — and the system reports no diagnosis.
- **Exact attribution.** The contribution chart is the model's own arithmetic — coefficient × standardised
  value, summing exactly to the logit — not a SHAP approximation.
- **A stress test in the UI.** Held-out samples are progressively corrupted; withhold rate rises to 100%
  while *confident error* (decisive, in-distribution, and wrong) does not rise with it. It fails by going
  quiet rather than by guessing.

**Results.** Wisconsin Diagnostic Breast Cancer (569 samples, 30 features, 37.3% malignant), three-way
train/calibration/test split. Averaged over **40 random splits**, mean ± sd: AUC 0.985 ± 0.012, Brier
0.025, ECE 0.026 (improved from 0.032 uncalibrated); **conformal coverage 93.5% ± 3.5%** against a 90%
target; decisive on 95.1% of cases, abstaining on 5.0%. Novelty detection flags **100%** of deliberately
scrambled inputs versus 6.8% of genuine held-out samples. Single-split coverage ranges from 86% to 99%,
which is why no single split is quoted. Every figure regenerates with `python scripts/evaluate.py`;
32 claim-level tests assert each claim made in the README and the UI.

**Who it helps.** Clinicians and triage workflows, who get a refusal they can act on instead of a
confident number they cannot audit. Lab and QC staff, for whom a flagged measurement is a transcription
check. Deployment and validation teams, who get a built-in monitor for the silent degradation that
breaks models at a second site. Patients, who are not assigned a number produced by extrapolation.

**Honest limitations.** Not a medical device; never clinically validated; must not inform a real medical
decision. 569 samples from one institution collected in the early 1990s, with no demonstrated transfer.
It classifies **pre-extracted nuclear measurements, not images** — the upstream slide-to-features problem
is assumed solved. The conformal guarantee is *marginal*, not conditional (Angelopoulos & Bates, 2023):
subgroup-conditional coverage is the obvious next step and **is not implemented**. A per-feature spline
GAM was tried and reverted after a 25-seed controlled A/B — linear won on all eight metrics (ECE 0.031
vs 0.111) — and is kept in `gam.py` as the record of what was tried.

**Methods trace to published sources:** Street, Wolberg & Mangasarian (1993) · Guo et al. (2017) ·
Hastie & Tibshirani (1986) · Lee et al. (2018) · Sadinle, Lei & Wasserman (2019) · Angelopoulos & Bates (2023).
Full strings in `second_opinion/references.py`.

**Run it:** `pip install -r requirements.txt && streamlit run app.py` — then pick *Implausible geometry*
in the cell-profile dropdown and watch the diagnosis disappear.
