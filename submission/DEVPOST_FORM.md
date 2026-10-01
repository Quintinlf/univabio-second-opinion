# Devpost project page — ready-to-paste copy

Devpost's project form has standard fields. Paste these in. Keep the
not-a-medical-device line in the first screenful; judges read the top.

---

## Tagline (one line)

> A diagnostic model that reports when it should not be trusted — it withholds
> the diagnosis instead of guessing.

---

## Inspiration

Ask almost any health-AI demo for a prediction and it will give you one. Feed
it a patient unlike anyone in its training data, a mistyped measurement, or a
sample from a different lab, and it still gives you one — with the same
confident percentage attached. The number carries no signal about whether the
model had any business producing it.

That failure mode is not hypothetical. It is a central reason clinical ML
struggles to deploy: a model validated at one hospital degrades at the next,
and nothing in its output says so. We wanted to build the opposite — a model
whose headline output can be *silence*.

## What it does

Second Opinion answers **three separate questions** about every sample, and
withholds the diagnosis when the last two say it should:

1. **What is the probability?** — isotonic-calibrated logistic regression.
   Always answers, even for nonsense.
2. **Is that probability trustworthy *for this sample*?** — split conformal
   prediction (LAC), which gives a prediction set with a coverage guarantee.
3. **Is this sample even the kind of thing I've seen?** — class-conditional
   Mahalanobis novelty detection.

Each is unreliable alone. Reporting all three — and gating the headline number
on the last two — is the product. So the model has **four** possible outputs,
not two: `{malignant}`, `{benign}`, `{malignant, benign}` (genuinely
borderline, a human should look), and `{}` — **abstain**.

That empty set is the point of the project.

When it withholds, it does not stop at "I don't know". The two withholds get
two different, *computed* next steps:

- a **conformal abstain** names the smallest single measurement change that
  would make the call decisive — "if mean concave points drops 37%, this
  becomes a 90%-confident benign";
- a **novelty withhold** instead names the measurement that looks like a typo,
  and refuses to offer a diagnosis on the far side of that edit.

The sharpest demo is the **Implausible geometry** preset: *mean area* at the
dataset maximum while *radius* and *perimeter* sit at their medians. Every
value is individually in range — a per-feature bounds check passes it without
comment — but the combination is geometrically impossible. The base classifier
is **100% confident it is malignant**. The system reports **no diagnosis**,
because Mahalanobis distance accounts for how features covary and puts the
sample 52× beyond the radius containing 99% of training data.

## How we built it

Python, scikit-learn, Streamlit. Wisconsin Diagnostic Breast Cancer (569
samples, 30 features), split three ways — train / calibration / test — where
the calibration split is never trained on and the test split is touched only
for reported numbers.

- `conformal.py` — split conformal prediction (LAC / THR), finite-sample
  corrected
- `ood.py` — class-conditional Mahalanobis novelty score
- `counterfactual.py` — the computed next step for each kind of withhold
- `shift.py` — noise and lab-scale stress tests
- `scripts/evaluate.py` — regenerates every number claimed, over 40 splits
- `tests/` — 32 claim-level tests, one per claim made in the README or the UI

Every method traces to a published source (Sadinle et al. 2019; Angelopoulos &
Bates 2023; Lee et al. 2018; Guo et al. 2017; Street et al. 1993), listed in
`second_opinion/references.py`.

## Challenges we ran into

**Calibration and accuracy are different quantities, and we had to pick.**
Isotonic calibration is a monotone reshape of the probability; it cost us a
little AUC (0.994 → 0.985) and bought a better ECE (0.032 → 0.026). For a
system whose entire premise is that the probability means what it says, that
was the right trade — but it is a trade, and we report both columns.

**A more flexible model made things worse.** We tried a per-feature spline GAM,
expecting it to win. A controlled 25-seed A/B over an identical pipeline said
otherwise: linear won on all eight metrics (ECE 0.031 vs 0.111, Brier 0.022 vs
0.049, coverage 93.5% vs 91.4%). At n=569 the extra flexibility is not
supported by the data, and calibration pays for it — the one property this
project rests on. We reverted it and kept `gam.py` as the record of what was
tried.

**One split is a coin flip, not a finding.** Conformal coverage on a single
569-sample split ranges from 86% to 99% depending on the seed. Quoting any one
of those would have been cherry-picking, so every headline number is averaged
over 40 random splits and reported with a standard deviation.

## Accomplishments that we're proud of

- The empty prediction set actually works end to end: **100%** of deliberately
  scrambled inputs are flagged, versus **6.8%** of genuine held-out samples.
- Under progressive corruption, the withhold rate climbs to 100% while
  *confident error* — decisive, in-distribution, and wrong — does **not** climb
  with it. The system fails by going quiet rather than by guessing.
- Conformal coverage of **93.5% ± 3.5%** against a 90% target, decisive on
  95.1% of cases.
- Every claim in the README and the UI has a test asserting it, and every
  number regenerates with one command.
- We reverted a model we had already built because the measurement said to.
- A reviewer walkthrough caught the UI falsely flagging a *textbook malignant
  sample* as out-of-distribution. The cause was real: the feature rebuild
  anchored each sample's error/worst columns on the dataset median, which
  double-counted a large nucleus's size and pushed it off the training
  manifold. Fixed, with a regression test that asserts every preset assesses
  identically to the raw sample it came from.

## What we learned

That the hard part of trustworthy medical AI is not making the model more
accurate — this model is already at 0.985 AUC — but making it *shut up* on the
cases where accuracy is not the operative question. Abstention has to be a
first-class output, designed into the interface, not a confidence threshold
bolted on afterwards.

## What's next for Second Opinion

- **Subgroup-conditional coverage.** The conformal guarantee here is *marginal*
  (Angelopoulos & Bates, 2023): it holds across the population, not necessarily
  within every subgroup. Making it conditional is the obvious next step and is
  not implemented.
- **A second dataset**, to test the claim that this transfers at all.
- **Upstream from the measurements** — the real clinical problem is getting
  from a slide to the 30 numbers, which this prototype assumes already solved.
- **Clinician-in-the-loop evaluation**: does a computed next step actually
  change what a human does with a withheld case?

## Built with

`python` · `scikit-learn` · `streamlit` · `numpy` · `pandas` · `pytest` ·
`conformal-prediction` · `mahalanobis-distance` · `isotonic-calibration`

---

## ⚠️ Required disclaimer — keep this on the project page

> **Research prototype, not a medical device.** Trained on 569 historical
> fine-needle-aspirate measurements from one institution. It cannot diagnose
> anyone, has never been clinically validated, and must not be used for a real
> medical decision.
