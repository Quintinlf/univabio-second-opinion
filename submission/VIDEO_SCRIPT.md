# Second Opinion — demo video script (v2)

Rewritten after the 30 Sep walkthrough notes. Every jargon term you flagged is
now explained **on screen**, so the narration can point at an explanation
instead of apologising for one.

**Target 3:00.** UnivaBio sets no length limit; 3 minutes is the safe length
for a judge watching dozens. Rows marked `[TRIM]` can be cut for a 2:00 version
that still touches all five judging criteria.

---

## Before you record

```bash
python -m pytest tests -q          # expect: 32 passed
python scripts/evaluate.py         # regenerates the 40-split charts
streamlit run app.py
```

Let the first load finish — it fits the model and builds the presets once
(**15–20s**, with a `Running get_presets()` spinner). After that every preset
switch is instant. Record at 1080p with the browser maximised.

**Cue cards** are in `submission/cue_cards/`. Each section below names the one
that should be on screen while you read it. They were captured from this build,
so if your screen disagrees with a cue card, the script is wrong, not your app.

> Regenerate them any time with:
> ```bash
> python scripts/capture_cue_cards.py --port 8501
> ```

---

## 1 · The problem — 0:00–0:22
**Cue card:** `01_hero.png`

Open on the title and the warning banner. Hold one beat before speaking.

> Ask almost any health-AI demo for a prediction and it will give you one.
> Feed it a patient unlike anyone in its training data, or a mistyped
> measurement, and it still gives you one — with the same confident
> percentage attached. The number carries no signal about whether the model
> had any business producing it. That is a real reason clinical machine
> learning struggles to deploy: a model validated at one hospital degrades
> at the next, and nothing in its output says so.

---

## 2 · Three questions, not one — 0:22–0:45
**Cue card:** `02_glossary.png`

Open the **Plain-English glossary**. Scroll it slowly while you speak — you do
not need to read it aloud, the judge just needs to see it exists.

> Second Opinion answers three separate questions about every sample. What is
> the probability — a calibrated logistic regression. Is that probability
> trustworthy *for this sample* — conformal prediction. And is this sample
> even the kind of thing the model has seen — Mahalanobis novelty detection.
>
> Every one of those terms is defined in the app itself, including what each
> method cannot do on its own. Because each one is unreliable alone. Reporting
> all three, and withholding the diagnosis when the last two say no, is the
> project.

---

## 3 · The normal case — 0:45–1:05
**Cue card:** `04_malignant_decisive.png`

Select **Typical malignant nucleus**. Let it land, then point at the three
metrics.

> On a sample that looks like its training data, it behaves like any good
> classifier: a decisive call at ninety-nine-point-six percent, in
> distribution, with the exact reason beside it. Those bars are the model's
> own coefficients times this sample's standardised measurements. Not an
> approximation of the model — the model.
>
> And notice it does not say one hundred percent. The calibration set has a
> hundred and twenty-eight samples; a hundred percent is a claim that sample
> size cannot buy.

**[TRIM]** — cut from "And notice" to reach 2:00.

---

## 4 · It abstains, and says what would change its mind — 1:05–1:40
**Cue card:** `05_abstain_with_next_step.png`

Select **Borderline (withhold)**. Zoom the "Why" line and the "What would make
this decisive" line.

> Here is a real patient from the dataset where it refuses. At the ninety
> percent confidence level neither label is plausible, so the prediction set
> is empty and the headline is an abstention, not a number.
>
> And it does not stop at "I don't know." It computes the smallest single
> measurement change that would make the call decisive: if mean concave
> points rises thirty-two percent, this becomes a confident malignant. That
> is something a clinician can act on — re-measure that one feature.

> **Read the numbers off your own screen.** Verified on this build: conformal
> **abstain**, novelty **0.27x**, next step *"if mean concave points rises by
> 32% (0.0533 → 0.0704)"*. Exact figures shift slightly with your
> scikit-learn build.

---

## 5 · It refuses nonsense it is maximally confident about — 1:40–2:15
**Cue card:** `06_out_of_distribution.png`

Select **Implausible geometry**. **Frame it so the 99.61% probability and the
red refusal are visible together** — that contrast is the whole video.

> This is the case the project exists for. Mean area is at the dataset
> maximum while radius and perimeter sit at their medians. Every value is
> individually in range — a per-feature bounds check passes it without
> comment — but the combination is geometrically impossible. A nucleus
> cannot have that area at that radius.
>
> The underlying classifier is ninety-nine-point-six percent confident this
> is malignant. Look at question one. And the system refuses to report it:
> fifty-two times beyond the distance containing ninety-nine percent of the
> training data. It names the measurement that looks like a typo, and offers
> no diagnosis on the far side of that edit.
>
> That percentage is exactly what a conventional demo would have shown you,
> with nothing attached to tell you not to believe it.

---

## 6 · Why believe any of it — 2:15–2:45
**Cue cards:** `07_calibration_charts.png` → `08_metrics_with_bands.png` →
`09_forty_splits.png`

Scroll through the three in order. Roughly ten seconds each.

> These are not numbers you have to take on faith. The calibration curve
> shows what the model promised against what actually happened — the dashed
> line is a perfectly honest model, and we sit on it. Beside it is what
> calibration is physically doing to every held-out sample.
>
> Every metric ships with what a good value looks like, so you are not left
> guessing whether point-zero-three is good.
>
> And because coverage on a single split of a 569-sample dataset ranges from
> eighty-six to ninety-nine percent, nothing here is quoted from one split.
> That is the distribution over forty random splits, against the ninety
> percent target. The mean is ninety-three-and-a-half.

---

## 7 · How it fails — 2:45–3:00
**Cue card:** `10_shift_charts.png`

Open **What happens under measurement noise**.

> Under progressive corruption, the withheld share climbs to one hundred
> percent — while *confident and wrong* stays flat on the floor. It fails by
> going quiet, not by guessing.
>
> To be clear about what is finished: this is a research prototype, not a
> medical device. One institution, early nineties, pre-extracted
> measurements rather than images. The conformal guarantee is marginal, not
> per-subgroup, and that is the next thing to build.
>
> A model that cannot say "I don't know" will hand a confident answer to a
> case it has never seen. This one says it.

---

## Recording notes

- **Never narrate over a spinner.** Switch preset, wait for it to land, speak.
- Sections 3–5 are one continuous move through the dropdown. Keep the three
  metric tiles in frame the whole time — they are the visual spine.
- **Section 5 is the one judges remember.** If you rehearse one section,
  rehearse that one, and check that `99.61%` and the red panel are legible in
  the same frame.
- Say "research prototype" before the halfway mark. The banner is already on
  screen in section 1.
- Do not claim accuracy numbers beyond section 6, and never call any output a
  diagnosis.
- The cue cards were captured in light theme. If you record in dark theme the
  layout is identical — only the colours differ.
