# Second Opinion — presentation script (v3)

Written after the 1 Oct run-through. That take was you reading the one-pager
aloud and checking it against the repo — useful, but it is not a presentation,
and you said so yourself about four minutes in.

This version is built to be **spoken**, not read. Three rules it follows:

1. **Nothing in it is a term you have to look up mid-sentence.** On the last
   take you stopped on "split conformal prediction" and on "Mahalanobis" and
   lost about ninety seconds. Every line below is sayable cold.
2. **It opens with your lung-versus-breast example**, not with the dataset.
   That was the clearest thing you said in twelve minutes and it is a better
   hook than anything I had written.
3. **It is three minutes.** At nine minutes on the last take you said "let's
   finish this, I'm getting tired" — and it is audible. Short enough to deliver
   with energy beats complete and flat.

---

## ⚠️ Read this before recording — three things you said that are wrong

These are not criticisms of the project. They are things that would cost you
marks if you said them on camera, and all three are easy to fix.

**1. Mahalanobis is not a cell type, a disease, or a kind of cancer.**
You said *"it looks like we're detecting Mahalanobis cells... Mahalanobis
cancer, no idea what that even is."* It is a **way of measuring distance**,
named after the statistician P.C. Mahalanobis. Say it like this:

> "Ordinary distance asks whether one measurement is weird. Mahalanobis
> distance asks whether the *combination* is weird — because in real cells,
> area and radius move together. That is how it catches a sample where every
> single number is in range but the shape is impossible."

That is the whole concept and you now have it in one sentence. The glossary in
the app says the same thing, with a warning label on it.

**2. The out-of-distribution panel is not saying "99.61% sure it's
impossible."** You read it that way at 9:47. It is saying the opposite: the
classifier *wants* to say 99.61% malignant, and the system is **refusing to
let that number count**. The percentage is the thing being withheld, not the
confidence in the refusal. That distinction is the single most important point
in your whole project — getting it backwards on camera would undercut
everything. The app now labels that marker explicitly.

**3. Calibration did not make the model "a little bit worse."**
You said that at 7:52, reading the calibration curve. What actually happened:
AUC dropped a hair (0.993 → 0.992) and **ECE improved** — meaning the
percentages got more honest. Those are different things, which is the entire
point of the Guo paper you cited. The honest line is:

> "Calibration cost us almost nothing in ranking and bought us percentages
> that mean what they say. That trade is the whole reason this project works."

**One thing you did not know, and should:** you asked how split conformal
prediction works. It is simpler than it sounds. You hold out a chunk of data
the model never trains on. You look at how wrong the model's probabilities
were on that chunk. That gives you a threshold. For a new sample, any label
whose probability clears the threshold goes in the set — and if nothing clears
it, the set is empty and the model abstains. The 90% guarantee comes from the
held-out chunk, not from the model's own opinion of itself. **You do not need
to say any of this in the video** — section 3 below says it in one line — but
you should know it in case a judge asks.

---

## Before you record

```bash
cd "C:\Users\Windows User\Documents\Work\univabio"
python -m pytest tests -q          # expect: 32 passed
streamlit run app.py
```

Kill any old Streamlit first or it will grab port 8501 and serve you a stale
build. First load takes 15–20s behind a spinner — wait it out before recording.

Cue cards are in `submission/cue_cards/`, captured from this exact build.

---

# THE SCRIPT

Total 3:00. Say it in your own words — these are the beats, not a teleprompter.

---

### 1 · The hook — 0:00–0:30
**Cue card:** `01_hero.png`

> Hey, I'm Quintin. This is Second Opinion.
>
> Here is the problem I kept running into. Say you build a model on breast
> tumour cells, and it works. Then someone feeds it a lung tumour cell.
> Those are both tumour cells. Under a microscope they look similar. But they
> are not the same thing, and a model trained on one has no business
> answering about the other.
>
> Almost every health-AI demo will answer anyway. Same confident percentage,
> no warning. And that is not a rare edge case — that is the normal way these
> models fail when they leave the hospital they were built in.
>
> So I built one that knows when to shut up.

---

### 2 · What it is, and what it was trained on — 0:30–0:50
**Cue card:** `01_hero.png` (point at the subtitle)

> First line on the page is what it was trained on: breast fine-needle-aspirate
> nuclear measurements, and nothing else. Not lung. Not images. I put that at
> the top on purpose, because a model's training scope is the first thing you
> should know about it and it is usually buried.
>
> And then the whole project is about what happens when you go outside it.

---

### 3 · Three questions, not one — 0:50–1:15
**Cue card:** `02_glossary.png`

Open the glossary and scroll it while you talk. You do not have to read it.

> Instead of one number, it answers three separate questions.
>
> One — what is the probability. That is a logistic regression, calibrated so
> the percentages mean what they say.
>
> Two — can that probability be trusted *for this specific sample*. That is
> conformal prediction. It uses data the model never trained on to work out a
> threshold, and if nothing clears that threshold, it returns an empty set and
> refuses.
>
> Three — is this sample even the kind of thing I have seen before. That is a
> distance measure that looks at whether the *combination* of measurements is
> strange, not just whether any one of them is.
>
> Every one of those is defined in plain English inside the app, including what
> each one can't do on its own. Because none of them is enough alone. Using all
> three together, and letting the last two veto the first, is the project.

---

### 4 · It works normally — 1:15–1:40
**Cue cards:** `04_malignant_decisive.png` → `06b_where_this_sample_sits.png`

Select **Typical malignant nucleus**. Then scroll to "Where this sample sits".

> On a normal sample it behaves like any good classifier. Decisive call,
> 99.61% malignant, in distribution.
>
> And you can see what that number actually means — this chart is every
> held-out sample, coloured by what it really turned out to be. Benign cases
> live down on the left, malignant on the right, and the dashed line is our
> sample sitting right in the middle of the malignant cluster. So the
> percentage isn't just a number, it's a position among cases we know the
> answers to.
>
> Also notice it does not say a hundred percent. It says 99.61. The honest
> ceiling is set by how much calibration data I have — claiming certainty I
> can't back up is exactly what I'm trying to avoid.

---

### 5 · It abstains, and tells you what to do — 1:40–2:05
**Cue card:** `05_abstain_with_next_step.png`

Select **Borderline (withhold)**.

> Here is a real patient where it refuses. Neither label is plausible at 90%
> confidence, so the set comes back empty and the headline is an abstention
> instead of a number.
>
> But it doesn't just say "I don't know" and leave you there. It works out the
> smallest single change that would settle it — if this one measurement rises
> about a third, it becomes a confident call. That is something you can
> actually act on. Go re-measure that one thing.

---

### 6 · The part that matters — 2:05–2:35
**Cue card:** `06_out_of_distribution.png`

Select **Implausible geometry**. Get the 99.61% and the red panel in frame
together.

> And this is the one I actually care about.
>
> This sample has area at the dataset maximum, with radius and perimeter at
> normal values. Every single number is inside the valid range — a simple
> bounds check waves it straight through. But that combination is
> geometrically impossible. A nucleus cannot have that area at that radius.
>
> Now look at the probability. The classifier underneath is 99.61% confident
> this is malignant. It is completely sure, and it is completely wrong to
> even be answering.
>
> And the system refuses to report it. Fifty-two times beyond the range the
> training data covers. It points at the exact measurement that looks like a
> typo, and it gives no diagnosis on the other side of that.
>
> That 99.61% is exactly what a normal demo would have shown you — with
> nothing attached to tell you not to believe it.

---

### 7 · Why you should believe me — 2:35–2:50
**Cue cards:** `08_metrics_with_bands.png` → `09_forty_splits.png`

> Quick on the evidence, because none of this is me asserting things.
>
> Every metric shows what a good value looks like, so you're not left guessing
> whether 0.03 is good.
>
> And coverage on any single split of a 569-sample dataset swings between 86
> and 99 percent depending purely on the random seed. So nothing here is quoted
> from one split — that's the spread across forty, against the 90% target. It
> averages 93.5.

---

### 8 · Close — 2:50–3:00
**Cue card:** `10_shift_charts.png`

> Last thing. Under increasing corruption, the amount it withholds climbs to
> a hundred percent — while "confident and wrong" stays flat on the floor. It
> fails by going quiet instead of guessing.
>
> This is a research prototype, not a medical device. One institution, early
> nineties, and it works on measurements rather than images.
>
> But a model that can't say "I don't know" will hand you a confident answer
> about a case it has never seen. This one says it.

---

## Delivery notes

- **Record section 6 first**, while you have the most energy. It is the one
  judges remember. You can record sections out of order and cut them together.
- If you fumble a line, stop and redo that section. Do not push through — the
  last take has a lot of "um, let's see" and it reads as uncertainty about your
  own work, which is the opposite of true here.
- Say numbers as words: "ninety-nine point six one percent", not reading digits.
- **Do not read the one-pager aloud.** You tried that and called it boring
  yourself at 11:21. It is written to be skimmed on paper, not spoken.
- Never call any output a diagnosis. Say "call", "assessment", or "the model
  says".
- You mentioned your NMR background. It is a genuine credibility point, but
  careful: "nuclear" in this dataset means *cell nucleus*, not nuclear physics.
  If you mention it, say "I work with measurement data rather than images, so
  this kind of problem is familiar" and leave it there.
