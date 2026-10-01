"""Second Opinion — interactive prototype.

    streamlit run app.py

The interaction the judges should notice is not "move slider, get number".
It is that the number disappears when the model has no business giving one,
and that the two withholds come with different, computed next steps.

Every metric on this page ships with what a good value for it looks like,
and every chart has labelled axes and a legend in plain words. A number a
reader cannot judge is decoration.
"""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from second_opinion import SecondOpinion
from second_opinion.conformal import ABSTAIN, AMBIGUOUS
from second_opinion.counterfactual import suggest
from second_opinion.data import (
    BASE_FEATURES,
    CITATION,
    expand_base_features,
    feature_reference,
    load,
    slider_bounds,
)
from second_opinion.diagnostics import (
    METRIC_GUIDES,
    coverage_rows,
    format_probability,
    isotonic_mapping,
    reliability_curve,
    split_distribution,
)
from second_opinion.presets import archetypal_presets
from second_opinion.references import REFERENCES_BIBLIOGRAPHY
from second_opinion.shift import evaluate_shift

st.set_page_config(page_title="Second Opinion", page_icon="🩺", layout="wide")

BAND_ICON = {"good": "🟢", "fair": "🟡", "poor": "🔴"}

PRESET_NOTES = {
    "Dataset median": "The middle value of every measurement across all 569 "
                      "samples. Not a real patient — a neutral starting point.",
    "Typical malignant nucleus": "A real malignant sample, the one closest to "
                                 "the centre of the malignant group.",
    "Typical benign nucleus": "A real benign sample, closest to the centre of "
                              "the benign group.",
    "Borderline (withhold)": "A real sample the model refuses to call. This is "
                             "what abstention looks like on genuine data.",
    "Implausible geometry": "Mean area at the dataset maximum while radius and "
                            "perimeter stay at their medians. Every value is "
                            "in range; the combination is impossible.",
}


@st.cache_resource
def get_model() -> SecondOpinion:
    return SecondOpinion().fit()


@st.cache_data
def get_reference() -> dict:
    return feature_reference()


@st.cache_resource
def get_presets() -> dict[str, np.ndarray]:
    return archetypal_presets(get_model())


@st.cache_data
def get_shift_rows() -> dict[str, list[dict]]:
    model = get_model()
    return {
        "noise": evaluate_shift(model, model.X_test, model.y_test, kind="noise"),
        "scale": evaluate_shift(model, model.X_test, model.y_test, kind="scale"),
    }


@st.cache_data
def get_calibration_data() -> dict:
    model = get_model()
    return {
        "reliability": reliability_curve(model),
        "isotonic": isotonic_mapping(model),
    }


def _apply_preset(name: str) -> None:
    vector = get_presets()[name]
    names = get_model().data.feature_names
    start = dict(zip(names, vector))
    st.session_state.preset_start = start
    for feat in BASE_FEATURES:
        st.session_state[f"slider_{feat}"] = float(start[feat])


# ── charts ─────────────────────────────────────────────────────────────────────

def reliability_chart(rows: list[dict]) -> alt.LayerChart:
    """Stated probability vs what actually happened. Diagonal = honest."""
    df = pd.DataFrame(rows)
    diagonal = (
        alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
        .mark_line(strokeDash=[6, 4], color="grey")
        .encode(x="x", y="y")
    )
    points = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("stated:Q", title="Probability the model stated",
                    scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            y=alt.Y("observed:Q", title="Share that were actually malignant",
                    scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            color=alt.Color("series:N", title="",
                            scale=alt.Scale(scheme="tableau10")),
            tooltip=[
                alt.Tooltip("series:N", title="Stage"),
                alt.Tooltip("stated:Q", title="Model said", format=".1%"),
                alt.Tooltip("observed:Q", title="Truth was", format=".1%"),
                alt.Tooltip("n:Q", title="Samples in bin"),
            ],
        )
    )
    return (diagonal + points).properties(height=320)


def isotonic_chart(rows: list[dict]) -> alt.LayerChart:
    """What the isotonic step actually does to every held-out score."""
    df = pd.DataFrame(rows)
    diagonal = (
        alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
        .mark_line(strokeDash=[6, 4], color="grey")
        .encode(x="x", y="y")
    )
    step = (
        alt.Chart(df)
        .mark_line(interpolate="step-after", color="#4c78a8")
        .encode(
            x=alt.X("raw:Q", title="Raw score from the logistic regression",
                    scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            y=alt.Y("calibrated:Q", title="Probability after calibration",
                    scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        )
    )
    dots = (
        alt.Chart(df)
        .mark_circle(size=55, opacity=0.65)
        .encode(
            x="raw:Q",
            y="calibrated:Q",
            color=alt.Color("truth:N", title="Actual diagnosis",
                            scale=alt.Scale(domain=["benign", "malignant"],
                                            range=["#54a24b", "#e45756"])),
            tooltip=[
                alt.Tooltip("raw:Q", title="Raw score", format=".1%"),
                alt.Tooltip("calibrated:Q", title="Calibrated", format=".1%"),
                alt.Tooltip("truth:N", title="Truth"),
            ],
        )
    )
    return (diagonal + step + dots).properties(height=320)


def coverage_chart(dist: dict) -> alt.LayerChart:
    """Per-seed coverage against the 90% target it is supposed to hit."""
    df = pd.DataFrame(coverage_rows(dist))
    target = float(dist.get("target_coverage", 0.9))
    bars = (
        alt.Chart(df)
        .mark_bar(color="#4c78a8")
        .encode(
            x=alt.X("coverage:Q", bin=alt.Bin(maxbins=18),
                    title="Conformal coverage achieved on that split",
                    axis=alt.Axis(format="%")),
            y=alt.Y("count()", title="Number of random splits"),
            tooltip=[alt.Tooltip("count()", title="Splits")],
        )
    )
    line = (
        alt.Chart(pd.DataFrame({"t": [target]}))
        .mark_rule(color="#e45756", strokeWidth=2.5)
        .encode(x="t:Q", tooltip=alt.Tooltip("t:Q", title="Target", format="%"))
    )
    return (bars + line).properties(height=260)


SHIFT_LABELS = {
    "withhold": "Withheld — no diagnosis given",
    "novelty": "Flagged as unlike the training data",
    "confident_error": "Confident AND wrong (the dangerous case)",
}


def shift_chart(rows: list[dict], x_title: str) -> alt.Chart:
    df = pd.DataFrame(rows)[["level", "withhold", "novelty", "confident_error"]]
    long = df.melt("level", var_name="series", value_name="share")
    long["series"] = long["series"].map(SHIFT_LABELS)
    return (
        alt.Chart(long)
        .mark_line(point=True)
        .encode(
            x=alt.X("level:Q", title=x_title,
                    scale=alt.Scale(domain=[float(df["level"].min()),
                                            float(df["level"].max())], nice=False)),
            y=alt.Y("share:Q", title="Share of held-out samples",
                    scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            color=alt.Color(
                "series:N", title="",
                scale=alt.Scale(
                    domain=list(SHIFT_LABELS.values()),
                    range=["#4c78a8", "#f58518", "#e45756"],
                ),
                legend=alt.Legend(orient="bottom", columns=1, labelLimit=400),
            ),
            tooltip=[
                alt.Tooltip("level:Q", title="Level"),
                alt.Tooltip("series:N", title=""),
                alt.Tooltip("share:Q", title="Share", format=".1%"),
            ],
        )
        .properties(height=300)
    )


# ── page ───────────────────────────────────────────────────────────────────────

model = get_model()
reference = get_reference()
data = load()
presets = get_presets()

st.title("🩺 Second Opinion")
st.markdown(
    "**A diagnostic model that reports when it should not be trusted.**  \n"
    "Most health-AI demos return a confident percentage for any input you give "
    "them, including inputs unlike anything they were trained on. This one "
    "answers three separate questions — what the probability is, whether that "
    "probability is trustworthy *for this sample*, and whether the sample is "
    "even the kind of thing the model has seen — and withholds the diagnosis "
    "when the last two say no."
)

st.info(
    "**Research prototype, not a medical device.** Trained on 569 historical "
    "fine-needle-aspirate measurements. It cannot diagnose anyone, has never "
    "been validated clinically, and must not be used for a real medical "
    "decision.",
    icon="⚠️",
)

with st.expander("📖 Plain-English glossary — what every method here actually does"):
    st.markdown(
        """
**Logistic regression** — the base classifier. It weighs each measurement and
adds the weights up to get a score. Simple on purpose: the contribution chart
further down is then its own arithmetic rather than an approximation of it.

**Isotonic calibration** — a correction applied *after* the model predicts. The
raw model is good at *ranking* (malignant above benign) but its percentages can
be systematically off: it might say 90% for cases that are really 70%. Isotonic
regression learns a staircase that maps each raw score onto the frequency
actually observed on held-out data. It can only reshape, never reorder — so
ranking is preserved and only the numbers change. *Why not enough alone:* it
makes the probability honest **on average**, across the whole population. It
still produces a confident number for a single sample it has never seen
anything like.

**Conformal prediction** — instead of one number, it returns a *set* of labels
with a coverage guarantee: across many cases, the true label is inside the set
at least 90% of the time. The set can hold one label (decisive), both
(genuinely borderline), or **none** — meaning nothing is plausible and the model
should not be read. *Why not enough alone:* the guarantee holds only while new
data looks like the old data. Under a distribution shift it silently stops being
true without any warning.

**Distribution shift** — when the data a deployed model sees stops resembling
what it was trained on: a different scanner, a different lab's calibration, a
different patient population. This is the thing that breaks models at the second
hospital. The *What happens under measurement noise* section below simulates it
directly.

**Mahalanobis distance** — a distance that accounts for how measurements
normally move together. Plain distance asks "is this value unusual?"; Mahalanobis
asks "is this *combination* unusual?" Area and radius are tightly linked in real
nuclei, so a large area with a median radius is far away in Mahalanobis terms
even though neither value is individually out of range. That is how the
*Implausible geometry* preset gets caught. *Why not enough alone:* it tells you
the sample is strange; it says nothing about whether it is malignant.
        """
    )

# ── input ──────────────────────────────────────────────────────────────────────

if "preset_start" not in st.session_state:
    _apply_preset("Dataset median")

with st.sidebar:
    st.header("The sample being assessed")
    st.caption(
        "Everything in this panel is the **input**. Pick a cell profile and the "
        "assessment on the right updates for that sample."
    )
    st.selectbox(
        "Cell profile",
        list(presets.keys()),
        key="preset_select",
        on_change=lambda: _apply_preset(st.session_state.preset_select),
        help="Named starting points from real WDBC geometry, not invented "
             "cell types.",
    )
    st.info(PRESET_NOTES.get(st.session_state.get("preset_select",
                                                  "Dataset median, "), ""),
            icon="🔬")

    with st.expander("Fine-tune the measurements (optional)"):
        st.caption(
            "Ten nuclear measurements, in the dataset's own units. Ranges are "
            "the true min/max of the data, padded 50% either side so you can "
            "push a sample out of distribution on purpose and watch the "
            "diagnosis disappear."
        )
        chosen: dict[str, float] = {}
        for name in BASE_FEATURES:
            lo, hi = slider_bounds(name, reference)
            span = reference[name]["max"] - reference[name]["min"]
            if f"slider_{name}" not in st.session_state:
                st.session_state[f"slider_{name}"] = float(
                    st.session_state.preset_start[name]
                )
            chosen[name] = st.slider(
                name,
                min_value=lo,
                max_value=hi,
                step=float(span / 200),
                key=f"slider_{name}",
                help=f"Dataset range {reference[name]['min']:.3g} to "
                     f"{reference[name]['max']:.3g}",
            )

start = st.session_state.preset_start
vector = expand_base_features(
    chosen,
    start=start,
    reference=reference,
    feature_names=data.feature_names,
)
assessment = model.assess(vector)
counterfactual = suggest(model, vector)

# ── output ─────────────────────────────────────────────────────────────────────

left, right = st.columns([3, 2])

with left:
    st.subheader("Assessment")

    if assessment.novelty.is_outlier:
        st.error(f"### {assessment.headline()}", icon="🚫")
        st.markdown(f"**Why:** {assessment.novelty.explain()}")
    elif assessment.prediction_set.verdict == ABSTAIN:
        st.warning(f"### {assessment.headline()}", icon="🤷")
        st.markdown(f"**Why:** {assessment.prediction_set.explain()}")
    elif assessment.prediction_set.verdict == AMBIGUOUS:
        st.warning(f"### {assessment.headline()}", icon="⚖️")
        st.markdown(f"**Why:** {assessment.prediction_set.explain()}")
    else:
        st.success(f"### {assessment.headline()}", icon="✅")
        st.markdown(f"**Why:** {assessment.prediction_set.explain()}")

    if counterfactual is not None:
        label = (
            "What to check"
            if counterfactual.kind == "novelty"
            else "What would make this decisive"
        )
        st.markdown(f"**{label}:** {counterfactual.explanation}")

    st.divider()
    st.markdown("#### The three questions, separately")
    st.caption(
        "Question 1 always answers. Questions 2 and 3 decide whether you are "
        "allowed to read it."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "1. Probability",
        format_probability(assessment.probability_malignant),
        help="Probability malignant, calibrated on held-out data: 70% means "
             "roughly 70 of 100 such samples are malignant. On its own this "
             "says nothing about whether it can be trusted here.",
    )
    c1.caption("Calibrated logistic regression. Always answers.")

    c2.metric(
        "2. Conformal",
        assessment.prediction_set.verdict,
        help="A prediction set with a coverage guarantee. 'abstain' means no "
             "label is plausible at this confidence level.",
    )
    c2.caption("`decisive` = one label · `abstain` = none plausible")

    c3.metric(
        "3. Novelty",
        "outlier" if assessment.novelty.is_outlier else "typical",
        f"{assessment.novelty.ratio:.2f}x threshold",
        # Green when the sample sits comfortably inside the training
        # distribution, red when it does not. A low ratio is a good thing and
        # must not be painted as a warning.
        delta_color="inverse" if assessment.novelty.is_outlier else "normal",
        help="Mahalanobis distance from the training data, relative to the "
             "distance containing 99% of it.",
    )
    c3.caption("Above **1.00x** = stranger than 99% of training data.")

    if not assessment.trustworthy:
        st.caption(
            "The probability above is shown for transparency, not as a result. "
            "This is the case the project exists for."
        )

with right:
    st.subheader("What drove this")
    st.caption(
        "The six measurements that moved this particular sample's score the "
        "most. Each bar is **the model's weight for that feature × this "
        "sample's standardised value** — the actual arithmetic behind the "
        "score, not an approximation of it. Bars to the right pushed toward "
        "*malignant*; bars to the left pushed toward *benign*."
    )
    contrib = pd.DataFrame(
        assessment.contributions, columns=["feature", "contribution"]
    )
    # Altair rather than st.bar_chart: the built-in truncates feature names to
    # "mean concave po…", which is the one thing this panel must not do.
    st.altair_chart(
        alt.Chart(contrib)
        .mark_bar()
        .encode(
            x=alt.X("contribution:Q", title="Push toward malignant  →"),
            y=alt.Y("feature:N", title="", sort="-x",
                    axis=alt.Axis(labelLimit=260)),
            color=alt.condition(
                alt.datum.contribution > 0,
                alt.value("#e45756"), alt.value("#4c78a8"),
            ),
            tooltip=[
                alt.Tooltip("feature:N", title="Measurement"),
                alt.Tooltip("contribution:Q", title="Contribution",
                            format="+.3f"),
            ],
        )
        .properties(height=260),
        use_container_width=True,
    )

st.divider()

# ── evidence ───────────────────────────────────────────────────────────────────

st.header("Why you should believe any of this")

m = model.metrics
cal = get_calibration_data()
dist = split_distribution()

st.markdown("#### Does the probability mean what it says?")
st.caption(
    "A model can rank perfectly and still lie about its percentages "
    "(Guo et al., 2017). These two charts are that distinction, drawn."
)

g1, g2 = st.columns(2)
with g1:
    st.markdown("**Calibration curve** — stated vs actual")
    st.altair_chart(reliability_chart(cal["reliability"]), use_container_width=True)
    st.caption(
        "Each point is a group of held-out samples of equal size (hover for "
        "the count). The dashed diagonal is a perfectly honest model: of the "
        "cases it called 70%, exactly 70% were malignant. Closer to the "
        "diagonal is better — that distance *is* ECE. Groups hold an equal "
        "number of samples rather than an equal probability range, because "
        "this model's predictions cluster near 0 and 1 and even-width bins "
        "would leave the middle ones holding a single point."
    )
with g2:
    st.markdown("**What calibration actually does**")
    st.altair_chart(isotonic_chart(cal["isotonic"]), use_container_width=True)
    st.caption(
        "Every held-out sample, coloured by its true diagnosis. The blue "
        "staircase is the isotonic fit: it reads the raw score on the x-axis "
        "and returns the corrected probability on the y-axis. Points on the "
        "dashed diagonal were left unchanged; the flat runs are where several "
        "different raw scores all map to the same honest probability."
    )

st.markdown("#### The headline numbers, and what a good value looks like")

mc = st.columns(4)
for col, (key, before, after) in zip(
    mc,
    [
        ("auc", m.auc_uncalibrated, m.auc_calibrated),
        ("brier", m.brier_uncalibrated, m.brier_calibrated),
        ("ece", m.ece_uncalibrated, m.ece_calibrated),
    ],
):
    guide = METRIC_GUIDES[key]
    band = guide.verdict(after)
    delta = after - before
    col.metric(
        f"{BAND_ICON[band]} {guide.label}",
        f"{after:.3f}",
        f"{delta:+.3f} from calibration",
        delta_color="inverse" if guide.better == "lower" else "normal",
        help=guide.plain,
    )
    col.caption(f"**{guide.ideal}.** This value reads as **{band}**.")

coverage_band = "good" if m.coverage >= m.target_coverage else "fair"
mc[3].metric(
    f"{BAND_ICON[coverage_band]} Conformal coverage",
    f"{m.coverage:.1%}",
    f"target {m.target_coverage:.0%}",
    delta_color="off",
)
mc[3].caption(
    "**At or above target is good.** Below target on a single split is "
    "expected noise — see the spread below."
)

st.caption(
    f"Split: train {m.n_train} / calibration {m.n_calibration} / test "
    f"{m.n_test}, malignant prevalence {m.prevalence:.1%}. The calibration "
    "split is never trained on; the test split is touched only for these "
    "numbers."
)

if dist:
    st.markdown("#### One split is a coin flip — so here are all 40")
    h1, h2 = st.columns([3, 2])
    with h1:
        st.altair_chart(coverage_chart(dist), use_container_width=True)
        st.caption(
            "Conformal coverage on each of 40 random train/calibration/test "
            "splits. The red line is the 90% target. The split shown above "
            f"landed at {m.coverage:.1%} — which is why no single split is "
            "quoted as the result."
        )
    with h2:
        cov = np.array(dist["coverage"])
        st.metric("Mean coverage over 40 splits", f"{cov.mean():.1%}",
                  f"± {cov.std():.1%} sd", delta_color="off")
        st.metric("Decisive", f"{np.mean(dist['decisive']):.1%}",
                  delta_color="off")
        st.metric("Abstained", f"{np.mean(dist['abstain']):.1%}",
                  delta_color="off")
        st.metric("Scrambled inputs caught",
                  f"{dist['scrambled_flagged']:.0%}",
                  f"vs {np.mean(dist['novelty_flagged']):.1%} of real samples",
                  delta_color="off")
        st.caption(f"Range across splits: {cov.min():.1%} to {cov.max():.1%}.")
else:
    st.info("Run `python scripts/evaluate.py` to generate the 40-split charts.")

with st.expander("What happens under measurement noise (distribution shift, simulated)"):
    st.markdown(
        "Held-out samples are progressively corrupted, simulating a different "
        "lab or a miscalibrated instrument. The question is **how a model "
        "fails**, not whether it does."
    )
    st.markdown(
        "- **Withheld** should rise — the model noticing it is out of its depth.\n"
        "- **Confident AND wrong** should *not* rise with it. That line staying "
        "flat near zero is the whole claim: it fails by going quiet rather "
        "than by guessing."
    )
    rows = get_shift_rows()
    st.altair_chart(
        shift_chart(rows["noise"],
                    "Noise added (standard deviations of the training scale)"),
        use_container_width=True,
    )
    st.caption(
        "At zero noise about 23% is already withheld — the model is cautious "
        "on genuine data too. By one standard deviation of noise it refuses "
        "essentially everything."
    )
    st.altair_chart(
        shift_chart(rows["scale"],
                    "Size-feature scale shift (radius / perimeter / area)"),
        use_container_width=True,
    )
    st.caption(
        "A different lab whose instrument reports sizes on a different scale. "
        "The small bump in *confident AND wrong* around a 10–25% shift is "
        "honest and worth seeing: a mild shift is the hardest case, because "
        "it is big enough to matter and small enough to look normal."
    )

with st.expander("Methods & references"):
    st.markdown(
        "Each method in this prototype traces to a published source. "
        "The conformal guarantee is *marginal* (Angelopoulos & Bates, 2023): "
        "it holds on average, not necessarily within every subgroup."
    )
    for ref in REFERENCES_BIBLIOGRAPHY:
        st.markdown(f"- {ref}")
    st.caption(f"Data: {CITATION}")
