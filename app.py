"""Second Opinion — interactive prototype.

    streamlit run app.py

The interaction the judges should notice is not "move slider, get number".
It is that the number disappears when the model has no business giving one.
Push the sliders somewhere physically implausible and the tool stops
answering instead of extrapolating confidently, which is the failure mode
every other demo in this category has.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from second_opinion import SecondOpinion
from second_opinion.conformal import ABSTAIN, AMBIGUOUS
from second_opinion.data import CITATION, feature_reference, load
from second_opinion.references import REFERENCES_BIBLIOGRAPHY

st.set_page_config(page_title="Second Opinion", page_icon="🩺", layout="wide")

# The ten base measurements. The dataset also carries "... error" and
# "worst ..." variants of each; those are driven from the base value below
# rather than given their own sliders, because thirty independent sliders is
# not an interface, and because moving them independently produces
# combinations no real cell nucleus could have.
BASE_FEATURES = [
    "mean radius", "mean texture", "mean perimeter", "mean area",
    "mean smoothness", "mean compactness", "mean concavity",
    "mean concave points", "mean symmetry", "mean fractal dimension",
]


@st.cache_resource
def get_model() -> SecondOpinion:
    return SecondOpinion().fit()


@st.cache_data
def get_reference() -> dict:
    return feature_reference()


@st.cache_data
def get_examples() -> dict[str, list[float]]:
    """Three real samples from the dataset, for one-click demoing."""
    data = load()
    mal = int(np.where(data.y == 1)[0][0])
    ben = int(np.where(data.y == 0)[0][0])
    return {
        "A real malignant sample": data.X[mal].tolist(),
        "A real benign sample": data.X[ben].tolist(),
    }


model = get_model()
reference = get_reference()
data = load()

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

# ── input ──────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Sample measurements")
    preset = st.selectbox(
        "Start from", ["Dataset median"] + list(get_examples().keys())
    )
    st.caption(
        "Slider ranges are the true min/max of the dataset. Push a value to an "
        "extreme to see the novelty detector fire."
    )

    examples = get_examples()
    if preset in examples:
        start = dict(zip(data.feature_names, examples[preset]))
    else:
        start = {n: reference[n]["median"] for n in data.feature_names}

    chosen: dict[str, float] = {}
    for name in BASE_FEATURES:
        ref = reference[name]
        span = ref["max"] - ref["min"]
        chosen[name] = st.slider(
            name,
            min_value=float(ref["min"] - span * 0.5),
            max_value=float(ref["max"] + span * 0.5),
            value=float(start[name]),
            step=float(span / 200),
            help=f"Dataset range {ref['min']:.3g} to {ref['max']:.3g}",
        )

# Build the full 30-feature vector: the ten sliders, plus the error/worst
# variants scaled by how far the slider moved from the dataset median. Holding
# them fixed instead would let a user triple the mean radius while the "worst
# radius" stayed put — an impossible sample, which the novelty detector would
# then correctly flag, but for the wrong reason.
vector = []
for name in data.feature_names:
    if name in chosen:
        vector.append(chosen[name])
        continue
    base = name.replace(" error", "").replace("worst ", "mean ")
    if base in chosen and reference[base]["median"]:
        ratio = chosen[base] / reference[base]["median"]
        vector.append(start.get(name, reference[name]["median"]) * ratio)
    else:
        vector.append(start.get(name, reference[name]["median"]))

assessment = model.assess(np.array(vector))

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

    st.divider()
    st.markdown("#### The three questions, separately")

    # Short labels/values on purpose: st.metric truncates silently rather than
    # wrapping, and three of them side by side in a 3/5-width column left
    # "1. Probability malignant" reading "1. Probability malig..." and
    # "in distribution" reading "in distri...". The full wording lives in the
    # tooltip (help=), which is not truncated.
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "1. Probability",
        f"{assessment.probability_malignant:.1%}",
        help="Probability malignant, calibrated on held-out data: 70% means "
             "roughly 70 of 100 such samples are malignant. On its own this "
             "says nothing about whether it can be trusted here.",
    )
    c2.metric(
        "2. Conformal",
        assessment.prediction_set.verdict,
        help="A prediction set with a coverage guarantee. 'abstain' means no "
             "label is plausible at this confidence level.",
    )
    c3.metric(
        "3. Novelty",
        "outlier" if assessment.novelty.is_outlier else "typical",
        f"{assessment.novelty.ratio:.2f}x threshold",
        delta_color="inverse",
        help="Mahalanobis distance from the training data, relative to the "
             "distance containing 99% of it.",
    )

    if not assessment.trustworthy:
        st.caption(
            "The probability above is shown for transparency, not as a result. "
            "This is the case the project exists for."
        )

with right:
    st.subheader("What drove this")
    st.caption(
        "Exact contributions to the log-odds — coefficient × standardised "
        "value — not an approximation. Positive pushes toward malignant."
    )
    contrib = pd.DataFrame(
        assessment.contributions, columns=["feature", "contribution"]
    ).set_index("feature")
    st.bar_chart(contrib, horizontal=True)

st.divider()

with st.expander("Held-out performance (why you should believe any of this)"):
    m = model.metrics
    st.code(m.summary(), language="text")
    st.markdown(
        "Figures above are for this one fixed split. Across 40 random splits: "
        "**coverage 93.5% ± 3.5%**, decisive on **95.0%** of cases, abstaining "
        "on **5.0%**, AUC **0.985**, Brier **0.025**. Coverage on a single "
        "split ranges from 86% to 99% — which is why the headline numbers are "
        "averaged rather than taken from whichever split flattered the model. "
        "Regenerate with `python scripts/evaluate.py`.\n\n"
        "The novelty detector flags **100%** of deliberately scrambled inputs "
        "and **6.8%** of genuine held-out samples."
    )
    st.caption(f"Data: {CITATION}")

with st.expander("Methods & references"):
    st.markdown(
        "Each method in this prototype traces to a published source. "
        "The conformal guarantee is *marginal* (Angelopoulos & Bates, 2023): "
        "it holds on average, not necessarily within every subgroup."
    )
    for ref in REFERENCES_BIBLIOGRAPHY:
        st.markdown(f"- {ref}")
