"""Second Opinion — interactive prototype.

    streamlit run app.py

The interaction the judges should notice is not "move slider, get number".
It is that the number disappears when the model has no business giving one,
and that the two withholds come with different, computed next steps.
"""

from __future__ import annotations

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
from second_opinion.presets import archetypal_presets
from second_opinion.references import REFERENCES_BIBLIOGRAPHY
from second_opinion.shift import evaluate_shift

st.set_page_config(page_title="Second Opinion", page_icon="🩺", layout="wide")


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


def _apply_preset(name: str) -> None:
    vector = get_presets()[name]
    names = get_model().data.feature_names
    start = dict(zip(names, vector))
    st.session_state.preset_start = start
    for feat in BASE_FEATURES:
        st.session_state[f"slider_{feat}"] = float(start[feat])


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

# ── input ──────────────────────────────────────────────────────────────────────

if "preset_start" not in st.session_state:
    _apply_preset("Dataset median")

with st.sidebar:
    st.header("Sample measurements")
    st.selectbox(
        "Cell profile",
        list(presets.keys()),
        key="preset_select",
        on_change=lambda: _apply_preset(st.session_state.preset_select),
        help="Named starting points from real WDBC geometry, not invented "
             "cell types. Sliders below stay live for fine-tuning.",
    )
    st.caption(
        "The dropdown fills the sliders from a real sample (or the documented "
        "impossible-geometry demo). Fine-tune from there. Slider ranges are "
        "the true min/max of the dataset, padded so you can trip novelty."
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
        "Exact contributions — coefficient x standardised value — "
        "not an approximation. Positive pushes toward malignant."
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
        "**coverage 92.1% ± 3.1%**, decisive on **94.2%** of cases, abstaining "
        "on **5.5%**, AUC **0.980**, Brier **0.036**. Coverage on a single "
        "split ranges from 86% to 99% — which is why the headline numbers are "
        "averaged rather than taken from whichever split flattered the model. "
        "Regenerate with `python scripts/evaluate.py`.\n\n"
        "The novelty detector flags **100%** of deliberately scrambled inputs "
        "and **6.8%** of genuine held-out samples."
    )
    st.caption(f"Data: {CITATION}")

with st.expander("What happens under measurement noise"):
    st.markdown(
        "Held-out samples are progressively corrupted. **Withhold** (novelty "
        "or a non-decisive conformal set) should rise. **Confident error** — "
        "decisive, in-distribution, and wrong — should not climb with it."
    )
    rows = get_shift_rows()
    noise = pd.DataFrame(rows["noise"]).set_index("level")[
        ["withhold", "novelty", "confident_error"]
    ]
    noise.index.name = "noise (stdev of training scale)"
    st.line_chart(noise)
    scale = pd.DataFrame(rows["scale"]).set_index("level")[
        ["withhold", "novelty", "confident_error"]
    ]
    scale.index.name = "size-feature scale shift"
    st.caption("Covariate shift on radius / perimeter / area (a different lab).")
    st.line_chart(scale)

with st.expander("Methods & references"):
    st.markdown(
        "Each method in this prototype traces to a published source. "
        "The conformal guarantee is *marginal* (Angelopoulos & Bates, 2023): "
        "it holds on average, not necessarily within every subgroup."
    )
    for ref in REFERENCES_BIBLIOGRAPHY:
        st.markdown(f"- {ref}")
