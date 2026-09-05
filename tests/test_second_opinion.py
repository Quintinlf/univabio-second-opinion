"""Tests for the claims this project makes.

Each test pins a claim that appears in the README or the UI. If a claim is
not checkable it should not be made.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer

from second_opinion import SecondOpinion
from second_opinion.conformal import (
    ABSTAIN, AMBIGUOUS, DECISIVE, calibrate, coverage, predict_set,
)
from second_opinion.counterfactual import (
    conformal_counterfactual, novelty_counterfactual, suggest,
)
from second_opinion.data import (
    BASE_FEATURES, BENIGN, MALIGNANT, expand_base_features, feature_reference,
    load,
)
from second_opinion.model import expected_calibration_error
from second_opinion.ood import NoveltyDetector
from second_opinion.presets import archetypal_presets
from second_opinion.references import REFERENCES, REFERENCES_BIBLIOGRAPHY
from second_opinion.shift import evaluate_shift


@pytest.fixture(scope="module")
def model() -> SecondOpinion:
    return SecondOpinion().fit()


# ── data ───────────────────────────────────────────────────────────────────────

def test_label_direction_is_flipped_so_one_means_malignant():
    """sklearn ships this with 0 == malignant. A screening tool whose positive
    class is 'healthy' will report high confidence for the wrong thing."""
    raw = load_breast_cancer()
    data = load()
    assert data.y.sum() == (raw.target == 0).sum()
    assert data.prevalence == pytest.approx(0.3726, abs=1e-3)
    # The malignant class must be the minority; if this flips, the direction
    # was silently reversed somewhere.
    assert data.prevalence < 0.5


# ── conformal ──────────────────────────────────────────────────────────────────

def test_calibrate_uses_the_finite_sample_correction():
    """With n=9 and alpha=0.1 the level is ceil(10*0.9)/9 = 1.0, not 0.9.
    Dropping the correction undercovers on small calibration sets."""
    p = np.column_stack([np.linspace(0.9, 0.1, 9), np.linspace(0.1, 0.9, 9)])
    labels = np.ones(9, dtype=int)
    qhat = calibrate(p, labels, alpha=0.10)
    scores = 1 - p[np.arange(9), labels]
    assert qhat == pytest.approx(scores.max())


def test_prediction_set_verdicts():
    # qhat=0.3 -> threshold 0.7. Only labels at/above 0.7 are included.
    assert predict_set(0.95, qhat=0.3).verdict == DECISIVE
    assert predict_set(0.05, qhat=0.3).verdict == DECISIVE
    assert predict_set(0.50, qhat=0.3).verdict == ABSTAIN
    # qhat=0.6 -> threshold 0.4, so 0.5 admits both labels.
    assert predict_set(0.50, qhat=0.6).verdict == AMBIGUOUS


def test_abstention_is_the_empty_set_not_a_low_probability():
    """The distinguishing feature of the project: abstaining is a structural
    outcome, not a threshold on the probability."""
    p = predict_set(0.5, qhat=0.05)
    assert p.labels == []
    assert p.verdict == ABSTAIN
    assert "withholding" in p.explain()


def test_decisive_set_names_the_right_label():
    assert predict_set(0.99, qhat=0.3).includes_malignant
    assert not predict_set(0.99, qhat=0.3).includes_benign
    assert predict_set(0.01, qhat=0.3).includes_benign
    assert not predict_set(0.01, qhat=0.3).includes_malignant


def test_calibrate_rejects_malformed_input():
    with pytest.raises(ValueError):
        calibrate(np.zeros((5, 3)), np.zeros(5, dtype=int))
    with pytest.raises(ValueError):
        calibrate(np.zeros((5, 2)), np.zeros(4, dtype=int))
    with pytest.raises(ValueError):
        calibrate(np.zeros((5, 2)), np.zeros(5, dtype=int), alpha=0)


def test_coverage_guarantee_holds_on_held_out_data(model):
    """The whole point. Coverage on data the threshold was not fitted on must
    land near the target -- conservatively, not below it by much."""
    assert model.metrics.coverage >= 0.85
    assert model.metrics.target_coverage == 0.90


def test_coverage_holds_on_average_across_many_splits():
    """One split of 569 samples is noise; the README quotes the average, so
    the average is what gets tested. Measured range is 86%-99%."""
    covs = [SecondOpinion(seed=s).fit().metrics.coverage for s in range(12)]
    assert np.mean(covs) >= 0.90, f"mean coverage {np.mean(covs):.3f} below target"


# ── novelty ────────────────────────────────────────────────────────────────────

def test_novelty_detector_flags_scrambled_inputs(model):
    """A detector that never fires is decoration."""
    data = load()
    rng = np.random.default_rng(0)
    junk = data.X[rng.integers(0, len(data), 100)] * rng.uniform(2.5, 4.0, (100, 30))
    flagged = (model.novelty.score(model.scaler.transform(junk)) > model.novelty.threshold)
    assert flagged.mean() > 0.95


def test_novelty_detector_accepts_genuine_samples(model):
    """...and one that always fires is equally useless."""
    data = load()
    flagged = model.novelty.score(model.scaler.transform(data.X)) > model.novelty.threshold
    assert flagged.mean() < 0.10


def test_novelty_detector_requires_fitting():
    with pytest.raises(RuntimeError):
        NoveltyDetector().threshold
    with pytest.raises(RuntimeError):
        NoveltyDetector().score(np.zeros((1, 30)))


def test_novelty_detector_requires_both_classes():
    data = load()
    X_mal = data.X[data.y == MALIGNANT]
    y_mal = data.y[data.y == MALIGNANT]
    with pytest.raises(ValueError, match="no training samples"):
        NoveltyDetector().fit(X_mal, y_mal)


# ── the assembled model ────────────────────────────────────────────────────────

def test_assessment_withholds_trust_on_an_outlier(model):
    """The headline behaviour: an impossible sample gets no diagnosis, however
    confident the underlying probability happens to look."""
    data = load()
    assessment = model.assess(data.X[0] * 4.0)
    assert assessment.novelty.is_outlier
    assert not assessment.trustworthy
    assert "Out of distribution" in assessment.headline()


def test_assessment_trusts_a_genuine_sample(model):
    data = load()
    trusted = [model.assess(x).trustworthy for x in data.X[:40]]
    assert sum(trusted) > 25, "model should trust most genuine samples"


def test_assess_rejects_wrong_feature_count(model):
    with pytest.raises(ValueError, match="expected 30 features"):
        model.assess(np.zeros(5))


def test_contributions_sum_toward_the_logit(model):
    """Claimed in the UI as exact rather than approximate, so it is checked:
    all 30 contributions plus the intercept must reproduce the model's logit."""
    data = load()
    x = data.X[0]
    total = float(model.feature_contributions(x).sum() + model.base.intercept_)
    expected = float(model.base.decision_function(x.reshape(1, -1))[0])
    assert total == pytest.approx(expected, abs=1e-8)


def test_contributions_are_ranked_by_magnitude(model):
    data = load()
    contrib = model.contributions(data.X[0], top=6)
    assert len(contrib) == 6
    magnitudes = [abs(v) for _, v in contrib]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_calibration_does_not_wreck_ranking(model):
    """Isotonic calibration trades a little AUC for a better Brier score.
    'A little' is the claim; this pins it."""
    m = model.metrics
    assert m.auc_calibrated > 0.95
    assert abs(m.auc_uncalibrated - m.auc_calibrated) < 0.05


def test_unfitted_model_refuses_to_predict():
    with pytest.raises(RuntimeError):
        SecondOpinion().assess(np.zeros(30))


def test_references_include_all_load_bearing_papers():
    expected = {
        "street1993": ("Street", "1993"),
        "guo2017": ("Guo", "2017"),
        "lee2018": ("Lee", "2018"),
        "sadinle2019": ("Sadinle", "2019"),
        "hastie1986": ("Hastie", "1986"),
        "angelopoulos2021": ("Angelopoulos", "2021"),
    }
    for key, (author, year) in expected.items():
        assert key in REFERENCES
        assert author in REFERENCES[key]
        assert year in REFERENCES[key]
    assert len(REFERENCES_BIBLIOGRAPHY) == len(expected)


def test_ece_computed_and_calibrated(model):
    """Guo et al.: calibration and accuracy are different quantities."""
    m = model.metrics
    assert m.ece_uncalibrated >= 0
    assert m.ece_calibrated >= 0
    assert expected_calibration_error(
        model.base.predict_proba(model.data.X)[:, 1], model.data.y
    ) >= 0


# ── feature reconstruction ─────────────────────────────────────────────────────

def test_expand_base_features_keeps_chosen_means_and_scales_the_rest():
    data = load()
    reference = feature_reference(data)
    start = {n: reference[n]["median"] for n in data.feature_names}
    chosen = {n: start[n] for n in BASE_FEATURES}
    chosen["mean radius"] = reference["mean radius"]["median"] * 1.5
    vec = expand_base_features(
        chosen, start=start, reference=reference, feature_names=data.feature_names
    )
    names = data.feature_names
    assert vec[names.index("mean radius")] == pytest.approx(chosen["mean radius"])
    assert vec[names.index("radius error")] == pytest.approx(
        start["radius error"] * 1.5
    )
    assert vec[names.index("worst radius")] == pytest.approx(
        start["worst radius"] * 1.5
    )
    assert vec[names.index("mean texture")] == pytest.approx(chosen["mean texture"])


# ── counterfactuals ────────────────────────────────────────────────────────────

def _in_distribution_withhold(model) -> np.ndarray:
    for x in model.data.X:
        a = model.assess(x)
        if not a.novelty.is_outlier and a.prediction_set.verdict != DECISIVE:
            return x
    presets = archetypal_presets(model)
    x = presets["Borderline (withhold)"]
    a = model.assess(x)
    if not a.novelty.is_outlier and a.prediction_set.verdict != DECISIVE:
        return x
    pytest.skip("no in-distribution withhold on this split")


def test_conformal_counterfactual_makes_a_decisive_in_distribution_call(model):
    x = _in_distribution_withhold(model)
    cf = conformal_counterfactual(model, x)
    assert cf is not None
    assert cf.kind == "conformal"
    assert novelty_counterfactual(model, x) is None
    if cf.new_value is None:
        return
    names = model.data.feature_names
    reference = feature_reference(model.data)
    start = dict(zip(names, x))
    chosen = {n: start[n] for n in BASE_FEATURES}
    chosen[cf.feature] = cf.new_value
    vec = expand_base_features(
        chosen, start=start, reference=reference, feature_names=names
    )
    a = model.assess(vec)
    assert a.prediction_set.verdict == DECISIVE
    assert not a.novelty.is_outlier


def test_novelty_counterfactual_names_a_feature_and_refuses_a_diagnosis(model):
    x = model.data.X[0] * 4.0
    cf = novelty_counterfactual(model, x)
    assert cf is not None
    assert cf.kind == "novelty"
    assert cf.feature is not None
    assert cf.drivers
    assert "not a diagnosis" in cf.explanation
    routed = suggest(model, x)
    assert routed.kind == "novelty"
    assert conformal_counterfactual(model, x) is None


def test_suggest_is_silent_on_a_trusted_decisive_sample(model):
    for x in model.data.X:
        a = model.assess(x)
        if a.trustworthy and a.prediction_set.verdict == DECISIVE:
            assert suggest(model, x) is None
            return
    pytest.fail("expected at least one trusted decisive sample")


def test_presets_are_thirty_feature_vectors(model):
    presets = archetypal_presets(model)
    assert "Implausible geometry" in presets
    for name, vec in presets.items():
        assert vec.shape == (30,), name
    implausible = model.assess(presets["Implausible geometry"])
    assert implausible.novelty.is_outlier


# ── shift stress test ──────────────────────────────────────────────────────────

def test_withhold_rises_under_heavy_noise_and_confident_error_does_not(model):
    rows = evaluate_shift(
        model, model.X_test, model.y_test, levels=(0.0, 2.0), kind="noise"
    )
    clean, noisy = rows[0], rows[1]
    assert noisy["withhold"] > clean["withhold"]
    assert noisy["confident_error"] <= clean["confident_error"] + 0.08
