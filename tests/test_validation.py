import numpy as np

from painpredict.synthetic import make_synthetic_cohort
from painpredict.validation import bootstrap_performance, decision_curve, fit_final_model, repeated_nested_cv


def test_synthetic_cohort_shape():
    frame = make_synthetic_cohort(356, seed=1)
    assert len(frame) == 356
    assert 0.12 < frame.event.mean() < 0.30
    assert (frame.event == (frame.nrs_24h >= 4)).all()
    assert frame["D-dimer"].isna().any()


def test_nested_cv_runs_without_leakage():
    frame = make_synthetic_cohort(200, seed=3)
    result = repeated_nested_cv(frame, repeats=1)
    assert len(result.predictions) == len(frame)
    assert result.folds.train_test_overlap.eq(0).all()
    assert result.predictions.probability.between(0, 1).all()
    summary = bootstrap_performance(result.predictions.event, result.predictions.probability, n_bootstrap=50)
    auc = summary.set_index("metric").loc["auc"]
    assert 0.6 < auc.estimate < 1.0
    assert auc.ci_low <= auc.estimate <= auc.ci_high


def test_decision_curve_and_refit():
    frame = make_synthetic_cohort(200, seed=4)
    model = fit_final_model(frame)
    risk = model.predict_proba(frame[["Pre_PainScore", "D-dimer", "AFP", "Albumin"]])[:, 1]
    curve = decision_curve(frame.event.to_numpy(), risk, np.array([0.1, 0.2, 0.3]))
    assert list(curve.columns) == ["threshold", "model", "treat_all", "treat_none"]
    assert curve.treat_none.eq(0).all()
