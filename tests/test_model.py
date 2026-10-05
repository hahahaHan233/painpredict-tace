from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from painpredict import FEATURES, formula_risk, load_model, model_card, predict_risk
from painpredict.nomogram import points_table
from painpredict.schema import SchemaError

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def examples():
    return pd.read_csv(ROOT / "examples" / "example_patients.csv")


def test_packaged_model_matches_card():
    model = load_model()
    names = list(model.named_steps["preprocess"].get_feature_names_out())
    assert names[:4] == list(FEATURES)
    assert model.named_steps["model"].C == pytest.approx(model_card()["estimator"]["C"])


def test_predictions_are_probabilities(examples):
    result = predict_risk(examples)
    assert result.index.equals(examples.index)
    assert result.predicted_risk.between(0, 1).all()
    assert (result.above_threshold == (result.predicted_risk >= 0.20)).all()


def test_published_equation_reproduces_pipeline(examples):
    synthetic = pd.read_csv(ROOT / "examples" / "synthetic_cohort.csv")
    for frame in (examples, synthetic):
        assert np.allclose(predict_risk(frame).predicted_risk, formula_risk(frame), atol=1e-9)


def test_missing_laboratory_values_are_imputed(examples):
    assert examples[["D-dimer", "AFP", "Albumin"]].isna().any().any()
    assert predict_risk(examples).predicted_risk.notna().all()


def test_risk_increases_with_preprocedural_pain():
    base = pd.DataFrame({"Pre_PainScore": [0, 1, 2, 3], "D-dimer": 226.0, "AFP": 5.2, "Albumin": 40.0})
    assert np.all(np.diff(predict_risk(base).predicted_risk) > 0)


def test_schema_errors():
    with pytest.raises(SchemaError):
        predict_risk(pd.DataFrame({"Pre_PainScore": [1], "AFP": [5.0], "Albumin": [40.0]}))
    with pytest.raises(SchemaError):
        predict_risk(pd.DataFrame({"Pre_PainScore": [None], "D-dimer": [200.0], "AFP": [5.0], "Albumin": [40.0]}))
    with pytest.raises(SchemaError):
        predict_risk(pd.DataFrame({"Pre_PainScore": [11], "D-dimer": [200.0], "AFP": [5.0], "Albumin": [40.0]}))


def test_extrapolation_warning():
    with pytest.warns(UserWarning, match="extrapolation"):
        predict_risk(pd.DataFrame({"Pre_PainScore": [6], "D-dimer": [200.0], "AFP": [5.0], "Albumin": [40.0]}))


def test_nomogram_points_match_published_table():
    table, _, _ = points_table()
    published = pd.read_csv(ROOT / "results" / "nomogram_points.csv")
    assert len(table) == len(published)
    for term, expected in published.groupby("term"):
        computed = table.loc[table.term.eq(term)]
        assert np.allclose(computed.value, expected.value, atol=1e-9)
        assert np.allclose(computed.points, expected.points, atol=1e-8)
