"""Score new patients with the packaged four-variable ridge logistic model."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from importlib import resources
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .schema import FEATURES, prepare_predictors

DEFAULT_THRESHOLD = 0.20


def _asset(name: str) -> Path:
    return Path(str(resources.files("painpredict") / "assets" / name))


@lru_cache(maxsize=1)
def model_card() -> dict:
    """Metadata, coefficients and development summary of the packaged model."""

    return json.loads(_asset("model_card.json").read_text(encoding="utf-8"))


def load_model(path: str | Path | None = None):
    """Load the fitted scikit-learn pipeline (packaged model by default)."""

    model_path = Path(path) if path is not None else _asset("model.joblib")
    if path is None:
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if digest != model_card()["artifact"]["sha256"]:
            raise RuntimeError("Packaged model file does not match the checksum in model_card.json.")
    return joblib.load(model_path)


def predict_risk(frame: pd.DataFrame, *, model=None, threshold: float = DEFAULT_THRESHOLD) -> pd.DataFrame:
    """Return the predicted probability of NRS >= 4 at 24 h for each row.

    The output keeps the input index and adds ``predicted_risk`` and
    ``above_threshold`` (risk >= ``threshold``; 0.20 is the operating point
    reported for the development cohort).
    """

    x = prepare_predictors(frame)
    model = load_model() if model is None else model
    risk = model.predict_proba(x)[:, 1]
    return pd.DataFrame({"predicted_risk": risk, "above_threshold": risk >= threshold}, index=frame.index)


def formula_risk(frame: pd.DataFrame) -> pd.Series:
    """Compute the same risk by hand from the published raw-scale equation.

    Each predictor is clipped to its capping bounds, a missing value is replaced
    by the development median and its missing indicator is set to 1, and the
    linear predictor is passed through the logistic function.
    """

    card = model_card()["equation"]
    x = prepare_predictors(frame, warn=False)
    logit = np.full(len(x), card["intercept"], dtype=float)
    for term in card["terms"]:
        values = x[term["variable"]]
        missing = values.isna().to_numpy()
        filled = values.fillna(term["median"]).clip(term["cap_lower"], term["cap_upper"]).to_numpy()
        logit += term["coefficient"] * filled + term["missing_coefficient"] * missing
    return pd.Series(1 / (1 + np.exp(-logit)), index=frame.index, name="predicted_risk")


__all__ = ["FEATURES", "DEFAULT_THRESHOLD", "formula_risk", "load_model", "model_card", "predict_risk"]
