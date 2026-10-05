"""Synthetic cohorts for demonstrations and tests.

Values are random draws shaped to resemble the published cohort summary
(medians, interquartile ranges, missingness and an event rate near 21%).
They do not correspond to any real patient and carry no clinical meaning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .predict import formula_risk

TARGET_EVENT_RATE = 0.21


def _logit(p):
    return np.log(p / (1 - p))


def make_synthetic_cohort(n: int = 356, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "Pre_PainScore": rng.choice([0, 1, 2, 3], size=n, p=[0.80, 0.10, 0.07, 0.03]).astype(float),
            "D-dimer": np.round(np.exp(rng.normal(np.log(226), 0.95, n)).clip(14, 5000)),
            "AFP": np.round(np.exp(rng.normal(np.log(5.2), 2.26, n)).clip(1.0, 60000), 2),
            "Albumin": np.round(rng.normal(39.5, 6.0, n).clip(24, 55), 1),
        }
    )
    for column, rate in (("D-dimer", 0.076), ("AFP", 0.059), ("Albumin", 0.003)):
        frame.loc[rng.random(n) < rate, column] = np.nan

    base = _logit(formula_risk(frame).to_numpy())
    shift = _calibrate_shift(base, TARGET_EVENT_RATE)
    risk = 1 / (1 + np.exp(-(base + shift)))
    event = rng.random(n) < risk
    nrs = np.where(
        event,
        rng.choice([4, 5, 6, 7], size=n, p=[0.55, 0.30, 0.10, 0.05]),
        rng.choice([0, 1, 2, 3], size=n, p=[0.13, 0.30, 0.35, 0.22]),
    )
    frame.insert(0, "synthetic_id", [f"S{i:04d}" for i in range(1, n + 1)])
    frame["nrs_24h"] = nrs
    frame["event"] = event.astype(int)
    return frame


def _calibrate_shift(base_logit: np.ndarray, target: float) -> float:
    low, high = -10.0, 10.0
    for _ in range(80):
        mid = (low + high) / 2
        if np.mean(1 / (1 + np.exp(-(base_logit + mid)))) < target:
            low = mid
        else:
            high = mid
    return (low + high) / 2
