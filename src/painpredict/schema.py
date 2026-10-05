"""Input schema for the four predictors and the 24-hour outcome."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Predictor:
    column: str
    label: str
    unit: str
    lower: float
    upper: float
    development_range: tuple[float, float]


# development_range: values observed in the development cohort after IQR capping.
PREDICTORS = (
    Predictor("Pre_PainScore", "Preprocedural pain (NRS)", "0-10 points", 0, 10, (0, 3)),
    Predictor("D-dimer", "D-dimer", "ng/mL", 0, np.inf, (14, 962)),
    Predictor("AFP", "Alpha-fetoprotein", "ng/mL", 0, np.inf, (1.05, 152.57)),
    Predictor("Albumin", "Albumin", "g/L", 0, np.inf, (24.1, 55.35)),
)
FEATURES = tuple(p.column for p in PREDICTORS)

EVENT = "event"
OUTCOME_NRS = "nrs_24h"
EVENT_THRESHOLD = 4


class SchemaError(ValueError):
    """Raised when an input table cannot be scored or validated."""


def prepare_predictors(frame: pd.DataFrame, *, warn: bool = True) -> pd.DataFrame:
    """Return the four predictor columns as floats, checking names and ranges.

    Missing D-dimer, AFP or albumin values are allowed; the model imputes them.
    Preprocedural NRS must be present.
    """

    absent = [column for column in FEATURES if column not in frame.columns]
    if absent:
        raise SchemaError(f"Missing predictor columns: {absent}. Required columns: {list(FEATURES)}.")
    x = frame.loc[:, list(FEATURES)].apply(pd.to_numeric, errors="coerce").astype(float)
    if x["Pre_PainScore"].isna().any():
        raise SchemaError("Pre_PainScore must be recorded for every row.")
    for predictor in PREDICTORS:
        values = x[predictor.column].dropna()
        if ((values < predictor.lower) | (values > predictor.upper)).any():
            raise SchemaError(f"{predictor.column} has values outside {predictor.lower}-{predictor.upper} ({predictor.unit}).")
    if warn:
        low, high = PREDICTORS[0].development_range
        outside = x["Pre_PainScore"].gt(high).sum()
        if outside:
            warnings.warn(
                f"{outside} row(s) have preprocedural NRS > {high:g}, above the range seen in the development "
                "cohort; predicted risks for these rows are extrapolations.",
                stacklevel=2,
            )
    return x


def outcome_events(frame: pd.DataFrame) -> np.ndarray:
    """Binary outcome from ``event`` (0/1) or from ``nrs_24h`` (event = NRS >= 4)."""

    if EVENT in frame.columns:
        event = pd.to_numeric(frame[EVENT], errors="coerce")
        if event.isna().any() or not event.isin([0, 1]).all():
            raise SchemaError("Column 'event' must contain only 0 and 1.")
        return event.to_numpy(dtype=int)
    if OUTCOME_NRS in frame.columns:
        nrs = pd.to_numeric(frame[OUTCOME_NRS], errors="coerce")
        if nrs.isna().any() or not nrs.between(0, 10).all():
            raise SchemaError("Column 'nrs_24h' must contain NRS values from 0 to 10.")
        return nrs.ge(EVENT_THRESHOLD).to_numpy(dtype=int)
    raise SchemaError("Provide the outcome as 'event' (0/1) or 'nrs_24h' (0-10).")
