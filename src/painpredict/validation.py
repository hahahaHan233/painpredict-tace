"""Repeated nested cross-validation and out-of-fold performance metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .modeling import RANDOM_STATE, make_search
from .schema import FEATURES, outcome_events, prepare_predictors


@dataclass
class NestedCVResult:
    predictions: pd.DataFrame
    folds: pd.DataFrame


def repeated_nested_cv(
    frame: pd.DataFrame,
    *,
    repeats: int = 5,
    outer_folds: int = 5,
    inner_folds: int = 4,
    seed: int = RANDOM_STATE,
    n_jobs: int | None = None,
) -> NestedCVResult:
    """Out-of-fold risks from repeated stratified nested cross-validation.

    All preprocessing (capping, imputation, scaling) and the choice of C are
    learned inside each outer training partition. Each row's out-of-fold risk
    is averaged over repeats.
    """

    x = prepare_predictors(frame, warn=False).reset_index(drop=True)
    y = outcome_events(frame)
    rows, folds = [], []
    for repeat in range(repeats):
        outer = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed + repeat)
        for fold, (train, test) in enumerate(outer.split(x, y), start=1):
            inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed + repeat * 31 + fold)
            search = make_search(FEATURES, list(inner.split(x.iloc[train], y[train])), n_jobs=n_jobs)
            search.fit(x.iloc[train], y[train])
            rows.append(
                pd.DataFrame(
                    {
                        "row": test,
                        "repeat": repeat + 1,
                        "fold": fold,
                        "event": y[test],
                        "probability": search.predict_proba(x.iloc[test])[:, 1],
                    }
                )
            )
            folds.append(
                {
                    "repeat": repeat + 1,
                    "fold": fold,
                    "n_train": len(train),
                    "n_test": len(test),
                    "events_test": int(y[test].sum()),
                    "train_test_overlap": len(np.intersect1d(train, test)),
                    "best_C": float(search.best_params_["model__C"]),
                }
            )
    raw = pd.concat(rows, ignore_index=True)
    predictions = raw.groupby("row", as_index=False).agg(event=("event", "first"), probability=("probability", "mean"))
    return NestedCVResult(predictions=predictions, folds=pd.DataFrame(folds))


def calibration_intercept_slope(event: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    fit = LogisticRegression(penalty=None, solver="lbfgs", max_iter=20_000).fit(logit, event)
    return float(fit.intercept_[0]), float(fit.coef_[0, 0])


def performance(event: np.ndarray, probability: np.ndarray, threshold: float = 0.20) -> dict[str, float]:
    """Discrimination, probability error, calibration and threshold metrics."""

    event = np.asarray(event, dtype=int)
    probability = np.asarray(probability, dtype=float)
    tn, fp, fn, tp = confusion_matrix(event, probability >= threshold, labels=[0, 1]).ravel()
    intercept, slope = calibration_intercept_slope(event, probability)
    return {
        "auc": roc_auc_score(event, probability),
        "pr_auc": average_precision_score(event, probability),
        "brier": brier_score_loss(event, probability),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "ppv": tp / (tp + fp) if tp + fp else np.nan,
        "npv": tn / (tn + fn) if tn + fn else np.nan,
    }


def bootstrap_performance(
    event: np.ndarray,
    probability: np.ndarray,
    *,
    n_bootstrap: int = 2000,
    threshold: float = 0.20,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Point estimates with percentile 95% CIs from patient-level bootstrap."""

    event = np.asarray(event, dtype=int)
    probability = np.asarray(probability, dtype=float)
    point = performance(event, probability, threshold)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_bootstrap):
        index = rng.integers(0, len(event), len(event))
        if np.unique(event[index]).size < 2:
            continue
        draws.append(performance(event[index], probability[index], threshold))
    draws = pd.DataFrame(draws)
    return pd.DataFrame(
        {
            "metric": list(point),
            "estimate": list(point.values()),
            "ci_low": [draws[name].quantile(0.025) for name in point],
            "ci_high": [draws[name].quantile(0.975) for name in point],
        }
    )


def decision_curve(event: np.ndarray, probability: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    """Net benefit of the model, treat-all and treat-none across risk thresholds."""

    event = np.asarray(event, dtype=bool)
    probability = np.asarray(probability, dtype=float)
    n, prevalence = len(event), event.mean()
    rows = []
    for t in thresholds:
        positive = probability >= t
        tp, fp = np.sum(positive & event), np.sum(positive & ~event)
        rows.append(
            {
                "threshold": t,
                "model": tp / n - fp / n * t / (1 - t),
                "treat_all": prevalence - (1 - prevalence) * t / (1 - t),
                "treat_none": 0.0,
            }
        )
    return pd.DataFrame(rows)


def fit_final_model(frame: pd.DataFrame, *, folds: int = 5, seed: int = RANDOM_STATE, n_jobs: int | None = None):
    """Refit on all rows with C tuned by stratified CV; report performance from nested CV, not from this fit."""

    x = prepare_predictors(frame, warn=False)
    y = outcome_events(frame)
    cv = list(StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(x, y))
    return make_search(FEATURES, cv, n_jobs=n_jobs).fit(x, y).best_estimator_
