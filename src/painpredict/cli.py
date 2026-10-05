"""Command-line interface: ``painpredict predict | validate | synth | nomogram``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

from . import __version__
from .predict import DEFAULT_THRESHOLD, load_model, predict_risk
from .schema import SchemaError


def _predict(args) -> None:
    frame = pd.read_csv(args.input)
    model = load_model(args.model) if args.model else None
    result = frame.join(predict_risk(frame, model=model, threshold=args.threshold))
    if args.output:
        result.to_csv(args.output, index=False)
        print(f"Wrote {len(result)} predictions to {args.output}")
    else:
        result.to_csv(sys.stdout, index=False)


def _validate(args) -> None:
    from .validation import bootstrap_performance, fit_final_model, repeated_nested_cv

    frame = pd.read_csv(args.input)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    result = repeated_nested_cv(frame, repeats=args.repeats, n_jobs=args.jobs)
    result.predictions.to_csv(out / "oof_predictions.csv", index=False)
    result.folds.to_csv(out / "fold_audit.csv", index=False)
    summary = bootstrap_performance(
        result.predictions.event.to_numpy(), result.predictions.probability.to_numpy(),
        n_bootstrap=args.bootstrap, threshold=args.threshold,
    )
    summary.to_csv(out / "performance.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if args.fit_final:
        joblib.dump(fit_final_model(frame, n_jobs=args.jobs), out / "model.joblib")
        print(f"Saved refitted model to {out / 'model.joblib'}")


def _synth(args) -> None:
    from .synthetic import make_synthetic_cohort

    frame = make_synthetic_cohort(args.n, args.seed)
    frame.to_csv(args.output, index=False)
    print(f"Wrote {len(frame)} synthetic rows ({frame.event.mean():.1%} events) to {args.output}")


def _nomogram(args) -> None:
    from .nomogram import draw_nomogram

    draw_nomogram(args.output)
    print(f"Saved nomogram to {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="painpredict", description="Risk of NRS >= 4 at 24 h after TACE.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    predict = commands.add_parser("predict", help="score a CSV of patients with the packaged model")
    predict.add_argument("input", help="CSV with Pre_PainScore, D-dimer, AFP and Albumin columns")
    predict.add_argument("-o", "--output", help="output CSV (default: print to stdout)")
    predict.add_argument("--model", help="use another fitted pipeline (joblib) instead of the packaged model")
    predict.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    predict.set_defaults(func=_predict)

    validate = commands.add_parser("validate", help="repeated nested CV of the model specification on your data")
    validate.add_argument("input", help="CSV with the four predictors and 'event' (0/1) or 'nrs_24h' (0-10)")
    validate.add_argument("-o", "--output", default="runs/validation", help="output directory")
    validate.add_argument("--repeats", type=int, default=5)
    validate.add_argument("--bootstrap", type=int, default=2000)
    validate.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    validate.add_argument("--jobs", type=int, default=None, help="parallel jobs for the inner grid search")
    validate.add_argument("--fit-final", action="store_true", help="also refit on all rows and save model.joblib")
    validate.set_defaults(func=_validate)

    synth = commands.add_parser("synth", help="write a synthetic demonstration cohort")
    synth.add_argument("-o", "--output", default="synthetic_cohort.csv")
    synth.add_argument("-n", type=int, default=356)
    synth.add_argument("--seed", type=int, default=1)
    synth.set_defaults(func=_synth)

    nomogram = commands.add_parser("nomogram", help="draw the points nomogram into png, pdf and svg subfolders")
    nomogram.add_argument("-o", "--output", default="figures")
    nomogram.set_defaults(func=_nomogram)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except SchemaError as error:
        print(f"Input error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
