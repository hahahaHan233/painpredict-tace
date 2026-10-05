from pathlib import Path

import pandas as pd

from painpredict.cli import main

ROOT = Path(__file__).resolve().parents[1]


def test_predict_command(tmp_path):
    output = tmp_path / "predictions.csv"
    assert main(["predict", str(ROOT / "examples" / "example_patients.csv"), "-o", str(output)]) == 0
    result = pd.read_csv(output)
    assert {"predicted_risk", "above_threshold"} <= set(result.columns)


def test_predict_reports_bad_input(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"Pre_PainScore": [1]}).to_csv(bad, index=False)
    assert main(["predict", str(bad)]) == 2


def test_synth_and_validate_commands(tmp_path):
    cohort = tmp_path / "cohort.csv"
    assert main(["synth", "-o", str(cohort), "-n", "150", "--seed", "5"]) == 0
    out = tmp_path / "run"
    assert main(["validate", str(cohort), "-o", str(out), "--repeats", "1", "--bootstrap", "30"]) == 0
    assert (out / "performance.csv").exists() and (out / "oof_predictions.csv").exists()
