import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from painpredict import formula_risk, predict_risk
from painpredict.schema import FEATURES

ROOT = Path(__file__).resolve().parents[1]
SCORE = ROOT / "tests" / "web" / "score_cases.mjs"


def _browser_inputs(frame):
    rows = []
    for record in frame.loc[:, list(FEATURES)].to_dict(orient="records"):
        rows.append({name: "" if pd.isna(value) else format(float(value), ".17g") for name, value in record.items()})
    return rows


def _browser_risks(cases):
    completed = subprocess.run(
        ["node", str(SCORE)],
        input=json.dumps(cases),
        text=True,
        capture_output=True,
        cwd=ROOT,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(completed.stderr)
    return np.asarray(json.loads(completed.stdout), dtype=float)


def test_browser_equation_matches_python_and_joblib():
    extra = pd.DataFrame(
        [
            {"Pre_PainScore": 6, "D-dimer": 5000, "AFP": 1000, "Albumin": 20},
            {"Pre_PainScore": 0, "D-dimer": np.nan, "AFP": np.nan, "Albumin": np.nan},
            {"Pre_PainScore": 3, "D-dimer": 14, "AFP": 1.05, "Albumin": 55.35},
            {"Pre_PainScore": 1, "D-dimer": 0, "AFP": 0, "Albumin": 4.2},
        ]
    )
    cohorts = [
        pd.read_csv(ROOT / "examples" / "example_patients.csv"),
        pd.read_csv(ROOT / "examples" / "synthetic_cohort.csv"),
        extra,
    ]
    for frame in cohorts:
        browser = _browser_risks(_browser_inputs(frame))
        assert np.allclose(browser, formula_risk(frame), atol=1e-12, rtol=0)
        assert np.allclose(browser, predict_risk(frame).predicted_risk, atol=1e-9, rtol=0)


def test_static_site_uses_the_published_model_card():
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_web.py")], cwd=ROOT, check=True)
    source = json.loads((ROOT / "src" / "painpredict" / "assets" / "model_card.json").read_text(encoding="utf-8"))
    built = json.loads((ROOT / "site" / "model_card.json").read_text(encoding="utf-8"))
    page = (ROOT / "site" / "index.html").read_text(encoding="utf-8")
    assert built == source
    assert "model_card.json" not in page
    assert (ROOT / "site" / "app.js").is_file()
    assert (ROOT / "site" / ".nojekyll").is_file()
