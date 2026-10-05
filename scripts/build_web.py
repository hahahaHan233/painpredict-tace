"""Assemble the static calculator for local preview and GitHub Pages."""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "web"
SITE = ROOT / "site"
MODEL_CARD = ROOT / "src" / "painpredict" / "assets" / "model_card.json"
PAGES = ("index.html", "styles.css", "app.js", "model.js", "i18n.js")


def build() -> Path:
    if not MODEL_CARD.is_file():
        raise FileNotFoundError(MODEL_CARD)
    if SITE.exists():
        shutil.rmtree(SITE)
    SITE.mkdir()
    for name in PAGES:
        shutil.copy2(SOURCE / name, SITE / name)
    shutil.copy2(MODEL_CARD, SITE / "model_card.json")
    (SITE / ".nojekyll").write_text("", encoding="utf-8")
    return SITE


if __name__ == "__main__":
    print(f"Built {build()}")
