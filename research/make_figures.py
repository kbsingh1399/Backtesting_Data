"""Assemble the research figure from the saved experiment outputs.

    .venv/bin/python research/make_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantlab import config as C            # noqa: E402
from quantlab.reporting import research_figure  # noqa: E402


def main() -> None:
    R = C.REPORT_DIR
    controls = pd.DataFrame(json.loads((R / "diagnostics.json").read_text()))

    rr = json.loads((R / "reversal_robustness.json").read_text())
    lag_key = next(k for k in rr if k.startswith("T1"))
    lag = pd.DataFrame(rr[lag_key])

    vr = json.loads((R / "vwap_robustness.json").read_text())
    uni_key = next(k for k in vr if k.startswith("U"))
    uni = pd.DataFrame(vr[uni_key])

    p = research_figure(controls, lag, uni, R / "research_findings.png")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
