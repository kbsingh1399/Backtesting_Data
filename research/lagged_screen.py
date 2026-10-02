"""
Second-pass alpha screen with implementation lag built in from the start.
=========================================================================

`reversal_robustness.py` showed that the headline reversal Sharpes were an
artefact of filling at the same close that produced the signal: a single
hour of lag removes ~96% of the gross edge.  That is the signature of
bid-ask bounce / non-synchronous quoting, not of a tradable anomaly.

This pass re-screens every candidate with
  * a mandatory 1-bar (1h) implementation lag,
  * longer horizons where the ~0.8-1.2bp round-turn cost is amortised, and
  * a sub-period split, because a signal that only works in one half of a
    10-month dev block has told us nothing.

Run:
    .venv/bin/python research/lagged_screen.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantlab import config as C                               # noqa: E402
from quantlab.universe import screen_universe                  # noqa: E402
from research.alpha_screen import build_panel, make_signals    # noqa: E402
from research.reversal_robustness import ls_backtest           # noqa: E402

DEV_START, DEV_END = C.WF_TRAIN_START, "2024-06-30"
HORIZONS = {"4h": 4, "1d": 24, "2d": 48, "5d": 120, "10d": 240}


def main() -> None:
    tbl, cms = screen_universe(verbose=False)
    symbols = tbl.loc[tbl["status"] == "keep", "symbol"].tolist()
    P, feats, Cst = build_panel(symbols, cms, DEV_START, DEV_END)
    print(f"panel {P.shape}\n")

    sigs = make_signals(P, feats)
    mid = P.index[len(P) // 2]

    rows = []
    for name, s in sigs.items():
        for hn, h in HORIZONS.items():
            full = ls_backtest(s, P, Cst, h=h, lag=1)
            if not full:
                continue
            h1 = ls_backtest(s[P.index < mid], P[P.index < mid], Cst[P.index < mid], h=h, lag=1)
            h2 = ls_backtest(s[P.index >= mid], P[P.index >= mid], Cst[P.index >= mid], h=h, lag=1)
            rows.append({
                "signal": name, "horizon": hn,
                "gross_bps": full["gross_bps"], "net_bps": full["net_bps"],
                "net_sharpe": full["net_sharpe"],
                "sharpe_H1": h1.get("net_sharpe", np.nan),
                "sharpe_H2": h2.get("net_sharpe", np.nan),
                "n": full["n"],
            })
    res = pd.DataFrame(rows).sort_values("net_sharpe", ascending=False)
    res["both_halves_pos"] = (res["sharpe_H1"] > 0) & (res["sharpe_H2"] > 0)
    C.REPORT_DIR.mkdir(exist_ok=True)
    res.to_csv(C.REPORT_DIR / "lagged_screen.csv", index=False)

    pd.set_option("display.width", 200)
    print("=" * 100)
    print("LAGGED ALPHA SCREEN  (1-bar implementation lag, dev block only)")
    print("=" * 100)
    print(res.head(25).to_string(index=False))
    surv = res[(res["net_sharpe"] > 0.5) & res["both_halves_pos"]]
    print(f"\nsurvivors (net Sharpe > 0.5 AND positive in both halves): {len(surv)} / {len(res)}")
    if len(surv):
        print(surv.to_string(index=False))
    print(f"\ntrials evaluated: {len(res)}")


if __name__ == "__main__":
    main()
