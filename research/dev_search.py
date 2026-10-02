"""
In-sample design search for the S5 primary rule.
================================================

**This script is the only place where strategy parameters are allowed to be
chosen.**  It runs exclusively on the development block

        2023-09-15  ..  2024-06-30

and the bar series itself is truncated at the boundary, so no path simulation
can run into out-of-sample data.  Whatever this script selects is frozen into
`quantlab.config.FROZEN_PARAMS` and never revisited.

The search is **coordinate-wise, not a full grid**, for two reasons:
  1. it is ~8x cheaper, and
  2. the number of configurations actually evaluated is the `n_trials` that
     goes into the Deflated Sharpe Ratio.  Honest accounting of trials is the
     whole point; a 10,000-cell grid would make the DSR meaningless.

Run:
    .venv/bin/python research/dev_search.py
"""
from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantlab import config as C                     # noqa: E402
from quantlab.pipeline import build_trades, summarise  # noqa: E402
from quantlab.universe import screen_universe        # noqa: E402

DEV_START = C.WF_TRAIN_START
DEV_END = "2024-06-30"


def evaluate(symbols, cms, params: C.StrategyParams, label: str) -> Dict:
    tr = build_trades(symbols, params, cms, end=DEV_END)
    if tr.empty:
        return {"label": label, "n": 0}
    tr = tr[tr["entry_time"] >= pd.Timestamp(DEV_START, tz="UTC")]
    s = summarise(tr, label)
    s["n_symbols"] = int(tr["symbol"].nunique())
    return s


def main() -> None:
    t0 = time.time()
    tbl, cms = screen_universe(verbose=True)
    symbols = tbl.loc[tbl["status"] == "keep", "symbol"].tolist()

    trials: List[Dict] = []
    base = C.StrategyParams()

    def run(tag: str, **over) -> Dict:
        p = dataclasses.replace(base, **over)
        s = evaluate(symbols, cms, p, tag)
        s.update(over)
        trials.append(s)
        print(f"  {tag:<34s} n={s.get('n',0):>5}  WR={s.get('win_rate',0):>5.1f}%  "
              f"E[R]={s.get('avg_R',0):+.4f}  PF={s.get('profit_factor',0):.3f}  "
              f"t={s.get('t_stat',float('nan')):+.2f}")
        return s

    # ---- Stage A: direction + payoff -----------------------------------
    print("\n[Stage A] polarity x take-profit")
    best_a, best_score = None, -np.inf
    for pol in ("fade", "continuation"):
        for tp in (1.5, 2.0, 3.0):
            s = run(f"A pol={pol} tp={tp}", polarity=pol, take_profit_r=tp)
            score = s.get("avg_R", -9) * np.sqrt(max(s.get("n", 0), 1))
            if s.get("n", 0) > 300 and score > best_score:
                best_a, best_score = {"polarity": pol, "take_profit_r": tp}, score
    base = dataclasses.replace(base, **best_a)
    print(f"  -> {best_a}")

    # ---- Stage B: exit management --------------------------------------
    print("\n[Stage B] ratchet")
    ratchets = {
        "none": (),
        "breakeven@1R": ((1.0, 0.0),),
        "lock .10@1R / .75@1.5R": ((1.0, 0.10), (1.5, 0.75)),
        "lock .50@1.5R": ((1.5, 0.50),),
    }
    best_b, best_score = None, -np.inf
    for tag, rr in ratchets.items():
        s = run(f"B ratchet={tag}", ratchet=rr)
        score = s.get("avg_R", -9) * np.sqrt(max(s.get("n", 0), 1))
        if s.get("n", 0) > 300 and score > best_score:
            best_b, best_score = rr, score
    base = dataclasses.replace(base, ratchet=best_b)
    print(f"  -> ratchet={best_b}")

    # ---- Stage C: sweep geometry ---------------------------------------
    print("\n[Stage C] penetration band + reclaim strength")
    best_c, best_score = None, -np.inf
    for mn, mx in ((0.05, 1.5), (0.10, 1.0), (0.20, 1.5), (0.10, 2.5)):
        for rec in (0.40, 0.55, 0.70):
            s = run(f"C pen=[{mn},{mx}] reclaim={rec}",
                    min_penetration_atr=mn, max_penetration_atr=mx, min_reclaim_frac=rec)
            score = s.get("avg_R", -9) * np.sqrt(max(s.get("n", 0), 1))
            if s.get("n", 0) > 300 and score > best_score:
                best_c = {"min_penetration_atr": mn, "max_penetration_atr": mx,
                          "min_reclaim_frac": rec}
                best_score = score
    base = dataclasses.replace(base, **best_c)
    print(f"  -> {best_c}")

    # ---- Stage D: holding horizon --------------------------------------
    print("\n[Stage D] time stop")
    best_d, best_score = None, -np.inf
    for ts in (16, 32, 64, 96):
        s = run(f"D time_stop={ts}", time_stop_bars=ts)
        score = s.get("avg_R", -9) * np.sqrt(max(s.get("n", 0), 1))
        if s.get("n", 0) > 300 and score > best_score:
            best_d, best_score = ts, score
    base = dataclasses.replace(base, time_stop_bars=best_d)
    print(f"  -> time_stop_bars={best_d}")

    # ---- Stage E: stop floor (a cost decision) --------------------------
    print("\n[Stage E] stop floor")
    best_e, best_score = None, -np.inf
    for ms in (1.0, 1.5, 2.0, 2.5):
        s = run(f"E min_stop_atr={ms}", min_stop_atr=ms)
        score = s.get("avg_R", -9) * np.sqrt(max(s.get("n", 0), 1))
        if s.get("n", 0) > 300 and score > best_score:
            best_e, best_score = ms, score
    base = dataclasses.replace(base, min_stop_atr=best_e)
    print(f"  -> min_stop_atr={best_e}")

    final = evaluate(symbols, cms, base, "FROZEN")
    trials.append(final)

    out = {
        "dev_window": [DEV_START, DEV_END],
        "n_trials": len(trials),
        "frozen_params": dataclasses.asdict(base),
        "final_dev_summary": final,
        "trials": trials,
    }
    C.REPORT_DIR.mkdir(exist_ok=True)
    (C.REPORT_DIR / "dev_search.json").write_text(json.dumps(out, indent=2, default=str))
    pd.DataFrame(trials).to_csv(C.REPORT_DIR / "dev_search_trials.csv", index=False)

    print("\n" + "=" * 78)
    print(f"TRIALS EVALUATED (feeds Deflated Sharpe): {len(trials)}")
    print(json.dumps(dataclasses.asdict(base), indent=2, default=str))
    print(f"\nFrozen rule on DEV: {final}")
    print(f"elapsed {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
