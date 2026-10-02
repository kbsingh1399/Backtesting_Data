"""
Control experiments for the S5 primary rule (development block only).
=====================================================================

Before believing *or* dismissing a backtest number, you have to know what the
same machinery produces on inputs that cannot contain alpha.  This script runs
four controls on identical exit logic and identical costs:

  1. RULE        - the sweep events, as the rule specifies them
  2. RULE-FLIP   - the same events with the side reversed
  3. RANDOM      - random entry bars drawn from the same session/hour mix
  4. EVERY-N     - a systematic entry every N bars inside the session

If RULE and RANDOM are statistically indistinguishable, the rule carries no
information and nothing downstream can fix it.  If RULE and RULE-FLIP are both
negative by the same amount, the loss is a *cost/geometry* artefact, not a
directional signal.

It also sanity-checks that the price series themselves are real.

Run:
    .venv/bin/python research/diagnostics.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantlab import config as C                       # noqa: E402
from quantlab.labeling import simulate_events          # noqa: E402
from quantlab.pipeline import cost_per_atr, get_enriched, summarise  # noqa: E402
from quantlab.signals import detect_sweep_events, _session_mask      # noqa: E402
from quantlab.universe import screen_universe          # noqa: E402

DEV_END = "2024-06-30"
RNG = np.random.default_rng(12345)


def synthetic_events(enr: pd.DataFrame, params, n_target: int, mode: str) -> pd.DataFrame:
    """Entries that contain no sweep information but share session structure."""
    sess = _session_mask(enr, params)
    ok = np.where(sess & np.isfinite(enr["atr14"].to_numpy()) & (enr["atr14"].to_numpy() > 0))[0]
    ok = ok[(ok > 300) & (ok < len(enr) - 2)]
    if len(ok) == 0:
        return pd.DataFrame()
    if mode == "random":
        k = min(n_target, len(ok))
        idx = np.sort(RNG.choice(ok, size=k, replace=False))
    else:
        step = max(1, len(ok) // max(n_target, 1))
        idx = ok[::step]
    side = RNG.choice([-1, 1], size=len(idx))
    a = enr["atr14"].to_numpy()[idx]
    o = enr["open"].to_numpy()[idx + 1]
    lo = enr["low"].to_numpy()[idx]
    hi = enr["high"].to_numpy()[idx]
    struct = np.where(side == 1, o - (lo - params.stop_buffer_atr * a),
                      (hi + params.stop_buffer_atr * a) - o)
    return pd.DataFrame({
        "i": idx, "entry_i": idx + 1,
        "signal_time": enr["datetime"].to_numpy()[idx],
        "entry_time": enr["datetime"].to_numpy()[idx + 1],
        "side": side, "atr": a, "struct_r_atr": struct / a,
    })


def main() -> None:
    tbl, cms = screen_universe(verbose=False)
    symbols = tbl.loc[tbl["status"] == "keep", "symbol"].tolist()
    params = C.FROZEN_PARAMS

    print("=" * 96)
    print("PRICE-SERIES SANITY (pure 15m era)")
    print("=" * 96)
    for s in ["XAUUSD", "EURUSD", "USDJPY", "NAS100"]:
        e = get_enriched(s)
        px = e["close"]
        lr = np.log(px).diff().dropna()
        print(f"  {s:8s} {e.datetime.iloc[0].date()}..{e.datetime.iloc[-1].date()} "
              f"first={px.iloc[0]:>10.2f} last={px.iloc[-1]:>10.2f} "
              f"tot={px.iloc[-1]/px.iloc[0]-1:+7.1%}  ann.vol={lr.std()*np.sqrt(96*252):5.1%}  "
              f"AC(1)={lr.autocorr(1):+.4f}  AC(2)={lr.autocorr(2):+.4f}")

    buckets = {k: [] for k in ("RULE", "RULE-FLIP", "RANDOM", "EVERY-N")}
    for sym in symbols:
        cm = cms[sym]
        enr = get_enriched(sym)
        enr = enr[enr["datetime"] <= pd.Timestamp(DEV_END, tz="UTC")].reset_index(drop=True)
        if len(enr) < 2000:
            continue
        cpa = cost_per_atr(cm, enr)
        ev = detect_sweep_events(enr, params, cm.point, 5, cost_per_atr=cpa, enriched=enr)
        if ev.empty:
            continue
        n = len(ev)
        tr = simulate_events(enr, ev, params, cm); tr["symbol"] = sym
        buckets["RULE"].append(tr)
        ev2 = ev.copy(); ev2["side"] = -ev2["side"]
        # recompute the structural stop for the flipped side
        a = ev2["atr"].to_numpy(); i = ev2["i"].to_numpy()
        o = enr["open"].to_numpy()[i + 1]
        lo = enr["low"].to_numpy()[i]; hi = enr["high"].to_numpy()[i]
        ev2["struct_r_atr"] = np.where(
            ev2["side"] == 1, o - (lo - params.stop_buffer_atr * a),
            (hi + params.stop_buffer_atr * a) - o) / a
        t2 = simulate_events(enr, ev2, params, cm); t2["symbol"] = sym
        buckets["RULE-FLIP"].append(t2)
        for mode, key in (("random", "RANDOM"), ("everyn", "EVERY-N")):
            se = synthetic_events(enr, params, n, mode)
            if se.empty:
                continue
            t3 = simulate_events(enr, se, params, cm); t3["symbol"] = sym
            buckets[key].append(t3)

    print("\n" + "=" * 96)
    print(f"CONTROL EXPERIMENTS  |  dev block {C.WF_TRAIN_START} .. {DEV_END}  |  identical exits & costs")
    print("=" * 96)
    print(f"{'bucket':<12} {'n':>6} {'WR%':>6} {'E[R_net]':>10} {'E[R_gross]':>11} "
          f"{'t(net)':>7} {'t(gross)':>9} {'PF':>6} {'cost_R':>7}")
    rows = []
    for k, fr in buckets.items():
        if not fr:
            continue
        d = pd.concat(fr, ignore_index=True)
        d = d[d["entry_time"] >= pd.Timestamp(C.WF_TRAIN_START, tz="UTC")]
        rn, rg = d["r_net"].to_numpy(), d["r_gross"].to_numpy()
        tn = rn.mean() / (rn.std(ddof=1) / np.sqrt(len(rn)))
        tg = rg.mean() / (rg.std(ddof=1) / np.sqrt(len(rg)))
        gp, gl = rn[rn > 0].sum(), -rn[rn <= 0].sum()
        print(f"{k:<12} {len(d):>6} {100*(rn>0).mean():>6.1f} {rn.mean():>+10.4f} "
              f"{rg.mean():>+11.4f} {tn:>+7.2f} {tg:>+9.2f} {gp/max(gl,1e-9):>6.3f} "
              f"{d['cost_r'].mean():>7.4f}")
        rows.append(dict(bucket=k, n=int(len(d)), win_rate=float((rn > 0).mean()),
                         E_R_net=float(rn.mean()), E_R_gross=float(rg.mean()),
                         t_net=float(tn), t_gross=float(tg)))

    # Difference-in-means test: RULE vs RANDOM on gross R
    if buckets["RULE"] and buckets["RANDOM"]:
        A = pd.concat(buckets["RULE"])["r_gross"].to_numpy()
        B = pd.concat(buckets["RANDOM"])["r_gross"].to_numpy()
        se = np.sqrt(A.var(ddof=1) / len(A) + B.var(ddof=1) / len(B))
        print(f"\n  RULE - RANDOM gross edge = {A.mean()-B.mean():+.4f}R   "
              f"(SE {se:.4f},  t = {(A.mean()-B.mean())/se:+.2f})")

    C.REPORT_DIR.mkdir(exist_ok=True)
    (C.REPORT_DIR / "diagnostics.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
