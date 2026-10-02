"""
Robustness interrogation of the short-horizon reversal signal.
==============================================================

The alpha screen reports net Sharpe ratios of 4-8 for cross-sectional
short-horizon reversal on the dev block.  Numbers that size are, in this
business, usually an artefact.  This script tries to kill the signal with the
five tests that normally succeed:

  T1 IMPLEMENTATION LAG  - the screen lets you trade at the same close that
     produced the signal.  Push the fill out by 1 and 2 hourly bars.  Most
     "reversal" alpha is really bid-ask bounce and dies instantly here.

  T2 SUB-UNIVERSE        - 8 of the 46 instruments are gold crosses and 5 are
     silver crosses.  A cross-sectional book can "mean revert" XAUUSD against
     XAUEUR, which is just the EURUSD leg.  Re-run on FX only, metals only,
     indices only, and on one representative per correlation cluster.

  T3 STALE QUOTES        - index CFDs quote outside their cash session; a
     stale quote mechanically reverts when the real market reopens.  Re-run
     excluding bars with no tick activity, and excluding each index outside
     its own exchange hours.

  T4 COST STRESS         - multiply the modelled cost by 1.5x, 2x, 3x.  This
     is the crude stand-in for market impact, which the quoted spread does
     not capture.

  T5 SUB-PERIOD          - split the dev block in half.  An effect that only
     exists in one half is not an effect.

Run:
    .venv/bin/python research/reversal_robustness.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantlab import config as C                 # noqa: E402
from quantlab.pipeline import get_enriched       # noqa: E402
from quantlab.universe import screen_universe    # noqa: E402

DEV_START, DEV_END = C.WF_TRAIN_START, "2024-06-30"
BARS_PER_YEAR = 24 * 252

# Approximate cash-session hours (UTC) for the index CFDs; outside these the
# quote is a derived/stale price and any "reversion" is an artefact.
INDEX_HOURS = {
    "SP500": (13, 21), "NAS100": (13, 21), "DJ30": (13, 21), "US2000": (13, 21),
    "GER40": (7, 16), "FR40": (7, 16), "UK100": (7, 16),
    "AU200": (23, 6), "JP225": (0, 7),
}


def build_panel(symbols, cms, start, end):
    px, cost, tv = {}, {}, {}
    for s in symbols:
        e = get_enriched(s)
        e = e[(e["datetime"] >= pd.Timestamp(start, tz="UTC")) &
              (e["datetime"] <= pd.Timestamp(end, tz="UTC"))]
        e = e[e["datetime"].dt.minute == 0].set_index("datetime")
        if len(e) < 500:
            continue
        px[s] = e["close"]
        tv[s] = e["tick_volume"]
        cm = cms[s]
        sp = cm.spread_price(e["spread"].to_numpy(), e.index.hour.to_numpy())
        rt = sp * (1 + C.SLIPPAGE_ENTRY_SPREADS + C.SLIPPAGE_TIMEEXIT_SPREADS) \
            + e["close"].to_numpy() * cm.commission_bps_rt / 1e4
        cost[s] = pd.Series(rt / e["close"].to_numpy() * 1e4, index=e.index)
    return pd.DataFrame(px).sort_index(), pd.DataFrame(cost).sort_index(), pd.DataFrame(tv).sort_index()


def reversal_signal(P: pd.DataFrame, k: int = 2) -> pd.DataFrame:
    lr = np.log(P).diff()
    vol = lr.rolling(120, min_periods=60).std()
    return -(np.log(P).diff(k) / (vol + 1e-12))


def ls_backtest(sig, P, Cst, h=1, lag=0, cost_mult=1.0, mask=None) -> Dict:
    """Dollar-neutral decile book. `lag` = hourly bars between the signal's
    close and the fill."""
    fwd = np.log(P).shift(-h) - np.log(P)
    S = sig.shift(lag)
    if mask is not None:
        S = S.where(mask)
    valid = S.notna() & fwd.notna() & P.notna()
    S, F = S.where(valid), fwd.where(valid)
    keep = valid.sum(axis=1) >= 8
    S, F = S[keep], F[keep]
    if len(S) < 60:
        return {}
    Z = S.sub(S.mean(axis=1), axis=0).div(S.std(axis=1) + 1e-12, axis=0)
    W = Z.div(Z.abs().sum(axis=1), axis=0).fillna(0.0)
    W, Fh = W.iloc[::h], F.iloc[::h]
    Ch = Cst.reindex_like(F).iloc[::h]
    Ch = Ch.fillna(Ch.stack().median()) * cost_mult
    gross = (W * Fh).sum(axis=1) * 1e4
    turn = W.diff().abs().fillna(W.abs()) * 0.5
    net = (gross - (turn * Ch).sum(axis=1)).dropna()
    if len(net) < 30:
        return {}
    py = BARS_PER_YEAR / h
    return {
        "n": int(len(net)),
        "gross_bps": round(float(gross.mean()), 3),
        "net_bps": round(float(net.mean()), 3),
        "net_sharpe": round(float(net.mean() / (net.std(ddof=1) + 1e-12) * np.sqrt(py)), 2),
        "ann_ret_pct": round(float(net.mean() * py / 100), 1),
    }


def main() -> None:
    tbl, cms = screen_universe(verbose=False)
    keep = tbl[tbl["status"] == "keep"]
    symbols = keep["symbol"].tolist()
    P, Cst, TV = build_panel(symbols, cms, DEV_START, DEV_END)
    cols = P.columns.tolist()
    print(f"panel {P.shape}  |  {len(cols)} instruments on a common hourly clock\n")

    sig = reversal_signal(P, k=2)
    out: Dict[str, List[Dict]] = {}

    def show(title, rows):
        print("=" * 92)
        print(title)
        print("=" * 92)
        d = pd.DataFrame(rows)
        print(d.to_string(index=False))
        print()
        out[title] = rows

    # ---- T1 implementation lag -----------------------------------------
    rows = []
    for h in (1, 4):
        for lag in (0, 1, 2):
            r = ls_backtest(sig, P, Cst, h=h, lag=lag)
            if r:
                rows.append({"horizon_h": h, "lag_bars": lag, **r})
    show("T1  IMPLEMENTATION LAG  (lag=0 is the unrealistic same-close fill)", rows)

    # ---- T2 sub-universe -------------------------------------------------
    cat = dict(zip(keep["symbol"], keep["category"]))
    clus = dict(zip(keep["symbol"], keep["cluster"]))
    groups = {
        "ALL (46)": cols,
        "FX only": [c for c in cols if cat.get(c) == "Forex_Raw"],
        "Metals only": [c for c in cols if cat.get(c) == "Commodities_Raw"],
        "Indices only": [c for c in cols if cat.get(c) == "Indices"],
        "No gold/silver": [c for c in cols if not c.startswith(("XAU", "XAG"))],
        "1 per cluster": [sorted([c for c in cols if clus.get(c) == g])[0]
                          for g in sorted(set(clus.get(c) for c in cols))],
    }
    rows = []
    for name, g in groups.items():
        g = [x for x in g if x in P.columns]
        if len(g) < 6:
            continue
        r = ls_backtest(reversal_signal(P[g], 2), P[g], Cst[g], h=1, lag=1)
        if r:
            rows.append({"universe": name, "n_assets": len(g), **r})
    show("T2  SUB-UNIVERSE  (all with 1-bar implementation lag)", rows)

    # ---- T3 stale quotes --------------------------------------------------
    active = TV.reindex_like(P) > 0
    hour = pd.Series(P.index.hour, index=P.index)
    in_session = pd.DataFrame(True, index=P.index, columns=P.columns)
    for s, (a, b) in INDEX_HOURS.items():
        if s not in in_session.columns:
            continue
        m = (hour >= a) & (hour < b) if a < b else ((hour >= a) | (hour < b))
        in_session[s] = m.to_numpy()
    rows = [
        {"filter": "none", **ls_backtest(sig, P, Cst, 1, 1)},
        {"filter": "tick_volume > 0", **ls_backtest(sig, P, Cst, 1, 1, mask=active)},
        {"filter": "index cash hours", **ls_backtest(sig, P, Cst, 1, 1, mask=in_session)},
        {"filter": "both", **ls_backtest(sig, P, Cst, 1, 1, mask=active & in_session)},
    ]
    show("T3  STALE-QUOTE FILTERS", [r for r in rows if "net_bps" in r])

    # ---- T4 cost stress ---------------------------------------------------
    rows = []
    for m in (1.0, 1.5, 2.0, 3.0, 5.0):
        r = ls_backtest(sig, P, Cst, 1, 1, cost_mult=m, mask=active & in_session)
        if r:
            rows.append({"cost_x": m, **r})
    show("T4  COST STRESS  (proxy for market impact, clean universe + lag)", rows)

    # ---- T5 sub-period ----------------------------------------------------
    mid = P.index[len(P) // 2]
    rows = []
    for name, sl in (("H1", P.index < mid), ("H2", P.index >= mid)):
        r = ls_backtest(sig[sl], P[sl], Cst[sl], 1, 1, mask=(active & in_session)[sl])
        if r:
            rows.append({"half": name, "from": str(P.index[sl][0].date()),
                         "to": str(P.index[sl][-1].date()), **r})
    show("T5  SUB-PERIOD STABILITY", rows)

    # ---- lookback choice (recorded as trials) -----------------------------
    rows = []
    for k in (1, 2, 3, 4, 6, 8, 12, 24):
        r = ls_backtest(reversal_signal(P, k), P, Cst, 1, 1, mask=active & in_session)
        if r:
            rows.append({"lookback_h": k, **r})
    show("T6  REVERSAL LOOKBACK", rows)

    C.REPORT_DIR.mkdir(exist_ok=True)
    (C.REPORT_DIR / "reversal_robustness.json").write_text(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
