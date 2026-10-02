"""
Hostile testing of the VWAP-deviation reversion signal.
=======================================================

`lagged_screen.py` left exactly one candidate standing: fading the deviation
of price from its own 12-hour volume-weighted average, cross-sectionally,
rebalanced every 4 hours.  Unlike the 1-2 bar reversal, it survives an
implementation lag -- its anchor is a 48-bar average, so it is not just
yesterday's bid-ask bounce.

A 3.9 dev Sharpe is still not credible on its face, so this script applies
every test that normally kills a statistical-arbitrage signal:

  L  longer implementation lags (1..8 hourly bars)
  U  sub-universes, including one-name-per-correlation-cluster, which removes
     the "XAUUSD vs XAUEUR" self-arbitrage degeneracy
  S  stale-quote filters (tick activity, index cash hours)
  C  cost multipliers up to 5x as a market-impact proxy
  W  anchor window (6h / 12h / 24h / 48h) -- a signal that only works at one
     window is a fit, not an effect
  P  sub-period split
  N  neutralisation: strip the cross-sectional mean and the cluster mean, so
     the book cannot be covertly long a common factor
  R  per-cluster attribution

Run:
    .venv/bin/python research/vwap_robustness.py
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
from research.reversal_robustness import INDEX_HOURS, ls_backtest  # noqa: E402

DEV_START, DEV_END = C.WF_TRAIN_START, "2024-06-30"


def panel(symbols, cms, start, end):
    px, cost, tv, bars = {}, {}, {}, {}
    for s in symbols:
        e = get_enriched(s)
        e = e[(e["datetime"] >= pd.Timestamp(start, tz="UTC")) &
              (e["datetime"] <= pd.Timestamp(end, tz="UTC"))]
        if len(e) < 2000:
            continue
        bars[s] = e.set_index("datetime")
        h = e[e["datetime"].dt.minute == 0].set_index("datetime")
        px[s] = h["close"]
        tv[s] = h["tick_volume"]
        cm = cms[s]
        sp = cm.spread_price(h["spread"].to_numpy(), h.index.hour.to_numpy())
        rt = sp * (1 + C.SLIPPAGE_ENTRY_SPREADS + C.SLIPPAGE_TIMEEXIT_SPREADS) \
            + h["close"].to_numpy() * cm.commission_bps_rt / 1e4
        cost[s] = pd.Series(rt / h["close"].to_numpy() * 1e4, index=h.index)
    return (pd.DataFrame(px).sort_index(), pd.DataFrame(cost).sort_index(),
            pd.DataFrame(tv).sort_index(), bars)


def vwap_signal(bars: Dict[str, pd.DataFrame], index: pd.DatetimeIndex, window: int = 48) -> pd.DataFrame:
    """-(close - VWAP_window) / ATR14, sampled on the hourly clock."""
    out = {}
    for s, b in bars.items():
        tp = (b["high"] + b["low"] + b["close"]) / 3.0
        v = b["tick_volume"]
        vwap = (tp * v).rolling(window).sum() / (v.rolling(window).sum() + 1e-12)
        sig = -(b["close"] - vwap) / (b["atr14"] + 1e-12)
        out[s] = sig.reindex(index)
    return pd.DataFrame(out)


def neutralise(sig: pd.DataFrame, clusters: Dict[str, int]) -> pd.DataFrame:
    """Remove the cross-sectional mean and each cluster's mean."""
    z = sig.sub(sig.mean(axis=1), axis=0)
    out = z.copy()
    for c in set(clusters.values()):
        cols = [k for k, v in clusters.items() if v == c and k in z.columns]
        if len(cols) > 1:
            out[cols] = z[cols].sub(z[cols].mean(axis=1), axis=0)
    return out


def main() -> None:
    tbl, cms = screen_universe(verbose=False)
    keep = tbl[tbl["status"] == "keep"]
    symbols = keep["symbol"].tolist()
    P, Cst, TV, bars = panel(symbols, cms, DEV_START, DEV_END)
    cols = P.columns.tolist()
    cat = dict(zip(keep["symbol"], keep["category"]))
    clus = {k: int(v) for k, v in zip(keep["symbol"], keep["cluster"]) if k in cols}
    print(f"panel {P.shape}, {len(cols)} instruments\n")

    base = vwap_signal(bars, P.index, 48)
    active = TV.reindex_like(P) > 0
    hour = pd.Series(P.index.hour, index=P.index)
    in_sess = pd.DataFrame(True, index=P.index, columns=P.columns)
    for s, (a, b) in INDEX_HOURS.items():
        if s in in_sess.columns:
            m = (hour >= a) & (hour < b) if a < b else ((hour >= a) | (hour < b))
            in_sess[s] = m.to_numpy()
    clean = active & in_sess

    out: Dict[str, object] = {}

    def show(title, rows):
        print("=" * 96); print(title); print("=" * 96)
        print(pd.DataFrame(rows).to_string(index=False)); print()
        out[title] = rows

    show("L  IMPLEMENTATION LAG (h=4, clean universe)",
         [{"lag_h": g, **ls_backtest(base, P, Cst, 4, g, mask=clean)} for g in (0, 1, 2, 4, 8)])

    groups = {
        "ALL": cols,
        "FX only": [c for c in cols if cat.get(c) == "Forex_Raw"],
        "Metals only": [c for c in cols if cat.get(c) == "Commodities_Raw"],
        "Indices only": [c for c in cols if cat.get(c) == "Indices"],
        "No gold/silver": [c for c in cols if not c.startswith(("XAU", "XAG"))],
        "1 per cluster": sorted({min(c for c in cols if clus.get(c) == g)
                                 for g in sorted(set(clus.values()))}),
    }
    rows = []
    for n, g in groups.items():
        g = [x for x in g if x in cols]
        if len(g) < 6:
            continue
        r = ls_backtest(base[g], P[g], Cst[g], 4, 1, mask=clean[g])
        if r:
            rows.append({"universe": n, "n_assets": len(g), **r})
    show("U  SUB-UNIVERSE (lag=1, h=4)", rows)

    show("S  STALE-QUOTE FILTERS (lag=1, h=4)", [
        {"filter": "none", **ls_backtest(base, P, Cst, 4, 1)},
        {"filter": "tick_volume>0", **ls_backtest(base, P, Cst, 4, 1, mask=active)},
        {"filter": "index cash hrs", **ls_backtest(base, P, Cst, 4, 1, mask=in_sess)},
        {"filter": "both", **ls_backtest(base, P, Cst, 4, 1, mask=clean)},
    ])

    show("C  COST STRESS (market-impact proxy)",
         [{"cost_x": m, **ls_backtest(base, P, Cst, 4, 1, cost_mult=m, mask=clean)}
          for m in (1.0, 1.5, 2.0, 3.0, 5.0)])

    rows = []
    for w in (24, 48, 96, 192):
        for h in (4, 8, 24):
            r = ls_backtest(vwap_signal(bars, P.index, w), P, Cst, h, 1, mask=clean)
            if r:
                rows.append({"anchor_bars_15m": w, "rebal_h": h, **r})
    show("W  ANCHOR WINDOW x REBALANCE", rows)

    mid = P.index[len(P) // 2]
    rows = []
    for n, sl in (("H1", P.index < mid), ("H2", P.index >= mid)):
        r = ls_backtest(base[sl], P[sl], Cst[sl], 4, 1, mask=clean[sl])
        if r:
            rows.append({"half": n, "from": str(P.index[sl][0].date()),
                         "to": str(P.index[sl][-1].date()), **r})
    show("P  SUB-PERIOD", rows)

    show("N  NEUTRALISATION", [
        {"variant": "raw", **ls_backtest(base, P, Cst, 4, 1, mask=clean)},
        {"variant": "xs-demeaned", **ls_backtest(base.sub(base.mean(axis=1), axis=0),
                                                 P, Cst, 4, 1, mask=clean)},
        {"variant": "cluster-neutral", **ls_backtest(neutralise(base, clus),
                                                     P, Cst, 4, 1, mask=clean)},
    ])

    rows = []
    for g in sorted(set(clus.values())):
        cl = [c for c in cols if clus.get(c) == g]
        if len(cl) < 2:
            continue
        r = ls_backtest(base[cl], P[cl], Cst[cl], 4, 1, mask=clean[cl])
        if r:
            rows.append({"cluster": g, "members": " ".join(cl), **r})
    show("R  WITHIN-CLUSTER ATTRIBUTION", rows)

    C.REPORT_DIR.mkdir(exist_ok=True)
    (C.REPORT_DIR / "vwap_robustness.json").write_text(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
