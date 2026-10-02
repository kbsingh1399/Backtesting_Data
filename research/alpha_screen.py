"""
Cost-aware alpha screen on the development block.
=================================================

The sweep-event rule tested in `diagnostics.py` is indistinguishable from a
random entry (t = -0.15 against the matched control).  Rather than keep
tuning a rule with no information content, we screen a panel of candidate
predictors directly, in the only currency that matters: **basis points of
forward return net of the modelled round-turn cost.**

Method
------
* Panel is sampled hourly (every 4th 15m bar) over 2023-09-15..2024-06-30.
* For each candidate signal and each horizon h in {1h, 4h, 24h} we compute
    - pooled Spearman IC against the forward log return, and
    - a decile long/short portfolio rebalanced every h, charged one
      round-turn cost per leg per rebalance, reported as an annualised
      Sharpe on the per-rebalance spread series.
* Signals are cross-sectionally z-scored per timestamp so the long/short
  book is (approximately) dollar neutral, which is how any of this would be
  traded at scale.

The decile Sharpe net of costs is the screen that matters.  A signal with a
beautiful IC and a negative net Sharpe is a signal you cannot trade.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantlab import config as C                 # noqa: E402
from quantlab.pipeline import get_enriched       # noqa: E402
from quantlab.universe import screen_universe    # noqa: E402

DEV_START, DEV_END = C.WF_TRAIN_START, "2024-06-30"
HORIZONS = {"1h": 1, "4h": 4, "1d": 24, "2d": 48, "5d": 120}   # in hourly steps
BARS_PER_YEAR = 24 * 252


def build_panel(symbols: List[str], cms, start: str, end: str):
    px, feats, costs = {}, {}, {}
    for s in symbols:
        e = get_enriched(s)
        e = e[(e["datetime"] >= pd.Timestamp(start, tz="UTC")) &
              (e["datetime"] <= pd.Timestamp(end, tz="UTC"))]
        if len(e) < 2000:
            continue
        # Sample on a COMMON clock (top of the hour), not every Nth row of each
        # symbol's own frame -- otherwise each instrument lands on a different
        # grid and the cross-sectional panel is almost entirely NaN.
        e = e[e["datetime"].dt.minute == 0].set_index("datetime")
        px[s] = e["close"]
        cm = cms[s]
        sp = cm.spread_price(e["spread"].to_numpy(), e.index.hour.to_numpy())
        rt = sp * (1 + C.SLIPPAGE_ENTRY_SPREADS + C.SLIPPAGE_TIMEEXIT_SPREADS) \
            + e["close"].to_numpy() * cm.commission_bps_rt / 1e4
        costs[s] = pd.Series(rt / e["close"].to_numpy() * 1e4, index=e.index)  # bps
        feats[s] = e[["rsi14", "er40", "vol_expansion", "vwap_dist_atr", "hurst",
                      "tickvol_z", "atr14", "roc16_atr", "roc64_atr",
                      "pos_in_pd_range", "px_vs_h4ema200_atr", "close"]]
    P = pd.DataFrame(px).sort_index()
    Cst = pd.DataFrame(costs).reindex_like(P)
    return P, feats, Cst


def make_signals(P: pd.DataFrame, feats: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    lr = np.log(P).diff()
    vol = lr.rolling(120, min_periods=60).std()

    def risk_adj(x):
        return x / (vol + 1e-12)

    sig: Dict[str, pd.DataFrame] = {}
    # --- short-horizon reversal -------------------------------------------
    for k in (1, 2, 4, 8, 24):
        sig[f"rev_{k}h"] = -risk_adj(np.log(P).diff(k))
    # --- time-series momentum ---------------------------------------------
    for k in (24, 120, 480):
        sig[f"mom_{k}h"] = risk_adj(np.log(P).diff(k))
    # --- cross-sectional momentum (demeaned per timestamp) ----------------
    for k in (24, 120, 480):
        m = risk_adj(np.log(P).diff(k))
        sig[f"xsmom_{k}h"] = m.sub(m.mean(axis=1), axis=0)
    # --- breakout ----------------------------------------------------------
    for k in (24, 120):
        hi = P.rolling(k).max()
        lo = P.rolling(k).min()
        sig[f"donch_{k}h"] = (P - (hi + lo) / 2) / ((hi - lo) / 2 + 1e-12)
    # --- indicator-based ----------------------------------------------------
    def stack(col):
        return pd.DataFrame({s: f[col] for s, f in feats.items()}).reindex_like(P)

    sig["rsi_rev"] = -(stack("rsi14") - 50.0)
    sig["vwap_rev"] = -stack("vwap_dist_atr")
    sig["pdrange_rev"] = -(stack("pos_in_pd_range") - 0.5)
    sig["h4_trend"] = stack("px_vs_h4ema200_atr")
    sig["vol_exp"] = stack("vol_expansion")
    sig["tickvol"] = stack("tickvol_z")
    sig["carry_proxy"] = risk_adj(np.log(P).diff(480)).rank(axis=1, pct=True) - 0.5
    return sig


def zscore_xs(df: pd.DataFrame) -> pd.DataFrame:
    m = df.mean(axis=1)
    s = df.std(axis=1)
    return df.sub(m, axis=0).div(s + 1e-12, axis=0)


def evaluate(sig: pd.DataFrame, P: pd.DataFrame, Cst: pd.DataFrame, h: int) -> Dict:
    fwd = np.log(P).shift(-h) - np.log(P)            # log return over h steps
    valid = sig.notna() & fwd.notna()
    S = sig.where(valid)
    F = fwd.where(valid)
    cnt = valid.sum(axis=1)
    keep = cnt >= 10
    S, F = S[keep], F[keep]
    if len(S) < 50:
        return {}

    ic = S.corrwith(F, axis=1, method="spearman")

    Z = zscore_xs(S)
    # long/short book, gross exposure 1.0, rebalanced every h steps
    W = Z.div(Z.abs().sum(axis=1), axis=0).fillna(0.0)
    W = W.iloc[::h]
    Fh = F.iloc[::h]
    Ch = Cst.reindex_like(F).iloc[::h].fillna(Cst.stack().median())
    gross = (W * Fh).sum(axis=1) * 1e4                                   # bps
    turn = W.diff().abs().fillna(W.abs()).mul(0.5)
    # `turn` is one-way turnover as a fraction of gross exposure; charging
    # the full round-turn cost on it already covers entry + exit.
    cost = (turn * Ch).sum(axis=1)                                       # bps
    net = (gross - cost).dropna()
    if len(net) < 20:
        return {}
    per_year = BARS_PER_YEAR / h
    return {
        "ic_mean": float(ic.mean()),
        "ic_t": float(ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic)))) if ic.std() > 0 else np.nan,
        "gross_bps": float(gross.mean()),
        "cost_bps": float(cost.mean()),
        "net_bps": float(net.mean()),
        "net_sharpe": float(net.mean() / (net.std(ddof=1) + 1e-12) * np.sqrt(per_year)),
        "n_rebal": int(len(net)),
    }


def main() -> None:
    tbl, cms = screen_universe(verbose=False)
    symbols = tbl.loc[tbl["status"] == "keep", "symbol"].tolist()
    print(f"panel: {len(symbols)} instruments, {DEV_START}..{DEV_END}, hourly sampling")
    P, feats, Cst = build_panel(symbols, cms, DEV_START, DEV_END)
    print(f"panel shape {P.shape}, median cost {Cst.stack().median():.3f} bps round turn")

    sigs = make_signals(P, feats)
    rows = []
    for name, s in sigs.items():
        for hname, h in HORIZONS.items():
            r = evaluate(s, P, Cst, h)
            if r:
                rows.append({"signal": name, "horizon": hname, **r})
    res = pd.DataFrame(rows).sort_values("net_sharpe", ascending=False)
    C.REPORT_DIR.mkdir(exist_ok=True)
    res.to_csv(C.REPORT_DIR / "alpha_screen.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n" + "=" * 104)
    print("COST-AWARE ALPHA SCREEN  (dev block only)   net_sharpe = annualised Sharpe of the L/S book after costs")
    print("=" * 104)
    print(res.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    print(f"\ntrials evaluated: {len(res)}")


if __name__ == "__main__":
    main()
