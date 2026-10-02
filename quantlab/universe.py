"""
quantlab.universe
=================
Turn 156 raw files into a defensible tradable universe.

Three successive screens:
  1. **Integrity**  - enough genuine 15m history after the padded block is cut.
  2. **Cost**       - modelled round-turn cost must be small vs. ATR(14),
                      otherwise the instrument cannot pay for the signal.
  3. **Redundancy** - hierarchical clustering on daily returns; correlated
                      instruments form one cluster so the portfolio engine can
                      cap concurrent risk per cluster instead of pretending
                      XAUUSD / XAUEUR / XAUGBP are three independent bets.

Every screen writes its reason into the output table, so the universe is
auditable rather than asserted.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import config as C
from . import datafeed as D
from .costs import CostModel, build_cost_model


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def screen_universe(
    symbols: Optional[List[str]] = None,
    manifest: Optional[Dict] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, CostModel]]:
    manifest = manifest or C.load_manifest()
    symbols = symbols or C.base_universe(manifest)

    rows: List[Dict] = []
    cost_models: Dict[str, CostModel] = {}

    for sym in symbols:
        meta = manifest.get(sym, {})
        row: Dict = {
            "symbol": sym,
            "category": meta.get("category", "?"),
            "description": meta.get("description", ""),
        }
        try:
            bars = D.load_bars(sym, "15m", enforce_pure=True)
        except FileNotFoundError:
            row.update(status="drop", reason="no 15m file")
            rows.append(row)
            continue

        row["pure_start"] = bars["datetime"].iloc[0] if len(bars) else pd.NaT
        row["pure_end"] = bars["datetime"].iloc[-1] if len(bars) else pd.NaT
        row["n_pure_bars"] = len(bars)

        if len(bars) < C.MIN_PURE_BARS_15M:
            row.update(status="drop", reason=f"only {len(bars)} genuine 15m bars")
            rows.append(row)
            continue

        cm = build_cost_model(sym, meta, bars)
        cost_models[sym] = cm

        hours = bars["datetime"].dt.hour.to_numpy()
        sp_px = cm.spread_price(bars["spread"].to_numpy(), hours)
        atr = _atr(bars, 14)
        med_atr = float(np.nanmedian(atr.to_numpy()))
        med_px = float(bars["close"].median())
        rt_cost = float(np.median(sp_px)) * (1 + C.SLIPPAGE_ENTRY_SPREADS + C.SLIPPAGE_STOP_SPREADS) \
            + med_px * cm.commission_bps_rt / 1e4

        assumed_r = med_atr * C.ASSUMED_R_IN_ATR
        row.update(
            point=cm.point,
            live_spread=cm.has_live_spread,
            med_spread_pts=round(cm.spread_median, 2),
            med_spread_bps=round(float(np.median(sp_px)) / med_px * 1e4, 3),
            med_atr14_bps=round(med_atr / med_px * 1e4, 2),
            rt_cost_bps=round(rt_cost / med_px * 1e4, 3),
            cost_to_atr=round(rt_cost / med_atr, 4) if med_atr > 0 else np.nan,
            cost_to_R=round(rt_cost / assumed_r, 4) if assumed_r > 0 else np.nan,
        )

        if not np.isfinite(row["cost_to_R"]):
            row.update(status="drop", reason="cannot estimate ATR")
        elif row["cost_to_R"] > C.MAX_COST_TO_R:
            row.update(status="drop",
                       reason=f"cost/R {row['cost_to_R']:.3f} > {C.MAX_COST_TO_R}")
        else:
            row.update(status="keep", reason="")
        rows.append(row)

    tbl = pd.DataFrame(rows)
    keep = tbl[tbl["status"] == "keep"]["symbol"].tolist()
    if verbose:
        print(f"  integrity+cost screen: {len(keep)}/{len(symbols)} instruments retained")

    clusters = cluster_universe(keep)
    tbl["cluster"] = tbl["symbol"].map(clusters)
    cost_models = {k: v for k, v in cost_models.items() if k in keep}
    return tbl, cost_models


def cluster_universe(symbols: List[str], corr_cut: float = 0.70) -> Dict[str, int]:
    """Group instruments by daily-return correlation.

    Uses the D1 files (genuine back to 2015/2020) restricted to the common
    2023+ window so the correlation reflects the regime we actually trade.
    """
    if len(symbols) < 2:
        return {s: i for i, s in enumerate(symbols)}

    series = {}
    for s in symbols:
        try:
            d = D.load_bars(s, "d1", enforce_pure=False, start="2022-01-01")
        except FileNotFoundError:
            continue
        if len(d) < 200:
            continue
        r = np.log(d.set_index("datetime")["close"]).diff()
        series[s] = r[~r.index.duplicated(keep="last")]

    if len(series) < 2:
        return {s: i for i, s in enumerate(symbols)}

    R = pd.DataFrame(series).resample("1D").sum(min_count=1)
    R = R.dropna(axis=0, how="all")
    R = R.loc[:, R.notna().sum() > 150]
    corr = R.corr(min_periods=120).fillna(0.0)

    try:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform

        dist = np.sqrt(np.clip(0.5 * (1.0 - corr.to_numpy()), 0, None))
        np.fill_diagonal(dist, 0.0)
        dist = (dist + dist.T) / 2.0
        Z = linkage(squareform(dist, checks=False), method="average")
        cut = np.sqrt(0.5 * (1.0 - corr_cut))
        labels = fcluster(Z, t=cut, criterion="distance")
        out = {sym: int(lab) for sym, lab in zip(corr.columns, labels)}
    except Exception:
        out = {sym: i for i, sym in enumerate(corr.columns)}

    nxt = max(out.values(), default=0) + 1
    for s in symbols:
        if s not in out:
            out[s] = nxt
            nxt += 1
    return out
