"""
quantlab.pipeline
=================
Glue: universe -> cached causal features -> sweep events -> simulated trades.

The engineered feature frame is expensive and does *not* depend on
`StrategyParams`, so it is cached on disk.  Event detection and path
simulation are cheap, which makes an honest in-sample parameter search
affordable without ever touching out-of-sample bars.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import config as C
from . import datafeed as D
from .costs import CostModel, build_cost_model
from .features import atr
from .labeling import simulate_events, uniqueness_weights
from .signals import detect_sweep_events, engineer

CACHE_DIR = C.REPO_ROOT / ".cache"
CACHE_VERSION = "v3"


def _cache_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol}_{CACHE_VERSION}.parquet"


def get_enriched(symbol: str, use_cache: bool = True) -> pd.DataFrame:
    p = _cache_path(symbol)
    if use_cache and p.exists():
        return pd.read_parquet(p)
    bars = D.load_bars(symbol, "15m", enforce_pure=True)
    df = engineer(bars)
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p, index=False)
    return df


def cost_per_atr(cm: CostModel, enriched: pd.DataFrame) -> float:
    hrs = enriched["datetime"].dt.hour.to_numpy()
    rt = cm.round_turn_cost(
        enriched["close"].to_numpy(), enriched["spread"].to_numpy(), hrs, "stop"
    )
    med_atr = float(np.nanmedian(enriched["atr14"].to_numpy()))
    return float(np.nanmedian(rt)) / med_atr if med_atr > 0 else np.nan


def build_trades(
    symbols: List[str],
    params: C.StrategyParams,
    cost_models: Dict[str, CostModel],
    use_cache: bool = True,
    verbose: bool = False,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Run the full primary-rule pipeline over a symbol list.

    `end` truncates the *bar series* (not just the trade list) so that an
    in-sample design search genuinely cannot see later bars - not even
    through a path simulation that would have run past the boundary.
    """
    frames: List[pd.DataFrame] = []
    for sym in symbols:
        cm = cost_models.get(sym)
        if cm is None:
            continue
        try:
            enr = get_enriched(sym, use_cache=use_cache)
        except FileNotFoundError:
            continue
        if end is not None:
            enr = enr[enr["datetime"] <= pd.Timestamp(end, tz="UTC")].reset_index(drop=True)
        if len(enr) < 1000:
            continue
        cpa = cost_per_atr(cm, enr)
        ev = detect_sweep_events(
            enr, params, cm.point, 5, cost_per_atr=cpa, enriched=enr
        )
        if ev.empty:
            continue
        tr = simulate_events(enr, ev, params, cm)
        tr["symbol"] = sym
        tr["w_uniq"] = uniqueness_weights(
            tr["entry_i"].to_numpy(int), tr["exit_i"].to_numpy(int), len(enr)
        )
        frames.append(tr)
        if verbose:
            print(f"    {sym:8s} {len(tr):5d} events  E[R]={tr.r_net.mean():+.3f}")
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values("entry_time").reset_index(drop=True)


def summarise(trades: pd.DataFrame, label: str = "") -> Dict:
    """Trade-level (not portfolio-level) summary of a primary rule."""
    if trades.empty:
        return {"label": label, "n": 0}
    r = trades["r_net"].to_numpy()
    wins = r > 0
    gp = r[wins].sum()
    gl = -r[~wins].sum()
    return {
        "label": label,
        "n": int(len(r)),
        "win_rate": round(float(wins.mean()) * 100, 2),
        "avg_R": round(float(r.mean()), 4),
        "median_R": round(float(np.median(r)), 4),
        "sum_R": round(float(r.sum()), 1),
        "profit_factor": round(float(gp / gl), 3) if gl > 0 else np.inf,
        "avg_cost_R": round(float(trades["cost_r"].mean()), 4),
        "t_stat": round(float(r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))), 2) if len(r) > 2 else np.nan,
        "target_hit": round(float((trades["exit_kind"] == "target").mean()) * 100, 1),
        "avg_bars": round(float(trades["bars_held"].mean()), 1),
    }
