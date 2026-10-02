"""
quantlab.signals
================
S5 primary rule: **liquidity-sweep stop-run events**.

Economic rationale
------------------
Osler (2003, *Journal of Finance* 58(5)) documents, using actual order books
from a large FX dealer, that (a) stop-loss orders cluster just **beyond**
round numbers and recent price extremes, and (b) take-profit orders cluster
**at** round numbers.  The implication is a specific, testable microstructure
pattern: price reaches a known liquidity pool, triggers a cascade of resting
stops, overshoots, and then either

  * **reverts**, because the cascade exhausts into resting take-profit/limit
    interest (the "stop-run reversion" / turtle-soup case), or
  * **continues**, because the stops were the fuel for a genuine
    repricing (the breakout case).

Which of the two happens is exactly the question a meta-model can answer, and
it is why this module emits *events* with a rich geometric description rather
than a hard-coded directional opinion.  `StrategyParams.polarity` selects
which side the primary rule takes; the ML layer decides which events to
actually trade.

Liquidity pools used (all strictly causal):
  PD  - previous completed day's high/low
  AS  - current day's Asian-session (00:00-06:00 UTC) high/low, visible 06:00+
  PW  - previous completed week's high/low
  SW  - rolling intraday swing extreme (default 48 bars = 12h), shifted 1 bar
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import StrategyParams
from .features import add_bar_features, add_context, round_number_distance

POOL_CODES = {"PD": 0, "AS": 1, "PW": 2, "SW": 3}

EVENT_FEATURES = [
    # sweep geometry
    "pen_atr", "reclaim_frac", "body_frac", "rng_atr", "sweep_wick",
    "struct_r_atr", "pool_conf", "pool_code", "rn_dist_atr",
    # where we are in the day / week
    "hour", "dow", "minute_of_day", "sweeps_today",
    # local state
    "rsi14", "rsi_64", "er40", "vol_expansion", "vwap_dist_atr", "ou_hl",
    "hurst", "roc16_atr", "roc64_atr", "tickvol_z", "spread_z", "dir_run",
    # higher-timeframe regime
    "px_vs_h4ema200_atr", "h4_trend_atr", "h4_er", "d_atr_pctile",
    "pd_range_atr", "pos_in_pd_range",
    # cost state (the model should learn to avoid expensive bars)
    "cost_to_r_est",
]


def _session_mask(df: pd.DataFrame, p: StrategyParams) -> np.ndarray:
    hour = df["hour"].to_numpy()
    dow = df["dow"].to_numpy()
    m = np.isin(hour, np.array(p.session_hours))
    m &= ~((dow == 4) & (hour >= p.exclude_friday_after))   # Friday late
    m &= ~((dow == 6) & (hour < p.exclude_sunday_before))   # Sunday open
    m &= dow != 5                                            # Saturday
    return m


def engineer(bars: pd.DataFrame) -> pd.DataFrame:
    """Full causal feature frame for one instrument.  Cached by callers: this
    is the expensive step and it is independent of StrategyParams."""
    return add_bar_features(add_context(bars))


def detect_sweep_events(
    bars: pd.DataFrame,
    params: StrategyParams,
    point: float,
    digits: int,
    cost_per_atr: float = 0.0,
    enriched: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Return one row per sweep event, with features evaluated at the signal
    bar's close.  Execution index is `entry_i = i + 1`.
    """
    df = engineer(bars) if enriched is None else enriched
    n = len(df)
    if n < 500:
        return pd.DataFrame()

    a = df["atr14"].to_numpy()
    low = df["low"].to_numpy()
    high = df["high"].to_numpy()
    close = df["close"].to_numpy()
    rngv = df["rng"].to_numpy()

    ub = params.unbreached_lookback
    prior_min = df["low"].rolling(ub).min().shift(1).to_numpy()
    prior_max = df["high"].rolling(ub).max().shift(1).to_numpy()
    swing_lo = df["low"].rolling(params.swing_lookback).min().shift(1)
    swing_hi = df["high"].rolling(params.swing_lookback).max().shift(1)

    sess = _session_mask(df, params)
    valid_atr = np.isfinite(a) & (a > 0)

    pools_low = {"PD": df["pdl"].to_numpy(), "AS": df["asian_low"].to_numpy(),
                 "PW": df["pwl"].to_numpy(), "SW": swing_lo.to_numpy()}
    pools_high = {"PD": df["pdh"].to_numpy(), "AS": df["asian_high"].to_numpy(),
                  "PW": df["pwh"].to_numpy(), "SW": swing_hi.to_numpy()}
    enabled = {"PD": params.use_prev_day, "AS": params.use_asian_range,
               "PW": params.use_prev_week, "SW": params.use_swing}

    with np.errstate(invalid="ignore", divide="ignore"):
        pen_low = {k: (v - low) / a for k, v in pools_low.items() if enabled[k]}
        pen_high = {k: (high - v) / a for k, v in pools_high.items() if enabled[k]}

    close_pos = (close - low) / np.where(rngv > 0, rngv, np.nan)

    # ---- candidate masks -------------------------------------------------
    def _side_candidates(pen: Dict[str, np.ndarray], pools: Dict[str, np.ndarray], is_low_side: bool):
        best_pen = np.full(n, -np.inf)
        best_code = np.full(n, -1, dtype=int)
        best_level = np.full(n, np.nan)
        for k, pv in pen.items():
            lvl = pools[k]
            ok = (
                np.isfinite(pv)
                & (pv >= params.min_penetration_atr)
                & (pv <= params.max_penetration_atr)
            )
            # the pool must have been intact immediately before this bar,
            # otherwise every subsequent bar re-fires the same event
            if is_low_side:
                ok &= np.isfinite(prior_min) & (prior_min > lvl)
            else:
                ok &= np.isfinite(prior_max) & (prior_max < lvl)
            better = ok & (pv > best_pen)
            best_pen = np.where(better, pv, best_pen)
            best_code = np.where(better, POOL_CODES[k], best_code)
            best_level = np.where(better, lvl, best_level)
        return best_pen, best_code, best_level

    pen_l, code_l, lvl_l = _side_candidates(pen_low, pools_low, True)
    pen_h, code_h, lvl_h = _side_candidates(pen_high, pools_high, False)

    fade = params.polarity == "fade"
    if fade:
        # sweep the LOW pool and close back above it -> long
        long_ok = (code_l >= 0) & (close > lvl_l) & (close_pos >= params.min_reclaim_frac)
        short_ok = (code_h >= 0) & (close < lvl_h) & ((1.0 - close_pos) >= params.min_reclaim_frac)
    else:
        # sweep the LOW pool and hold below it -> short (breakout)
        short_ok = (code_l >= 0) & (close < lvl_l) & (close_pos <= 1.0 - params.min_reclaim_frac)
        long_ok = (code_h >= 0) & (close > lvl_h) & (close_pos >= params.min_reclaim_frac)

    base = sess & valid_atr
    long_ok = np.nan_to_num(long_ok, nan=False).astype(bool) & base
    short_ok = np.nan_to_num(short_ok, nan=False).astype(bool) & base
    # a bar cannot be both; prefer the deeper raid
    both = long_ok & short_ok
    if both.any():
        deeper_low = pen_l >= pen_h
        if fade:
            long_ok = long_ok & (~both | deeper_low)
            short_ok = short_ok & (~both | ~deeper_low)
        else:
            short_ok = short_ok & (~both | deeper_low)
            long_ok = long_ok & (~both | ~deeper_low)

    idx = np.where(long_ok | short_ok)[0]
    idx = idx[idx < n - 2]
    if len(idx) == 0:
        return pd.DataFrame()

    # ---- cooldown --------------------------------------------------------
    kept: List[int] = []
    last = -10 ** 9
    for i in idx:
        if i - last >= params.cooldown_bars:
            kept.append(int(i))
            last = i
    idx = np.array(kept, dtype=int)

    side = np.where(long_ok[idx], 1, -1)
    if fade:
        pool_code = np.where(side == 1, code_l[idx], code_h[idx])
        pool_level = np.where(side == 1, lvl_l[idx], lvl_h[idx])
        pen = np.where(side == 1, pen_l[idx], pen_h[idx])
    else:
        pool_code = np.where(side == -1, code_l[idx], code_h[idx])
        pool_level = np.where(side == -1, lvl_l[idx], lvl_h[idx])
        pen = np.where(side == -1, pen_l[idx], pen_h[idx])

    # pool confluence: how many *other* enabled pools sit within tolerance
    tol = params.pool_tolerance_atr * a[idx]
    conf = np.zeros(len(idx))
    for k in POOL_CODES:
        if not enabled[k]:
            continue
        lv = np.where(side == 1, pools_low[k][idx], pools_high[k][idx])
        conf += (np.abs(lv - pool_level) <= tol).astype(float)
    conf -= 1.0  # discount the pool itself

    cp = close_pos[idx]
    reclaim = np.where(side == 1, cp, 1.0 - cp)
    wick = np.where(side == 1, df["lower_wick"].to_numpy()[idx], df["upper_wick"].to_numpy()[idx])

    # structural stop distance implied by the sweep extreme (pre-floor)
    struct = np.where(
        side == 1,
        df["open"].to_numpy()[idx + 1] - (low[idx] - params.stop_buffer_atr * a[idx]),
        (high[idx] + params.stop_buffer_atr * a[idx]) - df["open"].to_numpy()[idx + 1],
    )
    struct_r_atr = struct / a[idx]

    rn = round_number_distance(pd.Series(pool_level), point, digits).to_numpy()

    sweeps_today = (
        pd.Series(1, index=df["date"].iloc[idx]).groupby(level=0).cumcount().to_numpy()
    )

    out = pd.DataFrame(
        {
            "i": idx,
            "entry_i": idx + 1,
            "signal_time": df["datetime"].to_numpy()[idx],
            "entry_time": df["datetime"].to_numpy()[idx + 1],
            "side": side,
            "atr": a[idx],
            "pool_level": pool_level,
            "pen_atr": pen,
            "reclaim_frac": reclaim,
            "sweep_wick": wick,
            "struct_r_atr": struct_r_atr,
            "pool_conf": conf,
            "pool_code": pool_code.astype(float),
            "rn_dist_atr": rn / a[idx],
            "sweeps_today": sweeps_today.astype(float),
            "cost_to_r_est": cost_per_atr / max(params.min_stop_atr, 1e-6),
        }
    )

    carry = [
        "body_frac", "rng_atr", "hour", "dow", "minute_of_day", "rsi14", "rsi_64",
        "er40", "vol_expansion", "vwap_dist_atr", "ou_hl", "hurst", "roc16_atr",
        "roc64_atr", "tickvol_z", "spread_z", "dir_run", "px_vs_h4ema200_atr",
        "h4_trend_atr", "h4_er", "d_atr_pctile", "pd_range_atr", "pos_in_pd_range",
    ]
    for c in carry:
        out[c] = df[c].to_numpy()[idx]

    out["hour"] = out["hour"].astype(float)
    out["dow"] = out["dow"].astype(float)
    out["minute_of_day"] = out["minute_of_day"].astype(float)
    return out.reset_index(drop=True)
