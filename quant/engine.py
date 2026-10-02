"""
engine.py — Single-market trade execution engine.

Mission invariants baked in as DEFAULTS (overridable via sim_kwargs for
controlled iteration experiments only — the certified/baseline run must use
the defaults):

  - Entries at NEXT bar open only (signal observed at bar close -> fill at
    the following bar's open). Zero lookahead.
  - SL = 1.5 x ATR(14) at the signal bar  -> defines 1R in price terms.
  - TP = 3.0 R.
  - Ratchet: once unrealized >= +1.2R, stop is moved to lock +0.2R (one-way,
    never loosened).
  - Time-decay exit: if trade is still open after 24 bars and unrealized R
    is < +0.2R, flatten at the NEXT bar's open (decision made on bar close,
    executed on next open -> no lookahead).
  - Friction: 41 bps round-trip, charged as 20.5 bps notional on entry and
    20.5 bps notional on exit (dollar drag, independent of direction).
  - Risk per trade: fixed dollar amount (default $12.5), sized off the SL
    distance -> position size = risk_dollars / (1.5*ATR).

This module only produces *candidate* trades for a single instrument. It is
unaware of portfolio-level capital, concurrency caps or the hard drawdown
circuit breaker -- those are enforced by wfo.PortfolioSim across the whole
sleeve universe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

DEFAULT_SIM_KWARGS = dict(
    sl_atr_mult=1.5,          # SL = 1.5 x ATR -> defines 1R
    tp_R=3.0,                 # TP = 3.0R
    ratchet_trigger_R=1.2,    # once unrealized >= 1.2R...
    ratchet_lock_R=0.2,       # ...lock stop at +0.2R
    time_decay_bars=24,       # after 24 bars...
    time_decay_R=0.2,         # ...if unrealized R < 0.2 -> exit next open
    risk_dollars=12.5,        # fixed $ risk per trade
    friction_bps_roundtrip=41.0,  # 41bps total RT friction
    max_hold_bars=500,        # safety cap so a dead trade can't run forever
)


def wilder_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)


@dataclass
class Trade:
    symbol: str
    side: int               # +1 long, -1 short
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    entry_price: float
    sl0: float
    r_distance: float        # price distance == 1R
    size: float
    risk_dollars: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    bars_held: int = 0
    gross_pnl: float = 0.0
    friction_cost: float = 0.0
    net_pnl: float = 0.0
    r_multiple: float = 0.0
    ratcheted: bool = False


def generate_trades(df: pd.DataFrame, entries: pd.Series, sides: pd.Series,
                     atr: pd.Series, symbol: str, sim_kwargs: dict | None = None,
                     start_idx: int = 0, end_idx: int | None = None) -> list[Trade]:
    """Scan a single-instrument OHLC frame and turn boolean `entries` (aligned
    to df.index) into executed Trade objects using next-bar-open fills and
    the mandated exit geometry. `sides` is +1/-1 aligned to df.index.

    entries/sides are only honoured for signal bars whose index is within
    [start_idx, end_idx) (used by WFO to constrain signal generation to an
    OOS window while still allowing warmup lookback for indicators).
    """
    kw = dict(DEFAULT_SIM_KWARGS)
    if sim_kwargs:
        kw.update(sim_kwargs)

    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    c = df["close"].to_numpy()
    idx = df.index
    atr_arr = atr.to_numpy()
    entries_arr = entries.to_numpy()
    sides_arr = sides.to_numpy()
    n = len(df)
    if end_idx is None:
        end_idx = n

    friction_half = (kw["friction_bps_roundtrip"] / 2.0) / 10000.0

    trades: list[Trade] = []
    i = max(start_idx, 0)
    last_exit_i = -1  # don't allow overlapping trades on the SAME instrument
    while i < min(end_idx, n - 1):
        if not entries_arr[i] or np.isnan(atr_arr[i]) or atr_arr[i] <= 0:
            i += 1
            continue
        if i <= last_exit_i:
            i += 1
            continue
        entry_i = i + 1  # next bar open
        if entry_i >= n:
            break
        side = int(sides_arr[i])
        entry_price = o[entry_i]
        r_distance = kw["sl_atr_mult"] * atr_arr[i]
        if r_distance <= 0 or np.isnan(r_distance):
            i += 1
            continue
        size = kw["risk_dollars"] / r_distance

        sl = entry_price - side * r_distance
        tp = entry_price + side * kw["tp_R"] * r_distance
        ratchet_trigger_price = entry_price + side * kw["ratchet_trigger_R"] * r_distance
        ratchet_lock_price = entry_price + side * kw["ratchet_lock_R"] * r_distance
        ratcheted = False

        tr = Trade(symbol=symbol, side=side, signal_time=idx[i], entry_time=idx[entry_i],
                   entry_price=entry_price, sl0=sl, r_distance=r_distance, size=size,
                   risk_dollars=kw["risk_dollars"])

        exit_price = None
        exit_reason = None
        exit_time = None
        bars_held = 0
        pending_time_exit = False
        j = entry_i
        max_j = min(n - 1, entry_i + kw["max_hold_bars"])
        while j <= max_j:
            bars_held = j - entry_i
            # On the entry bar itself we still allow same-bar SL/TP checks
            # (resting orders triggered intrabar after the open fill).
            if pending_time_exit:
                exit_price = o[j]
                exit_reason = "time_decay"
                exit_time = idx[j]
                break
            bar_hit_sl = (l[j] <= sl) if side == 1 else (h[j] >= sl)
            bar_hit_tp = (h[j] >= tp) if side == 1 else (l[j] <= tp)
            if bar_hit_sl and bar_hit_tp:
                # conservative: assume SL first on ambiguous wide bars
                exit_price = sl
                exit_reason = "sl_ratchet" if ratcheted else "sl"
                exit_time = idx[j]
                break
            if bar_hit_sl:
                # handle gap-through
                gap_through = (o[j] < sl) if side == 1 else (o[j] > sl)
                exit_price = o[j] if gap_through else sl
                exit_reason = "sl_ratchet" if ratcheted else "sl"
                exit_time = idx[j]
                break
            if bar_hit_tp:
                gap_through = (o[j] > tp) if side == 1 else (o[j] < tp)
                exit_price = o[j] if gap_through else tp
                exit_reason = "tp"
                exit_time = idx[j]
                break
            # ratchet check (based on intrabar extreme)
            if not ratcheted:
                triggered = (h[j] >= ratchet_trigger_price) if side == 1 else (l[j] <= ratchet_trigger_price)
                if triggered:
                    ratcheted = True
                    sl = ratchet_lock_price
            # time decay check (based on bar close), executes next open
            if bars_held >= kw["time_decay_bars"]:
                unreal_R = (c[j] - entry_price) * side / r_distance
                if unreal_R < kw["time_decay_R"]:
                    if j == max_j:
                        exit_price = c[j]
                        exit_reason = "time_decay_final_bar"
                        exit_time = idx[j]
                        break
                    pending_time_exit = True
            j += 1
        if exit_price is None:
            # ran out of bars (max_hold) -> mark-to-close, flag as timeout
            exit_price = c[max_j]
            exit_reason = "timeout"
            exit_time = idx[max_j]
            j = max_j

        gross_pnl = (exit_price - entry_price) * side * size
        friction_cost = (entry_price * size + exit_price * size) * friction_half
        net_pnl = gross_pnl - friction_cost

        tr.exit_time = exit_time
        tr.exit_price = exit_price
        tr.exit_reason = exit_reason
        tr.bars_held = bars_held
        tr.gross_pnl = gross_pnl
        tr.friction_cost = friction_cost
        tr.net_pnl = net_pnl
        tr.r_multiple = net_pnl / kw["risk_dollars"]
        tr.ratcheted = ratcheted
        trades.append(tr)

        last_exit_i = j
        i = max(i + 1, j)  # no overlapping trades on same instrument
    return trades
