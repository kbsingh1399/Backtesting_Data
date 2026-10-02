"""
statarb_engine.py — OU/z-score-native trade generator for mean-reversion
pairs trades. Deliberately NOT engine.generate_trades (which is an
ATR-breakout risk box built for trending/pullback sleeves): a mean-reverting
spread's own daily noise (ATR) is tiny by construction (that's *why* it's a
good stat-arb candidate), so forcing a "SL = 1.5xATR" box onto it makes the
stop almost meaningless relative to round-trip friction -- see
FINAL_REPORT_STATARB.md for the quantified failure of that mismatch.

Classic pairs-trading risk management instead (Gatev/Goetzmann/Rouwenhorst
2006; Avellaneda & Lee 2010 style): entry at |z|>=entry_z, take profit when
the spread reverts to exit_z (usually 0), hard stop if the spread diverges
FURTHER to stop_z (cointegration-breakdown protection), and a time stop at
max_hold_days (half-life-scaled) because a cointegrating relationship can
structurally break and must not be held forever waiting for a reversion that
will never come.
"""
from __future__ import annotations
import numpy as np
from engine import Trade

DEFAULT_STATARB_KWARGS = dict(
    risk_dollars=12.5,
    friction_bps_roundtrip=82.0,  # 2-leg round trip (41bps x 2 legs)
    max_hold_days=60,
)


def generate_statarb_trades(df, entries, sides, std_series, pair_id, sim_kwargs=None,
                             start_idx=0, end_idx=None):
    kw = dict(DEFAULT_STATARB_KWARGS)
    if sim_kwargs:
        kw.update(sim_kwargs)

    o = df["open"].to_numpy()
    c = df["close"].to_numpy()
    idx = df.index
    std_arr = std_series.to_numpy()
    entries_arr = entries.to_numpy()
    sides_arr = sides.to_numpy()
    n = len(df)
    if end_idx is None:
        end_idx = n

    entry_z = kw["entry_z"]
    exit_z = kw.get("exit_z", 0.0)
    stop_z = entry_z + kw.get("stop_extra_z", 1.5)
    max_hold = kw.get("max_hold_days", 60)
    friction_half = (kw["friction_bps_roundtrip"] / 2.0) / 10000.0

    trades = []
    i = max(start_idx, 0)
    last_exit_i = -1
    while i < min(end_idx, n - 1):
        if not entries_arr[i] or np.isnan(std_arr[i]) or std_arr[i] <= 0:
            i += 1
            continue
        if i <= last_exit_i:
            i += 1
            continue
        entry_i = i + 1
        if entry_i >= n:
            break
        side = int(sides_arr[i])
        entry_price = o[entry_i]
        std_at_entry = std_arr[i]

        dist_profit = abs(entry_z - exit_z) * std_at_entry   # spread-units of reversion needed
        dist_stop = abs(stop_z - entry_z) * std_at_entry     # spread-units of further divergence tolerated
        profit_price = entry_price * np.exp(side * dist_profit)
        stop_price = entry_price * np.exp(-side * dist_stop)
        risk_price_dist = abs(entry_price - stop_price)
        if risk_price_dist <= 0:
            i += 1
            continue
        size = kw["risk_dollars"] / risk_price_dist

        tr = Trade(symbol=pair_id, side=side, signal_time=idx[i], entry_time=idx[entry_i],
                   entry_price=entry_price, sl0=stop_price, r_distance=risk_price_dist, size=size,
                   risk_dollars=kw["risk_dollars"])

        exit_price = exit_reason = exit_time = None
        bars_held = 0
        max_j = min(n - 1, entry_i + max_hold)
        j = entry_i
        while j <= max_j:
            bars_held = j - entry_i
            price = c[j]
            hit_stop = (price <= stop_price) if side == 1 else (price >= stop_price)
            hit_profit = (price >= profit_price) if side == 1 else (price <= profit_price)
            if hit_stop:
                exit_price, exit_reason, exit_time = price, "stop_divergence", idx[j]
                break
            if hit_profit:
                exit_price, exit_reason, exit_time = price, "profit_reversion", idx[j]
                break
            if bars_held >= max_hold:
                exit_price, exit_reason, exit_time = price, "time_stop", idx[j]
                break
            j += 1
        if exit_price is None:
            exit_price, exit_reason, exit_time, j = c[max_j], "time_stop", idx[max_j], max_j

        gross_pnl = (exit_price - entry_price) * side * size
        friction_cost = (entry_price * size + exit_price * size) * friction_half
        net_pnl = gross_pnl - friction_cost

        tr.exit_time, tr.exit_price, tr.exit_reason, tr.bars_held = exit_time, exit_price, exit_reason, bars_held
        tr.gross_pnl, tr.friction_cost, tr.net_pnl = gross_pnl, friction_cost, net_pnl
        tr.r_multiple = net_pnl / kw["risk_dollars"]
        trades.append(tr)

        last_exit_i = j
        i = max(i + 1, j)
    return trades
