"""
quantlab.portfolio
==================
Event-driven portfolio simulation over the pooled trade list.

Why this exists rather than `equity += r * fixed_risk`:

A per-signal R-multiple list is **not** a portfolio.  The reference S4 script
compounds every signal sequentially as if trades never overlap, which both
(a) invents capital that was already committed, and (b) destroys the serial
structure that drawdown and Sharpe are computed from.  In reality, 46
instruments firing a 15m rule produce heavy clustering: ten correlated
signals in the same hour are one bet, not ten.

This engine therefore enforces:
  * a hard cap on concurrent positions overall, per instrument, and per
    correlation cluster,
  * risk budgeted as a fraction of *current* equity,
  * optional probability-scaled sizing with a hard cap (no raw Kelly),
  * a daily loss circuit breaker,
and produces a genuine daily equity curve.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import PortfolioParams


@dataclass
class PortfolioResult:
    trades: pd.DataFrame
    daily: pd.DataFrame
    skipped: Dict[str, int]

    @property
    def equity(self) -> pd.Series:
        return self.daily["equity"]

    @property
    def returns(self) -> pd.Series:
        return self.daily["ret"]


def run_portfolio(
    trades: pd.DataFrame,
    pp: PortfolioParams,
    clusters: Optional[Dict[str, int]] = None,
    use_prob: bool = True,
) -> PortfolioResult:
    if trades.empty:
        return PortfolioResult(trades, pd.DataFrame(columns=["equity", "ret"]), {})

    df = trades.sort_values("entry_time").reset_index(drop=True).copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    clusters = clusters or {}

    equity = pp.initial_equity
    open_pos: List[Dict] = []
    taken: List[Dict] = []
    skipped = {"max_total": 0, "max_asset": 0, "max_cluster": 0,
               "prob_gate": 0, "daily_stop": 0}

    realised: List[Dict] = []
    day_start_equity = equity
    cur_day = df["entry_time"].iloc[0].date()
    day_locked = False

    def close_due(now: pd.Timestamp) -> None:
        nonlocal equity, open_pos
        still = []
        for p in open_pos:
            if p["exit_time"] <= now:
                pnl = p["risk_cash"] * p["r_net"]
                equity += pnl
                realised.append({**p, "pnl": pnl, "equity_after": equity})
            else:
                still.append(p)
        open_pos = still

    for _, row in df.iterrows():
        now = row["entry_time"]
        close_due(now)

        if now.date() != cur_day:
            cur_day = now.date()
            day_start_equity = equity
            day_locked = False
        if not day_locked and equity < day_start_equity * (1 - pp.daily_loss_limit_pct / 100.0):
            day_locked = True
        if day_locked:
            skipped["daily_stop"] += 1
            continue

        prob = row.get("prob", np.nan)
        if use_prob and np.isfinite(prob):
            if prob < pp.prob_floor:
                skipped["prob_gate"] += 1
                continue
            scale = np.clip((prob - pp.prob_floor) / max(pp.prob_cap - pp.prob_floor, 1e-9), 0.0, 1.0)
            risk_pct = pp.risk_per_trade_pct * (0.5 + 0.5 * scale) if pp.prob_sizing else pp.risk_per_trade_pct
        else:
            risk_pct = pp.risk_per_trade_pct
        risk_pct = min(risk_pct, pp.max_risk_per_trade_pct)

        sym = row["symbol"]
        cl = clusters.get(sym, -1)
        if len(open_pos) >= pp.max_concurrent_positions:
            skipped["max_total"] += 1
            continue
        if sum(1 for p in open_pos if p["symbol"] == sym) >= pp.max_concurrent_per_asset:
            skipped["max_asset"] += 1
            continue
        if cl >= 0 and sum(1 for p in open_pos if p["cluster"] == cl) >= pp.max_concurrent_per_cluster:
            skipped["max_cluster"] += 1
            continue

        risk_cash = equity * risk_pct / 100.0
        open_pos.append({
            "symbol": sym, "cluster": cl, "entry_time": now,
            "exit_time": row["exit_time"], "r_net": row["r_net"],
            "cost_r": row.get("cost_r", np.nan), "side": row.get("side", 0),
            "prob": prob, "risk_pct": risk_pct, "risk_cash": risk_cash,
            "exit_kind": row.get("exit_kind", ""),
        })

    close_due(pd.Timestamp.max.tz_localize("UTC"))

    tk = pd.DataFrame(realised)
    if tk.empty:
        return PortfolioResult(tk, pd.DataFrame(columns=["equity", "ret"]), skipped)

    tk = tk.sort_values("exit_time").reset_index(drop=True)
    tk["equity_after"] = pp.initial_equity + tk["pnl"].cumsum()

    # Daily curve: PnL is booked on the exit date.  Intraday marks are not
    # available from a trade list, so daily vol is slightly understated; the
    # alternative (compounding per trade) would be far worse.
    daily_pnl = tk.set_index("exit_time")["pnl"].resample("1D").sum()
    idx = pd.date_range(tk["entry_time"].min().floor("D"),
                        tk["exit_time"].max().ceil("D"), freq="1D", tz="UTC")
    daily_pnl = daily_pnl.reindex(idx).fillna(0.0)
    eq = pp.initial_equity + daily_pnl.cumsum()
    ret = eq.pct_change().fillna(0.0)
    daily = pd.DataFrame({"pnl": daily_pnl, "equity": eq, "ret": ret})
    # drop weekends with no activity so Sharpe is on trading days
    daily = daily[~((daily["pnl"] == 0) & (daily.index.dayofweek >= 5))]

    return PortfolioResult(tk, daily, skipped)
