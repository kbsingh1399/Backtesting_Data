"""
quantlab.labeling
=================
Triple-barrier path simulation with a ratcheting stop, executed bar-by-bar on
the 15m series, with the dataset's own spread charged as a cost.

Barrier conventions (deliberately pessimistic where ambiguous)
--------------------------------------------------------------
* Entry is the **open of the bar after** the signal bar.  Never the signal
  bar's close.
* Within a bar we cannot see the path, so if both the stop and the target are
  inside the bar's range we assume the **stop** filled.  This is the standard
  conservative convention and it matters: the optimistic convention inflates
  hit rate by several points on 15m FX.
* A bar that **gaps through** the stop fills at that bar's open, not at the
  stop level.  Weekend gaps are therefore paid for honestly.
* Positions are force-closed before the weekend.

Costs
-----
Barrier *touch* detection runs on the quoted (bid) series; all friction is
charged once per round turn as a price-unit deduction
(`CostModel.round_turn_cost`), covering spread + slippage + commission.  The
only residual approximation is short-side barrier timing, which is optimistic
by about one spread (<2% of R on the screened universe).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .config import StrategyParams
from .costs import CostModel

EXIT_STOP, EXIT_TARGET, EXIT_TIME, EXIT_WEEKEND, EXIT_EOD = 0, 1, 2, 3, 4
EXIT_NAMES = {0: "stop", 1: "target", 2: "time", 3: "weekend", 4: "eod"}
_COST_KIND = {0: "stop", 1: "target", 2: "time", 3: "time", 4: "time"}


def simulate_events(
    bars: pd.DataFrame,
    events: pd.DataFrame,
    params: StrategyParams,
    cost_model: CostModel,
) -> pd.DataFrame:
    """Attach realised outcomes to each sweep event."""
    if events.empty:
        return events

    o = bars["open"].to_numpy(float)
    h = bars["high"].to_numpy(float)
    l = bars["low"].to_numpy(float)
    c = bars["close"].to_numpy(float)
    n = len(bars)
    dt = bars["datetime"]
    dow = dt.dt.dayofweek.to_numpy()
    hour = dt.dt.hour.to_numpy()
    spread_pts = cost_model.spread_points(bars["spread"].to_numpy(float), hour)
    spread_px = spread_pts * cost_model.point

    ei = events["entry_i"].to_numpy(int)
    side = events["side"].to_numpy(int)
    atr = events["atr"].to_numpy(float)
    struct = events["struct_r_atr"].to_numpy(float) * atr

    r_dist = np.clip(struct, params.min_stop_atr * atr, params.max_stop_atr * atr)
    entry_px = o[ei]
    stop0 = entry_px - side * r_dist
    tp = entry_px + side * params.take_profit_r * r_dist

    trig = np.array([t for t, _ in params.ratchet], dtype=float)
    lock = np.array([k for _, k in params.ratchet], dtype=float)

    m = len(ei)
    exit_i = np.zeros(m, dtype=int)
    exit_px = np.zeros(m)
    exit_kind = np.full(m, EXIT_TIME, dtype=int)
    mfe = np.zeros(m)
    mae = np.zeros(m)

    horizon = params.time_stop_bars

    for k in range(m):
        s = side[k]
        e = entry_px[k]
        R = r_dist[k]
        stop = stop0[k]
        target = tp[k]
        start = ei[k]
        last = min(start + horizon, n - 1)
        best = 0.0
        worst = 0.0
        done = False
        for j in range(start, last + 1):
            # ---- adverse excursion within the bar --------------------
            fav = (h[j] - e) / R if s == 1 else (e - l[j]) / R
            adv = (l[j] - e) / R if s == 1 else (e - h[j]) / R
            best = max(best, fav)
            worst = min(worst, adv)

            # ---- gap through the stop at the open --------------------
            if (s == 1 and o[j] <= stop) or (s == -1 and o[j] >= stop):
                exit_i[k], exit_px[k], exit_kind[k] = j, o[j], EXIT_STOP
                done = True
                break
            # ---- stop before target (pessimistic tie-break) ----------
            if (s == 1 and l[j] <= stop) or (s == -1 and h[j] >= stop):
                exit_i[k], exit_px[k], exit_kind[k] = j, stop, EXIT_STOP
                done = True
                break
            if (s == 1 and h[j] >= target) or (s == -1 and l[j] <= target):
                exit_i[k], exit_px[k], exit_kind[k] = j, target, EXIT_TARGET
                done = True
                break
            # ---- weekend flat ----------------------------------------
            if dow[j] == 4 and hour[j] >= 20:
                exit_i[k], exit_px[k], exit_kind[k] = j, c[j], EXIT_WEEKEND
                done = True
                break
            # ---- ratchet (evaluated on the closed bar) ---------------
            for t_i in range(len(trig)):
                if fav >= trig[t_i]:
                    cand = e + s * lock[t_i] * R
                    stop = max(stop, cand) if s == 1 else min(stop, cand)
        if not done:
            exit_i[k], exit_px[k], exit_kind[k] = last, c[last], EXIT_TIME
        mfe[k], mae[k] = best, worst

    gross_px = side * (exit_px - entry_px)
    cost_px = np.array(
        [
            cost_model.round_turn_cost(
                np.array([entry_px[k]]),
                np.array([spread_px[ei[k]] / cost_model.point]),
                np.array([hour[ei[k]]]),
                _COST_KIND[int(exit_kind[k])],
            )[0]
            for k in range(m)
        ]
    )

    ev = events.copy()
    ev["entry_px"] = entry_px
    ev["stop_px"] = stop0
    ev["tp_px"] = tp
    ev["r_dist"] = r_dist
    ev["r_dist_atr"] = r_dist / atr
    ev["exit_i"] = exit_i
    ev["exit_time"] = dt.to_numpy()[exit_i]
    ev["exit_px"] = exit_px
    ev["exit_kind"] = [EXIT_NAMES[int(x)] for x in exit_kind]
    ev["bars_held"] = exit_i - ei + 1
    ev["r_gross"] = gross_px / r_dist
    ev["cost_r"] = cost_px / r_dist
    ev["r_net"] = (gross_px - cost_px) / r_dist
    ev["mfe_r"] = mfe
    ev["mae_r"] = mae
    ev["label"] = (ev["r_net"] > 0).astype(int)
    return ev


# --------------------------------------------------------------------------
# Sample weighting (Lopez de Prado, AFML ch.4)
# --------------------------------------------------------------------------
def uniqueness_weights(entry_i: np.ndarray, exit_i: np.ndarray, n_bars: int) -> np.ndarray:
    """Average uniqueness of each label, given overlapping holding periods.

    Overlapping triple-barrier labels are not IID; training on them unweighted
    over-counts crowded periods.  We compute, for each bar, how many open
    labels span it, then weight each label by the mean of 1/concurrency over
    its own span.
    """
    if len(entry_i) == 0:
        return np.array([])
    conc = np.zeros(n_bars + 2, dtype=float)
    for a, b in zip(entry_i, exit_i):
        conc[a] += 1.0
        conc[b + 1] -= 1.0
    conc = np.cumsum(conc)[: n_bars + 1]
    conc = np.maximum(conc, 1.0)
    inv = 1.0 / conc
    cs = np.concatenate([[0.0], np.cumsum(inv)])
    out = np.empty(len(entry_i))
    for k, (a, b) in enumerate(zip(entry_i, exit_i)):
        b = min(b, n_bars - 1)
        out[k] = (cs[b + 1] - cs[a]) / max(b - a + 1, 1)
    return out


def time_decay_weights(times: pd.Series, half_life_days: float = 365.0) -> np.ndarray:
    """Exponential recency weighting so stale regimes do not dominate."""
    t = pd.to_datetime(times)
    age_days = (t.max() - t).dt.total_seconds().to_numpy() / 86400.0
    return np.exp(-np.log(2.0) * age_days / half_life_days)
