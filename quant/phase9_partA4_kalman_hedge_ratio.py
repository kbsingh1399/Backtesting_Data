"""
Phase 9, Part A step 4: Kalman-filter dynamic hedge ratio for the 8
cointegrated FX pairs used by Sleeves F/G (both certified DEAD: F=0 trades,
G Sharpe -3.849). STATARB_REPORT.md explicitly named "a time-varying hedge
ratio via a Kalman filter instead of a rolling-OLS window" as a recommended
next step -- this closes that gap.

Standard 2-state local-level Kalman filter (Ernie Chan-style): state
x_t=[alpha_t, beta_t], random-walk transition, observation
log(A)_t = alpha_t + beta_t*log(B)_t + noise. At each t we use the
PREDICTED state (built only from data through t-1) to construct the day-t
spread -- identical no-lookahead discipline to the rolling-OLS version
already in strategies.py (which also lags its beta/alpha by one day).

Direct comparison: raw mean-reversion edge (bps, t-stat) of the
z-scored spread using (a) the existing rolling-252-day-OLS hedge ratio vs
(b) the Kalman-filtered adaptive hedge ratio, on the same 8 pairs, same OOS
window (>=2020-01-01, matching Sleeve F/G's own IS/OOS split).
"""
from __future__ import annotations
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

import strategies as strat
from phase7_screen_part1 import tstat

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OOS_START = pd.Timestamp("2020-01-01", tz="UTC")


def kalman_hedge_ratio(la, lb, delta=1e-5, r_var=None):
    """Returns (alpha_pred, beta_pred) series: the PREDICTED (pre-update,
    i.e. built only from data through t-1) state at each t -- this is what
    must be used to build the no-lookahead spread at time t."""
    n = len(la)
    x = np.array([0.0, 1.0])          # [alpha, beta] initial guess
    P = np.eye(2) * 1.0
    Q = np.eye(2) * delta             # process noise (random-walk step size)
    if r_var is None:
        r_var = float(np.nanvar(la.values[:60] - lb.values[:60])) or 1e-4
    alpha_pred = np.full(n, np.nan)
    beta_pred = np.full(n, np.nan)
    la_v, lb_v = la.values, lb.values
    for t in range(n):
        # Predict step (F=I, random walk)
        x_p = x.copy()
        P_p = P + Q
        alpha_pred[t] = x_p[0]
        beta_pred[t] = x_p[1]
        if np.isnan(la_v[t]) or np.isnan(lb_v[t]):
            x, P = x_p, P_p
            continue
        H = np.array([1.0, lb_v[t]])
        y = la_v[t]
        y_pred = H @ x_p
        e = y - y_pred
        S = H @ P_p @ H.T + r_var
        K = (P_p @ H) / S
        x = x_p + K * e
        P = P_p - np.outer(K, H) @ P_p
    return pd.Series(alpha_pred, index=la.index), pd.Series(beta_pred, index=la.index)


def screen_pair(pair_id, z_lookback=20, z_entry=1.5):
    a_sym, b_sym = pair_id.split("~")
    da = strat.load_forex(a_sym, "d1")["close"]
    db = strat.load_forex(b_sym, "d1")["close"]
    idx = da.index.intersection(db.index)
    la, lb = np.log(da.loc[idx]), np.log(db.loc[idx])

    # (a) existing rolling-OLS hedge ratio (as in strategies.py prep_sleeve_f)
    w = 252
    cov = la.rolling(w).cov(lb).shift(1)
    var = lb.rolling(w).var().shift(1)
    beta_ols = cov / var.replace(0, np.nan)
    mean_a = la.rolling(w).mean().shift(1)
    mean_b = lb.rolling(w).mean().shift(1)
    alpha_ols = mean_a - beta_ols * mean_b
    spread_ols = la - (alpha_ols + beta_ols * lb)

    # (b) Kalman-filtered adaptive hedge ratio
    alpha_kf, beta_kf = kalman_hedge_ratio(la, lb)
    spread_kf = la - (alpha_kf + beta_kf * lb)

    oos_mask = idx >= OOS_START
    hold = 5

    def edge(spread, beta_series):
        """IMPORTANT FIX: a real pairs trade enters at t with hedge ratio
        beta_t FIXED for the life of the trade (exactly how engine.py's
        synthetic index is built -- beta is locked at entry, not re-marked
        daily). Measuring fwd = spread(t+h) - spread(t) when beta itself is
        still moving during the hold conflates "the spread reverted" with
        "the hedge ratio estimate itself drifted", which is not a tradeable
        P&L. This computes the ACTUAL fixed-hedge trade return instead:
        pnl = (logA_{t+h}-logA_t) - beta_t*(logB_{t+h}-logB_t)."""
        m = spread.rolling(z_lookback).mean()
        s = spread.rolling(z_lookback).std()
        z = (spread - m) / s.replace(0, np.nan)
        d_la = la.shift(-hold) - la
        d_lb = lb.shift(-hold) - lb
        fixed_hedge_fwd = d_la - beta_series * d_lb
        long_sig = z <= -z_entry
        short_sig = z >= z_entry
        pnl = pd.Series(np.nan, index=spread.index)
        pnl[long_sig] = fixed_hedge_fwd[long_sig]
        pnl[short_sig] = -fixed_hedge_fwd[short_sig]
        pnl_oos = pnl[oos_mask].dropna()
        return pnl_oos

    pnl_ols = edge(spread_ols, beta_ols)
    pnl_kf = edge(spread_kf, beta_kf)
    mean_ols, t_ols, n_ols = tstat(pnl_ols) if len(pnl_ols) > 5 else (np.nan, np.nan, 0)
    mean_kf, t_kf, n_kf = tstat(pnl_kf) if len(pnl_kf) > 5 else (np.nan, np.nan, 0)

    # Half-life of the KF spread (OU AR(1) regression) for reference
    try:
        from statsmodels.regression.linear_model import OLS
        from statsmodels.tools import add_constant
        s = spread_kf.dropna()
        ds = s.diff().dropna()
        slag = s.shift(1).loc[ds.index]
        m = OLS(ds.values, add_constant(slag.values)).fit()
        theta = -m.params[1]
        half_life_kf = np.log(2) / theta if theta > 0 else np.nan
    except Exception:
        half_life_kf = np.nan

    return dict(pair=pair_id, n_oos_ols=int(n_ols), mean_ols_bps=round(float(mean_ols) * 1e4, 2) if n_ols else None,
                t_ols=round(float(t_ols), 2) if n_ols else None,
                n_oos_kf=int(n_kf), mean_kf_bps=round(float(mean_kf) * 1e4, 2) if n_kf else None,
                t_kf=round(float(t_kf), 2) if n_kf else None,
                half_life_kf_days=round(float(half_life_kf), 1) if not np.isnan(half_life_kf) else None,
                beta_kf_final=round(float(beta_kf.dropna().iloc[-1]), 4) if beta_kf.dropna().size else None,
                beta_ols_final=round(float(beta_ols.dropna().iloc[-1]), 4) if beta_ols.dropna().size else None)


def main():
    pairs = strat.SLEEVE_F_PAIRS
    results = [screen_pair(p) for p in pairs]
    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    pooled_ols_mean = np.nanmean([r["mean_ols_bps"] for r in results if r["mean_ols_bps"] is not None])
    pooled_kf_mean = np.nanmean([r["mean_kf_bps"] for r in results if r["mean_kf_bps"] is not None])
    print(f"\nPooled mean edge (OLS hedge ratio):     {pooled_ols_mean:.2f} bps per signal")
    print(f"Pooled mean edge (Kalman hedge ratio):   {pooled_kf_mean:.2f} bps per signal")
    print("(Note: this measures spread-reversion P&L in log-price-spread units, pre-friction, as a")
    print(" relative-quality comparison of hedge-ratio estimation methods -- not a full WFO P&L.)")

    with open(os.path.join(RESULTS_DIR, "PHASE9_kalman_vs_ols_hedge_ratio.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nWrote results/PHASE9_kalman_vs_ols_hedge_ratio.json")


if __name__ == "__main__":
    main()
