"""
quantlab.metrics
================
Performance statistics, with the ones that actually discriminate between
"found something" and "looked at a lot of things".

In particular:
  * **Deflated Sharpe Ratio** (Bailey & Lopez de Prado, 2014) - adjusts the
    observed Sharpe for the number of configurations tried, the length of
    the sample, and the skew/kurtosis of returns.  If you ran 100 trials, a
    Sharpe of 1.5 on 2 years of data is not evidence of anything, and the DSR
    says so numerically.
  * **Probabilistic Sharpe Ratio** - P(true SR > benchmark).
  * **Stationary bootstrap CI** on the Sharpe, which respects serial
    dependence that an IID bootstrap would destroy.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

EULER = 0.5772156649015329
TRADING_DAYS = 252


# --------------------------------------------------------------------------
def drawdown_curve(equity: np.ndarray) -> np.ndarray:
    peak = np.maximum.accumulate(equity)
    return (equity - peak) / np.where(peak > 0, peak, np.nan)


def max_drawdown(equity: np.ndarray) -> float:
    return float(np.nanmin(drawdown_curve(equity))) if len(equity) else 0.0


def sharpe(returns: np.ndarray, periods_per_year: int = TRADING_DAYS) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 3 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year))


def sortino(returns: np.ndarray, periods_per_year: int = TRADING_DAYS) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    dn = r[r < 0]
    if len(r) < 3 or len(dn) == 0 or dn.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / dn.std(ddof=1) * np.sqrt(periods_per_year))


# --------------------------------------------------------------------------
def probabilistic_sharpe(sr: float, n: int, skew: float, kurt: float, sr_bench: float = 0.0) -> float:
    """P(true SR > sr_bench) given the observed SR and its higher moments."""
    if n < 3:
        return np.nan
    denom = np.sqrt(max(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2, 1e-12))
    return float(stats.norm.cdf((sr - sr_bench) * np.sqrt(n - 1) / denom))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """E[max SR] under the null that all `n_trials` strategies have SR = 0."""
    n = max(int(n_trials), 2)
    z1 = stats.norm.ppf(1.0 - 1.0 / n)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n * np.e))
    return float(np.sqrt(max(sr_variance, 0.0)) * ((1 - EULER) * z1 + EULER * z2))


def deflated_sharpe(
    returns: np.ndarray,
    n_trials: int,
    periods_per_year: int = TRADING_DAYS,
    trial_sharpes: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """Deflated Sharpe Ratio.

    `trial_sharpes` lets you use the *observed* dispersion of Sharpes across
    the configurations you actually tried, which is the correct variance for
    the null; otherwise we fall back to the asymptotic 1/(n-1) estimate.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 10:
        return {"sharpe": 0.0, "dsr": np.nan, "psr": np.nan, "sr0": np.nan, "n_trials": n_trials}

    sr_period = r.mean() / r.std(ddof=1) if r.std(ddof=1) > 0 else 0.0
    sr_ann = sr_period * np.sqrt(periods_per_year)
    sk = float(stats.skew(r))
    ku = float(stats.kurtosis(r, fisher=False))

    if trial_sharpes is not None and len(trial_sharpes) > 2:
        srv = float(np.var(np.asarray(trial_sharpes, dtype=float) / np.sqrt(periods_per_year), ddof=1))
    else:
        srv = 1.0 / max(n - 1, 1)
    sr0 = expected_max_sharpe(n_trials, srv)

    return {
        "sharpe": round(sr_ann, 3),
        "psr": round(probabilistic_sharpe(sr_period, n, sk, ku, 0.0), 4),
        "sr0_period": round(sr0, 4),
        "sr0_ann": round(sr0 * np.sqrt(periods_per_year), 3),
        "dsr": round(probabilistic_sharpe(sr_period, n, sk, ku, sr0), 4),
        "n_trials": int(n_trials),
        "skew": round(sk, 3),
        "kurtosis": round(ku, 3),
    }


def stationary_bootstrap_sharpe_ci(
    returns: np.ndarray,
    n_boot: int = 2000,
    mean_block: int = 10,
    periods_per_year: int = TRADING_DAYS,
    seed: int = 0,
) -> Dict[str, float]:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 30:
        return {}
    rng = np.random.default_rng(seed)
    p = 1.0 / mean_block
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.empty(n, dtype=int)
        i = rng.integers(n)
        for t in range(n):
            idx[t] = i
            i = rng.integers(n) if rng.random() < p else (i + 1) % n
        s = r[idx]
        out[b] = s.mean() / s.std(ddof=1) * np.sqrt(periods_per_year) if s.std(ddof=1) > 0 else 0.0
    return {
        "sharpe_ci_lo": round(float(np.percentile(out, 2.5)), 3),
        "sharpe_ci_hi": round(float(np.percentile(out, 97.5)), 3),
        "p_sharpe_gt_0": round(float((out > 0).mean()), 4),
    }


# --------------------------------------------------------------------------
def performance_summary(
    daily_returns: pd.Series,
    equity: pd.Series,
    trades: Optional[pd.DataFrame] = None,
    n_trials: int = 1,
    trial_sharpes: Optional[Sequence[float]] = None,
) -> Dict:
    r = daily_returns.dropna().to_numpy()
    eq = equity.to_numpy()
    years = max(len(r) / TRADING_DAYS, 1e-9)
    total = eq[-1] / eq[0] - 1.0 if len(eq) > 1 and eq[0] > 0 else 0.0
    cagr = (1.0 + total) ** (1.0 / years) - 1.0 if total > -1 else -1.0
    mdd = max_drawdown(eq)

    out: Dict = {
        "start": str(daily_returns.index[0].date()) if len(daily_returns) else None,
        "end": str(daily_returns.index[-1].date()) if len(daily_returns) else None,
        "days": int(len(r)),
        "total_return_pct": round(total * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "ann_vol_pct": round(float(r.std(ddof=1) * np.sqrt(TRADING_DAYS)) * 100, 2) if len(r) > 2 else 0.0,
        "sharpe": round(sharpe(r), 3),
        "sortino": round(sortino(r), 3),
        "max_drawdown_pct": round(mdd * 100, 2),
        "calmar": round(cagr / abs(mdd), 3) if mdd < 0 else np.nan,
        "best_day_pct": round(float(r.max()) * 100, 2) if len(r) else 0.0,
        "worst_day_pct": round(float(r.min()) * 100, 2) if len(r) else 0.0,
        "pct_positive_days": round(float((r > 0).mean()) * 100, 1) if len(r) else 0.0,
    }
    out.update(deflated_sharpe(r, n_trials, trial_sharpes=trial_sharpes))
    out.update(stationary_bootstrap_sharpe_ci(r))

    if trades is not None and len(trades):
        rn = trades["r_net"].to_numpy()
        pnl = trades["pnl"].to_numpy() if "pnl" in trades else rn
        gp, gl = pnl[pnl > 0].sum(), -pnl[pnl <= 0].sum()
        out.update({
            "n_trades": int(len(trades)),
            "trades_per_day": round(len(trades) / max(len(r), 1), 2),
            "win_rate_pct": round(float((rn > 0).mean()) * 100, 2),
            "avg_R": round(float(rn.mean()), 4),
            "expectancy_R": round(float(rn.mean()), 4),
            "profit_factor": round(float(gp / gl), 3) if gl > 0 else np.inf,
            "avg_cost_R": round(float(trades["cost_r"].mean()), 4) if "cost_r" in trades else None,
            "t_stat_R": round(float(rn.mean() / (rn.std(ddof=1) / np.sqrt(len(rn)))), 2) if len(rn) > 2 else None,
        })
    return out


def yearly_table(daily_returns: pd.Series) -> pd.DataFrame:
    if daily_returns.empty:
        return pd.DataFrame()
    g = daily_returns.groupby(daily_returns.index.year)
    return pd.DataFrame({
        "return_pct": (g.apply(lambda s: (1 + s).prod() - 1) * 100).round(2),
        "vol_pct": (g.std(ddof=1) * np.sqrt(TRADING_DAYS) * 100).round(2),
        "sharpe": g.apply(lambda s: round(sharpe(s.to_numpy()), 2)),
        "max_dd_pct": g.apply(lambda s: round(max_drawdown((1 + s).cumprod().to_numpy()) * 100, 2)),
        "days": g.size(),
    })
