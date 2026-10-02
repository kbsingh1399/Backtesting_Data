"""
Phase 8, Part 3 -- Regime-detection & event-structure battery (Section 6/8
taxonomy): Hidden Markov Models / Markov-switching regimes, change-point
detection (CUSUM via ruptures), jump detection (Barndorff-Nielsen/Shephard
bipower variation), and a Hawkes process fit for jump/event clustering.
Run on the Sleeve M4 pair (XAUUSD, XAGUSD) -- the one certified live result
-- plus a cross-sectional confirmation on a handful of other liquid symbols.
"""
from __future__ import annotations
import json, os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import ruptures as rpt
from scipy.optimize import minimize
from scipy.stats import norm

import strategies as strat

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
SYMS = ["XAUUSD", "XAGUSD", "EURUSD", "COPPER"]


# ---------------------------------------------------------------------------
# 1. HMM / Markov-switching regimes (2-state: calm vs turbulent)
# ---------------------------------------------------------------------------
def hmm_regimes(ret: pd.Series, n_states=2):
    X = ret.values.reshape(-1, 1)
    model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=200, random_state=42)
    model.fit(X)
    states = model.predict(X)
    means = model.means_.flatten()
    vols = np.sqrt(model.covars_.flatten())
    order = np.argsort(vols)  # low-vol state first
    state_map = {old: new for new, old in enumerate(order)}
    mapped = np.array([state_map[s] for s in states])
    return mapped, means[order], vols[order], model.transmat_


# ---------------------------------------------------------------------------
# 2. Change-point detection: CUSUM-style via ruptures (Pelt, RBF cost) --
#    proxy for "Bayesian online" changepoint flavor (both detect structural
#    shifts in the return-generating process without assuming a fixed window)
# ---------------------------------------------------------------------------
def changepoints(ret: pd.Series, penalty=8):
    z = (ret - ret.mean()) / ret.std()
    algo = rpt.Pelt(model="rbf").fit(z.values.reshape(-1, 1))
    bkps = algo.predict(pen=penalty)
    return bkps  # list of breakpoint indices (last = len(series))


# ---------------------------------------------------------------------------
# 3. Jump detection: Barndorff-Nielsen/Shephard bipower variation test.
#    RV (sum of squared daily returns in a rolling window) vs bipower
#    variation BV (robust to jumps) -> jump component = RV - BV; a day is
#    flagged as a "jump day" if its own squared return is a statistical
#    outlier relative to the local BV-implied continuous volatility.
# ---------------------------------------------------------------------------
def bns_jump_flags(ret: pd.Series, window=22, z_thresh=3.0):
    r = ret.values
    n = len(r)
    mu1 = np.sqrt(2 / np.pi)
    bv = pd.Series(np.nan, index=ret.index)
    for i in range(window, n):
        seg = r[i - window:i]
        bv.iloc[i] = (1.0 / mu1 ** 2) * np.sum(np.abs(seg[1:]) * np.abs(seg[:-1])) / (window - 1)
    local_sigma = np.sqrt(bv.clip(lower=1e-12))
    z = ret / local_sigma
    jump_flag = z.abs() > z_thresh
    return jump_flag, z


# ---------------------------------------------------------------------------
# 4. Hawkes process (self-exciting event clustering) fit on jump-day
#    timestamps via MLE of a 1D exponential-kernel Hawkes process.
#    lambda(t) = mu + sum_{t_i<t} alpha*exp(-beta*(t-t_i))
#    branching ratio n = alpha/beta: n close to/above 1 => jumps cluster
#    strongly (self-exciting); n near 0 => jumps arrive ~independently (Poisson).
# ---------------------------------------------------------------------------
def fit_hawkes(event_times_days: np.ndarray, T: float):
    """Bounded MLE (beta capped at 5/day -> excitation can't decay faster than
    ~5hr-equivalent, keeping the fit numerically meaningful for daily-spaced
    discrete event data instead of degenerating to an instantaneous spike)."""
    def neg_loglik(theta):
        mu, alpha, beta = theta
        if mu <= 1e-6 or alpha < 0 or beta <= 1e-3 or alpha >= beta:
            return 1e10
        n = len(event_times_days)
        ll = 0.0
        A = 0.0
        prev_t = 0.0
        for i in range(n):
            t = event_times_days[i]
            A = A * np.exp(-beta * (t - prev_t)) + (1 if i > 0 else 0)
            lam = mu + alpha * A
            if lam <= 0:
                return 1e10
            ll += np.log(lam)
            prev_t = t
        compensator = mu * T + (alpha / beta) * np.sum(1 - np.exp(-beta * (T - event_times_days)))
        return -(ll - compensator)

    bounds = [(1e-4, 1.0), (1e-4, 4.9), (1e-3, 5.0)]
    best = None
    for mu0, a0, b0 in [(0.02, 0.1, 0.5), (0.02, 0.3, 1.0), (0.05, 0.05, 0.3), (0.01, 0.5, 2.0)]:
        res = minimize(neg_loglik, [mu0, a0, b0], method="L-BFGS-B", bounds=bounds)
        if best is None or res.fun < best.fun:
            best = res
    mu, alpha, beta = best.x
    branching_ratio = alpha / beta if beta > 0 else np.nan

    # Robustness cross-check independent of the MLE: Fano factor (variance/mean
    # of event counts in fixed windows) and gaps-vs-exponential KS test --
    # Fano>1 / KS-reject => over-dispersed (clustered) relative to a Poisson
    # process, corroborating or contradicting the Hawkes branching ratio.
    gaps = np.diff(np.concatenate([[0.0], event_times_days]))
    rate = len(event_times_days) / T
    from scipy.stats import kstest, expon
    ks_stat, ks_p = kstest(gaps, expon(scale=1 / rate).cdf)
    win = 60
    nbins = int(T // win)
    counts = np.histogram(event_times_days, bins=np.linspace(0, nbins * win, nbins + 1))[0] if nbins > 2 else np.array([len(event_times_days)])
    fano = counts.var() / counts.mean() if counts.mean() > 0 else np.nan

    return {"mu": round(float(mu), 5), "alpha": round(float(alpha), 5), "beta": round(float(beta), 5),
            "branching_ratio": round(float(branching_ratio), 4), "neg_loglik": round(float(best.fun), 2),
            "fano_factor_60d_windows": round(float(fano), 3) if fano == fano else None,
            "ks_gaps_vs_exponential_p": round(float(ks_p), 4)}


def main():
    all_results = {}
    for sym in SYMS:
        df = strat.load_forex(sym, "d1")
        ret = (np.log(df["close"]).diff() * 100).dropna()
        ret = ret.tail(2500)  # cap for compute

        print(f"\n=== {sym} (n={len(ret)}) ===")

        # 1. HMM
        states, means, vols, transmat = hmm_regimes(ret)
        pct_turbulent = round(100 * (states == 1).mean(), 1)
        persistence_calm = round(float(transmat[0, 0]), 3)
        persistence_turb = round(float(transmat[1, 1]), 3)
        print(f"HMM: calm_vol={vols[0]:.3f} turbulent_vol={vols[1]:.3f} "
              f"%time_turbulent={pct_turbulent}% persistence(calm->calm)={persistence_calm} "
              f"persistence(turb->turb)={persistence_turb}")

        # 2. Changepoints (standardized series, dimensionless RBF penalty)
        bkps = changepoints(ret, penalty=5)
        n_breaks = len(bkps) - 1
        avg_segment_len = len(ret) / max(n_breaks, 1)
        print(f"Changepoints (Pelt/RBF): {n_breaks} structural breaks detected, "
              f"avg segment length {avg_segment_len:.0f} days")

        # 3. BN-S jump detection
        jump_flag, z = bns_jump_flags(ret)
        n_jumps = int(jump_flag.sum())
        pct_jumps = round(100 * jump_flag.mean(), 2)
        print(f"BN-S jump test: {n_jumps} jump days flagged ({pct_jumps}% of sample, |z|>3 vs local bipower-variation vol)")

        # 4. Hawkes on jump-day clustering
        jump_days_idx = np.where(jump_flag.values)[0]
        hawkes_res = None
        if len(jump_days_idx) >= 15:
            T = float(len(ret))
            hawkes_res = fit_hawkes(jump_days_idx.astype(float), T)
            print(f"Hawkes fit on jump events: mu={hawkes_res['mu']} alpha={hawkes_res['alpha']} "
                  f"beta={hawkes_res['beta']} branching_ratio={hawkes_res['branching_ratio']} "
                  f"({'SELF-EXCITING/clustered' if hawkes_res['branching_ratio']>0.3 else 'near-Poisson/independent'})")

        all_results[sym] = {
            "n_obs": len(ret),
            "hmm_calm_vol": round(float(vols[0]), 4), "hmm_turbulent_vol": round(float(vols[1]), 4),
            "hmm_pct_time_turbulent": pct_turbulent,
            "hmm_persistence_calm": persistence_calm, "hmm_persistence_turbulent": persistence_turb,
            "n_structural_breaks": n_breaks, "avg_segment_len_days": round(avg_segment_len, 1),
            "n_jump_days": n_jumps, "pct_jump_days": pct_jumps,
            "hawkes": hawkes_res,
        }

    path = os.path.join(RESULTS_DIR, "PHASE8_part3_hmm_changepoint_jumps_hawkes.json")
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
