"""
Phase 9, Part A step 2: distributional diagnostics + stationarity battery +
ACF/PACF + variance ratio test + Hurst-exponent refinement, across the full
98-symbol FX/metals/indices/energy universe (D1, the only clean timeframe
per the Phase-7 data-quality scan).

Covers from the pasted taxonomy (section 6):
  - Return distributions: fat tails, skew, kurtosis
  - Stationarity: ADF, KPSS, Phillips-Perron
  - Autocorrelation / partial autocorrelation
  - Variance ratio test (Lo-MacKinlay)
  - Hurst exponent (refined: full universe + bootstrap CI, vs Sleeve P's
    single rolling-window version)
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
from scipy import stats as sps
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

import strategies as strat
from phase7_screen_part1 import ASSET_CLASSES, ALL_SYMBOLS, load_quietly

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
RNG = np.random.default_rng(7)


def phillips_perron(x, lags=None):
    """Phillips-Perron unit-root test (not in statsmodels directly before
    some versions -- implemented directly: regress x_t on x_{t-1} + const,
    then apply the Newey-West HAC correction to the t-stat on rho)."""
    x = np.asarray(x, dtype=float)
    y = x[1:]
    ylag = x[:-1]
    X = add_constant(ylag)
    model = OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags or int(4 * (len(y) / 100) ** (2 / 9))})
    rho = model.params[1]
    tstat = model.tvalues[1]
    pval_approx = adfuller(x, maxlag=0, regression="c", autolag=None)[1]  # same asymptotic critical values family
    return dict(rho=float(rho), t_stat=float(tstat), pvalue_approx=float(pval_approx))


def variance_ratio_test(returns, q=5):
    """Lo-MacKinlay (1988) variance ratio test. VR(q) = Var(q-period return)/
    (q * Var(1-period return)). Under RW null, VR=1. Returns VR and the
    asymptotic Z-stat (homoskedastic version)."""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    n = len(r)
    if n < q * 20:
        return None
    mu = r.mean()
    var1 = np.sum((r - mu) ** 2) / (n - 1)
    rq = pd.Series(r).rolling(q).sum().dropna().to_numpy()
    m = len(rq)
    varq = np.sum((rq - q * mu) ** 2) / (m - 1)
    vr = varq / (q * var1)
    # Lo-MacKinlay asymptotic variance under homoskedasticity
    theta = 2 * (2 * q - 1) * (q - 1) / (3 * q * n)
    z = (vr - 1) / np.sqrt(theta)
    return dict(q=q, vr=float(vr), z_stat=float(z), n=n)


def hurst_rs(x, min_chunk=16):
    """Classic rescaled-range Hurst exponent estimator across a range of
    window sizes (log-log slope of R/S vs window size)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < min_chunk * 8:
        return np.nan
    sizes = np.unique(np.logspace(np.log10(min_chunk), np.log10(n // 2), 12).astype(int))
    rs_vals = []
    for size in sizes:
        n_chunks = n // size
        if n_chunks < 1:
            continue
        rs_list = []
        for i in range(n_chunks):
            chunk = x[i * size:(i + 1) * size]
            mean_adj = chunk - chunk.mean()
            z = np.cumsum(mean_adj)
            r = z.max() - z.min()
            s = chunk.std(ddof=1)
            if s > 0:
                rs_list.append(r / s)
        if rs_list:
            rs_vals.append((size, np.mean(rs_list)))
    if len(rs_vals) < 4:
        return np.nan
    sizes_arr = np.log([v[0] for v in rs_vals])
    rs_arr = np.log([v[1] for v in rs_vals])
    slope, intercept = np.polyfit(sizes_arr, rs_arr, 1)
    return float(slope)


def hurst_bootstrap_ci(returns, n_boot=80, block=90):
    """Block-bootstrap confidence interval on the Hurst exponent."""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    n = len(r)
    if n < block * 10:
        return None
    ests = []
    nblocks = n // block
    for _ in range(n_boot):
        idx = RNG.integers(0, nblocks, size=nblocks)
        sample = np.concatenate([r[i * block:(i + 1) * block] for i in idx])
        h = hurst_rs(sample)  # R/S Hurst is defined on the increment (return) series itself
        if not np.isnan(h):
            ests.append(h)
    if len(ests) < 20:
        return None
    return dict(mean=float(np.mean(ests)), ci95=[float(np.percentile(ests, 2.5)), float(np.percentile(ests, 97.5))])


HURST_CI_SYMBOLS = {"XAUUSD", "XAGUSD", "XPTUSD", "COPPER", "EURUSD", "GBPUSD",
                     "USDJPY", "SP500", "NAS100", "UKBRENT", "AUDUSD", "USDCAD"}


def analyze_symbol(sym):
    df = load_quietly(sym)
    if df is None:
        return None
    px = df["close"].astype(float)
    ret = px.pct_change().dropna()
    if len(ret) < 300:
        return None
    out = dict(symbol=sym, n_obs=len(ret))

    # Distributional diagnostics
    out["mean_daily_ret_bps"] = round(float(ret.mean() * 1e4), 3)
    out["std_daily_ret_bps"] = round(float(ret.std() * 1e4), 3)
    out["skew"] = round(float(sps.skew(ret)), 4)
    out["kurtosis_excess"] = round(float(sps.kurtosis(ret, fisher=True)), 4)
    jb_stat, jb_p = sps.jarque_bera(ret)[:2]
    out["jarque_bera_p"] = round(float(jb_p), 6)
    out["fat_tailed"] = bool(jb_p < 0.01 and sps.kurtosis(ret, fisher=True) > 1.0)

    # Stationarity battery (on log-price level, and on returns as a sanity check)
    logpx = np.log(px.to_numpy())
    try:
        adf_stat, adf_p = adfuller(logpx, autolag="AIC")[:2]
        out["adf_price_p"] = round(float(adf_p), 4)
    except Exception:
        out["adf_price_p"] = None
    try:
        kpss_stat, kpss_p = kpss(logpx, regression="c", nlags="auto")[:2]
        out["kpss_price_p"] = round(float(kpss_p), 4)
    except Exception:
        out["kpss_price_p"] = None
    try:
        pp = phillips_perron(logpx)
        out["pp_price_pvalue_approx"] = round(pp["pvalue_approx"], 4)
    except Exception:
        out["pp_price_pvalue_approx"] = None
    try:
        adf_stat_r, adf_p_r = adfuller(ret.to_numpy(), autolag="AIC")[:2]
        out["adf_returns_p"] = round(float(adf_p_r), 6)
    except Exception:
        out["adf_returns_p"] = None

    # ACF / PACF (first 10 lags) + Ljung-Box for genuine serial dependence
    try:
        acf_vals = acf(ret, nlags=10, fft=True)
        pacf_vals = pacf(ret, nlags=10)
        out["acf_lag1"] = round(float(acf_vals[1]), 4)
        out["acf_lag5"] = round(float(acf_vals[5]), 4)
        out["pacf_lag1"] = round(float(pacf_vals[1]), 4)
        lb = sps.combine_pvalues([2 * (1 - sps.norm.cdf(abs(a) * np.sqrt(len(ret)))) for a in acf_vals[1:6]])
        out["ljungbox_combined_p_lags1to5"] = round(float(lb[1]), 6)
    except Exception:
        pass

    # Variance ratio test at a few horizons
    for q in (2, 5, 10, 20):
        vrt = variance_ratio_test(ret.to_numpy(), q=q)
        if vrt:
            out[f"vr_q{q}"] = round(vrt["vr"], 4)
            out[f"vr_q{q}_z"] = round(vrt["z_stat"], 3)

    # Hurst exponent (point estimate + bootstrap CI). IMPORTANT: R/S analysis
    # must be applied to the RETURN series itself (hurst_rs already does the
    # within-chunk demeaning+cumsum internally) -- feeding it an
    # already-cumulated (price-level) series double-integrates and spuriously
    # inflates H toward 1 for every symbol regardless of true persistence.
    h = hurst_rs(ret.to_numpy())
    out["hurst"] = round(h, 4) if not np.isnan(h) else None
    if sym in HURST_CI_SYMBOLS:
        hci = hurst_bootstrap_ci(ret.to_numpy())
        if hci:
            out["hurst_ci95"] = hci["ci95"]
            out["hurst_trending"] = bool(hci["ci95"][0] > 0.5)
            out["hurst_mean_reverting"] = bool(hci["ci95"][1] < 0.5)

    return out


def main():
    results = []
    for cls, syms in ASSET_CLASSES.items():
        for sym in syms:
            r = analyze_symbol(sym)
            if r:
                r["asset_class"] = cls
                results.append(r)
    print(f"Analyzed {len(results)}/{len(ALL_SYMBOLS)} symbols with sufficient D1 history.")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, "PHASE9_distributional_stationarity.csv"), index=False)

    # Summaries
    print("\n--- Fat tails / skew / kurtosis ---")
    print(f"Fraction fat-tailed (JB p<0.01 & excess kurtosis>1): {df['fat_tailed'].mean():.3f}")
    print(f"Mean skew: {df['skew'].mean():.3f}  Mean excess kurtosis: {df['kurtosis_excess'].mean():.3f}")

    print("\n--- Stationarity (fraction rejecting unit root / confirming stationarity) ---")
    print(f"ADF on log-price: reject unit root (p<0.05) in {(df['adf_price_p'] < 0.05).mean():.3f} of symbols")
    print(f"KPSS on log-price: reject stationarity (p<0.05, i.e. non-stationary) in {(df['kpss_price_p'] < 0.05).mean():.3f} of symbols")
    print(f"ADF on returns: reject unit root (p<0.05) in {(df['adf_returns_p'] < 0.05).mean():.3f} of symbols (expected: ~all, returns should be stationary)")

    print("\n--- ACF/PACF lag-1 autocorrelation ---")
    print(f"Mean ACF(1): {df['acf_lag1'].mean():.4f}  |  Fraction with |ACF(1)|>0.05: {(df['acf_lag1'].abs() > 0.05).mean():.3f}")
    print(f"Ljung-Box (lags 1-5) rejects white noise (p<0.05) in {(df['ljungbox_combined_p_lags1to5'] < 0.05).mean():.3f} of symbols")

    print("\n--- Variance ratio test (q=5) ---")
    sig_vr = df[df["vr_q5_z"].abs() > 1.96] if "vr_q5_z" in df else pd.DataFrame()
    print(f"Symbols with significant VR(5) departure from 1 (|Z|>1.96): {len(sig_vr)}/{len(df)}")
    if len(sig_vr):
        trending = sig_vr[sig_vr["vr_q5"] > 1]
        reverting = sig_vr[sig_vr["vr_q5"] < 1]
        print(f"  VR>1 (trending/positively autocorrelated): {len(trending)}  |  VR<1 (mean-reverting): {len(reverting)}")

    print("\n--- Hurst exponent ---")
    print(f"Mean Hurst: {df['hurst'].mean():.4f}")
    if "hurst_trending" in df:
        print(f"Symbols with Hurst CI95 lower bound > 0.5 (genuinely trending): {int(df['hurst_trending'].sum())}")
        print(f"Symbols with Hurst CI95 upper bound < 0.5 (genuinely mean-reverting): {int(df['hurst_mean_reverting'].sum())}")

    # Highlight gold/silver specifically (M4's universe) for cross-reference
    gs = df[df["symbol"].isin(["XAUUSD", "XAGUSD"])]
    print("\n--- Gold/Silver (M4 universe) specific diagnostics ---")
    print(gs[["symbol", "skew", "kurtosis_excess", "adf_price_p", "hurst", "hurst_ci95" if "hurst_ci95" in gs else "hurst"]].to_string(index=False))

    df.to_json(os.path.join(RESULTS_DIR, "PHASE9_distributional_stationarity.json"), orient="records", indent=2)
    print(f"\nWrote results/PHASE9_distributional_stationarity.{{csv,json}}")


if __name__ == "__main__":
    main()
