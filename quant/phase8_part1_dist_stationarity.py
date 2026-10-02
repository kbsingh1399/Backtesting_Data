"""
Phase 8, Part 1 -- Return-distribution and stationarity battery (Section 6
taxonomy items: fat tails/skew/kurtosis, ADF/KPSS/Phillips-Perron, ACF/PACF,
variance-ratio test, Hurst exponent confirmation) run across the full
98-symbol D1 universe, with a focused deep-dive on the Sleeve M4 pair
(XAUUSD, XAGUSD) since that's the one certified live result everything else
should be checked against.
"""
from __future__ import annotations
import json, os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf
from arch.unitroot import ADF, KPSS, PhillipsPerron, VarianceRatio

import strategies as strat
from phase7_screen_part1 import ALL_SYMBOLS, load_quietly
from strategies import rolling_hurst

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def jarque_bera(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    m = x.mean()
    s2 = ((x - m) ** 2).mean()
    s3 = ((x - m) ** 3).mean()
    s4 = ((x - m) ** 4).mean()
    skew = s3 / s2 ** 1.5
    kurt = s4 / s2 ** 2  # excess kurtosis = kurt - 3
    jb = n / 6.0 * (skew ** 2 + (kurt - 3) ** 2 / 4.0)
    from scipy.stats import chi2
    p = 1 - chi2.cdf(jb, df=2)
    return skew, kurt - 3.0, jb, p


def per_symbol_battery(sym):
    df = load_quietly(sym)
    if df is None or len(df) < 300:
        return None
    ret = np.log(df["close"]).diff().dropna()
    out = {"symbol": sym, "n_obs": len(ret)}

    skew, exkurt, jb, jb_p = jarque_bera(ret.values)
    out["skew"] = round(float(skew), 4)
    out["excess_kurtosis"] = round(float(exkurt), 4)
    out["jarque_bera_stat"] = round(float(jb), 2)
    out["jarque_bera_p"] = round(float(jb_p), 6)
    out["fat_tailed"] = bool(jb_p < 0.01)

    # stationarity battery on PRICE (should be non-stationary -> random-walk-like)
    try:
        adf_p = ADF(df["close"].values).pvalue
    except Exception:
        adf_p = np.nan
    try:
        kpss_p = KPSS(df["close"].values).pvalue
    except Exception:
        kpss_p = np.nan
    out["price_adf_p"] = round(float(adf_p), 4) if adf_p == adf_p else None
    out["price_kpss_p"] = round(float(kpss_p), 4) if kpss_p == kpss_p else None

    # stationarity battery on RETURNS (should be stationary)
    try:
        r_adf_p = ADF(ret.values).pvalue
    except Exception:
        r_adf_p = np.nan
    try:
        r_kpss_p = KPSS(ret.values).pvalue
    except Exception:
        r_kpss_p = np.nan
    try:
        r_pp_p = PhillipsPerron(ret.values).pvalue
    except Exception:
        r_pp_p = np.nan
    out["ret_adf_p"] = round(float(r_adf_p), 6) if r_adf_p == r_adf_p else None
    out["ret_kpss_p"] = round(float(r_kpss_p), 4) if r_kpss_p == r_kpss_p else None
    out["ret_pp_p"] = round(float(r_pp_p), 6) if r_pp_p == r_pp_p else None
    out["ret_stationary_adf"] = bool(r_adf_p < 0.05) if r_adf_p == r_adf_p else None
    out["ret_stationary_kpss"] = bool(r_kpss_p > 0.05) if r_kpss_p == r_kpss_p else None  # KPSS null = stationary

    # ACF/PACF of returns -- first 5 lags, flag any |acf|>2/sqrt(n) (approx 95% band)
    try:
        acf_vals = acf(ret.values, nlags=10, fft=True)
        pacf_vals = pacf(ret.values, nlags=10)
        band = 1.96 / np.sqrt(len(ret))
        sig_acf_lags = [i for i in range(1, 11) if abs(acf_vals[i]) > band]
        sig_pacf_lags = [i for i in range(1, 11) if abs(pacf_vals[i]) > band]
        out["acf_lag1"] = round(float(acf_vals[1]), 4)
        out["pacf_lag1"] = round(float(pacf_vals[1]), 4)
        out["acf_significant_lags_1to10"] = sig_acf_lags
        out["pacf_significant_lags_1to10"] = sig_pacf_lags
    except Exception:
        pass

    # Variance ratio test (Lo-MacKinlay), q=2,5,10 -- VR=1 under random walk
    vr_results = {}
    for q in (2, 5, 10):
        try:
            vr = VarianceRatio(ret.values, lags=q)
            vr_results[f"q{q}"] = {"vr_stat": round(float(vr.vr), 4), "p_value": round(float(vr.pvalue), 4)}
        except Exception:
            pass
    out["variance_ratio"] = vr_results

    # Hurst exponent (confirm/extend Sleeve P's indices-only result to full universe)
    try:
        h = rolling_hurst(df["close"], window=min(250, len(df) - 5), lag_max=20)
        out["hurst_fullsample_est"] = round(float(h.dropna().iloc[-1]), 4) if h.dropna().size else None
        out["hurst_mean_rolling100"] = round(float(rolling_hurst(df["close"], window=100).dropna().mean()), 4)
    except Exception:
        pass

    return out


def main():
    print(f"Running distribution/stationarity battery on {len(ALL_SYMBOLS)} symbols (D1)...")
    rows = []
    for sym in ALL_SYMBOLS:
        r = per_symbol_battery(sym)
        if r is not None:
            rows.append(r)
    print(f"  completed {len(rows)}/{len(ALL_SYMBOLS)} symbols")

    df_out = pd.DataFrame(rows)

    summary = {
        "n_symbols": len(rows),
        "pct_fat_tailed_jb": round(100 * df_out["fat_tailed"].mean(), 1),
        "mean_excess_kurtosis": round(float(df_out["excess_kurtosis"].mean()), 3),
        "median_excess_kurtosis": round(float(df_out["excess_kurtosis"].median()), 3),
        "mean_skew": round(float(df_out["skew"].mean()), 3),
        "pct_ret_stationary_adf_p05": round(100 * df_out["ret_stationary_adf"].mean(), 1),
        "pct_ret_stationary_kpss_p05": round(100 * df_out["ret_stationary_kpss"].dropna().mean(), 1),
        "pct_price_nonstationary_adf_p05": round(100 * (df_out["price_adf_p"] > 0.05).mean(), 1),
        "pct_with_significant_acf_lag1": round(100 * (df_out["acf_lag1"].abs() > 1.96 / np.sqrt(df_out["n_obs"])).mean(), 1),
    }

    m4_syms = ["XAUUSD", "XAGUSD"]
    m4_detail = df_out[df_out["symbol"].isin(m4_syms)].to_dict(orient="records")

    out = {"summary": summary, "m4_pair_detail": m4_detail, "per_symbol": rows}
    path = os.path.join(RESULTS_DIR, "PHASE8_part1_dist_stationarity.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps(summary, indent=2))
    print("M4 pair detail:")
    for r in m4_detail:
        print(f"  {r['symbol']}: skew={r['skew']}, exkurt={r['excess_kurtosis']}, "
              f"JB_p={r['jarque_bera_p']}, ret_ADF_p={r['ret_adf_p']}, "
              f"ret_KPSS_p={r['ret_kpss_p']}, hurst={r.get('hurst_mean_rolling100')}, "
              f"VR_q5={r['variance_ratio'].get('q5')}")
    print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
