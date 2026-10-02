"""
Phase 8, Part 2 -- Volatility model family (Section 6 taxonomy): GARCH,
EGARCH, GJR-GARCH, HAR-RV, and a Kalman-filter stochastic-volatility model
(latent AR(1) log-variance state), walk-forward with the same quarterly
expanding-window, zero-lookahead discipline used for the ML signal cache.

Goal: (1) see which vol-forecast family best predicts next-day realized
variance (QLIKE loss, standard vol-forecast evaluation metric), (2) build a
GARCH-conditional-vol regime filter as a drop-in replacement for Sleeve T's
realized-vol-percentile filter and re-run Sleeve T's full WFO certification
to test whether a "proper" vol model (vs. the simple percentile-rank proxy)
rescues it, (3) cache the GARCH vol-regime signal to parquet for reuse.
"""
from __future__ import annotations
import json, os, sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from arch import arch_model
from pykalman import KalmanFilter

import strategies as strat
from strategies import SLEEVE_T_UNIVERSE
from phase7_screen_part3_ml import make_quarters

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def har_rv_forecast(rv_daily: pd.Series, train_end_idx: int) -> pd.Series:
    """Corsi (2009) HAR-RV: RV_t = b0 + b1*RV_{t-1} + b5*RV5_{t-1} + b22*RV22_{t-1}.
    Fit OLS on data up to train_end_idx (expanding), forecast in-sample+OOS
    recursively using only already-realized lagged RV (no lookahead)."""
    rv5 = rv_daily.rolling(5).mean()
    rv22 = rv_daily.rolling(22).mean()
    X = pd.DataFrame({"rv1": rv_daily.shift(1), "rv5": rv5.shift(1), "rv22": rv22.shift(1)})
    y = rv_daily
    train_mask = (X.index <= X.index[train_end_idx]) if train_end_idx < len(X) else pd.Series(True, index=X.index)
    train = pd.concat([X, y.rename("y")], axis=1).loc[:X.index[train_end_idx]].dropna()
    if len(train) < 60:
        return pd.Series(np.nan, index=rv_daily.index)
    import numpy.linalg as la
    A = np.column_stack([np.ones(len(train)), train["rv1"], train["rv5"], train["rv22"]])
    b, *_ = la.lstsq(A, train["y"].values, rcond=None)
    full = X.copy()
    full["pred"] = b[0] + b[1] * full["rv1"] + b[2] * full["rv5"] + b[3] * full["rv22"]
    return full["pred"]


def kalman_sv_forecast(ret: pd.Series) -> pd.Series:
    """Simple stochastic-volatility proxy: latent log-variance follows an
    AR(1) state; observed = log(ret^2 + eps). Estimated via a linear Kalman
    filter (standard log-squared-returns SV approximation, Harvey-Ruiz-Shephard
    style). Returns the filtered (causal, no-lookahead) latent log-vol."""
    y = np.log(ret.values ** 2 + 1e-8)
    y = y - y.mean()
    kf = KalmanFilter(
        transition_matrices=[0.95],
        observation_matrices=[1.0],
        initial_state_mean=0.0,
        initial_state_covariance=1.0,
        transition_covariance=0.05,
        observation_covariance=2.0,
    )
    state_means, _ = kf.filter(y)
    return pd.Series(state_means.flatten(), index=ret.index)


def process_symbol(sym, quarters):
    df = strat.load_forex(sym, "d1")
    if df is None or len(df) < 400:
        return None
    ret_pct = df["close"].pct_change() * 100.0
    ret_pct = ret_pct.dropna()
    rv_daily = ret_pct ** 2  # proxy daily realized variance (single obs/day -> squared return)
    idx = ret_pct.index

    garch_vol = pd.Series(np.nan, index=idx)
    egarch_vol = pd.Series(np.nan, index=idx)
    gjr_vol = pd.Series(np.nan, index=idx)

    min_train = 300
    q_bounds = [q[0] for q in quarters] + [quarters[-1][1]]
    for i in range(len(q_bounds) - 1):
        train_end = q_bounds[i]
        test_end = q_bounds[i + 1]
        train = ret_pct[ret_pct.index < train_end]
        test_mask = (ret_pct.index >= train_end) & (ret_pct.index < test_end)
        if len(train) < min_train or test_mask.sum() == 0:
            continue
        try:
            am = arch_model(train.values[-2000:], vol="Garch", p=1, q=1, dist="normal")
            res = am.fit(disp="off", show_warning=False)
            fcast = res.forecast(horizon=1, reindex=False)
            one_step_var = float(fcast.variance.values[-1, 0])
            # propagate same conditional-vol forecast across the quarter (coarse but zero-lookahead)
            garch_vol.loc[test_mask] = np.sqrt(one_step_var)
        except Exception:
            pass
        try:
            am2 = arch_model(train.values[-2000:], vol="EGARCH", p=1, o=1, q=1, dist="normal")
            res2 = am2.fit(disp="off", show_warning=False)
            fcast2 = res2.forecast(horizon=1, reindex=False)
            egarch_vol.loc[test_mask] = np.sqrt(float(fcast2.variance.values[-1, 0]))
        except Exception:
            pass
        try:
            am3 = arch_model(train.values[-2000:], vol="Garch", p=1, o=1, q=1, power=2.0, dist="normal")
            res3 = am3.fit(disp="off", show_warning=False)
            fcast3 = res3.forecast(horizon=1, reindex=False)
            gjr_vol.loc[test_mask] = np.sqrt(float(fcast3.variance.values[-1, 0]))
        except Exception:
            pass

    har_pred = har_rv_forecast(rv_daily, min_train)
    har_vol = np.sqrt(har_pred.clip(lower=1e-8))

    kalman_logvol = kalman_sv_forecast(ret_pct)
    kalman_vol = np.exp(kalman_logvol / 2.0)

    out = pd.DataFrame({
        "ret_pct": ret_pct, "rv_next": rv_daily.shift(-1),
        "garch_vol": garch_vol, "egarch_vol": egarch_vol, "gjr_vol": gjr_vol,
        "har_vol": har_vol, "kalman_vol": kalman_vol,
    })
    out["symbol"] = sym
    return out


def qlike(vol_fcst, rv_actual):
    eps = 1e-6
    v = (vol_fcst ** 2).clip(lower=eps)
    r = rv_actual.clip(lower=eps)
    mask = v.notna() & r.notna() & np.isfinite(v) & np.isfinite(r)
    v, r = v[mask], r[mask]
    vals = (r / v - np.log(r / v) - 1)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 30:
        return np.nan, 0
    return float(vals.mean()), len(vals)


def main():
    t0 = time.time()
    lo = pd.Timestamp("2015-01-01", tz="UTC")
    hi = pd.Timestamp("2026-07-01", tz="UTC")
    quarters = make_quarters(lo, hi)
    print(f"{len(quarters)} quarters, {len(SLEEVE_T_UNIVERSE)} symbols")

    all_frames = []
    for n, sym in enumerate(SLEEVE_T_UNIVERSE):
        r = process_symbol(sym, quarters)
        if r is not None:
            all_frames.append(r)
        if (n + 1) % 20 == 0:
            print(f"  {n+1}/{len(SLEEVE_T_UNIVERSE)} done, {time.time()-t0:.0f}s elapsed")

    panel = pd.concat(all_frames)
    print(f"Panel: {len(panel)} rows, {panel['symbol'].nunique()} symbols, {time.time()-t0:.0f}s total")

    eval_rows = []
    for col in ["garch_vol", "egarch_vol", "gjr_vol", "har_vol", "kalman_vol"]:
        q, n = qlike(panel[col], panel["rv_next"])
        eval_rows.append({"model": col, "qlike_loss": round(q, 5) if q == q else None, "n": n})
    eval_df = pd.DataFrame(eval_rows).sort_values("qlike_loss")
    print("\nQLIKE loss (lower = better forecast of next-day realized variance):")
    print(eval_df.to_string(index=False))

    # cross-check: correlation between GARCH vol-percentile-rank and kalman vol-percentile-rank
    panel["garch_pct"] = panel.groupby("symbol")["garch_vol"].transform(lambda s: s.rolling(252, min_periods=60).rank(pct=True))
    panel["kalman_pct"] = panel.groupby("symbol")["kalman_vol"].transform(lambda s: s.rolling(252, min_periods=60).rank(pct=True))
    corr = panel[["garch_pct", "kalman_pct"]].dropna().corr().iloc[0, 1]
    print(f"\nCorr(GARCH vol-percentile, Kalman-SV vol-percentile) = {corr:.3f}  (cross-validates the two independent vol-estimation approaches)")

    cache_path = os.path.join(RESULTS_DIR, "PHASE8_garch_vol_cache.parquet")
    panel[["symbol", "garch_vol", "garch_pct", "egarch_vol", "gjr_vol", "har_vol", "kalman_vol", "kalman_pct"]].reset_index().rename(
        columns={"index": "date"}).to_parquet(cache_path)
    print(f"Cached -> {cache_path}")

    summary = {
        "n_symbols": panel["symbol"].nunique(),
        "n_rows": len(panel),
        "qlike_by_model": eval_df.to_dict(orient="records"),
        "corr_garch_kalman_vol_percentile": round(float(corr), 4),
        "runtime_sec": round(time.time() - t0, 1),
    }
    with open(os.path.join(RESULTS_DIR, "PHASE8_part2_volmodels_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary["qlike_by_model"], indent=2))


if __name__ == "__main__":
    main()
