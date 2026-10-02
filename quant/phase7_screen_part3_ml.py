"""
Phase 7, Part 3 — ML ensemble screen: pooled cross-sectional panel, walk-
forward retrained quarterly (same quarterly cadence as wfo.py's certified
sleeves, so results are comparable), predicting next-day return direction.

"Different models of it" = three genuinely different model families:
  - Logistic Regression  (linear baseline, L2-regularized)
  - Random Forest        (bagged, nonlinear, feature-interaction-capable)
  - Gradient Boosting    (boosted trees, usually the strongest tabular model)

This is deliberately walk-forward, not k-fold cross-validation: for each
calendar quarter from 2018Q1 onward, every model is trained ONLY on data
strictly before that quarter (expanding window) and evaluated ONLY on that
quarter's out-of-sample predictions -- exactly the same IS/OOS discipline as
wfo.py's sleeve certification, applied to an ML model instead of a parameter
grid. No lookahead: every feature is built from data available at or before
the close of the bar it's computed on, and the label is the FOLLOWING bar's
return.

Features (all computable from OHLC alone -- no external data):
  ret_1, ret_5, ret_21, ret_63   : trailing log returns
  rsi14                          : Wilder RSI
  vol21                          : 21d realized vol of daily returns
  dist_ma21, dist_ma63           : (close - MA)/MA
  xs_rank_21                     : cross-sectional percentile rank of ret_21
                                    within the same asset class on that date
  asset_class                    : one-hot (FX / Indices / Metals / Energy_Base)

Target: sign(next 1-day return). Reported per-quarter and pooled: OOS hit
rate, mean realized bps per predicted-long/predicted-short signal, t-stat,
vs the 41bps round-trip friction floor.
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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from engine import rsi
import strategies as strat
from phase7_screen_part1 import ASSET_CLASSES, ALL_SYMBOLS

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

FRICTION_BPS_RT = 41.0


def build_symbol_features(symbol, asset_class):
    try:
        df = strat.load_forex(symbol, "d1")
    except Exception:
        return None
    if len(df) < 400:
        return None
    c = df["close"]
    feat = pd.DataFrame(index=df.index)
    feat["ret_1"] = c.pct_change(1)
    feat["ret_5"] = c.pct_change(5)
    feat["ret_21"] = c.pct_change(21)
    feat["ret_63"] = c.pct_change(63)
    feat["rsi14"] = rsi(c, 14) / 100.0
    feat["vol21"] = c.pct_change().rolling(21).std()
    ma21 = c.rolling(21).mean()
    ma63 = c.rolling(63).mean()
    feat["dist_ma21"] = (c - ma21) / ma21
    feat["dist_ma63"] = (c - ma63) / ma63
    feat["symbol"] = symbol
    feat["asset_class"] = asset_class
    feat["fwd_ret_1"] = c.pct_change(1).shift(-1)  # label source (next bar's return)
    return feat.dropna(subset=["ret_63", "vol21"])


def build_panel():
    frames = []
    for cls, syms in ASSET_CLASSES.items():
        for sym in syms:
            f = build_symbol_features(sym, cls)
            if f is not None:
                frames.append(f)
    panel = pd.concat(frames)
    panel.index.name = "date"
    panel = panel.reset_index()
    # cross-sectional rank of ret_21 within (date, asset_class)
    panel["xs_rank_21"] = panel.groupby(["date", "asset_class"])["ret_21"].rank(pct=True)
    panel = panel.dropna(subset=["xs_rank_21"])
    return panel


FEATURE_COLS = ["ret_1", "ret_5", "ret_21", "ret_63", "rsi14", "vol21",
                 "dist_ma21", "dist_ma63", "xs_rank_21"]


def make_quarters(lo, hi):
    qs = []
    start = pd.Timestamp(year=lo.year, month=((lo.month - 1) // 3) * 3 + 1, day=1, tz="UTC")
    cur = start
    while True:
        q_end = cur + pd.offsets.QuarterBegin(startingMonth=1)
        if q_end > hi:
            break
        qs.append((cur, q_end))
        cur = q_end
    return qs


def run_model_wfo(panel, model_name, model_factory, min_train_rows=5000,
                   retrain_every_n_quarters=1, max_train_rows=None, seed=42):
    """retrain_every_n_quarters>1 reuses the last-fitted model for the next
    N-1 quarters (pure compute optimization -- still zero lookahead, since
    the model is only ever applied to quarters strictly after its training
    cutoff). max_train_rows randomly subsamples the (expanding) training set
    with a fixed seed so per-quarter fit cost is bounded for the heavier
    tree ensembles; this does not change which rows are *eligible* (every
    row before q_start still could be sampled), only how many are used."""
    panel = panel.sort_values("date")
    lo = panel["date"].min() + pd.DateOffset(years=3)  # need enough history before first OOS quarter
    hi = panel["date"].max()
    quarters = make_quarters(lo, hi)

    asset_class_dummies = pd.get_dummies(panel["asset_class"], prefix="ac")
    X_full = pd.concat([panel[FEATURE_COLS], asset_class_dummies], axis=1)
    y_full = (panel["fwd_ret_1"] > 0).astype(int)
    rng = np.random.RandomState(seed)

    all_oos = []
    model, scaler = None, None
    for qi, (q_start, q_end) in enumerate(quarters):
        test_mask = (panel["date"] >= q_start) & (panel["date"] < q_end)
        if test_mask.sum() < 20:
            continue
        need_retrain = (model is None) or (qi % retrain_every_n_quarters == 0)
        if need_retrain:
            train_mask = panel["date"] < q_start
            if train_mask.sum() < min_train_rows:
                continue
            X_train, y_train = X_full[train_mask], y_full[train_mask]
            if max_train_rows is not None and len(X_train) > max_train_rows:
                sel = rng.choice(len(X_train), size=max_train_rows, replace=False)
                X_train, y_train = X_train.iloc[sel], y_train.iloc[sel]
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            model = model_factory()
            model.fit(X_train_s, y_train)

        X_test = X_full[test_mask]
        X_test_s = scaler.transform(X_test)
        proba = model.predict_proba(X_test_s)[:, 1]

        test_rows = panel[test_mask].copy()
        test_rows["pred_proba"] = proba
        test_rows["pred_long"] = proba > 0.55   # require some conviction, not just >0.5
        test_rows["pred_short"] = proba < 0.45
        all_oos.append(test_rows)

    if not all_oos:
        return None
    oos = pd.concat(all_oos)
    long_rets = oos.loc[oos["pred_long"], "fwd_ret_1"]
    short_rets = -oos.loc[oos["pred_short"], "fwd_ret_1"]
    combined = pd.concat([long_rets, short_rets])

    def stats(x):
        x = x.dropna()
        if len(x) < 20:
            return dict(n=len(x), mean_bps=None, t_stat=None)
        t = x.mean() / (x.std() / np.sqrt(len(x)))
        return dict(n=int(len(x)), mean_bps=round(float(x.mean() * 10000), 2), t_stat=round(float(t), 2))

    return dict(
        model=model_name,
        n_oos_quarters=len(all_oos),
        n_oos_rows=len(oos),
        hit_rate_long=round(float((oos.loc[oos["pred_long"], "fwd_ret_1"] > 0).mean()), 4) if oos["pred_long"].sum() else None,
        hit_rate_short=round(float((oos.loc[oos["pred_short"], "fwd_ret_1"] < 0).mean()), 4) if oos["pred_short"].sum() else None,
        long_signal_stats=stats(long_rets),
        short_signal_stats=stats(short_rets),
        combined_stats=stats(combined),
        friction_bps_rt=FRICTION_BPS_RT,
    )


def main():
    print("Building pooled cross-sectional panel (all 98 FX/metals/indices/energy symbols, D1)...")
    panel = build_panel()
    print(f"Panel shape: {panel.shape}, date range {panel['date'].min()} -> {panel['date'].max()}\n")

    models = {
        # (factory, retrain_every_n_quarters, max_train_rows) -- LR is cheap
        # enough to retrain every quarter on the full expanding window; the
        # tree ensembles retrain every 2 quarters on a capped random sample
        # purely to keep this screen's wall-clock time bounded in a shared
        # sandbox. This does not relax any WFO/lookahead discipline.
        "LogisticRegression": (lambda: LogisticRegression(max_iter=500, C=1.0), 1, None),
        "RandomForest": (lambda: RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=50, n_jobs=-1, random_state=42), 2, 40000),
        "GradientBoosting": (lambda: GradientBoostingClassifier(n_estimators=60, max_depth=3, learning_rate=0.05, random_state=42), 2, 40000),
    }

    results = []
    for name, (factory, retrain_n, max_rows) in models.items():
        print(f"=== {name} (walk-forward, retrain every {retrain_n}q, max_train_rows={max_rows}) ===")
        r = run_model_wfo(panel, name, factory, retrain_every_n_quarters=retrain_n, max_train_rows=max_rows)
        if r is None:
            print("  no usable OOS quarters")
            continue
        results.append(r)
        print(f"  n_oos_quarters={r['n_oos_quarters']} n_oos_rows={r['n_oos_rows']}")
        print(f"  hit_rate_long={r['hit_rate_long']} hit_rate_short={r['hit_rate_short']}")
        print(f"  LONG signals:     {r['long_signal_stats']}")
        print(f"  SHORT signals:    {r['short_signal_stats']}")
        print(f"  COMBINED signals: {r['combined_stats']}  (need >{FRICTION_BPS_RT}bps to clear round-trip cost)")
        print()

    out_path = os.path.join(RESULTS_DIR, "PHASE7_part3_ml.json")
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
