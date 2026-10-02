"""
Phase 7 — precomputes the walk-forward Logistic Regression long-signal
probability (the one model/signal from phase7_screen_part3_ml.py that showed
a statistically real, if tiny, edge: 8.02bps, t=3.61, n=10876) as a per-
(date, symbol) column, cached to parquet so strategies.py's Sleeve U can
merge it into each symbol's D1 frame cheaply and repeatedly across a WFO run
without re-fitting the model every time prep_sleeve_u() is called.

This is the SAME walk-forward discipline as phase7_screen_part3_ml.py (train
on an expanding window strictly before each quarter, predict only that
quarter) -- i.e. the probability attached to any given date was produced by
a model that had never seen that date's data. Using it inside wfo.py's own
IS/OOS quarter locking is not double-dipping future information; it's one
walk-forward-safe signal being fed into a second, independent walk-forward
parameter lock (the probability THRESHOLD used for entries).
"""
from __future__ import annotations
import os
import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from phase7_screen_part3_ml import build_panel, FEATURE_COLS, make_quarters

CACHE_PATH = os.path.join(os.path.dirname(__file__), "results", "PHASE7_ml_proba_cache.parquet")


def compute_and_cache():
    panel = build_panel().sort_values("date")
    lo = panel["date"].min() + pd.DateOffset(years=3)
    hi = panel["date"].max()
    quarters = make_quarters(lo, hi)

    asset_class_dummies = pd.get_dummies(panel["asset_class"], prefix="ac")
    X_full = pd.concat([panel[FEATURE_COLS], asset_class_dummies], axis=1)
    y_full = (panel["fwd_ret_1"] > 0).astype(int)

    out_rows = []
    for q_start, q_end in quarters:
        train_mask = panel["date"] < q_start
        test_mask = (panel["date"] >= q_start) & (panel["date"] < q_end)
        if train_mask.sum() < 5000 or test_mask.sum() < 20:
            continue
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_full[train_mask])
        model = LogisticRegression(max_iter=500, C=1.0)
        model.fit(X_train_s, y_full[train_mask])

        X_test_s = scaler.transform(X_full[test_mask])
        proba = model.predict_proba(X_test_s)[:, 1]
        rows = panel.loc[test_mask, ["date", "symbol"]].copy()
        rows["ml_long_proba"] = proba
        out_rows.append(rows)

    result = pd.concat(out_rows)
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    result.to_parquet(CACHE_PATH, index=False)
    print(f"Wrote {CACHE_PATH}: {len(result)} rows, "
          f"{result['date'].min()} -> {result['date'].max()}, "
          f"{result['symbol'].nunique()} symbols")
    return result


def load_cache():
    if not os.path.exists(CACHE_PATH):
        return compute_and_cache()
    return pd.read_parquet(CACHE_PATH)


if __name__ == "__main__":
    compute_and_cache()
