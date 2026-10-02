"""
quantlab.model
==============
Meta-labelling layer: a secondary model that decides *which* primary-rule
signals to take (Lopez de Prado, AFML ch.3).

The primary rule fixes the side; the model only ever says take / skip / size.
That is a strictly easier learning problem than predicting direction, and it
is the only way an unconditionally edgeless rule could still be profitable -
if, and only if, the edge is conditional on observable state.

Validation discipline
---------------------
* **Anchored walk-forward.**  Train on everything before the fold, predict
  the fold, never look back.
* **Purging.**  Training labels whose holding period overlaps the test fold
  are removed.  Triple-barrier labels span up to `time_stop_bars`, so a naive
  split leaks the test period's price path into the training labels.
* **Embargo.**  An additional buffer after the fold start is dropped, because
  serial correlation makes the bars immediately before a fold informative
  about it.
* **Sample weights.**  Overlapping labels are down-weighted by their average
  uniqueness, and old samples decay exponentially.
* **Threshold learned in-fold.**  The probability gate is a quantile of the
  *training* predictions, never of the test predictions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import ModelParams


def _fit_lgbm(X, y, w, seed):
    import lightgbm as lgb

    ds = lgb.Dataset(X, label=y, weight=w, free_raw_data=False)
    params = {
        "objective": "binary", "metric": "binary_logloss",
        "learning_rate": 0.03, "num_leaves": 15, "max_depth": 4,
        "min_data_in_leaf": 80, "feature_fraction": 0.7,
        "bagging_fraction": 0.8, "bagging_freq": 1,
        "lambda_l1": 0.5, "lambda_l2": 5.0,
        "verbose": -1, "seed": seed, "num_threads": 2,
    }
    return lgb.train(params, ds, num_boost_round=180)


def _fit_xgb(X, y, w, seed):
    import xgboost as xgb

    d = xgb.DMatrix(X, label=y, weight=w)
    params = {
        "objective": "binary:logistic", "eval_metric": "logloss",
        "max_depth": 3, "eta": 0.03, "subsample": 0.8,
        "colsample_bytree": 0.7, "min_child_weight": 30,
        "reg_alpha": 0.5, "reg_lambda": 5.0,
        "seed": seed, "verbosity": 0, "nthread": 2,
    }
    return xgb.train(params, d, num_boost_round=180)


def _fit_logit(X, y, w, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(C=0.05, max_iter=2000, random_state=seed)),
    ])
    pipe.fit(X, y, lr__sample_weight=w)
    return pipe


class MetaEnsemble:
    """Equal-weight average of three diverse learners.

    Deliberately *not* a stacked meta-learner trained on in-sample base
    predictions: that is what the reference S4 script does, and fitting the
    blender on the base models' own training predictions is a textbook route
    to an overconfident ensemble.  A simple average has no such failure mode.
    """

    def __init__(self, seed: int = 7):
        self.seed = seed
        self.models: Dict[str, object] = {}
        self.features: List[str] = []

    def fit(self, X: pd.DataFrame, y: np.ndarray, w: np.ndarray) -> "MetaEnsemble":
        self.features = list(X.columns)
        Xv = X.to_numpy(dtype=np.float32)
        w = w / (w.mean() + 1e-12)
        self.models = {
            "lgbm": _fit_lgbm(Xv, y, w, self.seed),
            "xgb": _fit_xgb(Xv, y, w, self.seed),
            "logit": _fit_logit(Xv, y, w, self.seed),
        }
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        import xgboost as xgb

        Xv = X[self.features].to_numpy(dtype=np.float32)
        p = [
            self.models["lgbm"].predict(Xv),
            self.models["xgb"].predict(xgb.DMatrix(Xv)),
            self.models["logit"].predict_proba(Xv)[:, 1],
        ]
        return np.mean(np.vstack(p), axis=0)

    def feature_importance(self) -> pd.Series:
        g = self.models["lgbm"].feature_importance(importance_type="gain")
        return pd.Series(g, index=self.features).sort_values(ascending=False)


# --------------------------------------------------------------------------
@dataclass
class FoldResult:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int
    n_test: int
    gate: float
    train_base_rate: float


def walk_forward_predict(
    trades: pd.DataFrame,
    features: List[str],
    mp: ModelParams,
    oos_start: str,
    oos_end: str,
    bar_minutes: int = 15,
) -> Tuple[pd.DataFrame, List[FoldResult], pd.DataFrame]:
    """Anchored, purged, embargoed walk-forward over the OOS period.

    Returns (trades_with_prob, fold_metadata, mean_feature_importance).
    """
    df = trades.sort_values("entry_time").reset_index(drop=True).copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)

    start = pd.Timestamp(oos_start, tz="UTC")
    end = pd.Timestamp(oos_end, tz="UTC")
    embargo = pd.Timedelta(minutes=bar_minutes * mp.embargo_bars)
    step = pd.Timedelta(days=mp.refit_every_days)

    df["prob"] = np.nan
    df["fold"] = -1
    folds: List[FoldResult] = []
    importances = []

    edges = []
    t = start
    while t < end:
        edges.append((t, min(t + step, end)))
        t += step

    for k, (a, b) in enumerate(edges):
        test_mask = (df["entry_time"] >= a) & (df["entry_time"] < b)
        if not test_mask.any():
            continue
        # PURGE: drop training labels whose holding period reaches into the
        # test fold, plus an embargo buffer before the fold opens.
        train_mask = (df["exit_time"] < a - embargo) & (df["entry_time"] < a - embargo)
        n_train = int(train_mask.sum())
        if n_train < mp.min_train_signals:
            continue

        tr = df.loc[train_mask]
        te = df.loc[test_mask]
        X_tr = tr[features].astype(float).fillna(0.0)
        X_te = te[features].astype(float).fillna(0.0)
        y_tr = tr["label"].to_numpy(int)
        if y_tr.mean() in (0.0, 1.0):
            continue

        age_days = (a - tr["entry_time"]).dt.total_seconds().to_numpy() / 86400.0
        w = tr["w_uniq"].to_numpy() * np.exp(-np.log(2.0) * age_days / 365.0)

        m = MetaEnsemble(seed=mp.seed).fit(X_tr, y_tr, w)
        p_tr = m.predict_proba(X_tr)
        gate = float(np.quantile(p_tr, mp.gate_quantile))

        df.loc[test_mask, "prob"] = m.predict_proba(X_te)
        df.loc[test_mask, "fold"] = k
        df.loc[test_mask, "gate"] = gate

        importances.append(m.feature_importance())
        folds.append(FoldResult(
            fold=k, train_start=tr["entry_time"].min(), train_end=tr["entry_time"].max(),
            test_start=a, test_end=b, n_train=n_train, n_test=int(test_mask.sum()),
            gate=round(gate, 4), train_base_rate=round(float(y_tr.mean()), 4),
        ))

    imp = (pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
           if importances else pd.Series(dtype=float))
    return df, folds, imp.rename("gain").to_frame()
