"""
Phase 8, Part 6 -- Meta-labeling + triple-barrier labeling + sample-weighting-
by-uniqueness (Section 7 taxonomy), applied to Sleeve M4's own primary signal
(Donchian breakout + ATR-rank filter on XAUUSD/XAGUSD).

Pipeline (de Prado's "Advances in Financial Machine Learning" recipe):
  1. PRIMARY MODEL generates raw entry signals (don_n=40, atr_rank_min=30 --
     the most common IS-locked choice across M4's certified quarters).
  2. TRIPLE-BARRIER LABELING: every raw signal is run through the project's
     actual SL/TP/ratchet/time-decay exit engine (engine.generate_trades) in
     ISOLATION (ignoring the 3-concurrent-position portfolio cap -- that's a
     portfolio-level constraint applied later by the real WFO, not a labeling
     concern) to get its true win/loss outcome (y=1 if r_multiple>0).
  3. SAMPLE WEIGHTING BY UNIQUENESS: trades with overlapping holding periods
     get down-weighted (de Prado's concurrency-based uniqueness weight) so
     the meta-model's training loss isn't dominated by a cluster of
     simultaneously-open, highly-correlated trades.
  4. META-MODEL: walk-forward (quarterly expanding, embargoed -- only trades
     whose EXIT occurred strictly before the current quarter start are used
     for training, closing the purged-k-fold-style leakage gap) RandomForest
     predicts P(win) from PRE-TRADE features only (zero lookahead).
  5. Cache P(win) per (symbol, signal_time) for use as a meta-filter in a new
     sleeve (M4_META) run through the full, real WFO engine.
"""
from __future__ import annotations
import json, os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import strategies as strat
from strategies import prep_sleeve_m, sig_sleeve_m, rolling_hurst
from engine import generate_trades, wilder_atr
from phase7_screen_part3_ml import make_quarters

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
UNIVERSE = ["XAUUSD", "XAGUSD"]
LABEL_PARAMS = dict(don_n=40, atr_rank_min=30)  # most common IS-locked choice in M4_geom_search_v2


def build_features_and_labels(symbol):
    df = prep_sleeve_m(symbol)
    entries, sides, atr = sig_sleeve_m(df, LABEL_PARAMS)

    trades = generate_trades(df, entries, sides, atr, symbol, sim_kwargs=None)
    if not trades:
        return None

    # pre-trade feature snapshot at signal_time (NOT entry_time -- entry is
    # next-bar-open, so signal-bar's close-based features are the last
    # genuinely available information, matching the project's no-lookahead rule)
    vol21 = df["close"].pct_change().rolling(21).std()
    hurst100 = rolling_hurst(df["close"], window=100)
    don_hi = df["high"].rolling(LABEL_PARAMS["don_n"]).max().shift(1)
    don_lo = df["low"].rolling(LABEL_PARAMS["don_n"]).min().shift(1)
    breakout_extension = (df["close"] - don_hi).abs() / df["atr14"]  # how far past the band, in ATR units

    rows = []
    for tr in trades:
        t = tr.signal_time
        if t not in df.index:
            continue
        feat = {
            "symbol": symbol, "signal_time": t, "exit_time": tr.exit_time, "side": tr.side,
            "atr_rank": df.loc[t, "atr_rank"] if "atr_rank" in df.columns else np.nan,
            "vol21": vol21.loc[t] if t in vol21.index else np.nan,
            "hurst100": hurst100.loc[t] if t in hurst100.index else np.nan,
            "breakout_extension_atr": breakout_extension.loc[t] if t in breakout_extension.index else np.nan,
            "weekday": t.weekday(),
            "bars_held": tr.bars_held, "r_multiple": tr.r_multiple,
            "label_win": int(tr.r_multiple > 0),
        }
        rows.append(feat)
    out = pd.DataFrame(rows).dropna(subset=["atr_rank", "vol21", "hurst100", "breakout_extension_atr"])
    return out


def uniqueness_weights(df):
    """de Prado average-uniqueness weight: for each trade, 1/(avg number of
    concurrently-open trades during its lifespan). Computed PER SYMBOL since
    overlapping trades are already forbidden within a symbol by
    generate_trades (last_exit_i gate) -- so this really captures cross-
    symbol (XAU vs XAG) concurrency overlap, which is the real correlation-
    clustering risk for a 2-asset sleeve."""
    df = df.sort_values("signal_time").reset_index(drop=True)
    n = len(df)
    starts = df["signal_time"].values
    ends = df["exit_time"].values
    concurrency = np.zeros(n)
    for i in range(n):
        overlap = ((starts < ends[i]) & (ends > starts[i])).sum()
        concurrency[i] = overlap
    weights = 1.0 / concurrency
    weights = weights / weights.mean()
    return weights


def walk_forward_meta_model(panel):
    panel = panel.sort_values("signal_time").reset_index(drop=True)
    lo, hi = panel["signal_time"].min(), panel["signal_time"].max()
    quarters = make_quarters(lo, hi)
    feat_cols = ["atr_rank", "vol21", "hurst100", "breakout_extension_atr", "weekday"]

    panel["meta_proba"] = np.nan
    min_train = 25
    for i in range(len(quarters)):
        q_start, q_end = quarters[i]
        # EMBARGO: only use trades whose exit fully resolved before q_start
        train_mask = panel["exit_time"] < q_start
        test_mask = (panel["signal_time"] >= q_start) & (panel["signal_time"] < q_end)
        train = panel[train_mask]
        if len(train) < min_train or test_mask.sum() == 0:
            continue
        w = uniqueness_weights(train)
        clf = RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_leaf=5,
                                      random_state=42, n_jobs=-1)
        clf.fit(train[feat_cols].values, train["label_win"].values, sample_weight=w)
        proba = clf.predict_proba(panel.loc[test_mask, feat_cols].values)[:, 1]
        panel.loc[test_mask, "meta_proba"] = proba
    return panel


def main():
    frames = []
    for sym in UNIVERSE:
        f = build_features_and_labels(sym)
        if f is not None:
            frames.append(f)
            print(f"{sym}: {len(f)} raw signals, win rate {f['label_win'].mean():.3f}, "
                  f"mean r_multiple {f['r_multiple'].mean():.4f}")
    panel = pd.concat(frames, ignore_index=True)

    panel = walk_forward_meta_model(panel)
    valid = panel.dropna(subset=["meta_proba"])
    print(f"\nMeta-model coverage: {len(valid)}/{len(panel)} signals got an OOS meta-probability")

    # does the meta-probability actually discriminate win/loss OOS?
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(valid["label_win"], valid["meta_proba"])
    print(f"OOS AUC of meta-model (win vs loss): {auc:.4f}  (0.5 = no skill)")

    for thresh in [0.0, 0.45, 0.5, 0.55, 0.6]:
        sub = valid[valid["meta_proba"] >= thresh]
        if len(sub) == 0:
            continue
        print(f"  thresh>={thresh}: n={len(sub)}, win_rate={sub['label_win'].mean():.3f}, "
              f"mean_r={sub['r_multiple'].mean():.4f}")

    cache_cols = ["symbol", "signal_time", "meta_proba"]
    out_path = os.path.join(RESULTS_DIR, "PHASE8_metalabel_cache.parquet")
    panel[cache_cols].to_parquet(out_path)
    print(f"\nCached -> {out_path}")

    summary = {
        "label_params": LABEL_PARAMS, "n_raw_signals": len(panel), "n_with_meta_proba": len(valid),
        "oos_auc": round(float(auc), 4),
        "threshold_sweep": [
            {"thresh": th, "n": int((valid["meta_proba"] >= th).sum()),
             "win_rate": round(float(valid.loc[valid["meta_proba"] >= th, "label_win"].mean()), 4) if (valid["meta_proba"] >= th).sum() else None,
             "mean_r": round(float(valid.loc[valid["meta_proba"] >= th, "r_multiple"].mean()), 4) if (valid["meta_proba"] >= th).sum() else None}
            for th in [0.0, 0.45, 0.5, 0.55, 0.6]
        ],
    }
    with open(os.path.join(RESULTS_DIR, "PHASE8_part6_metalabel_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)


if __name__ == "__main__":
    main()
