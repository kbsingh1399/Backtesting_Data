"""
ml_sleeve.py — Sleeve O: walk-forward machine-learning classifier.

This is the "modern quant" category not yet tried anywhere in this project:
every prior sleeve (A-N) used a hand-specified rule (z-score threshold,
Donchian breakout, RSI pullback, cointegration spread). Here a gradient-
boosted classifier is trained PER QUARTER on a trailing 6-month in-sample
window only (identical walk-forward discipline to every other sleeve: never
trained on data from the quarter it then trades), predicting the sign of the
next ~4h return from a feature set spanning technicals, orderflow, funding,
sentiment, and liquidation data -- deliberately using the master dataset's
columns that no prior sleeve touched (vwap_zscore, zc_div,
liq_imbalance_ratio, whale_index, top_account_ratio, long/short_liq_zs).

Trades are generated exactly like every other sleeve (engine.generate_trades:
real ATR-box SL/TP/ratchet/time-decay, mandated friction), fed through the
same wfo.simulate_portfolio (concurrency cap, DD halt) and compute_stats --
only the entry SIGNAL differs. Model: sklearn GradientBoostingClassifier,
trained pooled across all 10 symbols (more rows per quarter = less
overfitting than training one model per symbol on 6 months of data).
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

import strategies as strat
from engine import generate_trades, Trade
import wfo

CRYPTO_UNIVERSE = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
                    "AVAXUSDT", "LINKUSDT", "DOGEUSDT", "ADAUSDT", "DOTUSDT"]

FEATURE_COLS = [
    "ret_1", "ret_4", "ret_16", "ret_96",            # momentum at multiple lags
    "rsi_14", "vwap_zscore", "zc_div",                # mean-reversion/technical
    "volume_ratio", "taker_volume_ratio",             # volume/orderflow
    "funding_rate_pct", "ls_ratio_global",            # sentiment/positioning
    "oi_change_pct", "whale_index", "top_account_ratio",
    "liq_imbalance_ratio", "long_liq_zs", "short_liq_zs",
    "basis_index_bps",
]
FWD_HORIZON = 32   # bars (~8h on 15m data) forward label horizon


def build_features(symbol: str) -> pd.DataFrame:
    df = strat.load_binance(symbol)
    out = df[["open", "high", "low", "close"]].copy()
    out["atr14"] = df["atr14_w"]
    c = df["close"]
    out["ret_1"] = c.pct_change(1)
    out["ret_4"] = c.pct_change(4)
    out["ret_16"] = c.pct_change(16)
    out["ret_96"] = c.pct_change(96)
    for col in ["rsi_14", "vwap_zscore", "zc_div", "volume_ratio", "taker_volume_ratio",
                "funding_rate_pct", "ls_ratio_global", "oi_change_pct", "whale_index",
                "top_account_ratio", "liq_imbalance_ratio", "long_liq_zs", "short_liq_zs",
                "basis_index_bps"]:
        out[col] = df[col]
    out["fwd_ret"] = c.pct_change(FWD_HORIZON).shift(-FWD_HORIZON)
    out["symbol"] = symbol
    return out


def run_ml_sleeve(friction_bps=41.0, prob_threshold=0.58, model_type="gbm",
                   is_lookback_quarters=2, dd_limit=225.0, verbose=True):
    panels = {sym: build_features(sym) for sym in CRYPTO_UNIVERSE}
    lo = max(p.index.min() for p in panels.values())
    hi = min(p.index.max() for p in panels.values())
    quarters = wfo.make_quarters(lo, hi)
    n_is = is_lookback_quarters
    oos_quarters = quarters[n_is:]
    if verbose:
        print(f"[ML sleeve] {len(oos_quarters)} candidate OOS quarters, "
              f"model={model_type}, prob_threshold={prob_threshold}, friction={friction_bps}bps")

    all_candidates = []
    quarter_log = []
    for q_start, q_end in oos_quarters:
        is_start = q_start - pd.DateOffset(months=3 * n_is)
        is_end = q_start

        train_X, train_y = [], []
        for sym, p in panels.items():
            seg = p[(p.index >= is_start) & (p.index < is_end)].dropna(subset=FEATURE_COLS + ["fwd_ret"])
            if len(seg) < 200:
                continue
            train_X.append(seg[FEATURE_COLS].values)
            train_y.append((seg["fwd_ret"] > 0).astype(int).values)
        if not train_X:
            quarter_log.append(dict(quarter=str(q_start.date()), traded=False, reason="no IS data"))
            continue
        X = np.vstack(train_X)
        y = np.concatenate(train_y)
        if len(np.unique(y)) < 2 or len(y) < 500:
            quarter_log.append(dict(quarter=str(q_start.date()), traded=False, reason="insufficient IS labels"))
            continue

        if model_type == "gbm":
            model = GradientBoostingClassifier(n_estimators=80, max_depth=3, learning_rate=0.05,
                                                subsample=0.7, random_state=42)
        else:
            model = LogisticRegression(max_iter=500)
        model.fit(X, y)
        # quick IS sanity check: in-sample accuracy / confident-subset edge
        is_pred = model.predict_proba(X)[:, 1]
        is_acc_confident = ((is_pred > prob_threshold) == (y == 1)).mean() if (is_pred > prob_threshold).any() else np.nan

        q_trades = []
        for sym, p in panels.items():
            seg = p[(p.index >= q_start) & (p.index < q_end)].dropna(subset=FEATURE_COLS)
            if len(seg) < 10:
                continue
            proba = model.predict_proba(seg[FEATURE_COLS].values)[:, 1]
            long_cond = pd.Series(proba > prob_threshold, index=seg.index)
            short_cond = pd.Series(proba < (1 - prob_threshold), index=seg.index)
            # fire once per confident excursion (avoid re-entering every bar of a sustained signal)
            long_cond = long_cond & ~long_cond.shift(1).fillna(False)
            short_cond = short_cond & ~short_cond.shift(1).fillna(False)
            entries = (long_cond | short_cond)
            sides = pd.Series(np.where(long_cond, 1, np.where(short_cond, -1, 0)), index=seg.index)
            full_df = p.loc[seg.index[0]:seg.index[-1], ["open", "high", "low", "close", "atr14"]]
            entries_full = entries.reindex(full_df.index, fill_value=False)
            sides_full = sides.reindex(full_df.index, fill_value=0)
            trades = generate_trades(full_df, entries_full, sides_full, full_df["atr14"], sym,
                                      {"friction_bps_roundtrip": friction_bps})
            q_trades.extend(trades)
            all_candidates.extend(trades)
        quarter_log.append(dict(quarter=str(q_start.date()), traded=True,
                                 is_confident_acc=round(float(is_acc_confident), 4) if not np.isnan(is_acc_confident) else None,
                                 n_is_rows=len(y), oos_trades=len(q_trades)))

    wfo.DD_LIMIT = dd_limit
    result = wfo.simulate_portfolio(all_candidates)
    windows_traded = sum(1 for q in quarter_log if q.get("traded") and q.get("oos_trades", 0) > 0)
    stats = wfo.compute_stats(result["taken_trades"], result["equity_curve"], len(oos_quarters), windows_traded)
    stats["n_oos_quarters"] = len(oos_quarters)
    stats["windows_traded"] = windows_traded
    stats["halted"] = result["halted"]
    stats["halt_time"] = str(result["halt_time"]) if result["halt_time"] else None
    stats["model_type"] = model_type
    stats["prob_threshold"] = prob_threshold
    stats["friction_bps"] = friction_bps
    return dict(stats=stats, trades=result["taken_trades"], equity_curve=result["equity_curve"],
                quarter_log=quarter_log, universe=CRYPTO_UNIVERSE,
                oos_span=(oos_quarters[0][0] if oos_quarters else None,
                          oos_quarters[-1][1] if oos_quarters else None))



if __name__ == "__main__":
    import json
    for fbps in [41.0, 8.0, 4.0]:
        out = run_ml_sleeve(friction_bps=fbps, prob_threshold=0.58, model_type="gbm", dd_limit=1e12, verbose=True)
        s = out["stats"]
        print(f"friction={fbps}bps -> n_trades={s['n_trades']} roi%={s['roi_pct']:.2f} sharpe={s['sharpe']:.2f} "
              f"win%={s['win_rate']:.2f} windows={s['windows_traded']}/{s['n_oos_quarters']} "
              f"gross={s.get('gross_pnl')}")
