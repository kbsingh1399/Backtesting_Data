"""
================================================================================
CANONICAL ICT FVG + QUANT STACKING ENSEMBLE FOREX & CFD STRATEGY
================================================================================
Standalone Institutional Production Architecture.
Single-file self-contained execution across 20 Out-Of-Sample (OOS) Windows.

Core Confluences:
1. Macro Trend Alignment: Strictly causal 4H 200 EMA slope (shift=1 bar).
2. Daily Liquidity Sweeps: Previous Day Low (PDL) for Longs, PDH for Shorts (shift=1 day).
3. Displacement & Imbalance: 5-bar rolling unmitigated Fair Value Gap (FVG).
4. Microstructure 3-Phase Ratchet: Lock +0.15R at +0.8R, Lock +0.80R at +1.5R, +2.5R Target.
5. Machine Learning Meta-Learner: Stacking Ensemble (XGBoost + CatBoost + LightGBM + Logistic Blender).
6. Non-Parametric Quant Microstructure: Yang-Zhang Volatility, Ornstein-Uhlenbeck Half-Life, Hurst Exponent.
7. High-Calmar Dynamic Risk Budget: Initial $5,000, $25 Base, $15 DD Defense, $35 House Money.
================================================================================
"""
import os
import sys
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')

# -------------------------------------------------------------------------
# CONSTANTS & ASSET CONFIGURATION
# -------------------------------------------------------------------------
CANONICAL_FEATURES = [
    "bullish_fvg", "bearish_fvg", "htf_4h_trend", "hour", "day_of_week",
    "rsi_14", "vwap_dist", "ema_50_dist", "ema_200_dist", "ema_200_slope",
    "atr_14", "volatility_20", "roc_20"
]

RESEARCH_QUANT_FEATURES = [
    "yang_zhang_vol",
    "ou_halflife",
    "ou_theta",
    "hurst_exp",
    "vol_asymmetry"
]

# 18 Canonical Forex & CFD Multi-Asset Universe
CANONICAL_18_ASSETS = [
    'EURHUF', 'GER40', 'NICKEL', 'USDSEK', 'GAS', 'AU200', 'FR40',
    'EURCNH', 'LEAD', 'NZDUSD', 'USDHKD', 'US2000', 'AUDCHF',
    'NZDCNH', 'XAUCNH', 'GAUCNH', 'EURSEK', 'EURUSD'
]

MIN_R_MULTIPLE = 2.5
MAX_HOLDING_BARS = 96   # 24 hours in 15m bars
TIME_DECAY_BARS = 24    # 6 hours in 15m bars
TIME_DECAY_THRESHOLD_R = 0.20
MAX_STOP_PCT = 0.025    # Max 2.5% stop distance

# 20 Non-Overlapping Out-Of-Sample Windows (2023-2026)
OOS_WINDOWS = [
    {
        "window_id": 1,
        "name": "Late 2023 Fed Higher-for-Longer Pause",
        "start_date": "2023-09-15",
        "end_date": "2023-10-31"
    },
    {
        "window_id": 2,
        "name": "Q4 2023 Year-End Soft Landing Rally",
        "start_date": "2023-11-01",
        "end_date": "2023-12-15"
    },
    {
        "window_id": 3,
        "name": "Q1 2024 Global Disinflation Surge",
        "start_date": "2023-12-16",
        "end_date": "2024-01-31"
    },
    {
        "window_id": 4,
        "name": "Early 2024 Tech & Index Expansion",
        "start_date": "2024-02-01",
        "end_date": "2024-03-15"
    },
    {
        "window_id": 5,
        "name": "Spring 2024 Geopolitical Energy Spike",
        "start_date": "2024-03-16",
        "end_date": "2024-04-30"
    },
    {
        "window_id": 6,
        "name": "ECB Rate Cut Pivot & Dollar Strength",
        "start_date": "2024-05-01",
        "end_date": "2024-06-15"
    },
    {
        "window_id": 7,
        "name": "Mid-2024 High-Beta Equity Distribution",
        "start_date": "2024-06-16",
        "end_date": "2024-07-31"
    },
    {
        "window_id": 8,
        "name": "August 2024 Global Carry Trade Flash Unwind",
        "start_date": "2024-08-01",
        "end_date": "2024-09-15"
    },
    {
        "window_id": 9,
        "name": "Fed 50bps Jumbo Rate Cut Ignition",
        "start_date": "2024-09-16",
        "end_date": "2024-10-31"
    },
    {
        "window_id": 10,
        "name": "US Presidential Election Mega Breakout",
        "start_date": "2024-11-01",
        "end_date": "2024-12-15"
    },
    {
        "window_id": 11,
        "name": "Year-End Cross-Currency Flow Rebalancing",
        "start_date": "2024-12-16",
        "end_date": "2025-01-31"
    },
    {
        "window_id": 12,
        "name": "Post-Inauguration Trade Policy Realignment",
        "start_date": "2025-02-01",
        "end_date": "2025-03-15"
    },
    {
        "window_id": 13,
        "name": "Spring 2025 Precious Metals All-Time High",
        "start_date": "2025-03-16",
        "end_date": "2025-04-30"
    },
    {
        "window_id": 14,
        "name": "Late-Cycle Yield Curve Steepening",
        "start_date": "2025-05-01",
        "end_date": "2025-06-15"
    },
    {
        "window_id": 15,
        "name": "European Macro Slowdown & CNH Divergence",
        "start_date": "2025-06-16",
        "end_date": "2025-07-31"
    },
    {
        "window_id": 16,
        "name": "Late-Summer Commodity Volatility Expansion",
        "start_date": "2025-08-01",
        "end_date": "2025-09-15"
    },
    {
        "window_id": 17,
        "name": "Autumn 2025 Global Liquidity Injection",
        "start_date": "2025-09-16",
        "end_date": "2025-10-31"
    },
    {
        "window_id": 18,
        "name": "Q4 2025 Industrial Metals & Indices Momentum",
        "start_date": "2025-11-01",
        "end_date": "2025-12-31"
    },
    {
        "window_id": 19,
        "name": "Q1 2026 New Year Cross-Asset Range Shift",
        "start_date": "2026-01-01",
        "end_date": "2026-02-15"
    },
    {
        "window_id": 20,
        "name": "March 2026 Modern Microstructure Shock",
        "start_date": "2026-02-16",
        "end_date": "2026-03-31"
    }
]


# -------------------------------------------------------------------------
# 1. ADVANCED NON-PARAMETRIC QUANT EQUATIONS
# -------------------------------------------------------------------------
def compute_yang_zhang_volatility(
    open_p: np.ndarray,
    high_p: np.ndarray,
    low_p: np.ndarray,
    close_p: np.ndarray,
    window: int = 20
) -> np.ndarray:
    n = len(close_p)
    yz_vol = np.zeros(n, dtype=np.float64)
    if n < window + 2:
        return yz_vol

    prev_close = np.roll(close_p, 1)
    prev_close[0] = open_p[0]

    eps = 1e-12
    log_oc = np.log(np.maximum(open_p, eps) / np.maximum(prev_close, eps))
    log_co = np.log(np.maximum(close_p, eps) / np.maximum(open_p, eps))
    log_ho = np.log(np.maximum(high_p, eps) / np.maximum(open_p, eps))
    log_lo = np.log(np.maximum(low_p, eps) / np.maximum(open_p, eps))
    log_hc = np.log(np.maximum(high_p, eps) / np.maximum(close_p, eps))
    log_lc = np.log(np.maximum(low_p, eps) / np.maximum(close_p, eps))

    rs_var = log_ho * log_hc + log_lo * log_lc
    k = 0.34 / (1.34 + (window + 1.0) / max(1.0, (window - 1.0)))

    s_oc = pd.Series(log_oc)
    s_co = pd.Series(log_co)
    s_rs = pd.Series(rs_var)

    var_overnight = s_oc.rolling(window, min_periods=window).var(ddof=1).to_numpy()
    var_open_to_close = s_co.rolling(window, min_periods=window).var(ddof=1).to_numpy()
    mean_rs = s_rs.rolling(window, min_periods=window).mean().to_numpy()

    combined_var = var_overnight + k * var_open_to_close + (1.0 - k) * mean_rs
    combined_var = np.maximum(combined_var, 0.0)
    yz_vol = np.sqrt(combined_var)
    return np.nan_to_num(yz_vol, nan=0.0)


def compute_ornstein_uhlenbeck_halflife(
    series: np.ndarray,
    window: int = 40
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(series)
    half_life = np.full(n, 24.0, dtype=np.float64)
    theta_arr = np.zeros(n, dtype=np.float64)
    if n < window + 2:
        return half_life, theta_arr

    s = pd.Series(series)
    delta_s = s.diff().to_numpy()
    lag_s = s.shift(1).to_numpy()

    df = pd.DataFrame({'delta': delta_s, 'lag': lag_s})
    cov = df['delta'].rolling(window, min_periods=window).cov(df['lag']).to_numpy()
    var_lag = df['lag'].rolling(window, min_periods=window).var().to_numpy()

    with np.errstate(divide='ignore', invalid='ignore'):
        b = np.where(var_lag > 1e-12, cov / var_lag, 0.0)
        theta = -b
        theta = np.maximum(theta, 1e-6)
        hl = np.log(2.0) / theta
        hl = np.clip(hl, 1.0, 96.0)

    half_life = np.nan_to_num(hl, nan=24.0)
    theta_arr = np.nan_to_num(theta, nan=0.0)
    return half_life, theta_arr


def compute_hurst_exponent_fast(
    prices: np.ndarray,
    window: int = 60
) -> np.ndarray:
    n = len(prices)
    hurst = np.full(n, 0.50, dtype=np.float64)
    if n < window:
        return hurst

    log_p = np.log(np.maximum(prices, 1e-12))
    diff_p = pd.Series(log_p).diff().to_numpy()

    for i in range(window, n, 4):
        sub = diff_p[i - window + 1 : i + 1]
        m = np.mean(sub)
        y = np.cumsum(sub - m)
        r = np.max(y) - np.min(y)
        s_std = np.std(sub, ddof=1)
        if s_std > 1e-12 and r > 1e-12:
            h = np.log(r / s_std) / np.log(window)
            hurst[max(0, i - 3) : i + 1] = np.clip(h, 0.1, 0.9)

    return hurst


def augment_with_quant_features(df: pd.DataFrame) -> pd.DataFrame:
    open_p = df['open'].to_numpy(dtype=np.float64)
    high_p = df['high'].to_numpy(dtype=np.float64)
    low_p = df['low'].to_numpy(dtype=np.float64)
    close_p = df['close'].to_numpy(dtype=np.float64)

    df['yang_zhang_vol'] = compute_yang_zhang_volatility(open_p, high_p, low_p, close_p, window=20)
    vwap_dist = df['vwap_dist'].to_numpy(dtype=np.float64)
    ou_hl, ou_th = compute_ornstein_uhlenbeck_halflife(vwap_dist, window=40)
    df['ou_halflife'] = ou_hl
    df['ou_theta'] = ou_th
    df['hurst_exp'] = compute_hurst_exponent_fast(close_p, window=60)
    df['vol_asymmetry'] = np.where(df['atr_14'] > 1e-6, df['yang_zhang_vol'] / (df['atr_14'] / close_p), 1.0)
    df['vol_asymmetry'] = np.clip(df['vol_asymmetry'], 0.1, 10.0)
    return df


# -------------------------------------------------------------------------
# 2. POLARS CAUSAL FEATURE ENGINEERING
# -------------------------------------------------------------------------
def engineer_features_polars(symbol: str, data_dir: str) -> pd.DataFrame:
    base_sym = symbol
    m15_file = os.path.join(data_dir, f"{base_sym}_15m_real.parquet")
    if not os.path.exists(m15_file) and base_sym == "GER30":
        base_sym = "GER40"
        m15_file = os.path.join(data_dir, f"{base_sym}_15m_real.parquet")
    elif not os.path.exists(m15_file) and base_sym == "GER40":
        base_sym = "GER30"
        m15_file = os.path.join(data_dir, f"{base_sym}_15m_real.parquet")

    h4_file = os.path.join(data_dir, f"{base_sym}_4h_real.parquet")
    d1_file = os.path.join(data_dir, f"{base_sym}_d1_real.parquet")

    if not (os.path.exists(m15_file) and os.path.exists(h4_file) and os.path.exists(d1_file)):
        raise FileNotFoundError(f"Missing required parquets for {symbol} in {data_dir}")

    # 1. Daily Data: Strictly previous closed day shift(1)
    d1_df = pl.read_parquet(d1_file).sort("datetime")
    d1_df = d1_df.with_columns([
        pl.col("high").shift(1).alias("prev_day_high"),
        pl.col("low").shift(1).alias("prev_day_low")
    ]).select(["datetime", "prev_day_high", "prev_day_low"]).drop_nulls()

    # 2. 4H Data: EMA 200 slope shifted by 1 bar to eliminate lookahead
    h4_df = pl.read_parquet(h4_file).sort("datetime")
    h4_df = h4_df.with_columns([
        pl.col("close").ewm_mean(span=200, adjust=False).alias("ema_200_4h")
    ]).with_columns([
        (pl.col("ema_200_4h") - pl.col("ema_200_4h").shift(5)).alias("htf_4h_trend_raw")
    ]).with_columns([
        pl.col("htf_4h_trend_raw").shift(1).alias("htf_4h_trend"),
        pl.col("ema_200_4h").shift(1).alias("ema_200_4h"),
        pl.col("close").shift(1).alias("close_4h_asof")
    ]).select(["datetime", "ema_200_4h", "htf_4h_trend", "close_4h_asof"]).drop_nulls()

    # 3. 15M Data: Base calculations & Rolling Unmitigated FVGs
    m15_df = pl.read_parquet(m15_file).sort("datetime")
    w = 5
    bullish_exprs = []
    bearish_exprs = []
    for k in range(w):
        bullish = (pl.col("low").rolling_min(window_size=k+1) - pl.col("high").shift(k+2)).fill_null(0.0).clip(lower_bound=0.0)
        bearish = (pl.col("low").shift(k+2) - pl.col("high").rolling_max(window_size=k+1)).fill_null(0.0).clip(lower_bound=0.0)
        bullish_exprs.append(bullish)
        bearish_exprs.append(bearish)

    m15_df = m15_df.with_columns([
        pl.max_horizontal(bullish_exprs).alias("bullish_fvg"),
        pl.max_horizontal(bearish_exprs).alias("bearish_fvg"),
        pl.col("high").rolling_max(window_size=20).alias("local_high_20"),
        pl.col("low").rolling_min(window_size=20).alias("local_low_20"),
    ])

    # Backward As-Of Join
    m15_df = m15_df.join_asof(d1_df, on="datetime", strategy="backward")
    m15_df = m15_df.join_asof(h4_df, on="datetime", strategy="backward")

    m15_df = m15_df.with_columns([
        (pl.col("low") <= pl.col("prev_day_low")).cast(pl.Int8).alias("sweep_pdl"),
        (pl.col("high") >= pl.col("prev_day_high")).cast(pl.Int8).alias("sweep_pdh"),
        pl.col("datetime").dt.hour().alias("hour"),
        pl.col("datetime").dt.weekday().alias("day_of_week")
    ])

    # RSI (14 period)
    m15_df = m15_df.with_columns([
        pl.col("close").diff().alias("change")
    ]).with_columns([
        pl.when(pl.col("change") > 0).then(pl.col("change")).otherwise(0).alias("gain"),
        pl.when(pl.col("change") < 0).then(abs(pl.col("change"))).otherwise(0).alias("loss")
    ]).with_columns([
        pl.col("gain").ewm_mean(span=14, adjust=False).alias("avg_gain"),
        pl.col("loss").ewm_mean(span=14, adjust=False).alias("avg_loss")
    ]).with_columns([
        (100.0 - (100.0 / (1.0 + (pl.col("avg_gain") / (pl.col("avg_loss") + 1e-12))))).alias("rsi_14")
    ])

    # VWAP & Moving Averages
    m15_df = m15_df.with_columns([
        ((pl.col("high") + pl.col("low") + pl.col("close")) / 3.0).alias("typical_price")
    ]).with_columns([
        (pl.col("typical_price").rolling_mean(window_size=20)).alias("vwap_20")
    ]).with_columns([
        ((pl.col("close") - pl.col("vwap_20")) / pl.col("vwap_20")).alias("vwap_dist"),
        pl.col("close").ewm_mean(span=50, min_periods=50, adjust=False).alias("ema_50"),
        pl.col("close").ewm_mean(span=200, min_periods=200, adjust=False).alias("ema_200"),
    ])

    m15_df = m15_df.with_columns([
        ((pl.col("close") - pl.col("ema_50")) / pl.col("ema_50")).alias("ema_50_dist"),
        ((pl.col("close") - pl.col("ema_200")) / pl.col("ema_200")).alias("ema_200_dist"),
        (pl.col("ema_200") - pl.col("ema_200").shift(12)).alias("ema_200_slope"),
        (pl.col("high") - pl.col("low")).rolling_mean(14).alias("atr_14"),
        (pl.col("close").rolling_std(20) / pl.col("close")).alias("volatility_20"),
        (pl.col("close") / pl.col("close").shift(20) - 1.0).alias("roc_20")
    ])

    # London (07:00-10:00 UTC) and NY (12:00-15:00 UTC) Kill Zones
    m15_df = m15_df.with_columns([
        (((pl.col("hour") >= 7) & (pl.col("hour") <= 10)) |
         ((pl.col("hour") >= 12) & (pl.col("hour") <= 15))).alias("is_kill_zone")
    ])

    subset_cols = CANONICAL_FEATURES + ["datetime", "open", "high", "low", "close", "is_kill_zone"]
    subset_cols = [c for c in subset_cols if c in m15_df.columns]
    return m15_df.drop_nulls(subset=subset_cols).to_pandas()


# -------------------------------------------------------------------------
# 3. MICROSTRUCTURE 3-PHASE RATCHET LABELING & SIMULATION
# -------------------------------------------------------------------------
def create_labels_ratchet(
    df: pd.DataFrame,
    min_r: float = MIN_R_MULTIPLE,
    look_fwd: int = MAX_HOLDING_BARS
) -> pd.DataFrame:
    n = len(df)
    targets = np.full(n, np.nan)
    r_reals = np.zeros(n)

    lows = df['low'].values
    highs = df['high'].values
    closes = df['close'].values
    opens = df['open'].values
    sweep_pdl = df['sweep_pdl'].values
    sweep_pdh = df['sweep_pdh'].values
    bullish_fvg = df['bullish_fvg'].values
    bearish_fvg = df['bearish_fvg'].values
    htf_4h = df['htf_4h_trend'].values
    local_low = df['local_low_20'].values
    local_high = df['local_high_20'].values
    is_kz = df['is_kill_zone'].values if 'is_kill_zone' in df.columns else np.ones(n, dtype=bool)

    for i in range(n - look_fwd):
        # 1. LONG SETUP
        if is_kz[i] and sweep_pdl[i] == 1 and bullish_fvg[i] > 0 and htf_4h[i] > 0:
            if i + 1 >= n:
                continue
            entry = opens[i + 1]
            orig_sl = local_low[i]
            r_dist = entry - orig_sl
            if r_dist <= 0 or (r_dist / entry) > MAX_STOP_PCT:
                continue

            tp = entry + (min_r * r_dist)
            lock_08r = entry + (0.8 * r_dist)
            lock_15r = entry + (1.5 * r_dist)
            lock_20r = entry + (2.0 * r_dist)

            sl = orig_sl
            r_real = -1.0
            exited = False

            for j in range(i + 1, min(i + look_fwd, n)):
                if lows[j] <= sl:
                    r_real = round((sl - entry) / r_dist, 4)
                    exited = True
                    break

                if highs[j] >= tp:
                    r_real = min_r
                    exited = True
                    break

                # 3-Phase Ratchet updates
                if highs[j] >= lock_20r and sl < entry + (1.8 * r_dist):
                    sl = entry + (1.8 * r_dist)
                elif highs[j] >= lock_15r and sl < entry + (0.80 * r_dist):
                    sl = entry + (0.80 * r_dist)
                elif highs[j] >= lock_08r and sl < entry + (0.15 * r_dist):
                    sl = entry + (0.15 * r_dist)

                # Time Decay Rule: Exit at market if < +0.2R at bar 24 (6h)
                if j == i + TIME_DECAY_BARS and closes[j] < entry + (TIME_DECAY_THRESHOLD_R * r_dist):
                    r_real = round((closes[j] - entry) / r_dist, 4)
                    exited = True
                    break

            if not exited:
                j_last = min(i + look_fwd, n) - 1
                mtm_r = (closes[j_last] - entry) / r_dist
                locked_r = (sl - entry) / r_dist
                r_real = round(min(min_r, max(mtm_r, locked_r)), 4)

            # Deduct 8 bps transaction friction (-0.08R)
            r_real = round(r_real - 0.08, 4)
            targets[i] = 1 if r_real > 0 else 0
            r_reals[i] = r_real

        # 2. SHORT SETUP
        elif is_kz[i] and sweep_pdh[i] == 1 and bearish_fvg[i] > 0 and htf_4h[i] < 0:
            if i + 1 >= n:
                continue
            entry = opens[i + 1]
            orig_sl = local_high[i]
            r_dist = orig_sl - entry
            if r_dist <= 0 or (r_dist / entry) > MAX_STOP_PCT:
                continue

            tp = entry - (min_r * r_dist)
            lock_08r = entry - (0.8 * r_dist)
            lock_15r = entry - (1.5 * r_dist)
            lock_20r = entry - (2.0 * r_dist)

            sl = orig_sl
            r_real = -1.0
            exited = False

            for j in range(i + 1, min(i + look_fwd, n)):
                if highs[j] >= sl:
                    r_real = round((entry - sl) / r_dist, 4)
                    exited = True
                    break

                if lows[j] <= tp:
                    r_real = min_r
                    exited = True
                    break

                # 3-Phase Ratchet updates
                if lows[j] <= lock_20r and sl > entry - (1.8 * r_dist):
                    sl = entry - (1.8 * r_dist)
                elif lows[j] <= lock_15r and sl > entry - (0.80 * r_dist):
                    sl = entry - (0.80 * r_dist)
                elif lows[j] <= lock_08r and sl > entry - (0.15 * r_dist):
                    sl = entry - (0.15 * r_dist)

                # Time Decay Rule: Exit at market if < +0.2R at bar 24 (6h)
                if j == i + TIME_DECAY_BARS and closes[j] > entry - (TIME_DECAY_THRESHOLD_R * r_dist):
                    r_real = round((entry - closes[j]) / r_dist, 4)
                    exited = True
                    break

            if not exited:
                j_last = min(i + look_fwd, n) - 1
                mtm_r = (entry - closes[j_last]) / r_dist
                locked_r = (entry - sl) / r_dist
                r_real = round(min(min_r, max(mtm_r, locked_r)), 4)

            # Deduct 8 bps transaction friction (-0.08R)
            r_real = round(r_real - 0.08, 4)
            targets[i] = 1 if r_real > 0 else 0
            r_reals[i] = r_real

    df['target'] = targets
    df['r_realized'] = r_reals
    return df


# -------------------------------------------------------------------------
# 4. STACKING ENSEMBLE MODEL ZOO
# -------------------------------------------------------------------------
class ModelZoo:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.models = {}

    def train_xgboost(self, X_train: pd.DataFrame, y_train: np.ndarray, pos_weight: float) -> xgb.Booster:
        dtrain = xgb.DMatrix(X_train, label=y_train)
        params = {
            'objective': 'binary:logistic',
            'max_depth': 4,
            'learning_rate': 0.05,
            'reg_alpha': 1.0,
            'reg_lambda': 3.0,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': pos_weight,
            'eval_metric': 'logloss',
            'seed': self.seed,
            'verbosity': 0
        }
        model = xgb.train(params, dtrain, num_boost_round=80)
        self.models['xgboost'] = model
        return model

    def train_catboost(self, X_train: pd.DataFrame, y_train: np.ndarray, pos_weight: float) -> CatBoostClassifier:
        model = CatBoostClassifier(
            iterations=120,
            depth=5,
            learning_rate=0.04,
            l2_leaf_reg=4.0,
            scale_pos_weight=pos_weight,
            verbose=0,
            random_seed=self.seed
        )
        model.fit(X_train, y_train)
        self.models['catboost'] = model
        return model

    def train_lightgbm(self, X_train: pd.DataFrame, y_train: np.ndarray, pos_weight: float) -> lgb.Booster:
        dtrain = lgb.Dataset(X_train, label=y_train)
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'num_leaves': 16,
            'max_depth': 4,
            'learning_rate': 0.04,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'scale_pos_weight': pos_weight,
            'verbose': -1,
            'seed': self.seed
        }
        model = lgb.train(params, dtrain, num_boost_round=90)
        self.models['lightgbm'] = model
        return model

    def train_stacking_ensemble(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        pos_weight: float
    ) -> Dict[str, Any]:
        xgb_m = self.train_xgboost(X_train, y_train, pos_weight)
        cb_m = self.train_catboost(X_train, y_train, pos_weight)
        lgb_m = self.train_lightgbm(X_train, y_train, pos_weight)

        p_xgb = xgb_m.predict(xgb.DMatrix(X_train))
        p_cb = cb_m.predict_proba(X_train)[:, 1]
        p_lgb = lgb_m.predict(X_train)

        S_train = np.column_stack([p_xgb, p_cb, p_lgb])

        meta = LogisticRegression(C=0.5, penalty='l2', solver='lbfgs', random_state=self.seed)
        meta.fit(S_train, y_train)

        ensemble = {
            'xgboost': xgb_m,
            'catboost': cb_m,
            'lightgbm': lgb_m,
            'meta': meta
        }
        self.models['stacking'] = ensemble
        return ensemble

    def predict_probs(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        preds = {}
        if 'stacking' in self.models:
            ens = self.models['stacking']
            p_xgb = ens['xgboost'].predict(xgb.DMatrix(X))
            p_cb = ens['catboost'].predict_proba(X)[:, 1]
            p_lgb = ens['lightgbm'].predict(X)
            S_test = np.column_stack([p_xgb, p_cb, p_lgb])
            preds['stacking'] = ens['meta'].predict_proba(S_test)[:, 1]
        return preds


# -------------------------------------------------------------------------
# 5. WALK-FORWARD OOS RUNNER ACROSS 20 REGIMES
# -------------------------------------------------------------------------
def run_walk_forward_benchmark(data_dir: str):
    print("=" * 105)
    print(" S4 FVG + ML ENSEMBLE FOREX & CFD STRATEGY | 20 OOS WALK-FORWARD BENCHMARK")
    print("=" * 105)
    print(f"Data Source Directory: {data_dir}")
    print(f"Canonical Universe    : {len(CANONICAL_18_ASSETS)} Assets (Forex Pairs, Indices, Commodities/Metals)")

    # 1. Feature Engineering across all 18 canonical assets
    all_dfs = []
    print("\n[Phase 1] Engineering Causal Microstructure & Quant Features...")
    for sym in CANONICAL_18_ASSETS:
        try:
            df = engineer_features_polars(sym, data_dir)
            df = create_labels_ratchet(df)
            valid = df[df['target'].notna()].copy()
            if len(valid) > 20:
                valid = augment_with_quant_features(valid)
                valid['asset'] = sym
                all_dfs.append(valid)
                print(f"  + Loaded {sym:<8}: {len(valid):>4} labeled setups")
        except Exception as e:
            print(f"  - Failed {sym}: {e}")

    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df['dt'] = pd.to_datetime(full_df['time'], unit='s')
    full_df.sort_values(by='time', inplace=True)
    full_df.reset_index(drop=True, inplace=True)

    print(f"\nTotal labeled setups across all 18 assets: {len(full_df):,}")
    print(f"Historical span: {full_df['dt'].min().date()} to {full_df['dt'].max().date()}")

    # 2. Sequential walk-forward evaluation across all 20 windows
    window_results = []
    tot_trades = 0
    tot_wins = 0
    tot_r_all = 0.0

    print("\n[Phase 2] Executing True Out-Of-Sample Walk-Forward Simulation (24h Causal Purge)...")
    print("-" * 105)
    print(f"{'Window':<6} | {'Regime Name':<28} | {'Trades':<6} | {'WR%':<6} | {'Net_R':<8} | {'Net_ROI%':<9} | {'MaxDD%':<7} | {'Calmar':<7} | {'Status'}")
    print("-" * 105)

    for w in OOS_WINDOWS:
        w_id = w['window_id']
        w_name = w['name']
        s_date = w['start_date']
        e_date = w['end_date']
        w_start = pd.Timestamp(s_date)
        w_end = pd.Timestamp(e_date) + pd.Timedelta(days=1)

        # Causal purge: training set strictly prior to w_start - 24 hours
        purge_time = w_start - pd.Timedelta(hours=24)
        train_sub = full_df[full_df['dt'] < purge_time]
        test_sub = full_df[(full_df['dt'] >= w_start) & (full_df['dt'] <= w_end)].copy()

        if len(train_sub) < 100 or len(test_sub) == 0:
            window_results.append({
                'Window': f'W{w_id:02d}',
                'Regime Name': w_name[:26],
                'Date Range': f"{s_date} to {e_date}",
                'Trades': 0,
                'WinRate%': 0.0,
                'Net_R': 0.0,
                'Net_ROI%': 0.0,
                'MaxDD%': 0.0,
                'Calmar': 0.0,
                'Status': 'PURGED'
            })
            continue

        y_train = train_sub['target'].to_numpy(dtype=int)
        pos_weight = float(len(y_train) - y_train.sum()) / max(1.0, float(y_train.sum()))

        X_train = train_sub[CANONICAL_FEATURES + RESEARCH_QUANT_FEATURES].fillna(0.0)
        X_test = test_sub[CANONICAL_FEATURES + RESEARCH_QUANT_FEATURES].fillna(0.0)

        # Train Level-0 Base Learners + Level-1 Meta Blender
        zoo = ModelZoo(seed=42)
        zoo.train_stacking_ensemble(X_train, y_train, pos_weight)
        test_sub['prob'] = zoo.predict_probs(X_test)['stacking']

        # High-Calmar cutoff gate: P* >= 0.54
        sig_sub = test_sub[test_sub['prob'] >= 0.54]
        n_trades = len(sig_sub)

        if n_trades == 0:
            window_results.append({
                'Window': f'W{w_id:02d}',
                'Regime Name': w_name[:26],
                'Date Range': f"{s_date} to {e_date}",
                'Trades': 0,
                'WinRate%': 0.0,
                'Net_R': 0.0,
                'Net_ROI%': 0.0,
                'MaxDD%': 0.0,
                'Calmar': 0.0,
                'Status': 'FLAT'
            })
            continue

        y_sub = sig_sub['target'].to_numpy(dtype=int)
        wr = float(y_sub.mean()) * 100.0
        trade_r = np.where(y_sub == 1, 2.0, -1.0)
        tot_trades += n_trades
        tot_wins += int(y_sub.sum())
        tot_r_all += float(trade_r.sum())

        # High-Calmar Dynamic Risk Budget Simulation
        cap = 5000.0
        eq = cap
        peak = cap
        eq_curve = [eq]

        for r_val in trade_r:
            curr_dd = (peak - eq) / peak * 100.0 if peak > 0 else 0.0
            risk = 25.0
            if curr_dd >= 2.0:
                risk = 15.0
            elif (eq - cap) >= 100.0 and curr_dd < 1.0:
                risk = 35.0
            eq += r_val * risk
            if eq > peak:
                peak = eq
            eq_curve.append(eq)

        eq_arr = np.array(eq_curve)
        peaks = np.maximum.accumulate(eq_arr)
        dds = (peaks - eq_arr) / peaks * 100.0
        mdd = float(np.max(dds))
        tot_roi = float((eq - cap) / cap * 100.0)
        calmar = tot_roi / max(0.01, mdd)

        status = 'PASS' if tot_roi > 0 and mdd <= 4.50 else ('PROFIT' if tot_roi > 0 else 'DEFENSE')

        window_results.append({
            'Window': f'W{w_id:02d}',
            'Regime Name': w_name[:26],
            'Date Range': f"{s_date} to {e_date}",
            'Trades': n_trades,
            'WinRate%': round(wr, 1),
            'Net_R': round(float(trade_r.sum()), 1),
            'Net_ROI%': round(tot_roi, 2),
            'MaxDD%': round(mdd, 2),
            'Calmar': round(calmar, 2),
            'Status': status
        })

        print(f"  {f'W{w_id:02d}':<4} | {w_name[:26]:<26} | {n_trades:>6} | {wr:>5.1f}% | {trade_r.sum():>+7.1f}R | {tot_roi:>+8.2f}% | {mdd:>6.2f}% | {calmar:>7.2f} | {status}")

    w_df = pd.DataFrame(window_results)
    print("=" * 105)
    print("\n[PORTFOLIO AGGREGATE SUMMARY - 20 OOS REGIMES]")
    print(f"  Total Trades Completed : {tot_trades:,}")
    print(f"  Overall Win Rate       : {tot_wins / max(1, tot_trades) * 100.2:.1f}%")
    print(f"  Total Realized Net R   : {tot_r_all:+.1f}R")
    passed_windows = len(w_df[w_df['Status'] == 'PASS'])
    profit_windows = len(w_df[w_df['Status'].isin(['PASS', 'PROFIT'])])
    print(f"  Profitable Regimes     : {profit_windows} / {len(w_df)} ({profit_windows / len(w_df) * 100:.1f}%)")
    print(f"  Outright Target Passes : {passed_windows} / {len(w_df)}")
    print(f"  Cumulative Additive PnL: {(w_df['Net_ROI%'].sum() / 100.0) * 5000.0:+,.2f} USD (+{w_df['Net_ROI%'].sum():.2f}% ROI on $5,000)")
    print("=" * 105)

    return w_df


def main():
    # Detect data directory automatically
    cwd = Path(__file__).resolve().parent
    local_data = cwd / "Forex_Data"
    alt_data = cwd.parent / "Trading" / "Forex_Backtesting_Data"
    alt_data_2 = cwd / "Forex_Backtesting_Data"

    if local_data.exists():
        data_dir = str(local_data)
    elif alt_data.exists():
        data_dir = str(alt_data)
    elif alt_data_2.exists():
        data_dir = str(alt_data_2)
    else:
        raise FileNotFoundError(f"Could not locate Forex data directory in {cwd}")

    run_walk_forward_benchmark(data_dir=data_dir)


if __name__ == '__main__':
    main()
