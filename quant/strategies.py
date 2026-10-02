"""
strategies.py — data loaders + sleeve signal definitions.

Sleeves:
  A_CRYPTO_PULLBACK : d1 trend filter + 4H RSI pullback + CVD/taker confirm
  B_CRYPTO_SWEEP    : 15m prev-day VAH/VAL liquidity sweep + reclaim, CVD confirm
  C_SWING_BREAKOUT  : Donchian breakout (d1) + ATR-rank filter, FX majors + XAU
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

from engine import wilder_atr, rsi
from statarb_engine import generate_statarb_trades

DATA_ROOT = os.environ.get("BT_DATA_ROOT", os.path.join(os.path.dirname(__file__), "..", "..", "Backtesting_Data"))
if not os.path.isdir(DATA_ROOT):
    DATA_ROOT = os.path.join(os.path.dirname(__file__), "..")

BINANCE_DIR = os.path.join(DATA_ROOT, "Binance_Data")
FOREX_DIR = os.path.join(DATA_ROOT, "Forex_Data")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_binance(symbol: str) -> pd.DataFrame:
    path = os.path.join(BINANCE_DIR, f"{symbol}_15m_master_2020_2026.parquet")
    df = pd.read_parquet(path)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    df = df.sort_values("datetime_utc").drop_duplicates("datetime_utc").reset_index(drop=True)
    df = df.set_index("datetime_utc")
    # columns already include rsi_14, atr_14, future_cvd_15m, taker_volume_ratio,
    # prev_day_vah, prev_day_val, ema_200 etc. Re-derive ATR via Wilder for
    # consistency with engine defaults (dataset's atr_14 may use a different
    # smoothing convention).
    df["atr14_w"] = wilder_atr(df, 14)
    return df


def resample_binance_to_d1(df15: pd.DataFrame) -> pd.DataFrame:
    agg = {
        "open": "first", "high": "max", "low": "min", "close": "last",
        "volume_base": "sum",
    }
    d1 = df15.resample("1D").agg(agg).dropna(subset=["open"])
    d1["ema_200"] = d1["close"].ewm(span=200, adjust=False, min_periods=200).mean()
    d1["ema_50"] = d1["close"].ewm(span=50, adjust=False, min_periods=50).mean()
    d1["trend_up"] = d1["close"] > d1["ema_200"]
    d1["trend_dn"] = d1["close"] < d1["ema_200"]
    return d1


def resample_binance_to_4h(df15: pd.DataFrame) -> pd.DataFrame:
    agg = {
        "open": "first", "high": "max", "low": "min", "close": "last",
        "volume_base": "sum", "future_cvd_15m": "sum", "taker_volume_ratio": "mean",
    }
    h4 = df15.resample("4h").agg(agg).dropna(subset=["open"])
    h4["rsi14"] = rsi(h4["close"], 14)
    h4["atr14"] = wilder_atr(h4, 14)
    return h4


def load_forex(symbol: str, tf: str) -> pd.DataFrame:
    path = os.path.join(FOREX_DIR, f"{symbol}_{tf}_real.parquet")
    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
    df = df.set_index("datetime")
    df["atr14"] = wilder_atr(df, 14)
    df["rsi14"] = rsi(df["close"], 14)
    return df


# ---------------------------------------------------------------------------
# Sleeve A: crypto pullback (d1 trend + 4H RSI pullback + CVD/taker confirm)
# ---------------------------------------------------------------------------
SLEEVE_A_UNIVERSE = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

SLEEVE_A_PARAM_GRID = [
    dict(rsi_low=35, rsi_high=65, cvd_confirm=True),
    dict(rsi_low=40, rsi_high=60, cvd_confirm=True),
    dict(rsi_low=35, rsi_high=65, cvd_confirm=False),
    dict(rsi_low=30, rsi_high=70, cvd_confirm=False),
]


def prep_sleeve_a(symbol: str):
    """Returns (df15, d1, h4) with trend/pullback columns attached to h4."""
    df15 = load_binance(symbol)
    d1 = resample_binance_to_d1(df15)
    h4 = resample_binance_to_4h(df15)
    # map daily trend onto 4H bars (as-of previous completed daily bar -> no lookahead)
    d1_shift = d1[["trend_up", "trend_dn"]].shift(1)
    h4 = h4.join(d1_shift.reindex(h4.index, method="ffill"))
    h4["cvd_slope"] = h4["future_cvd_15m"].diff()
    return df15, h4


def sig_sleeve_a(h4: pd.DataFrame, params: dict):
    """Entries aligned to h4 index. Long on RSI pullback below rsi_low while
    daily uptrend; short on RSI pullback above rsi_high while daily downtrend.
    Optional CVD-slope confirmation (buyers/sellers stepping back in)."""
    rsi_low, rsi_high = params["rsi_low"], params["rsi_high"]
    long_cond = (h4["trend_up"].fillna(False)) & (h4["rsi14"] < rsi_low) & (h4["rsi14"].shift(1) >= rsi_low)
    short_cond = (h4["trend_dn"].fillna(False)) & (h4["rsi14"] > rsi_high) & (h4["rsi14"].shift(1) <= rsi_high)
    if params.get("cvd_confirm"):
        long_cond &= h4["cvd_slope"] > 0
        short_cond &= h4["cvd_slope"] < 0
    entries = (long_cond | short_cond).fillna(False)
    sides = pd.Series(np.where(long_cond, 1, np.where(short_cond, -1, 0)), index=h4.index)
    atr = h4["atr14"]
    return entries, sides, atr


# ---------------------------------------------------------------------------
# Sleeve B: 15m prev-day VAH/VAL sweep + reclaim (crypto)
# ---------------------------------------------------------------------------
SLEEVE_B_UNIVERSE = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "AVAXUSDT", "LINKUSDT", "DOGEUSDT"]

SLEEVE_B_PARAM_GRID = [
    dict(cvd_confirm=True, taker_confirm=False),
    dict(cvd_confirm=True, taker_confirm=True),
    dict(cvd_confirm=False, taker_confirm=False),
    dict(cvd_confirm=False, taker_confirm=True),
]


def prep_sleeve_b(symbol: str):
    df15 = load_binance(symbol)
    df15["cvd_slope"] = df15["future_cvd_15m"].diff()
    return df15


def sig_sleeve_b(df15: pd.DataFrame, params: dict):
    """Sweep prev-day VAL then reclaim back above it -> long (mean reversion).
    Sweep prev-day VAH then reclaim back below it -> short.
    'Sweep' = prior bar's low (high) pierced the level; 'reclaim' = current
    bar's close back on the right side of the level."""
    val = df15["prev_day_val"]
    vah = df15["prev_day_vah"]
    low, high, close = df15["low"], df15["high"], df15["close"]

    swept_val = (low.shift(1) < val.shift(1)) & (low < val)
    reclaim_long = swept_val & (close > val)

    swept_vah = (high.shift(1) > vah.shift(1)) & (high > vah)
    reclaim_short = swept_vah & (close < vah)

    long_cond = reclaim_long.fillna(False)
    short_cond = reclaim_short.fillna(False)

    if params.get("cvd_confirm"):
        long_cond &= df15["cvd_slope"] > 0
        short_cond &= df15["cvd_slope"] < 0
    if params.get("taker_confirm"):
        long_cond &= df15["taker_volume_ratio"] > 1.0
        short_cond &= df15["taker_volume_ratio"] < 1.0

    entries = (long_cond | short_cond).fillna(False)
    sides = pd.Series(np.where(long_cond, 1, np.where(short_cond, -1, 0)), index=df15.index)
    atr = df15["atr14_w"]
    return entries, sides, atr


# ---------------------------------------------------------------------------
# Sleeve C: FX/commodity swing breakout (Donchian + ATR-rank), daily
# ---------------------------------------------------------------------------
SLEEVE_C_UNIVERSE = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "XAUUSD"]

SLEEVE_C_PARAM_GRID = [
    dict(don_n=55, atr_rank_min=50),
    dict(don_n=40, atr_rank_min=50),
    dict(don_n=80, atr_rank_min=50),
    dict(don_n=55, atr_rank_min=70),
    dict(don_n=40, atr_rank_min=70),
]


def prep_sleeve_c(symbol: str):
    d1 = load_forex(symbol, "d1")
    d1["atr_rank"] = d1["atr14"].rolling(252, min_periods=60).rank(pct=True) * 100
    return d1


def sig_sleeve_c(d1: pd.DataFrame, params: dict):
    n = params["don_n"]
    hh = d1["high"].rolling(n).max().shift(1)
    ll = d1["low"].rolling(n).min().shift(1)
    long_cond = (d1["close"] > hh) & (d1["atr_rank"] >= params["atr_rank_min"])
    short_cond = (d1["close"] < ll) & (d1["atr_rank"] >= params["atr_rank_min"])
    entries = (long_cond | short_cond).fillna(False)
    sides = pd.Series(np.where(long_cond, 1, np.where(short_cond, -1, 0)), index=d1.index)
    atr = d1["atr14"]
    return entries, sides, atr


# ---------------------------------------------------------------------------
# Sleeve D: low-frequency crypto mean reversion (daily, z-score vs rolling
# mean, regime/asymmetry-aware). Motivated by SSRN/academic evidence that
# (a) short-horizon (sub-daily) crypto predictability is consumed by costs,
# (b) BTC mean reversion is asymmetric -- stronger/faster after down-moves
# (Corbet & Katsiampa 2020), and (c) naive MR bleeds hard in trend regimes
# it fights (bull/bear regime dependence, Poterba & Summers 1988 logic) --
# hence the trend-aligned variant only fades dips inside an uptrend / rips
# inside a downtrend, and the asymmetric variant only ever fades downside
# shocks (long-only), never shorts strength.
# ---------------------------------------------------------------------------
SLEEVE_D_UNIVERSE = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "AVAXUSDT", "LINKUSDT", "DOGEUSDT"]

SLEEVE_D_PARAM_GRID = [
    dict(z_lookback=20, z_entry=2.0, mode="trend_aligned_both"),
    dict(z_lookback=20, z_entry=1.5, mode="trend_aligned_both"),
    dict(z_lookback=14, z_entry=2.0, mode="trend_aligned_both"),
    dict(z_lookback=30, z_entry=2.0, mode="trend_aligned_both"),
    dict(z_lookback=20, z_entry=2.0, mode="long_only_downside"),
    dict(z_lookback=20, z_entry=1.5, mode="long_only_downside"),
    dict(z_lookback=14, z_entry=1.5, mode="long_only_downside"),
]


def prep_sleeve_d(symbol: str):
    df15 = load_binance(symbol)
    d1 = resample_binance_to_d1(df15)
    n_max = 30
    sma = d1["close"].rolling(n_max).mean()
    std = d1["close"].rolling(n_max).std()
    for n in {g["z_lookback"] for g in SLEEVE_D_PARAM_GRID}:
        m = d1["close"].rolling(n).mean()
        s = d1["close"].rolling(n).std()
        d1[f"z_{n}"] = (d1["close"] - m) / s.replace(0, np.nan)
    d1["atr14"] = wilder_atr(d1, 14)
    return d1


def sig_sleeve_d(d1: pd.DataFrame, params: dict):
    z = d1[f"z_{params['z_lookback']}"]
    mode = params["mode"]
    if mode == "trend_aligned_both":
        long_cond = (z <= -params["z_entry"]) & d1["trend_up"]
        short_cond = (z >= params["z_entry"]) & d1["trend_dn"]
    elif mode == "long_only_downside":
        long_cond = (z <= -params["z_entry"])
        short_cond = pd.Series(False, index=d1.index)
    else:
        raise ValueError(mode)
    # only fire on the FIRST bar the threshold is breached (avoid re-firing
    # every bar while still extended)
    long_cond = long_cond & ~long_cond.shift(1).fillna(False)
    short_cond = short_cond & ~short_cond.shift(1).fillna(False)
    entries = (long_cond | short_cond).fillna(False)
    sides = pd.Series(np.where(long_cond, 1, np.where(short_cond, -1, 0)), index=d1.index)
    atr = d1["atr14"]
    return entries, sides, atr


# ---------------------------------------------------------------------------
# Sleeve E: daily order-flow momentum (CVD flow z-score continuation).
# Motivated by SSRN evidence that aggressor order flow (CVD) predicts
# next-day crypto returns OOS -- i.e. a CONTINUATION signal, not reversion:
# trade in the direction of an unusually large daily net taker/CVD flow,
# optionally confirmed by the daily candle closing the same direction.
# ---------------------------------------------------------------------------
SLEEVE_E_UNIVERSE = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "AVAXUSDT", "LINKUSDT", "DOGEUSDT"]

SLEEVE_E_PARAM_GRID = [
    dict(cvd_lookback=20, cvd_z_entry=1.0, price_confirm=True),
    dict(cvd_lookback=20, cvd_z_entry=1.5, price_confirm=True),
    dict(cvd_lookback=10, cvd_z_entry=1.5, price_confirm=True),
    dict(cvd_lookback=30, cvd_z_entry=1.5, price_confirm=True),
    dict(cvd_lookback=20, cvd_z_entry=1.0, price_confirm=False),
    dict(cvd_lookback=20, cvd_z_entry=1.5, price_confirm=False),
]


def prep_sleeve_e(symbol: str):
    df15 = load_binance(symbol)
    agg = {
        "open": "first", "high": "max", "low": "min", "close": "last",
        "future_cvd_15m": "sum", "taker_volume_ratio": "mean", "funding_rate_pct": "mean",
    }
    d1 = df15.resample("1D").agg(agg).dropna(subset=["open"])
    d1["atr14"] = wilder_atr(d1, 14)
    d1["day_ret"] = d1["close"] / d1["open"] - 1.0
    for n in {g["cvd_lookback"] for g in SLEEVE_E_PARAM_GRID}:
        m = d1["future_cvd_15m"].rolling(n).mean()
        s = d1["future_cvd_15m"].rolling(n).std()
        d1[f"cvd_z_{n}"] = (d1["future_cvd_15m"] - m) / s.replace(0, np.nan)
    return d1


def sig_sleeve_e(d1: pd.DataFrame, params: dict):
    z = d1[f"cvd_z_{params['cvd_lookback']}"]
    long_cond = z >= params["cvd_z_entry"]
    short_cond = z <= -params["cvd_z_entry"]
    if params.get("price_confirm"):
        long_cond &= d1["day_ret"] > 0
        short_cond &= d1["day_ret"] < 0
    # only fire on the first bar the threshold is breached
    long_cond = long_cond.fillna(False) & ~long_cond.shift(1).fillna(False)
    short_cond = short_cond.fillna(False) & ~short_cond.shift(1).fillna(False)
    entries = (long_cond | short_cond).fillna(False)
    sides = pd.Series(np.where(long_cond, 1, np.where(short_cond, -1, 0)), index=d1.index)
    atr = d1["atr14"]
    return entries, sides, atr


# ---------------------------------------------------------------------------
# Sleeve F: statistical arbitrage FX pairs (Engle-Granger cointegration +
# OU mean-reversion of the stationary spread). Pair universe was selected by
# `statarb_scan.py` using ONLY data up to 2019-12-31 (Engle-Granger p<0.05,
# OU half-life 3-40 trading days) -- see results/statarb_coint_scan_*.csv.
# OOS testing for this sleeve is restricted to start 2020-01-01 or later so
# the pair-selection step never touches data used for performance evaluation.
#
# Mechanics:
#   - Hedge ratio beta_t estimated via a trailing 252-day ROLLING regression
#     of log(A) on log(B), LAGGED by one day (uses only data through t-1) --
#     no lookahead.
#   - Stationary spread_t = logA_t - (alpha_t + beta_t * logB_t): this is
#     what mean-reverts and what the z-score entry signal is computed from.
#   - A separate synthetic "price" series (the tradable NAV index) is built
#     from the *return* of being long $1 of A and short $beta of B each day
#     (index_t = 100*exp(cumsum(retA_t - beta_t*retB_t))) -- this is what
#     engine.py uses for ATR/SL/TP/ratchet bookkeeping, dollar P&L, and
#     position sizing, so going "long the index" means exactly "long A,
#     short beta*B", which is the correct trade when the spread is cheap.
#   - Round-trip friction is DOUBLED vs a single-instrument sleeve (82bps,
#     not 41bps) because a pairs trade executes two separate legs.
#   - EOD approximation: no true intrabar O/H/L exists for a synthetic
#     spread index, so open=high=low=close=index_t (SL/TP/ratchet are
#     checked once per day, at the close) -- a standard, explicitly
#     disclosed simplification for daily statistical arbitrage.
# ---------------------------------------------------------------------------
SLEEVE_F_PAIRS = [
    "CADCHF~USDCAD", "USDCHF~USDMXN", "NZDUSD~ZARJPY", "CHFJPY~GBPSEK",
    "GBPCHF~GBPUSD", "EURCAD~EURCHF", "CADCHF~EURUSD", "USDCHF~USDZAR",
]
SLEEVE_F_HEDGE_WINDOW = 252
SLEEVE_F_MIN_OOS_START = pd.Timestamp("2020-01-01", tz="UTC")

SLEEVE_F_PARAM_GRID = [
    dict(z_lookback=20, z_entry=2.0),
    dict(z_lookback=20, z_entry=1.5),
    dict(z_lookback=40, z_entry=2.0),
    dict(z_lookback=40, z_entry=1.5),
    dict(z_lookback=60, z_entry=2.0),
]


def prep_sleeve_f(pair_id: str):
    a_sym, b_sym = pair_id.split("~")
    da = load_forex(a_sym, "d1")["close"]
    db = load_forex(b_sym, "d1")["close"]
    idx = da.index.intersection(db.index)
    la, lb = np.log(da.loc[idx]), np.log(db.loc[idx])

    w = SLEEVE_F_HEDGE_WINDOW
    cov = la.rolling(w).cov(lb).shift(1)
    var = lb.rolling(w).var().shift(1)
    beta = (cov / var.replace(0, np.nan))
    mean_a = la.rolling(w).mean().shift(1)
    mean_b = lb.rolling(w).mean().shift(1)
    alpha = mean_a - beta * mean_b

    spread = la - (alpha + beta * lb)  # stationary spread -> drives the signal

    ret_a = la.diff()
    ret_b = lb.diff()
    spread_ret = ret_a - beta * ret_b
    index = 100.0 * np.exp(spread_ret.fillna(0).cumsum())

    df = pd.DataFrame(index=idx)
    df["open"] = index
    df["high"] = index
    df["low"] = index
    df["close"] = index
    df["atr14"] = wilder_atr(df, 14)
    for n in {g["z_lookback"] for g in SLEEVE_F_PARAM_GRID}:
        m = spread.rolling(n).mean()
        s = spread.rolling(n).std()
        df[f"z_{n}"] = (spread - m) / s.replace(0, np.nan)
    return df


def sig_sleeve_f(df: pd.DataFrame, params: dict):
    z = df[f"z_{params['z_lookback']}"]
    long_cond = z <= -params["z_entry"]     # spread cheap -> long A / short beta*B
    short_cond = z >= params["z_entry"]     # spread rich  -> short A / long beta*B
    long_cond = long_cond.fillna(False) & ~long_cond.shift(1).fillna(False)
    short_cond = short_cond.fillna(False) & ~short_cond.shift(1).fillna(False)
    entries = (long_cond | short_cond).fillna(False)
    sides = pd.Series(np.where(long_cond, 1, np.where(short_cond, -1, 0)), index=df.index)
    atr = df["atr14"]
    return entries, sides, atr


# ---------------------------------------------------------------------------
# Sleeve G: same cointegrated-pairs universe/signal as F, but with risk
# management NATIVE to the OU/mean-reversion process instead of an ATR
# breakout box: entry at |z|>=entry_z, profit-take on reversion to exit_z,
# hard stop on FURTHER divergence to stop_z (cointegration-breakdown
# protection), time-stop at max_hold_days. This is the textbook Gatev/
# Goetzmann/Rouwenhorst-style pairs-trading exit, and is the "think like a
# quant" fix for sleeve F's failure mode (ATR stop too tight relative to
# 2-leg friction on a low-daily-vol spread).
# ---------------------------------------------------------------------------
SLEEVE_G_PAIRS = SLEEVE_F_PAIRS
SLEEVE_G_HEDGE_WINDOW = SLEEVE_F_HEDGE_WINDOW
SLEEVE_G_MIN_OOS_START = SLEEVE_F_MIN_OOS_START

SLEEVE_G_PARAM_GRID = [
    dict(entry_z=2.0, exit_z=0.0, stop_extra_z=1.5, max_hold_days=60),
    dict(entry_z=1.5, exit_z=0.0, stop_extra_z=1.5, max_hold_days=60),
    dict(entry_z=2.5, exit_z=0.0, stop_extra_z=1.5, max_hold_days=60),
    dict(entry_z=2.0, exit_z=0.5, stop_extra_z=1.5, max_hold_days=45),
    dict(entry_z=2.0, exit_z=0.0, stop_extra_z=2.0, max_hold_days=90),
    dict(entry_z=1.75, exit_z=0.25, stop_extra_z=1.75, max_hold_days=60),
]


def prep_sleeve_g(pair_id: str):
    a_sym, b_sym = pair_id.split("~")
    da = load_forex(a_sym, "d1")["close"]
    db = load_forex(b_sym, "d1")["close"]
    idx = da.index.intersection(db.index)
    la, lb = np.log(da.loc[idx]), np.log(db.loc[idx])

    w = SLEEVE_G_HEDGE_WINDOW
    cov = la.rolling(w).cov(lb).shift(1)
    var = lb.rolling(w).var().shift(1)
    beta = cov / var.replace(0, np.nan)
    mean_a = la.rolling(w).mean().shift(1)
    mean_b = lb.rolling(w).mean().shift(1)
    alpha = mean_a - beta * mean_b

    spread = la - (alpha + beta * lb)
    ret_a, ret_b = la.diff(), lb.diff()
    spread_ret = ret_a - beta * ret_b
    index = 100.0 * np.exp(spread_ret.fillna(0).cumsum())

    df = pd.DataFrame(index=idx)
    df["open"], df["high"], df["low"], df["close"] = index, index, index, index
    z_lookback = 20  # fixed estimation window for the entry statistic itself
    df["spread_mean"] = spread.rolling(z_lookback).mean()
    df["spread_std"] = spread.rolling(z_lookback).std()
    df["z"] = (spread - df["spread_mean"]) / df["spread_std"].replace(0, np.nan)
    return df


def sig_sleeve_g(df: pd.DataFrame, params: dict):
    z = df["z"]
    long_cond = z <= -params["entry_z"]
    short_cond = z >= params["entry_z"]
    long_cond = long_cond.fillna(False) & ~long_cond.shift(1).fillna(False)
    short_cond = short_cond.fillna(False) & ~short_cond.shift(1).fillna(False)
    entries = (long_cond | short_cond).fillna(False)
    sides = pd.Series(np.where(long_cond, 1, np.where(short_cond, -1, 0)), index=df.index)
    std_series = df["spread_std"]
    return entries, sides, std_series


# ---------------------------------------------------------------------------
# Sleeve H: same OU-native mechanics as G, but a DIVERSIFIED 28-pair universe
# (vs G's concentrated 8) selected from the leak-free cointegration scan with
# a max-3-pairs-per-leg cap, to reduce the single-pair variance that tripped
# Sleeve G's drawdown halt. This directly tests the "more breadth = smoother
# equity" quant diversification hypothesis on the exact same validated edge.
# ---------------------------------------------------------------------------
SLEEVE_H_PAIRS = [
    "CADCHF~USDCAD", "USDCHF~USDMXN", "CADCHF~USDNOK", "NZDUSD~ZARJPY",
    "CADCHF~EURUSD", "CHFJPY~GBPSEK", "USDCHF~USDZAR", "GBPCHF~GBPUSD",
    "EURCAD~EURCHF", "GBPCAD~GBPCHF", "AUDCHF~AUDUSD", "CADJPY~CHFJPY",
    "AUDCAD~ZARJPY", "EURZAR~NZDCAD", "USDCHF~USDNOK", "CHFJPY~USDJPY",
    "AUDCHF~NZDUSD", "NZDCAD~NZDCHF", "NZDCAD~ZARJPY", "AUDNZD~EURJPY",
    "CADJPY~GBPSEK", "NZDCHF~NZDUSD", "AUDCAD~AUDCHF", "AUDNZD~GBPUSD",
    "AUDNZD~GBPJPY", "CADJPY~GBPNOK", "EURCHF~EURUSD", "GBPNOK~NOKJPY",
]
SLEEVE_H_MIN_OOS_START = SLEEVE_F_MIN_OOS_START
SLEEVE_H_PARAM_GRID = SLEEVE_G_PARAM_GRID
prep_sleeve_h = prep_sleeve_g
sig_sleeve_h = sig_sleeve_g


# ---------------------------------------------------------------------------
# Sleeve I: crypto long/short-ratio CONTRARIAN (fade the crowd). Raw-edge
# screen (screen_candidates.py / screen_candidates2.py) found this to be the
# strongest, cleanest, most intuitive new signal: when the exchange-wide
# long/short account ratio is extremely stretched long, forward 1-day raw
# returns are significantly negative (t=-23 over 116k 15m obs); extremely
# short, forward returns are significantly positive (t=+23). Both legs are
# profitable in ABSOLUTE (not just market-relative) terms, so this trades on
# the existing per-instrument ATR engine like sleeves A-E, at real prices
# (no synthetic-index ATR mismatch like sleeve F).
# ---------------------------------------------------------------------------
SLEEVE_I_UNIVERSE = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
                     "AVAXUSDT", "LINKUSDT", "DOGEUSDT", "ADAUSDT", "DOTUSDT"]
SLEEVE_I_PARAM_GRID = [
    dict(z_lookback_bars=672, entry_z=2.0),   # 672 bars = 7 days of 15m
    dict(z_lookback_bars=672, entry_z=2.5),
    dict(z_lookback_bars=288, entry_z=2.0),   # 3-day lookback
    dict(z_lookback_bars=288, entry_z=2.5),
    dict(z_lookback_bars=96, entry_z=2.0),    # 1-day lookback
]


def prep_sleeve_i(symbol: str):
    df = load_binance(symbol)
    out = df[["open", "high", "low", "close"]].copy()
    out["atr14"] = df["atr14_w"]
    ls = df["ls_ratio_global"]
    for w in {g["z_lookback_bars"] for g in SLEEVE_I_PARAM_GRID}:
        m = ls.rolling(w).mean()
        s = ls.rolling(w).std()
        out[f"lsz_{w}"] = (ls - m) / s.replace(0, np.nan)
    return out


def sig_sleeve_i(df: pd.DataFrame, params: dict):
    z = df[f"lsz_{params['z_lookback_bars']}"]
    long_cond = z <= -params["entry_z"]   # crowd extremely SHORT -> fade -> go long
    short_cond = z >= params["entry_z"]   # crowd extremely LONG -> fade -> go short
    long_cond = long_cond.fillna(False) & ~long_cond.shift(1).fillna(False)
    short_cond = short_cond.fillna(False) & ~short_cond.shift(1).fillna(False)
    entries = (long_cond | short_cond).fillna(False)
    sides = pd.Series(np.where(long_cond, 1, np.where(short_cond, -1, 0)), index=df.index)
    return entries, sides, df["atr14"]


# ---------------------------------------------------------------------------
# Sleeve J: crypto funding-rate extreme LONG-ONLY momentum-confirmation
# filter. Raw-edge screen found funding is NOT a clean contrarian signal
# (both high and low funding precede positive raw returns -- a market-beta
# confound), but the high-funding side alone is a strong, large, LONG-ONLY
# timing filter (raw 1d fwd +93.9bps, t=22.6, market-neutral excess +31.3bps,
# t=9.0): crowded, heavily-paid-for longs keep working short-term more often
# than not. Tested honestly here as a long-only entry filter (no short leg --
# the low-funding side showed no exploitable short edge).
# ---------------------------------------------------------------------------
SLEEVE_J_UNIVERSE = SLEEVE_I_UNIVERSE
SLEEVE_J_PARAM_GRID = [
    dict(z_lookback_bars=672, entry_z=2.0),
    dict(z_lookback_bars=672, entry_z=2.5),
    dict(z_lookback_bars=288, entry_z=2.0),
    dict(z_lookback_bars=288, entry_z=2.5),
    dict(z_lookback_bars=96, entry_z=2.0),
]


def prep_sleeve_j(symbol: str):
    df = load_binance(symbol)
    out = df[["open", "high", "low", "close"]].copy()
    out["atr14"] = df["atr14_w"]
    fr = df["funding_rate_pct"]
    for w in {g["z_lookback_bars"] for g in SLEEVE_J_PARAM_GRID}:
        m = fr.rolling(w).mean()
        s = fr.rolling(w).std()
        out[f"frz_{w}"] = (fr - m) / s.replace(0, np.nan)
    return out


def sig_sleeve_j(df: pd.DataFrame, params: dict):
    z = df[f"frz_{params['z_lookback_bars']}"]
    long_cond = z >= params["entry_z"]
    long_cond = long_cond.fillna(False) & ~long_cond.shift(1).fillna(False)
    entries = long_cond.fillna(False)
    sides = pd.Series(np.where(long_cond, 1, 0), index=df.index)
    return entries, sides, df["atr14"]


# ---------------------------------------------------------------------------
# Sleeve K: FX day-of-week seasonality. Per-symbol check (not just pooled)
# showed a consistent Monday-positive / Friday-negative daily-return pattern
# across 8/10 majors independently (not just a cross-symbol-correlation
# pooling artifact), t-stats in the double digits when pooled. Flagged
# honestly: this could be a genuine weekly flow/positioning effect, OR a
# data-vendor convention in how weekly OHLC candles are stitched across the
# weekend gap -- cannot fully rule out the latter without an independent data
# source. Tested as: long at Monday's open, exit at Monday's close; short at
# Friday's open, exit at Friday's close -- using the same ATR institutional
# risk box as sleeves A-E (so a real SL can still cut a bad Monday/Friday
# short).
# ---------------------------------------------------------------------------
SLEEVE_K_UNIVERSE = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF",
                     "NZDUSD", "USDMXN", "EURJPY", "GBPJPY"]
SLEEVE_K_PARAM_GRID = [
    dict(mode="mon_long_fri_short"),
    dict(mode="mon_long_only"),
    dict(mode="fri_short_only"),
]


def prep_sleeve_k(symbol: str):
    df = load_forex(symbol, "d1")
    out = df[["open", "high", "low", "close"]].copy()
    out["atr14"] = wilder_atr(out, 14)
    out["weekday"] = out.index.weekday
    return out


def sig_sleeve_k(df: pd.DataFrame, params: dict):
    mode = params["mode"]
    is_mon = df["weekday"] == 0
    is_fri = df["weekday"] == 4
    long_cond = is_mon if mode in ("mon_long_fri_short", "mon_long_only") else pd.Series(False, index=df.index)
    short_cond = is_fri if mode in ("mon_long_fri_short", "fri_short_only") else pd.Series(False, index=df.index)
    entries = (long_cond | short_cond)
    sides = pd.Series(np.where(long_cond, 1, np.where(short_cond, -1, 0)), index=df.index)
    return entries, sides, df["atr14"]


# ---------------------------------------------------------------------------
# Sleeve L: EQUITY INDICES, dedicated asset class. Indices have historically
# shown much stronger, more persistent secular trends than mean-reverting FX
# crosses (this exact dataset spans the entire 2015-2026 equity bull market,
# interrupted only by 2018Q4, 2020 COVID, and 2022), so a trend/breakout
# design -- not a mean-reversion one -- is the economically appropriate
# paradigm here (same Donchian + ATR-rank filter pattern validated in
# Sleeve C, but re-tested on a pure, dedicated index universe rather than
# folding XAUUSD into an FX-majors sleeve).
# ---------------------------------------------------------------------------
SLEEVE_L_UNIVERSE = ["SP500", "NAS100", "DJ30", "US2000", "GER40", "UK100",
                     "FR40", "JP225", "AU200", "HK50"]

SLEEVE_L_PARAM_GRID = [
    dict(don_n=55, atr_rank_min=50),
    dict(don_n=40, atr_rank_min=50),
    dict(don_n=80, atr_rank_min=50),
    dict(don_n=55, atr_rank_min=70),
    dict(don_n=40, atr_rank_min=30),
    dict(don_n=20, atr_rank_min=50),
]


def prep_sleeve_l(symbol: str):
    d1 = load_forex(symbol, "d1")
    d1["atr_rank"] = d1["atr14"].rolling(252, min_periods=60).rank(pct=True) * 100
    return d1


def sig_sleeve_l(d1: pd.DataFrame, params: dict):
    n = params["don_n"]
    hh = d1["high"].rolling(n).max().shift(1)
    ll = d1["low"].rolling(n).min().shift(1)
    long_cond = (d1["close"] > hh) & (d1["atr_rank"] >= params["atr_rank_min"])
    short_cond = (d1["close"] < ll) & (d1["atr_rank"] >= params["atr_rank_min"])
    entries = (long_cond | short_cond).fillna(False)
    sides = pd.Series(np.where(long_cond, 1, np.where(short_cond, -1, 0)), index=d1.index)
    return entries, sides, d1["atr14"]


# ---------------------------------------------------------------------------
# Sleeve M: METALS, dedicated asset class (precious + base). A classic
# cointegration test was run FIRST on the obvious a priori candidate pairs
# (gold/silver ratio XAUUSD~XAGUSD, platinum/palladium, copper/aluminium,
# zinc/lead) using statsmodels.coint over the full available history
# (2020-2026) -- none passed (p=0.40-0.91 across the board). That's an
# honest, important negative result: the textbook metals-ratio pairs trade
# does NOT hold statistically in this specific sample (plausibly because
# 2023-2025 saw gold de-couple from silver/platinum amid central-bank
# buying/de-dollarization flows unique to gold). Pivoted to the same
# trend/breakout design as Sleeve L instead, tested on its own dedicated
# metals universe, since this period's gold/silver/copper moves were
# large, persistent, and genuinely trending (not mean-reverting) in
# absolute price terms.
# ---------------------------------------------------------------------------
SLEEVE_M_UNIVERSE = ["XAUUSD", "XAGUSD", "XPTUSD",
                     "COPPER", "ALUMINIUM", "NICKEL", "ZINC", "LEAD"]
# XPDUSD excluded: history only starts 2023-01-20, which would cap the
# common-range intersection to ~11 OOS quarters, short of the >=20 mandate.

SLEEVE_M_PARAM_GRID = SLEEVE_L_PARAM_GRID
prep_sleeve_m = prep_sleeve_l
sig_sleeve_m = sig_sleeve_l


# ---------------------------------------------------------------------------
# Sleeve N: ENERGY, dedicated asset class -- WTI/Brent spread. Unlike
# metals, this pair IS genuinely cointegrated (Engle-Granger p=0.0010,
# OU half-life 16.9 trading days, beta=0.886, over the full 2016-2026
# sample) -- the WTI-Brent spread is one of the most famous, long-standing
# commodity term-structure trades in real energy markets (freight/quality/
# regional-supply arbitrage keeps the two benchmarks tethered). Only one
# genuinely cointegrated pair exists among this dataset's 3 energy
# instruments (GAS has no natural partner here), so this sleeve is
# necessarily a single-pair book -- built with the same OU-native
# mean-reversion engine validated in sleeves G/H (profit on reversion,
# stop on further divergence, time-stop), with the mandated 2-leg 82bps
# round-trip friction.
# ---------------------------------------------------------------------------
SLEEVE_N_PAIRS = ["UKBRENT~USWTI"]
SLEEVE_N_PARAM_GRID = SLEEVE_G_PARAM_GRID
sig_sleeve_n = sig_sleeve_g


def prep_sleeve_n(pair_id: str):
    a_sym, b_sym = pair_id.split("~")
    da = load_forex(a_sym, "d1")["close"]
    db = load_forex(b_sym, "d1")["close"]
    idx = da.index.intersection(db.index)
    la, lb = np.log(da.loc[idx]), np.log(db.loc[idx])

    w = SLEEVE_G_HEDGE_WINDOW
    cov = la.rolling(w).cov(lb).shift(1)
    var = lb.rolling(w).var().shift(1)
    beta = cov / var.replace(0, np.nan)
    mean_a = la.rolling(w).mean().shift(1)
    mean_b = lb.rolling(w).mean().shift(1)
    alpha = mean_a - beta * mean_b

    spread = la - (alpha + beta * lb)
    ret_a, ret_b = la.diff(), lb.diff()
    spread_ret = ret_a - beta * ret_b
    index = 100.0 * np.exp(spread_ret.fillna(0).cumsum())

    df = pd.DataFrame(index=idx)
    df["open"], df["high"], df["low"], df["close"] = index, index, index, index
    z_lookback = 20
    df["spread_mean"] = spread.rolling(z_lookback).mean()
    df["spread_std"] = spread.rolling(z_lookback).std()
    df["z"] = (spread - df["spread_mean"]) / df["spread_std"].replace(0, np.nan)
    return df


# ---------------------------------------------------------------------------
# Sleeve O: SESSION / OPENING-RANGE BREAKOUT (Asian-range breakout traded at
# London open), dedicated FX + metals universe. This is a distinct strategy
# family never tested anywhere in this project: "Breakout & Session-Based"
# (opening range breakout, Asian range breakout at London open, volatility
# contraction -> expansion / NR-style squeeze filter). Mechanics: each
# trading day's Asian-session high/low defines the range; if price breaks
# that range within the first N minutes of the London session, and the
# Asian range itself was in a *volatility-contraction* regime (narrow
# relative to its own trailing 20-day history -- the NR4/NR7 "squeeze"
# concept), take one breakout trade in the breakout direction. At most one
# signal per instrument per day. Uses the dataset's pre-computed `session`
# tags (normalized for case/label inconsistencies) on real 15m FX/metals
# bars -- genuine intraday session structure, not a daily-bar proxy.
# ---------------------------------------------------------------------------
SLEEVE_O_UNIVERSE = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
                     "USDCHF", "NZDUSD", "EURGBP", "XAUUSD", "XAGUSD"]

# DATA-QUALITY FINDING (discovered building this sleeve): Forex_Data's
# "15m"/"1h"/"4h" intraday files are NOT genuinely intraday for 2015-2022 --
# every symbol checked has exactly ~259-261 rows/year in that span (one bar
# per trading day, i.e. silently downsampled-to-daily data mislabeled as
# intraday), and only becomes true intraday-granularity data from 2023-01
# onward (10,851+ rows/year). A session-structure strategy (Asian range /
# London open) is meaningless on daily bars, so this sleeve's OOS window is
# restricted to >=2023-01-01. That leaves only ~13 OOS quarters available --
# BELOW the >=20 mandate. This sleeve is therefore run and reported as an
# explicit DIAGNOSTIC/EXPLORATORY result, not a certified one; the shortfall
# is a genuine dataset limitation, not a relaxed gate.
SLEEVE_O_MIN_OOS_START = pd.Timestamp("2023-01-01", tz="UTC")

SLEEVE_O_PARAM_GRID = [
    dict(window_mins=60, range_rank_max=100),   # no squeeze filter, 1h window
    dict(window_mins=60, range_rank_max=50),    # squeeze filter (below-median Asian range)
    dict(window_mins=60, range_rank_max=30),    # tight squeeze filter
    dict(window_mins=120, range_rank_max=100),  # wider 2h window, no filter
    dict(window_mins=120, range_rank_max=50),
    dict(window_mins=30, range_rank_max=50),    # tight window + squeeze filter
]

_SESSION_MAP = {"off": "off_hours", "new york": "new_york", "close": "off_hours"}


def prep_sleeve_o(symbol: str):
    df = load_forex(symbol, "15m")
    df["sess"] = df["session"].astype(str).str.lower().replace(_SESSION_MAP)
    df["date"] = df.index.date

    asian_mask = df["sess"] == "asian"
    asian = df[asian_mask].groupby("date").agg(a_hi=("high", "max"), a_lo=("low", "min"))
    asian["a_range"] = asian["a_hi"] - asian["a_lo"]
    asian["range_rank"] = asian["a_range"].rolling(20, min_periods=10).rank(pct=True) * 100

    df = df.join(asian[["a_hi", "a_lo", "range_rank"]], on="date")

    london_mask = df["sess"] == "london"
    first_london = df.loc[london_mask].groupby("date").apply(lambda g: g.index.min())
    df["london_open_time"] = df["date"].map(first_london)
    mins_since = (df.index.tz_localize(None) - pd.DatetimeIndex(df["london_open_time"]).tz_localize(None))
    df["mins_since_lo"] = mins_since.total_seconds() / 60.0
    return df


def sig_sleeve_o(df: pd.DataFrame, params: dict):
    w = params["window_mins"]
    rr_max = params["range_rank_max"]
    in_window = (df["sess"] == "london") & (df["mins_since_lo"] >= 0) & (df["mins_since_lo"] <= w)
    squeeze_ok = df["range_rank"] <= rr_max
    long_raw = in_window & squeeze_ok & (df["close"] > df["a_hi"])
    short_raw = in_window & squeeze_ok & (df["close"] < df["a_lo"])
    raw = (long_raw | short_raw).fillna(False)
    cum = raw.groupby(df["date"]).cumsum()
    entries = raw & (cum == 1)   # first qualifying breakout of the day only
    sides = pd.Series(np.where(long_raw & entries, 1, np.where(short_raw & entries, -1, 0)), index=df.index)
    return entries, sides, df["atr14"]


# ---------------------------------------------------------------------------
# Sleeve P: HURST-REGIME-SWITCHED INDICES. Retrofit of Sleeve L (which failed
# cleanly as a pure trend/breakout design) adding genuine mathematical
# regime detection: a rolling generalized-Hurst-exponent estimator (variance-
# scaling of price differences across multiple lags, Di Matteo-style) on
# each index's daily closes. When the exponent signals a persistent/trending
# regime (H >= hurst_trend_min) the sleeve runs the same Donchian breakout
# used in Sleeve L; when it signals an anti-persistent/mean-reverting regime
# (H <= hurst_mr_max) it switches to a Bollinger/z-score mean-reversion
# fade instead; in between (random-walk-like, H near 0.5) it stands down --
# directly testing "Trend persistence measures (Hurst, variance ratio)" and
# "Regime Analysis / regime-conditional strategy allocation" against the
# project's one clean asset-class kill (indices).
# ---------------------------------------------------------------------------

def rolling_hurst(series: pd.Series, window: int = 100, lag_max: int = 20) -> pd.Series:
    log_p = np.log(series)
    lags = np.arange(2, lag_max)
    log_lags = np.log(lags)

    def _h(x):
        tau = np.array([np.std(x[lag:] - x[:-lag]) for lag in lags])
        if np.any(tau <= 0) or np.any(np.isnan(tau)):
            return np.nan
        poly = np.polyfit(log_lags, np.log(np.sqrt(tau)), 1)
        return float(poly[0] * 2.0)

    return log_p.rolling(window).apply(_h, raw=True)


SLEEVE_P_UNIVERSE = SLEEVE_L_UNIVERSE

SLEEVE_P_PARAM_GRID = [
    dict(don_n=55, hurst_trend_min=0.55, hurst_mr_max=0.45, bb_window=20, z_entry=2.0),
    dict(don_n=40, hurst_trend_min=0.55, hurst_mr_max=0.45, bb_window=20, z_entry=2.0),
    dict(don_n=55, hurst_trend_min=0.60, hurst_mr_max=0.40, bb_window=20, z_entry=2.0),
    dict(don_n=55, hurst_trend_min=0.55, hurst_mr_max=0.45, bb_window=20, z_entry=2.5),
    dict(don_n=55, hurst_trend_min=0.55, hurst_mr_max=0.45, bb_window=14, z_entry=2.0),
    dict(don_n=80, hurst_trend_min=0.60, hurst_mr_max=0.40, bb_window=20, z_entry=2.5),
]


def prep_sleeve_p(symbol: str):
    d1 = load_forex(symbol, "d1")
    d1["hurst"] = rolling_hurst(d1["close"], window=100)
    return d1


def sig_sleeve_p(d1: pd.DataFrame, params: dict):
    n = params["don_n"]
    hh = d1["high"].rolling(n).max().shift(1)
    ll = d1["low"].rolling(n).min().shift(1)
    trend_regime = d1["hurst"] >= params["hurst_trend_min"]
    mr_regime = d1["hurst"] <= params["hurst_mr_max"]

    trend_long = trend_regime & (d1["close"] > hh)
    trend_short = trend_regime & (d1["close"] < ll)

    bbw = params["bb_window"]
    bb_mean = d1["close"].rolling(bbw).mean()
    bb_std = d1["close"].rolling(bbw).std()
    z = (d1["close"] - bb_mean) / bb_std.replace(0, np.nan)
    mr_long = mr_regime & (z < -params["z_entry"])
    mr_short = mr_regime & (z > params["z_entry"])

    long_cond = (trend_long | mr_long).fillna(False)
    short_cond = (trend_short | mr_short).fillna(False)
    entries = long_cond | short_cond
    sides = pd.Series(np.where(long_cond, 1, np.where(short_cond, -1, 0)), index=d1.index)
    return entries, sides, d1["atr14"]


# ---------------------------------------------------------------------------
# Sleeve Q: INTERMARKET-GATED INDICES TREND. Second retrofit of Sleeve L,
# this time adding a genuine cross-asset macro filter instead of a pure
# price-series regime detector: the Copper/Gold ratio, a textbook
# growth-vs-safe-haven risk gauge (industrial metal demand = growth/risk-on;
# gold bid = defensive/risk-off). Sleeve L's identical Donchian breakout
# signal is gated so longs only fire when the ratio's 20d MA is above its
# 100d MA (risk-on regime) and shorts only fire when it's below (risk-off)
# -- directly testing "Cross-Asset & Intermarket Relationships" (copper/gold
# ratio as growth proxy) and "Regime-conditional strategy allocation"
# against the project's clean indices kill.
# ---------------------------------------------------------------------------

def _copper_gold_risk_regime():
    cu = load_forex("COPPER", "d1")["close"]
    au = load_forex("XAUUSD", "d1")["close"]
    idx = cu.index.intersection(au.index)
    ratio = cu.loc[idx] / au.loc[idx]
    fast = ratio.rolling(20).mean()
    slow = ratio.rolling(100).mean()
    return (fast > slow).rename("risk_on")


SLEEVE_Q_UNIVERSE = SLEEVE_L_UNIVERSE
SLEEVE_Q_PARAM_GRID = SLEEVE_L_PARAM_GRID


def prep_sleeve_q(symbol: str):
    d1 = prep_sleeve_l(symbol)
    risk_on = _copper_gold_risk_regime()
    d1 = d1.join(risk_on, how="left")
    d1["risk_on"] = d1["risk_on"].ffill()
    return d1


def sig_sleeve_q(d1: pd.DataFrame, params: dict):
    n = params["don_n"]
    hh = d1["high"].rolling(n).max().shift(1)
    ll = d1["low"].rolling(n).min().shift(1)
    atr_ok = d1["atr_rank"] >= params["atr_rank_min"]
    long_cond = (d1["close"] > hh) & atr_ok & (d1["risk_on"] == True)
    short_cond = (d1["close"] < ll) & atr_ok & (d1["risk_on"] == False)
    entries = (long_cond | short_cond).fillna(False)
    sides = pd.Series(np.where(long_cond, 1, np.where(short_cond, -1, 0)), index=d1.index)
    return entries, sides, d1["atr14"]


# ---------------------------------------------------------------------------
# Sleeve R: VOLATILITY-REGIME-FILTERED L/S-RATIO CONTRARIAN. Retrofit of
# Sleeve I (the project's single best validated result, Sharpe 1.28 / +61%
# over 6yr @ realistic 4-5bps crypto friction) adding a realized-volatility-
# percentile regime filter: stand the strategy down when trailing 30-day
# realized vol is in its own top percentile band (the 2022-style
# crash/crypto-winter regime empirically documented to break mean-reversion
# strategies -- "Mean reversion strongly regime-dependent: +16% bull vs -41%
# bear" per the Phase-5 research review). Directly tests "Volatility regime
# switching" / "Regime Analysis" against the project's best strategy, aimed
# at its known weak spot (the 2022 drawdown).
# ---------------------------------------------------------------------------
SLEEVE_R_UNIVERSE = SLEEVE_I_UNIVERSE
SLEEVE_R_PARAM_GRID = [
    dict(z_lookback_bars=672, entry_z=2.0, vol_rank_max=100),   # control: filter off
    dict(z_lookback_bars=672, entry_z=2.0, vol_rank_max=70),
    dict(z_lookback_bars=672, entry_z=2.0, vol_rank_max=50),
    dict(z_lookback_bars=288, entry_z=2.0, vol_rank_max=70),
    dict(z_lookback_bars=288, entry_z=2.5, vol_rank_max=50),
    dict(z_lookback_bars=96, entry_z=2.0, vol_rank_max=70),
]


def prep_sleeve_r(symbol: str):
    out = prep_sleeve_i(symbol)
    rv_window = 30 * 96   # 30 days of 15m bars
    rank_window = 365 * 96  # trailing 1yr walk-forward percentile (no lookahead)
    rv = out["close"].pct_change().rolling(rv_window).std()
    out["vol_rank"] = rv.rolling(rank_window, min_periods=rv_window).rank(pct=True) * 100
    return out


def sig_sleeve_r(df: pd.DataFrame, params: dict):
    entries, sides, atr = sig_sleeve_i(df, params)
    ok = (df["vol_rank"] <= params["vol_rank_max"]).fillna(True)
    entries = entries & ok
    sides = sides.where(entries, 0)
    return entries, sides, atr


SLEEVES = {
    "A": dict(universe=SLEEVE_A_UNIVERSE, grid=SLEEVE_A_PARAM_GRID, prep=prep_sleeve_a,
              sig=sig_sleeve_a, name="A_CRYPTO_PULLBACK"),
    "B": dict(universe=SLEEVE_B_UNIVERSE, grid=SLEEVE_B_PARAM_GRID, prep=prep_sleeve_b,
              sig=sig_sleeve_b, name="B_CRYPTO_SWEEP"),
    "C": dict(universe=SLEEVE_C_UNIVERSE, grid=SLEEVE_C_PARAM_GRID, prep=prep_sleeve_c,
              sig=sig_sleeve_c, name="C_SWING_BREAKOUT"),
    "D": dict(universe=SLEEVE_D_UNIVERSE, grid=SLEEVE_D_PARAM_GRID, prep=prep_sleeve_d,
              sig=sig_sleeve_d, name="D_CRYPTO_MEANREV_DAILY"),
    "E": dict(universe=SLEEVE_E_UNIVERSE, grid=SLEEVE_E_PARAM_GRID, prep=prep_sleeve_e,
              sig=sig_sleeve_e, name="E_CRYPTO_ORDERFLOW_MOMENTUM"),
    "F": dict(universe=SLEEVE_F_PAIRS, grid=SLEEVE_F_PARAM_GRID, prep=prep_sleeve_f,
              sig=sig_sleeve_f, name="F_FX_STATARB_PAIRS",
              min_oos_start=SLEEVE_F_MIN_OOS_START, friction_bps_roundtrip=82.0),
    "G": dict(universe=SLEEVE_G_PAIRS, grid=SLEEVE_G_PARAM_GRID, prep=prep_sleeve_g,
              sig=sig_sleeve_g, name="G_FX_STATARB_PAIRS_OU_NATIVE",
              min_oos_start=SLEEVE_G_MIN_OOS_START, friction_bps_roundtrip=82.0,
              trade_fn=generate_statarb_trades),
    "H": dict(universe=SLEEVE_H_PAIRS, grid=SLEEVE_H_PARAM_GRID, prep=prep_sleeve_h,
              sig=sig_sleeve_h, name="H_FX_STATARB_DIVERSIFIED_28PAIR",
              min_oos_start=SLEEVE_H_MIN_OOS_START, friction_bps_roundtrip=82.0,
              trade_fn=generate_statarb_trades),
    "I": dict(universe=SLEEVE_I_UNIVERSE, grid=SLEEVE_I_PARAM_GRID, prep=prep_sleeve_i,
              sig=sig_sleeve_i, name="I_CRYPTO_LSRATIO_CONTRARIAN"),
    "J": dict(universe=SLEEVE_J_UNIVERSE, grid=SLEEVE_J_PARAM_GRID, prep=prep_sleeve_j,
              sig=sig_sleeve_j, name="J_CRYPTO_FUNDING_MOMENTUM_LONGONLY"),
    "K": dict(universe=SLEEVE_K_UNIVERSE, grid=SLEEVE_K_PARAM_GRID, prep=prep_sleeve_k,
              sig=sig_sleeve_k, name="K_FX_DOW_SEASONALITY"),
    "L": dict(universe=SLEEVE_L_UNIVERSE, grid=SLEEVE_L_PARAM_GRID, prep=prep_sleeve_l,
              sig=sig_sleeve_l, name="L_EQUITY_INDICES_TREND"),
    "M": dict(universe=SLEEVE_M_UNIVERSE, grid=SLEEVE_M_PARAM_GRID, prep=prep_sleeve_m,
              sig=sig_sleeve_m, name="M_METALS_TREND"),
    "N": dict(universe=SLEEVE_N_PAIRS, grid=SLEEVE_N_PARAM_GRID, prep=prep_sleeve_n,
              sig=sig_sleeve_n, name="N_ENERGY_WTI_BRENT_SPREAD",
              friction_bps_roundtrip=82.0, trade_fn=generate_statarb_trades),
    "O": dict(universe=SLEEVE_O_UNIVERSE, grid=SLEEVE_O_PARAM_GRID, prep=prep_sleeve_o,
              sig=sig_sleeve_o, name="O_FX_METALS_SESSION_ORB",
              min_oos_start=SLEEVE_O_MIN_OOS_START),
    "P": dict(universe=SLEEVE_P_UNIVERSE, grid=SLEEVE_P_PARAM_GRID, prep=prep_sleeve_p,
              sig=sig_sleeve_p, name="P_INDICES_HURST_REGIME_SWITCH"),
    "Q": dict(universe=SLEEVE_Q_UNIVERSE, grid=SLEEVE_Q_PARAM_GRID, prep=prep_sleeve_q,
              sig=sig_sleeve_q, name="Q_INDICES_COPPERGOLD_INTERMARKET_GATE"),
    "R": dict(universe=SLEEVE_R_UNIVERSE, grid=SLEEVE_R_PARAM_GRID, prep=prep_sleeve_r,
              sig=sig_sleeve_r, name="R_CRYPTO_LSRATIO_VOLREGIME_FILTERED"),
}
