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
}
