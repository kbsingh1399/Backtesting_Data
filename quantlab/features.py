"""
quantlab.features
=================
Strictly causal feature engineering on the 15m grid.

Design rules enforced here
--------------------------
1. **No higher-timeframe file joins.** The 4h/d1 parquet files have their own
   padding artefacts and their own bar-stamping quirks (some D1 bars are
   stamped 19:00 or 07:00).  Everything higher-timeframe is *resampled from
   the pure 15m series itself*, so the context can never disagree with the
   execution series and can never peek.
2. **Aggregates are shifted before use.** A daily/weekly/4h aggregate is only
   visible to a 15m bar once that aggregate's period has fully closed.
3. **Every rolling statistic ends at the current bar's close.** The signal bar
   is evaluated on its own close; execution happens at the *next* bar's open.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

EPS = 1e-12


# --------------------------------------------------------------------------
# Primitive indicators
# --------------------------------------------------------------------------
def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift(1)
    return pd.concat(
        [df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0.0)
    dn = (-d).clip(lower=0.0)
    au = up.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    ad = dn.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    return 100.0 - 100.0 / (1.0 + au / (ad + EPS))


def efficiency_ratio(close: pd.Series, n: int = 40) -> pd.Series:
    """Kaufman efficiency ratio: |net move| / sum|moves|. ~1 trending, ~0 chop."""
    net = (close - close.shift(n)).abs()
    path = close.diff().abs().rolling(n).sum()
    return net / (path + EPS)


def yang_zhang_vol(df: pd.DataFrame, n: int = 24) -> pd.Series:
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    pc = c.shift(1)
    lo_ = np.log(o / pc.replace(0, np.nan))
    lc_ = np.log(c / o.replace(0, np.nan))
    lho = np.log(h / o.replace(0, np.nan))
    llo = np.log(l / o.replace(0, np.nan))
    lhc = np.log(h / c.replace(0, np.nan))
    llc = np.log(l / c.replace(0, np.nan))
    rs = lho * lhc + llo * llc
    k = 0.34 / (1.34 + (n + 1.0) / (n - 1.0))
    v = lo_.rolling(n).var(ddof=1) + k * lc_.rolling(n).var(ddof=1) + (1 - k) * rs.rolling(n).mean()
    return np.sqrt(v.clip(lower=0.0))


def ou_halflife(x: pd.Series, n: int = 96) -> pd.Series:
    """Rolling Ornstein-Uhlenbeck half-life of a (stationary-ish) spread."""
    dx = x.diff()
    lx = x.shift(1)
    cov = dx.rolling(n).cov(lx)
    var = lx.rolling(n).var()
    beta = cov / (var + EPS)
    theta = (-beta).clip(lower=1e-6)
    return (np.log(2.0) / theta).clip(upper=4 * n)


def hurst_rs(close: pd.Series, n: int = 128, step: int = 8) -> pd.Series:
    """Rescaled-range Hurst, computed every `step` bars and forward-filled.

    Forward-filling is causal (we carry the last *computed* value forward),
    and the step keeps the cost linear.
    """
    lr = np.log(close.replace(0, np.nan)).diff().to_numpy()
    out = np.full(len(close), np.nan)
    for i in range(n, len(close), step):
        s = lr[i - n + 1 : i + 1]
        s = s[np.isfinite(s)]
        if len(s) < n // 2:
            continue
        y = np.cumsum(s - s.mean())
        rng = y.max() - y.min()
        sd = s.std(ddof=1)
        if sd > EPS and rng > EPS:
            out[i] = np.clip(np.log(rng / sd) / np.log(len(s)), 0.05, 0.95)
    return pd.Series(out, index=close.index).ffill()


def rolling_zscore(x: pd.Series, n: int) -> pd.Series:
    m = x.rolling(n).mean()
    s = x.rolling(n).std(ddof=1)
    return (x - m) / (s + EPS)


# --------------------------------------------------------------------------
# Higher-timeframe context, resampled from the 15m series
# --------------------------------------------------------------------------
def _period_agg(df: pd.DataFrame, key: pd.Series) -> pd.DataFrame:
    """OHLC aggregate per period label, plus the period's last 15m index."""
    g = df.groupby(key, sort=True)
    out = pd.DataFrame(
        {
            "open": g["open"].first(),
            "high": g["high"].max(),
            "low": g["low"].min(),
            "close": g["close"].last(),
            "vol": g["tick_volume"].sum(),
        }
    )
    return out


def add_context(df: pd.DataFrame) -> pd.DataFrame:
    """Attach daily / weekly / 4h context, each shifted so only *closed*
    periods are visible to the current 15m bar."""
    dt = df["datetime"]
    df = df.copy()
    df["date"] = dt.dt.floor("D")
    df["hour"] = dt.dt.hour
    df["dow"] = dt.dt.dayofweek          # Mon=0 .. Sun=6
    df["minute_of_day"] = dt.dt.hour * 60 + dt.dt.minute

    # ---- previous completed day ----------------------------------------
    daily = _period_agg(df, df["date"])
    prev_d = daily.shift(1)
    df["pdh"] = df["date"].map(prev_d["high"])
    df["pdl"] = df["date"].map(prev_d["low"])
    df["pdc"] = df["date"].map(prev_d["close"])
    df["pd_range"] = df["pdh"] - df["pdl"]

    # ---- previous completed week (FX week: Mon..Fri label) --------------
    wk = dt.dt.tz_localize(None).dt.to_period("W-SUN").astype(str)
    df["_wk"] = wk
    weekly = _period_agg(df, df["_wk"])
    prev_w = weekly.shift(1)
    df["pwh"] = df["_wk"].map(prev_w["high"])
    df["pwl"] = df["_wk"].map(prev_w["low"])

    # ---- Asian session range of the *current* day -----------------------
    asian_mask = df["hour"] < 6
    a = df[asian_mask].groupby(df.loc[asian_mask, "date"])
    ah = a["high"].max()
    al = a["low"].min()
    df["asian_high"] = df["date"].map(ah)
    df["asian_low"] = df["date"].map(al)
    # Not knowable before 06:00 UTC.
    not_yet = df["hour"] < 6
    df.loc[not_yet, ["asian_high", "asian_low"]] = np.nan

    # ---- 4h regime, visible only after the 4h bar closes ----------------
    h4key = dt.dt.floor("4h")
    df["_h4"] = h4key
    h4 = _period_agg(df, df["_h4"])
    h4_ema50 = h4["close"].ewm(span=50, adjust=False, min_periods=50).mean()
    h4_ema200 = h4["close"].ewm(span=200, adjust=False, min_periods=200).mean()
    h4_atr = atr(h4.rename_axis("p"), 14)
    h4_ctx = pd.DataFrame(
        {
            "h4_ema50": h4_ema50,
            "h4_ema200": h4_ema200,
            "h4_atr": h4_atr,
            "h4_close": h4["close"],
            "h4_er": efficiency_ratio(h4["close"], 20),
        }
    ).shift(1)                      # <- only the *previous closed* 4h bar
    for col in h4_ctx.columns:
        df[col] = df["_h4"].map(h4_ctx[col])

    # ---- daily volatility context ---------------------------------------
    d_atr = atr(daily.rename_axis("p"), 14)
    d_ctx = pd.DataFrame({"d_atr": d_atr, "d_close": daily["close"]}).shift(1)
    df["d_atr"] = df["date"].map(d_ctx["d_atr"])
    df["d_atr_pctile"] = df["date"].map(
        d_ctx["d_atr"].rolling(120, min_periods=40).rank(pct=True)
    )

    return df.drop(columns=["_wk", "_h4"])


def add_bar_features(df: pd.DataFrame) -> pd.DataFrame:
    """15m-grid indicators. All end at the current bar's close."""
    df = df.copy()
    c, h, l, o = df["close"], df["high"], df["low"], df["open"]

    df["atr14"] = atr(df, 14)
    df["atr64"] = atr(df, 64)
    df["rng"] = (h - l).replace(0, np.nan)
    df["body_frac"] = (c - o).abs() / df["rng"]
    df["close_pos"] = (c - l) / df["rng"]            # 1 = closed on the high
    df["upper_wick"] = (h - np.maximum(o, c)) / df["rng"]
    df["lower_wick"] = (np.minimum(o, c) - l) / df["rng"]
    df["rng_atr"] = df["rng"] / (df["atr14"] + EPS)

    df["rsi14"] = rsi(c, 14)
    df["rsi_64"] = rsi(c, 64)
    df["er40"] = efficiency_ratio(c, 40)
    df["yz24"] = yang_zhang_vol(df, 24)
    df["vol_expansion"] = df["atr14"] / (df["atr64"] + EPS)

    tp = (h + l + c) / 3.0
    vwap = (tp * df["tick_volume"]).rolling(48).sum() / (df["tick_volume"].rolling(48).sum() + EPS)
    df["vwap_dist_atr"] = (c - vwap) / (df["atr14"] + EPS)
    df["ou_hl"] = ou_halflife(df["vwap_dist_atr"].fillna(0.0), 96)
    df["hurst"] = hurst_rs(c, 128, 8)

    df["roc16_atr"] = (c - c.shift(16)) / (df["atr14"] + EPS)
    df["roc64_atr"] = (c - c.shift(64)) / (df["atr14"] + EPS)
    df["tickvol_z"] = rolling_zscore(df["tick_volume"], 96)
    df["spread_z"] = rolling_zscore(df["spread"].replace(0, np.nan).ffill(), 480)

    # consecutive directional bars (exhaustion proxy)
    sgn = np.sign(c.diff().fillna(0.0)).to_numpy()
    run = np.zeros(len(sgn))
    for i in range(1, len(sgn)):
        run[i] = run[i - 1] + sgn[i] if sgn[i] == sgn[i - 1] and sgn[i] != 0 else sgn[i]
    df["dir_run"] = np.clip(run, -10, 10)

    # regime relative to the previous closed 4h bar
    df["px_vs_h4ema200_atr"] = (c - df["h4_ema200"]) / (df["atr14"] + EPS)
    df["h4_trend_atr"] = (df["h4_ema50"] - df["h4_ema200"]) / (df["h4_atr"] + EPS)
    df["pd_range_atr"] = df["pd_range"] / (df["atr14"] + EPS)
    df["pos_in_pd_range"] = (c - df["pdl"]) / (df["pd_range"] + EPS)

    return df


def round_number_distance(price: pd.Series, point: float, digits: int) -> pd.Series:
    """Distance to the nearest 'big figure' in price units.

    Osler (2003) shows stop-loss orders cluster just beyond round numbers and
    take-profit orders cluster *at* them.  For a 5-digit FX quote the relevant
    grid is 0.0050 / 0.0100; we use a grid of 500 points, which generalises to
    JPY pairs, metals and indices.
    """
    grid = 500.0 * point
    rem = np.mod(price.to_numpy(), grid)
    return pd.Series(np.minimum(rem, grid - rem), index=price.index)
