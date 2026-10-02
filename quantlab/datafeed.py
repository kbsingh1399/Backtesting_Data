"""
quantlab.datafeed
=================
Loading + integrity screening for Forex_Data parquet files.

The headline issue this module exists to solve: the shipped
``*_15m_real.parquet`` files are **not** pure 15-minute series.  Each file is
a concatenation of

    [ DAILY bars 2015..2022 ]  +  [ hourly bars ]  +  [ true 15m bars ]

all carrying the same column schema and the ``_15m_`` filename.  Any feature
computed over the full file (EMA, ATR, FVG, rolling extremes, hour-of-day) is
silently computed across three different sampling frequencies.  We detect the
first timestamp from which the nominal spacing actually holds and clip there.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import config as C

OHLC = ["open", "high", "low", "close"]
READ_COLS = ["datetime", "open", "high", "low", "close", "tick_volume", "spread"]


# --------------------------------------------------------------------------
# Resolution integrity
# --------------------------------------------------------------------------
@dataclass
class ResolutionProfile:
    symbol: str
    timeframe: str
    n_bars: int
    first: pd.Timestamp
    last: pd.Timestamp
    frac_nominal: float
    pure_start: Optional[pd.Timestamp]
    n_pure: int
    n_padded: int
    padded_modal_seconds: float
    dup_timestamps: int
    ohlc_violations: int
    zero_spread_pct: float

    def as_dict(self) -> Dict:
        d = self.__dict__.copy()
        for k in ("first", "last", "pure_start"):
            d[k] = None if pd.isna(d[k]) else str(d[k])
        return d


def detect_pure_start(
    dt: pd.Series,
    nominal_seconds: int,
    window: int = C.PURE_RESOLUTION_WINDOW,
    min_frac: float = C.PURE_RESOLUTION_MIN_FRAC,
) -> Tuple[Optional[pd.Timestamp], float]:
    """First timestamp from which the series genuinely samples at `nominal`.

    We slide a window over the diff series and take the first position where
    at least `min_frac` of diffs equal the nominal spacing.  Weekend gaps and
    holidays keep the global fraction below 1.0 even for clean data, which is
    why this is a rolling test rather than a global one.
    """
    diffs = dt.diff().dt.total_seconds()
    is_nom = (diffs == nominal_seconds).astype(float)
    n = len(is_nom)
    frac_global = float(is_nom.mean()) if n else 0.0
    w = min(window, max(50, n // 20))
    if n <= w + 2:
        return (dt.iloc[0] if n else None), frac_global
    roll = is_nom.rolling(w).mean().to_numpy()
    hits = np.where(roll >= min_frac)[0]
    if not len(hits):
        return None, frac_global
    start_i = max(int(hits[0]) - w + 1, 0)
    return dt.iloc[start_i], frac_global


def profile_file(symbol: str, timeframe: str) -> Optional[ResolutionProfile]:
    path = C.FOREX_DIR / f"{symbol}_{timeframe}_real.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path, columns=READ_COLS)
    df = df.sort_values("datetime").reset_index(drop=True)
    nominal = C.NOMINAL_SECONDS[timeframe]
    pure_start, frac = detect_pure_start(df["datetime"], nominal)

    if pure_start is not None:
        pad = df[df["datetime"] < pure_start]
        pure = df[df["datetime"] >= pure_start]
    else:
        pad, pure = df, df.iloc[0:0]

    pad_modal = float(pad["datetime"].diff().dt.total_seconds().median()) if len(pad) > 2 else float("nan")

    hi, lo = df["high"], df["low"]
    viol = int(
        ((hi < df[["open", "close"]].max(axis=1)) | (lo > df[["open", "close"]].min(axis=1)) | (hi < lo)).sum()
    )

    return ResolutionProfile(
        symbol=symbol,
        timeframe=timeframe,
        n_bars=len(df),
        first=df["datetime"].iloc[0],
        last=df["datetime"].iloc[-1],
        frac_nominal=round(frac, 4),
        pure_start=pure_start,
        n_pure=int(len(pure)),
        n_padded=int(len(pad)),
        padded_modal_seconds=pad_modal,
        dup_timestamps=int(df["datetime"].duplicated().sum()),
        ohlc_violations=viol,
        zero_spread_pct=round(float((df["spread"] == 0).mean()) * 100.0, 2),
    )


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _pure_start_cached(symbol: str, timeframe: str) -> Optional[pd.Timestamp]:
    path = C.FOREX_DIR / f"{symbol}_{timeframe}_real.parquet"
    if not path.exists():
        return None
    dt = pd.read_parquet(path, columns=["datetime"])["datetime"].sort_values()
    start, _ = detect_pure_start(dt, C.NOMINAL_SECONDS[timeframe])
    return start


def load_bars(
    symbol: str,
    timeframe: str,
    enforce_pure: bool = True,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Load one instrument/timeframe as a clean, sorted, de-duplicated frame.

    `enforce_pure=True` clips off the mis-sampled padded block.  Always leave
    it on for anything that feeds a model.
    """
    path = C.FOREX_DIR / f"{symbol}_{timeframe}_real.parquet"
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_parquet(path, columns=READ_COLS)
    df = df.sort_values("datetime")
    df = df[~df["datetime"].duplicated(keep="last")].reset_index(drop=True)

    if enforce_pure:
        ps = _pure_start_cached(symbol, timeframe)
        if ps is None:
            return df.iloc[0:0]
        df = df[df["datetime"] >= ps].reset_index(drop=True)

    # Structural OHLC repair: a handful of bars in the raw export have
    # high/low inconsistent with open/close.  Clamp rather than drop so the
    # bar index stays contiguous.
    df["high"] = df[["high", "open", "close"]].max(axis=1)
    df["low"] = df[["low", "open", "close"]].min(axis=1)

    for col, dt in (("open", "float64"), ("high", "float64"), ("low", "float64"),
                    ("close", "float64"), ("tick_volume", "float64"), ("spread", "float64")):
        df[col] = df[col].astype(dt)

    if start is not None:
        df = df[df["datetime"] >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        df = df[df["datetime"] <= pd.Timestamp(end, tz="UTC")]

    return df.reset_index(drop=True)


def available_symbols(timeframe: str = "15m") -> List[str]:
    out = []
    suffix = f"_{timeframe}_real.parquet"
    for fn in os.listdir(C.FOREX_DIR):
        if fn.endswith(suffix):
            out.append(fn[: -len(suffix)])
    return sorted(out)
