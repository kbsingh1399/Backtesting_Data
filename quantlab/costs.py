"""
quantlab.costs
==============
Transaction-cost model built from the dataset's own ``spread`` column.

Forex_Data bars carry an integer ``spread`` in broker POINTS.  Multiplying by
the manifest ``point`` gives the quoted spread in price units for that bar.
That is a far better cost estimate than any flat bps assumption, and it is the
reason this backtest can claim to be cost-realistic.

Conventions
-----------
MT5 OHLC bars are BID prices.  Therefore:
  * long  entry  -> fill at ask  = open + spread            (pay full spread)
  * long  exit   -> fill at bid  = quoted level             (no spread)
  * short entry  -> fill at bid  = quoted level             (no spread)
  * short exit   -> fill at ask  = level + spread           (pay full spread)
Net: exactly one full spread per round turn, which we charge on exit-adjusted
R so it is visible in the trade blotter.

On top of the quoted spread we add
  * commission (round turn, bps of notional) by instrument category, and
  * slippage expressed as a multiple of the quoted spread, differentiated by
    exit type (stop orders slip, limit orders do not).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from . import config as C


@dataclass
class CostModel:
    symbol: str
    category: str
    point: float
    # median spread in POINTS, indexed by hour of day (UTC)
    spread_by_hour: Dict[int, float]
    spread_median: float
    commission_bps_rt: float
    has_live_spread: bool

    # ---------------------------------------------------------------
    def spread_points(self, bar_spread: np.ndarray, hours: np.ndarray) -> np.ndarray:
        """Per-bar spread in points, falling back to the hour-of-day median
        wherever the broker wrote 0 (the padded block and part of 2023-25)."""
        s = np.asarray(bar_spread, dtype=float).copy()
        fallback = np.array([self.spread_by_hour.get(int(h), self.spread_median) for h in hours])
        bad = ~np.isfinite(s) | (s <= 0)
        s[bad] = fallback[bad]
        return s

    def spread_price(self, bar_spread: np.ndarray, hours: np.ndarray) -> np.ndarray:
        return self.spread_points(bar_spread, hours) * self.point

    def commission_price(self, price: np.ndarray) -> np.ndarray:
        return np.asarray(price, dtype=float) * self.commission_bps_rt / 1e4

    def round_turn_cost(
        self,
        price: np.ndarray,
        bar_spread: np.ndarray,
        hours: np.ndarray,
        exit_kind: str = "stop",
    ) -> np.ndarray:
        """Total round-turn cost in PRICE units."""
        sp = self.spread_price(bar_spread, hours)
        slip_exit = {
            "stop": C.SLIPPAGE_STOP_SPREADS,
            "target": 0.0,
            "time": C.SLIPPAGE_TIMEEXIT_SPREADS,
        }.get(exit_kind, C.SLIPPAGE_TIMEEXIT_SPREADS)
        return sp * (1.0 + C.SLIPPAGE_ENTRY_SPREADS + slip_exit) + self.commission_price(price)


def build_cost_model(symbol: str, meta: Dict, bars: pd.DataFrame) -> CostModel:
    """Estimate the cost model for one instrument from its own pure-era bars."""
    point = float(meta.get("point", np.nan))
    if not np.isfinite(point) or point <= 0:
        digits = int(meta.get("digits", 5))
        point = 10.0 ** (-digits)
    category = meta.get("category", "Forex_Raw")

    live = bars.loc[bars["spread"] > 0, ["datetime", "spread"]]
    has_live = len(live) >= 500
    if has_live:
        hrs = live["datetime"].dt.hour
        by_hour = live.groupby(hrs)["spread"].median().to_dict()
        med = float(live["spread"].median())
    else:
        # No usable live spread: fall back to a conservative category default
        # expressed in bps of price, converted to points.
        default_bps = {"Forex_Raw": 1.2, "Indices": 2.0,
                       "Commodities_Raw": 1.5, "Commodities": 6.0}.get(category, 2.0)
        px = float(bars["close"].median()) if len(bars) else 1.0
        med = default_bps / 1e4 * px / point
        by_hour = {}

    by_hour = {int(k): float(v) for k, v in by_hour.items()}

    return CostModel(
        symbol=symbol,
        category=category,
        point=point,
        spread_by_hour=by_hour,
        spread_median=float(med),
        commission_bps_rt=float(C.COMMISSION_BPS_ROUND_TURN.get(category, 0.7)),
        has_live_spread=bool(has_live),
    )
