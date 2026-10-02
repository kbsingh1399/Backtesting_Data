"""
Local limit-order-book (L2) reconstruction engine.

This is exchange-agnostic: it maintains a price -> quantity map for bids and
asks, and applies the standard "snapshot + buffered diff" reconciliation
algorithm documented by Binance (and used by essentially every exchange that
publishes a diff-depth WebSocket stream):

  1. Open the diff-depth WebSocket stream and buffer events.
  2. Pull a REST snapshot (has a `last_update_id`).
  3. Discard any buffered event whose `final_update_id` <= snapshot's
     `last_update_id`.
  4. The first event applied must satisfy
     `first_update_id <= last_update_id + 1 <= final_update_id`.
  5. Apply remaining events in order; each event's `first_update_id` must be
     exactly the previous event's `final_update_id + 1` (else resync).
  6. A quantity of 0 at a price level means "remove that level".

This module has NO network dependency, so it can be fully unit-tested with
synthetic messages (see test_orderbook.py) in environments that have no
outbound internet access (like this one).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import time


class OrderBookResyncRequired(Exception):
    """Raised when a sequence gap is detected; caller must refetch a snapshot."""


@dataclass
class LocalOrderBook:
    symbol: str
    bids: Dict[float, float] = field(default_factory=dict)   # price -> qty
    asks: Dict[float, float] = field(default_factory=dict)
    last_update_id: Optional[int] = None
    initialized: bool = False
    n_updates_applied: int = 0
    last_update_ts: Optional[float] = None

    def load_snapshot(self, last_update_id: int, bids: List[Tuple[float, float]],
                       asks: List[Tuple[float, float]]) -> None:
        self.bids = {float(p): float(q) for p, q in bids if float(q) > 0}
        self.asks = {float(p): float(q) for p, q in asks if float(q) > 0}
        self.last_update_id = last_update_id
        self.initialized = True
        self.n_updates_applied = 0
        self.last_update_ts = time.time()

    def apply_diff(self, first_update_id: int, final_update_id: int,
                    bid_updates: List[Tuple[float, float]],
                    ask_updates: List[Tuple[float, float]]) -> None:
        """Apply one diff-depth event. Raises OrderBookResyncRequired on a gap."""
        if not self.initialized or self.last_update_id is None:
            raise OrderBookResyncRequired("book not initialized with a snapshot yet")

        if final_update_id <= self.last_update_id:
            # Stale event, predates our snapshot -> ignore, not an error.
            return

        if self.n_updates_applied == 0:
            # First event applied after snapshot must bracket last_update_id+1.
            if not (first_update_id <= self.last_update_id + 1 <= final_update_id):
                raise OrderBookResyncRequired(
                    f"first event does not bracket snapshot id "
                    f"(snapshot={self.last_update_id}, event=[{first_update_id},{final_update_id}])"
                )
        else:
            if first_update_id != self.last_update_id + 1:
                raise OrderBookResyncRequired(
                    f"sequence gap: expected first_update_id={self.last_update_id + 1}, "
                    f"got {first_update_id}"
                )

        for price, qty in bid_updates:
            price, qty = float(price), float(qty)
            if qty == 0.0:
                self.bids.pop(price, None)
            else:
                self.bids[price] = qty

        for price, qty in ask_updates:
            price, qty = float(price), float(qty)
            if qty == 0.0:
                self.asks.pop(price, None)
            else:
                self.asks[price] = qty

        self.last_update_id = final_update_id
        self.n_updates_applied += 1
        self.last_update_ts = time.time()

    def best_bid(self) -> Optional[Tuple[float, float]]:
        if not self.bids:
            return None
        p = max(self.bids)
        return p, self.bids[p]

    def best_ask(self) -> Optional[Tuple[float, float]]:
        if not self.asks:
            return None
        p = min(self.asks)
        return p, self.asks[p]

    def top_n(self, n: int = 20) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        bid_levels = sorted(self.bids.items(), key=lambda kv: -kv[0])[:n]
        ask_levels = sorted(self.asks.items(), key=lambda kv: kv[0])[:n]
        return bid_levels, ask_levels

    def mid_price(self) -> Optional[float]:
        bb, ba = self.best_bid(), self.best_ask()
        if bb is None or ba is None:
            return None
        return (bb[0] + ba[0]) / 2.0

    def snapshot_row(self, n_levels: int = 10) -> dict:
        """Flatten current book state into one row suitable for a parquet log."""
        bids, asks = self.top_n(n_levels)
        row = {
            "ts": self.last_update_ts,
            "symbol": self.symbol,
            "last_update_id": self.last_update_id,
            "mid": self.mid_price(),
        }
        for i in range(n_levels):
            bp, bq = bids[i] if i < len(bids) else (None, None)
            ap, aq = asks[i] if i < len(asks) else (None, None)
            row[f"bid_px_{i}"] = bp
            row[f"bid_qty_{i}"] = bq
            row[f"ask_px_{i}"] = ap
            row[f"ask_qty_{i}"] = aq
        return row
