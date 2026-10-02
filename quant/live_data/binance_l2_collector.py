#!/usr/bin/env python3
"""
Binance L2 order-book collector — FREE, no API key, no account needed.

What this gets you (verified against Binance's public docs, Oct 2026):
  - Real exchange-published Level-2 depth (price-level aggregated book),
    not a synthetic broker DOM. Binance is a real central limit order book.
  - Live, streamed over a public WebSocket. No auth, no cost, no rate-limit
    tier required for this.
  - This is the realistic ceiling for "free L2": crypto venues are the only
    asset class where a genuine central order book is published for free.
    FX/metals/indices have NO equivalent — those markets are OTC and have no
    central limit order book at all (see dukascopy_tick_downloader.py for the
    best *free* alternative there: tick-level bid/ask quotes, not depth).

How it works (standard exchange-documented snapshot+diff algorithm, see
orderbook.py docstring for the exact steps):
  1. Open <symbol>@depth@100ms diff streams for all requested symbols.
  2. Buffer events, then pull a REST snapshot per symbol.
  3. Reconcile and maintain a LocalOrderBook per symbol.
  4. Every --snapshot-interval seconds, flush a top-N-level snapshot row to
     a local parquet file (NOT committed to git — see ../.gitignore).

Requires (install on the machine that will actually run this — NOT this
sandbox, which has no outbound internet access to Binance):
    pip install websockets requests pandas pyarrow

Usage:
    python3 binance_l2_collector.py \
        --symbols BTCUSDT ETHUSDT SOLUSDT BNBUSDT XRPUSDT AVAXUSDT LINKUSDT DOGEUSDT ADAUSDT DOTUSDT \
        --levels 20 --snapshot-interval 5 --out-dir ./capture/binance_l2

Run `python3 test_orderbook.py` first (no network needed) to confirm the
book-reconstruction logic is sound in your environment.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from orderbook import LocalOrderBook, OrderBookResyncRequired

try:
    import websockets
except ImportError:
    websockets = None

try:
    import requests
except ImportError:
    requests = None

try:
    import pandas as pd
except ImportError:
    pd = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("binance_l2")

REST_BASE = "https://api.binance.com"
WS_BASE = "wss://stream.binance.com:9443"

DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "AVAXUSDT", "LINKUSDT", "DOGEUSDT", "ADAUSDT", "DOTUSDT",
]


class SymbolWorker:
    """Owns one symbol's local book, its diff-event buffer, and its output log."""

    def __init__(self, symbol: str, out_dir: Path, levels: int, snapshot_interval: float):
        self.symbol = symbol
        self.book = LocalOrderBook(symbol=symbol)
        self.buffer: deque = deque()
        self.out_dir = out_dir / symbol
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.levels = levels
        self.snapshot_interval = snapshot_interval
        self._last_flush = 0.0
        self._rows: List[dict] = []

    def fetch_snapshot(self):
        url = f"{REST_BASE}/api/v3/depth"
        resp = requests.get(url, params={"symbol": self.symbol, "limit": 1000}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        self.book.load_snapshot(
            last_update_id=data["lastUpdateId"],
            bids=[(p, q) for p, q in data["bids"]],
            asks=[(p, q) for p, q in data["asks"]],
        )
        # Drop any buffered events that predate (or don't bracket) this snapshot.
        while self.buffer and self.buffer[0]["u"] <= self.book.last_update_id:
            self.buffer.popleft()
        log.info("%s: snapshot loaded @ last_update_id=%s", self.symbol, self.book.last_update_id)

    def handle_event(self, event: dict):
        self.buffer.append(event)
        if not self.book.initialized:
            return  # waiting on fetch_snapshot() to run
        while self.buffer:
            ev = self.buffer.popleft()
            try:
                self.book.apply_diff(
                    first_update_id=ev["U"], final_update_id=ev["u"],
                    bid_updates=[(p, q) for p, q in ev["b"]],
                    ask_updates=[(p, q) for p, q in ev["a"]],
                )
            except OrderBookResyncRequired as e:
                log.warning("%s: resync required (%s) -> refetching snapshot", self.symbol, e)
                self.book.initialized = False
                self.buffer.appendleft(ev)
                raise
        self.maybe_flush()

    def maybe_flush(self):
        now = time.time()
        if now - self._last_flush < self.snapshot_interval:
            return
        self._last_flush = now
        row = self.book.snapshot_row(n_levels=self.levels)
        self._rows.append(row)
        if len(self._rows) >= 60:  # flush to disk roughly every N snapshots
            self._write_rows()

    def _write_rows(self):
        if not self._rows:
            return
        df = pd.DataFrame(self._rows)
        df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = self.out_dir / f"{day}.parquet"
        if path.exists():
            existing = pd.read_parquet(path)
            df = pd.concat([existing, df], ignore_index=True)
        df.to_parquet(path, index=False)
        log.info("%s: wrote %d rows -> %s (cumulative %d)", self.symbol, len(self._rows), path, len(df))
        self._rows = []

    def flush_final(self):
        self._write_rows()


async def run_symbol(symbol: str, out_dir: Path, levels: int, snapshot_interval: float):
    worker = SymbolWorker(symbol, out_dir, levels, snapshot_interval)
    stream_name = f"{symbol.lower()}@depth@100ms"
    url = f"{WS_BASE}/ws/{stream_name}"
    while True:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                log.info("%s: connected to %s", symbol, url)
                worker.fetch_snapshot()
                async for raw in ws:
                    event = json.loads(raw)
                    try:
                        worker.handle_event(event)
                    except OrderBookResyncRequired:
                        worker.fetch_snapshot()
        except asyncio.CancelledError:
            worker.flush_final()
            raise
        except Exception as e:  # noqa: BLE001 - this is a long-running collector, must not die
            log.error("%s: connection error (%s) -> reconnecting in 3s", symbol, e)
            worker.flush_final()
            await asyncio.sleep(3)


async def main_async(symbols: List[str], out_dir: Path, levels: int, snapshot_interval: float):
    tasks = [
        asyncio.create_task(run_symbol(sym, out_dir, levels, snapshot_interval))
        for sym in symbols
    ]
    await asyncio.gather(*tasks)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--levels", type=int, default=20, help="book depth levels to log per snapshot")
    parser.add_argument("--snapshot-interval", type=float, default=5.0, help="seconds between logged snapshots")
    parser.add_argument("--out-dir", type=str, default="./capture/binance_l2")
    args = parser.parse_args()

    if websockets is None or requests is None or pd is None:
        log.error("Missing dependencies. Run: pip install websockets requests pandas pyarrow")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Starting Binance L2 collector for %d symbols -> %s", len(args.symbols), out_dir)
    try:
        asyncio.run(main_async(args.symbols, out_dir, args.levels, args.snapshot_interval))
    except KeyboardInterrupt:
        log.info("Stopped by user.")


if __name__ == "__main__":
    main()
