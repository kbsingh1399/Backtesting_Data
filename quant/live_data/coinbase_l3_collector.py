#!/usr/bin/env python3
"""
Coinbase Exchange L3 (order-by-order, "full" channel) collector.

Coinbase, Bitfinex and Bitstamp are the only crypto venues that publish a
true Level-3 feed (every individual order: received/open/done/change/match,
with order IDs) rather than price-aggregated Level-2. This script uses
Coinbase's "full" channel.

COST REALITY CHECK (verified against Coinbase's own changelog, Oct 2026):
Since 2023-08-01 Coinbase requires authentication even for the full/level2/
level3 market-data channels. It is still FREE — you just need:
  1. A free Coinbase account (no funding required for market-data-only use).
  2. A free API key generated at https://www.coinbase.com/settings/api or
     via Coinbase Advanced Trade / Exchange API key management.
Set these as environment variables before running:
    COINBASE_API_KEY, COINBASE_API_SECRET, COINBASE_API_PASSPHRASE

Universe note: not every symbol in our existing 10-coin sleeve universe has
a USD order book on Coinbase (venue listings differ from Binance). Default
below covers majors that reliably do; pass --products to override.

Requires (install where this actually runs — not this sandbox):
    pip install websockets pandas pyarrow

Usage:
    export COINBASE_API_KEY=...
    export COINBASE_API_SECRET=...
    export COINBASE_API_PASSPHRASE=...
    python3 coinbase_l3_collector.py --products BTC-USD ETH-USD --out-dir ./capture/coinbase_l3
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

try:
    import websockets
except ImportError:
    websockets = None

try:
    import pandas as pd
except ImportError:
    pd = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("coinbase_l3")

WS_URL = "wss://ws-feed.exchange.coinbase.com"

DEFAULT_PRODUCTS = ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "LINK-USD",
                     "DOGE-USD", "ADA-USD", "DOT-USD"]
# Note: BNB and XRP are historically not both listed on Coinbase the way they
# are on Binance — verify current listings at https://exchange.coinbase.com
# before assuming full 10-symbol parity with the Binance L2 universe.


def sign_request(secret: str, timestamp: str, method: str, request_path: str, body: str = "") -> str:
    message = f"{timestamp}{method}{request_path}{body}".encode("utf-8")
    hmac_key = base64.b64decode(secret)
    signature = hmac.new(hmac_key, message, hashlib.sha256)
    return base64.b64encode(signature.digest()).decode("utf-8")


def build_subscribe_message(products: List[str]) -> dict:
    api_key = os.environ.get("COINBASE_API_KEY")
    api_secret = os.environ.get("COINBASE_API_SECRET")
    api_passphrase = os.environ.get("COINBASE_API_PASSPHRASE")
    if not all([api_key, api_secret, api_passphrase]):
        raise RuntimeError(
            "Missing COINBASE_API_KEY / COINBASE_API_SECRET / COINBASE_API_PASSPHRASE "
            "env vars. Full/Level3 channel requires auth since 2023-08-01 "
            "(still free — generate a key at https://www.coinbase.com/settings/api)."
        )
    timestamp = str(time.time())
    signature = sign_request(api_secret, timestamp, "GET", "/users/self/verify")
    return {
        "type": "subscribe",
        "product_ids": products,
        "channels": ["full", "heartbeat"],
        "signature": signature,
        "key": api_key,
        "passphrase": api_passphrase,
        "timestamp": timestamp,
    }


class L3Logger:
    """Buffers raw full-channel messages and periodically flushes to parquet.

    We log the raw order-level event stream (received/open/done/change/match)
    rather than trying to maintain a live book in this script — reconstructing
    exact queue position from the full channel is materially more involved
    (ordering guarantees, sequence gap handling) and out of scope for a first
    cut. This gives a faithful, replayable record of the real L3 event stream.
    """

    def __init__(self, out_dir: Path, flush_every: int = 200):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.flush_every = flush_every
        self.rows: List[dict] = []

    def add(self, msg: dict):
        if msg.get("type") in ("received", "open", "done", "change", "match"):
            self.rows.append(msg)
        if len(self.rows) >= self.flush_every:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        df = pd.DataFrame(self.rows)
        by_product = df.groupby("product_id") if "product_id" in df.columns else [(None, df)]
        for product_id, sub in by_product:
            day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            prod_dir = self.out_dir / (product_id or "unknown")
            prod_dir.mkdir(parents=True, exist_ok=True)
            path = prod_dir / f"{day}.parquet"
            if path.exists():
                existing = pd.read_parquet(path)
                sub = pd.concat([existing, sub], ignore_index=True)
            sub.to_parquet(path, index=False)
        log.info("flushed %d raw L3 events", len(self.rows))
        self.rows = []


async def run(products: List[str], out_dir: Path):
    logger = L3Logger(out_dir)
    while True:
        try:
            async with websockets.connect(WS_URL, ping_interval=20, ping_timeout=20) as ws:
                await ws.send(json.dumps(build_subscribe_message(products)))
                log.info("subscribed to full channel for %s", products)
                async for raw in ws:
                    msg = json.loads(raw)
                    if msg.get("type") == "error":
                        log.error("Coinbase error: %s", msg)
                        continue
                    logger.add(msg)
        except Exception as e:  # noqa: BLE001
            logger.flush()
            log.error("connection error (%s) -> reconnecting in 3s", e)
            await asyncio.sleep(3)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--products", nargs="+", default=DEFAULT_PRODUCTS)
    parser.add_argument("--out-dir", type=str, default="./capture/coinbase_l3")
    args = parser.parse_args()

    if websockets is None or pd is None:
        log.error("Missing dependencies. Run: pip install websockets pandas pyarrow")
        sys.exit(1)

    try:
        asyncio.run(run(args.products, Path(args.out_dir)))
    except KeyboardInterrupt:
        log.info("Stopped by user.")


if __name__ == "__main__":
    main()
