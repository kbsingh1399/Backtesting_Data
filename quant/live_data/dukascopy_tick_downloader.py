#!/usr/bin/env python3
"""
Dukascopy free historical tick (bid/ask) downloader for FX, metals & indices.

IMPORTANT — honest framing: this is NOT Level 2 or Level 3 data. FX/metals/
CFD markets traded off-exchange (OTC) have no central limit order book, so
"depth" in the Level 2/3 sense structurally does not exist for them, for
free or for any price. What Dukascopy publishes for free is their own ECN's
historical best-bid/best-ask TICK stream (every quote change, millisecond
timestamps, 15+ years of history) — a genuine, free, and large upgrade over
bar data, but still top-of-book only, not market depth.

Why this matters for this project: Forex_Data's existing intraday files are
confirmed (see PHASE5/PHASE6 notes) to be daily bars mislabeled as 15m bars
before 2023-2024 for several symbols. Dukascopy tick data is a real fix for
that specific defect — it gives genuinely granular, free, long-history data
for EURUSD/GBPUSD/XAUUSD/XAGUSD etc. Use this to rebuild clean intraday bars
at any resolution (tick -> 1m/5m/15m) with correct historical coverage.

Data format (reverse-engineered & widely documented; see header comment on
`decode_bi5` below): one .bi5 file per instrument/year/month/day/hour, LZMA-
compressed, containing fixed 20-byte big-endian tick records:
    uint32  time_delta_ms   (ms since start of the hour)
    uint32  ask_price_raw   (price * point_factor, integer)
    uint32  bid_price_raw
    float32 ask_volume      (in millions of base currency, per Dukascopy)
    float32 bid_volume

URL pattern:
    https://datafeed.dukascopy.com/datafeed/{SYMBOL}/{YYYY}/{MM-1:02d}/{DD:02d}/{HH:02d}h_ticks.bi5
(month is ZERO-indexed in the URL, e.g. January = "00")

Requires (install where this is actually run — NOT this sandbox, which has
no outbound internet access to Dukascopy):
    pip install requests pandas pyarrow

Usage:
    python3 dukascopy_tick_downloader.py --instrument EURUSD \
        --from 2024-01-01 --to 2024-01-02 --out-dir ./capture/dukascopy

Run `python3 test_dukascopy_decoder.py` first (no network needed) to confirm
the binary decoder is correct in your environment.
"""
from __future__ import annotations

import argparse
import lzma
import logging
import struct
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Tuple

try:
    import requests
except ImportError:
    requests = None

try:
    import pandas as pd
except ImportError:
    pd = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("dukascopy")

BASE_URL = "https://datafeed.dukascopy.com/datafeed"

# Point factor = how raw integer prices in the .bi5 file are scaled.
# Verified/commonly-documented values; if a decoded price looks implausible
# for an instrument not listed here, check Dukascopy's "point size" for it
# before trusting the output (see docstring warning in decode_bi5).
POINT_FACTOR = {
    "EURUSD": 100000, "GBPUSD": 100000, "AUDUSD": 100000, "NZDUSD": 100000,
    "USDCAD": 100000, "USDCHF": 100000,
    "USDJPY": 1000, "EURJPY": 1000, "GBPJPY": 1000,
    "XAUUSD": 1000,   # gold, 3 decimal places
    "XAGUSD": 10000,  # silver, 4 decimal places (verify against a known quote)
}

RECORD_STRUCT = struct.Struct(">IIIff")  # big-endian: u32,u32,u32,f32,f32
RECORD_SIZE = RECORD_STRUCT.size  # 20 bytes


def decode_bi5(raw_compressed: bytes, hour_start: datetime, point_factor: int) -> List[dict]:
    """Decompress + parse one hour's .bi5 file into a list of tick dicts.

    `raw_compressed` may be an empty bytes object (Dukascopy returns a 0-byte
    body for hours with no ticks, e.g. weekends) -> returns [].
    """
    if not raw_compressed:
        return []
    try:
        raw = lzma.decompress(raw_compressed)
    except lzma.LZMAError as e:
        raise ValueError(f"not a valid LZMA-compressed .bi5 payload: {e}")

    if len(raw) % RECORD_SIZE != 0:
        raise ValueError(
            f"decompressed size {len(raw)} is not a multiple of record size {RECORD_SIZE} "
            "(corrupt file or wrong format assumption)"
        )

    ticks = []
    for i in range(0, len(raw), RECORD_SIZE):
        delta_ms, ask_raw, bid_raw, ask_vol, bid_vol = RECORD_STRUCT.unpack_from(raw, i)
        ts = hour_start + timedelta(milliseconds=delta_ms)
        ticks.append({
            "ts": ts,
            "ask": ask_raw / point_factor,
            "bid": bid_raw / point_factor,
            "ask_volume_mm": ask_vol,
            "bid_volume_mm": bid_vol,
        })
    return ticks


def fetch_hour(instrument: str, hour_start: datetime) -> bytes:
    url = (
        f"{BASE_URL}/{instrument}/{hour_start.year:04d}/{hour_start.month - 1:02d}/"
        f"{hour_start.day:02d}/{hour_start.hour:02d}h_ticks.bi5"
    )
    resp = requests.get(url, timeout=20)
    if resp.status_code == 404:
        return b""  # instrument/date combo with no data published
    resp.raise_for_status()
    return resp.content


def download_range(instrument: str, start: datetime, end: datetime, out_dir: Path):
    point_factor = POINT_FACTOR.get(instrument)
    if point_factor is None:
        log.warning(
            "%s: no known point factor, defaulting to 100000 — VERIFY decoded "
            "prices look sane before trusting this output", instrument
        )
        point_factor = 100000

    out_dir = out_dir / instrument
    out_dir.mkdir(parents=True, exist_ok=True)

    cur_day = start.replace(hour=0, minute=0, second=0, microsecond=0)
    end_day = end.replace(hour=0, minute=0, second=0, microsecond=0)
    total_ticks = 0
    while cur_day <= end_day:
        day_ticks = []
        for hour in range(24):
            hour_start = cur_day.replace(hour=hour, tzinfo=timezone.utc)
            try:
                raw = fetch_hour(instrument, hour_start)
                day_ticks.extend(decode_bi5(raw, hour_start, point_factor))
            except Exception as e:  # noqa: BLE001
                log.error("%s %s hour=%02d: %s", instrument, cur_day.date(), hour, e)

        if day_ticks:
            df = pd.DataFrame(day_ticks)
            path = out_dir / f"{cur_day.strftime('%Y-%m-%d')}.parquet"
            df.to_parquet(path, index=False)
            total_ticks += len(df)
            log.info("%s %s: wrote %d ticks -> %s", instrument, cur_day.date(), len(df), path)
        else:
            log.info("%s %s: no ticks (weekend/holiday/no data)", instrument, cur_day.date())

        cur_day += timedelta(days=1)

    log.info("%s: done, %d total ticks written to %s", instrument, total_ticks, out_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--instrument", nargs="+", required=True,
                         help="e.g. EURUSD GBPUSD XAUUSD XAGUSD")
    parser.add_argument("--from", dest="date_from", required=True, help="YYYY-MM-DD")
    parser.add_argument("--to", dest="date_to", required=True, help="YYYY-MM-DD")
    parser.add_argument("--out-dir", type=str, default="./capture/dukascopy")
    args = parser.parse_args()

    if requests is None or pd is None:
        log.error("Missing dependencies. Run: pip install requests pandas pyarrow")
        sys.exit(1)

    start = datetime.strptime(args.date_from, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.date_to, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    out_dir = Path(args.out_dir)

    for instrument in args.instrument:
        download_range(instrument.upper(), start, end, out_dir)


if __name__ == "__main__":
    main()
