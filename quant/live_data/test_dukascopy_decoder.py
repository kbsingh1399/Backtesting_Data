"""
Offline unit test for the Dukascopy .bi5 binary tick decoder. No network
required — synthesizes a fake LZMA-compressed tick file matching the
documented format and verifies round-trip decoding.

Run with: python3 quant/live_data/test_dukascopy_decoder.py
"""
import lzma
import struct
from datetime import datetime, timezone

from dukascopy_tick_downloader import decode_bi5, RECORD_STRUCT, POINT_FACTOR


def make_fake_bi5(records):
    """records: list of (delta_ms, ask_raw, bid_raw, ask_vol, bid_vol)"""
    raw = b"".join(RECORD_STRUCT.pack(*r) for r in records)
    return lzma.compress(raw)


def test_decode_known_records():
    hour_start = datetime(2024, 1, 3, 10, tzinfo=timezone.utc)
    point_factor = POINT_FACTOR["EURUSD"]  # 100000
    # EURUSD ask=1.09345, bid=1.09330 at t+1500ms; then ask=1.09346 at t+2200ms
    records = [
        (1500, int(round(1.09345 * point_factor)), int(round(1.09330 * point_factor)), 1.5, 2.0),
        (2200, int(round(1.09346 * point_factor)), int(round(1.09331 * point_factor)), 0.5, 0.8),
    ]
    compressed = make_fake_bi5(records)
    ticks = decode_bi5(compressed, hour_start, point_factor)

    assert len(ticks) == 2
    assert abs(ticks[0]["ask"] - 1.09345) < 1e-9
    assert abs(ticks[0]["bid"] - 1.09330) < 1e-9
    assert ticks[0]["ts"] == datetime(2024, 1, 3, 10, 0, 1, 500000, tzinfo=timezone.utc)
    assert abs(ticks[1]["ask"] - 1.09346) < 1e-9
    assert ticks[1]["ts"] == datetime(2024, 1, 3, 10, 0, 2, 200000, tzinfo=timezone.utc)
    print("test_decode_known_records: PASS")


def test_empty_payload_returns_empty_list():
    assert decode_bi5(b"", datetime(2024, 1, 1, tzinfo=timezone.utc), 100000) == []
    print("test_empty_payload_returns_empty_list: PASS")


def test_corrupt_payload_raises():
    try:
        decode_bi5(b"not lzma data at all", datetime(2024, 1, 1, tzinfo=timezone.utc), 100000)
        raise AssertionError("expected ValueError")
    except ValueError:
        print("test_corrupt_payload_raises: PASS")


def test_wrong_length_after_decompress_raises():
    # Valid LZMA stream, but payload length not a multiple of 20 bytes
    bad = lzma.compress(b"\x00" * 13)
    try:
        decode_bi5(bad, datetime(2024, 1, 1, tzinfo=timezone.utc), 100000)
        raise AssertionError("expected ValueError")
    except ValueError:
        print("test_wrong_length_after_decompress_raises: PASS")


if __name__ == "__main__":
    test_decode_known_records()
    test_empty_payload_returns_empty_list()
    test_corrupt_payload_raises()
    test_wrong_length_after_decompress_raises()
    print("\nAll Dukascopy decoder tests passed.")
