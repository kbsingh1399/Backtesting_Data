"""
Offline unit tests for orderbook.py — no network required.
Run with: python3 quant/live_data/test_orderbook.py
"""
from orderbook import LocalOrderBook, OrderBookResyncRequired


def test_snapshot_then_normal_sequence():
    book = LocalOrderBook(symbol="BTCUSDT")
    book.load_snapshot(
        last_update_id=100,
        bids=[(100.0, 1.0), (99.5, 2.0)],
        asks=[(100.5, 1.5), (101.0, 2.5)],
    )
    assert book.best_bid() == (100.0, 1.0)
    assert book.best_ask() == (100.5, 1.5)
    assert book.mid_price() == 100.25

    # First diff must bracket last_update_id+1 == 101
    book.apply_diff(
        first_update_id=99, final_update_id=102,
        bid_updates=[(100.0, 0.5)],   # update qty
        ask_updates=[(100.5, 0.0)],   # remove level
    )
    assert book.bids[100.0] == 0.5
    assert 100.5 not in book.asks
    assert book.last_update_id == 102

    # Next diff continues the sequence
    book.apply_diff(
        first_update_id=103, final_update_id=104,
        bid_updates=[(99.0, 3.0)],    # new level
        ask_updates=[],
    )
    assert book.bids[99.0] == 3.0
    assert book.last_update_id == 104
    print("test_snapshot_then_normal_sequence: PASS")


def test_stale_event_before_snapshot_is_ignored():
    book = LocalOrderBook(symbol="ETHUSDT")
    book.load_snapshot(last_update_id=500, bids=[(10.0, 1.0)], asks=[(10.1, 1.0)])
    # Event entirely predates snapshot -> must be silently ignored, not applied
    book.apply_diff(first_update_id=490, final_update_id=499,
                     bid_updates=[(9999.0, 9999.0)], ask_updates=[])
    assert 9999.0 not in book.bids
    assert book.last_update_id == 500
    print("test_stale_event_before_snapshot_is_ignored: PASS")


def test_gap_triggers_resync():
    book = LocalOrderBook(symbol="SOLUSDT")
    book.load_snapshot(last_update_id=10, bids=[(1.0, 1.0)], asks=[(1.1, 1.0)])
    book.apply_diff(first_update_id=10, final_update_id=11,
                     bid_updates=[], ask_updates=[])
    try:
        # Skip update_id 12..14 -> gap -> must raise
        book.apply_diff(first_update_id=15, final_update_id=16,
                         bid_updates=[], ask_updates=[])
        raise AssertionError("expected OrderBookResyncRequired")
    except OrderBookResyncRequired:
        print("test_gap_triggers_resync: PASS")


def test_zero_qty_removes_level_and_top_n_sorted_correctly():
    book = LocalOrderBook(symbol="XRPUSDT")
    book.load_snapshot(
        last_update_id=1,
        bids=[(1.00, 5), (0.99, 10), (0.98, 1)],
        asks=[(1.01, 5), (1.02, 10), (1.03, 1)],
    )
    bids, asks = book.top_n(2)
    assert bids == [(1.00, 5.0), (0.99, 10.0)]
    assert asks == [(1.01, 5.0), (1.02, 10.0)]
    row = book.snapshot_row(n_levels=2)
    assert row["bid_px_0"] == 1.00 and row["ask_px_0"] == 1.01
    assert row["mid"] == (1.00 + 1.01) / 2
    print("test_zero_qty_removes_level_and_top_n_sorted_correctly: PASS")


if __name__ == "__main__":
    test_snapshot_then_normal_sequence()
    test_stale_event_before_snapshot_is_ignored()
    test_gap_triggers_resync()
    test_zero_qty_removes_level_and_top_n_sorted_correctly()
    print("\nAll orderbook.py unit tests passed.")
