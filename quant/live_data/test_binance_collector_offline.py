"""
Offline integration test for SymbolWorker (binance_l2_collector.py) using a
mocked REST snapshot and synthetic diff events. No network required.

Run with: python3 quant/live_data/test_binance_collector_offline.py
"""
import sys
import types
from pathlib import Path
from unittest import mock

# Provide lightweight stand-ins for optional deps if they are not installed
# in this sandbox, so the import-time dependency checks don't block the test
# of the *logic* (orderbook.py has zero deps and is the part being verified).
for modname in ("websockets",):
    if modname not in sys.modules:
        sys.modules[modname] = types.ModuleType(modname)

import pandas as pd  # noqa: E402  (must be real; required for snapshot_row/flush)

import binance_l2_collector as blc  # noqa: E402


def test_symbolworker_snapshot_and_diff_flow(tmp_path: Path = Path("/tmp/binance_l2_test")):
    tmp_path.mkdir(parents=True, exist_ok=True)
    worker = blc.SymbolWorker("BTCUSDT", tmp_path, levels=5, snapshot_interval=0.0)

    fake_snapshot = {
        "lastUpdateId": 1000,
        "bids": [["50000.00", "1.0"], ["49999.00", "2.0"]],
        "asks": [["50001.00", "1.5"], ["50002.00", "2.5"]],
    }
    fake_resp = mock.Mock()
    fake_resp.json.return_value = fake_snapshot
    fake_resp.raise_for_status.return_value = None

    # Buffer a diff event that arrives BEFORE the snapshot fetch completes
    # (this is the real-world race the snapshot+diff algorithm must handle).
    worker.buffer.append({"U": 995, "u": 1001, "b": [["50000.00", "0.5"]], "a": []})

    with mock.patch("binance_l2_collector.requests.get", return_value=fake_resp):
        worker.fetch_snapshot()

    assert worker.book.initialized
    assert worker.book.last_update_id == 1000
    # The pre-buffered event should still be present (its final id > snapshot id)
    assert len(worker.buffer) == 1

    # Now simulate the live stream handing us that same buffered event
    worker.handle_event(worker.buffer.popleft())
    assert worker.book.bids[50000.00] == 0.5
    assert worker.book.last_update_id == 1001

    # A normal follow-on event
    worker.handle_event({"U": 1002, "u": 1002, "b": [], "a": [["50001.00", "0.0"]]})
    assert 50001.00 not in worker.book.asks
    assert worker.book.last_update_id == 1002

    row = worker.book.snapshot_row(n_levels=2)
    assert row["symbol"] == "BTCUSDT"
    assert row["bid_px_0"] == 50000.00
    print("test_symbolworker_snapshot_and_diff_flow: PASS")


if __name__ == "__main__":
    test_symbolworker_snapshot_and_diff_flow()
    print("\nAll offline binance_l2_collector tests passed.")
