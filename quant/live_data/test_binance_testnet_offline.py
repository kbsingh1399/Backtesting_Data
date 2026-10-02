"""
Offline test for BinanceTestnetClient request construction/signing
(binance_testnet_trader.py). No network required — mocks requests.Session.

Run with: python3 quant/live_data/test_binance_testnet_offline.py
"""
import hashlib
import hmac
from unittest import mock
from urllib.parse import parse_qs, urlparse

from binance_testnet_trader import BinanceTestnetClient


def test_signed_request_includes_valid_signature():
    client = BinanceTestnetClient(api_key="testkey", api_secret="testsecret")

    captured = {}

    def fake_request(method, url, timeout=10):
        captured["method"] = method
        captured["url"] = url
        resp = mock.Mock()
        resp.ok = True
        resp.json.return_value = {"ok": True}
        return resp

    client.session.request = fake_request
    result = client._signed_request("GET", "/api/v3/account", {"foo": "bar"})
    assert result == {"ok": True}

    parsed = urlparse(captured["url"])
    qs = parse_qs(parsed.query)
    assert qs["foo"] == ["bar"]
    assert "timestamp" in qs
    assert "signature" in qs

    # Recompute the expected signature over everything except `signature` itself,
    # in the exact order it was appended (Binance requires signing fewer param
    # reorderings than the query string that's actually sent minus the sig).
    query_without_sig = captured["url"].split("?", 1)[1].rsplit("&signature=", 1)[0]
    expected_sig = hmac.new(b"testsecret", query_without_sig.encode(), hashlib.sha256).hexdigest()
    assert qs["signature"] == [expected_sig]
    print("test_signed_request_includes_valid_signature: PASS")


def test_api_key_header_set():
    client = BinanceTestnetClient(api_key="mykey123", api_secret="secret")
    assert client.session.headers["X-MBX-APIKEY"] == "mykey123"
    print("test_api_key_header_set: PASS")


def test_place_order_requires_price_for_limit():
    client = BinanceTestnetClient(api_key="k", api_secret="s")
    client._signed_request = mock.Mock(return_value={"status": "ok"})
    try:
        client.place_order("BTCUSDT", "BUY", "LIMIT", quantity=0.01, price=None)
        raise AssertionError("expected ValueError for missing price on LIMIT order")
    except ValueError:
        print("test_place_order_requires_price_for_limit: PASS")


def test_place_order_market_ok_without_price():
    client = BinanceTestnetClient(api_key="k", api_secret="s")
    client._signed_request = mock.Mock(return_value={"status": "FILLED"})
    result = client.place_order("BTCUSDT", "BUY", "MARKET", quantity=0.01)
    assert result["status"] == "FILLED"
    args, kwargs = client._signed_request.call_args
    assert args[2]["type"] == "MARKET"
    assert "price" not in args[2]
    print("test_place_order_market_ok_without_price: PASS")


if __name__ == "__main__":
    test_signed_request_includes_valid_signature()
    test_api_key_header_set()
    test_place_order_requires_price_for_limit()
    test_place_order_market_ok_without_price()
    print("\nAll Binance testnet trader offline tests passed.")
