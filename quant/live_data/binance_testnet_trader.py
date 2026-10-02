#!/usr/bin/env python3
"""
Binance SPOT TESTNET trading client — free, zero-risk paper trading.

This is intentionally wired ONLY to Binance's Spot Testnet
(https://testnet.binance.vision), per the explicit decision to use
paper/testnet trading, not real capital, for the live-trading side of this
project. There is no flag in this file to point it at the real exchange —
that is a deliberate safety choice, not an oversight. If you later decide to
go live with real money, that is a separate, explicit decision you make on
your own infrastructure with your own real API keys; this script does not
do it for you.

SETUP (you must do this yourself — it requires your own GitHub login, which
nobody else can do on your behalf):
    1. Go to https://testnet.binance.vision/
    2. Click "Log in with GitHub" and authorize.
    3. Click "Generate HMAC_SHA256 Key", save the API key + secret shown
       (the secret is only shown once).
    4. export BINANCE_TESTNET_API_KEY=...
       export BINANCE_TESTNET_API_SECRET=...
    5. The testnet account is pre-funded with fake USDT/BTC/etc. — nothing
       to deposit, nothing withdrawable, zero financial risk.

Requires (install where this actually runs — not this sandbox):
    pip install requests

Usage:
    python3 binance_testnet_trader.py balance
    python3 binance_testnet_trader.py price --symbol BTCUSDT
    python3 binance_testnet_trader.py order --symbol BTCUSDT --side BUY \
        --type MARKET --quantity 0.001
    python3 binance_testnet_trader.py open-orders --symbol BTCUSDT
    python3 binance_testnet_trader.py cancel --symbol BTCUSDT --order-id 12345
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import logging
import os
import sys
import time
from urllib.parse import urlencode

try:
    import requests
except ImportError:
    requests = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("binance_testnet")

TESTNET_BASE = "https://testnet.binance.vision"


class BinanceTestnetClient:
    def __init__(self, api_key: str, api_secret: str, base_url: str = TESTNET_BASE):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"X-MBX-APIKEY": api_key})

    def _signed_request(self, method: str, path: str, params: dict | None = None):
        params = dict(params or {})
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = 5000
        query = urlencode(params)
        signature = hmac.new(self.api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()
        query += f"&signature={signature}"
        url = f"{self.base_url}{path}?{query}"
        resp = self.session.request(method, url, timeout=10)
        if not resp.ok:
            raise RuntimeError(f"Binance testnet API error {resp.status_code}: {resp.text}")
        return resp.json()

    def _public_request(self, path: str, params: dict | None = None):
        url = f"{self.base_url}{path}"
        resp = self.session.get(url, params=params or {}, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def get_account(self) -> dict:
        return self._signed_request("GET", "/api/v3/account")

    def get_price(self, symbol: str) -> float:
        data = self._public_request("/api/v3/ticker/price", {"symbol": symbol})
        return float(data["price"])

    def place_order(self, symbol: str, side: str, order_type: str,
                     quantity: float | None = None, price: float | None = None,
                     time_in_force: str = "GTC") -> dict:
        params = {"symbol": symbol, "side": side, "type": order_type}
        if quantity is not None:
            params["quantity"] = quantity
        if order_type == "LIMIT":
            if price is None:
                raise ValueError("LIMIT orders require --price")
            params["price"] = price
            params["timeInForce"] = time_in_force
        return self._signed_request("POST", "/api/v3/order", params)

    def cancel_order(self, symbol: str, order_id: int) -> dict:
        return self._signed_request("DELETE", "/api/v3/order", {"symbol": symbol, "orderId": order_id})

    def get_open_orders(self, symbol: str | None = None) -> list:
        params = {"symbol": symbol} if symbol else {}
        return self._signed_request("GET", "/api/v3/openOrders", params)


def load_client() -> BinanceTestnetClient:
    api_key = os.environ.get("BINANCE_TESTNET_API_KEY")
    api_secret = os.environ.get("BINANCE_TESTNET_API_SECRET")
    if not api_key or not api_secret:
        log.error(
            "Missing BINANCE_TESTNET_API_KEY / BINANCE_TESTNET_API_SECRET.\n"
            "Get free testnet keys at https://testnet.binance.vision/ (log in with GitHub)."
        )
        sys.exit(1)
    return BinanceTestnetClient(api_key, api_secret)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("balance")

    p_price = sub.add_parser("price")
    p_price.add_argument("--symbol", required=True)

    p_order = sub.add_parser("order")
    p_order.add_argument("--symbol", required=True)
    p_order.add_argument("--side", required=True, choices=["BUY", "SELL"])
    p_order.add_argument("--type", dest="order_type", required=True, choices=["MARKET", "LIMIT"])
    p_order.add_argument("--quantity", type=float, required=True)
    p_order.add_argument("--price", type=float, default=None)

    p_open = sub.add_parser("open-orders")
    p_open.add_argument("--symbol", required=False)

    p_cancel = sub.add_parser("cancel")
    p_cancel.add_argument("--symbol", required=True)
    p_cancel.add_argument("--order-id", type=int, required=True)

    args = parser.parse_args()

    if requests is None:
        log.error("Missing dependency. Run: pip install requests")
        sys.exit(1)

    client = load_client()

    if args.command == "balance":
        acct = client.get_account()
        balances = [b for b in acct["balances"] if float(b["free"]) > 0 or float(b["locked"]) > 0]
        for b in balances:
            print(f"{b['asset']:>6}  free={b['free']:>15}  locked={b['locked']:>15}")
    elif args.command == "price":
        print(f"{args.symbol}: {client.get_price(args.symbol)}")
    elif args.command == "order":
        result = client.place_order(args.symbol, args.side, args.order_type, args.quantity, args.price)
        print(result)
    elif args.command == "open-orders":
        for o in client.get_open_orders(args.symbol):
            print(o)
    elif args.command == "cancel":
        print(client.cancel_order(args.symbol, args.order_id))


if __name__ == "__main__":
    main()
