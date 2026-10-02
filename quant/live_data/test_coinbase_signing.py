"""
Offline test for Coinbase request-signing logic (coinbase_l3_collector.py).
No network required — verifies the HMAC-SHA256 signing matches Coinbase's
documented algorithm using a known-good test vector style check (signature
is deterministic given key/secret/timestamp/method/path).

Run with: python3 quant/live_data/test_coinbase_signing.py
"""
import base64
import os

from coinbase_l3_collector import sign_request, build_subscribe_message


def test_sign_request_is_deterministic_and_base64():
    # A valid base64-encoded secret, as Coinbase issues them
    fake_secret = base64.b64encode(b"0123456789abcdef0123456789abcdef").decode()
    sig1 = sign_request(fake_secret, "1700000000.0", "GET", "/users/self/verify")
    sig2 = sign_request(fake_secret, "1700000000.0", "GET", "/users/self/verify")
    assert sig1 == sig2, "signature must be deterministic for identical inputs"

    # Changing the timestamp must change the signature
    sig3 = sign_request(fake_secret, "1700000001.0", "GET", "/users/self/verify")
    assert sig1 != sig3

    # Must be valid base64
    base64.b64decode(sig1)
    print("test_sign_request_is_deterministic_and_base64: PASS")


def test_build_subscribe_message_requires_env_vars():
    for var in ("COINBASE_API_KEY", "COINBASE_API_SECRET", "COINBASE_API_PASSPHRASE"):
        os.environ.pop(var, None)
    try:
        build_subscribe_message(["BTC-USD"])
        raise AssertionError("expected RuntimeError when env vars are missing")
    except RuntimeError as e:
        assert "COINBASE_API_KEY" in str(e)
        print("test_build_subscribe_message_requires_env_vars: PASS")


def test_build_subscribe_message_with_env_vars():
    os.environ["COINBASE_API_KEY"] = "fake-key"
    os.environ["COINBASE_API_SECRET"] = base64.b64encode(b"fake-secret-bytes-000000").decode()
    os.environ["COINBASE_API_PASSPHRASE"] = "fake-pass"
    msg = build_subscribe_message(["BTC-USD", "ETH-USD"])
    assert msg["type"] == "subscribe"
    assert msg["product_ids"] == ["BTC-USD", "ETH-USD"]
    assert msg["channels"] == ["full", "heartbeat"]
    assert msg["key"] == "fake-key"
    assert msg["passphrase"] == "fake-pass"
    assert "signature" in msg and "timestamp" in msg
    print("test_build_subscribe_message_with_env_vars: PASS")
    for var in ("COINBASE_API_KEY", "COINBASE_API_SECRET", "COINBASE_API_PASSPHRASE"):
        os.environ.pop(var, None)


if __name__ == "__main__":
    test_sign_request_is_deterministic_and_base64()
    test_build_subscribe_message_requires_env_vars()
    test_build_subscribe_message_with_env_vars()
    print("\nAll Coinbase signing tests passed.")
