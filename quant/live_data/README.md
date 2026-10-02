# Live L2/L3 Market Data — Reality Check + Working Pipeline

This directory answers the question "get us L2/L3 data, free, pushed to
GitHub, and live-tradeable" as honestly as the facts allow. Short version:
**real L2/L3 order-book data only exists, for free, for crypto.** FX, metals
and indices (our `Forex_Data` universe) are traded over-the-counter and have
no central limit order book at all — there is no "L2 we haven't found yet"
there; it's a structural fact about how those markets work, not a sourcing
gap. See the per-asset breakdown below before using anything in this folder.

## What's actually here

| File | What it does | Free? | Tested here? |
|---|---|---|---|
| `orderbook.py` | Exchange-agnostic local L2 book reconstruction engine (snapshot+diff reconciliation) | n/a (library) | ✅ 4/4 unit tests pass, no network needed |
| `binance_l2_collector.py` | Live L2 depth collector for Binance — **real exchange order book**, public WebSocket, no account needed | ✅ 100% free, no signup | ✅ offline integration test passes (mocked snapshot+diff); **cannot run live from this sandbox** (see below) |
| `coinbase_l3_collector.py` | Live L3 (order-by-order, with order IDs) collector via Coinbase's "full" channel — one of only 3 venues (Coinbase/Bitfinex/Bitstamp) that publish true L3 at all | ✅ free, but requires your own free Coinbase account + API key (Coinbase requires auth on this channel since Aug 2023) | ✅ offline signing-logic tests pass; **cannot run live from this sandbox** |
| `binance_testnet_trader.py` | Places/cancels orders on **Binance Spot Testnet** — live order execution, fake money, zero financial risk | ✅ free, requires your own free testnet API key (GitHub login) | ✅ offline request-signing tests pass; **cannot run live from this sandbox** |
| `dukascopy_tick_downloader.py` | Downloads Dukascopy's free historical tick (bid/ask) data for FX/metals — the best *free* alternative where real L2/L3 doesn't exist | ✅ 100% free, no signup, 15+ years of history | ✅ 4/4 decoder unit tests pass on synthetic data, no network needed |
| `test_*.py` | Offline unit/integration tests for everything above — the only thing actually *executed* in this environment | — | ✅ all pass |

## Why nothing here was run live, and why that's not a shortcut I took

This sandbox's network is locked down to an allowlist: `pypi.org`,
`registry.npmjs.org`, and `github.com`/`api.github.com` work; **everything
else — including `api.binance.com`, `datafeed.dukascopy.com`,
`api.exchange.coinbase.com`, and even `www.google.com` — is blocked at the
TLS layer** (verified directly: TCP connects, then the handshake is reset).
This is an environment limit, not a design choice: these scripts need to run
on a machine with normal internet access — your laptop, a VPS, a free-tier
cloud VM, GitHub Actions, etc.

What I *could* do here, and did:
- Write production-quality, fully documented, free-tier-correct code against
  each vendor's real, verified API docs (checked live via web search, not
  from training memory — Binance's WS schema, Coinbase's auth requirement
  change, Dukascopy's `.bi5` format, Binance Testnet's GitHub-login flow).
- Extract the hardest logic (order-book snapshot+diff reconciliation, HMAC
  request signing, binary tick decoding) into pure functions with zero
  network dependency, and write real unit tests with synthetic data that
  prove that logic is correct — not just "looks right."
- **Not** fabricate a "ran successfully" log from a connection that never
  happened, and not silently skip the honesty checkpoint about what's really
  free vs. not.

## The honest L2/L3/free matrix (verified Oct 2026)

| | L2 (price-level depth) | L3 (order-by-order) |
|---|---|---|
| **Crypto** (Binance, Coinbase, OKX, ...) | Real, free, live via public WebSocket. This is the actual ceiling for "free depth data." | Only 3 venues publish it at all (Coinbase, Bitfinex, Bitstamp); Coinbase's feed is free but needs your own API key since Aug 2023. **Bulk historical L3 is not free anywhere** — even Coinbase/Bitfinex only give you the *live* stream for free; multi-year historical L3 flat files are a paid product (e.g. CoinAPI), running multiple GB/day per active symbol. |
| **FX / Metals / Indices / Energy** (our `Forex_Data` universe) | **Does not exist, for anyone, at any price.** These are OTC markets — no central limit order book. What a broker's "DOM" shows is their own internal synthetic liquidity, not published, not downloadable, not the real market. | Same — structurally impossible. |
| Best free alternative for FX/metals | — | **Dukascopy tick (bid/ask) data** — real, free, 15+ years, millisecond timestamps. Not depth, but a genuine fix for the data-quality defect already identified in Phase 5/6 (several `Forex_Data` intraday files are daily bars mislabeled as 15m before 2023/2024). |

## Live trading: why it's wired to testnet only

Per your decision, `binance_testnet_trader.py` only ever points at
`https://testnet.binance.vision` — there is deliberately no flag to switch
it to the real exchange. Testnet gives you:
- A **real, live order book and matching engine** (same infra as production).
- **Fake pre-funded balances** — nothing to deposit, nothing withdrawable,
  genuinely zero financial risk.
- The exact same REST/WS API shape as live trading, so strategy code written
  against it is a same-code swap to go live later — that step is yours to
  take deliberately, with your own keys, when/if you choose to.

## Setup (you have to do these steps yourself — no one else can)

1. **Binance L2 data** — nothing to set up, it's fully public:
   ```bash
   pip install -r requirements.txt
   python3 binance_l2_collector.py --symbols BTCUSDT ETHUSDT SOLUSDT BNBUSDT \
       XRPUSDT AVAXUSDT LINKUSDT DOGEUSDT ADAUSDT DOTUSDT
   ```
2. **Coinbase L3 data** — needs a free API key:
   - Create a free account at coinbase.com (no funding needed).
   - Generate an API key (Settings → API, or Advanced Trade API key mgmt).
   - `export COINBASE_API_KEY=... COINBASE_API_SECRET=... COINBASE_API_PASSPHRASE=...`
   - `python3 coinbase_l3_collector.py --products BTC-USD ETH-USD`
3. **Binance Testnet paper trading** — needs a free testnet key:
   - Go to https://testnet.binance.vision/, "Log in with GitHub", authorize.
   - Click "Generate HMAC_SHA256 Key", save both values immediately.
   - `export BINANCE_TESTNET_API_KEY=... BINANCE_TESTNET_API_SECRET=...`
   - `python3 binance_testnet_trader.py balance`
4. **Dukascopy FX/metals tick data** — nothing to set up, it's fully public:
   ```bash
   python3 dukascopy_tick_downloader.py --instrument EURUSD XAUUSD \
       --from 2024-01-01 --to 2024-01-07
   ```

Run the offline test suite anywhere, including this sandbox, with no setup:
```bash
cd quant/live_data
python3 test_orderbook.py
python3 test_binance_collector_offline.py
python3 test_dukascopy_decoder.py
python3 test_coinbase_signing.py
python3 test_binance_testnet_offline.py
```

## Data storage policy (why nothing captured here gets committed to git)

Per the project's existing convention (and this folder's `.gitignore`
entries), only the collector/downloader **code** is version-controlled.
Captured order-book data is excluded from git (`quant/live_data/capture/`)
because:
- Live L2 capture at even modest snapshot frequency across 10 symbols
  accumulates fast; full L3 or tick-level capture is gigabytes/day for
  active symbols (confirmed via CoinAPI's own published sizing figures).
- GitHub is not designed as a data lake and has hard 100MB per-file limits.
- The reproducible, valuable asset is the *pipeline*, not a frozen data
  dump — anyone can regenerate the data by running the scripts.

If you want a standing historical archive, point `--out-dir` at external
storage (a mounted volume, S3/GCS bucket, etc.) instead of a path under this
repo.

## How this connects back to the rest of the project

This is infrastructure, not a new certified sleeve. The natural next step,
if you want it, is to mine the captured Binance L2 order-book imbalance /
microprice signals as **new predictive features** for the existing crypto
sleeves (I/R), run through the same WFO + friction + drawdown-halt discipline
as every other sleeve in this project before anything gets called
"certified." Nothing in this folder should be treated as validated alpha —
it's the data-acquisition layer those future experiments would sit on top of.
