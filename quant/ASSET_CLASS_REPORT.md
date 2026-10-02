# Phase 3 — Dedicated per-asset-class strategies (Indices / Metals / Energy)

**Request:** confirm `Forex_Data/` actually holds multiple asset classes, and
design *separate* strategies per class (indices like SP500, metals, etc.)
rather than lumping everything into "FX."

## 1. Confirmed: `Forex_Data/` is a multi-asset-class folder, not just FX

Despite the folder name, it contains five distinct asset classes (178
instruments total, each with 15m/1h/4h/d1 parquet files):

| Asset class | Count | Examples |
|---|---|---|
| **FX spot pairs** | 60 | EURUSD, GBPJPY, USDZAR, GBPNOK, ... |
| **Equity indices** | 16 | SP500, NAS100, DJ30, US2000, GER40, UK100, FR40, JP225, AU200, HK50, CHINA50, CHINAH, STOXX50, SWISS20, NETH25 |
| **Metals** | 19 | XAUUSD (gold), XAGUSD (silver), XPTUSD (platinum), XPDUSD (palladium), plus gold/silver cross-currency quotes, and base metals COPPER, ALUMINIUM, NICKEL, ZINC, LEAD |
| **Energy** | 3 | UKBRENT, USWTI, GAS |
| **Crypto CFDs** | 58 | BTCUSD, ETHUSD, SOLUSD, ... (same coins already covered far more richly by `Binance_Data/`'s 15m orderflow/funding/OI dataset used for sleeves A/B/D/E/I — not duplicated here) |

Data depth: indices mostly 2015-2026 (US2000 and AU200 start later, 2019-2020);
metals mostly 2020-2026 (XPDUSD only from 2023, excluded — see below); energy
2016-2026. All comfortably long enough for the mission's ≥20-OOS-quarter
standard except where noted.

## 2. Three new, genuinely separate asset-class sleeves built and honestly WFO-tested

Same institutional rigor as every other sleeve in this project: IS-locked
parameters per quarter, real ATR-box or OU-native risk mechanics, mandated
friction, Buy&Hold benchmark, equity PNGs, scorecards.

### Sleeve L — Equity Indices (trend/breakout)
**Design rationale:** indices show far more persistent secular trends than
mean-reverting FX crosses (this exact dataset spans nearly the entire
2015-2026 global equity bull market), so trend/breakout — not mean-reversion
— is the economically correct paradigm. Reused the Donchian + ATR-percentile
breakout filter validated in Sleeve C, on a dedicated 10-index universe
(SP500, NAS100, DJ30, US2000, GER40, UK100, FR40, JP225, AU200, HK50).

| | baseline | geom_search_v2 |
|---|---|---|
| ROI | **-4.71%** | -4.67% |
| Sharpe | -4.18 | -2.78 |
| windows traded | 5/21 | 8/21 |
| halted | Yes | Yes |
| Buy & Hold | **+59.5%** | +59.5% |

**Clean, complete failure.** See `results/L_baseline_equity.png`: the
strategy bleeds slowly and flatly for years while the simple buy-and-hold
benchmark triples. The honest diagnosis (consistent with sleeves A and D's
earlier finding on crypto): a tight 1.5×ATR stop with mandated friction
simply cannot survive the whipsaw of entering breakouts in a market that
grinds upward with frequent shallow pullbacks — only 5-8 of 21 quarters even
had enough in-sample edge to clear the honest gate at all. **For equity
indices specifically, this result argues FOR simple buy-and-hold /
long-bias exposure over active breakout trading, not against it** — a
legitimate, if anticlimactic, quant conclusion.

### Sleeve M — Metals (trend/breakout)
**Design rationale, with an honest pivot documented:** the obvious first
idea for metals is a cross-asset pairs trade (the classic gold/silver ratio,
platinum/palladium, copper/aluminium). Tested cointegration on all of these
*before* building anything (discipline: check the statistics before
designing the strategy around them) — **none passed** (Engle-Granger
p=0.40–0.91 across every candidate pair over the full 2020-2026 sample).
Plausible explanation: 2023-2025 saw gold structurally decouple from silver/
platinum amid central-bank buying and de-dollarization flows unique to
gold — a real regime break, not a data problem. Pivoted to the same trend/
breakout design as Sleeve L instead (universe: XAUUSD, XAGUSD, XPTUSD,
COPPER, ALUMINIUM, NICKEL, ZINC, LEAD; XPDUSD excluded — its 2023-only
history would cut the common OOS window to just 11 quarters).

| | baseline | geom_search_v2 |
|---|---|---|
| ROI | -0.01% (dead flat) | **+4.70%** |
| Sharpe | -0.003 | **1.06** |
| windows traded | 15/21 | 17/21 |
| halted | Yes (right at $225) | **No** |
| gross P&L | +$139.24 | +$485.92 |
| friction | $139.83 | $250.77 |
| Buy & Hold | +51.0% | +51.0% |

**Marginal but real.** The baseline run landed almost exactly on the
breakeven line (breakeven friction ≈40.8bps vs the 41bps mandate — a coin
flip). Allowing the institutional geometry grid (TP/ratchet variants) to be
IS-locked alongside the signal — the same disciplined "geom_search_v2"
process used throughout this project, never peeking at OOS to pick it —
turns this into a genuinely positive, Sharpe-1.06, never-halted result (see
`results/M_geom_search_v2_equity.png`). It is nowhere close to matching
Buy&Hold's +51% in this exact sample (that return was driven by gold's
historic 2025-2026 rally, which a mean-reversion-free trend filter only
partially captures), but it is a real, low-volatility, friction-surviving
edge — the same "modest but genuine" tier as sleeve E_baseline from the
original A-E mission.

### Sleeve N — Energy: WTI/Brent spread (statistical arbitrage)
**Design rationale:** with only 3 energy instruments and no natural partner
for GAS, cointegration was tested on the one plausible pair first — WTI vs
Brent crude, one of the most famous, long-standing spread trades in real
commodity markets (freight/quality/regional-supply arbitrage structurally
ties the two benchmarks together). Result: **p=0.0010, OU half-life 16.9
trading days, beta=0.886** over the full 2016-2026 sample — genuinely,
strongly cointegrated. Built with the same OU-native mean-reversion engine
validated in sleeves G/H (reversion target, divergence stop, time stop),
82bps 2-leg friction.

| | result |
|---|---|
| ROI | -0.75% |
| Sharpe | -0.89 |
| windows traded | 2/36 |
| gross P&L (full-sample, every param combo) | **negative in all 6 combinations tested** (-$53 to -$301) |

**Dead — and informatively so.** Despite strong *aggregate* cointegration
over the full decade, the trade-level P&L is negative even before friction
in every parameter combination tested. The likely explanation: this exact
sample contains two major structural shocks to the WTI-Brent relationship —
the April 2020 negative-WTI-price event (US storage crisis, unique to WTI)
and the 2022 Russia/Ukraine war's Brent-specific geopolitical risk premium —
both of which are large, *directional*, multi-month dislocations in the
spread, not short-lived mean-reverting noise. A statistical test run over
the *whole* sample can show "cointegrated on average" while the actual
trading experience is dominated by exactly the kind of regime-breaking event
an OU mean-reversion stop is designed to protect against (and did: the
stops did their job, cutting losses on each divergence, just at a net cost).
**Lesson: aggregate cointegration is necessary but not sufficient — a spread
can be statistically mean-reverting in the long run while still containing
large, structurally-driven dislocations that make short-horizon reversion
trading unprofitable net of costs.**

## 3. Where this leaves the full per-asset-class picture

| Asset class | Best strategy tried | Verdict |
|---|---|---|
| FX (spot) | Sleeves A/C/F/G/H/K (trend, breakout, cointegrated pairs, seasonality) | Mixed — thin stat-arb edge needs sub-5bps execution (Sleeve G/H); everything else dead. See `FINAL_REPORT.md`, `STATARB_REPORT.md`. |
| Crypto (perps) | Sleeve I (long/short-ratio contrarian) | **Best result of the entire project**: Sharpe 1.28, +61% over 6yr at realistic 4-5bps crypto execution cost. See `STRATEGY_SEARCH_PHASE2.md`. |
| **Equity indices** | Sleeve L (Donchian breakout) | Dead — buy-and-hold dominates; active trading only adds cost in a strong secular trend. |
| **Metals** | Sleeve M (trend/breakout, geometry-searched) | Marginal but genuine: Sharpe ~1.06, never halted, small but real edge — nowhere near Buy&Hold's return in this gold-bull sample, but a legitimate, friction-surviving, low-volatility result. |
| **Energy** | Sleeve N (WTI-Brent stat-arb) | Dead — long-run cointegration doesn't protect against the sample's two major structural dislocations (2020 negative WTI, 2022 Ukraine war). |

This directly answers the request: yes, the repository holds genuinely
separate asset classes, and treating each with the strategy design suited to
its own statistical behavior (trend for secularly-trending equities,
breakout for metals once the pairs-trade idea was honestly tested and
rejected, cointegration for the one energy pair that actually has it) is
the correct "think like a quant" approach — rather than forcing one
methodology onto all of them. Across indices, metals, and energy, only
metals produced a real (if modest) surviving edge; crypto's long/short-ratio
contrarian (Sleeve I) remains the strongest result in the whole project.

## Artifacts
- `quant/strategies.py` — sleeves L (`prep_sleeve_l`/`sig_sleeve_l`), M (reuses L's functions on a metals universe), N (`prep_sleeve_n`/`sig_sleeve_n`, OU-native).
- Results: `results/L_*`, `results/M_*`, `results/N_*` (baseline + geom_search_v2), all upserted into `results/scorecards.json`.
- Headline charts: `results/L_baseline_equity.png` (clean kill vs B&H), `results/M_geom_search_v2_equity.png` (marginal genuine edge).
