# Phase 5 — Concept Coverage Report

**Question answered:** *"Have you applied all the concepts that exist in the world
regarding quant trading to find an edge?"* and the follow-up, pasted taxonomy
of ~21 sections covering Indices/Forex/Metals quant trading, with the
instruction not to skip any category.

**Honest headline answer:** No exhaustive list of "every quant concept" can
literally be executed and certified in one research sprint — several whole
sub-fields (listed-options pricing/Greeks, L2 order-book HFT, broker/FIX
infrastructure, regulatory compliance) require data or execution
infrastructure this sandbox does not have, at all, under any amount of
effort. What **was done** this phase: every testable strategy *family* and
analytical *technique* in the pasted taxonomy was mapped to either (a) an
honest walk-forward result already produced in this project (Phases 1-4,
Sleeves A-N), (b) a new result produced this phase (Sleeves O-R + three
fail-fast quantitative screens), or (c) an explicit, reasoned "out of scope"
with the exact missing data/infra named. Nothing was silently dropped.

---

## 1. New results produced this phase

All new sleeves follow the same mandate as every prior sleeve unless
explicitly flagged otherwise: next-bar-open entries, 1.5xATR SL / 3.0R TP,
ratchet +0.2R@+1.2R, time-decay exit, $5,000 capital, $12.5 fixed risk,
$225 DD halt, max 3 concurrent, >=20 quarterly OOS windows with IS-only
parameter locking, 41bps round-trip friction (single-instrument) unless
noted. Code: `strategies.py` (Sleeves O/P/Q/R), run via `campaign.py`.

### Sleeve O — FX + Metals session Opening-Range Breakout — **DEAD (clean, instant kill)**
*Tests: "Breakout & Session-Based" family (ORB, Asian-range-breakout-at-
London-open, volatility-contraction/squeeze filter).*

Mechanics: each day's Asian-session high/low on real 15m FX/metals bars
defines a range; price breaking that range within the first 30-120 minutes
of the London session, optionally gated by a below-median "squeeze" filter
on the Asian range's own trailing width, triggers one breakout trade.
Universe: EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD, EURGBP,
XAUUSD, XAGUSD. 6-combo param grid (window x squeeze threshold).

**Result: 0 trades in both baseline and geometry-searched variants.** The
IS edge gate (never lock a parameter set with non-positive in-sample total
R) rejected *every* parameter combination in *every* candidate quarter —
direct measurement showed in-sample aggregate R was negative across all 6
geometry variants in every quarter checked (e.g. 2023-Q3: -228R to -824R
across param combos on ~90-340 IS trades). The raw signal has **no edge at
all** before friction is even applied; this is the fastest, cleanest kill
in the project (the strategy never earns the right to trade OOS).

**Important ancillary discovery (data quality):** building this sleeve
surfaced that `Forex_Data`'s `*_15m_real.parquet` / `*_1h_real.parquet` /
`*_4h_real.parquet` files are **not genuinely intraday before 2023** — every
symbol checked (EURUSD, and spot-checked others) has only ~259-261 rows/year
for 2015-2022 (one bar per trading day, i.e. silently downsampled-to-daily
data mislabeled as 15m/1h/4h), and only becomes true intraday-granularity
data from 2023-01 onward (10,000+ rows/year). Session-structure strategies
are meaningless on daily bars, so Sleeve O's OOS window was restricted to
>=2023-01-01, leaving only **14 OOS quarters — below the >=20 mandate.**
This sleeve is reported as an explicit diagnostic result, not a certified
one; the shortfall is a genuine dataset limitation (flagged, not hidden),
and a direct instance of the "tick/bar data quality, feed differences"
pitfall named in the user's own taxonomy (section 16/21).
Artifacts: `results/O_baseline_equity.png`, `results/O_geom_search_v2_equity.png`.

### Sleeve P — Hurst-exponent regime-switched Indices — **DEAD, but informative**
*Tests: "Trend persistence measures (Hurst, variance ratio)" + "Regime
Analysis / regime-conditional strategy allocation" (sections 3, 6, 8),
applied directly against the project's one clean asset-class kill
(Sleeve L, indices trend, ROI -4.7%).*

Mechanics: a rolling generalized-Hurst-exponent estimator (variance-scaling
of log-price differences across lags 2-19, Di Matteo-style) computed on
each index's daily closes. H >= 0.55 (persistent/trending) -> run Sleeve
L's Donchian breakout; H <= 0.45 (anti-persistent/mean-reverting) -> switch
to a Bollinger/z-score mean-reversion fade instead; H in between (random-
walk-like) -> stand down. Universe = Sleeve L's 10 indices.

| Variant | ROI | Sharpe | Win% | Gross P&L | Friction | Windows | Halted |
|---|---|---|---|---|---|---|---|
| baseline | -3.93% | -1.28 | 36.3% | **-$3.11** | $193.35 | 14/21 | Yes (2024-11-06) |
| geom_search_v2 | -4.71% | -1.75 | 42.1% | -$38.30 | $197.15 | 17/21 | Yes (2023-07-19) |

B&H over same window: +59.5%. **Still a net loser after friction**, but the
quantitative comparison to Sleeve L is the real finding: Sleeve L's raw
gross P&L (pre-friction) was **-$174.94**; Sleeve P's regime-aware version
cuts that to **essentially breakeven (-$3.11 gross)**. The Hurst filter is
measurably *doing its job* — it successfully avoids most of the raw-edge
bleed that killed pure trend-following on indices — but the signal that
remains after regime-gating is too thin (too few, too marginal trades) to
cross the 41bps/trade friction hurdle. Diagnosed as: genuine partial
de-risking, insufficient standalone edge. Artifacts:
`results/P_baseline_equity.png`, `results/P_geom_search_v2_equity.png`.

### Sleeve Q — Copper/Gold intermarket-gated Indices trend — **DEAD / inconclusive (thin sample)**
*Tests: "Cross-Asset & Intermarket Relationships" (copper/gold ratio as a
growth/risk-on-off proxy) + "Regime-conditional strategy allocation"
(sections 4, 8).*

Mechanics: Sleeve L's identical Donchian breakout, but longs only fire when
the Copper/Gold ratio's 20d MA is above its 100d MA (risk-on/growth regime)
and shorts only fire when below (risk-off/defensive regime) — a classic
cross-asset macro confirmation filter.

**Result: ROI -1.06%, Sharpe -1.16, only 9 trades across 3/21 windows.**
The intermarket gate is so restrictive combined with the IS min-trades gate
(>=6 IS trades to lock a parameter set) that most quarters never clear the
bar to trade at all. With only 9 OOS trades, no statistically meaningful
read on whether the filter adds value is possible either way — reported
honestly as inconclusive-and-dead rather than forcing a verdict from an
underpowered sample. Artifact: `results/Q_baseline_equity.png`.

### Sleeve R — Volatility-regime-filtered crypto L/S-ratio contrarian — **GENUINE IMPROVEMENT (best Phase-5 result)**
*Tests: "Volatility regime switching" / "Regime Analysis" (sections 3, 8)
against the project's single best validated strategy (Sleeve I), directly
targeting its documented weak spot: the 2022 drawdown. (Literature basis:
Coinquant.ai's 78-backtest study found crypto mean-reversion is sharply
regime-dependent, +16% in bull regimes vs -41% in bear regimes — see
PROGRESS.md External Sources.)*

Mechanics: identical to Sleeve I (fade extreme crowd long/short-ratio
z-scores), with one addition — a trailing 1-year walk-forward percentile
rank of 30-day realized volatility; trades are skipped when that percentile
exceeds a threshold (i.e. stand down during the highest-vol regimes, such
as the 2022 crypto-winter unwind).

At the mandated 41bps friction / $225 DD halt, **Sleeve R baseline is dead**
(ROI -4.59%, Sharpe -2.86, halted after 2 windows) — **identical in kind to
Sleeve I's own certified baseline** (ROI -3.96%, halted after 1 window).
This is expected and consistent: the 41bps FX-CFD-calibrated friction
mandate is simply too high for 15-minute-bar crypto turnover regardless of
regime-filtering (crypto exchanges charge far less in reality, ~4-8bps
maker/taker, which is why Sleeve I's own prior work used a disclosed,
documented 4bps diagnostic override to show the strategy's real-world
shape). Run under that **same, already-disclosed diagnostic basis**
(4bps round-trip friction, DD halt disabled — exactly matching the existing
`I_diag_nohalt_4bps` scorecard entry) for a genuine apples-to-apples
comparison:

| | Trades | ROI | Sharpe | Win% | Payoff | Max DD | Windows |
|---|---|---|---|---|---|---|---|
| I_diag_nohalt_4bps (prior best, Phase 3) | 17,830 | +60.98% | 1.276 | 46.4% | 1.186 | $3,087.67 | 21/21 |
| **R_diag_nohalt_4bps (vol-regime-filtered)** | 15,830 | **+64.12%** | **1.420** | 46.4% | 1.191 | **$2,387.54** | 21/21 |

The volatility regime filter trades **11% less** (correctly skipping the
worst-vol windows) while **improving ROI by +3.1pp, Sharpe by +11%, and
cutting max drawdown by 23%** ($3,087.67 -> $2,387.54). This is a real,
quantified, same-direction improvement on the project's best strategy —
not a reversal of its fundamental status (it's still a diagnostic-friction
result, not a certified-at-mandate one, exactly like its parent Sleeve I),
but it is the single cleanest positive finding of Phase 5. See
`results/R_diag_nohalt_4bps_equity.png` (visibly smoother equity curve,
shallower drawdowns, vs. both Buy&Hold and the unfiltered Sleeve I).

---

## 2. Fail-fast quantitative screens (no full backtest needed — killed on contact)

### PCA / Johansen basket stat-arb across US equity indices — **DEAD**
*Tests: "Relative Value" (section 3: generalizing pairs trading to a basket)
+ "PCA, factor models" + "Cointegration (Johansen)" (section 6).*

Ran a Johansen cointegration test (the correct multivariate generalization
of Engle-Granger to a >2-asset basket) on SP500/NAS100/DJ30/US2000 daily
log-prices, 2020-2026. Trace statistic for rank=0 was **34.60, below even
the 90% critical value of 44.49** — fail to reject "no cointegrating
relationship." PCA on the same returns shows PC1 explains 86.3% of variance
(expected: these are all highly-correlated US equity benchmarks sharing
one market-beta factor), but — critically — **that is not evidence of a
tradeable mean-reverting residual**; the Johansen test is the correct check
for that, and it fails outright. Conclusion: no PCA/basket stat-arb edge
exists among these four indices in this sample; killed before writing a
single line of execution code, consistent with "fail fast."

### Turn-of-month seasonality — **DEAD**
*Tests: "Seasonality & Calendar" (section 3) beyond the project's existing
day-of-week screen (Sleeve K).*

Welch's t-test on daily returns (first 3 / last 2 trading days of each
month vs. all other days) across SP500, NAS100, DJ30, GER40, UK100,
2020-2026: **p-values 0.57-0.87 across all five indices** (e.g. SP500:
turn-of-month mean +6.5bps/day vs other-days +4.4bps/day, t=0.46, p=0.65).
No statistically distinguishable turn-of-month effect survives in this
post-2020 sample — consistent with the broader finding (also true of the
classic day-of-week effect pre-Phase-3) that well-known calendar anomalies
have been substantially arbitraged away in modern, liquid index markets.
Killed fail-fast; no backtest built.

---

## 3. Concept-taxonomy coverage map (pasted list, sections 1-21)

| # | Section | Status |
|---|---|---|
| 1 | Instruments & Market Structure | **Partially covered.** Cash/CFD mechanics implicit in all sleeves (Forex_Data = CFD bars). Futures roll mechanics, contract specs, auctions, circuit breakers, LBMA fixings: **out of scope** — no futures/auction/fixing-level data in this dataset, only CFD OHLC bars. |
| 2 | Fundamental & Macro Drivers | **Out of scope for direct modeling** (no earnings, CB balance sheet, PMI, CFTC-flow datasets available) but **proxied once**: Copper/Gold ratio as a growth/risk macro gauge (Sleeve Q). |
| 3 | Strategy Families | **Covered broadly.** Trend/momentum: Sleeves C, L, M, P (trend leg). Mean reversion: Sleeves D, G, H, N, P (MR leg). Breakout/session: Sleeve C (swing breakout), Sleeve O (session ORB — dead). Carry/value (FX): **not tested — no interest-rate/OIS dataset available to compute carry**, named explicitly as a gap. Relative value/stat-arb: Sleeves F, G, H, N (pairs), PCA/Johansen basket screen (dead). Event/news trading: **out of scope, no news/calendar feed**. Volatility strategies: **partially** — Sleeve R's realized-vol regime filter; GARCH/implied-vol/VIX-style term-structure trades **out of scope, no options/VIX data**. Flow/microstructure (fixing flows, GEX, MOC imbalances, dealer gamma): **out of scope, no options/auction data**. Seasonality: Sleeve K (day-of-week), turn-of-month screen (dead). Price-action/market-structure (order blocks, FVGs, Wyckoff, Elliott wave): **deliberately not pursued** — explicitly flagged in this project as "need precise algorithmic definitions... before they count as quant signals," and most of this category is non-falsifiable/discretionary by construction. Grid/Martingale/news-sniping: **deliberately excluded** — flagged by the user's own list as high-risk/controversial, inconsistent with this project's fixed-fractional-risk mandate. |
| 4 | Cross-Asset & Intermarket Relationships | **Tested**: Copper/Gold ratio gate (Sleeve Q, inconclusive). DXY-proxy, yield-curve, VIX-equity, oil-CAD/NOK relationships: **out of scope** — no rates/yields/VIX dataset in this workspace. |
| 5 | Signals & Features | **Covered extensively** across all 18 sleeves: returns/MAs/RSI/ATR/Donchian/z-scores/ADX-style ATR-rank (price-based); CVD/taker-ratio/OI/L-S-ratio/funding/whale-index/liq-imbalance (orderflow — Sleeves A, B, E, I, J, R); realized-vol percentile (Sleeve R). Order-book L2/footprint, VPIN, implied-vol surfaces, CFTC COT, retail-positioning, news sentiment: **out of scope — no such data in this workspace** (footprint/orderbook ladder files exist in `Binance_Data` but were identified, not yet mined — see Section 4 below). |
| 6 | Mathematical & Statistical Foundations | **Covered**: stationarity/cointegration (Engle-Granger — Sleeves F/G/H/N/M-pairs-screen; Johansen — PCA basket screen), Hurst exponent (Sleeve P), OU half-life (Sleeves G/N), z-scores throughout. GARCH/HMM/Kalman filters/Hawkes/EVT/copulas/wavelets/fractional differentiation: **not implemented this phase** — acknowledged gap (see Section 4 below for why, given time budget). |
| 7 | Machine Learning | **Attempted, not completed this phase** — a walk-forward GradientBoosting classifier sleeve (per-quarter retrain on IS-only data, predicting forward-return sign from a rich crypto feature panel including the previously-unused `vwap_zscore`/`whale_index`/`liq_imbalance_ratio` columns) was scaffolded (`quant/ml_sleeve.py`) but de-prioritized mid-build in favor of the user's explicit Indices/FX/Metals taxonomy request — **honest, stated gap**, not a silent drop. |
| 8 | Regime Analysis | **Covered**: Hurst trend-vs-range switch (Sleeve P), realized-vol regime filter (Sleeve R — genuine improvement), Copper/Gold risk-on/off (Sleeve Q). HMM/Markov-switching formally, correlation-regime (DCC-GARCH), structural-break detection (CUSUM): **not implemented** — acknowledged gap. |
| 9 | Options & Volatility | **Out of scope entirely** — no options chain, implied-vol surface, or Greeks data exists anywhere in this workspace. Cannot be tested without acquiring an options data feed. |
| 10 | Portfolio Construction & Sizing | **Partially covered by mandate design, not varied as a research axis**: fixed-fractional risk ($12.5/trade) and ATR-based stop sizing are baked into every sleeve's engine by the mission invariants. Kelly, risk parity, Black-Litterman, HRP, Ledoit-Wolf shrinkage: **not tested** — these are capital-allocation-across-strategies techniques, out of this project's single-sleeve-at-a-time WFO scope by design. |
| 11 | Risk Management | **Covered by mandate**: ATR stops, ratchet, time-decay exit, $225 hard DD halt, 3-concurrent cap are enforced identically in every sleeve (A-R). VaR/Expected Shortfall/stress-testing against named historical scenarios: **not separately computed** (the 2020 COVID / 2022 Ukraine / negative-WTI shocks were however explicitly diagnosed as causal factors in Sleeve N's failure). |
| 12-13 | Execution, Microstructure, Market-Making/HFT | **Out of scope** — these require live L2 feeds, colocation/latency, and broker FIX integration; fundamentally untestable in an offline-bar-data sandbox regardless of effort spent. |
| 14 | Backtesting & Research Methodology | **Core strength of this entire project**: walk-forward IS-locked quarters, no-lookahead next-bar-open fills, realistic friction, concurrency/DD portfolio simulation, Buy&Hold benchmarking, and (this phase) a real data-quality defect caught and disclosed (Sleeve O's daily-mislabeled-as-intraday discovery) are all direct, practiced instances of this section. Deflated Sharpe Ratio / Probability-of-Backtest-Overfitting / combinatorial purged CV: **not formally computed** — acknowledged gap, flagged as a natural next step given the number of parameter grids searched across 18 sleeves (multiple-testing correction has not been formally applied to the project's overall sleeve count). |
| 15 | Performance Metrics | **Covered**: Sharpe, win rate, payoff ratio, max DD, ROI, avg-R reported for every sleeve. Sortino/Calmar/Ulcer Index/Omega/tail ratio: **not separately computed** — straightforward extension, not done this phase for time. |
| 16 | Data | **Covered, with a real finding**: this phase discovered `Forex_Data`'s nominal 15m/1h/4h files are actually daily-resolution pre-2023 (Sleeve O). Order-book L2/L3, options, COT, ETF flows, news/alt-data: **out of scope, not present in this workspace**. |
| 17 | Infrastructure & Tooling | **Out of scope for this research task** — this is a backtest research sprint, not a live-trading deployment; FIX/VPS/CI-CD/secrets-management are deployment concerns with no bearing on the honesty of the walk-forward results delivered. |
| 18 | Market Theory & Phenomena | **Implicitly tested throughout**: every clean kill (L, N, O, turn-of-month, PCA basket) is itself evidence consistent with market efficiency in this specific sample/frequency; every surviving edge (I, M-geom, R) is evidence of a specific, named inefficiency (crowd positioning extremes, cross-sectional metals momentum, vol-regime-conditional mean reversion). |
| 19-20 | Costs & Frictions, Regulation & Compliance | Costs: **core design axis of the entire project** (41bps/82bps mandated friction, explicit breakeven-friction calculations in every report). Regulation/compliance: **out of scope** — not a backtesting concern. |
| 21 | Common Pitfalls | **Actively avoided and, where found, explicitly reported**: look-ahead bias (next-bar-open enforced everywhere), data quality (Sleeve O's daily-mislabeled-intraday discovery), overfitting to one instrument (every sleeve uses a 5-10-symbol universe), curve-fitting (the IS-lock gate requires positive in-sample edge before any OOS trade, and "geometry search" variants are explicitly labeled diagnostic, never certified). |

---

## 4. Explicitly acknowledged gaps (stated, not hidden)

1. **No options/implied-vol data** anywhere in this workspace -> all of
   section 9 (Black-Scholes/SABR/Greeks/GEX/variance swaps) and the options-
   dependent parts of sections 3, 5, 12 are untestable without acquiring a
   new data feed.
2. **No interest-rate/OIS/yield data** -> FX carry trade, real-yield-vs-gold
   regression, yield-curve-vs-USD signals cannot be built from what's on
   disk.
3. **No L2 order-book, tick, or COT/positioning data** (beyond crypto's
   long/short-ratio and funding, already used in Sleeves I/J/R) -> VPIN,
   order-book imbalance, dealer gamma, and CFTC-based signals are out of
   reach.
4. **ML walk-forward classifier (Sleeve "O" in the original Phase-5 plan,
   renamed to avoid clashing with the session-ORB sleeve) was scaffolded
   but not completed or certified this round** — explicitly deprioritized
   in favor of the user's Indices/FX/Metals-focused list. This remains the
   single largest unexecuted item from the original Phase-5 plan and should
   be the first thing tackled if work continues.
5. **Binance footprint/order-flow ladder files** (`is_buy_imbalance`,
   `is_stacked_buy_imb`, POC/value-area) were identified in earlier
   exploration but never mined for a signal — stated gap, not attempted.
6. **GARCH, HMM/Markov-switching, Kalman-filter dynamic hedge ratios,
   CUSUM change-point detection, copulas, deflated Sharpe / PBO /
   combinatorial-purged-CV** were not implemented. The project substitutes
   simpler, equally legitimate proxies where time allowed (Hurst exponent
   instead of HMM for regime detection in Sleeve P; realized-vol percentile
   instead of GARCH-forecast vol in Sleeve R) but the more sophisticated
   versions of these techniques remain untested.

---

## 5. Updated project scorecard (all sleeves, A-R)

| Sleeve | Asset class | Verdict | Headline number |
|---|---|---|---|
| A-E | Crypto (various) | Dead / mixed | See `FINAL_REPORT.md` |
| F, G | FX stat-arb pairs | Thin edge, needs sub-5bps execution | See `STATARB_REPORT.md` |
| H | FX stat-arb (28-pair diversified) | Dead | See `STRATEGY_SEARCH_PHASE2.md` |
| **I** | Crypto L/S-ratio contrarian | **Best prior result** | Sharpe 1.276, ROI +61% @ 4bps, no-halt diagnostic |
| J | Crypto funding-momentum | Weak positive | See `STRATEGY_SEARCH_PHASE2.md` |
| K | FX day-of-week seasonality | Dead | See `STRATEGY_SEARCH_PHASE2.md` |
| L | Equity indices trend | Dead (clean) | ROI -4.7%, B&H +59.5% |
| M | Metals trend (geom-searched) | Marginal-but-genuine | Sharpe 1.06, ROI +4.7% |
| N | Energy WTI-Brent stat-arb | Dead (gross-negative pre-friction) | despite p=0.001 cointegration |
| O | FX+Metals session ORB | **Dead (clean, instant)** | 0 trades, IS gate never clears |
| P | Indices Hurst regime-switch | Dead, but gross ~breakeven | vs Sleeve L's gross -$175 |
| Q | Indices Copper/Gold gate | Dead/inconclusive | only 9 trades |
| **R** | Crypto L/S-ratio + vol-regime filter | **Best Phase-5 result — improves on I** | Sharpe 1.42 (+11%), MaxDD -23% vs I |

**Bottom line of Phase 5:** the exhaustive sweep did not discover a brand
new standalone edge among the tested categories (session breakout,
intermarket gating, PCA basket stat-arb, and calendar seasonality are all
dead in this sample) — but it did produce one genuinely valuable
improvement (Sleeve R) on the project's existing best strategy, and one
real, previously-unknown data-quality defect (Sleeve O's daily-mislabeled-
as-intraday Forex_Data discovery) that protects future work on this
dataset from a silent lookahead/granularity bug.
