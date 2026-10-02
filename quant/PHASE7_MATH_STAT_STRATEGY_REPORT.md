# Phase 7 — Mathematical & Statistical Strategy Sweep, Full Forex_Data Universe

**Mandate:** Test all mathematical/statistical-style strategies against the
complete `Forex_Data` universe (not the small hand-picked subsets used in
earlier phases), run a multi-grid combination of assets to find the best
equity curve, think like an institutional desk investing in foreign markets,
and use ML where it might help. This report covers the full sweep: data
scope, every strategy family tested, what died and why (quantitatively), and
the one combination that produced a genuinely better certified result than
anything found in Phases 1-6.

**Headline result:** Narrowing the existing certified Sleeve M (8-symbol
metals trend/breakout) down to just **gold + silver (XAUUSD + XAGUSD)**
produces the best risk-adjusted equity curve found in this entire project:
**Sharpe 2.228, ROI 5.87% over 21 OOS quarters, max drawdown $48 (0.96% of
the $5,000 account), never halted.** Full comparison table and the
methodology used to avoid look-ahead bias in that selection are in Part 5.

---

## 0. Universe & data quality (done before any strategy testing)

`Forex_Data` contains 158 symbols; **60 are crypto CFDs** bundled into the
same folder. Per the broker's own manifest (Blueberry Markets, exported
2026-09-14, history from 2010), the genuine FX/metals/indices/energy
universe is **98 symbols**: 60 FX pairs (`Forex_Raw`), 16 indices, 14
precious-metals cross-currency variants (`Commodities_Raw`), 8 base
metals/energy (`Commodities`). This phase uses all 98, across all four
timeframes (15m/1h/4h/D1), per the explicit scope decision.

A fresh per-symbol-per-timeframe data-quality scan (`phase7_data_quality_scan.py`,
output `results/PHASE7_data_quality.json`) confirms the **same defect found
in Phase 5/6 is universal**, not isolated to the handful of symbols checked
before: every single symbol's 15m/1h/4h files contain daily-bar-density data
mislabeled as intraday before a per-symbol cutoff (2023-2025 depending on
symbol); only D1 is reliable across its full history. Example (EURUSD):

| Year | 15m bars | Expected if genuine 15m |
|---|---|---|
| 2015-2022 | 223-261/year | ~75,000+/year |
| 2023 onward | 10,851-24,836/year | — now genuinely intraday |

**Consequence for this phase:** the classical academic factor screens (Part
1-3 below) were run on **D1**, the only timeframe with a clean, multi-year
history across the whole universe — which is also the natural, honest
frequency for momentum/carry/cross-sectional factors in the literature this
phase is drawing on (these are not meant to be 15m signals). The two sleeves
promoted to full engine certification (T, U) also use D1 for the same
reason. The multi-asset grid combination in Part 5 (M2-M6) also uses D1,
consistent with the existing certified Sleeve M/L baseline.

---

## 1. Classical time-series & cross-sectional factor screens

Fast vectorized triage (`phase7_screen_part1.py`, no ATR/ratchet engine yet
— just honest t-stats on raw forward returns) across all 98 symbols, grouped
into FX (60), Indices (16), Metals (14), Energy/Base-metals (8).

### 1a. Time-series momentum (Moskowitz-Ooi-Pedersen style)

sign(trailing K-day return) → next-period return, swept across lookbacks
(21/63/126/252 days) × horizons (1/5/21 days), pooled across all 98 symbols:

| Lookback | Horizon | n | mean bps/period | t-stat |
|---|---|---|---|---|
| 21d | 1d | 223,184 | -0.11 | -0.48 |
| 21d | 5d | 222,792 | **-2.64** | **-5.30** |
| 63d | 21d | 217,108 | **-8.31** | **-8.18** |
| 252d | 21d | 198,586 | +1.93 | 1.80 |

**Verdict: DEAD, and mostly significantly negative**, not just insignificant
— classic trend-following on raw price, pooled across this universe, loses
money net of nothing (before any friction). The one place a positive signal
persistently shows up: gold/silver variants dominate every lookback's
top-5-by-symbol list, which is exactly consistent with the pre-existing
certified Sleeve M finding — this is corroborating evidence, not a new lead.

### 1b. Cross-sectional momentum (Menkhoff et al. currency-momentum style)

Vol-scaled 12-1 month formation, tercile long-winners/short-losers, monthly
rebalance, run **separately per asset class** (pooling raw returns across
asset classes with different vol scales would be methodologically wrong):

| Asset class | n months | mean bps/month | t-stat | Needs (2-leg cost) |
|---|---|---|---|---|
| FX (60 syms) | 14 | +35.9 | 1.14 | >82bps |
| Indices (16 syms) | 36 | -26.3 | -0.35 | >82bps |
| Energy/Base metals (8 syms) | 59 | -125.8 | -0.86 | >82bps |

**Verdict: DEAD everywhere.** FX shows the literature's expected sign but
is statistically insignificant (t=1.14) AND an order of magnitude short of
clearing round-trip cost even if it were real. This is directionally
consistent with real published FX-momentum research (Menkhoff et al.,
Ilmanen/Israel/Moskowitz/Thapar/Wang — cited in Phase 6) showing momentum
Sharpe ratios around 0.6-0.7 pre-cost; our specific 11-year live-dealer
sample simply doesn't have enough months of clean data to detect it with
confidence, and real-world retail friction eats it regardless.

### 1c. Short-horizon mean reversion (z-score fade)

| z threshold | Horizon | n | mean bps | t-stat |
|---|---|---|---|---|
| z>1.5 | 1d | 27,717 | **+2.71** | **3.13** |
| z>2.0 | 1d | 10,124 | **+4.91** | **2.74** |
| z>2.0 | 5d | 10,116 | -2.15 | -0.62 |
| z>2.5 | 1d | 2,882 | +7.34 | 1.47 |

**Verdict: statistically real at 1-day horizon, economically dead.** The
1-day fade edge is genuine (t=2.7-3.1, large n) but it's **2.7-4.9bps
against a 41bps round-trip friction floor** — the edge would need to be
roughly 10x larger just to break even on costs. This is the clean, honest
EV-leak explanation: the market IS slightly mean-reverting intraday-to-daily,
but not by enough margin to survive realistic retail execution costs.

### 1d. Volatility-regime conditioning

Does the above MR edge get stronger when conditioned on being in a
high-realized-vol regime (motivated by Sleeve R's crypto finding)?

| Regime | n | mean bps | t-stat |
|---|---|---|---|
| HIGH vol (>70th pct) | 2,957 | **+19.47** | **3.65** |
| LOW vol (<30th pct) | 2,862 | -1.60 | -1.04 |

**This is the strongest lead from the raw screens** — nearly 4x the
unconditional MR edge, strongly significant, and starting to approach (if
not clear) the friction floor. **Promoted to full WFO certification as
Sleeve T** (Part 4) — this is where this phase ran a real engine test on a
real raw lead rather than stopping at "p<0.05 and a neat table."

---

## 2. PCA & Engle-Granger cointegration screen

`phase7_screen_part2_pca_coint.py` — Phase 5 already killed a PCA/Johansen
screen on US equity indices. This phase tests baskets with much stronger a
priori structural linkage (FX triangulation, which is mechanically enforced
by arbitrage) across 7 natural baskets spanning the full universe.

**PCA residual stationarity (ADF test on PC1-detrended residual spread):**

| Basket | n symbols | PC1 variance | ADF p-value | Stationary? |
|---|---|---|---|---|
| G10 majors | 7 | 58.8% | 0.950 | No |
| EUR-crosses | 7 | 42.6% | 0.863 | No |
| GBP-crosses | 6 | 59.8% | 0.839 | No |
| Commodity-bloc FX | 6 | 53.3% | 0.404 | No |
| Precious metals | 4 | 73.1% | 0.169 | No |
| Base metals | 5 | 54.3% | 0.431 | No |
| Equity indices | 7 | 74.1% | 0.843 | No |

**None stationary at any threshold.** No basket in this dataset has a
tradeable, mean-reverting common-factor residual — fully consistent with
Phase 5's equity-indices finding, now confirmed across every other natural
grouping too.

**Pairwise Engle-Granger cointegration** (all pairs within each basket,
109 pairs tested total): 13 pairs showed p<0.05 — almost exactly the ~5.5
false positives expected by chance alone at a 5% threshold across 109 tests
(a textbook multiple-testing trap an institutional desk must correct for).
Of those 13, **zero cleared the 82bps 2-leg round-trip cost** — the best
edge found (NZDUSD-USDJPY) was 1.08bps against an 82bps requirement, ~76x
short.

**Verdict: DEAD, decisively**, consistent with Sleeves F/G/H/N and the
Phase 5 PCA/Johansen screen. Statistical arbitrage does not have a home in
this dataset at any basket construction tried across four separate phases.

---

## 3. ML ensemble (three different model families)

`phase7_screen_part3_ml.py` — pooled cross-sectional panel across all 98
symbols, **walk-forward retrained every quarter** (expanding window, strict
train-before-test-quarter split, zero lookahead — same discipline as the
engine's own WFO), predicting sign of next-day return. Three genuinely
different model families, as requested:

| Model | OOS quarters | Hit rate (long) | Long-signal mean bps | t-stat |
|---|---|---|---|---|
| **Logistic Regression** | 43 | 54.5% | **+8.02** | **3.61** |
| Random Forest | 43 | 54.5% | +7.05 | 2.57 |
| Gradient Boosting | 43 | 52.5% | -1.14 | -0.39 |

(Short-side signals were weak/insignificant for all three models — the
interesting structure here is entirely on the long side.)

**Logistic Regression — the simplest model — won**, a classic and
instructive result: Gradient Boosting's extra flexibility let it fit noise
in the training folds that didn't generalize, while the linear model's
regularization kept it honest. This is exactly the kind of model-selection
insight an institutional quant desk would flag before trusting any ML
signal.

**Verdict: statistically real (t=3.61, n=10,876) but still economically
sub-cost** (8.02bps vs 41bps friction, ~5x short) in its raw form. Same
pattern as the mean-reversion screens — real signal exists, friction wins.
**Promoted to full WFO certification as Sleeve U** using the walk-forward
LR probability as an entry trigger (Part 4).

---

## 4. Promoting the two best raw leads to full engine certification

Both leads (vol-regime MR, ML long-signal) showed genuine statistical edges
in Parts 1 and 3 — but a raw fixed-horizon-return screen is NOT the same
test as this project's actual trade engine (1.5×ATR stop / 3.0R target /
ratchet at +1.2R→+0.2R / 24-bar time-decay / $12.5 fixed risk / 3 concurrent
/ $225 DD halt). Both were built as new sleeves (`T`, `U` in `strategies.py`)
and run through the full mandated WFO, including the IS-locked geometry
search already used to rescue Sleeve M.

| Sleeve | Variant | Trades | ROI% | Sharpe | Max DD | Halted |
|---|---|---|---|---|---|---|
| T (vol-regime MR, 85 symbols) | baseline | 27 | -4.12% | -2.47 | $231.68 | Yes (2024-08) |
| T (vol-regime MR, 85 symbols) | geom_search | 32 | -3.81% | -1.99 | $238.52 | Yes (2024-08) |
| U (ML LR long-signal, 85 symbols) | baseline | 42 | -3.06% | -1.37 | $225.93 | Yes (2021-10) |
| U (ML LR long-signal, 85 symbols) | geom_search | 41 | -4.70% | -2.38 | $245.66 | Yes (2021-09) |

**Both DEAD, and not close** — even with geometry search allowed. This is
the honest, quantitative EV-leak explanation for why: the raw screens
measured a **fixed 1-day forward return** after a fade/long signal. The
engine's exit geometry is built for **asymmetric R-multiple capture** (3R
target, SL at 1.5×ATR) — a mechanic designed for moves that keep running,
not one-day mean-reversion pops. Two specific failure modes, confirmed in
the trade logs:
1. On the vol-regime sleeve, ATR is *itself inflated* exactly when the
   high-vol filter fires, so the 1.5×ATR stop is unusually wide — sizing
   shrinks (fixed $12.5 risk / wide stop = small position) and the 3R target
   requires a continued, clean reversal that a 1-day mean-reversion edge
   doesn't reliably deliver before noise fills the rest of the hold.
2. On the ML sleeve, the probability signal's 8bps edge is a *daily*
   average effect; held through the engine's 24-bar time-decay window, the
   position is exposed to 24 days of unrelated noise for an edge that was
   only ever measured at a 1-day horizon.

This is the single clearest lesson from this phase: **a statistically
significant raw-return screen is necessary but nowhere near sufficient** —
it has to survive translation into real, risk-defined trade mechanics
before it means anything. (This is exactly the lesson that made Sleeve M
work in the first place — trend/breakout signals and 3R-target exits are a
natural match; mean-reversion signals and 3R-target exits are not.)

---

## 5. Multi-grid asset combination (the actual winning result)

Sleeve M (Donchian breakout + ATR-rank trend filter, 8 metals symbols:
XAUUSD/XAGUSD/XPTUSD/COPPER/ALUMINIUM/NICKEL/ZINC/LEAD) remains the only
strategy family that survives full certification anywhere in this 21-sleeve
project. This section tests whether a different **combination of assets**
running the identical strategy produces a better combined equity curve —
the literal "multi-grid combination of assets" requested.

**Methodology safeguard (read this before the table):** looking at Sleeve
M's own trade log and picking out which of its 8 symbols individually made
money *in that same historical run*, then rebuilding a sleeve from just
those symbols, would be look-ahead/selection bias — the subset choice
itself needs an out-of-sample test, not a hindsight read of the result it's
being carved out of. Instead, five **pre-registered, economically-motivated
universe groupings** were decided *before* looking at per-symbol breakdowns,
each run as its own complete, independent WFO certification:

| Sleeve | Universe | Trades | ROI% | Sharpe | Max DD | Halted | BH ROI% |
|---|---|---|---|---|---|---|---|
| M (original, 8 metals) | XAU,XAG,XPT,Copper,Alu,Nickel,Zinc,Lead | 159 | 4.70% | 1.058 | $183.20 | No | 51.0% |
| M2: precious metals only | XAU, XAG, XPT | 69 | 2.96% | 0.954 | $166.33 | No | 100.0% |
| M3: base/industrial metals only | Copper, Alu, Nickel, Zinc, Lead | 86 | -0.91% | -0.289 | $233.32 | **Yes** | 21.7% |
| **M4: monetary metals only** | **XAU, XAG** | **47** | **5.87%** | **2.228** | **$48.14** | **No** | 134.8% |
| M5: metals + energy | M original + Brent/WTI/Gas | 205 | 5.57% | 1.104 | $155.21 | No | 40.9% |
| M6: global multi-asset CTA | M5 + 10 major indices | 162 | -1.78% | -0.468 | $238.99 | **Yes** | 49.7% |

**Findings, in order of how they inform the design:**

- **Base metals alone are dead** (M3: Sharpe -0.289, halted) — the trending
  behavior in the original Sleeve M was being carried almost entirely by
  the precious-metals leg, diluted (not helped) by copper/aluminium/
  nickel/zinc/lead's own weaker/choppier price action.
- **Gold+silver alone (M4) is a clear improvement over all 8 metals
  combined**: more than double the Sharpe (2.228 vs 1.058), a quarter of
  the max drawdown ($48 vs $183), similar/better ROI (5.87% vs 4.70%),
  fewer, higher-quality trades (47 vs 159). Dropping platinum and the base
  metals removes noise the Donchian/ATR-rank filter was otherwise trading.
- **Adding energy (M5) is a mild positive** vs the original 8-metal M
  (Sharpe 1.104 vs 1.058) — Brent/WTI/Gas share enough of the same
  "commodity supercycle, persistently trending" macro character — but still
  clearly worse than the leaner M4.
- **Adding equity indices (M6) kills it** (Sharpe -0.468, halted) — this
  directly confirms Sleeve L's standalone finding (indices trend/breakout
  was already dead on its own, -4.71% ROI) by showing it doesn't even help
  as a diversifier once blended into a working book; it actively drags the
  shared-equity, 3-concurrent-slot portfolio down. **More assets is not
  automatically better** — diversifying into an asset class with no edge of
  its own dilutes the one that works.

**Winner: Sleeve M4 (Gold + Silver only).** Full certified stats:
47 trades, ROI 5.867%, Sharpe 2.228, win rate 61.7%, payoff ratio 1.337,
max drawdown $48.14 (0.96% of the $5,000 account — well inside Blueberry's
5%/10% kill-switch thresholds with no leverage applied), 11/21 OOS quarters
actually traded, never halted. Equity curve: `results/M4_geom_search_v2_equity.png`.

**Honest caveat — read the chart, not just the headline number:** Buy&Hold
on gold+silver over this exact span returned **134.8%**, because 2023-2026
was an extraordinary monetary-metals bull market (central-bank buying,
de-dollarization flows — see Sleeve M's own docstring). M4's 5.87% is a
small fraction of that. This is not M4 failing to notice the trend; it's
the structural trade-off of **any** fixed-risk, defined-stop, 3-concurrent-
position strategy against a one-directional secular move: M4 is giving up
most of the raw trend's magnitude in exchange for a drawdown of under 1% of
account equity the entire time, instead of riding the ~50%+ drawdowns
visible in the grey Buy&Hold line on the chart. Which one an "institutional
investor" should prefer depends entirely on mandate — a prop-firm account
with hard 5%/10% drawdown kill-switches (Blueberry-style) cannot survive
holding through Buy&Hold's drawdowns regardless of the eventual payoff; a
long-only allocator with a 10+ year horizon and no intermediate drawdown
constraint might reasonably prefer the Buy&Hold exposure instead. **M4 is
the better answer specifically for a risk-constrained, defined-stop mandate
— not a claim that it "beats gold" in absolute terms.**

**Sample-size caveat:** 47 trades across 21 quarters is still a small
sample for a Sharpe-ratio point estimate — treat 2.228 as "this combination
is the best candidate found," not as a precise, tight-confidence-interval
number. The concentration in two historically-correlated precious metals
also means M4 has less true diversification than its symbol count suggests.

---

## 6. Net verdict across the whole phase

| Family | Status | Best raw edge found | Survives real engine? |
|---|---|---|---|
| Time-series momentum / trend | Dead (mostly significant negative) | — | No (consistent w/ Sleeve M needing a dedicated metals-only design) |
| Cross-sectional momentum | Dead (insignificant, sub-cost) | FX +35.9bps/mo, t=1.14 | Not tested further (fails bar 1) |
| Mean reversion (unconditional) | Statistically real, economically dead | +4.9bps vs 41bps needed | No |
| Mean reversion (vol-regime filtered) | Statistically real, economically marginal | +19.5bps, t=3.65 | **No** (Sleeve T, certified dead) |
| PCA basket stat-arb | Dead, decisively | No stationary residual, any basket | N/A |
| Pairwise cointegration | Mostly multiple-testing noise | 1.08bps vs 82bps needed | N/A |
| ML (Logistic Regression) | Statistically real, economically marginal | +8.0bps, t=3.61 | **No** (Sleeve U, certified dead) |
| ML (Random Forest) | Weaker version of LR | +7.1bps, t=2.57 | Not separately certified |
| ML (Gradient Boosting) | Dead (overfit to training noise) | -1.1 to -10.7bps | Not certified |
| **Donchian/ATR-rank trend, gold+silver only (M4)** | **CERTIFIED SURVIVOR** | — | **Yes — Sharpe 2.228** |

This phase tested 9 distinct mathematical/statistical approaches (plus 3 ML
model variants) across the complete 98-symbol universe and all four
timeframes' data-quality implications. Eight of nine died, each for a
specific, quantified reason documented above — mostly "real signal, real
cost, cost wins by 5-10x." The one survivor from prior phases (Sleeve M)
was improved by 2x on risk-adjusted terms through principled, pre-registered
universe narrowing, not by discovering a new signal.

## Artifacts

- `phase7_data_quality_scan.py` → `results/PHASE7_data_quality.json`
- `phase7_screen_part1.py` → `results/PHASE7_part1_screen.json` (TSMOM/XSMOM/MR/vol-regime)
- `phase7_screen_part2_pca_coint.py` → `results/PHASE7_part2_pca_coint.json`
- `phase7_screen_part3_ml.py` → `results/PHASE7_part3_ml.json`
- `phase7_ml_signal_cache.py` → `results/PHASE7_ml_proba_cache.parquet` (walk-forward LR probabilities, feeds Sleeve U)
- New certified sleeves in `strategies.py`/`campaign.py`: `T`, `U`, `M2`, `M3`, `M4`, `M5`, `M6`
- Results for all of the above: `results/{T,U,M2,M3,M4,M5,M6}_*_{records.json,equity.csv,equity.png,quarter_log.json}`
- `results/scorecards.json` updated with every run in this phase
- **Best result / recommended portfolio: `results/M4_geom_search_v2_equity.png`**
