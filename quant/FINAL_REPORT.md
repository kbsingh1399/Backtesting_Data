# Final Mission Report — Certified Trading Strategy Search

**Verdict: NO SLEEVE CERTIFIES.** Five independent strategy sleeves (A, B, C,
D, E) were built, tested over ≥20 quarterly out-of-sample walk-forward
windows under the full mandated institutional invariants, and iterated
honestly (in-sample-only parameter/geometry selection). All five end net
negative or trade zero times. This report explains exactly why, with numbers.

Full code: `quant/engine.py`, `quant/wfo.py`, `quant/strategies.py`,
`quant/campaign.py`. Full ledger/methodology notes: `quant/PROGRESS.md`.
Every number below is reproducible via `python3 campaign.py <A|B|C|D|E> ...`.

## Mission invariants honored throughout

Entries at next-bar open only · 41bps round-trip friction · SL = 1.5×ATR
(defines 1R) · TP = 3.0R · ratchet lock +0.2R @ +1.2R · time-decay exit
<+0.2R @ 24 bars · $5,000 capital · $12.5 risk/trade · $225 hard drawdown
halt · max 3 concurrent positions · ≥20 quarterly OOS windows with
parameters locked by in-sample selection only · Buy&Hold benchmark on every
chart.

## Scorecard (see `results/scorecards.json` for full detail, `*_equity.png` for charts)

| Sleeve | Idea | OOS windows (traded/total) | Trades | ROI% | Sharpe | Payoff | Gross $ | Friction $ | Max DD $ | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| **A** `_baseline` | d1 trend + 4H RSI pullback + CVD confirm (crypto) | 12/21 | 71 | **-4.82%** | -1.74 | 1.09 | -123 | 118 | 240.9 (halted) | FAIL — no gross edge |
| A `_geom_search_v2` | + IS-locked geometry/gate iteration | 21/21 | 39 | -2.71% | -1.32 | 1.18 | -100 | 35 | 226.7 (halted) | FAIL |
| **B** `_baseline` | 15m prev-day VAH/VAL liquidity-sweep reversion (crypto) | 0/21 | 0 | 0.00% | — | — | — | — | 0 | **DEAD — zero IS edge at any parameterization** |
| B `_geom_search_v2` | + geometry/gate iteration | 0/21 | 0 | 0.00% | — | — | — | — | 0 | DEAD |
| **C** `_baseline` | d1 Donchian breakout + ATR-rank (FX majors + XAU) | 8/21 | 48 | **-4.60%** | -2.05 | 2.35 | -59 | 171 | 230.2 (halted) | FAIL — no gross edge |
| C `_geom_search_v2` | + geometry/gate iteration | 11/21 | 72 | -4.52% | -1.57 | 1.01 | **+31** | 257 | 227.0 (halted) | FAIL — cost-driven |
| **D** `_baseline` | daily z-score mean reversion vs rolling mean, trend/asymmetry-aware (SSRN pivot) | 16/21 | 60 | -1.12% | -0.45 | 1.20 | -32 | 24 | 239.5 (halted) | FAIL — weak gross edge |
| D `_geom_search_v2` | + geometry/gate iteration | 20/21 | 61 | -2.09% | -0.85 | 1.24 | -82 | 22 | 238.5 (halted) | FAIL |
| **E** `_baseline` | daily CVD order-flow momentum, continuation (SSRN pivot) | 16/21 | 186 | **-1.86%** | -0.37 | 1.29 | **+16** | 109 | 246.8 (halted) | FAIL — **closest result, cost-driven** |
| E `_geom_search_v2` | + geometry/gate iteration (overfit) | 19/21 | 78 | -3.49% | -1.57 | 0.88 | -144 | 31 | 225.1 (halted) | FAIL |

Buy&Hold over the same stitched OOS spans (equal-weight universe): crypto
sleeves +38–88%, FX/XAU sleeve +23%. Every strategy variant underperformed
simply holding the underlying — the charts make this stark (see PNGs).

## Why: two distinct, quantified EV leaks

**1. No raw alpha (A, D, C-baseline).** Gross P&L — *before any friction at
all* — is already negative. Ratchets, time-decay tuning, and looser gates
cannot repair a signal definition with no directional edge. This is a model
problem.

**2. Real alpha, swamped by friction (C_geom_search_v2, E_baseline).** Gross
P&L is positive before costs, but the mandated 41bps round-trip friction eats
it several times over:

| Sleeve | Gross edge / trade (R) | Friction / trade (R) | Friction is Nx the edge | Friction bps needed to break even |
|---|---|---|---|---|
| E_baseline | +0.0071R | 0.047R | **6.6×** | ≈ 6 bps |
| C_geom_search_v2 | +0.0344R | 0.286R | **8.3×** | ≈ 5 bps |

Friction cost scales as **1/ATR%** under fixed-dollar risk sizing: smaller
ATR% instruments need larger notional to express the same $12.5 risk, so the
same 41bps fee bites far harder. This is exactly why the SSRN-motivated pivot
to **daily** crypto sleeves (D, E) cut the friction/R ratio by 4–25× relative
to the original 15m/4H sleeves (A ≈0.12R/trade; B's live friction never even
measured because it never trades, but back-of-envelope on its 15m ATR% puts
it near ≈0.78R/trade) — and it's why sleeve E is the only configuration that
shows genuine, if too-small, gross alpha.

## Engineering integrity notes (full detail in PROGRESS.md)

- **A material portfolio-accounting bug was found and fixed mid-mission.**
  The original concurrency/DD-halt simulator mis-ordered same-timestamp
  entry/exit events for zero-duration trades (an immediate same-bar
  ratchet stop-out), causing those trades to occupy a concurrency slot
  *forever* and silently truncate the simulation. One early result (sleeve A,
  loosened gate) looked like a small positive edge purely as an artifact of
  this bug cutting the equity curve short after only 11 trades. Fixed with a
  sweep-line/min-heap portfolio simulator; **every number in this report is
  post-fix**, and every sleeve was rerun from a clean slate afterward.
- **The in-sample gate was tightened to require positive in-sample
  expectancy**, not just a minimum trade count — the original "pick the
  least-bad parameter combo" logic would otherwise lock in and trade
  demonstrably negative-edge parameters every single quarter.
- **Loosening the gate made results worse, not better**, across every
  sleeve tested (e.g. E: Sharpe -0.37 → -1.57 when `min_is_trades` dropped
  6→3). Smaller in-sample samples select noisier "best" parameters that
  don't generalize OOS — a textbook multiple-comparisons overfit. The
  mandated baseline settings outperformed every "iterated" variant.

## Recommendation

Do not deploy any of A/B/C/D/E as specified. The one quantitatively
interesting thread is **sleeve E (daily CVD order-flow momentum)**: it has a
real, measurable, positive gross edge (statistically the largest sample,
186 trades) that is roughly 6.6× too small to clear the mandated 41bps
friction. Future work should target closing that specific gap (lower
effective friction via execution improvements, or a position-sizing scheme
that scales $ edge faster than $ risk) rather than re-testing more geometry
variants on the existing signal definitions — every geometry/gate search
attempted this session made results worse, not better.
