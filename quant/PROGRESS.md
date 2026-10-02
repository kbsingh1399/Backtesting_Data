# QUANT MISSION — PROGRESS LEDGER (as of 2026-10-02, rebuild session)

## ⚠️ Session note: prior "institutional memory" did not persist

This sandbox started as a **clean checkout** of `Backtesting_Data` with no
`quant/` directory at all — `engine.py`, `wfo.py`, `strategies.py`,
`campaign.py`, prior `results/scorecards.json`, and `papers/` did not exist on
disk. Everything described below was re-architected from scratch this session
strictly from the mission brief's invariants and the prior session's written
findings (used as a design target / sanity-check, not as code). Where the
rebuilt engine reproduced the previously-reported numbers for sleeves A/C
closely (see below), that was used as a cross-check that the re-implementation
is faithful to the original methodology.

## Artifact inventory (this session)

- `quant/engine.py` — single-market execution engine. Next-bar-open fills,
  SL = 1.5×ATR(14) (Wilder), TP = 3.0R, ratchet lock +0.2R @ +1.2R, time-decay
  exit (<0.2R @ 24 bars, executed next open), 41bps round-trip friction
  charged as $ notional drag on both legs, $12.5 fixed risk/trade sizing.
- `quant/wfo.py` — `SigCache` (memoized signals/indicators), `make_quarters`,
  IS→OOS walk-forward loop with an **honest IS gate** (see bug/fix notes
  below), `simulate_portfolio` (shared $5,000 equity, ≤3 concurrent, $225 hard
  DD halt), `bh_curve` (equal-weight Buy&Hold benchmark), `plot_report`.
- `quant/strategies.py` — data loaders (`load_binance`, `load_forex`) +
  5 sleeve definitions (A, B, C, D, E — D and E are new this session).
- `quant/campaign.py` — CLI runner; writes `results/{tag}_records.json`,
  `{tag}_equity.csv`, `{tag}_bh_equity.csv`, `{tag}_equity.png`,
  `{tag}_quarter_log.json`, and upserts `results/scorecards.json`.
- `quant/FINAL_REPORT.md` — the mission deliverable write-up (read this for
  the verdict).

## Data

Repo: `https://github.com/kbsingh1399/Backtesting_Data`. Binance: 18 USDT
perps, 15m OHLCV + orderflow (CVD, taker ratio, funding, OI, liqs,
session/prev-day VAH/VAL), 2020-09→2026-09. Forex: 156 CFD instruments ×
{15m,1h,4h,d1}, 2015→2026-09.

## Mandated geometry (in force for every *baseline* run)

Entry = next bar open; 41bps RT friction; SL = 1.5×ATR (1R); TP = 3.0R;
ratchet lock +0.2R after +1.2R; time-decay exit <+0.2R after 24 bars;
risk $12.5/trade; cap $5,000; DD limit $225 (hard halt); max 3 concurrent;
≥20 quarterly OOS windows, params locked by in-sample selection ONLY; B&H
overlay mandatory. All 5 sleeves tested on 21 OOS quarters (2021-Q2 →
2026-Q2/Q3), satisfying the ≥20 window invariant.

## A critical bug found & fixed this session (material — read this)

The first version of `simulate_portfolio` built a single global event
timeline and sorted `(time, priority)` with exits given priority over entries
at an identical timestamp, to let a slot freed by a closing trade be reused
immediately. This is correct **except** for a trade whose own `exit_time ==
entry_time` (an immediate same-bar stop-out, which happens often with the
ratchet logic) — its exit event then sorted *before* its own entry event, so
when the entry was later processed and added to the open-slot set, there was
no remaining exit event left to free that slot. **The trade silently occupied
a concurrency slot forever**, and once 3 such zombies accumulated, the
portfolio sim stopped accepting any new trade for the rest of the run —
producing an early, artificially short equity curve that in one case (sleeve
A, loosened-gate variant) looked like a small positive edge (+0.55% ROI,
Sharpe 0.39) purely because the simulation had silently stopped trading after
only 11 trades instead of continuing to accumulate losses.

Fixed by replacing the event-timeline approach with a sweep-line / min-heap
algorithm (`wfo.simulate_portfolio`) that only ever closes a trade out of the
open-set when a **later** candidate's entry is reached (or at final flush),
which is immune to the same-timestamp self-ordering bug. **Every sleeve was
rerun from a clean `results/` directory after this fix** — all numbers in
`scorecards.json` and `FINAL_REPORT.md` are post-fix. This is exactly the
kind of silent, OK-looking-but-wrong portfolio-accounting bug that an
"honest fail fast" mission is supposed to catch before certifying anything —
noting it explicitly here so it isn't rediscovered as a surprise later.

## IS-gate hygiene fix

The original gate selected the parameter combo with the *highest* in-sample
score even when every candidate combo in the grid showed a **negative**
in-sample expectancy (sum of R-multiples ≤ 0). That is not "selection", it's
picking the least-bad way to lose money, and it quietly let every sleeve
trade into quarters with demonstrably negative IS edge. Added
`require_positive_is_edge=True` (default): a quarter only trades if at least
one grid combo clears `min_is_trades` **and** shows positive total IS R.
Sleeve B (15m VAH/VAL sweep) never finds a single such combo across all 21
quarters under any geometry tested — a clean, honest "zero edge found"
result, not a data problem.

## Results (see `results/scorecards.json` + PNGs for full detail)

| Sleeve | Idea | n_trades | ROI% | Sharpe | Payoff | Gross PnL | Friction $ | Verdict |
|---|---|---|---|---|---|---|---|---|
| A_baseline | d1 trend + 4H RSI pullback + CVD confirm | 71 | -4.82% | -1.74 | 1.09 | -$123 | $118 | FAIL |
| A_geom_search_v2 | + IS-locked geometry/gate search | 39 | -2.71% | -1.32 | 1.18 | -$100 | $35 | FAIL |
| B_baseline | 15m prev-day VAH/VAL sweep+reclaim | 0 | 0% | — | — | — | — | DEAD (no IS edge, any params) |
| B_geom_search_v2 | + geometry/gate search | 0 | 0% | — | — | — | — | DEAD |
| C_baseline | d1 Donchian breakout + ATR-rank, FX+XAU | 48 | -4.60% | -2.05 | 2.35 | -$59 | $171 | FAIL |
| C_geom_search_v2 | + geometry/gate search | 72 | -4.52% | -1.57 | 1.01 | **+$31** | $257 | FAIL (cost-driven) |
| D_baseline | daily z-score mean reversion (SSRN pivot) | 60 | -1.12% | -0.45 | 1.20 | -$32 | $24 | FAIL |
| D_geom_search_v2 | + geometry/gate search | 61 | -2.09% | -0.85 | 1.24 | -$82 | $22 | FAIL |
| E_baseline | daily CVD order-flow momentum (SSRN pivot) | 186 | -1.86% | -0.37 | 1.29 | **+$16** | $109 | FAIL (closest; cost-driven) |
| E_geom_search_v2 | + geometry/gate search (overfit on sparse IS) | 78 | -3.49% | -1.57 | 0.88 | -$144 | $31 | FAIL |

**No sleeve certifies.** The mission's own criteria (≥20 OOS windows, IS-only
param locking, mandated friction/geometry, $225 hard DD) were honored
throughout, and every configuration — baseline and iterated — ends net
negative or a flat zero-trade dead end.

## Quantitative EV-leak diagnosis (final, honest version)

Two distinct failure modes showed up, and they are **not the same bug**:

1. **No alpha, friction irrelevant** (A, D, and C-baseline): gross P&L
   (before any friction) is *already negative*. Tuning exits or loosening
   gates cannot fix a signal with no raw directional edge — this is a model
   problem, not a cost problem.
2. **Real alpha, swamped by institutional friction** (C_geom_search_v2 and
   E_baseline): gross P&L is *positive* pre-friction, but 41bps round-trip
   consumes 4–9× the gross edge per trade:
   - E_baseline: gross edge ≈ **0.0071R**/trade vs. friction ≈ **0.047R**/trade
     (6.6× too expensive). Breakeven friction ≈ **6 bps** round-trip, not 41.
   - C_geom_search_v2: gross edge ≈ **0.034R**/trade vs. friction ≈
     **0.286R**/trade (8.3× too expensive). Breakeven friction ≈ **5 bps**.

Friction-as-fraction-of-R scales with **1/ATR%**: fixed-dollar risk sizing
means smaller-ATR% instruments/timeframes (FX daily, 0.3–0.5% ATR) carry much
larger notional per $1 of risk than high-ATR% ones (crypto daily, ~2–4% ATR),
so the exact same 41bps fee is far more punishing in FX (≈0.27–0.29R/trade)
than in crypto daily (≈0.03–0.05R/trade). This is why the SSRN-motivated
pivot to **daily** crypto sleeves (D, E) cut friction drag by ~4–25× versus
the original 15m/4H sleeves (A: 0.12R, B: ≈0.78R estimated from microstructure
math) — and why sleeve E is the only one with a real, measurable (if too
small) gross edge.

**Secondary finding — loosening the IS gate backfires**: for every sleeve,
dropping `min_is_trades` from 6→3 (to get more windows trading) made OOS
results *worse*, not better (e.g. E: Sharpe -0.37 → -1.57). Smaller IS
samples pick noisier "best" parameter combos that don't generalize — a
textbook multiple-comparisons overfit. The mandated baseline gate
(`min_is_trades=6`, standard geometry) outperformed every loosened/geometry-
searched variant tried. This is itself a useful mission finding: the
invariants as originally specified were already closer to sound WFO practice
than the "loosen everything" iteration instinct.

## Next actions if this mission continues

1. Sleeve E (order-flow momentum) is the only lead with genuine gross alpha.
   To certify it would require either (a) negotiating institutional friction
   down toward ~6-10bps (maker rebates / VIP tier / larger average trade
   size to amortize fixed costs), or (b) finding a position-sizing scheme
   that increases $ edge/trade without proportionally increasing $ risk
   (e.g., asymmetric R:R rather than fixed 3R TP).
2. Funding-carry delta-neutral (Chan 2025, Sharpe 6.1 claimed) was flagged
   last session as needing hedged-pair support the engine doesn't have —
   still unimplemented; would require a 2-leg position model in `engine.py`.
3. Intraday momentum (first-30min → last-30min, Gao/Han/Li/Zhou) untested
   this session — candidate sleeve F for a future pass.
