# Audit — `s4_fvg_ml_strategy.py`

Reproduce with `.venv/bin/python run_s4_audit.py`.
Machine-readable output: [`reports/s4_audit.json`](../reports/s4_audit.json),
[`reports/s4_audit_waterfall.csv`](../reports/s4_audit_waterfall.csv).

> **What this audit does.** It re-runs S4's own feature engineering, S4's own
> labeller and S4's own stacked ensemble, keeps S4's own trade selection
> (`P ≥ 0.54`), and then changes **one accounting assumption at a time**. The
> trading idea is never modified. Every number below comes from S4's own code
> path.

---

## The reported result

```
[PORTFOLIO AGGREGATE SUMMARY - 20 OOS REGIMES]
  Total Trades Completed : 2,408
  Overall Win Rate       : 53.4%
  Total Realized Net R   : +1441.0R
  Profitable Regimes     : 19 / 20 (95.0%)
  Cumulative Additive PnL: +46,065.00 USD (+921.30% ROI on $5,000)
```

## The attribution waterfall

| variant | trades | mean R | total R | summed window ROI |
|---|---:|---:|---:|---:|
| **V0** as reported (label → +2R/−1R, 20 accounts) | 2,408 | +0.598 | **+1,441.0 R** | **+921.3 %** |
| **V1** + book the realised R | 2,408 | −0.076 | **−183.3 R** | −14.1 % |
| **V2** + realistic transaction costs | 2,408 | −0.342 | −823.6 R | −242.4 % |
| **V2b** … cost-viable instruments only (13/18) | 1,961 | −0.103 | −202.7 R | −23.3 % |
| **V3** + genuine 15m bars only | 2,423 | −0.407 | −987.0 R | −299.2 % |
| **V4** + one account, concurrency-capped | 935 | −0.519 | — | **−77.2 %** (Sharpe −4.46) |

**The entire +921 % is bookkeeping.** Not one of the four corrections involves
disagreeing with the trading idea.

---

## Defect 1 — PnL is booked from the binary label, not the simulated outcome

```python
# s4_fvg_ml_strategy.py:762
trade_r = np.where(y_sub == 1, 2.0, -1.0)
```

`y_sub` is the column `target`, defined 30 lines earlier as:

```python
# s4_fvg_ml_strategy.py:494
targets[i] = 1 if r_real > 0 else 0
r_reals[i]  = r_real          # <- the actual simulated R, computed and then ignored
```

So the script runs a careful bar-by-bar path simulation with a three-phase ratchet,
a time-decay rule and a 2.5 R target, stores the result in `r_realized` — and then
throws it away, replacing every outcome with ±2 R / −1 R based only on its sign.

On this run:

| | booked | realised |
|---|---:|---:|
| mean R | +0.598 | **−0.076** |
| total R | +1,441.0 | **−183.3** |
| mean R on "wins" | +2.000 | **+0.673** |
| mean R on "losses" | −1.000 | −0.930 |

A trade that gained +0.01 R is booked at +2.00 R. The difference —
**1,624 R of pure bookkeeping** — is the entire reported result.

There is a second, smaller inconsistency here: `MIN_R_MULTIPLE = 2.5`, so the
target is 2.5 R, but winners are booked at 2.0 R. The two numbers disagree about
what the strategy is.

*Fix:* `trade_r = sig_sub['r_realized'].to_numpy()`.

---

## Defect 2 — the win rate is multiplied by 100.2

```python
# s4_fvg_ml_strategy.py:813
print(f"  Overall Win Rate       : {tot_wins / max(1, tot_trades) * 100.2:.1f}%")
```

Reported 53.4 %; actual 53.28 %. Cosmetic on its own, but it is the kind of typo
that survives only when no one re-derives the summary from the trade blotter.

---

## Defect 3 — twenty separate \$5,000 accounts, summed as if they were one

```python
# s4_fvg_ml_strategy.py:768
cap = 5000.0
eq  = cap          # re-initialised inside the per-window loop
...
print(f"  Cumulative Additive PnL: {(w_df['Net_ROI%'].sum() / 100.0) * 5000.0:+,.2f} USD")
```

Each of the 20 windows starts from a fresh \$5,000 and its **percentage** return is
added to the others. A +122.7 % window and a +129.7 % window "add" to +252.4 % on a
\$5,000 base. Consequences:

- no capital is ever carried across windows, so compounding is neither applied nor
  modelled;
- **the maximum drawdown is never carried either** — the worst single-window DD is
  8.8 %, while the actual single-account path (V4) draws down **77.2 %**;
- the windows are contiguous (2023-09-15 → 2026-03-31), so there is no reason to
  treat them as separate accounts at all.

Within each window the simulation also compounds every signal sequentially,
`eq += r_val * risk`, as if trades never overlap. They overlap constantly: 18
instruments running a 15m rule produce clusters of ten-plus simultaneous positions.
Under a single account with realistic concurrency caps, only **935 of the 2,423
signals are actually takeable**.

---

## Defect 4 — the cost assumption, and the universe it is applied to

```python
# s4_fvg_ml_strategy.py:493
# Deduct 8 bps transaction friction (-0.08R)
r_real = round(r_real - 0.08, 4)
```

The comment conflates two different units: 8 bps of price and 0.08 R are only the
same thing if R happens to be 1 % of price. More importantly, the *actual* modelled
round-turn cost for S4's own instruments, computed from the dataset's own `spread`
column, is a median of **0.111 R** and a mean of **0.346 R**:

| asset | modelled cost per trade |
|---|---:|
| GAS | **1.440 R** |
| USDHKD | **1.093 R** |
| LEAD | **0.995 R** |
| EURHUF | 0.292 R |
| EURSEK | 0.291 R |
| NICKEL | 0.228 R |
| USDSEK | 0.147 R |
| EURCNH | 0.143 R |

For GAS, USDHKD and LEAD **the spread alone is wider than the entire stop distance**.
No 15-minute rule can trade them at any hit rate. USDHKD is additionally pegged
inside an HKMA band, where a "liquidity sweep" is not an economically meaningful
event.

This points at the deeper issue: `CANONICAL_18_ASSETS` — `EURHUF, GER40, NICKEL,
USDSEK, GAS, AU200, FR40, EURCNH, LEAD, NZDUSD, USDHKD, US2000, AUDCHF, NZDCNH,
XAUCNH, GAUCNH, EURSEK, EURUSD` — is a list of mostly *illiquid, wide-spread*
instruments. That is a strange universe to arrive at from first principles, and a
very natural one to arrive at by picking whatever scored best in-sample. Applying a
cost screen before looking at returns (`quantlab/universe.py`) retains **46 of 88**
candidate instruments and drops 12 of these 18.

---

## Defect 5 — the data the model is trained on

See [`DATA_INTEGRITY.md`](DATA_INTEGRITY.md). **8 of S4's 18 assets have no genuine
15-minute data at all when its first out-of-sample window opens on 2023-09-15**;
`GAUCNH`'s real 15m history starts 2025-01-23, after 17 of the 20 windows have
closed. Everything learned before those dates is learned from daily or hourly bars
being treated as 15-minute bars — so `hour` is always 0, the kill-zone filter is
degenerate, and a "3-bar FVG" spans three days.

Rebuilding the whole pipeline on clipped data (V3) does not rescue the result; it
makes it slightly worse, because the padded daily bars were contributing
artificially smooth, easy-looking setups.

---

## Defect 6 — the stacked ensemble is blended on its own training predictions

```python
# s4_fvg_ml_strategy.py:627-636
p_xgb = xgb_m.predict(xgb.DMatrix(X_train))      # in-sample
p_cb  = cb_m.predict_proba(X_train)[:, 1]        # in-sample
p_lgb = lgb_m.predict(X_train)                   # in-sample
S_train = np.column_stack([p_xgb, p_cb, p_lgb])
meta.fit(S_train, y_train)
```

Level-1 blenders must be fitted on **out-of-fold** base predictions. Fitted on
in-sample ones, the blender sees three nearly-perfect, nearly-identical predictors
and learns weights that do not transfer. This does not create the +921 %, but it
makes the `P ≥ 0.54` gate uncalibrated — the threshold means something different in
every window. `quantlab/model.py` uses a plain equal-weight average instead, which
has no such failure mode, and learns its gate as a quantile of the *training*
predictions.

---

## Smaller items

| | |
|---|---|
| `s4:711` | the 24-hour purge equals the 24-hour label horizon exactly, so a label starting one minute before the purge boundary still overlaps the test window. Purge should be `horizon + embargo`. |
| `s4:792` | `status = 'PASS' if tot_roi > 0 and mdd <= 4.50` — the 4.50 % threshold is applied to a drawdown that was computed on a per-window account that cannot drop below it by construction. |
| `s4:283` | `compute_hurst_exponent_fast` writes each computed value **backwards** over the preceding 3 bars (`hurst[max(0,i-3):i+1] = h`). The value at bar `i-3` is therefore derived from data up to bar `i`. This is a genuine 45-minute look-ahead in a feature. |
| `s4:256-258` | `hl = np.clip(hl, 1.0, 96.0)` then `np.nan_to_num(hl, nan=24.0)` — the fill value 24.0 sits inside the valid range, so "no estimate" and "half-life of 24 bars" are indistinguishable to the model. |
| naming | the OOS windows carry narrative labels ("August 2024 Global Carry Trade Flash Unwind") that are not used by any code. Harmless, but they create an impression of regime-conditioned validation that is not implemented. |

---

## What S4 gets right

It is worth being specific, because the structure is sound and the defects are all
in the accounting:

- entries are taken at `opens[i+1]`, never at the signal bar's close;
- the daily and 4h context are explicitly shifted (`shift(1)`) before the as-of join;
- the walk-forward is anchored and genuinely forward-only;
- the as-of joins use `strategy="backward"`, which is the correct direction;
- the ratchet/time-decay path simulation is bar-by-bar and conservative in ordering;
- the non-parametric feature block (Yang-Zhang, OU half-life, Hurst) is a reasonable
  microstructure feature set, the Hurst look-ahead aside.

**The idea was never tested.** Fix the booking line and S4 is a −183 R result, which
is roughly what the same event rule produces when tested cleanly in
[`S5_STRATEGY.md`](S5_STRATEGY.md) — and, more interestingly, roughly what a *random
entry* produces with the same barriers and costs.
