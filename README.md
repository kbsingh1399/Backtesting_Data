# Backtesting_Data

Institutional quantitative backtesting datasets for multi-asset algorithmic trading
(Forex, CFDs, and Binance USDT-M perpetuals) — plus `quantlab`, a research stack
built on top of them.

---

## ⚠️ Read this first

The `Forex_Data/*_15m_real.parquet` files are **not pure 15-minute series**. 142 of
156 begin with a block of daily or hourly bars carrying 15m filenames — 3 % of rows
but **60 % of the calendar span**. Anything computed over a whole file crosses a
resolution boundary silently.

Use `quantlab.datafeed.load_bars(symbol, "15m", enforce_pure=True)`, which detects
and clips the boundary. Full detail: **[docs/DATA_INTEGRITY.md](docs/DATA_INTEGRITY.md)**.

---

## 📁 Repository structure

### Data

| path | contents |
|---|---|
| `Forex_Data/` | 156 instruments × 4 timeframes (`15m`, `1h`, `4h`, `d1`), 626 parquet files. FX majors/crosses, equity indices, metals, energy, crypto CFDs. Includes a per-bar broker `spread` column in points — the most valuable field in the dataset. |
| `Binance_Data/` | 18 USDT-M perpetuals: 15m master bars with funding rates, plus verified tick footprint orderbook ladders. |
| `ssrn_pdfs/`, `ssrn_trading_strategies_filtered.csv` | Reference literature. |

### `quantlab/` — research stack

| module | role |
|---|---|
| `config.py` | universe definition, cost assumptions, frozen strategy parameters — one file, with the rationale next to each number |
| `datafeed.py` | loading + **resolution-integrity screening** |
| `costs.py` | transaction costs built from the dataset's own `spread` column |
| `universe.py` | integrity → cost → correlation-cluster screens |
| `features.py` | strictly causal 15m features; all higher-timeframe context resampled from the 15m series itself and shifted |
| `signals.py` | liquidity-sweep event detector (PD / Asian / prev-week / swing pools) |
| `labeling.py` | gap-aware triple-barrier simulation + AFML uniqueness weights |
| `controls.py` | **matched random-entry null models** |
| `model.py` | purged, embargoed, anchored walk-forward meta-labelling ensemble |
| `portfolio.py` | event-driven portfolio with concurrency and cluster caps |
| `metrics.py` | Sharpe/Sortino/Calmar + **Deflated Sharpe Ratio**, PSR, stationary-bootstrap CIs |
| `reporting.py` | tearsheets and figures |

### Entry points

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/python run_data_audit.py        # data integrity report
.venv/bin/python research/build_cache.py  # one-off feature cache (~35s, 640MB, gitignored)
.venv/bin/python run_s5_strategy.py       # S5 strategy, out-of-sample
.venv/bin/python run_s4_audit.py          # audit of s4_fvg_ml_strategy.py
```

Research scripts (in dependency order, all dev-block only):

```bash
.venv/bin/python research/dev_search.py           # in-sample parameter search (31 trials)
.venv/bin/python research/diagnostics.py          # rule vs matched random controls
.venv/bin/python research/alpha_screen.py         # 100-config cost-aware alpha screen
.venv/bin/python research/reversal_robustness.py  # execution-lag / stale-quote / cost stress
.venv/bin/python research/lagged_screen.py        # re-screen with lag built in
.venv/bin/python research/vwap_robustness.py      # hostile test of the one survivor
.venv/bin/python research/make_figures.py
```

### Documentation

| doc | what it covers |
|---|---|
| **[docs/DATA_INTEGRITY.md](docs/DATA_INTEGRITY.md)** | the 15m-files-are-not-15m problem, the spread column, what's clean |
| **[docs/S5_STRATEGY.md](docs/S5_STRATEGY.md)** | the strategy: hypothesis, construction, out-of-sample result |
| **[docs/S4_AUDIT.md](docs/S4_AUDIT.md)** | line-by-line audit of `s4_fvg_ml_strategy.py`'s +921 % result |
| **[docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md)** | everything tried, in order, including what failed |

---

## Headline findings

**1. The reference strategy's +921 % is bookkeeping.**
`s4_fvg_ml_strategy.py` runs a careful bar-by-bar path simulation, stores the result
in `r_realized`, then books PnL as `np.where(label == 1, 2.0, -1.0)` — every winner
at +2 R regardless of size. Its winners actually average +0.673 R. Correcting only
that line turns +1,441 R into **−183 R**. → [docs/S4_AUDIT.md](docs/S4_AUDIT.md)

**2. The liquidity-sweep event carries no information.**
Against a matched control (random entries, same session mix, same barriers, same
costs) the rule is **0.064 R worse, t = −3.30** out-of-sample. Its mirror image is
no better. → [docs/S5_STRATEGY.md](docs/S5_STRATEGY.md)

**3. Same-bar execution manufactures 4+ Sharpes.**
A 100-configuration alpha screen produced cross-sectional reversal books at net
Sharpe 4–8. One hour of execution lag removed **96 %** of the gross edge — it was
bid-ask bounce. → [docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md)

**4. Correlated universes manufacture fake breadth.**
The one signal that survived the lag test (VWAP-deviation reversion) had a +1.61
net Sharpe across all instruments, **−2.70 on FX only** and **−0.98 cluster-neutral**.
It was trading XAUUSD against XAUEUR.

**5. Meta-labelling works — on a rule that doesn't.**
The secondary model ranks unseen trades with Spearman +0.090 and lifts mean net R
from −0.132 R (bottom decile) to −0.003 R (top). It recovers the whole cost deficit
and has nothing left over. Its top features include `minute_of_day` and
`cost_to_r_est`: it learned *when trading is expensive*, not when the sweep works.

**6. On 15m FX, commission is the dominant term.**
Retail ECN commission alone (0.70 bp round turn) is ~14 % of one 15-minute ATR on
EURUSD. Any intraday rule risking ~1 ATR starts ~0.15 R down.

---

## `s4_fvg_ml_strategy.py`

Left in place unmodified so the audit is reproducible. See
[docs/S4_AUDIT.md](docs/S4_AUDIT.md) before using it or any number it prints.
