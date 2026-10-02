"""
quantlab.config
===============
Single source of truth for universe definition, cost assumptions, strategy
hyper-parameters and walk-forward schedule.

Everything that a reviewer would want to challenge lives here, in one place,
with the rationale written next to the number.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
FOREX_DIR = REPO_ROOT / "Forex_Data"
REPORT_DIR = REPO_ROOT / "reports"
DOCS_DIR = REPO_ROOT / "docs"

NOMINAL_SECONDS = {"15m": 900, "1h": 3600, "4h": 14400, "d1": 86400}

# --------------------------------------------------------------------------
# Universe
# --------------------------------------------------------------------------
# The user-requested slice is "FX + metals + indices".  Energy (GAS, UKBRENT,
# USWTI) and the crypto CFDs carried in Forex_Data/manifest_crypto.json are
# deliberately excluded: their cost/vol regimes are different enough that
# pooling them into one cross-sectional model is a modelling error, not
# diversification.
INCLUDED_CATEGORIES = ("Forex_Raw", "Indices", "Commodities_Raw", "Commodities")

ENERGY = {"GAS", "UKBRENT", "USWTI"}

# Hard exclusions: duplicate contracts for the same underlying.  Keeping both
# halves of a pair manufactures fake breadth and double-counts the same risk.
#   GER30 / GER40   -> same DAX contract, different tick convention
#   GAUUSD / XAUUSD -> same spot gold, different liquidity pool
#   GAUCNH / XAUCNH -> same
DUPLICATE_EXCLUSIONS = {"GER30", "GAUUSD", "GAUCNH"}

# Pegged / administered crosses.  USDHKD and EURHKD track an explicit HKMA
# band, USDTHB and the CNH crosses are managed floats.  A liquidity-sweep
# reversion edge is not economically meaningful inside a defended band.
PEGGED_EXCLUSIONS = {"USDHKD", "EURHKD", "GBPHKD", "USDTHB"}

# --------------------------------------------------------------------------
# Data integrity thresholds
# --------------------------------------------------------------------------
# The shipped *_15m_real.parquet files are NOT pure 15-minute series: every
# file is prefixed with a block of DAILY (and in some cases hourly) bars
# carrying 15m file names.  See docs/DATA_INTEGRITY.md.  We therefore detect
# the first timestamp from which the nominal bar spacing actually holds and
# refuse to use anything earlier.
PURE_RESOLUTION_WINDOW = 500      # bars in the rolling spacing test
PURE_RESOLUTION_MIN_FRAC = 0.80   # >= 80% of diffs must equal nominal spacing
MIN_PURE_BARS_15M = 20_000        # ~7 months of 24h 15m bars; below this, drop

# --------------------------------------------------------------------------
# Transaction costs
# --------------------------------------------------------------------------
# Forex_Data bars carry a per-bar `spread` column in integer broker POINTS
# (multiply by manifest `point` to get price units).  That is the single most
# valuable column in this dataset and we use it directly whenever it is
# populated.  It is zero for the padded historical block and for a share of
# 2023-2025 bars, so we fall back to the asset's own observed median spread
# conditioned on hour-of-day.
#
# Commission: raw-spread retail ECN pricing is ~USD 3.5 per 100k lot per side
# -> 0.35bp per side, 0.70bp round turn.  Index/metal CFDs at this broker are
# typically commission-free with a wider spread, which the spread column
# already reflects.
COMMISSION_BPS_ROUND_TURN = {
    "Forex_Raw": 0.70,
    "Commodities_Raw": 0.0,
    "Commodities": 0.0,
    "Indices": 0.0,
}

# Slippage model.  Market entries are assumed to pay a fraction of the spread
# on top of the quoted spread; stop-loss exits are *stop* orders and slip
# harder, especially on the gap-through case which the bar engine handles
# explicitly.  Limit (take-profit) fills are assumed to get the quoted price.
SLIPPAGE_ENTRY_SPREADS = 0.25
SLIPPAGE_STOP_SPREADS = 0.75
SLIPPAGE_TIMEEXIT_SPREADS = 0.25

# Universe cost screen.
#
# Measured on this dataset, retail ECN *commission alone* (0.70bp round turn)
# is ~14% of one 15-minute ATR on EURUSD.  That is the single most important
# economic fact for an intraday FX strategy and it is why the stop floor below
# is 1.5x ATR rather than something tighter: at R = 1.5 x ATR a 0.33 cost/ATR
# instrument costs 0.22R per round turn, which a 2:1 payoff can still carry.
#
# The screen is expressed as cost / R where R is the *assumed* stop distance,
# so the number means something to a trader.
ASSUMED_R_IN_ATR = 1.5
MAX_COST_TO_R = 0.22          # equivalently cost/ATR <= 0.33

# --------------------------------------------------------------------------
# Strategy: S5 liquidity-sweep reversion
# --------------------------------------------------------------------------


@dataclass
class StrategyParams:
    """S5 primary-rule parameters. All windows are in 15m bars unless noted."""

    # --- liquidity pools -------------------------------------------------
    use_prev_day: bool = True
    use_asian_range: bool = True
    use_prev_week: bool = True
    use_swing: bool = True             # rolling intraday swing extreme
    swing_lookback: int = 48           # 12h of 15m bars
    asian_start_hour: int = 0          # UTC
    asian_end_hour: int = 6            # exclusive; range is [00:00, 06:00)
    pool_tolerance_atr: float = 0.05   # pools within this distance are merged

    # --- sweep geometry ---------------------------------------------------
    min_penetration_atr: float = 0.10  # must actually take out the level
    max_penetration_atr: float = 1.50  # beyond this it is a breakout, not a raid
    require_reclaim: bool = True       # close must return inside the pool
    min_reclaim_frac: float = 0.50     # close position within the sweep bar range
    max_bars_since_pool: int = 480     # pool must be <5 days old

    # --- session filter ---------------------------------------------------
    # London 07:00-11:00 UTC and New York 12:00-16:00 UTC.  Stop clusters are
    # hit when liquidity arrives; the Asian session is excluded because the
    # Asian range is itself one of the pools.
    session_hours: tuple = (6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17)
    exclude_friday_after: int = 19     # UTC hour; avoid the weekend-gap tail
    exclude_sunday_before: int = 2

    # --- risk geometry ----------------------------------------------------
    stop_buffer_atr: float = 0.25
    # Stop floor is a cost decision, not a chart decision: see MAX_COST_TO_R.
    min_stop_atr: float = 1.50
    max_stop_atr: float = 4.00
    take_profit_r: float = 2.00
    time_stop_bars: int = 32           # 8 hours
    # Ratchet: once price has run +trigger R, stop moves to +lock R.
    ratchet: tuple = ((1.00, 0.10), (1.50, 0.75))

    # --- direction --------------------------------------------------------
    # "fade"        -> trade against the sweep (stop-run reversion, Osler 2003)
    # "continuation"-> trade with the sweep (classic breakout/ICT displacement)
    polarity: str = "fade"

    # --- cooldown ---------------------------------------------------------
    cooldown_bars: int = 8             # no second signal on the same asset for 2h
    unbreached_lookback: int = 6       # pool must be intact for this many bars


@dataclass
class PortfolioParams:
    initial_equity: float = 100_000.0
    risk_per_trade_pct: float = 0.30   # of current equity, pre-vol-scaling
    max_risk_per_trade_pct: float = 0.60
    max_concurrent_positions: int = 8
    max_concurrent_per_asset: int = 1
    max_concurrent_per_cluster: int = 2
    # Kelly-style scaling off the meta-model probability, clipped hard.
    prob_sizing: bool = True
    prob_floor: float = 0.50           # below this the trade is skipped
    prob_cap: float = 0.75             # sizing saturates here
    # Daily loss circuit breaker (fraction of start-of-day equity).
    daily_loss_limit_pct: float = 2.0


@dataclass
class ModelParams:
    features: List[str] = field(default_factory=list)
    # Purged, embargoed, anchored walk-forward.
    min_train_signals: int = 1000
    refit_every_days: int = 60
    embargo_bars: int = 64             # >= time_stop_bars; label horizon leakage guard
    seed: int = 7
    n_folds_cv: int = 4
    # Probability gate is *learned* on the training fold (quantile of train
    # predictions) rather than hard-coded, so it adapts to base-rate drift.
    gate_quantile: float = 0.55


# --------------------------------------------------------------------------
# Walk-forward schedule
# --------------------------------------------------------------------------
# True 15m history begins 2023-09 for ~66 instruments and 2024-07 for the
# rest.  We therefore burn 2023-09..2024-06 as the initial training block and
# run out-of-sample from 2024-07-01 to the end of the data.
WF_TRAIN_START = "2023-09-15"
WF_OOS_START = "2024-07-01"
WF_OOS_END = "2026-09-21"


def load_manifest() -> Dict:
    """Merge the FX/CFD manifest with the crypto manifest (we exclude crypto,
    but we still want the metadata to be able to say *why*)."""
    out: Dict[str, Dict] = {}
    for name in ("manifest.json", "manifest_crypto.json"):
        p = FOREX_DIR / name
        if not p.exists():
            continue
        m = json.loads(p.read_text())
        for sym, meta in m.get("symbols", {}).items():
            out[sym] = meta
    return out


def base_universe(manifest: Optional[Dict] = None) -> List[str]:
    """Category + duplicate + peg screen.  Liquidity/cost screening happens
    later in `quantlab.universe`, because it needs the price data."""
    manifest = manifest or load_manifest()
    syms = []
    for sym, meta in manifest.items():
        cat = meta.get("category", "")
        if cat not in INCLUDED_CATEGORIES:
            continue
        if sym in ENERGY or sym in DUPLICATE_EXCLUSIONS or sym in PEGGED_EXCLUSIONS:
            continue
        syms.append(sym)
    return sorted(syms)


# --------------------------------------------------------------------------
# FROZEN STRATEGY PARAMETERS
# --------------------------------------------------------------------------
# Selected by research/dev_search.py on the development block
# 2023-09-15..2024-06-30 ONLY, in a coordinate-wise search of 31 configs.
# Full trial log: reports/dev_search_trials.csv.
#
# These are frozen.  Nothing downstream re-tunes them, and the out-of-sample
# run from 2024-07-01 has never informed them.  The trial count is carried
# into the Deflated Sharpe Ratio so the reader can discount accordingly.
FROZEN_PARAMS = StrategyParams(
    polarity="fade",
    take_profit_r=1.5,
    ratchet=(),
    min_penetration_atr=0.20,
    max_penetration_atr=1.50,
    min_reclaim_frac=0.70,
    time_stop_bars=16,
    min_stop_atr=2.5,
)
N_DEV_TRIALS = 31
