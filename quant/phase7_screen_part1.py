"""
Phase 7, Part 1 — fast vectorized screen of classic, literature-grounded
mathematical/statistical strategy families across the FULL 98-symbol
FX/metals/indices/energy universe (not the small hand-picked subsets used in
earlier phases). This is a "fail fast" triage pass (like screen_candidates.py)
-- no ATR/ratchet trade simulation yet, just honest t-stats on raw forward
returns, net of a friction floor, so we know which family x asset-class
combos are even worth promoting to a full WFO sleeve build.

Strategy families covered here:
  1. Time-series momentum / trend-following (Moskowitz-Ooi-Pedersen style)
  2. Cross-sectional momentum (Menkhoff et al currency-momentum style),
     run separately per asset class (FX / Indices / Metals / Energy) since
     pooling raw returns across asset classes with very different vol scales
     would be methodologically wrong without vol-normalization
  3. Short-horizon mean reversion (z-score fade of extreme daily moves)
  4. Volatility regime persistence (does high realized vol predict high
     *subsequent* vol / does it change the sign or magnitude of the MR or
     momentum edge -- i.e. a regime-conditioning check, not a standalone bet)

PCA/cointegration statistical arbitrage and the ML ensemble are big enough
to warrant their own scripts (phase7_screen_part2_pca_coint.py and
phase7_screen_part3_ml.py).

Carry/value factors are NOT included: this dataset has no interest-rate/OIS
data to compute carry from (confirmed absent from the parquet schema --
only OHLC + tick_volume + spread + session columns), and fabricating a
carry proxy from price action alone would not be carry, it would be
re-labelled momentum. This gap is disclosed, not silently skipped.
"""
from __future__ import annotations
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import strategies as strat

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

FRICTION_BPS_RT = 41.0  # mission-mandated single-instrument round-trip friction

ASSET_CLASSES = {
    "FX": ["AUDCAD", "AUDCHF", "AUDCNH", "AUDJPY", "AUDNZD", "AUDSGD", "AUDUSD",
           "CADCHF", "CADJPY", "CHFJPY", "CHFSGD", "CNHJPY", "EURAUD", "EURCAD",
           "EURCHF", "EURCNH", "EURGBP", "EURHKD", "EURHUF", "EURJPY", "EURMXN",
           "EURNOK", "EURNZD", "EURSEK", "EURSGD", "EURUSD", "EURZAR", "GBPAUD",
           "GBPCAD", "GBPCHF", "GBPCNH", "GBPHKD", "GBPJPY", "GBPNOK", "GBPNZD",
           "GBPSEK", "GBPSGD", "GBPUSD", "NOKJPY", "NOKSEK", "NZDCAD", "NZDCHF",
           "NZDCNH", "NZDJPY", "NZDSGD", "NZDUSD", "SGDJPY", "USDCAD", "USDCHF",
           "USDCNH", "USDHKD", "USDHUF", "USDJPY", "USDMXN", "USDNOK", "USDSEK",
           "USDSGD", "USDTHB", "USDZAR", "ZARJPY"],
    "Indices": ["AU200", "CHINA50", "CHINAH", "DJ30", "FR40", "GER30", "GER40",
                "HK50", "JP225", "NAS100", "NETH25", "SP500", "STOXX50",
                "SWISS20", "UK100", "US2000"],
    "Metals": ["GAUCNH", "GAUUSD", "XAGAUD", "XAGEUR", "XAGSGD", "XAGUSD",
               "XAUAUD", "XAUCNH", "XAUEUR", "XAUGBP", "XAUSGD", "XAUUSD",
               "XPDUSD", "XPTUSD"],
    "Energy_Base": ["ALUMINIUM", "COPPER", "GAS", "LEAD", "NICKEL", "UKBRENT",
                     "USWTI", "ZINC"],
}
ALL_SYMBOLS = sorted(sum(ASSET_CLASSES.values(), []))


def tstat(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 10:
        return np.nan, np.nan, len(x)
    return x.mean(), x.mean() / (x.std(ddof=1) / np.sqrt(len(x))), len(x)


def load_quietly(symbol, tf="d1", min_start=None):
    try:
        df = strat.load_forex(symbol, tf)
    except Exception:
        return None
    if min_start is not None:
        df = df[df.index >= min_start]
    return df if len(df) > 100 else None


# ---------------------------------------------------------------------------
# 1. Time-series momentum (per symbol, D1): sign(trailing K-day return)
#    predicts next 1/5/21-day return. Classic CTA/TSMOM signal.
# ---------------------------------------------------------------------------
def screen_ts_momentum(results):
    lookbacks = [21, 63, 126, 252]   # ~1m, 3m, 6m, 12m
    horizons = [1, 5, 21]
    for lb in lookbacks:
        for hz in horizons:
            all_rets = []
            per_symbol = {}
            for sym in ALL_SYMBOLS:
                df = load_quietly(sym)
                if df is None:
                    continue
                c = df["close"]
                sig = np.sign(c.pct_change(lb)).shift(1)
                fwd = c.pct_change(hz).shift(-hz)
                strat_ret = (sig * fwd).dropna()
                if len(strat_ret) > 50:
                    all_rets.extend(strat_ret.tolist())
                    per_symbol[sym] = strat_ret.mean()
            mean_r, t_, n = tstat(all_rets)
            bps = mean_r * 10000 if not np.isnan(mean_r) else np.nan
            # annualize-ish: how many such horizon-periods per year, vs round-trip friction cost assumption
            top5 = sorted(per_symbol.items(), key=lambda kv: -kv[1])[:5] if per_symbol else []
            results.append(dict(family="ts_momentum", lookback=lb, horizon=hz,
                                 n_obs=n, mean_bps_per_period=round(bps, 2) if bps == bps else None,
                                 t_stat=round(t_, 2) if t_ == t_ else None,
                                 top5_symbols=[(s, round(v * 10000, 1)) for s, v in top5]))
            print(f"[TSMOM] lookback={lb}d horizon={hz}d: n={n} mean={bps:.2f}bps t={t_:.2f}  top5={top5[:3]}")


# ---------------------------------------------------------------------------
# 2. Cross-sectional momentum, per asset class (vol-scaling applied so
#    different instruments' raw return magnitudes are comparable within a
#    class -- standard practice, Menkhoff et al normalize similarly).
# ---------------------------------------------------------------------------
def screen_xs_momentum(results):
    for cls, syms in ASSET_CLASSES.items():
        if len(syms) < 6:
            continue
        closes = {}
        for sym in syms:
            df = load_quietly(sym)
            if df is not None:
                closes[sym] = df["close"]
        if len(closes) < 6:
            continue
        idx = sorted(set.intersection(*[set(c.index) for c in closes.values()]))
        idx = pd.DatetimeIndex(idx)
        if len(idx) < 300:
            continue
        panel = pd.DataFrame({s: c.reindex(idx) for s, c in closes.items()}).dropna(how="all").ffill()
        log_p = np.log(panel)
        daily_ret = log_p.diff()
        vol21 = daily_ret.rolling(21).std()
        formation = (log_p.shift(21) - log_p.shift(252)) / (vol21.shift(21) * np.sqrt(231))  # vol-scaled 12-1 momentum
        fwd_ret = log_p.shift(-21) - log_p

        monthly_idx = panel.index[::21]
        ls_rets = []
        for t in monthly_idx:
            if t not in formation.index:
                continue
            f = formation.loc[t].dropna()
            if len(f) < max(4, len(syms) // 3):
                continue
            ranked = f.sort_values()
            n = max(1, len(ranked) // 3)
            losers, winners = ranked.index[:n], ranked.index[-n:]
            r = fwd_ret.loc[t]
            ls = r[winners].mean() - r[losers].mean()
            if not np.isnan(ls):
                ls_rets.append(ls)
        mean_r, t_, n = tstat(ls_rets)
        bps = mean_r * 10000 if not np.isnan(mean_r) else np.nan
        results.append(dict(family="xs_momentum", asset_class=cls, n_months=n,
                             mean_bps_per_month=round(bps, 1) if bps == bps else None,
                             t_stat=round(t_, 2) if t_ == t_ else None,
                             n_symbols=len(closes),
                             needs_bps_to_clear_2leg_cost=FRICTION_BPS_RT * 2))
        print(f"[XSMOM-{cls}] n_months={n} n_syms={len(closes)} mean={bps:.1f}bps/mo t={t_:.2f} "
              f"(need >{FRICTION_BPS_RT*2:.0f}bps to clear 2-leg cost)")


# ---------------------------------------------------------------------------
# 3. Short-horizon mean reversion: z-score of 1-day return vs trailing 21d
#    distribution, fade extremes, check next 1/5-day forward return.
# ---------------------------------------------------------------------------
def screen_mean_reversion(results):
    for z_thresh in [1.5, 2.0, 2.5]:
        for hz in [1, 5]:
            all_rets = []
            for sym in ALL_SYMBOLS:
                df = load_quietly(sym)
                if df is None:
                    continue
                c = df["close"]
                ret = c.pct_change()
                z = (ret - ret.rolling(21).mean()) / ret.rolling(21).std()
                fwd = c.pct_change(hz).shift(-hz)
                fade_sig = -np.sign(z.where(z.abs() > z_thresh))
                strat_ret = (fade_sig * fwd).dropna()
                if len(strat_ret) > 20:
                    all_rets.extend(strat_ret.tolist())
            mean_r, t_, n = tstat(all_rets)
            bps = mean_r * 10000 if not np.isnan(mean_r) else np.nan
            results.append(dict(family="mean_reversion", z_thresh=z_thresh, horizon=hz,
                                 n_obs=n, mean_bps_per_trade=round(bps, 2) if bps == bps else None,
                                 t_stat=round(t_, 2) if t_ == t_ else None))
            print(f"[MR] z>{z_thresh} horizon={hz}d: n={n} mean={bps:.2f}bps t={t_:.2f}")


# ---------------------------------------------------------------------------
# 4. Volatility-regime conditioning: does the mean-reversion edge differ in
#    high-vol vs low-vol regimes? (motivated by Sleeve R's crypto finding
#    that MR is strongly regime-dependent)
# ---------------------------------------------------------------------------
def screen_vol_regime_conditioning(results):
    z_thresh, hz = 2.0, 1
    hi_rets, lo_rets = [], []
    for sym in ALL_SYMBOLS:
        df = load_quietly(sym)
        if df is None:
            continue
        c = df["close"]
        ret = c.pct_change()
        vol21 = ret.rolling(21).std()
        vol_pct = vol21.rolling(252).rank(pct=True)
        z = (ret - ret.rolling(21).mean()) / vol21
        fwd = c.pct_change(hz).shift(-hz)
        fade_sig = -np.sign(z.where(z.abs() > z_thresh))
        strat_ret = (fade_sig * fwd)
        hi_mask = vol_pct > 0.7
        lo_mask = vol_pct < 0.3
        hi_rets.extend(strat_ret[hi_mask].dropna().tolist())
        lo_rets.extend(strat_ret[lo_mask].dropna().tolist())
    m_hi, t_hi, n_hi = tstat(hi_rets)
    m_lo, t_lo, n_lo = tstat(lo_rets)
    results.append(dict(family="vol_regime_mr", high_vol_bps=round(m_hi * 10000, 2) if m_hi == m_hi else None,
                         high_vol_t=round(t_hi, 2) if t_hi == t_hi else None, high_vol_n=n_hi,
                         low_vol_bps=round(m_lo * 10000, 2) if m_lo == m_lo else None,
                         low_vol_t=round(t_lo, 2) if t_lo == t_lo else None, low_vol_n=n_lo))
    print(f"[VOLREGIME-MR] HIGH-vol regime: n={n_hi} mean={m_hi*10000:.2f}bps t={t_hi:.2f} | "
          f"LOW-vol regime: n={n_lo} mean={m_lo*10000:.2f}bps t={t_lo:.2f}")


def main():
    results = []
    print(f"Universe: {len(ALL_SYMBOLS)} symbols across {len(ASSET_CLASSES)} asset classes\n")
    screen_ts_momentum(results)
    print()
    screen_xs_momentum(results)
    print()
    screen_mean_reversion(results)
    print()
    screen_vol_regime_conditioning(results)

    out_path = os.path.join(RESULTS_DIR, "PHASE7_part1_screen.json")
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
