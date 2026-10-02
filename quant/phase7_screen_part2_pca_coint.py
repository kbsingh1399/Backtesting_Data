"""
Phase 7, Part 2 — PCA statistical-arbitrage + Engle-Granger pairwise
cointegration screen across natural FX/metals/indices baskets in the full
98-symbol universe. Phase 5 already ran a Johansen+PCA screen on US equity
indices and killed it (PC1 explained 86.3% of variance but residuals showed
no stationary mean-reverting combination). Here we test baskets with much
stronger structural linkage -- FX triangulation (EUR-crosses, GBP-crosses)
and a G10 majors basket -- where real cointegrating relationships are more
plausible a priori (e.g. EURUSD, EURGBP, GBPUSD are mechanically linked via
triangulation arbitrage enforced by the market itself).

Methods:
  - PCA factor decomposition + residual stationarity (ADF test) on each
    basket's log-price panel.
  - Pairwise Engle-Granger cointegration test (statsmodels) across ALL pairs
    within each basket, not just a hand-picked few -- this is the proper,
    broader version of what Sleeves F/G/H attempted on a narrower pair list.
  - For any statistically cointegrated pair/basket residual (ADF p<0.05),
    report the resulting z-score mean-reversion edge net of 2-leg 82bps
    friction, so we know whether statistical significance survives contact
    with realistic trading costs.
"""
from __future__ import annotations
import itertools
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint
from sklearn.decomposition import PCA

import strategies as strat

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

BASKETS = {
    "G10_majors": ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY"],
    "EUR_crosses": ["EURUSD", "EURGBP", "EURJPY", "EURCHF", "EURAUD", "EURCAD", "EURNZD"],
    "GBP_crosses": ["GBPUSD", "GBPJPY", "GBPCHF", "GBPAUD", "GBPCAD", "GBPNZD"],
    "Commodity_bloc_FX": ["AUDUSD", "NZDUSD", "USDCAD", "AUDCAD", "AUDNZD", "NZDCAD"],
    "Precious_metals": ["XAUUSD", "XAGUSD", "XPTUSD", "XPDUSD"],
    "Base_metals": ["ALUMINIUM", "COPPER", "NICKEL", "ZINC", "LEAD"],
    "Equity_indices_core": ["SP500", "NAS100", "DJ30", "US2000", "GER40", "UK100", "JP225"],
}


def load_panel(symbols, tf="d1", min_rows=500):
    closes = {}
    for sym in symbols:
        try:
            df = strat.load_forex(sym, tf)
        except Exception:
            continue
        if len(df) > min_rows:
            closes[sym] = df["close"]
    if len(closes) < 3:
        return None
    idx = sorted(set.intersection(*[set(c.index) for c in closes.values()]))
    idx = pd.DatetimeIndex(idx)
    if len(idx) < min_rows:
        return None
    return pd.DataFrame({s: c.reindex(idx) for s, c in closes.items()}).dropna()


def pca_residual_screen(basket_name, symbols, results):
    panel = load_panel(symbols)
    if panel is None:
        print(f"[PCA-{basket_name}] insufficient overlapping data, skipped")
        return
    log_p = np.log(panel)
    ret = log_p.diff().dropna()
    scaler_std = ret.std()
    ret_scaled = ret / scaler_std  # normalize so PCA isn't dominated by one noisy series

    pca = PCA(n_components=min(len(symbols), 5))
    pcs = pca.fit_transform(ret_scaled.values)
    explained = pca.explained_variance_ratio_

    # Reconstruct using only PC1 (the dominant common factor) and look at the
    # cumulative residual (the "statistical spread") for stationarity.
    pc1_loadings = pca.components_[0]
    common_factor_ret = pcs[:, 0:1] @ pc1_loadings.reshape(1, -1)
    residual_ret = ret_scaled.values - common_factor_ret
    # Use the FIRST asset's residual cumulative sum as the representative
    # "basket-relative" spread (any column works structurally; report the
    # one with the largest idiosyncratic residual variance as most likely
    # to show a tradeable, basket-hedged mean-reverting series).
    resid_var = residual_ret.var(axis=0)
    rep_idx = int(np.argmax(resid_var))
    rep_symbol = ret.columns[rep_idx]
    spread = np.cumsum(residual_ret[:, rep_idx])

    adf_stat, adf_p, *_ = adfuller(spread, autolag="AIC")
    results.append(dict(family="pca_stat_arb", basket=basket_name, n_symbols=len(panel.columns),
                         n_obs=len(panel), pc1_explained_variance=round(float(explained[0]), 3),
                         representative_symbol=rep_symbol, adf_pvalue=round(float(adf_p), 4),
                         stationary_at_5pct=bool(adf_p < 0.05)))
    print(f"[PCA-{basket_name}] n_syms={len(panel.columns)} n_obs={len(panel)} "
          f"PC1_var={explained[0]:.1%} residual_ADF_p={adf_p:.4f} "
          f"{'STATIONARY (tradeable spread candidate)' if adf_p < 0.05 else 'NOT stationary -> no basket stat-arb edge'}")


def pairwise_cointegration_screen(basket_name, symbols, results):
    panel = load_panel(symbols)
    if panel is None:
        return
    log_p = np.log(panel)
    cointegrated_pairs = []
    for a, b in itertools.combinations(log_p.columns, 2):
        try:
            score, pvalue, _ = coint(log_p[a], log_p[b])
        except Exception:
            continue
        if pvalue < 0.05:
            # Found a statistically cointegrated pair -> estimate hedge ratio via OLS,
            # build spread z-score, check net-of-cost mean-reversion edge.
            beta = np.polyfit(log_p[b], log_p[a], 1)[0]
            spread = log_p[a] - beta * log_p[b]
            z = (spread - spread.rolling(63).mean()) / spread.rolling(63).std()
            fwd = (log_p[a].diff().shift(-1) - beta * log_p[b].diff().shift(-1))
            fade_sig = -np.sign(z.where(z.abs() > 2.0))
            strat_ret = (fade_sig * fwd).dropna()
            mean_bps = strat_ret.mean() * 10000 if len(strat_ret) > 20 else np.nan
            t_ = (strat_ret.mean() / (strat_ret.std() / np.sqrt(len(strat_ret)))) if len(strat_ret) > 20 else np.nan
            cointegrated_pairs.append(dict(pair=f"{a}-{b}", pvalue=round(float(pvalue), 4),
                                            hedge_ratio=round(float(beta), 4), n_trades_signal=int(len(strat_ret)),
                                            mean_bps_per_signal=round(float(mean_bps), 2) if mean_bps == mean_bps else None,
                                            t_stat=round(float(t_), 2) if t_ == t_ else None,
                                            clears_82bps_2leg_cost=bool(mean_bps == mean_bps and mean_bps > 82.0)))
    results.append(dict(family="pairwise_cointegration", basket=basket_name,
                         n_pairs_tested=len(list(itertools.combinations(symbols, 2))),
                         n_cointegrated_at_5pct=len(cointegrated_pairs),
                         cointegrated_pairs=sorted(cointegrated_pairs, key=lambda d: d["pvalue"])))
    n_clear = sum(1 for p in cointegrated_pairs if p["clears_82bps_2leg_cost"])
    print(f"[COINT-{basket_name}] {len(cointegrated_pairs)}/{len(list(itertools.combinations(symbols,2)))} "
          f"pairs cointegrated @5%; {n_clear} of those clear 82bps 2-leg cost")
    for p in sorted(cointegrated_pairs, key=lambda d: d["pvalue"])[:5]:
        print(f"    {p['pair']}: p={p['pvalue']} beta={p['hedge_ratio']} "
              f"edge={p['mean_bps_per_signal']}bps t={p['t_stat']} n={p['n_trades_signal']}")


def main():
    results = []
    print("=== PCA residual-stationarity screen ===")
    for name, syms in BASKETS.items():
        pca_residual_screen(name, syms, results)
    print("\n=== Pairwise Engle-Granger cointegration screen ===")
    for name, syms in BASKETS.items():
        if len(syms) >= 3:
            pairwise_cointegration_screen(name, syms, results)

    out_path = os.path.join(RESULTS_DIR, "PHASE7_part2_pca_coint.json")
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
