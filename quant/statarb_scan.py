"""
statarb_scan.py — Quant-style statistical arbitrage pair discovery for FX.

Procedure (strictly in-sample, no peeking at OOS data used later by WFO):
  1. Load daily close for a curated G10+ FX cross universe.
  2. Engle-Granger cointegration test (OLS hedge ratio + ADF on residual)
     over an IN-SAMPLE window only (2016-01-01 .. 2022-06-30).
  3. For every pair with coint p-value < 0.05, estimate the OU half-life of
     the residual spread via AR(1): d(spread) = theta*(mu - spread) + eps.
  4. Keep pairs with 3 <= half-life <= 40 trading days (fast enough to be
     tradable inside typical holding windows, slow enough not to be noise).
  5. Rank by p-value and report full statistics for transparency.

This produces the candidate universe for quant/strategies.py Sleeve F. The
hedge ratio itself is RE-ESTIMATED out-of-sample on a rolling basis inside
the strategy (see prep_sleeve_f) -- this scan only answers "is this pair
structurally cointegrated at all", which is a property we lock in once,
the same way a human quant would pick a pairs universe before building a
trading rule on top of it.
"""
import itertools
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint, adfuller
import strategies as strat

warnings.filterwarnings("ignore")

IS_END = pd.Timestamp("2019-12-31", tz="UTC")

UNIVERSE = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "NZDUSD", "USDCAD",
    "EURGBP", "EURCHF", "EURJPY", "EURAUD", "EURCAD", "EURNZD", "EURNOK", "EURSEK",
    "GBPJPY", "GBPCHF", "GBPAUD", "GBPCAD", "GBPNZD", "GBPNOK", "GBPSEK",
    "AUDJPY", "AUDCHF", "AUDCAD", "AUDNZD", "NZDJPY", "NZDCHF", "NZDCAD",
    "CADJPY", "CADCHF", "CHFJPY", "USDNOK", "USDSEK", "NOKJPY", "NOKSEK",
    "USDMXN", "USDZAR", "EURMXN", "EURZAR", "ZARJPY",
]


def load_close(symbol):
    df = strat.load_forex(symbol, "d1")
    c = df["close"]
    c = c[c.index <= IS_END]
    return np.log(c)


def ou_half_life(spread: pd.Series):
    s = spread.dropna()
    lag = s.shift(1).dropna()
    delta = s.diff().dropna()
    lag = lag.loc[delta.index]
    x = np.vstack([np.ones(len(lag)), lag.values]).T
    beta, *_ = np.linalg.lstsq(x, delta.values, rcond=None)
    theta = -beta[1]
    if theta <= 0:
        return np.inf
    return np.log(2) / theta


def main():
    series = {}
    for sym in UNIVERSE:
        try:
            s = load_close(sym)
            if len(s) > 500:
                series[sym] = s
        except Exception as e:
            print(f"  [skip] {sym}: {e}")

    print(f"Loaded {len(series)}/{len(UNIVERSE)} instruments with sufficient IS history.\n")

    results = []
    for a, b in itertools.combinations(sorted(series.keys()), 2):
        sa, sb = series[a], series[b]
        idx = sa.index.intersection(sb.index)
        if len(idx) < 500:
            continue
        xa, xb = sa.loc[idx], sb.loc[idx]
        try:
            score, pvalue, _ = coint(xa, xb, trend="c")
        except Exception:
            continue
        # OLS hedge ratio on the SAME in-sample window (Engle-Granger step 1)
        X = np.vstack([np.ones(len(xb)), xb.values]).T
        beta, *_ = np.linalg.lstsq(X, xa.values, rcond=None)
        hedge_ratio = beta[1]
        spread = xa - (beta[0] + hedge_ratio * xb)
        hl = ou_half_life(spread)
        adf_stat, adf_p, *_ = adfuller(spread.dropna(), autolag="AIC")
        results.append(dict(pair=f"{a}~{b}", a=a, b=b, coint_p=pvalue, adf_p=adf_p,
                             hedge_ratio=round(hedge_ratio, 4), half_life_days=round(hl, 1)
                             if np.isfinite(hl) else None, n_obs=len(idx)))

    df = pd.DataFrame(results).sort_values("coint_p")
    tradable = df[(df["coint_p"] < 0.05) & (df["half_life_days"].notna()) &
                  (df["half_life_days"] >= 3) & (df["half_life_days"] <= 40)]
    print(f"Tested {len(df)} pairs. {len(df[df['coint_p']<0.05])} pass coint p<0.05. "
          f"{len(tradable)} ALSO have a tradable half-life (3-40d).\n")
    print("=== TOP CANDIDATES (coint p<0.05, half-life 3-40 trading days) ===")
    print(tradable.head(20).to_string(index=False))
    df.to_csv("results/statarb_coint_scan_full.csv", index=False)
    tradable.to_csv("results/statarb_coint_scan_tradable.csv", index=False)
    return tradable


if __name__ == "__main__":
    main()
