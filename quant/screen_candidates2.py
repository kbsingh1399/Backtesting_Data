"""
screen_candidates2.py — corrected crypto factor screens, market-neutral.

The first pass (screen_candidates.py items 5-9) showed universally POSITIVE
forward returns under every conditioning variable, including opposite-signed
ones (extreme high AND extreme low funding both "positive"). That's the
signature of a confound, not a signal: BTC/crypto had a strong net bull drift
over 2020-2026, so any filter applied to a long-only forward-return sample
inherits that drift as a positive bias. Fixed here by using *cross-sectionally
de-meaned* (market-neutral) forward returns: for every 15m timestamp, subtract
the equal-weight mean forward return across the whole crypto universe at that
same timestamp. What's left is each symbol's IDIOSYNCRATIC excess return --
this is what a funding/sentiment/liquidation factor should actually predict
if it has real selection power, independent of market beta.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import strategies as strat

CRYPTO_UNIVERSE = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
                    "AVAXUSDT", "LINKUSDT", "DOGEUSDT", "ADAUSDT", "DOTUSDT"]


def tstat(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 10:
        return np.nan, np.nan, len(x)
    return x.mean(), x.mean() / (x.std(ddof=1) / np.sqrt(len(x))), len(x)


def load_panel():
    data = {}
    for sym in CRYPTO_UNIVERSE:
        df = strat.load_binance(sym)
        data[sym] = df
    idx = sorted(set.intersection(*[set(d.index) for d in data.values()]))
    idx = pd.DatetimeIndex(idx)
    close = pd.DataFrame({s: d["close"].reindex(idx) for s, d in data.items()})
    fwd1d_raw = close.pct_change(96).shift(-96)
    mkt_fwd1d = fwd1d_raw.mean(axis=1)
    fwd1d_xs = fwd1d_raw.sub(mkt_fwd1d, axis=0)  # market-neutral excess fwd return

    fwd8h_raw = close.pct_change(32).shift(-32)
    mkt_fwd8h = fwd8h_raw.mean(axis=1)
    fwd8h_xs = fwd8h_raw.sub(mkt_fwd8h, axis=0)

    feats = {}
    for col in ["funding_rate_pct", "ls_ratio_global", "long_liq_usd", "short_liq_usd",
                "oi_change_pct", "basis_index_bps"]:
        feats[col] = pd.DataFrame({s: data[s][col].reindex(idx) for s in CRYPTO_UNIVERSE})
    price_up_1d = (close.pct_change(96) > 0)
    return idx, feats, fwd1d_xs, fwd8h_xs, price_up_1d


def zscore(panel, window):
    return (panel - panel.rolling(window).mean()) / panel.rolling(window).std()


def flat(df):
    return df.to_numpy().ravel()


def main():
    idx, feats, fwd1d_xs, fwd8h_xs, price_up_1d = load_panel()
    w = 96 * 7

    # 5. funding contrarian, market-neutral
    fr_z = zscore(feats["funding_rate_pct"], w)
    hi, lo = flat(fwd1d_xs[fr_z > 2.0]), flat(fwd1d_xs[fr_z < -2.0])
    m_h, t_h, n_h = tstat(hi); m_l, t_l, n_l = tstat(lo)
    print(f"[5-XS] Funding contrarian (mkt-neutral): extreme HIGH funding -> excess 1d={m_h*10000:.1f}bps "
          f"(n={n_h}, t={t_h:.2f}) | extreme LOW funding -> excess 1d={m_l*10000:.1f}bps (n={n_l}, t={t_l:.2f})")

    # 6. L/S ratio contrarian, market-neutral
    ls_z = zscore(feats["ls_ratio_global"], w)
    hi, lo = flat(fwd1d_xs[ls_z > 2.0]), flat(fwd1d_xs[ls_z < -2.0])
    m_h, t_h, n_h = tstat(hi); m_l, t_l, n_l = tstat(lo)
    print(f"[6-XS] L/S ratio contrarian (mkt-neutral): crowd very LONG -> excess 1d={m_h*10000:.1f}bps "
          f"(n={n_h}, t={t_h:.2f}) | crowd very SHORT -> excess 1d={m_l*10000:.1f}bps (n={n_l}, t={t_l:.2f})")

    # 7. liquidation cascade, market-neutral, 8h horizon
    ll_z = zscore(feats["long_liq_usd"], w)
    sl_z = zscore(feats["short_liq_usd"], w)
    ll_hits, sl_hits = flat(fwd8h_xs[ll_z > 3.0]), flat(fwd8h_xs[sl_z > 3.0])
    m_ll, t_ll, n_ll = tstat(ll_hits); m_sl, t_sl, n_sl = tstat(sl_hits)
    print(f"[7-XS] Liquidation cascade (mkt-neutral): LONG-liq flush -> excess 8h={m_ll*10000:.1f}bps "
          f"(n={n_ll}, t={t_ll:.2f}) | SHORT-liq squeeze -> excess 8h={m_sl*10000:.1f}bps (n={n_sl}, t={t_sl:.2f})")

    # 8. OI+price divergence, market-neutral
    oi_up = feats["oi_change_pct"].rolling(96).sum() > 0
    confirm = price_up_1d & oi_up
    diverge = price_up_1d & ~oi_up
    c_vals, d_vals = flat(fwd1d_xs[confirm]), flat(fwd1d_xs[diverge])
    m_c, t_c, n_c = tstat(c_vals); m_d, t_d, n_d = tstat(d_vals)
    print(f"[8-XS] OI+price divergence (mkt-neutral): confirm(price&OI up) -> excess 1d={m_c*10000:.1f}bps "
          f"(n={n_c}, t={t_c:.2f}) | diverge(price up,OI down) -> excess 1d={m_d*10000:.1f}bps (n={n_d}, t={t_d:.2f})")

    # 9. basis mean reversion, market-neutral
    b_z = zscore(feats["basis_index_bps"], w)
    hi, lo = flat(fwd1d_xs[b_z > 2.0]), flat(fwd1d_xs[b_z < -2.0])
    m_h, t_h, n_h = tstat(hi); m_l, t_l, n_l = tstat(lo)
    print(f"[9-XS] Basis mean reversion (mkt-neutral): extreme HIGH basis -> excess 1d={m_h*10000:.1f}bps "
          f"(n={n_h}, t={t_h:.2f}) | extreme LOW basis -> excess 1d={m_l*10000:.1f}bps (n={n_l}, t={t_l:.2f})")


if __name__ == "__main__":
    main()
