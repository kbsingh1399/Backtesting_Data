"""
screen_candidates.py — fast "fail fast" raw-edge screen across a slate of
10+ distinct, literature-grounded strategy ideas spanning FX and crypto,
using data actually present in this repo. This is NOT the certified WFO
pipeline (no IS/OOS locking, no position sizing/ratchet mechanics) -- it's
a cheap vectorized triage pass to decide which ideas are even worth the
cost of building a full honest WFO campaign (engine.py/statarb_engine.py +
strategies.py + wfo.py). Ideas that don't show a statistically significant,
cost-surviving raw edge here are killed immediately and documented with
numbers; survivors get promoted to a full sleeve.

Run: python3 screen_candidates.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import strategies as strat

FX_UNIVERSE = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
    "EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "EURCHF", "EURAUD", "EURCAD",
    "GBPCHF", "GBPCAD", "AUDCAD", "AUDCHF", "AUDNZD", "CADCHF", "CADJPY",
    "CHFJPY", "NZDCHF", "NZDJPY", "USDMXN", "USDZAR", "USDNOK", "USDSEK",
    "EURNOK", "EURSEK",
]
CRYPTO_UNIVERSE = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
                    "AVAXUSDT", "LINKUSDT", "DOGEUSDT", "ADAUSDT", "DOTUSDT"]

FX_FRICTION_BPS = 41.0     # single-instrument round trip, per mission mandate
CRYPTO_FRICTION_BPS = 41.0  # same mandate applied uniformly (conservative vs real ~8-10bps)


def tstat(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 10:
        return np.nan, np.nan, len(x)
    return x.mean(), x.mean() / (x.std(ddof=1) / np.sqrt(len(x))), len(x)


# ---------------------------------------------------------------------------
# 1. FX cross-sectional momentum (Menkhoff et al 2012 "Currency Momentum")
# ---------------------------------------------------------------------------
def screen_fx_xs_momentum():
    closes = {}
    for sym in FX_UNIVERSE:
        try:
            c = strat.load_forex(sym, "d1")["close"]
            closes[sym] = c
        except Exception:
            continue
    idx = sorted(set.intersection(*[set(c.index) for c in closes.values()]))
    idx = pd.DatetimeIndex(idx)
    panel = pd.DataFrame({s: c.reindex(idx) for s, c in closes.items()}).dropna()
    log_p = np.log(panel)
    formation = log_p.shift(21) - log_p.shift(252)   # 12M return, skip most recent month
    fwd_ret = log_p.shift(-21) - log_p               # subsequent 1M return

    monthly_idx = panel.index[::21]
    long_minus_short = []
    for t in monthly_idx:
        if t not in formation.index:
            continue
        f = formation.loc[t].dropna()
        if len(f) < 10:
            continue
        ranked = f.sort_values()
        n = len(ranked) // 3
        losers, winners = ranked.index[:n], ranked.index[-n:]
        r = fwd_ret.loc[t]
        ls = r[winners].mean() - r[losers].mean()
        if not np.isnan(ls):
            long_minus_short.append(ls)
    mean_r, t_, n = tstat(long_minus_short)
    bps = mean_r * 10000 if not np.isnan(mean_r) else np.nan
    print(f"[1] FX XS momentum (12-1, tercile L/S, monthly): n_months={n} "
          f"mean_spread={bps:.1f}bps/month t={t_:.2f}  (need >{FX_FRICTION_BPS*2:.0f}bps to clear 2-leg cost)")


# ---------------------------------------------------------------------------
# 2. FX time-series trend following (EMA50/200 cross, Moskowitz/Ooi/Pedersen style)
# ---------------------------------------------------------------------------
def screen_fx_ts_momentum():
    results = []
    for sym in FX_UNIVERSE:
        try:
            df = strat.load_forex(sym, "d1")
        except Exception:
            continue
        c = df["close"]
        ema_f = c.ewm(span=50, adjust=False).mean()
        ema_s = c.ewm(span=200, adjust=False).mean()
        sig = np.sign(ema_f - ema_s).shift(1)
        ret = c.pct_change()
        strat_ret = (sig * ret).dropna()
        results.extend(strat_ret.tolist())
    mean_r, t_, n = tstat(results)
    bps = mean_r * 10000 if not np.isnan(mean_r) else np.nan
    print(f"[2] FX TS trend (EMA50/200): n_days={n} mean_daily={bps:.2f}bps t={t_:.2f} "
          f"(need >{FX_FRICTION_BPS/252*100:.3f}bps/day-equivalent churn cost; "
          f"trades infrequent so compare to per-trade switch cost separately)")


# ---------------------------------------------------------------------------
# 3. Turn-of-month seasonality (documented small-sample calendar anomaly)
# ---------------------------------------------------------------------------
def screen_turn_of_month():
    results_tom, results_rest = [], []
    for sym in FX_UNIVERSE:
        try:
            df = strat.load_forex(sym, "d1")
        except Exception:
            continue
        c = df["close"]
        ret = c.pct_change().dropna()
        day = ret.index.day
        days_in_month = ret.index.days_in_month
        is_tom = (day >= days_in_month - 1) | (day <= 3)
        results_tom.extend(ret[is_tom].tolist())
        results_rest.extend(ret[~is_tom].tolist())
    m_tom, t_tom, n_tom = tstat(results_tom)
    m_rest, t_rest, n_rest = tstat(results_rest)
    print(f"[3] Turn-of-month: TOM mean={m_tom*10000:.2f}bps/day (n={n_tom}, t={t_tom:.2f}) "
          f"vs rest={m_rest*10000:.2f}bps/day (n={n_rest}, t={t_rest:.2f})")


# ---------------------------------------------------------------------------
# 4. Day-of-week seasonality
# ---------------------------------------------------------------------------
def screen_day_of_week():
    by_day = {i: [] for i in range(5)}
    for sym in FX_UNIVERSE:
        try:
            df = strat.load_forex(sym, "d1")
        except Exception:
            continue
        ret = df["close"].pct_change().dropna()
        for wd, r in zip(ret.index.weekday, ret.values):
            if wd in by_day:
                by_day[wd].append(r)
    names = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    line = "[4] Day-of-week means(bps)/t-stat: "
    for i, nm in enumerate(names):
        m, t_, n = tstat(by_day[i])
        line += f"{nm}={m*10000:.2f}/{t_:.2f}  "
    print(line)


# ---------------------------------------------------------------------------
# 5. Crypto funding-rate contrarian (crowded-positioning fade)
# ---------------------------------------------------------------------------
def screen_funding_contrarian():
    results_pos, results_neg = [], []
    for sym in CRYPTO_UNIVERSE:
        try:
            df = strat.load_binance(sym)
        except Exception:
            continue
        fr = df["funding_rate_pct"]
        fr_z = (fr - fr.rolling(96 * 7).mean()) / fr.rolling(96 * 7).std()  # 1wk lookback, 15m bars
        fwd_ret = df["close"].pct_change(96).shift(-96)  # next 1-day fwd return
        extreme_pos = fr_z > 2.0
        extreme_neg = fr_z < -2.0
        results_pos.extend(fwd_ret[extreme_pos].dropna().tolist())
        results_neg.extend(fwd_ret[extreme_neg].dropna().tolist())
    m_p, t_p, n_p = tstat(results_pos)
    m_n, t_n, n_n = tstat(results_neg)
    print(f"[5] Funding contrarian: after extreme HIGH funding (crowded long) 1d-fwd={m_p*10000:.1f}bps "
          f"(n={n_p}, t={t_p:.2f}, expect negative if fade works) | "
          f"after extreme LOW funding (crowded short) 1d-fwd={m_n*10000:.1f}bps (n={n_n}, t={t_n:.2f}, expect positive)")


# ---------------------------------------------------------------------------
# 6. Crypto long/short ratio contrarian
# ---------------------------------------------------------------------------
def screen_ls_ratio_contrarian():
    results_high, results_low = [], []
    for sym in CRYPTO_UNIVERSE:
        try:
            df = strat.load_binance(sym)
        except Exception:
            continue
        ls = df["ls_ratio_global"]
        ls_z = (ls - ls.rolling(96 * 7).mean()) / ls.rolling(96 * 7).std()
        fwd_ret = df["close"].pct_change(96).shift(-96)
        results_high.extend(fwd_ret[ls_z > 2.0].dropna().tolist())
        results_low.extend(fwd_ret[ls_z < -2.0].dropna().tolist())
    m_h, t_h, n_h = tstat(results_high)
    m_l, t_l, n_l = tstat(results_low)
    print(f"[6] L/S ratio contrarian: after crowd very LONG 1d-fwd={m_h*10000:.1f}bps (n={n_h}, t={t_h:.2f}) | "
          f"after crowd very SHORT 1d-fwd={m_l*10000:.1f}bps (n={n_l}, t={t_l:.2f})")


# ---------------------------------------------------------------------------
# 7. Liquidation-cascade mean reversion
# ---------------------------------------------------------------------------
def screen_liq_cascade():
    results_long_liq, results_short_liq = [], []
    for sym in CRYPTO_UNIVERSE:
        try:
            df = strat.load_binance(sym)
        except Exception:
            continue
        ll, sl = df["long_liq_usd"], df["short_liq_usd"]
        ll_z = (ll - ll.rolling(96 * 7).mean()) / ll.rolling(96 * 7).std()
        sl_z = (sl - sl.rolling(96 * 7).mean()) / sl.rolling(96 * 7).std()
        fwd_ret = df["close"].pct_change(32).shift(-32)  # next ~8h
        results_long_liq.extend(fwd_ret[ll_z > 3.0].dropna().tolist())   # long-liq flush -> expect bounce (+)
        results_short_liq.extend(fwd_ret[sl_z > 3.0].dropna().tolist())  # short-liq squeeze -> expect pullback (-)
    m_ll, t_ll, n_ll = tstat(results_long_liq)
    m_sl, t_sl, n_sl = tstat(results_short_liq)
    print(f"[7] Liquidation cascade: after LONG-liq flush 8h-fwd={m_ll*10000:.1f}bps (n={n_ll}, t={t_ll:.2f}, "
          f"expect positive=bounce) | after SHORT-liq squeeze 8h-fwd={m_sl*10000:.1f}bps "
          f"(n={n_sl}, t={t_sl:.2f}, expect negative=pullback)")


# ---------------------------------------------------------------------------
# 8. Open-interest + price divergence (trend confirmation filter)
# ---------------------------------------------------------------------------
def screen_oi_divergence():
    results_confirm, results_diverge = [], []
    for sym in CRYPTO_UNIVERSE:
        try:
            df = strat.load_binance(sym)
        except Exception:
            continue
        price_up = df["close"].pct_change(96) > 0
        oi_up = df["oi_change_pct"].rolling(96).sum() > 0
        fwd_ret = df["close"].pct_change(96).shift(-96)
        confirm = price_up & oi_up            # price up + OI up = healthy trend -> expect continuation
        diverge = price_up & ~oi_up           # price up but OI down = short-covering -> expect weaker/reversal
        results_confirm.extend(fwd_ret[confirm].dropna().tolist())
        results_diverge.extend(fwd_ret[diverge].dropna().tolist())
    m_c, t_c, n_c = tstat(results_confirm)
    m_d, t_d, n_d = tstat(results_diverge)
    print(f"[8] OI+price divergence: price_up & OI_up (confirm) 1d-fwd={m_c*10000:.1f}bps (n={n_c}, t={t_c:.2f}) | "
          f"price_up & OI_down (diverge) 1d-fwd={m_d*10000:.1f}bps (n={n_d}, t={t_d:.2f})")


# ---------------------------------------------------------------------------
# 9. Futures-spot basis mean reversion
# ---------------------------------------------------------------------------
def screen_basis_reversion():
    results_high, results_low = [], []
    for sym in CRYPTO_UNIVERSE:
        try:
            df = strat.load_binance(sym)
        except Exception:
            continue
        b = df["basis_index_bps"]
        b_z = (b - b.rolling(96 * 7).mean()) / b.rolling(96 * 7).std()
        fwd_ret = df["close"].pct_change(96).shift(-96)
        results_high.extend(fwd_ret[b_z > 2.0].dropna().tolist())
        results_low.extend(fwd_ret[b_z < -2.0].dropna().tolist())
    m_h, t_h, n_h = tstat(results_high)
    m_l, t_l, n_l = tstat(results_low)
    print(f"[9] Basis mean reversion: after basis extreme HIGH (rich future) 1d-fwd={m_h*10000:.1f}bps "
          f"(n={n_h}, t={t_h:.2f}, expect negative) | after extreme LOW (cheap future) 1d-fwd={m_l*10000:.1f}bps "
          f"(n={n_l}, t={t_l:.2f}, expect positive)")


if __name__ == "__main__":
    screen_fx_xs_momentum()
    screen_fx_ts_momentum()
    screen_turn_of_month()
    screen_day_of_week()
    screen_funding_contrarian()
    screen_ls_ratio_contrarian()
    screen_liq_cascade()
    screen_oi_divergence()
    screen_basis_reversion()
