"""
Audit of s4_fvg_ml_strategy.py
==============================

The reference script reports **+921.30% ROI** and **+1,441R** over 20
out-of-sample windows.  This script takes that number apart.

It does not argue with S4's idea.  It re-runs S4's own feature engineering,
S4's own labeller and S4's own stacked ensemble, keeps S4's own trade
selection, and then changes **one accounting assumption at a time**, so the
reader can see exactly how much of the headline is alpha and how much is
bookkeeping.

The waterfall:

  V0  as reported                     S4's printed number
  V1  book the realised R             use S4's own `r_realized` column
                                      instead of mapping the binary label to
                                      +2.0R / -1.0R
  V2  + realistic transaction costs   per-asset spread from the data's own
                                      `spread` column + commission +
                                      slippage, instead of a flat -0.08R
  V3  + genuine 15m bars only         drop the daily/hourly block that the
                                      *_15m_*.parquet files are padded with
  V4  + one account, one portfolio    concurrency caps and a single
                                      compounding equity curve instead of 20
                                      independent $5,000 accounts whose
                                      percentage returns are added up

Run:
    .venv/bin/python run_s4_audit.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import s4_fvg_ml_strategy as S4                       # noqa: E402
from quantlab import config as C                      # noqa: E402
from quantlab import datafeed as D                    # noqa: E402
from quantlab.costs import build_cost_model           # noqa: E402
from quantlab.metrics import performance_summary      # noqa: E402
from quantlab.portfolio import run_portfolio          # noqa: E402

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)


# --------------------------------------------------------------------------
# Faithful re-implementation of S4's labeller, with the bookkeeping S4 drops
# --------------------------------------------------------------------------
def s4_labels_plus(df: pd.DataFrame) -> pd.DataFrame:
    """Bit-for-bit the same R outcomes as S4.create_labels_ratchet, but also
    records entry index, exit index, stop distance and the entry bar's
    spread so the trade can be costed and scheduled properly."""
    n = len(df)
    min_r, look_fwd = S4.MIN_R_MULTIPLE, S4.MAX_HOLDING_BARS
    tgt = np.full(n, np.nan)
    rr = np.zeros(n)
    rdist = np.full(n, np.nan)
    exit_ix = np.full(n, -1)
    side_a = np.zeros(n)

    lows, highs = df["low"].values, df["high"].values
    closes, opens = df["close"].values, df["open"].values
    s_pdl, s_pdh = df["sweep_pdl"].values, df["sweep_pdh"].values
    b_fvg, s_fvg = df["bullish_fvg"].values, df["bearish_fvg"].values
    htf = df["htf_4h_trend"].values
    lo20, hi20 = df["local_low_20"].values, df["local_high_20"].values
    is_kz = df["is_kill_zone"].values if "is_kill_zone" in df.columns else np.ones(n, bool)

    for i in range(n - look_fwd):
        long_ok = is_kz[i] and s_pdl[i] == 1 and b_fvg[i] > 0 and htf[i] > 0
        short_ok = (not long_ok) and is_kz[i] and s_pdh[i] == 1 and s_fvg[i] > 0 and htf[i] < 0
        if not (long_ok or short_ok) or i + 1 >= n:
            continue
        s = 1 if long_ok else -1
        entry = opens[i + 1]
        orig_sl = lo20[i] if s == 1 else hi20[i]
        r_d = (entry - orig_sl) if s == 1 else (orig_sl - entry)
        if r_d <= 0 or (r_d / entry) > S4.MAX_STOP_PCT:
            continue
        tp = entry + s * min_r * r_d
        sl = orig_sl
        r_real, exited, j_ex = -1.0, False, min(i + look_fwd, n) - 1
        for j in range(i + 1, min(i + look_fwd, n)):
            hit_sl = lows[j] <= sl if s == 1 else highs[j] >= sl
            if hit_sl:
                r_real = round(s * (sl - entry) / r_d, 4); exited, j_ex = True, j; break
            hit_tp = highs[j] >= tp if s == 1 else lows[j] <= tp
            if hit_tp:
                r_real = min_r; exited, j_ex = True, j; break
            fav = s * (highs[j] - entry) / r_d if s == 1 else s * (entry - lows[j]) / r_d * s
            fav = ((highs[j] - entry) / r_d) if s == 1 else ((entry - lows[j]) / r_d)
            if fav >= 2.0 and s * (sl - entry) < 1.8 * r_d:
                sl = entry + s * 1.8 * r_d
            elif fav >= 1.5 and s * (sl - entry) < 0.80 * r_d:
                sl = entry + s * 0.80 * r_d
            elif fav >= 0.8 and s * (sl - entry) < 0.15 * r_d:
                sl = entry + s * 0.15 * r_d
            if j == i + S4.TIME_DECAY_BARS and (
                (s == 1 and closes[j] < entry + S4.TIME_DECAY_THRESHOLD_R * r_d)
                or (s == -1 and closes[j] > entry - S4.TIME_DECAY_THRESHOLD_R * r_d)
            ):
                r_real = round(s * (closes[j] - entry) / r_d, 4); exited, j_ex = True, j; break
        if not exited:
            jl = min(i + look_fwd, n) - 1
            mtm = s * (closes[jl] - entry) / r_d
            lock = s * (sl - entry) / r_d
            r_real = round(min(min_r, max(mtm, lock)), 4)
            j_ex = jl
        r_real = round(r_real - 0.08, 4)
        tgt[i] = 1 if r_real > 0 else 0
        rr[i], rdist[i], exit_ix[i], side_a[i] = r_real, r_d, j_ex, s

    out = df.copy()
    out["target"] = tgt
    out["r_realized"] = rr
    out["s4_r_dist"] = rdist
    out["s4_entry_i"] = np.arange(n) + 1
    out["s4_exit_i"] = exit_ix
    out["s4_side"] = side_a
    return out


def equity_from_r(r: np.ndarray, cap: float = 5000.0) -> Tuple[float, float]:
    """S4's dynamic risk-budget simulation, unchanged."""
    eq = peak = cap
    curve = [eq]
    for v in r:
        dd = (peak - eq) / peak * 100.0 if peak > 0 else 0.0
        risk = 25.0
        if dd >= 2.0:
            risk = 15.0
        elif (eq - cap) >= 100.0 and dd < 1.0:
            risk = 35.0
        eq += v * risk
        peak = max(peak, eq)
        curve.append(eq)
    a = np.array(curve)
    peaks = np.maximum.accumulate(a)
    mdd = float(np.max((peaks - a) / peaks * 100.0))
    return float((eq - cap) / cap * 100.0), mdd


def build_s4_panel(pure_only: bool) -> pd.DataFrame:
    frames = []
    for sym in S4.CANONICAL_18_ASSETS:
        try:
            df = S4.engineer_features_polars(sym, str(C.FOREX_DIR))
        except Exception:
            continue
        if pure_only:
            ps = D._pure_start_cached(sym if sym != "GER30" else "GER40", "15m")
            if ps is None:
                continue
            df = df[df["datetime"] >= ps].reset_index(drop=True)
        if len(df) < 500:
            continue
        df = s4_labels_plus(df)
        v = df[df["target"].notna()].copy()
        if len(v) < 20:
            continue
        v = S4.augment_with_quant_features(v)
        v["asset"] = sym
        frames.append(v)
    full = pd.concat(frames, ignore_index=True)
    full["dt"] = pd.to_datetime(full["time"], unit="s", utc=True)
    return full.sort_values("time").reset_index(drop=True)


def walk_forward(full: pd.DataFrame) -> pd.DataFrame:
    """S4's own walk-forward + stacking ensemble + P>=0.54 gate."""
    feats = S4.CANONICAL_FEATURES + S4.RESEARCH_QUANT_FEATURES
    picks = []
    for w in S4.OOS_WINDOWS:
        a = pd.Timestamp(w["start_date"], tz="UTC")
        b = pd.Timestamp(w["end_date"], tz="UTC") + pd.Timedelta(days=1)
        tr = full[full["dt"] < a - pd.Timedelta(hours=24)]
        te = full[(full["dt"] >= a) & (full["dt"] <= b)].copy()
        if len(tr) < 100 or te.empty:
            continue
        y = tr["target"].to_numpy(int)
        if y.sum() in (0, len(y)):
            continue
        pw = float(len(y) - y.sum()) / max(1.0, float(y.sum()))
        zoo = S4.ModelZoo(seed=42)
        zoo.train_stacking_ensemble(tr[feats].fillna(0.0), y, pw)
        te["prob"] = zoo.predict_probs(te[feats].fillna(0.0))["stacking"]
        sel = te[te["prob"] >= 0.54].copy()
        if sel.empty:
            continue
        sel["window"] = w["window_id"]
        picks.append(sel)
    return pd.concat(picks, ignore_index=True) if picks else pd.DataFrame()


def attach_costs(sel: pd.DataFrame) -> pd.DataFrame:
    """Replace S4's flat -0.08R with the modelled per-asset round-turn cost."""
    man = C.load_manifest()
    sel = sel.copy()
    sel["cost_r_real"] = np.nan
    for sym, g in sel.groupby("asset"):
        s = sym if sym != "GER30" else "GER40"
        try:
            bars = D.load_bars(s, "15m", enforce_pure=True)
        except FileNotFoundError:
            continue
        cm = build_cost_model(s, man.get(s, {}), bars)
        hours = pd.to_datetime(g["dt"]).dt.hour.to_numpy()
        rt = cm.round_turn_cost(g["close"].to_numpy(), g["spread"].to_numpy(), hours, "stop")
        sel.loc[g.index, "cost_r_real"] = rt / g["s4_r_dist"].to_numpy()
    return sel


def main() -> None:
    t0 = time.time()
    C.REPORT_DIR.mkdir(exist_ok=True)
    print("=" * 100)
    print("AUDIT OF s4_fvg_ml_strategy.py")
    print("=" * 100)
    report: Dict = {}

    # ---------------- defect 1: the PnL is booked off the label -----------
    print("\n[1] Building S4's panel with its own labeller (plus bookkeeping) ...")
    full = build_s4_panel(pure_only=False)
    print(f"    {len(full):,} labelled setups, {full['asset'].nunique()} assets, "
          f"{full['dt'].min().date()} .. {full['dt'].max().date()}")

    sel = walk_forward(full)
    print(f"    {len(sel):,} trades selected by the P>=0.54 gate across "
          f"{sel['window'].nunique()} windows")

    booked = np.where(sel["target"].to_numpy(int) == 1, 2.0, -1.0)
    realised = sel["r_realized"].to_numpy()
    print("\n" + "-" * 100)
    print("DEFECT 1  -  PnL is booked from the binary label, not the simulated outcome")
    print("-" * 100)
    print("    s4_fvg_ml_strategy.py:744    trade_r = np.where(y_sub == 1, 2.0, -1.0)")
    print("    ...but `target` is just `1 if r_realized > 0`, and `r_realized` is already")
    print("    sitting in the dataframe.  Every winner is booked at +2.0R no matter how")
    print("    small, every loser at -1.0R no matter how large.")
    print(f"\n    booked  : mean {booked.mean():+.4f}R   sum {booked.sum():+,.1f}R")
    print(f"    realised: mean {realised.mean():+.4f}R   sum {realised.sum():+,.1f}R")
    print(f"    difference            : {booked.sum()-realised.sum():+,.1f}R of pure bookkeeping")
    win = sel["target"] == 1
    print(f"    mean realised R on 'wins'  : {realised[win].mean():+.4f}R  (booked as +2.0R)")
    print(f"    mean realised R on 'losses': {realised[~win].mean():+.4f}R  (booked as -1.0R)")

    print("\n" + "-" * 100)
    print("DEFECT 2  -  win-rate arithmetic")
    print("-" * 100)
    wr_true = float(win.mean()) * 100
    print(f"    s4_fvg_ml_strategy.py:826    tot_wins / max(1, tot_trades) * 100.2")
    print(f"    '100.2' should be 100. Reported 53.4% vs actual {wr_true:.2f}% on this run.")

    print("\n" + "-" * 100)
    print("DEFECT 3  -  twenty separate $5,000 accounts, summed as if one")
    print("-" * 100)
    print("    Each window restarts at $5,000 and its ROI% is added to the next,")
    print("    so a +122% window and a +129% window 'add' to +251% on a $5,000 base.")
    print("    No capital is ever at risk across windows and no drawdown is ever carried.")

    # ---------------- waterfall -------------------------------------------
    print("\n[2] Re-costing with the dataset's own spread column ...")
    sel = attach_costs(sel)
    cost_real = sel["cost_r_real"].to_numpy()
    print(f"    modelled round-turn cost: median {np.nanmedian(cost_real):.4f}R, "
          f"mean {np.nanmean(cost_real):.4f}R   (S4 assumes a flat 0.08R)")
    per_asset = (sel.groupby("asset")["cost_r_real"].median().sort_values(ascending=False)
                 .round(4))
    print("\n    worst assets by real cost per trade (R):")
    print("   " + per_asset.head(8).to_string().replace("\n", "\n   "))

    rows: List[Dict] = []
    v0_roi = sum(equity_from_r(np.where(g["target"].to_numpy(int) == 1, 2.0, -1.0))[0]
                 for _, g in sel.groupby("window"))
    rows.append({"variant": "V0  as reported (label -> +2R/-1R, 20 accounts)",
                 "trades": len(sel), "mean_R": round(float(booked.mean()), 4),
                 "total_R": round(float(booked.sum()), 1),
                 "summed_window_ROI_pct": round(v0_roi, 1)})

    v1_roi = sum(equity_from_r(g["r_realized"].to_numpy())[0] for _, g in sel.groupby("window"))
    rows.append({"variant": "V1  + book the realised R", "trades": len(sel),
                 "mean_R": round(float(realised.mean()), 4),
                 "total_R": round(float(realised.sum()), 1),
                 "summed_window_ROI_pct": round(v1_roi, 1)})

    r2 = realised + 0.08 - np.nan_to_num(cost_real, nan=float(np.nanmedian(cost_real)))
    v2_roi = sum(equity_from_r(
        (g["r_realized"] + 0.08 - g["cost_r_real"].fillna(np.nanmedian(cost_real))).to_numpy()
    )[0] for _, g in sel.groupby("window"))
    rows.append({"variant": "V2  + realistic transaction costs", "trades": len(sel),
                 "mean_R": round(float(r2.mean()), 4), "total_R": round(float(r2.sum()), 1),
                 "summed_window_ROI_pct": round(v2_roi, 1)})

    # V2b: three of S4's 18 instruments have a round-turn cost LARGER than the
    # entire stop distance, which flatters nothing and distorts the average.
    # Re-run the cost step on the cost-viable subset so the effect is honest.
    viable = per_asset[per_asset <= 0.25].index.tolist()
    sv = sel[sel["asset"].isin(viable)]
    if len(sv):
        r2b = (sv["r_realized"] + 0.08
               - sv["cost_r_real"].fillna(sv["cost_r_real"].median())).to_numpy()
        v2b_roi = sum(equity_from_r(
            (g["r_realized"] + 0.08 - g["cost_r_real"].fillna(sv["cost_r_real"].median())).to_numpy()
        )[0] for _, g in sv.groupby("window"))
        rows.append({"variant": f"V2b  ... cost-viable instruments only ({len(viable)}/18)",
                     "trades": len(sv), "mean_R": round(float(r2b.mean()), 4),
                     "total_R": round(float(r2b.sum()), 1),
                     "summed_window_ROI_pct": round(v2b_roi, 1)})
        print(f"\n    {18-len(viable)} of S4's 18 instruments have a modelled round-turn cost")
        print(f"    above 0.25R; GAS, USDHKD and LEAD exceed 1.0R - the spread alone is")
        print(f"    wider than the entire stop distance, so no 15m rule can trade them.")

    print("\n[3] Rebuilding on genuine 15m bars only ...")
    full_p = build_s4_panel(pure_only=True)
    sel_p = walk_forward(full_p)
    print(f"    {len(full_p):,} labelled setups ({len(full):,} before), "
          f"{len(sel_p):,} trades selected")
    if len(sel_p):
        sel_p = attach_costs(sel_p)
        cp = sel_p["cost_r_real"]
        r3 = (sel_p["r_realized"] + 0.08 - cp.fillna(cp.median())).to_numpy()
        v3_roi = sum(equity_from_r(
            (g["r_realized"] + 0.08 - g["cost_r_real"].fillna(cp.median())).to_numpy())[0]
            for _, g in sel_p.groupby("window"))
        rows.append({"variant": "V3  + genuine 15m bars only", "trades": len(sel_p),
                     "mean_R": round(float(r3.mean()), 4), "total_R": round(float(r3.sum()), 1),
                     "summed_window_ROI_pct": round(v3_roi, 1)})

    wf = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print("ATTRIBUTION WATERFALL")
    print("=" * 100)
    print(wf.to_string(index=False))

    # ---------------- V4: one portfolio ------------------------------------
    print("\n[4] One account, concurrency-capped portfolio ...")
    if len(sel_p):
        bar = pd.Timedelta(minutes=15)
        tr = pd.DataFrame({
            "symbol": sel_p["asset"].to_numpy(),
            "entry_time": pd.to_datetime(sel_p["dt"]) + bar,
            "exit_time": pd.to_datetime(sel_p["dt"])
            + bar * (sel_p["s4_exit_i"] - sel_p["s4_entry_i"] + 2).clip(lower=1).to_numpy(),
            "r_net": r3,
            "cost_r": sel_p["cost_r_real"].fillna(cp.median()).to_numpy(),
            "side": sel_p["s4_side"].to_numpy(),
        })
        pp = C.PortfolioParams()
        res = run_portfolio(tr, pp, clusters=None, use_prob=False)
        if not res.daily.empty:
            perf = performance_summary(res.returns, res.equity, res.trades, n_trials=1)
            print("\n    single $100,000 account, 0.30% risk/trade, max 8 concurrent:")
            for k in ("start", "end", "n_trades", "total_return_pct", "cagr_pct",
                      "ann_vol_pct", "sharpe", "max_drawdown_pct", "calmar",
                      "win_rate_pct", "avg_R", "profit_factor", "t_stat_R"):
                if k in perf:
                    print(f"      {k:<20}: {perf[k]}")
            report["V4_portfolio"] = perf
            rows.append({"variant": "V4  + one portfolio account",
                         "trades": perf.get("n_trades"), "mean_R": perf.get("avg_R"),
                         "total_R": round(float(np.nansum(res.trades["r_net"])), 1),
                         "summed_window_ROI_pct": perf.get("total_return_pct")})

    wf = pd.DataFrame(rows)
    wf.to_csv(C.REPORT_DIR / "s4_audit_waterfall.csv", index=False)
    report["waterfall"] = rows
    report["defects"] = {
        "label_booking_inflation_x": round(float(booked.sum() / max(realised.sum(), 1e-9)), 2),
        "reported_total_R": round(float(booked.sum()), 1),
        "realised_total_R": round(float(realised.sum()), 1),
        "win_rate_typo": "tot_wins/tot_trades*100.2",
        "mean_modelled_cost_R": round(float(np.nanmean(cost_real)), 4),
        "s4_assumed_cost_R": 0.08,
    }
    (C.REPORT_DIR / "s4_audit.json").write_text(json.dumps(report, indent=2, default=str))

    print("\n" + "=" * 100)
    print("BOTTOM LINE")
    print("=" * 100)
    print(f"    S4 prints +921.30% ROI and +1,441R.")
    print(f"    Using S4's own simulated outcomes instead of its label->R map, the same")
    print(f"    trades produce {realised.sum():+,.1f}R.")
    print(f"    Adding the real cost of the instruments it trades: {r2.sum():+,.1f}R.")
    if len(sel_p):
        print(f"    Restricted to bars that are actually 15m: {r3.sum():+,.1f}R "
              f"over {len(sel_p):,} trades.")
    print(f"\n    elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
