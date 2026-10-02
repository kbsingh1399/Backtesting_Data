"""
S5 - Liquidity-Sweep Meta-Labelled Strategy : out-of-sample runner
==================================================================

Architecture (the same family as `s4_fvg_ml_strategy.py`, rebuilt so the
numbers mean something):

  1. Universe      integrity + cost + redundancy screen over Forex_Data
  2. Primary rule  causal liquidity-sweep events on genuine 15m bars
  3. Labels        triple barrier with gap-aware stops and the dataset's own
                   per-bar spread charged as cost
  4. Meta-model    purged, embargoed, anchored walk-forward ensemble that
                   decides which signals to take
  5. Portfolio     event-driven, concurrency-capped, equity-scaled risk
  6. Statistics    Sharpe / Sortino / Calmar / max DD, plus Deflated Sharpe
                   against the number of configurations actually tried

Everything that could be tuned was tuned on 2023-09-15..2024-06-30 and then
frozen (see `quantlab.config.FROZEN_PARAMS`).  The window reported below,
2024-07-01 onward, has never been used to make a decision.

Run:
    .venv/bin/python run_s5_strategy.py
"""
from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from quantlab import config as C
from quantlab.controls import control_trades, welch_diff
from quantlab.metrics import performance_summary, yearly_table
from quantlab.model import walk_forward_predict
from quantlab.pipeline import build_trades, get_enriched, summarise
from quantlab.portfolio import run_portfolio
from quantlab.reporting import tearsheet
from quantlab.signals import EVENT_FEATURES
from quantlab.universe import screen_universe

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)


def banner(t: str) -> None:
    print("\n" + "=" * 96)
    print(t)
    print("=" * 96)


def main() -> None:
    t0 = time.time()
    C.REPORT_DIR.mkdir(exist_ok=True)
    out: dict = {"generated": pd.Timestamp.now("UTC").isoformat()}

    banner("S5  |  LIQUIDITY-SWEEP META-LABELLED STRATEGY  |  OUT-OF-SAMPLE RUN")
    params = C.FROZEN_PARAMS
    print(f"frozen primary rule : {params.polarity}, TP={params.take_profit_r}R, "
          f"stop floor={params.min_stop_atr} ATR, time stop={params.time_stop_bars} bars")
    print(f"dev block (tuning)  : {C.WF_TRAIN_START} .. 2024-06-30   ({C.N_DEV_TRIALS} configs tried)")
    print(f"out-of-sample       : {C.WF_OOS_START} .. {C.WF_OOS_END}")

    # ---------------- 1. universe --------------------------------------
    banner("[1/6] UNIVERSE SCREEN")
    tbl, cms = screen_universe(verbose=True)
    tbl.to_csv(C.REPORT_DIR / "universe_screen.csv", index=False)
    keep = tbl[tbl["status"] == "keep"]
    symbols = keep["symbol"].tolist()
    clusters = {r.symbol: int(r.cluster) for r in keep.itertuples() if np.isfinite(r.cluster)}
    print(f"  {len(symbols)} instruments in {len(set(clusters.values()))} correlation clusters")
    print(f"  categories: {keep.groupby('category').size().to_dict()}")
    print(f"  median round-turn cost: {keep['rt_cost_bps'].median():.2f} bps "
          f"({keep['cost_to_R'].median():.3f} R at the frozen stop floor)")
    out["universe"] = {"n": len(symbols), "symbols": symbols,
                       "n_clusters": len(set(clusters.values()))}

    # ---------------- 2. primary rule ------------------------------------
    banner("[2/6] PRIMARY RULE  (sweep events + triple-barrier simulation)")
    trades = build_trades(symbols, params, cms)
    trades["cluster"] = trades["symbol"].map(clusters).fillna(-1).astype(int)
    print(f"  {len(trades):,} events, {trades['symbol'].nunique()} instruments, "
          f"{trades.entry_time.min().date()} .. {trades.entry_time.max().date()}")
    full = summarise(trades, "all events (dev + oos)")
    print(f"  all-sample  : {full}")

    oos_mask = trades["entry_time"] >= pd.Timestamp(C.WF_OOS_START, tz="UTC")
    oos_raw = trades[oos_mask]
    s_oos = summarise(oos_raw, "oos primary rule")
    print(f"  OOS raw rule: {s_oos}")
    out["primary_rule"] = {"all": full, "oos": s_oos}
    trades.to_parquet(C.REPORT_DIR / "s5_trades_raw.parquet", index=False)

    # ---- matched null: same barriers, same costs, no sweep information ----
    print("\n  matched control (random entries, identical barriers + costs):")
    enriched = {s: get_enriched(s) for s in symbols}
    counts = oos_raw.groupby("symbol").size().to_dict()
    ctrl = control_trades(enriched, cms, params, counts, seed=11)
    ctrl = ctrl[ctrl["entry_time"] >= pd.Timestamp(C.WF_OOS_START, tz="UTC")]
    s_ctrl = summarise(ctrl, "oos random control")
    print(f"  control     : {s_ctrl}")
    dg = welch_diff(oos_raw["r_gross"].to_numpy(), ctrl["r_gross"].to_numpy())
    dn = welch_diff(oos_raw["r_net"].to_numpy(), ctrl["r_net"].to_numpy())
    print(f"\n  RULE - CONTROL   gross {dg['diff']:+.4f}R (t={dg['t']:+.2f})   "
          f"net {dn['diff']:+.4f}R (t={dn['t']:+.2f})")
    print("  -> the rule can only claim credit for this difference, not for E[R] itself.")
    out["control"] = {"summary": s_ctrl, "gross_diff": dg, "net_diff": dn,
                      "rule_gross_ER": float(oos_raw["r_gross"].mean()),
                      "control_gross_ER": float(ctrl["r_gross"].mean())}

    # ---------------- 3. meta-model --------------------------------------
    banner("[3/6] META-LABELLING  (anchored, purged, embargoed walk-forward)")
    mp = C.ModelParams()
    feats = [f for f in EVENT_FEATURES if f in trades.columns]
    print(f"  {len(feats)} features | refit every {mp.refit_every_days}d | "
          f"embargo {mp.embargo_bars} bars | gate = train-quantile {mp.gate_quantile}")
    scored, folds, imp = walk_forward_predict(
        trades, feats, mp, C.WF_OOS_START, C.WF_OOS_END
    )
    print(f"  {len(folds)} folds fitted")
    fold_df = pd.DataFrame([dataclasses.asdict(f) for f in folds])
    if len(fold_df):
        print(fold_df[["fold", "test_start", "n_train", "n_test", "gate",
                       "train_base_rate"]].to_string(index=False))
        fold_df.to_csv(C.REPORT_DIR / "s5_folds.csv", index=False)
    imp.to_csv(C.REPORT_DIR / "s5_feature_importance.csv")
    print("\n  top features by mean gain:")
    print("   ", ", ".join(imp.head(10).index.tolist()))

    scored_oos = scored[scored["prob"].notna()].copy()
    if len(scored_oos):
        d = scored_oos.copy()
        d["decile"] = pd.qcut(d["prob"], 10, labels=False, duplicates="drop")
        dec = d.groupby("decile").agg(n=("r_net", "size"), mean_R=("r_net", "mean"),
                                      win=("label", "mean"))
        dec["mean_R"] = dec["mean_R"].round(4)
        dec["win"] = (dec["win"] * 100).round(1)
        print("\n  OOS mean net R by predicted-probability decile "
              "(monotonicity here is the whole thesis):")
        print(dec.to_string())
        dec.to_csv(C.REPORT_DIR / "s5_prob_deciles.csv")
        rho = d["prob"].corr(d["r_net"], method="spearman")
        print(f"\n  Spearman(prob, realised net R) = {rho:+.4f}  "
              f"(n={len(d):,})")
        out["meta_model"] = {"n_scored": int(len(d)), "spearman_prob_vs_R": float(rho),
                             "deciles": dec.reset_index().to_dict("records")}

    # ---------------- 4. portfolio ----------------------------------------
    banner("[4/6] PORTFOLIO SIMULATION")
    pp = C.PortfolioParams()
    print(f"  equity ${pp.initial_equity:,.0f} | risk {pp.risk_per_trade_pct}%/trade | "
          f"max {pp.max_concurrent_positions} concurrent, "
          f"{pp.max_concurrent_per_cluster}/cluster")

    oos_all = scored[scored["entry_time"] >= pd.Timestamp(C.WF_OOS_START, tz="UTC")]
    res_raw = run_portfolio(oos_all, pp, clusters, use_prob=False)
    res_ml = run_portfolio(oos_all, pp, clusters, use_prob=True)

    for name, res in (("A. primary rule, no ML gate", res_raw),
                      ("B. primary rule + ML gate", res_ml)):
        print(f"\n  {name}: {len(res.trades):,} trades taken, "
              f"skipped={res.skipped}")

    # ---------------- 5. statistics ----------------------------------------
    banner("[5/6] OUT-OF-SAMPLE PERFORMANCE")
    summaries = {}
    for name, res in (("primary_rule", res_raw), ("ml_gated", res_ml)):
        if res.daily.empty:
            continue
        s = performance_summary(res.returns, res.equity, res.trades,
                                n_trials=C.N_DEV_TRIALS)
        summaries[name] = s
        print(f"\n--- {name} ---")
        for k in ("start", "end", "n_trades", "total_return_pct", "cagr_pct",
                  "ann_vol_pct", "sharpe", "sortino", "max_drawdown_pct", "calmar",
                  "win_rate_pct", "avg_R", "profit_factor", "avg_cost_R", "t_stat_R",
                  "psr", "sr0_ann", "dsr", "sharpe_ci_lo", "sharpe_ci_hi"):
            if k in s and s[k] is not None:
                print(f"    {k:<22}: {s[k]}")
        yt = yearly_table(res.returns)
        if len(yt):
            print("\n  by year:")
            print("   " + yt.to_string().replace("\n", "\n   "))
    out["performance"] = summaries

    # ---------------- 6. artefacts ------------------------------------------
    banner("[6/6] ARTEFACTS")
    if not res_raw.daily.empty:
        p = tearsheet(res_raw.daily, res_raw.trades,
                      "S5 liquidity-sweep strategy - out-of-sample 2024-07 .. 2026-09",
                      C.REPORT_DIR / "s5_tearsheet.png",
                      gated=res_ml.daily if not res_ml.daily.empty else None)
        print(f"  {p}")
        res_raw.daily.to_csv(C.REPORT_DIR / "s5_daily_primary.csv")
        res_ml.daily.to_csv(C.REPORT_DIR / "s5_daily_mlgated.csv")
        res_raw.trades.to_csv(C.REPORT_DIR / "s5_trades_primary.csv", index=False)
        res_ml.trades.to_csv(C.REPORT_DIR / "s5_trades_mlgated.csv", index=False)

    (C.REPORT_DIR / "s5_results.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"  {C.REPORT_DIR/'s5_results.json'}")

    banner("VERDICT")
    a = summaries.get("primary_rule", {})
    b = summaries.get("ml_gated", {})
    print(f"  primary rule   Sharpe {a.get('sharpe')}, net R/trade {a.get('avg_R')}, "
          f"DSR {a.get('dsr')}")
    print(f"  + ML gate      Sharpe {b.get('sharpe')}, net R/trade {b.get('avg_R')}, "
          f"DSR {b.get('dsr')}")
    print(f"\n  elapsed {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
