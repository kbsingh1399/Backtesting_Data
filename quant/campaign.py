"""
campaign.py — CLI entrypoint. Runs one sleeve's full WFO campaign, writes
results/{family}_records.json, {family}_equity.csv, {family}_equity.png,
and upserts results/scorecards.json.

Usage:
    python3 campaign.py B
    python3 campaign.py A --variant loose_gate_tp2R
"""
from __future__ import annotations
import sys
import os
import json
import argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from wfo import run_wfo, bh_curve, plot_report
import strategies as strat

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def trade_to_dict(t):
    return dict(symbol=t.symbol, side=int(t.side), signal_time=str(t.signal_time),
                entry_time=str(t.entry_time), entry_price=float(t.entry_price),
                exit_time=str(t.exit_time), exit_price=float(t.exit_price),
                exit_reason=t.exit_reason, bars_held=int(t.bars_held),
                size=float(t.size), gross_pnl=round(float(t.gross_pnl), 4),
                friction_cost=round(float(t.friction_cost), 4), net_pnl=round(float(t.net_pnl), 4),
                r_multiple=round(float(t.r_multiple), 4), ratcheted=bool(t.ratcheted))


GEOMETRY_GRID = [
    dict(),  # mandated baseline: SL1.5xATR/TP3.0R, ratchet 1.2->0.2, decay<0.2@24
    dict(ratchet_trigger_R=1.5, ratchet_lock_R=0.5),
    dict(tp_R=2.0),
    dict(ratchet_trigger_R=1.5, ratchet_lock_R=0.5, tp_R=2.0),
    dict(tp_R=1.5, ratchet_trigger_R=0.8, ratchet_lock_R=0.3),
    dict(tp_R=1.5),
]


def run_sleeve(sleeve_key: str, sim_kwargs=None, min_is_trades=6, is_lookback_quarters=2,
               score_fn="sum_r", variant_name=None, tag=None, require_positive_edge=True,
               search_geometry=False):
    cfg = strat.SLEEVES[sleeve_key]
    label = tag or cfg["name"]
    print(f"\n=== Running sleeve {sleeve_key} ({cfg['name']}) variant={variant_name or 'baseline'} ===")
    out = run_wfo(sleeve_key, sim_kwargs=sim_kwargs, min_is_trades=min_is_trades,
                   is_lookback_quarters=is_lookback_quarters, score_fn=score_fn,
                   variant_name=variant_name, require_positive_edge=require_positive_edge,
                   sim_kwargs_grid=GEOMETRY_GRID if search_geometry else None)
    stats = out["stats"]
    print(json.dumps(stats, indent=2, default=str))

    records_path = os.path.join(RESULTS_DIR, f"{label}_records.json")
    with open(records_path, "w") as f:
        json.dump([trade_to_dict(t) for t in out["trades"]], f, indent=2)

    eq_csv = os.path.join(RESULTS_DIR, f"{label}_equity.csv")
    out["equity_curve"].to_csv(eq_csv, index=False)

    start, end = out["oos_span"]
    bh_df = pd.DataFrame()
    if start is not None:
        bh_df = bh_curve(out["universe"], out["cache"], start, end)
        bh_df.to_csv(os.path.join(RESULTS_DIR, f"{label}_bh_equity.csv"), index=False)

    if len(bh_df):
        bh_final = bh_df["equity"].iloc[-1]
        stats["bh_roi_pct"] = round((bh_final - 5000.0) / 5000.0 * 100, 3)
    else:
        stats["bh_roi_pct"] = None

    png_path = os.path.join(RESULTS_DIR, f"{label}_equity.png")
    plot_report(stats, out["equity_curve"], bh_df, png_path,
                title=f"{cfg['name']} [{variant_name or 'baseline'}] — OOS stitched equity vs Buy&Hold")

    quarter_log_path = os.path.join(RESULTS_DIR, f"{label}_quarter_log.json")
    with open(quarter_log_path, "w") as f:
        json.dump(out["quarter_log"], f, indent=2, default=str)

    update_scorecard(label, stats)
    print(f"[{label}] wrote: {records_path}, {eq_csv}, {png_path}")
    return out


def update_scorecard(label, stats):
    path = os.path.join(RESULTS_DIR, "scorecards.json")
    data = {}
    if os.path.exists(path):
        with open(path) as f:
            try:
                data = json.load(f)
            except Exception:
                data = {}
    data[label] = stats
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sleeve", choices=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N"])
    ap.add_argument("--variant", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--min-is-trades", type=int, default=6)
    ap.add_argument("--is-lookback-quarters", type=int, default=2)
    ap.add_argument("--score-fn", default="sum_r")
    ap.add_argument("--allow-negative-is-edge", action="store_true")
    ap.add_argument("--search-geometry", action="store_true",
                     help="IS-lock geometry (ratchet/TP variants) jointly with signal params per quarter")
    ap.add_argument("--sl-atr-mult", type=float, default=None)
    ap.add_argument("--tp-r", type=float, default=None)
    ap.add_argument("--ratchet-trigger-r", type=float, default=None)
    ap.add_argument("--ratchet-lock-r", type=float, default=None)
    ap.add_argument("--time-decay-bars", type=int, default=None)
    ap.add_argument("--time-decay-r", type=float, default=None)
    ap.add_argument("--friction-bps", type=float, default=None,
                     help="Override round-trip friction bps (diagnostic only; mandate is 41bps "
                          "single-instrument / 82bps for 2-leg pairs sleeves).")
    args = ap.parse_args()

    sim_kwargs = {}
    if args.sl_atr_mult is not None:
        sim_kwargs["sl_atr_mult"] = args.sl_atr_mult
    if args.tp_r is not None:
        sim_kwargs["tp_R"] = args.tp_r
    if args.ratchet_trigger_r is not None:
        sim_kwargs["ratchet_trigger_R"] = args.ratchet_trigger_r
    if args.ratchet_lock_r is not None:
        sim_kwargs["ratchet_lock_R"] = args.ratchet_lock_r
    if args.time_decay_bars is not None:
        sim_kwargs["time_decay_bars"] = args.time_decay_bars
    if args.time_decay_r is not None:
        sim_kwargs["time_decay_R"] = args.time_decay_r
    if args.friction_bps is not None:
        sim_kwargs["friction_bps_roundtrip"] = args.friction_bps

    run_sleeve(args.sleeve, sim_kwargs=sim_kwargs or None, min_is_trades=args.min_is_trades,
               is_lookback_quarters=args.is_lookback_quarters, score_fn=args.score_fn,
               variant_name=args.variant, tag=args.tag,
               require_positive_edge=not args.allow_negative_is_edge,
               search_geometry=args.search_geometry)
