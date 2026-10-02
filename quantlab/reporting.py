"""quantlab.reporting - charts and markdown for the strategy report."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 8.5,
    "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "white",
})

INK = "#1b2a41"
ACCENT = "#c1554a"
MUTED = "#8c9bab"


def tearsheet(
    daily: pd.DataFrame,
    trades: pd.DataFrame,
    title: str,
    out_path: Path,
    gated: Optional[pd.DataFrame] = None,
    gated_label: str = "ML-gated",
) -> Path:
    fig = plt.figure(figsize=(11.5, 8.0))
    gs = fig.add_gridspec(3, 3, hspace=0.55, wspace=0.28,
                          height_ratios=[1.35, 1.0, 1.0])

    # --- equity -----------------------------------------------------------
    ax = fig.add_subplot(gs[0, :])
    if len(daily):
        base = daily["equity"].iloc[0]
        ax.plot(daily.index, daily["equity"] / base * 100 - 100,
                color=INK, lw=1.4, label="primary rule")
    if gated is not None and len(gated):
        b2 = gated["equity"].iloc[0]
        ax.plot(gated.index, gated["equity"] / b2 * 100 - 100,
                color=ACCENT, lw=1.4, label=gated_label)
    ax.axhline(0, color=MUTED, lw=0.8, ls="--")
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.set_ylabel("cumulative return (%)")
    ax.legend(frameon=False, fontsize=8, loc="best")

    # --- drawdown ---------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    if len(daily):
        eq = daily["equity"].to_numpy()
        dd = (eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq) * 100
        ax.fill_between(daily.index, dd, 0, color=INK, alpha=0.3)
        ax.plot(daily.index, dd, color=INK, lw=0.8)
    ax.set_title("drawdown (%)", loc="left", fontsize=9)
    ax.tick_params(axis="x", rotation=30, labelsize=7)

    # --- R distribution ----------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    if len(trades):
        ax.hist(trades["r_net"].clip(-2, 3), bins=45, color=INK, alpha=0.75)
        ax.axvline(0, color=ACCENT, lw=1.0)
        ax.axvline(trades["r_net"].mean(), color=ACCENT, ls="--", lw=1.0,
                   label=f"mean {trades['r_net'].mean():+.3f}R")
        ax.legend(frameon=False, fontsize=7)
    ax.set_title("net R per trade", loc="left", fontsize=9)

    # --- rolling 90d Sharpe -------------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    if len(daily) > 100:
        r = daily["ret"]
        rs = r.rolling(90).mean() / r.rolling(90).std(ddof=1) * np.sqrt(252)
        ax.plot(rs.index, rs, color=INK, lw=1.0)
        ax.axhline(0, color=ACCENT, lw=0.8, ls="--")
    ax.set_title("rolling 90d Sharpe", loc="left", fontsize=9)
    ax.tick_params(axis="x", rotation=30, labelsize=7)

    # --- monthly returns ----------------------------------------------------
    ax = fig.add_subplot(gs[2, 0])
    if len(daily):
        m = daily["ret"].resample("1ME").apply(lambda s: (1 + s).prod() - 1) * 100
        ax.bar(range(len(m)), m.to_numpy(),
               color=[INK if v >= 0 else ACCENT for v in m.to_numpy()])
        ax.set_xticks(range(0, len(m), max(len(m) // 6, 1)))
        ax.set_xticklabels([str(d.date())[:7] for d in m.index[::max(len(m) // 6, 1)]],
                           rotation=30, fontsize=7)
    ax.set_title("monthly return (%)", loc="left", fontsize=9)

    # --- probability decile diagnostic --------------------------------------
    ax = fig.add_subplot(gs[2, 1])
    if len(trades) and "prob" in trades and trades["prob"].notna().sum() > 100:
        t = trades.dropna(subset=["prob"]).copy()
        t["dec"] = pd.qcut(t["prob"], 10, labels=False, duplicates="drop")
        g = t.groupby("dec")["r_net"].mean()
        ax.bar(g.index, g.to_numpy(),
               color=[INK if v >= 0 else ACCENT for v in g.to_numpy()])
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xlabel("meta-model probability decile", fontsize=7)
    ax.set_title("mean net R by model decile", loc="left", fontsize=9)

    # --- per-instrument -----------------------------------------------------
    ax = fig.add_subplot(gs[2, 2])
    if len(trades) and "symbol" in trades:
        g = trades.groupby("symbol")["pnl" if "pnl" in trades else "r_net"].sum().sort_values()
        g = pd.concat([g.head(8), g.tail(8)])
        ax.barh(range(len(g)), g.to_numpy(),
                color=[ACCENT if v < 0 else INK for v in g.to_numpy()])
        ax.set_yticks(range(len(g)))
        ax.set_yticklabels(g.index, fontsize=6.5)
    ax.set_title("best / worst instruments", loc="left", fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def research_figure(
    controls: pd.DataFrame,
    lag_decay: pd.DataFrame,
    subuniverse: pd.DataFrame,
    out_path: Path,
) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.1))

    ax = axes[0]
    if len(controls):
        ax.bar(controls["bucket"], controls["E_R_gross"],
               color=[ACCENT if v < 0 else INK for v in controls["E_R_gross"]])
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.tick_params(axis="x", rotation=20, labelsize=7)
    ax.set_title("1. sweep rule vs matched controls\n(gross E[R], dev block)",
                 loc="left", fontsize=9)
    ax.set_ylabel("E[R] gross")

    ax = axes[1]
    if len(lag_decay):
        for h, grp in lag_decay.groupby("horizon_h"):
            ax.plot(grp["lag_bars"], grp["gross_bps"], marker="o", lw=1.3,
                    label=f"{h}h horizon")
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.legend(frameon=False, fontsize=7)
    ax.set_xlabel("implementation lag (hours)")
    ax.set_ylabel("gross bps / rebalance")
    ax.set_title("2. reversal alpha vs execution lag", loc="left", fontsize=9)

    ax = axes[2]
    if len(subuniverse):
        ax.barh(subuniverse["universe"], subuniverse["net_sharpe"],
                color=[ACCENT if v < 0 else INK for v in subuniverse["net_sharpe"]])
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.tick_params(labelsize=7)
    ax.set_xlabel("net Sharpe")
    ax.set_title("3. VWAP reversion by sub-universe", loc="left", fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def fmt_table(d: Dict, keys: Optional[List[str]] = None) -> str:
    keys = keys or list(d.keys())
    rows = ["| metric | value |", "|---|---|"]
    for k in keys:
        if k in d and d[k] is not None:
            v = d[k]
            v = f"{v:,.4g}" if isinstance(v, float) else str(v)
            rows.append(f"| {k} | {v} |")
    return "\n".join(rows)
