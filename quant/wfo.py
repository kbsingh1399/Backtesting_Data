"""
wfo.py — Walk-forward orchestration: IS parameter selection -> locked OOS
quarters -> portfolio-level execution (shared $5k equity, <=3 concurrent,
$225 hard DD halt) -> stitched equity curve -> Buy&Hold benchmark -> report.

Approximation disclosed up front: the hard DD halt stops *new* entries for
the remainder of the run once breached, but does not retroactively force an
intrabar flatten of trades already open at the breach instant (those run to
their pre-computed, price-action-only exit). This is a standard WFO
simplification; it is conservative in spirit (halt still stops all further
risk-taking) and is noted in every scorecard.
"""
from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
from dataclasses import dataclass

from engine import generate_trades, DEFAULT_SIM_KWARGS
import strategies as strat

DD_LIMIT = 225.0
START_EQUITY = 5000.0
MAX_CONCURRENT = 3


class SigCache:
    """Caches prepped frames and per-param-combo (entries, sides, atr) so the
    WFO loop never recomputes indicators while scanning quarters."""

    def __init__(self, sleeve_key: str):
        self.sleeve_key = sleeve_key
        self.cfg = strat.SLEEVES[sleeve_key]
        self.prepped = {}   # symbol -> prepped frame (or tuple of frames)
        self.sig_cache = {}  # (symbol, param_key) -> (entries, sides, atr, df)

    def _df_for_sig(self, symbol):
        p = self.prepped[symbol]
        return p[1] if isinstance(p, tuple) else p

    def prep_all(self):
        for sym in self.cfg["universe"]:
            try:
                self.prepped[sym] = self.cfg["prep"](sym)
            except Exception as e:
                print(f"  [warn] prep failed for {sym}: {e}")
        return self

    def get_signals(self, symbol, params: dict):
        key = (symbol, tuple(sorted(params.items())))
        if key in self.sig_cache:
            return self.sig_cache[key]
        df = self._df_for_sig(symbol)
        entries, sides, atr = self.cfg["sig"](df, params)
        self.sig_cache[key] = (entries, sides, atr, df)
        return self.sig_cache[key]

    def common_range(self):
        lo, hi = None, None
        for sym in self.cfg["universe"]:
            if sym not in self.prepped:
                continue
            df = self._df_for_sig(sym)
            if len(df) == 0:
                continue
            a, b = df.index.min(), df.index.max()
            lo = a if lo is None or a > lo else lo  # want latest start (intersection)
            hi = b if hi is None or b < hi else hi  # earliest end
        return lo, hi


def make_quarters(lo: pd.Timestamp, hi: pd.Timestamp):
    """Non-overlapping calendar quarters fully inside [lo, hi]."""
    qs = []
    start = pd.Timestamp(year=lo.year, month=((lo.month - 1) // 3) * 3 + 1, day=1, tz="UTC")
    if start < lo:
        start = start + pd.offsets.QuarterBegin(startingMonth=1)
    cur = start
    while True:
        q_end = cur + pd.offsets.QuarterBegin(startingMonth=1)
        if q_end > hi:
            break
        qs.append((cur, q_end))
        cur = q_end
    return qs


def _score_params(trades, min_is_trades, score_fn="sum_r", require_positive_edge=True):
    n = len(trades)
    if n < min_is_trades:
        return None
    total_r = sum(t.r_multiple for t in trades)
    if require_positive_edge and total_r <= 0:
        # IS gate: never lock in a parameter set that shows a demonstrably
        # negative in-sample expectancy -- that is not "selection", that's
        # picking the least-bad way to lose money. No edge -> no trade.
        return None
    if score_fn == "sum_r":
        return total_r
    if score_fn == "sharpe_like":
        rs = np.array([t.r_multiple for t in trades])
        return rs.mean() / (rs.std() + 1e-9) * np.sqrt(n)
    return total_r


def run_wfo(sleeve_key: str, sim_kwargs: dict | None = None, min_is_trades: int = 6,
            is_lookback_quarters: int = 2, score_fn: str = "sum_r",
            variant_name: str | None = None, verbose: bool = True,
            require_positive_edge: bool = True, sim_kwargs_grid: list[dict] | None = None):
    """If sim_kwargs_grid is given, geometry variants are searched and LOCKED
    by the in-sample selector jointly with the signal-parameter grid (one
    combined grid search per quarter) -- this is the methodologically sound
    way to "iterate geometry via WFO": the IS data picks the geometry, never
    a human peeking at OOS results. `sim_kwargs` (if given without a grid)
    is treated as a fixed override applied uniformly (used for quick fixed
    stress tests, not for a certifiable run)."""
    cfg = strat.SLEEVES[sleeve_key]
    trade_fn = cfg.get("trade_fn", generate_trades)
    sleeve_default_sim_kwargs = {}
    if "friction_bps_roundtrip" in cfg:
        sleeve_default_sim_kwargs["friction_bps_roundtrip"] = cfg["friction_bps_roundtrip"]
    base_sim_kwargs = {**sleeve_default_sim_kwargs, **(sim_kwargs or {})}
    geom_grid = sim_kwargs_grid if sim_kwargs_grid else [{}]

    cache = SigCache(sleeve_key).prep_all()
    lo, hi = cache.common_range()
    if verbose:
        print(f"[{cfg['name']}] data range: {lo} -> {hi}")
    quarters = make_quarters(lo, hi)
    n_is = is_lookback_quarters
    oos_quarters = quarters[n_is:]
    min_oos_start = cfg.get("min_oos_start")
    if min_oos_start is not None:
        oos_quarters = [q for q in oos_quarters if q[0] >= min_oos_start]
        if verbose:
            print(f"[{cfg['name']}] restricting OOS to >= {min_oos_start.date()} "
                  f"(pair/feature-selection cutoff) -> {len(oos_quarters)} quarters")
    if verbose:
        print(f"[{cfg['name']}] {len(oos_quarters)} candidate OOS quarters (need >=20)")

    all_candidates = []
    quarter_log = []
    for qi, (q_start, q_end) in enumerate(oos_quarters):
        is_start = q_start - pd.DateOffset(months=3 * n_is)
        is_end = q_start

        best_params, best_geom, best_score, best_is_trades = None, None, -np.inf, 0
        for params in cfg["grid"]:
            for geom in geom_grid:
                sk = {**base_sim_kwargs, **geom}
                is_trades = []
                for sym in cfg["universe"]:
                    if sym not in cache.prepped:
                        continue
                    entries, sides, atr, df = cache.get_signals(sym, params)
                    try:
                        si = df.index.searchsorted(is_start)
                        ei = df.index.searchsorted(is_end)
                    except Exception:
                        continue
                    trades = trade_fn(df, entries, sides, atr, sym, {**params, **sk},
                                       start_idx=si, end_idx=ei)
                    is_trades.extend(trades)
                score = _score_params(is_trades, min_is_trades, score_fn, require_positive_edge)
                if score is not None and score > best_score:
                    best_score, best_params, best_geom, best_is_trades = score, params, geom, len(is_trades)

        traded = best_params is not None
        q_trades = []
        if traded:
            sk = {**base_sim_kwargs, **best_geom}
            for sym in cfg["universe"]:
                if sym not in cache.prepped:
                    continue
                entries, sides, atr, df = cache.get_signals(sym, best_params)
                si = df.index.searchsorted(q_start)
                ei = df.index.searchsorted(q_end)
                trades = trade_fn(df, entries, sides, atr, sym, {**best_params, **sk},
                                   start_idx=si, end_idx=ei)
                q_trades.extend(trades)
                all_candidates.extend(trades)
        quarter_log.append(dict(quarter=f"{q_start.date()}", locked_params=best_params,
                                 locked_geometry=best_geom,
                                 is_score=None if best_score == -np.inf else round(best_score, 3),
                                 is_trades=best_is_trades, oos_trades=len(q_trades), traded=traded))

    # ---- portfolio-level execution: concurrency cap + hard DD halt ----
    result = simulate_portfolio(all_candidates)

    windows_traded = sum(1 for q in quarter_log if q["traded"] and q["oos_trades"] > 0)
    stats = compute_stats(result["taken_trades"], result["equity_curve"], len(oos_quarters), windows_traded)
    stats["sleeve"] = cfg["name"]
    stats["variant"] = variant_name or "baseline"
    stats["n_oos_quarters"] = len(oos_quarters)
    stats["windows_traded"] = windows_traded
    stats["halted"] = result["halted"]
    stats["halt_time"] = str(result["halt_time"]) if result["halt_time"] else None
    stats["sim_kwargs"] = {**DEFAULT_SIM_KWARGS, **base_sim_kwargs}
    stats["geometry_grid_searched_is"] = sim_kwargs_grid if sim_kwargs_grid else None
    stats["min_is_trades_gate"] = min_is_trades
    stats["require_positive_is_edge"] = require_positive_edge

    return dict(stats=stats, trades=result["taken_trades"], equity_curve=result["equity_curve"],
                quarter_log=quarter_log, universe=cfg["universe"], cache=cache,
                oos_span=(oos_quarters[0][0] if oos_quarters else None,
                          oos_quarters[-1][1] if oos_quarters else None))


def simulate_portfolio(candidates):
    """Sweep-line portfolio sim with a correct event order even for
    zero-duration trades (entry_time == exit_time, e.g. an immediate
    same-bar ratchet stop-out). We never interleave a trade's own exit
    before its own entry: trades are only closed out of the open-heap when
    a LATER (or equal) candidate's entry_time is reached, or at the final
    flush -- so a same-bar in/out trade still occupies a slot for exactly
    one comparison step, never forever."""
    import heapq

    candidates_sorted = sorted(range(len(candidates)), key=lambda i: candidates[i].entry_time)

    equity = START_EQUITY
    peak = START_EQUITY
    halted = False
    halt_time = None
    open_heap = []  # (exit_time, i)
    taken_trades = []
    equity_curve = [(candidates[0].entry_time if candidates else pd.Timestamp.utcnow(), equity)]

    def close_due(cutoff_time):
        nonlocal equity, peak, halted, halt_time
        while open_heap and open_heap[0][0] <= cutoff_time:
            exit_time, i = heapq.heappop(open_heap)
            t = candidates[i]
            equity += t.net_pnl
            peak = max(peak, equity)
            equity_curve.append((exit_time, equity))
            if not halted and (peak - equity) > DD_LIMIT:
                halted = True
                halt_time = exit_time

    for i in candidates_sorted:
        t = candidates[i]
        close_due(t.entry_time)
        if halted:
            continue
        if len(open_heap) >= MAX_CONCURRENT:
            continue
        heapq.heappush(open_heap, (t.exit_time, i))
        taken_trades.append(t)

    # final flush: close out anything still open at the end of the run
    while open_heap:
        exit_time, i = heapq.heappop(open_heap)
        t = candidates[i]
        equity += t.net_pnl
        peak = max(peak, equity)
        equity_curve.append((exit_time, equity))
        if not halted and (peak - equity) > DD_LIMIT:
            halted = True
            halt_time = exit_time

    equity_curve_df = pd.DataFrame(equity_curve, columns=["time", "equity"]).sort_values("time").drop_duplicates("time")
    return dict(taken_trades=taken_trades, equity_curve=equity_curve_df, halted=halted, halt_time=halt_time,
                final_equity=equity, peak_equity=peak)


def compute_stats(trades, equity_curve_df, n_oos_quarters, windows_traded):
    n = len(trades)
    if n == 0:
        return dict(n_trades=0, roi_pct=0.0, final_equity=START_EQUITY, max_dd_dollars=0.0,
                    win_rate=0.0, payoff_ratio=float("nan"), sharpe=0.0, avg_r=0.0)
    net_pnls = np.array([t.net_pnl for t in trades])
    rs = np.array([t.r_multiple for t in trades])
    wins = net_pnls[net_pnls > 0]
    losses = net_pnls[net_pnls <= 0]
    win_rate = len(wins) / n
    avg_win = wins.mean() if len(wins) else 0.0
    avg_loss = abs(losses.mean()) if len(losses) else np.nan
    payoff = (avg_win / avg_loss) if (avg_loss and not np.isnan(avg_loss) and avg_loss > 0) else float("nan")
    final_equity = START_EQUITY + net_pnls.sum()
    roi_pct = (final_equity - START_EQUITY) / START_EQUITY * 100
    eq = equity_curve_df["equity"].to_numpy()
    run_peak = np.maximum.accumulate(eq)
    max_dd_dollars = float((run_peak - eq).max()) if len(eq) else 0.0
    sharpe = (rs.mean() / (rs.std() + 1e-9)) * np.sqrt(n) if n > 1 else 0.0
    return dict(n_trades=n, roi_pct=round(roi_pct, 3), final_equity=round(final_equity, 2),
                max_dd_dollars=round(max_dd_dollars, 2), win_rate=round(win_rate, 4),
                payoff_ratio=round(payoff, 3) if payoff == payoff else None,
                sharpe=round(sharpe, 3), avg_r=round(rs.mean(), 4),
                gross_pnl=round(sum(t.gross_pnl for t in trades), 2),
                total_friction=round(sum(t.friction_cost for t in trades), 2))


def bh_curve(universe, cache: SigCache, start, end):
    """Equal-weight Buy&Hold benchmark over the stitched OOS span."""
    closes = {}
    for sym in universe:
        if sym not in cache.prepped:
            continue
        df = cache._df_for_sig(sym)
        c = df["close"]
        c = c[(c.index >= start) & (c.index <= end)]
        if len(c) < 2:
            continue
        closes[sym] = c
    if not closes:
        return pd.DataFrame(columns=["time", "equity"])
    idx = sorted(set().union(*[set(c.index) for c in closes.values()]))
    idx = pd.DatetimeIndex(idx)
    panel = pd.DataFrame({s: c.reindex(idx).ffill() for s, c in closes.items()})
    panel = panel.dropna(how="all")
    panel = panel.ffill().bfill()
    weight_dollars = START_EQUITY / panel.shape[1]
    units = weight_dollars / panel.iloc[0]
    equity = (panel * units).sum(axis=1)
    return pd.DataFrame({"time": equity.index, "equity": equity.values})


def plot_report(stats, equity_curve_df, bh_df, out_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6))
    if len(equity_curve_df):
        ax.plot(equity_curve_df["time"], equity_curve_df["equity"], label="Strategy (OOS stitched)", color="#1f77b4", linewidth=1.6)
    if len(bh_df):
        ax.plot(bh_df["time"], bh_df["equity"], label="Buy & Hold (equal-weight universe)", color="#888888", linewidth=1.2, linestyle="--")
    ax.axhline(START_EQUITY, color="black", linewidth=0.6, linestyle=":")
    ax.set_title(title)
    ax.set_ylabel("Equity ($)")
    ax.legend(loc="best")
    txt = (f"ROI {stats.get('roi_pct')}%  |  Trades {stats.get('n_trades')}  |  "
           f"Sharpe {stats.get('sharpe')}  |  Payoff {stats.get('payoff_ratio')}  |  "
           f"MaxDD ${stats.get('max_dd_dollars')}  |  Windows traded {stats.get('windows_traded')}/{stats.get('n_oos_quarters')}")
    ax.text(0.01, -0.12, txt, transform=ax.transAxes, fontsize=9, va="top")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
