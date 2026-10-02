"""
quantlab.controls
=================
Matched null models.

A backtest number is meaningless without knowing what the *same* machinery
produces on an input that cannot contain alpha.  Two facts make this
essential here:

  * bar-discretised triple barriers are not a fair game even on a random
    walk - the pessimistic within-bar tie-break, gap-through-at-open fills
    and the weekend flatten all bleed a little; and
  * the cost model subtracts a further ~0.06-0.10 R per round turn.

So the honest question is never "is E[R] > 0" but "is E[R] better than a
random entry with the same session structure, the same barriers and the same
costs".  That difference is the only thing the rule can claim credit for.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import StrategyParams
from .costs import CostModel
from .labeling import simulate_events
from .signals import _session_mask


def random_entry_events(
    enr: pd.DataFrame,
    params: StrategyParams,
    n_target: int,
    seed: int = 0,
    mode: str = "random",
) -> pd.DataFrame:
    """Entries with no sweep information but the same session/hour mix."""
    rng = np.random.default_rng(seed)
    sess = _session_mask(enr, params)
    a_all = enr["atr14"].to_numpy()
    ok = np.where(sess & np.isfinite(a_all) & (a_all > 0))[0]
    ok = ok[(ok > 300) & (ok < len(enr) - 2)]
    if len(ok) == 0 or n_target <= 0:
        return pd.DataFrame()
    if mode == "random":
        idx = np.sort(rng.choice(ok, size=min(n_target, len(ok)), replace=False))
    else:
        step = max(1, len(ok) // n_target)
        idx = ok[::step]
    side = rng.choice([-1, 1], size=len(idx))
    a = a_all[idx]
    o = enr["open"].to_numpy()[idx + 1]
    lo, hi = enr["low"].to_numpy()[idx], enr["high"].to_numpy()[idx]
    struct = np.where(side == 1, o - (lo - params.stop_buffer_atr * a),
                      (hi + params.stop_buffer_atr * a) - o)
    return pd.DataFrame({
        "i": idx, "entry_i": idx + 1,
        "signal_time": enr["datetime"].to_numpy()[idx],
        "entry_time": enr["datetime"].to_numpy()[idx + 1],
        "side": side, "atr": a, "struct_r_atr": struct / a,
    })


def control_trades(
    enriched: Dict[str, pd.DataFrame],
    cost_models: Dict[str, CostModel],
    params: StrategyParams,
    counts: Dict[str, int],
    seed: int = 0,
    mode: str = "random",
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for k, (sym, enr) in enumerate(enriched.items()):
        cm = cost_models.get(sym)
        if cm is None or counts.get(sym, 0) <= 0:
            continue
        ev = random_entry_events(enr, params, counts[sym], seed=seed + k, mode=mode)
        if ev.empty:
            continue
        tr = simulate_events(enr, ev, params, cm)
        tr["symbol"] = sym
        frames.append(tr)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("entry_time").reset_index(drop=True)


def welch_diff(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """Difference in means with a Welch standard error."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return {"diff": float(d), "se": float(se), "t": float(d / se) if se > 0 else np.nan}
