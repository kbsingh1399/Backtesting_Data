"""
Phase 7 data-quality scan — before building any strategy, check every one of
the 98 FX/metals/indices/energy symbols x 4 timeframes for the same defect
found in Phase 5/6 (several symbols' "intraday" files are actually daily bars
repeated/mislabeled before some cutoff date). We compute bars/calendar-year
and flag the first year where bar density looks genuinely intraday, so every
downstream WFO sleeve can set a correct per-symbol `min_oos_start`.

Output: results/PHASE7_data_quality.json
"""
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import strategies as strat

# Expected bars/year if genuinely that granularity (FX trades ~5.5 days/week,
# ~24h/day during the week -> use generous thresholds with margin for
# holidays/weekends/broker downtime).
EXPECTED_BARS_PER_YEAR = {
    "15m": 96 * 5.5 * 52 * 0.5,   # ~half of theoretical max as a loose "genuinely intraday" floor
    "1h": 24 * 5.5 * 52 * 0.5,
    "4h": 6 * 5.5 * 52 * 0.5,
    "d1": 5 * 52 * 0.5,
}

FOREX_DIR = strat.FOREX_DIR


def get_universe():
    manifest = json.load(open(os.path.join(FOREX_DIR, "manifest.json")))
    return sorted(manifest["symbols"].keys())


def scan_symbol_tf(symbol, tf):
    try:
        df = strat.load_forex(symbol, tf)
    except Exception as e:
        return {"error": str(e)}
    if len(df) == 0:
        return {"error": "empty"}
    by_year = df.groupby(df.index.year).size()
    threshold = EXPECTED_BARS_PER_YEAR[tf]
    genuine_years = by_year[by_year >= threshold]
    first_genuine_year = int(genuine_years.index.min()) if len(genuine_years) else None
    return {
        "n_rows": int(len(df)),
        "start": str(df.index.min()),
        "end": str(df.index.max()),
        "bars_by_year": {int(y): int(c) for y, c in by_year.items()},
        "first_genuine_intraday_year": first_genuine_year,
    }


def main():
    universe = get_universe()
    print(f"Scanning {len(universe)} symbols x 4 timeframes = {len(universe)*4} series...")
    out = {}
    for i, sym in enumerate(universe):
        out[sym] = {}
        for tf in ["15m", "1h", "4h", "d1"]:
            out[sym][tf] = scan_symbol_tf(sym, tf)
        if (i + 1) % 20 == 0:
            print(f"  ...{i+1}/{len(universe)}")

    os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
    out_path = os.path.join(os.path.dirname(__file__), "results", "PHASE7_data_quality.json")
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"Wrote {out_path}")

    # Summary: how many symbols need a cutoff > 2020 for each timeframe (i.e.
    # are affected by the mislabeling defect in any meaningful way)
    for tf in ["15m", "1h", "4h", "d1"]:
        affected = []
        for sym in universe:
            fy = out[sym][tf].get("first_genuine_intraday_year")
            if fy is not None and fy > 2020:
                affected.append((sym, fy))
        print(f"\n[{tf}] {len(affected)}/{len(universe)} symbols genuinely intraday only from >2020:")
        for sym, fy in sorted(affected, key=lambda x: -x[1])[:15]:
            print(f"   {sym}: {fy}")


if __name__ == "__main__":
    main()
