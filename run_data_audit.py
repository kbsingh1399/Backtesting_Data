"""
Data-integrity audit of Forex_Data/
===================================

Produces reports/data_integrity.csv and a printed summary.

Headline issue this audit exists to surface: the ``*_15m_real.parquet``
files are not pure 15-minute series.  Every one of them is prefixed with a
block of coarser bars - typically DAILY - carrying the same schema and the
same ``_15m_`` filename.  Any indicator computed over the whole file is
therefore computed across a resolution change, silently.

Run:
    .venv/bin/python run_data_audit.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from quantlab import config as C
from quantlab.datafeed import available_symbols, profile_file

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)


def main() -> None:
    C.REPORT_DIR.mkdir(exist_ok=True)
    rows = []
    syms = available_symbols("15m")
    print(f"profiling {len(syms)} instruments x 4 timeframes ...")
    for i, s in enumerate(syms, 1):
        for tf in ("15m", "1h", "4h", "d1"):
            p = profile_file(s, tf)
            if p:
                rows.append(p.as_dict())
        if i % 40 == 0:
            print(f"  {i}/{len(syms)}")

    df = pd.DataFrame(rows)
    for c in ("first", "last", "pure_start"):
        df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    df["padded_pct"] = (df["n_padded"] / df["n_bars"] * 100).round(1)
    # Share of the file's CALENDAR SPAN that is mis-sampled.  This is the
    # number that matters: the padded block is sparse in rows (daily bars)
    # but covers most of the years the file claims to provide.
    span = (df["last"] - df["first"]).dt.total_seconds()
    pad_span = (df["pure_start"] - df["first"]).dt.total_seconds()
    df["padded_span_pct"] = (pad_span / span * 100).round(1)
    df.to_csv(C.REPORT_DIR / "data_integrity.csv", index=False)

    print("\n" + "=" * 100)
    print("A.  RESOLUTION INTEGRITY  -  is a '15m' file actually 15m?")
    print("=" * 100)
    agg = df.groupby("timeframe").agg(
        files=("symbol", "size"),
        median_bars=("n_bars", "median"),
        median_padded_bars=("n_padded", "median"),
        median_padded_pct=("padded_pct", "median"),
        median_padded_SPAN_pct=("padded_span_pct", "median"),
        median_pure_start=("pure_start", "median"),
        files_with_padding=("n_padded", lambda s: int((s > 50).sum())),
    ).reindex(["15m", "1h", "4h", "d1"])
    print(agg.to_string())

    m15 = df[df["timeframe"] == "15m"].copy()
    mod = m15["padded_modal_seconds"]
    print(f"\n  modal spacing INSIDE the padded block of the 15m files:")
    print(f"    daily  (86400s): {int((mod == 86400).sum())} files")
    print(f"    hourly ( 3600s): {int((mod == 3600).sum())} files")
    print(f"    other          : {int((~mod.isin([86400, 3600])).sum())} files")
    print(f"  -> {int((mod.isin([86400,3600])).sum())}/{len(m15)} of the *_15m_*.parquet files begin "
          f"with DAILY or HOURLY bars wearing a 15m filename.")
    print(f"  -> that block is only {m15['padded_pct'].median():.1f}% of ROWS but "
          f"{m15['padded_span_pct'].median():.0f}% of the CALENDAR SPAN the file advertises.")
    print(f"\n  genuine 15m start dates:")
    print(m15["pure_start"].dt.to_period("Q").astype(str).value_counts().sort_index().to_string())
    print(f"\n  usable 15m bars per instrument: "
          f"p10={m15['n_pure'].quantile(.1):,.0f}  median={m15['n_pure'].median():,.0f}  "
          f"p90={m15['n_pure'].quantile(.9):,.0f}")
    print(f"  discarded as mis-sampled: {m15['n_padded'].sum():,.0f} bars "
          f"({m15['n_padded'].sum()/m15['n_bars'].sum()*100:.1f}% of all 15m rows)")

    print("\n" + "=" * 100)
    print("B.  OTHER CHECKS")
    print("=" * 100)
    print(f"  duplicate timestamps        : {int(df['dup_timestamps'].sum())} rows across all files")
    print(f"  OHLC consistency violations : {int(df['ohlc_violations'].sum())} bars "
          f"(high < max(o,c) or low > min(o,c)) - clamped on load")
    print(f"  files with zero spread > 50%: "
          f"{int((df[df.timeframe=='15m']['zero_spread_pct'] > 50).sum())} / {len(m15)} (15m)")
    print(f"  median zero-spread share    : {m15['zero_spread_pct'].median():.1f}% of 15m rows "
          f"(mostly inside the padded block; cost model falls back to the hour-of-day median)")

    print("\n" + "=" * 100)
    print("C.  IMPACT ON THE REFERENCE STRATEGY (s4_fvg_ml_strategy.py)")
    print("=" * 100)
    s4 = ["EURHUF", "GER40", "NICKEL", "USDSEK", "GAS", "AU200", "FR40", "EURCNH",
          "LEAD", "NZDUSD", "USDHKD", "US2000", "AUDCHF", "NZDCNH", "XAUCNH",
          "GAUCNH", "EURSEK", "EURUSD"]
    sub = m15[m15["symbol"].isin(s4)][
        ["symbol", "n_bars", "n_padded", "padded_pct", "padded_span_pct",
         "pure_start", "n_pure"]
    ].sort_values("padded_span_pct", ascending=False)
    print(sub.to_string(index=False))
    print(f"\n  S4's first out-of-sample window opens 2023-09-15.")
    late = sub[sub["pure_start"] > pd.Timestamp("2023-09-15", tz="UTC")]
    print(f"  {len(late)}/{len(sub)} of its assets have NO genuine 15m data at that point: "
          f"{', '.join(late['symbol'].tolist())}")
    print(f"  mean share of each S4 input file that is mis-sampled: "
          f"{sub['padded_pct'].mean():.1f}% of rows / {sub['padded_span_pct'].mean():.0f}% of calendar span")

    (C.REPORT_DIR / "data_integrity_summary.json").write_text(json.dumps({
        "files_profiled": int(len(df)),
        "m15_padded_bars_total": int(m15["n_padded"].sum()),
        "m15_padded_share_pct": round(float(m15["n_padded"].sum() / m15["n_bars"].sum() * 100), 2),
        "m15_median_pure_start": str(m15["pure_start"].median()),
        "m15_median_padded_span_pct": float(m15["padded_span_pct"].median()),
        "m15_padded_modal_seconds": float(m15["padded_modal_seconds"].median()),
        "dup_timestamps": int(df["dup_timestamps"].sum()),
        "ohlc_violations": int(df["ohlc_violations"].sum()),
        "s4_assets_without_15m_at_oos_start": late["symbol"].tolist(),
    }, indent=2))
    print(f"\nwrote {C.REPORT_DIR/'data_integrity.csv'}")


if __name__ == "__main__":
    main()
