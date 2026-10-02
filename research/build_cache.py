"""Pre-compute and cache the causal feature frame for the screened universe.

    .venv/bin/python research/build_cache.py

Writes .cache/<SYMBOL>_<version>.parquet (gitignored).  Safe to delete.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantlab import config as C          # noqa: E402
from quantlab.pipeline import get_enriched  # noqa: E402
from quantlab.universe import screen_universe  # noqa: E402


def main() -> None:
    t0 = time.time()
    tbl, cms = screen_universe(verbose=True)
    C.REPORT_DIR.mkdir(exist_ok=True)
    tbl.to_csv(C.REPORT_DIR / "universe_screen.csv", index=False)
    keep = tbl.loc[tbl["status"] == "keep", "symbol"].tolist()
    print(f"caching features for {len(keep)} instruments ...")
    for i, s in enumerate(keep, 1):
        df = get_enriched(s, use_cache=True)
        print(f"  [{i:>2}/{len(keep)}] {s:10s} {len(df):>7,} bars")
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
