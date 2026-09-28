"""
ingest_breadth.py — Market breadth: % of S&P 500 names above 200DMA.

Strategy:
  - Try to read /home/jupiter/Lvl3Quant/wheel_strategy_v1/data/cache/prices.parquet
    The wheel pipeline ingests ~329 large-caps with daily OHLCV — that's a usable
    breadth panel.
  - Compute, per date, the share of names whose close > 200d-rolling-mean.
  - If wheel prices.parquet is not yet present, write a NaN-only placeholder and
    log TODO so the GA can still run (the GA simply won't get a breadth signal).

Output: data/cache/breadth.parquet  (date, pct_above_200dma)
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
WHEEL_PRICES = ROOT.parent / "wheel_strategy_v1" / "data" / "cache" / "prices.parquet"
WHEEL_PRICES_SMOKE = ROOT.parent / "wheel_strategy_v1" / "data" / "cache" / "prices_smoke.parquet"


def _compute_breadth(prices_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(prices_path, columns=["ticker", "date", "close"])
    df["date"] = pd.to_datetime(df["date"])
    # Pivot wide; many tickers x dates. Memory-OK at ~330 names x ~2700 days.
    wide = df.pivot_table(index="date", columns="ticker", values="close", aggfunc="last").sort_index()
    dma200 = wide.rolling(200, min_periods=100).mean()
    above = (wide > dma200).astype(float)
    # only count where dma200 is defined and price is non-null
    valid = (~wide.isna()) & (~dma200.isna())
    pct = (above.where(valid).sum(axis=1) / valid.sum(axis=1)).rename("pct_above_200dma")
    out = pct.reset_index()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    sfx = "_smoke" if args.smoke else ""

    if args.smoke and WHEEL_PRICES_SMOKE.exists():
        src = WHEEL_PRICES_SMOKE
    elif WHEEL_PRICES.exists():
        src = WHEEL_PRICES
    elif WHEEL_PRICES_SMOKE.exists():
        src = WHEEL_PRICES_SMOKE
    else:
        src = None

    if src is None:
        # TODO: replace with cached breadth feed once wheel ingest finishes.
        print("[breadth] wheel prices.parquet not present yet; writing NaN placeholder.  # TODO", flush=True)
        spine = pd.DataFrame({"date": pd.bdate_range("2010-01-01", pd.Timestamp.today())})
        spine["pct_above_200dma"] = np.nan
        out = spine
    else:
        print(f"[breadth] computing from {src}", flush=True)
        out = _compute_breadth(src)

    path = CACHE / f"breadth{sfx}.parquet"
    out.to_parquet(path, index=False)
    print(f"[breadth] wrote {len(out)} rows -> {path}  "
          f"(non-null pct: {out['pct_above_200dma'].notna().sum()})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
