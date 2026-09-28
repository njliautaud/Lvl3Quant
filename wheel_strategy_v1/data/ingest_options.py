"""
ingest_options.py — SYNTHESIZE option-price features for the backtest.

Historical options chains are paid data. For this v1 we approximate option
prices via Black-Scholes using realized volatility per name as the IV input,
with an IV-rank feature derived from rolling 252d realized-vol percentile.

This is documented as an approximation — real options ingest (Tradier free
tier / Polygon trial / paid vendor) is a Phase-2 TODO.

Approximation details:
  - sigma  := realized vol over the trailing 20 trading days (annualized).
  - iv_rank := percentile of sigma over the trailing 252 trading days, in [0,1].
  - r := 0.04 (4% risk-free; approximates the 2020-2025 average; ok for relative pricing).
  - q := dividend_yield from fundamentals (0 if missing).
  - The wheel engine uses BS to price CSPs / CCs at the target delta on the fly.

Output: data/cache/iv_features.parquet (date, ticker, sigma, iv_rank, term_proxy)
"""
from __future__ import annotations
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    px_path = CACHE / ("prices_smoke.parquet" if args.smoke else "prices.parquet")
    if not px_path.exists():
        print(f"[opt] missing {px_path}", file=sys.stderr); sys.exit(2)
    px = pd.read_parquet(px_path)
    if "rv_20" not in px.columns:
        print("[opt] prices missing rv_20", file=sys.stderr); sys.exit(2)

    px = px.sort_values(["ticker", "date"]).copy()
    # sigma = trailing 20d realized vol (annualized) — already computed
    px["sigma"] = px["rv_20"].astype(float)
    # iv_rank = percentile of sigma over trailing 252 days within each ticker
    def _rank(s):
        return s.rolling(252, min_periods=20).rank(pct=True)
    px["iv_rank"] = px.groupby("ticker")["sigma"].transform(_rank)
    # term proxy: ratio of 60d realized vol to 20d realized vol
    if "rv_60" in px.columns:
        px["term_proxy"] = px["rv_60"] / px["sigma"]
    else:
        px["term_proxy"] = 1.0

    out_cols = ["date", "ticker", "sigma", "iv_rank", "term_proxy"]
    iv = px[out_cols].copy()
    out_path = CACHE / ("iv_features_smoke.parquet" if args.smoke else "iv_features.parquet")
    iv.to_parquet(out_path, index=False)
    print(f"[opt] wrote {len(iv)} rows -> {out_path}")
    print(f"[opt] mean sigma={iv['sigma'].mean():.3f}  mean iv_rank={iv['iv_rank'].mean():.3f}")


if __name__ == "__main__":
    main()
