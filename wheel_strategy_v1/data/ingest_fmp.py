"""
ingest_fmp.py — HC #544 R1/R2 — FMP-first ingest for the wheel pipeline.

Produces the same parquet artifacts that the yfinance-based ingest scripts
produce, but sourced from the local FMP archive. This means the downstream GA
+ backtest + report stack doesn't need to know we switched data sources —
it just reads the same parquets, now populated with higher-quality data.

Artifacts (written to data/cache/):
  universe.parquet          ticker, name, sector, source
  prices.parquet            ticker, date, open, high, low, close, volume, ret, log_ret, rv_20, rv_60, rv_252
  price_features.parquet    per-ticker static features for GA scoring
  fundamentals.parquet      point-in-time fundamentals snapshot (as_of = today)

Run:
  python -m data.ingest_fmp                  # full
  python -m data.ingest_fmp --start 2015-01-01
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CACHE = ROOT / "data" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

from data.fmp_loader import (  # noqa: E402
    fmp_available_tickers,
    load_universe_from_fmp,
    load_prices_fmp,
    load_fundamentals_pit,
)


def _per_ticker_features(px_long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t, g in px_long.groupby("ticker"):
        if len(g) < 30:
            continue
        g = g.sort_values("date").set_index("date")
        ret = g["ret"]
        log_ret = g["log_ret"]
        # Realized vol windows
        rv_20 = ret.rolling(20).std().iloc[-1] * np.sqrt(252) if len(ret) >= 20 else np.nan
        rv_60 = ret.rolling(60).std().iloc[-1] * np.sqrt(252) if len(ret) >= 60 else np.nan
        rv_252 = ret.rolling(252).std().iloc[-1] * np.sqrt(252) if len(ret) >= 252 else np.nan
        # Max DD over full window
        eq = (1.0 + ret.fillna(0)).cumprod()
        dd = (eq / eq.cummax()) - 1.0
        max_dd = float(-dd.min() * 100.0) if len(dd) else np.nan
        # ATR-ish: 14d avg true range %
        tr = pd.concat([
            (g["high"] - g["low"]).abs(),
            (g["high"] - g["close"].shift(1)).abs(),
            (g["low"] - g["close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr14_pct = float((tr.rolling(14).mean() / g["close"]).iloc[-1] * 100.0) if len(tr) >= 14 else np.nan
        # Gap stats (overnight)
        gap = (g["open"] - g["close"].shift(1)) / g["close"].shift(1)
        gap_mean_abs_pct = float(gap.abs().mean() * 100.0) if len(gap) else np.nan
        gap_max_abs_pct = float(gap.abs().max() * 100.0) if len(gap) else np.nan
        # Total returns
        def _tr_yrs(n_yr):
            need = int(252 * n_yr)
            if len(g) < need + 1:
                return np.nan
            p_start = g["close"].iloc[-need - 1]
            p_end = g["close"].iloc[-1]
            return float((p_end / p_start) - 1.0) * 100.0
        rows.append({
            "ticker": t,
            "rv_20": rv_20, "rv_60": rv_60, "rv_252": rv_252,
            "max_dd_pct": max_dd,
            "atr14_pct": atr14_pct,
            "gap_mean_abs_pct": gap_mean_abs_pct,
            "gap_max_abs_pct": gap_max_abs_pct,
            "tot_ret_1y_pct": _tr_yrs(1),
            "tot_ret_3y_pct": _tr_yrs(3),
            "tot_ret_5y_pct": _tr_yrs(5),
            "last_close": float(g["close"].iloc[-1]),
            "adv20_usd": float((g["close"] * g["volume"]).rolling(20).mean().iloc[-1]),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--min-adv-usd", type=float, default=50_000_000.0,
                    help="HC #544 R2: only optionable names with ADV >= this (in $)")
    ap.add_argument("--min-price", type=float, default=10.0,
                    help="HC #544 R2: minimum share price")
    args = ap.parse_args()

    print(f"[fmp] available tickers in archive: {len(fmp_available_tickers())}")

    # 1) UNIVERSE — HC #544 R2: SPY/QQQ/IWM constituents (intersected with archive)
    universe = load_universe_from_fmp(("sp500", "nasdaq", "dowjones"))
    print(f"[fmp] universe (constituent-driven): {len(universe)}")
    universe[["ticker", "name", "sector", "source"]].to_parquet(
        CACHE / "universe.parquet", index=False)

    # 2) PRICES — long-format daily OHLCV
    print(f"[fmp] loading prices for {len(universe)} tickers...")
    px = load_prices_fmp(universe["ticker"].tolist(),
                         start=args.start, end=args.end)
    if px.empty:
        print("[fmp] ERROR: no price rows loaded", file=sys.stderr)
        sys.exit(1)

    # Rolling realized vol on prices (long-format)
    px = px.sort_values(["ticker", "date"]).reset_index(drop=True)
    px["rv_20"] = px.groupby("ticker")["ret"].transform(
        lambda s: s.rolling(20).std() * np.sqrt(252))
    px["rv_60"] = px.groupby("ticker")["ret"].transform(
        lambda s: s.rolling(60).std() * np.sqrt(252))
    px["rv_252"] = px.groupby("ticker")["ret"].transform(
        lambda s: s.rolling(252).std() * np.sqrt(252))

    px.to_parquet(CACHE / "prices.parquet", index=False)
    print(f"[fmp] wrote prices.parquet: {px.shape}  "
          f"span {px['date'].min().date()} -> {px['date'].max().date()}")

    # 3) PRICE FEATURES (per-ticker static snapshot)
    pf = _per_ticker_features(px)
    pf.to_parquet(CACHE / "price_features.parquet", index=False)
    print(f"[fmp] wrote price_features.parquet: {pf.shape}")

    # 4) LIQUIDITY FILTER for the GA universe
    pf_liquid = pf[(pf["adv20_usd"].fillna(0) >= args.min_adv_usd) &
                   (pf["last_close"].fillna(0) >= args.min_price)].copy()
    print(f"[fmp] liquidity filter (ADV >= ${args.min_adv_usd/1e6:.0f}M, "
          f"price >= ${args.min_price:.0f}): "
          f"{len(pf_liquid)}/{len(pf)} pass")
    pf_liquid.to_parquet(CACHE / "price_features_liquid.parquet", index=False)

    # 5) FUNDAMENTALS — point-in-time as_of the last date in the price panel
    # (HC #544 R5: this is the as_of date for backtest entries on the final day;
    # the GA + wheel engine should call load_fundamentals_pit() PER ENTRY DATE
    # during the backtest to remain leakage-free. This parquet is the snapshot
    # for the current-deploy report.)
    as_of = px["date"].max()
    fund = load_fundamentals_pit(universe["ticker"].tolist(), as_of=as_of)
    fund.to_parquet(CACHE / "fundamentals.parquet", index=False)
    print(f"[fmp] wrote fundamentals.parquet as_of {as_of.date()}: {fund.shape}")

    # 6) Combined "GA-ready" name table for the wheel
    name_table = (universe[["ticker", "name", "sector"]]
                  .merge(pf_liquid[["ticker", "rv_20", "rv_60", "max_dd_pct",
                                    "atr14_pct", "last_close", "adv20_usd"]],
                         on="ticker", how="inner")
                  .merge(fund[["ticker", "pe", "fcf_yield", "gross_margin",
                               "debt_to_equity", "fund_score"]],
                         on="ticker", how="left"))
    name_table.to_parquet(CACHE / "ga_name_table.parquet", index=False)
    print(f"[fmp] wrote ga_name_table.parquet: {name_table.shape}")
    print("\nTop 12 by fund_score:")
    print(name_table.sort_values("fund_score", ascending=False)
          .head(12)[["ticker", "sector", "pe", "fcf_yield", "fund_score"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
