"""
ingest_fundamentals.py — Per-name fundamentals snapshot.

Fields (via yfinance.Ticker.info — best-effort, NaN where missing):
  pe, ps, fcf_yield, debt_to_equity, gross_margin, roic, revenue_growth,
  earnings_date, dividend_yield, short_interest, beta

Cached to data/cache/fundamentals.parquet.

NOTE: yfinance.info is a static snapshot (current values). For a true
walk-forward backtest we'd want point-in-time fundamentals (Compustat, SEC
EDGAR). Phase-2 upgrade. For now we use the static snapshot as a "fundamental
score floor" filter — the GA gates entries on a fundamental score computed
from these values, accepting some lookahead bias documented here.
"""
from __future__ import annotations
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"


def _safe(d: dict, *keys, default=float("nan")):
    for k in keys:
        v = d.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                return v
    return default


def fetch_one(ticker: str) -> dict:
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}
    fcf = _safe(info, "freeCashflow")
    mcap = _safe(info, "marketCap")
    fcf_yield = (fcf / mcap) if (isinstance(fcf, float) and isinstance(mcap, float)
                                  and mcap and not (np.isnan(fcf) or np.isnan(mcap))) else float("nan")
    return {
        "ticker": ticker,
        "pe": _safe(info, "trailingPE", "forwardPE"),
        "ps": _safe(info, "priceToSalesTrailing12Months"),
        "fcf_yield": fcf_yield,
        "debt_to_equity": _safe(info, "debtToEquity"),
        "gross_margin": _safe(info, "grossMargins"),
        "roic": _safe(info, "returnOnAssets"),  # ROIC not available; use ROA proxy
        "revenue_growth": _safe(info, "revenueGrowth"),
        "dividend_yield": _safe(info, "dividendYield"),
        "short_interest": _safe(info, "shortPercentOfFloat"),
        "beta": _safe(info, "beta"),
        "market_cap": mcap,
    }


def fundamental_score(row: pd.Series) -> float:
    """
    Composite quality score 0..100. Higher = better.
    Rewards: positive FCF yield, gross margin > 30%, revenue growth > 0,
             ROIC/ROA > 5%, reasonable P/E (5-30), low debt/equity (<150).
    Penalizes: negative growth, high D/E, sky-high P/E.
    """
    score = 50.0
    fcf = row.get("fcf_yield", float("nan"))
    if pd.notna(fcf):
        score += np.clip(fcf * 200, -15, 15)  # 5% FCF yield -> +10
    gm = row.get("gross_margin", float("nan"))
    if pd.notna(gm):
        score += np.clip((gm - 0.30) * 30, -10, 10)
    rg = row.get("revenue_growth", float("nan"))
    if pd.notna(rg):
        score += np.clip(rg * 50, -15, 15)
    roic = row.get("roic", float("nan"))
    if pd.notna(roic):
        score += np.clip(roic * 100, -10, 10)
    pe = row.get("pe", float("nan"))
    if pd.notna(pe):
        if pe < 0:
            score -= 10
        elif pe < 30:
            score += 5
        elif pe > 60:
            score -= 5
    de = row.get("debt_to_equity", float("nan"))
    if pd.notna(de):
        if de < 50:
            score += 5
        elif de > 200:
            score -= 10
    return float(np.clip(score, 0, 100))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-names", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=0.25)
    args = ap.parse_args()

    uni_path = CACHE / "universe.parquet"
    if not uni_path.exists():
        print(f"[fund] missing {uni_path}", file=sys.stderr); sys.exit(2)
    uni = pd.read_parquet(uni_path)
    tickers = uni["ticker"].tolist()
    if args.smoke:
        tickers = ["SPY","QQQ","AAPL","MSFT","NVDA"]
    if args.max_names:
        tickers = tickers[: args.max_names]

    rows = []
    for i, t in enumerate(tickers):
        rows.append(fetch_one(t))
        if i % 25 == 0:
            print(f"[fund] {i+1}/{len(tickers)}", flush=True)
        time.sleep(args.sleep)
    df = pd.DataFrame(rows)
    df["fund_score"] = df.apply(fundamental_score, axis=1)
    out = CACHE / ("fundamentals_smoke.parquet" if args.smoke else "fundamentals.parquet")
    df.to_parquet(out, index=False)
    print(f"[fund] wrote {len(df)} -> {out}; mean fund_score={df['fund_score'].mean():.1f}")


if __name__ == "__main__":
    main()
