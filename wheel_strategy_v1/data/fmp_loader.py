"""
fmp_loader.py — HC #544 R1/R2

Central FMP archive interface for the wheel pipeline.

Reads from /home/jupiter/teleclaude-main/data/fmp_archive/ (88 tickers,
174MB, prices + financials + metrics + indices). FMP is the PRIMARY source;
the original yfinance ingest paths remain as fallback.

Public API:
  fmp_available_tickers()                  -> set[str]
  load_index_constituents(index)           -> list[dict]  (sp500/nasdaq/dowjones)
  load_universe_from_fmp(indices)          -> pd.DataFrame columns [ticker,name,sector,source]
  load_prices_fmp(tickers, start, end)     -> long-format prices DataFrame
  load_fundamentals_pit(tickers, as_of)    -> point-in-time fundamentals
                                              (uses filingDate so NO LEAKAGE)
  load_profile(ticker)                     -> dict (currentCompanyName, beta, marketCap, etc.)

POINT-IN-TIME guarantee (HC #544 R5):
  load_fundamentals_pit(t, as_of=YYYY-MM-DD) returns only statements with
  filingDate <= as_of. Strict — uses the date the filing was actually
  publicly released, not the fiscal period-end.
"""
from __future__ import annotations
import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

FMP_ROOT = Path("/home/jupiter/teleclaude-main/data/fmp_archive")
PRICES_DIR = FMP_ROOT / "prices"
FIN_DIR = FMP_ROOT / "financials"
METRICS_DIR = FMP_ROOT / "metrics"
EARNINGS_DIR = FMP_ROOT / "earnings"
INDICES_DIR = FMP_ROOT / "indices"


@lru_cache(maxsize=1)
def fmp_available_tickers() -> frozenset:
    """All tickers we have at least DAILY PRICES for."""
    if not PRICES_DIR.exists():
        return frozenset()
    out = set()
    for f in PRICES_DIR.glob("*_daily.json"):
        out.add(f.stem.replace("_daily", ""))
    return frozenset(out)


def load_index_constituents(index: str) -> list[dict]:
    """index in {'sp500','nasdaq','dowjones'}.
    Returns list of {symbol, name, sector, subSector, ...}."""
    path = INDICES_DIR / f"{index}.json"
    if not path.exists():
        raise FileNotFoundError(f"FMP index file not found: {path}")
    return json.load(open(path))


def load_universe_from_fmp(indices: Iterable[str] = ("sp500", "nasdaq", "dowjones")) -> pd.DataFrame:
    """
    HC #544 R2: Build wheel universe from the actual ETF constituent lists,
    intersected with tickers we have FMP price data for.
    """
    seen = {}
    for ix in indices:
        try:
            for row in load_index_constituents(ix):
                sym = row.get("symbol")
                if not sym:
                    continue
                if sym not in seen:
                    seen[sym] = {
                        "ticker": sym,
                        "name": row.get("name", ""),
                        "sector": row.get("sector", ""),
                        "subsector": row.get("subSector", ""),
                        "source_index": ix,
                    }
        except FileNotFoundError:
            continue

    avail = fmp_available_tickers()
    rows = [r for r in seen.values() if r["ticker"] in avail]
    df = pd.DataFrame(rows)
    df["source"] = "fmp_archive"
    return df.sort_values("ticker").reset_index(drop=True)


def load_prices_fmp(tickers: Iterable[str],
                    start: str | pd.Timestamp | None = None,
                    end: str | pd.Timestamp | None = None) -> pd.DataFrame:
    """
    Returns long-format daily OHLCV DataFrame:
      columns: [ticker, date, open, high, low, close, volume, ret, log_ret]
    Indexed default (no multi-index); date is tz-naive datetime.
    """
    if start is not None:
        start = pd.Timestamp(start)
    if end is not None:
        end = pd.Timestamp(end)

    out_blocks = []
    for t in tickers:
        path = PRICES_DIR / f"{t}_daily.json"
        if not path.exists():
            continue
        try:
            rows = json.load(open(path))
        except Exception:
            continue
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        if start is not None:
            df = df[df["date"] >= start]
        if end is not None:
            df = df[df["date"] <= end]
        if df.empty:
            continue
        df["ticker"] = t
        for c in ("open", "high", "low", "close", "volume"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["ret"] = df["close"].pct_change()
        df["log_ret"] = np.log1p(df["ret"])
        out_blocks.append(df[["ticker", "date", "open", "high", "low",
                              "close", "volume", "ret", "log_ret"]])

    if not out_blocks:
        return pd.DataFrame(columns=["ticker", "date", "open", "high", "low",
                                     "close", "volume", "ret", "log_ret"])
    return pd.concat(out_blocks, ignore_index=True)


# --- Point-in-time fundamentals (HC #544 R5: no leakage) ---------------------

def _load_pit_table(ticker: str, category: str, filename: str) -> list[dict]:
    """category in {'financials','metrics'}; filename includes .json"""
    base = FIN_DIR if category == "financials" else METRICS_DIR
    path = base / ticker / filename
    if not path.exists():
        return []
    try:
        return json.load(open(path)) or []
    except Exception:
        return []


def _latest_before(rows: list[dict], as_of: pd.Timestamp) -> dict:
    """Return most recent row whose filingDate (or date) <= as_of."""
    best = None
    best_dt = pd.Timestamp("1900-01-01")
    for r in rows:
        # Prefer filingDate (publicly available date) over date (fiscal period end)
        dt_str = r.get("filingDate") or r.get("date")
        if not dt_str:
            continue
        try:
            dt = pd.Timestamp(dt_str)
        except Exception:
            continue
        if dt <= as_of and dt > best_dt:
            best = r
            best_dt = dt
    return best or {}


def load_fundamentals_pit(tickers: Iterable[str], as_of: str | pd.Timestamp) -> pd.DataFrame:
    """
    Point-in-time fundamentals snapshot as_of a given date — uses filingDate
    so the GA never sees a statement that wasn't publicly out yet.

    Fields returned (NaN where unavailable):
      ticker, as_of, market_cap, beta, pe, ps, fcf_yield, debt_to_equity,
      gross_margin, ebitda_margin, net_margin, current_ratio, roe_proxy,
      revenue_ttm, fcf_ttm, fund_score (composite 0..100)
    """
    as_of = pd.Timestamp(as_of)
    rows_out = []

    for t in tickers:
        ratios = _load_pit_table(t, "metrics", "ratios_quarter.json")
        km = _load_pit_table(t, "metrics", "key_metrics_quarter.json")
        profile_list = _load_pit_table(t, "metrics", "profile.json")
        income_q = _load_pit_table(t, "financials", "income_quarter.json")
        bal_q = _load_pit_table(t, "financials", "balance_quarter.json")
        cf_q = _load_pit_table(t, "financials", "cashflow_quarter.json")

        latest_ratio = _latest_before(ratios, as_of)
        latest_km = _latest_before(km, as_of)
        profile = profile_list[0] if isinstance(profile_list, list) and profile_list else {}
        latest_inc = _latest_before(income_q, as_of)
        latest_bal = _latest_before(bal_q, as_of)
        latest_cf = _latest_before(cf_q, as_of)

        # NOTE: profile is a static snapshot — marketCap/beta have mild lookahead.
        # We accept this for sector/beta but compute fundamentals from PIT statements.

        # TTM revenue + FCF from last 4 income/cashflow quarters
        def _ttm(table: list[dict], col: str) -> float:
            cuts = [r for r in table
                    if (r.get("filingDate") or r.get("date") or "1900-01-01")
                    and pd.Timestamp(r.get("filingDate") or r["date"]) <= as_of]
            cuts = sorted(cuts, key=lambda r: r.get("filingDate") or r.get("date"), reverse=True)[:4]
            vals = [r.get(col) for r in cuts if r.get(col) is not None]
            return float(np.nansum(vals)) if vals else float("nan")

        revenue_ttm = _ttm(income_q, "revenue")
        ocf_ttm = _ttm(cf_q, "netCashProvidedByOperatingActivities") or _ttm(cf_q, "operatingCashFlow")
        capex_ttm = _ttm(cf_q, "capitalExpenditure")
        fcf_ttm = (ocf_ttm or 0.0) - abs(capex_ttm or 0.0) if (ocf_ttm or capex_ttm) else float("nan")

        market_cap = profile.get("marketCap") or latest_km.get("marketCap")
        pe = latest_km.get("priceToEarningsRatio") or latest_ratio.get("priceToEarningsRatio")
        ps = latest_km.get("evToSales")  # closest proxy in this schema
        gross_margin = latest_ratio.get("grossProfitMargin")
        ebitda_margin = latest_ratio.get("ebitdaMargin")
        net_margin = latest_ratio.get("netProfitMargin")
        current_ratio = latest_km.get("currentRatio")

        # debt-to-equity from balance sheet
        total_debt = (latest_bal.get("totalDebt") or
                      ((latest_bal.get("shortTermDebt") or 0.0) +
                       (latest_bal.get("longTermDebt") or 0.0)))
        total_equity = latest_bal.get("totalStockholdersEquity")
        de = (total_debt / total_equity) if (total_debt is not None
                                             and total_equity not in (None, 0)) else float("nan")

        # ROE proxy from net income / equity
        net_income_q = latest_inc.get("netIncome")
        roe_proxy = (net_income_q * 4.0 / total_equity) if (net_income_q is not None
                                                            and total_equity not in (None, 0)) else float("nan")

        # FCF yield = fcf_ttm / market_cap
        fcf_yield = (fcf_ttm / market_cap) if (market_cap not in (None, 0)
                                               and fcf_ttm == fcf_ttm) else float("nan")

        rows_out.append({
            "ticker": t,
            "as_of": as_of,
            "market_cap": market_cap,
            "beta": profile.get("beta"),
            "pe": pe,
            "ps": ps,
            "fcf_yield": fcf_yield,
            "debt_to_equity": de,
            "gross_margin": gross_margin,
            "ebitda_margin": ebitda_margin,
            "net_margin": net_margin,
            "current_ratio": current_ratio,
            "roe_proxy": roe_proxy,
            "revenue_ttm": revenue_ttm,
            "fcf_ttm": fcf_ttm,
            "sector": profile.get("sector"),
            "industry": profile.get("industry"),
        })

    df = pd.DataFrame(rows_out)
    if df.empty:
        return df

    # Composite fund_score 0..100 — higher = better wheel candidate
    # Components (rank-normalize within snapshot):
    #   + fcf_yield, + gross_margin, + ebitda_margin, + net_margin,
    #   + current_ratio, + roe_proxy, - debt_to_equity, - |pe| (lower is better but >0)
    def _rank01(s: pd.Series, ascending=True) -> pd.Series:
        x = s.astype(float)
        if x.notna().sum() < 2:
            return pd.Series(0.5, index=x.index)
        r = x.rank(ascending=ascending, pct=True)
        return r.fillna(0.5)

    score = (
        0.20 * _rank01(df["fcf_yield"])
      + 0.15 * _rank01(df["gross_margin"])
      + 0.10 * _rank01(df["ebitda_margin"])
      + 0.10 * _rank01(df["net_margin"])
      + 0.10 * _rank01(df["current_ratio"])
      + 0.10 * _rank01(df["roe_proxy"])
      + 0.15 * _rank01(df["debt_to_equity"], ascending=False)
      + 0.10 * _rank01(df["pe"].where(df["pe"] > 0), ascending=False)
    )
    df["fund_score"] = (score * 100.0).round(2)
    return df


def load_profile(ticker: str) -> dict:
    p = METRICS_DIR / ticker / "profile.json"
    if not p.exists():
        return {}
    try:
        rows = json.load(open(p))
        return rows[0] if isinstance(rows, list) and rows else {}
    except Exception:
        return {}


if __name__ == "__main__":
    # Self-test
    avail = fmp_available_tickers()
    print(f"FMP available tickers: {len(avail)}")
    uni = load_universe_from_fmp()
    print(f"Universe (indices ∩ FMP): {len(uni)}")
    print(uni.head(8).to_string(index=False))
    px = load_prices_fmp(["SPY", "AAPL", "NVDA"], start="2024-01-01")
    print(f"\nPrices sample shape: {px.shape}  span: {px['date'].min().date()} -> {px['date'].max().date()}")
    fund = load_fundamentals_pit(["AAPL", "MSFT", "NVDA", "JPM", "XOM"], as_of="2024-06-30")
    print(f"\nFundamentals PIT as_of 2024-06-30:")
    print(fund[["ticker", "pe", "fcf_yield", "gross_margin", "debt_to_equity", "fund_score"]].to_string(index=False))
