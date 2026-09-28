"""
ingest_fundamentals_pit.py

Builds the POINT-IN-TIME historical fundamentals timeseries the macro picker
needs to escape the static-snapshot look-ahead leak documented in
ingest_fundamentals.py.

For each (ticker, business_day) the row contains the snapshot of fundamentals
that was PUBLICLY AVAILABLE as of that day (uses filingDate, not period-end).

Strategy
--------
1.  For each ticker, load quarterly income / balance / cashflow / key_metrics
    / ratios / analyst_estimates tables from the local FMP archive.
2.  Walk every quarterly filingDate. At each filingDate, compute a fundamental
    snapshot containing:
        revenue_ttm, fcf_ttm, eps_ttm, gross_margin, net_margin, ebitda_margin,
        roe_proxy, debt_to_equity, current_ratio, market_cap_at_filing,
        rev_yoy_growth, eps_yoy_growth, fcf_yoy_growth, margin_delta_4q,
        beat_rate_4q, est_revisions_4q
3.  Build the daily timeseries by forward-filling each snapshot until the next
    filingDate. Trading-day calendar is the union of all dates in prices.parquet.
4.  Write data/cache/fundamentals_pit_daily.parquet.

Output schema (long-format, sorted by ticker, date):
    ticker | date | <feature columns above>

Cost: ~2-3 minutes for 247 names on Jupiter CPU.

Run:
    python -m data.ingest_fundamentals_pit              # full v2 universe
    python -m data.ingest_fundamentals_pit --tickers AAPL,MSFT,NVDA
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
FMP_ROOT = Path("/home/jupiter/teleclaude-main/data/fmp_archive")
FIN_DIR = FMP_ROOT / "financials"
METRICS_DIR = FMP_ROOT / "metrics"
EARN_DIR = FMP_ROOT / "earnings"


# ---------------------------------------------------------------------------
# Raw FMP loaders (defensive; missing file -> empty list)
# ---------------------------------------------------------------------------
def _load_json(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        with open(path) as f:
            d = json.load(f)
        return d if isinstance(d, list) else []
    except Exception:
        return []


def _ticker_tables(ticker: str) -> dict[str, list[dict]]:
    return {
        "income": _load_json(FIN_DIR / ticker / "income_quarter.json"),
        "balance": _load_json(FIN_DIR / ticker / "balance_quarter.json"),
        "cashflow": _load_json(FIN_DIR / ticker / "cashflow_quarter.json"),
        "ratios": _load_json(METRICS_DIR / ticker / "ratios_quarter.json"),
        "key_metrics": _load_json(METRICS_DIR / ticker / "key_metrics_quarter.json"),
        "est": _load_json(EARN_DIR / ticker / "analyst_estimates_quarter.json"),
    }


def _filing_date(r: dict) -> pd.Timestamp | None:
    s = r.get("filingDate") or r.get("date")
    if not s:
        return None
    try:
        return pd.Timestamp(s)
    except Exception:
        return None


def _period_end(r: dict) -> pd.Timestamp | None:
    s = r.get("date")
    if not s:
        return None
    try:
        return pd.Timestamp(s)
    except Exception:
        return None


def _sorted_by_filing(rows: list[dict]) -> list[dict]:
    """Return rows sorted ASCENDING by filing date, dropping rows w/o filingDate."""
    keyed = []
    for r in rows:
        fd = _filing_date(r)
        if fd is None:
            continue
        keyed.append((fd, r))
    keyed.sort(key=lambda x: x[0])
    return [r for _, r in keyed]


def _latest_at_or_before(rows: list[dict], cutoff: pd.Timestamp) -> dict:
    """Most recent row whose filingDate <= cutoff. Empty dict if none."""
    best, best_dt = {}, pd.Timestamp("1900-01-01")
    for r in rows:
        fd = _filing_date(r)
        if fd is None or fd > cutoff:
            continue
        if fd >= best_dt:
            best, best_dt = r, fd
    return best


def _ttm_sum(rows: list[dict], col: str, cutoff: pd.Timestamp) -> float:
    """Sum of `col` across the 4 most recently FILED quarters as of cutoff."""
    avail = [r for r in rows if (_filing_date(r) is not None and _filing_date(r) <= cutoff)]
    avail.sort(key=lambda r: _filing_date(r), reverse=True)
    vals = []
    for r in avail[:4]:
        v = r.get(col)
        if v is not None:
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                pass
    return float(np.sum(vals)) if vals else float("nan")


def _ttm_sum_yago(rows: list[dict], col: str, cutoff: pd.Timestamp) -> float:
    """Same as _ttm_sum but for the trailing-4Q ending ~1 year before cutoff."""
    y_ago = cutoff - pd.Timedelta(days=365)
    return _ttm_sum(rows, col, y_ago)


# ---------------------------------------------------------------------------
# Snapshot at a single filingDate event
# ---------------------------------------------------------------------------
def _snapshot(tables: dict, cutoff: pd.Timestamp) -> dict:
    income = tables["income"]
    balance = tables["balance"]
    cashflow = tables["cashflow"]
    ratios = tables["ratios"]
    km = tables["key_metrics"]
    est = tables["est"]

    inc = _latest_at_or_before(income, cutoff)
    bal = _latest_at_or_before(balance, cutoff)
    rat = _latest_at_or_before(ratios, cutoff)
    kmi = _latest_at_or_before(km, cutoff)

    # TTM totals
    rev_ttm = _ttm_sum(income, "revenue", cutoff)
    ni_ttm = _ttm_sum(income, "netIncome", cutoff)
    gp_ttm = _ttm_sum(income, "grossProfit", cutoff)
    ebitda_ttm = _ttm_sum(income, "ebitda", cutoff)
    eps_ttm = _ttm_sum(income, "eps", cutoff)
    ocf_ttm = _ttm_sum(cashflow, "netCashProvidedByOperatingActivities", cutoff)
    if not np.isfinite(ocf_ttm):
        ocf_ttm = _ttm_sum(cashflow, "operatingCashFlow", cutoff)
    capex_ttm = _ttm_sum(cashflow, "capitalExpenditure", cutoff)
    fcf_ttm = (ocf_ttm if np.isfinite(ocf_ttm) else 0.0) \
              - (abs(capex_ttm) if np.isfinite(capex_ttm) else 0.0)
    if not (np.isfinite(ocf_ttm) or np.isfinite(capex_ttm)):
        fcf_ttm = float("nan")

    # Year-ago TTMs for YoY growth
    rev_ttm_yago = _ttm_sum_yago(income, "revenue", cutoff)
    ni_ttm_yago = _ttm_sum_yago(income, "netIncome", cutoff)
    eps_ttm_yago = _ttm_sum_yago(income, "eps", cutoff)
    fcf_ocf_y = _ttm_sum_yago(cashflow, "netCashProvidedByOperatingActivities", cutoff)
    fcf_cap_y = _ttm_sum_yago(cashflow, "capitalExpenditure", cutoff)
    fcf_ttm_yago = (fcf_ocf_y if np.isfinite(fcf_ocf_y) else 0.0) \
                   - (abs(fcf_cap_y) if np.isfinite(fcf_cap_y) else 0.0)
    if not (np.isfinite(fcf_ocf_y) or np.isfinite(fcf_cap_y)):
        fcf_ttm_yago = float("nan")

    def _gr(now, then):
        if not (np.isfinite(now) and np.isfinite(then)) or abs(then) < 1.0:
            return float("nan")
        return (now / then) - 1.0

    rev_yoy = _gr(rev_ttm, rev_ttm_yago)
    ni_yoy = _gr(ni_ttm, ni_ttm_yago)
    eps_yoy = _gr(eps_ttm, eps_ttm_yago)
    fcf_yoy = _gr(fcf_ttm, fcf_ttm_yago)

    # Margins
    gross_margin = (gp_ttm / rev_ttm) if (np.isfinite(gp_ttm) and np.isfinite(rev_ttm) and rev_ttm > 0) else float("nan")
    net_margin = (ni_ttm / rev_ttm) if (np.isfinite(ni_ttm) and np.isfinite(rev_ttm) and rev_ttm > 0) else float("nan")
    ebitda_margin = (ebitda_ttm / rev_ttm) if (np.isfinite(ebitda_ttm) and np.isfinite(rev_ttm) and rev_ttm > 0) else float("nan")

    # Balance-sheet leverage / liquidity
    total_debt = bal.get("totalDebt")
    if total_debt is None:
        st = bal.get("shortTermDebt") or 0.0
        lt = bal.get("longTermDebt") or 0.0
        total_debt = st + lt if (st or lt) else None
    total_equity = bal.get("totalStockholdersEquity")
    de = (total_debt / total_equity) if (total_debt is not None
                                          and total_equity not in (None, 0)) else float("nan")
    curr_ratio = rat.get("currentRatio") or kmi.get("currentRatio") or float("nan")
    try:
        curr_ratio = float(curr_ratio) if curr_ratio is not None else float("nan")
    except (TypeError, ValueError):
        curr_ratio = float("nan")

    # ROE proxy from TTM NI / equity
    roe = (ni_ttm / total_equity) if (np.isfinite(ni_ttm)
                                       and total_equity not in (None, 0)) else float("nan")

    # 4Q margin trend: current GM minus avg-GM-prior-4Q
    margin_trend = float("nan")
    sorted_inc = sorted(
        [r for r in income if (_filing_date(r) is not None and _filing_date(r) <= cutoff)],
        key=lambda r: _filing_date(r), reverse=True,
    )
    if len(sorted_inc) >= 8:
        recent = sorted_inc[:4]
        prior = sorted_inc[4:8]
        def _gm(qs):
            rev = sum((r.get("revenue") or 0.0) for r in qs)
            gp = sum((r.get("grossProfit") or 0.0) for r in qs)
            return (gp / rev) if rev > 0 else float("nan")
        gm_recent = _gm(recent)
        gm_prior = _gm(prior)
        if np.isfinite(gm_recent) and np.isfinite(gm_prior):
            margin_trend = gm_recent - gm_prior

    # Analyst beat rate over last 4 reported quarters (uses estimate vs reported EPS)
    beat_rate = float("nan")
    if est and sorted_inc:
        # match estimate.date (fiscal period end) to income.date
        est_map = {}
        for e in est:
            d = e.get("date")
            if not d:
                continue
            est_eps = e.get("estimatedEpsAvg") or e.get("epsAvg")
            if est_eps is None:
                continue
            try:
                est_map[pd.Timestamp(d).normalize()] = float(est_eps)
            except (TypeError, ValueError):
                continue
        hits = 0
        n = 0
        for r in sorted_inc[:4]:
            pe = _period_end(r)
            if pe is None:
                continue
            eps_actual = r.get("eps")
            if eps_actual is None:
                continue
            # find estimate within 5 days of fiscal period end
            cand = None
            for d_est, v in est_map.items():
                if abs((d_est - pe).days) <= 5:
                    cand = v
                    break
            if cand is None:
                continue
            n += 1
            if float(eps_actual) >= cand:
                hits += 1
        beat_rate = (hits / n) if n > 0 else float("nan")

    # Market cap snapshot via key_metrics (uses period-end price; lookahead-free)
    market_cap = kmi.get("marketCap") if kmi else float("nan")
    try:
        market_cap = float(market_cap) if market_cap is not None else float("nan")
    except (TypeError, ValueError):
        market_cap = float("nan")
    fcf_yield = (fcf_ttm / market_cap) if (np.isfinite(fcf_ttm)
                                            and np.isfinite(market_cap) and market_cap > 0) else float("nan")

    return {
        "revenue_ttm": rev_ttm,
        "fcf_ttm": fcf_ttm,
        "ni_ttm": ni_ttm,
        "eps_ttm": eps_ttm,
        "ebitda_ttm": ebitda_ttm,
        "gross_margin": gross_margin,
        "net_margin": net_margin,
        "ebitda_margin": ebitda_margin,
        "roe": roe,
        "debt_to_equity": de,
        "current_ratio": curr_ratio,
        "market_cap_pit": market_cap,
        "fcf_yield": fcf_yield,
        "rev_yoy_growth": rev_yoy,
        "ni_yoy_growth": ni_yoy,
        "eps_yoy_growth": eps_yoy,
        "fcf_yoy_growth": fcf_yoy,
        "margin_trend_4q": margin_trend,
        "beat_rate_4q": beat_rate,
    }


# ---------------------------------------------------------------------------
# Per-ticker timeseries: one row per filingDate event
# ---------------------------------------------------------------------------
def build_ticker_events(ticker: str) -> pd.DataFrame:
    tables = _ticker_tables(ticker)
    income = tables["income"]
    if not income:
        return pd.DataFrame()

    # Every distinct filingDate that exists in ANY of the quarterly tables
    event_dates = set()
    for key in ("income", "balance", "cashflow", "ratios", "key_metrics"):
        for r in tables[key]:
            fd = _filing_date(r)
            if fd is not None:
                event_dates.add(fd.normalize())
    if not event_dates:
        return pd.DataFrame()

    snapshots = []
    for ev in sorted(event_dates):
        snap = _snapshot(tables, ev)
        snap["ticker"] = ticker
        snap["effective_date"] = ev
        snapshots.append(snap)
    return pd.DataFrame(snapshots)


# ---------------------------------------------------------------------------
# Daily forward-fill to trading calendar
# ---------------------------------------------------------------------------
def daily_from_events(events: pd.DataFrame, trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    out = []
    for t, g in events.groupby("ticker"):
        g = g.sort_values("effective_date").set_index("effective_date")
        g = g.drop(columns=["ticker"], errors="ignore")
        # Reindex onto trading calendar with forward fill — only days >= first filing
        idx = trading_days[trading_days >= g.index.min()]
        if len(idx) == 0:
            continue
        df = g.reindex(idx, method="ffill")
        df["ticker"] = t
        df.index.name = "date"
        out.append(df.reset_index())
    if not out:
        return pd.DataFrame()
    daily = pd.concat(out, ignore_index=True)
    cols = ["ticker", "date"] + [c for c in daily.columns if c not in ("ticker", "date")]
    return daily[cols]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="universe_v2.parquet",
                    help="Parquet under data/cache/ with a 'ticker' column")
    ap.add_argument("--tickers", default=None,
                    help="Override: comma-separated tickers (skips --universe)")
    ap.add_argument("--prices", default="prices_v2.parquet",
                    help="Parquet under data/cache/ used to build trading calendar")
    ap.add_argument("--out", default="fundamentals_pit_daily.parquet")
    ap.add_argument("--start", default="2015-01-01",
                    help="Earliest date to keep in daily output")
    args = ap.parse_args()

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        u = pd.read_parquet(CACHE / args.universe)
        tickers = sorted(u["ticker"].unique().tolist())

    # Trading calendar from prices
    prices_path = CACHE / args.prices
    if prices_path.exists():
        px = pd.read_parquet(prices_path, columns=["date"])
        trading_days = pd.DatetimeIndex(sorted(px["date"].unique()))
    else:
        trading_days = pd.bdate_range(args.start, pd.Timestamp.today())
    trading_days = trading_days[trading_days >= pd.Timestamp(args.start)]

    print(f"[ingest_fundamentals_pit] tickers={len(tickers)} trading_days={len(trading_days)} start={args.start}")
    all_events = []
    skipped = []
    for i, t in enumerate(tickers):
        ev = build_ticker_events(t)
        if ev.empty:
            skipped.append(t)
            continue
        all_events.append(ev)
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(tickers)}] events so far={sum(len(e) for e in all_events)}")

    if not all_events:
        print("[ingest_fundamentals_pit] NO EVENTS — aborting")
        return

    events = pd.concat(all_events, ignore_index=True)
    print(f"[ingest_fundamentals_pit] events_total={len(events):,} skipped_tickers={len(skipped)}")
    if skipped:
        print(f"  skipped first 20: {skipped[:20]}")

    daily = daily_from_events(events, trading_days)
    print(f"[ingest_fundamentals_pit] daily rows={len(daily):,} cols={len(daily.columns)}")

    out_path = CACHE / args.out
    daily.to_parquet(out_path, index=False)
    print(f"[ingest_fundamentals_pit] wrote {out_path}")

    # Coverage report
    cov = daily.groupby("ticker")["date"].agg(["min", "max", "count"]).reset_index()
    print("\nCoverage sample (first 8 tickers):")
    print(cov.head(8).to_string(index=False))
    print(f"\nMedian rows/ticker: {cov['count'].median():.0f}")
    print(f"Earliest coverage: {cov['min'].min()}  |  Latest: {cov['max'].max()}")


if __name__ == "__main__":
    main()
