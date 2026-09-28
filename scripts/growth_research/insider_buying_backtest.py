#!/usr/bin/env python3
"""
HC #694 -- Insider Buying Backtest (Comprehensive)
====================================================
Standalone backtest of the insider-buying alpha signal using SEC EDGAR Form 4
data and yfinance price data.

Strategy:
  - Universe: ~200 liquid large/mid-cap US stocks
  - Signal: cluster buying (>=2 insiders in 30d) OR 1 C-suite purchase >$100K
  - Bonus: contrarian flag when stock is below 20-day SMA
  - Entry: next day open after signal
  - Hold: test 5/10/20/40/60 trading days
  - Sizing: equal-weight, max 20 positions
  - Exit: at end of hold period
  - Benchmark: random stock selection from same universe (permutation test)
  - Commissions: ZERO (HC #694 -- Robinhood/IBKR are commission-free for stocks)

Data:
  - Insider transactions: SEC EDGAR Form 4 XML parsing
  - Prices: yfinance
  - Lookback: 2022-01-01 to present
  - Cache: all SEC responses cached to disk
"""

import json
import os
import re
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------- dependency check ----------
for pkg in ["requests", "yfinance", "scipy"]:
    try:
        __import__(pkg)
    except ImportError:
        os.system(f"{sys.executable} -m pip install {pkg} -q")

import requests
import yfinance as yf
from scipy import stats

# ---------- paths ----------
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research/insider_alpha")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = OUTPUT_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------- SEC EDGAR constants ----------
SEC_HEADERS = {
    "User-Agent": "ResearchBot research@example.com",
    "Accept": "application/json",
}
SEC_RATE_LIMIT = 0.11  # 10 req/sec


# ============================================================
#  UNIVERSE -- ~200 liquid large/mid-cap US equities
# ============================================================
UNIVERSE = [
    # --- Financials ---
    "JPM", "BAC", "WFC", "C", "GS", "MS", "USB", "PNC", "TFC", "COF",
    "SCHW", "BK", "STT", "AXP", "MET", "PRU", "ALL", "TRV", "AIG", "AFL",
    "ALLY", "KEY", "ZION", "CFG", "FHN", "HBAN", "RF", "FITB", "MTB", "CMA",
    # --- Technology ---
    "AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMD", "INTC", "QCOM", "MU", "MRVL",
    "CRM", "ORCL", "IBM", "ADBE", "NOW", "INTU", "AMAT", "LRCX", "KLAC", "TXN",
    "AVGO", "CSCO", "HPQ", "DELL", "CDNS", "SNPS", "FTNT", "PANW", "ZS", "CRWD",
    # --- Healthcare ---
    "JNJ", "PFE", "ABBV", "MRK", "BMY", "LLY", "UNH", "TMO", "ABT", "AMGN",
    "GILD", "REGN", "VRTX", "BIIB", "ISRG", "MDT", "SYK", "BDX", "EW", "ZBH",
    "OGN", "VTRS", "TEVA", "JAZZ", "NBIX",
    # --- Energy ---
    "XOM", "CVX", "COP", "EOG", "SLB", "OXY", "MPC", "VLO", "PSX", "HES",
    "HAL", "DVN", "FANG", "PXD", "APA", "RIG", "NOV",
    # --- Industrials ---
    "HON", "GE", "CAT", "DE", "UNP", "UPS", "RTX", "LMT", "GD", "NOC",
    "BA", "MMM", "EMR", "ITW", "ETN", "PH", "ROK", "FTV", "IR", "DOV",
    # --- Consumer Discretionary ---
    "AMZN", "TSLA", "HD", "LOW", "NKE", "SBUX", "MCD", "TGT", "DG", "DLTR",
    "GM", "F", "APTV", "CMG", "YUM", "DRI", "MAR", "HLT", "LVS", "WYNN",
    # --- Consumer Staples ---
    "PG", "KO", "PEP", "PM", "MO", "CL", "KMB", "GIS", "K", "HSY",
    "WMT", "COST", "KR", "SYY", "ADM",
    # --- Materials ---
    "LIN", "APD", "SHW", "ECL", "DD", "DOW", "NEM", "FCX", "NUE", "STLD",
    "CLF", "AA", "RS",
    # --- Utilities & REITs ---
    "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "WEC", "ES", "PEG",
    # --- Communications ---
    "DIS", "CMCSA", "T", "VZ", "NFLX", "CHTR", "PARA", "WBD", "NWSA", "LYV",
    # --- Airlines & Transport ---
    "DAL", "UAL", "LUV", "ALK",
]


# ============================================================
#  SEC EDGAR HELPERS
# ============================================================

_cik_map: Dict[str, str] = {}


def _load_cik_map() -> Dict[str, str]:
    """Load (or fetch) full ticker->CIK mapping from SEC."""
    global _cik_map
    if _cik_map:
        return _cik_map

    cache_file = CACHE_DIR / "ticker_cik_map.json"
    if cache_file.exists():
        with open(cache_file) as f:
            _cik_map = json.load(f)
        if _cik_map:
            return _cik_map

    url = "https://www.sec.gov/files/company_tickers.json"
    try:
        resp = requests.get(url, headers=SEC_HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        for entry in data.values():
            t = entry.get("ticker", "").upper()
            cik = str(entry.get("cik_str", ""))
            if t:
                _cik_map[t] = cik
        with open(cache_file, "w") as f:
            json.dump(_cik_map, f)
    except Exception as e:
        print(f"WARNING: CIK map fetch failed: {e}")

    return _cik_map


def get_cik(ticker: str) -> Optional[str]:
    m = _load_cik_map()
    return m.get(ticker.upper())


def _fetch_form4_xml(cik_padded: str, acc_clean: str) -> Optional[str]:
    """Try to fetch raw Form 4 XML from EDGAR archives."""
    for fname in ("form4.xml", "doc4.xml"):
        url = f"https://www.sec.gov/Archives/edgar/data/{cik_padded}/{acc_clean}/{fname}"
        try:
            time.sleep(SEC_RATE_LIMIT)
            resp = requests.get(url, headers=SEC_HEADERS, timeout=15)
            if resp.status_code == 200 and not resp.text.strip().startswith("<!DOCTYPE"):
                return resp.text
        except Exception:
            pass
    return None


C_SUITE_KEYWORDS = [
    "ceo", "cfo", "coo", "cto", "cio", "president", "chairman",
    "chief executive", "chief financial", "chief operating",
    "chief technology", "chief information",
]


def _is_c_suite(title: str) -> bool:
    t = title.lower()
    return any(kw in t for kw in C_SUITE_KEYWORDS)


def _parse_form4_xml(xml_text: str) -> List[dict]:
    """Parse transactions from a single Form 4 XML."""
    name_m = re.search(r"<rptOwnerName>([^<]+)</rptOwnerName>", xml_text)
    title_m = re.search(r"<officerTitle>([^<]+)</officerTitle>", xml_text)
    reporter = name_m.group(1).strip() if name_m else "Unknown"
    title = title_m.group(1).strip() if title_m else ""

    blocks = re.findall(
        r"<nonDerivativeTransaction>(.*?)</nonDerivativeTransaction>",
        xml_text, re.DOTALL,
    )

    rows = []
    for blk in blocks:
        date_m = re.search(r"<transactionDate>\s*<value>(\d{4}-\d{2}-\d{2})</value>", blk)
        code_m = re.search(r"<transactionCode>(\w)</transactionCode>", blk)
        shares_m = re.search(r"<transactionShares>\s*<value>([\d.]+)</value>", blk)
        price_m = re.search(r"<transactionPricePerShare>\s*<value>([\d.]+)</value>", blk)

        if not (date_m and code_m and shares_m):
            continue

        tx_code = code_m.group(1)
        if tx_code not in ("P", "S"):
            continue  # only care about open-market buys/sells

        shares = float(shares_m.group(1))
        price = float(price_m.group(1)) if price_m else 0.0

        rows.append({
            "transaction_date": date_m.group(1),
            "reporter": reporter,
            "title": title,
            "is_c_suite": _is_c_suite(title),
            "tx_type": "BUY" if tx_code == "P" else "SELL",
            "shares": shares,
            "price": price,
            "value": shares * price,
        })

    return rows


def fetch_insider_transactions(ticker: str, lookback_days: int = 1600) -> pd.DataFrame:
    """
    Fetch all Form 4 insider transactions for *ticker* going back *lookback_days*.
    Results are cached per ticker to avoid redundant SEC queries.
    """
    cache_file = CACHE_DIR / f"tx_{ticker}_{lookback_days}d.parquet"
    if cache_file.exists():
        try:
            df = pd.read_parquet(cache_file)
            if len(df) >= 0:
                return df
        except Exception:
            pass

    cik = get_cik(ticker)
    if not cik:
        return pd.DataFrame()

    cik_padded = cik.zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"

    try:
        time.sleep(SEC_RATE_LIMIT)
        resp = requests.get(url, headers=SEC_HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  WARNING: EDGAR fetch failed for {ticker}: {e}")
        # cache empty so we don't re-hit
        pd.DataFrame().to_parquet(cache_file, index=False)
        return pd.DataFrame()

    recent = data.get("filings", {}).get("recent", {})
    if not recent:
        pd.DataFrame().to_parquet(cache_file, index=False)
        return pd.DataFrame()

    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])

    cutoff = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    form4_accs = []
    for form, dt, acc in zip(forms, dates, accessions):
        if form in ("4", "4/A") and dt >= cutoff:
            form4_accs.append((dt, acc))

    if not form4_accs:
        pd.DataFrame().to_parquet(cache_file, index=False)
        return pd.DataFrame()

    # Limit XML fetches to avoid excessive runtime -- 80 most recent filings
    all_tx = []
    for filing_date, acc in form4_accs[:80]:
        acc_clean = acc.replace("-", "")
        xml = _fetch_form4_xml(cik_padded, acc_clean)
        if xml:
            rows = _parse_form4_xml(xml)
            for r in rows:
                r["filing_date"] = filing_date
                r["ticker"] = ticker
            all_tx.extend(rows)

    df = pd.DataFrame(all_tx)
    if df.empty:
        df = pd.DataFrame(columns=[
            "transaction_date", "reporter", "title", "is_c_suite",
            "tx_type", "shares", "price", "value", "filing_date", "ticker",
        ])
    df.to_parquet(cache_file, index=False)
    return df


# ============================================================
#  PRICE DATA
# ============================================================

def fetch_prices(tickers: List[str], start: str = "2021-06-01") -> pd.DataFrame:
    """Download daily OHLC for all tickers. Cache to disk."""
    cache_file = CACHE_DIR / "prices.parquet"
    if cache_file.exists():
        try:
            df = pd.read_parquet(cache_file)
            # check if reasonably fresh (within 3 days)
            if df.index.max() >= pd.Timestamp.now() - pd.Timedelta(days=3):
                # only return tickers we have
                available = [t for t in tickers if t in df.columns.get_level_values(1).unique()]
                if len(available) > len(tickers) * 0.8:
                    return df
        except Exception:
            pass

    print(f"Downloading price data for {len(tickers)} tickers from {start}...")
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=True, threads=True)
    if raw.empty:
        return pd.DataFrame()

    raw.to_parquet(cache_file)
    return raw


# ============================================================
#  SIGNAL GENERATION
# ============================================================

def generate_signals(all_tx: pd.DataFrame, prices_open: pd.DataFrame,
                     prices_close: pd.DataFrame) -> pd.DataFrame:
    """
    For every trading day in the backtest window, check each ticker for an
    insider-buying signal. Returns a DataFrame of (date, ticker, signal_type,
    is_contrarian).

    Signal triggers:
      A) cluster: >= 2 distinct insider PURCHASES with filing_date in the last 30 calendar days
      B) c_suite_large: 1 C-suite officer purchase > $100K
    """
    if all_tx.empty:
        return pd.DataFrame(columns=["signal_date", "ticker", "signal_type", "is_contrarian"])

    buys = all_tx[all_tx["tx_type"] == "BUY"].copy()
    if buys.empty:
        return pd.DataFrame(columns=["signal_date", "ticker", "signal_type", "is_contrarian"])

    buys["filing_date"] = pd.to_datetime(buys["filing_date"])
    buys["transaction_date"] = pd.to_datetime(buys["transaction_date"])

    # Use filing_date as the date when information becomes public
    # (transaction_date is when the insider actually traded, but it's not public yet)

    # Build a date range from prices
    trade_dates = prices_close.index.sort_values()

    signals = []
    tickers_with_buys = buys["ticker"].unique()

    for ticker in tickers_with_buys:
        tk_buys = buys[buys["ticker"] == ticker]

        if ticker not in prices_close.columns:
            continue

        close_series = prices_close[ticker].dropna()
        if len(close_series) < 25:
            continue

        # For each trading day, check if signal fires
        for dt in trade_dates:
            # Filings in the 30 calendar days ending on dt
            window_start = dt - pd.Timedelta(days=30)
            window_buys = tk_buys[
                (tk_buys["filing_date"] >= window_start) &
                (tk_buys["filing_date"] <= dt)
            ]

            if window_buys.empty:
                continue

            sig_type = None

            # Signal A: cluster buying (>=2 distinct buyers)
            n_distinct_buyers = window_buys["reporter"].nunique()
            if n_distinct_buyers >= 2:
                sig_type = "cluster"

            # Signal B: C-suite large purchase >$100K
            csuite_buys = window_buys[window_buys["is_c_suite"]]
            if not csuite_buys.empty and csuite_buys["value"].max() > 100_000:
                sig_type = "c_suite_large" if sig_type is None else "cluster+c_suite"

            if sig_type is None:
                continue

            # Contrarian check: price below 20-day SMA
            close_to_date = close_series[close_series.index <= dt]
            is_contrarian = False
            if len(close_to_date) >= 20:
                sma20 = close_to_date.iloc[-20:].mean()
                if close_to_date.iloc[-1] < sma20:
                    is_contrarian = True

            signals.append({
                "signal_date": dt,
                "ticker": ticker,
                "signal_type": sig_type,
                "is_contrarian": is_contrarian,
                "n_buyers": n_distinct_buyers,
                "max_buy_value": window_buys["value"].max(),
            })

    sig_df = pd.DataFrame(signals)

    if sig_df.empty:
        return sig_df

    # De-duplicate: keep only the FIRST signal per ticker per rolling 30-day window
    # (avoid re-entering the same position repeatedly)
    sig_df = sig_df.sort_values("signal_date")
    deduped = []
    last_signal: Dict[str, pd.Timestamp] = {}
    for _, row in sig_df.iterrows():
        tk = row["ticker"]
        dt = row["signal_date"]
        if tk in last_signal and (dt - last_signal[tk]).days < 30:
            continue
        last_signal[tk] = dt
        deduped.append(row)

    return pd.DataFrame(deduped)


# ============================================================
#  BACKTEST ENGINE
# ============================================================

def run_backtest(signals: pd.DataFrame, prices_open: pd.DataFrame,
                 prices_close: pd.DataFrame, hold_days: int,
                 max_positions: int = 20) -> pd.DataFrame:
    """
    Event-driven backtest.
    - Enter at next trading day's open after signal_date.
    - Exit at the close of the trading day that is *hold_days* trading days later.
    - Equal-weight, max_positions cap (first-come-first-served).
    - No commissions (HC #694).

    Returns DataFrame of trades: ticker, entry_date, entry_price, exit_date,
    exit_price, return, is_contrarian, signal_type.
    """
    if signals.empty:
        return pd.DataFrame()

    trade_dates = prices_open.index.sort_values()
    date_list = list(trade_dates)

    trades = []
    active_positions: Dict[str, pd.Timestamp] = {}  # ticker -> exit_date

    for _, sig in signals.iterrows():
        sig_date = sig["signal_date"]
        ticker = sig["ticker"]

        # Find next trading day for entry
        future_dates = [d for d in date_list if d > sig_date]
        if len(future_dates) < hold_days + 1:
            continue

        entry_date = future_dates[0]
        exit_date = future_dates[hold_days]  # hold_days trading days later

        # Expire old positions
        active_positions = {
            t: ed for t, ed in active_positions.items() if ed >= entry_date
        }

        # Check capacity
        if len(active_positions) >= max_positions:
            continue

        # Already have this ticker?
        if ticker in active_positions:
            continue

        # Get prices
        if ticker not in prices_open.columns or ticker not in prices_close.columns:
            continue

        try:
            entry_price = prices_open.loc[entry_date, ticker]
            exit_price = prices_close.loc[exit_date, ticker]
        except (KeyError, TypeError):
            continue

        if pd.isna(entry_price) or pd.isna(exit_price) or entry_price <= 0:
            continue

        ret = exit_price / entry_price - 1.0

        trades.append({
            "ticker": ticker,
            "signal_date": sig_date,
            "entry_date": entry_date,
            "entry_price": float(entry_price),
            "exit_date": exit_date,
            "exit_price": float(exit_price),
            "return": float(ret),
            "hold_days": hold_days,
            "signal_type": sig["signal_type"],
            "is_contrarian": sig["is_contrarian"],
        })

        active_positions[ticker] = exit_date

    return pd.DataFrame(trades)


# ============================================================
#  RANDOM BENCHMARK (permutation test)
# ============================================================

def random_benchmark(trade_df: pd.DataFrame, prices_open: pd.DataFrame,
                     prices_close: pd.DataFrame, universe: List[str],
                     hold_days: int, n_permutations: int = 100) -> dict:
    """
    For each real trade, randomly pick a different stock from the universe
    on the same entry date and hold for the same period. Repeat n_permutations
    times to build a null distribution of average returns.
    """
    if trade_df.empty:
        return {"mean_random": np.nan, "p_value": np.nan, "n_perms": 0}

    rng = np.random.default_rng(42)
    available_tickers = [t for t in universe if t in prices_open.columns and t in prices_close.columns]

    perm_means = []
    for _ in range(n_permutations):
        perm_returns = []
        for _, trade in trade_df.iterrows():
            # Pick a random ticker (different from the actual one)
            candidates = [t for t in available_tickers if t != trade["ticker"]]
            if not candidates:
                continue
            rand_ticker = rng.choice(candidates)

            try:
                entry_p = prices_open.loc[trade["entry_date"], rand_ticker]
                exit_p = prices_close.loc[trade["exit_date"], rand_ticker]
                if pd.isna(entry_p) or pd.isna(exit_p) or entry_p <= 0:
                    continue
                perm_returns.append(exit_p / entry_p - 1.0)
            except (KeyError, TypeError):
                continue

        if perm_returns:
            perm_means.append(np.mean(perm_returns))

    if not perm_means:
        return {"mean_random": np.nan, "p_value": np.nan, "n_perms": 0}

    actual_mean = trade_df["return"].mean()
    perm_means = np.array(perm_means)
    p_value = np.mean(perm_means >= actual_mean)  # one-sided: fraction of random >= actual

    return {
        "mean_random": float(np.mean(perm_means)),
        "std_random": float(np.std(perm_means)),
        "p_value": float(p_value),
        "n_perms": len(perm_means),
        "actual_mean": float(actual_mean),
    }


# ============================================================
#  METRICS
# ============================================================

def compute_metrics(trade_df: pd.DataFrame) -> dict:
    """Compute strategy metrics from a trade DataFrame."""
    if trade_df.empty or len(trade_df) < 2:
        return {
            "n_trades": 0, "avg_return": np.nan, "median_return": np.nan,
            "win_rate": np.nan, "sharpe": np.nan, "sortino": np.nan,
            "max_dd": np.nan, "cagr": np.nan, "profit_factor": np.nan,
        }

    rets = trade_df["return"].values
    n = len(rets)
    avg = float(np.mean(rets))
    med = float(np.median(rets))
    wr = float(np.mean(rets > 0))

    # Sharpe (annualized, assuming avg hold period from the trades)
    hold_days = trade_df["hold_days"].iloc[0]
    trades_per_year = 252.0 / hold_days
    std = np.std(rets, ddof=1) if n > 1 else 1e-9
    sharpe = (avg / std) * np.sqrt(trades_per_year) if std > 1e-9 else 0.0

    # Sortino
    downside = rets[rets < 0]
    down_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-9
    sortino = (avg / down_std) * np.sqrt(trades_per_year) if down_std > 1e-9 else 0.0

    # Profit factor
    gross_profit = float(np.sum(rets[rets > 0]))
    gross_loss = float(np.abs(np.sum(rets[rets < 0])))
    pf = gross_profit / gross_loss if gross_loss > 1e-9 else float("inf")

    # Approximate CAGR from cumulative compounded return
    cum = np.prod(1 + rets) - 1
    # Time span
    first = pd.Timestamp(trade_df["entry_date"].min())
    last = pd.Timestamp(trade_df["exit_date"].max())
    years = max((last - first).days / 365.25, 0.25)
    cagr = (1 + cum) ** (1 / years) - 1

    # Max drawdown on cumulative equity curve
    equity = np.cumprod(1 + rets)
    running_max = np.maximum.accumulate(equity)
    dd = (equity - running_max) / running_max
    max_dd = float(np.min(dd))

    return {
        "n_trades": n,
        "avg_return": float(avg),
        "median_return": float(med),
        "win_rate": float(wr),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_dd": float(max_dd),
        "cagr": float(cagr),
        "profit_factor": float(pf),
        "total_return": float(cum),
        "years": float(years),
    }


def regime_analysis(trade_df: pd.DataFrame, spy_close: pd.Series) -> dict:
    """Split trades into green/red/flat regimes based on SPY close-to-close."""
    if trade_df.empty:
        return {}

    results = {}
    for regime_name, condition_fn in [
        ("green", lambda r: r > 0.002),
        ("red", lambda r: r < -0.002),
        ("flat", lambda r: abs(r) <= 0.002),
    ]:
        regime_trades = []
        for _, trade in trade_df.iterrows():
            entry_dt = pd.Timestamp(trade["entry_date"])
            try:
                spy_today = spy_close.loc[entry_dt]
                prev_dates = spy_close.index[spy_close.index < entry_dt]
                if len(prev_dates) == 0:
                    continue
                spy_prev = spy_close.loc[prev_dates[-1]]
                spy_ret = spy_today / spy_prev - 1
                if condition_fn(spy_ret):
                    regime_trades.append(trade)
            except (KeyError, TypeError):
                continue

        if regime_trades:
            rdf = pd.DataFrame(regime_trades)
            m = compute_metrics(rdf)
            results[regime_name] = {
                "n_trades": m["n_trades"],
                "avg_return": m["avg_return"],
                "win_rate": m["win_rate"],
                "sharpe": m["sharpe"],
            }

    # Regime gap check
    if "green" in results and "red" in results:
        sg = results["green"].get("sharpe", 0)
        sr = results["red"].get("sharpe", 0)
        denom = max(abs(sg), abs(sr), 1e-9)
        gap = abs(sg - sr) / denom
        results["regime_gap"] = float(gap)
        results["regime_gap_pass"] = gap <= 0.50

    return results


# ============================================================
#  MAIN PIPELINE
# ============================================================

def main():
    t0 = time.time()
    print("=" * 80)
    print("INSIDER BUYING BACKTEST  (HC #694)")
    print("=" * 80)

    # ------ Phase 1: Fetch insider transaction data ------
    print(f"\n--- Phase 1: Fetching insider transactions for {len(UNIVERSE)} tickers ---")

    all_frames = []
    failed = []
    for i, ticker in enumerate(UNIVERSE):
        pct = (i + 1) / len(UNIVERSE) * 100
        sys.stdout.write(f"\r  [{i+1}/{len(UNIVERSE)}] {pct:.0f}%  {ticker:<6}")
        sys.stdout.flush()

        try:
            df = fetch_insider_transactions(ticker, lookback_days=1600)
            if not df.empty:
                all_frames.append(df)
        except Exception as e:
            failed.append((ticker, str(e)))

    print()

    if not all_frames:
        print("FATAL: no insider data collected. Exiting.")
        return

    all_tx = pd.concat(all_frames, ignore_index=True)

    buys_only = all_tx[all_tx["tx_type"] == "BUY"]
    print(f"  Total transactions: {len(all_tx)}  ({len(buys_only)} buys)")
    print(f"  Tickers with data: {all_tx['ticker'].nunique()}")
    print(f"  Failed tickers: {len(failed)}")
    if failed:
        print(f"    {[f[0] for f in failed[:10]]}{'...' if len(failed) > 10 else ''}")

    all_tx.to_parquet(OUTPUT_DIR / "all_insider_transactions.parquet", index=False)

    # ------ Phase 2: Fetch prices ------
    print("\n--- Phase 2: Downloading price data ---")
    tickers_needed = list(set(UNIVERSE))
    # Always include SPY for regime analysis
    if "SPY" not in tickers_needed:
        tickers_needed.append("SPY")

    raw_prices = fetch_prices(tickers_needed, start="2021-06-01")
    if raw_prices.empty:
        print("FATAL: price download failed. Exiting.")
        return

    # Handle multi-level columns from yfinance
    if isinstance(raw_prices.columns, pd.MultiIndex):
        prices_open = raw_prices["Open"]
        prices_close = raw_prices["Close"]
    else:
        # single ticker edge case
        prices_open = raw_prices[["Open"]].rename(columns={"Open": tickers_needed[0]})
        prices_close = raw_prices[["Close"]].rename(columns={"Close": tickers_needed[0]})

    spy_close = prices_close["SPY"].dropna() if "SPY" in prices_close.columns else pd.Series(dtype=float)

    print(f"  Price data: {len(prices_close)} trading days, "
          f"{len(prices_close.columns)} tickers, "
          f"{prices_close.index.min().date()} to {prices_close.index.max().date()}")

    # ------ Phase 3: Generate signals ------
    print("\n--- Phase 3: Generating insider-buying signals ---")
    signals = generate_signals(all_tx, prices_open, prices_close)
    if signals.empty:
        print("WARNING: No signals generated. Check data quality.")
        return

    print(f"  Total signals: {len(signals)}")
    print(f"  Signal types: {signals['signal_type'].value_counts().to_dict()}")
    print(f"  Contrarian: {signals['is_contrarian'].sum()} / {len(signals)}")
    print(f"  Date range: {signals['signal_date'].min()} to {signals['signal_date'].max()}")

    signals.to_parquet(OUTPUT_DIR / "signals.parquet", index=False)

    # ------ Phase 4: Run backtests for each hold period ------
    print("\n--- Phase 4: Running backtests ---")
    hold_periods = [5, 10, 20, 40, 60]

    all_results = {}
    all_trades_combined = []

    for hd in hold_periods:
        print(f"\n  === Hold period: {hd} trading days ===")

        trades = run_backtest(signals, prices_open, prices_close, hold_days=hd)
        if trades.empty:
            print(f"    No trades for hold={hd}")
            continue

        metrics = compute_metrics(trades)
        print(f"    Trades: {metrics['n_trades']}")
        print(f"    Avg return: {metrics['avg_return']*100:+.2f}%")
        print(f"    Win rate: {metrics['win_rate']*100:.1f}%")
        print(f"    Sharpe: {metrics['sharpe']:.2f}")
        print(f"    Sortino: {metrics['sortino']:.2f}")
        print(f"    Max DD: {metrics['max_dd']*100:.1f}%")
        print(f"    Profit factor: {metrics['profit_factor']:.2f}")
        print(f"    CAGR: {metrics['cagr']*100:.1f}%")

        # Contrarian subset
        contrarian_trades = trades[trades["is_contrarian"]]
        if len(contrarian_trades) >= 5:
            c_metrics = compute_metrics(contrarian_trades)
            print(f"    -- Contrarian subset ({c_metrics['n_trades']} trades): "
                  f"avg={c_metrics['avg_return']*100:+.2f}%, "
                  f"WR={c_metrics['win_rate']*100:.0f}%, "
                  f"Sharpe={c_metrics['sharpe']:.2f}")
        else:
            c_metrics = None

        # Permutation test
        print(f"    Running permutation test (100 shuffles)...")
        perm = random_benchmark(trades, prices_open, prices_close, UNIVERSE,
                                hold_days=hd, n_permutations=100)
        print(f"    Benchmark avg return: {perm['mean_random']*100:+.2f}%")
        print(f"    p-value (one-sided): {perm['p_value']:.3f}")

        # Regime analysis
        regime = regime_analysis(trades, spy_close)
        if regime:
            for rname in ["green", "red", "flat"]:
                if rname in regime:
                    r = regime[rname]
                    print(f"    Regime {rname}: {r['n_trades']} trades, "
                          f"avg={r['avg_return']*100:+.2f}%, Sharpe={r['sharpe']:.2f}")
            if "regime_gap" in regime:
                print(f"    Regime gap: {regime['regime_gap']:.2f} "
                      f"({'PASS' if regime.get('regime_gap_pass') else 'FAIL -- regime-dependent'})")

        all_results[hd] = {
            "metrics": metrics,
            "contrarian_metrics": c_metrics if c_metrics else {},
            "permutation": perm,
            "regime": regime,
        }

        trades.to_csv(OUTPUT_DIR / f"trades_hold{hd}.csv", index=False)
        all_trades_combined.append(trades)

    # ------ Phase 5: Summary ------
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\n{'Hold':>6} | {'Trades':>6} | {'Avg Ret':>8} | {'WR':>5} | "
          f"{'Sharpe':>7} | {'Sortino':>7} | {'PF':>5} | {'p-val':>6} | {'Regime Gap':>10}")
    print("-" * 80)

    for hd in hold_periods:
        if hd not in all_results:
            continue
        m = all_results[hd]["metrics"]
        p = all_results[hd]["permutation"]
        rg = all_results[hd].get("regime", {})
        gap_str = f"{rg.get('regime_gap', 0):.2f}" if "regime_gap" in rg else "N/A"
        print(f"{hd:>6} | {m['n_trades']:>6} | {m['avg_return']*100:>+7.2f}% | "
              f"{m['win_rate']*100:>4.0f}% | {m['sharpe']:>7.2f} | "
              f"{m['sortino']:>7.2f} | {m['profit_factor']:>5.2f} | "
              f"{p['p_value']:>5.3f} | {gap_str:>10}")

    # Save summary
    summary = {
        "run_timestamp": datetime.now().isoformat(),
        "universe_size": len(UNIVERSE),
        "total_signals": len(signals),
        "data_range": f"{signals['signal_date'].min()} to {signals['signal_date'].max()}",
        "results_by_hold_period": {},
    }

    for hd, res in all_results.items():
        summary["results_by_hold_period"][str(hd)] = {
            "metrics": {k: round(v, 6) if isinstance(v, float) else v
                        for k, v in res["metrics"].items()},
            "permutation": {k: round(v, 6) if isinstance(v, float) else v
                            for k, v in res["permutation"].items()},
            "regime": res.get("regime", {}),
        }

    with open(OUTPUT_DIR / "backtest_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Save combined trades
    if all_trades_combined:
        combined = pd.concat(all_trades_combined, ignore_index=True)
        combined.to_parquet(OUTPUT_DIR / "all_trades.parquet", index=False)

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed/60:.1f} minutes")
    print(f"Output saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
