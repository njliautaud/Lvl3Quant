#!/usr/bin/env python3
"""
HC #691 — Deeper Stock-Picking Alpha: Insider Transaction Features
===================================================================
Build features from SEC EDGAR Form 4 filings (insider buys/sells).

Hypothesis: Insiders buying their own stock is one of the few signals with
documented alpha in academic literature (Lakonishok & Lee 2001, Jeng et al 2003).
Key: it's not just "insiders bought" — it's the PATTERN of buying that matters.

Features engineered:
  1. insider_buy_ratio_30d — ratio of insider buys to total transactions (30d)
  2. insider_net_shares_30d — net shares purchased by insiders (buys - sells)
  3. insider_cluster_score — multiple insiders buying within a short window
  4. insider_buy_value_30d — dollar value of insider purchases
  5. insider_officer_buy — C-suite specifically buying (higher signal)
  6. insider_conviction — large purchases relative to officer's total holdings
  7. insider_contrarian — buying during stock price decline (highest signal)

Data: SEC EDGAR XBRL API (free, no API key needed, 10 req/sec limit).
"""

import json
import os
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import requests
except ImportError:
    os.system(f"{sys.executable} -m pip install requests -q")
    import requests

try:
    import yfinance as yf
except ImportError:
    os.system(f"{sys.executable} -m pip install yfinance -q")
    import yfinance as yf

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research/insider_alpha")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = OUTPUT_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# SEC EDGAR requires a User-Agent with contact info
SEC_HEADERS = {
    "User-Agent": "ResearchBot research@example.com",
    "Accept": "application/json",
}
SEC_BASE = "https://efts.sec.gov/LATEST"
EDGAR_FULL_TEXT = "https://efts.sec.gov/LATEST/search-index"
EDGAR_COMPANY = "https://data.sec.gov/submissions"

# Rate limit: 10 req/sec for SEC EDGAR
SEC_RATE_LIMIT = 0.12  # seconds between requests


# ============================================================
# SEC EDGAR DATA FETCHING
# ============================================================

def get_cik_for_ticker(ticker: str) -> Optional[str]:
    """Look up CIK number for a stock ticker from SEC EDGAR."""
    cache_file = CACHE_DIR / "ticker_cik_map.json"

    # Load cache
    cik_map = {}
    if cache_file.exists():
        with open(cache_file) as f:
            cik_map = json.load(f)

    if ticker.upper() in cik_map:
        return cik_map[ticker.upper()]

    # Fetch the full ticker-CIK mapping from SEC
    url = "https://www.sec.gov/files/company_tickers.json"
    try:
        resp = requests.get(url, headers=SEC_HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        for entry in data.values():
            t = entry.get("ticker", "").upper()
            cik = str(entry.get("cik_str", ""))
            if t:
                cik_map[t] = cik

        # Save cache
        with open(cache_file, "w") as f:
            json.dump(cik_map, f)

        return cik_map.get(ticker.upper())
    except Exception as e:
        print(f"  Warning: CIK lookup failed for {ticker}: {e}")
        return None


def fetch_insider_transactions(ticker: str, cik: str, lookback_days: int = 365) -> pd.DataFrame:
    """
    Fetch Form 4 filings (insider transactions) from SEC EDGAR.
    Returns DataFrame with: date, insider_name, title, transaction_type, shares, price, value
    """
    cache_file = CACHE_DIR / f"insider_{ticker}_{lookback_days}d.parquet"
    if cache_file.exists():
        df = pd.read_parquet(cache_file)
        # Check if cache is recent enough (within 7 days)
        if len(df) > 0:
            max_date = pd.to_datetime(df["filing_date"]).max()
            if max_date >= pd.Timestamp.now() - pd.Timedelta(days=7):
                return df

    # Pad CIK to 10 digits
    cik_padded = cik.zfill(10)

    # Fetch company filings from EDGAR
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"

    try:
        time.sleep(SEC_RATE_LIMIT)
        resp = requests.get(url, headers=SEC_HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  Warning: EDGAR fetch failed for {ticker} (CIK {cik}): {e}")
        return pd.DataFrame()

    # Extract recent filings
    recent = data.get("filings", {}).get("recent", {})
    if not recent:
        return pd.DataFrame()

    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])

    # Filter for Form 4 (insider transactions)
    cutoff = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    form4_entries = []
    for i, (form, date_str, acc, doc) in enumerate(zip(forms, dates, accessions, primary_docs)):
        if form in ("4", "4/A") and date_str >= cutoff:
            form4_entries.append({
                "filing_date": date_str,
                "accession": acc,
                "document": doc,
            })

    if not form4_entries:
        print(f"  No Form 4 filings found for {ticker} in last {lookback_days} days")
        return pd.DataFrame()

    # For each Form 4, we need to parse the XML to get transaction details
    # But that's slow — use the EDGAR full-text search API instead
    # For now, use the filing metadata to create aggregate features

    import re

    transactions = []
    for entry in form4_entries[:50]:  # Limit to most recent 50 filings
        acc_clean = entry["accession"].replace("-", "")

        # Fetch raw XML (NOT the XSLT-transformed HTML)
        # The primaryDocument often points to xslF345X06/form4.xml (HTML).
        # The raw XML is at the root: form4.xml or doc4.xml
        raw_xml_url = f"https://www.sec.gov/Archives/edgar/data/{cik_padded}/{acc_clean}/form4.xml"

        try:
            time.sleep(SEC_RATE_LIMIT)
            resp = requests.get(raw_xml_url, headers=SEC_HEADERS, timeout=15)
            if resp.status_code != 200:
                # Try alternative filename
                raw_xml_url = f"https://www.sec.gov/Archives/edgar/data/{cik_padded}/{acc_clean}/doc4.xml"
                time.sleep(SEC_RATE_LIMIT)
                resp = requests.get(raw_xml_url, headers=SEC_HEADERS, timeout=15)
                if resp.status_code != 200:
                    continue

            content = resp.text

            # Verify it's XML, not HTML
            if content.strip().startswith("<!DOCTYPE html"):
                continue

            # Extract reporter name and title
            name_match = re.search(r'<rptOwnerName>([^<]+)</rptOwnerName>', content)
            title_match = re.search(r'<officerTitle>([^<]+)</officerTitle>', content)

            reporter_name = name_match.group(1).strip() if name_match else "Unknown"
            officer_title = title_match.group(1).strip() if title_match else ""

            # Extract non-derivative transactions
            # Split content into individual transaction blocks
            tx_blocks = re.findall(
                r'<nonDerivativeTransaction>(.*?)</nonDerivativeTransaction>',
                content, re.DOTALL
            )

            for block in tx_blocks:
                date_match = re.search(r'<transactionDate>\s*<value>(\d{4}-\d{2}-\d{2})</value>', block)
                code_match = re.search(r'<transactionCode>(\w)</transactionCode>', block)
                shares_match = re.search(r'<transactionShares>\s*<value>([\d.]+)</value>', block)
                price_match = re.search(r'<transactionPricePerShare>\s*<value>([\d.]+)</value>', block)
                acq_disp_match = re.search(r'<transactionAcquiredDisposedCode>\s*<value>(\w)</value>', block)

                if not (date_match and code_match and shares_match):
                    continue

                tx_date = date_match.group(1)
                tx_code = code_match.group(1)
                shares = float(shares_match.group(1))
                price = float(price_match.group(1)) if price_match else 0.0
                acq_disp = acq_disp_match.group(1) if acq_disp_match else ""

                # Transaction codes:
                # P = Open market Purchase (strongest signal)
                # S = Open market Sale
                # M = Exercise of options
                # A = Award/grant
                # F = Tax withholding
                # G = Gift
                # We care most about P and S (open market transactions)
                if tx_code == "P":
                    tx_type = "BUY"
                elif tx_code == "S":
                    tx_type = "SELL"
                elif tx_code == "M":
                    tx_type = "EXERCISE"
                elif tx_code == "A":
                    tx_type = "AWARD"
                elif tx_code == "F":
                    tx_type = "TAX_WITHHOLD"
                else:
                    tx_type = f"OTHER_{tx_code}"

                transactions.append({
                    "filing_date": entry["filing_date"],
                    "transaction_date": tx_date,
                    "reporter": reporter_name,
                    "title": officer_title,
                    "type": tx_type,
                    "shares": shares,
                    "price": price,
                    "value": shares * price,
                })

        except Exception as e:
            continue  # Skip problematic filings

    df = pd.DataFrame(transactions)
    df["ticker"] = ticker

    # Cache
    df.to_parquet(cache_file, index=False)
    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def compute_insider_features(ticker: str, transactions: pd.DataFrame,
                            price_data: pd.Series, as_of_date: str) -> Dict:
    """
    Compute insider features for a ticker as of a specific date.

    Features:
      1. buy_ratio_30d: buys / (buys + sells) in last 30 days
      2. net_shares_30d: net shares purchased
      3. cluster_score: number of distinct insiders buying within 10 days
      4. buy_value_30d: total dollar value of insider purchases
      5. officer_buy: 1 if C-suite bought in last 30 days
      6. contrarian: 1 if insider bought while stock was in 20d downtrend
      7. buy_ratio_90d: longer-term buy ratio
      8. net_value_90d: net dollar value of transactions (90d)
    """
    features = {
        "buy_ratio_30d": 0.0,
        "net_shares_30d": 0.0,
        "cluster_score": 0,
        "buy_value_30d": 0.0,
        "officer_buy": 0,
        "contrarian": 0,
        "buy_ratio_90d": 0.0,
        "net_value_90d": 0.0,
        "total_filings_90d": 0,
    }

    if transactions.empty:
        return features

    cutoff_date = pd.Timestamp(as_of_date)
    tx = transactions.copy()
    tx["transaction_date"] = pd.to_datetime(tx["transaction_date"])

    # 30-day window
    mask_30d = tx["transaction_date"] >= cutoff_date - pd.Timedelta(days=30)
    tx_30d = tx[mask_30d]

    # 90-day window
    mask_90d = tx["transaction_date"] >= cutoff_date - pd.Timedelta(days=90)
    tx_90d = tx[mask_90d]

    # Feature 1: Buy ratio (30d) — P transactions are open-market purchases
    buys_30d = tx_30d[tx_30d["type"] == "BUY"]
    sells_30d = tx_30d[tx_30d["type"] == "SELL"]
    # Also count exercises that are held (not immediately sold) as bullish
    exercises_30d = tx_30d[tx_30d["type"] == "EXERCISE"]
    total_30d = len(buys_30d) + len(sells_30d)
    if total_30d > 0:
        features["buy_ratio_30d"] = len(buys_30d) / total_30d

    # Feature 2: Net shares (30d)
    buy_shares = buys_30d["shares"].sum() if len(buys_30d) > 0 else 0
    sell_shares = sells_30d["shares"].sum() if len(sells_30d) > 0 else 0
    features["net_shares_30d"] = buy_shares - sell_shares

    # Sell intensity: total dollar value of insider selling (high = bearish signal)
    features["sell_value_30d"] = sells_30d["value"].sum() if len(sells_30d) > 0 else 0

    # Feature 3: Cluster score — distinct buyers within 10-day windows
    if len(buys_30d) > 0:
        unique_buyers = buys_30d["reporter"].nunique()
        features["cluster_score"] = unique_buyers

    # Feature 4: Buy value (30d)
    features["buy_value_30d"] = buys_30d["value"].sum() if len(buys_30d) > 0 else 0

    # Feature 5: Officer buy
    c_suite_titles = ["CEO", "CFO", "COO", "CTO", "President", "Chairman",
                      "Chief Executive", "Chief Financial", "Chief Operating"]
    if len(buys_30d) > 0:
        for _, row in buys_30d.iterrows():
            if any(t.lower() in row.get("title", "").lower() for t in c_suite_titles):
                features["officer_buy"] = 1
                break

    # Feature 6: Contrarian buying (bought during price decline)
    if len(buys_30d) > 0 and price_data is not None and len(price_data) > 20:
        try:
            recent_price = price_data.iloc[-1]
            price_20d_ago = price_data.iloc[-20] if len(price_data) >= 20 else price_data.iloc[0]
            if recent_price < price_20d_ago * 0.95:  # 5%+ decline
                features["contrarian"] = 1
        except:
            pass

    # Feature 7: Buy ratio (90d)
    buys_90d = tx_90d[tx_90d["type"] == "BUY"]
    sells_90d = tx_90d[tx_90d["type"] == "SELL"]
    total_90d = len(buys_90d) + len(sells_90d)
    if total_90d > 0:
        features["buy_ratio_90d"] = len(buys_90d) / total_90d

    # Feature 8: Net value (90d)
    buy_val_90d = buys_90d["value"].sum() if len(buys_90d) > 0 else 0
    sell_val_90d = sells_90d["value"].sum() if len(sells_90d) > 0 else 0
    features["net_value_90d"] = buy_val_90d - sell_val_90d

    # Feature 9: Total filing count (activity level)
    features["total_filings_90d"] = len(tx_90d)

    return features


# ============================================================
# UNIVERSE + BACKTEST
# ============================================================

# Mix of large, mid, and small-cap stocks
# Insider buying (open market P transactions) is rare at mega-caps
# (execs get paid in stock and mostly sell). Need mid/small-caps where
# insiders actually buy with their own money — THAT's the signal.
TEST_UNIVERSE = [
    # Large-cap (some insider buying happens)
    "JPM", "BAC", "WFC", "C", "GS",  # Banks — officers buy shares
    "JNJ", "PFE", "ABBV", "MRK", "BMY",  # Pharma
    "XOM", "CVX", "COP", "EOG", "SLB",  # Energy
    "INTC", "AMD", "QCOM", "MU", "MRVL",  # Semis
    # Mid-cap (more insider buying activity)
    "ALLY", "KEY", "ZION", "CFG", "FHN",  # Regional banks
    "OGN", "VTRS", "TEVA", "JAZZ", "NBIX",  # Mid-cap pharma
    "CLF", "AA", "NUE", "STLD", "RS",  # Materials
    "PARA", "WBD", "LYV", "NWSA", "GTN",  # Media
    "RIG", "HAL", "NOV", "CHX", "WFRD",  # Oil services
    "ALK", "SAVE", "LUV", "JBLU", "HA",  # Airlines
]


def run_insider_data_collection():
    """
    Phase 1: Collect insider transaction data for the test universe.
    This is the slow part (SEC API rate limited) — cache aggressively.
    """
    print("=" * 80)
    print("HC #691 — INSIDER TRANSACTION DATA COLLECTION")
    print("=" * 80)

    all_data = []
    failed = []

    for i, ticker in enumerate(TEST_UNIVERSE):
        print(f"\n[{i+1}/{len(TEST_UNIVERSE)}] Processing {ticker}...")

        # Get CIK
        cik = get_cik_for_ticker(ticker)
        if not cik:
            print(f"  ❌ No CIK found for {ticker}")
            failed.append(ticker)
            continue

        print(f"  CIK: {cik}")

        # Fetch transactions
        tx = fetch_insider_transactions(ticker, cik, lookback_days=365)

        if tx.empty:
            print(f"  ⚠️ No transactions found")
            continue

        buys = len(tx[tx["type"] == "BUY"])
        sells = len(tx[tx["type"] == "SELL"])
        print(f"  ✅ {len(tx)} transactions ({buys} buys, {sells} sells)")

        all_data.append(tx)

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        combined.to_parquet(OUTPUT_DIR / "insider_transactions_all.parquet", index=False)
        print(f"\n✅ Total: {len(combined)} transactions across {len(all_data)} tickers")
        print(f"❌ Failed: {len(failed)} tickers: {failed}")
        return combined
    else:
        print("\n❌ No data collected")
        return pd.DataFrame()


def run_feature_backtest(transactions: pd.DataFrame):
    """
    Phase 2: Compute features and test predictive power.
    Simple test: do stocks with high insider buying outperform in next 30/60/90 days?
    """
    print("\n" + "=" * 80)
    print("HC #691 — INSIDER FEATURE PREDICTIVE POWER TEST")
    print("=" * 80)

    # Get price data for all tickers
    tickers_with_data = transactions["ticker"].unique().tolist()
    print(f"\nDownloading price data for {len(tickers_with_data)} tickers...")

    try:
        price_data = yf.download(tickers_with_data, period="2y", interval="1d",
                                 auto_adjust=True, progress=False)
        prices = price_data["Close"] if "Close" in price_data.columns.get_level_values(0) else price_data
    except Exception as e:
        print(f"  Price download failed: {e}")
        return

    # For each month-end, compute features and forward returns
    # Walk-forward: features at month-end, measure returns over next month

    dates = pd.date_range("2025-08-01", "2026-06-30", freq="ME")

    monthly_records = []

    for eval_date in dates:
        eval_str = eval_date.strftime("%Y-%m-%d")

        for ticker in tickers_with_data:
            # Get price series up to eval date
            if ticker not in prices.columns:
                continue

            price_series = prices[ticker].dropna()
            price_to_date = price_series[price_series.index <= eval_date]

            if len(price_to_date) < 30:
                continue

            # Compute features
            tx = transactions[transactions["ticker"] == ticker]
            features = compute_insider_features(ticker, tx, price_to_date, eval_str)

            # Compute forward returns (1m, 2m, 3m)
            future_prices = price_series[price_series.index > eval_date]

            fwd_1m = fwd_2m = fwd_3m = np.nan
            if len(future_prices) >= 21:
                fwd_1m = future_prices.iloc[20] / price_to_date.iloc[-1] - 1
            if len(future_prices) >= 42:
                fwd_2m = future_prices.iloc[41] / price_to_date.iloc[-1] - 1
            if len(future_prices) >= 63:
                fwd_3m = future_prices.iloc[62] / price_to_date.iloc[-1] - 1

            record = {
                "date": eval_str,
                "ticker": ticker,
                "fwd_return_1m": fwd_1m,
                "fwd_return_2m": fwd_2m,
                "fwd_return_3m": fwd_3m,
                **features,
            }
            monthly_records.append(record)

    if not monthly_records:
        print("No records generated")
        return

    df = pd.DataFrame(monthly_records)
    df.to_parquet(OUTPUT_DIR / "insider_features_panel.parquet", index=False)

    # ========================================
    # ANALYSIS: Do insider features predict returns?
    # ========================================
    print(f"\nPanel: {len(df)} observations, {df['ticker'].nunique()} tickers, "
          f"{df['date'].nunique()} months")

    # Correlation of each feature with forward returns
    feature_cols = ["buy_ratio_30d", "net_shares_30d", "cluster_score",
                    "buy_value_30d", "officer_buy", "contrarian",
                    "buy_ratio_90d", "net_value_90d", "total_filings_90d"]

    print("\n📊 Feature-Return Correlations (Spearman rank):")
    print(f"{'Feature':<25} {'1m Corr':>10} {'2m Corr':>10} {'3m Corr':>10}")
    print("─" * 55)

    from scipy import stats

    corr_results = {}
    for feat in feature_cols:
        corrs = {}
        for horizon, col in [("1m", "fwd_return_1m"), ("2m", "fwd_return_2m"), ("3m", "fwd_return_3m")]:
            valid = df[[feat, col]].dropna()
            if len(valid) > 10:
                r, p = stats.spearmanr(valid[feat], valid[col])
                corrs[horizon] = (r, p)
            else:
                corrs[horizon] = (0, 1)

        corr_results[feat] = corrs
        r1, p1 = corrs["1m"]
        r2, p2 = corrs["2m"]
        r3, p3 = corrs["3m"]
        sig1 = "*" if p1 < 0.05 else " "
        sig2 = "*" if p2 < 0.05 else " "
        sig3 = "*" if p3 < 0.05 else " "
        print(f"{feat:<25} {r1:>+8.3f}{sig1} {r2:>+8.3f}{sig2} {r3:>+8.3f}{sig3}")

    print("\n* = statistically significant (p < 0.05)")

    # Long-short test: top quintile insider buying vs bottom quintile
    print("\n📊 Long-Short Quintile Test (buy_ratio_30d):")

    for horizon, col in [("1m", "fwd_return_1m"), ("2m", "fwd_return_2m")]:
        valid = df[["buy_ratio_30d", col, "date"]].dropna()
        if len(valid) < 20:
            continue

        # By month, sort into quintiles
        monthly_spreads = []
        for date_str, group in valid.groupby("date"):
            if len(group) < 10:
                continue
            group = group.sort_values("buy_ratio_30d")
            n = len(group)
            q1 = group.iloc[:n//5][col].mean()  # Bottom quintile (least insider buying)
            q5 = group.iloc[-n//5:][col].mean()  # Top quintile (most insider buying)
            spread = q5 - q1
            monthly_spreads.append(spread)

        if monthly_spreads:
            avg_spread = np.mean(monthly_spreads)
            t_stat = np.mean(monthly_spreads) / (np.std(monthly_spreads) / np.sqrt(len(monthly_spreads)))
            print(f"  {horizon}: Avg Q5-Q1 spread = {avg_spread*100:+.2f}%/month, "
                  f"t-stat = {t_stat:.2f} "
                  f"({'significant' if abs(t_stat) > 2 else 'not significant'})")

    # Summary
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    # Find strongest features
    strong_features = []
    for feat, corrs in corr_results.items():
        for h, (r, p) in corrs.items():
            if p < 0.10 and abs(r) > 0.05:
                strong_features.append((feat, h, r, p))

    if strong_features:
        print("\n✅ Features with potential predictive power:")
        for feat, h, r, p in sorted(strong_features, key=lambda x: x[3]):
            print(f"   {feat} ({h}): r={r:+.3f}, p={p:.3f}")
        print("\n⚠️ Note: This is a preliminary test on limited data (1 year, 50 stocks).")
        print("   Full validation requires: walk-forward backtest, permutation test,")
        print("   survivorship correction, and regime independence check.")
    else:
        print("\n❌ No features show significant predictive power in this initial test.")
        print("   This could mean: (a) sample too small, (b) features too crude,")
        print("   (c) alpha already arbitraged away, or (d) need better feature engineering.")

    return df


# ============================================================
# MAIN
# ============================================================

def main():
    # Phase 1: Collect data
    transactions = run_insider_data_collection()

    if transactions.empty:
        print("No data collected. Exiting.")
        return

    # Phase 2: Test predictive power
    run_feature_backtest(transactions)


if __name__ == "__main__":
    main()
