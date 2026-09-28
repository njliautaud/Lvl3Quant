#!/usr/bin/env python3
"""
Earnings Transcript Fetcher
============================
Fetches earnings call transcripts from SEC EDGAR (8-K filings).
Falls back to extracting earnings-relevant content from 10-Q/10-K filings.

SEC EDGAR API is free but requires User-Agent header with contact info.
Rate limit: 10 requests/second.

Usage:
    python earnings_transcript_fetcher.py                    # Fetch all universe
    python earnings_transcript_fetcher.py --ticker AAPL      # Fetch single ticker
    python earnings_transcript_fetcher.py --days-back 90     # Look back 90 days
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# Paths
BASE_DIR = Path("/home/jupiter/Lvl3Quant")
UNIVERSE_FILE = BASE_DIR / "data" / "quality_universe.json"
TRANSCRIPTS_DIR = BASE_DIR / "data" / "earnings_transcripts"
LOG_DIR = BASE_DIR / "logs" / "earnings_nlp"

# SEC EDGAR settings
EDGAR_BASE = "https://efts.sec.gov/LATEST"
EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions"
EDGAR_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
USER_AGENT = "Lvl3Quant Research research@example.com"
RATE_LIMIT_DELAY = 0.12  # 10 req/s max

# Ensure dirs exist
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "transcript_fetcher.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# CIK lookup cache
CIK_CACHE = {}


def load_universe() -> list:
    """Load ticker universe from JSON file."""
    if UNIVERSE_FILE.exists():
        with open(UNIVERSE_FILE) as f:
            data = json.load(f)
            return data.get("tickers", [])
    # Fallback
    return [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "BRK-B", "LLY",
        "UNH", "JNJ", "JPM", "V", "MA", "PG", "HD", "COST", "ABBV", "MRK",
        "AVGO", "PEP", "KO", "TMO", "ACN", "MCD", "LIN", "AMD", "CRM",
        "ISRG", "NFLX", "INTU", "TXN", "LOW", "AMAT",
    ]


def sec_request(url: str, params: dict = None) -> requests.Response:
    """Make a rate-limited request to SEC EDGAR."""
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}
    time.sleep(RATE_LIMIT_DELAY)
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    return resp


def get_cik(ticker: str) -> str | None:
    """Get CIK number for a ticker from SEC EDGAR."""
    if ticker in CIK_CACHE:
        return CIK_CACHE[ticker]

    try:
        # Use the company_tickers.json endpoint (always works, no auth needed)
        resp = sec_request("https://www.sec.gov/files/company_tickers.json")
        tickers_data = resp.json()

        # Normalize ticker for matching (BRK-B -> BRK.B for SEC)
        check_tickers = {ticker.upper()}
        if "-" in ticker:
            check_tickers.add(ticker.replace("-", ".").upper())
            check_tickers.add(ticker.replace("-", "/").upper())

        for entry in tickers_data.values():
            entry_ticker = entry.get("ticker", "").upper()
            if entry_ticker in check_tickers:
                cik = str(entry["cik_str"]).zfill(10)
                CIK_CACHE[ticker] = cik
                log.info(f"{ticker}: Found CIK {cik}")
                return cik

        log.warning(f"CIK not found for {ticker}")
        return None
    except Exception as e:
        log.error(f"Error looking up CIK for {ticker}: {e}")
        return None


def fetch_recent_filings(cik: str, form_type: str = "8-K", days_back: int = 60) -> list:
    """Fetch recent filings of a given type from EDGAR."""
    try:
        url = f"{EDGAR_SUBMISSIONS}/CIK{cik}.json"
        resp = sec_request(url)
        data = resp.json()

        filings = data.get("filings", {}).get("recent", {})
        if not filings:
            return []

        forms = filings.get("form", [])
        dates = filings.get("filingDate", [])
        accessions = filings.get("accessionNumber", [])
        primary_docs = filings.get("primaryDocument", [])
        descriptions = filings.get("primaryDocDescription", [])

        cutoff = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        results = []

        for i in range(len(forms)):
            if forms[i] == form_type and dates[i] >= cutoff:
                results.append({
                    "form": forms[i],
                    "date": dates[i],
                    "accession": accessions[i],
                    "primary_doc": primary_docs[i] if i < len(primary_docs) else "",
                    "description": descriptions[i] if i < len(descriptions) else "",
                })

        return results
    except Exception as e:
        log.error(f"Error fetching filings for CIK {cik}: {e}")
        return []


def is_earnings_related(filing: dict) -> bool:
    """Check if an 8-K filing is earnings-related (Item 2.02)."""
    desc = filing.get("description", "").lower()
    earnings_keywords = [
        "results of operations",
        "financial condition",
        "earnings",
        "quarterly results",
        "press release",
        "item 2.02",
    ]
    return any(kw in desc for kw in earnings_keywords)


def fetch_filing_text(cik: str, accession: str, primary_doc: str) -> str | None:
    """Fetch the actual text content of a filing."""
    try:
        # Format accession number for URL (remove dashes)
        acc_no_dashes = accession.replace("-", "")
        url = f"{EDGAR_ARCHIVES}/{cik.lstrip('0')}/{acc_no_dashes}/{primary_doc}"

        resp = sec_request(url)
        content_type = resp.headers.get("Content-Type", "")

        if "html" in content_type or primary_doc.endswith(".htm") or primary_doc.endswith(".html"):
            soup = BeautifulSoup(resp.text, "html.parser")
            # Remove script and style elements
            for tag in soup(["script", "style"]):
                tag.decompose()
            text = soup.get_text(separator="\n")
        else:
            text = resp.text

        # Clean up whitespace
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(line for line in lines if line)

        return text
    except Exception as e:
        log.error(f"Error fetching filing text: {e}")
        return None


def extract_earnings_content(text: str) -> dict:
    """Extract earnings-relevant sections from filing text."""
    if not text:
        return {"raw_text": "", "sections": {}}

    sections = {}

    # Look for key sections
    section_patterns = {
        "financial_results": r"(?i)(financial results|results of operations|financial highlights).*?(?=\n[A-Z][A-Z]|\Z)",
        "revenue": r"(?i)(revenue|net sales|total revenue).*?(?=\n\n|\Z)",
        "guidance": r"(?i)(guidance|outlook|forecast|expectations for).*?(?=\n\n|\Z)",
        "management_commentary": r"(?i)(said|commented|stated|noted|remarked).*?(?=\n\n|\Z)",
        "key_metrics": r"(?i)(earnings per share|eps|net income|operating income|free cash flow).*?(?=\n\n|\Z)",
    }

    for name, pattern in section_patterns.items():
        matches = re.findall(pattern, text[:50000], re.DOTALL)  # Limit search
        if matches:
            # Take the best match (longest)
            sections[name] = max(matches, key=len)[:5000]

    return {
        "raw_text": text[:100000],  # Cap at 100K chars
        "sections": sections,
    }


def fetch_transcript_for_ticker(ticker: str, days_back: int = 60) -> dict | None:
    """Fetch the most recent earnings transcript/8-K for a ticker."""
    log.info(f"Fetching transcript for {ticker}")

    cik = get_cik(ticker)
    if not cik:
        log.warning(f"Could not find CIK for {ticker}, skipping")
        return None

    # First try 8-K filings (earnings releases)
    filings = fetch_recent_filings(cik, "8-K", days_back)
    earnings_filings = [f for f in filings if is_earnings_related(f)]

    if not earnings_filings:
        # Fall back to all 8-K filings
        earnings_filings = filings[:3]  # Take most recent 3
        log.info(f"{ticker}: No earnings-specific 8-K found, using {len(earnings_filings)} recent 8-Ks")

    if not earnings_filings:
        # Try 10-Q
        filings = fetch_recent_filings(cik, "10-Q", days_back)
        earnings_filings = filings[:1]
        log.info(f"{ticker}: Falling back to 10-Q, found {len(earnings_filings)}")

    if not earnings_filings:
        log.warning(f"{ticker}: No recent filings found within {days_back} days")
        return None

    # Take the most recent filing
    filing = earnings_filings[0]
    log.info(f"{ticker}: Processing {filing['form']} from {filing['date']}")

    text = fetch_filing_text(cik, filing["accession"], filing["primary_doc"])
    if not text:
        log.warning(f"{ticker}: Could not fetch filing text")
        return None

    content = extract_earnings_content(text)

    result = {
        "ticker": ticker,
        "cik": cik,
        "filing_type": filing["form"],
        "filing_date": filing["date"],
        "accession": filing["accession"],
        "description": filing.get("description", ""),
        "fetched_at": datetime.now().isoformat(),
        "content": content,
    }

    return result


def save_transcript(data: dict):
    """Save transcript data to JSON file."""
    ticker = data["ticker"]
    date = data["filing_date"]
    filename = f"{ticker}_{date}.json"
    filepath = TRANSCRIPTS_DIR / filename

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)

    log.info(f"Saved transcript: {filepath}")
    return filepath


def fetch_all(tickers: list = None, days_back: int = 60) -> dict:
    """Fetch transcripts for all tickers in universe."""
    if tickers is None:
        tickers = load_universe()

    results = {"fetched": [], "failed": [], "skipped": []}

    for ticker in tickers:
        # Check if we already have a recent transcript
        existing = list(TRANSCRIPTS_DIR.glob(f"{ticker}_*.json"))
        if existing:
            latest = max(existing)
            try:
                with open(latest) as f:
                    existing_data = json.load(f)
                fetched_at = datetime.fromisoformat(existing_data.get("fetched_at", "2000-01-01"))
                if (datetime.now() - fetched_at).days < 1:
                    log.info(f"{ticker}: Recent transcript exists, skipping")
                    results["skipped"].append(ticker)
                    continue
            except (json.JSONDecodeError, KeyError):
                pass

        try:
            data = fetch_transcript_for_ticker(ticker, days_back)
            if data:
                save_transcript(data)
                results["fetched"].append(ticker)
            else:
                results["failed"].append(ticker)
        except Exception as e:
            log.error(f"Error processing {ticker}: {e}")
            results["failed"].append(ticker)

    return results


def main():
    parser = argparse.ArgumentParser(description="Fetch earnings transcripts from SEC EDGAR")
    parser.add_argument("--ticker", type=str, help="Fetch for a single ticker")
    parser.add_argument("--days-back", type=int, default=60, help="How far back to look (days)")
    parser.add_argument("--force", action="store_true", help="Force re-fetch even if recent data exists")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("Earnings Transcript Fetcher Starting")
    log.info(f"Days back: {args.days_back}")

    if args.ticker:
        tickers = [args.ticker.upper()]
    else:
        tickers = load_universe()

    log.info(f"Processing {len(tickers)} tickers")

    if args.force:
        # Clear existing to force re-fetch
        for t in tickers:
            for f in TRANSCRIPTS_DIR.glob(f"{t}_*.json"):
                f.unlink()

    results = fetch_all(tickers, args.days_back)

    log.info("=" * 60)
    log.info(f"Results: {len(results['fetched'])} fetched, {len(results['failed'])} failed, {len(results['skipped'])} skipped")
    if results["failed"]:
        log.info(f"Failed tickers: {results['failed']}")

    # Save summary
    summary = {
        "run_at": datetime.now().isoformat(),
        "days_back": args.days_back,
        "results": results,
    }
    with open(TRANSCRIPTS_DIR / "_fetch_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return results


if __name__ == "__main__":
    main()
