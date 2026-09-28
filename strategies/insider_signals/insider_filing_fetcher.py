#!/usr/bin/env python3
"""
SEC Form 4 Insider Filing Fetcher
Fetches and parses Form 4 filings from SEC EDGAR for our quality universe.
"""

import json
import logging
import os
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path

import requests

# Paths
BASE_DIR = Path("/home/jupiter/Lvl3Quant")
UNIVERSE_FILE = BASE_DIR / "data" / "quality_universe.json"
OUTPUT_DIR = BASE_DIR / "data" / "insider_filings"
LOG_DIR = BASE_DIR / "logs" / "insider_signals"
STATE_FILE = OUTPUT_DIR / "_fetch_state.json"

# SEC EDGAR config
USER_AGENT = "Lvl3Quant Research qa@example.com"
HEADERS = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}
SEC_SUBMISSIONS = "https://data.sec.gov"
SEC_ARCHIVES = "https://www.sec.gov"
EFTS_BASE = "https://efts.sec.gov/LATEST"
RATE_LIMIT_DELAY = 0.12  # ~8 req/sec to stay safely under 10/sec

# Logging
LOG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "fetcher.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# CIK lookup cache
CIK_CACHE_FILE = OUTPUT_DIR / "_cik_cache.json"

# Ticker-to-CIK mapping for our universe (pre-populated for reliability)
TICKER_CIK_MAP = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "GOOGL": "0001652044",
    "AMZN": "0001018724",
    "NVDA": "0001045810",
    "META": "0001326801",
    "BRK-B": "0001067983",
    "LLY": "0000059478",
    "UNH": "0000731766",
    "JNJ": "0000200406",
    "JPM": "0000019617",
    "V": "0001403161",
    "MA": "0001141391",
    "PG": "0000080424",
    "HD": "0000354950",
    "COST": "0000909832",
    "ABBV": "0001551152",
    "MRK": "0000310158",
    "AVGO": "0001649338",
    "PEP": "0000077476",
    "KO": "0000021344",
    "TMO": "0000097745",
    "ACN": "0001281761",
    "MCD": "0000063908",
    "LIN": "0001707925",
    "AMD": "0000002488",
    "CRM": "0001108524",
    "ISRG": "0001035267",
    "NFLX": "0001065280",
    "INTU": "0000896878",
    "TXN": "0000097476",
    "LOW": "0000060667",
    "AMAT": "0000006951",
}


def load_universe():
    """Load tickers from quality universe file."""
    with open(UNIVERSE_FILE) as f:
        data = json.load(f)
    return data["tickers"]


def resolve_cik(ticker: str) -> str | None:
    """Resolve ticker to CIK. Use hardcoded map first, then EDGAR lookup."""
    if ticker in TICKER_CIK_MAP:
        return TICKER_CIK_MAP[ticker]

    # Fallback: EDGAR company search
    try:
        url = f"{EFTS_BASE}/search-index?q={ticker}&dateRange=custom&forms=4"
        resp = requests.get(url, headers=HEADERS, timeout=15)
        time.sleep(RATE_LIMIT_DELAY)
        if resp.ok:
            # Try to extract CIK from search results
            data = resp.json()
            if data.get("hits", {}).get("hits"):
                cik = data["hits"]["hits"][0].get("_source", {}).get("entity_id")
                if cik:
                    return str(cik).zfill(10)
    except Exception as e:
        log.warning(f"CIK lookup failed for {ticker}: {e}")
    return None


def fetch_company_filings(cik: str, ticker: str, lookback_days: int = 730) -> list[dict]:
    """Fetch recent Form 4 filings for a company from EDGAR submissions endpoint."""
    filings = []
    cutoff = datetime.now() - timedelta(days=lookback_days)

    url = f"{SEC_SUBMISSIONS}/submissions/CIK{cik}.json"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        time.sleep(RATE_LIMIT_DELAY)
        if not resp.ok:
            log.warning(f"Failed to fetch submissions for {ticker} (CIK {cik}): {resp.status_code}")
            return filings

        data = resp.json()
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])

        for i, form in enumerate(forms):
            if form != "4":
                continue
            filing_date = dates[i]
            if datetime.strptime(filing_date, "%Y-%m-%d") < cutoff:
                continue

            accession = accessions[i].replace("-", "")
            accession_dashed = accessions[i]
            primary_doc = primary_docs[i]
            # Strip xsl prefix (e.g., "xslF345X06/form4.xml" -> "form4.xml")
            raw_doc = primary_doc.split("/")[-1] if "/" in primary_doc else primary_doc

            filings.append({
                "ticker": ticker,
                "cik": cik,
                "filing_date": filing_date,
                "accession": accession_dashed,
                "doc_url": f"{SEC_ARCHIVES}/Archives/edgar/data/{cik.lstrip('0')}/{accession}/{raw_doc}",
            })

        # Check for older filings in additional files
        for extra_file in data.get("filings", {}).get("files", []):
            extra_url = f"{SEC_SUBMISSIONS}/submissions/{extra_file['name']}"
            try:
                resp2 = requests.get(extra_url, headers=HEADERS, timeout=30)
                time.sleep(RATE_LIMIT_DELAY)
                if resp2.ok:
                    extra_data = resp2.json()
                    forms2 = extra_data.get("form", [])
                    dates2 = extra_data.get("filingDate", [])
                    accessions2 = extra_data.get("accessionNumber", [])
                    primary_docs2 = extra_data.get("primaryDocument", [])

                    for j, form2 in enumerate(forms2):
                        if form2 != "4":
                            continue
                        filing_date2 = dates2[j]
                        if datetime.strptime(filing_date2, "%Y-%m-%d") < cutoff:
                            break  # Dates are descending, so we can stop
                        accession2 = accessions2[j].replace("-", "")
                        accession_dashed2 = accessions2[j]
                        primary_doc2 = primary_docs2[j]
                        raw_doc2 = primary_doc2.split("/")[-1] if "/" in primary_doc2 else primary_doc2

                        filings.append({
                            "ticker": ticker,
                            "cik": cik,
                            "filing_date": filing_date2,
                            "accession": accession_dashed2,
                            "doc_url": f"{SEC_ARCHIVES}/Archives/edgar/data/{cik.lstrip('0')}/{accession2}/{raw_doc2}",
                        })
            except Exception as e:
                log.warning(f"Failed to fetch extra filings for {ticker}: {e}")

    except Exception as e:
        log.error(f"Error fetching submissions for {ticker}: {e}")

    log.info(f"{ticker}: found {len(filings)} Form 4 filings")
    return filings


def parse_form4_xml(xml_text: str, ticker: str, filing_date: str) -> list[dict]:
    """Parse Form 4 XML to extract insider transactions."""
    transactions = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        # Try to find XML within HTML wrapper
        start = xml_text.find("<?xml")
        if start == -1:
            start = xml_text.find("<ownershipDocument")
        if start == -1:
            return transactions
        try:
            root = ET.fromstring(xml_text[start:])
        except ET.ParseError:
            log.warning(f"Could not parse XML for {ticker} filing {filing_date}")
            return transactions

    # Handle namespace
    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    # Extract reporting owner info
    owner_name = ""
    owner_title = ""
    is_director = False
    is_officer = False
    officer_title = ""

    for owner in root.findall(f".//{ns}reportingOwner"):
        owner_id = owner.find(f"{ns}reportingOwnerId")
        if owner_id is not None:
            name_el = owner_id.find(f"{ns}rptOwnerName")
            if name_el is not None and name_el.text:
                owner_name = name_el.text.strip()

        relationship = owner.find(f"{ns}reportingOwnerRelationship")
        if relationship is not None:
            dir_el = relationship.find(f"{ns}isDirector")
            if dir_el is not None and dir_el.text and dir_el.text.strip() in ("1", "true"):
                is_director = True
            off_el = relationship.find(f"{ns}isOfficer")
            if off_el is not None and off_el.text and off_el.text.strip() in ("1", "true"):
                is_officer = True
            title_el = relationship.find(f"{ns}officerTitle")
            if title_el is not None and title_el.text:
                officer_title = title_el.text.strip()

    # Classify title
    if is_officer and officer_title:
        owner_title = officer_title
    elif is_director:
        owner_title = "Director"
    else:
        owner_title = officer_title or "Unknown"

    # Classify role bucket
    title_upper = owner_title.upper()
    if any(x in title_upper for x in ["CEO", "CHIEF EXECUTIVE", "PRESIDENT"]):
        role_bucket = "CEO"
    elif any(x in title_upper for x in ["CFO", "CHIEF FINANCIAL", "FINANCE"]):
        role_bucket = "CFO"
    elif any(x in title_upper for x in ["COO", "CHIEF OPERATING"]):
        role_bucket = "COO"
    elif any(x in title_upper for x in ["CTO", "CHIEF TECHNOLOGY"]):
        role_bucket = "CTO"
    elif is_director:
        role_bucket = "Director"
    elif any(x in title_upper for x in ["VP", "VICE PRESIDENT", "SVP", "EVP"]):
        role_bucket = "VP"
    else:
        role_bucket = "Other"

    # Parse non-derivative transactions
    for txn in root.findall(f".//{ns}nonDerivativeTransaction"):
        txn_data = _parse_transaction(txn, ns, ticker, filing_date, owner_name, owner_title, role_bucket)
        if txn_data:
            transactions.append(txn_data)

    return transactions


def _parse_transaction(txn, ns, ticker, filing_date, owner_name, owner_title, role_bucket):
    """Parse a single transaction element."""
    coding = txn.find(f"{ns}transactionCoding")
    if coding is None:
        return None

    # Transaction code
    code_el = coding.find(f"{ns}transactionCode")
    if code_el is None or not code_el.text:
        return None
    txn_code = code_el.text.strip()

    # Skip non-open-market transactions
    # P = Purchase, S = Sale, A = Grant/Award, G = Gift, M = Exercise
    # We mainly care about P (open market purchase) and S (sale)
    form_type_el = coding.find(f"{ns}transactionFormType")

    # Check if 10b5-1
    is_10b5_1 = False
    eq_swap = coding.find(f"{ns}equitySwapInvolved")
    # 10b5-1 flag is in footnotes, but we can also check transactionTimeliness

    # Get amounts
    amounts = txn.find(f"{ns}transactionAmounts")
    if amounts is None:
        return None

    shares_el = amounts.find(f"{ns}transactionShares")
    shares = 0
    if shares_el is not None:
        val = shares_el.find(f"{ns}value")
        if val is not None and val.text:
            try:
                shares = float(val.text.strip())
            except ValueError:
                return None

    price_el = amounts.find(f"{ns}transactionPricePerShare")
    price = 0.0
    if price_el is not None:
        val = price_el.find(f"{ns}value")
        if val is not None and val.text:
            try:
                price = float(val.text.strip())
            except ValueError:
                price = 0.0

    acq_disp = amounts.find(f"{ns}transactionAcquiredDisposedCode")
    direction = ""
    if acq_disp is not None:
        val = acq_disp.find(f"{ns}value")
        if val is not None and val.text:
            direction = val.text.strip()  # A = acquired, D = disposed

    # Get transaction date
    date_el = txn.find(f"{ns}transactionDate")
    txn_date = filing_date
    if date_el is not None:
        val = date_el.find(f"{ns}value")
        if val is not None and val.text:
            txn_date = val.text.strip()

    total_value = shares * price

    return {
        "ticker": ticker,
        "insider_name": owner_name,
        "title": owner_title,
        "role_bucket": role_bucket,
        "transaction_code": txn_code,
        "direction": direction,
        "shares": shares,
        "price": round(price, 4),
        "total_value": round(total_value, 2),
        "transaction_date": txn_date,
        "filing_date": filing_date,
    }


def fetch_and_parse_filing(filing_info: dict) -> list[dict]:
    """Fetch a single Form 4 filing and parse it."""
    url = filing_info["doc_url"]
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        time.sleep(RATE_LIMIT_DELAY)
        if not resp.ok:
            log.warning(f"Failed to fetch filing {url}: {resp.status_code}")
            return []
        return parse_form4_xml(resp.text, filing_info["ticker"], filing_info["filing_date"])
    except Exception as e:
        log.warning(f"Error fetching filing {url}: {e}")
        return []


def load_state() -> dict:
    """Load fetch state to support incremental updates."""
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"last_fetch": {}, "total_filings": 0, "total_transactions": 0}


def save_state(state: dict):
    """Save fetch state."""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def run(lookback_days: int = 730, force: bool = False):
    """Main fetch loop for all tickers."""
    tickers = load_universe()
    state = load_state()
    all_transactions = []
    total_filings = 0

    # Load existing transactions if present
    master_file = OUTPUT_DIR / "all_transactions.json"
    existing_accessions = set()
    if master_file.exists() and not force:
        with open(master_file) as f:
            existing = json.load(f)
        all_transactions = existing
        # Track which accessions we already have
        # We'll use filing_date + insider_name as a rough dedup key

    log.info(f"Fetching Form 4 filings for {len(tickers)} tickers, lookback={lookback_days} days")

    for i, ticker in enumerate(tickers):
        cik = resolve_cik(ticker)
        if not cik:
            log.warning(f"Could not resolve CIK for {ticker}, skipping")
            continue

        log.info(f"[{i+1}/{len(tickers)}] Processing {ticker} (CIK: {cik})")

        # Check if we recently fetched this ticker
        last_fetch = state.get("last_fetch", {}).get(ticker, "")
        if last_fetch and not force:
            last_dt = datetime.strptime(last_fetch, "%Y-%m-%d")
            if (datetime.now() - last_dt).days < 1:
                log.info(f"  Skipping {ticker} - fetched today already")
                continue

        filings = fetch_company_filings(cik, ticker, lookback_days)
        total_filings += len(filings)

        ticker_transactions = []
        for filing in filings:
            txns = fetch_and_parse_filing(filing)
            ticker_transactions.extend(txns)

        if ticker_transactions:
            # Save per-ticker file
            ticker_file = OUTPUT_DIR / f"{ticker}_form4.json"
            with open(ticker_file, "w") as f:
                json.dump(ticker_transactions, f, indent=2, default=str)
            log.info(f"  {ticker}: {len(ticker_transactions)} transactions saved")

            all_transactions.extend(ticker_transactions)

        state.setdefault("last_fetch", {})[ticker] = datetime.now().strftime("%Y-%m-%d")
        save_state(state)

    # Deduplicate transactions
    seen = set()
    deduped = []
    for t in all_transactions:
        key = (t["ticker"], t["insider_name"], t["transaction_date"],
               t["transaction_code"], t["shares"], t["price"])
        if key not in seen:
            seen.add(key)
            deduped.append(t)

    # Save master file
    with open(master_file, "w") as f:
        json.dump(deduped, f, indent=2, default=str)

    # Filter purchases only and save separately
    purchases = [t for t in deduped if t["transaction_code"] == "P"]
    purchases_file = OUTPUT_DIR / "purchases_only.json"
    with open(purchases_file, "w") as f:
        json.dump(purchases, f, indent=2, default=str)

    state["total_filings"] = total_filings
    state["total_transactions"] = len(deduped)
    state["total_purchases"] = len(purchases)
    state["last_run"] = datetime.now().isoformat()
    save_state(state)

    log.info(f"DONE: {total_filings} filings processed, {len(deduped)} unique transactions, {len(purchases)} purchases")
    return deduped


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch SEC Form 4 insider filings")
    parser.add_argument("--lookback", type=int, default=730, help="Days to look back (default: 730)")
    parser.add_argument("--force", action="store_true", help="Force re-fetch all tickers")
    parser.add_argument("--ticker", type=str, help="Fetch single ticker only")
    args = parser.parse_args()

    if args.ticker:
        # Single ticker mode
        cik = resolve_cik(args.ticker)
        if cik:
            filings = fetch_company_filings(cik, args.ticker, args.lookback)
            txns = []
            for f in filings:
                txns.extend(fetch_and_parse_filing(f))
            purchases = [t for t in txns if t["transaction_code"] == "P"]
            log.info(f"{args.ticker}: {len(txns)} total transactions, {len(purchases)} purchases")
            for p in purchases:
                log.info(f"  BUY: {p['insider_name']} ({p['role_bucket']}) - {p['shares']} shares @ ${p['price']} = ${p['total_value']:,.0f} on {p['transaction_date']}")
        else:
            log.error(f"Could not resolve CIK for {args.ticker}")
    else:
        run(lookback_days=args.lookback, force=args.force)
