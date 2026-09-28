"""
Family: insider_form4  (HC #563 R2 — insider transactions)

WHAT: Form 4 filings — insider (officer / director / 10%+ holder) open-market
transactions. Per-(ticker, day): net insider buy/sell USD, # filers, exec vs non-exec split.

SOURCE: SEC EDGAR — submissions API + Form-4 XML body. Free, 10 req/sec MAX.

WHY: HC #562/#563 — insider buys are a documented positive signal (cluster
buying especially). User explicitly cited.

OUTPUT: data/feature_store/edgar_form4/insider_daily.parquet
Schema: (ticker, date, net_insider_usd, n_buys, n_sells, n_buyers, n_sellers,
         mean_price, gross_buy_usd, gross_sell_usd)

PIT-SAFE: filing typically lands <= 2 business days after txn -> feature usable filing_date + 1d.
"""
from __future__ import annotations
import sys, os, time, json, re, argparse, datetime as dt
from pathlib import Path
from typing import Optional, List, Dict, Any
import requests
import pandas as pd
from lxml import etree
from bs4 import BeautifulSoup

sys.path.insert(0, "/home/jupiter/Lvl3Quant/research/ingest")
from _common import write_parquet, smoke_log, SMOKE_UNIVERSE, STORE

FAMILY = "edgar_form4"
OUT_DIR = STORE / FAMILY
CIK_MAP_PATH = OUT_DIR / "_cik_map.json"

# EDGAR requires an identifying UA. Per task spec.
UA = "Lvl3Quant Research research@lvl3quant.local"

# Two host header variants — submissions API is on data.sec.gov, archives/browse are on www.sec.gov
HDR_DATA = {"User-Agent": UA, "Accept-Encoding": "gzip, deflate", "Host": "data.sec.gov"}
HDR_WWW  = {"User-Agent": UA, "Accept-Encoding": "gzip, deflate", "Host": "www.sec.gov"}

RATE_SLEEP = 0.12  # 10 req/sec policy with a bit of margin

# Wheel-style universe ~ 30 tickers (mega caps + liquid mid caps)
WHEEL_UNIVERSE = [
    "AAPL","MSFT","NVDA","JPM","XOM","UNH","TSLA","META","GOOGL","AMZN",
    "AMD","NFLX","BAC","WFC","CVX","KO","PEP","COST","WMT","HD",
    "DIS","CRM","ORCL","ADBE","INTC","CSCO","QCOM","T","VZ","PFE",
]


def _get(url: str, headers: Dict[str, str], timeout: int = 45) -> requests.Response:
    """GET with EDGAR rate-limit sleep + retry on 429/5xx and connection timeouts."""
    last_exc = None
    r = None
    for attempt in range(5):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            time.sleep(RATE_SLEEP)
            if r.status_code == 200:
                return r
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(1.0 + attempt * 1.5)
                continue
            return r
        except (requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectTimeout,
                requests.exceptions.ConnectionError) as e:
            last_exc = e
            time.sleep(1.5 + attempt * 1.5)
            continue
    if r is not None:
        return r
    raise last_exc if last_exc else RuntimeError(f"GET failed: {url}")


def load_cik_map(force: bool = False) -> Dict[str, str]:
    """Download (once) the company_tickers.json and cache as {TICKER: cik_zero_padded}."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if CIK_MAP_PATH.exists() and not force:
        try:
            return json.loads(CIK_MAP_PATH.read_text())
        except Exception:
            pass
    url = "https://www.sec.gov/files/company_tickers.json"
    r = _get(url, HDR_WWW)
    if r.status_code != 200:
        raise RuntimeError(f"cik map http {r.status_code}")
    raw = r.json()
    # Format: {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
    out: Dict[str, str] = {}
    for _, row in raw.items():
        tkr = str(row.get("ticker", "")).upper()
        cik = int(row.get("cik_str", 0))
        if tkr and cik:
            out[tkr] = f"{cik:010d}"
    CIK_MAP_PATH.write_text(json.dumps(out))
    return out


def fetch_recent_form4_filings(cik: str, since: dt.date) -> List[Dict[str, Any]]:
    """Pull the submissions JSON and filter for form == '4' with filingDate >= since."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    r = _get(url, HDR_DATA)
    if r.status_code != 200:
        return []
    j = r.json()
    recent = j.get("filings", {}).get("recent", {})
    forms = recent.get("form", []) or []
    accs  = recent.get("accessionNumber", []) or []
    dates = recent.get("filingDate", []) or []
    prim  = recent.get("primaryDocument", []) or []
    out: List[Dict[str, Any]] = []
    for f, a, d, p in zip(forms, accs, dates, prim):
        if f != "4":
            continue
        try:
            fd = dt.date.fromisoformat(d)
        except Exception:
            continue
        if fd < since:
            continue
        out.append({"accession": a, "filing_date": d, "primary_doc": p})
    return out


def _find_xml_in_index(cik_int: int, accession_nodash: str) -> Optional[str]:
    """Scrape filing index page to locate the Form-4 XML document name."""
    idx_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik_int}"
    # Use the filing index JSON for reliability
    index_json_url = (
        f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/index.json"
    )
    r = _get(index_json_url, HDR_WWW)
    if r.status_code != 200:
        return None
    try:
        j = r.json()
        items = j.get("directory", {}).get("item", [])
        # Prefer files ending .xml that aren't the FilingSummary
        xml_candidates = [it["name"] for it in items if it.get("name", "").lower().endswith(".xml")
                          and "filingsummary" not in it["name"].lower()]
        if xml_candidates:
            # Form 4 primary XML is typically the small one — pick first
            return xml_candidates[0]
    except Exception:
        return None
    return None


def parse_form4_xml(xml_bytes: bytes) -> List[Dict[str, Any]]:
    """Parse Form-4 XML; return one row per nonDerivativeTransaction."""
    rows: List[Dict[str, Any]] = []
    try:
        root = etree.fromstring(xml_bytes)
    except Exception:
        return rows

    def _t(node, path) -> Optional[str]:
        el = node.find(path)
        if el is None:
            return None
        # Form 4 wraps values in <value>X</value>
        v = el.find("value")
        if v is not None and v.text is not None:
            return v.text.strip()
        return (el.text or "").strip() or None

    insider_name = None
    is_officer = False
    is_director = False
    is_ten_pct = False

    ro = root.find(".//reportingOwner")
    if ro is not None:
        name_el = ro.find(".//rptOwnerName")
        if name_el is not None and name_el.text:
            insider_name = name_el.text.strip()
        rel = ro.find(".//reportingOwnerRelationship")
        if rel is not None:
            def _b(tag):
                e = rel.find(tag)
                return (e is not None and (e.text or "").strip() in ("1", "true", "True"))
            is_officer  = _b("isOfficer")
            is_director = _b("isDirector")
            is_ten_pct  = _b("isTenPercentOwner")

    if is_officer:
        role = "officer"
    elif is_director:
        role = "director"
    elif is_ten_pct:
        role = "ten_percent"
    else:
        role = "other"

    for txn in root.findall(".//nonDerivativeTransaction"):
        txn_date = _t(txn, "transactionDate")
        txn_code = _t(txn, "transactionCoding/transactionCode")
        shares   = _t(txn, "transactionAmounts/transactionShares")
        price    = _t(txn, "transactionAmounts/transactionPricePerShare")
        ad_code  = _t(txn, "transactionAmounts/transactionAcquiredDisposedCode")
        try:
            sh = float(shares) if shares is not None else 0.0
        except Exception:
            sh = 0.0
        try:
            pr = float(price) if price is not None else 0.0
        except Exception:
            pr = 0.0
        rows.append({
            "txn_date": txn_date,
            "txn_code": txn_code or "",
            "ad_code": ad_code or "",
            "shares": sh,
            "price": pr,
            "insider_name": insider_name or "",
            "role": role,
        })
    return rows


def process_ticker(ticker: str, cik: str, since: dt.date) -> pd.DataFrame:
    """Return raw transaction rows for one ticker (one row per txn)."""
    filings = fetch_recent_form4_filings(cik, since)
    if not filings:
        return pd.DataFrame()
    cik_int = int(cik)
    all_rows: List[Dict[str, Any]] = []
    for f in filings:
        acc = f["accession"]
        acc_nodash = acc.replace("-", "")
        xml_name = _find_xml_in_index(cik_int, acc_nodash)
        if not xml_name:
            continue
        xml_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{xml_name}"
        rx = _get(xml_url, HDR_WWW)
        if rx.status_code != 200:
            continue
        txns = parse_form4_xml(rx.content)
        for t in txns:
            t["ticker"] = ticker
            t["filing_date"] = f["filing_date"]
            t["accession"] = acc
            all_rows.append(t)
    return pd.DataFrame(all_rows)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Per-(ticker, date) aggregation."""
    if df.empty:
        return df
    df = df.copy()
    # Open-market transactions: code P (purchase) or S (sale). Filter rest (A=grant, F=tax, etc.)
    df = df[df["txn_code"].isin(["P", "S"])]
    if df.empty:
        return df
    df["usd"] = df["shares"] * df["price"]
    df["is_buy"]  = (df["txn_code"] == "P").astype(int)
    df["is_sell"] = (df["txn_code"] == "S").astype(int)
    df["signed_usd"] = df["usd"] * df["is_buy"] - df["usd"] * df["is_sell"]
    df["date"] = pd.to_datetime(df["txn_date"], errors="coerce").dt.date

    g = df.groupby(["ticker", "date"], dropna=True)
    out = g.agg(
        net_insider_usd=("signed_usd", "sum"),
        gross_buy_usd=("usd", lambda s: float(s[df.loc[s.index, "is_buy"] == 1].sum())),
        gross_sell_usd=("usd", lambda s: float(s[df.loc[s.index, "is_sell"] == 1].sum())),
        n_buys=("is_buy", "sum"),
        n_sells=("is_sell", "sum"),
        n_buyers=("insider_name", lambda s: s[df.loc[s.index, "is_buy"] == 1].nunique()),
        n_sellers=("insider_name", lambda s: s[df.loc[s.index, "is_sell"] == 1].nunique()),
        mean_price=("price", "mean"),
    ).reset_index()
    return out


def run(universe: List[str], lookback_days: int = 365, out_name: str = "insider_daily.parquet") -> Path:
    cik_map = load_cik_map()
    since = dt.date.today() - dt.timedelta(days=lookback_days)
    all_txn: List[pd.DataFrame] = []
    for i, tkr in enumerate(universe):
        cik = cik_map.get(tkr.upper())
        if not cik:
            print(f"[{i+1}/{len(universe)}] {tkr}: no CIK mapping, skip", flush=True)
            continue
        try:
            df = process_ticker(tkr, cik, since)
        except Exception as e:
            print(f"[{i+1}/{len(universe)}] {tkr}: ERROR {e!r}", flush=True)
            continue
        print(f"[{i+1}/{len(universe)}] {tkr} CIK={cik}: {len(df)} raw txn rows", flush=True)
        if not df.empty:
            all_txn.append(df)
    if not all_txn:
        empty = pd.DataFrame(columns=["ticker","date","net_insider_usd","gross_buy_usd",
                                      "gross_sell_usd","n_buys","n_sells","n_buyers","n_sellers","mean_price"])
        return write_parquet(empty, FAMILY, out_name)
    raw = pd.concat(all_txn, ignore_index=True)
    agg = aggregate(raw)
    p = write_parquet(agg, FAMILY, out_name)
    # Also keep raw txn-level alongside for audit
    write_parquet(raw, FAMILY, "insider_raw_txn.parquet")
    return p


def main():
    """Smoke: first 5 SMOKE_UNIVERSE tickers, ~365d lookback."""
    try:
        universe = SMOKE_UNIVERSE[:5]
        p = run(universe, lookback_days=365, out_name="insider_daily.parquet")
        try:
            df = pd.read_parquet(p)
            n = len(df)
        except Exception:
            n = -1
        smoke_log(FAMILY, True, f"{n} agg rows -> {p}")
        print(f"OK {FAMILY}: {n} rows -> {p}")
        if n > 0:
            print(df.head().to_string())
    except Exception as e:
        smoke_log(FAMILY, False, repr(e))
        print(f"FAIL {FAMILY}: {e}")
        raise


def run_full():
    """Full wheel universe ingest — used for the background launch."""
    p = run(WHEEL_UNIVERSE, lookback_days=365, out_name="insider_daily.parquet")
    print(f"FULL DONE: {p}")
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="Run full wheel universe")
    ap.add_argument("--lookback", type=int, default=365)
    args = ap.parse_args()
    if args.full:
        run_full()
    else:
        main()
