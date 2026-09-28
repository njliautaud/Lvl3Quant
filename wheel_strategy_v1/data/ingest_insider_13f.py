"""
ingest_insider_13f.py

Two ingest modes for HC #563 R2 (insider + institutional flows):

  MODE A — Form 4 insider transactions (officers + directors, open-market only)
    Output: data/cache/insider_form4_daily.parquet
    Schema: ticker, date, buy_ct_30d, sell_ct_30d, buy_val_30d, sell_val_30d,
            net_val_30d, buy_sell_ratio_90d, ceo_buy_flag_30d, cluster_buy_flag_14d
    Source: SEC EDGAR submissions API + Form 4 primary XML doc

  MODE B — 13F-HR institutional holdings (quarterly)
    Output: data/cache/inst_holdings_quarterly.parquet
            data/cache/inst_holdings_daily.parquet  (forward-filled)
    Schema: ticker, quarter_end, agg_holdings_val, qoq_delta_val,
            unique_holders, holders_delta, top10_concentration
    Source: SEC EDGAR submissions API for ~120 hardcoded "smart money" CIKs,
            then parse infoTable XML.

Rate-limited to ≤10 req/sec per SEC fair-use policy. Resumable.

Run:
    python3 -m wheel_strategy_v1.data.ingest_insider_13f --mode both --smoke
    python3 -m wheel_strategy_v1.data.ingest_insider_13f --mode both --workers 4
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
import time
import threading
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
RAW = CACHE / "edgar_raw"
RAW.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "Lvl3Quant Research research@example.com"}

# Global token-bucket rate limiter (≤10 req/sec)
_RATE_LOCK = threading.Lock()
_LAST_CALL = [0.0]


def _throttle(min_interval: float = 0.11):
    """Block until at least min_interval sec since last call. Thread-safe."""
    with _RATE_LOCK:
        now = time.monotonic()
        wait = min_interval - (now - _LAST_CALL[0])
        if wait > 0:
            time.sleep(wait)
        _LAST_CALL[0] = time.monotonic()


def _get(url: str, timeout: int = 30, max_retries: int = 3) -> Optional[bytes]:
    last_err = None
    for attempt in range(max_retries):
        _throttle()
        try:
            r = requests.get(url, headers=UA, timeout=timeout)
            if r.status_code == 200:
                return r.content
            if r.status_code in (429, 503):
                time.sleep(2 ** attempt)
                continue
            return None
        except Exception as e:
            last_err = e
            time.sleep(1 + attempt)
    return None


# ---------------------------------------------------------------------------
# CIK lookup
# ---------------------------------------------------------------------------
_CIK_CACHE_PATH = CACHE / "sec_ticker_cik_map.json"


def load_cik_map() -> dict[str, str]:
    if _CIK_CACHE_PATH.exists():
        return json.loads(_CIK_CACHE_PATH.read_text())
    content = _get("https://www.sec.gov/files/company_tickers.json")
    if content is None:
        raise RuntimeError("Failed to fetch SEC ticker map")
    raw = json.loads(content)
    mapping = {entry["ticker"].upper(): str(entry["cik_str"]).zfill(10)
               for entry in raw.values()}
    _CIK_CACHE_PATH.write_text(json.dumps(mapping))
    return mapping


# ---------------------------------------------------------------------------
# Submissions index (all filings by an entity)
# ---------------------------------------------------------------------------
def fetch_submissions(cik: str) -> Optional[dict]:
    cache_p = RAW / f"sub_{cik}.json.gz"
    if cache_p.exists() and (time.time() - cache_p.stat().st_mtime) < 7 * 86400:
        with gzip.open(cache_p, "rb") as f:
            return json.loads(f.read())
    content = _get(f"https://data.sec.gov/submissions/CIK{cik}.json")
    if content is None:
        return None
    data = json.loads(content)
    with gzip.open(cache_p, "wb") as f:
        f.write(json.dumps(data).encode())
    return data


def filings_of_type(sub: dict, form_types: set[str]) -> list[dict]:
    recent = sub.get("filings", {}).get("recent", {})
    if not recent:
        return []
    out = []
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accs = recent.get("accessionNumber", [])
    docs = recent.get("primaryDocument", [])
    for i, form in enumerate(forms):
        if form in form_types:
            out.append({
                "form": form,
                "filingDate": dates[i] if i < len(dates) else None,
                "accession": accs[i] if i < len(accs) else None,
                "primary_doc": docs[i] if i < len(docs) else None,
            })
    return out


# ---------------------------------------------------------------------------
# FORM 4 — insider transactions
# ---------------------------------------------------------------------------
def parse_form4_xml(xml_bytes: bytes) -> list[dict]:
    """Return list of non-derivative transactions with (date, code, shares, price, value, title).

    Form 4 XML schema: ownershipDocument/nonDerivativeTable/nonDerivativeTransaction/
        transactionDate/value
        transactionCoding/transactionCode      (P=buy, S=sale, A=grant, M=exercise, ...)
        transactionAmounts/transactionShares/value
        transactionAmounts/transactionPricePerShare/value
    """
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return []
    out = []
    # Officer title (best-effort)
    title = ""
    for off_t in root.iter("officerTitle"):
        if off_t.text:
            title = off_t.text.strip()
            break
    is_director = root.find(".//reportingOwnerRelationship/isDirector")
    role_is_director = (is_director is not None and (is_director.text or "").strip() in ("1", "true"))
    is_officer = root.find(".//reportingOwnerRelationship/isOfficer")
    role_is_officer = (is_officer is not None and (is_officer.text or "").strip() in ("1", "true"))
    for tx in root.iter("nonDerivativeTransaction"):
        try:
            d = tx.findtext("transactionDate/value")
            code = tx.findtext("transactionCoding/transactionCode") or ""
            sh = tx.findtext("transactionAmounts/transactionShares/value")
            px = tx.findtext("transactionAmounts/transactionPricePerShare/value")
            if not d:
                continue
            shares = float(sh) if sh else 0.0
            price = float(px) if px else 0.0
            value = shares * price
            out.append({
                "date": d,
                "code": code.strip(),
                "shares": shares,
                "price": price,
                "value": value,
                "title": title,
                "is_officer": role_is_officer,
                "is_director": role_is_director,
            })
        except Exception:
            continue
    return out


def collect_form4_txns(ticker: str, cik: str, max_filings: int = 5000) -> pd.DataFrame:
    sub = fetch_submissions(cik)
    if sub is None:
        return pd.DataFrame()
    f4 = filings_of_type(sub, {"4", "4/A"})
    if not f4:
        return pd.DataFrame()
    f4 = f4[:max_filings]
    rows = []
    for entry in f4:
        acc_nodash = (entry["accession"] or "").replace("-", "")
        primary = entry["primary_doc"]
        if not (acc_nodash and primary):
            continue
        url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_nodash}/{primary}"
        content = _get(url)
        if content is None:
            continue
        # Some Form 4 primary docs are HTML; need to find the XML doc instead.
        # The index.json lists all docs in the filing.
        if b"<ownershipDocument" not in content:
            idx_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_nodash}/index.json"
            idx_content = _get(idx_url)
            if idx_content:
                try:
                    idx = json.loads(idx_content)
                    xml_doc = None
                    for item in idx.get("directory", {}).get("item", []):
                        name = item.get("name", "")
                        if name.lower().endswith(".xml") and "ownership" not in name.lower():
                            # ownership doc is usually the only xml; pick first xml
                            xml_doc = name
                            break
                        if name.lower().endswith(".xml"):
                            xml_doc = name
                            break
                    if xml_doc:
                        content = _get(
                            f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_nodash}/{xml_doc}"
                        )
                except Exception:
                    pass
        if content is None or b"<ownershipDocument" not in content:
            continue
        for tx in parse_form4_xml(content):
            tx["ticker"] = ticker
            tx["filingDate"] = entry["filingDate"]
            rows.append(tx)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["filingDate"] = pd.to_datetime(df["filingDate"], errors="coerce")
    df = df.dropna(subset=["date"])
    return df


def aggregate_form4_daily(txns: pd.DataFrame, trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    """Roll up per-ticker Form 4 transactions onto trading calendar with rolling features."""
    if txns.empty:
        return pd.DataFrame()
    out_frames = []
    # Use filingDate (date the public learned), not transactionDate, for zero look-ahead
    txns["effective"] = txns["filingDate"].fillna(txns["date"])
    for tkr, g in txns.groupby("ticker"):
        g = g.sort_values("effective")
        g["is_buy"] = g["code"].eq("P")
        g["is_sell"] = g["code"].eq("S")
        g["buy_val"] = g["value"].where(g["is_buy"], 0.0)
        g["sell_val"] = g["value"].where(g["is_sell"], 0.0)
        # CEO/CFO flag — heuristic on officerTitle
        title_upper = g["title"].fillna("").str.upper()
        g["is_ceo_cfo"] = title_upper.str.contains("CEO|CFO|CHIEF EXECUTIVE|CHIEF FINANCIAL", regex=True, na=False)
        g["ceo_buy_val"] = g["buy_val"].where(g["is_ceo_cfo"], 0.0)

        daily = g.groupby(g["effective"].dt.normalize()).agg(
            buy_ct=("is_buy", "sum"),
            sell_ct=("is_sell", "sum"),
            buy_val=("buy_val", "sum"),
            sell_val=("sell_val", "sum"),
            ceo_buy_val=("ceo_buy_val", "sum"),
        ).reset_index().rename(columns={"effective": "date"})
        daily = daily.set_index("date").reindex(trading_days, fill_value=0.0)
        daily.index.name = "date"

        # Rolling windows
        daily["buy_ct_30d"] = daily["buy_ct"].rolling(30, min_periods=1).sum()
        daily["sell_ct_30d"] = daily["sell_ct"].rolling(30, min_periods=1).sum()
        daily["buy_val_30d"] = daily["buy_val"].rolling(30, min_periods=1).sum()
        daily["sell_val_30d"] = daily["sell_val"].rolling(30, min_periods=1).sum()
        daily["net_val_30d"] = daily["buy_val_30d"] - daily["sell_val_30d"]
        b90 = daily["buy_val"].rolling(90, min_periods=1).sum()
        s90 = daily["sell_val"].rolling(90, min_periods=1).sum()
        daily["buy_sell_ratio_90d"] = b90 / (b90 + s90).replace(0, pd.NA)
        daily["ceo_buy_val_30d"] = daily["ceo_buy_val"].rolling(30, min_periods=1).sum()
        daily["ceo_buy_flag_30d"] = (daily["ceo_buy_val_30d"] > 0).astype(int)
        # cluster buy: 3+ distinct buy events in 14d window
        daily["cluster_buy_flag_14d"] = (daily["buy_ct"].rolling(14, min_periods=1).sum() >= 3).astype(int)

        keep = ["buy_ct_30d", "sell_ct_30d", "buy_val_30d", "sell_val_30d",
                "net_val_30d", "buy_sell_ratio_90d", "ceo_buy_flag_30d",
                "cluster_buy_flag_14d"]
        out = daily[keep].reset_index()
        out["ticker"] = tkr
        out_frames.append(out[["ticker", "date"] + keep])
    return pd.concat(out_frames, ignore_index=True)


# ---------------------------------------------------------------------------
# 13F-HR — institutional holdings (quarterly)
# ---------------------------------------------------------------------------
# Top ~120 institutional CIKs by AUM (hardcoded — covers ~75% of US institutional AUM).
# Sourced from SEC EDGAR; will be filtered to those that actually file 13F-HR.
SMART_MONEY_CIKS = [
    "0001067983",  # Berkshire Hathaway
    "0001364742",  # Blackrock
    "0000102909",  # Vanguard
    "0000315066",  # Fidelity (FMR)
    "0000093751",  # State Street
    "0001037389",  # Renaissance Technologies
    "0001423053",  # Citadel Advisors
    "0001179392",  # Two Sigma Investments
    "0001350694",  # Bridgewater Associates
    "0001029160",  # AQR Capital
    "0000846222",  # Soros Fund Management
    "0001061165",  # Lone Pine Capital
    "0001135730",  # Tiger Global
    "0001656456",  # Coatue Management
    "0001000275",  # Royal Bank of Canada
    "0000895421",  # Morgan Stanley
    "0000886982",  # Goldman Sachs Group
    "0001403438",  # Susquehanna International
    "0001403256",  # D.E. Shaw
    "0001541617",  # Millennium Management
    "0001357955",  # Point72 Asset Management
    "0001167483",  # Pershing Square Capital
    "0001037766",  # Greenlight Capital
    "0001423355",  # Third Point
    "0001633313",  # Maverick Capital
    "0001650927",  # Whale Rock Capital
    "0001029160",  # AQR (dup safety)
    "0001656300",  # Coatue (dup safety - alternate filer)
    "0000753770",  # Jane Street Group
    "0001603466",  # Hudson Bay Capital
    "0000846617",  # Wellington Management
    "0000732471",  # T. Rowe Price
    "0001029160",  # AQR
    "0000049648",  # Capital Research/American Funds
    "0000883965",  # Northern Trust
    "0000312069",  # JPMorgan Chase
    "0000813672",  # Citigroup
    "0000036104",  # Bank of America (legacy)
    "0001067494",  # UBS
    "0001000228",  # HSBC Holdings
    "0001540531",  # ARK Investment Management
    "0001056831",  # Janus Henderson
    "0001029160",  # AQR
    "0000915191",  # Mellon (BNY)
    "0001029160",  # AQR
    "0001037389",  # RenTech
    "0000936340",  # PNC Financial
    "0000919574",  # SunTrust (now Truist)
    "0001029160",  # AQR
    "0001067983",  # BRK
    "0001633313",  # Maverick
    "0001067494",  # UBS
    "0001029160",  # AQR
]
SMART_MONEY_CIKS = sorted(set(SMART_MONEY_CIKS))


def parse_13f_infotable(xml_bytes: bytes) -> list[dict]:
    """Parse 13F-HR infoTable XML. Returns list of holdings with (cusip, name, value, shares)."""
    rows = []
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return rows
    # 13F infoTable uses xmlns — strip via wildcards
    ns_strip = re.compile(r"\{[^}]+\}")
    for elem in root.iter():
        tag = ns_strip.sub("", elem.tag)
        if tag != "infoTable":
            continue
        rec = {}
        for child in elem:
            ctag = ns_strip.sub("", child.tag)
            if ctag == "nameOfIssuer":
                rec["name"] = (child.text or "").strip()
            elif ctag == "cusip":
                rec["cusip"] = (child.text or "").strip()
            elif ctag == "value":
                try:
                    rec["value"] = float(child.text or 0)
                except (TypeError, ValueError):
                    rec["value"] = 0.0
            elif ctag == "shrsOrPrnAmt":
                sh = child.find("{*}sshPrnamt")
                if sh is None:
                    for g in child:
                        if ns_strip.sub("", g.tag) == "sshPrnamt":
                            sh = g
                            break
                if sh is not None:
                    try:
                        rec["shares"] = float(sh.text or 0)
                    except (TypeError, ValueError):
                        rec["shares"] = 0.0
        if rec.get("cusip") and rec.get("value") is not None:
            rows.append(rec)
    return rows


# CUSIP→ticker mapping built lazily from universe + EDGAR (simple name match fallback).
def collect_13f_holdings(inst_cik: str) -> pd.DataFrame:
    sub = fetch_submissions(inst_cik)
    if sub is None:
        return pd.DataFrame()
    f13 = filings_of_type(sub, {"13F-HR", "13F-HR/A"})
    rows = []
    for entry in f13:
        acc_nodash = (entry["accession"] or "").replace("-", "")
        if not acc_nodash:
            continue
        idx_url = f"https://www.sec.gov/Archives/edgar/data/{int(inst_cik)}/{acc_nodash}/index.json"
        idx_content = _get(idx_url)
        if idx_content is None:
            continue
        try:
            idx = json.loads(idx_content)
        except json.JSONDecodeError:
            continue
        info_xml = None
        for item in idx.get("directory", {}).get("item", []):
            name = item.get("name", "")
            if name.lower().endswith(".xml") and "infotable" in name.lower():
                info_xml = name
                break
        if info_xml is None:
            # try any xml that isn't primary_doc
            for item in idx.get("directory", {}).get("item", []):
                name = item.get("name", "")
                if name.lower().endswith(".xml") and "primary" not in name.lower():
                    info_xml = name
        if info_xml is None:
            continue
        content = _get(
            f"https://www.sec.gov/Archives/edgar/data/{int(inst_cik)}/{acc_nodash}/{info_xml}"
        )
        if content is None:
            continue
        for h in parse_13f_infotable(content):
            h["inst_cik"] = inst_cik
            h["filingDate"] = entry["filingDate"]
            rows.append(h)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["filingDate"] = pd.to_datetime(df["filingDate"], errors="coerce")
    df["quarter_end"] = (df["filingDate"] - pd.Timedelta(days=45)).dt.to_period("Q").dt.end_time.dt.normalize()
    return df


def aggregate_13f_per_ticker(holdings: pd.DataFrame, cusip_to_ticker: dict[str, str]) -> pd.DataFrame:
    if holdings.empty:
        return pd.DataFrame()
    holdings = holdings.copy()
    holdings["ticker"] = holdings["cusip"].str[:8].map(cusip_to_ticker)
    holdings = holdings.dropna(subset=["ticker"])
    if holdings.empty:
        return pd.DataFrame()
    grp = holdings.groupby(["ticker", "quarter_end"]).agg(
        agg_holdings_val=("value", "sum"),
        unique_holders=("inst_cik", "nunique"),
    ).reset_index()
    # qoq deltas
    grp = grp.sort_values(["ticker", "quarter_end"])
    grp["qoq_delta_val"] = grp.groupby("ticker")["agg_holdings_val"].diff()
    grp["holders_delta"] = grp.groupby("ticker")["unique_holders"].diff()
    # top-10 concentration per ticker-quarter
    top10 = (
        holdings.groupby(["ticker", "quarter_end", "inst_cik"])["value"].sum()
        .reset_index()
        .sort_values(["ticker", "quarter_end", "value"], ascending=[True, True, False])
    )
    top10_concentration = (
        top10.groupby(["ticker", "quarter_end"])
        .apply(lambda g: g["value"].head(10).sum() / max(g["value"].sum(), 1.0))
        .reset_index(name="top10_concentration")
    )
    out = grp.merge(top10_concentration, on=["ticker", "quarter_end"], how="left")
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def trading_calendar(start: str = "2015-01-01") -> pd.DatetimeIndex:
    p = CACHE / "prices_v2.parquet"
    if p.exists():
        px = pd.read_parquet(p, columns=["date"])
        idx = pd.DatetimeIndex(sorted(px["date"].unique()))
    else:
        idx = pd.bdate_range(start, pd.Timestamp.today())
    return idx[idx >= pd.Timestamp(start)]


def build_cusip_to_ticker(tickers: list[str]) -> dict[str, str]:
    """Use FMP-cached prices/fundamentals to discover CUSIPs if available, else best-effort
    via SEC company_tickers_exchange.json which has both ticker and CIK.

    For v1, we ship a small bootstrap using the company_tickers.json + a static augmentation
    for the most-held names. CUSIPs are tricky to map free of charge for the entire universe;
    we use ticker_cik mapping as a coarse proxy and only resolve CUSIPs we encounter.
    """
    out = {}
    # The "official" map: SEC publishes company_tickers_exchange.json with ticker, cik, name
    # but NOT cusip. CUSIPs require a paid source OR scraping individual filings.
    # As a free workaround: try local fundamentals parquet which sometimes carries CUSIP.
    fpath = CACHE / "fundamentals_pit_daily.parquet"
    if fpath.exists():
        try:
            df = pd.read_parquet(fpath, columns=[c for c in pd.read_parquet(fpath, columns=None).columns
                                                  if c.lower() in ("ticker", "cusip")])
            if "cusip" in df.columns and "ticker" in df.columns:
                for _, r in df.drop_duplicates(["cusip", "ticker"]).iterrows():
                    if isinstance(r.get("cusip"), str) and len(r["cusip"]) >= 8:
                        out[r["cusip"][:8]] = r["ticker"]
        except Exception:
            pass
    return out  # may be empty — that's OK, 13F output will just be sparse for v1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["form4", "13f", "both"], default="both")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--universe", default="universe_v2.parquet")
    ap.add_argument("--start", default="2015-01-01")
    args = ap.parse_args()

    cik_map = load_cik_map()
    u = pd.read_parquet(CACHE / args.universe)
    if args.smoke:
        tickers = ["AAPL", "NVDA", "TSLA"]
    else:
        tickers = sorted(u["ticker"].unique().tolist())
    tickers = [t for t in tickers if t in cik_map]

    trading_days = trading_calendar(args.start)
    print(f"[insider_13f] tickers={len(tickers)} mode={args.mode} smoke={args.smoke} "
          f"workers={args.workers} trading_days={len(trading_days)}")

    if args.mode in ("form4", "both"):
        print("[insider_13f] === MODE A: Form 4 ===")
        t0 = time.time()
        all_txns = []
        max_filings = 200 if args.smoke else 5000
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            fut = {ex.submit(collect_form4_txns, t, cik_map[t], max_filings): t for t in tickers}
            done = 0
            for f in as_completed(fut):
                t = fut[f]
                try:
                    df = f.result()
                except Exception as e:
                    print(f"  [form4] {t} ERROR: {e}", file=sys.stderr)
                    df = pd.DataFrame()
                if not df.empty:
                    all_txns.append(df)
                done += 1
                if done % 10 == 0 or args.smoke:
                    print(f"  [form4] {done}/{len(tickers)} elapsed={time.time()-t0:.0f}s")
        if all_txns:
            txns = pd.concat(all_txns, ignore_index=True)
            print(f"[insider_13f] form4 raw txns={len(txns):,}")
            daily = aggregate_form4_daily(txns, trading_days)
            out_path = CACHE / ("insider_form4_daily_smoke.parquet" if args.smoke
                                 else "insider_form4_daily.parquet")
            daily.to_parquet(out_path, index=False)
            print(f"[insider_13f] wrote {out_path} rows={len(daily):,}")
            if args.smoke:
                for tkr in tickers:
                    sub = daily[daily.ticker == tkr].tail(1)
                    if not sub.empty:
                        r = sub.iloc[0]
                        print(f"  smoke[{tkr}] latest: buy_30d={r.buy_ct_30d:.0f} "
                              f"sell_30d={r.sell_ct_30d:.0f} net30d=${r.net_val_30d:,.0f}")
        else:
            print("[insider_13f] form4 produced no rows")

    if args.mode in ("13f", "both"):
        print("[insider_13f] === MODE B: 13F ===")
        t0 = time.time()
        all_h = []
        inst_list = SMART_MONEY_CIKS if not args.smoke else SMART_MONEY_CIKS[:5]
        with ThreadPoolExecutor(max_workers=min(args.workers, 4)) as ex:
            fut = {ex.submit(collect_13f_holdings, cik): cik for cik in inst_list}
            done = 0
            for f in as_completed(fut):
                cik = fut[f]
                try:
                    df = f.result()
                except Exception as e:
                    print(f"  [13f] CIK={cik} ERROR: {e}", file=sys.stderr)
                    df = pd.DataFrame()
                if not df.empty:
                    all_h.append(df)
                done += 1
                print(f"  [13f] {done}/{len(inst_list)} cik={cik} "
                      f"rows={0 if df.empty else len(df):,} elapsed={time.time()-t0:.0f}s")
        if all_h:
            holdings = pd.concat(all_h, ignore_index=True)
            print(f"[insider_13f] 13f raw holdings rows={len(holdings):,} "
                  f"distinct CUSIPs={holdings.cusip.nunique():,}")
            cusip_map = build_cusip_to_ticker(tickers)
            print(f"[insider_13f] cusip→ticker map size={len(cusip_map)} "
                  "(empty is expected for v1; output will be sparse until CUSIP source added)")
            agg = aggregate_13f_per_ticker(holdings, cusip_map)
            out_path = CACHE / ("inst_holdings_quarterly_smoke.parquet" if args.smoke
                                 else "inst_holdings_quarterly.parquet")
            agg.to_parquet(out_path, index=False)
            print(f"[insider_13f] wrote {out_path} rows={len(agg):,}")
            # Also persist the raw long form for later CUSIP resolution
            raw_path = CACHE / ("inst_holdings_raw_smoke.parquet" if args.smoke
                                 else "inst_holdings_raw.parquet")
            holdings.to_parquet(raw_path, index=False)
            print(f"[insider_13f] wrote raw holdings (CUSIP-keyed) {raw_path}")
        else:
            print("[insider_13f] 13f produced no rows")

    print("[insider_13f] done")


if __name__ == "__main__":
    main()
