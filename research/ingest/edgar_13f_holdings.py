"""
Family: ticker_flows_13f  (HC #563 R2 — institutional holdings deltas)

WHAT: 13F-HR filings (quarterly institutional holdings) — pulls the holdings
informationTable XML for a roster of major asset managers, normalises the
positions (nameOfIssuer, cusip, value, shares), and computes Q-over-Q deltas
per (manager, cusip) and the aggregate net institutional flow per cusip.

SOURCE: SEC EDGAR — submissions JSON + per-filing informationTable XML.
Free; rate-limit 10 req/sec with polite User-Agent.

OUTPUT:
  data/feature_store/edgar_13f/holdings_quarterly.parquet
    (manager_cik, manager_name, period_of_report, cusip, name_of_issuer,
     shares, value_usd, put_call)
  data/feature_store/edgar_13f/deltas_per_manager.parquet
    (manager_cik, manager_name, period_of_report, cusip, shares,
     shares_delta_qoq, value_usd, value_delta_qoq)
  data/feature_store/edgar_13f/flow_per_cusip.parquet
    (period_of_report, cusip, name_of_issuer, n_managers,
     net_shares_delta, net_value_delta)

PIT-SAFE: 13F has 45-day filing deadline → feature usable filing_date + 1d,
not quarter_end + 1d. We retain filing_date alongside period_of_report.

NOTE: CUSIP→ticker resolution is NOT done here. Downstream consumers can join
on cusip via the OpenFIGI / CUSIP-to-ticker mapping when needed.
"""
from __future__ import annotations
import sys, time, re
from xml.etree import ElementTree as ET
import requests
import pandas as pd
sys.path.insert(0, "/home/jupiter/Lvl3Quant/research/ingest")
from _common import write_parquet, smoke_log, SEC_HEADERS  # type: ignore

FAMILY = "edgar_13f"

# Major 13F filers (CIK, name) — covers ~$10T AUM combined.
FILERS = [
    ("0001067983", "BERKSHIRE HATHAWAY INC"),
    ("0000102909", "VANGUARD GROUP INC"),
    ("0001364742", "BLACKROCK INC"),
    ("0000093751", "STATE STREET CORP"),
    ("0000315066", "FIDELITY MGMT & RESEARCH"),
    ("0001350694", "BRIDGEWATER ASSOCIATES LP"),
    ("0001037389", "RENAISSANCE TECHNOLOGIES LLC"),
    ("0000846222", "SOROS FUND MANAGEMENT LLC"),
    ("0001029160", "TWO SIGMA INVESTMENTS LP"),
    ("0001112520", "CITADEL ADVISORS LLC"),
]

EDGAR_DATA = "https://data.sec.gov"
EDGAR_WWW  = "https://www.sec.gov"

# Look back this many quarterly 13F filings per manager (5 = 5 quarters → 4 deltas).
N_FILINGS_PER_FILER = 5

NS = {"ns": "http://www.sec.gov/edgar/document/thirteenf/informationtable"}


def _req(url: str, host: str, retries: int = 3, pause: float = 0.15):
    h = {**SEC_HEADERS, "Host": host}
    last_err = None
    for i in range(retries):
        try:
            r = requests.get(url, headers=h, timeout=20)
            if r.status_code == 200:
                return r
            last_err = f"HTTP {r.status_code}"
        except Exception as e:
            last_err = repr(e)
        time.sleep(pause * (2 ** i))
    raise RuntimeError(f"EDGAR fetch failed for {url}: {last_err}")


def list_13f_filings(cik10: str) -> pd.DataFrame:
    url = f"{EDGAR_DATA}/submissions/CIK{cik10}.json"
    j = _req(url, host="data.sec.gov").json()
    recs = j.get("filings", {}).get("recent", {})
    df = pd.DataFrame({
        "form":          recs.get("form", []),
        "filing_date":   recs.get("filingDate", []),
        "report_date":   recs.get("reportDate", []),
        "accession":     recs.get("accessionNumber", []),
        "primary_doc":   recs.get("primaryDocument", []),
    })
    df = df[df["form"].str.startswith("13F-HR", na=False)].copy()
    df = df.sort_values("filing_date", ascending=False).head(N_FILINGS_PER_FILER)
    return df


def fetch_information_table(cik10: str, accession: str) -> list[dict] | None:
    acc_nodash = accession.replace("-", "")
    # The informationTable XML filename varies; we list the filing index json.
    idx_url = f"{EDGAR_WWW}/cgi-bin/browse-edgar?action=getcompany&CIK={cik10}&type=13F-HR&dateb=&owner=include&count=40"
    # Simpler: directly try the standard informationtable XML candidates.
    candidates = [
        f"{EDGAR_WWW}/Archives/edgar/data/{int(cik10)}/{acc_nodash}/informationtable.xml",
        f"{EDGAR_WWW}/Archives/edgar/data/{int(cik10)}/{acc_nodash}/form13fInfoTable.xml",
        f"{EDGAR_WWW}/Archives/edgar/data/{int(cik10)}/{acc_nodash}/infotable.xml",
    ]
    # Fallback: enumerate the filing index for an XML matching info*table*
    idx_json = f"{EDGAR_WWW}/Archives/edgar/data/{int(cik10)}/{acc_nodash}/"
    body = None
    for u in candidates:
        try:
            body = _req(u, host="www.sec.gov", retries=1).content
            break
        except Exception:
            body = None
    if body is None:
        # Crawl the index page for a file matching infotable/table
        try:
            r = _req(idx_json, host="www.sec.gov", retries=1)
            html = r.text
            for m in re.findall(r'href="([^"]+\.xml)"', html, flags=re.IGNORECASE):
                if "table" in m.lower() or "info" in m.lower():
                    full = m if m.startswith("http") else f"{EDGAR_WWW}{m}"
                    try:
                        body = _req(full, host="www.sec.gov", retries=1).content
                        break
                    except Exception:
                        pass
        except Exception:
            pass
    if body is None:
        return None

    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        return None

    rows = []
    # Handle namespaced and non-namespaced XML
    info_tags = root.findall(".//ns:infoTable", NS) or root.findall(".//infoTable")
    for it in info_tags:
        def _t(tag):
            for path in (f"ns:{tag}", tag):
                el = it.find(path, NS) if path.startswith("ns:") else it.find(path)
                if el is not None and el.text is not None:
                    return el.text.strip()
            return None
        def _sh():
            # shrsOrPrnAmt / sshPrnamt
            for path in ("ns:shrsOrPrnAmt/ns:sshPrnamt", "shrsOrPrnAmt/sshPrnamt"):
                el = it.find(path, NS) if path.startswith("ns:") else it.find(path)
                if el is not None and el.text is not None:
                    return el.text.strip()
            return None
        def _pc():
            for path in ("ns:putCall", "putCall"):
                el = it.find(path, NS) if path.startswith("ns:") else it.find(path)
                if el is not None and el.text is not None:
                    return el.text.strip()
            return None
        rows.append({
            "name_of_issuer": _t("nameOfIssuer"),
            "title_of_class": _t("titleOfClass"),
            "cusip":          _t("cusip"),
            "value_usd":      _t("value"),
            "shares":         _sh(),
            "put_call":       _pc() or "",
        })
    return rows


def main():
    all_rows = []
    for cik, name in FILERS:
        try:
            print(f"[{FAMILY}] {name} CIK={cik} …")
            fils = list_13f_filings(cik)
            for _, row in fils.iterrows():
                acc = row["accession"]
                report_date = row["report_date"]
                filing_date = row["filing_date"]
                try:
                    holdings = fetch_information_table(cik, acc)
                except Exception as e:
                    print(f"  filing {acc} fetch err: {e}")
                    holdings = None
                if not holdings:
                    print(f"  {report_date} no holdings parsed")
                    continue
                for h in holdings:
                    h.update({
                        "manager_cik":   cik,
                        "manager_name":  name,
                        "period_of_report": report_date,
                        "filing_date":   filing_date,
                        "accession":     acc,
                    })
                all_rows.extend(holdings)
                print(f"  {report_date} {len(holdings)} positions")
                time.sleep(0.12)
        except Exception as e:
            print(f"[{FAMILY}] {name} ERROR {e!r}")
            continue

    if not all_rows:
        smoke_log(FAMILY, False, "no rows pulled")
        raise RuntimeError("13F ingest returned 0 rows across all filers")

    df = pd.DataFrame(all_rows)
    # Coerce numeric
    df["value_usd"] = pd.to_numeric(df["value_usd"], errors="coerce")
    df["shares"]    = pd.to_numeric(df["shares"], errors="coerce")
    df["period_of_report"] = pd.to_datetime(df["period_of_report"])
    df["filing_date"]      = pd.to_datetime(df["filing_date"])
    # Note: value is reported in thousands USD pre-2023, raw USD post-2023.
    # Heuristic — values < 1e9 likely thousands; multiply.
    df["value_usd"] = df["value_usd"].where(df["value_usd"] >= 1e9, df["value_usd"] * 1000)

    df = df[["manager_cik","manager_name","period_of_report","filing_date",
             "accession","cusip","name_of_issuer","title_of_class",
             "shares","value_usd","put_call"]].sort_values(
        ["manager_cik","period_of_report","cusip"]).reset_index(drop=True)

    p1 = write_parquet(df, FAMILY, "holdings_quarterly.parquet")
    print(f"OK {FAMILY} holdings: {len(df)} rows -> {p1}")

    # Compute per-(manager, cusip) Q-over-Q deltas
    df_d = df.sort_values(["manager_cik","cusip","period_of_report"]).copy()
    g = df_d.groupby(["manager_cik","cusip"], group_keys=False)
    df_d["shares_delta_qoq"]  = g["shares"].diff()
    df_d["value_delta_qoq"]   = g["value_usd"].diff()
    p2 = write_parquet(df_d, FAMILY, "deltas_per_manager.parquet")
    print(f"OK {FAMILY} deltas: {len(df_d)} rows -> {p2}")

    # Aggregate to per-(period, cusip) net institutional flow
    agg = (df_d.dropna(subset=["shares_delta_qoq"])
              .groupby(["period_of_report","cusip","name_of_issuer"], as_index=False)
              .agg(n_managers=("manager_cik","nunique"),
                   net_shares_delta=("shares_delta_qoq","sum"),
                   net_value_delta=("value_delta_qoq","sum")))
    p3 = write_parquet(agg, FAMILY, "flow_per_cusip.parquet")
    print(f"OK {FAMILY} flow: {len(agg)} rows -> {p3}")

    smoke_log(FAMILY, True, f"holdings={len(df)} deltas={len(df_d)} flow={len(agg)}")
    return p1, p2, p3


def run_full():
    return main()


if __name__ == "__main__":
    main()
