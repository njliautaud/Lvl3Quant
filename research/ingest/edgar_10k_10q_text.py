"""
Family: fundamentals_text_10k  (HC #563 R2 — fundamentals text)

REAL INGEST (HC #563 R7(c)) — pulls 10-K/10-Q filings for the wheel universe,
extracts Item 1A (Risk Factors) and Item 7 (MD&A), computes lightweight text
features (word counts, Loughran-McDonald tone, risk-factor delta, flags).

PIT-safe: every row's `available_from` = filing_date (NOT period-end).

Sources (free, polite UA, ~10 req/sec):
  - https://www.sec.gov/files/company_tickers.json  (ticker -> CIK)
  - https://data.sec.gov/submissions/CIK{cik10}.json  (filings index)
  - https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{primary_doc}

Output:
  data/feature_store/edgar_10k_text/{ticker}.parquet  (per-ticker, incremental)
  data/feature_store/edgar_10k_text/_all.parquet      (unified at end)
"""
from __future__ import annotations
import sys, os, time, json, re, traceback, html as html_lib
from pathlib import Path
import pandas as pd
import requests

sys.path.insert(0, "/home/jupiter/Lvl3Quant/research/ingest")
from _common import SEC_HEADERS, STORE

FAMILY = "edgar_10k_text"
OUT_DIR = STORE / FAMILY
OUT_DIR.mkdir(parents=True, exist_ok=True)

UNIVERSE_PARQUET = Path("/home/jupiter/Lvl3Quant/wheel_strategy_v1/data/cache/universe_v2.parquet")
START_DATE = "2018-01-01"
MAX_FILINGS_PER_TICKER = 40
SLEEP = 0.12  # 10 req/sec cap

# ----- Loughran-McDonald (abbreviated; ~50 pos + ~70 neg of the most common terms)
LM_POSITIVE = {
    "able","achieve","achieved","achievement","advancement","attain","attained","benefit",
    "boost","collaborate","collaboration","compliment","conducive","constructive","creative",
    "delight","desirable","despite","effective","efficiency","efficient","enhance","enhanced",
    "enhancement","enjoy","enjoyed","excellent","exclusive","exceptional","favorable","gain",
    "gained","good","great","greater","highest","honor","improve","improved","improvement",
    "improving","innovate","innovation","innovative","leading","loyal","opportunity",
    "outperform","outperformed","pleased","positive","progress","prosper","prosperous",
    "reward","rewarded","satisfactory","satisfy","strength","strong","stronger","strongest",
    "succeed","success","successful","successfully","superior","tremendous","upturn","valuable","won",
}
LM_NEGATIVE = {
    "abandon","abandoned","abandoning","abandons","abnormal","abolish","abrupt","absence",
    "accident","accuse","accused","adverse","adversely","aggravate","alleged","allegation",
    "annul","anomaly","bad","bankrupt","bankruptcy","barred","breach","breached","breaches",
    "burden","burdened","cancel","cancelled","catastrophe","catastrophic","caution","cease",
    "challenge","challenged","claim","claimed","claims","collapse","collusion","complaint",
    "concern","concerned","concerns","confess","confiscate","conspire","contempt","contract",
    "corrupt","corruption","crisis","critical","criticism","criticize","damage","damaged",
    "damages","danger","dangerous","decline","declined","declines","deficient","deficiency",
    "defendant","defer","deferred","delay","delayed","delays","demolish","deny","denied",
    "deprive","destroy","destroyed","deteriorate","deteriorated","detrimental","difficult",
    "difficulty","diminish","diminished","disappoint","disappointed","disaster","discontinue",
    "discontinued","dispute","disputed","disruption","downgrade","downgraded","downturn",
    "drag","drop","dropped","drought","dysfunction","erode","eroded","erosion","fail",
    "failed","failure","failures","fall","fallen","falsified","fault","faulty","fear",
    "fired","force","forced","forfeit","fraud","fraudulent","harm","harmed","harmful",
    "hazard","hazardous","hostile","hurt","illegal","impair","impaired","impairment",
    "impede","impossible","inability","inaccurate","inadequate","incident","insolvent",
    "instability","interfere","interference","investigation","investigations","lack","lacked",
    "lacking","lawsuit","lawsuits","layoff","layoffs","liability","liable","limitation",
    "limited","litigation","lose","loses","losing","loss","losses","lost","malfunction",
    "mislead","misleading","misled","misstate","misstated","misstatement","negative","neglect",
    "obstruct","obstruction","penalty","peril","plaintiff","plead","preclude","problem",
    "problems","prohibit","prohibited","prosecute","prosecuted","question","questionable",
    "recall","recession","reclassification","reduce","reduced","reduction","reject","rejected",
    "restate","restated","restatement","restrict","restricted","restriction","restructure",
    "restructured","restructuring","scrutiny","seize","seized","setback","severe","shortfall",
    "shut","shutdown","slow","slowdown","slowed","stoppage","strain","strained","subpoena",
    "subpoenaed","suffer","suffered","suit","suspend","suspended","terminate","terminated",
    "termination","threat","threaten","threatened","threats","tragedy","trouble","troubled",
    "uncertain","uncertainty","unable","unanticipated","underestimate","undermine",
    "unfavorable","unforeseen","unlawful","unprofitable","unstable","unsuccessful",
    "violate","violated","violation","volatile","volatility","vulnerable","warn","warned",
    "weak","weakened","weakness","worse","worsen","worsened","worst","wrong","wrongful",
}

WORD_RE = re.compile(r"[A-Za-z']+")

# Item header regex -- finds every "Item NN" anchor with start position.
# Used to enumerate candidate section starts, then score by content length.
RE_ITEM_HEADER = re.compile(
    r"item\s*(\d+)([a-z]?)\b",
    re.IGNORECASE,
)
# Specific anchors (must be followed by the section title to avoid generic references).
RE_ITEM_1A_ANCHOR = re.compile(
    r"item\s*1a[\.\s\u2014\-:]*\s*risk\s*factors",
    re.IGNORECASE,
)
RE_ITEM_7_ANCHOR = re.compile(
    r"item\s*7[\.\s\u2014\-:]*\s*management(?:['\u2019]s|s)?\s*discussion",
    re.IGNORECASE,
)
# Item 2 in a 10-Q is "Management's Discussion and Analysis ..."
RE_ITEM_2_MDA_ANCHOR = re.compile(
    r"item\s*2[\.\s\u2014\-:]*\s*management(?:['\u2019]s|s)?\s*discussion",
    re.IGNORECASE,
)

TAG_RE = re.compile(r"<[^>]+>")
SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
WS_RE = re.compile(r"\s+")

HOST_WWW = {**SEC_HEADERS, "Host": "www.sec.gov"}
HOST_DATA = {**SEC_HEADERS, "Host": "data.sec.gov"}


def polite_get(url: str, host: str = "www") -> requests.Response | None:
    headers = HOST_WWW if host == "www" else HOST_DATA
    try:
        r = requests.get(url, headers=headers, timeout=30)
        time.sleep(SLEEP)
        return r
    except Exception as e:
        print(f"  request error {url}: {e}", flush=True)
        time.sleep(SLEEP)
        return None


def fetch_ticker_cik_map() -> dict[str, str]:
    """Returns {ticker_upper: cik10}."""
    url = "https://www.sec.gov/files/company_tickers.json"
    r = polite_get(url, host="www")
    if r is None or r.status_code != 200:
        raise RuntimeError(f"failed to fetch company_tickers.json: {r.status_code if r else 'no resp'}")
    j = r.json()
    out = {}
    for _, rec in j.items():
        tk = str(rec.get("ticker", "")).upper().strip()
        cik = int(rec.get("cik_str", 0))
        if tk and cik:
            out[tk] = f"{cik:010d}"
    return out


def fetch_submissions(cik10: str) -> dict | None:
    url = f"https://data.sec.gov/submissions/CIK{cik10}.json"
    r = polite_get(url, host="data")
    if r is None or r.status_code != 200:
        return None
    try:
        return r.json()
    except Exception:
        return None


def fetch_filing_doc(cik10: str, accession: str, primary_doc: str) -> str | None:
    cik_int = int(cik10)
    acc_nodash = accession.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{primary_doc}"
    r = polite_get(url, host="www")
    if r is None:
        return None
    if r.status_code != 200:
        return None
    try:
        return r.text
    except Exception:
        return None


def strip_html(text: str) -> str:
    text = SCRIPT_STYLE_RE.sub(" ", text)
    text = TAG_RE.sub(" ", text)
    # Decode HTML entities (handles &#8217;, &amp;, &nbsp;, etc.)
    text = html_lib.unescape(text)
    # Replace non-breaking and other unicode whitespace
    text = text.replace("\xa0", " ").replace("\u2003", " ").replace("\u2002", " ")
    text = WS_RE.sub(" ", text)
    return text.strip()


def _is_inline_reference(text: str, header_start: int, header_end: int) -> bool:
    """Heuristic: a real header is followed by section title or period/colon.
    An inline reference is followed by 'of ', 'and ', 'to ', etc."""
    follow = text[header_end:header_end + 30].lstrip()
    follow_lower = follow.lower()
    # Inline reference patterns
    for prefix in ("of part", "of this", "of our", "of the form", "and item", "and part",
                   "to part", "to this", "is incorporated", "for information",
                   "for additional", "for more", "for the fiscal", "for a discussion",
                   "of our annual"):
        if follow_lower.startswith(prefix):
            return True
    return False


def _find_best_section(text: str, anchor_re: re.Pattern, end_item_numbers: tuple[str, ...]) -> str:
    """Find ALL anchor occurrences, pick the one yielding the longest body (skips ToC entries).

    end_item_numbers: e.g. ("1B", "2") -- the next-item headers that terminate this section.
    """
    matches = list(anchor_re.finditer(text))
    if not matches:
        return ""
    best = ""
    for am in matches:
        start = am.end()
        # Find next REAL item header after this start position (skip inline references)
        end = len(text)
        for hm in RE_ITEM_HEADER.finditer(text, start + 50):
            num = hm.group(1)
            letter = (hm.group(2) or "").upper()
            tag = f"{num}{letter}"
            if tag not in end_item_numbers:
                continue
            if _is_inline_reference(text, hm.start(), hm.end()):
                continue
            end = hm.start()
            break
        body = text[start:end].strip()
        if len(body) > len(best):
            best = body
    if len(best) > 500_000:
        best = best[:500_000]
    return best


def extract_sections(raw: str, form_type: str = "10-K") -> tuple[str, str]:
    """Return (item_1a_text, mdna_text). Empty strings if not found.

    For 10-K: Item 1A -> 1B/2; Item 7 -> 7A/8.
    For 10-Q: Item 1A (Part II) -> 2/3/4/5/6; Item 2 (Part I MD&A) -> 3/4.
    """
    text = strip_html(raw)
    is_10k = form_type.upper().startswith("10-K")

    # Item 1A is present in both 10-K (Part I) and 10-Q (Part II, often shorter / "no material changes")
    rf = _find_best_section(text, RE_ITEM_1A_ANCHOR, ("1B", "2", "3"))

    if is_10k:
        md = _find_best_section(text, RE_ITEM_7_ANCHOR, ("7A", "8"))
    else:
        md = _find_best_section(text, RE_ITEM_2_MDA_ANCHOR, ("3", "4"))

    return rf, md


def word_tokens(text: str) -> list[str]:
    return [w.lower() for w in WORD_RE.findall(text)]


def compute_features(rf: str, md: str) -> dict:
    rf_tokens = word_tokens(rf)
    md_tokens = word_tokens(md)
    all_tokens = rf_tokens + md_tokens
    pos = sum(1 for w in all_tokens if w in LM_POSITIVE)
    neg = sum(1 for w in all_tokens if w in LM_NEGATIVE)
    tone = (pos - neg) / (pos + neg + 1)
    combined_lower = (rf + " " + md).lower()
    return {
        "rf_word_count": len(rf_tokens),
        "mdna_word_count": len(md_tokens),
        "lm_positive_count": pos,
        "lm_negative_count": neg,
        "tone_score": tone,
        "flag_going_concern": ("going concern" in combined_lower),
        "flag_accounting_change": ("change in accounting" in combined_lower),
        "flag_restatement": ("restatement" in combined_lower),
        "rf_text_excerpt": rf[:2000],
        "mdna_text_excerpt": md[:2000],
    }


def load_universe() -> list[str]:
    df = pd.read_parquet(UNIVERSE_PARQUET)
    return [str(t).upper().strip() for t in df["ticker"].tolist() if str(t).strip()]


def process_ticker(ticker: str, cik10: str) -> pd.DataFrame | None:
    """Pull and process all 10-K/10-Q filings >= START_DATE for one ticker."""
    sub = fetch_submissions(cik10)
    if sub is None:
        return None
    recs = sub.get("filings", {}).get("recent", {})
    forms = recs.get("form", [])
    dates = recs.get("filingDate", [])
    accs = recs.get("accessionNumber", [])
    primaries = recs.get("primaryDocument", [])
    reports = recs.get("reportDate", [])

    filings = []
    for f, d, a, p, rd in zip(forms, dates, accs, primaries, reports):
        if f not in ("10-K", "10-Q"):
            continue
        if d < START_DATE:
            continue
        filings.append({"form": f, "filing_date": d, "accession": a, "primary": p, "period": rd})

    # Also check older "files" entries -- additional JSON files referenced for deep history
    older_files = sub.get("filings", {}).get("files", [])
    for ofile in older_files:
        fname = ofile.get("name")
        if not fname:
            continue
        url = f"https://data.sec.gov/submissions/{fname}"
        r = polite_get(url, host="data")
        if r is None or r.status_code != 200:
            continue
        try:
            j2 = r.json()
        except Exception:
            continue
        forms2 = j2.get("form", [])
        dates2 = j2.get("filingDate", [])
        accs2 = j2.get("accessionNumber", [])
        prims2 = j2.get("primaryDocument", [])
        reps2 = j2.get("reportDate", [])
        for f, d, a, p, rd in zip(forms2, dates2, accs2, prims2, reps2):
            if f not in ("10-K", "10-Q"):
                continue
            if d < START_DATE:
                continue
            filings.append({"form": f, "filing_date": d, "accession": a, "primary": p, "period": rd})

    if not filings:
        return None

    # Sort oldest -> newest, cap
    filings.sort(key=lambda r: r["filing_date"])
    filings = filings[-MAX_FILINGS_PER_TICKER:]

    rows = []
    prev_rf_count = None
    for fl in filings:
        if not fl["primary"]:
            continue
        try:
            doc = fetch_filing_doc(cik10, fl["accession"], fl["primary"])
        except Exception as e:
            print(f"  {ticker} {fl['accession']} fetch err: {e}", flush=True)
            continue
        if doc is None:
            continue
        try:
            rf, md = extract_sections(doc, fl["form"])
        except Exception as e:
            print(f"  {ticker} {fl['accession']} extract err: {e}", flush=True)
            continue
        if not rf and not md:
            # Can't extract -- skip
            continue
        feats = compute_features(rf, md)
        rf_delta = (feats["rf_word_count"] - prev_rf_count) if prev_rf_count is not None else 0
        prev_rf_count = feats["rf_word_count"]
        row = {
            "ticker": ticker,
            "cik": cik10,
            "form_type": fl["form"],
            "filing_date": fl["filing_date"],
            "period_of_report": fl["period"],
            "accession": fl["accession"],
            "primary_doc": fl["primary"],
            "available_from": fl["filing_date"],  # PIT-safe
            "rf_delta_word_count": rf_delta,
            **feats,
        }
        rows.append(row)

    if not rows:
        return None
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] EDGAR 10-K/10-Q ingest start", flush=True)
    print(f"  universe: {UNIVERSE_PARQUET}", flush=True)
    print(f"  output:   {OUT_DIR}", flush=True)

    universe = load_universe()
    print(f"  loaded {len(universe)} tickers from universe", flush=True)

    print("  fetching ticker->CIK map from SEC...", flush=True)
    cik_map = fetch_ticker_cik_map()
    print(f"  CIK map has {len(cik_map)} tickers", flush=True)

    n_done = 0
    n_skipped = 0
    n_failed = 0
    n_resumed = 0
    n_new = 0
    for i, ticker in enumerate(universe, 1):
        out_path = OUT_DIR / f"{ticker}.parquet"
        if out_path.exists():
            n_resumed += 1
            n_done += 1
            if i % 10 == 0:
                rate = i / (time.time() - t0)
                eta = (len(universe) - i) / max(rate, 1e-6)
                print(f"[{time.strftime('%H:%M:%S')}] {i}/{len(universe)} (resume cache) rate={rate:.2f} t/s eta={eta/60:.1f}min", flush=True)
            continue

        cik10 = cik_map.get(ticker)
        if cik10 is None:
            print(f"  {ticker}: no CIK -- skip", flush=True)
            n_skipped += 1
            continue

        try:
            df = process_ticker(ticker, cik10)
        except Exception as e:
            print(f"  {ticker}: FAILED {e}", flush=True)
            traceback.print_exc()
            n_failed += 1
            continue

        if df is None or df.empty:
            print(f"  {ticker}: 0 filings extracted", flush=True)
            n_skipped += 1
            # Write empty marker so we don't retry every run
            pd.DataFrame([{"ticker": ticker, "filing_date": None, "form_type": None,
                           "_empty": True}]).to_parquet(out_path, index=False)
            continue

        df.to_parquet(out_path, index=False)
        n_new += 1
        n_done += 1

        if i % 10 == 0 or i == len(universe):
            rate = i / (time.time() - t0)
            eta = (len(universe) - i) / max(rate, 1e-6)
            print(f"[{time.strftime('%H:%M:%S')}] {i}/{len(universe)} done={n_done} new={n_new} skip={n_skipped} fail={n_failed} resume={n_resumed} rate={rate:.2f} t/s eta={eta/60:.1f}min", flush=True)

    # Unified parquet
    print(f"[{time.strftime('%H:%M:%S')}] writing unified _all.parquet ...", flush=True)
    parts = []
    for p in sorted(OUT_DIR.glob("*.parquet")):
        if p.name == "_all.parquet":
            continue
        try:
            d = pd.read_parquet(p)
            if "_empty" in d.columns and d["_empty"].all():
                continue
            parts.append(d)
        except Exception as e:
            print(f"  read err {p}: {e}", flush=True)
    if parts:
        all_df = pd.concat(parts, ignore_index=True)
        all_path = OUT_DIR / "_all.parquet"
        all_df.to_parquet(all_path, index=False)
        print(f"  wrote {all_path}  rows={len(all_df)}  tickers={all_df['ticker'].nunique()}", flush=True)
    else:
        print("  no parts to unify", flush=True)

    dt = time.time() - t0
    print(f"[{time.strftime('%H:%M:%S')}] DONE in {dt/60:.1f}min  done={n_done} new={n_new} skip={n_skipped} fail={n_failed} resume={n_resumed}", flush=True)


if __name__ == "__main__":
    main()
