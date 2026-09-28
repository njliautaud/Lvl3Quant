"""
ingest_gdelt_sentiment.py

Builds daily news-sentiment features from the GDELT 2.0 Global Knowledge Graph
(GKG) for the wheel research feature store. Free, no auth.

GDELT GKG is published every 15 min as TSV.zip at:
  http://data.gdeltproject.org/gdeltv2/{YYYYMMDDHHMMSS}.gkg.csv.zip
The master list of all files lives at:
  http://data.gdeltproject.org/gdeltv2/masterfilelist.txt

GKG columns we use (tab-separated, 27 fields):
   2  DATE              YYYYMMDDHHMMSS
   8  V2Themes          theme;theme;...
  14  V2Names           Name1,offset;Name2,offset;...     (used for ticker match)
  16  V2Tone            tone,pos,neg,polarity,act,grp,wc  (first val is avg tone)

We aggregate per (ticker, UTC date):
  mention_ct          count of GKG records mentioning the ticker
  avg_tone            mean V2Tone (column 0 of the V2Tone tuple) across mentions
  pos_ct              count of records with tone > +1
  neg_ct              count of records with tone < -1
  themes_top5         5 most common themes among the records (";"-joined)
And derive:
  tone_zscore_20d     (avg_tone - 20d_mean) / 20d_std

Output: data/cache/gdelt_sentiment_daily.parquet (long, keyed ticker+date)

Run:
    python -m wheel_strategy_v1.data.ingest_gdelt_sentiment --smoke
    python -m wheel_strategy_v1.data.ingest_gdelt_sentiment --workers 8
"""
from __future__ import annotations
import argparse
import io
import re
import sys
import time
import threading
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
OUT_PATH = CACHE / "gdelt_sentiment_daily.parquet"

GDELT_BASE = "http://data.gdeltproject.org/gdeltv2"
MASTERLIST_URL = f"{GDELT_BASE}/masterfilelist.txt"

# ---------------------------------------------------------------------------
# Polite-rate throttle (<= 10 req/sec across all worker threads)
# ---------------------------------------------------------------------------
class RateLimiter:
    def __init__(self, rps: float = 10.0):
        self.min_interval = 1.0 / rps
        self.lock = threading.Lock()
        self.last = 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            dt = now - self.last
            if dt < self.min_interval:
                time.sleep(self.min_interval - dt)
            self.last = time.monotonic()


# ---------------------------------------------------------------------------
# Ticker matching: GKG V2Names is general entity names (people, orgs).
# For ticker matching we look for the literal ticker symbol or the company name
# as a whole word in the joined Themes+Names+Organizations fields.
# We pre-compile a regex per ticker for case-sensitive ticker symbol matching
# (ALL-CAPS) plus a case-insensitive company-name match (first word of name).
# ---------------------------------------------------------------------------
_NAME_STOP = {"inc", "corp", "co", "company", "ltd", "plc", "the", "&", "group",
              "holdings", "international", "industries"}

def _name_first_word(name: str) -> str | None:
    if not isinstance(name, str):
        return None
    w = re.split(r"[\s,.]+", name.strip())[0].strip()
    if not w or w.lower() in _NAME_STOP:
        # take 2nd word
        parts = re.split(r"[\s,.]+", name.strip())
        for p in parts[1:]:
            if p and p.lower() not in _NAME_STOP and len(p) >= 4:
                return p
        return None
    return w if len(w) >= 3 else None


def build_ticker_patterns(universe: pd.DataFrame) -> dict[str, re.Pattern]:
    """For each ticker, a regex matching the ticker symbol OR the company first-word.

    Symbol must appear as a standalone all-caps token to reduce false positives.
    Name match is case-insensitive on the first significant word of the name.
    """
    pats: dict[str, re.Pattern] = {}
    for _, r in universe.iterrows():
        tic = str(r["ticker"]).strip().upper()
        nm = _name_first_word(r.get("name") or "")
        # ticker as standalone uppercase word, OR optional dollar-prefix ($AAPL)
        alts = [rf"(?<![A-Z]){re.escape(tic)}(?![A-Z])"]
        if nm:
            # company first-word, case-insensitive, word-boundary
            alts.append(rf"(?i)\b{re.escape(nm)}\b")
        pats[tic] = re.compile("|".join(alts))
    return pats


# ---------------------------------------------------------------------------
# Master list parsing -> set of (timestamp_str, url) for all 15-min GKG slots
# ---------------------------------------------------------------------------
_GKG_LINE = re.compile(r"\s(\S+\.gkg\.csv\.zip)\s*$")

def fetch_masterlist(session: requests.Session) -> pd.DataFrame:
    r = session.get(MASTERLIST_URL, timeout=60)
    r.raise_for_status()
    rows = []
    for line in r.text.splitlines():
        if "gkg.csv.zip" not in line:
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        url = parts[-1]
        ts = url.rsplit("/", 1)[-1].split(".")[0]  # YYYYMMDDHHMMSS
        if len(ts) != 14 or not ts.isdigit():
            continue
        rows.append({"timestamp": ts, "url": url})
    df = pd.DataFrame(rows)
    df["dt_utc"] = pd.to_datetime(df["timestamp"], format="%Y%m%d%H%M%S", utc=True)
    df["date"] = df["dt_utc"].dt.date
    return df


# ---------------------------------------------------------------------------
# Parse a single GKG 15-min slot -> per-ticker partial aggregates
# Returns: { ticker: dict(mentions=int, tone_sum=float, pos=int, neg=int,
#                         themes=Counter) }
# ---------------------------------------------------------------------------
def parse_gkg_slot(content: bytes, patterns: dict[str, re.Pattern]) -> dict:
    out: dict[str, dict] = defaultdict(
        lambda: {"mentions": 0, "tone_sum": 0.0, "pos": 0, "neg": 0,
                 "themes": Counter()}
    )
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            name = z.namelist()[0]
            raw = z.read(name).decode("utf-8", errors="ignore")
    except (zipfile.BadZipFile, OSError, IndexError):
        return out

    for line in raw.split("\n"):
        if not line:
            continue
        cols = line.split("\t")
        if len(cols) < 16:
            continue
        themes = cols[7] or ""
        names = cols[13] or ""
        orgs = cols[6] or ""
        v2tone = cols[15] or ""
        # text we scan for ticker references = themes + names + orgs + URL/title hints
        haystack = " ".join((themes, names, orgs, cols[4]))
        if not haystack.strip():
            continue
        # parse tone (first comma-sep field)
        try:
            tone = float(v2tone.split(",")[0]) if v2tone else 0.0
        except ValueError:
            tone = 0.0
        # match each ticker pattern against the haystack
        for tic, pat in patterns.items():
            if pat.search(haystack):
                a = out[tic]
                a["mentions"] += 1
                a["tone_sum"] += tone
                if tone > 1.0:
                    a["pos"] += 1
                elif tone < -1.0:
                    a["neg"] += 1
                # top themes from this record
                if themes:
                    for th in themes.split(";"):
                        th = th.strip()
                        if th:
                            a["themes"][th] += 1
    return out


# ---------------------------------------------------------------------------
# Worker: download one slot URL, parse, return per-ticker dict
# ---------------------------------------------------------------------------
def fetch_and_parse(url: str, session: requests.Session, limiter: RateLimiter,
                    patterns: dict, retries: int = 3) -> tuple[str, dict]:
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            limiter.wait()
            r = session.get(url, timeout=60)
            if r.status_code == 404:
                # Missing slot — return empty
                return (url, {})
            r.raise_for_status()
            return (url, parse_gkg_slot(r.content, patterns))
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    print(f"[gdelt] FAIL {url}: {last_err}", file=sys.stderr)
    return (url, {})


# ---------------------------------------------------------------------------
# Aggregate slot results -> per (ticker, date) row
# ---------------------------------------------------------------------------
def aggregate_day(slot_results: list[dict]) -> dict[str, dict]:
    agg: dict[str, dict] = defaultdict(
        lambda: {"mentions": 0, "tone_sum": 0.0, "pos": 0, "neg": 0,
                 "themes": Counter()}
    )
    for slot in slot_results:
        for tic, v in slot.items():
            a = agg[tic]
            a["mentions"] += v["mentions"]
            a["tone_sum"] += v["tone_sum"]
            a["pos"] += v["pos"]
            a["neg"] += v["neg"]
            a["themes"].update(v["themes"])
    return agg


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def existing_keys() -> set[tuple[str, pd.Timestamp]]:
    if not OUT_PATH.exists():
        return set()
    try:
        df = pd.read_parquet(OUT_PATH, columns=["ticker", "date"])
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        return set(zip(df["ticker"].astype(str), df["date"]))
    except Exception:
        return set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="universe_v2.parquet")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default=None,
                    help="UTC end date, exclusive of today. Default = today.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rps", type=float, default=10.0,
                    help="Max GDELT requests per second across all workers")
    ap.add_argument("--smoke", action="store_true",
                    help="30 days, 10 tickers, write to gdelt_sentiment_smoke.parquet")
    args = ap.parse_args()

    universe = pd.read_parquet(CACHE / args.universe)
    if args.smoke:
        # 10 well-known names guaranteed to have coverage
        smoke_tics = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOG",
                      "JPM", "XOM", "SPY"]
        universe = universe[universe["ticker"].isin(smoke_tics)].copy()
        # if some are missing, add a synthetic row so we still have patterns
        have = set(universe["ticker"])
        for t in smoke_tics:
            if t not in have:
                universe = pd.concat([universe, pd.DataFrame(
                    [{"ticker": t, "name": t, "sector": "", "source": "smoke"}])],
                    ignore_index=True)
        out_path = CACHE / "gdelt_sentiment_smoke.parquet"
        n_days = 30
    else:
        out_path = OUT_PATH
        n_days = None  # use --start

    patterns = build_ticker_patterns(universe)
    print(f"[gdelt] universe={len(patterns)} tickers, output={out_path}")

    today_utc = datetime.now(timezone.utc).date()
    end_date = pd.Timestamp(args.end).date() if args.end else today_utc
    if args.smoke:
        # smoke: most recent 30 days that GDELT certainly has
        end_date = today_utc
        start_date = end_date - timedelta(days=n_days)
    else:
        start_date = pd.Timestamp(args.start).date()

    print(f"[gdelt] date range: {start_date} -> {end_date} UTC")

    session = requests.Session()
    session.headers["User-Agent"] = "wheel-research-ingest/1.0"
    limiter = RateLimiter(rps=args.rps)

    print("[gdelt] fetching masterfilelist ...")
    master = fetch_masterlist(session)
    master = master[(master["date"] >= start_date) & (master["date"] < end_date)]
    print(f"[gdelt] master slots in range: {len(master):,}")

    # Resume: skip dates that already have FULL coverage (we just check date-level
    # presence — a partial day is rare since we aggregate all slots together).
    done = existing_keys()
    done_dates = {d for _, d in done}
    if done_dates and not args.smoke:
        before = master["date"].nunique()
        master = master[~master["date"].apply(lambda d: pd.Timestamp(d) in done_dates)]
        after = master["date"].nunique()
        print(f"[gdelt] resume: skipping {before - after} already-cached days")

    if master.empty:
        print("[gdelt] nothing to do.")
        return

    # Group slots by UTC date so we can flush per-day and keep memory bounded
    all_rows: list[dict] = []
    dates = sorted(master["date"].unique())
    t0 = time.time()
    for di, d in enumerate(dates):
        slot_urls = master[master["date"] == d]["url"].tolist()
        slot_results = []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(fetch_and_parse, u, session, limiter, patterns)
                    for u in slot_urls]
            for f in as_completed(futs):
                _, parsed = f.result()
                if parsed:
                    slot_results.append(parsed)
        agg = aggregate_day(slot_results)
        for tic, v in agg.items():
            if v["mentions"] == 0:
                continue
            avg_tone = v["tone_sum"] / v["mentions"]
            top5 = ";".join(t for t, _ in v["themes"].most_common(5))
            all_rows.append({
                "ticker": tic,
                "date": pd.Timestamp(d),
                "mention_ct": v["mentions"],
                "avg_tone": avg_tone,
                "pos_ct": v["pos"],
                "neg_ct": v["neg"],
                "themes_top5": top5,
            })

        elapsed = time.time() - t0
        eta = elapsed / (di + 1) * (len(dates) - di - 1)
        if (di + 1) % 5 == 0 or di == len(dates) - 1:
            print(f"[gdelt] day {di+1}/{len(dates)} ({d}) "
                  f"rows_so_far={len(all_rows):,} "
                  f"elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m")

    new_df = pd.DataFrame(all_rows)
    if new_df.empty:
        print("[gdelt] no rows produced.")
        return

    # 20d rolling tone z-score per ticker
    new_df = new_df.sort_values(["ticker", "date"]).reset_index(drop=True)
    def _z(g: pd.DataFrame) -> pd.DataFrame:
        m = g["avg_tone"].rolling(20, min_periods=5).mean()
        s = g["avg_tone"].rolling(20, min_periods=5).std()
        g["tone_zscore_20d"] = (g["avg_tone"] - m) / s
        return g
    new_df = new_df.groupby("ticker", group_keys=False).apply(_z)

    # Merge with existing if not smoke
    if out_path.exists() and not args.smoke:
        old = pd.read_parquet(out_path)
        old["date"] = pd.to_datetime(old["date"])
        combined = pd.concat([old, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["ticker", "date"], keep="last")
        combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)
    else:
        combined = new_df

    combined.to_parquet(out_path, index=False)
    print(f"[gdelt] wrote {out_path} rows={len(combined):,} "
          f"tickers={combined['ticker'].nunique()}")

    # Validation print
    if "AAPL" in combined["ticker"].values:
        aapl = combined[combined["ticker"] == "AAPL"].tail(10)
        print("\n[gdelt] AAPL last 10 rows:")
        print(aapl[["date", "mention_ct", "avg_tone", "pos_ct", "neg_ct"]]
              .to_string(index=False))
        print(f"\n[gdelt] AAPL mean avg_tone overall: "
              f"{combined[combined['ticker']=='AAPL']['avg_tone'].mean():.3f}")


if __name__ == "__main__":
    main()
