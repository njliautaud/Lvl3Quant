"""
ingest_reddit_sentiment.py

Builds daily Reddit-mention sentiment features for the wheel research feature
store from three subreddits: r/wallstreetbets, r/stocks, r/investing.

Primary backend: Arctic-shift (https://arctic-shift.photon-reddit.com/api).
  Posts endpoint:  /posts/search
  Params:  subreddit, after (unix), before (unix), limit, title (regex/substring),
           selftext, sort, fields
Fallback:        https://api.pullpush.io/reddit/search/submission
  Params:  subreddit, q, after, before, size

VADER (vaderSentiment) provides crude sentiment on (title + selftext).

Per (ticker, UTC date) we record:
  wsb_mentions, stocks_mentions, investing_mentions, total_upvotes,
  avg_vader_compound, mention_zscore_20d

Output: data/cache/reddit_sentiment_daily.parquet  (long, ticker+date keyed)

Run:
    python -m wheel_strategy_v1.data.ingest_reddit_sentiment --smoke
    python -m wheel_strategy_v1.data.ingest_reddit_sentiment --workers 4
"""
from __future__ import annotations
import argparse
import re
import sys
import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError as e:
    print("[reddit] vaderSentiment missing — pip install vaderSentiment", file=sys.stderr)
    raise

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
OUT_PATH = CACHE / "reddit_sentiment_daily.parquet"

ARCTIC_BASE = "https://arctic-shift.photon-reddit.com/api/posts/search"
PULLPUSH_BASE = "https://api.pullpush.io/reddit/search/submission"

SUBS = ["wallstreetbets", "stocks", "investing"]
SUB_COL = {
    "wallstreetbets": "wsb_mentions",
    "stocks": "stocks_mentions",
    "investing": "investing_mentions",
}


# ---------------------------------------------------------------------------
# Rate limiter shared across worker threads (polite default 4 rps)
# ---------------------------------------------------------------------------
class RateLimiter:
    def __init__(self, rps: float = 4.0):
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
# Ticker pattern: $TICKER OR standalone-uppercase TICKER OR company first-word
# ---------------------------------------------------------------------------
_NAME_STOP = {"inc", "corp", "co", "company", "ltd", "plc", "the", "&",
              "group", "holdings", "international", "industries"}
# Common false-positive 3-letter words masquerading as tickers
_FALSE_POS = {"FOR", "AND", "THE", "ARE", "YOU", "ALL", "NEW", "USA", "CEO",
              "IPO", "ETF", "USD", "GDP", "FED", "DOJ", "FBI", "CIA", "PSA",
              "DM", "PM", "AM", "LOL", "LMAO", "WSB", "FUD", "EPS", "P/E",
              "OG", "TLDR", "EOD", "ATH", "ATL", "MOON", "YOLO"}

def _name_first_word(name: str) -> str | None:
    if not isinstance(name, str):
        return None
    parts = re.split(r"[\s,.]+", name.strip())
    for w in parts:
        w = w.strip()
        if w and w.lower() not in _NAME_STOP and len(w) >= 4:
            return w
    return None


def build_ticker_matchers(universe: pd.DataFrame) -> dict[str, re.Pattern]:
    pats: dict[str, re.Pattern] = {}
    for _, r in universe.iterrows():
        tic = str(r["ticker"]).strip().upper()
        nm = _name_first_word(r.get("name") or "")
        # Always allow $TICKER (case-insensitive)
        alts = [rf"\${re.escape(tic)}\b"]
        if tic not in _FALSE_POS:
            # standalone uppercase (must be word-bound by non-letter)
            alts.append(rf"(?<![A-Za-z])\b{re.escape(tic)}\b(?![A-Za-z])")
        if nm:
            alts.append(rf"(?i)\b{re.escape(nm)}\b")
        pats[tic] = re.compile("|".join(alts))
    return pats


# ---------------------------------------------------------------------------
# HTTP layer with fallback
# ---------------------------------------------------------------------------
def _arctic_fetch(session, sub, after_ts, before_ts, limiter):
    """Pull ALL posts in a (sub, day) window from Arctic-shift, paginated."""
    out = []
    cursor = after_ts
    for _ in range(40):  # cap pages at 40 * 100 = 4000 posts/day/sub
        limiter.wait()
        params = {
            "subreddit": sub,
            "after": cursor,
            "before": before_ts,
            "limit": 100,
            "sort": "asc",
            "fields": "id,created_utc,title,selftext,score,subreddit",
        }
        try:
            r = session.get(ARCTIC_BASE, params=params, timeout=60)
            if r.status_code == 429:
                time.sleep(5)
                continue
            r.raise_for_status()
            j = r.json()
        except Exception as e:  # noqa: BLE001
            raise
        items = j.get("data") or []
        if not items:
            break
        out.extend(items)
        if len(items) < 100:
            break
        # next page = after the latest created_utc we saw
        cursor = int(max(it.get("created_utc", cursor) for it in items)) + 1
        if cursor >= before_ts:
            break
    return out


def _pullpush_fetch(session, sub, after_ts, before_ts, limiter):
    """Fallback to pullpush — paginate by shrinking 'before' to oldest seen."""
    out = []
    before = before_ts
    for _ in range(40):
        limiter.wait()
        params = {
            "subreddit": sub,
            "after": after_ts,
            "before": before,
            "size": 100,
            "sort": "desc",
        }
        try:
            r = session.get(PULLPUSH_BASE, params=params, timeout=60)
            if r.status_code == 429:
                time.sleep(5)
                continue
            r.raise_for_status()
            j = r.json()
        except Exception:
            raise
        items = j.get("data") or []
        if not items:
            break
        out.extend(items)
        if len(items) < 100:
            break
        # next page = older than oldest seen
        before = int(min(it.get("created_utc", before) for it in items)) - 1
        if before <= after_ts:
            break
    return out


def fetch_day_sub(session, sub, day_utc, limiter):
    """Pull all posts in [day_utc, day_utc + 1d) for subreddit sub.

    Tries Arctic-shift first, falls back to pullpush on failure.
    """
    start_dt = datetime.combine(day_utc, datetime.min.time(), tzinfo=timezone.utc)
    after_ts = int(start_dt.timestamp())
    before_ts = after_ts + 86400
    try:
        items = _arctic_fetch(session, sub, after_ts, before_ts, limiter)
        return items, "arctic"
    except Exception as e:  # noqa: BLE001
        # fallback
        try:
            items = _pullpush_fetch(session, sub, after_ts, before_ts, limiter)
            return items, "pullpush"
        except Exception as e2:
            print(f"[reddit] FAIL {sub} {day_utc}: arctic={e} pullpush={e2}",
                  file=sys.stderr)
            return [], "none"


# ---------------------------------------------------------------------------
# Aggregate one day's posts -> per-ticker counts + sentiment
# ---------------------------------------------------------------------------
def aggregate_posts(posts_by_sub: dict[str, list[dict]],
                    patterns: dict[str, re.Pattern],
                    sia: SentimentIntensityAnalyzer) -> dict[str, dict]:
    agg: dict[str, dict] = defaultdict(lambda: {
        "wsb_mentions": 0, "stocks_mentions": 0, "investing_mentions": 0,
        "total_upvotes": 0, "vader_sum": 0.0, "vader_n": 0,
    })
    # Precompute compound per post -> avoid recomputing per ticker
    for sub, posts in posts_by_sub.items():
        col = SUB_COL.get(sub)
        if not col:
            continue
        for p in posts:
            text = ((p.get("title") or "") + " " + (p.get("selftext") or "")).strip()
            if not text:
                continue
            comp = sia.polarity_scores(text[:2000])["compound"]
            score = int(p.get("score") or 0)
            # find which tickers are mentioned
            matched = []
            for tic, pat in patterns.items():
                if pat.search(text):
                    matched.append(tic)
            for tic in matched:
                a = agg[tic]
                a[col] += 1
                a["total_upvotes"] += score
                a["vader_sum"] += comp
                a["vader_n"] += 1
    return agg


# ---------------------------------------------------------------------------
# Resume helper
# ---------------------------------------------------------------------------
def existing_dates(path: Path) -> set:
    if not path.exists():
        return set()
    try:
        df = pd.read_parquet(path, columns=["date"])
        return set(pd.to_datetime(df["date"]).dt.normalize().unique())
    except Exception:
        return set()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="universe_v2.parquet")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default=None,
                    help="UTC end date (exclusive). Default = today.")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rps", type=float, default=4.0)
    ap.add_argument("--smoke", action="store_true",
                    help="30 days, 10 popular tickers, separate output path.")
    args = ap.parse_args()

    universe = pd.read_parquet(CACHE / args.universe)
    if args.smoke:
        popular = ["SPY", "AAPL", "TSLA", "NVDA", "GME", "AMC", "MSFT",
                   "AMZN", "META", "GOOG"]
        universe = universe[universe["ticker"].isin(popular)].copy()
        have = set(universe["ticker"])
        for t in popular:
            if t not in have:
                universe = pd.concat([universe, pd.DataFrame(
                    [{"ticker": t, "name": t, "sector": "", "source": "smoke"}])],
                    ignore_index=True)
        out_path = CACHE / "reddit_sentiment_smoke.parquet"
        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=30)
    else:
        out_path = OUT_PATH
        start_date = pd.Timestamp(args.start).date()
        end_date = (pd.Timestamp(args.end).date() if args.end
                    else datetime.now(timezone.utc).date())

    patterns = build_ticker_matchers(universe)
    sia = SentimentIntensityAnalyzer()
    print(f"[reddit] universe={len(patterns)} tickers, output={out_path}")
    print(f"[reddit] date range: {start_date} -> {end_date} UTC ({(end_date-start_date).days} days)")

    session = requests.Session()
    session.headers["User-Agent"] = "wheel-research-ingest/1.0 (contact: research@local)"
    limiter = RateLimiter(rps=args.rps)

    # Build day list, skip already-cached
    all_days = [start_date + timedelta(days=i)
                for i in range((end_date - start_date).days)]
    if not args.smoke:
        cached = {d.date() for d in existing_dates(out_path)}
        before = len(all_days)
        all_days = [d for d in all_days if d not in cached]
        print(f"[reddit] resume: skipping {before - len(all_days)} cached days, "
              f"{len(all_days)} to fetch")

    if not all_days:
        print("[reddit] nothing to do.")
        return

    all_rows: list[dict] = []
    t0 = time.time()

    # Each "task" = (day, sub). Parallelize across (day, sub).
    tasks = [(d, s) for d in all_days for s in SUBS]

    # group posts by day before aggregating so we can flush per day
    posts_by_day: dict = defaultdict(lambda: {s: [] for s in SUBS})
    fetched_backends = defaultdict(int)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_day_sub, session, sub, d, limiter): (d, sub)
                for (d, sub) in tasks}
        done = 0
        for f in as_completed(futs):
            d, sub = futs[f]
            try:
                items, backend = f.result()
            except Exception as e:  # noqa: BLE001
                print(f"[reddit] task error {sub} {d}: {e}", file=sys.stderr)
                items, backend = [], "err"
            posts_by_day[d][sub] = items
            fetched_backends[backend] += 1
            done += 1
            if done % 30 == 0:
                elapsed = time.time() - t0
                eta = elapsed / done * (len(tasks) - done)
                print(f"[reddit] {done}/{len(tasks)} tasks done "
                      f"elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m "
                      f"backends={dict(fetched_backends)}")

    print(f"[reddit] all fetched. backends={dict(fetched_backends)}. aggregating ...")

    for d in sorted(posts_by_day.keys()):
        per_sub = posts_by_day[d]
        agg = aggregate_posts(per_sub, patterns, sia)
        for tic, v in agg.items():
            total_mentions = (v["wsb_mentions"] + v["stocks_mentions"]
                              + v["investing_mentions"])
            if total_mentions == 0:
                continue
            avg_compound = (v["vader_sum"] / v["vader_n"]) if v["vader_n"] else float("nan")
            all_rows.append({
                "ticker": tic,
                "date": pd.Timestamp(d),
                "wsb_mentions": v["wsb_mentions"],
                "stocks_mentions": v["stocks_mentions"],
                "investing_mentions": v["investing_mentions"],
                "total_upvotes": v["total_upvotes"],
                "avg_vader_compound": avg_compound,
            })

    if not all_rows:
        print("[reddit] no rows produced.")
        return

    new_df = pd.DataFrame(all_rows).sort_values(["ticker", "date"]).reset_index(drop=True)
    new_df["total_mentions"] = (new_df["wsb_mentions"] + new_df["stocks_mentions"]
                                + new_df["investing_mentions"])

    def _z(g: pd.DataFrame) -> pd.DataFrame:
        m = g["total_mentions"].rolling(20, min_periods=5).mean()
        s = g["total_mentions"].rolling(20, min_periods=5).std()
        g["mention_zscore_20d"] = (g["total_mentions"] - m) / s
        return g
    new_df = new_df.groupby("ticker", group_keys=False).apply(_z)
    new_df = new_df.drop(columns=["total_mentions"])

    if out_path.exists() and not args.smoke:
        old = pd.read_parquet(out_path)
        old["date"] = pd.to_datetime(old["date"])
        combined = pd.concat([old, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["ticker", "date"], keep="last")
        combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)
    else:
        combined = new_df

    combined.to_parquet(out_path, index=False)
    print(f"[reddit] wrote {out_path} rows={len(combined):,} "
          f"tickers={combined['ticker'].nunique()}")

    # Validation: show TSLA recent
    if "TSLA" in combined["ticker"].values:
        tsla = combined[combined["ticker"] == "TSLA"].tail(10)
        print("\n[reddit] TSLA last 10 rows:")
        print(tsla[["date", "wsb_mentions", "stocks_mentions",
                    "investing_mentions", "total_upvotes",
                    "avg_vader_compound"]].to_string(index=False))


if __name__ == "__main__":
    main()
