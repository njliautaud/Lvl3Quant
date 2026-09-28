"""
Family: social_hype_reddit  (HC #563 R7(d) — daily Reddit hype features)

REAL INGEST — pulls Reddit post mentions for the wheel universe across
six finance-relevant subreddits using the Arctic-shift mirror
(https://arctic-shift.photon-reddit.com/api/posts/search).

Per (ticker, date) aggregates:
    - mention_count          : number of posts mentioning the ticker
    - upvote_sum             : sum of post.score
    - comment_count          : sum of post.num_comments
    - sentiment_polarity     : mean polarity of post title+selftext
                               (VADER compound if available, else lexicon score)
    - sentiment_volatility   : std of per-post polarity
    - top_subreddit          : modal subreddit for that ticker-date

PIT-safe: post.created_utc is the timestamp; daily aggregate row is
keyed by (ticker, date_utc) and is "available_from" the same date.

Ticker matching discipline:
    - Multi-char "safe" tickers: regex r"\\$?\\b<TICKER>\\b" case-insensitive.
    - Ambiguous (single letter, or common English word like ON/GO/F/T):
      require explicit "$" prefix, i.e. r"\\$<TICKER>\\b".

Polite: ~30 req/min ratelimit (Arctic-shift "slow" scope). We pace at
~2.2 sec between requests and watch x-ratelimit-* headers, with
exponential backoff on 429.

Cache: every API response is cached to disk under
    data/feature_store/social_hype_reddit/_cache/<sha>.json
so re-runs don't re-hit the API.

Output:
    data/feature_store/social_hype_reddit/{ticker}.parquet   (per-ticker, incremental)
    data/feature_store/social_hype_reddit/_all.parquet       (unified at end)
"""
from __future__ import annotations
import sys, os, time, json, re, hashlib, math, traceback
import datetime as dt
from pathlib import Path
from collections import Counter, defaultdict
from urllib.parse import urlencode

import pandas as pd
import requests

sys.path.insert(0, "/home/jupiter/Lvl3Quant/research/ingest")
from _common import STORE

FAMILY = "social_hype_reddit"
OUT_DIR = STORE / FAMILY
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = OUT_DIR / "_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

UNIVERSE_PARQUET = Path("/home/jupiter/Lvl3Quant/wheel_strategy_v1/data/cache/universe_v2.parquet")

SUBREDDITS = [
    "wallstreetbets",
    "stocks",
    "investing",
    "options",
    "ValueInvesting",
    "SecurityAnalysis",
]

START_DATE = "2018-01-01"
START_EPOCH = int(dt.datetime(2018, 1, 1, tzinfo=dt.timezone.utc).timestamp())
END_EPOCH = int(dt.datetime.now(dt.timezone.utc).timestamp())

API_BASE = "https://arctic-shift.photon-reddit.com/api/posts/search"
HEADERS = {
    "User-Agent": "Lvl3Quant-Research/1.0 (research@relentlessrobotics)",
    "Accept": "application/json",
    "Accept-Encoding": "gzip, deflate",
}
# Arctic-shift slow scope = 30 req/min. Pace ~2.1s between requests.
MIN_SLEEP = 2.1
MAX_BACKOFF = 120.0
PAGE_LIMIT = 100  # max allowed

FIELDS = "id,created_utc,score,num_comments,title,selftext,subreddit"

# Ambiguous tickers — require explicit $ prefix
COMMON_WORDS = {
    "A","I","ON","GO","IT","BE","NO","OR","SO","TO","AT","BY","DO","IF",
    "OF","UP","US","WE","ME","HE","IN","AN","AS","FOR","ARE","ALL","ANY",
    "ONE","TWO","HAS","HAD","WAS","SEE","NEW","OUT","OUR","WHO","WHY",
    "HOW","NOW","OWN","SAY","GET","TRY","WAY","HER","HIM","HIS","SHE",
    "THE","NOT","NUM","BIG","BUY","ANY","ICE","FUN","HOT","KEY","LAB",
    "MAX","MIN","PAY","PIE","REG","RUN","SET","SUN","TOP","UPS","WIN",
    # finance noise
    "USD","CEO","CFO","COO","IPO","EPS","ETF","SEC","IRS","FED","CPI",
    "GDP","ATH","ATL","DD","FYI","TLDR","TBH","IMO","YOLO","FOMO","FUD",
    "TA","FA","PT","PR","NA","OK","ER","TR","REE","ALL","CALL","PUT",
    "SELL","HOLD","LONG","SHORT","BULL","BEAR","HIGH","LOW","OPEN","CLOSE",
}

# Build VADER once (preferred) — fall back to small finance lexicon.
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER = SentimentIntensityAnalyzer()
    USE_VADER = True
except Exception:
    _VADER = None
    USE_VADER = False

# Fallback finance lexicon (subset, mirrors edgar_10k_10q_text style).
LEX_POS = {
    "buy","bullish","bull","calls","moon","mooning","rocket","rip","pump",
    "up","gain","gains","green","strong","beat","beats","beating","crush",
    "crushed","upgrade","upgraded","outperform","positive","profit","profits",
    "winner","winning","squeeze","squeezed","rally","rallying","breakout",
    "long","longs","hold","holding","diamond","hands","hodl","yolo","tendies",
    "love","like","great","best","awesome","huge","massive","strong","solid",
    "growth","growing","beat","beating","surge","surging",
}
LEX_NEG = {
    "sell","bearish","bear","puts","crash","crashing","dump","dumping","tank",
    "tanking","tanked","red","down","drop","dropping","dropped","loss","losses",
    "loser","losing","weak","miss","missed","missing","downgrade","downgraded",
    "underperform","negative","bagholder","bag","scam","fraud","fake","short",
    "shorts","shorting","puts","panic","fear","afraid","scary","ugly","bad",
    "worst","terrible","awful","disaster","dead","dying","killed","destroyed",
    "fall","falling","fell","plunge","plunging","collapse","collapsing",
}

WORD_RE = re.compile(r"[A-Za-z']+")

# ---------- helpers ----------

def now_hms():
    return time.strftime("%H:%M:%S")

def cache_key(params: dict) -> Path:
    s = urlencode(sorted(params.items()))
    h = hashlib.sha1(s.encode()).hexdigest()
    return CACHE_DIR / f"{h}.json"

def polite_get(params: dict, retries: int = 5) -> dict | None:
    """GET arctic-shift with caching + backoff. Returns parsed JSON `data` list or None."""
    cf = cache_key(params)
    if cf.exists():
        try:
            with open(cf) as f:
                return json.load(f)
        except Exception:
            pass

    backoff = MIN_SLEEP
    for attempt in range(retries):
        try:
            r = requests.get(API_BASE, params=params, headers=HEADERS, timeout=45)
        except requests.RequestException as e:
            print(f"  net err: {e}; backoff {backoff:.1f}s", flush=True)
            time.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF)
            continue

        # Honor rate-limit headers
        remaining = r.headers.get("x-ratelimit-remaining")
        reset = r.headers.get("x-ratelimit-reset")
        try:
            remaining_i = int(remaining) if remaining is not None else None
            reset_i = int(reset) if reset is not None else None
        except Exception:
            remaining_i, reset_i = None, None

        if r.status_code == 429:
            wait = float(reset_i) if reset_i else backoff
            wait = max(wait, MIN_SLEEP)
            print(f"  429 — sleeping {wait:.1f}s", flush=True)
            time.sleep(wait)
            backoff = min(backoff * 2, MAX_BACKOFF)
            continue

        if r.status_code != 200:
            print(f"  HTTP {r.status_code} — backoff {backoff:.1f}s", flush=True)
            time.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF)
            continue

        try:
            j = r.json()
        except Exception as e:
            print(f"  json err: {e}; backoff {backoff:.1f}s", flush=True)
            time.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF)
            continue

        if j.get("error"):
            err = j["error"]
            if "Timeout" in err or "slow" in err.lower():
                print(f"  api timeout: {err}; backoff {backoff:.1f}s", flush=True)
                time.sleep(backoff)
                backoff = min(backoff * 2, MAX_BACKOFF)
                continue
            # Permanent error (bad param etc) — give up on this query
            print(f"  api error (permanent): {err}", flush=True)
            return {"data": [], "error": err}

        # Success — cache + sleep enough to stay under 30 req/min
        try:
            with open(cf, "w") as f:
                json.dump(j, f)
        except Exception:
            pass

        # If we're near the limit, pace harder
        if remaining_i is not None and remaining_i <= 2 and reset_i is not None and reset_i > 0:
            sleep_for = float(reset_i) + 1
            print(f"  pacing: {remaining_i} remaining, sleeping {sleep_for:.1f}s", flush=True)
            time.sleep(sleep_for)
        else:
            time.sleep(MIN_SLEEP)
        return j

    print(f"  giving up on {params} after {retries} retries", flush=True)
    return None


# ---------- ticker matching ----------

def is_ambiguous_ticker(t: str) -> bool:
    if len(t) <= 1:
        return True
    if t.upper() in COMMON_WORDS:
        return True
    return False

def build_ticker_regex(t: str) -> re.Pattern:
    t_esc = re.escape(t)
    if is_ambiguous_ticker(t):
        # Require $ prefix
        return re.compile(rf"\${t_esc}\b", re.IGNORECASE)
    # Safe ticker: either $TICKER or word-boundary TICKER
    return re.compile(rf"(?:\$|\b){t_esc}\b", re.IGNORECASE)

def ticker_matches(text: str, pat: re.Pattern) -> bool:
    if not text:
        return False
    return bool(pat.search(text))


# ---------- sentiment ----------

def polarity(text: str) -> float:
    if not text:
        return 0.0
    if USE_VADER:
        try:
            return float(_VADER.polarity_scores(text)["compound"])
        except Exception:
            pass
    # Fallback lexicon
    toks = [w.lower() for w in WORD_RE.findall(text)]
    if not toks:
        return 0.0
    pos = sum(1 for w in toks if w in LEX_POS)
    neg = sum(1 for w in toks if w in LEX_NEG)
    if pos + neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)


# ---------- per-ticker query ----------

def fetch_ticker_subreddit_posts(ticker: str, subreddit: str) -> list[dict]:
    """Paginate posts where title matches the ticker, in [START, END].
    Uses sort=asc + after cursor advanced past last post's created_utc."""
    posts = []
    # Two queries: with $ prefix and without — but Arctic-shift's `title` is a
    # substring match, so `title=AAPL` will already match `$AAPL`. We use the
    # bare ticker. We then re-filter in Python with the ticker regex to drop
    # false positives like "AAPL" appearing inside another word.
    query_value = ticker  # substring match
    after = START_EPOCH
    safety_pages = 0
    while True:
        safety_pages += 1
        if safety_pages > 5000:
            print(f"    {ticker}/{subreddit}: page safety cap hit", flush=True)
            break
        params = {
            "subreddit": subreddit,
            "title": query_value,
            "limit": PAGE_LIMIT,
            "after": after,
            "before": END_EPOCH,
            "sort": "asc",
            "fields": FIELDS,
        }
        j = polite_get(params)
        if j is None:
            break
        if j.get("error"):
            break
        arr = j.get("data") or []
        if not arr:
            break
        posts.extend(arr)
        if len(arr) < PAGE_LIMIT:
            break
        # Advance cursor
        last_ts = max(int(p.get("created_utc") or 0) for p in arr)
        if last_ts <= after:
            # No forward progress — abort to avoid loop
            break
        after = last_ts + 1
    return posts


# ---------- aggregation ----------

def aggregate_ticker(ticker: str, all_posts: list[dict]) -> pd.DataFrame:
    """Filter posts via ticker regex; aggregate per (ticker, date_utc)."""
    pat = build_ticker_regex(ticker)
    by_date: dict[str, dict] = {}
    polarities: dict[str, list] = defaultdict(list)
    sub_counts: dict[str, Counter] = defaultdict(Counter)

    seen_ids = set()
    for p in all_posts:
        pid = p.get("id")
        if pid in seen_ids:
            continue
        if pid:
            seen_ids.add(pid)
        title = p.get("title") or ""
        body = p.get("selftext") or ""
        combined_for_match = f"{title} {body}"
        if not ticker_matches(combined_for_match, pat):
            continue
        ts = p.get("created_utc")
        if not ts:
            continue
        try:
            d = dt.datetime.utcfromtimestamp(int(ts)).strftime("%Y-%m-%d")
        except Exception:
            continue
        score = int(p.get("score") or 0)
        ncom = int(p.get("num_comments") or 0)
        pol = polarity(f"{title} {body}")
        sub = p.get("subreddit") or ""
        row = by_date.setdefault(d, {
            "ticker": ticker,
            "date": d,
            "mention_count": 0,
            "upvote_sum": 0,
            "comment_count": 0,
        })
        row["mention_count"] += 1
        row["upvote_sum"] += score
        row["comment_count"] += ncom
        polarities[d].append(pol)
        sub_counts[d][sub] += 1

    rows = []
    for d, row in by_date.items():
        pols = polarities[d]
        if pols:
            mean_p = sum(pols) / len(pols)
            if len(pols) > 1:
                var = sum((x - mean_p) ** 2 for x in pols) / (len(pols) - 1)
                std_p = math.sqrt(var)
            else:
                std_p = 0.0
        else:
            mean_p = 0.0
            std_p = 0.0
        top_sub = sub_counts[d].most_common(1)[0][0] if sub_counts[d] else ""
        row["sentiment_polarity"] = mean_p
        row["sentiment_volatility"] = std_p
        row["top_subreddit"] = top_sub
        row["available_from"] = d
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


# ---------- driver ----------

def load_universe() -> list[str]:
    df = pd.read_parquet(UNIVERSE_PARQUET)
    out = []
    for t in df["ticker"].tolist():
        ts = str(t).upper().strip()
        if ts:
            out.append(ts)
    # Stable de-dup
    seen, ordered = set(), []
    for t in out:
        if t not in seen:
            ordered.append(t)
            seen.add(t)
    return ordered


def process_ticker(ticker: str) -> pd.DataFrame:
    """Pull posts across all subreddits + filter + aggregate."""
    all_posts: list[dict] = []
    for sub in SUBREDDITS:
        posts = fetch_ticker_subreddit_posts(ticker, sub)
        all_posts.extend(posts)
    if not all_posts:
        return pd.DataFrame()
    return aggregate_ticker(ticker, all_posts)


def main():
    t0 = time.time()
    print(f"[{now_hms()}] reddit hype ingest START", flush=True)
    print(f"  universe: {UNIVERSE_PARQUET}", flush=True)
    print(f"  output:   {OUT_DIR}", flush=True)
    print(f"  subreddits: {SUBREDDITS}", flush=True)
    print(f"  date range: {START_DATE} -> today", flush=True)
    print(f"  VADER available: {USE_VADER}", flush=True)

    universe = load_universe()
    print(f"  loaded {len(universe)} tickers", flush=True)

    n_resumed = n_new = n_empty = n_failed = 0
    n_ambig_skipped = 0
    for i, ticker in enumerate(universe, 1):
        out_path = OUT_DIR / f"{ticker}.parquet"
        if out_path.exists():
            n_resumed += 1
            if i % 10 == 0:
                rate = i / max(time.time() - t0, 1e-6)
                eta = (len(universe) - i) / max(rate, 1e-6)
                print(f"[{now_hms()}] {i}/{len(universe)} (cached) rate={rate:.3f} t/s eta={eta/60:.1f}min", flush=True)
            continue

        if is_ambiguous_ticker(ticker) and len(ticker) <= 1:
            # Single-letter ticker — too ambiguous, skip entirely (no $ prefix
            # will reliably appear). Write empty marker.
            print(f"  {ticker}: single-letter ambiguous — skip with empty marker", flush=True)
            pd.DataFrame([{
                "ticker": ticker, "date": None, "mention_count": 0,
                "upvote_sum": 0, "comment_count": 0,
                "sentiment_polarity": 0.0, "sentiment_volatility": 0.0,
                "top_subreddit": "", "available_from": None, "_empty": True,
            }]).to_parquet(out_path, index=False)
            n_ambig_skipped += 1
            continue

        try:
            df = process_ticker(ticker)
        except Exception as e:
            print(f"  {ticker}: FAILED {e}", flush=True)
            traceback.print_exc()
            n_failed += 1
            continue

        if df.empty:
            pd.DataFrame([{
                "ticker": ticker, "date": None, "mention_count": 0,
                "upvote_sum": 0, "comment_count": 0,
                "sentiment_polarity": 0.0, "sentiment_volatility": 0.0,
                "top_subreddit": "", "available_from": None, "_empty": True,
            }]).to_parquet(out_path, index=False)
            n_empty += 1
            print(f"  {ticker}: 0 mentions", flush=True)
        else:
            df.to_parquet(out_path, index=False)
            n_new += 1
            print(f"  {ticker}: {len(df)} ticker-days, sum_mentions={int(df['mention_count'].sum())}", flush=True)

        if i % 10 == 0 or i == len(universe):
            rate = i / max(time.time() - t0, 1e-6)
            eta = (len(universe) - i) / max(rate, 1e-6)
            print(f"[{now_hms()}] {i}/{len(universe)} new={n_new} empty={n_empty} ambig={n_ambig_skipped} fail={n_failed} resumed={n_resumed} rate={rate:.3f} t/s eta={eta/60:.1f}min", flush=True)

    # Unified parquet
    print(f"[{now_hms()}] writing unified _all.parquet ...", flush=True)
    parts = []
    for p in sorted(OUT_DIR.glob("*.parquet")):
        if p.name == "_all.parquet":
            continue
        try:
            d = pd.read_parquet(p)
            if "_empty" in d.columns and d["_empty"].all():
                continue
            # Drop _empty column if it leaked in
            if "_empty" in d.columns:
                d = d.drop(columns=["_empty"])
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

    el = time.time() - t0
    print(f"[{now_hms()}] DONE in {el/60:.1f}min  new={n_new} empty={n_empty} ambig={n_ambig_skipped} fail={n_failed} resumed={n_resumed}", flush=True)


if __name__ == "__main__":
    main()
