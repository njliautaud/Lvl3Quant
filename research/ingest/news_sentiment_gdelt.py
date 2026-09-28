"""
Family: news_sentiment  (HC #563 R2 — news sentiment via GDELT)

WHAT: GDELT 2.0 DOC API — global news article counts, tone, polarity per ticker
keyword (company name OR ticker symbol). Daily rollup: # articles, mean tone,
% positive, % negative.

SOURCE: GDELT 2.0 DOC API (free, no key).
https://api.gdeltproject.org/api/v2/doc/doc?query=...&mode=TimelineTone&format=json

WHY: HC #563 R2 — news_sentiment family is mandatory. GDELT covers global English
news with consistent tone scoring.

OUTPUT: data/feature_store/news_gdelt/daily.parquet
Schema: (ticker, date, n_articles, mean_tone, pct_pos_7d, pct_neg_7d)

PIT-SAFE: GDELT publishes near-real-time. Consumers must filter to dates
strictly less than the feature date (D−1) to avoid look-ahead.

USAGE:
  python3 research/ingest/news_sentiment_gdelt.py               # smoke (first 5 tickers, default)
  python3 research/ingest/news_sentiment_gdelt.py --full        # full universe -> daily.parquet
  python3 research/ingest/news_sentiment_gdelt.py --tickers AAPL MSFT
"""
from __future__ import annotations
import sys, time, json, argparse, urllib.parse
from pathlib import Path
import requests
import pandas as pd

sys.path.insert(0, "/home/jupiter/Lvl3Quant/research/ingest")
from _common import write_parquet, smoke_log, SMOKE_UNIVERSE, STORE

FAMILY = "news_gdelt"
GDELT_BASE = "https://api.gdeltproject.org/api/v2/doc/doc"
HEADERS = {"User-Agent": "Lvl3Quant-Research/0.2 qa@example.com"}
POLITE_SLEEP = 6.0  # GDELT requires >=5s between requests (their error msg)
MAX_RETRIES = 6
TIMEOUT = 30
INITIAL_BACKOFF = 6.0  # start backoff above the 5s GDELT requirement

# Extended universe used for the --full run. ~30 liquid mega/large caps that
# cover wheel-strategy candidates and key sector representatives.
FULL_UNIVERSE = [
    # Mega caps / mag-7-ish
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS",
    # Healthcare
    "UNH", "JNJ", "PFE", "LLY", "ABBV",
    # Energy
    "XOM", "CVX", "COP",
    # Consumer / retail
    "WMT", "HD", "COST", "MCD", "NKE",
    # Industrials / semi / tech-adjacent
    "BA", "CAT", "AMD", "INTC", "CRM",
    # Comms / media
    "DIS", "NFLX",
]


def _request_with_retry(url: str) -> dict | None:
    """GET with exponential backoff on 429/5xx. Returns parsed JSON or None."""
    delay = INITIAL_BACKOFF
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, timeout=TIMEOUT, headers=HEADERS)
        except requests.RequestException as e:
            print(f"  ! network error {e!r}; retry {attempt+1}/{MAX_RETRIES}")
            time.sleep(delay)
            delay *= 2
            continue
        if r.status_code == 200:
            # GDELT sometimes returns empty body or HTML on degraded responses.
            text = r.text.strip()
            if not text:
                return {}
            try:
                return r.json()
            except json.JSONDecodeError:
                # Not JSON (HTML error page). Treat as empty.
                return {}
        if r.status_code in (429, 500, 502, 503, 504):
            print(f"  ! HTTP {r.status_code}; backoff {delay:.1f}s "
                  f"(attempt {attempt+1}/{MAX_RETRIES})")
            time.sleep(delay)
            delay *= 2
            continue
        # Other codes: hard fail for this URL.
        print(f"  ! HTTP {r.status_code} (non-retryable) for {url}")
        return None
    return None


def _build_url(ticker: str, mode: str, timespan: str = "12m") -> str:
    # Query the literal ticker as a phrase combined with the word "stock" so
    # we filter out unrelated namespaces. GDELT expects URL-encoded query.
    raw_q = f'"{ticker}" stock'
    q = urllib.parse.quote(raw_q, safe="")
    return f"{GDELT_BASE}?query={q}&mode={mode}&timespan={timespan}&format=json"


def _parse_timeline(payload: dict, value_key: str = "value") -> pd.DataFrame:
    """Return DataFrame[date, value] from a GDELT timeline payload."""
    if not payload:
        return pd.DataFrame(columns=["date", value_key])
    tl = payload.get("timeline", [])
    rows = []
    for series in tl:
        for pt in series.get("data", []):
            d = pt.get("date")
            v = pt.get("value")
            if d is None:
                continue
            rows.append({"date": d, value_key: v})
    if not rows:
        return pd.DataFrame(columns=["date", value_key])
    df = pd.DataFrame(rows)
    # GDELT 'date' is YYYYMMDDTHHMMSSZ or YYYYMMDD; normalise to date.
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.date
    df = df.dropna(subset=["date"]).groupby("date", as_index=False).agg({value_key: "mean"})
    return df


def fetch_ticker(ticker: str) -> pd.DataFrame:
    """Fetch 12-month TimelineTone + TimelineVolRaw for one ticker.

    Returns DataFrame with columns: ticker, date, mean_tone, n_articles,
    pct_pos_7d, pct_neg_7d.  Empty DataFrame if GDELT has no data.
    """
    tone_url = _build_url(ticker, "TimelineTone")
    vol_url = _build_url(ticker, "TimelineVolRaw")

    print(f"[{ticker}] TimelineTone...")
    tone_payload = _request_with_retry(tone_url)
    time.sleep(POLITE_SLEEP)
    print(f"[{ticker}] TimelineVolRaw...")
    vol_payload = _request_with_retry(vol_url)
    time.sleep(POLITE_SLEEP)

    tone_df = _parse_timeline(tone_payload or {}, value_key="mean_tone")
    vol_df = _parse_timeline(vol_payload or {}, value_key="n_articles")

    if tone_df.empty and vol_df.empty:
        return pd.DataFrame()

    if tone_df.empty:
        merged = vol_df.copy()
        merged["mean_tone"] = float("nan")
    elif vol_df.empty:
        merged = tone_df.copy()
        merged["n_articles"] = float("nan")
    else:
        merged = pd.merge(tone_df, vol_df, on="date", how="outer")

    merged = merged.sort_values("date").reset_index(drop=True)
    merged["ticker"] = ticker

    # Rolling 7-day positive/negative day fractions.
    # GDELT tone scale: roughly [-100, +100], with most values in [-10, +10].
    # We classify a day as positive if mean_tone > 1, negative if < -1.
    pos_day = (merged["mean_tone"] > 1).astype(float)
    neg_day = (merged["mean_tone"] < -1).astype(float)
    merged["pct_pos_7d"] = pos_day.rolling(7, min_periods=1).mean()
    merged["pct_neg_7d"] = neg_day.rolling(7, min_periods=1).mean()
    # When tone is NaN for a day we shouldn't claim a positive/negative class.
    mask_nan = merged["mean_tone"].isna()
    merged.loc[mask_nan, ["pct_pos_7d", "pct_neg_7d"]] = float("nan")

    return merged[
        ["ticker", "date", "mean_tone", "n_articles", "pct_pos_7d", "pct_neg_7d"]
    ]


def run(tickers: list[str], out_name: str) -> Path:
    all_frames: list[pd.DataFrame] = []
    skipped: list[str] = []
    for i, tk in enumerate(tickers, 1):
        try:
            df = fetch_ticker(tk)
        except Exception as e:
            print(f"[{tk}] ERROR {e!r}; skipping")
            skipped.append(tk)
            continue
        if df.empty:
            print(f"[{tk}] empty timeline; skipped")
            skipped.append(tk)
            continue
        all_frames.append(df)
        print(f"[{tk}] {len(df)} rows  ({i}/{len(tickers)})")

    if not all_frames:
        raise RuntimeError(
            f"No data retrieved for any ticker. Skipped={skipped}"
        )
    out = pd.concat(all_frames, ignore_index=True)
    # Final stable sort.
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    path = write_parquet(out, FAMILY, out_name)
    print(f"WROTE {path}  ({len(out)} rows, {out['ticker'].nunique()} tickers)")
    if skipped:
        print(f"SKIPPED ({len(skipped)}): {skipped}")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="Run full universe and write daily.parquet")
    ap.add_argument("--tickers", nargs="+", default=None,
                    help="Explicit ticker list (overrides smoke/full)")
    ap.add_argument("--smoke-n", type=int, default=5,
                    help="How many SMOKE_UNIVERSE tickers to use in smoke mode")
    args = ap.parse_args()

    if args.tickers:
        tickers = args.tickers
        out_name = "custom.parquet"
        mode_label = f"custom({len(tickers)})"
    elif args.full:
        tickers = FULL_UNIVERSE
        out_name = "daily.parquet"
        mode_label = f"full({len(tickers)})"
    else:
        tickers = SMOKE_UNIVERSE[: args.smoke_n]
        out_name = "smoke.parquet"
        mode_label = f"smoke({len(tickers)})"

    print(f"GDELT ingest start — mode={mode_label}, out={out_name}")
    try:
        path = run(tickers, out_name)
        smoke_log(FAMILY, True, f"{mode_label} -> {path}")
        print(f"OK {FAMILY}: {mode_label} -> {path}")
    except Exception as e:
        smoke_log(FAMILY, False, repr(e))
        print(f"FAIL {FAMILY}: {e}")
        raise


if __name__ == "__main__":
    main()
