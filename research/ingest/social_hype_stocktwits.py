"""
Family: social_hype (StockTwits slice)  (HC #563 R2 — social hype)

WHAT: StockTwits per-symbol message stream — bullish/bearish sentiment ratio,
message volume, follower-weighted sentiment.

SOURCE: StockTwits public API (free tier, no auth for symbol streams).
GET https://api.stocktwits.com/api/2/streams/symbol/{TICKER}.json

WHY: HC #563 R2 — social hype family. StockTwits is the canonical retail
sentiment source (free) — user explicitly cited.

OUTPUT:
  data/feature_store/social_stocktwits/messages.parquet
    Raw per-message rows for the most recent stream (the API gives ~30
    most-recent messages per call; this script captures whatever is current).
  data/feature_store/social_stocktwits/daily.parquet
    Daily roll-up: (ticker, date, n_msgs, pct_bullish, pct_bearish, n_users,
    mean_followers, watcher_count)

NOTE: StockTwits free tier returns only the most-recent ~30 messages per stream,
so a single run produces ~last-day data per ticker. Build history by running
this script on cron (every few hours). Rate-limit: ~200 req/hr.
"""
from __future__ import annotations
import sys, time
import datetime as dt
import requests
import pandas as pd
sys.path.insert(0, "/home/jupiter/Lvl3Quant/research/ingest")
from _common import write_parquet, smoke_log, SMOKE_UNIVERSE  # type: ignore

FAMILY = "social_stocktwits"
UNIVERSE = SMOKE_UNIVERSE  # 10 megacaps; extend later.
PAUSE_SEC = 3.5  # ~200 req/hr cap → 18 sec/req max; 3.5s is conservative for ~10 tickers.


def fetch_stream(ticker: str):
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
    try:
        r = requests.get(
            url, timeout=15,
            headers={"User-Agent": "Lvl3Quant-Research/1.0 qa@example.com"},
        )
    except Exception as e:
        return None, repr(e)
    if r.status_code != 200:
        return None, f"HTTP {r.status_code}"
    try:
        return r.json(), None
    except Exception as e:
        return None, repr(e)


def parse_messages(ticker: str, payload: dict) -> tuple[list[dict], dict | None]:
    """Return (per-message rows, symbol_meta)."""
    rows = []
    sym_meta = (payload.get("symbol") or {})
    for m in payload.get("messages", []) or []:
        ent = (m.get("entities") or {}).get("sentiment") or {}
        usr = m.get("user") or {}
        rows.append({
            "ticker":         ticker,
            "message_id":     m.get("id"),
            "created_at":     m.get("created_at"),
            "sentiment":      ent.get("basic"),  # "Bullish" / "Bearish" / None
            "user_id":        usr.get("id"),
            "user_followers": usr.get("followers"),
            "user_likes":     (m.get("likes") or {}).get("total"),
            "body":           (m.get("body") or "")[:500],
        })
    return rows, sym_meta


def to_daily(messages: pd.DataFrame, watcher_counts: dict) -> pd.DataFrame:
    if messages.empty:
        return messages
    df = messages.copy()
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df["date"] = df["created_at"].dt.tz_convert("US/Eastern").dt.date
    df["is_bull"] = (df["sentiment"] == "Bullish").astype(int)
    df["is_bear"] = (df["sentiment"] == "Bearish").astype(int)
    grp = df.groupby(["ticker", "date"], as_index=False).agg(
        n_msgs=("message_id", "count"),
        n_bull=("is_bull", "sum"),
        n_bear=("is_bear", "sum"),
        n_users=("user_id", "nunique"),
        mean_followers=("user_followers", "mean"),
    )
    grp["pct_bullish"] = grp["n_bull"] / grp["n_msgs"]
    grp["pct_bearish"] = grp["n_bear"] / grp["n_msgs"]
    grp["watcher_count"] = grp["ticker"].map(watcher_counts)
    grp = grp[["ticker","date","n_msgs","pct_bullish","pct_bearish",
               "n_users","mean_followers","watcher_count"]]
    return grp


def main():
    msg_rows = []
    watcher_counts: dict = {}
    successes, failures = 0, 0
    for tk in UNIVERSE:
        payload, err = fetch_stream(tk)
        if payload is None:
            print(f"[{FAMILY}] {tk} FAIL {err}")
            failures += 1
            time.sleep(PAUSE_SEC)
            continue
        rows, meta = parse_messages(tk, payload)
        msg_rows.extend(rows)
        wc = meta.get("watchlist_count") if meta else None
        if wc is not None:
            watcher_counts[tk] = wc
        successes += 1
        print(f"[{FAMILY}] {tk} ok n_msgs={len(rows)} watchers={wc}")
        time.sleep(PAUSE_SEC)

    if not msg_rows:
        smoke_log(FAMILY, False, "no messages returned")
        raise RuntimeError("StockTwits returned 0 messages across universe")

    df_msgs = pd.DataFrame(msg_rows)
    p1 = write_parquet(df_msgs, FAMILY, "messages.parquet")
    print(f"OK {FAMILY} messages: {len(df_msgs)} -> {p1}")

    daily = to_daily(df_msgs, watcher_counts)
    p2 = write_parquet(daily, FAMILY, "daily.parquet")
    print(f"OK {FAMILY} daily: {len(daily)} rows ({successes} tickers ok, {failures} failed)")
    smoke_log(FAMILY, True, f"msgs={len(df_msgs)} daily={len(daily)} ok={successes} fail={failures}")
    return p1, p2


def run_full():
    return main()


if __name__ == "__main__":
    main()
