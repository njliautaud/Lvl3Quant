"""
Family: google_trends  (HC #563 R2, HC #564 R6(d))

Real production ingest. Pulls Google Trends search interest per ticker via
pytrends. PIT-safe daily resolution, ticker-by-ticker incremental parquet,
resume-on-restart, polite long delay to avoid 429 IP blocks.

OUTPUT: data/feature_store/google_trends/{ticker}.parquet  (per-ticker)
        data/feature_store/google_trends/_all.parquet      (unified, end-of-run)

Schema per row: ticker, date, search_interest (0-100), search_interest_z (252d rolling),
                related_breakout_count.

Cost ladder per HC #556 R2 / HC #563 R2: pytrends is FREE; unofficial; subject to
Google rate-limit. Conservative: 60s between primary queries; 30 min back-off on 429.
"""
from __future__ import annotations
import sys, time, json, os, traceback
from pathlib import Path
import pandas as pd
import numpy as np
sys.path.insert(0, "/home/jupiter/Lvl3Quant/research/ingest")
from _common import write_parquet, smoke_log, STORE

FAMILY = "google_trends"
OUT_DIR = STORE / FAMILY
OUT_DIR.mkdir(parents=True, exist_ok=True)

UNIVERSE_PARQUET = Path("/home/jupiter/Lvl3Quant/wheel_strategy_v1/data/cache/universe_v2.parquet")

# Conservative pacing — Google Trends will IP-block aggressively
SLEEP_BETWEEN_QUERIES = 60.0   # seconds, baseline
BACKOFF_429 = 1800.0           # 30 min cooldown on a 429
MAX_RETRIES = 3
TIMEFRAME = "today 5-y"        # 5 years of daily resolution
GEO = "US"


def load_universe() -> list[str]:
    df = pd.read_parquet(UNIVERSE_PARQUET)
    # universe_v2 has a 'ticker' column (lowercase) per build_universe_v2.py
    col = "ticker" if "ticker" in df.columns else df.columns[0]
    tickers = sorted({str(t).strip().upper() for t in df[col].dropna()})
    return tickers


def already_done(tk: str) -> bool:
    return (OUT_DIR / f"{tk}.parquet").exists() or (OUT_DIR / f"{tk}.empty").exists()


def write_ticker(tk: str, df: pd.DataFrame):
    df.to_parquet(OUT_DIR / f"{tk}.parquet", index=False)


def write_empty_marker(tk: str, reason: str):
    (OUT_DIR / f"{tk}.empty").write_text(f"{reason}\n")


def fetch_one(pt, tk: str) -> pd.DataFrame | None:
    """Query Trends for 'TICKER stock' (disambiguates short tickers). Returns
    long-form per-day frame or None on hard fail."""
    kw = f"{tk} stock"
    for attempt in range(MAX_RETRIES):
        try:
            pt.build_payload([kw], timeframe=TIMEFRAME, geo=GEO)
            iot = pt.interest_over_time()
            if iot is None or iot.empty:
                return None
            iot = iot.reset_index()
            if "isPartial" in iot.columns:
                iot = iot[~iot["isPartial"]].drop(columns=["isPartial"])
            iot = iot.rename(columns={kw: "search_interest", "date": "date"})
            iot["ticker"] = tk
            iot["date"] = pd.to_datetime(iot["date"])
            iot["search_interest"] = iot["search_interest"].astype(float)
            # rolling 252d z
            iot = iot.sort_values("date").reset_index(drop=True)
            rmean = iot["search_interest"].rolling(252, min_periods=60).mean()
            rstd = iot["search_interest"].rolling(252, min_periods=60).std()
            iot["search_interest_z"] = (iot["search_interest"] - rmean) / rstd.replace(0, np.nan)
            # related queries — count of "breakout" matches as quick spike proxy
            br_count = 0
            try:
                rq = pt.related_queries()
                rising = rq.get(kw, {}).get("rising")
                if rising is not None and not rising.empty:
                    br_count = int((rising["value"] == "Breakout").sum())
            except Exception:
                br_count = 0
            iot["related_breakout_count"] = br_count
            return iot[["ticker", "date", "search_interest", "search_interest_z", "related_breakout_count"]]
        except Exception as e:
            msg = str(e).lower()
            if "429" in msg or "too many" in msg or "rate" in msg:
                print(f"[{tk}] rate-limit hit (attempt {attempt+1}/{MAX_RETRIES}): {e!r} — sleeping {BACKOFF_429}s", flush=True)
                time.sleep(BACKOFF_429)
                continue
            print(f"[{tk}] error (attempt {attempt+1}/{MAX_RETRIES}): {e!r}", flush=True)
            time.sleep(30)
            continue
    return None


def main():
    from pytrends.request import TrendReq
    pt = TrendReq(hl="en-US", tz=360, timeout=(10, 30), retries=2, backoff_factor=0.5)
    tickers = load_universe()
    total = len(tickers)
    print(f"[start] universe={total} tickers  timeframe={TIMEFRAME}  pacing={SLEEP_BETWEEN_QUERIES}s/query", flush=True)

    done = new = skipped = failed = 0
    t0 = time.time()
    for i, tk in enumerate(tickers, 1):
        if already_done(tk):
            done += 1; skipped += 1
            continue
        try:
            df = fetch_one(pt, tk)
            if df is None or df.empty:
                write_empty_marker(tk, "no data")
                failed += 1
            else:
                write_ticker(tk, df)
                new += 1
                done += 1
            if i % 5 == 0:
                rate = (new + skipped) / max(time.time() - t0, 1) * 60
                eta_min = (total - i) / max(rate, 0.01)
                print(f"[{i}/{total}] done={done} new={new} skip={skipped} fail={failed}  rate={rate:.2f} t/min  eta={eta_min:.0f}min", flush=True)
        except Exception:
            traceback.print_exc()
            failed += 1
        time.sleep(SLEEP_BETWEEN_QUERIES)

    # unify
    parts = []
    for p in sorted(OUT_DIR.glob("*.parquet")):
        if p.name == "_all.parquet":
            continue
        try:
            parts.append(pd.read_parquet(p))
        except Exception:
            pass
    if parts:
        all_df = pd.concat(parts, ignore_index=True).sort_values(["ticker", "date"])
        all_df.to_parquet(OUT_DIR / "_all.parquet", index=False)
        print(f"[done] wrote _all.parquet rows={len(all_df)} tickers={all_df['ticker'].nunique()}", flush=True)
        smoke_log(FAMILY, True, f"full run: {len(all_df)} rows / {all_df['ticker'].nunique()} tickers")
    else:
        print("[done] no per-ticker frames to unify", flush=True)
        smoke_log(FAMILY, False, "no data after full run")


if __name__ == "__main__":
    main()
