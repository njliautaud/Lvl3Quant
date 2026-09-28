"""
Targeted refresh of prices_v2.parquet for the megacap-tech paper engine.

Updates only AAPL, MSFT, GOOGL, NVDA, META, AMZN, AVGO, TSLA + SPY (plus VIX
proxy ^VIX written separately into the cache for the regime gate).

Appends new dates onto the existing prices_v2.parquet without re-downloading
the full universe. Source: yfinance (free, no API limits). Idempotent.

HC #420 — user's own quant trading codebase, refresh utility for paper engine.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("yfinance not installed — pip install yfinance", flush=True)
    sys.exit(1)

ROOT = Path("/home/jupiter/Lvl3Quant")
CACHE = ROOT / "wheel_strategy_v1/data/cache"
PRICES_PATH = CACHE / "prices_v2.parquet"
VIX_PATH = CACHE / "vix_history.parquet"

TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMZN", "AVGO", "TSLA", "SPY"]


def update_one(ticker: str, start: pd.Timestamp) -> pd.DataFrame:
    """Pull yfinance OHLCV from start through today, return shaped df."""
    end = pd.Timestamp.today() + pd.Timedelta(days=1)
    df = yf.download(ticker, start=start.date().isoformat(),
                     end=end.date().isoformat(),
                     progress=False, auto_adjust=False, threads=False)
    if df is None or df.empty:
        print(f"  {ticker}: no data returned", flush=True)
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.reset_index()
    df.columns = [str(c).lower() for c in df.columns]
    df["ticker"] = ticker
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    keep = ["ticker", "date", "open", "high", "low", "close", "volume"]
    df = df[[c for c in keep if c in df.columns]].copy()
    return df


def main():
    if not PRICES_PATH.exists():
        print(f"FATAL: {PRICES_PATH} missing — run full build first", flush=True)
        sys.exit(1)

    existing = pd.read_parquet(PRICES_PATH)
    existing["date"] = pd.to_datetime(existing["date"])
    last_date = existing[existing["ticker"].isin(TICKERS)]["date"].max()
    print(f"existing cache last date for megacap+SPY: {last_date.date()}", flush=True)

    start = (last_date + pd.Timedelta(days=1)).normalize()
    today = pd.Timestamp.today().normalize()
    if start >= today:
        print("cache already current — nothing to do", flush=True)
        return

    print(f"refreshing {TICKERS} from {start.date()} -> {today.date()}", flush=True)
    new_rows = []
    for t in TICKERS:
        df = update_one(t, start)
        if df.empty:
            continue
        # only keep dates strictly after last cached for this ticker
        last_t = existing[existing["ticker"] == t]["date"].max()
        df = df[df["date"] > last_t]
        if df.empty:
            print(f"  {t}: nothing new", flush=True)
            continue
        print(f"  {t}: +{len(df)} rows ({df['date'].min().date()} -> {df['date'].max().date()})", flush=True)
        new_rows.append(df)

    if not new_rows:
        print("no new rows across any ticker — done", flush=True)
        return

    new_df = pd.concat(new_rows, ignore_index=True)
    # Compute the derived columns the megacap engine expects: ret, log_ret, rv_20/60/252
    # We need per-ticker continuity, so compute across (existing+new) and overwrite tail.
    merged_rows = []
    for t in new_df["ticker"].unique():
        old_t = existing[existing["ticker"] == t].copy()
        new_t = new_df[new_df["ticker"] == t].copy()
        full = pd.concat([old_t, new_t], ignore_index=True).sort_values("date")
        full["ret"] = full["close"].pct_change()
        full["log_ret"] = np.log1p(full["ret"].fillna(0)).replace(0, np.nan)
        full["rv_20"] = full["log_ret"].rolling(20).std() * np.sqrt(252)
        full["rv_60"] = full["log_ret"].rolling(60).std() * np.sqrt(252)
        full["rv_252"] = full["log_ret"].rolling(252).std() * np.sqrt(252)
        merged_rows.append(full)

    # Tickers we didn't touch keep their old rows
    untouched = existing[~existing["ticker"].isin(new_df["ticker"].unique())].copy()
    final = pd.concat([untouched] + merged_rows, ignore_index=True)
    final = final.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Backup then write
    backup = PRICES_PATH.with_suffix(".parquet.bak")
    PRICES_PATH.replace(backup)
    final.to_parquet(PRICES_PATH, index=False)
    print(f"wrote {PRICES_PATH} ({final.shape}); backup at {backup}", flush=True)

    # Also refresh VIX
    print("refreshing VIX...", flush=True)
    vix = yf.download("^VIX", start="2018-01-01",
                      end=(today + pd.Timedelta(days=1)).date().isoformat(),
                      progress=False, auto_adjust=False, threads=False)
    if vix is not None and not vix.empty:
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = [c[0] if isinstance(c, tuple) else c for c in vix.columns]
        vix = vix.reset_index()
        vix.columns = [str(c).lower() for c in vix.columns]
        vix["date"] = pd.to_datetime(vix["date"]).dt.normalize()
        vix[["date", "close"]].to_parquet(VIX_PATH, index=False)
        print(f"wrote {VIX_PATH} ({len(vix)} rows, last={vix['date'].max().date()}, last_close={vix['close'].iloc[-1]:.2f})", flush=True)


if __name__ == "__main__":
    main()
