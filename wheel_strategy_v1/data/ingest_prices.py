"""
ingest_prices.py — Daily OHLCV 2015-present + per-name features.

For each ticker in universe.parquet:
  - daily OHLCV via yfinance
  - realized vol (20/60/252d, annualized)
  - max drawdown over full window
  - ATR% (14d)
  - gap stats (mean abs overnight gap, max gap)
  - last earnings move (placeholder — use price-vs-7d gap proxy if no earnings date)

Output:
  data/cache/prices.parquet   (long format: ticker, date, open, high, low, close, volume, ret, log_ret, rv_20, rv_60, rv_252)
  data/cache/price_features.parquet (per-ticker static features used by GA scoring)
"""
from __future__ import annotations
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)


def _compute_features(px: pd.DataFrame) -> dict:
    """Per-name static feature snapshot."""
    if px.empty or len(px) < 20:
        return {}
    close = px["close"].astype(float)
    ret = close.pct_change()
    log_ret = np.log(close / close.shift(1))

    # realized vol annualized
    rv = lambda n: float(log_ret.rolling(n).std().iloc[-1] * np.sqrt(252)) if len(log_ret) >= n else float("nan")
    # max DD
    cum = (1 + ret.fillna(0)).cumprod()
    peak = cum.cummax()
    dd = (cum / peak) - 1
    max_dd = float(dd.min())
    # ATR% 14d
    tr = pd.concat([
        (px["high"] - px["low"]).abs(),
        (px["high"] - px["close"].shift()).abs(),
        (px["low"] - px["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean().iloc[-1] if len(tr) >= 14 else float("nan")
    atr_pct = float(atr14 / close.iloc[-1]) if not np.isnan(atr14) and close.iloc[-1] != 0 else float("nan")
    # gap stats: abs(open / prev_close - 1)
    gap = (px["open"] / close.shift() - 1).abs()
    gap_mean = float(gap.mean()) if not gap.empty else float("nan")
    gap_max = float(gap.max()) if not gap.empty else float("nan")
    # 1y / 3y / 5y total return
    def tr_pct(days):
        if len(close) < days + 1:
            return float("nan")
        return float(close.iloc[-1] / close.iloc[-1 - days] - 1)

    return {
        "rv_20": rv(20),
        "rv_60": rv(60),
        "rv_252": rv(252),
        "max_dd": max_dd,
        "atr_pct": atr_pct,
        "gap_mean": gap_mean,
        "gap_max": gap_max,
        "tr_1y": tr_pct(252),
        "tr_3y": tr_pct(252 * 3),
        "tr_5y": tr_pct(252 * 5),
        "last_close": float(close.iloc[-1]),
        "n_days": int(len(close)),
    }


def fetch_one(ticker: str, start: str, end: str, retries: int = 3) -> pd.DataFrame:
    import yfinance as yf
    for attempt in range(retries):
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True, threads=False)
            if df is None or df.empty:
                return pd.DataFrame()
            df = df.reset_index()
            # Flatten possible MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            df.columns = [str(c).lower() for c in df.columns]
            ren = {"date": "date", "open": "open", "high": "high", "low": "low",
                   "close": "close", "adj close": "close", "volume": "volume"}
            df = df.rename(columns=ren)
            keep = [c for c in ["date","open","high","low","close","volume"] if c in df.columns]
            df = df[keep]
            df["ticker"] = ticker
            return df
        except Exception as e:
            wait = 2 ** attempt
            print(f"[prices] {ticker} attempt {attempt+1} err: {e}; sleep {wait}s", flush=True)
            time.sleep(wait)
    return pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-names", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=0.15)
    args = ap.parse_args()

    if args.end is None:
        args.end = pd.Timestamp.today().strftime("%Y-%m-%d")
    if args.smoke:
        args.start = "2024-01-01"
        args.end = "2024-12-31"

    uni_path = CACHE / "universe.parquet"
    if not uni_path.exists():
        print(f"[prices] missing {uni_path}; run ingest_universe.py first", file=sys.stderr)
        sys.exit(2)
    uni = pd.read_parquet(uni_path)
    tickers = uni["ticker"].tolist()
    if args.max_names:
        tickers = tickers[: args.max_names]

    print(f"[prices] fetching {len(tickers)} tickers {args.start}..{args.end}", flush=True)
    all_px = []
    feats = []
    failed = []
    for i, t in enumerate(tickers):
        df = fetch_one(t, args.start, args.end)
        if df.empty:
            failed.append(t)
        else:
            df.rename(columns={"date":"date"}, inplace=True)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            df["ret"] = df["close"].pct_change()
            df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
            for w in (20, 60, 252):
                df[f"rv_{w}"] = df["log_ret"].rolling(w).std() * np.sqrt(252)
            all_px.append(df[["ticker","date","open","high","low","close","volume","ret","log_ret","rv_20","rv_60","rv_252"]])
            f = _compute_features(df)
            f["ticker"] = t
            feats.append(f)
        if i % 25 == 0:
            print(f"[prices] {i+1}/{len(tickers)} ok={len(all_px)} fail={len(failed)}", flush=True)
        time.sleep(args.sleep)

    if not all_px:
        print("[prices] no data fetched; bailing", file=sys.stderr)
        sys.exit(3)

    px = pd.concat(all_px, ignore_index=True)
    out_px = CACHE / ("prices_smoke.parquet" if args.smoke else "prices.parquet")
    px.to_parquet(out_px, index=False)
    feats_df = pd.DataFrame(feats)
    out_f = CACHE / ("price_features_smoke.parquet" if args.smoke else "price_features.parquet")
    feats_df.to_parquet(out_f, index=False)
    print(f"[prices] wrote {len(px)} rows -> {out_px}")
    print(f"[prices] wrote {len(feats_df)} feature rows -> {out_f}")
    if failed:
        print(f"[prices] failed: {len(failed)} (first 10: {failed[:10]})")


if __name__ == "__main__":
    main()
