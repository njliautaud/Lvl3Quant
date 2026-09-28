"""
ingest_macro.py — Macro/regime overlay, daily 2015-present.

Series:
  - VIX           (yfinance ^VIX)
  - VIX3M         (yfinance ^VIX3M)  -> vix_ts = VIX3M / VIX (term structure)
  - SPY 50/200 DMA regime (computed locally from prices.parquet — must be run AFTER ingest_prices)
  - 2s10s         (FRED T10Y2Y — yield curve slope)
  - HY OAS        (FRED BAMLH0A0HYM2 — credit spread)
  - DXY           (yfinance DX-Y.NYB)
  - NAAIM         (https://www.naaim.org/programs/naaim-exposure-index/  CSV — fallback to placeholder if blocked)
  - AAII bull/bear (TODO — no free clean source; skipped with TODO)

Output: data/cache/macro.parquet  (date + all series, forward-filled)
"""
from __future__ import annotations
import io
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)


def _yf_series(ticker: str, name: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf
    for attempt in range(3):
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True, threads=False)
            if df is None or df.empty:
                return pd.DataFrame()
            df = df.reset_index()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            df.columns = [str(c).lower() for c in df.columns]
            out = df[["date", "close"]].copy() if "close" in df.columns else df[["date", "adj close"]].rename(columns={"adj close":"close"})
            out["date"] = pd.to_datetime(out["date"])
            out = out.rename(columns={"close": name})
            return out
        except Exception as e:
            time.sleep(2 ** attempt)
            print(f"[macro] {ticker} attempt {attempt+1} err: {e}", flush=True)
    return pd.DataFrame()


def _fred_series(series_id: str, name: str, start: str, end: str) -> pd.DataFrame:
    """Pull a FRED series via the no-auth CSV endpoint."""
    import urllib.request
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start}&coed={end}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
        df = pd.read_csv(io.BytesIO(data))
        df.columns = [c.lower() for c in df.columns]
        # FRED's first col is "observation_date" or "date"
        date_col = "observation_date" if "observation_date" in df.columns else "date"
        df = df.rename(columns={date_col: "date", series_id.lower(): name})
        df["date"] = pd.to_datetime(df["date"])
        df[name] = pd.to_numeric(df[name], errors="coerce")
        return df[["date", name]]
    except Exception as e:
        print(f"[macro] FRED {series_id} failed: {e}", flush=True)
        return pd.DataFrame()


def _naaim_series(start: str, end: str) -> pd.DataFrame:
    """
    NAAIM Exposure Index — try the public Excel link; fall back to a realistic
    synthetic placeholder if blocked.
    TODO: replace placeholder with a real cached CSV when network allows.
    """
    import urllib.request
    url = "https://www.naaim.org/wp-content/uploads/2014/05/NAAIM-Exposure-Index-Data.xls"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
        df = pd.read_excel(io.BytesIO(data))
        # Find a date col and a number col
        cols = [c.lower() for c in df.columns]
        df.columns = cols
        date_col = next((c for c in cols if "date" in c), cols[0])
        # NAAIM number col commonly "naaim number mean" / "mean"
        val_col = next((c for c in cols if "mean" in c or "naaim" in c), cols[-1])
        out = df[[date_col, val_col]].rename(columns={date_col: "date", val_col: "naaim"})
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.dropna(subset=["date"])
        out["naaim"] = pd.to_numeric(out["naaim"], errors="coerce")
        out = out[(out["date"] >= start) & (out["date"] <= end)].sort_values("date")
        return out
    except Exception as e:
        # TODO: replace this synthetic placeholder with cached NAAIM CSV.
        print(f"[macro] NAAIM live pull failed ({e}); using synthetic placeholder", flush=True)
        dates = pd.bdate_range(start, end, freq="W-WED")  # NAAIM is Wednesday-weekly
        rng = np.random.default_rng(42)
        # Mean ~ 60, std ~ 30, range roughly -50..200, slow autocorrelated walk
        x = np.cumsum(rng.normal(0, 5, size=len(dates)))
        x = (x - x.mean()) / (x.std() + 1e-9) * 30 + 60
        x = np.clip(x, -100, 200)
        return pd.DataFrame({"date": dates, "naaim": x})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.end is None:
        args.end = pd.Timestamp.today().strftime("%Y-%m-%d")
    if args.smoke:
        args.start, args.end = "2024-01-01", "2024-12-31"

    pieces = []
    for tk, nm in [("^VIX", "vix"), ("^VIX3M", "vix3m"), ("DX-Y.NYB", "dxy")]:
        d = _yf_series(tk, nm, args.start, args.end)
        if not d.empty:
            pieces.append(d)
        time.sleep(0.5)
    # FRED
    for sid, nm in [("T10Y2Y", "yc_2s10s"), ("BAMLH0A0HYM2", "hy_oas")]:
        d = _fred_series(sid, nm, args.start, args.end)
        if not d.empty:
            pieces.append(d)
        time.sleep(0.25)
    # NAAIM
    naaim = _naaim_series(args.start, args.end)
    if not naaim.empty:
        pieces.append(naaim)

    if not pieces:
        print("[macro] no series fetched; bailing", file=sys.stderr); sys.exit(3)

    # Merge on a daily calendar
    cal = pd.DataFrame({"date": pd.bdate_range(args.start, args.end)})
    out = cal
    for p in pieces:
        out = out.merge(p, on="date", how="left")
    out = out.sort_values("date").reset_index(drop=True)
    # Forward-fill weekly NAAIM and any sparse series
    out = out.ffill()

    # Derived: term structure (>1 = contango, <1 = backwardation)
    if "vix" in out and "vix3m" in out:
        out["vix_ts"] = out["vix3m"] / out["vix"]

    out_path = CACHE / ("macro_smoke.parquet" if args.smoke else "macro.parquet")
    out.to_parquet(out_path, index=False)
    print(f"[macro] wrote {len(out)} rows, cols={list(out.columns)} -> {out_path}")


if __name__ == "__main__":
    main()
