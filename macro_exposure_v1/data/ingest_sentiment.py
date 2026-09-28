"""
ingest_sentiment.py — NAAIM + AAII + put/call ratio.

Priority order for NAAIM:
  1) Reuse wheel_strategy_v1/data/cache/macro.parquet column 'naaim' if present
     (the wheel pipeline already pulls NAAIM — don't duplicate the request).
  2) Fall back to live NAAIM .xls download (same logic as wheel_strategy_v1/data/ingest_macro.py).
  3) Synthetic placeholder (NaN with TODO note) if both fail — fitness function
     just doesn't get the NAAIM gene's signal contribution then.

AAII bull/bear: TODO — no free clean CSV without registration.
Put/call ratio: TODO — CBOE free historical is gone; use VIX-derived proxy as
fallback (high VIX -> elevated put hedging — noisy but better than nothing).

Output: data/cache/sentiment.parquet
        columns: date, naaim, aaii_bull, aaii_bear, aaii_bullbear, putcall_proxy
"""
from __future__ import annotations
import io
import argparse
from pathlib import Path
import urllib.request
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
WHEEL_CACHE = ROOT.parent / "wheel_strategy_v1" / "data" / "cache"


def _from_wheel_cache() -> pd.DataFrame:
    p = WHEEL_CACHE / "macro.parquet"
    if not p.exists():
        # wheel may have a smoke version while full is running
        p = WHEEL_CACHE / "macro_smoke.parquet"
        if not p.exists():
            return pd.DataFrame()
    try:
        df = pd.read_parquet(p)
        if "naaim" not in df.columns:
            return pd.DataFrame()
        out = df[["date", "naaim"]].dropna(subset=["date"]).copy()
        out["date"] = pd.to_datetime(out["date"])
        print(f"[sentiment] reused NAAIM from wheel cache {p.name} ({len(out)} rows)", flush=True)
        return out
    except Exception as e:
        print(f"[sentiment] wheel cache read failed: {e}", flush=True)
        return pd.DataFrame()


def _naaim_live(start: str, end: str) -> pd.DataFrame:
    url = "https://www.naaim.org/wp-content/uploads/2014/05/NAAIM-Exposure-Index-Data.xls"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
        df = pd.read_excel(io.BytesIO(data))
        df.columns = [str(c).lower() for c in df.columns]
        date_col = next((c for c in df.columns if "date" in c), df.columns[0])
        val_col = next((c for c in df.columns if "mean" in c or "naaim" in c), df.columns[-1])
        out = df[[date_col, val_col]].rename(columns={date_col: "date", val_col: "naaim"})
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out["naaim"] = pd.to_numeric(out["naaim"], errors="coerce")
        out = out.dropna(subset=["date"]).sort_values("date")
        out = out[(out["date"] >= start) & (out["date"] <= end)]
        print(f"[sentiment] NAAIM live pull {len(out)} rows", flush=True)
        return out
    except Exception as e:
        print(f"[sentiment] NAAIM live pull failed: {e}", flush=True)
        return pd.DataFrame()


def _naaim_synthetic(start: str, end: str) -> pd.DataFrame:
    # TODO: replace with cached NAAIM CSV when available.
    dates = pd.bdate_range(start, end, freq="W-WED")
    rng = np.random.default_rng(42)
    x = np.cumsum(rng.normal(0, 5, size=len(dates)))
    if len(dates) > 0:
        x = (x - x.mean()) / (x.std() + 1e-9) * 30 + 60
        x = np.clip(x, -100, 200)
    print(f"[sentiment] NAAIM SYNTHETIC PLACEHOLDER ({len(dates)} rows)  # TODO: real source", flush=True)
    return pd.DataFrame({"date": dates, "naaim": x})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.end is None:
        args.end = pd.Timestamp.today().strftime("%Y-%m-%d")
    if args.smoke:
        args.start, args.end = "2023-01-01", "2024-12-31"

    # --- NAAIM
    naaim = _from_wheel_cache()
    if naaim.empty:
        naaim = _naaim_live(args.start, args.end)
    if naaim.empty:
        naaim = _naaim_synthetic(args.start, args.end)

    # Build a daily date spine for forward-fill
    spine = pd.DataFrame({"date": pd.bdate_range(args.start, args.end)})
    out = spine.merge(naaim, on="date", how="left").sort_values("date")
    out["naaim"] = out["naaim"].ffill()

    # --- AAII (TODO)
    out["aaii_bull"] = np.nan
    out["aaii_bear"] = np.nan
    out["aaii_bullbear"] = np.nan  # TODO: aaii.com / investorsintelligence CSV

    # --- Put/call ratio (TODO — using a placeholder we will fold in later)
    out["putcall_proxy"] = np.nan  # TODO: CBOE PCR, fallback noisy VIX-based proxy in features step

    sfx = "_smoke" if args.smoke else ""
    path = CACHE / f"sentiment{sfx}.parquet"
    out.to_parquet(path, index=False)
    print(f"[sentiment] wrote {len(out)} rows -> {path}", flush=True)
    print(f"[sentiment] NAAIM non-null: {out['naaim'].notna().sum()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
