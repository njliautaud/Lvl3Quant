#!/usr/bin/env python3
"""Fetch SPY daily + VIX daily + VIX3M daily from Yahoo. Save parquet.
NAAIM is loaded from data/naaim_weekly.parquet if present, else skipped.
"""
import os
import sys
import time
import pandas as pd
import yfinance as yf

OUT = "/home/jupiter/Lvl3Quant/output/macro_swing_v1"
DATA = "/home/jupiter/Lvl3Quant/data"
START = "2010-01-01"
END = "2026-06-05"


def log(msg):
    print(msg, flush=True)
    with open(os.path.join(OUT, "run_log.txt"), "a") as f:
        f.write(msg + "\n")


def fetch(ticker, name):
    log(f"[fetch] {ticker} ({name}) {START} -> {END}")
    for attempt in range(3):
        try:
            df = yf.download(ticker, start=START, end=END, progress=False, auto_adjust=False)
            if df is None or len(df) == 0:
                raise RuntimeError("empty")
            # yfinance returns multi-level columns when single ticker -> flatten
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df = df.reset_index()
            df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
            log(f"  rows={len(df)}  cols={list(df.columns)}  first={df['date'].iloc[0]}  last={df['date'].iloc[-1]}")
            out_path = os.path.join(OUT, f"{name}_daily.parquet")
            df.to_parquet(out_path)
            log(f"  saved -> {out_path}")
            return df
        except Exception as e:
            log(f"  attempt {attempt+1} failed: {e}")
            time.sleep(2)
    log(f"  GIVING UP on {ticker}")
    return None


def main():
    open(os.path.join(OUT, "run_log.txt"), "w").close()  # truncate
    log(f"=== fetch_data.py START ===")
    spy = fetch("SPY", "spy")
    vix = fetch("^VIX", "vix")
    vix3m = fetch("^VIX3M", "vix3m")
    # NAAIM check
    naaim_path = os.path.join(DATA, "naaim_weekly.parquet")
    if os.path.exists(naaim_path):
        n = pd.read_parquet(naaim_path)
        log(f"[naaim] found {naaim_path}  rows={len(n)}  cols={list(n.columns)}")
    else:
        log(f"[naaim] NOT FOUND at {naaim_path} -- will run non-NAAIM cells only")
    log(f"=== fetch_data.py DONE ===")


if __name__ == "__main__":
    main()
