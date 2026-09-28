"""Shared helpers for ingest scaffolds (HC #563)."""
from __future__ import annotations
import os, sys, time, json, datetime as dt
from pathlib import Path
import pandas as pd

ROOT  = Path("/home/jupiter/Lvl3Quant")
STORE = ROOT / "data" / "feature_store"
LOGS  = ROOT / "logs" / "ingest_smoke"

# Default smoke universe — small, diverse mega caps used for smoke tests.
SMOKE_UNIVERSE = ["AAPL","MSFT","NVDA","JPM","XOM","UNH","TSLA","META","GOOGL","AMZN"]

# SEC EDGAR rate-limit: 10 req/sec, with a polite user agent header.
SEC_HEADERS = {
    "User-Agent": "Lvl3Quant Research qa@example.com",
    "Accept-Encoding": "gzip, deflate",
    "Host": "www.sec.gov",
}

def now_iso():
    return dt.datetime.now().isoformat(timespec="seconds")

def write_parquet(df: pd.DataFrame, family: str, name: str = "smoke.parquet"):
    out = STORE / family
    out.mkdir(parents=True, exist_ok=True)
    p = out / name
    df.to_parquet(p, index=False)
    return p

def smoke_log(family: str, ok: bool, msg: str = ""):
    LOGS.mkdir(parents=True, exist_ok=True)
    p = LOGS / f"{family}.log"
    with open(p, "a") as f:
        f.write(f"{now_iso()}  {'OK' if ok else 'FAIL'}  {msg}\n")
    return p
