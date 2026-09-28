"""
Sector Rotation Deep Research — Data Download
Downloads intraday + daily data for all sector ETFs via yfinance.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta

OUT_DIR = '/home/jupiter/Lvl3Quant/research/sector_rotation'
os.makedirs(OUT_DIR, exist_ok=True)

SECTOR_ETFS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLC', 'XLY', 'XLP', 'XLB', 'XLU', 'XLRE']
BENCHMARKS = ['SPY', 'VIX']  # VIX via ^VIX
ALL_TICKERS = SECTOR_ETFS + ['SPY']

# ============================================================
# 1. Daily data — as far back as possible (for long-term rotation analysis)
# ============================================================
print("=== Downloading daily data ===")
daily_data = {}
for tkr in ALL_TICKERS + ['^VIX']:
    label = tkr.replace('^', '')
    print(f"  {label}...", end=" ")
    try:
        df = yf.download(tkr, start='2000-01-01', end='2026-08-23', progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        daily_data[label] = df
        print(f"{len(df)} rows, {df.index.min().date()} to {df.index.max().date()}")
    except Exception as e:
        print(f"FAILED: {e}")

# Save closes
closes = pd.DataFrame({k: v['Close'] for k, v in daily_data.items()})
closes.to_parquet(os.path.join(OUT_DIR, 'sector_daily_closes.parquet'))

# Save volumes
volumes = pd.DataFrame({k: v['Volume'] for k, v in daily_data.items() if 'Volume' in v.columns})
volumes.to_parquet(os.path.join(OUT_DIR, 'sector_daily_volumes.parquet'))

# Save OHLCV for each
for label, df in daily_data.items():
    df.to_parquet(os.path.join(OUT_DIR, f'daily_{label}.parquet'))

print(f"\nDaily closes shape: {closes.shape}")
print(f"Date range: {closes.index.min().date()} to {closes.index.max().date()}")

# ============================================================
# 2. Hourly data — last 2 years (yfinance limit for hourly)
# ============================================================
print("\n=== Downloading hourly data (last 730 days) ===")
hourly_data = {}
for tkr in ALL_TICKERS:
    print(f"  {tkr}...", end=" ")
    try:
        df = yf.download(tkr, period='730d', interval='1h', progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        hourly_data[tkr] = df
        print(f"{len(df)} rows")
    except Exception as e:
        print(f"FAILED: {e}")

# Save hourly closes
hourly_closes = pd.DataFrame({k: v['Close'] for k, v in hourly_data.items()})
hourly_closes.to_parquet(os.path.join(OUT_DIR, 'sector_hourly_closes.parquet'))

# Save hourly volumes
hourly_volumes = pd.DataFrame({k: v['Volume'] for k, v in hourly_data.items()})
hourly_volumes.to_parquet(os.path.join(OUT_DIR, 'sector_hourly_volumes.parquet'))

print(f"\nHourly closes shape: {hourly_closes.shape}")
print(f"Date range: {hourly_closes.index.min()} to {hourly_closes.index.max()}")

# ============================================================
# 3. Download fundamental data (P/E, P/B, dividend yield)
# ============================================================
print("\n=== Downloading fundamental data ===")
fund_data = {}
for tkr in SECTOR_ETFS:
    print(f"  {tkr}...", end=" ")
    try:
        info = yf.Ticker(tkr).info
        fund_data[tkr] = {
            'trailingPE': info.get('trailingPE'),
            'forwardPE': info.get('forwardPE'),
            'priceToBook': info.get('priceToBook'),
            'dividendYield': info.get('dividendYield'),
            'beta': info.get('beta'),
            'totalAssets': info.get('totalAssets'),
            'category': info.get('category', ''),
            'longName': info.get('longName', tkr),
        }
        print(f"PE={fund_data[tkr]['trailingPE']}, PB={fund_data[tkr]['priceToBook']}")
    except Exception as e:
        print(f"FAILED: {e}")
        fund_data[tkr] = {}

with open(os.path.join(OUT_DIR, 'sector_fundamentals.json'), 'w') as f:
    json.dump(fund_data, f, indent=2, default=str)

print("\n=== Data download complete ===")
print(f"Files saved to {OUT_DIR}")
