"""
ingest_universe.py — Build the wheel-strategy universe.

S&P 500 + ~50 popular optionable mid-caps. Tries yfinance for ticker validation
and basic metadata; falls back to a hardcoded curated list if yfinance is
rate-limited (HC #542 R3: don't block — proceed with whatever we got).

Output: data/cache/universe.parquet  (ticker, name, sector, market_cap, source)
"""
from __future__ import annotations
import os
import sys
import time
import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

# -------- Curated S&P500 core (most-liquid) + mid-cap optionable extras --------
# Hand-curated, frozen list. Used as the fallback when yfinance is rate-limited.
SP500_CORE = [
    "AAPL","MSFT","NVDA","GOOGL","GOOG","AMZN","META","TSLA","BRK-B","UNH",
    "JPM","XOM","JNJ","V","PG","MA","HD","LLY","CVX","ABBV",
    "MRK","PEP","KO","BAC","AVGO","WMT","COST","DIS","ADBE","CSCO",
    "TMO","ACN","MCD","ABT","CRM","NFLX","LIN","AMD","TXN","DHR",
    "WFC","NEE","PM","UPS","RTX","NKE","ORCL","INTC","IBM","QCOM",
    "HON","T","LOW","INTU","CAT","SBUX","GS","MS","SPGI","BLK",
    "BA","DE","GE","MMM","AXP","C","USB","PNC","CME","ICE",
    "SCHW","COF","TGT","CMCSA","VZ","TMUS","CHTR","NOW","AMAT","LRCX",
    "KLAC","MU","ADI","SNPS","CDNS","PYPL","BKNG","UBER","ABNB","SQ",
    "SHOP","CRWD","PANW","NET","DDOG","SNOW","ZS","MDB","TEAM","WDAY",
    "GILD","AMGN","BMY","CVS","CI","ELV","HCA","ISRG","REGN","VRTX",
    "ZTS","SYK","BSX","BDX","MDT","EW","ILMN","DXCM","IDXX","MRNA",
    "PFE","COIN","HOOD","RIVN","LCID","F","GM","SLB","COP","EOG",
    "OXY","MPC","VLO","PSX","HAL","DVN","WMB","KMI","PXD","FANG",
    "DUK","SO","D","AEP","SRE","XEL","ED","EXC","PEG","WEC",
    "PLD","AMT","CCI","EQIX","SPG","O","WELL","PSA","DLR","SBAC",
    "AVB","EQR","ESS","MAA","UDR","CPT","ARE","BXP","HST","REG",
    "ECL","SHW","APD","LIN","FCX","NEM","NUE","STLD","CLF","X",
    "MOS","CF","DOW","DD","LYB","PPG","RPM","ALB","CTVA","FMC",
    "WM","RSG","WCN","REPL","WAT","DGX","LH","HOLX","TFX","ALGN",
    "F","TSN","K","GIS","HSY","CPB","CAG","SJM","HRL","MKC",
    "MO","KHC","STZ","BF-B","TAP","CCEP","KDP","MNST","CL","CHD",
    "KMB","CLX","EL","COTY","NWL","TPR","UAA","HBI","ROST","TJX",
    "ULTA","BURL","LULU","RH","WSM","BBY","KSS","M","JWN","DKS",
    "ETSY","EBAY","CHWY","W","PINS","SNAP","MTCH","ROKU","SPOT","TTD",
    "MELI","BIDU","JD","PDD","BABA","NTES","DIDI","NIO","XPEV","LI",
]

# Popular optionable mid-caps (frequently traded for premium, good liquidity)
MIDCAP_OPTIONABLE = [
    "PLTR","SOFI","RBLX","U","DKNG","PENN","CHPT","RUN","NOVA","ENPH",
    "FSLR","SEDG","BE","PLUG","BLDP","FCEL","BLNK","CHPT","EVGO","RIVN",
    "LCID","WKHS","HYZN","NKLA","GOEV","RIDE","FFIE","MULN","XPEV","NIO",
    "SOFI","UPST","AFRM","SOFI","OPEN","Z","ZG","RDFN","HOOD","COIN",
    "MARA","RIOT","CLSK","BTBT","HUT","BITF","GLXY","SI","SBNY","SLM",
]

# ETFs commonly used in wheel strategies (broad, sector, vol)
ETFS = [
    "SPY","QQQ","IWM","DIA","XLE","XLF","XLK","XLV","XLI","XLY",
    "XLP","XLU","XLB","XLRE","XLC","SMH","SOXX","ARKK","TLT","HYG",
    "GLD","SLV","USO","UNG","UVXY","SQQQ","TQQQ","SOXL","SOXS","LABU",
]


def _curated_fallback() -> pd.DataFrame:
    tickers = sorted(set(SP500_CORE + MIDCAP_OPTIONABLE + ETFS))
    df = pd.DataFrame({"ticker": tickers})
    df["name"] = ""
    df["sector"] = "Unknown"
    df["market_cap"] = float("nan")
    df["source"] = "curated_fallback"
    return df


def _enrich_with_yfinance(df: pd.DataFrame, max_per_call=1, sleep=0.25) -> pd.DataFrame:
    """
    Best-effort enrichment. Skips silently on yfinance errors (HC #542 R3:
    proceed with whatever we got — don't block on rate limits).
    """
    try:
        import yfinance as yf
    except Exception as e:
        print(f"[universe] yfinance import failed: {e}; using curated only", flush=True)
        return df

    enriched = []
    for i, row in df.iterrows():
        t = row["ticker"]
        try:
            info = yf.Ticker(t).info or {}
            enriched.append({
                "ticker": t,
                "name": info.get("shortName") or info.get("longName") or "",
                "sector": info.get("sector") or "Unknown",
                "market_cap": float(info.get("marketCap") or float("nan")),
                "source": "yfinance",
            })
        except Exception:
            enriched.append({"ticker": t, "name": "", "sector": "Unknown",
                             "market_cap": float("nan"), "source": "curated_fallback"})
        if i % 25 == 0:
            print(f"[universe] enriched {i}/{len(df)}", flush=True)
        time.sleep(sleep)
    return pd.DataFrame(enriched)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-enrich", action="store_true",
                    help="Skip yfinance enrichment; use curated list only.")
    ap.add_argument("--smoke", action="store_true",
                    help="Use a tiny smoke universe (5 names).")
    args = ap.parse_args()

    if args.smoke:
        df = pd.DataFrame({
            "ticker": ["SPY","QQQ","AAPL","MSFT","NVDA"],
            "name": ["SPDR S&P 500","Invesco QQQ","Apple","Microsoft","NVIDIA"],
            "sector": ["ETF","ETF","Technology","Technology","Technology"],
            "market_cap": [float("nan")]*5,
            "source": ["smoke"]*5,
        })
        out = CACHE / "universe.parquet"
        df.to_parquet(out, index=False)
        print(f"[universe] SMOKE wrote {len(df)} -> {out}")
        return

    df = _curated_fallback()
    if not args.no_enrich:
        df = _enrich_with_yfinance(df)
    out = CACHE / "universe.parquet"
    df.to_parquet(out, index=False)
    print(f"[universe] wrote {len(df)} tickers -> {out}")


if __name__ == "__main__":
    main()
