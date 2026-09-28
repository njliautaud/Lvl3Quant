"""
ingest_sector_flows.py

Builds per-day SECTOR-FLOW features from the existing sector_etfs.parquet.
Real money-flow data (ICI / ETF.com) is paid; we synthesize strong proxies
from price + dollar-volume that capture the same signal:

  * dv_z_20d        — 20-day Z-score of dollar volume (anomalous accumulation)
  * dv_chg_60d      — 60-day pct change in 20d-mean dollar volume (flow trend)
  * rs_60d_spy      — sector ETF 60d return MINUS SPY 60d return (leadership)
  * rs_252d_spy     — 252d return minus SPY (longer-cycle leadership)
  * px_200dma       — sector ETF price / 200-day MA (regime)
  * vol_252d        — 252-day realized vol
  * corr_tlt_60d    — 60-day corr w/ TLT (rate sensitivity)
  * corr_hyg_60d    — 60-day corr w/ HYG (credit risk-on)
  * corr_uup_60d    — 60-day corr w/ UUP (dollar)
  * corr_gld_60d    — 60-day corr w/ GLD (real assets)

Maps sector-name -> SPDR sector ETF for the join from universe to flow:
    Technology              -> XLK
    Financial Services      -> XLF
    Healthcare              -> XLV
    Energy                  -> XLE
    Consumer Cyclical       -> XLY
    Consumer Defensive      -> XLP
    Industrials             -> XLI
    Utilities               -> XLU
    Basic Materials         -> XLB
    Real Estate             -> XLRE
    Communication Services  -> XLC

Output:
    data/cache/sector_flows_daily.parquet
    columns: sector_etf | date | feature columns above

Run: python -m data.ingest_sector_flows
"""
from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"

SECTOR_ETFS = ["XLK", "XLF", "XLV", "XLE", "XLY", "XLP", "XLI", "XLU", "XLB", "XLRE", "XLC"]
MACRO_REF = ["SPY", "TLT", "HYG", "UUP", "GLD"]

SECTOR_TO_ETF = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Healthcare": "XLV",
    "Energy": "XLE",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Industrials": "XLI",
    "Utilities": "XLU",
    "Basic Materials": "XLB",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}


def _load_etf_panel() -> pd.DataFrame:
    df = pd.read_parquet(CACHE / "sector_etfs.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df


def _per_etf_features(etf: str, panel: pd.DataFrame, macro_returns: pd.DataFrame) -> pd.DataFrame:
    sub = panel[panel["ticker"] == etf].copy()
    sub = sub.sort_values("date").reset_index(drop=True)
    if sub.empty:
        return pd.DataFrame()

    sub["dv_ma20"] = sub["dollar_volume"].rolling(20).mean()
    sub["dv_sd20"] = sub["dollar_volume"].rolling(20).std()
    sub["dv_z_20d"] = (sub["dollar_volume"] - sub["dv_ma20"]) / sub["dv_sd20"]
    sub["dv_chg_60d"] = sub["dv_ma20"].pct_change(60)

    sub["ret_60d"] = sub["close"].pct_change(60)
    sub["ret_252d"] = sub["close"].pct_change(252)
    sub["px_ma200"] = sub["close"].rolling(200).mean()
    sub["px_200dma"] = sub["close"] / sub["px_ma200"] - 1.0
    sub["vol_252d"] = sub["log_ret_1d"].rolling(252).std() * np.sqrt(252)

    # Merge SPY returns for relative strength
    spy = macro_returns[macro_returns["ticker"] == "SPY"][["date", "ret_60d", "ret_252d"]].rename(
        columns={"ret_60d": "spy_60d", "ret_252d": "spy_252d"}
    )
    sub = sub.merge(spy, on="date", how="left")
    sub["rs_60d_spy"] = sub["ret_60d"] - sub["spy_60d"]
    sub["rs_252d_spy"] = sub["ret_252d"] - sub["spy_252d"]

    # 60d rolling correlations with macro reference ETFs
    log_ret_etf = sub.set_index("date")["log_ret_1d"]
    for ref in ("TLT", "HYG", "UUP", "GLD"):
        ref_lr = macro_returns[macro_returns["ticker"] == ref].set_index("date")["log_ret_1d"]
        joined = pd.concat([log_ret_etf.rename("a"), ref_lr.rename("b")], axis=1).dropna()
        corr = joined["a"].rolling(60).corr(joined["b"])
        sub[f"corr_{ref.lower()}_60d"] = sub["date"].map(corr)

    sub["sector_etf"] = etf
    keep_cols = [
        "sector_etf", "date",
        "dv_z_20d", "dv_chg_60d",
        "rs_60d_spy", "rs_252d_spy",
        "px_200dma", "vol_252d",
        "corr_tlt_60d", "corr_hyg_60d",
        "corr_uup_60d", "corr_gld_60d",
    ]
    return sub[keep_cols]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sector_flows_daily.parquet")
    args = ap.parse_args()

    panel = _load_etf_panel()
    print(f"[sector_flows] panel rows={len(panel):,} tickers={panel['ticker'].nunique()}")

    # Pre-compute macro reference returns
    macro = panel[panel["ticker"].isin(MACRO_REF)].copy()
    macro["ret_60d"] = macro.groupby("ticker")["close"].pct_change(60)
    macro["ret_252d"] = macro.groupby("ticker")["close"].pct_change(252)

    out = []
    for etf in SECTOR_ETFS:
        feats = _per_etf_features(etf, panel, macro)
        if feats.empty:
            print(f"  {etf}: empty — skipping")
            continue
        out.append(feats)
        print(f"  {etf}: {len(feats):,} rows")

    daily = pd.concat(out, ignore_index=True)
    out_path = CACHE / args.out
    daily.to_parquet(out_path, index=False)
    print(f"[sector_flows] wrote {out_path}  shape={daily.shape}")

    # Sanity: tail features per sector
    print("\nSpot-check: 2024-06-28 sector RS_60d vs SPY (positive = leading):")
    snap = daily[daily["date"] == "2024-06-28"][["sector_etf", "rs_60d_spy", "dv_z_20d", "px_200dma"]]
    print(snap.to_string(index=False))


if __name__ == "__main__":
    main()
