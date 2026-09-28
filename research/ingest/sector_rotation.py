"""
Family: sector_rotation  (HC #563 R2 — sector momentum + lead-lag)

WHAT: Per-(sector, date) rotation signals computed from the sector_etf_flows
parquet we just built:
  - rel_strength_spy (already in source — copied through)
  - momentum_cross_20_60  = sign(ret_20d - ret_60d)
  - rs_rank_among_sectors = daily cross-sectional rank of rel_strength_spy
  - lead_lag_score_5d     = mean (rolling-5d corr of this sector's returns vs
                            each other sector's returns leaded by +1 day) —
                            high score = this sector leads others.

SOURCE: data/feature_store/sector_etf_flows/daily.parquet

OUTPUT: data/feature_store/sector_rotation/daily.parquet
Schema: (etf, date, rel_strength_spy, momentum_cross_20_60,
         rs_rank_among_sectors, lead_lag_score_5d)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, "/home/jupiter/Lvl3Quant/research/ingest")
from _common import write_parquet, smoke_log  # type: ignore

FAMILY = "sector_rotation"
SRC = Path("/home/jupiter/Lvl3Quant/data/feature_store/sector_etf_flows/daily.parquet")
SECTOR_SPDRS = ["XLK","XLF","XLE","XLV","XLY","XLP","XLI","XLB","XLU","XLRE","XLC"]


def _lead_lag_score(returns_pivot: pd.DataFrame, target: str, window: int = 5) -> pd.Series:
    """For each date, compute rolling-window corr of target's return at t
       against every other sector's return at t+1. Aggregate to a single score:
       mean of these corrs. High score = target leads."""
    others = [c for c in returns_pivot.columns if c != target]
    out = pd.Series(index=returns_pivot.index, dtype="float64")
    tgt = returns_pivot[target]
    for d in returns_pivot.index:
        # for each window ending at d, look back `window` rows
        idx_end = returns_pivot.index.get_loc(d)
        if idx_end < window:
            out.loc[d] = np.nan
            continue
        sl = returns_pivot.iloc[idx_end - window + 1 : idx_end + 1]
        # Shift others by -1 within slice → they represent t+1 returns
        others_shifted = returns_pivot[others].shift(-1).iloc[idx_end - window + 1 : idx_end + 1]
        tgt_slice = sl[target]
        corrs = []
        for o in others:
            s = others_shifted[o]
            if s.notna().sum() < 3:
                continue
            c = tgt_slice.corr(s)
            if pd.notna(c):
                corrs.append(c)
        out.loc[d] = np.mean(corrs) if corrs else np.nan
    return out


def main():
    if not SRC.exists():
        raise FileNotFoundError(f"source missing: {SRC}; run sector_etf_flows.py first")
    src = pd.read_parquet(SRC)
    # Keep only the 11 sector SPDRs (sub-sectors and SPY excluded from rotation panel)
    df = src[src["etf"].isin(SECTOR_SPDRS)].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["etf","date"]).reset_index(drop=True)
    print(f"[{FAMILY}] input shape={df.shape}, {df['etf'].nunique()} SPDR sectors")

    df["momentum_cross_20_60"] = np.sign(df["ret_20d"] - df["ret_60d"])

    # Daily cross-sectional rank of rel_strength_spy among the 11 sectors
    df["rs_rank_among_sectors"] = df.groupby("date")["rel_strength_spy"].rank(method="dense")

    # Lead-lag — compute per sector using a returns pivot
    returns_pivot = df.pivot(index="date", columns="etf", values="ret_1d").sort_index()
    print(f"[{FAMILY}] computing lead-lag scores (this may take a minute) ...")
    ll_parts = []
    for s in SECTOR_SPDRS:
        if s not in returns_pivot.columns:
            continue
        ll = _lead_lag_score(returns_pivot, s, window=5).rename("lead_lag_score_5d")
        ll = ll.reset_index()
        ll["etf"] = s
        ll_parts.append(ll[["etf","date","lead_lag_score_5d"]])
    ll_df = pd.concat(ll_parts, ignore_index=True)
    df = df.merge(ll_df, on=["etf","date"], how="left")

    out = df[["etf","date","rel_strength_spy","momentum_cross_20_60",
              "rs_rank_among_sectors","lead_lag_score_5d"]].dropna(
        subset=["rel_strength_spy"]).reset_index(drop=True)

    p = write_parquet(out, FAMILY, "daily.parquet")
    smoke_log(FAMILY, True, f"{len(out)} rows, {out['etf'].nunique()} sectors -> {p}")
    print(f"OK {FAMILY}: {len(out)} rows, {out['etf'].nunique()} sectors -> {p}")
    print(out.tail(5).to_string())
    return p


def run_full():
    return main()


if __name__ == "__main__":
    main()
