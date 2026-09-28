"""
Step 1 — Fold 10-K text features into master_panel (PIT-safe).

For each (ticker, date) in master_panel/daily.parquet, attach the most recent
10-K filing whose available_from <= date (forward-fill within ticker).

Adds columns:
    lm_tone_score, lm_tone_score_z (252d rolling z within ticker),
    rf_delta_word_count, rf_delta_z,
    flag_going_concern, flag_accounting_change, flag_restatement,
    days_since_filing.

Rows with days_since_filing > 365 -> text features set to NaN (stale).
Writes to: data/feature_store/master_panel/master_panel_with_text.parquet
Keeps daily.parquet intact for diff.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/jupiter/Lvl3Quant")
PANEL = ROOT / "data/feature_store/master_panel/daily.parquet"
TEXT = ROOT / "data/feature_store/edgar_10k_text/_all.parquet"
OUT = ROOT / "data/feature_store/master_panel/master_panel_with_text.parquet"


def main():
    print("loading panel ...")
    p = pd.read_parquet(PANEL)
    p["date"] = pd.to_datetime(p["date"])
    print(f"  panel: {p.shape}")

    print("loading 10-K text ...")
    t = pd.read_parquet(TEXT)
    t["available_from"] = pd.to_datetime(t["available_from"])
    t = t.sort_values(["ticker", "available_from"]).reset_index(drop=True)
    # Drop the heavy text-excerpt columns; we only need the numeric features.
    t = t[[
        "ticker", "available_from",
        "tone_score", "rf_delta_word_count",
        "flag_going_concern", "flag_accounting_change", "flag_restatement",
    ]].copy()
    # rename to canonical
    t = t.rename(columns={"tone_score": "lm_tone_score"})
    t["flag_going_concern"] = t["flag_going_concern"].astype(int)
    t["flag_accounting_change"] = t["flag_accounting_change"].astype(int)
    t["flag_restatement"] = t["flag_restatement"].astype(int)
    print(f"  filings: {t.shape}, tickers={t['ticker'].nunique()}")

    # As-of merge per ticker: for each (ticker, date), find most recent filing
    # with available_from <= date.
    p = p.sort_values(["ticker", "date"]).reset_index(drop=True)
    parts = []
    universe = sorted(set(p["ticker"].unique()) & set(t["ticker"].unique()))
    print(f"  ticker overlap: {len(universe)} / panel={p['ticker'].nunique()} / text={t['ticker'].nunique()}")

    # Per-ticker as-of merge so we can carry filing_date for days_since_filing.
    for i, tk in enumerate(universe):
        if i % 50 == 0:
            print(f"  merging {i}/{len(universe)} ({tk})")
        pp = p[p["ticker"] == tk].copy()
        tt = t[t["ticker"] == tk].copy()
        if tt.empty:
            parts.append(pp)
            continue
        tt = tt.sort_values("available_from").reset_index(drop=True)
        # merge_asof needs sorted keys
        pp = pp.sort_values("date").reset_index(drop=True)
        merged = pd.merge_asof(
            pp, tt.drop(columns=["ticker"]),
            left_on="date", right_on="available_from",
            direction="backward",
        )
        merged["days_since_filing"] = (merged["date"] - merged["available_from"]).dt.days
        parts.append(merged)

    # Tickers with no 10-K coverage: append with NaN text cols
    no_text = sorted(set(p["ticker"].unique()) - set(t["ticker"].unique()))
    for tk in no_text:
        pp = p[p["ticker"] == tk].copy()
        for c in ["available_from", "lm_tone_score", "rf_delta_word_count",
                  "flag_going_concern", "flag_accounting_change", "flag_restatement",
                  "days_since_filing"]:
            pp[c] = np.nan
        parts.append(pp)

    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Staleness handling — > 365 days = NaN out the text signals
    stale = out["days_since_filing"] > 365
    for c in ["lm_tone_score", "rf_delta_word_count",
              "flag_going_concern", "flag_accounting_change", "flag_restatement"]:
        out.loc[stale, c] = np.nan

    # 252d rolling z-scores within ticker
    print("computing rolling z-scores ...")
    def _rollz(s: pd.Series, win: int = 252) -> pd.Series:
        mu = s.rolling(win, min_periods=60).mean()
        sd = s.rolling(win, min_periods=60).std()
        z = (s - mu) / sd.replace(0.0, np.nan)
        return z.replace([np.inf, -np.inf], np.nan)

    out["lm_tone_score_z"] = (
        out.groupby("ticker")["lm_tone_score"].transform(lambda s: _rollz(s, 252))
    )
    out["rf_delta_z"] = (
        out.groupby("ticker")["rf_delta_word_count"].transform(lambda s: _rollz(s, 252))
    )

    # Coverage diagnostics
    cov = out[["lm_tone_score", "rf_delta_word_count",
               "flag_going_concern", "lm_tone_score_z", "rf_delta_z"]].notna().mean()
    print("\nCoverage (fraction non-null) on panel rows:")
    for k, v in cov.items():
        print(f"  {k:>30s}: {v:.3f}")
    print(f"\nRows with days_since_filing<=180: {(out['days_since_filing']<=180).mean():.3f}")
    print(f"Rows with days_since_filing<=365: {(out['days_since_filing']<=365).mean():.3f}")

    print(f"\nwriting {OUT} ...")
    out.to_parquet(OUT, index=False)
    print(f"final shape: {out.shape}, cols: {len(out.columns)}")
    print("new cols:", [c for c in out.columns if c not in pd.read_parquet(PANEL).columns])


if __name__ == "__main__":
    main()
