"""
Master panel v2 — extends master_panel_with_text.parquet by joining the
feature families that the original master_panel.py builder missed.

Adds (PIT-safe, direct (ticker,date) or broadcast):
  - fundamentals_pit_daily.parquet — revenue/margins/ROE/growth/etc.
      (already daily-ffilled with filingDate semantics; direct merge)
  - sector_flows_daily.parquet     — ETF flow Z-scores + macro corrs
      (per-ETF/day; merged on sr_etf+date, prefix sf_)
  - edgar_form4/insider_daily_v2.parquet (if present) — full-universe Form 4
      (replaces the smoke-universe `ins_*` columns from v1 panel)

Input  : data/feature_store/master_panel/master_panel_with_text.parquet
Output : data/feature_store/master_panel/master_panel_v2.parquet

PIT safety: every join uses date <= D semantics. fundamentals_pit was already
built with filingDate (not period-end) so no look-ahead. sector_flows is a
daily snapshot of public ETF data, no look-ahead.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/jupiter/Lvl3Quant")
FS = ROOT / "data/feature_store"
CACHE = ROOT / "wheel_strategy_v1/data/cache"
OUT_DIR = FS / "master_panel"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "master_panel_v2.parquet"

BASE = OUT_DIR / "master_panel_with_text.parquet"


def _stage(name: str, df: pd.DataFrame) -> None:
    n_t = df["ticker"].nunique() if "ticker" in df.columns else 0
    print(f"  after {name:30s} shape={df.shape}  tickers={n_t}")


def _read(path: Path, label: str) -> pd.DataFrame | None:
    if not path.exists():
        print(f"  SKIP {label}: {path.name} not on disk")
        return None
    df = pd.read_parquet(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    print(f"  read {label:30s} rows={len(df):>9,} cols={len(df.columns):2d}")
    return df


def main():
    print(f"[master_panel_v2] base = {BASE.name}")
    panel = _read(BASE, "base panel (with text)")
    if panel is None:
        raise FileNotFoundError(f"base panel missing: {BASE}")
    panel["date"] = pd.to_datetime(panel["date"])
    _stage("base", panel)

    # ---------- fundamentals_pit (numeric fundamentals, daily ffilled) ----------
    fp = _read(CACHE / "fundamentals_pit_daily.parquet", "fundamentals_pit")
    if fp is not None and not fp.empty:
        # Prefix to namespace-protect against accidental name collision later.
        keep = [c for c in fp.columns if c not in ("ticker", "date")]
        rename = {c: f"fp_{c}" for c in keep}
        fp = fp.rename(columns=rename)
        before_cols = panel.shape[1]
        panel = panel.merge(fp, on=["ticker", "date"], how="left")
        print(f"    +fundamentals_pit added {panel.shape[1] - before_cols} cols")
        _stage("fundamentals_pit", panel)

    # ---------- sector_flows (per-ETF Z-scores + macro corrs, broadcast) ----------
    sf = _read(CACHE / "sector_flows_daily.parquet", "sector_flows")
    if sf is not None and not sf.empty and "sr_etf" in panel.columns:
        sf = sf.rename(columns={"sector_etf": "sr_etf"})
        keep = [c for c in sf.columns if c not in ("sr_etf", "date")]
        rename = {c: f"sf_{c}" for c in keep}
        sf = sf.rename(columns=rename)
        before_cols = panel.shape[1]
        panel = panel.merge(sf, on=["sr_etf", "date"], how="left")
        print(f"    +sector_flows added {panel.shape[1] - before_cols} cols")
        _stage("sector_flows", panel)

    # ---------- form4 insider v2 (full universe, if ingest finished) -----------
    f4v2 = FS / "edgar_form4/insider_daily_v2.parquet"
    if f4v2.exists():
        f4 = _read(f4v2, "edgar_form4_v2 (full)")
        if f4 is not None and not f4.empty:
            # Drop any existing ins_* cols from base panel — v2 supersedes.
            old_ins = [c for c in panel.columns if c.startswith("ins_")]
            if old_ins:
                print(f"    dropping {len(old_ins)} v1 ins_* cols (superseded)")
                panel = panel.drop(columns=old_ins)
            rename = {c: f"ins_{c}" for c in f4.columns if c not in ("ticker", "date")}
            f4 = f4.rename(columns=rename)
            before_cols = panel.shape[1]
            panel = panel.merge(f4, on=["ticker", "date"], how="left")
            for c in [c for c in panel.columns if c.startswith("ins_")]:
                if pd.api.types.is_numeric_dtype(panel[c]):
                    panel[c] = panel[c].fillna(0)
            print(f"    +form4_v2 added {panel.shape[1] - before_cols} cols")
            _stage("form4_v2 insider", panel)
    else:
        print(f"  SKIP form4_v2: {f4v2.name} not on disk yet (ingest in-flight)")

    # ---------- finalize ----------
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    print(f"\n[master_panel_v2] final shape: {panel.shape}")
    print(f"[master_panel_v2] tickers: {panel['ticker'].nunique()}")
    print(f"[master_panel_v2] date range: {panel['date'].min()} -> {panel['date'].max()}")

    # Null-rate report (HC #563 R1 — schema honesty)
    print("\n[master_panel_v2] null-rate by NEW column (>5%):")
    base_cols = set(pd.read_parquet(BASE, columns=None).columns)
    new_cols = [c for c in panel.columns if c not in base_cols]
    null_rate = panel[new_cols].isna().mean().sort_values(ascending=False)
    for c, r in null_rate.items():
        if r > 0.05:
            print(f"  {c:38s} {r*100:5.1f}%")
    if null_rate.empty or (null_rate <= 0.05).all():
        print("  (no NEW columns above 5% null)")

    panel.to_parquet(OUT, index=False)
    size_mb = OUT.stat().st_size / (1024 * 1024)
    print(f"\nOK master_panel_v2: wrote {len(panel):,} rows x {len(panel.columns)} cols "
          f"({size_mb:.1f} MB) -> {OUT}")
    print(f"   new families joined this build: fundamentals_pit, sector_flows"
          + (", form4_v2" if f4v2.exists() else " (form4_v2 pending)"))
    return OUT


if __name__ == "__main__":
    main()
