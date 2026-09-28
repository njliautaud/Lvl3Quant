"""
ingest_macro_extra.py — Loads the 28 FRED macro indicators sourced from
the MacroStrategy repo cache. Per HC #555 R1+R5.

Input:  data/cache/macro_extra/fred_*.csv  (date, <SERIES_ID>)
Output: data/cache/macro_extra.parquet     (date + all series, FFilled)

Series included (28):
  Rates:        T10Y2Y, FEDFUNDS, DGS2, DGS10, DGS30, TEDRATE
  Inflation:    CPIAUCSL, PCEPI, PPIACO, T5YIE, T10YIE, WPSFD49207
  Labor:        UNRATE, PAYEMS, JTSJOL, MANEMP, CIVPART, ICSA
  Activity:    INDPRO, IPMAN, TCU, DGORDER, BUSINV
  Housing:     HOUST, PERMIT, HSN1F
  Sentiment:   UMCSENT, CSCICP03USM665S

Lowercase column names for downstream consumption.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
FRED_DIR = CACHE / "macro_extra"


# Human-friendly column rename map (matches what regime_overlay.py expects)
RENAME = {
    "T10Y2Y": "yc_2s10s",
    "FEDFUNDS": "fed_funds",
    "DGS2": "ust_2y",
    "DGS10": "ust_10y",
    "DGS30": "ust_30y",
    "TEDRATE": "ted_spread",
    "CPIAUCSL": "cpi",
    "PCEPI": "pce",
    "PPIACO": "ppi",
    "T5YIE": "be_5y",
    "T10YIE": "be_10y",
    "WPSFD49207": "ppi_final_demand",
    "UNRATE": "unrate",
    "PAYEMS": "payems",
    "JTSJOL": "job_openings",
    "MANEMP": "mfg_emp",
    "CIVPART": "labor_part",
    "ICSA": "jobless_claims",
    "INDPRO": "indpro",
    "IPMAN": "ip_mfg",
    "TCU": "cap_util",
    "DGORDER": "durable_goods",
    "BUSINV": "biz_inv",
    "HOUST": "housing_starts",
    "PERMIT": "building_permits",
    "HSN1F": "new_home_sales",
    "UMCSENT": "umich_sent",
    "CSCICP03USM665S": "oecd_cci_us",
}


def _load_one(fp: Path) -> pd.DataFrame:
    series_id = fp.stem.replace("fred_", "")
    try:
        df = pd.read_csv(fp)
    except Exception as e:
        print(f"[macro_extra] {fp.name} read error: {e}", flush=True)
        return pd.DataFrame()
    if df.empty or "date" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    # the value column is whatever isn't date
    val_cols = [c for c in df.columns if c != "date"]
    if not val_cols:
        return pd.DataFrame()
    df = df[["date", val_cols[0]]].copy()
    out_name = RENAME.get(series_id, series_id.lower())
    df = df.rename(columns={val_cols[0]: out_name})
    df[out_name] = pd.to_numeric(df[out_name], errors="coerce")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default=None)
    args = ap.parse_args()
    end = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")

    if not FRED_DIR.exists():
        raise SystemExit(f"[macro_extra] cache dir not found: {FRED_DIR}")

    files = sorted(FRED_DIR.glob("fred_*.csv"))
    if not files:
        raise SystemExit("[macro_extra] no fred_*.csv files in cache")

    pieces = []
    for fp in files:
        d = _load_one(fp)
        if not d.empty:
            pieces.append(d)
            print(f"[macro_extra] loaded {fp.name}: {len(d)} rows, col={d.columns[1]}", flush=True)

    if not pieces:
        raise SystemExit("[macro_extra] all loads empty")

    # Merge onto a business-day calendar
    cal = pd.DataFrame({"date": pd.bdate_range(args.start, end)})
    out = cal
    for p in pieces:
        out = out.merge(p, on="date", how="left")

    # Forward-fill monthly/weekly indicators onto the daily calendar
    out = out.sort_values("date").reset_index(drop=True)
    value_cols = [c for c in out.columns if c != "date"]
    out[value_cols] = out[value_cols].ffill()

    out_path = CACHE / "macro_extra.parquet"
    out.to_parquet(out_path, index=False)
    print(f"[macro_extra] wrote {len(out)} rows × {len(value_cols)} cols -> {out_path}")
    print(f"[macro_extra] cols: {value_cols}")


if __name__ == "__main__":
    main()
