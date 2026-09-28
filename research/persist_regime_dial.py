"""
Persist regime modulator output (HC #561 R2).

Calls the public `regime_series()` API from strategy.macro_picker.regime_modulator
and writes the resulting dataframe to data/feature_store/macro_regime/dial_daily.parquet
so downstream consumers (master_panel, GA fit, wheel sizer) can read it.

This is a thin wrapper — it does NOT augment the regime modulator itself.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(ROOT))

from strategy.macro_picker.regime_modulator import regime_series  # type: ignore

OUT_DIR = ROOT / "data/feature_store/macro_regime"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "dial_daily.parquet"


def main():
    print("[persist_regime_dial] computing regime_series ...")
    df = regime_series()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df.to_parquet(OUT, index=False)
    print(f"OK persist_regime_dial: {len(df)} rows -> {OUT}")
    print(df.tail(3).to_string())
    print("\nstate distribution:")
    print(df["state"].value_counts())


if __name__ == "__main__":
    main()
