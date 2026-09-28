#!/usr/bin/env python3
"""
Download ES futures MBO data for Dec 2025 - Mar 2026 from Databento.

Purpose: Out-of-time validation for Strategy B-MA v2 (currently trained on Jul-Nov 2025).

Usage:
    # Cost estimate only (safe):
    python fetch_es_mbo_extension.py --cost-only

    # Download data:
    python fetch_es_mbo_extension.py

Requires:
    pip install databento
    export DATABENTO_KEY=db-xxxxx

After download, process with:
    python rust_cache_builder/... (or the existing pipeline)
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import date

KEY = os.environ.get("DATABENTO_KEY", "")
START = "2025-12-01"
END = "2026-03-07"  # Up to today
SYMBOL = "ES.FUT"  # ES continuous front month
DATASET = "GLBX.MDP3"  # CME Globex
SCHEMA = "mbo"  # Full order book

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw" / "es_mbo_extension"


def main():
    parser = argparse.ArgumentParser(description="Fetch ES MBO extension data")
    parser.add_argument("--cost-only", action="store_true", help="Only estimate cost")
    args = parser.parse_args()

    if not KEY:
        print("ERROR: Set DATABENTO_KEY environment variable first.")
        print("  export DATABENTO_KEY=db-xxxxx")
        sys.exit(1)

    try:
        import databento as db
    except ImportError:
        print("ERROR: pip install databento")
        sys.exit(1)

    client = db.Historical(KEY)

    # Cost estimate
    print(f"Dataset: {DATASET}, Symbol: {SYMBOL}, Schema: {SCHEMA}")
    print(f"Date range: {START} to {END}")

    try:
        cost = client.metadata.get_cost(
            dataset=DATASET,
            symbols=[SYMBOL],
            schema=SCHEMA,
            start=START,
            end=END,
        )
        print(f"\nEstimated cost: ${cost / 100:.2f}")
    except Exception as e:
        print(f"Cost estimate failed: {e}")
        print("Proceeding to download (you can cancel if cost is too high)")

    if args.cost_only:
        return

    # Download
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / f"es_mbo_{START}_{END}.dbn.zst"

    print(f"\nDownloading to {out_file}...")
    data = client.timeseries.get_range(
        dataset=DATASET,
        symbols=[SYMBOL],
        schema=SCHEMA,
        start=START,
        end=END,
    )
    data.to_file(str(out_file))
    print(f"Saved: {out_file} ({out_file.stat().st_size / 1e9:.1f} GB)")
    print("\nNext: Process with rust_cache_builder to create dl_book_cache files.")


if __name__ == "__main__":
    main()
