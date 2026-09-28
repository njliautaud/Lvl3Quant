#!/usr/bin/env python3
"""
Process raw .dbn.zst MBO files into _mbo_events.npz format for training.

Scans data/raw/mbo/ for unprocessed dates and converts each to the 6-feature
event format with forward-looking mid-price labels at 1s/5s/10s/30s horizons.

Usage:
    python process_missing_mbo.py                    # process all missing
    python process_missing_mbo.py --workers 4        # parallel
    python process_missing_mbo.py --force 20260313   # reprocess specific date
"""