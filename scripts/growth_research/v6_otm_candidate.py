#!/usr/bin/env python3
"""
V6 OTM Candidate — Tests v5 params + 2% OTM moneyness
=======================================================

Moneyness cross-validation (exp 173) showed 2% OTM has Sharpe 2.44 vs ATM 2.04.
This script tests the full v6 candidate: K=2, weekly (5d), DTE=21, 2% OTM.

Also tests DTE=28 and DTE=30 with 2% OTM (DTE interaction with OTM may differ).
And a production safety variant with 1% OTM (less aggressive).

6 Variants:
  v5_atm_baseline:  K=2, 5d, DTE=21, ATM (current production)
  v6a_otm2_dte21:   K=2, 5d, DTE=21, 2% OTM (main candidate)
  v6b_otm1_dte21:   K=2, 5d, DTE=21, 1% OTM (conservative)
  v6c_otm2_dte28:   K=2, 5d, DTE=28, 2% OTM (longer DTE interaction)
  v6d_otm3_dte21:   K=2, 5d, DTE=21, 3% OTM (aggressive)
  v6e_otm2_k1:      K=1, 5d, DTE=21, 2% OTM (concentrated + OTM)
"""

import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

_builtin_print = print

def fprint(*args, **kwargs):
    _builtin_print(*args, **kwargs)
    sys.stdout.flush()

sys.path.insert(0, "/home/nick/Lvl3Quant")
from research.tools.options_pricer import (
    price_bull_call_spread,
    price_bear_put_spread,
    estimate_iv,
    compute_atr,
    COMMISSION_RT_SPREAD,
    DEFAULT_HAIRCUT,
)
from research.tools.adversarial_validator import validate_trades

BASE = Path("/home/nick/Lvl3Quant")
OUTPUT_DIR = BASE / "output" / "v6_otm_candidate"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX", "^VIX3M", "TLT", "SHY", "HYG", "GLD"]
CAP = 645.0
SPREAD_PCT = 3.0
REGIME_BULL_THRESHOLD = 0.4

REGIME_FILE = BASE / "output" / "regime_detector_v1" / "regime_predictions_v1.npz"
WF_TRAIN_PERIODS = 12

MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "v6_otm_candidate"

MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen(MLFLOW_URI, timeout=3)
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    MLFLOW_OK = True
    fprint(f"MLflow connected: {MLFLOW_URI}")
except Exception:
    fprint("MLflow unavailable")

VARIANTS = [
    {"name": "v5_atm_baseline",  "top_k": 2, "rebal_days": 5,  "dte": 21, "moneyness_pct": 0.0,  "desc": "v5 production baseline (ATM)"},
    {"name": "v6a_otm2_dte21",  "top_k": 2, "rebal_days": 5,  "dte": 21, "moneyness_pct": 2.0,  "desc": "Main v6 candidate (2% OTM)"},
    {"name": "v6b_otm1_dte21",  "top_k": 2, "rebal_days": 5,  "dte": 21, "moneyness_pct": 1.0,  "desc": "Conservative OTM (1%)"},
    {"name": "v6c_otm2_dte28",  "top_k": 2, "rebal_days": 5,  "dte": 28, "moneyness_pct": 2.0,  "desc": "2% OTM + longer DTE"},
    {"name": "v6d_otm3_dte21",  "top_k": 2, "rebal_days": 5,  "dte": 21, "moneyness_pct": 3.0,  "desc": "Aggressive OTM (3%)"},
    {"name": "v6e_otm2_k1",     "top_k": 1, "rebal_days": 5,  "dte": 21, "moneyness_pct": 2.0,  "desc": "2% OTM + concentrated K=1"},
]

# ─── Import the rest from v5 crossval script ───
# (Re-use all the data download, regime, feature, LGBM, and trading logic)
# This is a simplified version that imports the core functions

import importlib.util
spec = importlib.util.spec_from_file_location("v5_crossval", 
    str(BASE / "scripts" / "growth_research" / "moneyness_crossval_v1.py"))
v5mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v5mod)

# Override the variants and experiment name
v5mod.VARIANTS = VARIANTS
v5mod.EXPERIMENT_NAME = EXPERIMENT_NAME
v5mod.OUTPUT_DIR = OUTPUT_DIR

# Run the main function
if hasattr(v5mod, 'main'):
    v5mod.main()
else:
    # If no main(), the module ran on import
    fprint("Module executed on import")
