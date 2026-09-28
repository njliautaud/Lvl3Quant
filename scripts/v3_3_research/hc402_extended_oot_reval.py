"""
HC #402 EXTENDED-OOT RE-VALIDATION

Same as hc402_24h_unlock.py but POINTS AT THE 15-DAY EXTENDED OOT NPZ
that already exists from May 14 inference run.

  Predictions: output/v3_3_extended_oot_20260514/extended_oot_predictions.npz
  Dates: 20260301 .. 20260319 (15 firing days)
  Samples: 673,184

This bypasses the need to wait for v3.4.2 to finish + dispatch a new inference run.
If trial 224 (or any of the top 50) passes HC #344 strict day_conc <= 0.20 on
this 15-day OOT, we have a STRICT PRODUCTION-READY config TONIGHT.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(PROJ))

# Monkey-patch the path before importing
EXTENDED_PREDS = PROJ / "output" / "v3_3_extended_oot_20260514" / "extended_oot_predictions.npz"
LABELS_DIR = PROJ / "data" / "processed" / "mbo_events_smart_v3_fifo_labels"

# Import the re-val machinery from the original script
import scripts.v3_3_research.hc402_24h_unlock as hc402

# Override the predictions path
hc402.DEFAULT_PREDS_PATH = EXTENDED_PREDS
hc402.DEFAULT_LABELS_DIR = LABELS_DIR

# Override OUT_DIR
OUT_DIR = PROJ / "output" / f"hc402_EXTENDED_OOT_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUT_DIR.mkdir(parents=True, exist_ok=True)
hc402.OUT_DIR = OUT_DIR

def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")

if __name__ == "__main__":
    print(f"[{_now()}] HC #402 EXTENDED-OOT (15-day) RE-VALIDATION")
    print(f"[{_now()}] Predictions: {EXTENDED_PREDS}")
    print(f"[{_now()}] OUT_DIR: {OUT_DIR}")
    print(f"[{_now()}] Canonical commission: {hc402.CANONICAL_COMMISSION}")
    print(f"[{_now()}] Strict day_conc gate: {hc402.STRICT_DAY_CONC_GATE} (HC #344)")
    print()

    t0 = time.time()
    df_reval = hc402.run_topK_reval()
    t1 = time.time()
    print(f"\n[{_now()}] (A) Done in {(t1-t0)/60:.1f} min\n")

    df_econ = hc402.run_mfe_mae_economics()
    t2 = time.time()
    print(f"\n[{_now()}] (B) Done in {(t2-t1)/60:.1f} min\n")

    summary = {
        "output_dir": str(OUT_DIR),
        "predictions_path": str(EXTENDED_PREDS),
        "n_oot_days_used": 15,
        "n_trials_revalidated": len(df_reval),
        "n_strict_passers": int(df_reval["hc344_strict_pass"].sum()) if len(df_reval) else 0,
        "n_relaxed_passers": int(df_reval["hc344_relaxed_pass"].sum()) if len(df_reval) else 0,
        "n_econ_rows": len(df_econ),
        "duration_minutes": (t2 - t0) / 60,
        "canonical_commission": hc402.CANONICAL_COMMISSION,
        "strict_day_conc_gate": hc402.STRICT_DAY_CONC_GATE,
        "hc_refs": ["HC #337", "HC #344", "HC #392", "HC #397B", "HC #402"],
    }
    (OUT_DIR / "SUMMARY.json").write_text(json.dumps(summary, indent=2))
    print(f"[{_now()}] === EXTENDED-OOT ALL DONE ===")
    print(json.dumps(summary, indent=2))
