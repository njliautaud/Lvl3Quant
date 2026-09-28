#!/usr/bin/env python3
"""
HC #363 first deliverable: v3.3 per-head per-band per-horizon TICK-NATIVE
execution metrics dashboard.

THIN WRAPPER — imports the v3.2 dashboard module, monkey-patches PRED_NPZ +
OUT_DIR + LOG_PATH globals to point at the v3.3 NPZ. ZERO modifications to
v32_per_head_tick_dashboard.py. Pure analysis tooling per HC #307D.

Why a wrapper: v3.3's NPZ uses the same ALL_HEAD_NAMES (32 heads) + same
evaluate_v32 output structure, because v3.3 imports CNNMambaV32 +
ALL_HEAD_NAMES + evaluate_v32 from train_cnn_mamba_v3_2 as a library. So the
dashboard methodology is identical; only the input path differs.

Outputs land under:
  output/v3_3_full_execution_analysis_20260514/per_head_dashboard/

Usage:
  python3 v33_run_per_head_dashboard.py
"""
from pathlib import Path
import sys

THIS_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(THIS_DIR))

import v32_per_head_tick_dashboard as dash  # type: ignore  # noqa: E402

# Patch globals to point at v3.3 inputs/outputs
dash.PRED_NPZ = Path(
    "/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_predictions.npz"
)
dash.OUT_DIR = Path(
    "/home/jupiter/Lvl3Quant/output/v3_3_full_execution_analysis_20260514/per_head_dashboard"
)
dash.OUT_DIR.mkdir(parents=True, exist_ok=True)
dash.LOG_PATH = dash.OUT_DIR / "build.log"

print(f"[v33_run_per_head_dashboard] PRED_NPZ = {dash.PRED_NPZ}", flush=True)
print(f"[v33_run_per_head_dashboard] OUT_DIR  = {dash.OUT_DIR}", flush=True)

if __name__ == "__main__":
    sys.exit(dash.main())
