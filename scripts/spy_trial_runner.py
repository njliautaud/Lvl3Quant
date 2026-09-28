#!/usr/bin/env python3
"""spy_trial_runner.py — SPY MBO trial walk-forward (HC #527 R1).

PURPOSE
=======
Run the SPY Mar 1–13 2026 trial in a way that respects every binding HC:
  * HC #0 (sliding WF — adapted to trial-length: 2d train / 1d OOT, NOT 60/1)
  * HC #428 R1 (regime-stratified reporting + gap gate)
  * HC #428 R2 (MFE-within-horizon caps on TP/SL/hold)
  * HC #344 (day-conc cap ≤ 0.70)
  * HC #515 R6 (MFE/MAE on RAW price path, not normalized features)

PROTOCOL
========
9 trading days available (Mar 2,3,4,5,6,9,10,11,12,13 2025).
Modes (run in order, report after each):

  1. zero_shot   — apply existing ES-trained CNN-Mamba v2 (or PatchTST) to SPY,
                   with feature stats refit on first 3 SPY days; measure raw IC.
                   No training. Output: IC at 1s/5s/10s/30s on the remaining 6 days.

  2. trial_wf    — 2d-train / 1d-OOT SLIDING WF over the 9 days.
                   Folds: (d1,d2 → d3), (d2,d3 → d4), ..., (d7,d8 → d9) = 7 folds.
                   Concat IC + per-day Sharpe/PF/WR + MFE/MAE distribution.

  3. regime_test — for whichever model wins, stratify the 9-day OOT predictions
                   by SPY close-to-close green/red/flat. Compute Sharpe per regime.
                   Apply the HC #428 R1 gap gate (reject if
                   |Sharpe_green - Sharpe_red| / max(|Sg|,|Sr|) > 0.50).

  4. mfe_cap    — verify TP candidates ≤ p90 of realized MFE within horizon h.
                   Reject any TP/hold combo that violates HC #428 R2.

OUTPUTS
=======
All under output/spy_trial/:
  norm_stats.npz                         z-score stats refit on first 3 SPY days
  zero_shot_ic.json                      per-horizon IC, zero-shot
  fold_NN/preds.npz                      per-fold OOT predictions
  fold_NN/metrics.json                   per-fold metrics
  concat_oot.npz                         all OOT predictions concatenated
  regime_stratified.json                 green/red/flat slicing + gap-gate result
  mfe_cap_report.json                    realized MFE distribution + cap recommendations
  SUMMARY.md                             human-readable decision report

USAGE (when SPY MBO data has landed under data/processed/spy_mbo_events/)
========================================================================
  python scripts/spy_trial_runner.py --mode zero_shot
  python scripts/spy_trial_runner.py --mode trial_wf
  python scripts/spy_trial_runner.py --mode regime_test
  python scripts/spy_trial_runner.py --mode mfe_cap
  python scripts/spy_trial_runner.py --mode all       # default

THIS SCRIPT IS SKELETON-COMPLETE BUT DATA-BLOCKED.
Each function has a TODO marking the exact wire-up needed when data lands.
NO TRAINING IS LAUNCHED. Per HC #527 + HC #527 R4: alignment & research only.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

LVL3 = Path("/home/jupiter/Lvl3Quant")
if str(LVL3) not in sys.path:
    sys.path.insert(0, str(LVL3))

# Project imports — verified to exist:
import cost_constants_spy as costs                              # READY
# from feeds.schema_adapter import SchemaAdapter                # READY (not needed at runtime; ingest already done)
# from feeds.spy_walkforward_smoke import check_file            # READY

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SPY_DATA_DIR = LVL3 / "data" / "processed" / "spy_mbo_events"
SPY_OUT_DIR  = LVL3 / "output" / "spy_trial"
NORM_STATS_PATH = SPY_OUT_DIR / "norm_stats.npz"

# Trial date list — Mar 3..13 2025, weekdays only, 9 RTH days
TRIAL_DATES = [
    "20250303", "20250304", "20250305", "20250306", "20250307",
    "20250310", "20250311", "20250312", "20250313",
]
ZERO_SHOT_FIT_DATES = TRIAL_DATES[:3]      # first 3 days = z-score fit only
ZERO_SHOT_EVAL_DATES = TRIAL_DATES[3:]     # remaining 6 = zero-shot IC measurement
WF_TRAIN_LEN = 2                           # 2-day train (trial-only; HC #0 still binds prod)
WF_OOT_LEN = 1                             # 1-day OOT
HORIZONS_SEC = [1, 5, 10, 30]

# HC #428 R1 gate
REGIME_GAP_THRESHOLD = 0.50
# HC #344
DAY_CONC_CAP = 0.70
# HC #428 R2
MFE_PERCENTILE_CAP = 90.0
HOLD_HORIZON_MULTIPLIER = 1.5
CANCEL_HORIZON_MULTIPLIER = 1.0

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
SPY_OUT_DIR.mkdir(parents=True, exist_ok=True)
log = logging.getLogger("spy_trial")
log.setLevel(logging.INFO)
for _h in (logging.FileHandler(SPY_OUT_DIR / "trial.log"),
           logging.StreamHandler(sys.stdout)):
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(_h)


# ===========================================================================
# Helpers
# ===========================================================================
@dataclass
class FoldResult:
    fold: int
    train_dates: list
    oot_date: str
    n_preds: int = 0
    ic_1s: Optional[float] = None
    ic_5s: Optional[float] = None
    ic_10s: Optional[float] = None
    ic_30s: Optional[float] = None
    sharpe: Optional[float] = None
    win_rate: Optional[float] = None
    issues: list = field(default_factory=list)


def load_spy_day(date_str: str) -> dict:
    """Load one SPY MBO NPZ. Returns dict with events, timestamps, mid path."""
    path = SPY_DATA_DIR / f"{date_str}_mbo_events.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"SPY NPZ for {date_str} not found at {path}. "
            f"Run feeds/spy_databento_trial.py --mode historical for that date first."
        )
    d = np.load(path, allow_pickle=True)
    events = d["events"].astype(np.float32)
    ts = d["timestamps"].astype(np.int64)
    # Mid-price reconstruction at SPY tick size (HC #515 R6 — raw, not normalized)
    # events[:, 3] = price_rel_ticks. Reconstruct ABSOLUTE mid via running BBO
    # is not available here (BBO was tracked in adapter but not persisted).
    # Instead use price_rel_ticks * tick_size as a centered series; absolute
    # level comes from the seed mid in metadata when needed for $-PnL.
    return {
        "events": events,
        "timestamps": ts,
        "n": int(events.shape[0]),
        "path": str(path),
    }


# ===========================================================================
# MODE 1 — refit z-score stats on first 3 SPY days
# ===========================================================================
def refit_norm_stats() -> Path:
    """Compute per-column mean/std on cols [0,3,4,5] over fit-dates.

    Cols 1 (event_type) and 2 (side) are categorical — keep raw.
    """
    log.info("REFIT z-score stats on %s", ZERO_SHOT_FIT_DATES)
    chunks = []
    for d in ZERO_SHOT_FIT_DATES:
        ev = load_spy_day(d)["events"]
        chunks.append(ev)
    all_ev = np.concatenate(chunks, axis=0)
    log.info("  total events for stats fit: %d", all_ev.shape[0])

    means = all_ev.mean(axis=0).astype(np.float32)
    stds = all_ev.std(axis=0).astype(np.float32)
    # Avoid div-by-zero on categorical cols (we'll overwrite std=1 there)
    stds[1] = 1.0  # event_type_id
    stds[2] = 1.0  # side_id
    means[1] = 0.0
    means[2] = 0.0
    stds[stds < 1e-6] = 1.0

    np.savez(NORM_STATS_PATH, means=means, stds=stds,
             fit_dates=np.array(ZERO_SHOT_FIT_DATES))
    log.info("  wrote %s  means=%s  stds=%s",
             NORM_STATS_PATH, means.tolist(), stds.tolist())
    return NORM_STATS_PATH


# ===========================================================================
# MODE 2 — zero-shot inference: ES-trained model on SPY (norm refit only)
# ===========================================================================
def run_zero_shot():
    """Apply existing CNN-Mamba v2 / PatchTST checkpoints to SPY with refit z-scores.

    TODO when data lands:
      1. Determine input dim of available checkpoints (see PIPELINE_READINESS §5).
         If checkpoints expect smart_v3 (25 cols): derive smart_v3 features
         on SPY via scripts/compute_smart_v3_features.py (verify name on disk).
         If checkpoints expect canonical 6 cols: use events directly.
      2. For each eval date in ZERO_SHOT_EVAL_DATES:
           - load_spy_day(date)
           - apply norm: (events - means) / stds
           - window into (B, L=1000, F) with stride=250
           - model.eval(); preds = model(x).cpu().numpy()
           - align preds to forward-mid returns at horizons [1s, 5s, 10s, 30s]
           - rank-correlate -> IC per horizon
      3. Aggregate concat IC across all eval dates.

    Output: zero_shot_ic.json with {per_date_ic: {...}, concat_ic: {...}}
    """
    if not NORM_STATS_PATH.exists():
        refit_norm_stats()
    log.info("ZERO-SHOT: SKELETON — wiring deferred until SPY data lands")
    # TODO: implement once checkpoint input-dim audit is done.
    out = {
        "status": "stub",
        "fit_dates": ZERO_SHOT_FIT_DATES,
        "eval_dates": ZERO_SHOT_EVAL_DATES,
        "next_step": "wire model loader + windowing once SPY data on disk",
    }
    with open(SPY_OUT_DIR / "zero_shot_ic.json", "w") as f:
        json.dump(out, f, indent=2)
    log.info("Wrote zero_shot_ic.json (stub)")
    return out


# ===========================================================================
# MODE 3 — trial walk-forward: 2d-train / 1d-OOT, 7 folds
# ===========================================================================
def make_trial_folds():
    """Generate the 7 sliding folds over 9 days."""
    folds = []
    for i in range(len(TRIAL_DATES) - WF_TRAIN_LEN - WF_OOT_LEN + 1):
        train = TRIAL_DATES[i:i + WF_TRAIN_LEN]
        oot = TRIAL_DATES[i + WF_TRAIN_LEN]
        folds.append({"fold": i, "train": train, "oot": oot})
    return folds


def run_trial_wf():
    """Walk forward with 2d train / 1d OOT.

    TODO when data lands:
      1. For each fold:
           - load train days, build dataset, train CNN-Mamba v2 from scratch
             OR fine-tune from ES checkpoint (default: fine-tune, faster + uses prior)
           - infer on OOT day, save preds.npz
           - compute IC at 1s/5s/10s/30s, Sharpe (on raw returns ranked by pred),
             win-rate of long/short signals
      2. Concatenate OOT preds across folds -> concat_oot.npz
      3. Write per-fold metrics.json + aggregate

    NOTE: 2 training days is THIN for a 1.5M-param model. Expect over-fit.
    Mitigations: heavy dropout (0.3+), early stopping on a held-out 10% slice
    of train, learning-rate 5x smaller than ES regime. Document in fold metrics.
    """
    folds = make_trial_folds()
    log.info("TRIAL WF: %d folds planned", len(folds))
    for fld in folds:
        log.info("  fold %d  train=%s  oot=%s",
                 fld["fold"], fld["train"], fld["oot"])
    # TODO: wire training loop. Skeleton only.
    out = {"status": "stub", "folds": folds,
           "next_step": "wire model training + inference per fold"}
    with open(SPY_OUT_DIR / "trial_wf_plan.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


# ===========================================================================
# MODE 4 — regime stratification (HC #428 R1)
# ===========================================================================
def load_spy_daily_closes():
    """TODO: load SPY daily closes for TRIAL_DATES + 1 prior day for diff.

    Default source: download once into data/external/spy_daily_2026Q1.csv
    via yfinance ('SPY', start='2025-02-28', end='2025-03-14').
    """
    path = LVL3 / "data" / "external" / "spy_daily_2026Q1.csv"
    if path.exists():
        import csv
        out = {}
        with open(path) as f:
            for row in csv.DictReader(f):
                out[row["date"].replace("-", "")] = float(row["close"])
        return out
    log.warning("SPY daily closes not on disk at %s — regime stratification "
                "will be stubbed. Action: pull once via yfinance.", path)
    return {}


def classify_regime(date_str: str, closes: dict, flat_thresh_pct: float = 0.1) -> str:
    """Green if close > prior close * (1 + 0.1%), Red if <, Flat otherwise."""
    if not closes:
        return "unknown"
    idx = TRIAL_DATES.index(date_str)
    if idx == 0:
        return "unknown"  # need prior day
    prev = TRIAL_DATES[idx - 1]
    c, p = closes.get(date_str), closes.get(prev)
    if c is None or p is None:
        return "unknown"
    ret = (c - p) / p * 100
    if ret > flat_thresh_pct:
        return "green"
    if ret < -flat_thresh_pct:
        return "red"
    return "flat"


def regime_gate(sharpe_per_regime: dict) -> dict:
    """HC #428 R1: reject if |Sg - Sr| / max(|Sg|,|Sr|) > 0.50."""
    sg = sharpe_per_regime.get("green")
    sr = sharpe_per_regime.get("red")
    if sg is None or sr is None:
        return {"verdict": "insufficient_data", "gap": None}
    denom = max(abs(sg), abs(sr), 1e-9)
    gap = abs(sg - sr) / denom
    verdict = "PASS" if gap <= REGIME_GAP_THRESHOLD else "REJECT"
    return {"verdict": verdict, "gap": gap, "sharpe_green": sg,
            "sharpe_red": sr, "threshold": REGIME_GAP_THRESHOLD}


def run_regime_test():
    closes = load_spy_daily_closes()
    classifications = {d: classify_regime(d, closes) for d in TRIAL_DATES}
    log.info("REGIME classifications: %s", classifications)
    # TODO: load concat_oot.npz, compute per-day Sharpe, group by regime,
    # apply regime_gate.
    out = {"status": "stub", "classifications": classifications,
           "next_step": "wire after trial_wf produces concat_oot.npz"}
    with open(SPY_OUT_DIR / "regime_stratified.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


# ===========================================================================
# MODE 5 — MFE-within-horizon caps (HC #428 R2)
# ===========================================================================
def run_mfe_cap():
    """For each horizon h, compute p90 of realized MFE within h.

    Recommended TP cap = p90(MFE_h). Hold cap = 1.5 * h. Cancel cap = h.
    Reject any candidate config violating these.

    TODO when data lands:
      1. For each TRIAL_DATE, load events + reconstruct mid path
      2. For each prediction emit-point (every 250 events stride):
           - look ahead h seconds via searchsorted on timestamps
           - compute MFE = max(mid[t:t+h]) - mid[t]  (in ticks)
           - compute MAE = mid[t] - min(mid[t:t+h])  (in ticks)
      3. Aggregate per-horizon MFE distribution -> p90
      4. Emit mfe_cap_report.json with {h: {p50, p90, p99, mean,
         recommended_tp_ticks, recommended_hold_seconds, recommended_cancel_seconds}}
    """
    log.info("MFE-CAP: SKELETON — needs SPY data")
    out = {"status": "stub", "horizons_sec": HORIZONS_SEC,
           "percentile_cap": MFE_PERCENTILE_CAP,
           "hold_multiplier": HOLD_HORIZON_MULTIPLIER,
           "cancel_multiplier": CANCEL_HORIZON_MULTIPLIER,
           "next_step": "wire after events on disk"}
    with open(SPY_OUT_DIR / "mfe_cap_report.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


# ===========================================================================
# Orchestration
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description="SPY trial walk-forward runner")
    ap.add_argument("--mode", choices=["refit_stats", "zero_shot", "trial_wf",
                                       "regime_test", "mfe_cap", "all"],
                    default="all")
    args = ap.parse_args()

    # Sanity: confirm at least the first SPY NPZ exists before running anything
    have_data = any((SPY_DATA_DIR / f"{d}_mbo_events.npz").exists()
                    for d in TRIAL_DATES)
    if not have_data:
        log.warning("NO SPY DATA on disk for any TRIAL_DATE. Each mode will "
                    "stub out cleanly. Re-run after running "
                    "feeds/spy_databento_trial.py --mode historical for "
                    "Mar 3..13 2025.")

    if args.mode in ("refit_stats", "all") and have_data:
        refit_norm_stats()
    if args.mode in ("zero_shot", "all"):
        run_zero_shot()
    if args.mode in ("trial_wf", "all"):
        run_trial_wf()
    if args.mode in ("regime_test", "all"):
        run_regime_test()
    if args.mode in ("mfe_cap", "all"):
        run_mfe_cap()

    log.info("DONE. Outputs in %s", SPY_OUT_DIR)


if __name__ == "__main__":
    main()
