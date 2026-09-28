#!/usr/bin/env python3
"""
backfill_vol_lgbm_v3_oot32.py

HC #538 R3 backfill — extend vol_lgbm_v3 per-day predictions to the full
32-day OOT window so confluence/filter experiments evaluate on a full-coverage
trade tape (no more pass-through gates on missing days).

Reuses fold_train() from alpha_discovery/deep_models/train_vol_lgbm_v3.py
unchanged. Only difference: drives it over OOT dates not in the original
hard-coded TEST_DATES list. Output schema matches existing
vol_v3_{date}_predictions.npz exactly so confluence_filter_v1 ingests it
without any code change.

Run:
  python3 scripts/backfill_vol_lgbm_v3_oot32.py

CPU-only. Writes to output/vol_lgbm_v3/.
"""
from __future__ import annotations

import json
import os
import sys
import time
import types
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(ROOT))

# Use the existing MBO data dir (the script default points at a non-existent
# legacy path; override via env var as the script supports).
os.environ.setdefault(
    "VOL_LGBM_DATA_DIR", str(ROOT / "data/processed/mbo_events_smart_v3")
)

# Import after env var so the module picks up the right DATA_DIR.
from alpha_discovery.deep_models import train_vol_lgbm_v3 as tv  # noqa: E402

OOT_DIR = ROOT / "output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate"
OUT_DIR = ROOT / "output/vol_lgbm_v3"
DATA_DIR = Path(os.environ["VOL_LGBM_DATA_DIR"])

# Force module's OUT_DIR to canonical location (it's a module-level global
# the original script mutates inside main(); we set it directly here).
tv.OUT_DIR = OUT_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)


def discover_oot_dates() -> list[str]:
    return sorted(p.stem.replace("oot_", "") for p in OOT_DIR.glob("oot_*.npz"))


def existing_pred_dates() -> set[str]:
    return {
        p.stem.split("_")[2] for p in OUT_DIR.glob("vol_v3_*_predictions.npz")
    }


def main() -> int:
    t0 = time.time()
    print(f"[backfill_vol] start {datetime.now().isoformat()}", flush=True)
    print(f"  DATA_DIR : {DATA_DIR}", flush=True)
    print(f"  OUT_DIR  : {OUT_DIR}", flush=True)

    if not DATA_DIR.exists():
        print(f"[fatal] DATA_DIR missing: {DATA_DIR}", file=sys.stderr)
        return 2

    oot_dates = discover_oot_dates()
    done = existing_pred_dates()
    missing = [d for d in oot_dates if d not in done]
    print(
        f"[plan] OOT total: {len(oot_dates)}; already done: {len(done)}; "
        f"to backfill: {len(missing)}",
        flush=True,
    )
    if not missing:
        print("[plan] nothing to do.", flush=True)
        return 0
    print(f"[plan] missing: {missing}", flush=True)

    # Available training files (sorted by date stem)
    files = sorted(DATA_DIR.glob("*_mbo_events.npz"))
    print(
        f"[plan] available MBO days: {len(files)} "
        f"({files[0].stem[:8]}..{files[-1].stem[:8]})",
        flush=True,
    )

    # Minimal Namespace-like args mirroring train_vol_lgbm_v3 defaults
    args = types.SimpleNamespace(
        folds="custom",
        n_estimators=600,
        lr=0.02,
        half_life=tv.HALF_LIFE_DAYS,
        n_jobs=8,
        output_dir=str(OUT_DIR),
    )
    # train_vol_lgbm_v3.fold_train reads args.half_life / args.n_estimators /
    # args.lr / args.n_jobs — that's it.

    results = []
    for td in missing:
        td_t0 = time.time()
        try:
            res = tv.fold_train(td, files, args, mlflow_run=None)
        except Exception as e:
            print(f"[ERR] fold {td} failed: {e}", flush=True)
            results.append({"test_date": td, "status": "error", "err": str(e)})
            continue
        td_dt = time.time() - td_t0
        res["wall_time_sec"] = td_dt
        results.append(res)
        print(f"[ok] {td}  status={res.get('status')}  dt={td_dt:.1f}s", flush=True)

    summary = {
        "completed_at": datetime.now().isoformat(),
        "n_attempted": len(missing),
        "n_ok": int(sum(r.get("status") == "ok" for r in results)),
        "wall_time_min": (time.time() - t0) / 60.0,
        "folds": results,
    }
    out_summary = (
        OUT_DIR
        / f"vol_v3_backfill_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(out_summary, "w") as fh:
        def _clean(o):
            if isinstance(o, dict):
                return {k: _clean(v) for k, v in o.items()}
            if isinstance(o, list):
                return [_clean(v) for v in o]
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, (np.integer,)):
                return int(o)
            return o

        json.dump(_clean(summary), fh, indent=2)
    print(f"[done] summary: {out_summary}  total={summary['wall_time_min']:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
