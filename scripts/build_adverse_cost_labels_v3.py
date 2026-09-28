#!/usr/bin/env python3
"""
Build realized adverse-cost labels per signal event for the v3 adverse-cost head.

Honest definition (HC #498 R3, HC #420 authorized):
  For each signal event at (event_id, ts_ns) with side ∈ {+1=buy/long, -1=sell/short},
  the realized adverse-cost-in-ticks at horizon h is:

      adverse_h = max(0, -side * labels_h[event_id])

  where labels_h is the precomputed signed signal price excursion at horizon h in
  tick units (from data/processed/mbo_events_smart_v3/*_mbo_events.npz).

  Interpretation: how many ticks the market moved AGAINST a passive long/short
  positioned at the touch at time t, measured at horizon h. NaN events skipped.

This is NOT a closed-form function of the predictor features used by v3 head
(queue depth, OFI, microprice imbalance, etc) — it is realized future price.
So no self-distillation.

Output: output/adverse_cost_labels_v3/labels_YYYYMMDD.parquet with columns:
  event_id, ts_ns, side, adverse_1s, adverse_5s, adverse_10s
"""
from __future__ import annotations
import sys, glob, os
from pathlib import Path
import numpy as np
import pandas as pd

LVL3_ROOT = Path('/home/jupiter/Lvl3Quant')
MBO_DIR   = LVL3_ROOT / 'data' / 'processed' / 'mbo_events_smart_v3'
LBL_DIR   = LVL3_ROOT / 'output' / 'mbo_walker_labels'
OUT_DIR   = LVL3_ROOT / 'output' / 'adverse_cost_labels_v3'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_for_date(date_str: str) -> dict:
    lbl_path = LBL_DIR / f'labels_{date_str}.parquet'
    mbo_path = MBO_DIR / f'{date_str}_mbo_events.npz'
    if not lbl_path.exists() or not mbo_path.exists():
        return {'date': date_str, 'status': 'missing'}
    lbl = pd.read_parquet(lbl_path, columns=['event_id', 'ts_ns', 'side'])
    with np.load(mbo_path, allow_pickle=False) as src:
        l1 = src['labels_1s'].astype(np.float32)
        l5 = src['labels_5s'].astype(np.float32)
        l10 = src['labels_10s'].astype(np.float32)
    eid = lbl['event_id'].to_numpy(dtype=np.int64)
    n_max = len(l1)
    if eid.max() >= n_max:
        bad = (eid >= n_max).sum()
        eid = np.clip(eid, 0, n_max - 1)
    side = lbl['side'].to_numpy(dtype=np.float32)
    # labels_h is signal-price excursion in ticks. side=+1 long => loses when labels<0.
    # adverse cost in ticks = max(0, -side * labels_h)
    s1 = l1[eid]; s5 = l5[eid]; s10 = l10[eid]
    a1 = np.maximum(0.0, -side * s1)
    a5 = np.maximum(0.0, -side * s5)
    a10 = np.maximum(0.0, -side * s10)
    # Mask NaNs
    valid = np.isfinite(s1) & np.isfinite(s5) & np.isfinite(s10)
    out = pd.DataFrame({
        'event_id': lbl['event_id'].to_numpy(),
        'ts_ns':    lbl['ts_ns'].to_numpy(),
        'side':     lbl['side'].to_numpy(),
        'adverse_1s':  a1,
        'adverse_5s':  a5,
        'adverse_10s': a10,
        'valid': valid.astype(np.int8),
    })
    out.to_parquet(OUT_DIR / f'labels_{date_str}.parquet', index=False)
    return {
        'date': date_str, 'status': 'ok', 'n': int(len(out)),
        'n_valid': int(valid.sum()),
        'mean_adv_1s': float(np.nanmean(a1[valid])),
        'mean_adv_5s': float(np.nanmean(a5[valid])),
        'mean_adv_10s': float(np.nanmean(a10[valid])),
        'p99_adv_5s': float(np.nanpercentile(a5[valid], 99)),
    }


def main():
    files = sorted(glob.glob(str(LBL_DIR / 'labels_*.parquet')))
    dates = [Path(f).stem.replace('labels_', '') for f in files]
    print(f'Building adverse-cost labels for {len(dates)} dates -> {OUT_DIR}')
    rows = []
    for d in dates:
        r = build_for_date(d)
        rows.append(r)
        print(f'  {d}: {r}')
    pd.DataFrame(rows).to_csv(OUT_DIR / '_summary.csv', index=False)
    ok = sum(1 for r in rows if r.get('status') == 'ok')
    print(f'\nDone. {ok}/{len(dates)} dates built.')


if __name__ == '__main__':
    main()
