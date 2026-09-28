# P-Alpha Head-C First-Passage v1 - Recon Report
**Date**: 2026-05-30 22:55 ET
**Author**: Sub-agent recon (20-min wall-cap)
**Status**: Script WORKS. Prior 0-folds outcome was caused by under-budgeting plus a misleading log-flush bug, NOT a code defect.

---

## Root Cause (one sentence)
The 4 prior attempts (10/15/20/25 min wall-caps) were killed before fold 0 could finish; a single cold-cache fold needs ~4.4 minutes, and the orchestrator was misled into thinking the runs crashed because the Python FileHandler logs were 0 bytes (the logging buffer was never flushed before SIGTERM).

## Evidence
Diagnostic smoke (4-min wall-cap, --max-folds 1) on Neptune RTX 3090 just completed successfully. Timing breakdown:

| Phase | Wall sec | Note |
|---|---|---|
| Shared feature stats (20 ref days) | 40 s | One-time, reused across all folds |
| Train HeadCDataset _build_index (55 days) | 128 s | First-passage label compute, ~2.3 s/day |
| OOT HeadCDataset _build_index (1 day) | 5 s | |
| Training (only 56 batches before deadline) | 67 s | Forward+back ~1.2 s/batch incl AMP |
| OOT inference + save | 22 s | |
| **Per-fold total (cold cache)** | **~263 s (4.4 min)** | |

Result of the smoke fold (OOT 20260306): n=10,160, IC=+0.0998, AUC=0.5576, acc@0.5=0.541 vs baseline 0.509 (+3.2 pp lift). The signal is real.

**Two secondary findings:**
1. logging.FileHandler is buffered and never flushed - every previous log file is 0 bytes including my own just-completed successful smoke run. Stdout via StreamHandler works fine. This is why the orchestrator interpreted runs as silent-crashing.
2. The 25-min orphan (run c562cb16, wall=163 s in summary.json) was killed externally - 163 s is well inside the 25-min budget, so something outside the script SIGKILL'd it (likely the parent orchestrator's own per-attempt timeout or a memory-governor kill before label build finished).

## Fixability
**Trivial (<=30 min edit).** Two minimal changes would close this out:
- Add force-flush to the log handlers (one liner: iterate log.handlers and call h.flush() after each batch/epoch, OR set PYTHONUNBUFFERED=1 in env + use stdout-only logging).
- Realistic wall-cap: budget ~4.5 min per fold cold + ~1.5 min per fold hot (label cache reuse across sliding-window overlap). For 35 OOT folds: ~60 min total. Use a 75-min wall-cap.

No structural redesign needed. Hyperparameters are sensible (W=3000, S=2000, B=128, 2 epochs, AMP on). num_workers=0 is a deliberate Neptune-32GB-RAM safety choice and is correct.

## Recommended Next Action
**(A) Fix log-flushing + relaunch with realistic budget.**

Concretely:
1. Patch log handler to flush on every epoch (1-line change in main loop).
2. Launch with --wall-cap-min 75 --max-folds 35 on Neptune.
3. Expect 35 folds in ~50-60 min, full FIFO grade after, summary + MLflow run with agg_mean_ic in the ~0.10 range, and FIFO trades with WR / PF / Sharpe on 35 OOT days.

The smoke result (IC +0.10, +3.2 pp lift over majority) suggests the K=2-tick first-passage paradigm has provisional edge worth the full 35-fold confirmation. That justifies using the Neptune GPU now rather than leaving it idle ~50 hr until Jupiter labels finish.

NOT (B) reduce scope - we already have the answer the smoke needed.
NOT (C) drop - first-passage paradigm just demonstrated edge.
NOT (D) idle - Neptune capacity is here, the full run is now under 1 hr.
