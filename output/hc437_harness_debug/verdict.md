# HC #437 — Verdict

**Date**: 2026-05-19 (Jupiter, ~3hr investigation)
**Status**: Two root-cause bugs identified. One patched. Second escalated (out of scope today).

## TL;DR

The HC #432 FIFO harness disagrees with HC #413 because of **two independent bugs**:

1. **NPZ source bug (PATCHED)** — `hc432_v2_baseline_runner.py` sourced predictions from
   the documented-corrupted inference directory `cnn_mamba_v2_all_oot/`
   (window_size=3000, IC_1s ≈ 0.01) instead of the canonical
   `cnn_mamba_v2_bulk_oot_v2/` (window_size=1000, IC_1s ≈ 0.19). One-line fix applied.

2. **Exit methodology gap (NOT PATCHED)** — even after fix 1, HC #432's
   FIFOReplayEngine evaluates exits intra-event with single TP/SL, while HC #413's
   backtester uses TP1/TP2/SL brackets evaluated at 1s/5s/10s **horizon checkpoints**
   against pre-computed labels. The FIFO engine is strictly more conservative on the
   exit path because it triggers SL on transient drawdowns that HC #413 never sees.
   Result: even with the corrected NPZ source, the patched runner produces
   -0.27 tk/fill (vs HC #413's +0.27 tk/fill). Both numbers are wrong in opposite
   directions; the truth is somewhere in between.

## Evidence

### Bug 1 — wrong NPZ source

47-day signal-only IC comparison (no fill sim, just realized-label mean of top-0.5%
short signals):

| NPZ source | mean IC_1s | N-weighted top-0.5% short realized pnl (tk) |
|---|---:|---:|
| `cnn_mamba_v2_all_oot/` (window=3000) | +0.013 | **-0.157** |
| `cnn_mamba_v2_bulk_oot_v2/` (window=1000) | +0.163 | **+0.730** |

The corrupted-window NPZ has near-zero IC and produces realized -0.16 tk/fill at the
signal level. HC #432's reported -0.17 tk/fill is the FIFO engine faithfully
executing a no-edge signal. The fix is a one-line directory swap (see
`fix_applied.diff`).

Background: this corruption was discovered May 3 and documented in
`scripts/v3_3_research/regenerate_v2_bulk_oot.py` (which CREATED the corrected
`bulk_oot_v2/` directory). The HC #432 runner was written later and accidentally
pointed at the older, still-corrupted `all_oot/` dir.

### Bug 2 — exit-methodology gap

3-date FIFO replay with patched NPZ source:

| run config | n_fills | net tk/fill | Sharpe | PF | WR | TP_hits | SL_hits |
|---|---:|---:|---:|---:|---:|---:|---:|
| `tp=1.0 sl=0.5` (HC #432 baseline) | 160 | -0.27 | -4.63 | 0.48 | 41% | 64 | 94 |
| `tp=0.478 sl=0.569` (HC #413 TP1/SL) | 160 | -0.52 | -12.7 | 0.07 | 41% | 64 | 94 |

With raw realized signal pnl of +1.45 tk/short, the FIFO engine throws away ~1.7 tk
of edge to intra-event SL hits. HC #413 doesn't see these — it only checks `label_h`
at h=1s/5s/10s endpoints. This is a fundamental methodology delta, not a tunable
parameter.

## Implications for HC #432's v3.4.x verdicts

All HC #432 v3.4.x config rejections from today should be considered **suspect** but
for a different reason than the v2 baseline:

- The v3.4.2 configs use the Neptune-supplied per-date NPZs concatenated into
  `fold_00_ep1_oot_inference_47day_hc432.npz`. Those NPZs were freshly generated
  on Neptune for HC #432; they are NOT subject to bug 1 (NPZ-source mismatch).
- They ARE subject to bug 2: the FIFOReplayEngine's intra-event SL evaluation is
  strictly more conservative than HC #413's horizon-checkpoint methodology. Any
  config that "passed" under HC #413's earlier methodology might fail HC #432 not
  because the model is bad, but because the new harness is harder.

**HC #437 R1 (forbid FAIL verdicts until v2 sanity reproduces) is NOT satisfied
by today's work.** Patching only the NPZ source moves the number from
-0.17 → -0.27 (still negative). Full reproduction would require porting HC #413's
TP1/TP2/SL-horizon-checkpoint exit logic into the FIFOReplayEngine as a new
`order_management` mode.

## What was done

| Action | Status |
|---|---|
| Located HC #413 ground-truth output | DONE — `output/hc417_hc413_v2native_mfe/` |
| Located HC #432 harness | DONE — `scripts/v3_4_research/hc432_*` |
| Identified prediction-NPZ-source mismatch as primary cause | DONE |
| Applied one-line patch to `hc432_v2_baseline_runner.py:V2_OOT_DIR` | DONE |
| 3-date FIFO repro under patched source | DONE (`v2_short_1s_top0.5_baseline_HC437_REPRO_3day_*`) |
| Identified exit-methodology gap as secondary cause | DONE |
| Port HC #413 bracket exits into FIFOReplayEngine | **NOT DONE** (out of 3hr scope) |
| 47-day full repro under fully-patched harness | NOT DONE (blocked on previous item) |

## Recommendation to user

1. **Mark all HC #432 v3.4.x FAIL verdicts from today as suspect.** They were
   generated under a strictly-more-conservative-than-HC #413 exit model.
2. **Schedule a follow-up engineering task** to port HC #413's TP1/TP2/SL bracket
   into `FIFOReplayEngine.simulate(order_management='hc413_bracket')`. After that
   change, re-run the v2 sanity baseline — if it produces +0.20 to +0.30 tk/fill
   then re-run the v3.4.x configs.
3. **The fix-applied today** restores correctness of the IC of the v2 baseline
   harness input, so any future use of `hc432_v2_baseline_runner.py` is at least
   evaluating the correct model predictions.

## Files

- `compare.md` — candidate-causes diff table
- `repro_diff.md` — single-date prediction comparison evidence
- `fix_applied.diff` — the one-line NPZ-source patch
- `sanity_repro_47day.md` — Phase 3 reproduction results
- `per_date_ic_compare.json` — 47-date IC and realized-label comparison data
- `verdict.md` — this file

