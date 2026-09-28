# v3.5 Multi-Head Continuation/Pressure CNN-Mamba — Scaffold + Smoke Test Report
HC #497 — 2026-05-30

## Summary
All three deliverables built and smoke-tested PASS. Ready for Monday full-multifold launch.

## Deliverables (built)
1. `/home/jupiter/Lvl3Quant/scripts/build_continuation_labels.py` (+ Neptune copy)
2. `/home/jupiter/Lvl3Quant/scripts/train_cnn_mamba_v35_multihead.py` (+ Neptune copy at `/home/nick/Lvl3Quant/alpha_discovery/deep_models/`)
3. `/home/jupiter/Lvl3Quant/scripts/fifo_grade_v35_variable_horizon.py`
4. `/tmp/launch_v35_full_multifold.sh` (Monday dispatch wrapper, bash-syntax clean)

## Smoke results

### 1. Label builder (Neptune, 50,000-event subsample on 20250714)
- Runtime: 5.5 s (per 50k events). Full day = ~13 min projected (6.5M events). 200+ days ETA ~36 hours single-threaded; **~5 hours with --workers 8**.
- Output keys verified: label_A_{5s,10s,30s,60s}, label_B_persistence_s, label_B_censored, label_C_{K2,K4,K8}, label_D_regime_60s
- Sanity: A_5s pos_frac = 0.534 (balanced), B median = 0.70s with 3.5% censored, C_K4 class dist = (neg 23k / zero 16k / pos 11k), D regime split (trending 1.7k / MR 43k / noise 5k).
- Note: D-head MR threshold is currently too permissive (43k/50k flagged MR). Tune Monday — likely needs flips > 8 or range > 3 ticks. Labels file remains valid; rebuild with adjusted thresholds is cheap.
- Note: label_A_60s synthesized from sign(labels_30s) as a temporary placeholder — needs true 60s lookahead label in next iteration.

### 2. v3.5 multi-head trainer (Neptune CUDA, smoke 1 fold × 1 epoch × W=256)
- Backbone import OK (smart_v3, 25 features, MAMBA_FEATURE_SET env override works)
- Model params: 701,842 (about 20k more than v2 single-head baseline — 4 small task heads added)
- Smoke run completed in 21.3 s
- All 4 head losses computed without NaN: train total = 4.08, val total = 3.57
- OOT inference dumps verified — prob_A (N,4), pred_B (N,), prob_C (N,3,3), prob_D (N,3)
- v2 baseline NOT modified (only imported)

### 3. Variable-horizon FIFO grader (Jupiter CPU, synthetic preds × 50k events)
- 7,831 trades generated from uniform-random predictions (as expected, unprofitable: -0.35 ticks/trade) — algorithm wired correctly.
- Exit reason distribution: a_flip / b_persistence / c_first_passage / d_cap recorded.
- Regime classification works (day tagged "red", skew check ran).
- Output: REPORT.md, per_day.csv, summary.json.
- TODO: wire `FIFOReplayEngine.simulate()` from `alpha_discovery/deep_models/fifo_market_replay.py` for full canonical queue-position realism. Current grader uses labels_1s diff as price-step proxy — sufficient for smoke and configuration-tuning, not for final pre-deploy validation.

## Constraint compliance
- v2 baseline NOT modified (only imported as `train_cnn_mamba_v2`)
- No full multi-fold training launched
- Razer LIVE host untouched
- HC #428 R1 (regime-skew) check implemented in grader
- HC #428 R2 (cancel-window <= persistence, exit on b_persistence) implemented

## Monday TODO before launching full
1. Run `python3 scripts/build_continuation_labels.py --workers 8` on Neptune (~5 hr ETA).
2. Tune Head-D thresholds (lower MR sensitivity).
3. Add true 60s lookahead label (requires extending the smart_v3 preprocessing OR computing from labels_30s plus a 30s extrapolation pass).
4. After labels build, run `bash /tmp/launch_v35_full_multifold.sh` (or copy to a permanent path).
5. Initial head_weights uniform; first MLflow run will reveal per-head loss magnitudes — re-weight if any head dominates.

## Self-check (HC #497)
1. All 3 scripts ran their smoke tests without error: YES.
2. Monday wrapper bash-syntax clean: YES (`bash -n` passes).
3. SESSION_STATE.md appended: YES.
4. Full Neptune training NOT launched: YES.
