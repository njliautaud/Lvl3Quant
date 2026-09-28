# STATE OF THE PROJECT — 2026-05-19 22:55 ET

## TL;DR — Honest Project Position

- **Friday deadline (2026-05-22) for production-ready tradeable system: AT RISK.**
- **No config to date trades positively under canonical realtime_sl FIFO replay on the CNN-Mamba v2 proven signal model.**
- The HC #441 "champion" headline (+0.71 tk/fill) was produced by a buggy analytic resolver and is FALSE.
- The CNN-Mamba v2 model has real label-level edge (IC_1s = 0.222, top-0.5% short signals show ~21 ticks mean MFE inside 1.5s).
- The gap between label edge and tradeable edge is the entire problem we have to solve in 3 days.

## Three Numbers That Define Tonight

| Source | Engine | cancel_s | n fills | net tk/fill | PF | WR | Verdict |
|---|---|---|---|---|---|---|---|
| HC441 verdict report | analytic resolver | 10s | 3,455 | **+0.71** | 2.54 | 45.8% | **BUG — invalid** |
| HC442 canonical c1 | realtime_sl FIFO | 1s | 2,431 | **−0.326** | 0.51 | 22.8% | TRUE — LOSING |
| HC442 canonical c10 | realtime_sl FIFO | 10s | 3,455 | **−0.313** | 0.51 | 25.1% | TRUE — LOSING |

The c10 canonical run produced exactly 3,455 fills — identical to the analytic — on the same model with the same TP/SL/hold/cancel. Same trades, opposite sign P&L. This is unambiguous proof of an analytic-resolver bug, not a methodology difference.

## What's Proven Real (and Reusable)

1. **CNN-Mamba v2 label-level edge** — concat IC_1s = 0.222, IC_5s = 0.141, IC_10s = 0.106 across 47 OOT days. Model decay analysis (2026-05-01) shows edge persists ~30s per MFE/MAE. NOT model decay.
2. **Top-0.5% short signals show 1.56 ticks of mean directional move inside 1s** (HC #428 analysis). Real but small.
3. **p90 of realized MFE on top-0.5% short within 1s = ~21 ticks** (HC #441 R2 verification). Real upside exists if we can capture it.
4. **CNN-Mamba v2 model weights, predictions, MBO data** all preserved in `output/cnn_mamba_v2_bulk_oot_v2/`.
5. **Canonical realtime_sl FIFO engine** at `scripts/v3_4_research/hc432_v2_baseline_runner.py` is the trustworthy harness. Use this for ALL future P&L claims.

## What's Broken / Not To Use

1. ❌ **Analytic resolver** in `scripts/v3_4_research/hc437_pathB_exit_sweep.py` `resolve_with_mfe_mae()` — sign error or direction-flip bug producing +0.71 when truth is −0.31. DO NOT use until diagnosed and fixed.
2. ❌ **HC #441 R2-STRICT (SL=0.5, TP=3.0, H=1.5s passive)** — confirmed losing under canonical. Not a champion.
3. ❌ **HC #437 TP=0.96, SL=0.57, hold=10s baseline** — already known losing (−0.27 tk/fill).
4. ❌ **HC #428 violation configs (cancel_s > h)** — survivorship bias on entry, not deployable.

## Open Question (Diagnose Soon)

**Why does the analytic resolver get the sign wrong?** Hypothesis: `resolve_with_mfe_mae` may be summing MFE as positive contribution for short trades when MFE represents the favorable direction (price moving DOWN for shorts) — but if the trajectory is recorded as raw price movement, MFE for a short should be a NEGATIVE price delta. The resolver might be adding |MFE| as winnings to short trades when actually the trade exits at TP=3 which is a 3-tick FAVORABLE move (price down). If the resolver is computing `net = MFE - SL_hit_cost` instead of `net = TP_when_hit_before_SL`, it accumulates the entire MFE peak as profit rather than the TP exit. That would explain the magnitude and the sign-consistency. Diagnose by hand-resolving 10 fills and comparing.

## Paths Forward — Ranked by Likelihood of Friday Success

### 1. **Aggressive market-order execution (FASTEST, MOST LIKELY)**
- The label edge is +21 ticks mean MFE within 1.5s on top-0.5% short.
- A market-order entry pays ~1.4 ticks (commission + 1-tick spread). Net theoretical edge ~+19 ticks IF model is right.
- A market-order exit at TP=3 also pays ~1.4 ticks. Net = 3 − 2.8 = +0.2 ticks per win.
- Or chase TP from inside the spread. Higher fill rate, lower cost.
- **Canonical-validate aggressive entry + chase exit on the same 32-day OOT TOMORROW.**

### 2. **Multi-horizon confluence (1s ∧ 5s ∧ 10s short agreement)**
- Drastically fewer trades but each one survives adverse selection.
- Need a sweep of confluence thresholds + canonical re-validation.
- Build time: 1-2 days. Tight on Friday.

### 3. **Wider geometry with passive entry**
- Try SL=3, TP=8, hold=10s passive — gives the trade room to breathe.
- Violates HC #428 R2 nominally; need to re-derive p90 MFE at longer horizons.
- 1 day to canonical-validate.

### 4. **Different model entirely (PatchTST or LGBM-Vol meta)**
- PatchTST was trained but never canonical-validated. Worth a baseline run.
- 1 day to canonical-test top-0.5% PatchTST short.

### 5. **HONEST FALLBACK: paper-trade label-level signals only**
- If nothing canonical-validates by Thursday EOD, ship a DATA-COLLECTION harness that records what would happen if we traded the model's top-0.5% short on every signal, and gather 1 week of live evidence before committing capital.
- This is a Friday-deliverable "production-aware system" even if it's not yet a P&L-positive one.

## Action Queue for Autonomous Execution

| Order | Action | Owner | Engine | Status |
|---|---|---|---|---|
| 1 | Diagnose analytic resolver bug (sign/direction error) | Jupiter | n/a | PENDING |
| 2 | Canonical run: market-order entry + TP=3, SL=0.5, hold=1.5s | Jupiter | hc432_v2_baseline_runner.py --order-type market | PENDING |
| 3 | Canonical run: chase entry + chase exit, same geometry | Jupiter | hc432_v2_baseline_runner.py --order-type chase | PENDING |
| 4 | Canonical run: passive + SL=3, TP=8, H=10s (wider) | Jupiter | hc432_v2_baseline_runner.py | PENDING |
| 5 | Multi-h confluence: require pred_log_ret_1s & 5s & 10s all in top-30% short | Jupiter | new script | PENDING |
| 6 | If anything passes: HC #428 R1 regime stratification + 21/15 holdout split | Jupiter | resolver script | PENDING |
| 7 | If nothing passes by Thursday EOD: ship the data-collection fallback | Jupiter | new live recorder | PENDING |

## Key File Paths

- **Canonical engine entry**: `scripts/v3_4_research/hc432_v2_baseline_runner.py`
- **CNN-Mamba v2 predictions**: `output/cnn_mamba_v2_bulk_oot_v2/{YYYYMMDD}_predictions.npz`
- **MBO data**: `data/raw_mbo/` and `data/processed/mbo_event_cache/`
- **Canonical results landing zone**: `output/hc432_v342_47day_validation/{config_name}_fifo_fills.csv` + `_summary.json`
- **Broken analytic verdict**: `output/hc441_full_verdict/` — labeled INVALID
- **This checkpoint**: `output/state_of_project/STATE_OF_THE_PROJECT_20260519.md`

## Commit Hashes / Reproducibility

- Canonical c1 result CSV: `output/hc432_v342_47day_validation/hc442_v2_canon_c1_fifo_fills.csv` (2,431 rows, written 22:40 ET)
- Canonical c10 result CSV: `output/hc432_v342_47day_validation/hc442_v2_canon_c10_fifo_fills.csv` (3,455 rows, written 22:55 ET)
- Buggy analytic CSV: `output/hc441_full_verdict/per_fill_PRIMARY.csv` (3,455 rows, written 22:05 ET — DO NOT TRUST P&L numbers)

## For Friday Claude

If you're reading this on Friday morning:
1. The +0.71 verdict report from Tuesday night is WRONG. Do not promise that number to the user.
2. The canonical engine is `hc432_v2_baseline_runner.py`. Use it for EVERY P&L claim.
3. Check `output/state_of_project/STATE_OF_THE_PROJECT_*.md` for the latest results — newest file wins.
4. Action queue above is your starting point. Order #1 (diagnose analytic bug) can wait; orders #2-5 are P&L searches.
5. If nothing canonical-passes by EOD Thursday, ship order #7 — the live recorder fallback.
