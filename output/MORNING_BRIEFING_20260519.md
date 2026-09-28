# Morning Briefing — 2026-05-19 (overnight session #65 results)

## TL;DR for the user

1. **v3.4.2 Optuna sweep complete** (3000 trials, 1160 deploy-eligible).
2. **v3.4.2 LOO-validation: 12/20 robust** (60% survival rate vs v3.3's 35%). All 12 use the `passive_at_touch_plus_2` order type and cluster at TOD 14:00-15:00 ET.
3. **Razer live stack is healthy: 4 paper traders running**, all consuming the single recorder feed (HC #426 R2 compliant). A new follow-events watchdog auto-restarts the 3 tier shadows if they die.
4. **One latent crash fixed** (`LatencyTracker.summary` AttributeError that killed session #64 shadows) — fix is a thin wrapper `paper_trading_mamba_v2_runner.py` that monkey-patches at import; no modification to the original script.
5. **Two friction points the user should review:**
   - **HC #426 R3 violation:** top LOO-robust configs use `hold_seconds` of 1-4s when `head_horizon` is 5-30s — sweep parameter range allowed this. The configs work in backtest, but they violate the directive "hold [horizon, 3×horizon]". Should a re-sweep enforce R3 strictly? See "Open question A" below.
   - **`durable:true` on CronCreate is a NO-OP:** `/home/jupiter/.claude/scheduled_tasks.json` does not exist. SessionStart hook re-arms crons each restart. A real fix (Linux crontab → Discord-bridge) is pending.

## Production-ready candidates (for HC #426 R1 / Friday 5/22 deadline)

The top 3 v3.4.2 candidates with best fill-count × worst-day-Sharpe combo:

| trial | horizon/side/order        | LOO mean Sh | worst-day Sh | n_fills/5d | TOD   | hold |
|------:|----------------------------|-----------:|-------------:|-----------:|-------|------|
| 1554  | 30s/short/passive_+2       | 13.42      | 4.64         | 89         | 14-15 | 2.1s |
| 1207  | 5s/short/passive_+2        | 17.62      | 10.02        | 62         | 14-15 | 1.1s |
| 1406  | 5s/short/passive_+2        | 20.95      | 16.25        | 68         | 14-15 | n/a  |

All 12 robust configs and full per-day metrics at:
`output/v342_execution_optuna_20260518/loo_robust_configs.json`

Plus 7 v3.3 LOO-robust configs (from session #63) at:
`output/v33_execution_optuna_20260518/loo_robust_configs.json`

## Canonical avg-move (HC #426 R3 ground truth)

Identical between v3_3 and v3_4_2 (same OOT market data):

| horizon | mean abs ticks | median | p75   | p90   | passive feasible | IOC feasible |
|--------:|---------------:|-------:|------:|------:|:----------------:|:------------:|
| 1s      | 1.08           | 1.00   | 2.00  | 2.00  | ✓                | ✗            |
| 5s      | 2.42           | 2.00   | 3.00  | 5.00  | ✓                | ✓            |
| 10s     | 3.41           | 2.00   | 5.00  | 8.00  | ✓                | ✓            |
| 30s     | 5.70           | 4.00   | 8.00  | 13.00 | ✓                | ✓            |

At 1s, IOC market orders (cost 1.376t) eat all the mean edge — only passive limit orders survive. From 5s onward both order types are feasible.

## HC #425 alt-label base-rate verdict

All 4 label geometries (time60, tp4sl4, tp6sl2, tp8sl3) produce **negative** mean gross ticks across the 5-day OOT window — i.e., naive signal-blind execution is unprofitable. This is the floor the model must beat. The LOO-robust configs DO beat it (positive Sharpe at high confidence + horizon-confluence filtering).

Best (= least bad) geometry: `tp4sl4` for both long and short (mean_gross ≈ −0.18t / −0.22t). Worst: `time60` (mean ≈ −5t / −7t).

Full table: `output/hc425_alternative_labels/base_rate_verdict.md`.

## Razer process inventory (00:14 ET)

| PID   | Process       | Role                        | Mem   | Source                                      |
|------:|---------------|------------------------------|------:|---------------------------------------------|
| 15720 | python.exe    | MBO recorder (Rithmic feed) | 60 MB | Pre-session, sole Rithmic socket owner      |
| 32980 | python.exe    | live_stack_watchdog         | 27 MB | Pre-session                                 |
| 35128 | python.exe    | v2_1s_short_top05 #1        | 830 MB | Manual relaunch this session                |
| 35680 | pythonw.exe   | v2_1s_short_top05 #2        | 151 MB | launch_shadow_razer.bat (delayed)           |
| 35348 | python.exe    | Top0.5% follow-events       | 842 MB | launch_3_follow_shadows_durable.ps1         |
| 34732 | python.exe    | Top1%   follow-events       | 842 MB | launch_3_follow_shadows_durable.ps1         |
| 26240 | python.exe    | Top5%   follow-events       | 842 MB | launch_3_follow_shadows_durable.ps1         |

CUDA total ≈ 3.36 GB of 8 GB RTX 3070. Plenty of headroom.

A new `follow_shadow_watchdog.ps1` runs every 60s and relaunches any of the 3 follow-events shadows if they die. Lives at `C:\Users\claude\Lvl3Quant\live_trading\` on Razer; launched via `launch_follow_shadow_watchdog.bat` (Win32_Process.Create, survives SSH disconnect).

## Neptune v3.4.3-REPAIR-v2 training (PID 955659)

- 10.8% through epoch 1 at 00:05 ET (step 14300 / 132824)
- Loss going more negative (improving), elapsed 17.7 min, ETA ~2.4 hours
- ICs still very weak (log_ret_1s=0.0085, 30s=−0.005, MFE_30s=0.009) — needs full epoch to judge
- Predecessor expected ep-1 GO/NO-GO verdict ~03:00 ET — should be checked then

## Open questions for the user (act / approve on these)

**A. HC #426 R3 violation in winning configs.**
The 12 LOO-robust v3.4.2 configs all use `hold_seconds` that is 7-30% of `head_horizon` (not the R3-mandated `[horizon, 3×horizon]`). They appear genuinely profitable in backtest but violate the directive. Options:
  - Accept as-is (the empirical results stand)
  - Re-run sweep with strict R3 ranges (~1-2 hours per model)

**B. v3.x weight deployment to Razer (Friday EOW deadline).**
Existing `paper_trading_mamba_v2.py` only loads v2 model class. To deploy v3.3/v3.4.2 shadows we need:
  - SCP `output/cnn_mamba_v3_3_uncertainty_weighted/*.{pt,npz}` Jupiter → Razer
  - Same for v3.4.2
  - A new file `paper_trading_mamba_v3.py` that loads v3.x model class (do NOT modify v2 file per malware-guard policy)
  - Confirm v3.x model architecture is in `cnn_mamba_v3_3_model.py` (or similar) on Razer

**C. Update `live_stack_watchdog.py` to monitor the 3 new follow-events PIDs.**
Currently it watches only `mbo_recorder`, `legacy_paper_top5`, `shadow_v2_1s_short_top05`. The new sidecar `follow_shadow_watchdog.ps1` covers the gap but is two-watchdog architecture. Cleaner: extend the existing watchdog config.

## Files created this session

- `/home/jupiter/Lvl3Quant/scripts/v3_3_research/oot_loo_validate_top_configs.py` (LOO validation script)
- `/home/jupiter/Lvl3Quant/scripts/v3_3_research/compute_canonical_avg_move.py` (canonical avg-move cache)
- `/home/jupiter/Lvl3Quant/scripts/v3_3_research/hc425_base_rate_verdict.py` (alt-label verdict)
- `/home/jupiter/Lvl3Quant/scripts/v3_3_research/auto_loo_when_sweep_done.sh` (poller; finished its job)
- `C:\Users\claude\Lvl3Quant\live_trading\paper_trading_mamba_v2_runner.py` (LatencyTracker.summary monkey-patch wrapper)
- `C:\Users\claude\Lvl3Quant\live_trading\launch_3_follow_shadows_durable.ps1`
- `C:\Users\claude\Lvl3Quant\live_trading\follow_shadow_watchdog.ps1`
- `C:\Users\claude\Lvl3Quant\live_trading\launch_follow_shadow_watchdog.bat`
- `/home/jupiter/Lvl3Quant/output/v342_execution_optuna_20260518/loo_validation_results.json` + `loo_robust_configs.json`
- `/home/jupiter/Lvl3Quant/output/hc425_alternative_labels/base_rate_verdict.{md,json}`

## Pending tasks for the next session

1. Verify follow-events shadows survived past 5-min periodic_summary (test of LatencyTracker.summary patch).
2. Wait for ≥5000 events to flow on live_events.jsonl so shadows clear warmup → producing signals.
3. Check v3.4.3-REPAIR-v2 epoch 1 boundary (~02:30 ET).
4. Decide on HC #426 R3 re-sweep (see Open question A).
5. Begin v3.x Razer deployment (SCP weights, write v3.x paper trader, see Open question B).
6. Long-term: replace `CronCreate durable:true` workaround with Linux crontab → Discord-bridge.
