# HC #417 Razer Paper Trader Config Audit (READ-ONLY)

**Date**: 2026-05-18
**Spec compared**: `output/hc417_v2_1s_short_top05_DEPLOYMENT_SPEC.md`
**Razer PID 25512 (paper trader) + PID 15720 (mbo recorder)**: untouched.

## Live process command lines (from `Get-CimInstance Win32_Process`)

- **PID 25512** (paper trader, started 2026-05-14 09:29:47):
  `python.exe C:\Users\claude\Lvl3Quant\live_trading\paper_trading_mamba_v2.py --symbol ESM6 --exchange CME --weights C:\Users\claude\Lvl3Quant\output\cnn_mamba_v2_smart_v3_mar\fold_10_best.pt --stats C:\Users\claude\Lvl3Quant\output\cnn_mamba_v2_smart_v3_mar\fold_09_feature_stats.npz --device cuda --min-tier Top5%`
- **PID 15720** (mbo recorder, started 2026-05-14 07:33:24):
  `python.exe C:\Users\claude\Lvl3Quant\mbo_recorder.py --symbol ESM6 --exchange CME --flush-minutes 5`

## IPC mechanism
**Inference is in-process** — `paper_trading_mamba_v2.py` instantiates `CNNMambaV2Inference` directly and connects to Rithmic itself via `run_live()` / `rithmic_client.RithmicClient`. There is **no separate inference engine process**. The `mbo_recorder` writes `live_events.jsonl` for `run_follow()` mode only — that mode is NOT in use by PID 25512.
Heartbeat sidecar JSON: `C:\Users\claude\Lvl3Quant\live_trading\status\razer_heartbeat.json` (60s cadence).

## Comparison table — current vs HC #417 spec

| Field | Spec (`v2_1s_short_top05`) | Current Razer (PID 25512) | Status |
|---|---|---|---|
| Model arch | CNN-Mamba v2 fold_10 | CNN-Mamba v2 fold_10 (loaded) | GREEN |
| Weights path | `models\fold_10_best.pt` (spec says) | `output\cnn_mamba_v2_smart_v3_mar\fold_10_best.pt` | YELLOW — spec path does NOT exist on Razer; loaded file SHA256 = `300E338D…1281E`. **Spec text is wrong; live path is the only one present.** |
| Symbol | ES (implied — ES_TICK_VALUE) | **ESM6** (CME) | GREEN |
| Signal head | `pred_log_ret_1s` only (head 0) | All 3 heads logged; entry uses combined `direction` from inference engine (not pure 1s head) | RED |
| Direction filter | SHORT only | **Both long & short** (no side filter in code) | RED |
| Confidence gate | Top 0.5% per-day percentile (≤ -0.6926 cold-start) | `--min-tier Top5%` tier flag (Top5/1/0.5/0.1 buckets pre-calibrated from concat OOT preds NPZ) | RED — wrong tier (Top5% vs Top0.5%) AND wrong gate type (tier bucket vs per-day percentile) |
| TP1 / TP2 / SL | +0.4782 / +0.9564 / -0.5686 ticks | No bracket TP/SL. Exits = signal_flip / direction_reversal / signal_decay / trailing_stop(MFE≥4t lock 1t) / max_hold(60s) / max_daily_loss(-$3000) | RED — no TP1/TP2/SL brackets implemented |
| Order type | passive_at_touch | Implicit market at mid/touch (`entry_price = mid_price or ask/bid`); no order lifecycle, no cancel | RED |
| Cancel window | 40 evals × 250ms = 10s | N/A (no resting limits) | RED |
| Position size | 1 contract | 1 (hard-coded, no scaling) | GREEN |
| Trading hours | RTH 09:30–16:00 ET | **No hours filter in script** (spec also lists "Rule 4: no entries 9:25–9:40" in docstring but I see no implementation check) | RED |
| Re-entry cooldown | 5s | None explicit; re-enters immediately after flip if tier/risk allow | RED |
| Stride / window | 250ms stride (spec implied) | window=1000, **stride=500** events (not time-based) | YELLOW |
| Kill: daily loss | -10 ticks ($125) | -$3000 (24× looser) | RED |
| Kill: weekly loss | -25 ticks ($312.50) | Not implemented | RED |
| Kill: consec losses | 5 → 1h pause | 3 → 10 min pause (CLI default) | YELLOW |
| Kill: stale signal | >30s → pause entries | Not implemented | RED |
| Kill: connectivity | flatten + pause | Not implemented (relies on Rithmic client only) | RED |
| Kill: SHA-256 check | refuse start on mismatch | Not implemented | RED |
| Kill: IC drift | rolling 5-day IC<0.15 disable | Not implemented | RED |
| Alerts | Discord/Telegram on fill/exit/kill | Heartbeat JSON only (no webhook wired in script) | RED |

## Greenlight gates: 3/19 pass cleanly (model arch, symbol, position size). 16 RED/YELLOW.

## Recommendation
**Phase A greenlight is NOT a config-swap-and-restart operation; it requires code changes.** The running script is the legacy HC #46 risk-managed trader (signal-flip entry, no TP/SL brackets, both sides, Top5% tier, no RTH/cooldown/connectivity/SHA/IC/weekly-loss/alert kill-switches). Implementing the `v2_1s_short_top05` spec needs: (a) SHORT-only filter, (b) Top0.5% per-day percentile gate with global cold-start fallback, (c) TP1/TP2/SL bracket execution logic with passive_at_touch + 10s cancel, (d) RTH window + 5s re-entry cooldown, (e) 7 additional kill-switches, (f) Discord webhook wiring. Estimated effort: **~6–10 engineer-hours of new code + a fresh test pass**, not a swap. Also fix DEPLOYMENT_SPEC.md row 15 — the canonical weights path is `output\cnn_mamba_v2_smart_v3_mar\fold_10_best.pt` (SHA `300E338D3C16137FC587B10CCE92204E8FE0A486FC3C7AAF53856FDD8C21281E`), not `models\fold_10_best.pt`.
