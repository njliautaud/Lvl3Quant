# Friday Harness Sidecar — Deploy Procedure (For User Review, 2026-05-22 AM)

**Status (2026-05-21 19:25 ET)**: All artifacts staged on Razer. NOT YET LAUNCHED.
**Recommended launch window**: 2026-05-22 between 06:30-09:00 ET (pre-market, light monitoring on v2 paper trader, time to verify before market open).

## What ships

1. **`harness_aux_sidecar.py`** (Razer: `C:\Users\claude\Lvl3Quant\live_trading_linux\`)
   - Runs as a SEPARATE Windows process from the v2 paper trader.
   - Tails the same `live_events.jsonl` (read-only, multi-reader safe).
   - Records v3.3 + v3.4.2 predictions to `harness_aux_predictions_<SESSION_TAG>.jsonl`.
   - Joins to v2 JSONL post-process by `ts_ns`.

2. **Adapter modules** (Razer: `C:\Users\claude\Lvl3Quant\live_trading_linux\`)
   - `v3_3_inference.py` — CNN-Mamba v3.3 inference wrapper (exposes 1s/5s/10s/30s + diagnostic 60s/5min).
   - `v3_4_2_inference.py` — CNN-Mamba v3.4.2 wrapper (omits 60s/5min per HC #477 broken labels; returns NaN with `reason="book_features_missing"` until book pipeline is built).

3. **Model checkpoints** (Razer: `C:\Users\claude\Lvl3Quant\output\`)
   - `cnn_mamba_v3_3_uncertainty_weighted/fold_00_intra_ckpt.pt` (18.7 MB)
   - `cnn_mamba_v3_3_uncertainty_weighted/fold_00_feature_stats.npz` (824 B)
   - `cnn_mamba_v3_4_2_fixedmtl/fold_00_intra_ckpt.pt` (21.3 MB)
   - `cnn_mamba_v3_4_2_fixedmtl/fold_00_feature_stats.npz` (824 B)

## Launch sequence

```powershell
# 1. Smoke test (verifies imports + paths on Razer)
ssh claude@razer "cd C:\Users\claude\Lvl3Quant\live_trading_linux && C:\Python311\python.exe harness_aux_sidecar.py --smoke"
# Expected: prints [smoke] PASS

# 2. Live launch (detached, survives SSH disconnect)
ssh claude@razer "cd C:\Users\claude\Lvl3Quant\live_trading_linux && start /B C:\Python311\python.exe harness_aux_sidecar.py --device cuda > logs\harness_sidecar_$(Get-Date -Format yyyyMMdd_HHmmss).out 2>&1"
# OR use a spawn_harness_sidecar.py wrapper modeled on spawn_shadow_v2.py
```

## Verification within 5 min of launch

1. Process running:
   ```powershell
   ssh claude@razer "tasklist | findstr python"
   # Expect: ≥4 python procs (MBO recorder, watchdog, v2 paper trader, NEW sidecar)
   ```

2. JSONL written:
   ```powershell
   ssh claude@razer "dir C:\Users\claude\Lvl3Quant\live_trading\logs\harness_aux_predictions_*.jsonl"
   ```

3. Sample row well-formed:
   ```powershell
   ssh claude@razer "powershell -Command \"Get-Content C:\Users\claude\Lvl3Quant\live_trading\logs\harness_aux_predictions_*.jsonl -Tail 3\""
   # Expect: each line is JSON with v33_pred_*, v342_pred_*, ts_ns, v33_reason=ok, v342_reason=book_features_missing
   ```

## Known acceptable issues

1. **v3.4.2 always NaN**: book_pyramid features aren't built on the live stream yet. Sidecar logs `v342_reason="book_features_missing"` 100% of the time. Per plan Step 4 fallback. v3.3 carries the multi-model signal.

2. **v3.3 feature padding bias**: streaming_features_smart_v3 emits 25 features; v3.3 model expects 39 (= 25 events + 4 PT-meta + 10 book-history). Sidecar zero-pads the trailing 14. Verified via model code: t1_adapter was initialized to identity on first 25 dims + tiny random (std=0.01) on the trailing 14. Bias is bounded by however much those near-zero weights drifted during training (likely small).

3. **No SHA-256 integrity check**: sidecar is OBSERVE-ONLY, not a tradable engine. SHA-256 is the v2 paper trader's job.

## Watchdog hook (after first verified launch)

Update `live_stack_watchdog.py` `processes` dict to add:
```python
"harness_aux_sidecar": {
    "cmdline_contains": "harness_aux_sidecar.py",
    "restart": True,
},
```

## Rollback (if sidecar misbehaves)

```powershell
ssh claude@razer "powershell -Command \"Get-Process python | Where-Object {$_.MainWindowTitle -eq '' -and ($_.CommandLine -like '*harness_aux_sidecar*')} | Stop-Process\""
```

Sidecar is read-only against `live_events.jsonl`. Killing it has ZERO impact on the v2 paper trader, MBO recorder, or watchdog.

## File hashes (post-scp verification)

```
v3_3_inference.py        — 10,800 bytes
v3_4_2_inference.py      — 10,807 bytes
harness_aux_sidecar.py   — 23.7 KB (~541 lines)
v3.3 ckpt                — 18,715,410 bytes
v3.4.2 ckpt              — 21,267,789 bytes
```
