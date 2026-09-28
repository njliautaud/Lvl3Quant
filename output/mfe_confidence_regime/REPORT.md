# Deploy-gate diagnostic: MFE/MAE × signal-confidence × regime

Generated: 2026-05-22 04:34 ET — wall: 6.7s
Source fills: `output/hc475_ab/symmetric_gate_fills.parquet` (31086 rows, 15 OOT dates 2026-02-23 → 2026-03-13).
Signal confidence: real |pred| recovered from v3.4.2 47-day OOT NPZ (1s/5s/10s/30s heads only — HC #477).
Realized MFE/MAE: NPZ `target_pred_{mfe,mae}_30s_ticks` (true within-30s extrema). For 1s/5s/10s horizons the per-horizon realized return (`target_log_ret_*`) is used as a proxy — true intra-horizon extrema not stored for h<30s.
Cost: passive limit at touch = 0.376 ticks (commission only). Bracket from fills: TP=4t, SL=3t, hold=30s, cancel=10s.

## Pipeline
- fills loaded: 31086, joined to NPZ: 31086 (100.0% — coverage limited only by 60s/5min NPZ horizons being unavailable / fills falling outside NPZ window).
- cells produced (config × side × horizon × confidence-quintile): 198
- cells passing day-conc gate (≤0.70): 4/198
- cells passing regime gate (imbalance ≤0.50): 78/198
- cells passing MFE-within-horizon gate (TP≤p90(MFE_h) and hold≤1.5h): 46/198
- cells passing ALL three gates: **0/198**

## Headline cell
**Status: FAIL (best=2/3 gates)**

| Field | Value |
|---|---|
| Config | trip01_pup5s+logret1s+logret10s |
| Side | short |
| Horizon | 5s |
| Confidence quintile | Q3 |
| n_trades | 49 |
| Sharpe | 0.143 |
| Sortino | 0.000 |
| PF | 1.340 |
| WR | 0.571 |
| Mean net ticks | 0.491 |
| Day-conc | 0.503 (gate ≤0.70 → PASS) |
| Regime imbalance | 0.278 (gate ≤0.50 → PASS) |
| p90 MFE within h | 5.200 ticks vs TP=4.0 → PASS |
| hold 30.0s vs 1.5×h (7.5s) → FAIL |
| Long/short balance for config | long_share=0.943 (n_long=3823, n_short=229) |

## Long/short separated summary (HC #475 R1)

Per-config aggregate (across all confidence quintiles), top horizon shown for each side:

| Config | Side | n | Sharpe | Sortino | PF | WR | Mean ticks |
|---|---|---|---|---|---|---|---|
| pair01_logret1s+pup5s | long | 6063 | -0.104 | -2.058 | 0.811 | 0.431 | -0.361 |
| pair01_logret1s+pup5s | short | 1228 | -0.053 | -0.984 | 0.898 | 0.458 | -0.185 |
| pair07_logret10s+logret60sq50 | long | 66 | -0.314 | -2321042848972626.500 | 0.537 | 0.333 | -1.043 |
| pair07_logret10s+logret60sq50 | short | 2229 | -0.176 | -1.430 | 0.701 | 0.401 | -0.591 |
| pair08_logret5s+pup5s | long | 8522 | -0.108 | -2.262 | 0.806 | 0.430 | -0.372 |
| pair08_logret5s+pup5s | short | 1827 | -0.057 | -1.142 | 0.891 | 0.456 | -0.199 |
| trip01_pup5s+logret1s+logret10s | long | 3823 | -0.093 | -2.276 | 0.830 | 0.438 | -0.321 |
| trip01_pup5s+logret1s+logret10s | short | 229 | -0.136 | -1.774 | 0.760 | 0.424 | -0.463 |
| trip03_logret5s+pup5s+logret1s | long | 5919 | -0.101 | -2.003 | 0.817 | 0.433 | -0.349 |
| trip03_logret5s+pup5s+logret1s | short | 1180 | -0.046 | -0.820 | 0.913 | 0.461 | -0.158 |

## Top-10 cells by Sharpe (any gate status)

| Config | Side | Horizon | Quintile | n | Sharpe | PF | WR | day_conc | regime_imb | MFE_h | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pair01_logret1s+pup5s | short | 10s | Q4 | 84 | 0.164 | 1.391 | 0.560 | 0.545 | 0.896 | 5.00 | D.. |
| trip01_pup5s+logret1s+logret10s | short | 5s | Q3 | 49 | 0.143 | 1.340 | 0.571 | 0.503 | 0.278 | 5.20 | DR. |
| pair01_logret1s+pup5s | short | 30s | Q2 | 228 | 0.128 | 1.293 | 0.548 | 0.483 | 0.251 | 1.00 | DR. |
| trip03_logret5s+pup5s+logret1s | short | 30s | Q2 | 228 | 0.128 | 1.293 | 0.548 | 0.483 | 0.251 | 1.00 | DR. |
| trip01_pup5s+logret1s+logret10s | short | 10s | Q4 | 41 | 0.098 | 1.220 | 0.537 | 0.807 | 1.691 | 6.00 | ... |
| trip01_pup5s+logret1s+logret10s | short | 1s | Q3 | 46 | 0.077 | 1.172 | 0.543 | 0.911 | 1.018 | 2.00 | ... |
| trip03_logret5s+pup5s+logret1s | short | 1s | Q5 | 236 | 0.057 | 1.120 | 0.513 | 0.753 | 0.806 | 2.00 | ... |
| trip01_pup5s+logret1s+logret10s | short | 5s | Q2 | 15 | 0.035 | 1.079 | 0.533 | 2.082 | 1.960 | 4.60 | ... |
| pair07_logret10s+logret60sq50 | long | 10s | Q3 | 8 | 0.033 | 1.073 | 0.500 | 7.306 | 1.833 | 2.40 | ... |
| trip03_logret5s+pup5s+logret1s | short | 30s | Q4 | 177 | 0.019 | 1.038 | 0.497 | 3.209 | 1.427 | 8.00 | ..M |

Gate legend: D=day-conc≤0.70, R=regime-imbalance≤0.50, M=MFE-within-horizon+hold-within-1.5h. `.` = fail.

## Verdict

**ZERO cells pass all three deploy gates.** Symmetric_gate output is not deploy-ready in any (config × side × horizon × confidence) slicing.

Dominant failure modes:
- **Unprofitable cells:** 184/198 cells have mean_net ≤ 0 — the symmetric gate is bleeding through the bracket cost on most slices. Only 14 cells are profitable at all before applying gates.
- **MFE-within-horizon failures:** 152/198 cells. TP=4 ticks exceeds p90(MFE) within the predictive horizon for nearly every short-horizon cell (typical p90 MFE_h ≈ 1–2 ticks at h=1s/5s/10s). Also hold=30s violates the 1.5×h limit for every h<20s head (prediction is stale by exit).
- **Regime imbalance failures:** 120/198. Many cells show Sharpe sign-flip between green and red days → regime-tailored, not edge.
- **Day-concentration failures:** 194/198. Auto-rejected when cell is unprofitable (day-conc undefined) or when one day contributes >70% of profit.

## Caveats
- For h ∈ {1s, 5s, 10s} we used realized return at horizon end (NPZ `target_log_ret_h`) as a proxy for MFE-within-h. True intra-horizon extrema are only stored for the 30s and 60s heads (and 60s is HC #477-banned).
- Confidence quintiles are formed within each (config, side) cell on |pred|. Q5 = strongest, Q1 = weakest. With small short-side fill counts (66–2,229) some quintiles collapse to fewer bins.
- Regime classification uses the existing cache (`output/stream_backtest_v2/top10_per_day_pair.parquet`) where available; fallback computes net 30s-return sum per day with ±5-tick thresholds.
- HC #74 binding: only FIFO market replay results (`net_ticks` column from fills) are used as the headline performance metric. NPZ-based "net_at_h" is reported as a sanity column.
