# HC #428 R2: MFE-Within-Horizon Audit (per top-3 config)

_Generated: 2026-05-19 08:35 ET_

## Rule (HC #428 R2)

- TP/hold/cancel/confluence MUST be bounded by predicted horizon
- hold_seconds ≤ 1.5 × horizon_sec
- cancel_window (sec) ≤ horizon_sec
- confluence_horizon ≤ trade_horizon
- mean realized payoff at h ≤ p90 of MFE@h (no luck/momentum carry)

## v3.4.2 trial 1554 · 30s short

- n_signals: 65720
- horizon: 30s (30.0s)
- MFE@h p50 / p75 / p90 / p95 (ticks): 3.000 / 7.000 / 12.000 / 17.000
- mean_payoff @ h (signed): 0.193 ticks
- hold_seconds: 2.14 (cap=1.5×30.0=45.0s) → **PASS**
- cancel_window_seconds: 9.5 (cap=30.0s) → **PASS**
- confluence_horizon: 10s (cap=30s) → **PASS**
- tp_check (mean payoff ≤ p90 MFE): **PASS**
- **R2 OVERALL: PASS**
- explanation: hold 2.14s vs 1.5×30.0=45.0s: PASS; cancel 9.5s vs 30.0s: PASS; confluence 10s ≤ 30s: PASS; mean_payoff 0.193 ≤ p90_MFE 12.000: PASS

## v3.4.2 trial 1422 · 5s short

- n_signals: 63580
- horizon: 5s (5.0s)
- MFE@h p50 / p75 / p90 / p95 (ticks): 1.000 / 3.000 / 5.000 / 7.000
- mean_payoff @ h (signed): 0.561 ticks
- hold_seconds: 1.10 (cap=1.5×5.0=7.5s) → **PASS**
- cancel_window_seconds: 20.0 (cap=5.0s) → **FAIL**
- confluence_horizon: 5s (cap=5s) → **PASS**
- tp_check (mean payoff ≤ p90 MFE): **PASS**
- **R2 OVERALL: FAIL**
- explanation: hold 1.10s vs 1.5×5.0=7.5s: PASS; cancel 20.0s vs 5.0s: FAIL; confluence 5s ≤ 5s: PASS; mean_payoff 0.561 ≤ p90_MFE 5.000: PASS

## v3.3 trial 2831 · 10s short

- n_signals: 15079
- horizon: 10s (10.0s)
- MFE@h p50 / p75 / p90 / p95 (ticks): 2.000 / 3.000 / 6.000 / 8.000
- mean_payoff @ h (signed): 0.597 ticks
- hold_seconds: 2.37 (cap=1.5×10.0=15.0s) → **PASS**
- cancel_window_seconds: 14.2 (cap=10.0s) → **FAIL**
- confluence_horizon: 10s (cap=10s) → **PASS**
- tp_check (mean payoff ≤ p90 MFE): **PASS**
- **R2 OVERALL: FAIL**
- explanation: hold 2.37s vs 1.5×10.0=15.0s: PASS; cancel 14.2s vs 10.0s: FAIL; confluence 10s ≤ 10s: PASS; mean_payoff 0.597 ≤ p90_MFE 6.000: PASS
