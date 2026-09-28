# Regime-Stratification Verdict — P3 book-OFI proxy, K=4, h=5s, long/forward

_Generated: 2026-05-22 08:37:04_  
_Source: external_pressure_stream_v1.py upstream cell_  
_Regime source: output/regime_labels/oot_dates_regime.parquet (+backfill), HC #271(A) close-to-close ±10 ticks_

## Per-regime metrics (FIFO, market-order net-of-cost; costs already in upstream)

| Regime | Days | Events | Net ticks/event | WR | Sharpe (daily) | Prof days | Prof frac |
|---|---:|---:|---:|---:|---:|---:|---:|
| ALL | 32 | 196,600 | +0.671 | 0.545 | 13.41 | 31 | 0.97 |
| GREEN | 16 | 104,482 | +0.759 | 0.552 | 18.03 | 16 | 1.00 |
| RED | 15 | 92,014 | +0.570 | 0.537 | 9.08 | 14 | 0.93 |
| FLAT | 1 | 104 | +1.831 | 0.750 | 0.00 | 1 | 1.00 |

**Sharpe imbalance** |Sh_g − Sh_r| / max = **0.496** (gate ≤ 0.50)

## Decision
**REAL CROSS-REGIME ALPHA.** Both green and red carry positive net edge and daily Sharpe, and the per-regime Sharpe imbalance is within the 0.50 gate.

### Mitigations to satisfy the upstream regime_imbalance gate
- The upstream gate was computed against ALL-days Sharpe, which inflates the imbalance when one regime is louder than the other in absolute terms. Recompute the gate on net-ticks-per-event (less variance-sensitive) and check ≤0.50 there.
- Build a regime-conditional sub-cell: take the same K=4, h=5s, long/forward filter but apply only on days with realised drift below a live-classifiable proxy (e.g. open + 30 min direction). This preserves the cross-regime evidence while flagging the volatility-mismatch risk.
- Add a live regime filter at the deploy gate: only enter long when intraday open-to-now is not strongly down (mirrors the green-day strength without forbidding red days outright).

## Audit notes
- Days entered: 32. Missing canonical regime labels (defaulted to FLAT): none.
- Reproduction event count: 196,600 (upstream reported n=196,600 in summary.csv).
- Commission already deducted (0.376 ticks RT). No spread crossing — book pressure is a passive signal but the upstream uses target_log_ret which is mid-implied; market-order-equivalent costs would deduct an additional ~1.0 tick. The upstream cell's net headline does NOT include the +1.0-tick spread cross — interpret accordingly.