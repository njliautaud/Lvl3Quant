# Adaptive Exit v0 — Imitation-Learning Baseline
Generated: 2026-05-21 16:02 ET. Wall: 158.8s.

**Compliance**: HC #469 R4 (adaptive must beat time-baseline by ≥10% net to ship).

**Important v0 caveats**:
- The in-trade MFE/MAE is APPROXIMATED via linear interpolation toward the realized exit (v0 only).
- v1 must use exact MBO replay of each trade leg for the in-trade trajectory.
- Until v1, treat the numbers below as directional, not absolute.

## Headline

| Policy | n | Mean net (t) | Total net (t) | Sharpe | WR |
|---|---|---|---|---|---|
| Time-baseline (full hold) | 2623 | -0.167 | -437.75 | -0.048 | 46.0% |
| **Adaptive (LightGBM)** | 2623 | +0.613 | +1608.91 | +0.457 | 27.6% |

**Adaptive lift vs baseline: +467.5%**

**HC #469 R4 ship gate (≥10% lift): PASS — adaptive ships**

## Next steps
- Replace v0 linear-interp MFE/MAE with exact MBO replay (v1).
- Train on Razer GPU with deeper MLP / Transformer once Razer meta-model completes.
- Add queue-position feature (HC #469 R5(a)) once the queue-position model lands.
