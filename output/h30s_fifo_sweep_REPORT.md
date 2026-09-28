# HC #494 R3 — H30s FIFO Sweep: FINAL REPORT

**Date**: 2026-05-28 19:41 ET
**Status**: COMPLETE — h30s longer-horizon retrain is **FIFO-DEAD**
**Verdict**: All 12 cells FAIL HC #494 R1 viability bar

---

## Lead Finding (HC #494 R4)

**Every cell of the 4 horizons × 3 confidence band sweep is FIFO-net-negative across 46 OOT days.**

The CNN-Mamba v3 h30s-target retrain does NOT survive canonical FIFO replay. The signal that the model learns at h=30s does not translate to executable profit at any horizon-confidence combination under realistic queue-aware fills with the canonical 0.376-tick commission cost.

## Results: All 12 Cells (Full 46-day OOT, 2026-02-24 → 2026-04-27)

| Horizon | Top % | Net Ticks | Trades | Days | Pos Days | Trades/Day | Sharpe | Verdict |
|---------|-------|-----------|--------|------|----------|------------|--------|---------|
| 1s  | 5%  | **-16,492** | 48,153  | 46 | 4 | 1,047 | -387.4 | FAIL |
| 1s  | 10% | **-34,680** | 95,737  | 46 | 2 | 2,081 | -413.7 | FAIL |
| 1s  | 20% | **-69,430** | 191,089 | 46 | 1 | 4,154 | -414.1 | FAIL |
| 5s  | 5%  | **-14,711** | 47,734  | 46 | 4 | 1,038 | -166.0 | FAIL |
| 5s  | 10% | **-33,276** | 95,233  | 46 | 5 | 2,070 | -192.4 | FAIL |
| 5s  | 20% | **-68,704** | 190,443 | 46 | 4 | 4,140 | -199.4 | FAIL |
| 10s | 5%  | **-15,934** | 47,239  | 46 | 4 | 1,027 | -139.8 | FAIL |
| 10s | 10% | **-34,718** | 94,746  | 46 | 4 | 2,060 | -152.2 | FAIL |
| 10s | 20% | **-69,477** | 189,991 | 46 | 3 | 4,130 | -150.5 | FAIL |
| 30s | 5%  | **-21,370** | 47,192  | 46 | 12 | 1,026 | -108.4 | FAIL |
| 30s | 10% | **-41,789** | 94,173  | 46 | 11 | 2,047 | -110.0 | FAIL |
| 30s | 20% | **-80,523** | 189,769 | 46 | 9 | 4,125 | -105.9 | FAIL |

**Best cell** (least bad): `5s_top5%` — net -14,711 ticks, 4/46 positive days, Sharpe -166. Still a clear FAIL.

## HC #494 R1 Viability Bar Check

A cell PASSES only if ALL conditions met:
- ✗ Net ticks > 0 → ALL CELLS NEGATIVE
- ✗ ≥30 OOT positive days → max positive days = 12/46 (30s_top5%)
- ✗ Sharpe ≥ 1.0 → all cells deeply negative
- — Regime skew ≤ 0.50 → cannot evaluate (no regime parquet on Neptune; all-green fallback used)
- ✗ Trades/day ≥ 5 → ALL cells fire 1000+ trades/day at top-5% — over-firing relative to typical execution bar

## Key Observations

1. **30s cells have most positive days (9-12 of 46)** but still FIFO-net-negative — model has *some* directional information at its native horizon, but not enough to overcome the 0.376-tick commission cost even with the maximum-edge passive-limit assumption.

2. **Net P&L worsens as confidence band widens (5% → 20%)** at every horizon — high-confidence shorts perform marginally less badly than the broader signal pool, but still lose. No "tail-of-confidence" alpha.

3. **The signal direction may even be wrong**: positive days are 1-12 out of 46 (2-26%). Random would be ~23 (50%). The model is systematically *anti-correlated* with realized within-horizon move, or the labels are being interpreted with inverted sign convention. **Worth a sign-flip sanity check before declaring fully dead.**

4. **Over-firing**: top-5% = 1,000+ trades/day. Real execution stack cannot handle this. Even if signal were viable, position-sizing & cooldown gates would be needed.

## Verdict (Plain English)

**The h30s longer-horizon CNN-Mamba v3 retrain is FIFO-DEAD.** The model's predictions at every horizon (1s, 5s, 10s, 30s) and every confidence band (top-5%, 10%, 20%) lose money under canonical FIFO replay across all 46 OOT days.

## Recommended Next Branches

### Branch A: Sign-Flip Sanity Check (1-2 hours, fastest)
- The pattern of 1-12/46 positive days (well below 50%) suggests possible label-sign mismatch.
- Re-run the sweep with `top-pct LONGS` (most positive predictions) on the same NPZ.
- If LONGS produce mirror-image POSITIVE results → label inversion bug, fix and re-evaluate.
- If LONGS also negative → signal is truly absent.

### Branch B: h5s-Target Retrain (1-2 days, established baseline)
- CNN-Mamba v2 at h=1s/5s/10s already proven (IC_1s=0.222, IC_5s=0.141, IC_10s=0.106).
- Decay analysis (2026-05-01) showed signal strongest at h=1s. The h30s target is **past the model's natural edge horizon** per HC #428 R2.
- Retrain v3 architecture at h=5s with same MFE/MAE-bounded targets.

### Branch C: Confluence-Gated h30s (2-3 days, novel)
- Use existing CNN-Mamba v2 (h=1s) + PatchTST + LGBM-Vol confluence to gate the h30s signal.
- Only trade h30s when confluence agrees → trade count drops 90%+, but expected edge per trade rises.
- Risk: still anchored to a dead signal. Lower priority than Branch B.

### Recommendation: **Branch A → Branch B**.

Branch A is a 30-minute sanity check that could rescue this entire effort if it's a sign bug. If Branch A confirms signal is dead, move to Branch B (retrain at h=5s) since h=5s is closer to the model's natural edge horizon per the decay analysis.

---

## Methodology

- **Predictions**: `/home/nick/Lvl3Quant/output/cnn_mamba_v3_h30s_56day_inference/predictions_56day.npz` (2,392,379 preds × 4 horizons)
- **MBO events**: `/home/nick/Lvl3Quant/data/processed/mbo_events_smart_v3/` (per-day NPZ with labels_1s/5s/10s/30s)
- **Execution model**: Passive limit at touch (HC #74 FIFO baseline), commission = 0.376 ticks RT
- **Label interpretation**: realized ES tick move within horizon (sign: positive = long-profitable, negative = short-profitable)
- **Confidence selection**: top-pct most negative predictions per horizon (short side per HC decay analysis)
- **Ran on**: Neptune (data locality, ~10 min wall time)

## Files

- Sweep script: `/home/jupiter/Lvl3Quant/scripts/h30s_fifo_sweep.py` (also at `/home/nick/Lvl3Quant/scripts/`)
- Per-cell CSV: `/home/nick/Lvl3Quant/output/h30s_fifo_sweep/summary.csv`
- Run log: `/home/nick/Lvl3Quant/output/h30s_fifo_sweep/run.log`
- Predictions: `/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_h30s_56day_inference_FETCHED.npz`

## MLflow

Experiment: `h30s_fifo_sweep` on http://localhost:5000 — all 12 cells logged with net_ticks, sharpe, positive_days, trades_per_day, verdict.
