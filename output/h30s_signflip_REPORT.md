# HC #494 R3 — H30s SIGN-FLIP Sanity Check: FINAL REPORT

**Date**: 2026-05-28 19:54 ET
**Status**: COMPLETE
**Verdict**: **OUTCOME 2 — Signal genuinely absent. h30s retrain confirmed FIFO-dead.**

---

## Lead Finding

**Sign-flipping the predictions did NOT rescue the signal.** All 12 cells remain FIFO-net-negative at very similar magnitude to the original sweep. The original (unflipped) result was NOT a label-sign inversion bug; the model simply has no exploitable edge at any horizon × confidence combination.

## Side-by-Side: Original vs Sign-Flipped (Full 46-day OOT)

| Horizon | Top % | Original Net Ticks | Sign-Flipped Net Ticks | Δ | Conclusion |
|---------|-------|-------------------|------------------------|---|------------|
| 1s  | 5%  | -16,492 | -17,885 | -1,393 | Worse |
| 1s  | 10% | -34,680 | -35,178 | -498 | Worse |
| 1s  | 20% | -69,430 | -70,592 | -1,162 | Worse |
| 5s  | 5%  | -14,711 | -16,223 | -1,512 | Worse |
| 5s  | 10% | -33,276 | -30,125 | +3,151 | Marginally better, still FAIL |
| 5s  | 20% | -68,704 | -66,391 | +2,313 | Marginally better, still FAIL |
| 10s | 5%  | -15,934 | -16,757 | -823 | Worse |
| 10s | 10% | -34,718 | -30,844 | +3,874 | Marginally better, still FAIL |
| 10s | 20% | -69,477 | -64,283 | +5,194 | Marginally better, still FAIL |
| 30s | 5%  | -21,370 | -17,915 | +3,455 | Marginally better, still FAIL |
| 30s | 10% | -41,789 | -30,792 | +10,997 | Notably better, still FAIL |
| 30s | 20% | -80,523 | -64,406 | +16,117 | Notably better, still FAIL |

**Best sign-flipped cell**: `30s_top10%` net **-30,792 ticks**, 11/46 positive days, Sharpe -78.7. **Still FAIL.**

**Symmetric loss pattern**: both sides lose ~similar magnitude. Commission drag (0.376 ticks × ~50K-200K trades = 19K-75K ticks of pure cost) dominates either direction. There is no edge to recover.

## Interpretation

1. **No sign-inversion bug**. Training and inference pipeline labels are oriented correctly. The model's predictions are simply uninformative about within-horizon price direction once you account for the realistic FIFO cost structure.

2. **Slight asymmetry at longer horizons (30s)** — sign-flipped 30s cells lose ~30% less than originals. This is consistent with the model having captured **a tiny long-bias signal at 30s** (i.e., predictions are slightly more useful as long signals than short signals), but the effect is far too small to overcome 0.376 ticks/trade commission. Margin of ~16K ticks improvement on ~200K trades = 0.08 ticks/trade. Noise.

3. **The original "1-12/46 positive days" anti-correlation was sample-size noise**, not bug evidence. Daily aggregation of many tiny noisy trades + commission drag = nearly-deterministic loss days regardless of sign.

4. **HC #494 R1 viability bar**: still 0 of 12 cells pass on either sign.

## Outcome Selection (per task spec)

- ❌ Outcome 1 (mirror positive) — REJECTED
- ✅ **Outcome 2 (still negative both sides)** — CONFIRMED
- ❌ Outcome 3 (mixed) — REJECTED

## Implications

- **No v7-production-Razer audit needed.** No bug was found, so there is nothing to propagate-check on Razer's v7 model.
- **h30s longer-horizon retrain is officially KILLED.** Both directions tested. No alpha recoverable.
- **Recommendation re-affirmed**: proceed to **Branch B (h5s-target retrain)** — closer to CNN-Mamba's natural edge horizon per the 2026-05-01 decay analysis. The h=30s target trained the model to predict noise.

## Methodology

- Same sweep script as `h30s_fifo_sweep.py`, only modification: `preds = -data["predictions"]` at load time.
- Same 46 OOT days (2026-02-24 → 2026-04-27), same MBO event labels, same 0.376-tick commission, same passive-limit assumption.
- Selection logic unchanged (top-pct most negative AFTER sign-flip = top-pct LONGS in original space).

## Files

- Sweep script: `/home/nick/Lvl3Quant/scripts/h30s_signflip_sweep.py`
- Per-cell CSV: `/home/jupiter/Lvl3Quant/output/h30s_signflip/summary.csv`
- Neptune run log: `/home/nick/Lvl3Quant/output/h30s_signflip/run.log`

## MLflow

Experiment: `h30s_signflip` on http://localhost:5000 — all 12 sign-flipped cells logged with net_ticks, sharpe, positive_days, trades_per_day, verdict.
