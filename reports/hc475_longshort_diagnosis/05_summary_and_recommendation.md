# 05 — Summary, Attribution, and Recommendation

## Final attribution (HC #475 R1)

| Root cause | Attribution | Verdict |
|---|---|---|
| (a) Signal asymmetry — model found real short edge | **~10%** | Mostly false. Model's LONG-side IC > SHORT-side IC at 1s/5s/10s. Short-side IC is near zero or negative. |
| (b) Label asymmetry — training labels are short-skewed | **~5%** | Rejected. Labels are within ±5pp of balanced at every horizon. Not a bug. |
| (c) Threshold asymmetry — top-quantile policy selects the negative tail | **~75%** | The dominant cause. The prediction-magnitude distribution is heavily negatively skewed at the trading horizons, so "top X% by magnitude per tail" + confluence collapses the long side to ~0 triggers. |
| (d) Execution asymmetry — passive-at-touch kills the long fills | **~10%** | Final amplifier. Drops the ~90 long triggers/16 days to 0 fills. Important secondary effect but not the primary cause. |

**Net headline (HC #474 risk-adjusted style): the 99.84% short-fill ratio is a model+policy artifact, not a real edge.**

## Both-sides competency gate (HC #475 R2)

The v3.4.2 multi-head model **FAILS** the both-sides competency bar at the 10s trading horizon:
- |IC_long| = 0.0217
- |IC_short| = 0.0100
- Ratio = 0.46 — below the required 0.5×.
- Short-side IC has the WRONG SIGN (negative at 5s and 10s).

Per R2, the model is rejected for production "trade both sides" use.

A short-only DEPLOY (R3) is also dubious here because the short side is the WEAKER side, not the stronger one — short-only would be deploying the worse of two anti-correlated signals.

## What we KNOW caused the bias

1. The model emits 2.17× more long than short predictions at 10s (correctly reflecting ES long-drift).
2. BUT when it emits a SHORT, the magnitude is meaningfully larger than when it emits a LONG (negative-tail p99 is 1.21–1.59× the positive-tail p99 at 1s/5s/10s).
3. Selection policies that say "top 5% by magnitude per direction" plus "all heads agree" mathematically force ~99% short selection.
4. Execution mechanics (passive-at-touch into the long-drift) kills the remaining 90 long triggers.

This is consistent with the user's prior: "the model OUGHT to predict long more often than short, because ES drifts up". The model in fact does predict long more often. We are then throwing away the long predictions in our trade-selection and execution layers.

## Top recommended next action

**Retrain the directional heads with sign-balanced loss + magnitude-calibrated targets, then re-evaluate.** Specifically:

- Replace the regression-MSE loss with a sign-balanced MSE (re-weight so the positive-tail and negative-tail magnitudes contribute equal gradient mass), OR add an explicit asymmetric-loss penalty that pulls the predicted-magnitude distribution toward symmetry.
- Train continuation-style targets per HC #470 R1 (stream-integrated returns, sign-stability) so the head learns "magnitude until reversal" not "snapshot return", which by construction tends to be more symmetric than snapshot returns.
- Validate after retraining by running this exact diagnostic again — competency ratio ≥ 0.5× and trigger-stage short-share between 40-60% are the acceptance gates.

This is HC #475 R4-compliant alpha redevelopment, with execution-tailored output design.

## Two interim policy fixes (cheap, can run TODAY without retraining)

If retraining is the right strategic move but takes days, then these two policy changes can be applied to the existing v3.4.2 model immediately for a Friday-EOD tradable read:

1. **Replace top-X%-per-direction with a calibrated probability gate.** Use the LGBM-vol or auxiliary `p_up` heads to compute calibrated probabilities, then trigger when `P(realized > k ticks) ≥ p_min` regardless of magnitude. This decouples gate selectivity from magnitude skew.
2. **Symmetric absolute-magnitude threshold.** Pick a single threshold `k` such that |pred| ≥ k generates a balanced number of long and short triggers (empirically tune until short-share ∈ [0.4, 0.6]). This is a one-line change in the FIFO sweep code.

Both can be done without GPU. Both can be A/B-tested against the current short-only setup on the 16-day OOT to give a Friday read on whether longs are actually tradable.

## Short-only deploy decision

Per HC #475 R3, short-only DEPLOY would require:
- (a) R1 diagnosis complete — **DONE** in this document.
- (b) Diagnosis shows a real, not pathological reason for short-only — **FAILS**. The reason is "threshold policy is broken", not "short edge is real and long edge is dead".
- (c) HC #428 R1 regime-agnostic test passes for shorts alone — current FIFO summary shows |Sharpe_green − Sharpe_red| / max = 0.91 to 1.87 for the high-fill configs → **FAILS regime gate**.
- (d) HC #474 R2 win-rate floor met (55% unless Sharpe ≥ 2.0 + PF ≥ 1.8) — best surviving config (trip10) has WR 57.4% but Sharpe only +0.42 → **TECHNICAL PASS on WR alone**, FAILS the broader risk-adjusted bar.

**Recommendation: do NOT short-only deploy.** Take the interim policy fix path, get a balanced read by Friday EOD, and pursue HC #475 R4 alpha redevelopment in parallel.

## Specific files / lines responsible

There is no "label-construction bug" — labels are fine. The responsible engineering surface is:
- The trade-trigger policy: `scripts/surviving_confluence_canonical_fifo.py::confluence_mask` (lines 100-141) — uses per-tail top-quantile selection, the policy that interacts pathologically with the model's asymmetric magnitude distribution.
- The model's loss function: the v3.4.2 dispatch (`dispatch_v34_2_fixedmtl.py` + `train_cnn_mamba_v3_2.py`) trains with unweighted MSE on log-return targets. There is no explicit penalty against the predicted-magnitude distribution being asymmetric. That's where R4 retraining should focus.

## Caveats / what's NOT covered

- Only 16 OOT days available — HC #428 R1's 40-day floor not met. This diagnostic is directionally robust but the numerical attribution percentages should be re-checked once the full ~40-day OOT inference lands.
- The signed IC computation restricts to the matching sub-sample (e.g., long-side IC uses only `pred > 0` events). This is the "is the model right when it says X?" view. An alternative is to compute IC across the WHOLE sample using only sign of prediction — that's a different question (binary directional skill).
- Per HC #472 R1, the per-day IC was reported in Report 02; no significant decay slope observed across the 16 OOT days. Decay is not driving the bias.
- The execution analysis assumes the existing canonical-FIFO engine is correctly implemented. A separate question.

## Quick-reference numbers (for the message to user)

- Raw model predicts 68.5% long / 31.5% short at 10s — consistent with ES long-drift.
- After confluence + top-5% magnitude gate: 0.5% long / 99.5% short triggers across the surviving configs.
- After canonical FIFO fill: 0% long / 100% short fills (14,107 short / 0 long across the 5 configs in this diagnostic; matches the 20,939 figure across all 9 surviving configs).
- Long-side IC is the BETTER side (0.022 vs 0.010 at 10s).
- Label distribution: balanced (±5pp from 50/50 at every horizon).
- **Conclusion: the model is mildly long-biased in skill but heavily short-biased in selection. The trade-trigger policy is broken, not the model.**
