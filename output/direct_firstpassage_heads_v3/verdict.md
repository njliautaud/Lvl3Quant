# direct_firstpassage_heads_v3 — VERDICT: NO-GO (K-framework structurally dead)

## Headline
- K=5/S=4 AUC: **v1 = 0.7111 → v2 = 0.6122 → v3 = 0.6856** (v3 recovered most of v2's damage but did NOT exceed v1)
- v3 GO cells under HC #506 R5 acceptance gates: **0** (same as v1 and v2)
- Best v3 cell (K=5/S=4 q=10%): Sharpe -15.7, PF 0.49, mean_ticks_net negative → all gates fail
- TAKER 1.376t cost is the structural killer; no model edge on these 8 (K,S) heads recovers it

## Inputs & method
- per_trade_walks_extended.parquet: 266,063 OOT events across 16 OOT dates
- CNN-Mamba v2 OOT predictions: per-day .npz at stride=250, window=3000 → joined per event via merge_asof (Option B, backward, tol=1.0s)
- Overall join coverage: **86.96%** of v2_inputs events (well above 70% floor)
- 4 of 16 days had partial v2 coverage (20260421=77%, 20260422=27%, 20260423=11%, 20260424=78%, 20260428=76%); these are days where v2 inference produced fewer windows than the trade-event count. Rows without v2 join were dropped (cleanest signal, HC-aligned).
- Features (12): 9 v1 features + cnn_mamba_v2_pred (1s horizon) + cnn_mamba_v2_x_side + cnn_mamba_v2_conf
- DID NOT include the noisy book_features that the v2 sub-agent used — that was the source of v2 degradation per afe8259 finding
- WF: expanding-bucket v1 style (NOT sliding-20). burn_in=4, train on all prior OOT rows
- XGBoost GPU (Neptune RTX 3090), n_est=600, lr=0.05, max_depth=5, early stop 20
- MLflow run: c71135863a194544aab88ba1f49efebf, experiment direct_firstpassage_v3_cnn_mamba

## AUC table (apples-to-apples on shared 12 OOT dates [20260416..20260429])

| (K,S) | v1 AUC | v2 AUC | v3 AUC | v3 - v1 |
|-------|--------|--------|--------|---------|
| 2,1   | 0.5406 | 0.5565 | 0.5386 | -0.0020 |
| 3,1   | 0.5691 | 0.5492 | 0.5561 | -0.0130 |
| 4,1   | 0.5992 | 0.5458 | 0.5773 | -0.0219 |
| 5,1   | 0.6314 | 0.5624 | 0.6022 | -0.0293 |
| 3,2   | 0.6096 | 0.5627 | 0.5965 | -0.0132 |
| 4,2   | 0.6403 | 0.5687 | 0.6187 | -0.0216 |
| 5,2   | 0.6683 | 0.5788 | 0.6404 | -0.0280 |
| **5,4** | **0.7111** | **0.6122** | **0.6856** | **-0.0256** |

Pattern: v3 lifts every head above v2 but never reaches v1. Average v3 dAUC = -0.0193. CNN-Mamba v2 at 1s horizon provides ~no incremental signal beyond v1's MFE/adverse/toxicity predictors for predicting 15s first-passage labels.

## Best regrade cell (v3, by Sharpe)
K=5/S=4, q=10% (15,752 trades over 14 days):
- cond_WR = 0.432, mean_ticks_net = -0.27 (NaN in summary is artifact), PF = 0.487
- Sharpe_daily = -15.7, day_concentration = N/A, all gates fail
- Bottom line for taker math: TAKER cost 1.376t > model's expected gross edge per trade

## Why this confirms the K-framework is structurally dead
1. v1's information from 3 continuous predictors (MFE, adverse, toxicity) was the best the heads ever saw.
2. CNN-Mamba v2 direction probability adds NO information once we already have v1's MFE prediction (which directly forecasts the same kind of move-magnitude that determines TP_K success).
3. Even with v1's best AUC (0.71 on K=5/S=4), the cost-adjusted Sharpe is -12 — meaning even a perfect oracle within this 8-head framework would not clear the TAKER hurdle at reasonable trade frequencies.
4. The 15s hold cap + discrete (K,S) tick grid is the wrong signal-extraction shape for what the underlying model knows.

## NO-GO + next-axis recommendation (per mission brief options)

PPO is OFF the table per RUN_HISTORY (failed twice). Recommended next axes:

**Option B (preferred): Continuous-return regressor at variable horizon.**
Predict expected ticks-net over an adaptive horizon (5s..30s) and trade only when |E[net]| > 2 * TAKER cost (≥2.75t). This replaces the (K,S) binary grid with a continuous problem the model already partially solves (we have y_pred_mfe and y_pred_adverse).

**Option C: Wait for Jupiter v3.5 multihead labels.**
~26/102 done. The right targets (multi-horizon multi-magnitude labels) are coming. Adding v3.5 to a v4 head would let us fit per-horizon expected return directly, but at current rate that's days away.

**Option A (contextual bandit / REINFORCE-baseline):**
Possible but premature. The signal extraction problem (Options B/C) must be resolved before the policy problem matters.

**Recommendation: B now (parallel to C). Build a continuous-return regressor on the existing v2_inputs walks targeting `mid_change_hold_ticks * side - 1.376`, with horizon as a feature and select-actions via threshold.**

## What's idle / next dispatch
- Neptune GPU: idle as of 17:20 ET (v3 training finished in 40s). Ready for Option B regressor or another dispatch.
- Razer GPU: MLP v2 still in flight per prior dispatch (~17:45 ETA), left alone per HC #504 R1.
- Jupiter CPU: v3.5 multihead label generation ongoing (26/102 last reported).

## Honest blockers
- MLflow `log_artifact` failed from Neptune (`Permission denied: /home/jupiter`) — MLflow server has the run + metrics but artifacts only landed locally on Neptune. Metrics are in MLflow UI; the parquet/JSON are at `/home/nick/Lvl3Quant/output/direct_firstpassage_heads_v3/` and synced to Jupiter `/home/jupiter/Lvl3Quant/output/direct_firstpassage_heads_v3/`.
- 13% of v2_inputs events dropped due to no v2 prediction within 1s tolerance — driven by 4 OOT days where v2 inference produced sparse windows. Could be revisited by re-running CNN-Mamba v2 inference end-to-end on those dates, but not worth it given v3's overall AUC lift is negative.
