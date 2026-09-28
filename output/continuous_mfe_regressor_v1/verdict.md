# continuous_mfe_regressor_v1 — VERDICT: NO-GO (continuous-return ALSO dead)

## Headline
- **0 GO cells across all strategies, buckets, gates, SL options** under honest regrade with REAL first-passage SL fills.
- Continuous-return reframe of K-framework FAILS just as hard.
- **Critical structural failure: model predicts MAGNITUDE (Spearman 0.21 on |MFE|) but NOT DIRECTION (Spearman -0.02 on signed terminal).**
- Magnitude-only gating + 1.376t taker cost + real SL fills = catastrophic. High-magnitude trades hit SL faster than they reach terminal MFE.

## Per-horizon (hold_npts quartile bucket) Spearman

| bucket | n | spearman(signed_terminal) | spearman(MFE_magnitude) |
|---|---|---|---|
| ALL | 161,519 | -0.020 | **+0.211** |
| q1_short (hold<772 events) | 40,380 | -0.006 | +0.120 |
| q2 | 40,380 | -0.023 | +0.108 |
| q3 | 40,381 | -0.033 | +0.054 |
| q4_long (hold>15k events) | 40,378 | -0.014 | +0.095 |

**Compare to v3 best AUC** (K=5/S=4 = 0.686 ≈ rank correlation ~0.37): our signed-terminal predictor is WORSE than random for direction. Our MFE-magnitude predictor has rank correlation 0.21 but provides no exploitable edge because it picks the most volatile (both-sided) trades.

## Best regrade cell per strategy (honest, REAL SL via sl{N}_dt_ns / tp{N}_dt_ns)

| strategy | gate | dir | bucket | q | SL | mean_net | PF | Sharpe_d | cond_WR | GO? |
|---|---|---|---|---|---|---|---|---|---|---|
| A_signed | abs(pred_signed) | sign(pred_signed) | q1_short | 2.5% | 1 | **-2.03** | 0.03 | -5.65 | 4% | NO |
| B_mfe_dirSigned | abs(pred_MFE) | sign(pred_signed) | q4_long | 0.1% | 2 | **-5.90** | 0.00 | -12.27 | 0% | NO |
| C_mfe_dirSide | abs(pred_MFE) | side (always) | q1_short | 0.1% | 1 | **-1.36** | 0.34 | -9.22 | 12.5% | NO |

## Why the brief's "naive clip" regrade gave fake GOs and why we discarded it
- Brief's regrade rule: "Realized P&L = realized_mfe_ticks if abs ≤ |predicted|; else clip at -SL" — clips TERMINAL move at -SL.
- This IGNORES intra-trade drawdown that actually triggered SL exit BEFORE terminal.
- Our walks parquet has sl{N}_dt_ns (true first-passage hit time of -N tick barrier in side-favored frame).
- Using REAL SL fills: SL=1 trigger rate is **80.8%** on top-MFE-magnitude trades. The naive clip understated SL hits by ~5x, manufacturing positive E from pure path-dependence noise.
- The v1 of our regrade (naive clip) produced 13 fake GO cells incl. "ALL q=10% sh=14.5". That number is not real.

## ORACLE upper bound (perfect direction, top by |true signed terminal|, REAL SL)
- q=1% SL=1: mean_net = +3.09, cond_WR 19% (still hurts because SL hits a lot)
- q=5% SL=1: mean_net = +1.01, cond_WR 21%
- q=10% SL=2: mean_net = +1.64, cond_WR 38%

**Direction is the bottleneck.** With an oracle direction sign, q=1-5% trades clear taker cost handsomely. Our regressor's Spearman(-0.02) on direction means we have ZERO signed alpha — only volatility-predictability. Volatility-predictability without directional skill cannot beat TAKER 1.376 + first-passage SL fills.

## What's idle / what's running

- Neptune GPU: idle (CMR run finished in 7.6s)
- Razer GPU: MLP v2 training in flight per prior dispatch (~17:45 ETA), untouched
- Jupiter CPU: v3.5 multihead label generation ongoing per prior agents

## Honest blockers
- MLflow log_artifact failed from Neptune again (`Permission denied: /home/jupiter`); MLflow has metrics + run ID `ebf3f5f97b8d48308d14d8f39be279ef` but artifacts only landed locally. Output synced to Jupiter manually.
- Walks parquet only has single-horizon MFE/MAE (per-trade with variable hold_npts up to 15s cap), not multi-h. Bucketed by hold_npts quartile as horizon proxy. Per-bucket Spearman shows magnitude predictability degrades with longer holds (q1=0.12 → q4=0.10) — consistent with signal decay.

## Inputs joined
- per_trade_walks.parquet: 262,726 rows after NA-drop, 15 OOT dates in [20260317..20260428]
- CNN-Mamba v2 1s-horizon predictions joined via merge_asof (backward, 1s tol). Overall coverage **89.59%** (high). 27,338 rows dropped without v2 join.
- Final training set: 161,519 OOS rows across 11 OOT dates (after burn-in 4 fold).

## Bottom line for taker math
Both the K-framework binary first-passage AND the continuous-return regressor have now been comprehensively closed. The 1.376t TAKER round-trip is unbeatable with this feature set (9 v1 head preds + CNN-Mamba v2 1s pred). Predicting direction is the constraint, not predicting magnitude.

## NO-GO + decision escalated to user (per HC #393 escalation criteria)

This is a **genuine novel decision** (not a routine engineering call) per HC #393:
- It contradicts active research direction (continuous regression was the "Option B" suggested by v3 verdict)
- Both K-framework and continuous reframe are now closed
- The next axes have material trade-offs requiring user-level guidance:
  - **RL with non-PPO algo** (TD3? SAC? A2C?): high R&D cost, unknown if action-space framing helps given direction-signal is absent in the features
  - **Wait for Jupiter v3.5 multihead labels** (~26/102 done): right targets in 1-2 days; would let us train a single multi-horizon multi-magnitude head against richer labels, possibly recovering directional skill
  - **Re-feature-engineer**: try v3.5 alpha_labels-v3 or pressure_labels datasets the Mar pipeline produced as features (may carry directional info that v1's MFE/adverse heads do not)
- Per HC #393: escalating to user. No specific action proposed.
