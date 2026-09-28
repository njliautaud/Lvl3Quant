# Trade Classifier v1 — Report

Candidate: **short_10s_thr55** (FIFO realized survivor)
Trades joined to features: **3,078**
WF folds run: 4 (sliding, time-ordered trade rows)

## OOS AUC (LightGBM): 0.5352 ± 0.1183
OOS AUC (LogReg robustness): 0.4882

### Per-fold
| fold | n_tr | n_va | n_te | auc_lgbm | auc_lr | base_rate | best_iter |
|------|------|------|------|----------|--------|-----------|-----------|
| 0 | 522 | 93 | 615 | 0.4014 | 0.4025 | 0.800 | 27 |
| 1 | 522 | 93 | 615 | 0.4356 | 0.3303 | 0.524 | 60 |
| 2 | 522 | 93 | 615 | 0.6305 | 0.6768 | 0.766 | 72 |
| 3 | 522 | 93 | 615 | 0.6733 | 0.5432 | 0.582 | 1 |

## Gated P&L Sweep (OOS only)
| label | tau | n | tpt | wr | profit_days | total_days | pdays_ratio | sharpe |
|-------|-----|---|-----|----|----|----|----|----|
| baseline (all OOS trades) | — | 2460 | 4.487 | 0.668 | 5 | 7 | 0.714 | 5.30 |
| clf_prob>=0.4 | 0.40 | 2140 | 4.701 | 0.668 | 5 | 7 | 0.714 | 5.28 |
| clf_prob>=0.45 | 0.45 | 2114 | 4.687 | 0.667 | 5 | 7 | 0.714 | 5.24 |
| clf_prob>=0.5 | 0.50 | 2076 | 4.624 | 0.662 | 5 | 7 | 0.714 | 5.14 |
| clf_prob>=0.55 | 0.55 | 2034 | 4.571 | 0.657 | 5 | 7 | 0.714 | 5.06 |
| clf_prob>=0.6 | 0.60 | 1991 | 4.678 | 0.655 | 5 | 7 | 0.714 | 5.25 |
| clf_prob>=0.65 | 0.65 | 1942 | 4.706 | 0.651 | 5 | 7 | 0.714 | 5.28 |
| clf_prob>=0.7 | 0.70 | 1897 | 4.704 | 0.646 | 4 | 7 | 0.571 | 5.25 |

## Verdict: **REJECT**
Reasons:
- auc_mean=0.535<0.55
- tpt_pooled=4.71<5.0
- tpt_pooled drops below baseline 5.03

Best tau on OOS: **0.65** → tpt=4.706, pdays_ratio=0.714, n_trades=1942, sharpe=5.28

## Top features (sum LGBM gain across folds)
f_ofi_10s_now(29.7%), mins_from_open(29.6%), f_mean_abs_pred_K20(10.6%)

## Honest Caveats
- 3,078 realized trades (short_10s @ thr=0.55) across 15 OOT-day pool. Sample is large enough that LGBM is fitting *trade-level* noise; this is in-sample-fold CV, not true forward-only future.
- The original 4 folds (from the meta booster) are NOT respected here. Trade-classifier uses a NEW 5-fold sliding split on signal_ts_ns. Each test slice is later in time than its train slice.
- meta_prob is re-derived by calling the *original fold's* meta booster on the meta-layer features at the signal event. So meta_prob entering the trade classifier is causal w.r.t. the meta model that generated the trade.
- Microstructure features used here are the 20 already-precomputed meta_layer_v1 features (sign-consistency, OFI, spread, queue_imb at signal). They were precomputed strictly causally upstream.
- LGBM overfit profile differs from 15-day classifier: at thousands of samples, gradient boosting can find genuine micro-edges but also memorize trade-cluster signatures. LogReg AUC is the robustness anchor.
- Per-trade alpha (baseline +5.03 t/trade FIFO realized, HC #428) is the floor. Any tau that drops pooled tpt below baseline is rejected even if profit-days ratio improves.