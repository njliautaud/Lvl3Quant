# Sparse-K sweep (HC #565+ — sparse cardinality fix for v5 dense ridge)

Same panel (master_panel_v2, 76-feature pool), hold=10d, train/OOT walk-forward unchanged. Only K (number of features kept per fold by |Spearman IC|) varies.

| K | Pooled Sharpe | Pooled Calmar | Pooled CAGR | MaxDD | Deployable (Calmar>=1) |
|---:|---:|---:|---:|---:|---:|
|  5 | 0.30 | 0.11 | 1.5% | -13.9% | 0/11 |
|  8 | 0.27 | 0.09 | 1.3% | -13.8% | 0/11 |
| 10 | 0.34 | 0.13 | 1.7% | -13.0% | 0/11 |
| 15 | 0.19 | 0.06 | 0.9% | -14.9% | 0/11 |

## Top-5 picked features per K

### K=5
- `flag_accounting_change`: picked 63x across folds-x-sectors
- `rh_sentiment_volatility`: picked 38x across folds-x-sectors
- `flag_restatement`: picked 38x across folds-x-sectors
- `rv_cc_252d`: picked 37x across folds-x-sectors
- `rv_pk_20d`: picked 34x across folds-x-sectors

### K=8
- `flag_accounting_change`: picked 73x across folds-x-sectors
- `rv_cc_252d`: picked 56x across folds-x-sectors
- `rh_sentiment_volatility`: picked 50x across folds-x-sectors
- `fp_market_cap_pit`: picked 49x across folds-x-sectors
- `flag_restatement`: picked 47x across folds-x-sectors

### K=10
- `flag_accounting_change`: picked 77x across folds-x-sectors
- `rv_cc_252d`: picked 67x across folds-x-sectors
- `fp_market_cap_pit`: picked 61x across folds-x-sectors
- `rh_sentiment_volatility`: picked 58x across folds-x-sectors
- `flag_restatement`: picked 55x across folds-x-sectors

### K=15
- `flag_accounting_change`: picked 95x across folds-x-sectors
- `rv_cc_252d`: picked 92x across folds-x-sectors
- `rv_cc_60d`: picked 77x across folds-x-sectors
- `rv_yz_20d`: picked 76x across folds-x-sectors
- `rv_pk_20d`: picked 73x across folds-x-sectors

_total wall: 130.6s_