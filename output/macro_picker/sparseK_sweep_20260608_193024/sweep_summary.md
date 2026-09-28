# Sparse-K sweep (HC #565+ — sparse cardinality fix for v5 dense ridge)

Same panel (master_panel_v2, 76-feature pool), hold=10d, train/OOT walk-forward unchanged. Only K (number of features kept per fold by |Spearman IC|) varies.

| K | Pooled Sharpe | Pooled Calmar | Pooled CAGR | MaxDD | Deployable (Calmar>=1) |
|---:|---:|---:|---:|---:|---:|
|  5 | 0.53 | 0.21 | 2.7% | -13.1% | 0/11 |
|  8 | 0.43 | 0.16 | 2.2% | -13.8% | 0/11 |
| 10 | 0.36 | 0.14 | 1.9% | -13.1% | 0/11 |
| 15 | 0.35 | 0.14 | 1.8% | -13.0% | 0/11 |

## Top-5 picked features per K

### K=5
- `flag_accounting_change`: picked 64x across folds-x-sectors
- `flag_restatement`: picked 41x across folds-x-sectors
- `rv_cc_252d`: picked 40x across folds-x-sectors
- `fp_margin_trend_4q`: picked 34x across folds-x-sectors
- `rv_pk_20d`: picked 34x across folds-x-sectors

### K=8
- `flag_accounting_change`: picked 80x across folds-x-sectors
- `rv_cc_252d`: picked 59x across folds-x-sectors
- `fp_market_cap_pit`: picked 54x across folds-x-sectors
- `flag_restatement`: picked 50x across folds-x-sectors
- `rv_cc_60d`: picked 47x across folds-x-sectors

### K=10
- `flag_accounting_change`: picked 86x across folds-x-sectors
- `rv_cc_252d`: picked 74x across folds-x-sectors
- `fp_market_cap_pit`: picked 63x across folds-x-sectors
- `fp_margin_trend_4q`: picked 61x across folds-x-sectors
- `fp_ni_ttm`: picked 59x across folds-x-sectors

### K=15
- `flag_accounting_change`: picked 101x across folds-x-sectors
- `rv_cc_252d`: picked 99x across folds-x-sectors
- `rv_pk_20d`: picked 88x across folds-x-sectors
- `rv_cc_60d`: picked 86x across folds-x-sectors
- `fp_eps_ttm`: picked 84x across folds-x-sectors

_total wall: 124.5s_