# Wheel LGBM Ticker Ranker v1 — Report

Walk-forward: 24m train / 6m OOT / 3m step (SLIDING, HC #0)
Features: 24 (PIT fundamentals + IV + price momentum)
Top-K per rebalance: 20

## OOT Information Coefficient
- Mean Spearman IC: 0.525 ± 0.107
- Folds with positive IC: 10 / 10

## Selection Alpha (vs Universe)
- Months evaluated: 33
- Mean selected 30d return: 7.74%
- Mean universe 30d return: 2.26%
- Mean excess 30d return: 5.48%
- Annualized Sharpe (selected basket): 2.11
- Annualized Sharpe (excess vs universe): 1.85
- Monthly hit rate (selected > universe): 66.7%

## Verdict
DEPLOY-CANDIDATE: IC positive, excess Sharpe > 0.30, hit rate > 55%. Replace static fund_score with LGBM ranker.