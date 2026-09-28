# Wheel LGBM Assignment Risk Ranker v1 — Report

Walk-forward: 24m train / 1m OOT / 1m step (SLIDING, HC #0)
Features: 36 (PIT + IV + momentum + beta + max_dd + VIX + sector dummies)
Top-K safe tickers per rebalance: 20  (ranked by LOWEST predicted assignment probability)

## OOT Information Coefficient
- Mean Spearman IC (vs binary assignment event): 0.125 ± 0.109
- Mean Spearman IC (vs max adverse drawdown):   0.072
- Folds with positive IC: 32 / 35

## Assignment Hit Rate
- Universe assignment rate:  46.0%
- Selected basket rate:      30.1%
- Reduction vs universe:     15.8% pp  (PASS — need >3pp)

## Regime Split (HC #428 R1)
- Green days — selected basket assignment rate: 24.2%
- Red days   — selected basket assignment rate: 27.5%
- Regime skew |red-green|/max = 0.121  [PASS — threshold 0.50]

## Comparison vs Yield Ranker v1
- Yield-basket assignment rate:  39.8%
- Safety-basket assignment rate: 30.1%
- Improvement: 9.6% pp reduction in assignment events

## Top Features (full-sample importance)
  r_1m                                 471
  vix_close                            181
  term_ratio                           157
  vix_z60                              129
  fund_pit_ni_growth_z                 117
  fund_pit_fcf_yield_z                 114
  fund_pit_roe_z                       106
  iv_rv_ratio                          105
  fund_pit_de_z                        102
  fund_pit_fcf_growth_z                98

## Verdict
DEPLOY-CANDIDATE: positive IC, assignment rate reduction confirmed, regime symmetry PASS. Replace or supplement yield ranker.