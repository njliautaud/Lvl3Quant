# LGBM meta-gate v1 — Regime-Stratified Threshold Sweep Verdict

Source sweep: `/home/jupiter/Lvl3Quant/output/lgbm_meta_gate_v1/threshold_sweep_per_day.csv`
Regime source: `/tmp/oot_dates_regime_renamed.parquet`
Output: `/home/jupiter/Lvl3Quant/output/lgbm_meta_gate_v1/regime_stratified_sweep.csv`

## Per-threshold mean per-day Sharpe by regime

|   threshold |     all |    flat |   green |     red |   regime_imbalance |
|------------:|--------:|--------:|--------:|--------:|-------------------:|
|        0.1  | -13.784 |  -2.464 | -15.757 | -12.381 |              0.214 |
|        0.15 |  -9.398 |  -1.657 | -11.242 |  -7.614 |              0.323 |
|        0.2  |  -3.353 |  -1.367 |  -4.032 |  -2.553 |              0.367 |
|        0.25 |  -0.648 | nan     |  -1.175 |   0.3   |              1.255 |
|        0.3  |  -0.777 | nan     |  -1.313 |   0.188 |              1.143 |
|        0.35 |  -2.675 | nan     |  -3.343 |   0     |              1     |
|        0.4  |   0     | nan     |   0     | nan     |            nan     |
|        0.45 | nan     | nan     | nan     | nan     |            nan     |
|        0.5  | nan     | nan     | nan     | nan     |            nan     |

## Trade counts by regime

|   threshold |    all |   flat |   green |    red |
|------------:|-------:|-------:|--------:|-------:|
|        0.1  | 434131 |   1632 |  291083 | 141416 |
|        0.15 | 199165 |   1057 |  146587 |  51521 |
|        0.2  |  27735 |     15 |   23657 |   4063 |
|        0.25 |   1046 |      0 |     849 |    197 |
|        0.3  |    124 |      0 |      91 |     33 |
|        0.35 |     30 |      0 |      28 |      2 |
|        0.4  |      2 |      0 |       2 |      0 |
|        0.45 |      0 |      0 |       0 |      0 |
|        0.5  |      0 |      0 |       0 |      0 |

## Profitable-days fraction by regime

|   threshold |   all |   flat |   green |   red |
|------------:|------:|-------:|--------:|------:|
|        0.1  |     0 |      0 |       0 |     0 |
|        0.15 |     0 |      0 |       0 |     0 |
|        0.2  |     0 |      0 |       0 |     0 |
|        0.25 |     0 |      0 |       0 |     0 |
|        0.3  |     0 |      0 |       0 |     0 |
|        0.35 |     0 |      0 |       0 |     0 |
|        0.4  |     0 |      0 |       0 |     0 |
|        0.45 |     0 |      0 |       0 |     0 |
|        0.5  |     0 |      0 |       0 |     0 |

## Verdict (HC #428 R1)

**REJECT** — no threshold produced ALL-regime Sharpe > 0 with regime imbalance ≤ 0.50.
- Best (least-bad) threshold: 0.4
- All-regime Sharpe at best: 0.000
- Regime imbalance: nan

Conclusion: base v3.4.2 OOT preds carry no positive-edge information about FIFO-net profitability at ANY threshold or regime. The meta-gate cannot manufacture edge from base predictions that lack it. Next path: HC #486 Step 3 (new model with 250ms head + supervised stream-stability auxiliary loss).