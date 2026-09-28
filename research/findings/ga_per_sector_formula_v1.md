# Per-Sector GA Formula Report — tag=v1
_Generated 2026-06-07T02:37:25.887266_

## Headline (HC #559 verdict)

- Sectors fit: **1**
- Pass Calmar≥1.0 in-sample: **1** (Technology)
- Pass Calmar≥1.0 OOT     : **0** (—)
- Deployable (Calmar≥1 OOT + regime-stable OOT + Sharpe>0.7): **0** (—)

## Per-sector summary

| Sector | N | Train Sh / Calmar / CAGR / DD | OOT Sh / Calmar / CAGR / DD | Regime (G/R) OOT | Verdict |
|---|---:|---|---|---|---|
| Technology | 17 | +2.05 / +4.20 / +35.6% / -8.5% | — / — / — / — | — | ⚠ IS-only |

## Feature consensus (which signals matter across sectors)

| Feature | # sectors using | avg |w| | sign consensus |
|---|---:|---:|---|
| flow_sectorRet_r20 | 1 | 0.11 | - (0+/1-) |
| flow_sectorRet_r60 | 1 | 0.58 | + (1+/0-) |
| flow_sectorAUM_z60 | 1 | 0.40 | - (0+/1-) |
| flow_sectorRel_r20 | 1 | 0.14 | + (1+/0-) |
| factor_momentum_load | 1 | 0.14 | + (1+/0-) |
| factor_lowvol_load | 1 | 0.17 | - (0+/1-) |
| factor_momentum_rank | 1 | 0.18 | + (1+/0-) |
| factor_quality_rank | 1 | 0.72 | + (1+/0-) |
| fund_debtEquity_z | 1 | 0.64 | - (0+/1-) |
| fund_earningsYield_z | 1 | 0.95 | + (1+/0-) |
| fund_evEbitda_z | 1 | 0.62 | + (1+/0-) |
| fund_fcfMargin_z | 1 | 0.58 | + (1+/0-) |
| fund_fcfYield_z | 1 | 1.10 | - (0+/1-) |
| fund_gross_margin_z | 1 | 0.41 | - (0+/1-) |
| fund_net_margin_z | 1 | 0.49 | + (1+/0-) |
| fund_revScale_log_z | 1 | 0.60 | - (0+/1-) |
| fund_roicProxy_z | 1 | 0.72 | + (1+/0-) |

## Formulas (top-weighted terms shown)

### Technology [IS-only]
`score = -1.105*fund_fcfYield_z +0.950*fund_earningsYield_z +0.725*factor_quality_rank +0.721*fund_roicProxy_z -0.644*fund_debtEquity_z +0.617*fund_evEbitda_z -0.603*fund_revScale_log_z +0.584*flow_sectorRet_r60 +0.581*fund_fcfMargin_z +0.492*fund_net_margin_z -0.413*fund_gross_margin_z -0.404*flow_sectorAUM_z60 +0.178*factor_momentum_rank -0.168*factor_lowvol_load +0.141*factor_momentum_load +0.137*flow_sectorRel_r20 -0.108*flow_sectorRet_r20`
