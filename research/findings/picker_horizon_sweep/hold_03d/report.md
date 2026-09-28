# Sector picker v4 — with 10-K text features (HC #564 R6(a))

**Pooled-OOT**: 1896 trading days, 2018-01-02 -> 2025-12-31
**Sectors used**: 11 (Basic Materials, Communication Services, Consumer Cyclical, Consumer Defensive, Energy, Financial Services, Healthcare, Industrials, Real Estate, Technology, Utilities)

## Pooled-OOT vs 4:43 PM baseline (v3 shelf, no text)

| Metric | v4 (with text) | v3 baseline (4:43 PM) | Delta |
|---|---|---|---|
| Sharpe | -0.43 | 0.69 | -1.12 |
| Sortino | -0.69 | 0.98 | -1.67 |
| CAGR | -1.6% | 7.0% | -8.6 pp |
| MaxDD | -18.4% | -18.2% | -0.2 pp |
| Calmar | -0.09 | 0.39 | -0.48 |

## vs SPY (levered, with financing)

| Metric | Strategy | SPY 1x | SPY 1.5x | SPY 2x |
|---|---|---|---|---|
| Sharpe | -0.43 | 0.81 | 0.74 | 0.71 |
| CAGR | -1.6% | 15.1% | 19.2% | 22.2% |
| MaxDD | -18.4% | -34.1% | -47.8% | -59.4% |
| Calmar | -0.09 | 0.44 | 0.40 | 0.37 |

## Per-sector pooled-OOT

| Sector | Sharpe | CAGR | MaxDD | Calmar | PF | WR |
|---|---|---|---|---|---|---|
| Technology | 0.37 | 4.4% | -24.4% | 0.18 | 1.09 | 31.9% |
| Consumer Cyclical | 0.27 | 2.9% | -32.3% | 0.09 | 1.06 | 30.6% |
| Real Estate | 0.10 | 0.4% | -22.0% | 0.02 | 1.02 | 31.2% |
| Financial Services | 0.02 | -0.4% | -33.9% | -0.01 | 1.00 | 30.6% |
| Industrials | -0.04 | -1.0% | -33.0% | -0.03 | 0.99 | 30.9% |
| Healthcare | -0.14 | -2.3% | -46.7% | -0.05 | 0.97 | 30.6% |
| Energy | -0.26 | -2.9% | -36.0% | -0.08 | 0.95 | 30.1% |
| Communication Services | -0.33 | -4.0% | -33.8% | -0.12 | 0.93 | 29.9% |
| Utilities | -0.47 | -3.3% | -29.0% | -0.11 | 0.90 | 29.9% |
| Basic Materials | -0.64 | -5.8% | -46.0% | -0.13 | 0.88 | 29.2% |
| Consumer Defensive | -1.34 | -10.6% | -58.6% | -0.18 | 0.75 | 28.1% |

## Feature importance (avg |coef| across sectors)

### Top 5
- `fp_ebitda_ttm`: 0.0144
- `fp_ni_ttm`: 0.0136
- `fp_ebitda_margin`: 0.0099
- `fp_net_margin`: 0.0093
- `fp_market_cap_pit`: 0.0082

### Bottom 5
- `sf_corr_hyg_60d`: 0.0000
- `sf_corr_uup_60d`: 0.0000
- `sf_corr_gld_60d`: 0.0000
- `ins_gross_buy_usd`: 0.0000
- `ins_n_buyers`: 0.0000

### Where text features landed
- `lm_tone_score_z`: rank 32/76, avg |coef|=0.0015
- `rf_delta_z`: rank 33/76, avg |coef|=0.0015
- `flag_going_concern`: rank 34/76, avg |coef|=0.0014
- `flag_accounting_change`: rank 26/76, avg |coef|=0.0026
- `flag_restatement`: rank 29/76, avg |coef|=0.0021

## Verdict

- Calmar floor (>=1.0): FAIL (-0.09)
- Beats SPY 1x on Sharpe: NO
- Beats SPY 1.5x on Sharpe: NO
- Beats SPY 1.5x on CAGR: NO
- 10-K text meaningfully moved the result: YES (dSharpe=-1.12, dCAGR=-8.6pp)
