# Sector picker v4 — with 10-K text features (HC #564 R6(a))

**Pooled-OOT**: 1824 trading days, 2018-01-02 -> 2025-12-29
**Sectors used**: 11 (Basic Materials, Communication Services, Consumer Cyclical, Consumer Defensive, Energy, Financial Services, Healthcare, Industrials, Real Estate, Technology, Utilities)

## Pooled-OOT vs 4:43 PM baseline (v3 shelf, no text)

| Metric | v4 (with text) | v3 baseline (4:43 PM) | Delta |
|---|---|---|---|
| Sharpe | 0.32 | 0.69 | -0.37 |
| Sortino | 0.51 | 0.98 | -0.47 |
| CAGR | 1.3% | 7.0% | -5.7 pp |
| MaxDD | -11.8% | -18.2% | +6.4 pp |
| Calmar | 0.11 | 0.39 | -0.28 |

## vs SPY (levered, with financing)

| Metric | Strategy | SPY 1x | SPY 1.5x | SPY 2x |
|---|---|---|---|---|
| Sharpe | 0.32 | 0.63 | 0.56 | 0.53 |
| CAGR | 1.3% | 11.2% | 13.2% | 14.0% |
| MaxDD | -11.8% | -34.1% | -47.8% | -59.4% |
| Calmar | 0.11 | 0.33 | 0.28 | 0.24 |

## Per-sector pooled-OOT

| Sector | Sharpe | CAGR | MaxDD | Calmar | PF | WR |
|---|---|---|---|---|---|---|
| Technology | 0.89 | 14.4% | -29.5% | 0.49 | 1.20 | 40.7% |
| Basic Materials | 0.40 | 3.6% | -15.9% | 0.23 | 1.08 | 39.5% |
| Consumer Cyclical | 0.39 | 5.5% | -52.0% | 0.11 | 1.08 | 39.9% |
| Real Estate | 0.37 | 2.9% | -15.6% | 0.19 | 1.08 | 38.7% |
| Energy | 0.15 | 1.1% | -27.2% | 0.04 | 1.03 | 38.4% |
| Healthcare | 0.09 | 0.3% | -24.2% | 0.01 | 1.02 | 38.0% |
| Financial Services | -0.20 | -3.1% | -31.2% | -0.10 | 0.96 | 38.9% |
| Communication Services | -0.20 | -3.1% | -29.4% | -0.11 | 0.96 | 38.4% |
| Industrials | -0.22 | -3.1% | -39.7% | -0.08 | 0.96 | 37.9% |
| Utilities | -0.27 | -2.3% | -18.9% | -0.12 | 0.95 | 37.6% |
| Consumer Defensive | -0.78 | -7.4% | -46.0% | -0.16 | 0.86 | 36.4% |

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

- Calmar floor (>=1.0): FAIL (0.11)
- Beats SPY 1x on Sharpe: NO
- Beats SPY 1.5x on Sharpe: NO
- Beats SPY 1.5x on CAGR: NO
- 10-K text meaningfully moved the result: YES (dSharpe=-0.37, dCAGR=-5.7pp)
