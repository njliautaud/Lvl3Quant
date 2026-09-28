# Sector picker v4 — with 10-K text features (HC #564 R6(a))

**Pooled-OOT**: 1954 trading days, 2018-01-02 -> 2025-12-26
**Sectors used**: 11 (Basic Materials, Communication Services, Consumer Cyclical, Consumer Defensive, Energy, Financial Services, Healthcare, Industrials, Real Estate, Technology, Utilities)

## Pooled-OOT vs 4:43 PM baseline (v3 shelf, no text)

| Metric | v4 (with text) | v3 baseline (4:43 PM) | Delta |
|---|---|---|---|
| Sharpe | 0.77 | 0.69 | +0.08 |
| Sortino | 1.14 | 0.98 | +0.16 |
| CAGR | 3.6% | 7.0% | -3.4 pp |
| MaxDD | -8.5% | -18.2% | +9.7 pp |
| Calmar | 0.42 | 0.39 | +0.03 |

## vs SPY (levered, with financing)

| Metric | Strategy | SPY 1x | SPY 1.5x | SPY 2x |
|---|---|---|---|---|
| Sharpe | 0.77 | 0.69 | 0.62 | 0.58 |
| CAGR | 3.6% | 12.2% | 14.8% | 16.3% |
| MaxDD | -8.5% | -34.1% | -47.8% | -59.4% |
| Calmar | 0.42 | 0.36 | 0.31 | 0.27 |

## Per-sector pooled-OOT

| Sector | Sharpe | CAGR | MaxDD | Calmar | PF | WR |
|---|---|---|---|---|---|---|
| Energy | 0.57 | 6.5% | -22.7% | 0.29 | 1.11 | 45.9% |
| Real Estate | 0.53 | 4.1% | -13.1% | 0.31 | 1.10 | 46.7% |
| Healthcare | 0.52 | 6.5% | -21.7% | 0.30 | 1.10 | 46.1% |
| Technology | 0.44 | 6.5% | -37.5% | 0.17 | 1.09 | 46.3% |
| Communication Services | 0.31 | 3.2% | -24.2% | 0.13 | 1.06 | 45.5% |
| Consumer Cyclical | 0.31 | 3.9% | -50.5% | 0.08 | 1.06 | 46.3% |
| Industrials | 0.26 | 2.5% | -19.9% | 0.13 | 1.05 | 44.9% |
| Financial Services | 0.22 | 2.1% | -27.4% | 0.08 | 1.05 | 44.9% |
| Basic Materials | 0.21 | 1.7% | -25.7% | 0.07 | 1.04 | 44.5% |
| Utilities | 0.00 | -0.3% | -23.1% | -0.01 | 1.00 | 44.9% |
| Consumer Defensive | -0.47 | -4.9% | -40.9% | -0.12 | 0.92 | 43.3% |

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

- Calmar floor (>=1.0): FAIL (0.42)
- Beats SPY 1x on Sharpe: YES
- Beats SPY 1.5x on Sharpe: YES
- Beats SPY 1.5x on CAGR: NO
- 10-K text meaningfully moved the result: YES (dSharpe=+0.08, dCAGR=-3.4pp)
