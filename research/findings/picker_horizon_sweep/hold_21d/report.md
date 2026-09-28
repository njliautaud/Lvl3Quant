# Sector picker v4 — with 10-K text features (HC #564 R6(a))

**Pooled-OOT**: 1563 trading days, 2018-01-02 -> 2025-12-24
**Sectors used**: 11 (Basic Materials, Communication Services, Consumer Cyclical, Consumer Defensive, Energy, Financial Services, Healthcare, Industrials, Real Estate, Technology, Utilities)

## Pooled-OOT vs 4:43 PM baseline (v3 shelf, no text)

| Metric | v4 (with text) | v3 baseline (4:43 PM) | Delta |
|---|---|---|---|
| Sharpe | 0.77 | 0.69 | +0.08 |
| Sortino | 1.15 | 0.98 | +0.17 |
| CAGR | 3.8% | 7.0% | -3.2 pp |
| MaxDD | -12.6% | -18.2% | +5.6 pp |
| Calmar | 0.30 | 0.39 | -0.09 |

## vs SPY (levered, with financing)

| Metric | Strategy | SPY 1x | SPY 1.5x | SPY 2x |
|---|---|---|---|---|
| Sharpe | 0.77 | 0.60 | 0.53 | 0.50 |
| CAGR | 3.8% | 10.6% | 12.2% | 12.7% |
| MaxDD | -12.6% | -31.3% | -44.3% | -55.6% |
| Calmar | 0.30 | 0.34 | 0.28 | 0.23 |

## Per-sector pooled-OOT

| Sector | Sharpe | CAGR | MaxDD | Calmar | PF | WR |
|---|---|---|---|---|---|---|
| Consumer Cyclical | 0.89 | 16.5% | -21.0% | 0.78 | 1.18 | 48.5% |
| Energy | 0.86 | 10.6% | -24.5% | 0.43 | 1.17 | 47.7% |
| Technology | 0.71 | 12.0% | -33.5% | 0.36 | 1.14 | 48.9% |
| Healthcare | 0.25 | 2.5% | -18.3% | 0.14 | 1.05 | 47.5% |
| Industrials | 0.23 | 2.1% | -22.8% | 0.09 | 1.04 | 46.9% |
| Utilities | 0.17 | 1.0% | -18.7% | 0.06 | 1.03 | 46.3% |
| Communication Services | 0.05 | -0.1% | -37.0% | -0.00 | 1.01 | 46.9% |
| Real Estate | 0.04 | -0.0% | -21.3% | -0.00 | 1.01 | 47.8% |
| Financial Services | -0.10 | -2.2% | -39.1% | -0.06 | 0.98 | 46.5% |
| Basic Materials | -0.16 | -2.4% | -30.6% | -0.08 | 0.97 | 45.1% |
| Consumer Defensive | -0.44 | -4.6% | -31.1% | -0.15 | 0.92 | 43.1% |

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

- Calmar floor (>=1.0): FAIL (0.30)
- Beats SPY 1x on Sharpe: YES
- Beats SPY 1.5x on Sharpe: YES
- Beats SPY 1.5x on CAGR: NO
- 10-K text meaningfully moved the result: YES (dSharpe=+0.08, dCAGR=-3.2pp)
