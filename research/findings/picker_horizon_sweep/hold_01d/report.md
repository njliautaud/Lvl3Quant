# Sector picker v4 — with 10-K text features (HC #564 R6(a))

**Pooled-OOT**: 2011 trading days, 2018-01-02 -> 2025-12-31
**Sectors used**: 11 (Basic Materials, Communication Services, Consumer Cyclical, Consumer Defensive, Energy, Financial Services, Healthcare, Industrials, Real Estate, Technology, Utilities)

## Pooled-OOT vs 4:43 PM baseline (v3 shelf, no text)

| Metric | v4 (with text) | v3 baseline (4:43 PM) | Delta |
|---|---|---|---|
| Sharpe | -73190041762708816.00 | 0.69 | -73190041762708816.00 |
| Sortino | -73190041762708816.00 | 0.98 | -73190041762708816.00 |
| CAGR | -11.8% | 7.0% | -18.8 pp |
| MaxDD | -63.4% | -18.2% | -45.2 pp |
| Calmar | -0.19 | 0.39 | -0.58 |

## vs SPY (levered, with financing)

| Metric | Strategy | SPY 1x | SPY 1.5x | SPY 2x |
|---|---|---|---|---|
| Sharpe | -73190041762708816.00 | 0.70 | 0.63 | 0.60 |
| CAGR | -11.8% | 12.5% | 15.2% | 16.9% |
| MaxDD | -63.4% | -34.1% | -47.8% | -59.4% |
| Calmar | -0.19 | 0.37 | 0.32 | 0.28 |

## Per-sector pooled-OOT

| Sector | Sharpe | CAGR | MaxDD | Calmar | PF | WR |
|---|---|---|---|---|---|---|
| Basic Materials | -36595020881354408.00 | -11.8% | -63.4% | -0.19 | 0.00 | 0.0% |
| Communication Services | -36595020881354408.00 | -11.8% | -63.4% | -0.19 | 0.00 | 0.0% |
| Consumer Cyclical | -36595020881354408.00 | -11.8% | -63.4% | -0.19 | 0.00 | 0.0% |
| Consumer Defensive | -36595020881354408.00 | -11.8% | -63.4% | -0.19 | 0.00 | 0.0% |
| Energy | -36595020881354408.00 | -11.8% | -63.4% | -0.19 | 0.00 | 0.0% |
| Financial Services | -36595020881354408.00 | -11.8% | -63.4% | -0.19 | 0.00 | 0.0% |
| Healthcare | -36595020881354408.00 | -11.8% | -63.4% | -0.19 | 0.00 | 0.0% |
| Industrials | -36595020881354408.00 | -11.8% | -63.4% | -0.19 | 0.00 | 0.0% |
| Real Estate | -36595020881354408.00 | -11.8% | -63.4% | -0.19 | 0.00 | 0.0% |
| Technology | -36595020881354408.00 | -11.8% | -63.4% | -0.19 | 0.00 | 0.0% |
| Utilities | -36595020881354408.00 | -11.8% | -63.4% | -0.19 | 0.00 | 0.0% |

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

- Calmar floor (>=1.0): FAIL (-0.19)
- Beats SPY 1x on Sharpe: NO
- Beats SPY 1.5x on Sharpe: NO
- Beats SPY 1.5x on CAGR: NO
- 10-K text meaningfully moved the result: YES (dSharpe=-73190041762708816.00, dCAGR=-18.8pp)
