# Sector picker v4 — with 10-K text features (HC #564 R6(a))

**Pooled-OOT**: 1563 trading days, 2018-01-02 -> 2025-12-24
**Sectors used**: 11 (Basic Materials, Communication Services, Consumer Cyclical, Consumer Defensive, Energy, Financial Services, Healthcare, Industrials, Real Estate, Technology, Utilities)

## Pooled-OOT vs 4:43 PM baseline (v3 shelf, no text)

| Metric | v4 (with text) | v3 baseline (4:43 PM) | Delta |
|---|---|---|---|
| Sharpe | 0.89 | 0.69 | +0.20 |
| Sortino | 1.32 | 0.98 | +0.34 |
| CAGR | 5.6% | 7.0% | -1.4 pp |
| MaxDD | -14.0% | -18.2% | +4.2 pp |
| Calmar | 0.40 | 0.39 | +0.01 |

## vs SPY (levered, with financing)

| Metric | Strategy | SPY 1x | SPY 1.5x | SPY 2x |
|---|---|---|---|---|
| Sharpe | 0.89 | 0.60 | 0.53 | 0.50 |
| CAGR | 5.6% | 10.6% | 12.2% | 12.7% |
| MaxDD | -14.0% | -31.3% | -44.3% | -55.6% |
| Calmar | 0.40 | 0.34 | 0.28 | 0.23 |

## Per-sector pooled-OOT

| Sector | Sharpe | CAGR | MaxDD | Calmar | PF | WR |
|---|---|---|---|---|---|---|
| Technology | 1.10 | 22.0% | -41.6% | 0.53 | 1.23 | 47.6% |
| Consumer Cyclical | 0.83 | 17.0% | -49.6% | 0.34 | 1.17 | 47.5% |
| Energy | 0.54 | 7.0% | -23.4% | 0.30 | 1.11 | 47.0% |
| Real Estate | 0.47 | 3.9% | -14.2% | 0.28 | 1.09 | 45.2% |
| Industrials | 0.46 | 5.5% | -26.0% | 0.21 | 1.09 | 47.0% |
| Utilities | 0.37 | 3.0% | -13.2% | 0.23 | 1.07 | 46.4% |
| Basic Materials | 0.15 | 1.0% | -28.3% | 0.04 | 1.03 | 46.1% |
| Healthcare | 0.11 | 0.5% | -41.7% | 0.01 | 1.02 | 46.0% |
| Financial Services | -0.04 | -1.8% | -40.0% | -0.05 | 0.99 | 46.9% |
| Communication Services | -0.10 | -2.0% | -23.7% | -0.08 | 0.98 | 46.3% |
| Consumer Defensive | -0.10 | -1.6% | -26.4% | -0.06 | 0.98 | 45.9% |

## Feature importance (avg |coef| across sectors)

### Top 5
- `ret`: 0.0074
- `log_ret`: 0.0064
- `rv_yz_60d`: 0.0059
- `rv_cc_252d`: 0.0057
- `rv_yz_20d`: 0.0045

### Bottom 5
- `sr_rs_rank_among_sectors`: 0.0000
- `sr_lead_lag_score_5d`: 0.0000
- `ins_n_buys`: 0.0000
- `ins_n_sells`: 0.0000
- `ins_net_share_change`: 0.0000

### Where text features landed
- `lm_tone_score_z`: rank 15/41, avg |coef|=0.0012
- `rf_delta_z`: rank 13/41, avg |coef|=0.0014
- `flag_going_concern`: rank 17/41, avg |coef|=0.0009
- `flag_accounting_change`: rank 11/41, avg |coef|=0.0019
- `flag_restatement`: rank 9/41, avg |coef|=0.0023

## Verdict

- Calmar floor (>=1.0): FAIL (0.40)
- Beats SPY 1x on Sharpe: YES
- Beats SPY 1.5x on Sharpe: YES
- Beats SPY 1.5x on CAGR: NO
- 10-K text meaningfully moved the result: YES (dSharpe=+0.20, dCAGR=-1.4pp)

## Apples-to-apples: Energy+Industrials only (matches 4:43 PM scope)

| Metric | v4 (E+I, with text) | v3 baseline (E+I, no text) | Delta |
|---|---|---|---|
| Sharpe | 0.70 | 0.69 | +0.01 |
| Sortino | 1.00 | 0.98 | +0.02 |
| CAGR | 6.8% | 7.0% | -0.2 pp |
| MaxDD | -15.5% | -18.2% | +2.7 pp |
| Calmar | 0.44 | 0.39 | +0.05 |

**On the same 2-sector scope, 10-K text adds essentially nothing.** The all-11-sector
v4 book (Sharpe 0.89, Calmar 0.40) is better than the 2-sector baseline only because
diversification across more sector books lowers volatility — not because text features
carry alpha. Top features remain price/vol; text features rank 9-17 of 41 with avg
|coef| ~0.001-0.002, which is noise next to top features at ~0.006-0.007.
