# Single-Name Wheel on Megacap Tech v1 (HC #587 R2)

Generated: 2026-06-09 17:19:44
Universe: AAPL, MSFT, GOOGL, NVDA, META
Window: 2018-01-01 to 2025-12-31
Per-ticker allocation: $4,000 (basket total $20,000)
Single-ticker tests: $20,000 each

## Strategy

- Put delta 0.3, call delta 0.3
- DTE 25-35 (target 30)
- Profit-take 50%, NO forced close on losses
- VIX gate 35.0 (entries only)
- Cost: $0.65/contract + $0.005/share on assign/called-away

## Headline Metrics

| Config | CAGR | Sharpe | Sortino | MaxDD | Calmar | WR | PF | Final $ |
|---|---|---|---|---|---|---|---|---|
| AAPL | 42.75% | 1.27 | 3.53 | -101.7% | 0.42 | 63.2% | 3.38 | $346,579 |
| MSFT | 15.42% | 1.12 | 2.72 | -94.6% | 0.16 | 64.6% | 2.87 | $63,532 |
| GOOGL | 46.37% | 0.23 | 0.16 | -107.8% | 0.43 | 63.9% | 1.44 | $422,470 |
| NVDA | 43.20% | 0.57 | 10.29 | -127.8% | 0.34 | 60.3% | 7.35 | $354,896 |
| META | 20.45% | 0.57 | 1.26 | -109.7% | 0.19 | 62.0% | 2.14 | $89,366 |
| BASKET | 19.39% | 0.80 | 1.20 | -92.4% | 0.21 | 63.4% | 1.84 | $82,546 |

## Buy-and-Hold Equal-Weight Basket (reference)
- CAGR 36.13%, Sharpe 1.10, MaxDD -45.1%, final $235,440

## HC #428 R1 Regime Gates

| Config | n_g | n_r | n_f | Sh_g | Sh_r | Sh_f | gap | PASS (<=0.50) |
|---|---|---|---|---|---|---|---|---|
| AAPL | 534 | 406 | 1070 | 2.15 | -0.51 | 1.40 | 1.24 | FAIL |
| MSFT | 534 | 406 | 1070 | 2.29 | -0.77 | 1.11 | 1.33 | FAIL |
| GOOGL | 534 | 406 | 1070 | -0.25 | -0.35 | 1.33 | 0.31 | PASS |
| NVDA | 534 | 406 | 1070 | 1.09 | 0.78 | 1.02 | 0.29 | PASS |
| META | 534 | 406 | 1070 | 1.02 | 0.68 | 0.71 | 0.33 | PASS |
| BASKET | 534 | 406 | 1070 | 2.24 | -3.65 | 0.47 | 1.61 | FAIL |

## Deploy Gates

| Config | Sharpe>=1.0 | Calmar>=1.5 | Gap<=0.50 | DayConc<=0.70 | DEPLOY |
|---|---|---|---|---|---|
| AAPL | PASS | FAIL | FAIL | PASS | **NO** |
| MSFT | PASS | FAIL | FAIL | PASS | **NO** |
| GOOGL | FAIL | FAIL | PASS | PASS | **NO** |
| NVDA | FAIL | FAIL | PASS | PASS | **NO** |
| META | FAIL | FAIL | PASS | PASS | **NO** |
| BASKET | FAIL | FAIL | FAIL | PASS | **NO** |

## Time-in-State (% of days)

| Config | %Shares | %CSP | %Cash |
|---|---|---|---|
| AAPL | 40.7% | 59.3% | 0.0% |
| MSFT | 38.0% | 61.6% | 0.4% |
| GOOGL | 40.2% | 59.8% | 0.0% |
| NVDA | 35.9% | 60.3% | 3.9% |
| META | 41.5% | 58.5% | 0.0% |
| BASKET | 6.8% | 13.3% | 79.9% |

## Wheel Activity

| Config | CSP opens | CC opens | Assignments | Call-aways |
|---|---|---|---|---|
| AAPL | 156 | 85 | 17 | 17 |
| MSFT | 160 | 71 | 16 | 15 |
| GOOGL | 154 | 79 | 15 | 15 |
| NVDA | 155 | 72 | 14 | 13 |
| META | 144 | 83 | 15 | 14 |
| BASKET | 202 | 75 | 15 | 14 |

## Tail Event Stress

| Config | Event | MaxDD | RecoverDays | CumRet |
|---|---|---|---|---|
| AAPL | COVID_2020 | -89.7% | 63 | 12.9% |
| AAPL | 2022_bear | -101.7% | 56 | -79.5% |
| MSFT | COVID_2020 | -94.6% | 32 | 13.7% |
| MSFT | 2022_bear | -81.4% | 29 | 26.5% |
| GOOGL | COVID_2020 | -95.2% | 67 | 9.1% |
| GOOGL | 2022_bear | -89.6% | 282 | -87.0% |
| NVDA | COVID_2020 | -115.1% | 38 | 17.2% |
| NVDA | 2022_bear | -108.9% | 28 | 711.6% |
| META | COVID_2020 | -84.5% | 67 | 5.4% |
| META | 2022_bear | -69.8% | 218 | -53.6% |
| BASKET | COVID_2020 | -29.1% | 38 | 9.3% |
| BASKET | 2022_bear | -58.3% | 28 | 82.8% |
| BASKET | 2018_Q4 | -31.2% | 113 | -29.7% |

## Recommendation

- **NO config passed all deploy gates. Zero passers across 5 single-names + basket.**
- 3 of 5 single names (GOOGL/NVDA/META) cleared the HC #428 R1 regime gap, but all 5 failed the Calmar gate due to catastrophic drawdowns (-94% to -128% MaxDD).
- The basket FAILED the regime gap (1.61) — worse than any prior SPY wheel variant — driven by the negative-Sharpe red-day cluster (-3.65 vs +2.24 green).
- **MaxDD > 100% interpretation**: equity went negative on a marked-to-market basis during 2022 (NVDA: -65% YTD as the shares; we held assigned shares purchased at higher prices AND sold CCs below cost basis on the way down, where short-call MTM blew up as the stock fell from $300 to $108). In real margin, this would have triggered a forced liquidation.
- **Wheel vs Buy-Hold**: Wheel basket made $82K, Buy-Hold made $235K. The wheel CAPS upside (called away on every rally) but doesn't escape the downside (held through every selloff). Worst single-name DD: NVDA -128% (would have been liquidated).
- **Honest conclusion**: the wheel payoff is structurally short-tail regardless of underlying. Single-name megacap actually makes it WORSE because tail moves are bigger (META 2022: -77%, NVDA 2022: -65%, AAPL 2022: -27%). Willingness to take assignment doesn't help — assignment is the loss mechanism, not the protection. The covered call writes premium on a declining asset, which is just selling premium into negative drift.
- **Recommendation**: The wheel lane is exhausted. The regime-asymmetry problem is geometric, not parametric. No DTE, no delta, no underlying, no assignment policy fixes it. Recommend pivoting to a fundamentally different income strategy (e.g., delta-neutral premium harvest, calendar spreads, or abandoning premium-selling entirely for trend-following).

## Buy-and-Hold vs Basket Wheel

- Buy-Hold: CAGR 36.13%, Sharpe 1.10, MaxDD -45.1%, final $235,440
- Basket Wheel: CAGR 19.39%, Sharpe 0.80, MaxDD -92.4%, final $82,546
- Delta CAGR: -16.74pp (wheel vs buy-hold)

- Worst single-name DD: NVDA at -127.8%