# Kelly Sizing v1 — Position-Sizing Test on short_10s @ 0.55 Survivor

**Overall verdict: REJECT**

**Best scheme: `B_linear`** — sized t/trade = +5.689,
profit-days = 7/15 (46.7%),
notional retained = 41.3% of baseline,
day-Sharpe = 6.57 (bootstrap 95% CI [1.28, 11.02]).

## Hypothesis
At 15-day OOT every filter axis has rejected for short_10s @ 0.55. Instead of removing
trades, weight them by meta-classifier confidence. If high-prob trades carry larger true
edge, sized day-PnL should lift profit-days share without dropping total notional.

## Data Caveats
- 15 unique trading days in FIFO (3/16 – 4/14).
- meta_prob is only available for the **walk-forward OOT subset** (2456 of 3074
  trades, dates 4/7 – 4/14). Trades on dates BEFORE 4/7 have no meta_prob — they
  predate the meta-classifier OOT window and were training data. For those trades we
  fall back to size=1 (baseline behaviour), so the sizing schemes only diverge from
  baseline on dates 4/7 – 4/14.
- 16 days never materialised: per_trade_diagnostics is the canonical replay and shows 15.

## meta_prob distribution
Range: [0.489, 0.808], mean=0.633, std=0.070.
Histogram:
- (0.489, 0.55]: 0
- (0.55, 0.6]: 1007
- (0.6, 0.65]: 582
- (0.65, 0.7]: 329
- (0.7, 0.75]: 328
- (0.75, 0.81]: 209

Dynamic range is moderate (std ≈ 0.070 on a [0.49, 0.81] support) — sizing
schemes that depend on (meta_prob − 0.5) have limited contrast.

## Schemes
- **A_unit**: size=1 (baseline)
- **B_linear**: size = clip((mp − 0.5)/0.5, [0.1, 1.0])
- **C_quadratic**: size = clip(((mp − 0.5)/0.5)², [0.05, 1.0])
- **D_threshold_linear**: size = 0 if mp < 0.55, else clip((mp − 0.55)/0.45, [0.1, 1.0])
- **E_inverse_vol**: confidence × inverse-vol-z (within-day proxy)
- **F_kelly_quarter**: edge=(mp − 0.5)·|net_t|_mean / var(net_t), capped at 0.25 full Kelly, mapped to [0.05, 1.0]

Trades without meta_prob: size=1 (baseline fallback) in all schemes B–F.

## Per-scheme results

| scheme | verdict | notional_frac | sized t/trade | profit_days | day-Sharpe | bootstrap-95%-CI |
|---|---|---|---|---|---|---|
| A_unit | REJECT | 1.00 | +5.021 | 8/15 (53.3%) | 6.50 | [3.86, 11.65] |
| B_linear | REJECT | 0.41 | +5.689 | 7/15 (46.7%) | 6.57 | [1.28, 11.02] |
| C_quadratic | REJECT | 0.28 | +6.279 | 8/15 (53.3%) | 6.05 | [-2.40, 10.83] |
| D_threshold_linear | REJECT | 0.37 | +5.794 | 8/15 (53.3%) | 6.49 | [0.67, 11.07] |
| E_inverse_vol | REJECT | 0.32 | +6.237 | 7/15 (46.7%) | 6.44 | [0.17, 10.96] |
| F_kelly_quarter | REJECT | 0.24 | +6.741 | 8/15 (53.3%) | 5.55 | [-7.09, 10.46] |

## Honest checks
- **Day-PnL correlation with A_unit** (high values ⇒ sizing only rescales the same days):
  - B_linear: +0.960
  - C_quadratic: +0.907
  - D_threshold_linear: +0.943
  - E_inverse_vol: +0.945
  - F_kelly_quarter: +0.890
- **meta_prob bimodality**: see histogram above. The distribution is spread, not bimodal.
- **Bootstrap day-Sharpe CIs**: see table column. All schemes' CIs broadly overlap; sized
  sharpe does NOT statistically dominate baseline given n=15.

## Gates
- ACCEPT: profit_days_ratio ≥ 69%, sized t/trade ≥ +4.0, notional ≥ 60%.
- PARTIAL: profit_days_ratio ≥ 60%, sized t/trade ≥ +4.0, notional ≥ 50%.
- REJECT: neither.

## Conclusion
Sizing does NOT rescue the short_10s @ 0.55 survivor at 15-day sample. No scheme passes the PARTIAL bar.

The day-PnL correlation analysis matters most: if every sizing scheme correlates >0.95
with baseline day-PnL, sizing is just rescaling the same 15 days. Bootstrap CIs are wide
because n=15 — no scheme's day-Sharpe CI excludes baseline.

Researcher degrees of freedom: scheme menu was chosen pre-hoc but the choice itself
(linear/quadratic/threshold) is degrees of freedom. Treat any "best-scheme" finding as
hypothesis to test on the next OOT batch (≥40 days, HC #428 R1), not as deploy approval.
