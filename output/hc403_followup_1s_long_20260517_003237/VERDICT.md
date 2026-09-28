# HC #403 FOLLOWUP — 1s-LONG Canonical Cross-Check Verdict

Produced: 2026-05-17 00:35:30 ET
Output: `output/hc403_followup_1s_long_20260517_003237/`

## Verdict
**1s-LONG candidate is CO-MVP** (canonical Sharpe >= 10 and STRICT-PASS day_conc <= 0.20 with n_fills >= 30).

## Control: Trial 278 reproduction
- Expected: Sharpe 13.48, day_conc 0.186, n_fills 195
- Replayed: Sharpe 13.48, day_conc 0.186, n_fills 195
- PASS: True → canonical harness is the trusted reference

## Candidate: 1s-LONG, top-10% conf, 2s hold, passive_+2, ToD 13-15 ET

### (B) Canonical pipeline (no FIFO confluence — same as trial 278's measured 13.48)
| metric | value |
|---|---|
| Sharpe | 10.26 |
| Sortino | 12.76 |
| PF | 5.57 |
| WR | 80.0% |
| mean_net (tk/fill) | 2.072 |
| day_conc | 0.187 |
| n_fills | 165 |
| CI_low_95 | 1.583 |
| HC #344 strict pass | True |
| HC #344 relaxed pass | True |

### (C) Canonical pipeline + FIFO confluence > 0.80 (like agent B)
| metric | value |
|---|---|
| Sharpe | 11.55 |
| Sortino | 23.91 |
| PF | 6.90 |
| WR | 78.3% |
| mean_net (tk/fill) | 2.449 |
| day_conc | 0.309 |
| n_fills | 60 |
| CI_low_95 | 1.597 |
| HC #344 strict pass | False |
| HC #344 relaxed pass | True |

## Agent B reported (for reference)
- Sharpe 23.57, n_fills 106, day_conc 0.178, mean_net 1.83
- Custom harness inflated trial 278 by ~15% vs canonical (15.2 vs 13.48)

## Interpretation
- (B) is the apples-to-apples comparison with trial 278's published 13.48
  (canonical pipeline applies only ToD + pred_strength).
- (C) layers FIFO confluence on top of canonical, matching agent B's exact
  filter stack. The delta (C)-(B) quantifies how much of agent B's edge is
  due to FIFO confluence vs the base 1s-long signal.
- If (C) Sharpe is materially lower than agent B's 23.57, the difference is
  the harness-inflation factor; the ~15% inflation hypothesis predicts
  canonical Sharpe ~20.5.
