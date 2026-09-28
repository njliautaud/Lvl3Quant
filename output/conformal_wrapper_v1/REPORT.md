# Conformal Wrapper v1 - REPORT

Generated: 2026-05-22 12:25:42

- OOT days loaded: 32  (cal=16, test=16)
- Total events: 1,579,225
- Horizons: ['1s', '5s', '10s', '30s']
- alpha (miscoverage): 0.1

## Verdict

**REJECT**

## Width vs Edge (decile 1 vs decile 10)

| horizon   | side   |   d1_sharpe |   d10_sharpe |    d1_net |   d10_net |
|:----------|:-------|------------:|-------------:|----------:|----------:|
| 10s       | long   |    -7.99736 |    -12.3705  | -0.216867 | -0.392358 |
| 10s       | short  |    -9.68595 |     -5.43158 | -0.254757 | -0.187346 |
| 1s        | long   |   -17.7894  |    -48.2325  | -0.199448 | -0.558864 |
| 1s        | short  |   -24.4839  |    -22.5647  | -0.248249 | -0.312107 |
| 30s       | long   |   -11.4497  |     -5.45118 | -0.542653 | -0.4728   |
| 30s       | short  |    -9.23073 |     -7.78704 | -0.325371 | -0.363799 |
| 5s        | long   |   -10.4408  |    -20.434   | -0.219863 | -0.474876 |
| 5s        | short  |   -14.0331  |    -11.1985  | -0.27376  | -0.256562 |

Mean delta Sharpe(d1 - d10) = 3.545

Width is informative: **False**

## Interpretation

Split-conformal calibrates uncertainty against realized residuals on the CAL split, then the trained ridge predicts the residual magnitude per event on the TEST split. The bottom decile (decile 1) of predicted residual magnitude is the most 'reliable' set of predictions. If this bucket is meaningfully more profitable than the top decile, the conformal-width feature carries information that raw point-confidence missed. If it is not, then the residual is largely irreducible at this feature set and the 7-test rejection of v3.4.2 stands.

## Methodology Caveats

- Regime gap uses day NET-sum sign as a proxy (no ES close-to-close pulled here). This is conservative since the day's own trades influence the classification.
- Targets in npz are documented as ticks per HC #486; P&L net = side*target_h - 0.376.
- Causal split: calibration days strictly precede test days.
- Adaptive width uses |pred| heads + p_up + quantile-spreads; a richer feature set could change conclusions.
