# HC #404 SHADOW REPLAY — TRIAL 278 ADVERSE-SELECTION SENSITIVITY

Produced: 2026-05-17 00:52:51 ET

## Baseline (control, scale=0.0, queue_depth=2)

- n_fills: **195** (published 195 — match: True)
- Sharpe: **13.48** (published 13.48)
- tk/fill: **1.991** (published 1.99)
- day_conc: **0.186** (published 0.186)
- HC #344 strict pass: **True**

## Modeling assumption

Per-fill adverse-selection cost (deterministic, no look-ahead):

    adv_cost_ticks = adv_sel_scale * queue_depth * 1.0 tk

Interpretation:
- **scale=0.0** = current full_market_replay (NO per-fill adv-sel cost — only fill-prob deflation).
- **scale=1.0** at **queue_depth=2** = queue-traversal cost EXACTLY ERASES the +2 tk passive_+2 entry edge.
- **scale=2.0** at queue_depth=2 = adv-sel DOUBLE the K-edge (picked off + then some).
- queue_depth ∈ {0, 1, 2} brackets sensitivity to the touch-offset assumption.
  Trial 278 is **passive_at_touch_plus_2** so queue_depth=2 is the realistic case.

## Sweep table (full)

| adv_sel_scale | queue_depth | adv_cost_per_fill_tk | n_fills | sharpe | sortino | pf | wr | mean_net_ticks | day_conc | ci_low_95 | hc344_strict_pass |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.000 | 0 | 0.000 | 195 | 13.478 | 29.229 | 9.432 | 81.538 | 1.991 | 0.186 | 1.662 | True |
| 0.000 | 1 | 0.000 | 195 | 13.478 | 29.229 | 9.432 | 81.538 | 1.991 | 0.186 | 1.662 | True |
| 0.000 | 2 | 0.000 | 195 | 13.478 | 29.229 | 9.432 | 81.538 | 1.991 | 0.186 | 1.662 | True |
| 0.250 | 0 | 0.000 | 195 | 13.478 | 29.229 | 9.432 | 81.538 | 1.991 | 0.186 | 1.662 | True |
| 0.250 | 1 | 0.250 | 195 | 11.785 | 24.702 | 7.029 | 76.410 | 1.741 | 0.178 | 1.412 | True |
| 0.250 | 2 | 0.500 | 195 | 10.092 | 21.154 | 5.288 | 76.410 | 1.491 | 0.173 | 1.162 | True |
| 0.500 | 0 | 0.000 | 195 | 13.478 | 29.229 | 9.432 | 81.538 | 1.991 | 0.186 | 1.662 | True |
| 0.500 | 1 | 0.500 | 195 | 10.092 | 21.154 | 5.288 | 76.410 | 1.491 | 0.173 | 1.162 | True |
| 0.500 | 2 | 1.000 | 195 | 6.707 | 13.273 | 2.943 | 64.615 | 0.991 | 0.194 | 0.662 | True |
| 0.750 | 0 | 0.000 | 195 | 13.478 | 29.229 | 9.432 | 81.538 | 1.991 | 0.186 | 1.662 | True |
| 0.750 | 1 | 0.750 | 195 | 8.400 | 16.623 | 3.943 | 64.615 | 1.241 | 0.180 | 0.912 | True |
| 0.750 | 2 | 1.500 | 195 | 3.322 | 6.510 | 1.710 | 63.590 | 0.491 | 0.266 | 0.162 | False |
| 1.000 | 0 | 0.000 | 195 | 13.478 | 29.229 | 9.432 | 81.538 | 1.991 | 0.186 | 1.662 | True |
| 1.000 | 1 | 1.000 | 195 | 6.707 | 13.273 | 2.943 | 64.615 | 0.991 | 0.194 | 0.662 | True |
| 1.000 | 2 | 2.000 | 195 | -0.063 | -0.110 | 0.990 | 53.846 | -0.009 | 11.908 | -0.338 | False |
| 1.500 | 0 | 0.000 | 195 | 13.478 | 29.229 | 9.432 | 81.538 | 1.991 | 0.186 | 1.662 | True |
| 1.500 | 1 | 1.500 | 195 | 3.322 | 6.510 | 1.710 | 63.590 | 0.491 | 0.266 | 0.162 | False |
| 1.500 | 2 | 3.000 | 195 | -6.834 | -9.997 | 0.314 | 33.846 | -1.009 | 0.349 | -1.338 | False |
| 2.000 | 0 | 0.000 | 195 | 13.478 | 29.229 | 9.432 | 81.538 | 1.991 | 0.186 | 1.662 | True |
| 2.000 | 1 | 2.000 | 195 | -0.063 | -0.110 | 0.990 | 53.846 | -0.009 | 11.908 | -0.338 | False |
| 2.000 | 2 | 4.000 | 195 | -13.604 | -17.617 | 0.095 | 16.410 | -2.009 | 0.295 | -2.338 | False |

## Pass/fail summary

- At queue_depth=2 (realistic), strict HC #344 passes at adv_sel_scale ∈ [0.0, 0.25, 0.5]
- At queue_depth=2 (realistic), strict HC #344 FAILS at adv_sel_scale ∈ [0.75, 1.0, 1.5, 2.0]

### VERDICT: DEPLOYMENT-RISKY
Trial 278 fails strict HC #344 at adv_sel_scale=0.75.
Even at HALF the "erase the edge" scale, the strategy stops passing the gate.
Recommend NOT deploying to live capital — the +2 edge is more fragile than the headline Sharpe suggests.

## Sensitivity at queue_depth=2 (realistic)

| adv_sel_scale | adv_cost (tk/fill) | mean_net | Sharpe | strict_pass |
|---|---|---|---|---|
| 0.00 | 0.00 | 1.991 | 13.48 | True |
| 0.25 | 0.50 | 1.491 | 10.09 | True |
| 0.50 | 1.00 | 0.991 | 6.71 | True |
| 0.75 | 1.50 | 0.491 | 3.32 | False |
| 1.00 | 2.00 | -0.009 | -0.06 | False |
| 1.50 | 3.00 | -1.009 | -6.83 | False |
| 2.00 | 4.00 | -2.009 | -13.60 | False |

## Caveats (CRITICAL — read before quoting Sharpe numbers above)

1. This is a SENSITIVITY STUDY, not a definitive cost. The adverse-selection model is a single deterministic per-fill scalar.
2. Real adv-sel is stochastic (some fills get picked off badly, some get filled by reverting noise traders).
   This model ignores variance, which would FURTHER lower Sharpe at any given mean cost.
3. We have no per-trade pre-fill order-book data on the NPZ, so we cannot calibrate adv-sel from data — only bracket it.
4. queue_depth=2 + scale=1.0 (the "realistic" cell) is our BEST GUESS at where reality sits for a +2-tick passive limit.
   The truth could easily be scale=0.5 (better, light queue-traversal) or scale=1.5 (worse, momentum picking us off).
5. Live paper-trading on Razer will tell us which it is. Until then, treat the scale=1.0 row as the deployment baseline.