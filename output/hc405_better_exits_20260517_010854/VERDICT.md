# HC #405 — VERDICT: Can dynamic exit logic recover MFE on LONG side?

Produced: 2026-05-17 01:08:59 ET
Sanity check (trial 278 reproduction): PASS

## Bottom line

**PARTIAL.** 2 config(s) clear the gates on LONG side, but all rely on the passive_+K limit credit (NOT on dynamic exit value-add). The LONG signal alone is not capturable.

## Family-by-family summary (passive_at_touch entry only)

| head | exit_family | best_sharpe | best_tk | median_tk | worst_day_conc | any_pass | n_configs |
|---|---|---|---|---|---|---|---|
| 10s | fixed_hold | -2.01 | -0.526 | -0.526 | 0.585 | False | 1 |
| 10s | mfe_trigger_market | -1.73 | -0.414 | -0.470 | 0.775 | False | 6 |
| 10s | signal_flip | -2.52 | -0.552 | -0.628 | 0.647 | False | 4 |
| 10s | trailing_stop | -2.42 | -0.617 | -0.625 | 0.565 | False | 4 |
| 30s | fixed_hold | -1.26 | -0.477 | -0.477 | 0.812 | False | 1 |
| 30s | mfe_trigger_market | -0.35 | -0.135 | -0.388 | 1.216 | False | 6 |
| 30s | signal_flip | -1.72 | -0.362 | -0.520 | 0.749 | False | 4 |
| 30s | trailing_stop | -2.15 | -0.706 | -0.772 | 0.564 | False | 4 |

## Family-by-family summary (any entry_order, incl. passive +K credit)

| head | exit_family | entry_order | best_sharpe | best_tk | worst_day_conc | any_pass |
|---|---|---|---|---|---|---|
| 10s | fixed_hold | passive_at_touch | -2.01 | -0.526 | 0.585 | False |
| 10s | fixed_hold | passive_at_touch_plus_1 | 2.28 | +0.650 | 0.438 | False |
| 10s | fixed_hold | passive_at_touch_plus_2 | 5.77 | +1.328 | 0.252 | False |
| 10s | mfe_trigger_market | passive_at_touch | -1.73 | -0.414 | 0.775 | False |
| 10s | mfe_trigger_market | passive_at_touch_plus_1 | 3.18 | +0.674 | 0.749 | False |
| 10s | mfe_trigger_market | passive_at_touch_plus_2 | 8.44 | +1.644 | 0.336 | False |
| 10s | signal_flip | passive_at_touch | -2.52 | -0.552 | 0.647 | False |
| 10s | signal_flip | passive_at_touch_plus_1 | 4.25 | +0.617 | 0.442 | False |
| 10s | signal_flip | passive_at_touch_plus_2 | 13.65 | +1.446 | 0.302 | False |
| 10s | trailing_stop | passive_at_touch | -2.42 | -0.617 | 0.565 | False |
| 10s | trailing_stop | passive_at_touch_plus_1 | 1.91 | +0.538 | 0.545 | False |
| 10s | trailing_stop | passive_at_touch_plus_2 | 5.49 | +1.262 | 0.273 | False |
| 30s | fixed_hold | passive_at_touch | -1.26 | -0.477 | 0.812 | False |
| 30s | fixed_hold | passive_at_touch_plus_1 | 2.48 | +0.810 | 0.371 | False |
| 30s | fixed_hold | passive_at_touch_plus_2 | 5.16 | +1.677 | 0.245 | False |
| 30s | mfe_trigger_market | passive_at_touch | -0.35 | -0.135 | 1.216 | False |
| 30s | mfe_trigger_market | passive_at_touch_plus_1 | 3.54 | +1.261 | 0.392 | False |
| 30s | mfe_trigger_market | passive_at_touch_plus_2 | 6.71 | +1.882 | 0.268 | False |
| 30s | signal_flip | passive_at_touch | -1.72 | -0.362 | 0.749 | False |
| 30s | signal_flip | passive_at_touch_plus_1 | 4.13 | +0.755 | 0.359 | False |
| 30s | signal_flip | passive_at_touch_plus_2 | 14.93 | +1.957 | 0.357 | False |
| 30s | trailing_stop | passive_at_touch | -2.15 | -0.706 | 0.564 | False |
| 30s | trailing_stop | passive_at_touch_plus_1 | 1.52 | +0.448 | 0.663 | False |
| 30s | trailing_stop | passive_at_touch_plus_2 | 5.97 | +1.760 | 0.206 | True |

## Interpretation

- **MFE-trigger market exit**: profits if signal generates favorable excursion ≥ THRESHOLD within hold window. Pays 1.0 tick spread cost. Net edge requires THRESHOLD - 1.0 - 0.376 = THRESHOLD - 1.376 tk to be positive PER FILL, AND requires enough fills to actually trigger.
- **Trailing stop**: protects against giveback after favorable excursion. Pays 1.0 tick spread cost. Works only if signal consistently produces favorable peaks before reverting.
- **Signal-flip**: exit when model says the trade direction has reversed. Pays 1.0 tick spread cost. Requires the model's subsequent predictions to be informative beyond the entry horizon, not just AT it.

## Caveats

- Exit P&L is quantized to {1s, 5s, 10s, 30s} horizon checkpoints (realized cumulative tick moves), NOT true 250ms tick-level intra-horizon path. This matches full_market_replay's approximation. Same caveat as HC #403/404.
- Spread for market exits set to canonical 1.0 tick RTH (NOT trial 278's optuna-sampled 0.77 — we use canonical for honesty).
- LONG side does NOT apply the FIFO-confluence filter that trial 278's SHORT side uses (per HC #405 scope — we test exit logic value-add independently, with the same percentile + pred-strength entry gate).
- Queue-deflator fill model uses identical params to full_market_replay (passive_at_touch = 0.5, +1 = 0.25, +2 = 0.125, with slow-exit ×0.25 multiplier).
