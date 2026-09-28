# HC #404 — WATERFALL: Where does the alpha go? (trial 278, n=195 fills)

Per-fill decomposition, all values in ticks (1 tick = $12.50 on ES futures).

## TL;DR — the answer is NOT what the user expected

The +5-tick gross MFE the user cited is from the **30s LONG Top10 / 30s hold** cell, NOT trial 278. Trial 278 uses a **1.48s hold** on a 30s SHORT signal. Within that 1.48s window the gross MFE is only **+0.41 tk/fill** — there is barely any price travel to capture because the hold is tiny. The +1.99 tk/fill realized net is **almost entirely from the +2 tick passive-entry edge** (posting a limit 2 ticks INSIDE the touch), NOT from price travel.

## Waterfall

```
  Gross MFE (price travel)     + 0.41 tk  | ==
  - exit timing loss           - 0.04 tk  |    (peak minus realized exit)
  + entry edge (passive_+2)    + 2.00 tk  | ========   (limit posted 2 ticks better than touch)
  - commission (RT)            - 0.38 tk  | ==
  ----------------------------------------------------
  = REALIZED NET / fill        + 1.99 tk  | ========
```

**Total gross alpha = price-travel MFE (0.41) + passive entry edge (2.00) = 2.41 tk. We realize 1.99 tk = 83% of total available alpha.**

- 0.04 ticks lost to exit timing (small because hold is only 1.48s).
- 2.00 ticks captured by passive_+2 entry — the DOMINANT P&L driver.
- 0.376 ticks paid in commission.

## Why the user's '5-tick MFE = wildly profitable' intuition does NOT apply here

The 5-tick MFE figure is the 30s-horizon MFE for 30s LONG Top10. If you tried to capture that at a 30s hold with a fixed-hold market exit, exit-timing loss would explode because the path mean-reverts and you can't pick the peak. See the CONTROL config in `decomposition_per_config.csv` and the exit-mode sweep in `exit_mode_comparison.csv` for the long-hold story.

Sanity check: |residual| max = 0.00e+00 ticks (should be ~0 — verifies the decomposition arithmetic).

Reconstruction vs replay df max residual: 0.00e+00 ticks (verifies we are decomposing the SAME numbers the canonical harness reports).

## Honest caveats

1. **Entry adverse-selection is NOT separately modeled in the canonical harness for passive orders.** The harness uses a constant `+K` edge for passive_at_touch_plus_K fills; the realized fill price equals touch+K by construction. In real trading the fill latency between order placement and queue traversal would induce a price-against-us component (entry adverse selection). We report it as 0 here because that is what the harness uses; this is a known harness limitation, NOT a claim that adv-sel is zero in reality.

2. **Queue-position slippage is absorbed UPSTREAM as a fill-rate deflator** (the harness deflates fill probability by 0.5^K for passive_+K). The trades we see are the ones that *did* fill; the unfilled trades carry the queue-loss penalty as missed opportunity, not as a per-fill cost. We cannot separate it per-fill in this harness.

3. **Exit-timing loss is the biggest leak.** See `exit_mode_comparison.csv` for what happens if we swap the fixed-hold exit for an MFE-trigger exit policy.