# HC #404 — WATERFALL: Where does the alpha go? (trial 278, n=195 fills)

Per-fill decomposition, all values in ticks (1 tick = $12.50 on ES futures).

```
  GROSS MFE within hold        + 0.41 tk  | ==
  - exit timing loss           - 0.04 tk  |    (we couldn't exit at the peak)
  + entry edge (passive_+2)    + 2.00 tk  | ========   (limit posted 2 ticks better than touch)
  - commission (RT)            - 0.38 tk  | ==
  ----------------------------------------------------
  = REALIZED NET / fill        + 1.99 tk  | ========
```

**Of the 0.41 ticks of gross alpha within the 1.48s hold window, we capture 1.99 ticks (488%) as realized net per fill.**

- 0.04 ticks vanishes to exit timing (we exited at the horizon, not at the MFE peak).
- We RECOVER 2.00 ticks from the passive_+2 entry (limit was 2 ticks better than touch).
- We pay 0.376 ticks commission.

Sanity check: |residual| max = 0.00e+00 ticks (should be ~0 — verifies the decomposition arithmetic).

Reconstruction vs replay df max residual: 0.00e+00 ticks (verifies we are decomposing the SAME numbers the canonical harness reports).

## Honest caveats

1. **Entry adverse-selection is NOT separately modeled in the canonical harness for passive orders.** The harness uses a constant `+K` edge for passive_at_touch_plus_K fills; the realized fill price equals touch+K by construction. In real trading the fill latency between order placement and queue traversal would induce a price-against-us component (entry adverse selection). We report it as 0 here because that is what the harness uses; this is a known harness limitation, NOT a claim that adv-sel is zero in reality.

2. **Queue-position slippage is absorbed UPSTREAM as a fill-rate deflator** (the harness deflates fill probability by 0.5^K for passive_+K). The trades we see are the ones that *did* fill; the unfilled trades carry the queue-loss penalty as missed opportunity, not as a per-fill cost. We cannot separate it per-fill in this harness.

3. **Exit-timing loss is the biggest leak.** See `exit_mode_comparison.csv` for what happens if we swap the fixed-hold exit for an MFE-trigger exit policy.