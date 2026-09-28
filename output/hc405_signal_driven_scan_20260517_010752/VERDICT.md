# HC #405 — VERDICT

Generated 2026-05-17 01:07:59 ET

## Is there a clearly-better-than-trial-278 signal-driven config?

**NO — among the deploy-eligible Optuna configs scanned, none satisfied mean_gross_MFE_within_hold >= 1.0 tk while also passing HC #344 strict (day_conc <= 0.20) with n_fills >= 30 on the 15-day extended OOT.**

Trial 278 remains the headline candidate, but its P&L is STRUCTURAL (passive_+2 credit), not signal-driven.

## What the leaderboard tells us

- Configs scanned: **5** | with MFE data: 5
- Order-type distribution: passive_+2=5, passive_+1=0, passive_at_touch=0, ioc_market=0
- Mean gross MFE >= 0.5 tk: **2** configs
- Mean gross MFE >= 1.0 tk: **1** configs
- Mean gross MFE >= 2.0 tk: **0** configs
- Signal-driven (MFE>=1tk AND strict pass AND n_fills>=30): **0** configs

## Critical structural observation

Of the 5 deploy-eligible configs, **5** (100%) use `passive_at_touch_plus_2`. Only 0 'no-structural-credit' configs (ioc_market or passive_at_touch) have mean gross MFE >= 1 tk in their hold window.

**This means the Optuna search overwhelmingly converged on configs that monetize the passive_+K entry credit, not on configs that capture directional signal alpha.** Trial 278's signal-alpha weakness (HC #404 finding: only 0.41 tk of the +1.99 tk/fill is signal-driven) is a property of the entire deploy-eligible cohort, not a quirk of trial 278 alone.

## Recommendation

**Keep trial 278 as the Monday deployment** but treat its Sharpe as STRUCTURAL not SIGNAL-DRIVEN. The Optuna search did NOT find configs with strong signal alpha at sub-30s horizons in this predictions cohort. Future research should re-direct toward longer-hold signal-capture configs (30s+ holds with MFE-trigger exits) instead of further Optuna sweeps of the same parameter space.
