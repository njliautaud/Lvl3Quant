# HC #405 — VERDICT

Generated 2026-05-17 01:12:27 ET

## Is there a clearly-better-than-trial-278 signal-driven config?

**NO STRICT-PASSER, BUT YES RELAXED-PASSER — 4 config(s) capture meaningful signal alpha (mean MFE >= 1 tk) and pass the HC #344 RELAXED day_conc gate (<= 0.70) on the 15-day OOT replay, but FAIL the strict 0.20 gate.**

Best relaxed-pass signal-driven candidate by Sharpe: `trial_000708_sharpe_22.267` — side=long, horizon=5s, order=passive_at_touch_plus_2, hold=1.07s, Sharpe=13.92, mean MFE=1.04 tk, %MFE>=1tk=68%, n_fills=38, day_conc=0.471 (strict FAIL, relaxed PASS)

These configs would be signal-driven supplements (not replacements) to trial 278 — they carry real directional alpha but their P&L is concentrated on fewer days (a sign of overfitting to the original 5-day OOT used during Optuna's day_conc evaluation; HC #402 demonstrated 40/50 trials that strict-passed on 5d failed on 15d).

## What the leaderboard tells us

- Configs scanned: **100** | with MFE data: 100
- Order-type distribution: passive_+2=100, passive_+1=0, passive_at_touch=0, ioc_market=0
- Mean gross MFE >= 0.5 tk: **28** configs
- Mean gross MFE >= 1.0 tk: **4** configs
- Mean gross MFE >= 2.0 tk: **0** configs
- Signal-driven STRICT (MFE>=1tk AND day_conc<=0.20 AND n_fills>=30): **0** configs
- Signal-driven RELAXED (MFE>=1tk AND day_conc<=0.70 AND n_fills>=30): **4** configs

## Critical structural observation

Of the 100 deploy-eligible configs, **100** (100%) use `passive_at_touch_plus_2`. Only 0 'no-structural-credit' configs (ioc_market or passive_at_touch) have mean gross MFE >= 1 tk in their hold window.

**This means the Optuna search overwhelmingly converged on configs that monetize the passive_+K entry credit, not on configs that capture directional signal alpha.** Trial 278's signal-alpha weakness (HC #404 finding: only 0.41 tk of the +1.99 tk/fill is signal-driven) is a property of the entire deploy-eligible cohort, not a quirk of trial 278 alone.

## Recommendation

**KEEP trial 278 as the Monday strict-gate deployment**, but PAPER-TRADE `trial_000708_sharpe_22.267` — side=long, horizon=5s, order=passive_at_touch_plus_2, hold=1.07s, Sharpe=13.92, mean MFE=1.04 tk, %MFE>=1tk=68%, n_fills=38, day_conc=0.471 (strict FAIL, relaxed PASS) in parallel as a signal-driven diversifier. The relaxed-pass cohort is meaningfully signal-driven (mean MFE >= 1 tk, ~60-78% of fills have MFE >= 1 tk) and would diversify trial 278's structural-credit fragility. The high day_conc on 15d OOT is the trade-off; if 2-4 weeks of live paper trading shows day_conc stabilizing below 0.20, these are stronger replacement candidates than trial 278.
