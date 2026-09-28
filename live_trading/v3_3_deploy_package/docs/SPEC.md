# v3.3 Deploy Package — Spec

**Authorized by**: HC #368 (2026-05-14 23:32 ET, user verbatim).
**Status**: PACKAGE BUILT, awaiting "good config" production + user deploy-go.
**Operating contract**: v2 stays LIVE on Razer until user issues explicit `deploy v3.3` go-ahead.
  (Deploy is a hard-cost production op per HC #366(f) — NOT autonomous.)

## Layout

```
live_trading/v3_3_deploy_package/
├── docs/
│   └── SPEC.md                    ← this file (architecture + invariants)
├── configs/
│   ├── v33_paper_trader_config.json     ← CLI-configurable runtime config (HC #351)
│   ├── v33_config_template.json         ← annotated default
│   └── candidate_configs/               ← top-K config variants from HC #357 sweep
├── safety_nets/
│   ├── safety_nets.py             ← 8 watchdog classes (DD, stop, trade-cap, etc.)
│   ├── static_fallback_rules.yaml ← static-rules backup mode
│   └── README.md                  ← per-gate threshold rationale
├── inference/
│   ├── v33_inference_daemon.py    ← model loader + predict loop (reads arch.window_size per HC #360)
│   └── README.md                  ← input/output spec, head names, σ values
├── reconciliation/
│   └── eod_reconcile.py           ← HC #321/#341/#348/#354/#361-compliant EOD report
├── runbooks/
│   ├── deploy_v33_to_razer.ps1    ← one-button deploy + auto-rollback
│   └── rollback_v3_to_v2.ps1      ← instant revert
└── config_dev_pipeline/
    └── v33_config_search.py       ← Jupiter sweep over HC #358/#363 outputs under HC #357 cost stack
```

## Invariants (must hold for ANY deploy)

1. **Arch params read from .pt ckpt** (HC #360) — `window_size`, `feature_stats` path, head spec, normalization scheme.
2. **No FIFO-floor headline metrics** (HC #349) — all config metrics under FIFO + queue-position-on-arrival + adverse-selection cost + commission + cancel/replace.
3. **Price path block on every config** (HC #361) — realized post-fill at {0/1/5/10/30/60/120/300s} + MFE/MAE-with-times.
4. **8 safety nets ALL ENABLED** at startup; ANY gate failure halts new entries.
5. **Static fallback always armed** — if ML stack dies, paper trader does NOT stop; switches to `static_fallback_rules.yaml`.
6. **Position reconciliation every N seconds** — broker positions vs internal state; mismatch → halt.
7. **EOD report posted to Discord automatically** — HC #354 attribution per trade + HC #321 Sharpe/Sortino/PF/WR + HC #348 MFE/MAE in ticks + HC #361 full price-path block.

## Deploy gate (HC #344 LIVE-READINESS)

A config is **deploy-eligible** only if it passes ALL of these on the v3.3 OOT (17-day extended per HC #337):

| Gate | Requirement |
|------|-------------|
| FIFO | net ticks > 0 over OOT period |
| QUEUE | queue-position-on-arrival model applied; fill rate > 30% at config band |
| ADV-SEL | adverse-selection cost (post-fill price walk +1/5/10/30s) accounted; net edge still > 0 |
| PERF-GATING | Sharpe > 0.5 on full-cost basis, max-drawdown < 2× avg-daily-return |
| MONITORING | all 8 safety nets configured + EOD reconciliation runs cleanly |

## Build status (live)

| Artifact | State |
|----------|-------|
| docs/SPEC.md | ✅ this file |
| configs/v33_config_template.json | ✅ written |
| safety_nets/safety_nets.py | ✅ written (9 watchdogs + self-test) |
| safety_nets/static_fallback_rules.yaml | ✅ written |
| safety_nets/README.md | ✅ written (per-gate rationale) |
| inference/v33_inference_daemon.py | ✅ written (ckpt-aware per HC #360; reads arch from .pt) |
| reconciliation/eod_reconcile.py | ✅ written |
| runbooks/deploy_v33_to_razer.ps1 | ✅ written |
| runbooks/rollback_v3_to_v2.ps1 | ✅ written |
| config_dev_pipeline/v33_config_search.py | ✅ written + executed |
| configs/candidate_configs/ | ⚠️  0 deploy-eligible candidates as of 2026-05-14 23:58 ET |
| config_search_report.md | ✅ written (213 rows analyzed, 0 pass HC #344 gate) |

### Package readiness: 100% (infrastructure-complete)
### Deploy readiness: NOT YET — no config passes HC #344 PERF-GATING gate yet.

The blocker is **signal quality**, not infrastructure. v2 stays live until
v3.3 produces at least one config with `hc357_sharpe ≥ 0.50, hc357_net > 0,
n_fills ≥ 30, day_conc ≤ 0.95, ci_low_95 > -0.5, side=SHORT` on the
extended OOT under the HC #357 full cost stack (FIFO + queue + adv-sel +
commission + cancel/replace).

Next levers to produce a passing config (autonomous work in parallel):
1. v3.4 dual-trunk training is still epoch 0 — wait for full OOT verdict.
2. σ-recalibration sweep over MTL learnable σ priors (may unlock currently-buried heads).
3. Joint-head ensemble: re-run config_search.py with multi-head primary
   (head A + head B agreement gate) instead of single-head primary.
4. Lower the `min_percentile` floor on Top10% band (currently 90.0; may
   re-include enough fills to push n_fills × edge over the bar).

## Pre-deploy checklist (user-runnable)

```
1. cat live_trading/v3_3_deploy_package/configs/v33_paper_trader_config.json   # review active config
2. cat live_trading/v3_3_deploy_package/safety_nets/static_fallback_rules.yaml # review fallback
3. python live_trading/v3_3_deploy_package/safety_nets/safety_nets.py --self-test   # smoke-test all 8 watchdogs
4. # On Razer: ./runbooks/deploy_v33_to_razer.ps1 -ConfigName <candidate_name>
5. # Auto-rollback if any verification step fails
```

## References

- HC #344 (live-readiness gates), #347 (Razer autonomous relaunch), #349 (no FIFO-floor headline),
  #351 (CLI-configurable wrapper), #354 (current v2 best-evidence config), #357 (full market replay),
  #358 (Jupiter test-bench), #360 (arch params from ckpt), #361 (price-path block), #363 (v3.3
  full execution analysis), #367 (autonomy/no-pause), #368 (this package).
