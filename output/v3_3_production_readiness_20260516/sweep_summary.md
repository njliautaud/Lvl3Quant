# v3.3 Production Readiness Full Sweep — HC #395

- Predictions: `/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_predictions.npz` (n_samples=241351, dates=['20260223', '20260224', '20260225', '20260226', '20260227'])
- Heads in catalog: 32 (directional=23)
- Bands: ['P50', 'P75', 'P90', 'P95', 'P99', 'P99.5', 'P99.9']
- Order types: ['passive_at_touch', 'passive_at_touch_plus_1', 'ioc_market']
- Regimes: ['all', 'open', 'mid', 'close', 'vol_low', 'vol_mid', 'vol_high']
- HC #344 gate: day_conc ≤ 0.2, n_filled ≥ 30
- HC #392: commission-only (0.376 ticks RT); spread implicit in fills.
- Total cells: 27,048 | Errors: 0
- Cells passing HC #344: 0

## Top 20 cells by Sharpe (HC #344 gated)

**No cells pass HC #344.**

## Per-head best gated cell (regime=all)

No head produced a gated cell.

## Top 10 confluence pairs by joint Sharpe

No confluence pairs available.