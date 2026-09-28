# v5_REAL Ladder — HC #556 R3 Honesty Check

**Pricing input**: `iv_features_real_blend.parquet` (originally written to `iv_features_modeled.parquet` slot during this run).

- 51.1% of rows use REAL vendor IV from DoltHub `post-no-preference/options` (vol_current 30d ATM IV).
- 48.9% of rows fall back to modeled VIX-regime IV/RV ratio (pre-2019 + the 2 tickers missing from DOLT: ARM, SHOP).
- 68/70 universe tickers covered by DOLT.
- 2020-2026 coverage: 84%-97% real per year.
- DOLT date range: 2019-02-09 -> 2026-06-05.

**Calibration delta vs purely modeled:**
- mean REAL IV/RV ratio = 1.13  (modeled assumed 1.20)
- mean REAL sigma = 0.322       (modeled 0.359)

**Result**: REAL IV is moderately LOWER than the modeled assumption.
The modeled run was somewhat optimistic on premium received.

**Window**: 2020-01-01 -> 2025-12-31 (matches v3 for apples-to-apples).
**Regime overlay**: ON (HC #555 — 25.3% of days masked as risk-off).
**Mode**: full wheel (profit-take 0.65, roll DTE 1, allows assignment).

To re-create:
  python -m data.ingest_options_real --features-out data/cache/iv_features_modeled.parquet
  python -m strategy.tier_runner --modeled --full-wheel --regime-overlay \
    --start 2020-01-01 --end 2025-12-31 \
    --out results/tier_ladder_v5_REAL_2020_2025

(Remember to restore iv_features_modeled.parquet from
 iv_features_modeled.MODELED_BACKUP.parquet afterwards.)
