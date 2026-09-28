# Macro Stock-Picker — B-Track (HC #558 / #559 / #560)

Placeholder scope doc. The macro picker is a sibling to the wheel strategy that consumes the **shared feature store** (`data/feature_store/v1/`).

## Scope (HC #558 B-Track)

- **Universe**: full-market screener (not the wheel's 70-name shelf). Initial v1 will piggyback on the existing 70-name universe; v2 expands to the SPX1500 once ingestion is in place.
- **Per-sector models**: separate scoring formula per GICS sector. Sector cohorts are statistically distinct — pooling destroys signal.
- **Interpretable formula**: HC #558 requires an explicit, human-readable score
  (`score = w1·fund_revGrowthYoY_z + w2·flow_sectorRel_r60 + w3·factor_quality_load + …`).
  GA is one tool for searching the weight vector; not the only one.
- **Selection method**: top-K per sector by score, rebalanced monthly, with regime gate from `regime_features`.

## Acceptance Floor (HC #559)

- **Calmar ≥ 1.0** out-of-sample per sector cohort.
- **OOT ≥ 40 days** stratified across green / red / flat regime days (HC #428 R1).
- **Day-conc cap ≤ 0.70** (HC #344).
- **No all-short / all-long configs** unless cross-regime evidence justifies.

## Inputs

All from `data/feature_store/v1/`:
- `fund_features.parquet` — cross-sectional fundamental z-scores
- `flow_features.parquet` — sector ETF flow proxies
- `factor_features.parquet` — rolling factor loadings
- `regime_features.parquet` — macro regime gate
- `theme_features.parquet` — thematic basket exposure

## Outputs (planned)

- `strategy/macro_picker/scores/YYYY-MM-DD.parquet` — per-ticker daily scores
- `strategy/macro_picker/picks/YYYY-MM-DD.parquet` — top-K per sector
- Backtest + walk-forward harness lives in `strategy/macro_picker/backtest/`

## Status

Skeleton only. The feature store (v1) is the prerequisite; the scoring formula and walk-forward harness come next.

## Relationship to Wheel Strategy

- Wheel = options income on the 70-name shelf, ranked by IV/regime.
- Macro picker = directional long equity (or pair) selection from a broader universe, ranked by fundamentals + flow + factor + theme.
- Both read from the shared feature store. Neither owns the data.
