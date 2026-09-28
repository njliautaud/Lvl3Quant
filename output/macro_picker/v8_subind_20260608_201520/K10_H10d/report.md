# Sector picker v8 — Sub-industry (K=10, hold=10d)

**Taxonomy**: 44 sub-industries, 259 ticker mappings
**Buckets active / skipped**: 3 / 33 (min size = 4)
**Wall**: 29.7s

## Pooled-OOT

| Metric | Strategy | SPY 1x | SPY 1.5x | SPY 2x |
|---|---:|---:|---:|---:|
| Sharpe | 0.54 | 0.69 | 0.62 | 0.58 |
| Sortino | 0.77 | 0.83 | 0.75 | 0.71 |
| CAGR | 1.5% | 12.2% | 14.8% | 16.3% |
| MaxDD | -5.2% | -34.1% | -47.8% | -59.4% |
| Calmar | 0.30 | 0.36 | 0.31 | 0.27 |

**Deployable sub-industries (Sharpe ≥ 1.0)**: 0 / 3
**Pooled passes Calmar ≥ 1.5**: NO

## Per-sub-industry pooled-OOT (sorted by Sharpe)

| Sub-industry | N | Sharpe | CAGR | MaxDD | Calmar | PF | WR |
|---|---:|---:|---:|---:|---:|---:|---:|
| semiconductors | 7 | 0.74 | 3.8% | -7.6% | 0.50 | 1.15 | 48.0% |
| ai_software | 9 | 0.18 | 0.7% | -10.3% | 0.07 | 1.04 | 46.6% |
| cloud_saas | 7 | 0.02 | -0.0% | -14.7% | -0.00 | 1.00 | 45.3% |

## Skipped buckets

- `aerospace_industrial` — 1 tickers < min_bucket=4
- `biotech_large` — 2 tickers < min_bucket=4
- `biotech_smid` — 1 tickers < min_bucket=4
- `brokerage` — 2 tickers < min_bucket=4
- `copper` — 1 tickers < min_bucket=4
- `crypto_proxy` — 1 tickers < min_bucket=4
- `cybersecurity` — 3 tickers < min_bucket=4
- `defense_tech` — 2 tickers < min_bucket=4
- `e_commerce` — 1 tickers < min_bucket=4
- `ev_auto` — 3 tickers < min_bucket=4
- `glp1` — 1 tickers < min_bucket=4
- `gold` — 0 tickers < min_bucket=4
- `hydrogen` — 0 tickers < min_bucket=4
- `lithium` — 1 tickers < min_bucket=4
- `luxury` — 0 tickers < min_bucket=4
- `meddev` — 3 tickers < min_bucket=4
- `natgas` — 2 tickers < min_bucket=4
- `nuclear` — 0 tickers < min_bucket=4
- `oil_services` — 2 tickers < min_bucket=4
- `physical_ai_robotics` — 3 tickers < min_bucket=4
- `power_ipp` — 3 tickers < min_bucket=4
- `quantum` — 0 tickers < min_bucket=4
- `rare_earth` — 0 tickers < min_bucket=4
- `regional_banks` — 2 tickers < min_bucket=4
- `reits_datacenter` — 2 tickers < min_bucket=4
- `reits_industrial` — 0 tickers < min_bucket=4
- `reits_residential` — 3 tickers < min_bucket=4
- `reits_tower` — 2 tickers < min_bucket=4
- `restaurants` — 3 tickers < min_bucket=4
- `semi_ai_foundry` — 0 tickers < min_bucket=4
- `solar` — 1 tickers < min_bucket=4
- `space` — 0 tickers < min_bucket=4
- `uranium_miners` — 0 tickers < min_bucket=4
