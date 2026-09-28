# Sector picker v8 — Sub-industry (K=10, hold=10d)

**Taxonomy**: 44 sub-industries, 259 ticker mappings
**Buckets active / skipped**: 11 / 22 (min size = 4)
**Wall**: 56.2s

## Pooled-OOT

| Metric | Strategy | SPY 1x | SPY 1.5x | SPY 2x |
|---|---:|---:|---:|---:|
| Sharpe | 0.09 | 0.69 | 0.62 | 0.58 |
| Sortino | 0.12 | 0.83 | 0.75 | 0.71 |
| CAGR | 0.2% | 12.2% | 14.8% | 16.3% |
| MaxDD | -7.2% | -34.1% | -47.8% | -59.4% |
| Calmar | 0.02 | 0.36 | 0.31 | 0.27 |

**Deployable sub-industries (Sharpe ≥ 1.0)**: 0 / 11
**Pooled passes Calmar ≥ 1.5**: NO

## Per-sub-industry pooled-OOT (sorted by Sharpe)

| Sub-industry | N | Sharpe | CAGR | MaxDD | Calmar | PF | WR |
|---|---:|---:|---:|---:|---:|---:|---:|
| semiconductors | 18 | 0.53 | 1.0% | -3.6% | 0.28 | 1.10 | 46.4% |
| p_c_insurance | 6 | 0.34 | 0.9% | -5.8% | 0.15 | 1.07 | 46.5% |
| payments | 6 | 0.24 | 1.0% | -15.4% | 0.06 | 1.05 | 45.8% |
| defense_tech | 10 | 0.22 | 1.7% | -22.3% | 0.07 | 1.06 | 44.9% |
| big_pharma | 6 | 0.14 | 0.5% | -8.3% | 0.05 | 1.03 | 45.8% |
| ai_software | 12 | 0.09 | 0.2% | -10.6% | 0.02 | 1.02 | 46.4% |
| physical_ai_robotics | 9 | -0.02 | -0.9% | -28.1% | -0.03 | 1.00 | 44.2% |
| nuclear | 6 | -0.19 | -4.0% | -30.3% | -0.13 | 0.96 | 42.7% |
| cloud_saas | 9 | -0.22 | -0.7% | -12.4% | -0.06 | 0.96 | 44.7% |
| big_banks | 6 | -0.55 | -1.3% | -11.7% | -0.11 | 0.90 | 44.4% |
| meddev | 7 | -0.57 | -1.5% | -12.4% | -0.12 | 0.90 | 44.5% |

## Skipped buckets

- `biotech_smid` — 2 tickers < min_bucket=4
- `brokerage` — 3 tickers < min_bucket=4
- `copper` — 1 tickers < min_bucket=4
- `crypto_proxy` — 1 tickers < min_bucket=4
- `cybersecurity` — 3 tickers < min_bucket=4
- `e_commerce` — 1 tickers < min_bucket=4
- `ev_auto` — 3 tickers < min_bucket=4
- `glp1` — 1 tickers < min_bucket=4
- `gold` — 1 tickers < min_bucket=4
- `hydrogen` — 1 tickers < min_bucket=4
- `lithium` — 1 tickers < min_bucket=4
- `luxury` — 2 tickers < min_bucket=4
- `natgas` — 2 tickers < min_bucket=4
- `oil_services` — 3 tickers < min_bucket=4
- `quantum` — 3 tickers < min_bucket=4
- `rare_earth` — 1 tickers < min_bucket=4
- `reits_datacenter` — 2 tickers < min_bucket=4
- `reits_industrial` — 1 tickers < min_bucket=4
- `reits_tower` — 3 tickers < min_bucket=4
- `semi_ai_foundry` — 0 tickers < min_bucket=4
- `solar` — 2 tickers < min_bucket=4
- `uranium_miners` — 1 tickers < min_bucket=4
