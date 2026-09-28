# VIX Data Sourcing Feasibility — v3.3 side-research

**Date:** 2026-05-12
**Triggered by:** HC #304(D) #5

## Question
Can we get historical 1m VIX (cash index) bars 2023-2026 for CNN-Mamba v3.3 T3 tier feature?

## Options

| Source | Cost | 1m bar coverage | Hist depth | API quality | Verdict |
|--------|------|-----------------|------------|-------------|---------|
| **Yahoo Finance** (`^VIX`) | FREE | 1m bars | **~30 days only** (max for 1m granularity) | Public, no auth | **TOO SHORT** — useless for training |
| **Yahoo Finance** (`^VIX`) | FREE | 5m / hourly / daily | Decades | Public | OK for daily/hourly fallback |
| **Polygon.io** | $29/mo (Starter) → $99/mo (Developer) | **1m bars, full history** | Decades for indices | REST + WebSocket, well-documented | **RECOMMENDED** ✅ |
| **Databento** | already paying | Need to check catalog (TBD task #6) | Likely 2018+ | Same format as our ES MBO | **PREFERRED if catalog covers VIX cash** — single-vendor wins |
| **CBOE DataShop** | $$$$ enterprise | Authoritative source | Decades | Custom file formats | Too expensive for prototyping |
| **TradingView** | $14.95-$49.95/mo Pro | 1m bars, paywalled export | Decades | Not really an API — chart-based | Not for production pipeline |
| **historical-data.com** | $99 one-time / instrument | 1m, full history | Decades | CSV download | Cheap one-time option |

## Related instruments needed for v3.3 IV regime feature family:
- **VIX (^VIX)** — spot index, cash-settled
- **VIX9D (^VIX9D)** — 9-day VIX (short-end term structure)
- **VIX3M (^VIX3M)** — 3-month VIX (mid-term)
- **VX1 / VX2** — front and 2nd-month VIX futures (CFE, Databento covers)
- **VVIX (^VVIX)** — vol-of-vol, useful for regime shifts

## Recommended path (cheapest viable):
1. **First**: Check Databento catalog (side-research task #6) for VIX cash + VIX9D + VIX3M. If covered → use Databento, no new vendor.
2. **If Databento doesn't cover VIX cash**: Polygon.io Starter ($29/mo) for VIX/VIX9D/VIX3M/VVIX 1m bars + Databento for VX1/VX2 futures (CFE).
3. **Fallback**: 5m VIX from Yahoo (free) — coarser but covers training window. Use only if budget is hard-blocked.

## Storage estimate
- 1m bars, 4 instruments, RTH only (~390 min/day), 252 days/yr, 4 years (2023-2026)
- = 4 × 390 × 252 × 4 = ~1.6M rows
- ~150 MB as Parquet (with compression) — trivial

## Open questions for user
- Approve Polygon.io $29/mo IF Databento doesn't carry VIX cash?
- Or stick with daily/5m VIX from Yahoo as v3.3 v0 → upgrade to 1m later?

## Next step
- Run Databento catalog query (side-research task #6) tomorrow to determine if path 1 (no new vendor) is viable.

---
**STATUS:** Decision pending Databento catalog check (task #6). v3.3 design can proceed assuming 1m VIX is obtainable for $0-29/mo + small storage cost.
