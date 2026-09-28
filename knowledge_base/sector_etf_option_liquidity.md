# Sector ETF Option Liquidity — Bull Call Spreads at 4% OTM

**Validation date:** 2026-07-28  
**Strategy:** Bull call spreads / bear put spreads at 4% OTM, DTE ≈ 28 days  
**Methodology:** Live Robinhood option chain data pulled via MCP tools, nearest monthly expiry, 4% OTM short leg + ~3% wide long leg  
**Universe:** 11 sector ETFs in V10 paper engine

---

## Liquidity Tiers (as of 2026-07-28)

### Tier 1 — Tradeable (OI ≥ 10 on both legs)

| ETF | Sector | Short-leg OI | Notes |
|-----|--------|-------------|-------|
| XLE | Energy | 14–113 | Solid two-way market, tight spreads |
| XLF | Financials | 14–113 | Active, most liquid of the group |

### Tier 2 — Thin but Possible (OI 3–9; marginal)

| ETF | Sector | Short-leg OI | Notes |
|-----|--------|-------------|-------|
| XLK | Technology | 3–7 | Light OI — may slip on entry; monitor |

### Tier 3 — Untradeable (zero bids on short leg)

| ETF | Sector | Issue |
|-----|--------|-------|
| XLV | Healthcare | Zero bid on short leg at 4% OTM |
| XLC | Communication | Zero bid on short leg at 4% OTM |
| XLY | Consumer Disc. | Zero bid on short leg at 4% OTM |
| XLP | Consumer Staples | Likely untradeable (not explicitly tested) |
| XLI | Industrials | Likely untradeable (not explicitly tested) |
| XLB | Materials | Likely untradeable (not explicitly tested) |
| XLU | Utilities | Likely untradeable (not explicitly tested) |
| XLRE | Real Estate | Likely untradeable (not explicitly tested) |

---

## Engine Config Update

The V10 paper engine (`paper_engines/sector_combined_v10_optimal_paper.py`) was updated 2026-07-28 to gate entries on a minimum OI check:

- **`MIN_OPTION_OI = 10`** — both legs must meet this threshold
- Uses yfinance option chain data at runtime (same data source as price download)
- Tickers that fail are logged with action `SKIP_LIQUIDITY` in the trade log
- Fail-open on data errors (allows trade rather than silently blocking on a data glitch)

---

## Practical Implication for V10

V10 picks top-4 / bottom-4 from 11 sectors. Only XLE and XLF are reliably above the OI threshold. XLK is borderline. This means in practice:

- **High-VIX mode (top-2 bull picks):** Only works if LGBM selects XLE/XLF in the top 2
- **Low-VIX mode (4 long + 4 short):** Likely 2–3 of the 8 positions will be skipped for liquidity; effective portfolio will be smaller than the 8-position design assumes

**Recommendation:** Consider reducing the sector universe in V11 to just the liquid tickers (XLE, XLF, plus XLK with elevated caution), or lowering the OTM moneyness to 2–3% where a wider set of sectors has tradeable two-sided markets.

---

## Next Validation

Re-run this check at the next monthly rebalance (first Friday of August 2026) to see if summer liquidity improves as vol picks up.
