# Data Inventory — HC #558 R1

Single canonical inventory of all data feeding the wheel + macro stock-picker strategies. Both strategies read from these cached parquets in `data/cache/`.

## Available Now

| Dataset | Shape | Coverage | Frequency | Key columns | Notes |
|---|---|---|---|---|---|
| `universe.parquet` | (70, 4) | snapshot | static | ticker, name, sector, source | 70 large-cap names. 8 sectors. |
| `prices.parquet` | (183,795, 12) | 2015-01-02 → 2026-03-09 | daily | ticker, date, OHLCV, ret, log_ret, rv_20/60/252 | Adjusted, includes realized vol. |
| `fundamentals.parquet` | (70, 18) | snapshot 2026-03-09 | **static only** | ticker, market_cap, beta, pe, ps, fcf_yield, debt_to_equity, gross/ebitda/net_margin, current_ratio, roe_proxy, revenue_ttm, fcf_ttm, sector, industry, fund_score | **LIMITATION: snapshot, not time series. YoY growth signals require historical refresh.** |
| `price_features.parquet` | (70, 13) | snapshot | static | rv_20/60/252, max_dd_pct, atr14_pct, gap_*, tot_ret_1y/3y/5y, last_close | Per-ticker price-derived characteristics. |
| `iv_features_real_blend.parquet` | (183,795, 11) | 2015-01-02 → 2026-03-09 | daily | sigma, sigma_atm_30d, iv_rank, term_ratio, iv_rv_ratio, r_1m, pricing_source | Real DOLT IV blended with modeled fallback. 51% real / 49% modeled. |
| `macro.parquet` | (2,982, 6) | 2015-01-01 → 2026-06-05 | daily | vix, vix3m, dxy, naaim, vix_ts | Core risk state. |
| `macro_extra.parquet` | (4,286, 29) | 2010-01-01 → 2026-06-05 | daily | biz_inv, labor_part, cpi, oecd_cci_us, durable_goods, ust_2y/10y/30y, fed_funds, housing_starts, new_home_sales, claims, sentiment, BE inflation | FRED-sourced macro depth. Yields curve full. |
| `regime_overlay.parquet` | (4,286, 8) | 2010-01-01 → 2026-06-05 | daily | yc_inverted, claims_spike, be_inflation_hot, sentiment_shock, ff_tightening, gates_on, risk_off | Pre-computed regime flags. Used by wheel engine. |
| `ga_name_table.parquet` | (70, 14) | snapshot | static | adv20_usd, pe, fcf_yield, gross_margin | Per-name characteristics for GA. |

## Universe by Sector

| Sector | Count | Example tickers |
|---|---|---|
| Technology | 17 | AAPL, ADBE, AMD, ARM, CRM, GOOGL (also Comm), MSFT, NVDA, PLTR |
| Financial Services | 15 | AXP, BAC, BLK, BRK-B, C, JPM, V, MA, SCHW |
| Consumer Cyclical | 9 | ABNB, AMZN, F, GM, HD, NKE, SBUX, TSLA |
| Consumer Defensive | 7 | CL, COST, KO, PEP, PG, WMT |
| Communication Services | 7 | DIS, GOOGL, META, NFLX, T, VZ |
| Healthcare | 6 | ABBV, JNJ, LLY, MRNA, PFE, UNH |
| Industrials | 5 | BA, CAT, DE, GE, RTX |
| Energy | 4 | CVX, OXY, SLB, XOM |

## Required But Missing (HC #558 R3 prerequisites)

1. **Sector ETF prices** — XLK, XLF, XLV, XLE, XLY, XLP, XLI, XLU, XLB, XLRE, XLC (+ SMH for semis specifically, XBI for biotech, ARKK for innovation). Required for sector-rotation feature. **Action: ingest via yfinance, 2015-2026.**
2. **Fund flows** — true sector AUM flow data is paid (ETF.com, Bloomberg). **Proxy**: sector-ETF dollar-volume (volume × close) z-scored over 20/60/252 days = "money moving into the sector" signal. Action: derived from sector-ETF prices once ingested.
3. **Time-series fundamentals** — revenue YoY, eps YoY require quarterly back-fills. Current snapshot is single point. **Action: queue FMP / yfinance quarterly pull.** Not blocking for v1 since price-momentum + ETF-flow proxies carry the rotation signal.
4. **Thematic exposure tags** — "AI", "physical AI", "semis", "SaaS" tags per ticker. **Action: hand-curate small thematic baskets from universe + use sector-ETF flows as proxy.**

## Data Hygiene

- All dates UTC midnight, tz-naive.
- Prices already adjusted for splits/dividends (yfinance source).
- IV blend has `pricing_source` column — "real" or "modeled" — strategies should weight observation confidence accordingly.
- Macro is forward-filled to daily at ingest, raw frequency varies (monthly/weekly/daily).

## What Both Strategies Can Use Right Now

**Wheel ranking upgrade (HC #558 R2):** sector × fund_score × ret_3m × macro regime × IV-rank → ranked universe. All inputs available.

**Macro picker v1 (HC #558 R3):** prices (cross-sectional momentum) × sector_etfs (rotation, once ingested) × macro regime (gate). Buildable today after sector ETF ingest.

---
Last updated: 2026-06-07
