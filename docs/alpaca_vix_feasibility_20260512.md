# Alpaca VIX Feasibility Report — 2026-05-12 21:48 ET

**Context**: HC #305(D.7) — User mandate: NO new paid data subscriptions. Probe Alpaca free tier for VIX-related symbols.
**Probe script**: `/tmp/alpaca_vix_probe.py` (raw results: `/tmp/alpaca_vix_probe_results.json`)
**Credentials used**: existing paper-trading keys from `alpha_discovery/fetch_spy_alpaca.py` (already in repo, free tier)
**Data feed tested**: IEX (Alpaca free-tier default, sufficient for ETFs)

## Headline result
**FREE 1-min VIX-proxy data confirmed available with ≥2-year history depth via Alpaca free tier.** Zero new subscription required.

## Symbol-by-symbol findings

| Symbol | Type | Latest price (2026-05-12 19:59 UTC) | 1m bars 30d back | 1m bars 2y back | Notes |
|---|---|---|---|---|---|
| **VXX** | ETN | $27.84 | ✅ 200 | ✅ 200 | iPath VXX, front-month VIX futures ETN — canonical VIX proxy |
| **VIXY** | ETF | $26.86 | ✅ 200 | ✅ 200 | ProShares VIX Short-Term Futures, higher retail volume (n_trades=17 vs VXX's 4 in sample minute) |
| **UVXY** | ETF | $35.74 | ✅ 200 | ✅ 200 | ProShares Ultra VIX (1.5x leveraged) — useful for amplifying low-vol signal but decays heavily |
| **VIXM** | ETF | $15.82 | ✅ 200 | ⚠️ no bars in sample minute | Mid-term VIX futures (VX2-VX5 weighted) — useful for **term-structure** signal |
| **SVXY** | ETF | $51.44 | ✅ 200 | ✅ 200 | Inverse VIX (short front-month) — useful for sign-flipped check / curve carry |
| `^VIX` | INDEX | — | ❌ 400 invalid symbol | — | Alpaca does NOT serve VIX cash index (equity feed only — expected) |
| `VIX` | INDEX | — | ❌ 404 no bar found | — | Same — not available |

## Recommendation for v3.2.1 / v3.3 T3 features

Use **VXX as primary VIX-level proxy** (most-liquid VIX futures ETN, canonical reference). Pull 1m bars, resample to T3 1Hz cadence via forward-fill (VIX moves slow vs MBO event rate — no information lost).

### Proposed T3 features (4-7 features depending on inclusion of term structure)

| Feature name | Definition | Rationale |
|---|---|---|
| `vxx_level_z20d` | VXX close price, rolling 20d z-score | Vol-regime level (current vs recent baseline) |
| `vxx_change_5m_bps` | log(VXX_t / VXX_{t-5m}) × 10000, clipped ±500 bps | Short-term vol panic detection |
| `vxx_change_30m_bps` | log(VXX_t / VXX_{t-30m}) × 10000, clipped ±500 bps | Medium-term vol regime drift |
| `vxx_vs_realized_vol_gap` | VXX z-score minus LGBM-5m-vol z-score | IV vs RV gap (compressed = potential mean revert) |
| **Stretch — term structure (3 more if VIXM 2y history fills in)**: | | |
| `vix_term_slope_vxx_vs_vixm` | log(VXX/VIXM) z-score | Contango vs backwardation regime |
| `vix_term_change_5m` | log(VXX/VIXM)_t − log(VXX/VIXM)_{t-5m} | Term-structure dynamics |
| `vix_term_class_3way` | One-hot {steep_contango, flat, backwardation} from rolling 20d quantiles | Categorical regime gate |

Total: 4 features (basic) or 7 features (with term structure). Adds **4-7 features to T3** dim count (31 → 35-38, on top of HC #305(D.4)'s 6 vol-regime-class features and HC #305(D.5)'s 6 datetime/countdown features → final T3: 43-49).

## Data pipeline plan (Jupiter, post-v3.2-verdict)

1. **Backfill historical 1m bars** — `scripts/v3_3_research/fetch_alpaca_vix.py`:
   - Pull VXX, VIXY, VIXM 1m bars from 2024-01-01 (covers full ES MBO data range + 12mo buffer for rolling z-scores)
   - Output: `data/external/vix_proxies_1m_2024_2026.parquet` (estimated ~120k rows × 3 symbols × 5 cols OHLCV = trivial size ~3 MB)
   - Free tier rate limit: 200 req/min (Alpaca free), each request returns 10k bars — full backfill in <5 min
2. **Daily incremental update** — cron Jupiter 09:00 ET weekdays, pulls last 24h, appends to parquet
3. **Feature builder** — extend `alpha_discovery/features/build_v3_2_1_tier_features.py` to join VXX features into T3 snapshots at 1Hz cadence via `pd.merge_asof` (existing pattern from LGBM vol joins)

## Risks / caveats

1. **VXX is futures-tracking, not cash VIX** — decays via roll yield (~5-15% / year contango drag). Z-scoring against rolling 20d mitigates but doesn't eliminate.
2. **IEX feed may have sub-second gaps in low-volume periods** for less-liquid VIXM. Forward-fill resampling handles.
3. **2-year history depth confirmed** but coverage of pre-2024 data not tested (probe started at 2024-05 — sufficient for our MBO range starting 2024-07).
4. **No live-stream from Alpaca free tier** (15-min delayed quotes only on free; historical bars are fine, live trading needs Razer's existing pipeline).
5. **No paid Alpaca upgrade required** for any of this. Free tier IEX feed sufficient for 1m bar history.

## Cost: **$0 / month**. Effort: ~3-4 hours Jupiter CPU dev + 5 min API backfill. Inclusion in v3.2.1 = recommended.

---
Generated 2026-05-12 21:48 ET by Recovery #26 session, per HC #305(D.7) + HC #305(H).
