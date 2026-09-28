# SPY Shares Pivot — Scoping Report
**Date:** 2026-06-04
**Author:** Claude (Head of Quant)
**Trigger:** HC #518 R3 + R5 — ES taker execution dead (gross edge ~0.18 ticks vs 0.376-tick RT commission wall). Same MBO microstructure features may survive on SPY shares because the cost structure is fundamentally different.
**Scope:** Research only. No signups, no spend, no broker commitment.

---

## TL;DR

SPY shares is a viable pivot. The big unlock is commission cost: at IBKR Tiered we pay ~$0.0035/share with a $0.35/order minimum — on a 1000-share scalp that's ~$7 RT, or **0.7 cents/share = 0.7 of a 1-cent tick (~70%)** vs ES's 37.6%-of-tick commission burden alone (ignoring the 100%-of-tick spread cross). Plus the **PDT rule was eliminated as of TODAY (2026-06-04)** — $25k minimum gone, no more day-trade counting. Major timing tailwind.

The hard part is **data fragmentation**. SPY is NYSE Arca-listed but trades across ~15 lit exchanges + 30+ ATSs. Our ES queue/OFI/sweep features assumed one venue (CME). For SPY they need to be either single-venue (Arca depth only) or composite (NBBO + per-venue depth). Databento US Equities is the closest match to what we have on ES — same vendor, same API, similar schemas.

**Honest timeline to first SPY paper trade: 6–10 weeks.** Most of it is feature-pipeline rework and CNN-Mamba retraining on equity microstructure.

---

## 1. SPY Data Feeds

SPY's primary listing is **NYSE Arca**. To replicate "what Databento gives us for ES" (full MBO history + live), you need either Arca depth in MBO form or a consolidated multi-venue feed.

### Comparison Table

| Provider | Closest equivalent | Live monthly | Historical depth | What you get | Latency class | Fit for our use |
|---|---|---|---|---|---|---|
| **Databento US Equities** | EQUS family (XNYS.PILLAR, ARCX.PILLAR, XNAS.ITCH, EDGX, etc.) | Standard $199/mo (live unlimited; L2/L3 limited to 1 month historical on that tier; Plus/Unlimited = contract) | XNAS.ITCH from 2018; Arca PILLAR similar | Full MBO (L3), MBP-10, MBP-1, trades, BBO. Same Python/Rust client as our ES pipeline. | Tick-by-tick, microsecond ts | **BEST FIT.** Same vendor, same code path. |
| **Polygon.io / Massive** | Stocks Advanced | ~$199–$2k/mo depending on tier (pricing page now requires walking through plan selector; old "Advanced" tier was $199) | Multi-year trades + NBBO; **no true MBO** — order book is NBBO + per-exchange aggregated L2 | Trades, NBBO, aggregated L2 quotes, websocket | Millisecond | OK for trades + NBBO; **not MBO**. Won't reproduce queue-position features. |
| **IEX DEEP + TOPS (HIST)** | DEEP = depth-of-book aggregate per price; TOPS = top of book | **FREE** | Trailing 12 months T+1 download | Aggregated displayed orders per price/side (not per-order). IEX only (~2–3% of SPY volume). | T+1 historical; real-time also free | Free testbed for plumbing, but IEX is a tiny slice of SPY flow. Will NOT validate feature alpha. |
| **NASDAQ TotalView-ITCH** | XNAS.ITCH (full MBO) | Direct from Nasdaq: enterprise pricing, ~$3k+/mo + display fees. Via Databento: bundled in EQUS. | From 2014 | Full MBO for Nasdaq-listed and Nasdaq trading. **SPY is Arca-listed**, so XNAS only sees the Nasdaq-routed slice of SPY — not the primary book. | Nanosecond | Useful as one venue in a composite, not standalone. |
| **NYSE Integrated / Arca XDP** | ARCX.PILLAR via Databento, or direct from NYSE | Direct: enterprise. Via Databento: bundled. | Multi-year via Databento | Full MBO for Arca — SPY's primary listing venue. | Nanosecond | **The key feed if going single-venue first.** |
| **dxFeed** | Equities depth | $300–$1000+/mo per use case; enterprise quotes | Multi-year | Aggregated L2, trades; not strict MBO across all venues | Millisecond | Decent backup; less code-reuse vs Databento. |
| **Algoseek** | Historical only — tick + L2 reconstruction | $500–$5000 per dataset, one-time historical | Decades | Historical reconstructed full-book parquet | N/A (historical only) | Good for backtesting cheaply; no live feed. |

### Recommendation (data)
**Start with Databento US Equities Standard ($199/mo)** + use ARCX.PILLAR (Arca MBO) as the single-venue feed.
- Pros: same vendor as ES, our `rithmic_client.py` -> `BBOEvent`/`TradeEvent` -> `mbo_recorder.py` design generalizes cleanly because Databento ships the same MBO/MBP-10 schemas the recorder already understands.
- Cons: Standard tier only gives **1 month of L3/MBO history**. For multi-month training data we need the Plus tier (contract pricing — typical range $500–$2000/mo for serious depth coverage). Budget assumption: **~$500–$1500/mo for production usage** after we move past prototyping.
- Daily data size estimate: SPY MBO on a single venue (Arca) is roughly **1–3 GB/day compressed DBN**, vs ~500MB/day for ES MBO on CME. Storage is manageable.

If money matters: **bootstrap on free IEX HIST** for plumbing port (1 week), then graduate to Databento Standard for real training data.

---

## 2. Brokers

Cost wall analysis — the central question is: **on a 1000-share SPY scalp, what is round-trip cost as a fraction of a 1-cent tick?**

### Comparison Table

| Broker | Commission | 1000-share RT cost | Cost as % of $0.01 tick | API | Order types | Scalp friendly? |
|---|---|---|---|---|---|---|
| **Interactive Brokers (Tiered)** | $0.0035/share, $0.35 min/order, exchange fees passed through | ~$7 base + ~$2–4 routing/regulatory ≈ **$9–11 RT** | ~0.9–1.1 cents/share = **~90–110% of one tick** | TWS API, IB Gateway, FIX (CTCI for >$30k/mo), websocket via ib_insync | Limit, IOC, MOO, MOC, hidden, midpoint peg, ISO, adaptive | Yes. Tolerated up to very high msg rates. Directed orders must use Fixed pricing. |
| **Interactive Brokers (Fixed)** | $0.005/share, $1 min | ~$10 RT | **~100% of one tick** | Same | Same | Yes, but Tiered cheaper at our volume. |
| **Alpaca** | $0/commission | $0 (regulatory fees only) | **~0%** | REST + websocket | Limit, market, stop, bracket. **No hidden, no peg, no ISO, limited routing.** | "Commission-free" but Alpaca explicitly reserves the right to penalize non-retail flow. PFOF model means **fills can be worse than NBBO** and very high msg rates risk throttling/banning. Bad for HFT scalping; OK for paper. |
| **Tradier (Pro $10/mo)** | $0 equity | ~$10/mo flat | Excellent if you do lots of trades | REST | Limit, market, stop, OCO. Equity routing is basic. | Workable; less aggressive routing than IBKR. |
| **Lightspeed (per-share)** | $0.001–$0.0035/share, $0.25 min | ~$2–7 RT + routing | **~20–70% of tick** | DAS Trader, Sterling, FIX | Full pro suite (ISO, hidden, midpoint, peg) | Yes — built for active traders. Higher min monthly activity. |
| **CenterPoint** | $0.003/share + routing (entry) | ~$6 + routing | **~60%+** of tick | DAS Trader Pro ($120/mo, waived at 200k+ shares) | Full pro suite, locate desk for borrows | Yes — built for active traders & short sellers. |
| **DAS Trader Pro** | Platform, not broker (front-end for above) | N/A | N/A | DDE/FIX | All pro types | Use with Lightspeed/CenterPoint. |
| **Wedbush** | Enterprise prime-broker — $0.0015–$0.003/share with minimum monthly $1k–$5k commitment | Variable | Best at scale | FIX | Full | Prime-broker tier; overkill unless we're trading >1M shares/day. |

### PDT Rule — MAJOR UPDATE (2026-06-04)
**Effective today**, FINRA Rule 4210 amendment **eliminates the PDT framework**:
- No more $25k minimum equity for day-trader accounts (only $2k margin minimum).
- No more "4 day-trades in 5 business days" trigger.
- Replaced with real-time Intraday Margin Buying Power (IMBP), scaled by an intraday multiplier per security.
- Brokers have until 2027-10-20 to phase in. IBKR/Alpaca/Tradier may not flip the switch on Day 1 — verify with broker before assuming.

**Implication for us:** We can paper/live trade SPY scalps at small account sizes (e.g., $5–10k) without PDT friction. Removes a major historical objection to retail HFT.

### Recommendation (broker)
- **Paper trading**: Use Alpaca first (free, easy API) for initial wiring.
- **Live cutover**: **IBKR Tiered** as primary. The ~$10 RT cost on 1000 shares is the floor for retail. Lightspeed/CenterPoint are cheaper if we scale to >500k shares/month.
- **Hard truth**: Even at IBKR's $10 RT (~1 cent/share = ~1 tick), **passive limit fills only barely break even**. We need the model to generate >1 cent of expected edge per trade, OR scale to bigger size where commission becomes <0.1 tick per share equivalent. On 5000-share clips IBKR's pricing is ~$17.5 RT = 0.35 cents/share = **35% of tick** — closer to ES's structural cost but at 5x the per-trade size.

---

## 3. Infra Mapping — What Changes from ES to SPY

### Current ES Stack (Razer live host)
```
Rithmic feed -> rithmic_client.py -> BBOEvent/TradeEvent
            -> mbo_recorder.py (DBN-style fanout, parquet rotate)
            -> streaming_features_smart_v3.py (queue/OFI/sweep/cancel features)
            -> cnn_mamba_v2_inference.py (PyTorch fwd pass, 250ms stride)
            -> paper_trading_v2_1s_short_top05.py (FIFO sim)
```

### What Generalizes (good news)
- **`mbo_recorder.py`** consumes generic `BBOEvent` / `TradeEvent` dataclasses — venue-agnostic by design. Replace the Rithmic ingest with a Databento ingest and the recorder works untouched.
- **`cnn_mamba_v2_inference.py`** consumes a feature tensor. It does NOT know or care about venue. **Zero changes** to model code — only retraining.
- **`paper_engine.py` / FIFO simulator** logic is generic queue replay. Will work on equity book once events arrive in same schema.
- **`streaming_features_smart_v3.py`** queue/OFI/sweep math is the same on equities (it's just book deltas). **Needs re-parameterization** (tick size $0.01 vs $0.25, lot sizes 100 vs 1).

### What Needs to Change (work)
| Area | File / module | Effort |
|---|---|---|
| Feed client | NEW: `databento_equity_client.py` (replaces `rithmic_client.py`). Subscribe to ARCX.PILLAR MBO live + write same `BBOEvent`/`TradeEvent` shape. | 2–4 days |
| Tick/lot constants | `live_trading_linux/instrument_validator.py` and feature configs. Change tick size 0.25 → 0.01, contract value to $/share, point value to 1. | 1 day |
| Order/position model | NEW: equities-specific position tracker. Shares (not contracts), T+1 settlement (T+1 is now standard post-2024), no margin futures multiplier. | 3–5 days |
| Broker client | NEW: `ibkr_client.py` (or `alpaca_client.py` for paper). Send limit/IOC/market, handle partial fills, position reconciliation. | 1–2 weeks (IBKR TWS API is notoriously fiddly) |
| Short selling | NEW: locate / borrow check logic. SPY borrow is essentially free and locatable, but the plumbing must exist for the short-leg of our signals. | 2–3 days |
| Cost constants | `CLAUDE.md` cost section. Add `SPY_TICK = $0.01`, `SPY_RT_COMMISSION_PER_SHARE = $0.005–0.01` (broker-dependent). | <1 day |
| Fill simulator | `inference/fill_simulator.py` — adapt to equities queue. Adverse selection model may need recalibration (HFT firms picking off retail differently than CME locals do). | 1 week |
| Feature pipeline | `streaming_features_smart_v3.py` re-parameterize, validate sweep detection on lit equity book. | 1 week |
| Training data prep | NEW: scripts to pull Databento historical, build same training tensor format as ES. | 1 week |
| Model retraining | CNN-Mamba v2 on SPY MBO. Same code, new data, full WF run on Neptune RTX 3090. | 1–2 weeks compute |
| Multi-venue handling | OPEN QUESTION: do we use Arca-only book, or composite NBBO + per-venue depth? Big design decision (see §5). | 1–3 weeks if composite |

### Files to add / modify (concrete)
- ADD: `live_trading_linux/databento_equity_client.py`
- ADD: `live_trading_linux/ibkr_client.py`
- ADD: `live_trading_linux/equity_position_tracker.py`
- ADD: `live_trading_linux/locate_borrow_checker.py`
- MODIFY: `live_trading_linux/instrument_validator.py` (add SPY config)
- MODIFY: `live_trading_linux/mbo_recorder.py` (parameterize tick size)
- MODIFY: `live_trading_linux/streaming_features_smart_v3.py` (equity tick math)
- MODIFY: `inference/fill_simulator.py` (equity queue model + adverse selection)
- MODIFY: `inference/framework_config.json` (SPY tick/cost constants)
- ADD: `scripts/pull_databento_spy_historical.py`
- ADD: training config under `live_trading/configs/` for SPY CNN-Mamba v2

---

## 4. Realistic Timeline to First SPY Paper Trade

Honest range — assume one engineer (me) full-time, no major surprises.

| Phase | Days | What happens |
|---|---|---|
| 1. Data feed signup + initial pull | 3–5 | Open Databento account, pull 1 month of IEX HIST for free first (plumbing test), then pull paid 1 month Arca PILLAR MBO. |
| 2. Recorder + storage port | 3–5 | Wire Databento client into `mbo_recorder.py` shape. Validate event counts vs raw DBN. |
| 3. Feature pipeline port | 5–10 | Re-parameterize streaming features for $0.01 tick, lot-size 100, validate queue dynamics on equities (very different cancel/replace patterns vs CME). |
| 4. Historical training data build | 5–7 | Pull 6–12 months of SPY MBO history (will need Databento Plus tier to get >1mo; may need to negotiate or use Algoseek one-time). Build feature tensors. |
| 5. CNN-Mamba v2 retrain on SPY | 10–14 | Sliding-window WF on Neptune. 60d train, 1d OOT. Need ≥40 OOT days for HC #428 R1 regime-agnostic validation. **This is the long pole.** |
| 6. Fill sim recalibration | 5–7 | Equities adverse-selection model. Tune for SPY queue churn. |
| 7. Broker client (IBKR or Alpaca) | 7–14 | Build broker client, paper-trade-only mode. IBKR TWS is the time-sink; Alpaca is faster. |
| 8. Integration + first paper trade | 3–5 | Wire it all together, dry run, send first paper order. |
| 9. Buffer for unknowns | 5–10 | Fragmentation surprises, latency tuning, etc. |
| **TOTAL** | **~46–77 days** | **6–11 weeks to first SPY paper trade.** |

Best case (Alpaca paper, IEX-only data, no multi-venue work, smooth retrain): **~5 weeks.**
Realistic mid case: **~7–8 weeks.**
If we go composite multi-venue from day 1: **+3 weeks.**

---

## 5. Open Risks / Gotchas

### Microstructure differences vs CME futures
- **Reg NMS / Order Protection Rule**: every exchange must route to the best price. Means our model's "the book at venue X looks like Y" view is incomplete — flow goes wherever NBBO is best.
- **Fragmentation**: SPY trades on NYSE, Arca, Nasdaq, BX, BZX, EDGX, IEX, MEMX, MIAX Pearl, NYSE American + 30+ dark pools/ATSs. **A single-venue book (e.g., Arca) sees maybe 15–25% of actual SPY flow.** Our queue/OFI features may misread imbalance because we can't see the other 75%.
- **Dark pools / hidden midpoint orders**: ~35–40% of SPY volume executes off-exchange. Sweeps and OFI computed on lit book miss this entirely. Our sweep-intensity feature might generate false positives (lit-book "exhaustion" while dark pools are absorbing).
- **Odd lots**: pre-2025 odd-lot trades didn't print to SIP. Now they do (post Reg NMS odd-lot rule). Still, your average HFT scalp on SPY happens in 100-share round lots; sub-100 lots act differently.

### Does our edge survive?
**Unknown until we test.** Reasonable priors:
- Queue/OFI on a single liquid venue (Arca for SPY) should work — Arca is the primary book.
- Sweep intensity is harder — a "sweep" in equities means ISO orders hitting multiple venues simultaneously. Detection requires composite tape, not single-venue book.
- Cancel asymmetry should translate — equity market makers cancel/replace constantly, same as CME locals.
- **Honest expectation**: edge will be smaller than ES (more competition, faster decay, more fragmentation). But the cost wall is also lower if we scale share size, so net Sharpe could still be positive.

### Borrow / locate for shorts
- SPY is on every broker's Easy-To-Borrow list. Borrow rate ~0%. **Not a constraint** for our short-biased signals.
- Caveat: in extreme market stress days (1–2x/year) SPY borrow can briefly tighten. Plumbing must handle locate-rejected orders gracefully.

### Round-lot vs odd-lot economics
- IBKR Tiered min is $0.35/order. For SPY at ~$650/share, a 100-share order = $65,000 notional. So minimum order economically must be ≥100 shares to keep commission <1bp.
- Sub-100-share orders are commission-prohibitive at IBKR.

### Other flags
- **Latency**: SPY HFT is a microsecond game at the top. Razer over residential internet to NYC datacenters = 30–80ms round trip. **We will not compete with co-located HFT.** Our edge must be in horizons where latency doesn't matter (≥250ms hold time, like we already do).
- **PFOF concerns at Alpaca**: payment for order flow means our fills may be deliberately delayed/worsened. Bad for scalping signals at 1s horizon. Use IBKR for live.
- **Market data display fees**: if we redistribute or display data (even to ourselves in a UI), Nasdaq/NYSE charge per-user/per-month fees ($1–$50/mo). Single-developer use usually OK under "internal use" but worth reading the EULA.
- **Regulatory**: high-frequency message rates (orders+cancels > 1000/sec) can trigger SEC Rule 15c3-5 questions at the broker. We are nowhere near this; flag it for future scale.
- **Tax**: equity scalping = short-term cap gains (ordinary income) + wash-sale rules. Different from 60/40 1256 treatment on ES futures. **This is a material P&L consideration** the user should know about — losing ~10–15% more of profits to taxes vs futures.

---

## Recommended next step if user greenlights
- **Week 1 sprint:** Sign up for Databento free $125 credit + IEX HIST free download. Build `databento_equity_client.py` and validate `mbo_recorder.py` works end-to-end on real SPY Arca MBO data — proves the pipeline before spending any subscription dollars.
- **Decision gate at end of Week 1:** Is the data shape clean? Are queue/OFI features computable? If yes → commit to Databento Standard ($199/mo) and start historical pull. If no → re-evaluate.
- **Parallel track:** I begin Alpaca broker-client scaffolding (free, no commitment) so we have paper-trade plumbing ready by the time the retrained CNN-Mamba v2 produces signals. IBKR client is built later, only after paper trading validates the signal survives equity microstructure.

---

## Sources
- [Databento Pricing](https://databento.com/pricing)
- [Databento US Equities launch](https://databento.com/blog/introducing-databento-us-equities)
- [Databento XNAS.ITCH dataset](https://databento.com/datasets/XNAS.ITCH)
- [IEX DEEP/TOPS — free](https://iextrading.com/trading/market-data/)
- [Polygon.io / Massive pricing](https://massive.com/pricing)
- [IBKR commissions](https://www.interactivebrokers.com/en/pricing/commissions-stocks.php)
- [Alpaca market data + trading](https://alpaca.markets/data)
- [Tradier pricing](https://tradier.com/individuals/pricing)
- [Lightspeed pricing](https://lightspeed.com/pricing-fees/stocks-etfs)
- [CenterPoint review 2026](https://bullishbears.com/centerpoint-securities-review/)
- [FINRA PDT Rule eliminated 2026-06-04 — Notice 26-10](https://www.finra.org/rules-guidance/notices/26-10)
- [Stocktitan: PDT $25K eliminated](https://www.stocktitan.net/articles/pattern-day-trader-rule-eliminated-2026)
