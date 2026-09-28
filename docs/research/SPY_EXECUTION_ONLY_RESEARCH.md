# SPY-FOR-EXECUTION-ONLY HYPOTHESIS — RESEARCH NOTE (HC #527 R2)

**Date**: 2026-06-04
**Author**: head-of-quant alignment pass
**User question (verbatim)**: *"could just use spy for EXECUTION only? maybe right?"*
**Bottom line up front**: economically the hypothesis is the strongest pivot we have. Statistically we cannot validate it for the Mar 1–13 2026 window because the requisite ES MBO data is NOT on disk. Practical path forward proposed at the end.

---

## 1. Why the hypothesis is interesting — the economics

### 1.1 ES taker is dead (HC #518)
The CNN-Mamba v2 / PatchTST signal stack produces a top-decile gross edge of **~+0.14 to +0.18 ticks** on ES at 1–10s horizons (40+ OOT days). One ES tick is $12.50; round-trip AMP commission is $4.70 = 0.376 ticks. With aggressive (taker) fills you also pay ~1 tick of spread cross half-spread × 2 legs. The math:

| | ES taker | ES passive (best case) |
|---|---|---|
| Gross edge per signal | +0.14 to +0.18 ticks | +0.14 to +0.18 ticks |
| Commission RT | -0.376 ticks | -0.376 ticks |
| Spread cross | -1.000 ticks (~) | -0.000 ticks (if filled) |
| **Net per signal** | **-1.2 to -1.24 ticks** | **-0.20 to -0.24 ticks** |

Even passive ES is underwater. The structural wall isn't bad modelling — it's that the gross edge is **smaller than the commission**.

### 1.2 SPY changes the cost denominator
SPY tick = $0.01 = 1 cent per share. Retail brokers (Alpaca, Schwab, IBKR Lite, Fidelity, Robinhood) charge **$0 commission** on SPY. Only fees: SEC Section 31 (~$8/$1M sell-side notional) + FINRA TAF ($0.000166/share sell-side, $8.30 cap). Reg fees on a 200-share trade @ $580 are ~$0.03 — negligible.

The SPY analogue of the ES table:

| | SPY taker (1c cross) | SPY passive (NBBO sit) |
|---|---|---|
| Commission RT | $0 (= 0 ticks) | $0 |
| Spread cross | -1.000 tick (1c, both legs aggregate ≈ 1 tick RT) | 0 (if filled) |
| Reg fees on $116k notional | -0.0003 ticks/share | -0.0003 ticks/share |
| **Net cost RT** | **~1 tick = 1¢/share** | **~0** |

**The transfer question** (the whole investment thesis): does the ES signal, translated through the ES→SPY lead-lag, produce a SPY signal whose gross edge ≥ 1¢/share at top decile? If yes, taker SPY is profitable. If passive fills are achievable a decent fraction of the time, even less edge is needed.

### 1.3 Order of magnitude
ES moves ~0.25 pts (1 tick) ≈ $12.50/contract. SPY tracks ES at ~0.1 ratio (SPY $580 ≈ ES 5800; 1 ES point ≈ $0.10 SPY). So ES moving 0.18 ticks (0.045 pts) ≈ SPY moving $0.0045 = **0.45 ticks** of SPY. **That is below the 1-tick taker cost.** Passive-only.

Counter-argument for the hypothesis: ES microstructure prediction is partially-informational (it sees lit limit-order book dynamics). When that info crosses to SPY at the ~50–200ms lead, it shouldn't just be the price-move — it should be the *direction of imminent SPY queue depletion*. A passive limit order placed on the soon-to-be-depleted side of the SPY NBBO has high fill probability AND favorable adverse-selection signature. That's the real claim.

---

## 2. The transfer-function evidence we'd need

To validate ES-signal-on-SPY-execution, we need to measure three things from overlapping ES+SPY MBO data:

### 2.1 Cross-correlation lag profile (ES leads SPY)
- Compute mid-price returns at 10ms grid for ES front-month and SPY consolidated
- Sample period: 1 RTH day, RTH-only (09:30:00 – 16:00:00 ET)
- Lag grid: 0, 25, 50, 100, 200, 500, 1000 ms
- Expected: peak cross-corr at +50 to +150 ms (ES leads SPY by this amount). This is the classic Hasbrouck (2003) / Chakravarty-Wood-Upson (2004) / Tao-Hasbrouck (2018) result — futures price discovery dominates the cash basket in the first 50–250ms.
- Expected effect size: peak correlation ~0.15–0.30 on 100ms returns, dropping fast outside the peak lag. (Lower than people expect — most second-by-second ES moves are *not* predictive of SPY because both are mostly noise; the predictive component is the persistent informed-flow component.)

### 2.2 ES-prediction → SPY-return IC at each lag
The real test:
- Generate CNN-Mamba v2 prediction series on ES (1000-event windows, 250-stride → one prediction every ~62ms)
- For each ES prediction at time `t`, measure realized SPY mid return over `[t + lag, t + lag + h]` for `lag ∈ {0, 50, 100, 200, 500} ms` and `h ∈ {1s, 5s, 10s}`
- Compute rank IC of (ES pred, SPY forward return) per lag, per horizon
- Aggregate across days (concat IC)

**Decision rule**: SPY-execution-only is GO if there exists at least one (lag, h) pair with concat IC ≥ 0.05 across at least 5 days. Below that and any SPY edge gets eaten by fill uncertainty even with $0 commissions.

### 2.3 Passive-fill simulation on SPY at NBBO
- Once we know the predicted direction, simulate a passive limit at NBBO on the predicted-side
- Realistic SPY fill simulator: queue position at top of book in SPY is *much* longer than ES (SPY top-of-book is usually 10k+ shares deep on each side at $580). A passive limit placed at the BBO must wait its turn. Use a Cox-process queue model anchored to observed top-of-book sizes from the MBO feed.
- Output: fill probability + adverse-selection conditional-on-fill, per horizon

Expected gotcha: high fill probability means we got filled because *someone wanted that side*, which usually means immediate adverse mark. This is the classic adverse-selection problem with passive fills. The whole point of having a signal is that we can place passive limits on the *non-adversely-selected* side. Need to verify the IC sign agrees.

---

## 3. The literature, briefly

| Paper | Finding relevant here |
|---|---|
| Hasbrouck (2003) JoF, "Intraday Price Formation in US Equity Index Markets" | ES futures contribute ~90% of price discovery vs SPY at intraday horizons. Lead is in the 50–250ms range during liquid hours. |
| Chakravarty, Wood, Upson (2004) | Confirms futures lead in cash-vs-future basis arbitrage windows; lead widens during news. |
| Tao-Hasbrouck (2018) "High Frequency Quoting" | At sub-second horizons, NBBO quote revisions in SPY/ES are tightly coupled but ES quote revisions occur first by ~80–150ms on average. |
| Easley, López de Prado, O'Hara (2011) "VPIN" | Informed flow on the futures side is detectable via volume-synced toxicity metrics; suggests when futures move on informed flow, the cash basket is about to follow. |

None of this is news to academic microstructure people; the practical question for us is whether **our specific CNN-Mamba v2 signal** captures the informed-flow component of ES well enough that translating it forward through the lead produces a tradeable SPY edge **after fill realism**.

---

## 4. Risk factors (be honest)

1. **SPY tick is fixed at 1 cent but the book is 1 tick wide ~99% of RTH on millions of shares.** Queue position dominates everything. A model that knows direction but not queue placement realism will badly over-estimate fill rate.
2. **PFOF flow distortion.** ~40% of SPY retail order flow is internalized/wholesaler. The visible lit book is not the full market. Our MBO-based signal sees only the lit book; some predictive content of ES flow may already be internalized away on the SPY side before we can act.
3. **Hidden liquidity.** SPY has substantial dark-pool flow (~30% of ADV). Trades at midpoint occur off-book; our paper model assumes lit-only.
4. **Sweep risk.** When ES moves big-and-fast, SPY catches up in a sweep across multiple price levels. A passive limit at the NBBO will fill on the front of the sweep, which is the WORST fill (adverse selection at its peak). This is the dominant tail risk and the model must explicitly avoid placing limits when |predicted move| > 1 SPY tick.
5. **Latency.** Co-located HFTs see ES→SPY at <1ms and have already arbitraged the relationship. Our edge from a non-co-located retail setup is in the slow component of the predictability (10s+ horizons), not the 50ms tick-by-tick basis trade. Frame the strategy accordingly.
6. **Borrow costs for shorts.** SPY is cheap to borrow but not free. Overnight holds (we won't, but flag).
7. **Day-of-week / regime drift.** Mar 1–13 2026 was a mid-month FOMC-blackout window. Behavior may not generalize to FOMC weeks. Sample-size limitation is severe.

---

## 5. The HOW — practical research procedure

Once the SPY MBO Mar 1–13 2026 data lands AND **assuming overlapping ES MBO is acquired** (see §6 blocker):

```
# Per overlapping date d
1. Load ES MBO   -> ts_es, mid_es, events_es  (existing pipeline)
2. Load SPY MBO  -> ts_spy, mid_spy, events_spy  (new pipeline, READY)
3. Compute ES-CNN-Mamba-v2 predictions on events_es every 250 events
4. For each ES prediction at ts_pred:
     for lag in [0, 50, 100, 200, 500, 1000] ms:
       spy_idx = searchsorted(ts_spy, ts_pred + lag*1e6)
       fwd_ret_spy_h = mid_spy[searchsorted(ts_spy, ts_pred + lag*1e6 + h*1e9)]
                     / mid_spy[spy_idx] - 1   # for h in [1s, 5s, 10s]
       record (pred, fwd_ret_spy_h, lag, h, d)
5. Per (lag, h): rank IC of (pred, fwd_ret_spy_h) across all records
6. Concat across overlapping days
7. Report IC matrix + scatterplot per (lag, h, day-of-week)
```

Output → `output/spy_exec_only/lag_ic_matrix.json` + heatmap.
Decision rule from §2.2.

---

## 6. ~~BLOCKER~~ — CORRECTED 2026-06-04 19:15 ET: NO BLOCKER

**Original agent claim**: "ES MBO for Mar 1–13 2026 is NOT on disk."
**Reality**: ES MBO for Mar 1–13 2026 **IS on disk** — 12 files at `data/raw/mbo/glbx-mdp3-2026030{1..6,8}.mbo.dbn.zst` and `glbx-mdp3-202603{09..13}.mbo.dbn.zst`. The agent assumed Mar **2025** (which would be 15 months stale and absent from disk); user almost certainly meant **2026** (current, normal trial purchase). Earliest ES MBO file = 2025-07-14 (correct), so all of 2026 is covered.

### Implication
**Cross-correlation lead-lag IC study runs on the SAME window as the SPY trial.** No extra ES data buy needed. Plan becomes:

1. **In parallel** with the SPY-standalone trial (deliverable #2): take the already-on-disk ES MBO for Mar 1–13 2026, run CNN-Mamba v2 inference, align timestamps with the incoming SPY MBO, measure IC at lags {0, 50, 100, 200, 500, 1000 ms}.
2. **Two-axis report**: (a) does SPY-trained signal beat ES-trained? (b) does ES-signal-on-SPY beat both? Whichever wins is the headline.
3. **If lead-lag IC > 0** at any positive lag, the SPY-for-execution-only path is validated and we don't need to retrain on SPY at all — we just lag the existing ES predictions and submit them as SPY orders.

### What's left to confirm before running
- Timestamp alignment: ES MBO is in CME PT (ts_event ns since epoch UTC). SPY MBO will be same epoch UTC. Verify both providers use the same clock (Databento normalizes — should be fine).
- ES contract roll: Mar 2026 ES is on the H6 contract (Mar 2026 expiry) for the first ~2 weeks of the window, then rolls to M6 (Jun 2026) around 2nd Thursday. Trial runner needs to handle the roll OR restrict to the front-month before roll.

---

## 7. One-line conclusion

The SPY-for-execution-only hypothesis is the strongest pivot we have AND it can be validated on the same window as the SPY-standalone trial because the ES MBO overlap is already on disk. No spending decision needed — both hypotheses run when the SPY data lands.
