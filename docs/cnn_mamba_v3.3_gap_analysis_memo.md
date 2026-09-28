# CNN-Mamba v3.3 Gap Analysis Memo

**Author:** Head of Quant (Claude / autonomous)
**Date:** 2026-05-12
**Status:** PRE-LAUNCH SCOPING — non-blocking on v3.2 verdict (ETA tonight ~21:54 ET) + v3.2.1 launch (gated)
**Trigger:** HC #303 + HC #304 (user explicit deep-research request, in-channel 19:50 ET + 20:35 ET)

---

## 0. Status & Framing

- **Currently training**: v3.2 fold 0, PID 4940 on Neptune, batch 25,700/33,206 (~77.4% Ep 1), loss 114.3 oscillating, RAM 8G/22G free. **NOT v3.2.1** — user expressed slight confusion in 19:50 ET msg; clarified.
- **v3.2.1**: code built + smoke-tested 18:10 ET, ON DISK NOT RUNNING. Launches IF v3.2 verdict tonight fails the falsification gate. v3.2.1 = INPUT FIXES ONLY (19 norm fixes, 5 missing T2 feats, LGBM vol preds, event-type embedding, session-phase one-hot). Same architecture, same head set.
- **v3.3** = the version after v3.2.1, scoped HERE. Major-step iteration: new INPUTS (cross-asset, IV, iceberg, queue, Hawkes, vol-regime-class, term structure, DOM/countdown), new ARCHITECTURE (cross-tier attention, hierarchical fusion, vol-gated MoE, EMA self-distillation, masked-MBO aux task, Hawkes-prior init), new OUTPUTS (copula head, TP/SL/size chooser, time-stop, cancel-replace, counterfactual entry-timing, adversarial-selection meta head, risk heads).
- **WAIT FOR CURRENT FOLD** per user explicit. This memo is design + side-research scoping. No launches.

---

## 1. INPUT GAPS — Realistic Data Sourcing + Implementation Difficulty

| # | Gap | Data Source | Acquisition | LOC | Est. IC Lift | Difficulty | Priority for v3.3 |
|---|-----|-------------|-------------|-----|--------------|------------|-------------------|
| 1 | **Cross-asset futures** (NQ, YM, RTY) | Databento (existing vendor, CME contracts) | Marginal subscription bump | ~150 | **+0.02 to +0.05** | EASY (same pipeline) | **TIER 1** — biggest blind spot |
| 2 | **Bond / curve** (ZN, ZF, ZB) | Databento (CME futures) | Same | ~50 | +0.01 to +0.02 | EASY | TIER 1 |
| 3 | **DXY (Dollar Index)** | Databento (ICE, symbol `DX`) | Same | ~30 | +0.005 to +0.015 | EASY | TIER 1 |
| 4 | **VIX 1m bars** (cash index) | Polygon.io ($29-99/mo, full hx) OR Yahoo Finance (1m, 30-day lookback only — too short) OR CBOE direct ($$) | New vendor decision | ~100 | **+0.01 to +0.03** | EASY-MEDIUM (new vendor) | **TIER 1** — option flow regime |
| 5 | **VIX term structure** (VX1/VX2, VIX9D) | Databento (CFE — Cboe Futures Exchange) | Marginal | ~80 | +0.01 to +0.02 | EASY | TIER 1 |
| 6 | **Dealer gamma / charm / vanna** | SpotGamma ($5-15k/yr) **OR** OPRA full chain via Databento + custom Greeks compute (~weeks of eng) | Licensing decision OR multi-week project | ~500 (Greeks calc) | **+0.03 to +0.08** (theoretically biggest) | HARD (blocked on licensing or eng) | TIER 3 (defer to v3.4 unless SpotGamma approved) |
| 7 | **Realized-vs-implied vol gap** | RV from existing T1 events + IV from VIX | Trivial once VIX in | ~30 | +0.005 to +0.015 | EASY (once #4 lands) | TIER 1 |
| 8 | **Calendar event proximity** (FOMC/CPI/NFP/Fed speakers/major earnings) | Free: Investing.com, ForexFactory, Trading Economics (scrape), FRED (API), Yahoo earnings | Public free scrape, store JSON | ~80 builder + ~30 feature | **+0.02 to +0.04** | TRIVIAL | **TIER 1** — vol regime shifts deterministically at these |
| 9 | **Iceberg / hidden-order detection** (T1) | **EXISTING MBO** (we have all events) | NONE — feature eng on our tape | ~200 (detection rules) | +0.01 to +0.03 | TRIVIAL-MEDIUM | **TIER 1** — should've been in v3.2.1 |
| 10 | **Queue rank / time-in-queue** (T1) | **EXISTING MBO** (order_ids in every event) | NONE — state-tracking | ~300 (FIFO queue reconstruction) | +0.005 to +0.02 (signal) | MEDIUM | **TIER 1** — critical for paper trader fill probs |
| 11 | **QTR — quote-to-trade ratio** (T2) | **EXISTING T1** events | NONE | ~20 | +0.005 to +0.015 | TRIVIAL | TIER 1 |
| 12 | **Hawkes intensity** (T2) | **EXISTING T1** event arrivals | NONE — `tick` Python lib | ~150 | +0.01 to +0.02 | MEDIUM (kernel fitting) | TIER 1 |
| 13 | **Vol regime CLASS** (3-way: low/mid/high) (T3) | **EXISTING LGBM vol preds** | Bin on rolling quantiles | ~30 | +0.005 to +0.015 (via downstream conditioning) | TRIVIAL | TIER 1 |
| 14 | **Term structure** (ES front-back spread) | Databento (back-month contracts) | Marginal | ~80 | +0.001 to +0.005 | EASY but limited info value most of year | TIER 2 |
| 15 | **DOM sin/cos** (day-of-month cyclical) | Datetime math | NONE | ~10 | +0.001 to +0.005 | TRIVIAL | TIER 1 (oversight) |
| 16 | **Event countdown features** (sec_to_FOMC, sec_to_close, sec_to_lunch_lull) | Datetime + #8 calendar | Depends on #8 | ~40 | +0.005 to +0.015 | TRIVIAL | TIER 1 |
| 17 | **Quad-witching / settlement Tuesday / monthly-OPEX flags** | Datetime math | NONE | ~20 | +0.001 to +0.005 | TRIVIAL | TIER 1 |
| 18 | **Day-of-month / week-of-quarter** | Datetime math | NONE | ~15 | +0.001 to +0.003 | TRIVIAL | TIER 2 |

**SUMMARY TIER 1 (v3.3 cut)**: 13 items, ~1300 LOC, expected aggregate IC lift +0.08 to +0.20 (with diminishing-returns overlap), assumes Polygon.io VIX subscription approved ($29-99/mo) and Databento subscription extension to NQ/YM/RTY/ZN/DX (marginal).

---

## 2. WHY THESE WEREN'T IMPLEMENTED ALREADY — Honest Answer

The user asked directly: "why didn't you implement [these] then?" Per HC #304(C):

1. **v3.2 is the FIRST RUN with all 3 tiers actually populated.** Pre HC #295 (2026-05-11 19:45 ET), Tier 3 was zero-placeholder + Tier 2 was bucket-mean-of-Tier-1 (lazy proxy). We literally haven't seen if the **baseline** 3-tier architecture works yet. Adding cross-asset + IV + iceberg + Hawkes + ... simultaneously would make per-feature attribution impossible — we'd have no idea which add caused which delta in IC.

2. **Cross-asset + IV introduce NEW data pipelines.** Each new vendor / symbol / format = engineering tax (parser, aligner, normalization, storage). Reasonable engineering hygiene to defer until the baseline confirms architecture viability.

3. **Iceberg / queue-rank / Hawkes / QTR / vol-regime-class / DOM are feature-engineering on data we ALREADY HAVE.** These ARE the lowest-hanging-fruit adds. They SHOULD have been in v3.2.1. Why they weren't: HC #298/#299 narrowly scoped v3.2.1 to "fix what's wrong" (normalization fixes, missing T2 feats, event-type embedding) rather than "add what's missing". This is a **mea culpa** — the scoping was too conservative. They are the first additions for v3.3.

4. **Dealer gamma is genuinely blocked.** Either spend $5-15k/yr on SpotGamma (licensing decision needed from user) or undertake a multi-week data-engineering project to compute Greeks from OPRA. Neither happens in a "side research on Jupiter" timeframe. Reasonable v3.4+ work.

5. **Cross-tier attention** was always "next architecture iteration" but I picked input-fixes first because of user's HC #298/#299 explicit ranking ("data presentation is #1 leverage point > model choice > execution"). v3.2 + v3.2.1 are input-quality milestones; v3.3 architecture upgrade comes once inputs are clean.

---

## 3. ARCHITECTURE GAPS

| # | Architecture Idea | What It Does | Param Cost | LOC | Risk | Priority |
|---|-------------------|--------------|------------|-----|------|----------|
| A1 | **Cross-tier attention** | T1 queries attend to T2+T3 keys/values. Lets microstructure branch USE context. | ~50k | ~200 | LOW (well-understood) | **TIER 1** (user-named "most obvious next step") |
| A2 | **Hierarchical fusion** | T3 (slow) → conditions T2 layer-norm → conditions T1 layer-norm. Ladder of context. | ~30k | ~150 | MEDIUM (training stability) | TIER 2 |
| A3 | **Vol-gated MoE heads** | K=3 expert head banks {low/mid/high vol}, vol-regime classifier gates which fires. | ~3× heads (~1M extra) | ~200 | MEDIUM (more params, risk overfitting) | TIER 1 |
| A4 | **Output-head conditioning on vol** | Heads receive predicted vol as input → naturally widens quantiles in high vol. | ~5k | ~50 | LOW | TIER 1 |
| A5 | **EMA-teacher self-distillation** | Student matches EMA-teacher predictions. Well-validated regularizer for noisy financial data. | 0 extra (teacher is EMA of student) | ~100 | LOW (well-validated) | TIER 1 |
| A6 | **Masked-MBO self-supervised aux task** | Mask 10% of T2 buckets, predict from T1+T3 context. Forces predictive representations. | ~50k aux head | ~150 | MEDIUM | TIER 2 |
| A7 | **Hawkes-prior Mamba init** | Initialize SSM state-transition matrices to encode self-exciting event structure. Domain prior > random init. | 0 | ~50 | MEDIUM (theory-heavy) | TIER 2 |
| A8 | **Causal-mask audit** | Verify zero future leakage at every tier boundary. Not a feature — a correctness check. | 0 | 0 (test code) | LOW | TIER 1 (mandatory) |
| A9 | **Per-tier dropout rate sweep** | Tiers have different SNR; tier-specific dropout. | 0 | ~30 + sweep | LOW | TIER 2 |
| A10 | **Larger CNN-Mamba on T1** (deeper backbone) | If T1 branch is gradient-norm-rich, scale capacity. Gated on v3.2 attribution. | +500k | ~50 | MEDIUM | TIER 2 (decide post-v3.2 attribution) |

**TIER 1 ARCHITECTURE FOR v3.3**: Cross-tier attention (A1) + Vol-gated MoE heads (A3) + Output-head conditioning (A4) + EMA-teacher distillation (A5) + Causal-mask audit (A8). Total: ~600 LOC, ~1.05M extra params (~70% increase over v3.2's 756k), substantial representational gain.

---

## 4. OUTPUT GAPS — Decision-Relevant Heads (alpha-first)

Per HC #294F's alpha-first architecture vision: the model should DECIDE, not just predict. Currently 32 heads are all PREDICTION (log_ret, p_up, quantiles, MFE/MAE, time-to-peak, reversal, vol). Downstream rules + FIFO bracket-fit do the actual decisions. Goal for v3.3: shift toward **the model picks bracket, sizing, hold time per trade**.

| # | Output Head | What It Predicts | Training Target | LOC | Priority |
|---|-------------|------------------|-----------------|-----|----------|
| O1 | **Copula head (Gaussian copula)** | Joint distribution of returns at 5s/10s/30s/60s/5min (not independent marginals) | Cholesky factor of covariance, trained on observed joint returns | ~150 | TIER 1 |
| O2 | **Optimal TP head** (`pred_optimal_tp_ticks`) | Best TP-target per trade | Argmax over simulated FIFO replay outcomes (range 2-12 ticks) | ~100 | **TIER 1** |
| O3 | **Optimal SL head** (`pred_optimal_sl_ticks`) | Best SL-target per trade | Same FIFO replay sim | ~100 | **TIER 1** |
| O4 | **Optimal size head** (`pred_optimal_size_fraction` ∈ [0,1]) | Position size as fraction of vol-adjusted budget | Trained on Kelly-fraction of simulated PnL distribution | ~80 | TIER 1 |
| O5 | **Time-stop head** (`pred_optimal_hold_secs`) | When to exit if neither TP nor SL hit | Trained on time-of-max-MFE distribution | ~60 | TIER 1 |
| O6 | **Cancel-replace head** (`pred_replace_now_prob`) | Should a resting order be re-quoted? | Trained on whether re-quoting would have improved fill | ~80 | TIER 2 (live-only) |
| O7 | **Counterfactual entry-timing head** | Expected return if trade NOW vs WAIT 1s/5s | Synthetic targets from observed forward returns at shifted T0 | ~100 | TIER 2 |
| O8 | **Adversarial-selection meta head** | P(this signal is a winner) — end-to-end version of downstream meta-LGBM gate | Trained on realized P&L outcomes | ~80 | **TIER 1** |
| O9 | **VaR_95 / Expected Shortfall heads** | Tail risk for position sizing | Trained on observed P&L tail quantiles | ~60 | TIER 2 |
| O10 | **Max drawdown over hold head** | Forward path drawdown for risk-aware sizing | Trained on observed MAE-over-hold | ~50 | TIER 2 |

**TIER 1 OUTPUT FOR v3.3**: Copula (O1) + Optimal TP/SL/size/time-stop (O2-O5) + Adversarial-selection meta (O8). 6 new head families, ~570 LOC. Critical dependency: **FIFO replay simulator must be production-grade** to generate training targets for O2/O3/O4/O5 (per HC #74 — already in roadmap).

---

## 5. Methodology Gaps

| # | Item | Why It Matters | Priority |
|---|------|----------------|----------|
| M1 | **Multi-task loss balancing** (PCGrad / GradNorm / uncertainty-weighted) | 32 heads with different loss scales — some heads dominate gradient, others starve. Currently only λ=0.1 on legacy FIFO. | TIER 1 |
| M2 | **Curriculum learning** (start high-vol regime, expand to chop) | Speeds convergence — high-vol days have clearer signal | TIER 2 |
| M3 | **10-fold sliding stability** | v3.2 is 1-fold for speed; need 10 folds for IC variance bands + statistical significance | TIER 1 (post-v3.3 baseline lands) |
| M4 | **Online fine-tune post-deployment** | Combat decay; we have evidence of weekly drift (per HC #290 decay-window analysis) | TIER 1 (deployment-gated) |
| M5 | **Per-tier attribution analysis required for every run** | Per HC #297C — gradient-norm, weight-norm, activation-magnitude per tier branch | TIER 1 (mandatory) |
| M6 | **Falsification gates re-spec for v3.3** | v3.3 must beat v3.2 (which itself must beat v3) — gate per HC #295H needs lift | TIER 1 |

---

## 6. v3.3 GO/NO-GO DECISION FRAMEWORK (HC #304(F))

The v3.3 launch decision is downstream of v3.2 verdict + v3.2.1 result:

```
                          ┌──────────────────────────────┐
                          │  v3.2 verdict tonight ~21:54 │
                          └──────────────┬───────────────┘
                                         │
                  ┌──────────────────────┼──────────────────────┐
                  │                      │                      │
              PASS clean               PASS barely             FAIL
              (≥+0.02 on any        (~+0.005 on one)     (no improvement)
              of 5/10/30s)
                  │                      │                      │
                  ▼                      ▼                      ▼
        ┌────────────────┐      ┌────────────────┐    ┌────────────────┐
        │ v3.3 = HIGH-   │      │ v3.2.1 first   │    │ v3.2.1 first   │
        │ VALUE next     │      │ → validate     │    │ → MUST restore │
        │ iteration      │      │ input fixes    │    │ baseline first │
        │                │      │ → v3.3 on top  │    │ → v3.3 deferred│
        │ RECOMMEND GO   │      │                │    │                │
        └────────────────┘      └────────────────┘    └────────────────┘
                                                              │
                                                              ▼
                                                  ┌────────────────────┐
                                                  │ If v3.2.1 also fails│
                                                  │ → fundamental rethink│
                                                  │ → v3.3 as scoped is  │
                                                  │   the WRONG move    │
                                                  └────────────────────┘
```

**Verdict tonight + v3.2.1 result this week → v3.3 launch decision by end-of-week.**

---

## 7. v3.3 CUT — TIER 1 SCOPING (Recommendation if GO)

If v3.2 passes gate cleanly:

### Inputs added (12 items, ~1100 LOC, est. +0.08 to +0.18 IC):
1. Cross-asset futures (NQ/YM/RTY/ZN/DX) — rank-normed in T3
2. VIX 1m bars + term structure (VX1/VX2/VIX9D) + RV/IV gap — T3
3. Calendar event proximity (FOMC/CPI/NFP/Fed/earnings countdowns) — T3
4. Iceberg / hidden-order detection — T1
5. Queue rank / time-in-queue — T1
6. QTR + Hawkes intensity — T2
7. Vol regime CLASS (3-way softmax bins) — T3
8. DOM sin/cos + quad-witching/OPEX flags + countdown features — T3

### Architecture changes (5 items, ~600 LOC, +~1M params):
- A1: Cross-tier attention (T1 ↔ T2 ↔ T3)
- A3: Vol-gated MoE heads (K=3 experts)
- A4: Output-head conditioning on predicted vol
- A5: EMA-teacher self-distillation
- A8: Causal-mask audit (mandatory)

### Outputs added (6 new head families, ~570 LOC):
- O1: Copula head (joint return distribution)
- O2-O5: TP/SL/size/time-stop chooser heads (alpha-first)
- O8: Adversarial-selection meta head

### Methodology:
- M1: Multi-task loss balancing (GradNorm)
- M5: Per-tier attribution mandatory in verdict reports
- M6: Falsification gate updated: v3.3 must beat v3.2 by ≥+0.02 IC on any of 5/10/30s

### Deferred to v3.4+ (TIER 2/3):
- Dealer gamma / charm / vanna (licensing or multi-week eng)
- Term structure (ES front-back spread) — limited info value
- A2 Hierarchical fusion, A6 masked-MBO aux task, A7 Hawkes-prior init
- O6 Cancel-replace, O7 Counterfactual entry-timing, O9/O10 risk heads

---

## 8. JUPITER SIDE-RESEARCH DISPATCH (HC #304(D))

Per HC #300C (Jupiter never idle) + user explicit ("Maybe any of this can be researched on the side on Jupiter to determine viability"):

| # | Side-Research Task | Output | Status | ETA |
|---|--------------------|--------|--------|-----|
| 1 | **Iceberg/hidden-order detection prototype** (existing MBO, read-only) | `docs/iceberg_feasibility_20260512.md` + iceberg counts/day/size dist | QUEUED | Tonight |
| 2 | **Queue-rank / time-in-queue prototype** (existing MBO order_ids) | `docs/queue_rank_prototype_20260513.md` | QUEUED | Tomorrow |
| 3 | **Hawkes intensity computation prototype** (existing T1 events) | `docs/hawkes_prototype_20260513.md` + IC vs CNN-Mamba targets | QUEUED | Tomorrow |
| 4 | **Calendar event JSON build** (FRED, ForexFactory, Yahoo earnings) | `data/external/economic_calendar_2023_2026.json` | QUEUED | Tonight |
| 5 | **VIX 1m bar feasibility report** (Polygon/Yahoo/Databento probe) | `docs/vix_data_sourcing_20260512.md` | QUEUED | Tonight |
| 6 | **Cross-asset Databento catalog query** (NQ/YM/RTY/ZN/DX/CL/GC) | `docs/cross_asset_databento_20260513.md` | QUEUED | Tomorrow |
| 7 | **Vol regime CLASS labeling job** (3-way bins on LGBM preds) | `data/processed/vol_regime_class_v1.parquet` + transition matrix | QUEUED | Tonight |
| 8 | **DOM sin/cos + countdown feature builder** (pure datetime) | New columns in T3 feature builder | QUEUED (depends on #4 for countdowns) | Tonight |

Side-research is all read-only / new-file-write. No trainer code modifications (malware-guard reminders active this session per the harness).

---

## 9. Open Questions for User (Non-Blocking)

These don't block v3.3 design but need user input before launch:

1. **Polygon.io subscription** ($29-99/mo) — approve for VIX 1m bars? Or Yahoo (1m only ~30 days = too short, won't cover training window)?
2. **SpotGamma** ($5-15k/yr) — approve for dealer gamma/charm/vanna? Or defer to v3.4+ (build from OPRA, ~weeks of eng)?
3. **Databento subscription extension** — exact cost for NQ/YM/RTY/ZN/DX/CL/GC. Side-research task #6 will quote it.
4. **v3.3 launch trigger** — auto-launch on v3.2 PASS verdict? Or wait for explicit greenlight? (Recommend: explicit greenlight; v3.3 is a big jump, ~2300 LOC + 1M params, deserves user-in-the-loop.)

---

## 10. References

- HC #303 (v3.3 gap analysis seed)
- HC #304 (this memo's binding directive — data-access + feasibility + Jupiter dispatch)
- HC #299 (v3.2.1 design memo — input fixes only)
- HC #298 (per-feature normalization audit)
- HC #295 (3-tier proper engineering)
- HC #294F (alpha-first head set including 1s log_ret)
- HC #297C (falsification gate + per-tier attribution)
- HC #290 (decay-window analysis — signal edge ~30s)
- HC #74 (FIFO market replay mandatory for execution targets)
- HC #69 (risk-adjusted metrics primary, not raw P&L)

---

**END OF MEMO.** ~2300 LOC + ~1M extra params if all Tier 1 v3.3 items land. Expected aggregate IC lift +0.08 to +0.18 on top of whatever v3.2/v3.2.1 deliver. **Worth it if v3.2 passes gate cleanly tonight.**
