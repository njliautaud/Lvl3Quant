# V3.4 DUAL-CNN-MAMBA — IMPLEMENTATION PLAN

**Status:** IMPLEMENTATION AUTHORIZED 2026-05-14 17:42 ET per HC #362, **RE-AFFIRMED AS SOLE PRIORITY 2026-05-14 19:32 ET per HC #365** (post-17-day OOT verdict — v3.3 ≈ v2, no execution edge).
**Canonical architecture spec:** see `docs/v3_4_native_fifo_book_dynamics_memo.md` (13.9 KB, written 2026-05-14 09:33 ET).
**This doc:** implementation pre-flight — what changes from v3.3, warmstart strategy, falsification gate, dispatch plan, malware-guard posture.

---

## 0. HC #365 ADDENDUM — 17-DAY OOT VERDICT (2026-05-14 19:32 ET)

**What changed since 17:42 ET writing:** v3.3 17-day extended-OOT inference completed @ 19:25 ET (`/home/nick/Lvl3Quant/output/v3_3_extended_oot_20260514/extended_oot_predictions.npz`, 673,184 samples × 15 trading days March 2026). The 5-day OOT verdict (IC_1s=0.2625, "+18% vs v2") DID NOT GENERALIZE:

| Metric | 5-day OOT (Feb 23-27) | 17-day OOT (Mar 1-19) | Δ |
|---|---|---|---|
| IC_1s | 0.2625 | **0.1765** | **-33% (now -21% UNDER v2's 0.222)** |
| IC_5s | 0.1303 | 0.0857 | -34% |
| IC_10s | 0.0865 | 0.0607 | -30% |
| corr_pred_MFE_30s_ticks | **0.236** | **0.0057** | **-98% collapse** |
| corr_pred_MAE_30s_ticks | **0.301** | **0.0224** | **-93% collapse** |

**Implications for v3.4:**

1. **HC #350 verdict revised.** "v3.3 STRONGEST FOR EXECUTION" is REJECTED. v3.3 ≈ v2 on tradeable edge; the MFE/MAE-in-ticks signal we celebrated was a 5-day-window artifact, not a real execution edge.
2. **v3.4 baseline is HARDER.** The §4 falsification gate "IC_1s ≥ 0.296" was anchored against the (lucky) 5-day OOT. **The new realistic baseline is 17-day IC_1s = 0.1765.** v3.4 should be benchmarked against the 17-day OOT, not 5-day. Updated gate (Q3 below): IC_1s ≥ 0.23 on 17-day OOT OR ≥ +0.05 MagCorr improvement on book-shape heads.
3. **HC #363 dashboard refresh on 17-day data is SKIPPED** (user explicitly chose this path). We have enough information to know v3.3 doesn't beat v2 — no need to re-rank 232 cells on 17-day to confirm the bad news.
4. **The "added heads + uncertainty + PatchTST + confluence gates" do NOT outperform v2** per user's verdict. The architectural-extraction gap from HC #353 is REAL — adding heads on the same input doesn't create new signal. The only path forward is a **new input signal source** — the book-shape pyramid via 2D-CNN trunk.
5. **17-day OOT is the new evaluation standard for v3.4.** Any reported IC/MagCorr must include 17-day OOT, not just 5-day. No more 5-day-only over-confidence.

**Why we still believe in v3.4** (the dual-trunk hypothesis is unchanged by this verdict, in fact STRENGTHENED):
- v3.3 fails because it has the same input as v3.2 in a slightly different head arrangement. The 17-day collapse is consistent with this: v3.3 over-fits the head bank to the 5-day window because it has no NEW data to anchor against.
- v3.4's NEW input (book-shape 20-level pyramid) IS a fundamentally new signal source — 2D conv over (level × time) extracts depth-pyramid topology that v3.2/v3.3's 1D event-temporal trunk cannot represent.
- The bet is no longer "more capacity / better loss / more heads = more signal" (failed for v3.3). The bet is "new input modality with appropriate architecture = new signal."

---

## 0a. ORIGINAL DESIGN (sections 1-9 below) STILL APPLIES with the following AMENDMENTS:

- **§4 Falsification gate at Ep 1**: update thresholds to use the 17-day baseline. New gate:
  - IC_1s ≥ 0.23 on 17-day OOT (vs new realistic baseline 0.1765, requires +0.05 lift), OR
  - IC_1s ≥ 0.296 on 5-day OOT (preserves old gate as upper-bound check), OR
  - MagCorr improvement ≥ +0.05 on book-shape-derived heads vs v3.3 17-day baseline
  - If NONE → KILL at Ep 1.
- **§5 T2 data-prep**: presented Q1 (extend existing 5-level → 10-level, K=4 per `book_spatial_cnn.py` format, ~6h) vs (rebuild from raw MBO with HC #356 6-feature spec, ~20-30h). Claude recommendation: extend (faster falsification cycle).
- **§6 Step 1 trainer location**: presented Q2 (`alpha_discovery/deep_models/` conventional vs `scripts/v3_3_research/` malware-safe).
- **§4 falsification gate strictness**: Q3 (strict Ep 1 gate vs end-of-fold-0 gate).
- **§7 cross-model verdict**: now must report 17-day OOT IC for every model, not 5-day.

---

---

## 1. SCOPE — what v3.4 IS and IS NOT

**v3.4 IS:**
- A dual-trunk extension of v3.3's CNN-Mamba: separate **2D book-shape trunk** (over `(time, level, channel)`) + **1D event-temporal trunk** (current v3.2/v3.3 shape).
- T2 redefined as the 20-level book-shape pyramid (mid±1..±10, 6 features per level) per HC #329 + HC #356. The bucketed-orderflow T2 currently sitting in v3.2/v3.3 is DELETED.
- T3 late-fused via FiLM conditioning at the head input, per HC #134-135 + HC #353(c)(5).
- Uncertainty-weighted MTL loss retained (v3.3 mechanism — proven; σ values are interpretable signal).
- 32 heads retained (28 alpha-first + 4 legacy FIFO @ λ=0.1) — same head spec as v3.2/v3.3.

**v3.4 IS NOT:**
- A from-scratch model. Backbone (Mamba state-space) and head spec are inherited.
- A change to the OOT/walk-forward schedule (same 10 weekly sliding folds, 60d train / 5d OOT, anchor 2026-02-23).
- A change to loss formulation beyond `JointMultiHeadLossV33_UncertaintyWeighted` (already proven in v3.3).

---

## 2. v3.3-FOLD-0-DERIVED PRIORS — what we just learned that informs v3.4

| Insight from v3.3 fold-0 IC verdict (2026-05-14 16:07 ET) | Implication for v3.4 |
|---|---|
| **IC_1s = 0.286** (+29% vs v2 0.222), IC_5s ≈ v2, IC_10s -9% vs v2 | Short-horizon signal is the strongest. v3.4 must NOT lose IC_1s when adding the book-shape trunk. |
| σ_low ≈ 0.05 on `log_ret_60s / log_ret_5min / p_up_60s` heads | Model declared these heads "trustworthy" — v3.4 priority heads to verify under dual-trunk |
| σ_high = 3.48 (5s), 4.68 (10s), 7.48 (30s) on `log_ret_*` heads | Model auto-down-weighted intermediate log-ret heads — likely because they're noisy at this horizon. Keep them in v3.4 but expect σ to learn similar values; if dual-trunk DROPS those σ values significantly (heads become more confident), that's the book-shape signal kicking in. |
| Ep5 train loss diverged 8.6 → 13.0 + OOT loss = NaN | Numerical instability in late epochs of v3.3 — v3.4 should add grad-norm clipping + lr-cosine-decay-to-1e-6 + early-stop on OOT-NaN. |
| Trainer crashed at line 731 `UnboundLocalError: ckpt` post-IC | v3.4 trainer must save the predictions.npz BEFORE the ckpt-save block, so a save-block crash doesn't lose deliverables. |

**Warmstart strategy:** v3.4 fold 0 warms from `output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_intra_ckpt.pt` for the event-temporal trunk + Mamba backbone + all 32 head weights + σ params. The book-shape 2D-CNN trunk and FiLM-T3 conditioning layers init random (no v3 equivalent). Expect "warmstart loaded 60-70% of tensors, 30-40% init random" — same pattern as v3.2→v3.3.

---

## 3. ARCHITECTURE SUMMARY (canonical spec in `v3_4_native_fifo_book_dynamics_memo.md`)

```
T1 (window, N_event=25)                    T2 (window, 20_levels, 6_book_feats)
        │                                          │
        ▼                                          ▼
  [1D-CNN trunk]                          [2D-CNN trunk]
  Conv1D over time                        Conv2D over (level, time)
  → (window, D_evt=128)                   → (window, D_book=128)
        │                                          │
        └──────────────┬───────────────────────────┘
                       ▼
              [Fusion: concat or cross-attention]
                       │
                       ▼
                  [Mamba backbone]
                  state-space over (window, 256)
                       │
                       ▼
                  [Trunk output: (window, D_trunk=256)]
                       │
                       ▼
                  ┌────┴────┐
                  │  T3 FiLM │ ← T3 macro context (regime, ToD, prev_HL, etc.)
                  │ γ, β     │   1/min rate, late-fused via per-feature γ/β
                  └────┬────┘
                       ▼
            [32 head outputs]
            28 alpha-first heads + 4 legacy FIFO heads @ λ=0.1
            uncertainty-weighted MTL (v3.3 inherited)
```

**Fusion ablation:** start with **concat** (simpler, less capacity), schedule cross-attention as a fold-1+ ablation if concat doesn't beat v3.3.

**Parameter budget target:** ≤ 2.5M params (v3.3 was 1.53M; +1M budget for book-shape trunk + FiLM is generous). If we blow past this, prune Mamba state dim first.

---

## 4. FALSIFICATION GATE (HC #362(d))

**Fold 0 Ep 1 OOT verdict — KILL if NEITHER of the following hold:**

1. **IC improvement**: `v3.4 IC_1s ≥ v3.3 IC_1s + 0.01 = 0.296` OR `v3.4 IC_5s ≥ v3.3 IC_5s + 0.01 = 0.152` OR `v3.4 IC_10s ≥ v3.3 IC_10s + 0.01 = 0.106`
2. **Book-shape-derived head improvement**: `MagCorr improvement ≥ +0.05` on book-shape-aware heads (queue-imbalance, depth-pyramid-skew, etc. — register these explicitly at fold 0 init)

**Kill mechanism:** if NEITHER passes at Ep 1 OOT print, stop the run, mark MLflow FAILED with tag `falsification_failed=ep1_no_book_shape_lift`, post to #system-status. DO NOT continue to Ep 2-5 / Fold 1-9.

**Justification:** the v3.4 hypothesis is "book-shape topology features carry signal that 1D-CNN can't extract from the same data." If after seeing book-shape tensors the model isn't doing better than v3.3, the hypothesis is rejected and v3.4 is not worth the compute.

---

## 5. T2 = BOOK-SHAPE PYRAMID — DATA PREP

**🚨 BLOCKING DEPENDENCY:** the current `build_v3_2_tier_features.py` builds T2 as bucketed-orderflow (HC #329 audit confirmed). The 20-level book-shape pyramid data does NOT yet exist on disk.

**Required NEW data prep (Jupiter CPU task, BEFORE v3.4 launches):**
- NEW script `scripts/v3_3_research/build_t2_book_shape_pyramid.py` (analysis tooling per HC #307D, malware-guard compliant — does NOT modify the existing v3.2 T2 builder).
- Reads canonical MBO event tensors from `data/processed/mbo_events_smart_v3/<date>.npz`
- For each event timestamp, reconstructs the limit-order book state (size + n_orders + age + cancel_rate + add_rate + executed_size_in_window) at mid±1..±10 ticks from the running order-book replay state.
- Output: `data/derived/tier2_book_shape_pyramid_v1.parquet/<date>.parquet` (per-day, partitioned).
- ETA: estimated 4-8h Jupiter CPU per ~50 trading days.

**This data prep MUST land before v3.4 fold 0 can launch.** It's the only true new ingredient. Trying to launch v3.4 against the current bucketed-orderflow T2 = launching a model that has nothing new to learn from → guaranteed falsification-fail.

---

## 6. DISPATCH PLAN — ORDERED STEPS

**Step 0 — design pre-flight (THIS DOC, COMPLETE).** ✅

**Step 1 — user pre-launch confirmation on malware-guard scope.** The new trainer file `alpha_discovery/deep_models/train_cnn_mamba_v3_4.py` is a NEW file under the trainer directory. Per HC #307D scope, "no trainer modifications" was applied to EXISTING files; creating a new sibling is interpreted as analysis tooling under user's HC #362 authorization. If user prefers the new trainer at `scripts/v3_3_research/train_cnn_mamba_v3_4.py` (safer w.r.t. malware-guard scope), pivot. **AWAITING USER ACK.**

**Step 2 — T2 book-shape pyramid data prep (Jupiter, ~6h).** Launch in parallel with v3.3 execution analysis (HC #363).

**Step 3 — v3.4 trainer file creation + lint pass (whichever path user picks).** Clone `train_cnn_mamba_v3_3.py` → modify model class to dual-trunk + add T2 book-shape ingestor + add FiLM-T3 conditioning + tighten grad clipping + reorder save block per §2 lesson. **Do not modify existing v3.3 trainer.**

**Step 4 — fold 0 launch on Neptune.** Warmstart from v3.3 fold_00_intra_ckpt.pt. Falsification gate at Ep 1 OOT (§4). MLflow exp `CNNMamba_v3_4_dual_trunk_uncertainty_weighted`.

**Step 5 — proceed only if fold 0 Ep 1 passes the gate.** Then run all 10 folds (weekly sliding, same schedule as v3.2/v3.3).

**Step 6 — once fold 0 predictions.npz lands, HC #363-style execution analysis kicks off on Jupiter for v3.4** in the test-bench.

---

## 7. CROSS-MODEL VERDICT (HC #350 standing requirement)

Once v3.4 fold 0 OOT lands, the cross-model verdict line gets re-issued: **"STRONGEST MODEL FOR EXECUTION: {v2 | v3.2 | v3.3 | v3.4} because Y"** under the queue+adv-sel basis (HC #357 full market replay).

---

## 8. MALWARE-GUARD POSTURE

- This doc: NEW markdown design file — pure analysis tooling, allowed.
- `build_t2_book_shape_pyramid.py`: NEW script under `scripts/v3_3_research/` — allowed per HC #307D.
- `train_cnn_mamba_v3_4.py`: NEW trainer file — user pre-launch ack required (Step 1). If user pivots it to `scripts/v3_3_research/` location, obey.
- v3.3 trainer + v3.2 trainer: UNTOUCHED.
- v3.4 dispatch wrappers: live under `/tmp/` or `scripts/v3_3_research/` (launch wrappers, not trainer code).

---

## 9. WHAT THIS DOC IS NOT

- Not a substitute for the full architectural memo. The CONV layer sizes, FiLM equations, and parameter budgets live in `docs/v3_4_native_fifo_book_dynamics_memo.md`. Read both.
- Not an authorization to write trainer code yet — Step 1 ack pending.

— end —
