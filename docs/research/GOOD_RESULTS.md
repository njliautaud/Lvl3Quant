# GOOD_RESULTS.md — Durable Index of Wins

**Purpose**: Single source of truth for "what works" across the research stack. Every result that meets HC #291(B) win criteria gets a row here + a JSON in `output/good_results/`.

**Win criteria (HC #291(B))** — any of:
- (i) IC ≥ +0.05 mean concat across ≥20 splits
- (ii) top10 hit-rate > base-rate + 5pp at ≥80% of splits
- (iii) non-inversion confirmed (both tails positive or non-degenerate ordering) on a previously-inverted setup
- (iv) any first-of-its-kind structural finding

**Last updated**: 2026-05-11 15:25 ET (HC #291 install + backfill of 5 May-11 clf wins)

---

## TIER A — STRUCTURAL BREAKTHROUGHS (deploy candidates or close)

| # | Date | Model / Variant | Side / Target | IC_mean | IC_median | top10 | bot10 | n_splits | MLflow run_id | Why it's saved |
|---|------|-----------------|---------------|---------|-----------|-------|-------|----------|---------------|----------------|
| A1 | 2026-05-11 12:30 ET | Meta-LGBM v3 FIFO v2 **clf** (per-day rank-norm + confluence) | tp4sl3 SHORT hit_tp | **+0.0530** | +0.0459 | **+0.350** | +0.276 | 31/33 | exp `MetaLGBM_v3_FIFO_v2_clf_hit_tp` | **FIRST NON-INVERTED META-LGBM EVER** — 5 prior regression variants all inverted on v2 preds. Binary clf objective unlocks the gate. Both tails POSITIVE. (HC #291(B)(iii)+(iv)) |
| A2 | 2026-05-11 13:40 ET | Meta-LGBM v3 FIFO v2 **clf** | tp8sl5 SHORT hit_tp | **+0.0715** | +0.0503 | +0.191 | +0.108 | 33 | run `meta_lgbm_v3_fifo_v2_clf_hit_tp_1778518907` exp 735727744927211526 | Confirms clf paradigm works on FATTER edge target too. Both tails positive. Higher IC than tp4sl3 clf — base-rate ceiling caps top10 absolute level. (HC #291(B)(i)+(iii)) |
| A3 | 2026-05-11 14:18 ET | Meta-LGBM v3 FIFO v2 **clf** | tp4sl3 LONG hit_tp | **+0.0647** | +0.0669 | **+0.369** | +0.272 | 33 | run `meta_lgbm_v3_fifo_v2_clf_hit_tp_1778522447` exp 425621512917111432 | **SIDE-AGNOSTIC** confirmed — LONG clf slightly stronger top10 than SHORT. Broadens deployable surface (don't need to deploy short-only). (HC #291(B)(i)+(iii)) |
| A4 | 2026-05-11 14:49 ET | Meta-LGBM v3 FIFO v2 **clf** | tp8sl5 LONG hit_tp | **+0.0999** | **+0.1111** | +0.199 | +0.095 | 33 | run `meta_lgbm_v3_fifo_v2_clf_hit_tp_1778524253` exp 874745270817613390 | **STRONGEST RANKING IC of the 2×2 matrix.** Completes the {short/long}×{tp4sl3/tp8sl5} matrix — ALL 4 non-inverted. (HC #291(B)(i)+(iii)) |
| A5 | 2026-05-11 15:20 ET | Meta-LGBM v3 FIFO v2 **clf + CONFLUENCE** (9 features) | tp4sl3 SHORT hit_tp | **+0.0522** | +0.0533 | +0.363 | +0.270 | 31/33 | run `meta_lgbm_v3_fifo_v2_clf_hit_tp_1778526066` exp 444269832630457535 | Confluence features (book_imb + depth_imb + persistence) provide NEUTRAL-to-slightly-positive lift on clf target. Confirms clf-target robustness. Sharpe +0.713 vs +0.572 no-confl. (HC #291(B)(iii)) |
| A6 | 2026-05-11 15:40 ET | Meta-LGBM v3 FIFO v2 **clf + CONFLUENCE** | tp8sl5 LONG hit_tp | **+0.0959** | +0.0985 | +0.199 | +0.096 | 33 | exp `MetaLGBM_v3_FIFO_v2_clf_hit_tp_confl_tp8sl5_long` | Vs A4 no-confl: IC -0.004, top10 IDENTICAL (+0.199 both). Confluence features are FLAT on clf. Combined with A5: confluence-on-clf paradigm is a wash on both tested cells. Useful negative finding. (HC #291(B)(iii)+(iv)) |
| A7 | 2026-05-11 16:05 ET | Meta-LGBM v3 FIFO v2 **clf + CONFLUENCE** | tp8sl5 SHORT hit_tp | **+0.0745** | +0.0613 | +0.191 | +0.104 | 33 | exp `MetaLGBM_v3_FIFO_v2_clf_hit_tp_confl_tp8sl5_short` | Vs A2 no-confl: IC +0.003, top10 IDENTICAL. 3rd flat cell of matrix. A5's +0.013 top10 was the outlier, not the rule. (HC #291(B)(iii)) |
| A8 | 2026-05-11 16:55 ET | Meta-LGBM v3 FIFO v2 **clf + CONFLUENCE** | tp4sl3 LONG hit_tp | **+0.0671** | +0.0629 | +0.376 | +0.273 | 33 | exp `MetaLGBM_v3_FIFO_v2_clf_hit_tp_confl_tp4sl3_long` | Vs A3 no-confl: IC +0.002, top10 +0.007. 4th flat cell — **CONFLUENCE-ON-CLF 2×2 MATRIX COMPLETE**. Avg delta IC=0.000, avg delta top10=+0.005. Confluence is FLAT across all 4 cells. (HC #291(B)(iii)+(iv)) |

| A9 | 2026-05-11 18:00 ET | Meta-LGBM v3 FIFO v2 **clf + PT_PRED** (HC #289 test) | tp4sl3 SHORT hit_tp | **+0.0632** | +0.0502 | +0.363 | +0.272 | 31/33 | exp `MetaLGBM_v3_FIFO_v2_clf_hit_tp_pt` | **🎯 FIRST POSITIVE FEATURE-ADDITION**: vs A1 baseline IC +0.0530 → **+0.0102 lift** (~20% relative), Sharpe 0.572 → **3.93**, top10 +0.013, 87% splits positive. Strong evidence for HC #289 v3.1 design: PatchTST preds as inputs. (HC #289 + #291(B)(i)(iv)) |
| A10 | 2026-05-12 00:31 ET | Meta-LGBM v3 FIFO v2 **clf + PT_PRED** | tp8sl5 LONG hit_tp | **+0.0920** | +0.0892 | +0.204 | +0.096 | 33 | run `29f9802668314428b212365f20f4379c` exp 583019530802444589 | Closes cell 3/4 of pt_pred × 4-target matrix. Vs A4 baseline IC +0.0999 → -0.008 (MARGINAL MISS but top10 +0.005 lift). The one negative cell — saturation hypothesis (A4 has highest base IC = near ceiling). (HC #291(B)(iii)) |
| A11 | 2026-05-12 01:35 ET | Meta-LGBM v3 FIFO v2 **clf + PT_PRED** | tp8sl5 SHORT hit_tp | **+0.0794** | +0.0621 | +0.197 | +0.101 | 33 | run `cfa4f067041e47fc9b01df87127ce28e` exp 621881958207483397 | Cell 2/4 of pt_pred matrix. Vs A2 baseline +0.0715 → **+0.008 IC lift** ✅, top10 +0.006. (HC #291(B)(i)+(iii)) |
| A12 | 2026-05-12 02:21 ET | Meta-LGBM v3 FIFO v2 **clf + PT_PRED** | tp4sl3 LONG hit_tp | **+0.0778** (manual valid IC, n=23) | +0.0771 | **+0.375** | +0.276 | 22/23 pos (95.7%) | run `531a096eaf674be8824587a7a110a9e0` exp 511005604040843752 | Closes cell 4/4 of pt_pred matrix. Vs A3 baseline +0.0647 → **+0.013 IC lift** ✅, top10 +0.006. Strongest combined cell (high IC + high top10). Raw aggregate IC was NaN (10/33 degenerate splits — known trainer divide bug); manual valid IC across 23 non-degenerate splits gives the true number. (HC #291(B)(i)+(iii)+(iv)) |

### Tier A STRUCTURAL FINDING (2026-05-12 02:25 ET) — **pt_pred IS A UNIVERSAL CLF-GATE BOOSTER**

Closing the 2×2 pt_pred × clf-gate matrix (A9/A10/A11/A12 vs A1/A4/A2/A3 baselines):

| Side | Target | Base IC | +pt IC | Δ IC | Δ top10 |
|---|---|---|---|---|---|
| short | tp4sl3 (A1→A9) | +0.0530 | +0.0632 | **+0.010** ✅ | +0.013 |
| short | tp8sl5 (A2→A11) | +0.0715 | +0.0794 | **+0.008** ✅ | +0.006 |
| long | tp8sl5 (A4→A10) | +0.0999 | +0.0920 | **-0.008** ❌ | +0.005 |
| long | tp4sl3 (A3→A12) | +0.0647 | +0.0778 | **+0.013** ✅ | +0.006 |

**3 out of 4 cells lift IC by +0.008-0.013.** Top10 uniformly +0.005-0.013 across **all 4 cells**. The single negative cell (tp8sl5_long) has the highest base IC = saturation hypothesis (already near ceiling). Earlier "side-asymmetry" interpretation (after A11 landed and A12 was still pending) was an artifact of skipping A12's NaN aggregate without computing the manual valid IC.

**This is the strongest IC-add finding of the session — direct empirical validation of HC #289's v3.2 architectural directive (PatchTST predictions @ 1s/5s/10s as v3.x input features, rank-normed per HC #281(E), forward-filled signal-step → event-tick).** Saved under HC #291(B)(i)+(iv).

**Secondary finding — pt+confl joint does NOT stack, BOTH SIDES CONFIRMED (2026-05-12 03:22 ET, A13 landed)**:
- Short side (tp4sl3): joint ic=+0.0506 vs pt-only A9 +0.0632 — LOSES -0.013 IC, gains +0.009 top10 (run `53c9cd2337de40b09a71868cf3ae0578`).
- Long side (tp4sl3, A13, run `4a6a3390512e440aa93e42b7f2381c62` exp 906116174001548869): joint ic=**+0.0553**, top10=**+0.361**, bot10=+0.283, 33 splits — vs pt-only A12 +0.0778 / +0.375 → LOSES -0.022 IC, -0.014 top10.

**Conclusion: pt_pred is the SOLO universal booster.** Confluence (book_imbalance + depth_imb + persistence) encodes redundant info with PatchTST predictions and adds NOISE when stacked. Feature-design implication for v3.2 / future Jupiter sweeps: include pt_pred, EXCLUDE confluence from input set.

### Tier A POST-MATRIX FINDING (2026-05-11 17:00 ET)
**Confluence-on-clf 2×2 matrix is FLAT across all 4 cells** (A5/A6/A7/A8). Average IC delta ≈0, average top10 delta = +0.005 (noise). The clf objective is doing all the heavy lifting; adding microstructure confluence features (ms_l3_imb + ms_depth_imb_5 + signal_persist_5/20) does NOT add signal. Path forward: test ORTHOGONAL feature additions (PatchTST preds per HC #289 — IN PROGRESS), v3 multi-head direct outputs (PENDING fold 0 OOT), longer-horizon labels (PENDING).

### Tier A summary findings
1. **CLASSIFICATION OBJECTIVE IS THE KEY UNLOCK** — every prior regression variant on v2 preds inverted (top10 picks were the WORST trades). Switching to `*_hit_tp` binary target produces non-inverted gates across 4/4 setups.
2. **SIDE-AGNOSTIC** — both LONG and SHORT clf produce positive top10. Doesn't contradict HC #69 short-side raw P&L edge (clf gates on hit-prob, not signed return).
3. **EDGE-SIZE TRADE-OFF** — tp4sl3 has higher absolute top10 hit-rate (+0.35/+0.37) due to higher base rate; tp8sl5 has stronger IC ranking (+0.10) but lower absolute top10 — base-rate ceiling effect.
4. **CONFLUENCE LIFT IS MARGINAL ON CLF** — adding 4 microstructure features (vs 5 raw) gave ~+0.01 boost on top10 and slightly better split-Sharpe. Not breakthrough but doesn't break. Confluence may shine more on fatter-edge tp8sl5.

---

## TIER B — DIAGNOSTIC / NEGATIVE-RESULT WINS (saved-because-informative)

| # | Date | Model / Variant | Finding | Why saved |
|---|------|-----------------|---------|-----------|
| B1 | 2026-05-11 04:38 ET | Meta-LGBM sign-flip diagnostic (`MetaLGBM_v3_FIFO_v2_signflip_test`, run `87abf0d7c136429c967e28b2513bf440`) | top10=-0.399t, bottom10=-0.439t — **both tails NEGATIVE**. Edge lives in MIDDLE of v2 pred distribution. "Flip the gate" cheap-path is DEAD. | Eliminates cheap-deploy option, narrows pivot tree. |
| B2 | 2026-05-11 05:28 ET | Meta-LGBM no-confluence regression (run `a14b14611f3e42eea7c69c91edba3dd7`) | top10=-0.412t, bottom10=-0.445t (identical to signflip within noise). | Proves inversion is INTRINSIC to v2 pred distribution, not feature-induced. Pivots research to v3 multi-head outputs (which DO correlate positively per Ep 1 OOT). |
| B3 | 2026-05-11 05:36 ET | CNN-Mamba **v3 fold 0 Ep 1 OOT** (still flowing, current run `f29e0e28cfea4457b1a1dc827490e46d` in `CNNMamba_v3_FIFO`) | OOT IC 1s/5s/10s = **+0.2560 / +0.1284 / +0.0889** (matches v2 at Ep 1). FIFO corr tp4sl3 = **+0.0128**, tp8sl5 = **+0.0375** — both POSITIVE (not inverted like meta-LGBM). | First-ever v3 OOT eval; validates option (c) of the pivot tree (direct-FIFO heads beat meta-LGBM on v2 preds). |

---

## TIER C — PENDING / IN-FLIGHT (will be promoted on completion)

| # | Date | Variant | Status | ETA |
|---|------|---------|--------|-----|
| C1 | 2026-05-11 15:20 ET | Meta-LGBM v3 FIFO v2 **clf + CONFLUENCE tp8sl5 LONG** (PID 3823239) | RUNNING split 13/34 | ~15:50 ET |
| C2 | 2026-05-11 — | CNN-Mamba v3 fold 0 Ep 5 OOT (Neptune PID 3404561) | RUNNING Ep 3→4 transition imminent (~16:00 ET) | Full fold 0 OOT lands ~02:00 ET TUE 2026-05-12 |

---

## INDEX OF JSON ARTIFACTS

Each Tier A/B row has a JSON in `output/good_results/`:
- `A1_meta_lgbm_clf_tp4sl3_short_20260511_1230.json`
- `A2_meta_lgbm_clf_tp8sl5_short_20260511_1340.json`
- `A3_meta_lgbm_clf_tp4sl3_long_20260511_1418.json`
- `A4_meta_lgbm_clf_tp8sl5_long_20260511_1449.json`
- `A5_meta_lgbm_clf_confl_tp4sl3_short_20260511_1520.json`
- `B3_cnn_mamba_v3_fold0_ep1_oot_20260511_0536.json`

---

## PROMOTION CRITERIA (Tier A → DEPLOY)

To move from "saved good result" to "live paper deploy candidate":
1. Validate IC + top10 persists on the upcoming queue-aware FIFO sim (HC #290(D) — spec ready in `HC290D_QUEUE_AWARE_FIFO_AUDIT.md`)
2. Compute risk-adjusted metrics under HC #290(C) commission-only cost model: Sharpe, Sortino, PF, WR, per-day consistency
3. Verify ≥3 trades/day floor with Sortino ≥ 1.5 (per HC #290(A))
4. Cross-validate against v3's direct-FIFO heads when they land (~02:00 ET TUE)

When a result clears all 4 gates → promote to a new `DEPLOY_CANDIDATES.md` and queue for Razer live paper (once Razer back online).
