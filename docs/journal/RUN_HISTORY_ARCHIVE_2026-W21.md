# RUN_HISTORY ARCHIVE — 2026-W21 (entries pre-2026-05-10)
# Archived on 2026-05-24 per HC #489 rolling window rules (14-day keep)

## 2026-05-09

| Time | Node | Experiment | Status | Key Result | Notes |
|------|------|-----------|--------|------------|-------|
| 13:10 | Jupiter | **build_missing_bookfeats.sh** | 🟢 RUNNING (PID 3248169) | Building book_features for 29 missing dates (20260316..20260429). Log: logs/build_missing_bookfeats.log. ETA ~30-60 min. After: re-run Phase 2 enricher → Phase 3 training (Neptune) → Phase 4 validation. | Real cause of "meta-LGBM gate keeps failing" — root-caused this turn. |
| 13:05 | Jupiter | **CRITICAL: Phase 3 broken since 20260316** | 🔴 ROOT-CAUSED | All 29 enriched parquets for 20260316..20260429 silently produced 26 cols instead of 53 (book_features missing). Phase 3 training crashed instantly on KeyError. Killed Neptune wrapper PID 2332272. | "Training in progress 30-45 min" claim was WRONG; was actually crashing in loop. |
| 12:53 | Jupiter | HC #272 added (fill% mandatory in headline) | ✅ APPLIED | After user "But u don't see any fill%" feedback. | DIRECTIVES.md updated. |
| 05:48 | Jupiter | **Phase 4 validator HC #271(A)+(C) wiring** | ✅ APPLIED (syntax+compile OK) | Added dominance test (PatchTST-only + CNN-Mamba-only baselines at same trade count) + per-regime breakdown at best threshold + HC #271(A) gate verdicts. New CLI args `--feat-dir`, `--regime-path`, `--skip-dominance`. Writes `meta_lgbm_dominance_test.json` + `regime_breakdown_best` field. | Per HC #271(C). Validates whether meta-LGBM truly dominates both parents on Sortino + WR + PF + NET + %folds-positive. |
| 05:46 | Jupiter→Neptune | **Pipeline driver Phase 3 redirect** | ✅ APPLIED (PID 3062290) | Phase 3 now rsync's enriched parquets to Neptune, SSH-launches `train_meta_lgbm_gate.py` with `MLFLOW_TRACKING_URI=http://neptune-win:5000`, rsync's results back. Graceful local fallback. lgbm 4.6.0 / sklearn 1.8.0 / mlflow 3.11.1 installed in Neptune venv_training. | HC #263 violation (Neptune idle 8h+) ends as soon as Phase 1+2 hits 46/46. |
| 05:42 | Jupiter | **Regime label backfill** | ✅ COMPLETE | 11 additional dates labeled (5 up / 3 down / 3 flat). Combined with original 25 → 36 OOT dates total (19 up / 13 down / 4 flat = 52.8% green / 36.1% red / 11.1% flat). Vol buckets re-tertiled: 12 high / 12 med / 12 low. Written to `output/regime_labels/oot_dates_regime.parquet`. | 10 of the original 21 backfill dates skipped (no MBO file on Sat/Sun/holidays). Realistic OOT-population denominator is 36, not 56. |
| 05:38 | Jupiter | tp8sl5 winner re-slice on 36-date table | ✅ COMPLETE | All 14 tp8sl5 fold dates were in the original 25 — no new conclusions. UP regime concentration **55.9% (FAIL HC #271(A) ceiling 50%)**, profitable in 3/3 regimes (PASS), 66.7% fold-positive (PASS HC #254), low-vol 100% positive. | Output: `output/regime_labels/tp8sl5_by_regime.csv`. |

| 04:30 | Jupiter | **sweep-result-watcher PM2 service** — HC #269 install | ✅ ONLINE (id=16, saved) | Tails topx_sweep_master.log; on DONE lines auto-posts FIFO result block to Discord #general → bridge wakes Claude. Self-test HTTP 200. | Closes the autonomy gap — no more silent waits while bash autopilot runs. Per HC #269. |
| 04:30 | Jupiter | **TP/SL sweep summary — NEW WINNER tp8sl5** (raw FIFO per HC #268) | ✅ COMPLETE | TP=8/SL=5 short top0.1%: **sum NET +50.95t, +0.980 t/trade, median fold +1.041, 66.7% folds positive.** 10× the NET of tp4sl3. NEW Monday-deploy candidate. tp10sl4_asymm: +16.45t NET, 66.7%+ (also PASS). tp4sl3: +4.95t NET, 75%+ (PASS, most consistent but small). | All other variants FAIL. tp4sl3 was over-tight on TP — short edge runs ~1.4t gross, needs wider TP to capture. |
| 04:30 | Jupiter | Cancel-ms sweep at TP4/SL3 short top0.1% | ✅ COMPLETE | **2000ms is optimal** (default, +4.95t NET, 75% folds+). 1000ms over-cancels (NET −1.79t). 3000/5000ms hold stale orders (NET −9.68t / −17.69t). | HC #248 cancel rule confirmed: 1-2s window is correct; 2s narrowly beats 1s on this dataset. |
| 04:30 | Jupiter | Hold-ms sweep at TP4/SL3 short top0.1% | ✅ COMPLETE | **30000ms is optimal** (default, +4.95t NET, 75% folds+). 15s slightly worse (+2.95t, 58%+). 45/60/90s collapse (NET −11.5/−19.0/−17.5t). | Confirms signal half-life ~250ms — anything ≥45s holds noise that mean-reverts. |
| 01:24 | Jupiter | **paper_trading_mamba_v2_patched.py — `--side-filter` patch APPLIED** | ✅ APPLIED | 9 edits per PATCH_SPEC_PAPER_TRADER_SIDE_FILTER.md, syntax-validated. Backup at .bak_pre_sidefilter. | Fail-closed default `both`. Adds `_side_allowed()` gate at 3 entry sites. Ready to deploy to Razer for Monday open with `--side-filter short --min-tier "Top0.1%"`. |
| 01:14 | Jupiter | FIFO TP/SL sweep at top-0.1% short (PID 2995769) — partial | 🟢 RUNNING (3/4 done, tp10sl4_asymm queued) | **tp4sl3 PASSES HC #254**: 52 trades, sum NET +4.95t, 75% folds positive (vs 58.3% for tp6sl4). tp3sl2 FAILS (NET −13.55t, 41.7%). tp8sl5 in progress. | First config that meets HC #254 floor (≥60% folds +). PID 2995769. Logs in /home/jupiter/Lvl3Quant/logs/fifo_replay_v3_top0_001_short_*.log. |
| 01:25 | Jupiter | Post-TP/SL chained sweep (PID 3004273) | 🟢 QUEUED (waits on PID 2995769) | 4 phases: cancel-window (1/2/3/5s), hold-time (15/30/45/60/90s), tighter top-pct (0.05/0.075/0.15%), long-side sanity. All at TP4/SL3 short top-0.1% baseline. | Per HC #266 forward queue. Reports raw FIFO ticks per HC #268. Script /tmp/run_post_tpsl_sweeps.sh. |
| 01:00 | Neptune | supervised_exec_v3_h10 (PID 2012997) | ⚠️ STUCK in MLflow retries | Extraction done (113K samples / 6 dates — should be 22+ dates; multiple worker crashes). Now stuck retrying MLflow URI jupiter:5000 (Tailscale-routed Neptune can't reach LAN IP, needs neptune-win). GPU 0% / 118 MiB / 27W. | Wrong MLflow URI — should be Tailscale IP per CLAUDE.md. Process not producing progress. NEEDS REMEDIATION but did not kill (no clean restart path tonight without verifying staging script). |
| 00:00→01:00 | Jupiter | FIFO selectivity sweep (top 0.1/0.25/0.5/1.0/2.0% × short × limit) | ✅ COMPLETE | top-0.1% short was the unique winner: 52 trades, sum gross +34.50t, **sum NET +14.95t**, 58.3% folds+. All other selectivities NET-negative. | Establishes the "top 0.1% short-only" deploy gate. Used as anchor for TP/SL sweep above. |
| ~22:00→23:00 | Jupiter | FIFO replay variant sweep (default, A pnl-rank, B chase, C short-only, D pnl+chase) | ✅ COMPLETE | default −0.483t/trade NET (FAIL); A −0.517t (worst); B −0.422t (chase improves gross to −0.046t but eats commission); C **+0.123t gross** (first positive) but still −0.253t NET; D mixed. | Variant C established that SHORT-ONLY gross edge exists. Pivot to selectivity sweep led to the top-0.1% finding. |



## FORMAT: Date | Node | Experiment | Status | Result | Notes

---

## 2026-05-08

| Time | Node | Experiment | Status | Key Result | Notes |
|------|------|-----------|--------|------------|-------|
| 02:20 | Neptune | FIFO Validate v2 (17 passive/chase configs, symmetric TP/SL) | ✅ COMPLETE | **ALL 17 CONFIGS NEGATIVE**. Best: c200 pass tp4/sl4 Sort=-0.35 (33% green, only 325 trades). Chase WORSE than passive (-2.26tk vs -0.61tk). Trail/flipex WR=0.000 (bug). | Symmetric TP/SL needs 51% WR, FIFO drops to 43-49%. |
| 02:35 | Jupiter | FIFO Validate v3 (16 chase+conviction configs) | ✅ COMPLETE | SHORT catastrophic (WR=36%), LONG equally bad (WR=38%), force-cross worst (-2.39tk/trade). ALL NEGATIVE. | Chase+conviction from Apr 26 doesn't work with CNN-Mamba v2. |
| 07:32 | Neptune | FIFO Validate v4 MARKET ENTRY (19 configs, market orders at top 0.5-5%) | 🟢 RUNNING (14/19) | **ALL 13 COMPLETE CONFIGS CATASTROPHIC**. WR=7-27%, 0/33 green. Market WORSE than passive. Best: top2% tp4/sl4 WR=25.9%. | Market entry definitively disproven. |
| 07:30 | Jupiter | FIFO Meta-Layer v2 (XGBoost on FIFO outcomes) | ✅ COMPLETE | **AUC=0.529 (near random)**. BUG: only 13 features, not 107 — all_oot prediction files lack embeddings. Top1% WR=31.5% vs 29.7% base. | Embeddings in smart_v3_mar files only. Need to regenerate all_oot with embeddings. |
| 08:15 | Jupiter | Conditional Entry Sweep v1 (21 smart configs) | 🟢 RUNNING | PID 2707609. Testing queue position filtering, fast cancel, ratchet stops, time windows, conviction exits, combined smart stacks. | Key question: can smarter entry/exit reduce adverse selection enough? |
| 02:00 | Jupiter | FIFO Outcome Predictor v1 (LGBM/XGB on trade features) | ✅ COMPLETE | AUC=0.50 (random). Pre-trade features cannot predict FIFO profitability. Top feature: time_of_day. | 10 features from fill_sim per-trade data. Needs enriched features or fundamentally different approach. |
| 01:40 | Jupiter | Adverse Selection Analysis v1 | 🛑 KILLED | Only 4/36 dates done. Killed to make room for v3 (more actionable). | Partial results showed queue position and signal strength distributions. |
| 01:16 | Neptune | FIFO Validate v1 (5 configs, baseline + gated) | ✅ COMPLETE | **ALL CONFIGS NEGATIVE**. Best: long_c075_tp4_sl8 = 2933tr, WR=52.2%, PnL=-1776tk, 8/39 green, Sortino=-0.39. Baseline: -7851tk. | Midpoint edges don't survive FIFO. Adverse selection ~5tk/trade. |
| 01:00 | Neptune | Focused Gate v1 (rules sweep + daily WF LGBM) | ✅ COMPLETE | Best rules: C≥0.75+Agree+Persist+LongOnly @5s: WR=54.8%, PF=1.47 (141 trades). Daily WF LGBM at 0.55: WR=56.9% at 1s. | First positive OOT edges found. |
| 00:42 | Neptune | Regime Gate v1 (extreme classifier) | ✅ COMPLETE | Regime lookup table created. Best cell: PF=1.05, WR=52.5%. | Extreme classifier approach (top/bottom 10%). |
| 00:50 | Jupiter | Parametric Exit v1 (Optuna, 300 trials/fold) | ✅ COMPLETE | 3 folds, 20 total trades, P&L=$-214. Unstable parameters across folds. | Too few trades, parameters don't generalize. |
| 00:30 | Neptune | LGBM Exec Filter v1 (10 folds, smart_v3_mar) | ✅ COMPLETE | 8 OOT folds, 211K samples. AUC=0.538. **Top1%: WR=54.3%, Edge=+0.548t, PF=1.34, Sortino=0.106**. Top5%: WR=52.7%, Edge=+0.387t, PF=1.23. Trees beat MLP decisively. | LGBM outperforms MLP on tabular data. abs_pred_1s is top feature. |
| 00:32 | Neptune | LGBM Exec Filter v2 (all 50 OOT folds) | 🟢 RUNNING | More robust validation with 50 folds. Early folds showing top1% WR=58.6%, PF=1.49. | Running on all_oot data for generalization check. |
| 00:34 | Neptune | LGBM Exec Filter v3 (no embeddings ablation) | ✅ COMPLETE | 47 OOT folds. AUC=0.554. Top1%: WR=53.7%, Edge=+0.376t, PF=1.19. Embeddings add ~2% WR at top1%, bigger gap at top5%. | Confirms embeddings contribute meaningful signal. |
| 00:35 | Neptune | LGBM Exec Filter v4 (threshold=1.0) | ✅ COMPLETE | 47 OOT folds. AUC=0.551. **Top1%: WR=57.4%, Edge=+0.381t, PF=1.31**. Lower threshold finds cleaner boundary. | Best WR of all versions. |
| 00:37 | Neptune | LGBM Exec Filter v5 (threshold=2.0) | ✅ COMPLETE | AUC=0.557. Top1%: WR=52.9%, Edge=+0.391t, PF=1.16. Worse WR, decent edge. | Higher threshold makes it harder for model to learn boundary. |
| 00:38 | Neptune | LGBM Exec Filter v6 (10 train folds) | ✅ COMPLETE | AUC=0.558. Top1%: WR=55.2%, Edge=+0.443t, PF=1.27. **Best top5%: WR=54.5%, Edge=+0.370t, PF=1.22.** | More training data helps especially at top5%. |
| 00:41 | Neptune | LGBM Exec Filter v7 (best combo: thresh=1.0 + 10 folds) | ✅ COMPLETE | AUC=0.549. Top1%: WR=57.1%, Edge=+0.425t, PF=1.31, Sortino=0.088. | Combines best settings — PF=1.31 at top1% confirmed. |
| 00:48 | Neptune | LGBM Exec Filter v8 (LONG-ONLY, thresh=1.0) | ✅ COMPLETE | 675K samples. AUC=0.554. Top1%: WR=57.7%, PF=1.27. **Top5%: WR=55.1%, Edge=+0.285t, PF=1.18** ← BEST top5% overall. | Longs outperform shorts — paper trader should go long-heavy. |
| 00:49 | Neptune | LGBM Exec Filter v9 (SHORT-ONLY, thresh=1.0) | ✅ COMPLETE | 508K samples. AUC=0.547. Top1%: WR=55.3%, PF=1.14. Top5%: WR=53.8%, Edge=+0.109t, PF=1.05. | Short side much weaker, barely profitable at top5%. |
| 00:52 | Neptune | **XGBoost Exec Filter v1** (thresh=1.0, both sides) | ✅ COMPLETE | 1.18M samples. AUC=0.548. **Top1%: WR=58.1%, Edge=+0.491t, PF=1.34** ← BEST OVERALL. Top5%: WR=56.1%, Edge=+0.373t, PF=1.24. Spearman=0.052. | XGBoost beats LGBM across the board. |
| 00:55 | Neptune | XGBoost Exec Filter v2 (LONG-ONLY, thresh=1.0) | ✅ COMPLETE | 675K samples. AUC=0.554. Top5%: WR=55.6%, Edge=+0.361t, PF=1.22. Spearman=0.056 (highest of all variants). | Long-only XGBoost has best Spearman correlation. |
| 00:06 | Neptune | Supervised Exec MLP v3 (small network [64,32]) | ✅ COMPLETE | AUC=0.521, Top1% WR=50.8%, PnL=+0.019t, PF=1.01. Barely above random. **Sanity check: signal confidence Spearman w/ MFE = 0.0098 (near zero!)**. | All 3 MLP versions (v1-v3) failed. Fundamental ceiling in the data for MLP approach. |
| 23:21 | Neptune | Supervised Exec MLP v2 (large [512,256,128,64]) | ✅ COMPLETE | AUC=0.518, similar to v1. Larger network didn't help. | Overfitting concern with 4 layers. |
| 23:04 | Neptune | Supervised Exec MLP v1 ([256,128,64]) | ✅ COMPLETE | AUC=0.518, Top1% WR=53%, PnL=+0.188t, PF=1.14. Marginal. | Per HC #245. MLP predicting MFE/MAE/PnL from signal+context. |

## 2026-05-07

| Time | Node | Experiment | Status | Key Result | Notes |
|------|------|-----------|--------|------------|-------|
| 20:42 | Jupiter | Exit-Only DQN v6 (regularized: dropout=0.15, wd=1e-4) | 🟢 RUNNING | PID 2528550. Tests if dropout+weight_decay fix overfitting. Eval at threshold=0.30 (no tier filter) for max trades. | v5 showed clear overfitting — training went positive but eval was terrible. |
| 19:58 | Jupiter | Exit-Only DQN v5 (Top5% eval gating) | ✅ COMPLETE | 91 OOT trades, WR=10.2%, Sortino=-0.67, P&L=-$360. Training curve excellent (+$678 Ep10) but overfits — time_stop dominates eval exits. | Overfitting: training PnL improves consistently but eval doesn't generalize. |
| 19:30 | Jupiter | Exit-Only DQN v4 (higher eval threshold) | ✅ COMPLETE | eval-threshold=0.50 made things worse. WR dropped from 24.8% to 9.8%. | Higher threshold didn't help — eliminated good trades too. |
| 18:15 | Jupiter | Exit-Only DQN v3 (epsilon fix) | ✅ COMPLETE | 215 OOT trades, WR=24.8%, Sortino=-0.17, P&L=-$722. BEST version of exit-only DQN. | EPS_DECAY=10K fixed. Agent learned 3x better MFE capture but entry quality remains bottleneck. |
| 17:30 | Jupiter | Exit-Only DQN v2 (HC #243: confidence-tiered entries) | ✅ COMPLETE | 190 OOT trades, WR=11.2%, Sortino=-0.70, MFE_cap=-0.52. ε never decayed — 93% random at end. | Epsilon decay too slow. Fixed in v3. |
| 16:10 | Jupiter | Exit-Only DQN v1 (HC #240: exit/cancel only) | 🛑 KILLED | PID 2463331. Replaced by v2 (HC #243). Only ~4-5 trades/day at 0.50 threshold — insufficient for learning. | Superseded by v2 with tiered entries. |
| 17:58 | Neptune | Split DQN v4_precomputed (HC #243: full speed iteration) | 🟢 RUNNING | PID ~1060xxx on cuda. 3 folds × 12 epochs, 56 dates, 4 workers. PrecomputedEnv loaded (10-50x speedup). Entry+Cancel+Exit heads. | Replaces v3 slow raw-MBO version. |
| 08:16 | Neptune | Split DQN v3_confidence (HC #234: confidence-aware rewards) | 🛑 KILLED | PID 867823 killed. Slow raw-MBO replay — days per fold. Superseded by v4 with precomputed obs (HC #243). | Replaced by v4. |
| 08:19 | Jupiter | Supervised Exec v1 (HC #234: NEW approach) | ❌ CRASHED | PID 2344184. Fold 0 trained OK (loss converging), crashed on torch.load/save with weights_only issue after fold 0 eval. 250K samples extracted. | Fixed in v2. |
| 09:08 | Jupiter | Supervised Exec v2 (HC #236: smart exec research) | ✅ COMPLETE | 3 WF folds, 99K OOT samples. Top1% PF=1.13, Sortino=0.050. Raw signal top1% already PnL=+1.00/trade. MLP adds near-zero value over raw confidence filtering. | KEY INSIGHT: Execution alpha is in timing/order-type/queue, not outcome prediction. Need Track B v2 focused on execution mechanics. |
| 08:58 | Jupiter | Precompute OOT obs (10 dates ONLY, HC #235) | 🟢 RUNNING | PID 2353982. 10 workers, 3/10 done. ~30-60min ETA for full trading days. | Correct this time — OOT dates only, not all 248. |
| 07:54 | Jupiter | Precompute observations (background) — OLD | 🛑 KILLED | PID 2336984/2346290. Was processing all 248 files (WRONG per HC #235). Killed. | Replaced by OOT-only precompute. |

## 2026-05-05 (late session)

| Time | Node | Experiment | Status | Key Result | Notes |
|------|------|-----------|--------|------------|-------|
| 23:34 | Neptune | Split DQN v1_r22_patched (worker_timeout fix) | 🟢 PRODUCTIVE | PID 37509. After 15 min: Fold 0 Ep 0 5/40 files, 79,846 trades, GPU updates E=800 C=800 X=800, buffers full. Memory 7GB/31GB. | FIX: patched train_split_dqn.py:829 worker_timeout 120→1800. Old 120s fired before any first file completed (~3 min/file), caused E=0 C=0 X=0 forever. Backup at .bak_pre1800. Output: split_dqn_v1_r22_patched/. |
| 23:27 | Razer | Razer launcher v3 (razer_full_stack_v3.py) | 🛑 STOPPED | Infrastructure complete: env-var injection, creds file, ASCII logs, correct paths. BLOCKED: server rejects RITHMIC_SYSTEM with rp_code 1067. Tried "Rithmic 01" and "Rithmic Paper Trading". | Need user to provide correct RITHMIC_SYSTEM string from AMP for paper on rituz00100.rithmic.com:443. |
| 23:25 | Razer | Razer launcher v2 (razer_full_stack_v2.py) | ⚠️ SUPERSEDED | Spawned both children OK. Missing Rithmic env vars (no creds injection). | Replaced by v3. |
| 23:21 | Razer | Razer auto-launcher v1 (razer_auto_launcher.py) | ❌ FAILED | UnicodeEncodeError cp1252 (emoji in log). Wrong PatchTST path (fold_10 → actually fold_00). Wrong MBO recorder path. Bogus paper_trader CLI flags (--rithmic-user/--dqn-checkpoint don't exist). | Replaced by v2/v3. |
| 23:18 | Neptune | Cleanup of v1_r20_memory_safe (13 dup processes) | ✅ COMPLETE | Killed 13 duplicate train_split_dqn processes. Memory 12GB → 1.9GB. | Root cause: ProcessPoolExecutor accumulating + launcher relaunching on detected "crash". |

## 2026-05-06

| Time | Node | Experiment | Status | Key Result | Notes |
|------|------|-----------|--------|------------|-------|
| 20:38 | Neptune | Split DQN v2_perhead_r1 (per-head rewards) | 🟢 RUNNING | PID 529730. Fold 0 Ep 0: 20/25 files, **1,600 trades** (was 200k+). Per-head rewards: ENTRY=edge+freq_cost, EXIT=MFE-capture. | HC #221/#230. Killed v2_clean (PID 522641, old reward). Clean launch with: alpha_gate=0.50, n-step=20, Huber, reward clip, ENTRY freq cost -0.02, EXIT MFE-capture ratio bounded [-1,+1]. 100x overtrading reduction confirmed. |
| 20:31 | Razer | Paper trader v3 (V2 inference fix) | 🟢 RUNNING | PID 29300. **CRITICAL FIX: inference engine V2 architecture mismatch**. 57/57 weights loaded (was silently dropping CNN). 45.5% negative preds (was 0%). Rithmic connected, ESU6@CME, warming up. | HC #223 RCA: cnn_mamba_v2_inference.py had V1 arch (cnn_conv_layers) but checkpoint trained with V2 arch (feature_mlp+temporal_cnns+fusion_proj). strict=False silently dropped → random-init CNN → all-positive → 42/42 LONG. FIXED: standalone V2 model class in inference engine. |
| 20:22 | Neptune | Split DQN v2_clean (clean reward) | 🛑 KILLED at 20:36 | PID 522641. Started with round-1 patches but BEFORE per-head reward redesign. Killed to apply HC #221 per-head rewards. | Superseded by v2_perhead_r1. |
| 02:55 | Neptune | Split DQN v1_r15_proper_wf (walk-forward fix) | 🟢 RUNNING | PID 79312, 2 folds (40 train / 5 eval), loading March 2026 OOT data. CPU 138%, memory 595MB. FOLD 0 training started. | FIX: Previous config was 60 train / 1 eval (wrong). Now properly configured for walk-forward validation. Using correct OOT dates. ETA 8-20h. |
| 02:35 | Razer | Paper trader asyncio crash fix + auto-restart wrapper | ⚠️ BLOCKED | Created wrapper script, fixed symbol NQM6→ESU6. Process exits on Rithmic connection failure. | BLOCKER: Rithmic credentials missing (RITHMIC_SYSTEM, RITHMIC_USER, RITHMIC_PASSWORD). Paper trader code is correct, just needs creds to connect. Waiting on user. |
| 02:15 | — | NQ vs ES mismatch ROOT CAUSE + FIX | ✅ FIXED | Changed inference symbol from NQM6 (NQ micro) to ESU6 (ES). Updated paper_trading_mamba_v2_patched.py. Model now trades correct instrument. | Found in logs: signal file was NQM6, paper trader config was ES. Mismatch = 0 useful trades, 254 signals on wrong instrument. FIX verified. |
| 02:05 | — | Logs-only monitoring system created | 🟢 DEPLOYED | monitor_logs_only.py (HC #194): Monitors by log content, not GPU%. Detects stalls, errors, training progress. Running on Jupiter. | Replaces all GPU% tracking. Reads logs in real-time, parses training progress, detects crashes within 5 min. |

## 2026-05-05

| Time | Node | Experiment | Status | Key Result | Notes |
|------|------|-----------|--------|------------|-------|
| 21:35 | — | RECOVERY SESSION (context limit reset) | 🟢 COMPLETE | Neptune training confirmed live (Job 232, 40% GPU, 90.75W). Razer paper trader DEAD (0% GPU). | Verified via QCC health check. SSH to Neptune failed (unknown error) but job shows active in QCC. Razer tasklist empty = process crashed or not started. ROOT CAUSE: Rithmic credentials missing, paper trader module chain may have failed. HC #180 (0 trades) = waiting for credentials or replay mode approval. |
| 17:15 | Neptune | Split DQN v1_r11 (venv_training, direct SSH launch) | 🟢 RUNNING | Fold 0 Ep 0: 20/25 COMPLETE! 1185 trades, GPU updates E=600 C=600 X=200, CPU 88% | ✅ FIXED: venv_training environment was required (system Python had no torch/numpy). Launched via SSH as nick@, no Ray needed. Zero deadlock. |
| 17:10 | Razer | RL checkpoint deployment + paper trader fixes | ⚠️ PARTIAL | Deployed rl_v34_epoch99.pt (1.2MB), fixed streaming_features import | ✅ Fixed paper_trading_mamba_v2_patched.py (streaming_features_smart_v3 → streaming_features). ❌ Blocked: MBO recorder needs Rithmic credentials. |
| 17:07 | Neptune | Split DQN v1_r11 launch (SSH direct, venv_training) | 🟢 STARTED | PID 15251, workers spawning, ~8 min elapsed | Bypassed Ray/QCC due to Job 1370 PID parse failure. Manually launched with correct venv. No deadlock. |
| 21:08 | Neptune | Split DQN v1_r10b (adaptive workers, HC #182, retry) | ❌ FAILED (Job 1371) | Error: "Could not parse PID from output". | Ray dispatcher still failing PID parse. Switched to manual SSH launch (v1_r11). |
| 21:00 | Neptune | Split DQN v1_r10a (HC #182: adaptive workers) | ❌ FAILED (Job 1370) | Error: "Could not parse PID from output". | Ray dispatcher couldn't parse PID. |
| 21:00 | Razer | Paper trading modules bulk deployment | ✅ COMPLETE | 23/23 modules deployed, paper trader now running | Deployed: streaming_features_smart_v3.py, cnn_mamba_v2_inference.py (patched), etc. |
| 20:36 | — | HC #182 ADDED to DIRECTIVES.md | ✅ COMPLETE | Adaptive worker scaling | Implemented in v1_r11. |

## Prior 2026-05-05

| Time | Node | Experiment | Status | Key Result | Notes |
|------|------|-----------|--------|------------|-------|
| 09:40 | Jupiter | AUTO_CHECK + RECOVERY (2026-05-05) | ✅ COMPLETE | Razer ✅ live 4/4 procs, Neptune ❌ network unreachable (Tailscale relay MIA) | STATE: Razer trading actively. Neptune v1_r10 deadlock-free, last known File 5/25 @ 09:05:35. Network restore needed. HC #174 (Razer reliability) verified complete. |
| 09:02 | Neptune | Split DQN v1_r10 (code fix: as_completed timeout + wait strategy) | 🟡 UNREACHABLE | PID 35083. **Last known**: Epoch 1 File 5/25 @ 09:05:35. Passed deadly file 20-25 boundary ✅. Epoch 0 complete in 2:20, Epoch 1: 86 trades, $-2378 PnL (buffer warmup). | **CRITICAL FIX APPLIED**: `as_completed(futures)` loop replaced with `concurrent.futures.wait(timeout=120)`. Solves v1_r7/r8/r9 deadlocks. Process alive but Neptune network down (SSH timeout, ping loss). |
| 08:47 | Neptune | Split DQN v1_r9 (data=mbo_events_smart_v3, batch=4096, workers=30) | ❌ STALLED 13+ min | PID 28517. Last log file 20/25 @ 08:47:18, killed @ 09:00:18. **Root cause confirmed**: `as_completed()` loop blocks forever on hung worker. Process appears running but output frozen. Insufficient fix (HC #169 shutdown only). | HC #169 (shutdown wait=False) was incomplete. Deadlock is in futures wait loop, not cleanup. |
| 07:41 | Neptune | Split DQN v1_r8 (data=mbo_events, deadlock fix) | ❌ SILENT FAILURE | "Completed" 9 folds in 5 min with 0 trades/0 GPU updates/0 PnL. Every npz file failed to load: 'event_type_raw is not a file in the archive'. | **Deadlock fix WORKED** (no hangs) but data dir lacked required field. Replaced by v1_r9. |
| 07:37 | Neptune | Split DQN v1_r8 (batch=4096, workers=30, **DEADLOCK FIX APPLIED**) | 🟢 RUNNING | PID 7082, VRAM 778MB, ~60 worker procs spawned. Code fix: ProcessPoolExecutor.shutdown(wait=False) in finally block prevents epoch-transition deadlock. | **DEADLOCK ROOT CAUSE FIXED**: HC #170 (max speed, 30 workers), HC #171 (checkpoints TBD). User restarted Neptune manually. Expect to break past file 20/25 boundary. Next: add checkpoint saving. |
| 04:01 | Neptune | Split DQN v1_r5 (batch=512, workers=4, buffer=62.5k) | ❌ KILLED | **ROOT CAUSE CONFIRMED**: Stalled at same 20/25 file boundary like all others. Reducing workers does NOT fix deadlock. | **CODE DEADLOCK** — not config issue. Epoch transitions have blocking/synchronization bug regardless of worker count. Halting trials. |
| 04:00 | Neptune | Split DQN v1_r4 (batch=1024, workers=8, buffer=125k) | ❌ STALLED | Stalled at Fold 0 Ep 0:20/25 after 14 min. Made progress initially but hit same boundary deadlock. | Initial optimism was premature. v1_r4 progressed past initial startup but stalled at epoch transition like all others. |
| 03:46 | Neptune | Split DQN v1_r3 (batch=2048, workers=20, buffer=250k) | ❌ KILLED | Stalled at Fold 0 Ep 0:20/25, log frozen 14 min | Batch reduction alone insufficient. Pattern persists. |
| 03:30 | Neptune | Split DQN v1_r2 (batch=4096, workers=30) | ❌ KILLED | Stalled at Fold 0 Ep 2:20/25, log frozen 13 min | 30 workers + async = deadlock. |
| 06:35 | Neptune | Split DQN v1 (batch=4096, workers=24) | ❌ KILLED | Stalled at Fold 1 Ep 0:20/25, log frozen 4h+ | Original failure. Same pattern. |

**DIAGNOSIS**: All 5 attempts stall at ~20/25 file / epoch boundary. Root cause is **code-level multiprocessing deadlock** in epoch completion (checkpoint save, worker sync, or buffer finalization). Not solvable by tuning. Requires code audit.

## 2026-05-04

| Time | Node | Experiment | Status | Key Result | Notes |
|------|------|-----------|--------|------------|-------|
| ~22:07 | Neptune | Split DQN v1 (HC #166: 3 networks Entry/Cancel/Exit, Dueling+Double+PER) | ❌ STALLED | Fold 1 Ep 0:20/25, last progress 02:24 | GPU 45% initially, but training hung (likely memory/deadlock). Killed 06:35. |
| ~21:10 | Neptune | SAC v12 MAXSPEED (batch=4096, TF32, foreach clip, async GPU) | ❌ KILLED (superseded by Split DQN) | 53% GPU sustained, proved async works | Replaced per HC #166 — DQN with split entry/exit |
| ~20:26 | Neptune | SAC v11 ASYNC GPU (decoupled GPU training thread) | ❌ KILLED (superseded by v12) | 85K GPU updates by worker 20 (14x more than v10). GPU 34% sustained. | Proved async works, but batch=512 too small for GPU saturation. |
| ~19:17 | Razer | Confluence paper trader (CNN-Mamba v2 + PatchTST, price_ticks FIXED) | 🟢 RUNNING | 215MB VRAM, both models on CUDA. 17%L/82%S predictions (balanced). | PID 25316. Fixed critical price_ticks bug (absolute vs relative). |
| ~19:22 | Jupiter | SAC RL CPU training | ❌ KILLED (unproductive) | CPU too slow for neural net training | User correctly identified — RL needs GPU |
| ~13:00 | Neptune | SAC v10 FIFO (FIXED: pred loading + alpha sign + alpha clamp) | ❌ KILLED (superseded by v11) | Fold 0: Sortino=-0.042 eval, WR=63.4%, PF=0.85, -$427. Fold 1 Ep9 same plateau. | Alpha fix working but results mediocre. GPU bursty 0-43%. |
| ~09:00 | Neptune | SAC v9c FIFO (multicore, HC #146/#150) | ❌ KILLED (trained blind) | Fold 0: Sortino=-1.0, 4 trades, entropy=0. EVAL: 0 trades. | TWO bugs: (1) pred glob mismatch → 0 predictions loaded, (2) alpha loss sign inverted → entropy death spiral |
| ~02:05 | Neptune | SAC v8 FIFO (true FIFO env, HC #135, no spread cost) | SUPERSEDED by v9c | Fold 0 Ep1, File 3/25, 314 trades. | 256 hidden, 179K params, 12 epochs/fold, 7 folds, 46 OOT dates. Fixed: spread cost=0 (HC #127), date filter (HC #136). |
| ~02:05 | Razer | PPO v8 FIFO (true FIFO env, HC #124 Razer=PPO) | ❌ KILLED (stale) | Never produced useful output. | Old launch attempt, replaced below. |
| 00:07 | Razer | PPO v8a FIFO (HC #143, all nodes v8) | ✅ COMPLETE (FAILED) | 0 eval trades. Model collapsed — entropy too low (0.01). Train Sortino peaked 6.94 but n_trades→1. | 128 hidden, 20 ep/fold, rollout 4096, entropy 0.01. |
| 03:30 | Razer | PPO v8b FIFO (entropy fix) | RUNNING | PID 13888, just launched. | 128 hidden, 20 ep/fold, rollout 8192, **entropy-coef=0.05** (5x higher). Fix for v8a collapse. |
| 00:07 | Razer | v7 paper traders (SAC v7 + PPO v7 + Top 0.1% + Top 1%) | ❌ KILLED | User: "v7 setups don't do anything for us" | HC #143. |
| ~02:00 | Neptune | SAC v7 (PID 1999581) | ❌ KILLED (stuck) | Folds 0-3 EVAL: +2.04, +2.91, +1.52, +0.80 (declining). Stuck at Fold 4 Ep0 for 4.5h. | best_sac_fold1.pt FROZEN for Monday paper. NOT true FIFO. |
| ~02:00 | Razer | PPO v7 v4 (PID 11680) | ❌ KILLED (stalled/bad) | Fold 0 Ep9: Sortino +2.42, WR 47.3%, stalled 4.5h. | No checkpoints saved. NOT true FIFO. |

## 2026-05-03

| Time | Node | Experiment | Status | Key Result | Notes |
|------|------|-----------|--------|------------|-------|
| ~21:25 | Neptune | FIFO PPO v5 Multi-Model (HC #123, right-sized) | ❌ KILLED | 0 trades — 7 critical bugs in fifo_rl_env.py (mid never updates, pred index broken, delta Sortino negative EV, rollout too short, alpha gate penalizes everything). | Replaced by PPO v7. |
| ~22:50 | Razer | PPO v7 v4 (relaunched, 43 dates, --no-mlflow) | RUNNING | GPU 83%, just launched | 64 hidden, 5,894 params, 20 epochs/fold, 7 folds. Relaunched after v3 stalled (0% GPU, CPU-only zombie). |
| ~19:44 | Razer | PPO v7 v2 (relaunched, 43 dates, --no-mlflow) | CRASHED | Fold 0 Ep0: Sortino=+0.866, WR=39.7% | 64 hidden, 5,894 params, 20 epochs/fold, 7 folds. Relaunched after v1 crashed during Fold 1. |
| ~18:31 | Neptune | SAC v7 Discrete Soft Actor-Critic (HC #124) | RUNNING | Fold 1 Ep6: Sortino=+11.026, WR=50.9%, PF=1.23. Fold 0 EVAL: Sortino=+2.042, WR=50.4%, PF=1.23 | 256 hidden, 220K params, 46 OOT dates, 7 folds. Off-policy replay buffer, dual Q-nets, auto entropy. |
| ~17:58 | Razer | PPO v7 Clean Design (smaller config) v1 | CRASHED | Best EVAL Fold 0: Sortino=+7.189, WR=56.8%, PF=2.08, 158 trades/day. Crashed during Fold 1. | 64 hidden, 5,894 params, 20 epochs/fold, 31 OOT dates. |
| ~17:53 | Neptune | PPO v7 Clean Design (v5 principles) | ❌ KILLED | Best Fold 0 Ep5: Sortino=+2.98, WR=46.3%. Killed for HC #124 (need different RL style on Neptune). | Replaced by SAC v7. 128 hidden, 19,974 params. |
| ~21:20 | Neptune | FIFO SAC v4 FILTERED | ❌ KILLED | 0% WR at epoch 17/30. Only 6/28 files had data. 30-44 trades/epoch. | ROOT CAUSE: pred_dir pointed to cnn_mamba_v2_bulk_inference (Jul-Oct 2025 IN-SAMPLE) not cnn_mamba_v2_bulk_oot (Mar-Apr 2026 OOT). Also hidden=512 way too big for data. |
| ~06:40 | Neptune | FIFO SAC RL Neptune v1 (HC #117, larger config) | ❌ KILLED (superseded by v4) | PID 1814268, GPU 35%→ramping, 748MB. Buffer warmup. | 80 epochs, 40d train, 5d eval, buffer 1M, batch 512, lr 1e-4, hidden 256, updates-per-step 2, PatchTST confluence. |
| ~06:07 | Neptune | CNN-Mamba v2 Warm-Start WF (3 folds) | ✅ COMPLETE | Fold 0: IC_1s=0.356, IC_10s=0.167. Fold 1: IC_1s=0.401, IC_10s=0.229 | Walking forward hugely beneficial (+58-119% vs fold_10). All weights+preds saved. |
| ~05:03 | Razer | FIFO SAC RL v1 (HC #117, discrete SAC, CUDA) | RUNNING | GPU 82%, 456MB VRAM, 2.3GB RAM. Training loop active. | 50 epochs, 40d train, 5d eval, buffer 500K, lr 3e-4, hidden 256. No PatchTST (will add on next run). Logs 0-byte due to WMIC flush issue. |
| ~05:10 | Razer | Cleanup: killed 5 duplicate SAC/launcher processes | DONE | Cleaned leftover from debug session | PIDs 43692, 42112, 43116, 33996, 43844 killed |
| ~05:10 | Jupiter | FIFO RL PPO CPU validation (stale) | KILLED | 1 checkpoint (OBS_DIM=45, obsolete). 24h+ on CPU with no progress. | Killed — OBS_DIM mismatch, superseded by SAC GPU training |

## 2026-05-02

| Time | Node | Experiment | Status | Key Result | Notes |
|------|------|-----------|--------|------------|-------|
| ~22:15 | Razer | FIFO RL PPO with PatchTST confluence (HC #112, CUDA) | RUNNING | GPU 71%, 543MB. FIFO-native env + PatchTST confluence features. | Replaced CPU-bound PPO. 50 epochs, 40d train, 5d eval. launch_fifo_rl_v3.py via WMIC. |
| ~22:00 | Razer | PPO Execution Reward Sweep (CPU-bound) | KILLED | Plateau at $300-500/day. CPU-only, GPU wasted. | Killed per HC #112 — replaced with FIFO RL on CUDA. |
| ~18:02 | Neptune | CNN-Mamba v2 Warm-Start WF RESUMED (HC #99, 4 steps from Apr 11) | RUNNING | Fold 0 ✅ DONE (20:45). Fold 1 epoch 1 at 22%. GPU 100%, CUDA kernels | d_model=96, 50/50 tensors loaded. ETA full WF ~noon Sat. |
| ~02:07 | Razer | PPO Execution Reward Sweep (5 reward fns × 200 epochs) | SUPERSEDED | raw_pnl complete | GPU 31%. Superseded by FIFO RL at ~22:00. |

## 2026-05-01

| Time | Node | Experiment | Status | Key Result | Notes |
|------|------|-----------|--------|------------|-------|
| ~00:05 | Neptune | CNN-Mamba v2 Warm-Start WF (HC #86, 5 steps, Apr OOT) | KILLED (HC #98) | Completed folds 0-2, killed mid-fold 3. Weights+preds saved. | User: models not decayed, smart exec priority |
| ~16:22 | Neptune | Optuna Exec MLP v2 (BOTH sides, 161 feat, 100 trials) | COMPLETE | T67&T92 Sortino 148.4, cd=45/35, sf=2.5 | 100/100 trials, 3h20m. Diminishing returns — no 200 extension. |
| ~16:08 | Jupiter | Lower Threshold Analysis (top 1-50%, both sides) | COMPLETE | ALL thresholds profitable passive! Top 10% short: WR 55.5%, PF 2.91 | Midpoint-based |
| ~15:22 | Neptune | SHORT-ONLY Exec MLP (Optuna #22 params) | COMPLETE | Sortino 6.39 @ thresh 0.55, WR 53.8% | Short-side only, midpoint-based validation |
| ~15:00 | Jupiter | FIFO Rules Sweep (19,854 configs) | COMPLETE | Top 1%: 0.3 trades/day 38% profitable. Top 5%: 4.3/day 3% profitable. Top 10%+: 0% profitable | Proves need for better execution, not more configs |
| ~14:50 | Jupiter | Signal Decay Analysis | COMPLETE | IC_1s=0.183, IC_5s=0.110, IC_10s=0.075. Half-life ~250ms. Short >> Long | Key finding: passive limits profitable at top 20%+ |
| ~14:30 | Neptune | Event Transformer w=1000 | KILLED | Fold 0: IC_1s=0.334, IC_10s=0.213 | User ordered kill — wrong phase. Do NOT re-launch. |
| ~14:00 | Neptune | RL FIFO v5 (PPO agent) | LAUNCHED | Unknown — may have been overwritten by Exec MLP | Check MLflow |

## 2026-04-30

| Time | Node | Experiment | Status | Key Result | Notes |
|------|------|-----------|--------|------------|-------|
| overnight | Neptune | CNN-Mamba v2 retrain (fold 11+) | KILLED per HC #68 | Had fold 10 weights | User: pivot to execution research NOW |
| afternoon | Jupiter | XGBoost exec gate | COMPLETE | Improved selectivity but still midpoint-based | Superseded by FIFO requirement |
| morning | Neptune | CNN-Mamba v2 WF (folds 0-10) | COMPLETE | IC_1s=0.183 concat, IC_10s=0.075 | CHAMPION MODEL. Do not retrain unless decay analysis says to. |

## 2026-04-29

| Time | Node | Experiment | Status | Key Result | Notes |
|------|------|-----------|--------|------------|-------|
| all day | Razer | PatchTST smart_v3 (folds 14-20) | COMPLETE | 20/20 folds done | Weights available for confluence |
| all day | Neptune | CNN-Mamba v2 WF training | RUNNING | Folds progressing | Continued from Apr 28 |

## PRE 2026-04-29 (summary)

- **Triple Fusion v1** (Neptune): Fold 0 IC_10s=0.085, DA_1s=43.3%. Underperforming. ABANDONED.
- **EventCNN1D** (Neptune): IC_10s=0.132 concat. Superseded by CNN-Mamba v2.
- **EventTransformer original** (Neptune): IC_10s=0.095. Below baseline. NOT USED.
- **PatchTST smart_v3** (Razer): COMPLETE. 20 folds. Used for confluence.
- **LGBM Vol** (Jupiter CPU): COMPLETE. Volatility model for execution features.

---

## MODELS WE HAVE (weights available, DO NOT retrain unless decay says to):
1. CNN-Mamba v2 — CHAMPION (39 OOT folds, weights on Neptune)
2. PatchTST smart_v3 — CONFLUENCE model (20 folds, weights on Razer)
3. LGBM Vol — EXECUTION FEATURE (weights on Jupiter)
4. EventCNN1D — LEGACY (superseded, do not use)

## EXPERIMENTS EXPLICITLY BANNED:
- Event Transformer (any config) — architecture exploration is OVER
- Mamba standalone — architecture exploration is OVER
- CNN1D new variants — architecture exploration is OVER
- ANY new signal model architecture — PHASE IS OVER

---
