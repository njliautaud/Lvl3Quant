# PPO v3 (HC #399 Canonical-Replay Reward) — BUILD REPORT

**Date**: 2026-05-16
**Author**: autonomous build per HC #399 + SESSION_STATE.md PPO v3 design brief
**Status**: BUILD COMPLETE + UNIT TEST PASS — READY FOR HUMAN REVIEW + DISPATCH

---

## 1. Files Created

All on Jupiter (paths absolute):

| File | Lines | Purpose |
|------|------:|---------|
| `/home/jupiter/Lvl3Quant/scripts/rl_v3_3_smart_exec/env_v3_canonical.py` | 650 | Gymnasium env with canonical-replay reward (HC #399 #1) |
| `/home/jupiter/Lvl3Quant/scripts/rl_v3_3_smart_exec/test_env_v3_canonical.py` | 259 | Reward-equivalence unit test (env vs canonical) |
| `/home/jupiter/Lvl3Quant/scripts/rl_v3_3_smart_exec/train_ppo_v3_canonical.py` | 294 | PPO training driver, MLflow + checkpoints |
| `/home/jupiter/Lvl3Quant/output/ppo_v3_BUILD_REPORT.md` | this | This report |

NO existing v2 / v2.1 / v2.2 files were modified (constraint respected).
NO training was launched (constraint respected).

## 2. Unit Test Result — PASS

```
$ python3 scripts/rl_v3_3_smart_exec/test_env_v3_canonical.py
==============================================================================
HC #399 #1 UNIT TEST: env_v3_canonical reward == canonical_reprice
==============================================================================
Env loaded: N=241351, n_days=5, day0 = 20260223 [0, 49585)
Action sequence length: 49585
Action distribution: HOLD=49290, BID=50, ASK=71, MKT_BUY=50, MKT_SELL=49, CANCEL=50, EXIT=25

--- ENV RESULT ---
Sum of step rewards: +84.040000 ticks
n_events (all attempts): 127
n_filled: 85

--- CANONICAL DIRECT RESULT ---
Canonical net_total: +88.040000 ticks
Canonical n_filled: 85
Canonical adv_sel_total: -231.000000 ticks
HC #344 surrogate penalty: -4.000000 ticks
Canonical + penalty: +84.040000 ticks

--- VERDICT ---
|env_reward - (canon + penalty)| = 0.00000000
Tolerance: 0.0001

  ✅ PASS — env reward matches canonical to 0.0001 ticks
  HC #399 #1 binding requirement satisfied.
```

**Diff = 0.00000000 ticks** (well under the 0.0001 tolerance the brief specified).

The test feeds a deterministic mixed action sequence (49,585 steps containing 50 market buys, 49 market sells, 50 passive bids, 71 passive asks, 50 cancels, 25 exits, rest HOLDs) through the env, then re-prices every resulting trade event via the canonical primitives (`_queue_position_model`, `_entry_price_edge_ticks`, `target_log_ret_30s`, `COMMISSION_RT_TICKS`) — identical to what `canonical_reprice()` does in `ppo_v2_1_canonical_replay_eval.py`. Episode-end sum matches byte-for-byte.

## 3. Design Highlights

### State (47 dims)
- 32 v3.3 head outputs (identical PRED_HEADS list as `env_v2.py` — cross-comparable)
- 4 book features: queue_depth_proxy (vol30 surrogate), spread (=1.0 RTH constant), imbalance (p_up_30s - 0.5), vol_30s_pred
- 6 position context: in_pos_flag, side_sign, age_norm, unrealized_norm, cancel_norm, pending_flag
- **4 HC #399 #2a adverse-sel features**: p_reversal_30s, log_ret_5s (predicted), vol_30s (predicted), p_up_30s
- **1 HC #399 #2b queue position estimate**: |imbalance|*2 (observable at decision time)

### Action (Discrete 7)
HOLD, BID, ASK, MKT_BUY, MKT_SELL, CANCEL, **EXIT_POS** (new vs v2; explicit close request)

### Reward (HC #399 #1)
- **Eagerly credited per fill**: at the moment a market or passive trade fills, the env calls `_price_trade()` which uses the canonical primitives to compute the per-trade net ticks (`side * target_log_ret_30s - 0.376 commission`, with `_queue_position_model` filter for passive). That value is the immediate step reward.
- **At episode end**: HC #344 surrogate penalty (-4 ticks if n_fills ≥ 2 and net positive) + HC #399 #2a adverse-selection penalty (disabled by default; hook present).
- **MTM is NOT a reward component** in this env (deliberate departure from v2 which used MTM as proxy reward — that was the v2/v2.1 proxy HC #399 explicitly forbade).
- Per-step rewards are sparse-ish: 0 on HOLD/CANCEL/EXIT steps, non-zero only at fill events and at episode end. This is a legitimate dense decomposition because the SUM is provably canonical (see unit test).

### HC #392 Compliance
- Market order PnL: `side * ret_30s_ticks - 0.376` (commission only). NO extra spread tick added.
- `_price_trade()` uses `edge_offset = 0.0` for market actions (the fill price IS the reference per HC #392).
- HC #392 lint check (`scripts/lint/hc392_check.py`): **clean** on all three new files.

```
$ python3 scripts/lint/hc392_check.py
✅ HC #392 lint check: clean. No forbidden cost-constant violations found.
```

## 4. Wall-Time Estimate for 1M Timesteps on RTX 3070

**Env throughput (Jupiter CPU)**:
- all-HOLD episode: ~34,000 steps/sec
- 5%-active episode (realistic): ~24,000 steps/sec

For 1M total steps with 4 parallel envs, n_steps=2048:
- Env stepping alone: ~40-50s
- PPO update overhead (10 epochs × 2048 batch, 256x256 MLP, RTX 3070): dominant cost
- v2.1 reference wall-time: ~28 min for 1M steps on Razer
- v3 reward computation is slightly heavier per fill (calls `_queue_position_model` per passive trade), BUT trades are infrequent (~100-200 per episode). Marginal slowdown estimated ~10-20%.
- **Estimated wall time: 30-40 min for 1M timesteps on RTX 3070** (env is fast; ckpt+MLflow overhead small).

The build brief warned that canonical rollouts can be 5-20× slower; in practice this env's design (eager per-fill pricing rather than end-of-episode batch replay) keeps the overhead minimal because the canonical primitives are called only on trade events, not on every step. The vast majority of steps are pure-Python no-op HOLD/CANCEL/EXIT branches.

## 5. Exact Bash Command — Launch on Razer

On Razer (Windows, weekend training per HC #396), via SSH:

```bash
ssh claude@razer "cd C:\\Users\\claude\\Lvl3Quant && \
  python scripts\\rl_v3_3_smart_exec\\train_ppo_v3_canonical.py \
    --npz C:\\Users\\claude\\Lvl3Quant\\output\\cnn_mamba_v3_3_uncertainty_weighted\\fold_00_predictions.npz \
    --labels-dir C:\\Users\\claude\\Lvl3Quant\\data\\processed\\mbo_events_smart_v3_fifo_labels \
    --output-dir C:\\Users\\claude\\Lvl3Quant\\output\\rl_v3_3_smart_exec_v3 \
    --total-timesteps 1000000 \
    --n-envs 4 --n-steps 2048 --batch-size 64 \
    --learning-rate 3e-4 --gamma 0.995 --gae-lambda 0.95 --ent-coef 0.01 \
    --seed 0 \
    --checkpoint-every 50000 \
    --mlflow-uri http://jupiter:5000 \
    --mlflow-experiment RL_v3_3_smart_exec_v3_canonical_reward \
    --run-name ppo_v3_canonical_seed0"
```

Or equivalently on Jupiter (Linux) for a faster smoke-test:

```bash
cd /home/jupiter/Lvl3Quant && \
  python3 scripts/rl_v3_3_smart_exec/train_ppo_v3_canonical.py \
    --total-timesteps 50000 --n-envs 2 --seed 0
```

**Before launching on Razer**: must SCP the v3.3 predictions NPZ + FIFO labels to Razer first (or verify they're already there). The new files (env_v3_canonical.py + train_ppo_v3_canonical.py) also need to be SCP'd.

**Resume from checkpoint** (HC #398):
```bash
... --resume-from C:\\Users\\claude\\Lvl3Quant\\output\\rl_v3_3_smart_exec_v3\\ckpt_v3\\ppo_v3_canonical_<N>_steps.zip
```

## 6. Known Limitations / TODOs

### Documented Approximations
1. **MTM as reward = 0**: env reports `unrealized_ticks` for state context but does NOT credit MTM as per-step reward. The canonical primitives book the full +30s exit PnL at fill, so MTM-during-hold would double-count. Consequence: agent gets no fine-grained learning signal during the position-hold phase. **Mitigation**: gamma=0.995 keeps temporal credit-assignment relatively local; per-fill rewards already carry the full canonical info.
2. **Queue position estimate proxy**: env uses `|imbalance|*2` as queue_pos_estimate state feature. We do NOT have raw L2 book depth in the data (the canonical replay itself uses a parametric mean-of-queue overlay calibrated to realized fills — see `full_market_replay._queue_position_model`). This is a faithful mirror of the canonical engine's own honesty about queue position; not a hidden shortcut.
3. **HC #344 surrogate penalty**: single-day episodes have true day_conc=1.0 by definition, so a literal HC #344 gate would fire on every episode. We use a softer surrogate: -4 ticks penalty if n_fills ≥ 2 AND net is positive (a learning signal for the agent to discover non-degenerate, non-overconcentrated policies). Real HC #344 gate is enforced at EVAL time, not training time.
4. **Adverse-sel aggregate penalty**: disabled by default (`ADVERSE_SEL_PENALTY_MULT = 0.0`). Canonical per-trade PnL already includes realized 30s adverse-selection cost (when negative, lr is the loss). The aggregate penalty is a separate hook for future "stacked filter" learning per HC #399 #2a.
5. **PASSIVE_FILL_WINDOW = 50 evals**: matches the cancel window per HC #321 (~12.5s). A passive order that doesn't fill within this window is auto-cancelled (recorded as filled=False event).
6. **Episode = single trading day**: Sufficient for training, but a true HC #344 day-concentration gate requires multi-day evaluation. The companion eval script (TODO: ppo_v3_canonical_replay_eval.py — to be built post-training, mirroring `ppo_v2_1_canonical_replay_eval.py`) will compute true HC #344 across the held-out day set.

### Post-training TODOs (NOT part of this build)
- [ ] Build `ppo_v3_canonical_replay_eval.py` (clone of v2.1 eval script, swap env import). Required to produce the HC #397B-compliant verdict CSV with `adv_sel_30s_avg`, `avg_queue_pos`, `cancel_window`, `day_conc`, `pass_hc344` columns.
- [ ] Sync v3.3 predictions NPZ + FIFO labels to Razer before launch (if not already present).
- [ ] SCP the 3 new Python files to Razer.
- [ ] Add training run to RUN_HISTORY.md when launched.
- [ ] Once training completes and eval runs, post head-to-head comparison vs v2.1 canonical (+0.300 t/fill) and rules baseline.

### What Could Bite During Training
- **Sparse rewards**: most env steps return 0 reward. Combined with gamma=0.995 over RTH episodes (~23,400 steps/day), value learning may be slow. If training plateaus with degenerate all-HOLD policy, dense-reward variants worth trying:
  - **Variant A**: distribute episode-end aggregate evenly across all steps (constant shaping) — preserves SUM equality.
  - **Variant B**: credit canonical-replay re-pricing of pending passive orders at each step (eagerly), not just at PASSIVE_FILL_WINDOW expiry — would require additional event tracking.
- **Single-NPZ overfitting**: training reuses the same 5 days of predictions repeatedly. Once v3.3 60d champion's chunk1 extended-OOT inference NPZ lands (Neptune PID 418732 per SESSION_STATE.md), expand training data.

## 7. HC Compliance Summary

| HC | Requirement | Status |
|----|-------------|--------|
| HC #392 | commission only (0.376), no extra spread on market | ✅ verified — `_price_trade()` uses edge=0 for market; lint check clean |
| HC #397 | every number from canonical primitives | ✅ env reward uses canonical primitives only |
| HC #397B | full FIFO + adverse-sel + cancellation | ✅ `_queue_position_model` (FIFO+cancel) + adv_sel_30s tracked |
| HC #398 | batch-level checkpointing | ✅ CheckpointCallback every 50k steps |
| HC #399 #1 | canonical-replay reward (NOT proxy) | ✅ verified by unit test — diff=0.00000000 |
| HC #399 #2a | adverse-sel features in state | ✅ 4 features added to state vector |
| HC #399 #2b | queue position in state | ✅ queue_pos_estimate added |

## 8. Handoff Note

The build brief specified: "When done, hand off a launch-ready package (env wrapper + training script + unit-test pass log) for human review."

That is what this report represents. Three files. Unit test passes with zero numerical gap. HC #392 lint clean. Smoke test confirms the training script runs PPO end-to-end on the new env (200 timesteps, no crash, model.learn returns).

**Recommended next step (human approval)**: SCP files + data to Razer (or run on Jupiter for a faster CPU smoke-test of, say, 100k timesteps first to surface any training-dynamics issues before committing 24h on Razer). Add to RUN_HISTORY.md upon launch.
