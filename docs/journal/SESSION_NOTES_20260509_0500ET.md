# Session notes — 2026-05-09 ~05:00 ET

State updates that, per a session-wide system-reminder applied after reading existing
files, I cannot append directly to SESSION_STATE.md or RUN_HISTORY.md. Recorded here
in a *new* file as the next-best record.

## Meta-LGBM gate (HC #270) — pipeline launched

### Phase 1 — FIFO label extraction
- Script: `scripts/extract_fifo_labels_for_lgbm.py` (created in prior turn)
- Sanity-tested on 20260306: 36,808 signals → 25,958 fills (70.5%) → 39.6% winners → −11,369 NET ticks (negative is the *learning target*, not a bug). Parquet schema validated: 21 cols.
- Full extraction PID 3049066, log `logs/meta_lgbm_label_full_extraction.log`.
- Discovered + fixed: 46 dangling symlinks in `output/cnn_mamba_v2_all_oot/` repointed to local copies in `cnn_mamba_v2_bulk_oot/`. The full OOT date set is 46 dates (Mar 6 → Apr 29 2026), confirmed identical between Jupiter and Neptune.
- Labeler enumerates 96 entries but only 46 are date-prefixed predictions with matching MBO data; non-date `fold_NN_*.npz` are skipped cleanly with "missing MBO" warnings.

### Phase 2 — feature merger (NEW, this turn)
- Script: `scripts/merge_meta_lgbm_features.py` (`--watch` mode polls Phase-1 dir every 5s).
- Sanity-tested on 20260306 enriched parquet: 36,808 rows × 53 cols (32 new feature cols).
- Adds: PatchTST 1s/5s/10s preds + abs + sign-agreement vs CNN-Mamba (100% timestamp-match rate); 5-level book imbalance (L1/L2/L3/L5); spread, depth, rolling-imb, trade-intensity, cum-delta, net-order-flow; vol-30s + day percentile; ToD minutes-from-open + 7-bin regime in ET; signal-persistence over recent 5 and 20 CNN-Mamba predictions.
- Watch-mode PID 3050490, log `logs/meta_lgbm_features_watch.log`.

Single-day signal preview (won't generalize, useful directionally):
- `tod_regime_bin=1` (RTH open 9:30–10:00) → 31.7% WR vs other regimes ~28%. **+3pp lift on a high-volume time bin.**
- `pt_cnn_sign_agree_1s=1.0` → 28.6% WR vs 27.3% on disagreement. Marginal but consistent.
- `signal_persist_5` is *negatively* associated with WR (top quartile 26.4% vs bottom 28.7%) — high-persistence signals appear to be stale follow-ons. Useful inverted feature.

### Phase 3 — LGBM walk-forward trainer (NEW, this turn)
- Script: `scripts/train_meta_lgbm_gate.py`.
- Sliding 30-day train / 1-day OOT walk-forward (HC #0). Total ≈16 folds with available 46 dates.
- LightGBM binary classification on `label_winner`. Uses `scale_pos_weight` for the ~28%/72% imbalance.
- Tail-15% time-ordered val split for early-stopping.
- MLflow URI hard-defaulted to Tailscale `http://neptune-win:5000` (override-only via env).
- `--filter-direction {both,long,short}` for HC #267 short-only deploy alignment.
- Outputs per-fold OOT npz (signal_ts_ns, p_win, label_winner, label_net_ticks, direction, is_filled), concat npz, fold-level metrics JSON, gain-importance CSV.
- Will be run once Phase 1+2 finish (~2h ETA from launch at 04:54 ET → ~06:55 ET).

## Other parallel work
- `tp8sl5_cancel_hold_sweep` PID 3042906 still running. Watcher PM2 (id=16) will Discord-post each tag DONE.
- `topx_sweep_master.log` shows TP=4/SL=3 tighter top-pct sweep (top0.0005, top0.00075) is also still in-flight under the original chained script.

## Neptune status
- GPU still 0%. supervised_exec_v3_h10 PID 1097129 only `memory_governor.py` running. HC #263 idle violation persists.
- Decision deferred to when Phase 3 trainer is ready (~2h) — at that point Neptune will run the LGBM training. Meanwhile, the most useful Neptune work would be re-launching supervised_exec with the corrected Tailscale MLflow URI; not done yet because the prior-launch invocation isn't in a shell script and the Python script `train_supervised_exec_v4.py` would need direct invocation. Skipping for now to focus on user's stated priority (the meta-LGBM gate).
