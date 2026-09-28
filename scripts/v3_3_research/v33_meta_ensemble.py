#!/usr/bin/env python3
"""
HC #363 deliverable 4: v3.3 meta-MLP + contextual bandit on 32 head outputs.

Tests two ensembling strategies on the v3.3 head bank:

1. **META-MLP / RIDGE** — supervised second-stage regression. Inputs = 32
   head predictions (z-scored from train block). Target = `target_fifo_tp4sl3_net`
   (passive FIFO net, in ticks). Compares Ridge (linear) vs MLP (32→32→1).
   Chronological 60/20/20 train/val/test split — no per-sample dates in NPZ,
   so we approximate via row order (predictions saved in event-time order).

2. **LinUCB CONTEXTUAL BANDIT** — arms = top-K solo heads (by train-block
   Sharpe). Context = z-scored head vector. At each event, the bandit picks
   which head to "trust" (LinUCB) and earns the realized FIFO net if that
   head was in its SHORT band. Trained on first 80%, tested on last 20%.

All operations on FIFO-fillable samples only (mask present). Commission ticks
= 0.376 (ES round-trip). SHORT side only (LONG was untradeable per HC #363
deliverable 3).

Outputs:
  output/v3_3_full_execution_analysis_20260514/meta_ensemble/
    meta_summary.md
    meta_results.json
    bandit_arm_pulls.csv

Per HC #307D — NEW analysis script.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/jupiter/Lvl3Quant")
PRED_NPZ = ROOT / "output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_predictions.npz"
OUT_DIR = ROOT / "output/v3_3_full_execution_analysis_20260514/meta_ensemble"
COMMISSION_TICKS = 0.376
SHORT_HIGH_HEADS = {"p_reversal_15s", "p_reversal_30s", "p_reversal_60s"}
SHORT_TOP_BAND = 0.01  # Top 1%
K_BANDIT_ARMS = 8
RANDOM_SEED = 42


def safe_zscore(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0, ddof=1)
    sd[sd < 1e-9] = 1.0
    Z = (X - mu) / sd
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    return Z, mu, sd


def short_signal(pred: np.ndarray, head: str) -> np.ndarray:
    """Per-head 'larger means more SHORT-like' scalar. Flip if needed."""
    return pred if head in SHORT_HIGH_HEADS else -pred


def sharpe_net(r: np.ndarray) -> tuple[float, float, int]:
    if r.size < 2:
        return (0.0, 0.0, int(r.size))
    mu = float(np.mean(r))
    sd = float(np.std(r, ddof=1))
    sh = mu / sd if sd > 1e-9 else 0.0
    return (sh, mu, int(r.size))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not PRED_NPZ.exists():
        print(f"ERR: {PRED_NPZ} missing", file=sys.stderr)
        return 1

    np.random.seed(RANDOM_SEED)
    print(f"Loading {PRED_NPZ}")
    npz = np.load(PRED_NPZ, allow_pickle=True)
    keys = list(npz.keys())
    pred_keys = sorted([k for k in keys if k.startswith("pred_") and k != "pred_meta"])
    heads = [k[len("pred_"):] for k in pred_keys]

    target = npz["target_fifo_tp4sl3_net"].astype(np.float64)
    fmask_raw = npz.get("mask_fifo_tp4sl3_net")
    fmask = fmask_raw.astype(bool) if fmask_raw is not None else ~np.isnan(target)

    # Restrict to fillable samples
    fill_idx = np.where(fmask)[0]
    n_total = fill_idx.size
    print(f"FIFO-fillable samples: {n_total:,} of {target.size:,}")

    X = np.column_stack([npz[f"pred_{h}"].astype(np.float64) for h in heads])
    X = X[fill_idx]  # (N, 32)
    y_raw = target[fill_idx]  # passive long P&L net (in ticks, w/o commission)

    # SHORT side P&L = -y_raw - commission. We target SHORT-side because LONG is dead.
    y_short = -y_raw - COMMISSION_TICKS

    # Chronological splits
    n = X.shape[0]
    n_tr = int(0.60 * n)
    n_val = int(0.20 * n)
    idx_tr = np.arange(0, n_tr)
    idx_val = np.arange(n_tr, n_tr + n_val)
    idx_te = np.arange(n_tr + n_val, n)
    print(f"Train={idx_tr.size:,}  Val={idx_val.size:,}  Test={idx_te.size:,}")

    # Flip per-head sign so that "high feature value = SHORT signal" (positive corr with y_short on train)
    X_signed = np.empty_like(X)
    head_signs = []
    for j, h in enumerate(heads):
        x = X[:, j]
        # default: short signal direction
        x_dir = short_signal(x, h)
        # Verify direction on TRAIN block
        corr_tr = float(np.corrcoef(x_dir[idx_tr], y_short[idx_tr])[0, 1])
        if not np.isfinite(corr_tr):
            corr_tr = 0.0
        if corr_tr < 0:
            x_dir = -x_dir
            sign = -1
        else:
            sign = 1
        X_signed[:, j] = x_dir
        head_signs.append({"head": h, "sign": sign, "train_corr_signed": abs(corr_tr)})

    # Z-score using train block only
    Z, mu, sd = safe_zscore(X_signed[idx_tr])
    Z_full = np.nan_to_num((X_signed - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)

    # --- 1. RIDGE BASELINE ---
    from sklearn.linear_model import Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(Z_full[idx_tr], y_short[idx_tr])
    pred_ridge_val = ridge.predict(Z_full[idx_val])
    pred_ridge_te = ridge.predict(Z_full[idx_te])

    # --- 2. META MLP ---
    from sklearn.neural_network import MLPRegressor
    mlp = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=200, learning_rate_init=1e-3,
                       random_state=RANDOM_SEED, early_stopping=True,
                       validation_fraction=0.15, n_iter_no_change=10)
    mlp.fit(Z_full[idx_tr], y_short[idx_tr])
    pred_mlp_val = mlp.predict(Z_full[idx_val])
    pred_mlp_te = mlp.predict(Z_full[idx_te])

    # Evaluation helper: pick top-band by predicted score, compute Sharpe on realized
    def eval_top_band(pred_score: np.ndarray, y_realized: np.ndarray, band_frac: float) -> dict:
        n_pick = max(int(band_frac * pred_score.size), 1)
        order = np.argsort(-pred_score)
        picked = order[:n_pick]
        r = y_realized[picked]
        sh, mu, k = sharpe_net(r)
        return {"n_picked": k, "sharpe": sh, "net_t_per_fill": mu, "band_frac": band_frac}

    results = {"head_signs": head_signs, "ridge": {}, "mlp": {}, "solo_best": {}, "bandit": {}}
    for tag, pred_v, pred_t in [("ridge", pred_ridge_val, pred_ridge_te),
                                 ("mlp", pred_mlp_val, pred_mlp_te)]:
        results[tag]["val_top1pct"] = eval_top_band(pred_v, y_short[idx_val], 0.01)
        results[tag]["val_top5pct"] = eval_top_band(pred_v, y_short[idx_val], 0.05)
        results[tag]["test_top1pct"] = eval_top_band(pred_t, y_short[idx_te], 0.01)
        results[tag]["test_top5pct"] = eval_top_band(pred_t, y_short[idx_te], 0.05)
        results[tag]["test_top10pct"] = eval_top_band(pred_t, y_short[idx_te], 0.10)

    # Solo best head on test block: pick the head with highest signed-score on test top 1% by its own ranking
    best_solo_sh = -1e9
    best_solo_head = None
    for j, h in enumerate(heads):
        score_tr = X_signed[idx_tr, j]
        # Use TRAIN-block-fitted z-score
        z_te = Z_full[idx_te, j]
        cell_top1 = eval_top_band(z_te, y_short[idx_te], 0.01)
        if cell_top1["sharpe"] > best_solo_sh and cell_top1["n_picked"] >= 10:
            best_solo_sh = cell_top1["sharpe"]
            best_solo_head = h
            results["solo_best"]["test_top1pct"] = cell_top1
            results["solo_best"]["head"] = h

    # --- 3. LINUCB BANDIT ---
    # Arms: top-K heads by absolute train-corr (=most useful signal direction)
    head_score = sorted(head_signs, key=lambda d: -d["train_corr_signed"])
    arm_heads = [d["head"] for d in head_score[:K_BANDIT_ARMS]]
    arm_indices = [heads.index(h) for h in arm_heads]
    print(f"Bandit arms (top-{K_BANDIT_ARMS} by |train corr|): {arm_heads}")

    # LinUCB per-arm parameters
    d = Z_full.shape[1]
    A = [np.eye(d) for _ in arm_heads]
    b = [np.zeros(d) for _ in arm_heads]
    alpha_ucb = 1.0
    pulls_log = []  # list of (idx, arm, reward, ucb_at_pick)
    cum_rewards = np.zeros(len(arm_heads))
    cum_pulls = np.zeros(len(arm_heads), dtype=int)

    # Train block: random exploration (online learning)
    train_perm = idx_tr.copy()  # already chronological
    test_block = idx_te

    # Train phase — fit LinUCB online
    for t, i in enumerate(train_perm):
        x = Z_full[i].reshape(-1, 1)
        ucbs = []
        for a in range(len(arm_heads)):
            A_inv = np.linalg.solve(A[a], np.eye(d))
            theta = A_inv @ b[a]
            mu_hat = float((theta.T @ x).item())
            sigma = float(np.sqrt((x.T @ A_inv @ x).item()))
            ucbs.append(mu_hat + alpha_ucb * sigma)
        # Epsilon exploration during train to ensure all arms get pulled
        if t < 200 or np.random.random() < 0.1:
            arm = t % len(arm_heads)
        else:
            arm = int(np.argmax(ucbs))
        # Reward: realized SHORT pnl gating on whether this arm's signed score is in its Top 1%
        # Simpler: reward = signed pnl gated by "head fires SHORT this event"
        score_arm = X_signed[i, arm_indices[arm]]
        # Train top1pct threshold for arm
        thr = np.quantile(X_signed[idx_tr, arm_indices[arm]], 1 - SHORT_TOP_BAND)
        fires = score_arm >= thr
        reward = float(y_short[i]) if fires else 0.0  # if arm doesn't fire, no trade, no reward
        # LinUCB update
        A[arm] += x @ x.T
        b[arm] += reward * x.flatten()
        cum_rewards[arm] += reward
        cum_pulls[arm] += 1

    # Test phase — exploit only
    test_pnl = []
    test_picks = []
    for t, i in enumerate(test_block):
        x = Z_full[i].reshape(-1, 1)
        ucbs = []
        for a in range(len(arm_heads)):
            A_inv = np.linalg.solve(A[a], np.eye(d))
            theta = A_inv @ b[a]
            ucbs.append(float((theta.T @ x).item()))
        arm = int(np.argmax(ucbs))
        score_arm = X_signed[i, arm_indices[arm]]
        thr_tr = np.quantile(X_signed[idx_tr, arm_indices[arm]], 1 - SHORT_TOP_BAND)
        fires = score_arm >= thr_tr
        if fires:
            r = float(y_short[i])
            test_pnl.append(r)
            test_picks.append({"arm": arm, "head": arm_heads[arm], "pnl": r})

    if test_pnl:
        arr = np.array(test_pnl)
        sh, mu, kp = sharpe_net(arr)
        results["bandit"] = {
            "arms": arm_heads,
            "test_n_trades": int(kp),
            "test_total_events": int(test_block.size),
            "fire_rate": float(kp / test_block.size),
            "test_sharpe": sh,
            "test_net_t_per_fill": mu,
            "train_arm_pull_distribution": {arm_heads[a]: int(cum_pulls[a]) for a in range(len(arm_heads))},
        }
    else:
        results["bandit"] = {"test_n_trades": 0, "test_sharpe": 0.0, "test_net_t_per_fill": 0.0,
                              "fire_rate": 0.0, "arms": arm_heads}

    # Write outputs
    json_path = OUT_DIR / "meta_results.json"
    json_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"Wrote {json_path}")

    lines = []
    lines.append("# v3.3 META-ENSEMBLE — HC #363 deliverable 4\n")
    lines.append(f"Source: `{PRED_NPZ.name}` (FIFO-fillable {n_total:,} samples, chronological 60/20/20 split).")
    lines.append(f"Target: passive SHORT FIFO net P&L (target_fifo_tp4sl3_net × -1 − {COMMISSION_TICKS} commission).")
    lines.append(f"Train block z-score + sign-fit per head (correlation-with-target on train).\n")

    def fmt_block(name: str, r: dict) -> list[str]:
        out = []
        out.append(f"## {name}\n")
        out.append("| Block | Band | n_picked | Sharpe | Net t/fill |")
        out.append("|---|---|---|---|---|")
        for k, key in [("Val Top1%", "val_top1pct"), ("Val Top5%", "val_top5pct"),
                        ("Test Top1%", "test_top1pct"), ("Test Top5%", "test_top5pct"),
                        ("Test Top10%", "test_top10pct")]:
            v = r.get(key)
            if v:
                out.append(f"| {k} | {v['band_frac']*100:.0f}% | {v['n_picked']} | {v['sharpe']:.3f} | {v['net_t_per_fill']:.3f} |")
        return out

    lines.extend(fmt_block("Ridge baseline", results["ridge"]))
    lines.append("")
    lines.extend(fmt_block("Meta-MLP (32→32→16→1, MLPRegressor)", results["mlp"]))
    lines.append("")
    lines.append("## Solo-best head reference (test block)\n")
    sb = results["solo_best"]
    if sb.get("head"):
        v = sb["test_top1pct"]
        lines.append(f"- Best solo head on test block (n>=10, top1%): **{sb['head']}** — Sharpe {v['sharpe']:.3f}, net {v['net_t_per_fill']:.3f} ticks, n={v['n_picked']}")
    lines.append("")
    lines.append(f"## LinUCB bandit (top-{K_BANDIT_ARMS} arms by |train corr|)\n")
    b_ = results["bandit"]
    lines.append(f"- Arms: {', '.join(b_.get('arms', []))}")
    lines.append(f"- Test events: {b_['test_total_events']:,}  Fired (traded): {b_['test_n_trades']}  Fire rate: {b_['fire_rate']*100:.2f}%")
    lines.append(f"- **Test Sharpe: {b_['test_sharpe']:.3f}  Net t/fill: {b_['test_net_t_per_fill']:.3f}**")
    if b_.get('train_arm_pull_distribution'):
        lines.append("- Train arm-pull distribution:")
        for arm, n_p in b_['train_arm_pull_distribution'].items():
            lines.append(f"  - {arm}: {n_p}")
    lines.append("")
    lines.append("## Per-head sign + train correlation (z-scored signed feature vs SHORT target)\n")
    lines.append("| Head | Sign | |Train corr| |")
    lines.append("|---|---|---|")
    for d in sorted(head_signs, key=lambda r: -r["train_corr_signed"])[:20]:
        lines.append(f"| {d['head']} | {'+' if d['sign']>0 else '−'} | {d['train_corr_signed']:.4f} |")

    md_path = OUT_DIR / "meta_summary.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {md_path}")

    # Bandit pulls log
    import csv
    bp_path = OUT_DIR / "bandit_arm_pulls.csv"
    with open(bp_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["test_event_idx", "arm", "head", "pnl"])
        w.writeheader()
        for k, p in enumerate(test_picks):
            w.writerow({"test_event_idx": k, "arm": p["arm"], "head": p["head"], "pnl": f"{p['pnl']:.4f}"})
    print(f"Wrote {bp_path}")

    # Print summary to stdout
    print("\n=== SUMMARY ===")
    print(f"Ridge test top1%: Sharpe={results['ridge']['test_top1pct']['sharpe']:.3f} net={results['ridge']['test_top1pct']['net_t_per_fill']:.3f} n={results['ridge']['test_top1pct']['n_picked']}")
    print(f"Meta-MLP test top1%: Sharpe={results['mlp']['test_top1pct']['sharpe']:.3f} net={results['mlp']['test_top1pct']['net_t_per_fill']:.3f} n={results['mlp']['test_top1pct']['n_picked']}")
    if sb.get("head"):
        print(f"Solo best ({sb['head']}) test top1%: Sharpe={sb['test_top1pct']['sharpe']:.3f} net={sb['test_top1pct']['net_t_per_fill']:.3f}")
    print(f"LinUCB: n_trades={b_['test_n_trades']}, Sharpe={b_['test_sharpe']:.3f}, net={b_['test_net_t_per_fill']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
