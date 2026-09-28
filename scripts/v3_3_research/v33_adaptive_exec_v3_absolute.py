"""
v33_adaptive_exec_v3_absolute.py — HC #375 Track B v3 — Two new adaptive-exit hypotheses.

v1+v2 falsified the Δ-from-entry adaptive-exit on K=2 + universal entries.
v3 tests TWO new hypotheses NOT yet tried:

  HYPOTHESIS A: ABSOLUTE level of pred_log_ret_30s.
    Rationale: pred_log_ret_30s IS the directly-trained directional head with
    IC 0.286 (real signal). If the model's belief in the trade going up has
    dropped to near-zero or negative mid-trade, that's a direct exit signal —
    no need for Δ-from-entry tricks.
    Rule: EXIT when pred_log_ret_30s[entry+s] < θ_abs at first candidate stride.

  HYPOTHESIS B: pred_p_reversal_30s (directly-trained reversal head).
    Rationale: `p_reversal_30s` is a separately trained head predicting whether
    price will reverse direction in next 30s. Never tested as an exit signal.
    Rule: EXIT when pred_p_reversal_30s[entry+s] > θ_rev at first candidate stride.

Held-out TRAIN/TEST split within day. Permutation null on TEST.
HC #376 head-validity assertion.
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path("/home/jupiter/Lvl3Quant")
PRED_NPZ = ROOT / "output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_predictions.npz"
METRICS_JSON = ROOT / "output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_predictions.metrics.json"
OUT_DIR = ROOT / "output/v3_3_full_execution_analysis_20260514/adaptive_exec_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RT_COMMISSION_TICKS = 0.376
STRIDE_PER_SEC = 4
EXIT_HORIZONS_SEC = [1, 5, 10, 30]
EXIT_STRIDES = [h * STRIDE_PER_SEC for h in EXIT_HORIZONS_SEC]
STATIC_BASELINE_SEC = 30
UNIVERSAL_ENTRY_EVERY = 20
TRAIN_FRAC = 0.60

REQUIRED_TRAINED = ["ic_log_ret_30s", "corr_pred_mfe_30s_ticks"]


def validate():
    m = json.load(open(METRICS_JSON))
    for k in REQUIRED_TRAINED:
        if m.get(k) is None:
            raise RuntimeError(f"HC #376: head {k} untrained")
    print(f"[head-audit] PASS", flush=True)


def load():
    npz = np.load(PRED_NPZ, allow_pickle=True)
    n = int(npz["n_samples"])
    n_days = len(npz["oot_dates"])
    spd = n // n_days
    day_idx = np.array([min(i // spd, n_days - 1) for i in range(n)], dtype=np.int32)

    pred_log = npz["pred_log_ret_30s"]
    pred_rev = npz["pred_p_reversal_30s"]
    pred_hit = npz["pred_fifo_tp4sl3_hit_tp"]
    realized = {h: npz[f"target_log_ret_{h}s"] for h in EXIT_HORIZONS_SEC}
    return pred_log, pred_rev, pred_hit, realized, day_idx, n


def make_entries(n, day_idx, max_hold_strides):
    entries = []
    for d in np.unique(day_idx):
        ds = np.where(day_idx == d)[0]
        if len(ds) < max_hold_strides + 1:
            continue
        last = ds[-max_hold_strides - 1]
        e = ds[::UNIVERSAL_ENTRY_EVERY]
        entries.extend(e[e <= last].tolist())
    return np.array(sorted(entries), dtype=np.int64)


def filter_valid(entries, pred_log, pred_rev, pred_hit, realized):
    valid = np.ones(len(entries), dtype=bool)
    for h in EXIT_HORIZONS_SEC:
        valid &= ~np.isnan(realized[h][entries])
    for arr in [pred_log, pred_rev, pred_hit]:
        valid &= ~np.isnan(arr[entries])
    return entries[valid]


def policy_abs(entries, pred_signal, realized, theta, exit_when_below, y_static30):
    """exit_when_below=True for log_ret (exit when pred drops below θ);
       False for p_reversal (exit when pred RISES above θ)."""
    N = len(entries)
    pnl = y_static30.copy()
    decided = np.zeros(N, dtype=bool)
    for hi, s in enumerate(EXIT_STRIDES):
        idx = entries + s
        in_range = idx < len(pred_signal)
        if not in_range.any():
            continue
        valid = in_range & (~decided)
        if not valid.any():
            continue
        sig = pred_signal[idx[valid]]
        if exit_when_below:
            exit_now = sig < theta
        else:
            exit_now = sig > theta
        h_sec = EXIT_HORIZONS_SEC[hi]
        idx_global = np.where(valid)[0][exit_now]
        if h_sec != STATIC_BASELINE_SEC:
            pnl[idx_global] = realized[h_sec][entries[idx_global]] - RT_COMMISSION_TICKS
        decided[idx_global] = True
    return pnl


def sharpe(p):
    if len(p) < 2: return 0.0
    sd = p.std(ddof=1)
    return float(p.mean() / sd * np.sqrt(252.0)) if sd > 0 else 0.0


def test_hypothesis(name, entries_all, signal_arr, exit_when_below,
                    theta_grid, realized, day_idx, train_mask, test_mask):
    train_e = entries_all[train_mask]
    test_e = entries_all[test_mask]
    y_train_st = realized[STATIC_BASELINE_SEC][train_e] - RT_COMMISSION_TICKS
    y_test_st = realized[STATIC_BASELINE_SEC][test_e] - RT_COMMISSION_TICKS

    best_th = None
    best_train_sh = -1e9
    for th in theta_grid:
        pnl = policy_abs(train_e, signal_arr, realized, th, exit_when_below, y_train_st)
        sh = sharpe(pnl)
        if sh > best_train_sh:
            best_train_sh = sh
            best_th = th

    pnl_test = policy_abs(test_e, signal_arr, realized, best_th, exit_when_below, y_test_st)
    test_sh = sharpe(pnl_test)
    static_test_sh = sharpe(y_test_st)
    test_mean = float(pnl_test.mean())
    test_wr = float((pnl_test > 0).mean())
    test_days = day_idx[test_e]
    daily = {int(d): float(pnl_test[test_days == d].sum()) for d in np.unique(test_days)}
    total_abs = sum(abs(v) for v in daily.values())
    max_share = max(abs(v) for v in daily.values()) / total_abs if total_abs > 0 else 0.0

    # Null on TEST: shuffle test_e within day
    rng = np.random.default_rng(42)
    n_perm = 100
    null_sh = []
    pools = {}
    for d in np.unique(test_days):
        ds = np.where(day_idx == d)[0]
        ds = ds[ds + EXIT_STRIDES[-1] < len(signal_arr)]
        v = np.ones(len(ds), dtype=bool)
        for h in EXIT_HORIZONS_SEC:
            v &= ~np.isnan(realized[h][ds])
        v &= ~np.isnan(signal_arr[ds])
        pools[int(d)] = ds[v]
    for pi in range(n_perm):
        shuf = test_e.copy()
        for d in np.unique(test_days):
            mask = (test_days == d)
            pool = pools[int(d)]
            shuf[mask] = rng.choice(pool, size=mask.sum(), replace=False)
        y_shuf = realized[STATIC_BASELINE_SEC][shuf] - RT_COMMISSION_TICKS
        pnl_shuf = policy_abs(shuf, signal_arr, realized, best_th, exit_when_below, y_shuf)
        null_sh.append(sharpe(pnl_shuf))
    null_arr = np.array(null_sh)
    p_val = float((null_arr >= test_sh).mean())

    print(f"\n[{name}] best θ={best_th:.4f}  TEST: Sharpe={test_sh:+.3f}  mean={test_mean:+.4f}t  WR={test_wr:.3f}  day_conc={max_share:.3f}", flush=True)
    print(f"[{name}] static-30s TEST: Sharpe={static_test_sh:+.3f}  ΔSh={test_sh - static_test_sh:+.3f}", flush=True)
    print(f"[{name}] null: mean={null_arr.mean():+.3f}  p95={np.percentile(null_arr,95):+.3f}  max={null_arr.max():+.3f}  p={p_val:.3f}  PASS_p10={'YES' if p_val<0.10 else 'NO'}", flush=True)
    return dict(name=name, best_theta=best_th, train_sharpe=best_train_sh,
                test_sharpe=test_sh, static_test_sharpe=static_test_sh,
                delta_sharpe=test_sh - static_test_sh,
                test_mean=test_mean, test_wr=test_wr, day_conc=max_share,
                null_mean=float(null_arr.mean()), null_p95=float(np.percentile(null_arr,95)),
                p_value=p_val, pass_p10=bool(p_val < 0.10),
                daily_pnl=daily, n_train=int(train_mask.sum()), n_test=int(test_mask.sum()))


def main():
    t0 = time.time()
    validate()
    pred_log, pred_rev, pred_hit, realized, day_idx, n = load()
    entries_all = make_entries(n, day_idx, STATIC_BASELINE_SEC * STRIDE_PER_SEC)
    entries_all = filter_valid(entries_all, pred_log, pred_rev, pred_hit, realized)
    print(f"[entries] {len(entries_all)} universal long entries", flush=True)

    train_mask = np.zeros(len(entries_all), dtype=bool)
    for d in np.unique(day_idx[entries_all]):
        idx_in_day = np.where(day_idx[entries_all] == d)[0]
        cut = int(len(idx_in_day) * TRAIN_FRAC)
        train_mask[idx_in_day[:cut]] = True
    test_mask = ~train_mask

    print(f"\npred_log_ret_30s: mean={pred_log.mean():.4f} p1={np.percentile(pred_log,1):.4f} p99={np.percentile(pred_log,99):.4f}", flush=True)
    print(f"pred_p_reversal_30s: mean={pred_rev.mean():.4f} p1={np.percentile(pred_rev,1):.4f} p99={np.percentile(pred_rev,99):.4f}", flush=True)
    print(f"pred_fifo_tp4sl3_hit_tp: mean={pred_hit.mean():.4f} p1={np.percentile(pred_hit,1):.4f} p99={np.percentile(pred_hit,99):.4f}", flush=True)

    # H-A: pred_log_ret_30s low = exit. θ grid from observed quantiles
    th_grid_log = list(np.linspace(np.percentile(pred_log[~np.isnan(pred_log)], 5),
                                    np.percentile(pred_log[~np.isnan(pred_log)], 50), 8))
    A = test_hypothesis("H-A pred_log_ret_30s ABS exit", entries_all, pred_log, True,
                        th_grid_log, realized, day_idx, train_mask, test_mask)

    # H-B: pred_p_reversal_30s high = exit
    th_grid_rev = list(np.linspace(np.percentile(pred_rev[~np.isnan(pred_rev)], 50),
                                    np.percentile(pred_rev[~np.isnan(pred_rev)], 95), 8))
    B = test_hypothesis("H-B pred_p_reversal_30s exit", entries_all, pred_rev, False,
                        th_grid_rev, realized, day_idx, train_mask, test_mask)

    # H-C: pred_fifo_tp4sl3_hit_tp low = exit (alternative absolute)
    th_grid_hit = list(np.linspace(np.percentile(pred_hit[~np.isnan(pred_hit)], 5),
                                    np.percentile(pred_hit[~np.isnan(pred_hit)], 50), 8))
    C = test_hypothesis("H-C pred_fifo_tp4sl3_hit_tp ABS exit", entries_all, pred_hit, True,
                        th_grid_hit, realized, day_idx, train_mask, test_mask)

    out = dict(elapsed_sec=time.time() - t0, n_entries=int(len(entries_all)),
               hypotheses=[A, B, C])
    out_path = OUT_DIR / "adaptive_exec_v3_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[done] elapsed={out['elapsed_sec']:.1f}s output: {out_path}", flush=True)

    passes = [h for h in [A, B, C] if h["pass_p10"] and h["test_sharpe"] > h["static_test_sharpe"]]
    if passes:
        print(f"[VERDICT] ✓ {len(passes)}/3 hypotheses pass null: {[h['name'] for h in passes]}", flush=True)
    else:
        print(f"[VERDICT] ✗ 0/3 hypotheses pass — Track B adaptive-exit fully falsified on 5-day OOT.", flush=True)


if __name__ == "__main__":
    main()
