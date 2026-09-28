"""
HC #315 — Smoothing test as EXIT signal (NOT entry).

User clarification: smoothing belongs on the exit side, not entry. Given a position
opened at time t at confidence band X, does watching the rolling-avg of subsequent
predictions tell us a better exit time than fixed-horizon?

Methodology:
  Entry: top-1% |pred_log_ret_1s| at time t in direction sign(pred[t])
  Realized: walk forward through prediction stream up to MAX_HOLD_STEPS
  Exit policies tested per trade:
    A. Fixed horizon (control): exit at t+H  for H ∈ {1s=4steps, 5s=20, 10s=40, 30s=120}
    B. Smoothed disagreement exit: exit when rolling_mean(pred[t+1..t+k]) over W steps
       has SIGN OPPOSITE to trade direction (and magnitude > eps), or hits MAX_HOLD.
       W ∈ {4, 10, 20, 40}.
    C. Reversal-head exit: exit when p_reversal_15s[t+k] > 0.5 (or higher thresholds)
    D. Combined: B AND C signals.

For each exit, P&L = sign(entry_pred) * realized_log_ret_at_exit_horizon — cost
where realized at exit step k uses the closest available horizon target:
  - k <= 4 steps  → use target_log_ret_1s at trade time t (1s = 4 steps @ 250ms)
  - 4 < k <= 20   → use target_log_ret_5s at t  (5s forward)
  - 20 < k <= 40  → use target_log_ret_10s at t
  - 40 < k <= 120 → use target_log_ret_30s at t

This is an approximation — true exit P&L would be log_ret from t to t+k. But targets
are saved only at the 4 canonical horizons, so we use closest-bucket as approx.

OUT: output/v3_2_deep_sim_20260512/exit_smoothing_test.json
"""
from __future__ import annotations
import json, csv
import numpy as np
from pathlib import Path

PREDS = Path("/home/jupiter/Lvl3Quant/output/v3_2_deep_sim_20260512/fold_00_oot_predictions.npz")
OUT_JSON = Path("/home/jupiter/Lvl3Quant/output/v3_2_deep_sim_20260512/exit_smoothing_test.json")
OUT_CSV = Path("/home/jupiter/Lvl3Quant/output/v3_2_deep_sim_20260512/exit_smoothing_test.csv")

COST = 0.376  # passive RT
ANN = 252.0
MIN_GAP_STEPS = 4
ES_TICK_USD = 12.50

# Step-to-horizon mapping for P&L approximation
STEP_TO_H = [
    (4,  "1s"),
    (20, "5s"),
    (40, "10s"),
    (120, "30s"),
]


def perf(pnl):
    n = len(pnl)
    if n < 5:
        return {"n_trades": n, "mean_ticks": float("nan"), "sharpe": None,
                "sortino": None, "pf": None, "wr_pct": float("nan"), "total_usd": 0.0}
    mean = float(pnl.mean()); std = float(pnl.std(ddof=1))
    sharpe = (mean / std * np.sqrt(ANN)) if std > 1e-9 else None
    neg = pnl[pnl < 0]
    dn = float(neg.std(ddof=1)) if len(neg) > 1 else 0.0
    sortino = (mean / dn * np.sqrt(ANN)) if dn > 1e-9 else None
    gw = float(pnl[pnl > 0].sum()); gl = -float(pnl[pnl < 0].sum())
    pf = (gw / gl) if gl > 1e-9 else None
    wr = float((pnl > 0).mean() * 100.0)
    return {
        "n_trades": n,
        "mean_ticks": mean,
        "median_ticks": float(np.median(pnl)),
        "sharpe": float(sharpe) if sharpe and np.isfinite(sharpe) else None,
        "sortino": float(sortino) if sortino and np.isfinite(sortino) else None,
        "pf": float(pf) if pf and np.isfinite(pf) else None,
        "wr_pct": wr,
        "total_ticks": float(pnl.sum()),
        "total_usd": float(pnl.sum() * ES_TICK_USD),
        "max_win": float(pnl.max()), "max_loss": float(pnl.min()),
    }


def pnl_at_step(direction, k, targets_at_t):
    """Approx P&L at exit step k by selecting nearest-horizon target."""
    for thr, h in STEP_TO_H:
        if k <= thr:
            return direction * targets_at_t[h]
    return direction * targets_at_t["30s"]


def main():
    d = np.load(PREDS, allow_pickle=True)
    n = int(d["n_samples"])

    pred = d["pred_log_ret_1s"][:n]
    t_targets = {
        "1s":  d["target_log_ret_1s"][:n],
        "5s":  d["target_log_ret_5s"][:n],
        "10s": d["target_log_ret_10s"][:n],
        "30s": d["target_log_ret_30s"][:n],
    }
    p_rev = d["pred_p_reversal_15s"][:n]
    abs_pred = np.abs(pred)
    valid = d["mask_log_ret_1s"][:n].astype(bool) & np.isfinite(pred) & np.isfinite(t_targets["1s"])

    # Entry signal: top-1% |pred_log_ret_1s|
    thr_top1 = float(np.quantile(abs_pred[valid], 0.99))
    entry_mask = valid & (abs_pred >= thr_top1)
    entry_idx_all = np.where(entry_mask)[0]
    # Anti-churn
    taken = []; last = -10**9
    for i in entry_idx_all:
        if i - last < MIN_GAP_STEPS: continue
        taken.append(i); last = i
    entry_idx = np.array(taken, dtype=np.int64)
    print(f"Entry signals (top-1%, anti-churn): {len(entry_idx):,}")

    direction = np.sign(pred[entry_idx])
    findings = {"setup": {
        "n_predictions": n,
        "entry_count": int(len(entry_idx)),
        "entry_thr_top1pct": thr_top1,
        "cost_passive_rt": COST,
        "step_to_horizon_map": STEP_TO_H,
    }, "policies": {}}

    rows = []

    # ============== Policy A: Fixed horizon ==============
    for k_steps, h_label in [(4, "1s"), (20, "5s"), (40, "10s"), (120, "30s")]:
        # P&L at fixed step k = sign × target_{h_label}
        pnl_list = []
        for ix, dr in zip(entry_idx, direction):
            t = t_targets[h_label][ix]
            if not np.isfinite(t): continue
            pnl_list.append(dr * t - COST)
        pnl = np.array(pnl_list)
        s = perf(pnl)
        s_key = f"A_fixedH{h_label}"
        findings["policies"][s_key] = s
        rows.append({"policy": s_key, **s})

    # ============== Policy B: Smoothed-disagreement exit ==============
    # Pre-compute rolling means for each W
    # Use a forward-running cumulative sum trick on pred (signed)
    cumP = np.cumsum(np.where(np.isfinite(pred), pred, 0.0))
    cumN = np.cumsum(np.isfinite(pred).astype(np.int64))

    def rolling_mean(a, k, W):
        """mean of pred[a..k] where length up to W back from k. Returns mean over min(k-a+1, W) valid samples ending at k."""
        # mean of pred[k-W+1..k] if valid
        lo = max(0, k - W + 1)
        hi = k + 1
        nv = int(cumN[hi-1] - (cumN[lo-1] if lo > 0 else 0))
        if nv < max(1, W // 2):
            return float("nan")
        sv = float(cumP[hi-1] - (cumP[lo-1] if lo > 0 else 0))
        return sv / nv

    for W in [4, 10, 20, 40]:
        for MAX_HOLD, h_max_label in [(40, "10s"), (120, "30s")]:
            pnl_list = []
            holds = []
            for ix, dr in zip(entry_idx, direction):
                # Walk forward step-by-step from MIN_HOLD=W to MAX_HOLD
                exit_step = MAX_HOLD
                for k in range(W, MAX_HOLD + 1):
                    pos = ix + k
                    if pos >= n: break
                    rm = rolling_mean(pred, max(0, pos - W + 1), pos)
                    if not np.isfinite(rm): continue
                    # Exit if rolling mean has opposite sign of direction
                    if (dr > 0 and rm < 0) or (dr < 0 and rm > 0):
                        exit_step = k
                        break
                pnl = pnl_at_step(dr, exit_step, {h: t_targets[h][ix] for h in t_targets})
                if not np.isfinite(pnl): continue
                pnl_list.append(pnl - COST)
                holds.append(exit_step)
            pnl_arr = np.array(pnl_list)
            s = perf(pnl_arr)
            s["mean_hold_steps"] = float(np.mean(holds)) if holds else None
            s_key = f"B_smoothW{W}_maxH{h_max_label}"
            findings["policies"][s_key] = s
            rows.append({"policy": s_key, **s})

    # ============== Policy C: Reversal-head exit ==============
    for rev_thr in [0.3, 0.5, 0.7]:
        for MAX_HOLD, h_max_label in [(40, "10s"), (120, "30s")]:
            pnl_list = []; holds = []
            for ix, dr in zip(entry_idx, direction):
                exit_step = MAX_HOLD
                for k in range(1, MAX_HOLD + 1):
                    pos = ix + k
                    if pos >= n: break
                    pr = p_rev[pos] if np.isfinite(p_rev[pos]) else 0.0
                    if pr >= rev_thr:
                        exit_step = k; break
                pnl = pnl_at_step(dr, exit_step, {h: t_targets[h][ix] for h in t_targets})
                if not np.isfinite(pnl): continue
                pnl_list.append(pnl - COST)
                holds.append(exit_step)
            pnl_arr = np.array(pnl_list)
            s = perf(pnl_arr)
            s["mean_hold_steps"] = float(np.mean(holds)) if holds else None
            s_key = f"C_revThr{rev_thr}_maxH{h_max_label}"
            findings["policies"][s_key] = s
            rows.append({"policy": s_key, **s})

    # ============== Policy D: Combined (smoothing OR reversal head) ==============
    for W, rev_thr in [(20, 0.5), (10, 0.5), (40, 0.5)]:
        for MAX_HOLD, h_max_label in [(40, "10s"), (120, "30s")]:
            pnl_list = []; holds = []
            for ix, dr in zip(entry_idx, direction):
                exit_step = MAX_HOLD
                for k in range(W, MAX_HOLD + 1):
                    pos = ix + k
                    if pos >= n: break
                    # Smoothing exit
                    rm = rolling_mean(pred, max(0, pos - W + 1), pos)
                    rev_signal = (np.isfinite(p_rev[pos]) and p_rev[pos] >= rev_thr)
                    smooth_signal = (np.isfinite(rm)
                                     and ((dr > 0 and rm < 0) or (dr < 0 and rm > 0)))
                    if smooth_signal or rev_signal:
                        exit_step = k; break
                pnl = pnl_at_step(dr, exit_step, {h: t_targets[h][ix] for h in t_targets})
                if not np.isfinite(pnl): continue
                pnl_list.append(pnl - COST)
                holds.append(exit_step)
            pnl_arr = np.array(pnl_list)
            s = perf(pnl_arr)
            s["mean_hold_steps"] = float(np.mean(holds)) if holds else None
            s_key = f"D_W{W}_rev{rev_thr}_maxH{h_max_label}"
            findings["policies"][s_key] = s
            rows.append({"policy": s_key, **s})

    with open(OUT_JSON, "w") as f:
        json.dump(findings, f, indent=2,
                  default=lambda x: None if (isinstance(x, float) and not np.isfinite(x)) else x)

    fields = ["policy", "n_trades", "mean_ticks", "median_ticks",
              "sharpe", "sortino", "pf", "wr_pct", "total_ticks", "total_usd",
              "mean_hold_steps", "max_win", "max_loss"]
    rows.sort(key=lambda r: -(r.get("sharpe") or -1e9))
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows: w.writerow({k: r.get(k) for k in fields})

    print("\nALL EXIT POLICIES (Sharpe-ranked):")
    print(f"{'rank':>4}  {'policy':<32} {'n':>6} {'mean':>6} {'WR%':>5} {'PF':>5} {'Sharpe':>7} {'hold':>5} {'$total':>9}")
    for i, r in enumerate(rows, 1):
        if r.get("n_trades", 0) < 30: continue
        h_str = f"{r['mean_hold_steps']:.1f}" if r.get('mean_hold_steps') else "—"
        pf_str = f"{r['pf']:.2f}" if r.get('pf') is not None else "—"
        sh_str = f"{r['sharpe']:.3f}" if r.get('sharpe') is not None else "—"
        print(f"{i:>4}  {r['policy']:<32} {r['n_trades']:>6,} {r['mean_ticks']:>6.3f} {r['wr_pct']:>5.1f} {pf_str:>5} {sh_str:>7} {h_str:>5} {r['total_usd']:>9,.0f}")

    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
