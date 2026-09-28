"""
HC #318 — v3.2 market-replay paper-trader using NEW outputs.

Goal: convert v3.2 OOT predictions into realistic Sharpe/Sortino/PF/WR, leveraging
the full multi-head taxonomy (not just log_ret_1s). This is the first
performance-at-confidence read on v3.2 with execution-aware features.

Heads used (verified high-IC from head_by_head_ic.csv):
  - pred_log_ret_1s     (IC=0.274, DA=60.28%) — DIRECTIONAL ENTRY
  - pred_realized_vol_30s_ticks (IC=0.616) — SIZING (1/vol-targeted)
  - pred_mae_30s_ticks (IC=0.362)         — STOP DISTANCE
  - pred_mfe_30s_ticks (IC=0.247)         — TARGET DISTANCE
  - pred_p_reversal_15s (IC=0.148)        — EXIT OVERRIDE

Methodology (intentional simplifications — full FIFO .dbn replay is next step):
  * Per signal, exit P&L = sign(pred) * realized_log_ret @ HORIZON, in ticks.
    Labels in predictions.npz are ALREADY tick-scale (verified std≈1.65 for 1s).
  * TP/SL bracket: if abs(realized_30s) reaches MFE or MAE first, that fills.
    We approximate first-touch using sign of realized_30s vs predicted bracket.
  * Cost: 0.376 ticks passive limit RT (CLAUDE.md canonical) — 1.376 if market.
  * Confidence bands: top-1%, 5%, 10%, 20% of |pred_log_ret_1s| within OOT set.
  * Vol-targeted sizing: position_size = clip(target_vol_ticks / pred_realized_vol_30s_ticks, 0.25, 4)
    where target_vol_ticks = median(pred_realized_vol_30s_ticks).
  * Reversal exit: if p_reversal_15s > 0.5 at entry, treat as no-trade (signal-canceled)
    (deferred: dynamic re-evaluation at t+k).

OUT: output/v3_2_deep_sim_20260512/paper_trader_v32_hc318.json
"""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path

PREDS = Path("/home/jupiter/Lvl3Quant/output/v3_2_deep_sim_20260512/fold_00_oot_predictions.npz")
OUT_JSON = Path("/home/jupiter/Lvl3Quant/output/v3_2_deep_sim_20260512/paper_trader_v32_hc318.json")
OUT_CSV = Path("/home/jupiter/Lvl3Quant/output/v3_2_deep_sim_20260512/paper_trader_v32_hc318_strategies.csv")

# Canonical cost (CLAUDE.md)
COST_PASSIVE_RT = 0.376  # ticks
COST_MARKET_RT  = 1.376  # ticks
ES_TICK_USD = 12.50

# Anti-churn: minimum samples between trades (each step = 250ms; 4 steps = 1s gap)
MIN_GAP_STEPS = 4

# Signal selection bands (cumulative top-X% by |pred_log_ret_1s|)
CONF_BANDS = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00]


def perf_stats(pnl_ticks: np.ndarray, ann_factor: float) -> dict:
    """Sharpe, Sortino, PF, WR over per-trade pnl_ticks (after costs)."""
    n = int(len(pnl_ticks))
    if n < 5:
        return {"n_trades": n, "mean_ticks": float("nan"), "sharpe": float("nan"),
                "sortino": float("nan"), "pf": float("nan"), "wr_pct": float("nan"),
                "total_ticks": 0.0, "total_usd": 0.0}
    mean = float(pnl_ticks.mean())
    std = float(pnl_ticks.std(ddof=1))
    sharpe = (mean / std * np.sqrt(ann_factor)) if std > 1e-9 else float("nan")
    neg = pnl_ticks[pnl_ticks < 0]
    dn = float(neg.std(ddof=1)) if len(neg) > 1 else 0.0
    sortino = (mean / dn * np.sqrt(ann_factor)) if dn > 1e-9 else float("nan")
    gross_win = float(pnl_ticks[pnl_ticks > 0].sum())
    gross_loss = -float(pnl_ticks[pnl_ticks < 0].sum())
    pf = (gross_win / gross_loss) if gross_loss > 1e-9 else float("inf")
    wr = float((pnl_ticks > 0).mean() * 100.0)
    return {
        "n_trades": n,
        "mean_ticks": mean,
        "median_ticks": float(np.median(pnl_ticks)),
        "std_ticks": std,
        "sharpe": float(sharpe) if np.isfinite(sharpe) else None,
        "sortino": float(sortino) if np.isfinite(sortino) else None,
        "pf": float(pf) if np.isfinite(pf) else None,
        "wr_pct": wr,
        "total_ticks": float(pnl_ticks.sum()),
        "total_usd": float(pnl_ticks.sum() * ES_TICK_USD),
        "max_win_ticks": float(pnl_ticks.max()),
        "max_loss_ticks": float(pnl_ticks.min()),
    }


def simulate_fixed_horizon(
    pred_dir: np.ndarray,           # signed direction signal (typically pred_log_ret_1s)
    realized_at_h: np.ndarray,      # realized log-return at exit horizon (TICKS)
    mask_h: np.ndarray,             # valid mask for that horizon
    sel_idx: np.ndarray,            # candidate signal indices (after conf-band gate)
    size: np.ndarray | None = None, # per-signal size multiplier
    cost_rt: float = COST_PASSIVE_RT,
    min_gap: int = MIN_GAP_STEPS,
) -> np.ndarray:
    """Run fixed-horizon paper trades on sel_idx. Returns per-trade P&L in ticks (after costs)."""
    # Anti-churn: drop signals within MIN_GAP of prior taken signal
    if len(sel_idx) == 0:
        return np.zeros(0, dtype=np.float64)
    taken = []
    last = -10**9
    for i in sel_idx:
        if i - last < min_gap:
            continue
        if not mask_h[i]:
            continue
        if not np.isfinite(pred_dir[i]) or not np.isfinite(realized_at_h[i]):
            continue
        taken.append(i)
        last = i
    if not taken:
        return np.zeros(0, dtype=np.float64)
    taken = np.array(taken, dtype=np.int64)
    direction = np.sign(pred_dir[taken])
    # Per-trade gross = direction * realized
    gross = direction * realized_at_h[taken]
    if size is not None:
        sz = np.clip(size[taken], 0.25, 4.0)
        gross = gross * sz
        net = gross - cost_rt * sz  # cost scales with size
    else:
        net = gross - cost_rt
    return net


def simulate_brkt_30s(
    pred_dir: np.ndarray,
    pred_mfe_ticks: np.ndarray,
    pred_mae_ticks: np.ndarray,
    realized_30s: np.ndarray,
    mask_30s: np.ndarray,
    sel_idx: np.ndarray,
    size: np.ndarray | None = None,
    cost_rt: float = COST_PASSIVE_RT,
    min_gap: int = MIN_GAP_STEPS,
    floor_tp: float = 1.0,
    floor_sl: float = 1.0,
) -> np.ndarray:
    """
    TP/SL bracket trade using predicted MFE (target) and predicted MAE (stop).
    Bracket sized in direction of pred_dir.

    Simplification: 'first-touch' is approximated. We know only realized at 30s.
    Conservative rule:
      - If sign(realized_30s) == direction AND |realized_30s| >= tp: WIN at +tp ticks
      - Elif sign(realized_30s) != direction AND |realized_30s| >= sl: LOSS at -sl ticks
      - Else: exit at realized_30s (in direction of trade)
    This UNDER-counts double-touches (path-dependence ignored) — a known bias.
    """
    if len(sel_idx) == 0:
        return np.zeros(0, dtype=np.float64)
    taken = []
    last = -10**9
    for i in sel_idx:
        if i - last < min_gap:
            continue
        if not mask_30s[i]:
            continue
        if not (np.isfinite(pred_dir[i]) and np.isfinite(pred_mfe_ticks[i])
                and np.isfinite(pred_mae_ticks[i]) and np.isfinite(realized_30s[i])):
            continue
        taken.append(i)
        last = i
    if not taken:
        return np.zeros(0, dtype=np.float64)
    taken = np.array(taken, dtype=np.int64)
    direction = np.sign(pred_dir[taken])
    r = realized_30s[taken]
    tp = np.maximum(np.abs(pred_mfe_ticks[taken]), floor_tp)
    sl = np.maximum(np.abs(pred_mae_ticks[taken]), floor_sl)
    # Realized in trade direction
    r_dir = direction * r
    pnl = np.where(
        r_dir >= tp,
        tp,
        np.where(r_dir <= -sl, -sl, r_dir),
    )
    if size is not None:
        sz = np.clip(size[taken], 0.25, 4.0)
        pnl = pnl * sz
        return pnl - cost_rt * sz
    return pnl - cost_rt


def main():
    d = np.load(PREDS, allow_pickle=True)
    n = int(d["n_samples"])
    print(f"Loaded predictions: n_samples={n:,}")
    print(f"OOT dates: {d['oot_dates']}")

    # Heads we'll use
    pred_dir = d["pred_log_ret_1s"][:n]                      # entry direction signal
    rv = d["pred_pred_realized_vol_30s_ticks"][:n]           # vol sizing (head name: pred_realized_vol_30s_ticks)
    mfe = d["pred_pred_mfe_30s_ticks"][:n]
    mae = d["pred_pred_mae_30s_ticks"][:n]
    p_rev = d["pred_p_reversal_15s"][:n]
    # Realized labels (tick-scale per audit)
    realized = {
        "1s":  d["target_log_ret_1s"][:n],
        "5s":  d["target_log_ret_5s"][:n],
        "10s": d["target_log_ret_10s"][:n],
        "30s": d["target_log_ret_30s"][:n],
    }
    masks = {
        "1s":  d["mask_log_ret_1s"][:n].astype(bool),
        "5s":  d["mask_log_ret_5s"][:n].astype(bool),
        "10s": d["mask_log_ret_10s"][:n].astype(bool),
        "30s": d["mask_log_ret_30s"][:n].astype(bool),
    }
    mask_mfe = d["mask_pred_mfe_30s_ticks"][:n].astype(bool)
    mask_mae = d["mask_pred_mae_30s_ticks"][:n].astype(bool)
    mask_rev = d["mask_p_reversal_15s"][:n].astype(bool)
    mask_rv  = d["mask_pred_realized_vol_30s_ticks"][:n].astype(bool)
    # NOTE: target keys for head heads use the head's full name (pred_X), so target_pred_X.
    # Predicted realized vol head — also need to align mask for valid target reference.

    # Annualization factor: ES RTH ~6.5h × 5d/wk × 50wk
    # Per trade ≈ Sharpe annual basis. Conservative: trades_per_day×252 if we measured per-trade std.
    # Use n_trades-equivalent basis: ann ≈ n_trades_per_day_avg * 252. We compute ann from elapsed.
    # Simpler: just report Sharpe scaled to per-day-of-OOT (5 OOT days), call it "Sharpe (per-trade × √252)".
    ANN = 252.0  # per-trade basis (each trade independent unit)

    # Confidence-band gating on |pred_log_ret_1s|
    abs_sig = np.abs(pred_dir)
    abs_sig_valid = abs_sig[masks["1s"] & np.isfinite(abs_sig)]
    print(f"Signal magnitude p99: {np.quantile(abs_sig_valid, 0.99):.3f} ticks; median: {np.quantile(abs_sig_valid, 0.5):.3f}")

    # Size multiplier (vol-targeted)
    target_vol = float(np.median(rv[mask_rv & np.isfinite(rv) & (rv > 0)]))
    safe_rv = np.where(mask_rv & np.isfinite(rv) & (rv > 0), rv, target_vol)
    size_vol = target_vol / safe_rv
    print(f"Target vol (median pred_rv_30s): {target_vol:.3f} ticks  | size range example: {float(np.clip(size_vol, 0.25, 4).mean()):.3f} mean")

    findings = {"setup": {
        "n_samples": n,
        "target_vol_ticks": target_vol,
        "cost_passive_rt_ticks": COST_PASSIVE_RT,
        "cost_market_rt_ticks": COST_MARKET_RT,
        "min_gap_steps_250ms": MIN_GAP_STEPS,
        "annualization_factor": ANN,
        "methodology": "fixed-horizon and TP/SL bracket; first-touch approximated by sign of realized_30s",
    }, "strategies": {}}

    rows = []  # for CSV

    # ============================================================
    # Strategy family A: Fixed-horizon trades on |pred_log_ret_1s|
    # ============================================================
    for h_name in ["1s", "5s", "10s", "30s"]:
        rzd = realized[h_name]
        m_h = masks[h_name]
        valid_signal = m_h & np.isfinite(pred_dir) & np.isfinite(rzd) & np.isfinite(abs_sig)
        v_idx = np.where(valid_signal)[0]
        thresholds = {}
        if len(v_idx) > 100:
            mag_sorted = np.sort(abs_sig[v_idx])
            for b in CONF_BANDS:
                # top-b fraction → threshold at quantile (1-b)
                q = max(0.0, 1.0 - b)
                thresholds[b] = float(np.quantile(abs_sig[v_idx], q))
        for b, thr in thresholds.items():
            sel_mask = valid_signal & (abs_sig >= thr)
            sel_idx = np.where(sel_mask)[0]
            for variant, sz in [("flat", None), ("vol_sized", size_vol)]:
                for cost_label, cost_val in [("passive", COST_PASSIVE_RT), ("market", COST_MARKET_RT)]:
                    pnl = simulate_fixed_horizon(pred_dir, rzd, m_h, sel_idx,
                                                 size=sz, cost_rt=cost_val)
                    s = perf_stats(pnl, ANN)
                    s_key = f"H{h_name}_top{b*100:g}pct_{variant}_{cost_label}"
                    findings["strategies"][s_key] = s
                    rows.append({"strategy": s_key, **s})

    # ============================================================
    # Strategy family B: TP/SL bracket using pred_mfe/mae
    # ============================================================
    # Need: signal direction, MFE/MAE prediction, realized_30s
    m30 = masks["30s"] & mask_mfe & mask_mae & mask_rev
    valid_B = m30 & np.isfinite(pred_dir) & np.isfinite(mfe) & np.isfinite(mae) & np.isfinite(realized["30s"])
    abs_sig_B = abs_sig.copy()
    # Confidence-band gate
    if valid_B.sum() > 100:
        for b in CONF_BANDS:
            thr = float(np.quantile(abs_sig_B[valid_B], 1.0 - b))
            for rev_filter in ["none", "p_rev_lt_0p5"]:
                sel_mask = valid_B & (abs_sig_B >= thr)
                if rev_filter == "p_rev_lt_0p5":
                    sel_mask = sel_mask & (p_rev < 0.5)
                sel_idx = np.where(sel_mask)[0]
                for variant, sz in [("flat", None), ("vol_sized", size_vol)]:
                    pnl = simulate_brkt_30s(pred_dir, mfe, mae, realized["30s"], masks["30s"],
                                            sel_idx, size=sz, cost_rt=COST_PASSIVE_RT)
                    s = perf_stats(pnl, ANN)
                    s_key = f"BRKT30s_top{b*100:g}pct_{variant}_{rev_filter}"
                    findings["strategies"][s_key] = s
                    rows.append({"strategy": s_key, **s})

    # ============================================================
    # Strategy family C: Short-side ONLY (HC-prior: shorts have better edge)
    # ============================================================
    for h_name in ["1s", "5s", "10s"]:
        rzd = realized[h_name]
        m_h = masks[h_name]
        valid_C = m_h & np.isfinite(pred_dir) & np.isfinite(rzd) & (pred_dir < 0)
        if valid_C.sum() < 200:
            continue
        for b in CONF_BANDS:
            thr = float(np.quantile(abs_sig[valid_C], 1.0 - b))
            sel_mask = valid_C & (abs_sig >= thr)
            sel_idx = np.where(sel_mask)[0]
            for variant, sz in [("flat", None), ("vol_sized", size_vol)]:
                pnl = simulate_fixed_horizon(pred_dir, rzd, m_h, sel_idx, size=sz)
                s = perf_stats(pnl, ANN)
                s_key = f"SHORTonly_H{h_name}_top{b*100:g}pct_{variant}_passive"
                findings["strategies"][s_key] = s
                rows.append({"strategy": s_key, **s})

    # ============================================================
    # Strategy family D: Realized-vol head used as confidence gate
    # ============================================================
    # Hypothesis: high predicted realized vol → wider expected move → bigger edge
    rv_p90 = float(np.quantile(rv[mask_rv & np.isfinite(rv)], 0.90))
    rv_p10 = float(np.quantile(rv[mask_rv & np.isfinite(rv)], 0.10))
    for tag, mask_rv_gate in [("high_rv", rv >= rv_p90), ("low_rv", rv <= rv_p10)]:
        for h_name in ["1s", "5s"]:
            rzd = realized[h_name]
            m_h = masks[h_name]
            valid_D = m_h & np.isfinite(pred_dir) & np.isfinite(rzd) & mask_rv_gate
            if valid_D.sum() < 100:
                continue
            for b in [0.01, 0.05, 0.10]:
                thr = float(np.quantile(abs_sig[valid_D], 1.0 - b))
                sel_mask = valid_D & (abs_sig >= thr)
                sel_idx = np.where(sel_mask)[0]
                pnl = simulate_fixed_horizon(pred_dir, rzd, m_h, sel_idx, size=None)
                s = perf_stats(pnl, ANN)
                s_key = f"RVgate_{tag}_H{h_name}_top{b*100:g}pct_flat_passive"
                findings["strategies"][s_key] = s
                rows.append({"strategy": s_key, **s})

    # =====================================
    # Write outputs
    # =====================================
    with open(OUT_JSON, "w") as f:
        json.dump(findings, f, indent=2,
                  default=lambda x: None if (isinstance(x, float) and not np.isfinite(x)) else x)

    # CSV ranked by Sharpe
    import csv
    fieldnames = ["strategy", "n_trades", "mean_ticks", "median_ticks", "std_ticks",
                  "sharpe", "sortino", "pf", "wr_pct", "total_ticks", "total_usd",
                  "max_win_ticks", "max_loss_ticks"]
    rows.sort(key=lambda r: -(r["sharpe"] or -1e9))
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})

    # Console: top-20 strategies by Sharpe (with n>=50)
    print("\nTOP 20 STRATEGIES BY SHARPE (n_trades>=50):")
    print(f"{'rank':>4}  {'strategy':<55} {'n':>6} {'mean_t':>7} {'WR%':>5} {'PF':>5} {'Sharpe':>7} {'$total':>10}")
    rank = 1
    for r in rows:
        if (r.get("n_trades") or 0) < 50:
            continue
        if r.get("sharpe") is None:
            continue
        print(f"{rank:>4}  {r['strategy']:<55} {r['n_trades']:>6,} {r['mean_ticks']:>7.3f} {r['wr_pct']:>5.1f} {r['pf'] if r['pf'] is not None else float('nan'):>5.2f} {r['sharpe']:>7.3f} {r['total_usd']:>10,.0f}")
        rank += 1
        if rank > 20:
            break

    print(f"\nWrote {len(rows)} strategies → {OUT_JSON}")
    print(f"Ranked CSV → {OUT_CSV}")


if __name__ == "__main__":
    main()
