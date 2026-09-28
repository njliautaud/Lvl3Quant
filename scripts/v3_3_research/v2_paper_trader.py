"""
v2 paper-trader — direct $-comparison to HC #318 v3.2 paper-trader.

Strategy: same horizon-exit P&L as v3.2 paper-trader, but using v2 predictions.
v2 has NO realized_vol / MFE / MAE heads (only 3 horizons: 1s, 5s, 10s), so:
  - No vol-targeted sizing variant
  - No TP/SL bracket variant
  - No RV-gate
  - Just fixed-horizon directional on each of 1s/5s/10s, with confidence bands.

For direct apples-to-apples vs v3.2 (which got dominant Sharpe on RV-gated H1s),
we are testing whether v2's BETTER top-1%-band IC translates into BETTER trade economics
even WITHOUT the new exec-features.

OUT: output/v3_2_deep_sim_20260512/paper_trader_v2_compare.json
"""
from __future__ import annotations
import json, csv
from pathlib import Path
import numpy as np

V2_DIR = Path("/home/jupiter/Lvl3Quant/output/cnn_mamba_v2_smart_v3_mar")
MBO_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3")
OUT_JSON = Path("/home/jupiter/Lvl3Quant/output/v3_2_deep_sim_20260512/paper_trader_v2_compare.json")
OUT_CSV = Path("/home/jupiter/Lvl3Quant/output/v3_2_deep_sim_20260512/paper_trader_v2_compare.csv")

OOT_DATES = ["20260223", "20260224", "20260225", "20260226", "20260227"]
V2_STRIDE, V2_WINDOW = 500, 1000
HORIZONS = {"1s": 0, "5s": 1, "10s": 2}

COST_PASSIVE_RT = 0.376
COST_MARKET_RT  = 1.376
ES_TICK_USD = 12.50
# v2 stride is 500ms ≈ 2 steps/s. min gap 2 steps = 1s anti-churn.
MIN_GAP_STEPS = 2

CONF_BANDS = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]


def perf_stats(pnl, ANN=252.0):
    n = len(pnl)
    if n < 5:
        return {"n_trades": n, "mean_ticks": float("nan"), "sharpe": float("nan"),
                "sortino": float("nan"), "pf": float("nan"), "wr_pct": float("nan"),
                "total_ticks": 0.0, "total_usd": 0.0}
    mean = float(pnl.mean()); std = float(pnl.std(ddof=1))
    sharpe = (mean / std * np.sqrt(ANN)) if std > 1e-9 else float("nan")
    neg = pnl[pnl < 0]
    dn = float(neg.std(ddof=1)) if len(neg) > 1 else 0.0
    sortino = (mean / dn * np.sqrt(ANN)) if dn > 1e-9 else float("nan")
    gw = float(pnl[pnl > 0].sum()); gl = -float(pnl[pnl < 0].sum())
    pf = (gw / gl) if gl > 1e-9 else float("inf")
    wr = float((pnl > 0).mean() * 100.0)
    return {
        "n_trades": n, "mean_ticks": mean, "median_ticks": float(np.median(pnl)),
        "std_ticks": std,
        "sharpe": float(sharpe) if np.isfinite(sharpe) else None,
        "sortino": float(sortino) if np.isfinite(sortino) else None,
        "pf": float(pf) if np.isfinite(pf) else None,
        "wr_pct": wr,
        "total_ticks": float(pnl.sum()), "total_usd": float(pnl.sum() * ES_TICK_USD),
        "max_win_ticks": float(pnl.max()), "max_loss_ticks": float(pnl.min()),
    }


def session_masks(ts_ns: np.ndarray):
    dt64 = ts_ns.astype("datetime64[ns]")
    minutes_utc = (dt64.astype("datetime64[m]") - dt64.astype("datetime64[D]").astype("datetime64[m]")).astype(int)
    minutes_et = (minutes_utc - 5 * 60) % (24 * 60)
    rth_open = (minutes_et >= 9*60+30) & (minutes_et < 10*60+30)
    return {
        "all": np.ones(len(ts_ns), dtype=bool),
        "ex_first_30m_rth": ~rth_open,
    }


def trade_fixed_horizon(pred, target, sel_idx, cost, min_gap=MIN_GAP_STEPS):
    if len(sel_idx) == 0:
        return np.zeros(0, dtype=np.float64)
    taken = []; last = -10**9
    for i in sel_idx:
        if i - last < min_gap: continue
        if not np.isfinite(pred[i]) or not np.isfinite(target[i]): continue
        taken.append(i); last = i
    if not taken: return np.zeros(0, dtype=np.float64)
    taken = np.array(taken, dtype=np.int64)
    direction = np.sign(pred[taken])
    return direction * target[taken] - cost


def main():
    # Load v2 — concat all 5 folds
    preds = {h: [] for h in HORIZONS}; targs = {h: [] for h in HORIZONS}
    ts_parts = []
    for fold, date in enumerate(OOT_DATES):
        d = np.load(V2_DIR / f"fold_{fold:02d}_oot_predictions.npz", allow_pickle=True)
        p = d["predictions"]; lab = d["labels"]; n = p.shape[0]
        mbo = np.load(MBO_DIR / f"{date}_mbo_events.npz", allow_pickle=True)
        ts = mbo["timestamps"]; n_ev = len(ts)
        ts_pred = np.zeros(n, dtype=np.int64)
        for i in range(n):
            ev_idx = min(i * V2_STRIDE + V2_WINDOW - 1, n_ev - 1)
            ts_pred[i] = ts[ev_idx]
        ts_parts.append(ts_pred)
        for h, hi in HORIZONS.items():
            preds[h].append(p[:, hi])
            targs[h].append(lab[:, hi])
    v2_ts = np.concatenate(ts_parts)
    preds = {h: np.concatenate(preds[h]) for h in HORIZONS}
    targs = {h: np.concatenate(targs[h]) for h in HORIZONS}
    masks = session_masks(v2_ts)
    print(f"v2 total predictions: {len(v2_ts):,}")
    print(f"  ex_open: {masks['ex_first_30m_rth'].sum():,}")

    findings = {"setup": {
        "model": "CNN-Mamba v2 (smart_v3_mar, 3 horizons)",
        "n_predictions": int(len(v2_ts)),
        "v2_stride_window": [V2_STRIDE, V2_WINDOW],
        "cost_passive_rt": COST_PASSIVE_RT,
        "cost_market_rt": COST_MARKET_RT,
        "oot_dates": OOT_DATES,
    }, "strategies": {}}
    rows = []

    for bucket_name, bucket_mask in masks.items():
        for h in HORIZONS:
            p = preds[h]; t = targs[h]
            valid = bucket_mask & np.isfinite(p) & np.isfinite(t)
            v_idx = np.where(valid)[0]
            if len(v_idx) < 100:
                continue
            abs_p = np.abs(p)
            for b in CONF_BANDS:
                thr = float(np.quantile(abs_p[v_idx], 1.0 - b))
                sel_mask = valid & (abs_p >= thr)
                sel_idx = np.where(sel_mask)[0]
                for cost_label, cost_val in [("passive", COST_PASSIVE_RT), ("market", COST_MARKET_RT)]:
                    pnl = trade_fixed_horizon(p, t, sel_idx, cost_val)
                    s = perf_stats(pnl)
                    s_key = f"v2_H{h}_{bucket_name}_top{b*100:g}pct_{cost_label}"
                    findings["strategies"][s_key] = s
                    rows.append({"strategy": s_key, **s})

    with open(OUT_JSON, "w") as f:
        json.dump(findings, f, indent=2,
                  default=lambda x: None if (isinstance(x, float) and not np.isfinite(x)) else x)

    fields = ["strategy", "n_trades", "mean_ticks", "median_ticks", "std_ticks",
              "sharpe", "sortino", "pf", "wr_pct", "total_ticks", "total_usd",
              "max_win_ticks", "max_loss_ticks"]
    rows.sort(key=lambda r: -(r["sharpe"] or -1e9))
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows: w.writerow({k: r.get(k) for k in fields})

    print("\nTOP 20 v2 STRATEGIES (n>=50, Sharpe-ranked):")
    print(f"{'rank':>4}  {'strategy':<55} {'n':>6} {'mean_t':>7} {'WR%':>5} {'PF':>5} {'Sharpe':>7} {'$total':>10}")
    rank = 1
    for r in rows:
        if (r.get("n_trades") or 0) < 50 or r.get("sharpe") is None:
            continue
        print(f"{rank:>4}  {r['strategy']:<55} {r['n_trades']:>6,} {r['mean_ticks']:>7.3f} {r['wr_pct']:>5.1f} {r['pf']:>5.2f} {r['sharpe']:>7.3f} {r['total_usd']:>10,.0f}")
        rank += 1
        if rank > 20: break

    print(f"\nWrote {len(rows)} v2 strategies → {OUT_JSON}")


if __name__ == "__main__":
    main()
