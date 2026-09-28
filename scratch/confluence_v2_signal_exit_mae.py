"""Layer-2 confluence on tp9_z2.3 trades:
  - Signal strength quintile breakdown
  - Exit reason breakdown (PnL contribution by exit type)
  - MAE/MFE asymmetry diagnostic
  - Combined: prime hours + top signal quintile + worst-MAE-trim
  
Capital = $50K/contract, ES futures.
"""
import json
from pathlib import Path
import numpy as np

CAPITAL = 50_000.0
SWEEP_DIR = Path("/home/jupiter/Lvl3Quant/execution/results/fifo_validation/sim_tp_z_finegrain_9day_20260427_101143")

def load_trades(tp_tag, z_tag):
    by_day = {}
    for fp in sorted(SWEEP_DIR.glob(f"{tp_tag}_{z_tag}_*.json")):
        date = fp.stem.split("_")[-1]
        by_day[date] = json.loads(fp.read_text()).get("trades", [])
    return by_day

def metrics(daily_pnls, all_trades):
    arr_d = np.array(daily_pnls, dtype=float)
    arr_t = np.array([t["pnl_dollars"] for t in all_trades], dtype=float)
    if arr_t.size == 0:
        return None
    cum = np.cumsum(arr_d)
    drawdown = cum - np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
    mdd_dollars = float(-drawdown.min()) if drawdown.size else 0.0
    neg = arr_d[arr_d < 0]
    sortino_d = float(arr_d.mean() / neg.std(ddof=1)) if neg.size > 1 and neg.std(ddof=1) > 0 else (float("inf") if arr_d.mean() > 0 else 0.0)
    return {
        "n_trades": int(arr_t.size), "n_days": int(arr_d.size),
        "n_pos_d": int((arr_d > 0).sum()),
        "total_pnl": float(arr_t.sum()), "pnl_pct": float(arr_t.sum()/CAPITAL*100.0),
        "sortino_d": sortino_d, "mdd_dollars": mdd_dollars, "mdd_pct": mdd_dollars/CAPITAL*100.0,
        "wr": float((arr_t > 0).mean()),
    }

def aggregate(filtered_by_day):
    daily = []
    flat = []
    for date in sorted(filtered_by_day.keys()):
        daily.append(sum(t["pnl_dollars"] for t in filtered_by_day[date]))
        flat.extend(filtered_by_day[date])
    return metrics(daily, flat)

def main():
    base = load_trades("tp9", "z2.3")
    flat = [t for d in base.values() for t in d]
    print(f"Loaded {len(flat)} trades across {len(base)} days\n")

    # === 1. Signal strength quintile ===
    sigs = np.array([abs(t["signal_strength"]) for t in flat])
    qs = np.quantile(sigs, [0.2, 0.4, 0.6, 0.8])
    print(f"Signal strength quintile boundaries: {qs.round(3)}")
    print(f"\n{'quintile':<12}{'n_tr':>6} {'pos_d':>6}  {'pnl_%':>8} {'sortino':>9} {'mdd_%':>8} {'wr':>7}")
    print("-" * 65)
    for q_idx in range(5):
        lo = qs[q_idx-1] if q_idx > 0 else -np.inf
        hi = qs[q_idx] if q_idx < 4 else np.inf
        filt = {date: [t for t in trades if lo < abs(t["signal_strength"]) <= hi]
                for date, trades in base.items()}
        m = aggregate(filt)
        if m and m["n_trades"]:
            print(f"Q{q_idx+1}          {m['n_trades']:>6} {m['n_pos_d']:>2}/{m['n_days']:<2}   "
                  f"{m['pnl_pct']:>6.2f}% {m['sortino_d']:>8.3f}  {m['mdd_pct']:>6.2f}%  {m['wr']:>5.1%}")

    # === 2. Exit reason breakdown ===
    print(f"\n{'exit_reason':<20}{'n_tr':>6}  {'mean_$':>8}  {'tot_pnl_$':>11}  {'tot_%':>7}  {'WR':>6}")
    print("-" * 70)
    by_reason = {}
    for t in flat:
        by_reason.setdefault(t["exit_reason"], []).append(t["pnl_dollars"])
    for reason in sorted(by_reason.keys(), key=lambda r: -sum(by_reason[r])):
        pnls = np.array(by_reason[reason])
        print(f"{reason:<20}{pnls.size:>6}  ${pnls.mean():>6.1f}  ${pnls.sum():>9,.0f}  "
              f"{pnls.sum()/CAPITAL*100:>5.2f}%  {(pnls>0).mean():>5.1%}")

    # === 3. MAE/MFE diagnostic ===
    print("\n=== MAE/MFE asymmetry on winners vs losers ===")
    winners = [t for t in flat if t["pnl_dollars"] > 0]
    losers = [t for t in flat if t["pnl_dollars"] < 0]
    for label, group in [("WIN", winners), ("LOSE", losers)]:
        if not group: continue
        mae = np.array([t["mae_ticks"] for t in group])
        mfe = np.array([t["mfe_ticks"] for t in group])
        print(f"  {label}: n={len(group)}  MAE p50={np.median(mae):.1f}  p90={np.quantile(mae,0.9):.1f}  "
              f"MFE p50={np.median(mfe):.1f}  p90={np.quantile(mfe,0.9):.1f}")

    # === 4. Combined: top quintile signal × prime hours (timestamp-aware) ===
    from datetime import datetime
    def in_prime(t):
        ts_ns = t.get("fill_time_ns") or t.get("post_time_ns") or t.get("signal_time_ns")
        if ts_ns is None: return True
        dt = datetime.utcfromtimestamp(ts_ns/1e9)
        et_h = (dt.hour - 5) % 24
        return (et_h == 9 and dt.minute >= 30) or et_h == 10 or (et_h == 11 and dt.minute < 30)
    q5_lo = qs[3]
    print(f"\n=== TOP-QUINTILE SIGNAL (sig>{q5_lo:.2f}) × PRIME HOURS combined ===")
    filt_combined = {
        date: [t for t in trades if abs(t["signal_strength"]) > q5_lo and in_prime(t)]
        for date, trades in base.items()
    }
    m = aggregate(filt_combined)
    if m:
        print(f"n_tr={m['n_trades']}  pos_d={m['n_pos_d']}/{m['n_days']}  "
              f"pnl=${m['total_pnl']:,.0f} ({m['pnl_pct']:.2f}%)  "
              f"Sortino={m['sortino_d']:.3f}  MDD={m['mdd_pct']:.2f}%  WR={m['wr']:.1%}")

    # === 5. Save layered configs ===
    out = {
        "quintile_signal_breakdown": "see stdout",
        "exit_reason_breakdown": {r: {"n": len(p), "total_pnl": float(sum(p))} for r, p in by_reason.items()},
        "top_quintile_x_prime": m,
    }
    Path("/home/jupiter/Lvl3Quant/scratch/confluence_v2_results.json").write_text(json.dumps(out, indent=2, default=str))
    print("\nWrote /home/jupiter/Lvl3Quant/scratch/confluence_v2_results.json")

if __name__ == "__main__":
    main()
