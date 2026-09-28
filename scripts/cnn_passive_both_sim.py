"""
CNN-Mamba Passive Entry + Passive Exit Trade Sim
=================================================
Tests the crucial question: if we can get filled passively on BOTH sides,
does CNN-Mamba's directional signal overcome the reduced 0.752t RT cost?

Prior tests showed:
- 16 configs gross-profitable at 1.752t cost (passive entry + market exit)
- If we can exit passively too, cost = 0.752t (commission only, both sides)

Passive exit model:
- After entry fill, immediately post limit exit on the other side
- LONG: bought at bid → sell at ask (mid + 0.5)
- SHORT: sold at ask → buy at bid (mid - 0.5)
- Exit fills when price reaches our limit within max_hold window
- If not filled within window → market exit at 1.376t cost
- This is realistic for ES which is 1 tick wide during RTH

Also tests: maker entry + maker exit with cancel-on-signal-reversal
"""
from __future__ import annotations
import gc, json, sys, time, warnings
from pathlib import Path
from typing import Optional
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
try:
    import mlflow; HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

CNN_MAMBA_DIR = Path("/home/nick/Lvl3Quant/output/cnn_mamba_v2_bulk_oot")
RELABEL_DIR = Path("/home/nick/Lvl3Quant/data/relabel")
OUTPUT_DIR = Path("/home/nick/Lvl3Quant/output/cnn_passive_both_sim_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MLFLOW_URI = "http://localhost:5000"

ES_TICK_VALUE = 12.50
COMMISSION_TICKS = 0.376
SPREAD_TICKS = 1.0
PASSIVE_RT_COST = 2 * COMMISSION_TICKS          # 0.752 (both sides passive)
MIXED_RT_COST = COMMISSION_TICKS + COMMISSION_TICKS + SPREAD_TICKS  # 1.752 (passive entry + market exit)

CNN_WINDOW = 3000
CNN_STRIDE = 250

# Parameters
CONFIDENCE_PCTILES = [90, 92, 95, 97, 99]
PASSIVE_EXIT_WINDOWS = [100, 250, 500, 1000, 2000]  # events to wait for passive fill
MAX_HOLD_EVENTS_LIST = [500, 1000, 2000]
MIN_EVENTS_BETWEEN = 100
PASSIVE_FILL_WINDOW = 40  # entry fill window


def load_day(date_str: str) -> Optional[dict]:
    """Load CNN-Mamba predictions + price data."""
    pred_path = CNN_MAMBA_DIR / f"{date_str}_predictions.npz"
    if not pred_path.exists():
        return None
    d = np.load(pred_path, allow_pickle=True)
    preds = d["predictions"]
    n = len(preds)
    event_indices = CNN_WINDOW + np.arange(n) * CNN_STRIDE
    
    rl_path = RELABEL_DIR / f"mfe_mae_h10s_{date_str}.parquet"
    if not rl_path.exists():
        rl_path = RELABEL_DIR / f"mfe_mae_h30s_{date_str}.parquet"
    if not rl_path.exists():
        return None
    rl = pd.read_parquet(rl_path, columns=["mid_t_ticks", "ts_ns"])
    mid_prices = rl["mid_t_ticks"].values.astype(np.float64)
    ts_ns = rl["ts_ns"].values
    
    max_idx = len(mid_prices) - 1
    valid_mask = event_indices <= max_idx
    event_indices = event_indices[valid_mask]
    preds = preds[valid_mask]
    
    return {
        "date": date_str,
        "event_indices": event_indices,
        "preds_1s": preds[:, 0],
        "preds_5s": preds[:, 1],
        "preds_10s": preds[:, 2],
        "mid_prices": mid_prices,
        "ts_ns": ts_ns,
    }


def simulate_day(data: dict, conf_pctile: float, passive_exit_window: int,
                 max_hold: int, require_5s_agree: bool = True) -> list[dict]:
    """Simulate with passive entry AND passive exit."""
    trades = []
    date_str = data["date"]
    event_indices = data["event_indices"]
    preds_10s = data["preds_10s"]
    preds_5s = data["preds_5s"]
    mid_prices = data["mid_prices"]
    ts_ns = data["ts_ns"]
    n_preds = len(preds_10s)
    n_prices = len(mid_prices)
    
    abs_preds = np.abs(preds_10s)
    conf_threshold = np.percentile(abs_preds, conf_pctile)
    
    last_exit_idx = -MIN_EVENTS_BETWEEN
    
    for i in range(n_preds):
        ei = event_indices[i]
        if ei - last_exit_idx < MIN_EVENTS_BETWEEN:
            continue
        
        p10 = preds_10s[i]
        if abs(p10) < conf_threshold:
            continue
        
        direction = 1 if p10 > 0 else -1
        
        # Optional 5s agreement
        if require_5s_agree:
            p5 = preds_5s[i]
            if np.sign(p5) != np.sign(p10):
                continue
        
        # ─── Passive Entry ───
        entry_mid = mid_prices[ei]
        fill_eidx = None
        end_fill = min(ei + PASSIVE_FILL_WINDOW, n_prices)
        
        if direction == 1:
            limit_price = entry_mid - 0.5
            for fi in range(ei + 1, end_fill):
                if mid_prices[fi] <= limit_price:
                    fill_eidx = fi; break
        else:
            limit_price = entry_mid + 0.5
            for fi in range(ei + 1, end_fill):
                if mid_prices[fi] >= limit_price:
                    fill_eidx = fi; break
        
        if fill_eidx is None:
            continue
        
        entry_price = mid_prices[fill_eidx]
        entry_ts = ts_ns[fill_eidx]
        
        # ─── Passive Exit ───
        # Post limit on opposite side immediately after fill
        # LONG: sell at ask = entry_price + 1.0 (1 tick profit target passive)
        # SHORT: buy at bid = entry_price - 1.0 (1 tick profit target passive)
        # Also try: exit at same price (breakeven gross, profit from passive spread capture)
        
        # For passive exit, we need to actually make money GROSS.
        # Strategy: post exit limit at entry_price + direction * 1.0 (1 tick in our favor)
        # This means we need price to move 1 tick in our direction for exit fill.
        
        # Actually, simplest passive exit: post at the OTHER side of the book
        # LONG (bought at bid): sell at ask = entry_mid + 0.5. Since we entered at bid = entry_mid - 0.5,
        # exit at ask = entry_mid + 0.5 gives us gross = 1.0 tick (full spread capture!)
        # Net = 1.0 - 0.752 = +0.248 ticks per trade if filled.
        
        # But we got filled at entry_price which was when mid dropped to our bid level.
        # So entry_price = mid at fill time. Our exit should be at ask = entry_fill_mid + 0.5
        # But entry_fill_mid might be < original entry_mid if price dropped to fill us.
        
        # Most accurate: entry_price = fill_price (bid/ask level when filled)
        # For LONG: we bought at bid. exit at ask = entry_price + 1.0 (since ES spread = 1 tick)
        # For SHORT: we sold at ask. exit at bid = entry_price - 1.0
        
        if direction == 1:
            exit_limit = entry_price + 1.0  # sell at ask = bid + spread
        else:
            exit_limit = entry_price - 1.0  # buy at bid = ask - spread
        
        # Check if passive exit fills within window
        passive_filled = False
        exit_eidx = None
        max_exit = min(fill_eidx + passive_exit_window, n_prices - 1)
        
        max_favorable = 0.0
        max_adverse = 0.0
        
        for j in range(fill_eidx + 1, max_exit + 1):
            pnl = (mid_prices[j] - entry_price) * direction
            if pnl > max_favorable: max_favorable = pnl
            if pnl < max_adverse: max_adverse = pnl
            
            # Hard stop at -3 ticks (cancel passive exit, market out)
            if pnl < -3.0:
                exit_eidx = j
                break
            
            # Check passive exit fill
            if direction == 1:
                # LONG: exit fills when mid >= exit_limit (someone hits our ask)
                if mid_prices[j] >= exit_limit:
                    exit_eidx = j
                    passive_filled = True
                    break
            else:
                # SHORT: exit fills when mid <= exit_limit (someone hits our bid)
                if mid_prices[j] <= exit_limit:
                    exit_eidx = j
                    passive_filled = True
                    break
        
        if exit_eidx is None:
            exit_eidx = max_exit  # timeout → market exit
        
        # Calculate P&L
        if passive_filled:
            # Gross = 1.0 ticks (spread capture), cost = passive RT = 0.752
            gross_ticks = 1.0
            net_ticks = gross_ticks - PASSIVE_RT_COST
            exit_reason = "passive_fill"
            exit_cost = PASSIVE_RT_COST
        elif (mid_prices[exit_eidx] - entry_price) * direction < -3.0:
            # Hard stop triggered → market exit
            gross_ticks = (mid_prices[exit_eidx] - entry_price) * direction
            net_ticks = gross_ticks - MIXED_RT_COST
            exit_reason = "hard_stop"
            exit_cost = MIXED_RT_COST
        else:
            # Timeout → market exit
            gross_ticks = (mid_prices[exit_eidx] - entry_price) * direction
            net_ticks = gross_ticks - MIXED_RT_COST
            exit_reason = "timeout_market"
            exit_cost = MIXED_RT_COST
        
        exit_ts = ts_ns[exit_eidx]
        hold_time_s = (exit_ts - entry_ts) / 1e9
        
        trades.append({
            "date": date_str,
            "direction": direction,
            "gross_ticks": float(gross_ticks),
            "net_ticks": float(net_ticks),
            "mfe_ticks": float(max_favorable),
            "mae_ticks": float(max_adverse),
            "hold_events": exit_eidx - fill_eidx,
            "hold_time_s": float(hold_time_s),
            "exit_reason": exit_reason,
            "passive_exit": passive_filled,
            "exit_cost": float(exit_cost),
            "entry_pred_10s": float(p10),
        })
        
        last_exit_idx = exit_eidx
    
    return trades


def compute_metrics(trades_df: pd.DataFrame) -> dict:
    if len(trades_df) == 0:
        return {"n_trades": 0, "total_net_ticks": 0, "sharpe_annualized": 0,
                "win_rate": 0, "profit_factor": 0}
    net = trades_df["net_ticks"].values
    n = len(net)
    wr = float((net > 0).mean())
    gp = float(net[net > 0].sum()) if (net > 0).any() else 0
    gl = float(abs(net[net < 0].sum())) if (net < 0).any() else 1e-9
    daily = trades_df.groupby("date")["net_ticks"].sum()
    dm, ds = daily.mean(), daily.std() if len(daily) > 1 else (daily.mean(), 1)
    sharpe = (dm / ds) * np.sqrt(252) if ds > 0 else 0
    dd = daily[daily < 0]
    dds = dd.std() if len(dd) > 1 else ds
    sortino = (dm / dds) * np.sqrt(252) if dds > 0 else 0
    passive_rate = float(trades_df["passive_exit"].mean()) if "passive_exit" in trades_df.columns else 0
    return {
        "n_trades": n, "n_days": len(daily),
        "trades_per_day": n / max(len(daily), 1),
        "total_net_ticks": float(net.sum()),
        "total_net_dollars": float(net.sum()) * ES_TICK_VALUE,
        "win_rate": wr, "profit_factor": gp / gl,
        "sharpe_annualized": sharpe, "sortino_annualized": sortino,
        "avg_net": float(net.mean()),
        "passive_exit_rate": passive_rate,
        "avg_mfe": float(trades_df["mfe_ticks"].mean()),
        "avg_mae": float(trades_df["mae_ticks"].mean()),
        "avg_hold_s": float(trades_df["hold_time_s"].mean()),
    }


def main():
    t_start = time.time()
    print("=" * 70)
    print("CNN-MAMBA PASSIVE ENTRY + PASSIVE EXIT TRADE SIM")
    print(f"Passive RT cost: {PASSIVE_RT_COST:.3f}t | Mixed RT cost: {MIXED_RT_COST:.3f}t")
    print(f"Strategy: buy bid → sell ask (spread capture + directional gate)")
    print("=" * 70)
    
    # Load data
    pred_files = sorted(CNN_MAMBA_DIR.glob("*_predictions.npz"))
    dates = [f.stem.replace("_predictions", "") for f in pred_files]
    
    day_data = {}
    for d in dates:
        data = load_day(d)
        if data is not None:
            day_data[d] = data
    print(f"Loaded {len(day_data)} days ({min(day_data.keys())} to {max(day_data.keys())})")
    
    # Sweep
    results = []
    combos = []
    for conf_p in CONFIDENCE_PCTILES:
        for pew in PASSIVE_EXIT_WINDOWS:
            for agree in [True, False]:
                combos.append((conf_p, pew, agree))
    
    print(f"\nSweeping {len(combos)} configs...\n")
    
    for idx, (conf_p, pew, agree) in enumerate(combos):
        all_trades = []
        for d, data in sorted(day_data.items()):
            trades = simulate_day(data, conf_p, pew, pew, agree)
            all_trades.extend(trades)
        
        if all_trades:
            tdf = pd.DataFrame(all_trades)
            m = compute_metrics(tdf)
        else:
            m = compute_metrics(pd.DataFrame())
        
        result = {"conf_pctile": conf_p, "exit_window": pew,
                  "require_5s_agree": agree, **m}
        results.append(result)
        
        marker = "+" if m["total_net_ticks"] > 0 else "-"
        agree_str = "agree" if agree else "no_agree"
        if m["total_net_ticks"] > 0 or (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{len(combos)}] p{conf_p} ew={pew:4d} {agree_str:8s} | "
                  f"n={m['n_trades']:4d} WR={m['win_rate']:.1%} "
                  f"passive_exit={m.get('passive_exit_rate',0):.1%} "
                  f"net={m['total_net_ticks']:+8.1f}t "
                  f"Sharpe={m['sharpe_annualized']:+6.2f} "
                  f"PF={m['profit_factor']:.2f} {marker}")
    
    rdf = pd.DataFrame(results)
    rdf.to_csv(OUTPUT_DIR / "sweep_results.csv", index=False)
    
    # Results
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    
    valid = rdf[rdf["n_trades"] >= 30]
    profitable = valid[valid["total_net_ticks"] > 0]
    
    if len(profitable) > 0:
        print(f"\n✅ {len(profitable)} PROFITABLE CONFIGURATIONS FOUND!")
        top = profitable.nlargest(15, "sharpe_annualized")
        for _, row in top.iterrows():
            agree_str = "5s+10s" if row["require_5s_agree"] else "10s_only"
            print(f"  p{int(row['conf_pctile'])} ew={int(row['exit_window']):4d} {agree_str:7s} | "
                  f"n={int(row['n_trades']):4d} ({row['trades_per_day']:.1f}/d) "
                  f"WR={row['win_rate']:.1%} "
                  f"passive_exit={row['passive_exit_rate']:.1%} "
                  f"Sharpe={row['sharpe_annualized']:+.2f} "
                  f"Sortino={row['sortino_annualized']:+.2f} "
                  f"PF={row['profit_factor']:.2f} "
                  f"net={row['total_net_ticks']:+.1f}t "
                  f"(${row['total_net_dollars']:+,.0f})")
        
        # Best config detailed analysis
        best = top.iloc[0]
        print(f"\n  BEST: p{int(best['conf_pctile'])} ew={int(best['exit_window'])}")
        print(f"  Net: {best['total_net_ticks']:+.1f} ticks (${best['total_net_dollars']:+,.2f})")
        print(f"  Passive exit rate: {best['passive_exit_rate']:.1%}")
        print(f"  Avg MFE: {best['avg_mfe']:.2f}t | Avg MAE: {best['avg_mae']:.2f}t")
        print(f"  Avg hold: {best['avg_hold_s']:.1f}s")
        
        # Save best trades for detailed analysis
        best_params = (best["conf_pctile"], int(best["exit_window"]), best["require_5s_agree"])
        all_best_trades = []
        for d, data in sorted(day_data.items()):
            trades = simulate_day(data, best_params[0], best_params[1], best_params[1], best_params[2])
            all_best_trades.extend(trades)
        if all_best_trades:
            btdf = pd.DataFrame(all_best_trades)
            btdf.to_parquet(OUTPUT_DIR / "best_trades.parquet", index=False)
            
            # Per-day breakdown
            print(f"\n  Per-day P&L:")
            daily = btdf.groupby("date").agg(
                n=("net_ticks", "count"), net=("net_ticks", "sum"),
                wr=("net_ticks", lambda x: (x > 0).mean()),
                passive_rate=("passive_exit", "mean"))
            for dt, row in daily.iterrows():
                m = "+" if row["net"] > 0 else "-"
                print(f"    {dt}: {int(row['n']):3d} trades | net={row['net']:+7.1f}t | "
                      f"WR={row['wr']:.0%} | passive_exit={row['passive_rate']:.0%} {m}")
            
            # Exit reason breakdown
            print(f"\n  Exit reasons:")
            for reason, count in btdf["exit_reason"].value_counts().items():
                subset = btdf[btdf["exit_reason"] == reason]
                avg_net = subset["net_ticks"].mean()
                print(f"    {reason:20s}: {count:4d} ({count/len(btdf):.1%}) avg_net={avg_net:+.3f}t")
            
            # Direction breakdown
            print(f"\n  Direction:")
            for d_val, label in [(1, "LONG"), (-1, "SHORT")]:
                subset = btdf[btdf["direction"] == d_val]
                if len(subset) > 0:
                    print(f"    {label}: {len(subset)} trades | WR={float((subset['net_ticks']>0).mean()):.1%} | "
                          f"avg_net={subset['net_ticks'].mean():+.3f}t | total={subset['net_ticks'].sum():+.1f}t")
    else:
        print(f"\n  ❌ No profitable configurations found.")
        print(f"\n  Analysis of passive exit rates:")
        for conf in CONFIDENCE_PCTILES:
            subset = rdf[rdf["conf_pctile"] == conf]
            avg_pe = subset["passive_exit_rate"].mean()
            avg_net = subset["total_net_ticks"].mean()
            print(f"    p{conf}: avg_passive_exit_rate={avg_pe:.1%} avg_net={avg_net:+.1f}t")
        
        print(f"\n  The passive exit approach may still fail because:")
        print(f"  1. Adverse selection: fills happen AGAINST our predicted direction")
        print(f"  2. Spread capture only works ~50% of the time without edge")
        print(f"  3. Losses from hard stops / timeouts swamp the small spread-capture gains")
    
    # MLflow
    if HAS_MLFLOW:
        try:
            mlflow.set_tracking_uri(MLFLOW_URI)
            mlflow.set_experiment("cnn_passive_both_sim")
            with mlflow.start_run(run_name="v1_sweep"):
                mlflow.log_param("n_configs", len(combos))
                n_profitable = len(profitable) if len(profitable) > 0 else 0
                mlflow.log_metric("n_profitable", n_profitable)
                if len(valid) > 0:
                    best = valid.nlargest(1, "sharpe_annualized").iloc[0]
                    mlflow.log_metric("best_sharpe", best["sharpe_annualized"])
                    mlflow.log_metric("best_net_ticks", best["total_net_ticks"])
                mlflow.log_artifact(str(OUTPUT_DIR / "sweep_results.csv"))
        except Exception as e:
            print(f"MLflow error: {e}")
    
    elapsed = time.time() - t_start
    print(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f}m)")


if __name__ == "__main__":
    main()
