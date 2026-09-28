"""
CNN-Mamba Confidence-Gated Trade Sim — Baseline
================================================
Uses CNN-Mamba v2 prediction magnitude as the PRIMARY entry gate.
No intensity gating. Pure test of CNN-Mamba directional signal profitability
under FIFO-realistic costs.

Key insight: CNN-Mamba top 10% shorts showed +1.56 ticks at 60.5% WR in prior analysis.
This test validates that signal under realistic passive-fill + market-exit costs.

Sweep: confidence percentile thresholds × hold periods × short-only vs both sides
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

# Paths
CNN_MAMBA_DIR = Path("/home/nick/Lvl3Quant/output/cnn_mamba_v2_bulk_oot")
RELABEL_DIR = Path("/home/nick/Lvl3Quant/data/relabel")
EVENTS_DIR = Path("/home/nick/Lvl3Quant/data/processed/mbo_events_smart_v3")
OUTPUT_DIR = Path("/home/nick/Lvl3Quant/output/cnn_confidence_sim_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MLFLOW_URI = "http://localhost:5000"

# Cost
ES_TICK_VALUE = 12.50
COMMISSION_TICKS = 0.376
SPREAD_TICKS = 1.0
PASSIVE_ENTRY_COST = COMMISSION_TICKS
MARKET_EXIT_COST = COMMISSION_TICKS + SPREAD_TICKS
TOTAL_RT_COST = PASSIVE_ENTRY_COST + MARKET_EXIT_COST  # 1.752

CNN_WINDOW = 3000
CNN_STRIDE = 250

# Sweep grid
CONFIDENCE_PCTILES = [90, 92, 95, 97, 99]  # top N% abs(pred)
HOLD_EVENTS = [100, 250, 500, 1000]  # fixed hold in events
SIDES = ["both", "short_only", "long_only"]
MAX_HOLD_MODES = ["fixed", "trailing"]  # fixed = hold N events then exit; trailing = exit on retrace

# Passive fill sim
PASSIVE_FILL_WINDOW = 40
MIN_EVENTS_BETWEEN = 100


def load_day(date_str: str) -> Optional[dict]:
    """Load CNN-Mamba predictions + price data for one day."""
    pred_path = CNN_MAMBA_DIR / f"{date_str}_predictions.npz"
    if not pred_path.exists():
        return None
    d = np.load(pred_path, allow_pickle=True)
    preds = d["predictions"]  # (n, 3) for 1s/5s/10s
    n = len(preds)
    event_indices = CNN_WINDOW + np.arange(n) * CNN_STRIDE
    
    # Load mid prices from events file
    
    # mid_t_ticks is typically column 0 or we can derive from bid/ask
    # Let's use the relabel file for mid prices (more reliable)
    rl_path = RELABEL_DIR / f"mfe_mae_h10s_{date_str}.parquet"
    if not rl_path.exists():
        rl_path = RELABEL_DIR / f"mfe_mae_h30s_{date_str}.parquet"
    if not rl_path.exists():
        return None
    rl = pd.read_parquet(rl_path, columns=["mid_t_ticks", "ts_ns"])
    mid_prices = rl["mid_t_ticks"].values.astype(np.float64)
    ts_ns = rl["ts_ns"].values
    
    # Ensure event indices are within bounds
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


def simulate_day(data: dict, params: dict) -> list[dict]:
    """Simulate trades for one day."""
    trades = []
    date_str = data["date"]
    event_indices = data["event_indices"]
    preds_10s = data["preds_10s"]
    preds_5s = data["preds_5s"]
    mid_prices = data["mid_prices"]
    ts_ns = data["ts_ns"]
    n_preds = len(preds_10s)
    
    conf_pctile = params["confidence_pctile"]
    hold_events = params["hold_events"]
    side_filter = params["side"]
    hold_mode = params["hold_mode"]
    
    # Compute confidence threshold (based on abs(pred))
    abs_preds = np.abs(preds_10s)
    conf_threshold = np.percentile(abs_preds, conf_pctile)
    
    last_exit_idx = -MIN_EVENTS_BETWEEN
    
    for i in range(n_preds):
        ei = event_indices[i]
        
        if ei - last_exit_idx < MIN_EVENTS_BETWEEN:
            continue
        
        p10 = preds_10s[i]
        
        # Confidence gate: abs prediction above threshold
        if abs(p10) < conf_threshold:
            continue
        
        # Direction
        direction = 1 if p10 > 0 else -1
        
        # Side filter
        if side_filter == "short_only" and direction == 1:
            continue
        if side_filter == "long_only" and direction == -1:
            continue
        
        # Optional: 5s agreement
        p5 = preds_5s[i]
        if np.sign(p5) != np.sign(p10):
            continue  # skip when 5s and 10s disagree
        
        # Passive fill
        entry_mid = mid_prices[ei]
        fill_eidx = None
        end_fill = min(ei + PASSIVE_FILL_WINDOW, len(mid_prices))
        
        if direction == 1:
            limit_price = entry_mid - 0.5
            for fi in range(ei + 1, end_fill):
                if mid_prices[fi] <= limit_price:
                    fill_eidx = fi
                    break
        else:
            limit_price = entry_mid + 0.5
            for fi in range(ei + 1, end_fill):
                if mid_prices[fi] >= limit_price:
                    fill_eidx = fi
                    break
        
        if fill_eidx is None:
            continue
        
        entry_price = mid_prices[fill_eidx]
        entry_ts = ts_ns[fill_eidx]
        
        # Hold and exit
        max_exit_eidx = min(fill_eidx + hold_events, len(mid_prices) - 1)
        
        exit_eidx = max_exit_eidx
        exit_reason = "max_hold"
        max_favorable = 0.0
        max_adverse = 0.0
        
        if hold_mode == "trailing":
            # Trail with 50% giveback
            for j in range(fill_eidx + 1, max_exit_eidx + 1):
                pnl = (mid_prices[j] - entry_price) * direction
                if pnl > max_favorable:
                    max_favorable = pnl
                if pnl < max_adverse:
                    max_adverse = pnl
                # Hard stop at -4 ticks
                if pnl < -4.0:
                    exit_eidx = j
                    exit_reason = "hard_stop"
                    break
                # Trail: give back 50% of MFE when MFE > 2
                if max_favorable > 2.0 and pnl < max_favorable * 0.5:
                    exit_eidx = j
                    exit_reason = "trailing"
                    break
            else:
                # Compute MFE/MAE for max_hold case
                for j in range(fill_eidx + 1, max_exit_eidx + 1):
                    pnl = (mid_prices[j] - entry_price) * direction
                    if pnl > max_favorable:
                        max_favorable = pnl
                    if pnl < max_adverse:
                        max_adverse = pnl
        else:
            # Fixed hold — just compute MFE/MAE
            for j in range(fill_eidx + 1, max_exit_eidx + 1):
                pnl = (mid_prices[j] - entry_price) * direction
                if pnl > max_favorable:
                    max_favorable = pnl
                if pnl < max_adverse:
                    max_adverse = pnl
        
        exit_price = mid_prices[exit_eidx]
        exit_ts = ts_ns[exit_eidx]
        gross_ticks = (exit_price - entry_price) * direction
        net_ticks = gross_ticks - TOTAL_RT_COST
        hold_time_s = (exit_ts - entry_ts) / 1e9
        
        trades.append({
            "date": date_str,
            "direction": direction,
            "entry_price_t": float(entry_price),
            "exit_price_t": float(exit_price),
            "gross_ticks": float(gross_ticks),
            "net_ticks": float(net_ticks),
            "mfe_ticks": float(max_favorable),
            "mae_ticks": float(max_adverse),
            "hold_events": exit_eidx - fill_eidx,
            "hold_time_s": float(hold_time_s),
            "exit_reason": exit_reason,
            "entry_pred_10s": float(p10),
            "entry_pred_5s": float(p5),
            "confidence": float(abs(p10)),
        })
        
        last_exit_idx = exit_eidx
    
    return trades


def compute_metrics(trades_df: pd.DataFrame) -> dict:
    if len(trades_df) == 0:
        return {"n_trades": 0, "total_net_ticks": 0, "sharpe_annualized": 0,
                "win_rate": 0, "profit_factor": 0, "sortino_annualized": 0}
    net = trades_df["net_ticks"].values
    n = len(net)
    total_net = float(net.sum())
    wr = (net > 0).mean()
    gp = float(net[net > 0].sum()) if (net > 0).any() else 0
    gl = float(abs(net[net < 0].sum())) if (net < 0).any() else 1e-9
    pf = gp / gl
    daily = trades_df.groupby("date")["net_ticks"].sum()
    if len(daily) > 1:
        dm, ds = daily.mean(), daily.std()
        sharpe = (dm / ds) * np.sqrt(252) if ds > 0 else 0
        dd = daily[daily < 0]
        dds = dd.std() if len(dd) > 1 else ds
        sortino = (dm / dds) * np.sqrt(252) if dds > 0 else 0
    else:
        sharpe = sortino = 0
    return {
        "n_trades": n, "n_days": len(daily),
        "trades_per_day": n / max(len(daily), 1),
        "total_net_ticks": total_net,
        "total_net_dollars": total_net * ES_TICK_VALUE,
        "win_rate": float(wr), "profit_factor": pf,
        "sharpe_annualized": sharpe, "sortino_annualized": sortino,
        "avg_win_ticks": float(net[net > 0].mean()) if (net > 0).any() else 0,
        "avg_loss_ticks": float(net[net < 0].mean()) if (net < 0).any() else 0,
        "avg_mfe": float(trades_df["mfe_ticks"].mean()),
        "avg_mae": float(trades_df["mae_ticks"].mean()),
        "avg_hold_s": float(trades_df["hold_time_s"].mean()),
    }


def main():
    t_start = time.time()
    print("=" * 70)
    print("CNN-MAMBA CONFIDENCE-GATED TRADE SIM — BASELINE EDGE TEST")
    print(f"Cost: {TOTAL_RT_COST:.3f}t RT (passive entry + market exit)")
    print("=" * 70)
    
    # Discover dates
    pred_files = sorted(CNN_MAMBA_DIR.glob("*_predictions.npz"))
    dates = [f.stem.replace("_predictions", "") for f in pred_files]
    print(f"\nCNN-Mamba prediction dates: {len(dates)} ({dates[0]} to {dates[-1]})")
    
    # Filter to dates with relabel data
    valid_dates = []
    for d in dates:
        if (RELABEL_DIR / f"mfe_mae_h10s_{d}.parquet").exists() or \
           (RELABEL_DIR / f"mfe_mae_h30s_{d}.parquet").exists():
            valid_dates.append(d)
    print(f"Dates with price data: {len(valid_dates)}")
    
    # Load all days
    print(f"\nLoading data...")
    day_data = {}
    for d in valid_dates:
        data = load_day(d)
        if data is not None:
            day_data[d] = data
            print(f"  {d}: {len(data['event_indices']):,} predictions")
    print(f"Loaded {len(day_data)} days")
    
    # Run sweep
    from itertools import product
    
    combos = list(product(CONFIDENCE_PCTILES, HOLD_EVENTS, SIDES, MAX_HOLD_MODES))
    print(f"\nSweeping {len(combos)} configurations...")
    
    results = []
    
    for idx, (conf_p, hold_e, side, hmode) in enumerate(combos):
        params = {
            "confidence_pctile": conf_p,
            "hold_events": hold_e,
            "side": side,
            "hold_mode": hmode,
        }
        
        all_trades = []
        for d, data in sorted(day_data.items()):
            trades = simulate_day(data, params)
            all_trades.extend(trades)
        
        if all_trades:
            tdf = pd.DataFrame(all_trades)
            m = compute_metrics(tdf)
        else:
            m = compute_metrics(pd.DataFrame())
        
        result = {**params, **m}
        results.append(result)
        
        marker = "+" if m["total_net_ticks"] > 0 else "-"
        if (idx + 1) % 10 == 0 or m["total_net_ticks"] > 0:
            print(f"  [{idx+1}/{len(combos)}] "
                  f"p{conf_p} h={hold_e:4d} {side:10s} {hmode:8s} | "
                  f"n={m['n_trades']:4d} WR={m['win_rate']:.1%} "
                  f"net={m['total_net_ticks']:+8.1f}t "
                  f"Sharpe={m['sharpe_annualized']:+6.2f} "
                  f"PF={m['profit_factor']:.2f} {marker}")
    
    # Save all results
    rdf = pd.DataFrame(results)
    rdf.to_csv(OUTPUT_DIR / "sweep_results.csv", index=False)
    
    # Find best configurations
    print(f"\n{'='*70}")
    print("TOP 10 CONFIGURATIONS (by Sharpe, min 30 trades):")
    print(f"{'='*70}")
    
    valid = rdf[rdf["n_trades"] >= 30].copy()
    if len(valid) > 0:
        top10 = valid.nlargest(10, "sharpe_annualized")
        for _, row in top10.iterrows():
            marker = "+" if row["total_net_ticks"] > 0 else "-"
            print(f"  p{int(row['confidence_pctile'])} "
                  f"h={int(row['hold_events']):4d} "
                  f"{row['side']:10s} {row['hold_mode']:8s} | "
                  f"n={int(row['n_trades']):4d} "
                  f"WR={row['win_rate']:.1%} "
                  f"PF={row['profit_factor']:.2f} "
                  f"Sharpe={row['sharpe_annualized']:+.2f} "
                  f"Sortino={row['sortino_annualized']:+.2f} "
                  f"net={row['total_net_ticks']:+.1f}t "
                  f"(${row['total_net_dollars']:+,.0f}) {marker}")
        
        # Detailed output for THE best
        best = top10.iloc[0]
        print(f"\n{'='*70}")
        print(f"BEST CONFIG DETAIL:")
        print(f"  Confidence: top {100-best['confidence_pctile']:.0f}%")
        print(f"  Hold: {int(best['hold_events'])} events ({best['hold_mode']})")
        print(f"  Side: {best['side']}")
        print(f"  Trades: {int(best['n_trades'])} over {int(best['n_days'])} days ({best['trades_per_day']:.1f}/day)")
        print(f"  WR: {best['win_rate']:.1%} | PF: {best['profit_factor']:.2f}")
        print(f"  Sharpe: {best['sharpe_annualized']:+.2f} | Sortino: {best['sortino_annualized']:+.2f}")
        print(f"  Net: {best['total_net_ticks']:+.1f} ticks (${best['total_net_dollars']:+,.2f})")
        print(f"  Avg win: {best['avg_win_ticks']:+.2f}t | Avg loss: {best['avg_loss_ticks']:+.2f}t")
        print(f"  Avg MFE: {best['avg_mfe']:.2f}t | Avg MAE: {best['avg_mae']:.2f}t")
        print(f"  Avg hold: {best['avg_hold_s']:.1f}s")
        
        # Also show: best profitable config if any
        profitable = valid[valid["total_net_ticks"] > 0]
        if len(profitable) > 0:
            print(f"\n{'='*70}")
            print(f"ALL PROFITABLE CONFIGURATIONS ({len(profitable)}):")
            for _, row in profitable.nlargest(20, "sharpe_annualized").iterrows():
                print(f"  p{int(row['confidence_pctile'])} "
                      f"h={int(row['hold_events']):4d} "
                      f"{row['side']:10s} {row['hold_mode']:8s} | "
                      f"n={int(row['n_trades']):4d} WR={row['win_rate']:.1%} "
                      f"PF={row['profit_factor']:.2f} Sharpe={row['sharpe_annualized']:+.2f} "
                      f"net={row['total_net_ticks']:+.1f}t")
        else:
            print(f"\n  ⚠️ NO PROFITABLE CONFIGURATIONS FOUND across {len(combos)} combos")
            print(f"  This means CNN-Mamba v2 directional signal alone cannot overcome")
            print(f"  {TOTAL_RT_COST:.3f}t round-trip cost under realistic FIFO fill simulation.")
            
            # Check if gross is profitable
            gross_positive = valid[valid.get("total_net_ticks", 0) + valid["n_trades"] * TOTAL_RT_COST > 0] if "n_trades" in valid.columns else pd.DataFrame()
            # Actually let's compute gross profitability
            valid_copy = valid.copy()
            valid_copy["total_gross_approx"] = valid_copy["total_net_ticks"] + valid_copy["n_trades"] * TOTAL_RT_COST
            gross_prof = valid_copy[valid_copy["total_gross_approx"] > 0]
            if len(gross_prof) > 0:
                print(f"\n  But {len(gross_prof)} configs are GROSS-profitable (before costs):")
                for _, row in gross_prof.nlargest(5, "total_gross_approx").iterrows():
                    gt = row["total_gross_approx"]
                    cost = row["n_trades"] * TOTAL_RT_COST
                    print(f"    p{int(row['confidence_pctile'])} h={int(row['hold_events'])} "
                          f"{row['side']:10s} | gross={gt:+.1f}t costs={cost:.1f}t → net={row['total_net_ticks']:+.1f}t")
                print(f"\n  → Need passive exit (maker-maker) to be profitable: cost would be 0.752t instead of 1.752t")
    
    # Short vs Long analysis
    print(f"\n{'='*70}")
    print("SIDE COMPARISON (averaged across configs):")
    for side in SIDES:
        subset = rdf[rdf["side"] == side]
        if len(subset) > 0:
            avg_net = subset["total_net_ticks"].mean()
            avg_wr = subset["win_rate"].mean()
            avg_trades = subset["n_trades"].mean()
            print(f"  {side:10s}: avg_net={avg_net:+.1f}t | avg_WR={avg_wr:.1%} | avg_trades={avg_trades:.0f}")
    
    # MLflow
    if HAS_MLFLOW:
        try:
            mlflow.set_tracking_uri(MLFLOW_URI)
            mlflow.set_experiment("cnn_confidence_sim")
            with mlflow.start_run(run_name="v1_sweep"):
                mlflow.log_param("n_configs", len(combos))
                mlflow.log_param("n_dates", len(day_data))
                mlflow.log_param("cost_rt", TOTAL_RT_COST)
                if len(valid) > 0:
                    best = valid.nlargest(1, "sharpe_annualized").iloc[0]
                    mlflow.log_metric("best_sharpe", best["sharpe_annualized"])
                    mlflow.log_metric("best_net_ticks", best["total_net_ticks"])
                    mlflow.log_metric("best_wr", best["win_rate"])
                mlflow.log_artifact(str(OUTPUT_DIR / "sweep_results.csv"))
        except Exception as e:
            print(f"MLflow error: {e}")
    
    elapsed = time.time() - t_start
    print(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f}m)")


if __name__ == "__main__":
    main()
