"""
Streaming Trade Simulator v2 — CNN-Mamba Direction + Intensity Gate
===================================================================
Key improvement over v1: direction comes from CNN-Mamba v2 predictions
(proven IC_1s=0.222, IC_10s=0.106) instead of crude order-flow heuristics.

Composition:
  - CNN-Mamba v2 (10s horizon): DIRECTION + confidence (sign = side, |pred| = strength)
  - Streaming continuation h=30s (Spearman 0.463): INTENSITY gate (how much edge)
  - Streaming continuation h=60s (Spearman 0.408): HOLD confirmation

Entry:  CNN-Mamba direction confident + streaming intensity top N%
Hold:   intensity stays above threshold + direction doesn't flip
Exit:   intensity fades OR direction reverses OR trailing stop OR max hold

FIFO Cost Model:
  - Passive entry: 0.376 ticks (commission)
  - Market exit: 1.376 ticks (commission + spread)
  - Total RT: 1.752 ticks
"""
from __future__ import annotations

import gc
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

# ─── Paths ────────────────────────────────────────────────────────────────
DATA_ROOT = Path("/home/nick/Lvl3Quant/data")
PRED_DIR = DATA_ROOT / "models/streaming_continuation_v1"
RELABEL_DIR = DATA_ROOT / "relabel"
CNN_MAMBA_DIR = Path("/home/nick/Lvl3Quant/output/cnn_mamba_v2_bulk_oot")
EVENTS_DIR = DATA_ROOT / "processed/mbo_events_smart_v3"
BOOK_DIR = DATA_ROOT / "processed/mbo_book_features"
OUTPUT_DIR = Path("/home/nick/Lvl3Quant/output/streaming_trade_sim_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MLFLOW_URI = "http://localhost:5000"
EXPERIMENT_NAME = "streaming_trade_sim"

# ─── Cost Constants (ES Futures, AMP/Rithmic) ────────────────────────────
ES_TICK_VALUE = 12.50
COMMISSION_TICKS = 0.376
SPREAD_TICKS = 1.0
PASSIVE_ENTRY_COST = COMMISSION_TICKS
MARKET_EXIT_COST = COMMISSION_TICKS + SPREAD_TICKS
TOTAL_RT_COST = PASSIVE_ENTRY_COST + MARKET_EXIT_COST  # 1.752 ticks

# ─── CNN-Mamba v2 config ─────────────────────────────────────────────────
CNN_WINDOW = 3000
CNN_STRIDE = 250
CNN_HORIZON_IDX = 2  # index 2 = 10s horizon (strongest edge vs cost)

# ─── OOS Subsampling (must match training code) ──────────────────────────
OOT_MAX_EVENTS = 500_000

# ─── Sweep parameters (will grid-search) ─────────────────────────────────
PARAM_GRID = {
    "intensity_pctile": [80, 85, 90, 95],     # top N% intensity required
    "cnn_abs_threshold": [0.1, 0.15, 0.2, 0.3],  # |cnn_pred| minimum for direction confidence
    "max_hold_events": [600, 1200],           # max hold period
    "reversal_stop_ticks": [3.0, 4.0, 6.0],
    "trailing_exit_frac": [0.4, 0.5, 0.6],
}

# Default params (used for single-run mode)
DEFAULT_PARAMS = {
    "intensity_pctile": 90,
    "cnn_abs_threshold": 0.15,
    "max_hold_events": 1200,
    "reversal_stop_ticks": 4.0,
    "trailing_exit_frac": 0.5,
    "min_events_between": 200,
    "min_hold_events": 20,
    "passive_fill_window": 40,
    "intensity_exit_pctile": 30,   # exit when intensity drops below this
    "intensity_hold_pctile": 50,   # hold while above this
}


def replay_subsampling(date_str: str, horizon_s: int) -> np.ndarray:
    """Replay the exact subsampling used during OOT prediction."""
    rl_path = RELABEL_DIR / f"mfe_mae_h{horizon_s}s_{date_str}.parquet"
    if not rl_path.exists():
        return np.array([], dtype=np.int64)
    rl = pd.read_parquet(rl_path, columns=["mfe_minus_mae_ticks"])
    target = rl["mfe_minus_mae_ticks"].values
    valid_idx = np.where(~np.isnan(target))[0]
    if len(valid_idx) == 0:
        return np.array([], dtype=np.int64)
    if len(valid_idx) > OOT_MAX_EVENTS:
        rng = np.random.default_rng(hash(date_str) % 2**32)
        chosen = rng.choice(valid_idx, OOT_MAX_EVENTS, replace=False)
        chosen.sort()
    else:
        chosen = valid_idx
    return chosen


def load_cnn_mamba_predictions(date_str: str) -> Optional[dict]:
    """Load CNN-Mamba v2 directional predictions for a date.
    Returns dict with 'event_indices' and 'predictions' (shape [n, 3])."""
    pred_path = CNN_MAMBA_DIR / f"{date_str}_predictions.npz"
    if not pred_path.exists():
        return None
    d = np.load(pred_path, allow_pickle=True)
    preds = d["predictions"]  # (n_windows, 3) for 1s/5s/10s
    n_windows = len(preds)
    # CNN-Mamba event indices: window_start at CNN_WINDOW + i * CNN_STRIDE
    event_indices = CNN_WINDOW + np.arange(n_windows) * CNN_STRIDE
    return {"event_indices": event_indices, "predictions": preds}


def load_day_data(date_str: str) -> Optional[dict]:
    """Load all prediction sources + price data for one day."""
    # 1. Streaming continuation predictions (intensity)
    horizon_preds = {}
    horizon_indices = {}
    for h in [30, 60, 120]:
        pred_path = PRED_DIR / f"oos_preds_h{h}s.parquet"
        if not pred_path.exists():
            continue
        preds_all = pd.read_parquet(pred_path)
        day_preds = preds_all[preds_all["date"] == date_str]
        if len(day_preds) == 0:
            del preds_all
            continue
        indices = replay_subsampling(date_str, h)
        if len(indices) != len(day_preds):
            print(f"  WARNING: {date_str} h={h}s index mismatch ({len(indices)} vs {len(day_preds)})")
            del preds_all
            continue
        horizon_preds[h] = day_preds["pred"].values
        horizon_indices[h] = indices
        del preds_all, day_preds
        gc.collect()

    if 30 not in horizon_preds:
        return None

    # 2. CNN-Mamba v2 directional predictions
    cnn = load_cnn_mamba_predictions(date_str)
    if cnn is None:
        return None

    # 3. Price data from relabel file
    rl_path = RELABEL_DIR / f"mfe_mae_h30s_{date_str}.parquet"
    if not rl_path.exists():
        return None
    rl = pd.read_parquet(rl_path, columns=["event_idx", "ts_ns", "mid_t_ticks"])

    # Build unified event-level DataFrame using h=30s base grid
    base_indices = horizon_indices[30]
    rl_subset = rl.iloc[base_indices].copy().reset_index(drop=True)

    # Map CNN-Mamba predictions to nearest base grid point
    cnn_idx = cnn["event_indices"]
    cnn_preds = cnn["predictions"]  # (n, 3)
    
    # For each base index, find nearest CNN-Mamba prediction
    # CNN predictions are on regular grid, so we can use searchsorted
    cnn_10s = np.full(len(base_indices), np.nan, dtype=np.float32)
    cnn_5s = np.full(len(base_indices), np.nan, dtype=np.float32)
    cnn_1s = np.full(len(base_indices), np.nan, dtype=np.float32)
    
    for i, bi in enumerate(base_indices):
        # Find nearest CNN prediction (within CNN_STRIDE/2 events)
        j = np.searchsorted(cnn_idx, bi)
        best_j = None
        best_dist = CNN_STRIDE  # max acceptable distance
        for candidate in [j-1, j]:
            if 0 <= candidate < len(cnn_idx):
                dist = abs(int(cnn_idx[candidate]) - int(bi))
                if dist < best_dist:
                    best_dist = dist
                    best_j = candidate
        if best_j is not None:
            cnn_1s[i] = cnn_preds[best_j, 0]
            cnn_5s[i] = cnn_preds[best_j, 1]
            cnn_10s[i] = cnn_preds[best_j, 2]

    df = pd.DataFrame({
        "event_idx": rl_subset["event_idx"].values,
        "ts_ns": rl_subset["ts_ns"].values,
        "mid_t_ticks": rl_subset["mid_t_ticks"].values.astype(np.float64),
        "pred_30s": horizon_preds[30],
        "cnn_1s": cnn_1s,
        "cnn_5s": cnn_5s, 
        "cnn_10s": cnn_10s,
    })

    # Map other intensity horizons
    for h in [60, 120]:
        col = f"pred_{h}s"
        if h not in horizon_preds:
            df[col] = np.nan
            continue
        idx_to_pred = dict(zip(horizon_indices[h], horizon_preds[h]))
        df[col] = [idx_to_pred.get(ei, np.nan) for ei in base_indices]

    df = df.sort_values("event_idx").reset_index(drop=True)
    
    # Report coverage
    cnn_coverage = (~np.isnan(cnn_10s)).mean()
    print(f"  CNN-Mamba coverage: {cnn_coverage:.1%} of intensity grid points have directional predictions")
    
    del rl, cnn
    gc.collect()
    return df


def run_simulation(df: pd.DataFrame, date_str: str, params: dict) -> list[dict]:
    """Run streaming trade simulation with CNN-Mamba direction + intensity gate."""
    trades = []
    n = len(df)
    
    intensity_pctile = params["intensity_pctile"]
    cnn_threshold = params["cnn_abs_threshold"]
    max_hold = params["max_hold_events"]
    min_hold = params["min_hold_events"]
    stop_ticks = params["reversal_stop_ticks"]
    trailing_frac = params["trailing_exit_frac"]
    min_between = params["min_events_between"]
    passive_window = params["passive_fill_window"]
    exit_pctile = params["intensity_exit_pctile"]
    hold_pctile = params["intensity_hold_pctile"]
    
    # Extract arrays
    mid_prices = df["mid_t_ticks"].values
    pred_30 = df["pred_30s"].values
    pred_60 = df["pred_60s"].values if "pred_60s" in df.columns else np.full(n, np.nan)
    pred_120 = df["pred_120s"].values if "pred_120s" in df.columns else np.full(n, np.nan)
    cnn_10s = df["cnn_10s"].values
    cnn_5s = df["cnn_5s"].values
    ts_ns = df["ts_ns"].values
    
    # Compute per-day thresholds
    valid_30 = pred_30[~np.isnan(pred_30)]
    entry_thresh_30 = np.percentile(valid_30, intensity_pctile) if len(valid_30) > 0 else np.inf
    exit_thresh_30 = np.percentile(valid_30, exit_pctile) if len(valid_30) > 0 else -np.inf
    hold_thresh_30 = np.percentile(valid_30, hold_pctile) if len(valid_30) > 0 else -np.inf
    
    valid_60 = pred_60[~np.isnan(pred_60)]
    hold_thresh_60 = np.percentile(valid_60, hold_pctile) if len(valid_60) > 100 else -np.inf
    
    i = 0
    last_exit_idx = -min_between
    
    while i < n - max_hold:
        if i - last_exit_idx < min_between:
            i += 1
            continue
        
        # ── ENTRY GATE 1: High intensity predicted ──
        p30 = pred_30[i]
        if np.isnan(p30) or p30 < entry_thresh_30:
            i += 1
            continue
        
        # ── ENTRY GATE 2: CNN-Mamba direction is confident ──
        c10 = cnn_10s[i]
        c5 = cnn_5s[i]
        if np.isnan(c10):
            i += 1
            continue
        
        # Use 10s prediction for direction, require minimum confidence
        if abs(c10) < cnn_threshold:
            i += 1
            continue
        
        # Direction from CNN-Mamba (negative pred = price going down = SHORT)
        direction = 1 if c10 > 0 else -1
        
        # Optional: require 5s to agree with 10s for stronger confirmation
        if not np.isnan(c5) and np.sign(c5) != np.sign(c10):
            # 5s and 10s disagree — skip for safety
            i += 1
            continue
        
        # ── Passive Entry ──
        entry_mid = mid_prices[i]
        fill_idx = None
        end_fill = min(i + passive_window, n)
        
        if direction == 1:  # buy at bid
            limit_price = entry_mid - 0.5
            for fi in range(i + 1, end_fill):
                if mid_prices[fi] <= limit_price:
                    fill_idx = fi
                    break
        else:  # sell at ask
            limit_price = entry_mid + 0.5
            for fi in range(i + 1, end_fill):
                if mid_prices[fi] >= limit_price:
                    fill_idx = fi
                    break
        
        if fill_idx is None:
            i += 1
            continue
        
        entry_price = mid_prices[fill_idx]
        entry_ts = ts_ns[fill_idx]
        
        # ── Hold & Exit Loop ──
        j = fill_idx + min_hold
        exit_reason = "max_hold"
        max_favorable = 0.0
        max_adverse = 0.0
        
        while j < min(fill_idx + max_hold, n):
            current_price = mid_prices[j]
            pnl_ticks = (current_price - entry_price) * direction
            
            if pnl_ticks > max_favorable:
                max_favorable = pnl_ticks
            if pnl_ticks < max_adverse:
                max_adverse = pnl_ticks
            
            # Exit 1: Hard stop
            if pnl_ticks < -stop_ticks:
                exit_reason = "hard_stop"
                break
            
            # Exit 2: Trailing exit
            if max_favorable > 2.0 and pnl_ticks < max_favorable * trailing_frac:
                exit_reason = "trailing"
                break
            
            # Exit 3: Intensity fade (h=30s below exit threshold)
            p30_j = pred_30[j]
            if not np.isnan(p30_j) and p30_j < exit_thresh_30:
                exit_reason = "fade_30s"
                break
            
            # Exit 4: Multi-horizon intensity fade
            p60_j = pred_60[j]
            p120_j = pred_120[j]
            if (not np.isnan(p60_j) and p60_j < hold_thresh_60 and
                not np.isnan(p30_j) and p30_j < hold_thresh_30):
                exit_reason = "multi_h_fade"
                break
            
            # Exit 5: CNN-Mamba direction reversal (direction flipped)
            c10_j = cnn_10s[j]
            if not np.isnan(c10_j):
                if direction == 1 and c10_j < -cnn_threshold:
                    exit_reason = "cnn_reversal"
                    break
                if direction == -1 and c10_j > cnn_threshold:
                    exit_reason = "cnn_reversal"
                    break
            
            j += 1
        
        exit_idx = min(j, n - 1)
        exit_price = mid_prices[exit_idx]
        exit_ts = ts_ns[exit_idx]
        gross_ticks = (exit_price - entry_price) * direction
        net_ticks = gross_ticks - TOTAL_RT_COST
        hold_time_s = (exit_ts - entry_ts) / 1e9
        
        trades.append({
            "date": date_str,
            "direction": direction,
            "entry_idx": int(fill_idx),
            "exit_idx": int(exit_idx),
            "entry_price_t": float(entry_price),
            "exit_price_t": float(exit_price),
            "gross_ticks": float(gross_ticks),
            "net_ticks": float(net_ticks),
            "mfe_ticks": float(max_favorable),
            "mae_ticks": float(max_adverse),
            "hold_events": exit_idx - fill_idx,
            "hold_time_s": float(hold_time_s),
            "exit_reason": exit_reason,
            "entry_pred_30s": float(pred_30[i]),
            "entry_cnn_10s": float(c10),
            "entry_cnn_5s": float(c5) if not np.isnan(c5) else None,
        })
        
        last_exit_idx = exit_idx
        i = exit_idx + 1
    
    return trades


def compute_metrics(trades_df: pd.DataFrame) -> dict:
    """Compute risk-adjusted performance metrics."""
    if len(trades_df) == 0:
        return {"n_trades": 0}
    net = trades_df["net_ticks"].values
    gross = trades_df["gross_ticks"].values
    n_trades = len(net)
    total_gross = float(gross.sum())
    total_net = float(net.sum())
    winners = (net > 0).sum()
    losers = (net < 0).sum()
    win_rate = winners / n_trades
    gross_profit = float(net[net > 0].sum()) if winners > 0 else 0
    gross_loss = float(abs(net[net < 0].sum())) if losers > 0 else 1e-9
    profit_factor = gross_profit / gross_loss
    daily_pnl = trades_df.groupby("date")["net_ticks"].sum()
    if len(daily_pnl) > 1:
        daily_mean = daily_pnl.mean()
        daily_std = daily_pnl.std()
        sharpe = (daily_mean / daily_std) * np.sqrt(252) if daily_std > 0 else 0
        downside_returns = daily_pnl[daily_pnl < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 1 else daily_std
        sortino = (daily_mean / downside_std) * np.sqrt(252) if downside_std > 0 else 0
    else:
        sharpe = sortino = 0
    avg_win = float(net[net > 0].mean()) if winners > 0 else 0
    avg_loss = float(net[net < 0].mean()) if losers > 0 else 0
    return {
        "n_trades": n_trades, "n_days": len(daily_pnl),
        "trades_per_day": n_trades / max(len(daily_pnl), 1),
        "total_gross_ticks": total_gross, "total_net_ticks": total_net,
        "win_rate": win_rate, "profit_factor": profit_factor,
        "sharpe_annualized": sharpe, "sortino_annualized": sortino,
        "avg_win_ticks": avg_win, "avg_loss_ticks": avg_loss,
        "avg_hold_s": float(trades_df["hold_time_s"].mean()),
        "avg_mfe_ticks": float(trades_df["mfe_ticks"].mean()),
        "avg_mae_ticks": float(trades_df["mae_ticks"].mean()),
        "total_net_dollars": total_net * ES_TICK_VALUE,
    }


def regime_analysis(trades_df: pd.DataFrame) -> dict:
    """Regime-agnostic validation per HC #428."""
    if len(trades_df) == 0:
        return {}
    trades_df = trades_df.copy()
    trades_df["month"] = trades_df["date"].str[:6]
    regime_metrics = {}
    for m in sorted(trades_df["month"].unique()):
        regime_metrics[m] = compute_metrics(trades_df[trades_df["month"] == m])
    sharpes = {m: v.get("sharpe_annualized", 0) for m, v in regime_metrics.items()
               if isinstance(v, dict) and v.get("n_trades", 0) > 5}
    if len(sharpes) >= 2:
        vals = list(sharpes.values())
        max_abs = max(abs(s) for s in vals) if vals else 1
        divergence = (max(vals) - min(vals)) / max_abs if max_abs > 0 else 0
        regime_metrics["_divergence"] = divergence
        regime_metrics["_pass_hc428"] = str(divergence <= 0.50)
    return regime_metrics


def main():
    t_start = time.time()
    print("=" * 70)
    print("STREAMING TRADE SIMULATOR v2 — CNN-Mamba DIRECTION + INTENSITY GATE")
    print("=" * 70)
    
    params = DEFAULT_PARAMS.copy()
    
    # Check for sweep mode
    sweep_mode = "--sweep" in sys.argv
    
    # Discover overlapping dates (both CNN-Mamba and streaming preds available)
    preds_30 = pd.read_parquet(PRED_DIR / "oos_preds_h30s.parquet")
    intensity_dates = set(preds_30["date"].unique())
    del preds_30; gc.collect()
    
    cnn_dates = set()
    for f in CNN_MAMBA_DIR.glob("*_predictions.npz"):
        cnn_dates.add(f.stem.replace("_predictions", ""))
    
    overlap_dates = sorted(intensity_dates & cnn_dates)
    print(f"\nIntensity prediction dates: {len(intensity_dates)}")
    print(f"CNN-Mamba prediction dates: {len(cnn_dates)}")
    print(f"Overlap dates: {len(overlap_dates)}")
    
    if not overlap_dates:
        print("ERROR: No overlapping dates between intensity and CNN-Mamba predictions.")
        return
    
    # Filter to dates with relabel data
    overlap_dates = [d for d in overlap_dates 
                     if (RELABEL_DIR / f"mfe_mae_h30s_{d}.parquet").exists()]
    print(f"Dates with relabel data: {len(overlap_dates)}")
    print(f"Date range: {overlap_dates[0]} to {overlap_dates[-1]}")
    
    # Load all days (reused across sweep)
    print(f"\nLoading {len(overlap_dates)} days of data...")
    day_data = {}
    for date_str in overlap_dates:
        print(f"  {date_str}...", end=" ", flush=True)
        df = load_day_data(date_str)
        if df is not None:
            day_data[date_str] = df
            print(f"OK ({len(df):,} events)")
        else:
            print("SKIP")
        gc.collect()
    
    print(f"\nLoaded {len(day_data)} days successfully")
    
    if sweep_mode:
        # Grid search over key parameters
        from itertools import product
        
        grid_keys = ["intensity_pctile", "cnn_abs_threshold", "reversal_stop_ticks"]
        grid_vals = [PARAM_GRID[k] for k in grid_keys]
        combos = list(product(*grid_vals))
        print(f"\n{'='*70}")
        print(f"SWEEP MODE: {len(combos)} parameter combinations")
        print(f"{'='*70}")
        
        results_list = []
        
        for combo_idx, combo in enumerate(combos):
            p = params.copy()
            for k, v in zip(grid_keys, combo):
                p[k] = v
            
            all_trades = []
            for date_str, df in sorted(day_data.items()):
                trades = run_simulation(df, date_str, p)
                all_trades.extend(trades)
            
            if all_trades:
                tdf = pd.DataFrame(all_trades)
                m = compute_metrics(tdf)
            else:
                m = {"n_trades": 0, "total_net_ticks": 0, "sharpe_annualized": 0,
                     "win_rate": 0, "profit_factor": 0}
            
            result = {**{k: v for k, v in zip(grid_keys, combo)}, **m}
            results_list.append(result)
            
            marker = "+" if m.get("total_net_ticks", 0) > 0 else "-"
            print(f"  [{combo_idx+1}/{len(combos)}] "
                  f"int_p={combo[0]} cnn_t={combo[1]:.2f} stop={combo[2]:.1f} | "
                  f"n={m.get('n_trades',0):4d} WR={m.get('win_rate',0):.1%} "
                  f"net={m.get('total_net_ticks',0):+.1f}t "
                  f"Sharpe={m.get('sharpe_annualized',0):+.2f} "
                  f"PF={m.get('profit_factor',0):.2f} {marker}")
        
        # Save sweep results
        sweep_df = pd.DataFrame(results_list)
        sweep_df.to_csv(OUTPUT_DIR / "sweep_results.csv", index=False)
        
        # Find best by Sharpe
        if len(sweep_df) > 0 and sweep_df["n_trades"].max() > 0:
            valid = sweep_df[sweep_df["n_trades"] >= 50]
            if len(valid) > 0:
                best = valid.loc[valid["sharpe_annualized"].idxmax()]
                print(f"\n{'='*70}")
                print(f"BEST CONFIG (by Sharpe, min 50 trades):")
                for k in grid_keys:
                    print(f"  {k}: {best[k]}")
                print(f"  Trades: {int(best['n_trades'])} | WR: {best['win_rate']:.1%}")
                print(f"  Sharpe: {best['sharpe_annualized']:+.2f} | Sortino: {best.get('sortino_annualized',0):+.2f}")
                print(f"  PF: {best['profit_factor']:.2f} | Net: {best['total_net_ticks']:+.1f} ticks")
                print(f"  Net $: ${best['total_net_ticks'] * ES_TICK_VALUE:+,.2f}")
                print(f"{'='*70}")
                
                # Run best config with full output
                best_params = params.copy()
                for k in grid_keys:
                    best_params[k] = best[k]
                
                all_trades = []
                for date_str, df in sorted(day_data.items()):
                    trades = run_simulation(df, date_str, best_params)
                    all_trades.extend(trades)
                
                if all_trades:
                    trades_df = pd.DataFrame(all_trades)
                    trades_df.to_parquet(OUTPUT_DIR / "best_trades.parquet", index=False)
                    trades_df.to_csv(OUTPUT_DIR / "best_trades.csv", index=False)
                    
                    metrics = compute_metrics(trades_df)
                    regime = regime_analysis(trades_df)
                    
                    print(f"\nBEST CONFIG — DETAILED RESULTS:")
                    print_detailed(trades_df, metrics, regime)
                    
                    # Save results
                    with open(OUTPUT_DIR / "best_results.json", "w") as f:
                        json.dump({"params": best_params, "metrics": metrics,
                                   "regime": {k:v for k,v in regime.items() if not isinstance(v, dict) or k.startswith("_")},
                                   "regime_detail": {k:v for k,v in regime.items() if isinstance(v, dict) and not k.startswith("_")}
                                  }, f, indent=2, default=str)
    
    else:
        # Single run with default params
        print(f"\nParams: intensity_p={params['intensity_pctile']} "
              f"cnn_thresh={params['cnn_abs_threshold']} "
              f"stop={params['reversal_stop_ticks']}t")
        
        all_trades = []
        for date_str, df in sorted(day_data.items()):
            t_day = time.time()
            trades = run_simulation(df, date_str, params)
            all_trades.extend(trades)
            if trades:
                net = sum(t["net_ticks"] for t in trades)
                wr = sum(1 for t in trades if t["net_ticks"] > 0) / len(trades)
                nl = sum(1 for t in trades if t["direction"] == 1)
                ns = len(trades) - nl
                print(f"  {date_str}: {len(trades)} trades (L={nl} S={ns}) | "
                      f"net={net:+.1f}t | WR={wr:.1%} | {time.time()-t_day:.1f}s")
            else:
                print(f"  {date_str}: 0 trades | {time.time()-t_day:.1f}s")
        
        if not all_trades:
            print("\nNO TRADES GENERATED.")
            return
        
        trades_df = pd.DataFrame(all_trades)
        trades_df.to_parquet(OUTPUT_DIR / "trades_v2.parquet", index=False)
        trades_df.to_csv(OUTPUT_DIR / "trades_v2.csv", index=False)
        
        metrics = compute_metrics(trades_df)
        regime = regime_analysis(trades_df)
        print_detailed(trades_df, metrics, regime)
        
        with open(OUTPUT_DIR / "results_v2.json", "w") as f:
            json.dump({"params": params, "metrics": metrics,
                       "regime": {k:v for k,v in regime.items() if not isinstance(v, dict) or k.startswith("_")},
                       "regime_detail": {k:v for k,v in regime.items() if isinstance(v, dict) and not k.startswith("_")}
                      }, f, indent=2, default=str)
    
    # MLflow logging
    if HAS_MLFLOW:
        try:
            mlflow.set_tracking_uri(MLFLOW_URI)
            mlflow.set_experiment(EXPERIMENT_NAME)
            run_name = "v2_sweep" if sweep_mode else "v2_default"
            with mlflow.start_run(run_name=run_name):
                mlflow.log_params({k: str(v) for k, v in params.items()})
                if 'metrics' in dir():
                    for k, v in metrics.items():
                        if isinstance(v, (int, float)):
                            mlflow.log_metric(k, v)
                for f in OUTPUT_DIR.glob("*.json"):
                    mlflow.log_artifact(str(f))
                for f in OUTPUT_DIR.glob("*.csv"):
                    mlflow.log_artifact(str(f))
        except Exception as e:
            print(f"MLflow error: {e}")
    
    elapsed = time.time() - t_start
    print(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f}m)")


def print_detailed(trades_df, metrics, regime):
    """Print detailed results."""
    print(f"\n{'='*70}")
    print(f"STREAMING TRADE SIM v2 — CNN-Mamba Direction + Intensity Gate")
    print(f"{'='*70}")
    print(f"  Trades: {metrics['n_trades']} over {metrics.get('n_days',0)} days ({metrics.get('trades_per_day',0):.1f}/day)")
    print(f"  Win rate: {metrics.get('win_rate',0):.1%}")
    print(f"  Profit factor: {metrics.get('profit_factor',0):.2f}")
    print(f"  Sharpe: {metrics.get('sharpe_annualized',0):+.2f}")
    print(f"  Sortino: {metrics.get('sortino_annualized',0):+.2f}")
    print(f"  Net: {metrics.get('total_net_ticks',0):+.1f} ticks (${metrics.get('total_net_dollars',0):+,.2f})")
    print(f"  Avg win: {metrics.get('avg_win_ticks',0):+.2f}t | Avg loss: {metrics.get('avg_loss_ticks',0):+.2f}t")
    print(f"  Avg MFE: {metrics.get('avg_mfe_ticks',0):.2f}t | Avg MAE: {metrics.get('avg_mae_ticks',0):.2f}t")
    print(f"  Avg hold: {metrics.get('avg_hold_s',0):.1f}s")
    print(f"  Cost: {TOTAL_RT_COST:.3f}t RT × {metrics['n_trades']} = {metrics['n_trades']*TOTAL_RT_COST:.1f}t total")
    
    if len(trades_df) > 0:
        print(f"\n  Exit reasons:")
        for reason, count in trades_df["exit_reason"].value_counts().items():
            pct = count / len(trades_df)
            subset = trades_df[trades_df["exit_reason"] == reason]
            avg_net = subset["net_ticks"].mean()
            wr = (subset["net_ticks"] > 0).mean()
            print(f"    {reason:20s}: {count:4d} ({pct:.1%}) | WR={wr:.1%} | net={avg_net:+.2f}t")
        
        print(f"\n  Direction breakdown:")
        for d, label in [(1, "LONG"), (-1, "SHORT")]:
            subset = trades_df[trades_df["direction"] == d]
            if len(subset) > 0:
                wr = (subset["net_ticks"] > 0).mean()
                avg = subset["net_ticks"].mean()
                total = subset["net_ticks"].sum()
                print(f"    {label}: {len(subset)} trades | WR={wr:.1%} | avg={avg:+.2f}t | total={total:+.1f}t")
        
        print(f"\n  Regime analysis:")
        for m, v in sorted(regime.items()):
            if isinstance(v, dict) and not m.startswith("_") and v.get("n_trades", 0) > 0:
                print(f"    {m}: {v['n_trades']:4d} trades | WR={v.get('win_rate',0):.1%} | "
                      f"Sharpe={v.get('sharpe_annualized',0):+.2f} | PF={v.get('profit_factor',0):.2f}")
        
        div = regime.get("_divergence")
        if div is not None:
            status = "PASS" if regime.get("_pass_hc428") in [True, "True"] else "FAIL"
            print(f"    Divergence: {div:.2f} ({status}, threshold 0.50)")
        
        print(f"\n  Per-day P&L:")
        daily = trades_df.groupby("date").agg(
            n=("net_ticks", "count"), net=("net_ticks", "sum"),
            wr=("net_ticks", lambda x: (x > 0).mean()))
        for dt, row in daily.iterrows():
            m = "+" if row["net"] > 0 else "-"
            print(f"    {dt}: {int(row['n']):3d} trades | net={row['net']:+7.1f}t | WR={row['wr']:.0%} {m}")
    
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
