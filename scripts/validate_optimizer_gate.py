#!/usr/bin/env python3
"""
Validate Passive Execution Optimizer as a GATE via FIFO Fill Sim
================================================================
Compares unfiltered vs optimizer-gated signal predictions through
the Rust fill_sim_cli (FIFO market replay).
"""

import os
import sys
import json
import subprocess
import datetime
import numpy as np
import pandas as pd
from pathlib import Path

# ============ PATHS ============
BASE = Path("/home/nick/Lvl3Quant")
CNN_PRED_DIR = BASE / "output" / "cnn_mamba_v2_bulk_oot"
EVENTS_DIR = BASE / "data" / "processed" / "mbo_events_smart_v3"
FEATURES_DIR = BASE / "output" / "queue_augmented_features"
LABELS_DIR = BASE / "output" / "mbo_walker_labels"
OPT_PREDS = BASE / "output" / "passive_exec_optimizer_v1" / "oot_predictions.npz"
OPT_RESULTS = BASE / "output" / "passive_exec_optimizer_v1" / "results.json"
MBO_DIR = BASE / "data" / "raw" / "mbo"
FILL_SIM = BASE / "rust_cache_builder" / "target" / "release" / "fill_sim_cli"
OUTPUT_DIR = BASE / "output" / "optimizer_gate_validation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_RTH_BARS = 234000

FILLSIM_CONFIGS = {
    "passive_tp8_hold30m": {
        "signal_threshold": 0.3,
        "take_profit_ticks": 8,
        "stop_loss_ticks": 16,
        "hold_ms": 1800000,
        "latency_ms": 10,
    },
    "passive_tp13_hold2h": {
        "signal_threshold": 0.3,
        "take_profit_ticks": 13,
        "stop_loss_ticks": 20,
        "hold_ms": 7200000,
        "latency_ms": 10,
    },
    "passive_tp6_scalp5m": {
        "signal_threshold": 0.3,
        "take_profit_ticks": 6,
        "stop_loss_ticks": 12,
        "hold_ms": 300000,
        "latency_ms": 10,
    },
}


def get_rth_bounds(date_str):
    y, m, d = int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8])
    rth_start = datetime.datetime(y, m, d, 13, 30, 0)
    rth_end = datetime.datetime(y, m, d, 20, 0, 0)
    return int(rth_start.timestamp() * 1e9), int(rth_end.timestamp() * 1e9)


def cnn_event_preds_to_bars(date_str):
    """Convert CNN-Mamba event-window predictions to bar-level (100ms) predictions."""
    cnn_path = CNN_PRED_DIR / f"{date_str}_predictions.npz"
    events_path = EVENTS_DIR / f"{date_str}_mbo_events.npz"

    if not cnn_path.exists() or not events_path.exists():
        return None

    d = np.load(str(cnn_path))
    preds_3h = d['predictions']
    stride = int(d['stride'])
    ws = int(d['window_size'])

    ev = np.load(str(events_path))
    ts = ev['timestamps']

    pred_10s = preds_3h[:, 2]  # 10s horizon
    n_win = len(pred_10s)

    # Prediction timestamp = ts at end of window
    pred_ts = np.array([ts[min(i * stride + ws - 1, len(ts) - 1)] for i in range(n_win)])

    rth_start, rth_end = get_rth_bounds(date_str)
    rth_mask = (pred_ts >= rth_start) & (pred_ts < rth_end)
    rth_pred_ts = pred_ts[rth_mask]
    rth_pred_vals = pred_10s[rth_mask]

    bar_indices = ((rth_pred_ts - rth_start) // 100_000_000).astype(int)

    bar_preds = np.zeros(N_RTH_BARS, dtype=np.float32)
    for idx, val in zip(bar_indices, rth_pred_vals):
        if 0 <= idx < N_RTH_BARS:
            bar_preds[idx] = val

    # Forward-fill
    last_val = np.float32(0.0)
    for i in range(N_RTH_BARS):
        if bar_preds[i] != 0:
            last_val = bar_preds[i]
        else:
            bar_preds[i] = last_val

    return bar_preds, rth_start


def get_optimizer_gate_bars(date_str, opt_scores, threshold, rth_start):
    """Get bar-level gate mask using merged features+labels (matching optimizer training)."""
    feat_path = FEATURES_DIR / f"features_{date_str}.parquet"
    lab_path = LABELS_DIR / f"labels_{date_str}.parquet"
    if not feat_path.exists() or not lab_path.exists():
        return None

    feat = pd.read_parquet(str(feat_path))
    lab = pd.read_parquet(str(lab_path))

    # Replicate the optimizer's data loading: inner join on event_id
    merged = feat.merge(lab[['event_id']], on='event_id', how='inner')

    if len(merged) != len(opt_scores):
        print(f"  WARNING {date_str}: merged={len(merged)} vs opt_scores={len(opt_scores)}, skipping")
        return None

    ts = merged['ts_ns'].values
    bar_indices = ((ts - rth_start) // 100_000_000).astype(int)
    rth_mask = (bar_indices >= 0) & (bar_indices < N_RTH_BARS)

    keep_mask = np.ones(N_RTH_BARS, dtype=bool)

    rth_bars = bar_indices[rth_mask]
    rth_scores = opt_scores[rth_mask]
    below_thresh = rth_scores < threshold

    for bar, is_bad in zip(rth_bars, below_thresh):
        if is_bad and 0 <= bar < N_RTH_BARS:
            keep_mask[max(0, bar):min(N_RTH_BARS, bar + 3)] = False

    return keep_mask


def run_fillsim(mbo_path, pred_npz_path, output_path, config):
    cmd = [
        str(FILL_SIM),
        "--mbo-file", str(mbo_path),
        "--predictions", str(pred_npz_path),
        "--output", str(output_path),
        "--signal-threshold", str(config["signal_threshold"]),
        "--take-profit-ticks", str(config["take_profit_ticks"]),
        "--stop-loss-ticks", str(config["stop_loss_ticks"]),
        "--hold-ms", str(config["hold_ms"]),
        "--latency-ms", str(config.get("latency_ms", 10)),
        "--quiet",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"  fill_sim ERROR: {result.stderr[:200]}")
            return None
        if os.path.exists(output_path):
            with open(output_path) as f:
                return json.load(f)
    except Exception as e:
        print(f"  fill_sim EXCEPTION: {e}")
    return None


def compute_metrics(results_list):
    if not results_list:
        return {}

    all_trades = []
    total_pnl = 0
    total_trades = 0
    total_signals = 0
    total_filled = 0
    daily_pnls = []

    for r in results_list:
        pnl = r.get("total_pnl_dollars", 0)
        total_pnl += pnl
        total_trades += r.get("total_trades", 0)
        total_signals += r.get("total_signals", 0)
        total_filled += r.get("total_filled", 0)
        daily_pnls.append(pnl)
        for t in r.get("trades", []):
            all_trades.append(t.get("pnl_dollars", 0))

    if not all_trades:
        return {"total_pnl_dollars": total_pnl, "n_trades": 0}

    trade_pnls = np.array(all_trades)
    wins = trade_pnls[trade_pnls > 0]
    losses = trade_pnls[trade_pnls <= 0]
    mean_pnl = trade_pnls.mean()
    std_pnl = trade_pnls.std() if len(trade_pnls) > 1 else 1.0
    sharpe = mean_pnl / std_pnl if std_pnl > 0 else 0
    daily_arr = np.array(daily_pnls)
    daily_sharpe = (daily_arr.mean() / daily_arr.std() * np.sqrt(252)) if daily_arr.std() > 0 else 0
    downside = trade_pnls[trade_pnls < 0]
    downside_std = downside.std() if len(downside) > 1 else 1.0
    sortino = mean_pnl / downside_std if downside_std > 0 else 0
    gross_profit = wins.sum() if len(wins) > 0 else 0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 1
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    wr = len(wins) / len(trade_pnls)

    return {
        "total_pnl_dollars": round(total_pnl, 2),
        "n_trades": total_trades,
        "n_signals": total_signals,
        "n_filled": total_filled,
        "fill_rate": round(total_filled / total_signals, 4) if total_signals > 0 else 0,
        "win_rate": round(wr, 4),
        "mean_pnl_per_trade": round(mean_pnl, 2),
        "mean_net_ticks": round(mean_pnl / 12.50, 4),
        "sharpe_per_trade": round(sharpe, 4),
        "daily_sharpe": round(daily_sharpe, 2),
        "sortino_per_trade": round(sortino, 4),
        "profit_factor": round(pf, 4),
        "avg_win": round(wins.mean(), 2) if len(wins) > 0 else 0,
        "avg_loss": round(losses.mean(), 2) if len(losses) > 0 else 0,
        "n_days": len(results_list),
    }


def main():
    print("=" * 70)
    print("OPTIMIZER GATE VALIDATION - FIFO Fill Sim")
    print("=" * 70)

    opt_data = np.load(str(OPT_PREDS))
    opt_preds_all = opt_data['predictions']
    with open(str(OPT_RESULTS)) as f:
        opt_results = json.load(f)

    folds = sorted(opt_results['fold_results'], key=lambda x: x['fold'])
    date_opt_preds = {}
    offset = 0
    for fold_info in folds:
        date = fold_info['oot_date']
        n = fold_info['n_oot']
        date_opt_preds[date] = opt_preds_all[offset:offset + n]
        offset += n
    print(f"Optimizer predictions: {len(date_opt_preds)} dates, {len(opt_preds_all)} total")

    threshold_p90 = float(np.percentile(opt_preds_all, 90))
    threshold_p80 = float(np.percentile(opt_preds_all, 80))
    threshold_p70 = float(np.percentile(opt_preds_all, 70))
    print(f"Thresholds: p90={threshold_p90:.4f}, p80={threshold_p80:.4f}, p70={threshold_p70:.4f}")

    thresholds = {
        "top10pct": threshold_p90,
        "top20pct": threshold_p80,
        "top30pct": threshold_p70,
    }

    oot_dates = [f['oot_date'] for f in folds]

    # Step 1: Convert CNN predictions to bar-level
    print("\n--- Step 1: Converting CNN predictions to bar-level ---")
    bar_preds_cache = {}
    rth_starts = {}
    for date in oot_dates:
        result = cnn_event_preds_to_bars(date)
        if result is not None:
            bar_preds, rth_start = result
            bar_preds_cache[date] = bar_preds
            rth_starts[date] = rth_start
            n_active = (np.abs(bar_preds) > 0.3).sum()
            print(f"  {date}: {n_active} active bars (|pred|>0.3)")

    valid_dates = sorted(bar_preds_cache.keys())
    print(f"\nValid dates: {len(valid_dates)}")

    # Step 2: Create prediction NPZ files
    print("\n--- Step 2: Creating prediction NPZ files ---")
    pred_dir = OUTPUT_DIR / "pred_npzs"
    pred_dir.mkdir(exist_ok=True)

    for date in valid_dates:
        # Unfiltered
        uf_path = pred_dir / f"{date}_unfiltered.npz"
        np.savez(str(uf_path), predictions=bar_preds_cache[date])

        # Gated versions
        for gate_name, threshold in thresholds.items():
            opt_scores = date_opt_preds[date]
            gate_mask = get_optimizer_gate_bars(date, opt_scores, threshold, rth_starts[date])
            if gate_mask is not None:
                gated_preds = bar_preds_cache[date].copy()
                gated_preds[~gate_mask] = 0.0
                gated_path = pred_dir / f"{date}_gated_{gate_name}.npz"
                np.savez(str(gated_path), predictions=gated_preds)
                n_active_before = (np.abs(bar_preds_cache[date]) > 0.3).sum()
                n_active_after = (np.abs(gated_preds) > 0.3).sum()
                print(f"  {date} {gate_name}: active bars {n_active_before} -> {n_active_after} "
                      f"({(1 - n_active_after/max(n_active_before,1))*100:.0f}% reduction)")
            else:
                # Fallback: use unfiltered if merge fails
                gated_path = pred_dir / f"{date}_gated_{gate_name}.npz"
                np.savez(str(gated_path), predictions=bar_preds_cache[date])
                print(f"  {date} {gate_name}: FALLBACK to unfiltered (merge mismatch)")

    # Step 3: Run fill_sim
    print("\n--- Step 3: Running fill_sim ---")
    all_results = {}

    for cfg_name, cfg in FILLSIM_CONFIGS.items():
        print(f"\n  Config: {cfg_name}")
        variants = ["unfiltered"] + [f"gated_{g}" for g in thresholds.keys()]

        for variant in variants:
            key = f"{cfg_name}__{variant}"
            results_list = []

            for date in valid_dates:
                pred_path = pred_dir / f"{date}_{variant}.npz"
                if not pred_path.exists():
                    continue
                mbo_path = MBO_DIR / f"glbx-mdp3-{date}.mbo.dbn.zst"
                if not mbo_path.exists():
                    continue

                out_path = OUTPUT_DIR / f"fillsim_{cfg_name}_{variant}_{date}.json"
                r = run_fillsim(mbo_path, pred_path, out_path, cfg)
                if r is not None:
                    results_list.append(r)
                    pnl = r.get("total_pnl_dollars", 0)
                    trades = r.get("total_trades", 0)
                    wr = r.get("win_rate", 0)
                    print(f"    {date} [{variant[:12]}]: PnL=${pnl:.0f} trades={trades} WR={wr*100:.1f}%")

            metrics = compute_metrics(results_list)
            all_results[key] = metrics

    # Step 4: Summary
    print("\n" + "=" * 70)
    print("SUMMARY: OPTIMIZER GATE IMPACT")
    print("=" * 70)

    summary = {}
    for cfg_name in FILLSIM_CONFIGS:
        print(f"\n{'='*60}")
        print(f"Config: {cfg_name}")
        print(f"{'='*60}")
        header = f"{'Variant':<20} {'PnL':>10} {'Trades':>8} {'WR':>7} {'Sharpe':>8} {'Sortino':>8} {'PF':>7} {'NetTicks':>10}"
        print(header)
        print("-" * len(header))

        cfg_summary = {}
        for variant_key in ["unfiltered"] + [f"gated_{g}" for g in thresholds.keys()]:
            key = f"{cfg_name}__{variant_key}"
            m = all_results.get(key, {})
            if not m or m.get("n_trades", 0) == 0:
                continue

            label = variant_key[:20]
            print(f"{label:<20} "
                  f"${m.get('total_pnl_dollars', 0):>9.0f} "
                  f"{m.get('n_trades', 0):>8d} "
                  f"{m.get('win_rate', 0)*100:>6.1f}% "
                  f"{m.get('sharpe_per_trade', 0):>8.4f} "
                  f"{m.get('sortino_per_trade', 0):>8.4f} "
                  f"{m.get('profit_factor', 0):>7.2f} "
                  f"{m.get('mean_net_ticks', 0):>10.4f}")

            cfg_summary[variant_key] = m

        # Improvement analysis
        uf = cfg_summary.get("unfiltered", {})
        for gate_name in thresholds.keys():
            gated = cfg_summary.get(f"gated_{gate_name}", {})
            if uf and gated and uf.get("n_trades", 0) > 0:
                pnl_delta = gated.get("total_pnl_dollars", 0) - uf.get("total_pnl_dollars", 0)
                wr_delta = (gated.get("win_rate", 0) - uf.get("win_rate", 0)) * 100
                trade_reduction = 1 - (gated.get("n_trades", 1) / max(uf.get("n_trades", 1), 1))
                sharpe_delta = gated.get("sharpe_per_trade", 0) - uf.get("sharpe_per_trade", 0)
                print(f"\n  {gate_name} vs unfiltered:")
                print(f"    PnL delta: ${pnl_delta:+.0f}")
                print(f"    WR delta: {wr_delta:+.1f}pp")
                print(f"    Sharpe delta: {sharpe_delta:+.4f}")
                print(f"    Trade reduction: {trade_reduction*100:.0f}%")

        summary[cfg_name] = cfg_summary

    # Save
    summary_path = OUTPUT_DIR / "gate_validation_summary.json"
    with open(str(summary_path), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults saved to {summary_path}")
    print("DONE.")


if __name__ == "__main__":
    main()
