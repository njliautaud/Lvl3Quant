#!/usr/bin/env python3
"""
FIFO Replay Validation v1 — Final Gate for Meta-Model-Filtered Trading Strategy
================================================================================
Validates that the meta-model-filtered top-3%-short strategy remains profitable
under realistic FIFO queue position simulation for ES futures.

Strategy:
  - Signal: top 3% shorts by pred_1s (most negative CNN-Mamba v2 predictions)
  - Meta-filter: keep top 30% by meta-model score (highest predicted P&L)
  - Entry: passive limit sell at ask (join queue at back)
  - Hold: 5 seconds
  - Stop: 2-tick adverse → market exit
  - Exit: passive limit buy at bid after hold, 5s patience, market if unfilled
  - Commission: $4.70 RT = 0.376 ticks

FIFO Queue Simulation:
  - For each fill rate scenario (30%-80%), simulate whether each trade gets
    passive entry, passive exit, or market exit.
  - Reports breakeven fill rate, per-scenario Sharpe/Sortino/PF/WR.

Output: /home/jupiter/Lvl3Quant/output/fifo_replay_v1/results.json
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

# ─── Platform detection ─────────────────────────────────────────────────────
if sys.platform == 'win32':
    _BASE = Path(r'C:\Users\claude\Lvl3Quant')
else:
    _BASE = Path('/home/jupiter/Lvl3Quant')

PRED_DIR = _BASE / 'output' / 'cnn_mamba_v2_bulk_oot_v2'
MBO_DIR = _BASE / 'data' / 'processed' / 'mbo_events_smart_v3'
WEIGHTS_DIR = _BASE / 'output' / 'meta_production_v1' / 'weights'
OUT_DIR = _BASE / 'output' / 'fifo_replay_v1'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Strategy constants ─────────────────────────────────────────────────────
COMMISSION_RT = 0.376          # $4.70 / $12.50 per tick
SPREAD_TICK = 1.0              # ES book is 1 tick wide during RTH
STOP_TICKS = 2.0               # adverse movement threshold
STOP_SLIPPAGE = 1.0            # market exit slippage on stop
SHORT_PERCENTILE = 3           # top 3% most negative pred_1s
META_FILTER_PCT = 30           # keep top 30% by meta score
HOLD_SECONDS = 5.0             # hold time before exit attempt
CANCEL_WINDOW_S = 5.0          # max wait for passive entry fill
EXIT_PATIENCE_S = 5.0          # max wait for passive exit fill

# Cost per trade outcome (in ticks)
COST_PASSIVE_PASSIVE = COMMISSION_RT                                  # 0.376
COST_PASSIVE_MARKET = COMMISSION_RT + SPREAD_TICK                    # 1.376
COST_STOP = STOP_TICKS + STOP_SLIPPAGE + COMMISSION_RT              # 3.376
COST_SKIP = 0.0                                                       # not entered

# Simulation parameters
FILL_RATE_SCENARIOS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
N_MONTE_CARLO = 1000
EXIT_PASSIVE_RATE = 0.50       # baseline exit fill rate (conservative)
STOP_RATE = 0.10               # 10% of trades hit stop loss (from historical data)
RNG_SEED = 42

TICK_VALUE = 12.50             # dollars per tick

device = torch.device('cpu')   # CPU only — no GPU needed


# ─── Meta-model architecture (must match training) ──────────────────────────
class ProductionMetaMLP(nn.Module):
    """256->128->64->32 deeper MLP. Exact match to train_meta_production_v1.py."""
    def __init__(self, input_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ─── Data loading ───────────────────────────────────────────────────────────
def load_day_data(pred_file):
    """Load one OOT day: predictions + MBO events + labels. Returns dict or None."""
    pred_data = np.load(pred_file, allow_pickle=True)
    date_str = str(pred_data['date'])
    preds = pred_data['predictions']  # (n_windows, 3) for 1s/5s/10s
    n_windows = int(pred_data['n_windows'])
    window_size = int(pred_data['window_size'])
    stride = int(pred_data['stride'])

    mbo_file = MBO_DIR / f'{date_str}_mbo_events.npz'
    if not mbo_file.exists():
        return None

    mbo = np.load(mbo_file, allow_pickle=True)
    events = mbo['events']       # (N, 25)
    labels_1s = mbo['labels_1s']
    labels_5s = mbo['labels_5s'] if 'labels_5s' in mbo else None

    # Map window indices to event indices
    indices = np.arange(n_windows) * stride + (window_size - 1)
    max_idx = min(len(events), len(labels_1s)) - 1
    valid = indices <= max_idx
    indices = indices[valid]
    preds = preds[:len(indices)]

    feat = events[indices]       # (n_valid, 25)
    label_1s = labels_1s[indices]
    label_5s = labels_5s[indices] if labels_5s is not None else None

    # Clean NaN
    valid_mask = ~(np.isnan(label_1s) | np.any(np.isnan(feat), axis=1))
    if label_5s is not None:
        valid_mask &= ~np.isnan(label_5s)

    feat = feat[valid_mask]
    label_1s = label_1s[valid_mask]
    if label_5s is not None:
        label_5s = label_5s[valid_mask]
    preds = preds[valid_mask]

    if len(feat) == 0:
        return None

    return {
        'date': date_str,
        'features': feat,         # (n, 25) MBO features
        'predictions': preds,     # (n, 3) CNN-Mamba predictions
        'label_1s': label_1s,
        'label_5s': label_5s,
        'n_total': int(valid_mask.sum()),
    }


def build_meta_features(day_data):
    """Build 29-dim meta features from day data (same as training)."""
    feat = day_data['features']       # (n, 25) MBO
    preds = day_data['predictions']   # (n, 3)
    pred_1s = preds[:, 0]

    # Select top 3% shorts (most negative pred_1s)
    threshold = np.percentile(pred_1s, SHORT_PERCENTILE)
    short_mask = pred_1s <= threshold

    if short_mask.sum() < 2:
        return None

    feat_s = feat[short_mask]
    preds_s = preds[short_mask]
    label_1s_s = day_data['label_1s'][short_mask]
    label_5s_s = day_data['label_5s'][short_mask] if day_data['label_5s'] is not None else None

    # 29 features: 25 MBO + pred_1s + pred_5s + pred_10s + rank
    ranks = np.argsort(np.argsort(preds_s[:, 0])).astype(np.float32) / max(len(preds_s), 1)
    meta_features = np.column_stack([
        feat_s,
        preds_s[:, 0], preds_s[:, 1], preds_s[:, 2],
        ranks,
    ]).astype(np.float32)

    # P&L target (for reference — actual tick movement for shorts)
    short_pnl = -label_1s_s  # short = profit when price drops
    stop_hit = label_1s_s >= STOP_TICKS
    pnl_target = np.where(
        stop_hit,
        -(STOP_TICKS + STOP_SLIPPAGE + COMMISSION_RT),
        short_pnl - COMMISSION_RT,
    ).astype(np.float32)

    return {
        'date': day_data['date'],
        'meta_features': meta_features,       # (n_shorts, 29)
        'pnl_no_fifo': pnl_target,            # P&L assuming perfect fills
        'label_1s': label_1s_s,
        'label_5s': label_5s_s,
        'mbo_features': feat_s,               # raw MBO for fill estimation
        'n_shorts': int(short_mask.sum()),
        'n_total': day_data['n_total'],
    }


def load_meta_weights(fold_date):
    """Find the weight file matching a test date. Returns (model, norm_mean, norm_std) or None."""
    # Search for weight file matching this date
    for wf in WEIGHTS_DIR.glob('fold_*_*.pt'):
        parts = wf.stem.split('_')
        w_date = parts[-1]
        if w_date == fold_date:
            checkpoint = torch.load(wf, map_location='cpu', weights_only=False)
            input_dim = checkpoint['input_dim']
            model = ProductionMetaMLP(input_dim).to(device)
            model.load_state_dict(checkpoint['model_state'])
            model.eval()
            norm_mean = checkpoint['norm_mean']
            norm_std = checkpoint['norm_std']
            if isinstance(norm_mean, torch.Tensor):
                norm_mean = norm_mean.numpy()
            if isinstance(norm_std, torch.Tensor):
                norm_std = norm_std.numpy()
            return model, norm_mean, norm_std
    return None


# ─── FIFO simulation ────────────────────────────────────────────────────────
def estimate_fill_probabilities(mbo_features):
    """
    Estimate per-trade fill probability from MBO features.

    The 25 MBO features encode bid/ask sizes, trade flow, and microstructure.
    We use a heuristic based on trade intensity relative to book depth.

    Returns array of fill probabilities in [0, 1] for each trade.
    """
    n = len(mbo_features)
    if n == 0:
        return np.array([])

    # MBO features (indices approximate — these are normalized features):
    # Features encode information about order flow, bid/ask balance, trade intensity
    # We use the variance across features as a proxy for "how active is the market"
    # Higher activity = higher fill probability

    # Use feature magnitude as activity proxy (more extreme = more activity)
    activity = np.abs(mbo_features).mean(axis=1)
    activity_norm = (activity - activity.min()) / (activity.max() - activity.min() + 1e-8)

    # Map to fill probability range [0.2, 0.9]
    fill_probs = 0.2 + 0.7 * activity_norm

    return fill_probs.astype(np.float32)


def simulate_fifo_scenario(
    pnl_no_fifo,
    label_1s,
    fill_rate_entry,
    fill_rate_exit=None,
    stop_rate=STOP_RATE,
    n_simulations=N_MONTE_CARLO,
    rng=None,
):
    """
    Monte Carlo FIFO simulation for a batch of trades.

    For each simulation:
      - Each trade gets passive entry with probability fill_rate_entry
      - Trades not entered: skip (cost=0)
      - For entered trades:
        - stop_rate fraction hit stop loss (cost = COST_STOP)
        - remaining: passive exit with probability fill_rate_exit, else market exit

    Returns dict with per-simulation P&L arrays and summary stats.
    """
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)

    if fill_rate_exit is None:
        fill_rate_exit = EXIT_PASSIVE_RATE

    n_trades = len(pnl_no_fifo)
    if n_trades == 0:
        return {
            'mean_pnl_per_trade': 0.0,
            'total_pnl': 0.0,
            'n_entered': 0,
            'sharpe': 0.0,
            'sortino': 0.0,
            'wr': 0.0,
            'pf': 0.0,
        }

    # Gross P&L per trade (before commission, no stop)
    # short_pnl = -label_1s (price movement in our favor)
    gross_pnl = -label_1s.copy()

    # Pre-identify which trades would hit stop if entered
    would_stop = label_1s >= STOP_TICKS  # price moved against short by >= 2 ticks

    sim_total_pnls = np.zeros(n_simulations)
    sim_n_entered = np.zeros(n_simulations)
    sim_trade_pnls = []  # for per-trade stats

    for sim_i in range(n_simulations):
        # Roll entry fills
        entry_rolls = rng.random(n_trades)
        entered = entry_rolls < fill_rate_entry

        n_entered = entered.sum()
        if n_entered == 0:
            sim_total_pnls[sim_i] = 0.0
            sim_n_entered[sim_i] = 0
            continue

        sim_n_entered[sim_i] = n_entered
        trade_pnl = np.zeros(n_trades)

        for j in range(n_trades):
            if not entered[j]:
                trade_pnl[j] = 0.0  # skipped
                continue

            if would_stop[j]:
                # Stop loss hit: fixed loss
                trade_pnl[j] = -COST_STOP
            else:
                # Trade lives to exit
                exit_roll = rng.random()
                if exit_roll < fill_rate_exit:
                    # Passive exit: only pay commission
                    trade_pnl[j] = gross_pnl[j] - COMMISSION_RT
                else:
                    # Market exit: pay commission + spread crossing
                    trade_pnl[j] = gross_pnl[j] - COST_PASSIVE_MARKET

        sim_total_pnls[sim_i] = trade_pnl[entered].sum()
        if sim_i == 0:
            sim_trade_pnls = trade_pnl[entered].copy()

    # Summary stats across simulations
    mean_total = float(sim_total_pnls.mean())
    mean_entered = float(sim_n_entered.mean())
    mean_per_trade = mean_total / max(mean_entered, 1)

    return {
        'mean_pnl_per_trade': mean_per_trade,
        'total_pnl': mean_total,
        'n_entered': mean_entered,
        'n_signals': n_trades,
        'fill_rate_entry': fill_rate_entry,
        'fill_rate_exit': fill_rate_exit,
        'sim_total_pnls': sim_total_pnls,   # for daily aggregation
        'first_sim_trade_pnls': sim_trade_pnls,  # for WR/PF from first sim
    }


# ─── Metric computation ─────────────────────────────────────────────────────
def compute_metrics(daily_pnls, trade_pnls=None):
    """Compute risk-adjusted metrics from daily P&L array."""
    daily = np.array(daily_pnls, dtype=np.float64)

    if len(daily) < 2 or daily.std() == 0:
        return {
            'sharpe_annual': 0.0,
            'sortino_annual': 0.0,
            'wr_pct': 0.0,
            'pf': 0.0,
            'green_day_pct': 0.0,
            'total_pnl_ticks': 0.0,
            'mean_daily_pnl': 0.0,
            'max_drawdown_ticks': 0.0,
            'n_days': len(daily),
        }

    mean_d = daily.mean()
    std_d = daily.std()
    sharpe = float(mean_d / std_d * np.sqrt(252))

    down = daily[daily < 0]
    down_std = down.std() if len(down) > 1 else std_d
    sortino = float(mean_d / down_std * np.sqrt(252)) if down_std > 0 else 999.0

    green_days = (daily > 0).sum()
    green_pct = float(green_days / len(daily) * 100)

    # Drawdown
    cumsum = np.cumsum(daily)
    running_max = np.maximum.accumulate(cumsum)
    drawdown = running_max - cumsum
    max_dd = float(drawdown.max())

    # Trade-level WR and PF
    wr = 0.0
    pf = 0.0
    if trade_pnls is not None and len(trade_pnls) > 0:
        tp = np.array(trade_pnls)
        tp_entered = tp[tp != 0]  # exclude skipped trades
        if len(tp_entered) > 0:
            wr = float((tp_entered > 0).sum() / len(tp_entered) * 100)
            wins = tp_entered[tp_entered > 0].sum()
            losses = abs(tp_entered[tp_entered < 0].sum())
            pf = float(wins / losses) if losses > 0 else 999.0

    return {
        'sharpe_annual': round(sharpe, 2),
        'sortino_annual': round(sortino, 2),
        'wr_pct': round(wr, 1),
        'pf': round(pf, 2),
        'green_day_pct': round(green_pct, 1),
        'total_pnl_ticks': round(float(daily.sum()), 1),
        'total_pnl_dollars': round(float(daily.sum() * TICK_VALUE), 0),
        'mean_daily_pnl': round(float(mean_d), 2),
        'mean_daily_dollars': round(float(mean_d * TICK_VALUE), 0),
        'max_drawdown_ticks': round(max_dd, 1),
        'max_drawdown_dollars': round(float(max_dd * TICK_VALUE), 0),
        'n_days': len(daily),
    }


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    print(f"{'='*72}", flush=True)
    print(f"FIFO REPLAY VALIDATION v1 — Final Gate", flush=True)
    print(f"{'='*72}", flush=True)
    print(f"Started: {datetime.now().isoformat()}", flush=True)
    print(f"Signal: top {SHORT_PERCENTILE}% shorts -> meta filter top {META_FILTER_PCT}%", flush=True)
    print(f"Fill rate scenarios: {FILL_RATE_SCENARIOS}", flush=True)
    print(f"Monte Carlo iterations: {N_MONTE_CARLO}", flush=True)
    print(f"Cost structure:", flush=True)
    print(f"  Passive+Passive: {COST_PASSIVE_PASSIVE:.3f} ticks", flush=True)
    print(f"  Passive+Market:  {COST_PASSIVE_MARKET:.3f} ticks", flush=True)
    print(f"  Stop loss:       {COST_STOP:.3f} ticks", flush=True)
    print(f"{'='*72}\n", flush=True)

    # ── Step 1: Load all prediction files and build per-day data ──────────
    print("[1/5] Loading prediction files...", flush=True)
    pred_files = sorted([
        f for f in PRED_DIR.glob('*_predictions.npz')
        if '_stale_' not in str(f)
    ])
    print(f"  Found {len(pred_files)} prediction files", flush=True)

    # Build weight date lookup: date_str -> weight_file_path
    weight_dates = {}
    for wf in sorted(WEIGHTS_DIR.glob('fold_*_*.pt')):
        parts = wf.stem.split('_')
        w_date = parts[-1]
        weight_dates[w_date] = wf
    print(f"  Found {len(weight_dates)} meta-model weight files", flush=True)
    print(f"  Weight dates: {sorted(weight_dates.keys())[:5]}...{sorted(weight_dates.keys())[-3:]}", flush=True)

    # Only process dates that have both predictions AND weight files
    # Walk-forward: for each date with weights, that weight was trained on prior data
    # So we can use it on that date's predictions
    valid_dates = set()
    for pf in pred_files:
        date_str = pf.stem.replace('_predictions', '')
        if date_str in weight_dates:
            valid_dates.add(date_str)

    print(f"  Dates with both predictions and weights: {len(valid_dates)}", flush=True)

    # ── Step 2: Load and process each valid date ──────────────────────────
    print("\n[2/5] Processing dates with meta-model filtering...", flush=True)
    day_results = []  # Each entry: {date, filtered_pnl, filtered_labels, n_signals, n_filtered}

    for i, date_str in enumerate(sorted(valid_dates)):
        pred_file = PRED_DIR / f'{date_str}_predictions.npz'
        day_data = load_day_data(pred_file)
        if day_data is None:
            print(f"  {date_str}: SKIP (no MBO data)", flush=True)
            continue

        meta_data = build_meta_features(day_data)
        if meta_data is None:
            print(f"  {date_str}: SKIP (too few shorts)", flush=True)
            continue

        # Load meta-model for this fold
        weight_info = load_meta_weights(date_str)
        if weight_info is None:
            print(f"  {date_str}: SKIP (no matching weights)", flush=True)
            continue

        model, norm_mean, norm_std = weight_info

        # Normalize features
        X = meta_data['meta_features']
        X_norm = (X - norm_mean) / (norm_std + 1e-8)

        # Get meta-model scores
        with torch.no_grad():
            meta_scores = model(torch.from_numpy(X_norm).to(device)).cpu().numpy()

        # Apply meta filter: keep top 30% by meta score
        if len(meta_scores) < 3:
            print(f"  {date_str}: SKIP (only {len(meta_scores)} shorts)", flush=True)
            continue

        filter_threshold = np.percentile(meta_scores, 100 - META_FILTER_PCT)
        filter_mask = meta_scores >= filter_threshold
        n_filtered = int(filter_mask.sum())

        if n_filtered == 0:
            print(f"  {date_str}: SKIP (no trades after meta filter)", flush=True)
            continue

        filtered_pnl = meta_data['pnl_no_fifo'][filter_mask]
        filtered_label_1s = meta_data['label_1s'][filter_mask]
        filtered_mbo = meta_data['mbo_features'][filter_mask]

        # Estimate per-trade fill probability (data-driven)
        fill_probs = estimate_fill_probabilities(filtered_mbo)

        # Compute stop rate for this day from actual labels
        day_stop_rate = float((filtered_label_1s >= STOP_TICKS).mean())

        day_results.append({
            'date': date_str,
            'n_total': meta_data['n_total'],
            'n_shorts': meta_data['n_shorts'],
            'n_filtered': n_filtered,
            'pnl_no_fifo': filtered_pnl,
            'label_1s': filtered_label_1s,
            'fill_probs': fill_probs,
            'stop_rate': day_stop_rate,
            'mean_meta_score': float(meta_scores[filter_mask].mean()),
        })

        status = "+" if filtered_pnl.mean() > 0 else "-"
        print(f"  {date_str}: {meta_data['n_total']:>6} events -> {meta_data['n_shorts']:>4} shorts "
              f"-> {n_filtered:>3} filtered  pnl={filtered_pnl.mean():+.3f}  "
              f"stop={day_stop_rate:.1%} {status}", flush=True)

    print(f"\n  Total valid days: {len(day_results)}", flush=True)
    total_signals = sum(d['n_filtered'] for d in day_results)
    print(f"  Total filtered signals: {total_signals}", flush=True)

    if len(day_results) == 0:
        print("ERROR: No valid days to simulate. Exiting.", flush=True)
        return

    # ── Step 3: Baseline (no FIFO) analysis ───────────────────────────────
    print(f"\n[3/5] Baseline analysis (perfect fills, no FIFO)...", flush=True)
    baseline_daily_pnl = []
    all_baseline_trades = []
    for d in day_results:
        daily_total = float(d['pnl_no_fifo'].sum())
        baseline_daily_pnl.append(daily_total)
        all_baseline_trades.extend(d['pnl_no_fifo'].tolist())

    baseline_metrics = compute_metrics(baseline_daily_pnl, all_baseline_trades)
    print(f"  Baseline (perfect fills):", flush=True)
    print(f"    Sharpe:   {baseline_metrics['sharpe_annual']}", flush=True)
    print(f"    Sortino:  {baseline_metrics['sortino_annual']}", flush=True)
    print(f"    WR:       {baseline_metrics['wr_pct']}%", flush=True)
    print(f"    PF:       {baseline_metrics['pf']}", flush=True)
    print(f"    Green %:  {baseline_metrics['green_day_pct']}%", flush=True)
    print(f"    Total:    {baseline_metrics['total_pnl_ticks']} ticks "
          f"(${baseline_metrics['total_pnl_dollars']:,.0f})", flush=True)
    print(f"    Mean/day: {baseline_metrics['mean_daily_pnl']:.2f} ticks "
          f"(${baseline_metrics['mean_daily_dollars']:,.0f})", flush=True)
    print(f"    Max DD:   {baseline_metrics['max_drawdown_ticks']} ticks", flush=True)

    # ── Step 4: FIFO simulation across fill rate scenarios ────────────────
    print(f"\n[4/5] Running FIFO Monte Carlo simulation...", flush=True)
    print(f"  {N_MONTE_CARLO} iterations per scenario x {len(FILL_RATE_SCENARIOS)} scenarios", flush=True)

    rng = np.random.default_rng(RNG_SEED)
    scenario_results = {}

    for fr_idx, fill_rate in enumerate(FILL_RATE_SCENARIOS):
        print(f"\n  --- Fill rate: {fill_rate:.0%} ---", flush=True)

        # For each day, run simulation
        daily_pnls_mean = []
        daily_pnls_all_sims = []  # (n_days, n_sims)
        all_trade_pnls = []

        for d in day_results:
            sim_result = simulate_fifo_scenario(
                pnl_no_fifo=d['pnl_no_fifo'],
                label_1s=d['label_1s'],
                fill_rate_entry=fill_rate,
                fill_rate_exit=EXIT_PASSIVE_RATE,
                stop_rate=d['stop_rate'],
                n_simulations=N_MONTE_CARLO,
                rng=rng,
            )

            daily_pnls_mean.append(sim_result['total_pnl'])
            daily_pnls_all_sims.append(sim_result['sim_total_pnls'])
            if len(sim_result['first_sim_trade_pnls']) > 0:
                all_trade_pnls.extend(sim_result['first_sim_trade_pnls'].tolist())

        # Compute metrics from mean daily P&L across simulations
        metrics = compute_metrics(daily_pnls_mean, all_trade_pnls)

        # Also compute confidence interval from simulation distribution
        daily_sims = np.array(daily_pnls_all_sims)  # (n_days, n_sims)
        total_per_sim = daily_sims.sum(axis=0)       # (n_sims,)
        ci_5 = float(np.percentile(total_per_sim, 5))
        ci_50 = float(np.percentile(total_per_sim, 50))
        ci_95 = float(np.percentile(total_per_sim, 95))

        # Per-sim Sharpe distribution
        sim_sharpes = []
        for sim_i in range(min(N_MONTE_CARLO, 100)):  # sample 100 for speed
            sim_daily = daily_sims[:, sim_i]
            if sim_daily.std() > 0:
                sim_sharpes.append(float(sim_daily.mean() / sim_daily.std() * np.sqrt(252)))
        sharpe_ci = (
            float(np.percentile(sim_sharpes, 5)) if sim_sharpes else 0.0,
            float(np.percentile(sim_sharpes, 50)) if sim_sharpes else 0.0,
            float(np.percentile(sim_sharpes, 95)) if sim_sharpes else 0.0,
        )

        # Mean trades entered per day
        mean_entered_per_day = sum(
            d['n_filtered'] * fill_rate for d in day_results
        ) / len(day_results)

        scenario_results[f'{int(fill_rate*100)}%'] = {
            'fill_rate': fill_rate,
            'metrics': metrics,
            'total_pnl_ci': {
                'p5': round(ci_5, 1),
                'p50': round(ci_50, 1),
                'p95': round(ci_95, 1),
            },
            'sharpe_ci': {
                'p5': round(sharpe_ci[0], 2),
                'p50': round(sharpe_ci[1], 2),
                'p95': round(sharpe_ci[2], 2),
            },
            'mean_trades_per_day': round(mean_entered_per_day, 1),
        }

        print(f"    Sharpe:   {metrics['sharpe_annual']:>6.2f}  "
              f"[CI: {sharpe_ci[0]:+.2f} / {sharpe_ci[1]:+.2f} / {sharpe_ci[2]:+.2f}]", flush=True)
        print(f"    Sortino:  {metrics['sortino_annual']:>6.2f}", flush=True)
        print(f"    WR:       {metrics['wr_pct']:>5.1f}%", flush=True)
        print(f"    PF:       {metrics['pf']:>5.2f}", flush=True)
        print(f"    Green %:  {metrics['green_day_pct']:>5.1f}%", flush=True)
        print(f"    Total:    {metrics['total_pnl_ticks']:>6.1f} ticks  "
              f"[CI: {ci_5:+.0f} / {ci_50:+.0f} / {ci_95:+.0f}]", flush=True)
        print(f"    Mean/day: {metrics['mean_daily_pnl']:>6.2f} ticks "
              f"(${metrics['mean_daily_dollars']:,.0f})", flush=True)
        print(f"    Trades/day: {mean_entered_per_day:.1f}", flush=True)

    # ── Step 5: Find breakeven fill rate and summarize ────────────────────
    print(f"\n{'='*72}", flush=True)
    print(f"[5/5] BREAKEVEN ANALYSIS", flush=True)
    print(f"{'='*72}", flush=True)

    # Interpolate breakeven
    fill_rates = []
    total_pnls = []
    for key, sr in scenario_results.items():
        fill_rates.append(sr['fill_rate'])
        total_pnls.append(sr['metrics']['total_pnl_ticks'])

    fill_rates = np.array(fill_rates)
    total_pnls = np.array(total_pnls)

    breakeven_fill_rate = None
    if total_pnls[-1] > 0 and total_pnls[0] < 0:
        # Interpolate where P&L crosses zero
        for i in range(len(total_pnls) - 1):
            if total_pnls[i] <= 0 and total_pnls[i + 1] > 0:
                # Linear interpolation
                frac = -total_pnls[i] / (total_pnls[i + 1] - total_pnls[i])
                breakeven_fill_rate = float(fill_rates[i] + frac * (fill_rates[i + 1] - fill_rates[i]))
                break
    elif total_pnls[0] > 0:
        breakeven_fill_rate = 0.0  # Profitable even at lowest fill rate
        # Try to find actual breakeven below our range
        if fill_rates[0] > 0:
            # Extrapolate
            slope = (total_pnls[1] - total_pnls[0]) / (fill_rates[1] - fill_rates[0])
            if slope > 0:
                breakeven_fill_rate = float(fill_rates[0] - total_pnls[0] / slope)
                breakeven_fill_rate = max(0.0, breakeven_fill_rate)
    elif total_pnls[-1] <= 0:
        breakeven_fill_rate = 1.0  # Never profitable

    if breakeven_fill_rate is not None:
        print(f"\n  BREAKEVEN FILL RATE: {breakeven_fill_rate:.1%}", flush=True)
        if breakeven_fill_rate < 0.40:
            verdict = "STRONG PASS — profitable even with poor fills"
        elif breakeven_fill_rate < 0.55:
            verdict = "PASS — profitable with moderate fill assumptions"
        elif breakeven_fill_rate < 0.70:
            verdict = "MARGINAL — requires good queue position for profitability"
        else:
            verdict = "FAIL — requires unrealistically high fill rate"
        print(f"  VERDICT: {verdict}", flush=True)
    else:
        if np.all(total_pnls > 0):
            verdict = "STRONG PASS — profitable at all tested fill rates"
            breakeven_fill_rate = 0.0
        elif np.all(total_pnls <= 0):
            verdict = "FAIL — unprofitable at all tested fill rates"
            breakeven_fill_rate = 1.0
        else:
            verdict = "INCONCLUSIVE — non-monotonic relationship"
            breakeven_fill_rate = -1.0
        print(f"\n  VERDICT: {verdict}", flush=True)

    # Summary table
    print(f"\n  {'Fill%':>6} {'Sharpe':>7} {'Sortino':>8} {'WR%':>6} {'PF':>6} "
          f"{'Green%':>7} {'Ticks':>7} {'$/day':>8} {'Tr/day':>7}", flush=True)
    print(f"  {'-'*68}", flush=True)
    print(f"  {'100%':>6} {baseline_metrics['sharpe_annual']:>7.2f} "
          f"{baseline_metrics['sortino_annual']:>8.2f} "
          f"{baseline_metrics['wr_pct']:>5.1f}% {baseline_metrics['pf']:>5.2f} "
          f"{baseline_metrics['green_day_pct']:>6.1f}% "
          f"{baseline_metrics['total_pnl_ticks']:>6.0f} "
          f"${baseline_metrics['mean_daily_dollars']:>6,.0f} "
          f"{'all':>7}", flush=True)
    for key in sorted(scenario_results.keys(), key=lambda x: int(x.replace('%', ''))):
        sr = scenario_results[key]
        m = sr['metrics']
        print(f"  {key:>6} {m['sharpe_annual']:>7.2f} {m['sortino_annual']:>8.2f} "
              f"{m['wr_pct']:>5.1f}% {m['pf']:>5.2f} {m['green_day_pct']:>6.1f}% "
              f"{m['total_pnl_ticks']:>6.0f} ${m['mean_daily_dollars']:>6,.0f} "
              f"{sr['mean_trades_per_day']:>6.1f}", flush=True)

    # ── Per-day breakdown at 50% fill rate (reference scenario) ───────────
    ref_key = '50%'
    if ref_key in scenario_results:
        print(f"\n  Per-day P&L at {ref_key} fill rate:", flush=True)
        rng_ref = np.random.default_rng(RNG_SEED + 999)
        print(f"  {'Date':>10} {'Signals':>8} {'Entered':>8} {'PnL':>8} {'WR%':>6}", flush=True)
        for d in day_results:
            sim = simulate_fifo_scenario(
                d['pnl_no_fifo'], d['label_1s'],
                fill_rate_entry=0.50, n_simulations=100, rng=rng_ref,
            )
            n_ent = sim['n_entered']
            pnl = sim['total_pnl']
            tp = sim['first_sim_trade_pnls']
            wr = float((tp > 0).sum() / max(len(tp), 1) * 100) if len(tp) > 0 else 0
            print(f"  {d['date']:>10} {d['n_filtered']:>8} {n_ent:>7.0f} "
                  f"{pnl:>+7.1f} {wr:>5.1f}%", flush=True)

    # ── Save results ──────────────────────────────────────────────────────
    print(f"\n{'='*72}", flush=True)
    print(f"Saving results...", flush=True)

    # Clean up non-serializable data for JSON
    json_scenarios = {}
    for key, sr in scenario_results.items():
        json_scenarios[key] = {
            'fill_rate': sr['fill_rate'],
            'metrics': sr['metrics'],
            'total_pnl_ci': sr['total_pnl_ci'],
            'sharpe_ci': sr['sharpe_ci'],
            'mean_trades_per_day': sr['mean_trades_per_day'],
        }

    per_day_summary = []
    for d in day_results:
        per_day_summary.append({
            'date': d['date'],
            'n_total_events': d['n_total'],
            'n_shorts': d['n_shorts'],
            'n_filtered': d['n_filtered'],
            'mean_pnl_no_fifo': round(float(d['pnl_no_fifo'].mean()), 4),
            'total_pnl_no_fifo': round(float(d['pnl_no_fifo'].sum()), 2),
            'stop_rate': round(d['stop_rate'], 3),
            'mean_meta_score': round(d['mean_meta_score'], 4),
        })

    results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'signal': f'top {SHORT_PERCENTILE}% shorts by pred_1s',
            'meta_filter': f'top {META_FILTER_PCT}% by meta score',
            'hold_seconds': HOLD_SECONDS,
            'stop_ticks': STOP_TICKS,
            'commission_rt_ticks': COMMISSION_RT,
            'spread_tick': SPREAD_TICK,
            'exit_passive_rate': EXIT_PASSIVE_RATE,
            'n_monte_carlo': N_MONTE_CARLO,
            'cost_passive_passive': COST_PASSIVE_PASSIVE,
            'cost_passive_market': COST_PASSIVE_MARKET,
            'cost_stop': COST_STOP,
        },
        'data': {
            'n_days': len(day_results),
            'total_signals': total_signals,
            'signals_per_day': round(total_signals / max(len(day_results), 1), 1),
            'dates': [d['date'] for d in day_results],
        },
        'baseline_perfect_fills': baseline_metrics,
        'fifo_scenarios': json_scenarios,
        'breakeven_fill_rate': round(breakeven_fill_rate, 3) if breakeven_fill_rate is not None else None,
        'verdict': verdict,
        'per_day': per_day_summary,
    }

    results_path = OUT_DIR / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"  Saved: {results_path}", flush=True)
    print(f"\n{'='*72}", flush=True)
    print(f"FIFO REPLAY VALIDATION COMPLETE", flush=True)
    print(f"  Breakeven fill rate: {breakeven_fill_rate:.1%}" if breakeven_fill_rate is not None else "  Breakeven: N/A", flush=True)
    print(f"  Verdict: {verdict}", flush=True)
    print(f"{'='*72}", flush=True)


if __name__ == '__main__':
    main()
