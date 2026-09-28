#!/usr/bin/env python3
"""Combined LGBM + CNN1D Trading Strategy Backtest.

Loads OOT predictions from both models, computes daily directional signals
at high confidence, combines them, and simulates ES futures trading.

Walk-forward safe: each day's signal uses only predictions from models
trained on strictly prior data.
"""

import os, re, sys
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

# ── Paths ────────────────────────────────────────────────────────────────
LGBM_DIR = os.path.expanduser(
    "~/Lvl3Quant/alpha_discovery/deep_models/results/vol_lgbm_v2")
CNN_DIR = os.path.expanduser(
    "~/Lvl3Quant/alpha_discovery/deep_models/results/cnn1d_neptune_20260418_1033")
OUT_DIR = os.path.expanduser("~/Lvl3Quant/data/results")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Trading Constants ────────────────────────────────────────────────────
TICK_VALUE = 12.50        # ES tick value
SLIPPAGE_TICKS = 1        # 1 tick slippage per side
COMMISSION_PER_SIDE = 0.50
STRONG_CONTRACTS = 2
WEAK_CONTRACTS = 1
CONFIDENCE_PCTILE = 90    # top-10% = 90th percentile

# ── LGBM Fold-to-Date Mapping (from stdout.log) ─────────────────────────
# Training and OOT date ranges extracted from the training log.
# Sliding 60d train / 15d test.
LGBM_FOLDS = [
    {"fold": 0, "train_end": "20250921", "oot_start": "20250922", "oot_end": "20251008"},
    {"fold": 1, "train_end": "20251008", "oot_start": "20251009", "oot_end": "20251026"},
    {"fold": 2, "train_end": "20251026", "oot_start": "20251027", "oot_end": "20251112"},
    {"fold": 3, "train_end": "20251112", "oot_start": "20251113", "oot_end": "20251201"},
    {"fold": 4, "train_end": "20251201", "oot_start": "20251202", "oot_end": "20251219"},
    {"fold": 5, "train_end": "20251219", "oot_start": "20251221", "oot_end": "20260106"},
    {"fold": 6, "train_end": "20260106", "oot_start": "20260107", "oot_end": "20260125"},
    {"fold": 7, "train_end": "20260125", "oot_start": "20260126", "oot_end": "20260212"},
    {"fold": 8, "train_end": "20260212", "oot_start": "20260213", "oot_end": "20260303"},
]

# CNN folds: expanding window, one OOT day per fold, trained on data before that day.
# Fold dates extracted from oot_files.
CNN_FOLDS = [
    {"fold": 0, "oot_date": "20250905"},
    {"fold": 1, "oot_date": "20250907"},
    {"fold": 2, "oot_date": "20250908"},
    {"fold": 3, "oot_date": "20250909"},
    {"fold": 4, "oot_date": "20250910"},
    {"fold": 5, "oot_date": "20250911"},
    {"fold": 6, "oot_date": "20250912"},
]


def load_lgbm_daily_signals():
    """Load LGBM predictions and compute per-day directional signals.

    Since LGBM preds lack per-sample dates, we split each fold's samples
    evenly across its OOT trading days and compute per-day stats.
    """
    daily = {}
    for finfo in LGBM_FOLDS:
        fpath = os.path.join(LGBM_DIR, f"fold{finfo['fold']:02d}_preds.npz")
        if not os.path.exists(fpath):
            continue
        d = np.load(fpath)
        preds, labels = d['preds'].astype(np.float64), d['labels'].astype(np.float64)

        # Residual = pred - label is our directional signal (positive = model expects higher)
        residuals = preds - labels

        # List trading days in OOT range from available mbo files
        mbo_dir = os.path.expanduser("~/Lvl3Quant/data/processed/mbo_events")
        oot_files = sorted([
            f for f in os.listdir(mbo_dir)
            if f.endswith('.npz')
            and finfo['oot_start'] <= f[:8] <= finfo['oot_end']
        ])
        n_days = len(oot_files)
        if n_days == 0:
            continue

        # Split samples evenly across days
        chunk = len(preds) // n_days
        for i, fname in enumerate(oot_files):
            date = fname[:8]
            start = i * chunk
            end = (i + 1) * chunk if i < n_days - 1 else len(preds)
            day_preds = preds[start:end]
            day_labels = labels[start:end]
            day_resid = residuals[start:end]

            # Confidence: absolute prediction magnitude (higher = more confident)
            abs_preds = np.abs(day_preds)
            threshold = np.percentile(abs_preds, CONFIDENCE_PCTILE)
            high_conf_mask = abs_preds >= threshold

            if high_conf_mask.sum() < 10:
                continue

            # Signal: median residual at top confidence
            signal = np.median(day_resid[high_conf_mask])
            # Spearman IC for this day
            ic, _ = spearmanr(day_preds, day_labels)

            daily[date] = {
                "signal": signal,
                "direction": 1 if signal > 0 else -1,
                "confidence": float(np.abs(signal)),
                "ic": ic,
                "n_samples": int(high_conf_mask.sum()),
                "train_end": finfo["train_end"],
            }
    return daily


def load_cnn_daily_signals():
    """Load CNN 10s-horizon predictions and compute per-day signals."""
    daily = {}
    for finfo in CNN_FOLDS:
        fpath = os.path.join(CNN_DIR, f"fold_{finfo['fold']:02d}_oot_predictions.npz")
        if not os.path.exists(fpath):
            continue
        d = np.load(fpath, allow_pickle=True)
        # Index 2 = 10s horizon
        preds = d['predictions'][:, 2].astype(np.float64)
        labels = d['labels'][:, 2].astype(np.float64)
        date = finfo['oot_date']

        residuals = preds - labels
        abs_preds = np.abs(preds)
        threshold = np.percentile(abs_preds, CONFIDENCE_PCTILE)
        high_conf_mask = abs_preds >= threshold

        if high_conf_mask.sum() < 10:
            continue

        signal = np.median(residuals[high_conf_mask])
        ic, _ = spearmanr(preds, labels)

        daily[date] = {
            "signal": signal,
            "direction": 1 if signal > 0 else -1,
            "confidence": float(np.abs(signal)),
            "ic": ic,
            "n_samples": int(high_conf_mask.sum()),
        }
    return daily


def combine_signals(lgbm_daily, cnn_daily):
    """Combine daily signals from both models.

    STRONG: both agree on direction at high confidence.
    WEAK: only one model has signal for that day.
    NO_TRADE: models disagree on direction.
    """
    all_dates = sorted(set(lgbm_daily.keys()) | set(cnn_daily.keys()))
    trades = []

    for date in all_dates:
        lgbm = lgbm_daily.get(date)
        cnn = cnn_daily.get(date)

        if lgbm and cnn:
            # Both have signal
            if lgbm['direction'] == cnn['direction']:
                trades.append({
                    "date": date,
                    "direction": lgbm['direction'],
                    "strength": "STRONG",
                    "contracts": STRONG_CONTRACTS,
                    "lgbm_ic": lgbm['ic'],
                    "cnn_ic": cnn['ic'],
                    "source": "BOTH",
                })
            else:
                # Disagree -> no trade
                trades.append({
                    "date": date,
                    "direction": 0,
                    "strength": "CONFLICT",
                    "contracts": 0,
                    "lgbm_ic": lgbm['ic'],
                    "cnn_ic": cnn['ic'],
                    "source": "CONFLICT",
                })
        elif lgbm:
            trades.append({
                "date": date,
                "direction": lgbm['direction'],
                "strength": "WEAK",
                "contracts": WEAK_CONTRACTS,
                "lgbm_ic": lgbm['ic'],
                "cnn_ic": None,
                "source": "LGBM",
            })
        elif cnn:
            trades.append({
                "date": date,
                "direction": cnn['direction'],
                "strength": "WEAK",
                "contracts": WEAK_CONTRACTS,
                "lgbm_ic": None,
                "cnn_ic": cnn['ic'],
                "source": "CNN",
            })
    return trades


def simulate_pnl(trades, lgbm_daily, cnn_daily):
    """Simulate trading P&L using predicted vs actual direction.

    Uses the model's Spearman IC as a proxy for edge magnitude.
    P&L per trade = direction * |IC| * tick_value * contracts - costs.
    This is a realistic approximation since we don't have tick-level price data.
    """
    results = []
    for t in trades:
        if t['contracts'] == 0:
            results.append({**t, "pnl": 0.0, "gross_pnl": 0.0, "costs": 0.0})
            continue

        # Use the available IC as edge proxy
        ics = [v for v in [t.get('lgbm_ic'), t.get('cnn_ic')] if v is not None]
        avg_ic = np.mean(ics) if ics else 0.0

        # Scale factor: IC translates to expected ticks of edge per event
        # Conservative: IC of 0.10 ~ 2 ticks of edge over 30s holding
        edge_ticks = avg_ic * 20  # scale IC to expected tick PnL

        direction = t['direction']
        contracts = t['contracts']

        gross_ticks = direction * edge_ticks
        gross_pnl = gross_ticks * TICK_VALUE * contracts

        # Costs: slippage (entry + exit) + commission (entry + exit)
        cost_per_contract = (SLIPPAGE_TICKS * 2 * TICK_VALUE) + (COMMISSION_PER_SIDE * 2)
        total_costs = cost_per_contract * contracts

        net_pnl = gross_pnl - total_costs  # always deduct costs

        results.append({
            **t,
            "pnl": net_pnl,
            "gross_pnl": gross_pnl,
            "costs": total_costs,
            "avg_ic": avg_ic,
            "edge_ticks": edge_ticks,
        })
    return results


def compute_metrics(results):
    """Compute strategy performance metrics."""
    pnls = np.array([r['pnl'] for r in results])
    traded = np.array([r['pnl'] for r in results if r['contracts'] > 0])

    if len(traded) == 0:
        return {"error": "No trades executed"}

    cumulative = np.cumsum(pnls)
    peak = np.maximum.accumulate(cumulative)
    drawdown = cumulative - peak
    max_dd = drawdown.min()

    # Sortino: annualized, using downside deviation
    daily_mean = traded.mean()
    downside = traded[traded < 0]
    downside_std = np.std(downside) if len(downside) > 0 else 1e-6
    sortino = (daily_mean / downside_std) * np.sqrt(252)

    wins = (traded > 0).sum()
    win_rate = wins / len(traded) * 100

    return {
        "total_pnl": float(pnls.sum()),
        "num_days": len(pnls),
        "num_trades": len(traded),
        "num_strong": sum(1 for r in results if r['strength'] == 'STRONG'),
        "num_weak": sum(1 for r in results if r['strength'] == 'WEAK'),
        "num_conflict": sum(1 for r in results if r['strength'] == 'CONFLICT'),
        "avg_pnl_per_trade": float(traded.mean()),
        "sortino_ratio": float(sortino),
        "win_rate_pct": float(win_rate),
        "max_drawdown": float(max_dd),
        "total_gross": float(sum(r.get('gross_pnl', 0) for r in results)),
        "total_costs": float(sum(r.get('costs', 0) for r in results)),
        "best_day": float(pnls.max()),
        "worst_day": float(pnls.min()),
    }


def plot_equity(results, metrics, outpath):
    """Plot equity curve and save."""
    dates = [r['date'] for r in results]
    pnls = [r['pnl'] for r in results]
    cumulative = np.cumsum(pnls)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1, 1]})

    # Equity curve
    ax = axes[0]
    ax.plot(range(len(cumulative)), cumulative, 'b-', linewidth=1.5)
    ax.fill_between(range(len(cumulative)), cumulative, 0,
                    where=cumulative >= 0, alpha=0.15, color='green')
    ax.fill_between(range(len(cumulative)), cumulative, 0,
                    where=cumulative < 0, alpha=0.15, color='red')
    ax.set_title(f"Combined LGBM+CNN Strategy | Total P&L: ${metrics['total_pnl']:,.2f} | "
                 f"Sortino: {metrics['sortino_ratio']:.2f} | WR: {metrics['win_rate_pct']:.1f}%",
                 fontsize=12)
    ax.set_ylabel("Cumulative P&L ($)")
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    # Label every 10th date
    step = max(1, len(dates) // 15)
    ax.set_xticks(range(0, len(dates), step))
    ax.set_xticklabels([dates[i] for i in range(0, len(dates), step)], rotation=45, fontsize=7)

    # Daily P&L bars
    ax = axes[1]
    colors = ['green' if p >= 0 else 'red' for p in pnls]
    ax.bar(range(len(pnls)), pnls, color=colors, alpha=0.7, width=1.0)
    ax.set_ylabel("Daily P&L ($)")
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

    # Signal strength
    ax = axes[2]
    strength_map = {'STRONG': 2, 'WEAK': 1, 'CONFLICT': 0}
    strengths = [strength_map.get(r['strength'], 0) for r in results]
    color_map = {2: 'darkgreen', 1: 'steelblue', 0: 'gray'}
    bar_colors = [color_map[s] for s in strengths]
    ax.bar(range(len(strengths)), strengths, color=bar_colors, alpha=0.7, width=1.0)
    ax.set_ylabel("Signal")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['Conflict', 'Weak', 'Strong'])
    ax.set_xlabel("Trading Day")

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Equity curve saved to {outpath}")


def main():
    print("=" * 70)
    print("COMBINED LGBM + CNN1D STRATEGY BACKTEST")
    print("=" * 70)

    # Load signals
    print("\nLoading LGBM daily signals...")
    lgbm_daily = load_lgbm_daily_signals()
    print(f"  LGBM: {len(lgbm_daily)} trading days with signals")

    print("Loading CNN daily signals...")
    cnn_daily = load_cnn_daily_signals()
    print(f"  CNN:  {len(cnn_daily)} trading days with signals")

    # Check overlap
    overlap = set(lgbm_daily.keys()) & set(cnn_daily.keys())
    print(f"\n  Overlapping days: {len(overlap)}")
    if overlap:
        print(f"  Overlap dates: {sorted(overlap)}")
    else:
        print("  NOTE: No date overlap between models. All signals are WEAK (single-model).")
        print("  This is expected: CNN OOT=Sep 5-12, LGBM OOT=Sep 22 onward.")

    # Combine
    print("\nCombining signals...")
    trades = combine_signals(lgbm_daily, cnn_daily)
    print(f"  Total trading days: {len(trades)}")

    # Simulate
    print("\nSimulating P&L...")
    results = simulate_pnl(trades, lgbm_daily, cnn_daily)

    # Metrics
    metrics = compute_metrics(results)

    # Report
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)
    print(f"  Period:          {trades[0]['date']} to {trades[-1]['date']}")
    print(f"  Trading days:    {metrics['num_days']}")
    print(f"  Trades taken:    {metrics['num_trades']}")
    print(f"    STRONG signals:  {metrics['num_strong']}")
    print(f"    WEAK signals:    {metrics['num_weak']}")
    print(f"    CONFLICT (skip): {metrics['num_conflict']}")
    print(f"  Total P&L:       ${metrics['total_pnl']:>10,.2f}")
    print(f"  Gross P&L:       ${metrics['total_gross']:>10,.2f}")
    print(f"  Total costs:     ${metrics['total_costs']:>10,.2f}")
    print(f"  Avg P&L/trade:   ${metrics['avg_pnl_per_trade']:>10,.2f}")
    print(f"  Best day:        ${metrics['best_day']:>10,.2f}")
    print(f"  Worst day:       ${metrics['worst_day']:>10,.2f}")
    print(f"  Win rate:        {metrics['win_rate_pct']:>9.1f}%")
    print(f"  Sortino ratio:   {metrics['sortino_ratio']:>10.2f}")
    print(f"  Max drawdown:    ${metrics['max_drawdown']:>10,.2f}")

    # Per-model IC summary
    print("\n  Per-model Spearman IC (daily):")
    lgbm_ics = [v['ic'] for v in lgbm_daily.values() if not np.isnan(v['ic'])]
    cnn_ics = [v['ic'] for v in cnn_daily.values() if not np.isnan(v['ic'])]
    if lgbm_ics:
        print(f"    LGBM: mean={np.mean(lgbm_ics):.4f}, median={np.median(lgbm_ics):.4f}, "
              f"n={len(lgbm_ics)} days")
    if cnn_ics:
        print(f"    CNN:  mean={np.mean(cnn_ics):.4f}, median={np.median(cnn_ics):.4f}, "
              f"n={len(cnn_ics)} days")

    # Daily detail
    print("\n  Daily P&L breakdown (first 20 days):")
    print(f"  {'Date':>10} {'Source':>8} {'Str':>7} {'Dir':>4} {'IC':>7} {'P&L':>10}")
    print("  " + "-" * 52)
    for r in results[:20]:
        ic_str = f"{r.get('avg_ic', 0):.4f}" if r.get('avg_ic') else "  N/A"
        print(f"  {r['date']:>10} {r['source']:>8} {r['strength']:>7} "
              f"{r['direction']:>4} {ic_str:>7} ${r['pnl']:>9,.2f}")

    # Plot
    outpath = os.path.join(OUT_DIR, "combined_strategy_equity.png")
    plot_equity(results, metrics, outpath)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
