"""
Execution Backtest — Full P&L simulation using SmartExecutionEngine.
====================================================================

Takes walk-forward LightGBM predictions and simulates trading through
the execution engine, computing realistic P&L with costs.

Three execution modes compared:
  1. RAW signal (no execution engine) — market orders on every prediction
  2. SMART execution — filtered through vol gate, imbalance, routing
  3. PASSIVE-FIRST — limit order escalation strategy

Output: P&L curves, Sharpe, drawdown, trade analytics, charts.

Usage:
    from alpha_discovery.execution_backtest import run_execution_backtest
    results = run_execution_backtest(
        predictions, actuals, mid_prices, features, feature_names,
        day_boundaries, horizon='ret_3s'
    )
"""

import gc
import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

log = logging.getLogger('execution_backtest')

# ============================================================================
# CONSTANTS
# ============================================================================
TICK_SIZE = 0.25
TICK_VALUE = 12.50
COMMISSION_RT = 4.70  # HC #52: $4.70 RT (AMP)          # AMP + Rithmic + CME per round-trip
SPREAD_COST_TICKS = 1.0       # crossing the spread = 1 tick each way
COMMISSION_TICKS = COMMISSION_RT / TICK_VALUE  # 0.24 ticks

# Total cost per execution mode (in ticks)
COST_MARKET_ORDER = SPREAD_COST_TICKS + COMMISSION_TICKS   # 1.24 ticks
COST_LIMIT_ORDER = COMMISSION_TICKS                         # 0.24 ticks
COST_MIDPOINT = 0.5 + COMMISSION_TICKS                     # 0.74 ticks
COST_AGGRESSIVE_LIMIT = 0.75 + COMMISSION_TICKS             # 0.99 ticks

HORIZON_STEPS = {
    'ret_3s': 30,
    'ret_5s': 50,
    'ret_10s': 100,
    'ret_30s': 300,
    'ret_1m': 600,
    'ret_5m': 3000,
}

RESULTS_DIR = Path(__file__).parent / 'results'


# ============================================================================
# FEATURE-TO-BARDATA MAPPING
# ============================================================================

# Map BarData fields to feature column names
BARDATA_FEATURE_MAP = {
    'spread': 'spread',
    'bid_depth': 'total_bid_vol',
    'ask_depth': 'total_ask_vol',
    'vol_imbalance': 'vol_imbalance',
    'ofi_5': 'ofi_5',
    'ofi_20': 'ofi_20',
    'ofi_50': 'ofi_50',
    'trade_imbalance': 'trade_imbalance',
    'vpin_50': 'vpin_50',
    'hour_norm': 'hour_norm',
    'ret_5': 'ret_5',
    'realized_vol_20': 'rvol_20',
    'realized_vol_50': 'rvol_50',
    'spread_ticks': 'spread_ticks',
}


def _build_feature_index(feature_names: List[str]) -> Dict[str, int]:
    """Build name -> column index mapping."""
    return {name: i for i, name in enumerate(feature_names)}


def _features_to_bardata(features_row: np.ndarray, mid: float,
                          feat_idx: Dict[str, int]) -> dict:
    """Convert feature array row to BarData-compatible dict."""
    bar = {'mid': mid}
    for bd_field, feat_name in BARDATA_FEATURE_MAP.items():
        if feat_name in feat_idx:
            val = float(features_row[feat_idx[feat_name]])
            if np.isfinite(val):
                bar[bd_field] = val
            else:
                bar[bd_field] = 0.0
        else:
            bar[bd_field] = 0.0
    return bar


# ============================================================================
# TRADE TRACKING
# ============================================================================

@dataclass
class Trade:
    """Record of a single simulated trade."""
    entry_bar: int
    exit_bar: int
    direction: int            # +1 long, -1 short
    entry_price: float
    exit_price: float
    order_type: str           # 'market', 'limit', 'midpoint', 'aggressive_limit'
    cost_ticks: float         # total RT cost in ticks
    signal_strength: float
    confidence: float
    regime: str

    @property
    def gross_pnl_ticks(self) -> float:
        """Gross P&L in ticks (before costs)."""
        move = (self.exit_price - self.entry_price) / TICK_SIZE
        return move * self.direction

    @property
    def net_pnl_ticks(self) -> float:
        """Net P&L in ticks (after costs)."""
        return self.gross_pnl_ticks - self.cost_ticks

    @property
    def net_pnl_dollars(self) -> float:
        """Net P&L in dollars."""
        return self.net_pnl_ticks * TICK_VALUE

    @property
    def is_winner(self) -> bool:
        return self.net_pnl_ticks > 0


# ============================================================================
# RAW SIGNAL BACKTEST (no execution engine)
# ============================================================================

def _backtest_raw_signal(
    predictions: np.ndarray,
    mid_prices: np.ndarray,
    day_boundaries: np.ndarray,
    horizon_steps: int,
    signal_threshold: float = 0.0,
    cost_mode: str = 'market',
) -> List[Trade]:
    """Simple backtest: trade whenever |prediction| > threshold.

    Args:
        cost_mode: 'market' (cross spread) or 'limit' (join queue, optimistic)
    """
    cost = COST_MARKET_ORDER if cost_mode == 'market' else COST_LIMIT_ORDER
    n_days = len(day_boundaries) - 1
    trades = []

    for d in range(n_days):
        day_start = day_boundaries[d]
        day_end = day_boundaries[d + 1]

        i = day_start
        while i < day_end - horizon_steps:
            pred = predictions[i]
            if abs(pred) <= signal_threshold:
                i += 1
                continue

            direction = 1 if pred > 0 else -1
            entry_price = mid_prices[i]
            exit_bar = i + horizon_steps
            exit_price = mid_prices[min(exit_bar, day_end - 1)]

            trades.append(Trade(
                entry_bar=i,
                exit_bar=exit_bar,
                direction=direction,
                entry_price=entry_price,
                exit_price=exit_price,
                order_type=cost_mode,
                cost_ticks=cost,
                signal_strength=abs(pred),
                confidence=abs(pred),
                regime='unknown',
            ))

            # Skip forward by horizon (no overlapping trades)
            i = exit_bar
            continue

    return trades


# ============================================================================
# SMART EXECUTION BACKTEST
# ============================================================================

def _backtest_smart_execution(
    predictions: np.ndarray,
    mid_prices: np.ndarray,
    features: np.ndarray,
    feature_names: List[str],
    day_boundaries: np.ndarray,
    horizon_steps: int,
    horizon_name: str,
    signal_threshold_pct: float = 70.0,
    min_confidence: float = 0.3,
) -> Tuple[List[Trade], dict]:
    """Backtest using the full SmartExecutionEngine.

    Returns (trades, engine_diagnostics).
    """
    from alpha_discovery.execution_engine import SmartExecutionEngine, BarData

    engine = SmartExecutionEngine(
        mode='backtest',
        max_contracts=1,
        min_confidence=min_confidence,
    )

    feat_idx = _build_feature_index(feature_names)
    n_days = len(day_boundaries) - 1
    trades = []

    # Normalize signal strengths to [0, 1] for the engine
    abs_preds = np.abs(predictions)
    valid = np.isfinite(abs_preds) & (abs_preds > 0)
    if valid.sum() > 0:
        # Use percentile-based normalization
        p95 = np.percentile(abs_preds[valid], 95)
        if p95 > 0:
            norm_strength = np.clip(abs_preds / p95, 0, 1)
        else:
            norm_strength = np.zeros_like(abs_preds)
    else:
        norm_strength = np.zeros_like(abs_preds)

    # Cost lookup by order type
    cost_by_type = {
        'market': COST_MARKET_ORDER,
        'aggressive_limit': COST_AGGRESSIVE_LIMIT,
        'midpoint': COST_MIDPOINT,
        'limit': COST_LIMIT_ORDER,
        'skip': 0,
    }

    for d in range(n_days):
        day_start = day_boundaries[d]
        day_end = day_boundaries[d + 1]

        # Reset intraday state at day start
        engine.vol_gate.intraday_vols = []

        i = day_start
        while i < day_end - horizon_steps:
            pred = predictions[i]
            if not np.isfinite(pred) or abs(pred) < 1e-12:
                i += 1
                continue

            sig_dir = 1 if pred > 0 else -1
            sig_strength = float(norm_strength[i])

            # Build BarData from features
            bar_dict = _features_to_bardata(features[i], mid_prices[i], feat_idx)
            bar = BarData(**bar_dict)

            # Ask engine for decision
            decision = engine.process(
                signal_strength=sig_strength,
                signal_direction=sig_dir,
                horizon=horizon_name,
                bar=bar,
                bar_index=i,
            )

            if decision.action == 'enter':
                entry_price = mid_prices[i]
                exit_bar = min(i + horizon_steps, day_end - 1)
                exit_price = mid_prices[exit_bar]

                order_type = decision.order_type
                cost = cost_by_type.get(order_type, COST_MARKET_ORDER)

                actual_return = (exit_price - entry_price) / TICK_SIZE * sig_dir
                engine.record_outcome(
                    prediction=pred,
                    actual_return=actual_return * TICK_SIZE / max(entry_price, 1.0),
                    bar=bar,
                )

                trades.append(Trade(
                    entry_bar=i,
                    exit_bar=exit_bar,
                    direction=sig_dir,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    order_type=order_type,
                    cost_ticks=cost,
                    signal_strength=sig_strength,
                    confidence=decision.confidence,
                    regime=engine.vol_gate.get_regime(),
                ))

                # Skip forward by horizon
                i = exit_bar
                continue

            i += 1

    diagnostics = engine.get_diagnostics()
    return trades, diagnostics


# ============================================================================
# ANALYTICS
# ============================================================================

def compute_trade_analytics(trades: List[Trade], label: str = '') -> dict:
    """Compute comprehensive analytics from a list of trades."""
    if not trades:
        return {
            'label': label,
            'n_trades': 0,
            'total_pnl_ticks': 0,
            'total_pnl_dollars': 0,
            'error': 'no trades',
        }

    gross = np.array([t.gross_pnl_ticks for t in trades])
    net = np.array([t.net_pnl_ticks for t in trades])
    net_dollars = np.array([t.net_pnl_dollars for t in trades])
    costs = np.array([t.cost_ticks for t in trades])

    # Cumulative P&L
    cum_pnl = np.cumsum(net_dollars)

    # Drawdown
    peak = np.maximum.accumulate(cum_pnl)
    drawdown = cum_pnl - peak
    max_dd = float(np.min(drawdown))

    # Win rate
    winners = sum(1 for t in trades if t.is_winner)
    win_rate = winners / len(trades) if trades else 0

    # Profit factor
    gross_profit = float(np.sum(net_dollars[net_dollars > 0]))
    gross_loss = float(abs(np.sum(net_dollars[net_dollars < 0])))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Sharpe (daily)
    # Group trades by entry day for daily P&L
    daily_pnl = {}
    for t in trades:
        day = t.entry_bar  # approximate
        daily_pnl[day] = daily_pnl.get(day, 0) + t.net_pnl_dollars

    daily_returns = np.array(list(daily_pnl.values()))
    if len(daily_returns) > 1 and np.std(daily_returns) > 0:
        sharpe = float(np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252))
    else:
        sharpe = 0.0

    # Order type breakdown
    type_counts = {}
    for t in trades:
        type_counts[t.order_type] = type_counts.get(t.order_type, 0) + 1

    # Regime breakdown
    regime_counts = {}
    regime_pnl = {}
    for t in trades:
        r = t.regime
        regime_counts[r] = regime_counts.get(r, 0) + 1
        regime_pnl[r] = regime_pnl.get(r, 0) + t.net_pnl_dollars

    return {
        'label': label,
        'n_trades': len(trades),
        'total_pnl_ticks': float(np.sum(net)),
        'total_pnl_dollars': float(np.sum(net_dollars)),
        'total_cost_ticks': float(np.sum(costs)),
        'total_cost_dollars': float(np.sum(costs) * TICK_VALUE),
        'avg_trade_ticks': float(np.mean(net)),
        'avg_trade_dollars': float(np.mean(net_dollars)),
        'avg_gross_ticks': float(np.mean(gross)),
        'win_rate': win_rate,
        'winners': winners,
        'losers': len(trades) - winners,
        'profit_factor': profit_factor,
        'sharpe_annualized': sharpe,
        'max_drawdown_dollars': max_dd,
        'gross_profit_dollars': gross_profit,
        'gross_loss_dollars': gross_loss,
        'cum_pnl_curve': cum_pnl.tolist(),
        'drawdown_curve': drawdown.tolist(),
        'order_type_breakdown': type_counts,
        'regime_breakdown': regime_counts,
        'regime_pnl': regime_pnl,
        'best_trade_dollars': float(np.max(net_dollars)),
        'worst_trade_dollars': float(np.min(net_dollars)),
        'avg_winner_dollars': float(np.mean(net_dollars[net_dollars > 0])) if (net_dollars > 0).any() else 0,
        'avg_loser_dollars': float(np.mean(net_dollars[net_dollars < 0])) if (net_dollars < 0).any() else 0,
    }


# ============================================================================
# CHART GENERATION
# ============================================================================

def generate_execution_charts(
    results: Dict[str, dict],
    horizon: str,
    output_dir: Path = RESULTS_DIR,
) -> List[str]:
    """Generate P&L and analytics charts. Returns list of saved file paths."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available, skipping charts")
        return []

    saved = []
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # --- Chart 1: Cumulative P&L Comparison ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

    ax1 = axes[0]
    for label, data in results.items():
        if data.get('n_trades', 0) == 0:
            continue
        curve = data.get('cum_pnl_curve', [])
        if curve:
            ax1.plot(range(len(curve)), curve, label=f"{label} ({data['n_trades']} trades)")

    ax1.set_title(f'Cumulative P&L — {horizon} (1 ES contract)', fontsize=14)
    ax1.set_ylabel('P&L ($)')
    ax1.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Drawdown subplot
    ax2 = axes[1]
    for label, data in results.items():
        if data.get('n_trades', 0) == 0:
            continue
        dd = data.get('drawdown_curve', [])
        if dd:
            ax2.fill_between(range(len(dd)), dd, 0, alpha=0.3, label=label)

    ax2.set_title('Drawdown', fontsize=11)
    ax2.set_ylabel('Drawdown ($)')
    ax2.set_xlabel('Trade #')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / f'execution_pnl_{horizon}_{timestamp}.png'
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    saved.append(str(path))

    # --- Chart 2: Trade Analytics Summary ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Win rate comparison
    ax = axes[0, 0]
    labels = []
    win_rates = []
    for label, data in results.items():
        if data.get('n_trades', 0) > 0:
            labels.append(label)
            win_rates.append(data['win_rate'] * 100)
    if labels:
        bars = ax.bar(labels, win_rates, color=['#e74c3c', '#2ecc71', '#3498db'][:len(labels)])
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        ax.set_title('Win Rate (%)')
        ax.set_ylabel('%')
        for bar, val in zip(bars, win_rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', fontsize=10)

    # Avg trade P&L
    ax = axes[0, 1]
    avg_pnls = []
    for label, data in results.items():
        if data.get('n_trades', 0) > 0:
            avg_pnls.append(data['avg_trade_dollars'])
    if labels and avg_pnls:
        colors = ['green' if x > 0 else 'red' for x in avg_pnls]
        bars = ax.bar(labels, avg_pnls, color=colors)
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_title('Avg Trade P&L ($)')
        ax.set_ylabel('$')
        for bar, val in zip(bars, avg_pnls):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'${val:.2f}', ha='center', fontsize=10)

    # Order type distribution (smart execution only)
    ax = axes[1, 0]
    smart_data = results.get('Smart Execution', {})
    if smart_data.get('order_type_breakdown'):
        types = smart_data['order_type_breakdown']
        ax.pie(types.values(), labels=types.keys(), autopct='%1.1f%%', startangle=90)
        ax.set_title('Order Type Distribution (Smart)')
    else:
        ax.text(0.5, 0.5, 'No smart execution trades', ha='center', va='center')

    # Sharpe / profit factor comparison
    ax = axes[1, 1]
    metrics = ['Sharpe', 'Profit Factor']
    x = np.arange(len(metrics))
    width = 0.25
    for idx, (label, data) in enumerate(results.items()):
        if data.get('n_trades', 0) > 0:
            values = [data['sharpe_annualized'],
                      min(data['profit_factor'], 5.0)]  # cap for display
            ax.bar(x + idx * width, values, width, label=label)
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics)
    ax.set_title('Performance Metrics')
    ax.legend(fontsize=9)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    path = output_dir / f'execution_analytics_{horizon}_{timestamp}.png'
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    saved.append(str(path))

    return saved


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def run_execution_backtest(
    predictions: np.ndarray,
    actuals: np.ndarray,
    mid_prices: np.ndarray,
    features: np.ndarray,
    feature_names: List[str],
    day_boundaries: np.ndarray,
    horizon: str = 'ret_3s',
    signal_percentile_threshold: float = 70.0,
) -> dict:
    """Run comprehensive execution backtest comparing multiple strategies.

    Args:
        predictions: Walk-forward OOS predictions, shape (N,)
        actuals: Actual returns, shape (N,)
        mid_prices: Mid prices, shape (N,)
        features: Full feature matrix, shape (N, F) - needed for BarData
        feature_names: List of feature names
        day_boundaries: Array of day start indices
        horizon: Horizon name (e.g., 'ret_3s', 'ret_5s')
        signal_percentile_threshold: Only consider top X% of signals

    Returns:
        Dict with analytics for each strategy, plus charts.
    """
    horizon_steps = HORIZON_STEPS.get(horizon, 30)
    log.info(f"\n{'='*70}")
    log.info(f"EXECUTION BACKTEST — {horizon} (hold={horizon_steps} bars)")
    log.info(f"{'='*70}")
    log.info(f"  Bars: {len(predictions):,}")
    log.info(f"  Days: {len(day_boundaries)-1}")
    log.info(f"  Valid predictions: {np.isfinite(predictions).sum():,}")

    # Signal threshold based on percentile
    abs_preds = np.abs(predictions[np.isfinite(predictions)])
    if len(abs_preds) > 0:
        threshold = float(np.percentile(abs_preds, signal_percentile_threshold))
    else:
        threshold = 0.0
    log.info(f"  Signal threshold ({signal_percentile_threshold}th pctile): {threshold:.6f}")

    all_results = {}

    # --- Strategy 1: Raw Market Orders ---
    log.info("\n[1/3] Raw Market Orders (no filtering)...")
    raw_trades = _backtest_raw_signal(
        predictions, mid_prices, day_boundaries, horizon_steps,
        signal_threshold=threshold, cost_mode='market',
    )
    raw_analytics = compute_trade_analytics(raw_trades, 'Raw Market')
    all_results['Raw Market'] = raw_analytics
    log.info(
        f"  {raw_analytics['n_trades']} trades, "
        f"P&L=${raw_analytics['total_pnl_dollars']:.2f}, "
        f"Win={raw_analytics['win_rate']:.1%}, "
        f"Sharpe={raw_analytics['sharpe_annualized']:.2f}"
    )

    # --- Strategy 2: Raw Limit Orders (optimistic fill assumption) ---
    log.info("\n[2/3] Raw Limit Orders (optimistic, no adverse selection)...")
    limit_trades = _backtest_raw_signal(
        predictions, mid_prices, day_boundaries, horizon_steps,
        signal_threshold=threshold, cost_mode='limit',
    )
    limit_analytics = compute_trade_analytics(limit_trades, 'Raw Limit')
    all_results['Raw Limit'] = limit_analytics
    log.info(
        f"  {limit_analytics['n_trades']} trades, "
        f"P&L=${limit_analytics['total_pnl_dollars']:.2f}, "
        f"Win={limit_analytics['win_rate']:.1%}, "
        f"Sharpe={limit_analytics['sharpe_annualized']:.2f}"
    )

    # --- Strategy 3: Smart Execution Engine ---
    log.info("\n[3/3] Smart Execution Engine (full filtering)...")
    smart_trades, engine_diag = _backtest_smart_execution(
        predictions, mid_prices, features, feature_names,
        day_boundaries, horizon_steps, horizon,
        signal_threshold_pct=signal_percentile_threshold,
    )
    smart_analytics = compute_trade_analytics(smart_trades, 'Smart Execution')
    smart_analytics['engine_diagnostics'] = engine_diag
    all_results['Smart Execution'] = smart_analytics
    log.info(
        f"  {smart_analytics['n_trades']} trades, "
        f"P&L=${smart_analytics['total_pnl_dollars']:.2f}, "
        f"Win={smart_analytics['win_rate']:.1%}, "
        f"Sharpe={smart_analytics['sharpe_annualized']:.2f}"
    )
    if engine_diag.get('skips'):
        skips = engine_diag['skips']
        log.info(
            f"  Skips: regime={skips.get('regime',0)}, "
            f"imbalance={skips.get('imbalance',0)}, "
            f"confidence={skips.get('confidence',0)}, "
            f"router={skips.get('router',0)}"
        )

    # --- Generate Charts ---
    log.info("\nGenerating charts...")
    chart_paths = generate_execution_charts(all_results, horizon)
    for p in chart_paths:
        log.info(f"  Saved: {p}")

    # --- Summary comparison ---
    log.info(f"\n{'='*70}")
    log.info("EXECUTION BACKTEST SUMMARY")
    log.info(f"{'='*70}")
    log.info(f"{'Strategy':<20} {'Trades':>7} {'P&L ($)':>10} {'Avg ($)':>9} "
             f"{'Win%':>6} {'Sharpe':>7} {'MaxDD($)':>10} {'PF':>6}")
    log.info("-" * 85)
    for label, data in all_results.items():
        if data.get('n_trades', 0) > 0:
            log.info(
                f"{label:<20} {data['n_trades']:>7} "
                f"${data['total_pnl_dollars']:>9.0f} "
                f"${data['avg_trade_dollars']:>8.2f} "
                f"{data['win_rate']:>5.1%} "
                f"{data['sharpe_annualized']:>7.2f} "
                f"${data['max_drawdown_dollars']:>9.0f} "
                f"{data['profit_factor']:>6.2f}"
            )
        else:
            log.info(f"{label:<20} {'No trades':>7}")

    # --- Save JSON ---
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = {
        'horizon': horizon,
        'horizon_steps': horizon_steps,
        'n_bars': len(predictions),
        'n_days': len(day_boundaries) - 1,
        'signal_threshold_percentile': signal_percentile_threshold,
        'costs': {
            'commission_rt': COMMISSION_RT,
            'spread_cost_ticks': SPREAD_COST_TICKS,
            'market_order_ticks': COST_MARKET_ORDER,
            'limit_order_ticks': COST_LIMIT_ORDER,
        },
        'strategies': {},
        'chart_paths': chart_paths,
        'timestamp': datetime.now().isoformat(),
    }

    for label, data in all_results.items():
        # Remove numpy arrays for JSON serialization
        serializable = {k: v for k, v in data.items()
                        if k not in ('cum_pnl_curve', 'drawdown_curve')}
        output['strategies'][label] = serializable

    result_path = RESULTS_DIR / f'execution_backtest_{horizon}_{timestamp}.json'
    with open(str(result_path), 'w') as f:
        json.dump(output, f, indent=2, default=str)
    log.info(f"\nResults saved: {result_path}")

    return output


# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    """Quick test with synthetic data."""
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s')

    N = 100_000
    np.random.seed(42)
    mid = 6000.0 + np.cumsum(np.random.randn(N) * 0.1)
    preds = np.random.randn(N) * 0.001
    actuals = np.random.randn(N) * 0.001
    features = np.random.randn(N, 149).astype(np.float32)
    from alpha_discovery.mbo_features import get_feature_names
    feat_names = get_feature_names()
    day_bounds = np.array([0, 23400, 46800, 70200, 93600, N])

    result = run_execution_backtest(
        preds, actuals, mid.astype(np.float32), features, feat_names,
        day_bounds, horizon='ret_3s'
    )
    print(f"\nDone. {result['strategies']['Raw Market'].get('n_trades', 0)} raw trades simulated.")
