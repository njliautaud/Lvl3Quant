"""
Test raw microstructure signals for predictive power.
=====================================================
No ML needed — just check if order book features predict future price moves.

Feature map (from src/features/engineering.py global_features):
  Base (0-9):   [mid, spread, imbalance, microprice, total_bid, total_ask, avg_bid_size, avg_ask_size, best_bid, best_ask]
  Flow (10-17): [trade_imbalance, buy_vol, sell_vol, add_count, cancel_count, trade_count, cancel_to_add, trade_to_add]
  Micro(18-25): [bid_pressure, ask_pressure, pressure_imbalance, depth_concentration, bid_slope, ask_slope, spread_ticks, depth_ratio]

Usage:
    python alpha_discovery/test_raw_signals.py
    python alpha_discovery/test_raw_signals.py --days 20
"""

import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.parent
SNAP_DIR = ROOT / 'data' / 'processed' / 'medium_snapshots_cache'
MBO_DIR = ROOT / 'data' / 'processed' / 'mbo_features_cache'

# Signal definitions: (name, source, column_index)
# source: 'global' = snapshots_cache global_features, 'mbo' = mbo_features_cache
SIGNALS = [
    ('imbalance',          'global', 2),
    ('microprice_dev',     'global', 3),   # microprice - mid = directional signal
    ('trade_imbalance',    'global', 10),
    ('pressure_imbalance', 'global', 20),
    ('bid_slope',          'global', 22),
    ('ask_slope',          'global', 23),
]

# Forward return horizons (in 100ms bars)
HORIZONS = {
    '1s':  10,
    '3s':  30,
    '5s':  50,
    '10s': 100,
    '30s': 300,
    '60s': 600,
}


def compute_ic(signal: np.ndarray, forward_ret: np.ndarray) -> float:
    """Pearson correlation between signal and forward return."""
    mask = np.isfinite(signal) & np.isfinite(forward_ret)
    if mask.sum() < 100:
        return 0.0
    s, r = signal[mask], forward_ret[mask]
    # Demean
    s = s - s.mean()
    r = r - r.mean()
    denom = np.sqrt((s**2).sum() * (r**2).sum())
    if denom < 1e-12:
        return 0.0
    return float((s * r).sum() / denom)


def compute_rank_ic(signal: np.ndarray, forward_ret: np.ndarray) -> float:
    """Spearman rank IC."""
    from scipy.stats import spearmanr
    mask = np.isfinite(signal) & np.isfinite(forward_ret)
    if mask.sum() < 100:
        return 0.0
    corr, _ = spearmanr(signal[mask], forward_ret[mask])
    return float(corr) if np.isfinite(corr) else 0.0


def load_day_data(npz_path: Path):
    """Load features and mid prices for a day."""
    data = np.load(str(npz_path))
    global_feats = data.get('global_features')
    mid_prices = data.get('mid_prices')
    return global_feats, mid_prices


def test_signals_one_day(npz_path: Path, date_str: str):
    """Test all signals on one day, return IC dict."""
    global_feats, mid_prices = load_day_data(npz_path)
    if global_feats is None or mid_prices is None:
        return None

    n = len(mid_prices)
    results = {}

    # Compute forward returns for each horizon
    fwd_rets = {}
    for hz_name, hz_bars in HORIZONS.items():
        ret = np.full(n, np.nan)
        ret[:n - hz_bars] = (mid_prices[hz_bars:] - mid_prices[:n - hz_bars]) / mid_prices[:n - hz_bars]
        fwd_rets[hz_name] = ret

    for sig_name, source, col_idx in SIGNALS:
        if source == 'global':
            if col_idx >= global_feats.shape[1]:
                continue
            raw_signal = global_feats[:, col_idx].astype(np.float64)

            # Special handling: microprice_dev = microprice - mid
            if sig_name == 'microprice_dev':
                raw_signal = raw_signal - mid_prices

        results[sig_name] = {}
        for hz_name, ret in fwd_rets.items():
            ic = compute_ic(raw_signal, ret)
            results[sig_name][hz_name] = ic

    return results


def pnl_from_signal(signal: np.ndarray, mid_prices: np.ndarray,
                     threshold: float, hold_bars: int,
                     cost_ticks: float = 1.24) -> dict:
    """
    Simple vectorized PnL estimate (NOT MBO sim, just directional accuracy).
    Provides quick check before running expensive MBO sim.

    cost_ticks: 1.24 ticks = $3 RT commission + half-spread
    """
    n = len(signal)
    tick_value = 12.50
    cost_dollars = cost_ticks * tick_value  # $15.50 per trade

    trades = []
    i = 0
    while i < n - hold_bars:
        if abs(signal[i]) > threshold:
            direction = 1 if signal[i] > 0 else -1
            entry_price = mid_prices[i]
            exit_price = mid_prices[i + hold_bars]
            pnl_points = direction * (exit_price - entry_price)
            pnl_dollars = pnl_points * 50.0 - cost_dollars  # ES point value = $50
            trades.append(pnl_dollars)
            i += hold_bars  # Skip hold period
        else:
            i += 1

    if not trades:
        return {'n_trades': 0, 'total_pnl': 0, 'avg_pnl': 0, 'win_rate': 0, 'sharpe': 0}

    trades = np.array(trades)
    return {
        'n_trades': len(trades),
        'total_pnl': round(float(trades.sum()), 2),
        'avg_pnl': round(float(trades.mean()), 2),
        'win_rate': round(float((trades > 0).mean()), 4),
        'sharpe': round(float(trades.mean() / max(trades.std(), 0.01) * np.sqrt(252)), 2),
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=50, help='Max days to test')
    parser.add_argument('--pnl', action='store_true', help='Also run quick PnL estimates')
    args = parser.parse_args()

    # Find all snapshot cache files
    snap_files = sorted(SNAP_DIR.glob('*.npz'))[:args.days]
    print(f"Testing {len(snap_files)} days from snapshots cache")
    print(f"Signals: {[s[0] for s in SIGNALS]}")
    print(f"Horizons: {list(HORIZONS.keys())}")
    print()

    # Aggregate ICs across days
    all_ics = defaultdict(lambda: defaultdict(list))

    for f in snap_files:
        date_str = f.stem.replace('_snapshots', '').replace('_features', '')
        result = test_signals_one_day(f, date_str)
        if result is None:
            continue
        for sig_name, hz_ics in result.items():
            for hz_name, ic in hz_ics.items():
                all_ics[sig_name][hz_name].append(ic)

    # Print IC table
    hz_names = list(HORIZONS.keys())
    header = f"{'Signal':<25s}" + "".join(f"{'IC_'+h:>10s}" for h in hz_names) + f"  {'Consistency':>12s}"
    print("=" * len(header))
    print("INFORMATION COEFFICIENT (mean across days)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    best_signal = None
    best_ic = 0
    best_hz = None

    for sig_name, _ , _ in SIGNALS:
        row = f"{sig_name:<25s}"
        for hz in hz_names:
            ics = all_ics[sig_name][hz]
            if ics:
                mean_ic = np.mean(ics)
                row += f"{mean_ic:>10.4f}"
                if abs(mean_ic) > abs(best_ic):
                    best_ic = mean_ic
                    best_signal = sig_name
                    best_hz = hz
            else:
                row += f"{'N/A':>10s}"
        # Consistency = fraction of days with same sign IC
        best_hz_ics = all_ics[sig_name].get('10s', [])
        if best_hz_ics:
            signs = np.sign(best_hz_ics)
            consistency = max((signs > 0).mean(), (signs < 0).mean())
            row += f"  {consistency:>10.1%}"
        print(row)

    print()
    print(f"BEST: {best_signal} at {best_hz} horizon, IC = {best_ic:.4f}")

    # Detailed per-day IC for best signal
    print(f"\n{'='*60}")
    print(f"Per-day IC for {best_signal} @ 10s horizon:")
    print(f"{'='*60}")
    ics_10s = all_ics[best_signal]['10s']
    for i, ic in enumerate(ics_10s):
        marker = ' ***' if abs(ic) > 0.05 else ''
        print(f"  Day {i+1:3d}: IC = {ic:+.4f}{marker}")
    pos_days = sum(1 for x in ics_10s if x > 0)
    print(f"\n  Positive IC days: {pos_days}/{len(ics_10s)} ({pos_days/max(len(ics_10s),1)*100:.0f}%)")
    print(f"  Mean IC: {np.mean(ics_10s):.4f}, Std: {np.std(ics_10s):.4f}")
    print(f"  IC t-stat: {np.mean(ics_10s) / max(np.std(ics_10s)/np.sqrt(len(ics_10s)), 1e-6):.2f}")

    if args.pnl:
        print(f"\n{'='*60}")
        print("QUICK PnL ESTIMATES (vectorized, NOT MBO sim)")
        print(f"{'='*60}")

        for sig_name, source, col_idx in SIGNALS:
            print(f"\n--- {sig_name} ---")
            all_day_pnl = []
            for f in snap_files[:20]:  # First 20 days
                gf, mp = load_day_data(f)
                if gf is None or mp is None:
                    continue
                raw = gf[:, col_idx].astype(np.float64)
                if sig_name == 'microprice_dev':
                    raw = raw - mp

                for thresh in [0.0, 0.1, 0.2, 0.3, 0.5]:
                    for hold in [100, 300]:  # 10s, 30s
                        result = pnl_from_signal(raw, mp, thresh, hold)
                        if result['n_trades'] > 0:
                            all_day_pnl.append((thresh, hold, result))

            # Summarize best config
            if all_day_pnl:
                best = max(all_day_pnl, key=lambda x: x[2]['total_pnl'])
                print(f"  Best: thresh={best[0]}, hold={best[1]}bars")
                print(f"    Trades: {best[2]['n_trades']}, PnL: ${best[2]['total_pnl']}, "
                      f"WR: {best[2]['win_rate']:.1%}, Sharpe: {best[2]['sharpe']}")
