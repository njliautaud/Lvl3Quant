#!/usr/bin/env python3
"""
Pairs Cointegration Backtest — Gatev, Goetzmann & Rouwenhorst (2006)
Statistical pairs trading on cointegrated stock pairs.
6 variants (A-F) with 5-gate validation framework.
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import coint
from scipy import stats

warnings.filterwarnings('ignore')

# ── Custom JSON encoder for numpy types ──────────────────────────────────────
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, pd.Timestamp):
            return str(obj)
        if isinstance(obj, (datetime,)):
            return obj.isoformat()
        return super().default(obj)

# ── Configuration ────────────────────────────────────────────────────────────
PAIRS = [
    ('MSFT', 'GOOGL'),
    ('AAPL', 'MSFT'),
    ('AMD', 'NVDA'),
    ('CRM', 'ADBE'),
    ('JPM', 'GS'),
    ('V', 'MA'),
    ('AMZN', 'WMT'),
    ('HD', 'LOW'),
    ('JNJ', 'PFE'),
    ('ABBV', 'MRK'),
]

START_DATE = '2022-01-01'
END_DATE = '2026-07-28'
CAPITAL = 645.0
TRADE_SIZE = 200.0  # per leg
SLIPPAGE_PCT = 0.0002  # 0.02% per leg per trade
LOOKBACK = 60  # rolling z-score lookback
COINT_LOOKBACK = 252  # rolling cointegration test window
MAX_HOLD = 20  # max holding days

# ── Download Data ────────────────────────────────────────────────────────────
def download_data():
    tickers = sorted(set(t for pair in PAIRS for t in pair))
    tickers.append('SPY')  # for regime classification
    print(f"Downloading {len(tickers)} tickers: {tickers}")
    data = yf.download(tickers, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
    # Handle multi-level columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Close']
    else:
        prices = data
    prices = prices.dropna(how='all')
    prices = prices.ffill()
    print(f"Downloaded {len(prices)} trading days from {prices.index[0].date()} to {prices.index[-1].date()}")
    return prices

# ── Compute spread z-score ───────────────────────────────────────────────────
def compute_zscore(prices, t1, t2, lookback=60):
    """Compute z-score of log price ratio spread."""
    log_ratio = np.log(prices[t1] / prices[t2])
    mean = log_ratio.rolling(lookback).mean()
    std = log_ratio.rolling(lookback).std()
    zscore = (log_ratio - mean) / std
    return zscore, log_ratio

# ── Rolling cointegration p-value ────────────────────────────────────────────
def rolling_coint_pvalue(prices, t1, t2, window=252):
    """Compute rolling Engle-Granger cointegration test p-value."""
    p1 = prices[t1].values
    p2 = prices[t2].values
    n = len(p1)
    pvals = pd.Series(np.nan, index=prices.index)
    for i in range(window, n):
        s1 = p1[i-window:i]
        s2 = p2[i-window:i]
        if np.any(np.isnan(s1)) or np.any(np.isnan(s2)):
            continue
        try:
            _, pval, _ = coint(s1, s2)
            pvals.iloc[i] = pval
        except Exception:
            continue
    return pvals

# ── Single pair backtest engine ──────────────────────────────────────────────
def backtest_pair(prices, t1, t2, entry_z=2.0, exit_z=0.0, stop_z=3.5,
                  max_hold=20, short_only=False, coint_filter=False,
                  coint_pvals=None, random_entry=False, rng=None,
                  avg_hold_days=None):
    """
    Backtest a single pair.
    Returns list of trade dicts.
    """
    zscore, log_ratio = compute_zscore(prices, t1, t2, LOOKBACK)
    trades = []
    in_trade = False
    entry_date = None
    direction = None  # 1 = long t1/short t2 (z was negative), -1 = short t1/long t2 (z was positive)
    entry_prices = None
    hold_days = 0

    valid_mask = ~zscore.isna()
    dates = prices.index[valid_mask]

    if random_entry and rng is not None:
        # Generate random entry dates with roughly same frequency
        # First run the normal strategy to count trades
        normal_trades = backtest_pair(prices, t1, t2, entry_z=entry_z, exit_z=exit_z,
                                      stop_z=stop_z, max_hold=max_hold)
        n_trades = len(normal_trades)
        if n_trades == 0:
            return []
        # Calculate average hold from normal trades
        if avg_hold_days is None:
            avg_hold_days = int(np.mean([t['hold_days'] for t in normal_trades]))
        avg_hold_days = max(avg_hold_days, 1)

        # Pick random entry points
        available_dates = dates.tolist()
        if len(available_dates) < 2:
            return []
        random_entries = sorted(rng.choice(len(available_dates), size=min(n_trades, len(available_dates)//2), replace=False))

        for idx in random_entries:
            entry_idx = available_dates[idx]
            exit_day = min(idx + rng.integers(1, 2 * avg_hold_days + 1), len(available_dates) - 1)
            exit_idx = available_dates[exit_day]

            if t1 not in prices.columns or t2 not in prices.columns:
                continue

            ep1 = prices.loc[entry_idx, t1]
            ep2 = prices.loc[entry_idx, t2]
            xp1 = prices.loc[exit_idx, t1]
            xp2 = prices.loc[exit_idx, t2]

            if np.isnan(ep1) or np.isnan(ep2) or np.isnan(xp1) or np.isnan(xp2):
                continue

            # Random direction
            d = rng.choice([-1, 1])
            # d=1: long t1, short t2. d=-1: short t1, long t2
            ret1 = (xp1 / ep1 - 1) * d
            ret2 = (xp2 / ep2 - 1) * (-d)

            slippage_cost = 2 * SLIPPAGE_PCT  # entry + exit, both legs
            pnl1 = TRADE_SIZE * (ret1 - slippage_cost)
            pnl2 = TRADE_SIZE * (ret2 - slippage_cost)

            trades.append({
                'pair': f"{t1}/{t2}",
                'entry_date': str(entry_idx.date()),
                'exit_date': str(exit_idx.date()),
                'direction': 'long_t1' if d == 1 else 'short_t1',
                'entry_z': float(zscore.loc[entry_idx]) if entry_idx in zscore.index and not np.isnan(zscore.loc[entry_idx]) else 0.0,
                'exit_z': float(zscore.loc[exit_idx]) if exit_idx in zscore.index and not np.isnan(zscore.loc[exit_idx]) else 0.0,
                'pnl': float(pnl1 + pnl2),
                'hold_days': int(exit_day - idx),
                'exit_reason': 'random',
            })
        return trades

    for i, date in enumerate(dates):
        z = zscore.loc[date]

        if not in_trade:
            # Check cointegration filter
            if coint_filter and coint_pvals is not None:
                pv = coint_pvals.loc[date] if date in coint_pvals.index else np.nan
                if np.isnan(pv) or pv >= 0.05:
                    continue

            # Entry conditions
            if short_only:
                if z > entry_z:
                    direction = -1  # short t1, long t2
                    in_trade = True
                    entry_date = date
                    entry_prices = (prices.loc[date, t1], prices.loc[date, t2])
                    hold_days = 0
            else:
                if z > entry_z:
                    direction = -1  # short t1 (overperformer), long t2
                    in_trade = True
                    entry_date = date
                    entry_prices = (prices.loc[date, t1], prices.loc[date, t2])
                    hold_days = 0
                elif z < -entry_z:
                    direction = 1   # long t1 (underperformer), short t2
                    in_trade = True
                    entry_date = date
                    entry_prices = (prices.loc[date, t1], prices.loc[date, t2])
                    hold_days = 0
        else:
            hold_days += 1
            exit_reason = None

            # Exit conditions
            if direction == -1 and z <= exit_z:
                exit_reason = 'mean_reversion'
            elif direction == 1 and z >= -exit_z:
                exit_reason = 'mean_reversion'
            elif abs(z) > stop_z:
                exit_reason = 'stop_loss'
            elif hold_days >= max_hold:
                exit_reason = 'max_hold'

            if exit_reason:
                exit_prices = (prices.loc[date, t1], prices.loc[date, t2])

                # Calculate P&L
                ret1 = (exit_prices[0] / entry_prices[0] - 1) * direction  # direction applied to t1
                ret2 = (exit_prices[1] / entry_prices[1] - 1) * (-direction)  # opposite for t2

                slippage_cost = 2 * SLIPPAGE_PCT  # entry + exit
                pnl1 = TRADE_SIZE * (ret1 - slippage_cost)
                pnl2 = TRADE_SIZE * (ret2 - slippage_cost)

                trades.append({
                    'pair': f"{t1}/{t2}",
                    'entry_date': str(entry_date.date()),
                    'exit_date': str(date.date()),
                    'direction': 'long_t1' if direction == 1 else 'short_t1',
                    'entry_z': float(zscore.loc[entry_date]),
                    'exit_z': float(z),
                    'pnl': float(pnl1 + pnl2),
                    'hold_days': hold_days,
                    'exit_reason': exit_reason,
                })

                in_trade = False
                direction = None
                entry_date = None
                entry_prices = None

    return trades

# ── Portfolio metrics ────────────────────────────────────────────────────────
def compute_metrics(trades, capital=CAPITAL):
    """Compute performance metrics from trade list."""
    if len(trades) == 0:
        return {
            'n_trades': 0, 'total_pnl': 0, 'sharpe': 0, 'sortino': 0,
            'profit_factor': 0, 'win_rate': 0, 'max_drawdown_pct': 0,
            'avg_hold_days': 0, 'avg_pnl_per_trade': 0,
        }

    pnls = np.array([t['pnl'] for t in trades])
    total_pnl = float(np.sum(pnls))
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    win_rate = float(len(wins) / len(pnls)) if len(pnls) > 0 else 0
    profit_factor = float(np.sum(wins) / abs(np.sum(losses))) if len(losses) > 0 and np.sum(losses) != 0 else float('inf') if len(wins) > 0 else 0

    # Equity curve for Sharpe/Sortino/DD
    equity = capital + np.cumsum(pnls)
    returns = pnls / capital

    # Annualize: assume ~250 trading days, estimate trades per year
    dates = sorted(set(t['entry_date'] for t in trades))
    if len(dates) > 1:
        first = pd.Timestamp(dates[0])
        last = pd.Timestamp(dates[-1])
        years = max((last - first).days / 365.25, 0.1)
        trades_per_year = len(trades) / years
    else:
        trades_per_year = len(trades)
        years = 1.0

    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1) if len(returns) > 1 else 1e-9

    # Sharpe: annualized
    sharpe = float((mean_ret / std_ret) * np.sqrt(trades_per_year)) if std_ret > 1e-9 else 0

    # Sortino
    downside = returns[returns < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-9
    sortino = float((mean_ret / downside_std) * np.sqrt(trades_per_year)) if downside_std > 1e-9 else 0

    # Max drawdown
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = float(np.min(dd)) * 100  # as percentage

    avg_hold = float(np.mean([t['hold_days'] for t in trades]))

    return {
        'n_trades': int(len(trades)),
        'total_pnl': round(total_pnl, 2),
        'total_return_pct': round(total_pnl / capital * 100, 2),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'profit_factor': round(profit_factor, 3) if profit_factor != float('inf') else 999.0,
        'win_rate': round(win_rate, 4),
        'max_drawdown_pct': round(max_dd, 2),
        'avg_hold_days': round(avg_hold, 1),
        'avg_pnl_per_trade': round(float(np.mean(pnls)), 4),
        'n_wins': int(len(wins)),
        'n_losses': int(len(losses)),
    }

# ── Regime classification ────────────────────────────────────────────────────
def classify_regime(prices):
    """Bull = SPY > 200-SMA, Bear = SPY < 200-SMA."""
    spy = prices['SPY']
    sma200 = spy.rolling(200).mean()
    regime = pd.Series('bull', index=prices.index)
    regime[spy < sma200] = 'bear'
    return regime

def regime_gap(trades, regime_series):
    """Compute regime gap: |Sharpe_bull - Sharpe_bear| / max(|Sharpe_bull|, |Sharpe_bear|)."""
    bull_trades = [t for t in trades if regime_series.get(pd.Timestamp(t['entry_date']), 'bull') == 'bull']
    bear_trades = [t for t in trades if regime_series.get(pd.Timestamp(t['entry_date']), 'bull') == 'bear']

    bull_metrics = compute_metrics(bull_trades)
    bear_metrics = compute_metrics(bear_trades)

    s_bull = bull_metrics['sharpe']
    s_bear = bear_metrics['sharpe']

    denom = max(abs(s_bull), abs(s_bear))
    gap = abs(s_bull - s_bear) / denom if denom > 1e-9 else 0

    return {
        'regime_gap': round(float(gap), 4),
        'bull_sharpe': s_bull,
        'bear_sharpe': s_bear,
        'bull_trades': bull_metrics['n_trades'],
        'bear_trades': bear_metrics['n_trades'],
    }

# ── Permutation test ────────────────────────────────────────────────────────
def permutation_test(trades, n_perms=1000, seed=42):
    """Shuffle entry dates, recompute Sharpe. Return p-value."""
    if len(trades) < 5:
        return 1.0

    actual_sharpe = compute_metrics(trades)['sharpe']
    pnls = np.array([t['pnl'] for t in trades])

    rng = np.random.default_rng(seed)
    count_better = 0

    for _ in range(n_perms):
        shuffled = rng.permutation(pnls)
        returns = shuffled / CAPITAL
        mean_r = np.mean(returns)
        std_r = np.std(returns, ddof=1) if len(returns) > 1 else 1e-9

        dates = sorted(set(t['entry_date'] for t in trades))
        if len(dates) > 1:
            first = pd.Timestamp(dates[0])
            last = pd.Timestamp(dates[-1])
            years = max((last - first).days / 365.25, 0.1)
            tpy = len(trades) / years
        else:
            tpy = len(trades)

        perm_sharpe = (mean_r / std_r) * np.sqrt(tpy) if std_r > 1e-9 else 0
        if perm_sharpe >= actual_sharpe:
            count_better += 1

    return round(float(count_better / n_perms), 4)

# ── Validation gates ─────────────────────────────────────────────────────────
def validate(metrics, perm_p, regime_info):
    """5-gate validation."""
    gates = {
        'sharpe_gt_0.5': metrics['sharpe'] > 0.5,
        'permutation_p_lt_0.05': perm_p < 0.05,
        'regime_gap_lt_0.5': regime_info['regime_gap'] < 0.5,
        'max_dd_gt_neg50': metrics['max_drawdown_pct'] > -50,
        'min_20_trades': metrics['n_trades'] >= 20,
    }
    gates['pass_count'] = int(sum(gates.values()))
    gates['all_pass'] = all(v for k, v in gates.items() if k not in ('pass_count', 'all_pass'))
    return gates

# ── Variant runners ──────────────────────────────────────────────────────────
def run_variant_a(prices, regime):
    """Standard Z-Score: entry 2.0, exit 0."""
    print("\n=== Variant A: Standard Z-Score (entry 2.0, exit 0) ===")
    all_trades = []
    for t1, t2 in PAIRS:
        trades = backtest_pair(prices, t1, t2, entry_z=2.0, exit_z=0.0)
        all_trades.extend(trades)
        print(f"  {t1}/{t2}: {len(trades)} trades")
    metrics = compute_metrics(all_trades)
    perm_p = permutation_test(all_trades)
    regime_info = regime_gap(all_trades, regime)
    gates = validate(metrics, perm_p, regime_info)
    print(f"  Total: {metrics['n_trades']} trades, Sharpe={metrics['sharpe']}, PnL=${metrics['total_pnl']}")
    return {'trades': all_trades, 'metrics': metrics, 'perm_p': perm_p, 'regime': regime_info, 'gates': gates}

def run_variant_b(prices, regime):
    """Tight Entry: z=2.5, exit 0.5."""
    print("\n=== Variant B: Tight Entry (z=2.5, exit 0.5) ===")
    all_trades = []
    for t1, t2 in PAIRS:
        trades = backtest_pair(prices, t1, t2, entry_z=2.5, exit_z=0.5)
        all_trades.extend(trades)
        print(f"  {t1}/{t2}: {len(trades)} trades")
    metrics = compute_metrics(all_trades)
    perm_p = permutation_test(all_trades)
    regime_info = regime_gap(all_trades, regime)
    gates = validate(metrics, perm_p, regime_info)
    print(f"  Total: {metrics['n_trades']} trades, Sharpe={metrics['sharpe']}, PnL=${metrics['total_pnl']}")
    return {'trades': all_trades, 'metrics': metrics, 'perm_p': perm_p, 'regime': regime_info, 'gates': gates}

def run_variant_c(prices, regime):
    """Best 3 Pairs Only: Cointegration-filtered (p < 0.05)."""
    print("\n=== Variant C: Best 3 Pairs Only (coint p < 0.05) ===")
    # Precompute cointegration p-values for all pairs
    all_trades = []
    for t1, t2 in PAIRS:
        print(f"  Computing cointegration for {t1}/{t2}...")
        pvals = rolling_coint_pvalue(prices, t1, t2, window=COINT_LOOKBACK)
        trades = backtest_pair(prices, t1, t2, entry_z=2.0, exit_z=0.0,
                               coint_filter=True, coint_pvals=pvals)
        all_trades.extend(trades)
        print(f"  {t1}/{t2}: {len(trades)} trades (coint-filtered)")
    metrics = compute_metrics(all_trades)
    perm_p = permutation_test(all_trades)
    regime_info = regime_gap(all_trades, regime)
    gates = validate(metrics, perm_p, regime_info)
    print(f"  Total: {metrics['n_trades']} trades, Sharpe={metrics['sharpe']}, PnL=${metrics['total_pnl']}")
    return {'trades': all_trades, 'metrics': metrics, 'perm_p': perm_p, 'regime': regime_info, 'gates': gates}

def run_variant_d(prices, regime):
    """Asymmetric: Short-the-overperformer only (z > 2.0)."""
    print("\n=== Variant D: Asymmetric Short-Biased (z > 2.0 only) ===")
    all_trades = []
    for t1, t2 in PAIRS:
        trades = backtest_pair(prices, t1, t2, entry_z=2.0, exit_z=0.0, short_only=True)
        all_trades.extend(trades)
        print(f"  {t1}/{t2}: {len(trades)} trades")
    metrics = compute_metrics(all_trades)
    perm_p = permutation_test(all_trades)
    regime_info = regime_gap(all_trades, regime)
    gates = validate(metrics, perm_p, regime_info)
    print(f"  Total: {metrics['n_trades']} trades, Sharpe={metrics['sharpe']}, PnL=${metrics['total_pnl']}")
    return {'trades': all_trades, 'metrics': metrics, 'perm_p': perm_p, 'regime': regime_info, 'gates': gates}

def run_variant_e(prices, regime):
    """Multi-Pair Portfolio: all pairs, equal weight, portfolio-level metrics."""
    print("\n=== Variant E: Multi-Pair Portfolio ===")
    # Same as A but we compute portfolio-level daily returns
    all_trades = []
    for t1, t2 in PAIRS:
        trades = backtest_pair(prices, t1, t2, entry_z=2.0, exit_z=0.0)
        all_trades.extend(trades)
        print(f"  {t1}/{t2}: {len(trades)} trades")

    # Build daily P&L series for portfolio-level Sharpe
    if len(all_trades) > 0:
        # Group trades by exit date for daily P&L
        daily_pnl = {}
        for t in all_trades:
            d = t['exit_date']
            daily_pnl[d] = daily_pnl.get(d, 0) + t['pnl']

        dates_sorted = sorted(daily_pnl.keys())
        pnl_series = pd.Series({pd.Timestamp(d): daily_pnl[d] for d in dates_sorted})

        # Fill in zero-PnL days
        full_idx = prices.index[(prices.index >= pd.Timestamp(dates_sorted[0])) &
                                (prices.index <= pd.Timestamp(dates_sorted[-1]))]
        pnl_full = pnl_series.reindex(full_idx, fill_value=0.0)

        daily_returns = pnl_full / CAPITAL
        ann_sharpe = float(np.mean(daily_returns) / np.std(daily_returns, ddof=1) * np.sqrt(252)) if np.std(daily_returns) > 1e-9 else 0
        down = daily_returns[daily_returns < 0]
        ann_sortino = float(np.mean(daily_returns) / np.std(down, ddof=1) * np.sqrt(252)) if len(down) > 1 and np.std(down) > 1e-9 else 0

        equity_curve = CAPITAL + np.cumsum(pnl_full.values)
        peak = np.maximum.accumulate(equity_curve)
        dd = (equity_curve - peak) / peak
        max_dd = float(np.min(dd)) * 100
    else:
        ann_sharpe = 0
        ann_sortino = 0
        max_dd = 0

    metrics = compute_metrics(all_trades)
    # Override with portfolio-level metrics
    metrics['portfolio_daily_sharpe'] = round(ann_sharpe, 3)
    metrics['portfolio_daily_sortino'] = round(ann_sortino, 3)
    metrics['portfolio_max_dd_pct'] = round(max_dd, 2)

    perm_p = permutation_test(all_trades)
    regime_info = regime_gap(all_trades, regime)
    gates = validate(metrics, perm_p, regime_info)
    print(f"  Total: {metrics['n_trades']} trades, Sharpe={metrics['sharpe']}, Portfolio Daily Sharpe={ann_sharpe:.3f}, PnL=${metrics['total_pnl']}")
    return {'trades': all_trades, 'metrics': metrics, 'perm_p': perm_p, 'regime': regime_info, 'gates': gates}

def run_variant_f(prices, regime):
    """ADVERSARIAL: Random pair entry, same pairs, same avg hold."""
    print("\n=== Variant F: ADVERSARIAL Random Entry ===")
    rng = np.random.default_rng(42)
    all_trades = []
    for t1, t2 in PAIRS:
        trades = backtest_pair(prices, t1, t2, entry_z=2.0, exit_z=0.0,
                               random_entry=True, rng=rng)
        all_trades.extend(trades)
        print(f"  {t1}/{t2}: {len(trades)} random trades")
    metrics = compute_metrics(all_trades)
    perm_p = permutation_test(all_trades)
    regime_info = regime_gap(all_trades, regime)
    gates = validate(metrics, perm_p, regime_info)
    print(f"  Total: {metrics['n_trades']} trades, Sharpe={metrics['sharpe']}, PnL=${metrics['total_pnl']}")
    return {'trades': all_trades, 'metrics': metrics, 'perm_p': perm_p, 'regime': regime_info, 'gates': gates}

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("PAIRS COINTEGRATION BACKTEST")
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Capital: ${CAPITAL}, Trade size: ${TRADE_SIZE}/leg, Slippage: {SLIPPAGE_PCT*100:.2f}%")
    print("=" * 70)

    prices = download_data()
    regime = classify_regime(prices)
    # Convert to dict for lookup
    regime_dict = regime.to_dict()

    # Run cointegration snapshot for reporting
    print("\n--- Static Cointegration Tests (full sample) ---")
    coint_results = {}
    for t1, t2 in PAIRS:
        p1 = prices[t1].dropna()
        p2 = prices[t2].dropna()
        idx = p1.index.intersection(p2.index)
        try:
            _, pval, _ = coint(p1.loc[idx].values, p2.loc[idx].values)
            coint_results[f"{t1}/{t2}"] = round(float(pval), 4)
            status = "COINTEGRATED" if pval < 0.05 else "not cointegrated"
            print(f"  {t1}/{t2}: p={pval:.4f} ({status})")
        except Exception as e:
            coint_results[f"{t1}/{t2}"] = None
            print(f"  {t1}/{t2}: FAILED ({e})")

    # Run all variants
    results = {}
    results['A'] = run_variant_a(prices, regime_dict)
    results['B'] = run_variant_b(prices, regime_dict)
    results['C'] = run_variant_c(prices, regime_dict)
    results['D'] = run_variant_d(prices, regime_dict)
    results['E'] = run_variant_e(prices, regime_dict)
    results['F'] = run_variant_f(prices, regime_dict)

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    header = f"{'Var':<5} {'Trades':>7} {'PnL':>10} {'Sharpe':>8} {'Sortino':>8} {'PF':>8} {'WR':>6} {'MaxDD':>8} {'PermP':>7} {'RGap':>7} {'Gates':>6}"
    print(header)
    print("-" * len(header))

    for var_name in ['A', 'B', 'C', 'D', 'E', 'F']:
        r = results[var_name]
        m = r['metrics']
        print(f"{var_name:<5} {m['n_trades']:>7} {m['total_pnl']:>10.2f} {m['sharpe']:>8.3f} "
              f"{m['sortino']:>8.3f} {m['profit_factor']:>8.3f} {m['win_rate']:>6.2%} "
              f"{m['max_drawdown_pct']:>7.1f}% {r['perm_p']:>7.4f} "
              f"{r['regime']['regime_gap']:>7.4f} {r['gates']['pass_count']:>3}/5")

    print("\nVariant Descriptions:")
    print("  A: Standard z-score (entry 2.0, exit 0)")
    print("  B: Tight entry (z=2.5, exit 0.5)")
    print("  C: Cointegration-filtered (p < 0.05)")
    print("  D: Asymmetric short-biased (z > 2.0 only)")
    print("  E: Multi-pair portfolio (daily Sharpe)")
    print("  F: ADVERSARIAL — random entry (baseline)")

    # Gate results
    print("\nValidation Gates:")
    for var_name in ['A', 'B', 'C', 'D', 'E', 'F']:
        g = results[var_name]['gates']
        status = "PASS" if g['all_pass'] else "FAIL"
        print(f"  {var_name}: {status} ({g['pass_count']}/5) — "
              f"Sharpe>0.5:{g['sharpe_gt_0.5']}, Perm<0.05:{g['permutation_p_lt_0.05']}, "
              f"RGap<0.5:{g['regime_gap_lt_0.5']}, DD>-50%:{g['max_dd_gt_neg50']}, "
              f"≥20trades:{g['min_20_trades']}")

    # ── Save results ─────────────────────────────────────────────────────
    output = {
        'metadata': {
            'strategy': 'Pairs Cointegration (Gatev et al. 2006)',
            'period': f"{START_DATE} to {END_DATE}",
            'capital': CAPITAL,
            'trade_size_per_leg': TRADE_SIZE,
            'slippage_pct': SLIPPAGE_PCT,
            'pairs': [f"{t1}/{t2}" for t1, t2 in PAIRS],
            'run_timestamp': datetime.now().isoformat(),
        },
        'cointegration_pvalues': coint_results,
        'variants': {},
    }

    for var_name in ['A', 'B', 'C', 'D', 'E', 'F']:
        r = results[var_name]
        output['variants'][var_name] = {
            'metrics': r['metrics'],
            'permutation_p': r['perm_p'],
            'regime': r['regime'],
            'gates': r['gates'],
            'trade_count_by_pair': {},
            'sample_trades': r['trades'][:5] if r['trades'] else [],
        }
        # Trade count by pair
        pair_counts = {}
        for t in r['trades']:
            p = t['pair']
            pair_counts[p] = pair_counts.get(p, 0) + 1
        output['variants'][var_name]['trade_count_by_pair'] = pair_counts

    output_path = '/home/jupiter/Lvl3Quant/data/pairs_cointegration_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)

    print(f"\nResults saved to {output_path}")

if __name__ == '__main__':
    main()
