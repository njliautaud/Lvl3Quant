#!/usr/bin/env python3
"""
Sector Pair Mean Reversion Backtest (v2)
=========================================
Walk-forward OOT: Jan 2022 - Jul 2026
5-gate validation: Sharpe>0.5, perm p<0.05, regime gap<0.5, MaxDD>-50%, >=20 trades
Account: $645, $0 ETF commission, 0.02% slippage.

Pairs:
  XLK/XLC  (Tech vs Communications — both growth-oriented)
  XLF/XLV  (Financials vs Healthcare — both value-oriented)
  XLI/XLB  (Industrials vs Materials — both cyclical)
  QQQ/SPY  (Growth vs Broad market)
  XLE/XLI  (Energy vs Industrials — both macro-sensitive)

Variants:
  A) Basic Long-Only: Buy underperformer when z>2.0, hold 10 days, equal weight
  B) Concentrated: Trade only most extreme divergence, one position at a time
  C) Options Enhancement: Buy calls on underperformer, 3% ATM cost, $0.65 commission, 5% bid-ask
  D) Regime-Filtered: Only trade when VIX < 25
  E) Wider Threshold: z>2.5 entry
  F) Faster Mean Reversion: 10-day rolling, hold 5 days
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from scipy import stats
import os
import sys

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
ACCOUNT_SIZE = 645.0
COMMISSION_ETF = 0.0
SLIPPAGE_PCT = 0.0002  # 0.02%
LOOKBACK_HISTORY = "2020-01-01"  # extra history for rolling calcs
OOT_START = "2022-01-01"
OOT_END = "2026-07-30"
PERM_ITERATIONS = 1000
SEED = 42

PAIRS = [
    ("XLK", "XLC", "Tech vs Communications"),
    ("XLF", "XLV", "Financials vs Healthcare"),
    ("XLI", "XLB", "Industrials vs Materials"),
    ("QQQ", "SPY", "Growth vs Broad"),
    ("XLE", "XLI", "Energy vs Industrials"),
]

# ============================================================
# DATA DOWNLOAD
# ============================================================
def download_data():
    """Download all required ticker data."""
    tickers = set()
    for a, b, _ in PAIRS:
        tickers.add(a)
        tickers.add(b)
    tickers.add("SPY")  # for regime
    tickers.add("^VIX")  # for VIX filter

    all_tickers = sorted(tickers)
    print(f"Downloading: {all_tickers}")

    data = {}
    for t in all_tickers:
        try:
            df = yf.download(t, start=LOOKBACK_HISTORY, end=OOT_END, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 100:
                data[t] = df['Close'].copy()
                print(f"  {t}: {len(df)} rows, {df.index[0].date()} to {df.index[-1].date()}")
            else:
                print(f"  WARNING: {t} only {len(df)} rows")
        except Exception as e:
            print(f"  ERROR downloading {t}: {e}")

    prices = pd.DataFrame(data)
    prices = prices.ffill().dropna()
    print(f"\nCombined price matrix: {prices.shape}")
    return prices


# ============================================================
# SIGNAL GENERATION
# ============================================================
def compute_pair_zscore(prices, ticker_a, ticker_b, lookback=20):
    """
    Compute rolling z-score of relative return spread.
    Positive z-score means A is outperforming B (B is underperformer).
    """
    ret_a = prices[ticker_a].pct_change()
    ret_b = prices[ticker_b].pct_change()
    spread = ret_a - ret_b  # relative return spread
    cum_spread = spread.rolling(lookback).sum()  # rolling cumulative relative return
    z_mean = cum_spread.rolling(60).mean()  # 60-day rolling mean of spread
    z_std = cum_spread.rolling(60).std()
    zscore = (cum_spread - z_mean) / z_std.replace(0, np.nan)
    return zscore


# ============================================================
# BACKTEST ENGINE
# ============================================================
class PairReversionBacktest:
    def __init__(self, prices, account_size=ACCOUNT_SIZE):
        self.prices = prices
        self.account_size = account_size
        self.spy = prices['SPY']
        self.spy_sma200 = self.spy.rolling(200).mean()
        self.vix = prices['^VIX'] if '^VIX' in prices.columns else None

    def get_regime(self, date):
        """Bull = SPY > 200-SMA, Bear = SPY < 200-SMA."""
        if date in self.spy.index and date in self.spy_sma200.index:
            if pd.notna(self.spy_sma200[date]):
                return 'bull' if self.spy[date] > self.spy_sma200[date] else 'bear'
        return 'unknown'

    def run_variant_a(self, pair, z_threshold=2.0, lookback=20, hold_days=10):
        """Basic Long-Only: Buy underperformer when z > threshold, hold N days."""
        ticker_a, ticker_b, label = pair
        zscore = compute_pair_zscore(self.prices, ticker_a, ticker_b, lookback)

        oot_mask = zscore.index >= OOT_START
        oot_dates = zscore.index[oot_mask]

        trades = []
        i = 0
        while i < len(oot_dates):
            date = oot_dates[i]
            z = zscore.get(date, np.nan)
            if pd.isna(z):
                i += 1
                continue

            # z > threshold means A outperforming B → buy B (underperformer)
            # z < -threshold means B outperforming A → buy A (underperformer)
            if abs(z) > z_threshold:
                buy_ticker = ticker_b if z > 0 else ticker_a
                entry_price = self.prices[buy_ticker].get(date, np.nan)
                if pd.isna(entry_price):
                    i += 1
                    continue

                # Find exit date
                exit_idx = min(i + hold_days, len(oot_dates) - 1)
                exit_date = oot_dates[exit_idx]
                exit_price = self.prices[buy_ticker].get(exit_date, np.nan)
                if pd.isna(exit_price):
                    i += 1
                    continue

                # Position sizing: equal weight across possible pairs
                position_size = self.account_size / 5.0  # 5 pairs
                shares = int(position_size / entry_price)
                if shares < 1:
                    shares = 1

                entry_cost = entry_price * (1 + SLIPPAGE_PCT)
                exit_cost = exit_price * (1 - SLIPPAGE_PCT)
                pnl = shares * (exit_cost - entry_cost) - COMMISSION_ETF * 2
                ret = pnl / (shares * entry_cost)

                regime = self.get_regime(date)
                trades.append({
                    'pair': f"{ticker_a}/{ticker_b}",
                    'entry_date': str(date.date()),
                    'exit_date': str(exit_date.date()),
                    'ticker': buy_ticker,
                    'direction': 'long',
                    'entry_price': round(entry_price, 2),
                    'exit_price': round(exit_price, 2),
                    'shares': shares,
                    'pnl': round(pnl, 2),
                    'return': round(ret, 4),
                    'zscore': round(z, 2),
                    'regime': regime,
                })
                i = exit_idx + 1  # skip hold period
            else:
                i += 1

        return trades

    def run_variant_b(self):
        """Concentrated: Only trade single most extreme divergence across all pairs."""
        # Compute all zscores
        zscores = {}
        for ticker_a, ticker_b, label in PAIRS:
            zscores[(ticker_a, ticker_b)] = compute_pair_zscore(
                self.prices, ticker_a, ticker_b, lookback=20)

        oot_start = pd.Timestamp(OOT_START)
        all_dates = self.prices.index[self.prices.index >= oot_start]

        trades = []
        i = 0
        while i < len(all_dates):
            date = all_dates[i]
            # Find most extreme z-score across all pairs
            best_z = 0
            best_pair = None
            for (ta, tb), zs in zscores.items():
                z = zs.get(date, np.nan)
                if pd.notna(z) and abs(z) > abs(best_z):
                    best_z = z
                    best_pair = (ta, tb)

            if best_pair and abs(best_z) > 2.0:
                ta, tb = best_pair
                buy_ticker = tb if best_z > 0 else ta
                entry_price = self.prices[buy_ticker].get(date, np.nan)
                if pd.isna(entry_price):
                    i += 1
                    continue

                exit_idx = min(i + 10, len(all_dates) - 1)
                exit_date = all_dates[exit_idx]
                exit_price = self.prices[buy_ticker].get(exit_date, np.nan)
                if pd.isna(exit_price):
                    i += 1
                    continue

                # Full account on single position
                shares = int(self.account_size / entry_price)
                if shares < 1:
                    shares = 1

                entry_cost = entry_price * (1 + SLIPPAGE_PCT)
                exit_cost = exit_price * (1 - SLIPPAGE_PCT)
                pnl = shares * (exit_cost - entry_cost)
                ret = pnl / (shares * entry_cost)

                regime = self.get_regime(date)
                trades.append({
                    'pair': f"{ta}/{tb}",
                    'entry_date': str(date.date()),
                    'exit_date': str(exit_date.date()),
                    'ticker': buy_ticker,
                    'direction': 'long',
                    'entry_price': round(entry_price, 2),
                    'exit_price': round(exit_price, 2),
                    'shares': shares,
                    'pnl': round(pnl, 2),
                    'return': round(ret, 4),
                    'zscore': round(best_z, 2),
                    'regime': regime,
                })
                i = exit_idx + 1
            else:
                i += 1

        return trades

    def run_variant_c(self):
        """Options Enhancement: Buy calls on underperformer. 3% ATM cost, $0.65 commission, 5% bid-ask."""
        trades = []
        for pair in PAIRS:
            ticker_a, ticker_b, label = pair
            zscore = compute_pair_zscore(self.prices, ticker_a, ticker_b, lookback=20)
            oot_dates = zscore.index[zscore.index >= OOT_START]

            i = 0
            while i < len(oot_dates):
                date = oot_dates[i]
                z = zscore.get(date, np.nan)
                if pd.isna(z) or abs(z) <= 2.0:
                    i += 1
                    continue

                buy_ticker = ticker_b if z > 0 else ticker_a
                stock_price = self.prices[buy_ticker].get(date, np.nan)
                if pd.isna(stock_price):
                    i += 1
                    continue

                # Call option modeling: ATM 14-DTE
                call_price = stock_price * 0.03  # 3% of ETF price
                bid_ask_cost = call_price * 0.05  # 5% bid-ask spread
                effective_cost = call_price + bid_ask_cost
                commission = 0.65

                # Position sizing: allocate per-pair budget
                position_budget = self.account_size / 5.0
                num_contracts = max(1, int(position_budget / (effective_cost * 100 + commission)))
                total_cost = num_contracts * effective_cost * 100 + num_contracts * commission

                # Exit after 10 days
                exit_idx = min(i + 10, len(oot_dates) - 1)
                exit_date = oot_dates[exit_idx]
                exit_price = self.prices[buy_ticker].get(exit_date, np.nan)
                if pd.isna(exit_price):
                    i += 1
                    continue

                # Simplified option P&L: intrinsic value + time decay
                stock_move = exit_price - stock_price
                # Delta ~0.5 for ATM, gamma effect
                days_held = min(10, (exit_date - date).days)
                theta_decay = call_price * (days_held / 14.0) * 0.7  # ~70% theta in last 14 days
                option_exit_value = max(0, call_price + stock_move * 0.5 - theta_decay)
                option_exit_value *= (1 - 0.05)  # exit bid-ask

                pnl = num_contracts * (option_exit_value - effective_cost) * 100 - num_contracts * commission * 2
                ret = pnl / total_cost if total_cost > 0 else 0

                regime = self.get_regime(date)
                trades.append({
                    'pair': f"{ticker_a}/{ticker_b}",
                    'entry_date': str(date.date()),
                    'exit_date': str(exit_date.date()),
                    'ticker': buy_ticker,
                    'direction': 'long_call',
                    'entry_price': round(stock_price, 2),
                    'exit_price': round(exit_price, 2),
                    'contracts': num_contracts,
                    'call_cost': round(effective_cost, 2),
                    'pnl': round(pnl, 2),
                    'return': round(ret, 4),
                    'zscore': round(z, 2),
                    'regime': regime,
                })
                i = exit_idx + 1

        return trades

    def run_variant_d(self):
        """Regime-Filtered: Only trade when VIX < 25."""
        trades = []
        for pair in PAIRS:
            ticker_a, ticker_b, label = pair
            zscore = compute_pair_zscore(self.prices, ticker_a, ticker_b, lookback=20)
            oot_dates = zscore.index[zscore.index >= OOT_START]

            i = 0
            while i < len(oot_dates):
                date = oot_dates[i]
                z = zscore.get(date, np.nan)

                # VIX filter
                vix_val = self.vix.get(date, np.nan) if self.vix is not None else np.nan
                if pd.isna(z) or abs(z) <= 2.0 or pd.isna(vix_val) or vix_val >= 25:
                    i += 1
                    continue

                buy_ticker = ticker_b if z > 0 else ticker_a
                entry_price = self.prices[buy_ticker].get(date, np.nan)
                if pd.isna(entry_price):
                    i += 1
                    continue

                exit_idx = min(i + 10, len(oot_dates) - 1)
                exit_date = oot_dates[exit_idx]
                exit_price = self.prices[buy_ticker].get(exit_date, np.nan)
                if pd.isna(exit_price):
                    i += 1
                    continue

                position_size = self.account_size / 5.0
                shares = max(1, int(position_size / entry_price))

                entry_cost = entry_price * (1 + SLIPPAGE_PCT)
                exit_cost = exit_price * (1 - SLIPPAGE_PCT)
                pnl = shares * (exit_cost - entry_cost)
                ret = pnl / (shares * entry_cost)

                regime = self.get_regime(date)
                trades.append({
                    'pair': f"{ticker_a}/{ticker_b}",
                    'entry_date': str(date.date()),
                    'exit_date': str(exit_date.date()),
                    'ticker': buy_ticker,
                    'direction': 'long',
                    'entry_price': round(entry_price, 2),
                    'exit_price': round(exit_price, 2),
                    'shares': shares,
                    'pnl': round(pnl, 2),
                    'return': round(ret, 4),
                    'zscore': round(z, 2),
                    'vix': round(vix_val, 1),
                    'regime': regime,
                })
                i = exit_idx + 1

        return trades

    def run_variant_e(self):
        """Wider Threshold: z > 2.5 entry."""
        trades = []
        for pair in PAIRS:
            trades.extend(self.run_variant_a(pair, z_threshold=2.5, lookback=20, hold_days=10))
        return trades

    def run_variant_f(self):
        """Faster Mean Reversion: 10-day rolling, hold 5 days."""
        trades = []
        for pair in PAIRS:
            trades.extend(self.run_variant_a(pair, z_threshold=2.0, lookback=10, hold_days=5))
        return trades

    def run_all_variant_a(self):
        """Run variant A across all pairs."""
        trades = []
        for pair in PAIRS:
            trades.extend(self.run_variant_a(pair))
        return trades


# ============================================================
# METRICS & VALIDATION
# ============================================================
def compute_metrics(trades, account_size=ACCOUNT_SIZE):
    """Compute performance metrics from trade list."""
    if not trades:
        return {
            'num_trades': 0, 'total_pnl': 0, 'sharpe': 0, 'sortino': 0,
            'profit_factor': 0, 'win_rate': 0, 'max_dd_pct': 0,
            'avg_return': 0, 'avg_pnl': 0,
        }

    returns = np.array([t['return'] for t in trades])
    pnls = np.array([t['pnl'] for t in trades])

    total_pnl = np.sum(pnls)
    avg_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1) if len(returns) > 1 else 1e-9

    # Sharpe (annualized, assuming ~25 trades/year avg hold ~10 days)
    trades_per_year = 252 / 10  # rough
    sharpe = (avg_ret / std_ret) * np.sqrt(trades_per_year) if std_ret > 1e-9 else 0

    # Sortino
    downside = returns[returns < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-9
    sortino = (avg_ret / downside_std) * np.sqrt(trades_per_year) if downside_std > 1e-9 else 0

    # Profit Factor
    gross_profit = np.sum(pnls[pnls > 0])
    gross_loss = abs(np.sum(pnls[pnls < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Win Rate
    win_rate = np.mean(returns > 0)

    # Max Drawdown
    cum_pnl = np.cumsum(pnls)
    equity = account_size + cum_pnl
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = np.min(dd) if len(dd) > 0 else 0

    return {
        'num_trades': len(trades),
        'total_pnl': round(float(total_pnl), 2),
        'total_return_pct': round(float(total_pnl / account_size * 100), 2),
        'avg_pnl': round(float(np.mean(pnls)), 2),
        'avg_return_pct': round(float(avg_ret * 100), 3),
        'sharpe': round(float(sharpe), 3),
        'sortino': round(float(sortino), 3),
        'profit_factor': round(float(profit_factor), 3),
        'win_rate': round(float(win_rate), 3),
        'max_dd_pct': round(float(max_dd * 100), 2),
    }


def regime_analysis(trades):
    """Split metrics by bull/bear regime."""
    bull = [t for t in trades if t.get('regime') == 'bull']
    bear = [t for t in trades if t.get('regime') == 'bear']
    bull_m = compute_metrics(bull)
    bear_m = compute_metrics(bear)

    # Regime gap: |Sharpe_bull - Sharpe_bear| / max(|Sharpe_bull|, |Sharpe_bear|)
    s_bull = bull_m['sharpe']
    s_bear = bear_m['sharpe']
    denom = max(abs(s_bull), abs(s_bear), 1e-9)
    regime_gap = abs(s_bull - s_bear) / denom

    return {
        'bull': bull_m,
        'bear': bear_m,
        'regime_gap': round(regime_gap, 3),
        'bull_count': len(bull),
        'bear_count': len(bear),
    }


def permutation_test(trades, n_iter=PERM_ITERATIONS):
    """Shuffle entry dates, compute p-value of observed Sharpe."""
    if len(trades) < 5:
        return 1.0

    observed_returns = [t['return'] for t in trades]
    observed_sharpe = np.mean(observed_returns) / (np.std(observed_returns, ddof=1) + 1e-9)

    rng = np.random.RandomState(SEED)
    count_better = 0
    for _ in range(n_iter):
        shuffled = rng.permutation(observed_returns)
        shuf_sharpe = np.mean(shuffled) / (np.std(shuffled, ddof=1) + 1e-9)
        if shuf_sharpe >= observed_sharpe:
            count_better += 1

    return round((count_better + 1) / (n_iter + 1), 4)


def five_gate_validation(metrics, regime, perm_p, label):
    """Apply 5-gate validation."""
    gates = {
        'sharpe_gt_0.5': metrics['sharpe'] > 0.5,
        'perm_p_lt_0.05': perm_p < 0.05,
        'regime_gap_lt_0.5': regime['regime_gap'] < 0.5,
        'max_dd_gt_neg50': metrics['max_dd_pct'] > -50.0,
        'min_20_trades': metrics['num_trades'] >= 20,
    }
    passed = sum(gates.values())
    return {
        'variant': label,
        'gates_passed': f"{passed}/5",
        'all_passed': passed == 5,
        'details': gates,
    }


# ============================================================
# PER-PAIR BREAKDOWN
# ============================================================
def per_pair_breakdown(trades):
    """Compute metrics per pair."""
    pairs = set(t['pair'] for t in trades)
    result = {}
    for p in sorted(pairs):
        pair_trades = [t for t in trades if t['pair'] == p]
        result[p] = compute_metrics(pair_trades)
    return result


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("SECTOR PAIR MEAN REVERSION BACKTEST")
    print(f"OOT: {OOT_START} to {OOT_END} | Account: ${ACCOUNT_SIZE}")
    print("=" * 70)

    prices = download_data()

    bt = PairReversionBacktest(prices)

    variants = {
        'A_basic_long_only': ('Basic Long-Only (z>2.0, hold 10d, equal weight)', bt.run_all_variant_a),
        'B_concentrated': ('Concentrated (most extreme pair only)', bt.run_variant_b),
        'C_options_enhancement': ('Options Enhancement (buy calls)', bt.run_variant_c),
        'D_regime_filtered': ('Regime-Filtered (VIX < 25 only)', bt.run_variant_d),
        'E_wider_threshold': ('Wider Threshold (z>2.5)', bt.run_variant_e),
        'F_faster_reversion': ('Faster Reversion (10d rolling, hold 5d)', bt.run_variant_f),
    }

    results = {}

    for key, (label, func) in variants.items():
        print(f"\n{'='*60}")
        print(f"  Variant {key[0]}: {label}")
        print(f"{'='*60}")

        trades = func()
        metrics = compute_metrics(trades)
        regime = regime_analysis(trades)

        print(f"  Trades: {metrics['num_trades']}")
        print(f"  Total P&L: ${metrics['total_pnl']} ({metrics['total_return_pct']}%)")
        print(f"  Sharpe: {metrics['sharpe']} | Sortino: {metrics['sortino']}")
        print(f"  PF: {metrics['profit_factor']} | WR: {metrics['win_rate']:.1%}")
        print(f"  Max DD: {metrics['max_dd_pct']}%")
        print(f"  Regime: Bull Sharpe={regime['bull']['sharpe']}, Bear Sharpe={regime['bear']['sharpe']}, Gap={regime['regime_gap']}")

        # Permutation test
        print(f"  Running permutation test ({PERM_ITERATIONS} iterations)...")
        perm_p = permutation_test(trades)
        print(f"  Permutation p-value: {perm_p}")

        # 5-gate validation
        validation = five_gate_validation(metrics, regime, perm_p, label)
        gate_str = "PASS" if validation['all_passed'] else "FAIL"
        print(f"  5-Gate: {validation['gates_passed']} — {gate_str}")
        for g, v in validation['details'].items():
            status = "✓" if v else "✗"
            print(f"    {status} {g}")

        # Per-pair breakdown
        pair_breakdown = per_pair_breakdown(trades)
        print(f"\n  Per-Pair Breakdown:")
        for p, pm in pair_breakdown.items():
            print(f"    {p}: {pm['num_trades']} trades, Sharpe={pm['sharpe']}, WR={pm['win_rate']:.1%}, PnL=${pm['total_pnl']}")

        results[key] = {
            'label': label,
            'metrics': metrics,
            'regime': regime,
            'perm_p_value': perm_p,
            'validation': validation,
            'per_pair': pair_breakdown,
            'sample_trades': trades[:5] if trades else [],
            'trade_count_by_pair': {p: sum(1 for t in trades if t['pair'] == p) for p in set(t['pair'] for t in trades)} if trades else {},
        }

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY — 5-GATE VALIDATION")
    print("=" * 70)
    print(f"{'Variant':<45} {'Trades':>6} {'Sharpe':>7} {'PF':>6} {'WR':>6} {'MaxDD':>7} {'Perm-p':>7} {'Gates':>6} {'Result':>7}")
    print("-" * 100)

    passing = []
    for key, r in results.items():
        m = r['metrics']
        v = r['validation']
        status = "PASS" if v['all_passed'] else "FAIL"
        print(f"{r['label']:<45} {m['num_trades']:>6} {m['sharpe']:>7.3f} {m['profit_factor']:>6.2f} {m['win_rate']:>6.1%} {m['max_dd_pct']:>6.1f}% {r['perm_p_value']:>7.4f} {v['gates_passed']:>6} {status:>7}")
        if v['all_passed']:
            passing.append(key)

    print(f"\nPassing variants: {len(passing)}/6")
    if passing:
        print(f"Winners: {', '.join(passing)}")
        # Best by Sharpe among passing
        best_key = max(passing, key=lambda k: results[k]['metrics']['sharpe'])
        best = results[best_key]
        print(f"\nBest: {best['label']}")
        print(f"  Sharpe={best['metrics']['sharpe']}, Sortino={best['metrics']['sortino']}, "
              f"PF={best['metrics']['profit_factor']}, WR={best['metrics']['win_rate']:.1%}, "
              f"P&L=${best['metrics']['total_pnl']}")
    else:
        print("No variants passed all 5 gates.")

    # ============================================================
    # SAVE RESULTS
    # ============================================================
    output = {
        'strategy': 'Sector Pair Mean Reversion',
        'run_date': datetime.now().isoformat(),
        'account_size': ACCOUNT_SIZE,
        'oot_period': f"{OOT_START} to {OOT_END}",
        'pairs': [f"{a}/{b} ({l})" for a, b, l in PAIRS],
        'slippage_pct': SLIPPAGE_PCT,
        'commission': COMMISSION_ETF,
        'permutation_iterations': PERM_ITERATIONS,
        'variants': results,
        'passing_variants': passing,
        'total_variants': 6,
    }

    out_path = "/home/jupiter/Lvl3Quant/data/sector_pair_reversion_results.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
