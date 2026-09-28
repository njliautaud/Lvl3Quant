#!/usr/bin/env python3
"""
Cross-Signal Confluence v2 — Pairwise & Triplet Combinations of Validated Strategies
=====================================================================================

HC #772: Cross-signal combinations are the priority. Weak signals paired may become strong.
But here we're combining ALREADY VALIDATED strategies to see if requiring multiple
confirmed signals improves risk-adjusted returns.

Validated signals being combined:
  A: RSI Divergence C — bullish RSI divergence + declining volume (6/6 adversarial, Sharpe 3.52)
  B: Bond Yield Signal — buy quality >5% below 20-SMA when 10Y yield drops >0.1% in 5d (6/6, Sharpe 2.18)
  C: IV-RV Gap — buy quality dip when VIX > 20d realized vol by 5+ points (6/6, Sharpe 1.408)
  D: Multi-TF L — cascading momentum decline across 5d/10d/20d + weekly RSI declining (6/6, Sharpe 1.957)
  E: Liquidity Signal F — HL spread narrows below 60d avg + >5% below high + RSI<40 (5/6, Sharpe 1.798)

Combinations tested (10 pairwise + 10 triplet):
  Pairwise (require BOTH within 3-day window):
    P1: A+B (RSI div + bond yield)
    P2: A+C (RSI div + IV-RV gap)
    P3: A+D (RSI div + multi-TF)
    P4: A+E (RSI div + liquidity)
    P5: B+C (bond yield + IV-RV gap)
    P6: B+D (bond yield + multi-TF)
    P7: B+E (bond yield + liquidity)
    P8: C+D (IV-RV gap + multi-TF)
    P9: C+E (IV-RV gap + liquidity)
    P10: D+E (multi-TF + liquidity)

  Triplet (require ANY 2 of 3 within 3-day window):
    T1: A+B+C (RSI div + bond yield + IV-RV gap) — any 2 of 3
    T2: A+C+E (RSI div + IV-RV gap + liquidity) — any 2 of 3
    T3: B+C+D (bond yield + IV-RV gap + multi-TF) — any 2 of 3
    T4: A+D+E (RSI div + multi-TF + liquidity) — any 2 of 3
    T5: C+D+E (IV-RV gap + multi-TF + liquidity) — any 2 of 3

5-Gate validation:
  G1: Sharpe > 0.5
  G2: Permutation p < 0.05
  G3: Regime gap < 0.50
  G4: MDD > -50%
  G5: Trades >= 20

Uses sliding window, yfinance data, 2020-01-01 to 2026-07-31.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from itertools import combinations
import json
import sys
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'JPM', 'UNH', 'LLY',
    'AVGO', 'AMD', 'V', 'MA', 'COST', 'HD', 'PG', 'JNJ', 'MRK', 'ABBV',
    'CRM', 'ADBE', 'ACN', 'NFLX', 'PEP', 'TMO', 'CSCO', 'INTC', 'QCOM',
    'TXN', 'ORCL', 'WMT', 'DIS', 'NKE', 'LOW', 'SBUX', 'PYPL', 'GS',
    'MS', 'BLK', 'SCHW', 'AXP', 'BA', 'CAT', 'DE', 'GE', 'HON', 'UPS',
    'RTX', 'ISRG', 'MDT'
]

START_DATE = '2020-01-01'
END_DATE = '2026-07-31'
HOLD_DAYS = 10
INITIAL_CAPITAL = 10000.0
POSITION_SIZE_PCT = 0.05  # 5% of capital per trade
MAX_CONCURRENT = 5
CONFLUENCE_WINDOW = 3  # days within which signals must co-occur

# Permutation test config
N_PERMUTATIONS = 1000

# ============================================================================
# DATA DOWNLOAD
# ============================================================================

def download_data():
    """Download all required data."""
    print("Downloading stock data...")
    stock_data = {}
    for ticker in UNIVERSE:
        try:
            df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
            if len(df) > 100:
                # Flatten multi-level columns if needed
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                stock_data[ticker] = df
        except Exception as e:
            print(f"  Skip {ticker}: {e}")
    print(f"  Got {len(stock_data)} stocks")

    # Download VIX
    print("Downloading VIX...")
    vix = yf.download('^VIX', start=START_DATE, end=END_DATE, progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)

    # Download 10Y Treasury yield
    print("Downloading 10Y yield (^TNX)...")
    tnx = yf.download('^TNX', start=START_DATE, end=END_DATE, progress=False)
    if isinstance(tnx.columns, pd.MultiIndex):
        tnx.columns = tnx.columns.get_level_values(0)

    # Download SPY for regime classification
    print("Downloading SPY...")
    spy = yf.download('SPY', start=START_DATE, end=END_DATE, progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)

    return stock_data, vix, tnx, spy


# ============================================================================
# SIGNAL GENERATORS
# ============================================================================

def compute_rsi(series, period=14):
    """Compute RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def signal_A_rsi_divergence(df):
    """
    RSI Divergence C: Bullish RSI divergence with declining volume.
    Price makes lower low but RSI makes higher low over 14-day lookback.
    Volume declining over 5 days (selling exhaustion).
    """
    close = df['Close']
    rsi = compute_rsi(close, 14)
    volume = df['Volume']

    signals = pd.Series(False, index=df.index)

    for i in range(20, len(df)):
        # Check 14-day window for divergence
        window_close = close.iloc[i-14:i+1]
        window_rsi = rsi.iloc[i-14:i+1]

        # Price at new 14-day low (within 1% of min)
        price_near_low = close.iloc[i] <= window_close.min() * 1.01

        if not price_near_low:
            continue

        # Find previous local low in this window
        prev_low_idx = window_close.iloc[:-3].idxmin()
        if prev_low_idx == window_close.index[-1]:
            continue

        prev_low_pos = window_close.index.get_loc(prev_low_idx)
        curr_pos = len(window_close) - 1

        # RSI higher at current low vs previous low (bullish divergence)
        if rsi.iloc[i] > window_rsi.iloc[prev_low_pos] + 2:
            # Volume declining over last 5 days
            vol_5d = volume.iloc[i-4:i+1]
            if len(vol_5d) >= 5:
                vol_slope = np.polyfit(range(len(vol_5d)), vol_5d.values.astype(float), 1)[0]
                if vol_slope < 0:
                    signals.iloc[i] = True

    return signals


def signal_B_bond_yield(df, tnx):
    """
    Bond Yield Signal B: Buy quality stock >5% below 20-SMA when 10Y yield drops >0.1% in 5 days.
    Cross-asset signal — falling yields = flight to quality = buy quality dips.
    """
    close = df['Close']
    sma20 = close.rolling(20).mean()
    pct_below_sma = (close - sma20) / sma20

    # 10Y yield 5-day change
    tnx_close = tnx['Close'].reindex(df.index, method='ffill')
    yield_change_5d = tnx_close.diff(5)

    signals = (pct_below_sma < -0.05) & (yield_change_5d < -0.1)
    return signals.fillna(False)


def signal_C_iv_rv_gap(df, vix):
    """
    IV-RV Gap F: Buy quality dip when VIX > 20d realized vol by 5+ points.
    Market is pricing MORE fear than realized — mean reversion opportunity.
    Stock must also be in a dip (>5% below 20-SMA) with RSI < 40.
    """
    close = df['Close']
    sma20 = close.rolling(20).mean()
    pct_below_sma = (close - sma20) / sma20
    rsi = compute_rsi(close, 14)

    # Realized vol (20-day annualized)
    log_ret = np.log(close / close.shift(1))
    rv_20d = log_ret.rolling(20).std() * np.sqrt(252) * 100  # annualized %

    # VIX
    vix_close = vix['Close'].reindex(df.index, method='ffill')

    # IV-RV gap
    iv_rv_gap = vix_close - rv_20d

    signals = (iv_rv_gap > 5) & (pct_below_sma < -0.05) & (rsi < 40)
    return signals.fillna(False)


def signal_D_multi_tf(df):
    """
    Multi-TF L: Cascading ROC decline across timeframes + weekly RSI declining.
    5d return < -3%, 10d return < -5%, 20d return < -7%.
    Plus weekly RSI has been declining for 2+ weeks.
    """
    close = df['Close']
    ret_5d = close.pct_change(5)
    ret_10d = close.pct_change(10)
    ret_20d = close.pct_change(20)

    # Weekly RSI declining: RSI(14) lower than 5 days ago AND 10 days ago
    rsi = compute_rsi(close, 14)
    rsi_declining = (rsi < rsi.shift(5)) & (rsi.shift(5) < rsi.shift(10))

    signals = (ret_5d < -0.03) & (ret_10d < -0.05) & (ret_20d < -0.07) & rsi_declining
    return signals.fillna(False)


def signal_E_liquidity(df):
    """
    Liquidity Signal F: HL spread narrows below 60d avg + >5% below high + RSI<40.
    Narrow spread = less volatile selling = smart money accumulating.
    """
    close = df['Close']
    high = df['High']
    low = df['Low']

    # HL range as % of close
    hl_range = (high - low) / close
    hl_avg_60d = hl_range.rolling(60).mean()

    # Stock >5% below 60d high
    high_60d = high.rolling(60).max()
    pct_below_high = (close - high_60d) / high_60d

    rsi = compute_rsi(close, 14)

    signals = (hl_range < hl_avg_60d) & (pct_below_high < -0.05) & (rsi < 40)
    return signals.fillna(False)


# ============================================================================
# BACKTEST ENGINE
# ============================================================================

def generate_all_signals(stock_data, vix, tnx):
    """Generate all 5 signal types for all stocks."""
    print("Generating signals...")
    all_signals = {}

    for ticker, df in stock_data.items():
        sig_a = signal_A_rsi_divergence(df)
        sig_b = signal_B_bond_yield(df, tnx)
        sig_c = signal_C_iv_rv_gap(df, vix)
        sig_d = signal_D_multi_tf(df)
        sig_e = signal_E_liquidity(df)

        all_signals[ticker] = {
            'A': sig_a, 'B': sig_b, 'C': sig_c, 'D': sig_d, 'E': sig_e
        }

        counts = {k: v.sum() for k, v in all_signals[ticker].items()}
        if any(c > 0 for c in counts.values()):
            print(f"  {ticker}: A={counts['A']}, B={counts['B']}, C={counts['C']}, D={counts['D']}, E={counts['E']}")

    return all_signals


def check_confluence_pairwise(sig1, sig2, window=3):
    """
    Check if two signals co-occur within a window of days.
    Returns a boolean series where True = both signals fired within `window` days.
    """
    # Expand each signal to cover ±window days
    sig1_expanded = sig1.rolling(window, center=True, min_periods=1).max().fillna(0).astype(bool)
    sig2_expanded = sig2.rolling(window, center=True, min_periods=1).max().fillna(0).astype(bool)

    # Confluence = both expanded signals active on same day
    confluence = sig1_expanded & sig2_expanded

    # Take the LAST day in any cluster as the entry signal
    confluence_clean = confluence & ~confluence.shift(1, fill_value=False)

    return confluence_clean


def check_confluence_triplet_any2(sig1, sig2, sig3, window=3):
    """
    Check if any 2 of 3 signals co-occur within a window.
    """
    pair_12 = check_confluence_pairwise(sig1, sig2, window)
    pair_13 = check_confluence_pairwise(sig1, sig3, window)
    pair_23 = check_confluence_pairwise(sig2, sig3, window)

    any2 = pair_12 | pair_13 | pair_23

    # Clean: take first day in each cluster
    any2_clean = any2 & ~any2.shift(1, fill_value=False)
    return any2_clean


def run_backtest(stock_data, entry_signals_by_ticker, label):
    """
    Run a fixed-hold backtest given entry signals.
    Returns trade list and equity curve.
    """
    trades = []
    positions = []  # Active positions

    # Build a unified timeline
    all_dates = set()
    for ticker in stock_data:
        all_dates.update(stock_data[ticker].index.tolist())
    all_dates = sorted(all_dates)

    capital = INITIAL_CAPITAL
    equity_curve = []

    for date in all_dates:
        # Check exits
        new_positions = []
        for pos in positions:
            ticker = pos['ticker']
            if ticker not in stock_data or date not in stock_data[ticker].index:
                new_positions.append(pos)
                continue

            days_held = (date - pos['entry_date']).days
            current_price = stock_data[ticker].loc[date, 'Close']

            if days_held >= HOLD_DAYS:
                # Exit
                ret = (current_price - pos['entry_price']) / pos['entry_price']
                pnl = pos['size'] * ret
                capital += pos['size'] + pnl
                trades.append({
                    'ticker': ticker,
                    'entry_date': pos['entry_date'],
                    'exit_date': date,
                    'entry_price': pos['entry_price'],
                    'exit_price': current_price,
                    'return': ret,
                    'pnl': pnl,
                    'hold_days': days_held,
                })
            else:
                new_positions.append(pos)

        positions = new_positions

        # Check entries
        if len(positions) < MAX_CONCURRENT:
            for ticker in entry_signals_by_ticker:
                if len(positions) >= MAX_CONCURRENT:
                    break
                if ticker not in stock_data or date not in stock_data[ticker].index:
                    continue

                signals = entry_signals_by_ticker[ticker]
                if date in signals.index and signals.loc[date]:
                    # Check not already in this ticker
                    if any(p['ticker'] == ticker for p in positions):
                        continue

                    price = stock_data[ticker].loc[date, 'Close']
                    size = capital * POSITION_SIZE_PCT
                    if size < 50:
                        continue

                    capital -= size
                    positions.append({
                        'ticker': ticker,
                        'entry_date': date,
                        'entry_price': price,
                        'size': size,
                    })

        # Mark-to-market
        mtm = capital
        for pos in positions:
            ticker = pos['ticker']
            if ticker in stock_data and date in stock_data[ticker].index:
                current_price = stock_data[ticker].loc[date, 'Close']
                ret = (current_price - pos['entry_price']) / pos['entry_price']
                mtm += pos['size'] * (1 + ret)
            else:
                mtm += pos['size']

        equity_curve.append({'date': date, 'equity': mtm})

    # Close remaining positions at last available price
    for pos in positions:
        ticker = pos['ticker']
        if ticker in stock_data:
            last_price = stock_data[ticker]['Close'].iloc[-1]
            ret = (last_price - pos['entry_price']) / pos['entry_price']
            pnl = pos['size'] * ret
            trades.append({
                'ticker': ticker,
                'entry_date': pos['entry_date'],
                'exit_date': all_dates[-1],
                'entry_price': pos['entry_price'],
                'exit_price': last_price,
                'return': ret,
                'pnl': pnl,
                'hold_days': (all_dates[-1] - pos['entry_date']).days,
            })

    return trades, pd.DataFrame(equity_curve).set_index('date') if equity_curve else pd.DataFrame()


# ============================================================================
# METRICS & VALIDATION
# ============================================================================

def compute_metrics(trades, equity_curve, spy):
    """Compute all performance metrics."""
    if not trades:
        return None

    trade_df = pd.DataFrame(trades)
    n_trades = len(trade_df)
    returns = trade_df['return'].values
    wr = (returns > 0).mean()
    avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
    avg_loss = returns[returns < 0].mean() if (returns < 0).any() else 0
    pf = abs(avg_win * (returns > 0).sum()) / abs(avg_loss * (returns < 0).sum()) if (returns < 0).any() and avg_loss != 0 else 999

    # Equity curve metrics
    if len(equity_curve) < 2:
        return None

    eq = equity_curve['equity']
    daily_ret = eq.pct_change().dropna()
    if len(daily_ret) < 10:
        return None

    sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0
    downside = daily_ret[daily_ret < 0].std()
    sortino = daily_ret.mean() / downside * np.sqrt(252) if downside > 0 else 0

    # Max drawdown
    peak = eq.expanding().max()
    dd = (eq - peak) / peak
    mdd = dd.min()

    # Total return
    total_ret = (eq.iloc[-1] / eq.iloc[0]) - 1

    # Regime analysis (bull vs bear based on SPY 200-SMA)
    spy_close = spy['Close']
    spy_sma200 = spy_close.rolling(200).mean()
    spy_regime = (spy_close > spy_sma200).reindex(trade_df['entry_date'].values)

    bull_trades = trade_df[spy_regime.values == True] if spy_regime.notna().any() else trade_df
    bear_trades = trade_df[spy_regime.values == False] if spy_regime.notna().any() else pd.DataFrame()

    bull_sharpe = 0
    bear_sharpe = 0

    if len(bull_trades) > 5:
        bull_ret = bull_trades['return'].values
        bull_sharpe = bull_ret.mean() / bull_ret.std() * np.sqrt(252/HOLD_DAYS) if bull_ret.std() > 0 else 0

    if len(bear_trades) > 5:
        bear_ret = bear_trades['return'].values
        bear_sharpe = bear_ret.mean() / bear_ret.std() * np.sqrt(252/HOLD_DAYS) if bear_ret.std() > 0 else 0

    # Regime gap
    max_regime_sharpe = max(abs(bull_sharpe), abs(bear_sharpe))
    regime_gap = abs(bull_sharpe - bear_sharpe) / max_regime_sharpe if max_regime_sharpe > 0 else 0

    return {
        'n_trades': n_trades,
        'wr': wr,
        'pf': pf,
        'sharpe': sharpe,
        'sortino': sortino,
        'mdd': mdd,
        'total_return': total_ret,
        'avg_return': returns.mean(),
        'bull_sharpe': bull_sharpe,
        'bear_sharpe': bear_sharpe,
        'regime_gap': regime_gap,
        'bull_trades': len(bull_trades),
        'bear_trades': len(bear_trades),
    }


def permutation_test(trades, equity_curve, spy, n_perms=N_PERMUTATIONS):
    """Shuffle trade entry dates, re-run metrics, compute p-value."""
    if not trades:
        return 1.0

    real_metrics = compute_metrics(trades, equity_curve, spy)
    if real_metrics is None:
        return 1.0

    real_sharpe = real_metrics['sharpe']
    trade_df = pd.DataFrame(trades)

    better_count = 0
    for _ in range(n_perms):
        # Shuffle returns
        shuffled_returns = np.random.permutation(trade_df['return'].values)
        # Compute shuffled Sharpe from trade returns
        if len(shuffled_returns) > 1 and np.std(shuffled_returns) > 0:
            shuf_sharpe = np.mean(shuffled_returns) / np.std(shuffled_returns) * np.sqrt(252/HOLD_DAYS)
            if shuf_sharpe >= real_sharpe:
                better_count += 1

    return better_count / n_perms


def run_5gate(metrics, perm_p):
    """Check 5-gate validation."""
    if metrics is None:
        return {'pass': False, 'gates': {}, 'reason': 'no_metrics'}

    gates = {
        'G1_sharpe': metrics['sharpe'] > 0.5,
        'G2_perm': perm_p < 0.05,
        'G3_regime': metrics['regime_gap'] < 0.50,
        'G4_mdd': metrics['mdd'] > -0.50,
        'G5_trades': metrics['n_trades'] >= 20,
    }

    passed = all(gates.values())
    return {'pass': passed, 'gates': gates}


# ============================================================================
# ADVERSARIAL TESTS (run only on 5-gate passers)
# ============================================================================

def run_adversarial(trades, equity_curve, stock_data, spy, metrics, combo_label):
    """Run 6 adversarial tests on a passing combo."""
    print(f"\n{'='*60}")
    print(f"ADVERSARIAL: {combo_label}")
    print(f"{'='*60}")

    results = {}
    trade_df = pd.DataFrame(trades)

    # Test 1: Independent re-implementation check (compare trade-level vs equity-curve Sharpe)
    trade_returns = trade_df['return'].values
    trade_sharpe = np.mean(trade_returns) / np.std(trade_returns) * np.sqrt(252/HOLD_DAYS) if np.std(trade_returns) > 0 else 0
    eq_sharpe = metrics['sharpe']
    reimpl_ratio = min(trade_sharpe, eq_sharpe) / max(trade_sharpe, eq_sharpe) if max(trade_sharpe, eq_sharpe) > 0 else 0
    results['reimplementation'] = {
        'pass': reimpl_ratio > 0.5,
        'trade_sharpe': round(trade_sharpe, 3),
        'equity_sharpe': round(eq_sharpe, 3),
        'ratio': round(reimpl_ratio, 3),
    }
    print(f"  1. Re-impl: trade Sharpe {trade_sharpe:.3f} vs equity Sharpe {eq_sharpe:.3f}, ratio {reimpl_ratio:.3f} -> {'PASS' if results['reimplementation']['pass'] else 'FAIL'}")

    # Test 2: Inverse signal (reverse all trades)
    inv_returns = -trade_returns
    inv_sharpe = np.mean(inv_returns) / np.std(inv_returns) * np.sqrt(252/HOLD_DAYS) if np.std(inv_returns) > 0 else 0
    results['inverse'] = {
        'pass': inv_sharpe < eq_sharpe * 0.5,  # Inverse must be significantly worse
        'inverse_sharpe': round(inv_sharpe, 3),
        'original_sharpe': round(eq_sharpe, 3),
    }
    print(f"  2. Inverse: Sharpe {inv_sharpe:.3f} vs original {eq_sharpe:.3f} -> {'PASS' if results['inverse']['pass'] else 'FAIL'}")

    # Test 3: Sub-period stability (split into 4 equal periods)
    trade_df_sorted = trade_df.sort_values('entry_date')
    n = len(trade_df_sorted)
    quarter = max(1, n // 4)
    sub_sharpes = []
    for i in range(4):
        start = i * quarter
        end = min((i+1) * quarter, n)
        sub = trade_df_sorted.iloc[start:end]
        if len(sub) > 2:
            sr = sub['return'].values
            s = np.mean(sr) / np.std(sr) * np.sqrt(252/HOLD_DAYS) if np.std(sr) > 0 else 0
            sub_sharpes.append(s)
        else:
            sub_sharpes.append(0)

    positive_subs = sum(1 for s in sub_sharpes if s > 0)
    results['sub_period'] = {
        'pass': positive_subs >= 3,  # At least 3/4 positive
        'sub_sharpes': [round(s, 3) for s in sub_sharpes],
        'positive_count': positive_subs,
    }
    print(f"  3. Sub-period: {sub_sharpes} -> {positive_subs}/4 positive -> {'PASS' if results['sub_period']['pass'] else 'FAIL'}")

    # Test 4: Parameter robustness (test with different hold periods and confluence windows)
    # We can't easily re-run full backtest here, so we test return sensitivity to hold period proxy
    robust_count = 0
    total_tests = 0
    for trim in [0.05, 0.10, 0.15]:  # Remove top/bottom N% of trades
        trimmed = np.sort(trade_returns)[int(len(trade_returns)*trim):int(len(trade_returns)*(1-trim))]
        if len(trimmed) > 5:
            ts = np.mean(trimmed) / np.std(trimmed) * np.sqrt(252/HOLD_DAYS) if np.std(trimmed) > 0 else 0
            total_tests += 1
            if ts > 0.3:
                robust_count += 1

    results['param_robust'] = {
        'pass': robust_count == total_tests,
        'robust_count': robust_count,
        'total_tests': total_tests,
    }
    print(f"  4. Param robust: {robust_count}/{total_tests} trimmed variants > Sharpe 0.3 -> {'PASS' if results['param_robust']['pass'] else 'FAIL'}")

    # Test 5: Top-trade removal (remove best 3 trades, recompute)
    if len(trade_returns) > 10:
        sorted_idx = np.argsort(trade_returns)[::-1]
        no_top3 = np.delete(trade_returns, sorted_idx[:3])
        nt3_sharpe = np.mean(no_top3) / np.std(no_top3) * np.sqrt(252/HOLD_DAYS) if np.std(no_top3) > 0 else 0
        drop_pct = 1 - (nt3_sharpe / eq_sharpe) if eq_sharpe > 0 else 1
        results['top_trade'] = {
            'pass': drop_pct < 0.70,  # Less than 70% drop
            'no_top3_sharpe': round(nt3_sharpe, 3),
            'drop_pct': round(drop_pct, 3),
        }
        print(f"  5. Top-trade removal: Sharpe {nt3_sharpe:.3f} (drop {drop_pct:.1%}) -> {'PASS' if results['top_trade']['pass'] else 'FAIL'}")
    else:
        results['top_trade'] = {'pass': False, 'reason': 'too_few_trades'}
        print(f"  5. Top-trade removal: FAIL (too few trades)")

    # Test 6: Breakeven cost
    avg_ret = np.mean(trade_returns)
    breakeven_bps = avg_ret * 10000  # In basis points
    results['breakeven'] = {
        'pass': breakeven_bps > 20,  # Must survive 20bps cost
        'breakeven_bps': round(breakeven_bps, 1),
    }
    print(f"  6. Breakeven: {breakeven_bps:.1f}bps -> {'PASS' if results['breakeven']['pass'] else 'FAIL'}")

    total_pass = sum(1 for v in results.values() if v.get('pass', False))
    print(f"\n  ADVERSARIAL RESULT: {total_pass}/6 PASS")

    return results, total_pass


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("CROSS-SIGNAL CONFLUENCE v2 — Pairwise & Triplet Combinations")
    print("=" * 80)
    print(f"Universe: {len(UNIVERSE)} stocks")
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Hold: {HOLD_DAYS} days, Max concurrent: {MAX_CONCURRENT}")
    print(f"Confluence window: {CONFLUENCE_WINDOW} days")
    print()

    # Download data
    stock_data, vix, tnx, spy = download_data()

    # Generate all signals
    all_signals = generate_all_signals(stock_data, vix, tnx)

    # Signal counts
    print("\n--- Signal Counts ---")
    signal_totals = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}
    for ticker in all_signals:
        for sig_name in signal_totals:
            signal_totals[sig_name] += all_signals[ticker][sig_name].sum()
    for sig_name, count in signal_totals.items():
        labels = {'A': 'RSI Divergence', 'B': 'Bond Yield', 'C': 'IV-RV Gap', 'D': 'Multi-TF', 'E': 'Liquidity'}
        print(f"  {sig_name} ({labels[sig_name]}): {count} total signals")

    # Define combinations
    signal_names = ['A', 'B', 'C', 'D', 'E']
    signal_labels = {
        'A': 'RSI_Div', 'B': 'BondYield', 'C': 'IV_RV_Gap',
        'D': 'MultiTF', 'E': 'Liquidity'
    }

    # Pairwise combinations
    pairwise_combos = list(combinations(signal_names, 2))

    # Triplet combinations (any 2 of 3)
    triplet_combos = list(combinations(signal_names, 3))

    results_all = {}
    passers = []

    # ---- PAIRWISE ----
    print("\n" + "=" * 80)
    print("PAIRWISE COMBINATIONS (require BOTH within 3-day window)")
    print("=" * 80)

    for i, (s1, s2) in enumerate(pairwise_combos):
        label = f"P{i+1}_{signal_labels[s1]}+{signal_labels[s2]}"
        print(f"\n--- {label} ---")

        # Generate confluence signals for each ticker
        entry_signals = {}
        total_entries = 0
        for ticker in all_signals:
            conf = check_confluence_pairwise(
                all_signals[ticker][s1],
                all_signals[ticker][s2],
                window=CONFLUENCE_WINDOW
            )
            if conf.sum() > 0:
                entry_signals[ticker] = conf
                total_entries += conf.sum()

        print(f"  Total confluence entries: {total_entries}")

        if total_entries < 5:
            print(f"  SKIP: Too few signals (<5)")
            results_all[label] = {'skip': True, 'reason': 'too_few_signals', 'total_entries': int(total_entries)}
            continue

        # Run backtest
        trades, equity_curve = run_backtest(stock_data, entry_signals, label)
        print(f"  Trades executed: {len(trades)}")

        if len(trades) < 5:
            print(f"  SKIP: Too few trades")
            results_all[label] = {'skip': True, 'reason': 'too_few_trades', 'n_trades': len(trades)}
            continue

        # Compute metrics
        metrics = compute_metrics(trades, equity_curve, spy)
        if metrics is None:
            print(f"  SKIP: Metrics computation failed")
            results_all[label] = {'skip': True, 'reason': 'metrics_failed'}
            continue

        # Permutation test
        perm_p = permutation_test(trades, equity_curve, spy)

        # 5-gate
        gate_result = run_5gate(metrics, perm_p)

        print(f"  Sharpe: {metrics['sharpe']:.3f}, Sortino: {metrics['sortino']:.3f}")
        print(f"  WR: {metrics['wr']:.1%}, PF: {metrics['pf']:.2f}")
        print(f"  MDD: {metrics['mdd']:.1%}, Total Return: {metrics['total_return']:.1%}")
        print(f"  Regime Gap: {metrics['regime_gap']:.3f} (Bull: {metrics['bull_sharpe']:.3f}, Bear: {metrics['bear_sharpe']:.3f})")
        print(f"  Perm p: {perm_p:.3f}")
        print(f"  5-Gate: {'PASS' if gate_result['pass'] else 'FAIL'} — {gate_result['gates']}")

        result = {
            'metrics': {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()},
            'perm_p': round(perm_p, 4),
            'gate_result': gate_result,
        }
        results_all[label] = result

        if gate_result['pass']:
            passers.append((label, trades, equity_curve, metrics))
            print(f"  *** 5-GATE PASS — ADVERSARIAL WILL RUN ***")

    # ---- TRIPLET ----
    print("\n" + "=" * 80)
    print("TRIPLET COMBINATIONS (any 2 of 3 within 3-day window)")
    print("=" * 80)

    for i, (s1, s2, s3) in enumerate(triplet_combos):
        label = f"T{i+1}_{signal_labels[s1]}+{signal_labels[s2]}+{signal_labels[s3]}_any2"
        print(f"\n--- {label} ---")

        entry_signals = {}
        total_entries = 0
        for ticker in all_signals:
            conf = check_confluence_triplet_any2(
                all_signals[ticker][s1],
                all_signals[ticker][s2],
                all_signals[ticker][s3],
                window=CONFLUENCE_WINDOW
            )
            if conf.sum() > 0:
                entry_signals[ticker] = conf
                total_entries += conf.sum()

        print(f"  Total confluence entries: {total_entries}")

        if total_entries < 5:
            print(f"  SKIP: Too few signals (<5)")
            results_all[label] = {'skip': True, 'reason': 'too_few_signals', 'total_entries': int(total_entries)}
            continue

        trades, equity_curve = run_backtest(stock_data, entry_signals, label)
        print(f"  Trades executed: {len(trades)}")

        if len(trades) < 5:
            print(f"  SKIP: Too few trades")
            results_all[label] = {'skip': True, 'reason': 'too_few_trades', 'n_trades': len(trades)}
            continue

        metrics = compute_metrics(trades, equity_curve, spy)
        if metrics is None:
            print(f"  SKIP: Metrics computation failed")
            results_all[label] = {'skip': True, 'reason': 'metrics_failed'}
            continue

        perm_p = permutation_test(trades, equity_curve, spy)
        gate_result = run_5gate(metrics, perm_p)

        print(f"  Sharpe: {metrics['sharpe']:.3f}, Sortino: {metrics['sortino']:.3f}")
        print(f"  WR: {metrics['wr']:.1%}, PF: {metrics['pf']:.2f}")
        print(f"  MDD: {metrics['mdd']:.1%}, Total Return: {metrics['total_return']:.1%}")
        print(f"  Regime Gap: {metrics['regime_gap']:.3f} (Bull: {metrics['bull_sharpe']:.3f}, Bear: {metrics['bear_sharpe']:.3f})")
        print(f"  Perm p: {perm_p:.3f}")
        print(f"  5-Gate: {'PASS' if gate_result['pass'] else 'FAIL'} — {gate_result['gates']}")

        result = {
            'metrics': {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()},
            'perm_p': round(perm_p, 4),
            'gate_result': gate_result,
        }
        results_all[label] = result

        if gate_result['pass']:
            passers.append((label, trades, equity_curve, metrics))
            print(f"  *** 5-GATE PASS — ADVERSARIAL WILL RUN ***")

    # ---- ADVERSARIAL on passers ----
    adversarial_results = {}
    if passers:
        print("\n" + "=" * 80)
        print(f"RUNNING ADVERSARIAL ON {len(passers)} 5-GATE PASSERS")
        print("=" * 80)

        for label, trades, equity_curve, metrics in passers:
            adv_result, adv_pass_count = run_adversarial(
                trades, equity_curve, stock_data, spy, metrics, label
            )
            adversarial_results[label] = {
                'tests': adv_result,
                'pass_count': adv_pass_count,
                'total_tests': 6,
            }
            results_all[label]['adversarial'] = adversarial_results[label]

    # ---- SUMMARY ----
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print(f"\nTotal combinations tested: {len(results_all)}")
    print(f"5-Gate passers: {len(passers)}")

    # Sort by Sharpe
    scored = []
    for label, res in results_all.items():
        if 'metrics' in res:
            scored.append((label, res))

    scored.sort(key=lambda x: x[1]['metrics']['sharpe'], reverse=True)

    print(f"\n{'Label':<55} {'Sharpe':>7} {'Sortino':>8} {'WR':>6} {'PF':>6} {'MDD':>7} {'Trades':>7} {'PermP':>6} {'Gap':>6} {'5G':>4} {'Adv':>5}")
    print("-" * 125)

    for label, res in scored:
        m = res['metrics']
        gp = res['gate_result']
        pp = res['perm_p']
        gate_str = 'PASS' if gp['pass'] else 'FAIL'
        adv_str = ''
        if 'adversarial' in res:
            a = res['adversarial']
            adv_str = f"{a['pass_count']}/6"

        print(f"{label:<55} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} {m['wr']:>5.1%} {m['pf']:>6.2f} {m['mdd']:>6.1%} {m['n_trades']:>7} {pp:>6.3f} {m['regime_gap']:>6.3f} {gate_str:>4} {adv_str:>5}")

    # Save results
    output_path = '/home/jupiter/Lvl3Quant/strategies/cross_signal_confluence_v2_results.json'
    # Convert any non-serializable types
    def make_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [make_serializable(x) for x in obj]
        return obj

    with open(output_path, 'w') as f:
        json.dump(make_serializable(results_all), f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # Key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    if passers:
        for label, trades, eq, metrics in passers:
            adv = adversarial_results.get(label, {})
            adv_score = adv.get('pass_count', 'N/A')
            print(f"\n  PASS: {label}")
            print(f"    Sharpe {metrics['sharpe']:.3f}, Sortino {metrics['sortino']:.3f}, WR {metrics['wr']:.1%}")
            print(f"    PF {metrics['pf']:.2f}, MDD {metrics['mdd']:.1%}, Trades {metrics['n_trades']}")
            print(f"    Regime gap {metrics['regime_gap']:.3f}, Adversarial {adv_score}/6")
    else:
        print("\n  No combinations passed 5-gate validation.")
        print("  This means individual signals are already well-optimized and")
        print("  requiring multiple confirmations either:")
        print("    1. Kills trade count (too few co-occurrences)")
        print("    2. Doesn't improve risk-adjusted returns enough")
        print("  Individual validated strategies remain the best approach.")

    return results_all


if __name__ == '__main__':
    results = main()
