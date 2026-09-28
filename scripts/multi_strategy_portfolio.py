#!/usr/bin/env python3
"""
Multi-Strategy Portfolio Diversification Test
---------------------------------------------
Hypothesis: combining multiple partially-correlated earnings/event strategies
reduces regime gap below 0.5 even if individual strategies fail that gate.

Strategies:
1. Earnings Surprise Momentum (hold 60d after >3% gap)
2. Analyst Revision Proxy D (>3% gap + above 200-SMA, hold 20d)
3. PEAD 40-day (all >3% gappers, hold 40d)
4. Vol-Timed UVXY Short (short UVXY via SVXY when VIX contango, hold 5d)
5. Composite Signal C (scoring: gap + 200-SMA + 50-SMA + sector mom, hold 30d)
"""

import json
import warnings
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings('ignore')
np.random.seed(42)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
OOT_START = '2022-01-01'
OOT_END = '2026-07-25'
DATA_START = '2021-01-01'  # need lookback for 200-SMA
TOTAL_EQUITY = 645.0  # per-strategy capital
GAP_THRESHOLD = 0.03  # 3% gap = earnings proxy
REGIME_GAP_GATE = 0.50
SHARPE_GATE = 0.50
PERM_P_GATE = 0.05
MAXDD_GATE = -0.50
MIN_TRADES_GATE = 20
N_PERM = 500

# Large-cap universe
TICKERS = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA', 'JPM', 'JNJ', 'V',
    'PG', 'UNH', 'HD', 'MA', 'DIS', 'PYPL', 'BAC', 'ADBE', 'CRM', 'NFLX',
    'CSCO', 'PFE', 'INTC', 'ABT', 'KO', 'PEP', 'TMO', 'AVGO', 'COST', 'WMT',
    'MRK', 'CVX', 'ABBV', 'LLY', 'AMD', 'QCOM', 'TXN', 'LOW', 'MDT', 'NEE',
    'AMGN', 'AMAT', 'ISRG', 'BKNG', 'MU', 'LRCX', 'GILD', 'SBUX', 'GS', 'CAT'
]

VOL_TICKERS = ['SVXY', '^VIX']

print("=" * 70)
print("MULTI-STRATEGY PORTFOLIO DIVERSIFICATION TEST")
print("=" * 70)

# ─── DATA DOWNLOAD ────────────────────────────────────────────────────────────
print("\n[1/6] Downloading price data...")

def download_data():
    """Download all needed price data."""
    # Stock data
    all_tickers = TICKERS + ['SPY']
    stock_data = {}

    # Download in batches
    batch_size = 10
    for i in range(0, len(all_tickers), batch_size):
        batch = all_tickers[i:i+batch_size]
        try:
            df = yf.download(batch, start=DATA_START, end=OOT_END,
                           progress=False, auto_adjust=True, threads=True)
            if len(batch) == 1:
                stock_data[batch[0]] = df
            else:
                for t in batch:
                    try:
                        sub = df.xs(t, level=1, axis=1) if isinstance(df.columns, pd.MultiIndex) else df
                        if not sub.empty:
                            stock_data[t] = sub
                    except:
                        pass
        except Exception as e:
            print(f"  Warning: batch {batch} failed: {e}")

    # VIX + SVXY
    vol_data = {}
    for t in VOL_TICKERS:
        try:
            df = yf.download(t, start=DATA_START, end=OOT_END, progress=False, auto_adjust=True)
            if not df.empty:
                vol_data[t] = df
        except:
            pass

    return stock_data, vol_data

stock_data, vol_data = download_data()
print(f"  Downloaded {len(stock_data)} stocks, {len(vol_data)} vol instruments")

# SPY for regime classification
spy = stock_data.get('SPY')
if spy is None or spy.empty:
    print("ERROR: Could not download SPY data")
    sys.exit(1)

spy_close = spy['Close'].squeeze().dropna()
spy_sma200 = spy_close.rolling(200).mean()

# ─── HELPER FUNCTIONS ─────────────────────────────────────────────────────────

def detect_gaps(ticker_data, threshold=GAP_THRESHOLD):
    """Detect days where open gaps up >threshold from prior close."""
    if ticker_data is None or ticker_data.empty:
        return pd.Series(dtype=float)
    close = ticker_data['Close'].squeeze()
    opn = ticker_data['Open'].squeeze()
    prev_close = close.shift(1)
    gap_pct = (opn - prev_close) / prev_close
    # Only positive gaps (earnings beats)
    gaps = gap_pct[gap_pct > threshold].dropna()
    return gaps

def compute_sma(series, window):
    """Compute SMA."""
    return series.rolling(window).mean()

def get_regime(date, spy_close=spy_close, spy_sma200=spy_sma200):
    """Bull = SPY > 200-SMA, Bear = SPY < 200-SMA."""
    if date in spy_sma200.index:
        idx = spy_sma200.index.get_loc(date)
        return 'bull' if spy_close.iloc[idx] > spy_sma200.iloc[idx] else 'bear'
    # Find nearest prior date
    prior = spy_sma200.index[spy_sma200.index <= date]
    if len(prior) == 0:
        return 'bull'
    idx = spy_sma200.index.get_loc(prior[-1])
    return 'bull' if spy_close.iloc[idx] > spy_sma200.iloc[idx] else 'bear'

def daily_returns_from_trades(trades, start_date, end_date):
    """Convert list of (entry_date, exit_date, return_pct) to daily return series.

    Returns are normalized by the number of concurrent positions so that
    the portfolio return on any day represents an equal-weight allocation
    across all active positions.
    """
    dates = pd.bdate_range(start=start_date, end=end_date)
    daily_sum = pd.Series(0.0, index=dates)
    daily_count = pd.Series(0, index=dates)

    for entry, exit_d, ret in trades:
        if exit_d > dates[-1]:
            exit_d = dates[-1]
        hold_days = pd.bdate_range(start=entry, end=exit_d)
        hold_days = hold_days[(hold_days >= dates[0]) & (hold_days <= dates[-1])]
        if len(hold_days) > 0:
            daily_ret = ret / len(hold_days)
            for d in hold_days:
                if d in daily_sum.index:
                    daily_sum[d] += daily_ret
                    daily_count[d] += 1

    # Normalize by number of concurrent positions (equal-weight across active positions)
    daily_count = daily_count.replace(0, 1)  # avoid div by zero on inactive days
    daily = daily_sum / daily_count
    return daily

def compute_metrics(daily_returns, trades, label=""):
    """Compute all strategy metrics."""
    dr = daily_returns.dropna()
    if len(dr) == 0:
        return None

    # Sharpe (annualized)
    mean_r = dr.mean()
    std_r = dr.std()
    sharpe = (mean_r / std_r * np.sqrt(252)) if std_r > 0 else 0.0

    # Sortino
    downside = dr[dr < 0]
    downside_std = downside.std() if len(downside) > 0 else 1e-10
    sortino = (mean_r / downside_std * np.sqrt(252)) if downside_std > 0 else 0.0

    # Profit Factor
    gains = dr[dr > 0].sum()
    losses = abs(dr[dr < 0].sum())
    pf = gains / losses if losses > 0 else float('inf')

    # Win Rate (on trade level)
    if trades:
        wins = sum(1 for _, _, r in trades if r > 0)
        wr = wins / len(trades) if len(trades) > 0 else 0
    else:
        wins = (dr > 0).sum()
        wr = wins / len(dr) if len(dr) > 0 else 0

    # Max Drawdown
    cum = (1 + dr).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    maxdd = dd.min()

    # Total return
    total_ret = cum.iloc[-1] - 1 if len(cum) > 0 else 0

    # Regime analysis
    bull_rets = []
    bear_rets = []
    for date, ret in dr.items():
        regime = get_regime(date)
        if regime == 'bull':
            bull_rets.append(ret)
        else:
            bear_rets.append(ret)

    bull_rets = np.array(bull_rets)
    bear_rets = np.array(bear_rets)

    bull_sharpe = (bull_rets.mean() / bull_rets.std() * np.sqrt(252)) if len(bull_rets) > 1 and bull_rets.std() > 0 else 0.0
    bear_sharpe = (bear_rets.mean() / bear_rets.std() * np.sqrt(252)) if len(bear_rets) > 1 and bear_rets.std() > 0 else 0.0

    max_abs = max(abs(bull_sharpe), abs(bear_sharpe))
    regime_gap = abs(bull_sharpe - bear_sharpe) / max_abs if max_abs > 0 else 0.0

    n_trades = len(trades) if trades else int((dr != 0).sum())

    return {
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'profit_factor': round(pf, 3),
        'win_rate': round(wr, 3),
        'max_drawdown': round(maxdd, 3),
        'total_return': round(total_ret, 4),
        'n_trades': n_trades,
        'bull_sharpe': round(bull_sharpe, 3),
        'bear_sharpe': round(bear_sharpe, 3),
        'regime_gap': round(regime_gap, 3),
        'bull_days': len(bull_rets),
        'bear_days': len(bear_rets),
    }

def permutation_test_trades(trades, n_perm=N_PERM):
    """Permutation test on TRADE-level returns (not daily, which are autocorrelated).

    Null hypothesis: trade returns are random (sign doesn't matter).
    We shuffle the signs of trade returns and compare mean return.
    """
    if not trades or len(trades) < 5:
        return 1.0

    trade_rets = np.array([r for _, _, r in trades])
    actual_mean = trade_rets.mean()

    count_above = 0
    for _ in range(n_perm):
        # Randomly flip signs of trade returns
        signs = np.random.choice([-1, 1], size=len(trade_rets))
        perm_mean = (trade_rets * signs).mean()
        if perm_mean >= actual_mean:
            count_above += 1

    p_value = (count_above + 1) / (n_perm + 1)
    return round(p_value, 4)

def passes_gates(metrics, p_value):
    """Check if strategy/portfolio passes all 5 gates."""
    gates = {
        'sharpe_gate': metrics['sharpe'] > SHARPE_GATE,
        'perm_p_gate': p_value < PERM_P_GATE,
        'regime_gap_gate': metrics['regime_gap'] < REGIME_GAP_GATE,
        'maxdd_gate': metrics['max_drawdown'] > MAXDD_GATE,
        'min_trades_gate': metrics['n_trades'] >= MIN_TRADES_GATE,
    }
    gates['all_pass'] = all(gates.values())
    return gates

def get_col(df, col):
    """Safely extract a column as a Series (handles multi-index yfinance output)."""
    return df[col].squeeze()

# ─── STRATEGY IMPLEMENTATIONS ─────────────────────────────────────────────────
print("\n[2/6] Generating strategy trades...")

oot_start = pd.Timestamp(OOT_START)
oot_end = pd.Timestamp(OOT_END)

def strategy_1_earnings_momentum(hold_days=60):
    """Buy top gappers (>3%), hold 60 trading days."""
    trades = []
    for ticker in TICKERS:
        data = stock_data.get(ticker)
        if data is None or data.empty:
            continue
        gaps = detect_gaps(data)
        close = get_col(data, 'Close')
        opn = get_col(data, 'Open')
        for gap_date, gap_pct in gaps.items():
            if gap_date < oot_start or gap_date > oot_end:
                continue
            entry_idx = close.index.get_loc(gap_date)
            exit_idx = min(entry_idx + hold_days, len(close) - 1)
            entry_price = opn.iloc[entry_idx]
            exit_price = close.iloc[exit_idx]
            ret = (exit_price - entry_price) / entry_price
            exit_date = close.index[exit_idx]
            trades.append((gap_date, exit_date, float(ret)))
    return trades

def strategy_2_analyst_revision_proxy(hold_days=20):
    """Buy gappers above 200-SMA, hold 20 trading days."""
    trades = []
    for ticker in TICKERS:
        data = stock_data.get(ticker)
        if data is None or data.empty:
            continue
        gaps = detect_gaps(data)
        close = get_col(data, 'Close')
        opn = get_col(data, 'Open')
        sma200 = compute_sma(close, 200)
        for gap_date, gap_pct in gaps.items():
            if gap_date < oot_start or gap_date > oot_end:
                continue
            idx = close.index.get_loc(gap_date)
            if pd.isna(sma200.iloc[idx]) or close.iloc[idx] <= sma200.iloc[idx]:
                continue
            entry_price = opn.iloc[idx]
            exit_idx = min(idx + hold_days, len(close) - 1)
            exit_price = close.iloc[exit_idx]
            ret = (exit_price - entry_price) / entry_price
            exit_date = close.index[exit_idx]
            trades.append((gap_date, exit_date, float(ret)))
    return trades

def strategy_3_pead_40d(hold_days=40):
    """Buy all gappers, hold 40 trading days."""
    trades = []
    for ticker in TICKERS:
        data = stock_data.get(ticker)
        if data is None or data.empty:
            continue
        gaps = detect_gaps(data)
        close = get_col(data, 'Close')
        opn = get_col(data, 'Open')
        for gap_date, gap_pct in gaps.items():
            if gap_date < oot_start or gap_date > oot_end:
                continue
            idx = close.index.get_loc(gap_date)
            entry_price = opn.iloc[idx]
            exit_idx = min(idx + hold_days, len(close) - 1)
            exit_price = close.iloc[exit_idx]
            ret = (exit_price - entry_price) / entry_price
            exit_date = close.index[exit_idx]
            trades.append((gap_date, exit_date, float(ret)))
    return trades

def strategy_4_vol_timed_svxy(hold_days=5):
    """Short UVXY (via long SVXY) when VIX in contango. Hold 5 days."""
    trades = []
    vix = vol_data.get('^VIX')
    svxy = vol_data.get('SVXY')
    if vix is None or svxy is None:
        print("  Warning: VIX or SVXY data missing, strategy 4 will have 0 trades")
        return trades

    # Flatten to Series if needed (yfinance can return DataFrame with ticker column)
    vix_close = vix['Close'].squeeze().dropna()
    svxy_close = svxy['Close'].squeeze().dropna()

    vix_ma10 = vix_close.rolling(10).mean()
    vix_ma30 = vix_close.rolling(30).mean()

    # Flatten index to remove timezone if present
    vix_ma10.index = vix_ma10.index.tz_localize(None) if vix_ma10.index.tz else vix_ma10.index
    vix_ma30.index = vix_ma30.index.tz_localize(None) if vix_ma30.index.tz else vix_ma30.index
    svxy_close.index = svxy_close.index.tz_localize(None) if svxy_close.index.tz else svxy_close.index

    # Entry when VIX 10MA > VIX 30MA per user spec
    for date in pd.bdate_range(start=OOT_START, end=OOT_END):
        if date not in vix_ma10.index or date not in vix_ma30.index:
            continue
        if date not in svxy_close.index:
            continue
        v10 = vix_ma10.loc[date]
        v30 = vix_ma30.loc[date]
        if pd.isna(v10) or pd.isna(v30):
            continue

        if v10 > v30:
            idx = svxy_close.index.get_loc(date)
            exit_idx = min(idx + hold_days, len(svxy_close) - 1)
            entry_price = svxy_close.iloc[idx]
            exit_price = svxy_close.iloc[exit_idx]
            if entry_price > 0:
                ret = (exit_price - entry_price) / entry_price
                exit_date = svxy_close.index[exit_idx]
                trades.append((date, exit_date, ret))

    # Too many signals — subsample to ~weekly entries (avoid massive overlap)
    if len(trades) > 200:
        # Keep only one entry per 5 bdays
        filtered = []
        last_entry = None
        for t in sorted(trades, key=lambda x: x[0]):
            if last_entry is None or (t[0] - last_entry).days >= 7:
                filtered.append(t)
                last_entry = t[0]
        trades = filtered

    return trades

def strategy_5_composite_signal(hold_days=30):
    """Composite scoring: gap size + above 200-SMA + above 50-SMA + sector momentum. Buy score >60."""
    trades = []

    # Sector ETFs for momentum
    sector_map = {
        'AAPL': 'XLK', 'MSFT': 'XLK', 'NVDA': 'XLK', 'ADBE': 'XLK', 'CRM': 'XLK',
        'CSCO': 'XLK', 'INTC': 'XLK', 'AVGO': 'XLK', 'AMD': 'XLK', 'QCOM': 'XLK',
        'TXN': 'XLK', 'AMAT': 'XLK', 'LRCX': 'XLK', 'MU': 'XLK',
        'AMZN': 'XLY', 'TSLA': 'XLY', 'HD': 'XLY', 'LOW': 'XLY', 'BKNG': 'XLY',
        'SBUX': 'XLY', 'COST': 'XLY', 'WMT': 'XLP', 'PG': 'XLP', 'KO': 'XLP', 'PEP': 'XLP',
        'JPM': 'XLF', 'BAC': 'XLF', 'GS': 'XLF', 'V': 'XLF', 'MA': 'XLF',
        'JNJ': 'XLV', 'UNH': 'XLV', 'PFE': 'XLV', 'ABT': 'XLV', 'TMO': 'XLV',
        'MRK': 'XLV', 'ABBV': 'XLV', 'LLY': 'XLV', 'AMGN': 'XLV', 'GILD': 'XLV',
        'ISRG': 'XLV', 'MDT': 'XLV',
        'GOOGL': 'XLC', 'META': 'XLC', 'NFLX': 'XLC', 'DIS': 'XLC',
        'PYPL': 'XLK', 'CVX': 'XLE', 'NEE': 'XLU', 'CAT': 'XLI',
    }

    # Use SPY as sector proxy for simplicity (20d momentum)
    spy_mom = spy_close.pct_change(20)

    for ticker in TICKERS:
        data = stock_data.get(ticker)
        if data is None or data.empty:
            continue
        gaps = detect_gaps(data)
        close = get_col(data, 'Close')
        opn = get_col(data, 'Open')
        sma200 = compute_sma(close, 200)
        sma50 = compute_sma(close, 50)

        for gap_date, gap_pct in gaps.items():
            if gap_date < oot_start or gap_date > oot_end:
                continue
            idx = close.index.get_loc(gap_date)

            # Score components (0-100)
            score = 0

            # Gap size score (3% = 20, 5% = 35, 10%+ = 50)
            score += min(gap_pct * 500, 50)

            # Above 200-SMA (+20)
            if not pd.isna(sma200.iloc[idx]) and close.iloc[idx] > sma200.iloc[idx]:
                score += 20

            # Above 50-SMA (+15)
            if not pd.isna(sma50.iloc[idx]) and close.iloc[idx] > sma50.iloc[idx]:
                score += 15

            # Sector momentum (+15 if SPY 20d mom > 0)
            if gap_date in spy_mom.index and not pd.isna(spy_mom.loc[gap_date]) and spy_mom.loc[gap_date] > 0:
                score += 15

            if score >= 60:
                entry_price = opn.iloc[idx]
                exit_idx = min(idx + hold_days, len(close) - 1)
                exit_price = close.iloc[exit_idx]
                ret = (exit_price - entry_price) / entry_price
                exit_date = close.index[exit_idx]
                trades.append((gap_date, exit_date, ret))

    return trades


# Generate all strategy trades
strat_trades = {}
strat_names = {
    1: 'Earnings Surprise Momentum (60d)',
    2: 'Analyst Revision Proxy D (20d)',
    3: 'PEAD 40-day',
    4: 'Vol-Timed SVXY (5d)',
    5: 'Composite Signal C (30d)',
}

strat_trades[1] = strategy_1_earnings_momentum()
print(f"  Strategy 1: {len(strat_trades[1])} trades")
strat_trades[2] = strategy_2_analyst_revision_proxy()
print(f"  Strategy 2: {len(strat_trades[2])} trades")
strat_trades[3] = strategy_3_pead_40d()
print(f"  Strategy 3: {len(strat_trades[3])} trades")
strat_trades[4] = strategy_4_vol_timed_svxy()
print(f"  Strategy 4: {len(strat_trades[4])} trades")
strat_trades[5] = strategy_5_composite_signal()
print(f"  Strategy 5: {len(strat_trades[5])} trades")

# ─── DAILY RETURNS PER STRATEGY ──────────────────────────────────────────────
print("\n[3/6] Computing daily returns per strategy...")

strat_daily = {}
for s_id in range(1, 6):
    strat_daily[s_id] = daily_returns_from_trades(strat_trades[s_id], OOT_START, OOT_END)
    active_days = (strat_daily[s_id] != 0).sum()
    print(f"  Strategy {s_id}: {active_days} active days out of {len(strat_daily[s_id])}")

# ─── INDIVIDUAL STRATEGY METRICS ─────────────────────────────────────────────
print("\n[4/6] Computing individual strategy metrics + permutation tests...")

individual_results = {}
for s_id in range(1, 6):
    metrics = compute_metrics(strat_daily[s_id], strat_trades[s_id])
    if metrics is None:
        print(f"  Strategy {s_id}: NO DATA")
        individual_results[s_id] = {'error': 'no data'}
        continue

    p_val = permutation_test_trades(strat_trades[s_id])
    gates = passes_gates(metrics, p_val)

    individual_results[s_id] = {
        'name': strat_names[s_id],
        'metrics': metrics,
        'perm_p_value': p_val,
        'gates': gates,
    }

    print(f"  Strategy {s_id} ({strat_names[s_id]}):")
    print(f"    Sharpe={metrics['sharpe']}, Sortino={metrics['sortino']}, PF={metrics['profit_factor']}")
    print(f"    WR={metrics['win_rate']}, MaxDD={metrics['max_drawdown']}, Trades={metrics['n_trades']}")
    print(f"    Bull Sharpe={metrics['bull_sharpe']}, Bear Sharpe={metrics['bear_sharpe']}, Regime Gap={metrics['regime_gap']}")
    print(f"    Perm p={p_val}, All Gates Pass={gates['all_pass']}")

# ─── PORTFOLIO COMBINATIONS ──────────────────────────────────────────────────
print("\n[5/6] Building portfolio combinations...")

def build_portfolio(strat_ids, method='equal_weight'):
    """Combine strategies into a portfolio."""
    n = len(strat_ids)
    if n == 0:
        return pd.Series(dtype=float), []

    if method == 'equal_weight':
        weights = {s: 1.0 / n for s in strat_ids}
    elif method == 'risk_parity':
        # Inverse volatility weighting
        vols = {}
        for s in strat_ids:
            vol = strat_daily[s].std()
            vols[s] = vol if vol > 0 else 1e-10
        inv_vols = {s: 1.0 / v for s, v in vols.items()}
        total_inv = sum(inv_vols.values())
        weights = {s: iv / total_inv for s, iv in inv_vols.items()}
    else:
        weights = {s: 1.0 / n for s in strat_ids}

    # Weighted daily returns
    portfolio_daily = sum(weights[s] * strat_daily[s] for s in strat_ids)

    # Aggregate trades (for counting)
    all_trades = []
    for s in strat_ids:
        for t in strat_trades[s]:
            all_trades.append(t)

    return portfolio_daily, all_trades, weights

# Define combinations
combos = {
    'all_5_equal': {'ids': [1, 2, 3, 4, 5], 'method': 'equal_weight', 'label': 'All 5 Strategies (Equal Weight)'},
    'all_5_riskparity': {'ids': [1, 2, 3, 4, 5], 'method': 'risk_parity', 'label': 'All 5 Strategies (Risk Parity)'},
    'best_3_equal': {'ids': [1, 2, 5], 'method': 'equal_weight', 'label': 'Best 3: S1+S2+S5 (Equal Weight)'},
    'best_3_riskparity': {'ids': [1, 2, 5], 'method': 'risk_parity', 'label': 'Best 3: S1+S2+S5 (Risk Parity)'},
    'earnings_only_equal': {'ids': [1, 2, 3], 'method': 'equal_weight', 'label': 'Earnings Only: S1+S2+S3 (Equal Weight)'},
    'earnings_only_riskparity': {'ids': [1, 2, 3], 'method': 'risk_parity', 'label': 'Earnings Only: S1+S2+S3 (Risk Parity)'},
    'earnings_vol_equal': {'ids': [1, 2, 3, 4], 'method': 'equal_weight', 'label': 'Earnings+Vol: S1+S2+S3+S4 (Equal Weight)'},
    'earnings_vol_riskparity': {'ids': [1, 2, 3, 4], 'method': 'risk_parity', 'label': 'Earnings+Vol: S1+S2+S3+S4 (Risk Parity)'},
}

portfolio_results = {}
for combo_name, combo_cfg in combos.items():
    portfolio_daily, all_trades, weights = build_portfolio(combo_cfg['ids'], combo_cfg['method'])
    metrics = compute_metrics(portfolio_daily, all_trades)

    if metrics is None:
        portfolio_results[combo_name] = {'error': 'no data'}
        continue

    p_val = permutation_test_trades(all_trades)
    gates = passes_gates(metrics, p_val)

    # Correlation matrix of constituent strategies
    strat_corrs = {}
    ids = combo_cfg['ids']
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            corr = strat_daily[ids[i]].corr(strat_daily[ids[j]])
            strat_corrs[f"S{ids[i]}_S{ids[j]}"] = round(corr, 3)

    avg_corr = np.mean(list(strat_corrs.values())) if strat_corrs else 0

    portfolio_results[combo_name] = {
        'label': combo_cfg['label'],
        'strategy_ids': combo_cfg['ids'],
        'method': combo_cfg['method'],
        'weights': {f"S{k}": round(v, 3) for k, v in weights.items()},
        'metrics': metrics,
        'perm_p_value': p_val,
        'gates': gates,
        'correlations': strat_corrs,
        'avg_correlation': round(avg_corr, 3),
    }

    print(f"\n  {combo_cfg['label']}:")
    print(f"    Weights: {', '.join(f'S{k}={v:.1%}' for k, v in weights.items())}")
    print(f"    Sharpe={metrics['sharpe']}, Sortino={metrics['sortino']}, PF={metrics['profit_factor']}")
    print(f"    WR={metrics['win_rate']}, MaxDD={metrics['max_drawdown']}, Trades={metrics['n_trades']}")
    print(f"    Bull Sharpe={metrics['bull_sharpe']}, Bear Sharpe={metrics['bear_sharpe']}")
    print(f"    Regime Gap={metrics['regime_gap']} {'PASS' if metrics['regime_gap'] < REGIME_GAP_GATE else 'FAIL'}")
    print(f"    Perm p={p_val}, Avg Corr={avg_corr:.3f}")
    print(f"    All Gates: {'PASS' if gates['all_pass'] else 'FAIL'} ({gates})")

# ─── SUMMARY & SAVE ──────────────────────────────────────────────────────────
print("\n[6/6] Generating summary...")

# Find best portfolio by regime gap (among those with Sharpe > 0.5)
best_regime_gap = None
best_combo = None
for combo_name, res in portfolio_results.items():
    if 'error' in res:
        continue
    if res['metrics']['sharpe'] > SHARPE_GATE:
        if best_regime_gap is None or res['metrics']['regime_gap'] < best_regime_gap:
            best_regime_gap = res['metrics']['regime_gap']
            best_combo = combo_name

# Diversification benefit analysis
print("\n" + "=" * 70)
print("DIVERSIFICATION ANALYSIS SUMMARY")
print("=" * 70)

# Individual regime gaps
print("\nIndividual Strategy Regime Gaps:")
for s_id in range(1, 6):
    if 'error' not in individual_results[s_id]:
        m = individual_results[s_id]['metrics']
        print(f"  S{s_id} ({strat_names[s_id]}): Regime Gap = {m['regime_gap']} | Sharpe = {m['sharpe']}")

print("\nPortfolio Regime Gaps:")
for combo_name, res in portfolio_results.items():
    if 'error' in res:
        continue
    m = res['metrics']
    g = res['gates']
    status = "ALL GATES PASS" if g['all_pass'] else "FAILS"
    print(f"  {res['label']}: Regime Gap = {m['regime_gap']} | Sharpe = {m['sharpe']} | {status}")

if best_combo:
    br = portfolio_results[best_combo]
    print(f"\nBest portfolio for regime gap: {br['label']}")
    print(f"  Regime Gap: {br['metrics']['regime_gap']} (gate: <{REGIME_GAP_GATE})")
    print(f"  Sharpe: {br['metrics']['sharpe']}, Sortino: {br['metrics']['sortino']}")
    print(f"  Avg strategy correlation: {br['avg_correlation']}")

# Hypothesis verdict
any_portfolio_passes = any(
    'error' not in r and r['gates']['all_pass']
    for r in portfolio_results.values()
)

diversification_reduces_gap = False
if individual_results:
    indiv_gaps = [r['metrics']['regime_gap'] for r in individual_results.values() if 'error' not in r]
    port_gaps = [r['metrics']['regime_gap'] for r in portfolio_results.values() if 'error' not in r]
    if indiv_gaps and port_gaps:
        avg_indiv_gap = np.mean(indiv_gaps)
        min_port_gap = min(port_gaps)
        diversification_reduces_gap = min_port_gap < avg_indiv_gap
        print(f"\nAvg individual regime gap: {avg_indiv_gap:.3f}")
        print(f"Best portfolio regime gap: {min_port_gap:.3f}")
        print(f"Diversification reduces gap: {'YES' if diversification_reduces_gap else 'NO'} (by {avg_indiv_gap - min_port_gap:.3f})")

hypothesis_result = "CONFIRMED" if any_portfolio_passes else "REJECTED"
print(f"\n{'=' * 70}")
print(f"HYPOTHESIS: Portfolio diversification fixes regime gap")
print(f"VERDICT: {hypothesis_result}")
if any_portfolio_passes:
    passing = [r['label'] for r in portfolio_results.values() if 'error' not in r and r['gates']['all_pass']]
    print(f"Passing portfolios: {passing}")
else:
    print("No portfolio combination passes all 5 gates simultaneously.")
    # Show which gates fail
    for combo_name, res in portfolio_results.items():
        if 'error' in res:
            continue
        failing = [k for k, v in res['gates'].items() if not v and k != 'all_pass']
        if failing:
            print(f"  {res['label']}: fails {failing}")
print(f"{'=' * 70}")

# Save results
output = {
    'metadata': {
        'generated': datetime.now().isoformat(),
        'oot_period': f'{OOT_START} to {OOT_END}',
        'universe': f'{len(TICKERS)} large-cap stocks + SVXY',
        'gap_threshold': GAP_THRESHOLD,
        'n_permutations': N_PERM,
        'equity_per_strategy': TOTAL_EQUITY,
        'regime_definition': 'SPY > 200-SMA = bull, < 200-SMA = bear',
    },
    'gates': {
        'sharpe_min': SHARPE_GATE,
        'perm_p_max': PERM_P_GATE,
        'regime_gap_max': REGIME_GAP_GATE,
        'maxdd_min': MAXDD_GATE,
        'min_trades': MIN_TRADES_GATE,
    },
    'individual_strategies': {
        f'S{k}': v for k, v in individual_results.items()
    },
    'portfolio_combinations': portfolio_results,
    'hypothesis': {
        'statement': 'Portfolio diversification of partially-correlated earnings strategies reduces regime gap below 0.5',
        'verdict': hypothesis_result,
        'diversification_reduces_gap': diversification_reduces_gap,
        'any_portfolio_passes_all_gates': any_portfolio_passes,
    },
}

# Convert Timestamps to strings for JSON serialization
def clean_for_json(obj):
    if isinstance(obj, dict):
        return {str(k): clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    elif isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, float) and (np.isinf(obj) or np.isnan(obj)):
        return str(obj)
    return obj

output = clean_for_json(output)

output_path = '/home/jupiter/Lvl3Quant/data/multi_strategy_portfolio_results.json'
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nResults saved to {output_path}")
print("DONE.")
