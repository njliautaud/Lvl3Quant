#!/usr/bin/env python3
"""
Volume Breakout from Consolidation — Debit Call Spread Backtest v1
====================================================================
Buy call debit spreads when stocks break out of tight consolidation
ranges on high volume. Distinct from momentum breakout: requires
PRIOR consolidation (tight range / low vol) before the breakout.

Entry signals tested:
  A) "Tight Range Breakout" — 20d ATR < 50th pctile of 100d ATR
     + price breaks above 20d high + volume > 2x 20d avg
  B) "BB Squeeze Breakout" — BB width (20d, 2σ) in bottom 20th pctile
     of trailing 252d + close above upper band
  C) "Compressed Breakout" — 10d range (H-L)/close < 5%
     + today's move > 2% on 2x volume

Trade setup:
  - Equity sim first (validate signal)
  - Then BS-priced debit call spread (ATM / ATM+5%)

Hold periods: 5, 10, 20 trading days

HC #705 Adversarial checks (ALL INLINE):
  - Permutation test (200 shuffles, p < 0.05)
  - Regime test (SPY green/red/flat, reject if |gap| > 0.50)
  - Sub-period consistency (both halves profitable)
  - Outlier removal (5th-95th percentile)
  - Ticker concentration (no single ticker > 30% of P&L)

Universe: 30 large-cap stocks (2019-2026)
"""

import sys, json, warnings, os, time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta, datetime
from scipy.stats import norm
from collections import defaultdict

warnings.filterwarnings("ignore")

ROOT   = Path("/home/jupiter/Lvl3Quant")
OUTPUT = ROOT / "output" / "consolidation_breakout_v1"
OUTPUT.mkdir(parents=True, exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════════════════════

STARTING_CAPITAL       = 10_000
RISK_PER_TRADE         = 250       # $250 per trade
MAX_CONCURRENT         = 3
EQUITY_SLIPPAGE_PCT    = 0.001     # 0.1%
OPTIONS_COMMISSION_LEG = 0.65      # $0.65 per leg
RISK_FREE_RATE         = 0.04
N_PERMUTATIONS         = 200

TICKERS = [
    'AAPL','MSFT','GOOGL','AMZN','META','NVDA','TSLA','NFLX','AMD','INTC',
    'BA','DIS','SBUX','HD','LOW','MCD','NKE','COST','WMT',
    'JPM','GS','BAC','MS','JNJ','PG','KO','UNH','ABBV','CRM','NOW',
]

ENTRY_SIGNALS = {
    'tight_range_breakout': {
        'type': 'atr_consolidation_breakout',
        'atr_period': 20,
        'atr_baseline': 100,
        'atr_pctile_threshold': 50,  # ATR < 50th pctile = consolidation
        'high_lookback': 20,
        'vol_multiple': 2.0,
        'vol_lookback': 20,
    },
    'bb_squeeze_breakout': {
        'type': 'bb_squeeze',
        'bb_period': 20,
        'bb_std': 2.0,
        'width_lookback': 252,
        'width_pctile_threshold': 20,  # BB width in bottom 20th pctile
    },
    'compressed_breakout': {
        'type': 'compressed_range',
        'range_lookback': 10,
        'range_threshold': 0.05,     # 10d range < 5% of close
        'move_threshold': 0.02,      # today's move > 2%
        'vol_multiple': 2.0,
        'vol_lookback': 20,
    },
}

HOLD_PERIODS = [5, 10, 20]

# ═════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

def fetch_prices(tickers, cache_path):
    """Fetch daily OHLCV from yfinance with caching."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        print(f"  Loaded cached prices: {len(df)} rows, {df['ticker'].nunique()} tickers")
        return df

    import yfinance as yf
    print(f"  Downloading prices for {len(tickers)} tickers...")
    all_frames = []
    for ticker in tickers:
        try:
            data = yf.download(ticker, start='2019-01-01', end='2026-07-15',
                               progress=False, auto_adjust=True)
            if len(data) > 100:
                data = data.reset_index()
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]
                data['ticker'] = ticker
                data.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high',
                                     'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
                data.columns = [c.lower() if isinstance(c, str) else c for c in data.columns]
                all_frames.append(data[['date','open','high','low','close','volume','ticker']])
                print(f"    {ticker}: {len(data)} days")
        except Exception as e:
            print(f"    {ticker}: error — {e}")
        time.sleep(0.2)

    df = pd.concat(all_frames, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    df.to_parquet(cache_path, index=False)
    print(f"  Saved {len(df)} rows to cache")
    return df


def fetch_spy(cache_path):
    """Fetch SPY for regime classification."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    import yfinance as yf
    spy = yf.download('SPY', start='2019-01-01', end='2026-07-15',
                       progress=False, auto_adjust=True).reset_index()
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = [c[0] if isinstance(c, tuple) else c for c in spy.columns]
    spy.rename(columns={'Date': 'date', 'Close': 'close'}, inplace=True)
    spy.columns = [c.lower() if isinstance(c, str) else c for c in spy.columns]
    spy = spy[['date','close']].copy()
    spy['date'] = pd.to_datetime(spy['date'])
    spy.to_parquet(cache_path, index=False)
    return spy


# ═════════════════════════════════════════════════════════════════════════════
# INDICATORS — CONSOLIDATION-SPECIFIC
# ═════════════════════════════════════════════════════════════════════════════

def compute_atr(high, low, close, period):
    """Average True Range."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_bb_width(close, period, std_mult):
    """Bollinger Band width as % of middle band."""
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    width = (upper - lower) / sma
    return width, upper, lower, sma


def compute_signals(df_ticker):
    """Add all signal columns for a single ticker's dataframe."""
    df = df_ticker.sort_values('date').copy()
    df['ret_1d'] = df['close'].pct_change()

    # ATR(20)
    df['atr_20'] = compute_atr(df['high'], df['low'], df['close'], 20)

    # Rolling percentile of ATR(20) over last 100 days
    df['atr_pctile'] = df['atr_20'].rolling(100).apply(
        lambda x: (x.iloc[-1] <= x).mean() * 100 if len(x) == 100 else np.nan,
        raw=False
    )

    # 20-day high of close
    df['high_20d'] = df['close'].rolling(20).max()

    # Volume averages
    df['vol_avg_20d'] = df['volume'].rolling(20).mean()

    # Bollinger Bands (20d, 2σ)
    df['bb_width'], df['bb_upper'], df['bb_lower'], df['bb_mid'] = compute_bb_width(
        df['close'], 20, 2.0
    )

    # Rolling percentile of BB width over 252 days
    df['bb_width_pctile'] = df['bb_width'].rolling(252).apply(
        lambda x: (x.iloc[-1] <= x).mean() * 100 if len(x) == 252 else np.nan,
        raw=False
    )

    # 10-day range as % of close
    df['range_10d'] = (df['high'].rolling(10).max() - df['low'].rolling(10).min()) / df['close']

    # Today's absolute move
    df['abs_move'] = df['ret_1d'].abs()

    # Previous close
    df['prev_close'] = df['close'].shift(1)

    return df


def check_entry(row, signal_cfg):
    """Check if a row triggers a consolidation breakout entry signal."""
    stype = signal_cfg['type']

    if stype == 'atr_consolidation_breakout':
        # ATR(20) < 50th percentile of trailing 100-day ATR = consolidation
        # + price breaks above 20-day high + volume > 2x average
        atr_pctile = row.get('atr_pctile', np.nan)
        high_20d = row.get('high_20d', np.nan)
        vol = row.get('volume', 0)
        vol_avg = row.get('vol_avg_20d', np.nan)
        close = row.get('close', 0)

        if pd.isna(atr_pctile) or pd.isna(high_20d) or pd.isna(vol_avg) or vol_avg <= 0:
            return False

        consolidation = atr_pctile < signal_cfg['atr_pctile_threshold']
        breakout = close >= high_20d
        high_volume = vol >= signal_cfg['vol_multiple'] * vol_avg

        return consolidation and breakout and high_volume

    elif stype == 'bb_squeeze':
        # BB width in bottom 20th percentile of trailing year = squeeze
        # + close above upper band = breakout
        bb_width_pctile = row.get('bb_width_pctile', np.nan)
        close = row.get('close', 0)
        bb_upper = row.get('bb_upper', np.nan)

        if pd.isna(bb_width_pctile) or pd.isna(bb_upper):
            return False

        squeeze = bb_width_pctile <= signal_cfg['width_pctile_threshold']
        breakout = close > bb_upper

        return squeeze and breakout

    elif stype == 'compressed_range':
        # 10-day range < 5% of close + today's move > 2% on 2x volume
        range_10d = row.get('range_10d', np.nan)
        abs_move = row.get('abs_move', 0)
        vol = row.get('volume', 0)
        vol_avg = row.get('vol_avg_20d', np.nan)
        ret_1d = row.get('ret_1d', 0)

        if pd.isna(range_10d) or pd.isna(vol_avg) or vol_avg <= 0 or pd.isna(ret_1d):
            return False

        compressed = range_10d < signal_cfg['range_threshold']
        big_move = abs_move >= signal_cfg['move_threshold']
        high_volume = vol >= signal_cfg['vol_multiple'] * vol_avg
        # We want direction: if move is positive, we go long
        bullish = ret_1d > 0

        return compressed and big_move and high_volume and bullish

    return False


# ═════════════════════════════════════════════════════════════════════════════
# BLACK-SCHOLES FOR OPTIONS PRICING
# ═════════════════════════════════════════════════════════════════════════════

def bs_call_price(S, K, T, sigma, r=0.04):
    """Black-Scholes European call price."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def estimate_iv(ticker, date, prices_df):
    """Estimate IV from realized vol with premium."""
    hist = prices_df[(prices_df['ticker'] == ticker) & (prices_df['date'] <= date)]
    if len(hist) < 35:
        return 0.35
    rets = hist['close'].pct_change().dropna().tail(30)
    rv = rets.std() * np.sqrt(252)
    iv = max(rv * 1.3, 0.15)
    return min(iv, 1.5)


def price_debit_call_spread(S, spread_pct, T_entry, T_exit, S_exit, iv_entry, iv_exit_factor=0.9):
    """Price a bull call spread (buy ATM, sell OTM by spread_pct)."""
    K_long  = S
    K_short = S * (1 + spread_pct)

    long_entry  = bs_call_price(S, K_long,  T_entry, iv_entry)
    short_entry = bs_call_price(S, K_short, T_entry, iv_entry)
    debit_paid  = long_entry - short_entry

    iv_exit = iv_entry * iv_exit_factor
    T_remain = max(T_entry - T_exit, 1/252)
    long_exit  = bs_call_price(S_exit, K_long,  T_remain, iv_exit)
    short_exit = bs_call_price(S_exit, K_short, T_remain, iv_exit)
    exit_value = long_exit - short_exit

    spread_width = K_short - K_long
    max_profit = spread_width - debit_paid
    max_loss   = debit_paid

    return debit_paid, exit_value, max_profit, max_loss


# ═════════════════════════════════════════════════════════════════════════════
# EQUITY SIMULATION
# ═════════════════════════════════════════════════════════════════════════════

def run_equity_backtest(all_data, signal_name, signal_cfg, hold_days, spy_regime):
    """Simulate buying stock on signal, selling N days later."""
    trades = []
    tickers = all_data['ticker'].unique()

    for ticker in tickers:
        df = all_data[all_data['ticker'] == ticker].sort_values('date').reset_index(drop=True)
        if len(df) < 260:  # need ~1yr for BB width percentile
            continue

        df = compute_signals(df)
        i = 0
        while i < len(df) - hold_days - 1:
            row = df.iloc[i]
            if check_entry(row, signal_cfg):
                entry_date = row['date']
                entry_price = df.iloc[i+1]['open']  # buy next day open
                exit_idx = min(i + 1 + hold_days, len(df) - 1)
                exit_price = df.iloc[exit_idx]['close']

                entry_price *= (1 + EQUITY_SLIPPAGE_PCT)
                exit_price  *= (1 - EQUITY_SLIPPAGE_PCT)

                ret = (exit_price - entry_price) / entry_price
                entry_dt = pd.Timestamp(entry_date)
                regime = spy_regime.get(entry_dt.normalize(), 'unknown')

                # Track sub-period info
                year = entry_dt.year

                trades.append({
                    'ticker': ticker,
                    'entry_date': entry_dt,
                    'exit_date': pd.Timestamp(df.iloc[exit_idx]['date']),
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'return_pct': ret,
                    'regime': regime,
                    'signal': signal_name,
                    'hold_days': hold_days,
                    'year': year,
                })
                i = exit_idx + 1  # no overlapping trades for same ticker
            else:
                i += 1

    return trades


# ═════════════════════════════════════════════════════════════════════════════
# OPTIONS P&L ESTIMATION
# ═════════════════════════════════════════════════════════════════════════════

def estimate_options_pnl(trades, all_data):
    """For each equity trade, estimate corresponding debit call spread P&L."""
    options_trades = []
    for t in trades:
        ticker = t['ticker']
        entry_date = t['entry_date']
        S = t['entry_price']
        S_exit = t['exit_price']
        hold = t['hold_days']

        iv = estimate_iv(ticker, entry_date, all_data)
        spread_pct = 0.05
        T_entry = 30 / 252
        T_exit  = hold / 252

        debit, exit_val, max_profit, max_loss = price_debit_call_spread(
            S, spread_pct, T_entry, T_exit, S_exit, iv
        )

        if debit <= 0:
            continue

        commission = 4 * OPTIONS_COMMISSION_LEG
        n_contracts = max(1, int(RISK_PER_TRADE / (debit * 100)))
        n_contracts = min(n_contracts, 5)

        pnl_per_contract = (exit_val - debit) * 100
        total_pnl = pnl_per_contract * n_contracts - commission

        options_trades.append({
            **t,
            'iv_entry': iv,
            'debit_paid': debit,
            'exit_value': exit_val,
            'spread_width': S * spread_pct,
            'n_contracts': n_contracts,
            'pnl_per_contract': pnl_per_contract,
            'total_pnl': total_pnl,
            'commission': commission,
            'options_return_pct': total_pnl / (debit * 100 * n_contracts) if debit > 0 else 0,
        })

    return options_trades


# ═════════════════════════════════════════════════════════════════════════════
# PORTFOLIO SIMULATION (with capital constraints)
# ═════════════════════════════════════════════════════════════════════════════

def simulate_portfolio(trades_list, starting_capital=STARTING_CAPITAL):
    """Simulate portfolio with max concurrent position limit."""
    if not trades_list:
        return [], []

    trades = sorted(trades_list, key=lambda t: t['entry_date'])
    capital = starting_capital
    open_positions = []
    executed = []
    equity_curve = [{'date': trades[0]['entry_date'], 'equity': capital}]

    for trade in trades:
        still_open = []
        for pos in open_positions:
            if trade['entry_date'] >= pos['exit_date']:
                capital += pos.get('total_pnl', pos['return_pct'] * RISK_PER_TRADE)
            else:
                still_open.append(pos)
        open_positions = still_open

        if len(open_positions) >= MAX_CONCURRENT:
            continue

        cost = trade.get('debit_paid', 0) * 100 * trade.get('n_contracts', 1) if 'debit_paid' in trade else RISK_PER_TRADE
        if capital < cost:
            continue

        open_positions.append(trade)
        executed.append(trade)
        equity_curve.append({'date': trade['entry_date'], 'equity': capital})

    for pos in open_positions:
        capital += pos.get('total_pnl', pos['return_pct'] * RISK_PER_TRADE)

    if executed:
        equity_curve.append({'date': executed[-1]['exit_date'], 'equity': capital})

    return equity_curve, executed


# ═════════════════════════════════════════════════════════════════════════════
# QUALITY GATES — ALL INLINE PER HC #705
# ═════════════════════════════════════════════════════════════════════════════

def classify_spy_regime(spy_df):
    """Classify each day as green/red/flat based on SPY daily return."""
    spy = spy_df.sort_values('date').copy()
    spy['ret'] = spy['close'].pct_change()
    regime = {}
    for _, row in spy.iterrows():
        dt = pd.Timestamp(row['date']).normalize()
        if pd.isna(row['ret']):
            regime[dt] = 'flat'
        elif row['ret'] > 0.003:
            regime[dt] = 'green'
        elif row['ret'] < -0.003:
            regime[dt] = 'red'
        else:
            regime[dt] = 'flat'
    return regime


def regime_agnostic_test(trades):
    """
    R1: Regime-agnostic test.
    Reject if |Sharpe_green - Sharpe_red| / max(...) > 0.50
    """
    if not trades:
        return {'pass': False, 'reason': 'no trades'}

    df = pd.DataFrame(trades)
    results = {}
    for regime in ['green', 'red', 'flat']:
        subset = df[df['regime'] == regime]
        if len(subset) < 5:
            results[regime] = {'sharpe': 0, 'n': 0}
            continue
        rets = subset['return_pct']
        sharpe = rets.mean() / rets.std() * np.sqrt(252 / subset['hold_days'].mean()) if rets.std() > 0 else 0
        results[regime] = {
            'sharpe': round(sharpe, 3),
            'n': len(subset),
            'wr': round((rets > 0).mean(), 3),
            'avg_ret': round(rets.mean() * 100, 3),
        }

    sg = results.get('green', {}).get('sharpe', 0)
    sr = results.get('red', {}).get('sharpe', 0)
    denom = max(abs(sg), abs(sr), 0.001)
    gap = abs(sg - sr) / denom

    passed = gap <= 0.50
    return {
        'pass': passed,
        'gap': round(gap, 3),
        'regimes': results,
        'note': 'PASS' if passed else f'FAIL: gap={gap:.3f} > 0.50',
    }


def permutation_test(trades, n_perms=N_PERMUTATIONS):
    """
    Shuffle returns to get null distribution.
    p-value = fraction of shuffled means >= observed.
    """
    if len(trades) < 10:
        return {'pass': False, 'p_value': 1.0, 'reason': 'too few trades'}

    df = pd.DataFrame(trades)
    rets = df['return_pct'].values
    obs_mean = rets.mean()

    count_ge = 0
    rng = np.random.RandomState(42)
    for _ in range(n_perms):
        shuffled = rng.permutation(rets)
        if shuffled.mean() >= obs_mean:
            count_ge += 1

    p_value = count_ge / n_perms
    return {
        'pass': p_value < 0.05,
        'p_value': round(p_value, 4),
        'observed_mean': round(obs_mean * 100, 4),
        'n_perms': n_perms,
    }


def adversarial_tests(trades):
    """
    HC #705 adversarial checks:
    1. Sub-period consistency (split in half — both halves must be profitable)
    2. Outlier removal (drop top/bottom 5% — must remain profitable)
    3. Ticker concentration (no single ticker > 30% of total positive P&L)
    """
    if len(trades) < 20:
        return {'overall_pass': False, 'reason': 'too few trades for adversarial'}

    df = pd.DataFrame(trades)
    results = {}

    # ── 1. Sub-period consistency ──
    df_sorted = df.sort_values('entry_date')
    mid = len(df_sorted) // 2
    first_half = df_sorted.iloc[:mid]['return_pct']
    second_half = df_sorted.iloc[mid:]['return_pct']
    wr1 = (first_half > 0).mean()
    wr2 = (second_half > 0).mean()
    avg1 = first_half.mean()
    avg2 = second_half.mean()
    sub_pass = avg1 > 0 and avg2 > 0
    results['sub_period'] = {
        'pass': sub_pass,
        'first_half': {'wr': round(wr1, 3), 'avg_ret': round(avg1*100, 3), 'n': len(first_half)},
        'second_half': {'wr': round(wr2, 3), 'avg_ret': round(avg2*100, 3), 'n': len(second_half)},
    }

    # ── 2. Outlier removal ──
    rets = df['return_pct'].values
    p5, p95 = np.percentile(rets, [5, 95])
    trimmed = rets[(rets >= p5) & (rets <= p95)]
    outlier_pass = trimmed.mean() > 0
    results['outlier_removal'] = {
        'pass': outlier_pass,
        'full_mean': round(rets.mean()*100, 3),
        'trimmed_mean': round(trimmed.mean()*100, 3),
        'n_removed': len(rets) - len(trimmed),
    }

    # ── 3. Ticker concentration ──
    ticker_pnl = df.groupby('ticker')['return_pct'].sum()
    total_positive = ticker_pnl[ticker_pnl > 0].sum()
    if total_positive > 0:
        max_conc = ticker_pnl.max() / total_positive
    else:
        max_conc = 0
    conc_pass = max_conc < 0.30
    top_tickers = ticker_pnl.nlargest(5)
    results['ticker_concentration'] = {
        'pass': conc_pass,
        'max_concentration': round(max_conc, 3),
        'top_5': {k: round(v*100, 2) for k, v in top_tickers.items()},
    }

    overall = all(r.get('pass', False) for r in results.values())
    results['overall_pass'] = overall
    return results


# ═════════════════════════════════════════════════════════════════════════════
# METRICS
# ═════════════════════════════════════════════════════════════════════════════

def compute_metrics(trades, label=''):
    """Compute risk-adjusted metrics for a set of trades."""
    if not trades:
        return {'label': label, 'n_trades': 0}

    df = pd.DataFrame(trades)
    rets = df['return_pct']
    n = len(rets)
    avg_ret = rets.mean()
    std_ret = rets.std()
    wins = rets[rets > 0]
    losses = rets[rets <= 0]
    wr = (rets > 0).mean()

    avg_hold = df['hold_days'].mean()
    ann_factor = np.sqrt(252 / max(avg_hold, 1))

    sharpe = avg_ret / std_ret * ann_factor if std_ret > 0 else 0
    downside = rets[rets < 0].std()
    sortino = avg_ret / downside * ann_factor if downside > 0 and len(rets[rets < 0]) > 2 else 0

    gross_wins = wins.sum() if len(wins) > 0 else 0
    gross_losses = abs(losses.sum()) if len(losses) > 0 else 0.001
    pf = gross_wins / gross_losses if gross_losses > 0 else float('inf')

    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0

    cum_rets = (1 + rets).cumprod()
    running_max = cum_rets.cummax()
    drawdowns = (cum_rets / running_max - 1)
    max_dd = drawdowns.min() * 100

    return {
        'label': label,
        'n_trades': n,
        'win_rate': round(wr, 4),
        'avg_return_pct': round(avg_ret * 100, 3),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'profit_factor': round(pf, 3),
        'avg_win_pct': round(avg_win * 100, 3),
        'avg_loss_pct': round(avg_loss * 100, 3),
        'max_drawdown_pct': round(max_dd, 2),
        'best_trade_pct': round(rets.max() * 100, 2),
        'worst_trade_pct': round(rets.min() * 100, 2),
    }


def per_ticker_breakdown(trades):
    """Breakdown by ticker."""
    df = pd.DataFrame(trades)
    results = {}
    for ticker, grp in df.groupby('ticker'):
        rets = grp['return_pct']
        results[ticker] = {
            'n': len(grp),
            'wr': round((rets > 0).mean(), 3),
            'avg_ret': round(rets.mean() * 100, 3),
            'total_ret': round(rets.sum() * 100, 2),
        }
    return dict(sorted(results.items(), key=lambda x: -x[1]['total_ret']))


def per_year_breakdown(trades):
    """Breakdown by year for sub-period analysis."""
    df = pd.DataFrame(trades)
    results = {}
    for year, grp in df.groupby('year'):
        rets = grp['return_pct']
        std_ret = rets.std()
        avg_hold = grp['hold_days'].mean()
        ann_factor = np.sqrt(252 / max(avg_hold, 1))
        sharpe = rets.mean() / std_ret * ann_factor if std_ret > 0 else 0
        results[int(year)] = {
            'n': len(grp),
            'wr': round((rets > 0).mean(), 3),
            'avg_ret': round(rets.mean() * 100, 3),
            'sharpe': round(sharpe, 3),
            'total_ret': round(rets.sum() * 100, 2),
        }
    return results


# ═════════════════════════════════════════════════════════════════════════════
# WARNING BANNER PRINTER
# ═════════════════════════════════════════════════════════════════════════════

def print_warning_banner(title, msg):
    """Print a prominent warning banner."""
    print()
    print("!" * 80)
    print(f"!!! WARNING: {title}")
    print(f"!!! {msg}")
    print("!" * 80)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("VOLUME BREAKOUT FROM CONSOLIDATION — DEBIT CALL SPREAD BACKTEST v1")
    print("=" * 80)
    print("HC #705: All adversarial checks built inline.")
    print("Concept: Prior consolidation (tight range/low vol) + breakout on volume.")

    # ── Load data ──
    print("\n[1/7] Loading price data...")
    # Reuse cached data from other income_research backtests if available
    for cache_dir in ['oversold_bounce_v1', 'momentum_breakout_v1']:
        candidate = ROOT / "output" / cache_dir / "prices_cache.parquet"
        if candidate.exists():
            prices_cache = candidate
            break
    else:
        prices_cache = OUTPUT / "prices_cache.parquet"

    for cache_dir in ['oversold_bounce_v1', 'momentum_breakout_v1']:
        candidate = ROOT / "output" / cache_dir / "spy_cache.parquet"
        if candidate.exists():
            spy_cache = candidate
            break
    else:
        spy_cache = OUTPUT / "spy_cache.parquet"

    all_data = fetch_prices(TICKERS, prices_cache)
    spy_df   = fetch_spy(spy_cache)

    print(f"  Universe: {all_data['ticker'].nunique()} tickers, "
          f"{all_data['date'].min().date()} to {all_data['date'].max().date()}")

    # ── Classify SPY regime ──
    print("\n[2/7] Classifying SPY regime...")
    spy_regime = classify_spy_regime(spy_df)
    regime_counts = defaultdict(int)
    for v in spy_regime.values():
        regime_counts[v] += 1
    print(f"  Regime distribution: {dict(regime_counts)}")

    # ── Run equity backtests ──
    print("\n[3/7] Running equity backtests (signal validation)...")
    all_results = {}
    best_combo = None
    best_sharpe = -999

    for sig_name, sig_cfg in ENTRY_SIGNALS.items():
        for hold in HOLD_PERIODS:
            label = f"{sig_name}_hold{hold}"
            print(f"  Testing {label}...", end='')

            trades = run_equity_backtest(all_data, sig_name, sig_cfg, hold, spy_regime)
            metrics = compute_metrics(trades, label)
            print(f"  n={metrics['n_trades']}, WR={metrics.get('win_rate',0):.1%}, "
                  f"Sharpe={metrics.get('sharpe',0):.2f}, PF={metrics.get('profit_factor',0):.2f}")

            # Immediate warning if too few trades
            if metrics['n_trades'] < 20:
                print_warning_banner(
                    f"LOW TRADE COUNT: {label}",
                    f"Only {metrics['n_trades']} trades — results statistically unreliable."
                )

            all_results[label] = {
                'metrics': metrics,
                'trades': trades,
            }

            if metrics.get('sharpe', 0) > best_sharpe and metrics['n_trades'] >= 20:
                best_sharpe = metrics['sharpe']
                best_combo = label

    # ── Quality gates ──
    print("\n[4/7] Quality gates (HC #705 — ALL adversarial checks inline)...")
    quality_results = {}
    for label, res in all_results.items():
        trades = res['trades']
        if len(trades) < 10:
            quality_results[label] = {'skip': True, 'reason': f'only {len(trades)} trades'}
            continue

        r1 = regime_agnostic_test(trades)
        perm = permutation_test(trades)
        adv = adversarial_tests(trades) if len(trades) >= 20 else {'overall_pass': False, 'reason': 'too few'}

        all_pass = r1['pass'] and perm['pass'] and adv.get('overall_pass', False)
        quality_results[label] = {
            'regime_test': r1,
            'permutation_test': perm,
            'adversarial': adv,
            'ALL_PASS': all_pass,
        }

        status = "PASS" if all_pass else "FAIL"
        print(f"  {label}: {status} (R1={r1['pass']}, perm p={perm['p_value']:.3f}, adv={adv.get('overall_pass', False)})")

        # Print WARNING banners for failures
        if not r1['pass']:
            print_warning_banner(
                f"REGIME TEST FAILED: {label}",
                f"Gap={r1['gap']:.3f} > 0.50 — strategy is regime-dependent, NOT robust edge."
            )
        if not perm['pass']:
            print_warning_banner(
                f"PERMUTATION TEST FAILED: {label}",
                f"p={perm['p_value']:.3f} — returns NOT statistically significant vs random."
            )
        if isinstance(adv, dict) and not adv.get('overall_pass', False) and 'reason' not in adv:
            failures = []
            if not adv.get('sub_period', {}).get('pass', True):
                failures.append("sub-period (one half unprofitable)")
            if not adv.get('outlier_removal', {}).get('pass', True):
                failures.append("outlier removal (unprofitable after trimming extremes)")
            if not adv.get('ticker_concentration', {}).get('pass', True):
                conc = adv.get('ticker_concentration', {}).get('max_concentration', 0)
                failures.append(f"ticker concentration ({conc:.1%} > 30%)")
            if failures:
                print_warning_banner(
                    f"ADVERSARIAL TESTS FAILED: {label}",
                    f"Failed: {', '.join(failures)}"
                )

    # ── Per-year breakdown for best combo ──
    print("\n[5/7] Sub-period (yearly) analysis...")
    yearly_results = {}
    if best_combo and all_results[best_combo]['trades']:
        yearly = per_year_breakdown(all_results[best_combo]['trades'])
        yearly_results[best_combo] = yearly
        print(f"  Best combo: {best_combo}")
        print(f"  {'Year':<8} {'N':>5} {'WR':>7} {'AvgRet':>8} {'Sharpe':>7} {'TotalRet':>9}")
        print("  " + "-" * 50)
        unprofitable_years = 0
        for year, ym in sorted(yearly.items()):
            flag = " <<<" if ym['avg_ret'] < 0 else ""
            print(f"  {year:<8} {ym['n']:>5} {ym['wr']:>6.1%} {ym['avg_ret']:>7.2f}% "
                  f"{ym['sharpe']:>7.2f} {ym['total_ret']:>8.1f}%{flag}")
            if ym['avg_ret'] < 0:
                unprofitable_years += 1
        if unprofitable_years >= len(yearly) // 2:
            print_warning_banner(
                "YEARLY INCONSISTENCY",
                f"{unprofitable_years}/{len(yearly)} years are UNPROFITABLE — strategy is NOT consistent."
            )

    # ── Options P&L estimation ──
    print("\n[6/7] Options P&L estimation (BS debit call spreads)...")
    options_results = {}

    ranked = sorted(all_results.items(),
                    key=lambda x: x[1]['metrics'].get('sharpe', -999), reverse=True)

    for label, res in ranked[:5]:
        trades = res['trades']
        if len(trades) < 10:
            continue

        opt_trades = estimate_options_pnl(trades, all_data)
        if not opt_trades:
            continue

        opt_df = pd.DataFrame(opt_trades)
        total_pnl = opt_df['total_pnl'].sum()
        avg_pnl = opt_df['total_pnl'].mean()
        wr = (opt_df['total_pnl'] > 0).mean()
        n = len(opt_df)

        options_results[label] = {
            'n_trades': n,
            'total_pnl': round(total_pnl, 2),
            'avg_pnl_per_trade': round(avg_pnl, 2),
            'win_rate': round(wr, 4),
            'avg_debit': round(opt_df['debit_paid'].mean(), 2),
            'avg_contracts': round(opt_df['n_contracts'].mean(), 1),
            'total_commission': round(opt_df['commission'].sum(), 2),
        }

        eq_curve, executed = simulate_portfolio(opt_trades, STARTING_CAPITAL)
        if executed:
            final_equity = STARTING_CAPITAL + sum(t['total_pnl'] for t in executed)
            options_results[label]['portfolio_final_equity'] = round(final_equity, 2)
            options_results[label]['portfolio_return_pct'] = round((final_equity / STARTING_CAPITAL - 1) * 100, 2)
            options_results[label]['n_executed'] = len(executed)

        print(f"  {label}: {n} trades, total P&L=${total_pnl:+,.0f}, WR={wr:.1%}, "
              f"avg=${avg_pnl:+,.0f}/trade")

    # ── Compile report ──
    print("\n[7/7] Compiling report...")

    signal_comparison = {}
    for sig_name in ENTRY_SIGNALS:
        sig_trades = []
        for label, res in all_results.items():
            if label.startswith(sig_name):
                sig_trades.extend(res['trades'])
        if sig_trades:
            signal_comparison[sig_name] = compute_metrics(sig_trades, sig_name)

    ticker_breakdown = {}
    if best_combo and all_results[best_combo]['trades']:
        ticker_breakdown = per_ticker_breakdown(all_results[best_combo]['trades'])

    report = {
        'backtest': 'Volume Breakout from Consolidation — Debit Call Spread v1',
        'date_run': str(datetime.now()),
        'universe': f'{len(TICKERS)} large-cap stocks',
        'period': f"{all_data['date'].min().date()} to {all_data['date'].max().date()}",
        'starting_capital': STARTING_CAPITAL,
        'risk_per_trade': RISK_PER_TRADE,
        'max_concurrent': MAX_CONCURRENT,
        'best_combo': best_combo,
        'best_sharpe': round(best_sharpe, 3) if best_sharpe > -999 else None,

        'equity_sim_results': {
            label: res['metrics']
            for label, res in sorted(all_results.items(),
                                     key=lambda x: x[1]['metrics'].get('sharpe', -999),
                                     reverse=True)
        },

        'signal_comparison': signal_comparison,
        'quality_gates': quality_results,
        'options_pnl_estimation': options_results,
        'ticker_breakdown_best': ticker_breakdown,
        'yearly_breakdown_best': yearly_results,
    }

    report_path = OUTPUT / "backtest_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved: {report_path}")

    # ═════════════════════════════════════════════════════════════════════════
    # PRINT SUMMARY
    # ═════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 80)
    print("SUMMARY — EQUITY SIMULATION (Signal Validation)")
    print("=" * 80)

    print(f"\n{'Config':<35} {'N':>5} {'WR':>7} {'AvgRet':>8} {'Sharpe':>7} {'Sortino':>8} {'PF':>6}")
    print("-" * 85)
    for label, res in sorted(all_results.items(),
                             key=lambda x: x[1]['metrics'].get('sharpe', -999), reverse=True):
        m = res['metrics']
        if m['n_trades'] < 5:
            continue
        print(f"{label:<35} {m['n_trades']:>5} {m.get('win_rate',0):>6.1%} "
              f"{m.get('avg_return_pct',0):>7.2f}% {m.get('sharpe',0):>7.2f} "
              f"{m.get('sortino',0):>7.2f} {m.get('profit_factor',0):>6.2f}")

    print("\n" + "=" * 80)
    print("SIGNAL COMPARISON (aggregated across hold periods)")
    print("=" * 80)
    for sig, m in sorted(signal_comparison.items(), key=lambda x: x[1].get('sharpe', 0), reverse=True):
        print(f"  {sig:<25} n={m['n_trades']:>5}  WR={m.get('win_rate',0):.1%}  "
              f"Sharpe={m.get('sharpe',0):.2f}  PF={m.get('profit_factor',0):.2f}  "
              f"AvgRet={m.get('avg_return_pct',0):.2f}%")

    print("\n" + "=" * 80)
    print("QUALITY GATES (HC #705)")
    print("=" * 80)
    for label, qr in quality_results.items():
        if 'skip' in qr:
            print(f"  {label:<35} SKIP ({qr['reason']})")
            continue
        status = "PASS" if qr['ALL_PASS'] else "FAIL"
        r1_note = qr['regime_test'].get('note', '')
        perm_p = qr['permutation_test'].get('p_value', 1)
        adv_pass = qr['adversarial'].get('overall_pass', False)
        print(f"  {label:<35} {status}  R1: {r1_note}  perm_p={perm_p:.3f}  adv={adv_pass}")

    if options_results:
        print("\n" + "=" * 80)
        print("OPTIONS P&L ESTIMATION (BS Debit Call Spreads)")
        print("=" * 80)
        for label, opr in options_results.items():
            print(f"  {label}:")
            print(f"    Trades: {opr['n_trades']}, WR: {opr['win_rate']:.1%}, "
                  f"Total P&L: ${opr['total_pnl']:+,.0f}")
            print(f"    Avg P&L/trade: ${opr['avg_pnl_per_trade']:+,.0f}, "
                  f"Avg debit: ${opr['avg_debit']:.2f}, Avg contracts: {opr['avg_contracts']:.1f}")
            if 'portfolio_final_equity' in opr:
                print(f"    Portfolio: ${opr['portfolio_final_equity']:,.0f} "
                      f"({opr['portfolio_return_pct']:+.1f}%), "
                      f"{opr['n_executed']} executed trades")

    if ticker_breakdown:
        print(f"\n{'='*80}")
        print(f"TOP/BOTTOM TICKERS (best combo: {best_combo})")
        print(f"{'='*80}")
        items = list(ticker_breakdown.items())
        for ticker, tb in items[:5]:
            print(f"  {ticker:<6} n={tb['n']:>3}  WR={tb['wr']:.1%}  avg={tb['avg_ret']:+.2f}%  total={tb['total_ret']:+.1f}%")
        if len(items) > 10:
            print("  ...")
            for ticker, tb in items[-3:]:
                print(f"  {ticker:<6} n={tb['n']:>3}  WR={tb['wr']:.1%}  avg={tb['avg_ret']:+.2f}%  total={tb['total_ret']:+.1f}%")

    # ── Final verdict ──
    any_pass = any(qr.get('ALL_PASS', False) for qr in quality_results.values() if not qr.get('skip'))
    print("\n" + "=" * 80)
    if any_pass:
        passing = [l for l, qr in quality_results.items() if qr.get('ALL_PASS', False)]
        print(f"VERDICT: {len(passing)} config(s) PASSED all quality gates: {passing}")
        print("These configs show a regime-agnostic, statistically significant edge")
        print("that survives adversarial tests.")
    else:
        print("VERDICT: NO configs passed all quality gates.")
        print("Volume breakout from consolidation does NOT show a robust, regime-agnostic edge")
        print("in this backtest with these parameters.")
    print("=" * 80)


if __name__ == '__main__':
    main()
