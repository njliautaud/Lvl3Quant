#!/usr/bin/env python3
"""
Gap-and-Fade (Contrarian) Backtest v1
=======================================
When a stock gaps on LOW/NORMAL volume (no institutional conviction),
the gap often fills. Fade the gap with options.

Entry signals tested:
  A) Gap UP 2-5% on BELOW-avg volume → buy puts (fade gap up)
  B) Gap DOWN 2-5% on BELOW-avg volume → buy calls (fade gap down)
  C) Gap UP >5% on BELOW-avg volume → buy puts (overextended fade)
  D) Gap UP 2-5% AND RSI(14)>70 → buy puts (overbought + gap = fade)

Hold periods: 1, 3, 5, 10 trading days

HC #705: ALL adversarial checks built inline:
  - Permutation test (200 shuffles)
  - Regime test (SPY green/red/flat, reject if gap>0.50)
  - Sub-period consistency
  - Outlier removal
  - Ticker concentration
  - Pricing sanity

Universe: 30 large-cap stocks, 2019-2026, yfinance daily.
Equity simulation first, then BS-priced options overlay.
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
OUTPUT = ROOT / "output" / "gap_fade_v1"
OUTPUT.mkdir(parents=True, exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════════════════════

STARTING_CAPITAL       = 10_000
RISK_PER_TRADE         = 250       # $250 per trade
MAX_CONCURRENT         = 3
EQUITY_SLIPPAGE_PCT    = 0.001     # 0.1% slippage
OPTIONS_COMMISSION_LEG = 0.65      # $0.65 per leg (Robinhood is $0 equity, $0.65/contract options)
RISK_FREE_RATE         = 0.04
N_PERMUTATIONS         = 200

TICKERS = [
    'AAPL','MSFT','GOOGL','AMZN','META','NVDA','TSLA','NFLX','AMD','INTC',
    'BA','DIS','SBUX','HD','LOW','MCD','NKE','COST','WMT',
    'JPM','GS','BAC','MS','JNJ','PG','KO','UNH','ABBV','CRM','NOW',
]

ENTRY_SIGNALS = {
    'gap_up_low_vol': {
        'type': 'gap_up_low_vol',
        'gap_min': 0.02,
        'gap_max': 0.05,
        'direction': 'fade_down',  # expect price to come back down → buy puts
    },
    'gap_down_low_vol': {
        'type': 'gap_down_low_vol',
        'gap_min': -0.05,
        'gap_max': -0.02,
        'direction': 'fade_up',    # expect price to bounce → buy calls
    },
    'gap_up_big_low_vol': {
        'type': 'gap_up_big_low_vol',
        'gap_min': 0.05,
        'gap_max': 0.15,
        'direction': 'fade_down',  # overextended gap → buy puts
    },
    'gap_up_overbought': {
        'type': 'gap_up_overbought',
        'gap_min': 0.02,
        'gap_max': 0.05,
        'rsi_threshold': 70,
        'direction': 'fade_down',  # overbought + gap = fade
    },
}

HOLD_PERIODS = [1, 3, 5, 10]

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
            print(f"    {ticker}: error -- {e}")
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
# INDICATORS
# ═════════════════════════════════════════════════════════════════════════════

def compute_rsi(series, period):
    """Wilder's RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_signals(df_ticker):
    """Add all signal columns for a single ticker's dataframe."""
    df = df_ticker.sort_values('date').copy()

    # Previous close for gap computation
    df['prev_close'] = df['close'].shift(1)
    df['gap_pct'] = (df['open'] - df['prev_close']) / df['prev_close']

    # RSI(14)
    df['rsi14'] = compute_rsi(df['close'], 14)

    # Volume: 20-day trailing average (shifted to avoid look-ahead)
    df['vol_avg_20d'] = df['volume'].shift(1).rolling(20).mean()

    # Is today's volume below the 20-day average?
    df['low_volume'] = df['volume'] < df['vol_avg_20d']

    return df


def check_entry(row, signal_cfg):
    """Check if a row triggers a gap-and-fade entry signal."""
    stype = signal_cfg['type']
    gap = row.get('gap_pct', np.nan)
    low_vol = row.get('low_volume', False)

    if pd.isna(gap) or not low_vol:
        return False

    if stype == 'gap_up_low_vol':
        return signal_cfg['gap_min'] <= gap <= signal_cfg['gap_max']

    elif stype == 'gap_down_low_vol':
        return signal_cfg['gap_min'] <= gap <= signal_cfg['gap_max']

    elif stype == 'gap_up_big_low_vol':
        return signal_cfg['gap_min'] <= gap <= signal_cfg['gap_max']

    elif stype == 'gap_up_overbought':
        rsi14 = row.get('rsi14', np.nan)
        if pd.isna(rsi14):
            return False
        return (signal_cfg['gap_min'] <= gap <= signal_cfg['gap_max']
                and rsi14 > signal_cfg['rsi_threshold'])

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


def bs_put_price(S, K, T, sigma, r=0.04):
    """Black-Scholes European put price via put-call parity."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(K - S, 0)
    call = bs_call_price(S, K, T, sigma, r)
    return call - S + K * np.exp(-r * T)


def estimate_iv(ticker, date, prices_df):
    """
    Estimate implied vol from realized vol with a premium.
    Uses 30-day realized vol * 1.3 as a proxy for IV.
    """
    hist = prices_df[(prices_df['ticker'] == ticker) & (prices_df['date'] <= date)]
    if len(hist) < 35:
        return 0.35  # default
    rets = hist['close'].pct_change().dropna().tail(30)
    rv = rets.std() * np.sqrt(252)
    iv = max(rv * 1.3, 0.15)
    return min(iv, 1.5)


# ═════════════════════════════════════════════════════════════════════════════
# EQUITY SIMULATION (validates the signal edge before options overlay)
# ═════════════════════════════════════════════════════════════════════════════

def run_equity_backtest(all_data, signal_name, signal_cfg, hold_days, spy_regime):
    """
    Simulate the FADE trade as equity:
    - fade_down: short the stock (entry next open, cover N days later)
    - fade_up:   buy the stock  (entry next open, sell N days later)
    Returns list of trade dicts.
    """
    direction = signal_cfg.get('direction', 'fade_down')
    trades = []
    tickers = all_data['ticker'].unique()

    for ticker in tickers:
        df = all_data[all_data['ticker'] == ticker].sort_values('date').reset_index(drop=True)
        if len(df) < 60:
            continue

        df = compute_signals(df)
        i = 0
        while i < len(df) - hold_days - 1:
            row = df.iloc[i]
            if check_entry(row, signal_cfg):
                entry_date = row['date']
                entry_price = df.iloc[i+1]['open']  # enter next day open

                exit_idx = min(i + 1 + hold_days, len(df) - 1)
                exit_price = df.iloc[exit_idx]['close']

                # Slippage
                if direction == 'fade_down':
                    # Shorting: enter at slightly worse price (lower), cover at slightly higher
                    entry_price *= (1 - EQUITY_SLIPPAGE_PCT)
                    exit_price  *= (1 + EQUITY_SLIPPAGE_PCT)
                    ret = (entry_price - exit_price) / entry_price  # short P&L
                else:
                    # Going long: enter higher, exit lower
                    entry_price *= (1 + EQUITY_SLIPPAGE_PCT)
                    exit_price  *= (1 - EQUITY_SLIPPAGE_PCT)
                    ret = (exit_price - entry_price) / entry_price

                entry_dt = pd.Timestamp(entry_date).normalize()
                regime = spy_regime.get(entry_dt, 'unknown')

                trades.append({
                    'ticker': ticker,
                    'entry_date': entry_dt,
                    'exit_date': pd.Timestamp(df.iloc[exit_idx]['date']),
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'gap_pct': row['gap_pct'],
                    'return_pct': ret,
                    'regime': regime,
                    'signal': signal_name,
                    'hold_days': hold_days,
                    'direction': direction,
                })
                i = exit_idx + 1  # no overlapping trades for same ticker
            else:
                i += 1

    return trades


# ═════════════════════════════════════════════════════════════════════════════
# OPTIONS P&L ESTIMATION
# ═════════════════════════════════════════════════════════════════════════════

def estimate_options_pnl(trades, all_data):
    """
    For each equity trade, estimate the corresponding options P&L:
    - fade_down trades: buy put (or bear put spread)
    - fade_up trades:   buy call (or bull call spread)
    """
    options_trades = []
    for t in trades:
        ticker = t['ticker']
        entry_date = t['entry_date']
        S = t['entry_price']
        S_exit = t['exit_price']
        hold = t['hold_days']
        direction = t.get('direction', 'fade_down')

        iv = estimate_iv(ticker, entry_date, all_data)
        spread_pct = 0.05  # 5% wide spread
        T_entry = 30 / 252  # ~30 DTE
        T_exit  = hold / 252
        T_remain = max(T_entry - T_exit, 1/252)
        iv_exit = iv * 0.9  # IV typically contracts as time passes

        if direction == 'fade_down':
            # Bear put spread: buy ATM put, sell OTM put (lower strike)
            K_long  = S                       # ATM put (buy)
            K_short = S * (1 - spread_pct)    # OTM put (sell)

            long_entry  = bs_put_price(S, K_long, T_entry, iv)
            short_entry = bs_put_price(S, K_short, T_entry, iv)
            debit_paid  = long_entry - short_entry

            long_exit  = bs_put_price(S_exit, K_long, T_remain, iv_exit)
            short_exit = bs_put_price(S_exit, K_short, T_remain, iv_exit)
            exit_value = long_exit - short_exit

        else:
            # Bull call spread: buy ATM call, sell OTM call (higher strike)
            K_long  = S                       # ATM call (buy)
            K_short = S * (1 + spread_pct)    # OTM call (sell)

            long_entry  = bs_call_price(S, K_long, T_entry, iv)
            short_entry = bs_call_price(S, K_short, T_entry, iv)
            debit_paid  = long_entry - short_entry

            long_exit  = bs_call_price(S_exit, K_long, T_remain, iv_exit)
            short_exit = bs_call_price(S_exit, K_short, T_remain, iv_exit)
            exit_value = long_exit - short_exit

        if debit_paid <= 0:
            continue

        # Pricing sanity check (HC #705)
        spread_width = abs(K_long - K_short)
        if debit_paid > spread_width:
            # Debit cannot exceed spread width — skip this trade
            continue
        if exit_value < 0:
            exit_value = 0  # spread can't go negative
        if exit_value > spread_width:
            exit_value = spread_width  # cap at max value

        commission = 4 * OPTIONS_COMMISSION_LEG  # open 2 legs + close 2 legs
        n_contracts = max(1, int(RISK_PER_TRADE / (debit_paid * 100)))
        n_contracts = min(n_contracts, 5)  # cap contracts

        pnl_per_contract = (exit_value - debit_paid) * 100
        total_pnl = pnl_per_contract * n_contracts - commission

        options_trades.append({
            **t,
            'iv_entry': iv,
            'debit_paid': round(debit_paid, 4),
            'exit_value': round(exit_value, 4),
            'spread_width': round(spread_width, 2),
            'n_contracts': n_contracts,
            'pnl_per_contract': round(pnl_per_contract, 2),
            'total_pnl': round(total_pnl, 2),
            'commission': commission,
            'options_return_pct': total_pnl / (debit_paid * 100 * n_contracts) if debit_paid > 0 else 0,
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
# QUALITY GATES (HC #705 — ALL INLINE)
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
    R1: Regime-agnostic validation (HC #428).
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
        'note': 'PASS' if passed else f'FAIL: regime_gap={gap:.3f} > 0.50',
    }


def permutation_test(trades, n_perms=N_PERMUTATIONS):
    """
    HC #705: Shuffle returns 200 times. p-value = fraction of shuffled means >= observed.
    Tests whether the observed mean return is statistically different from random.
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
        'observed_mean_pct': round(obs_mean * 100, 4),
        'n_perms': n_perms,
    }


def adversarial_tests(trades):
    """
    HC #705 adversarial checks:
    1. Sub-period consistency (split in half by time)
    2. Outlier removal (drop top/bottom 5% of returns)
    3. Ticker concentration (no single ticker > 30% of gross P&L)
    4. Pricing sanity (already enforced in options estimation)
    """
    if len(trades) < 20:
        return {'overall_pass': False, 'reason': 'too few trades for adversarial'}

    df = pd.DataFrame(trades)
    results = {}

    # 1. Sub-period consistency
    df_sorted = df.sort_values('entry_date')
    mid = len(df_sorted) // 2
    first_half = df_sorted.iloc[:mid]['return_pct']
    second_half = df_sorted.iloc[mid:]['return_pct']
    wr1 = (first_half > 0).mean()
    wr2 = (second_half > 0).mean()
    avg1 = first_half.mean()
    avg2 = second_half.mean()
    # Both halves must be profitable
    sub_pass = avg1 > 0 and avg2 > 0
    results['sub_period'] = {
        'pass': sub_pass,
        'first_half': {'wr': round(wr1, 3), 'avg_ret_pct': round(avg1*100, 3), 'n': len(first_half)},
        'second_half': {'wr': round(wr2, 3), 'avg_ret_pct': round(avg2*100, 3), 'n': len(second_half)},
    }

    # 2. Outlier removal
    rets = df['return_pct'].values
    p5, p95 = np.percentile(rets, [5, 95])
    trimmed = rets[(rets >= p5) & (rets <= p95)]
    outlier_pass = trimmed.mean() > 0
    results['outlier_removal'] = {
        'pass': outlier_pass,
        'full_mean_pct': round(rets.mean()*100, 3),
        'trimmed_mean_pct': round(trimmed.mean()*100, 3),
        'n_removed': len(rets) - len(trimmed),
    }

    # 3. Ticker concentration
    ticker_pnl = df.groupby('ticker')['return_pct'].sum()
    total_positive = ticker_pnl[ticker_pnl > 0].sum()
    if total_positive > 0:
        max_conc = ticker_pnl.max() / total_positive
    else:
        max_conc = 1.0  # all negative = fail
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

    # Max drawdown from cumulative returns
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


# ═════════════════════════════════════════════════════════════════════════════
# GAP FILL ANALYSIS (diagnostic: how often do gaps actually fill?)
# ═════════════════════════════════════════════════════════════════════════════

def gap_fill_analysis(all_data, spy_regime):
    """
    Diagnostic: For gap-up days with below-avg volume, how often does the
    stock close below the previous close (gap fill) within 1/3/5/10 days?
    """
    print("\n  Gap fill diagnostic (gap up 2-5%, below-avg volume):")
    results = {}

    for horizon in [1, 3, 5, 10]:
        fills = 0
        total = 0
        for ticker in all_data['ticker'].unique():
            df = all_data[all_data['ticker'] == ticker].sort_values('date').reset_index(drop=True)
            if len(df) < 60:
                continue
            df = compute_signals(df)

            for i in range(len(df) - horizon - 1):
                row = df.iloc[i]
                gap = row.get('gap_pct', np.nan)
                low_vol = row.get('low_volume', False)
                prev_close = row.get('prev_close', np.nan)

                if pd.isna(gap) or pd.isna(prev_close) or not low_vol:
                    continue
                if not (0.02 <= gap <= 0.05):
                    continue

                total += 1
                # Check if price touches/crosses below prev_close within horizon
                for j in range(1, horizon + 1):
                    if i + j >= len(df):
                        break
                    if df.iloc[i + j]['low'] <= prev_close:
                        fills += 1
                        break

        fill_rate = fills / total if total > 0 else 0
        results[horizon] = {'fill_rate': round(fill_rate, 3), 'fills': fills, 'total': total}
        print(f"    {horizon}d horizon: {fills}/{total} = {fill_rate:.1%} gap fills")

    return results


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("GAP-AND-FADE (CONTRARIAN) BACKTEST v1")
    print("=" * 80)

    # ── Load data ──
    print("\n[1/7] Loading price data...")
    bounce_cache = ROOT / "output" / "oversold_bounce_v1" / "prices_cache.parquet"
    prices_cache = bounce_cache if bounce_cache.exists() else OUTPUT / "prices_cache.parquet"
    spy_bounce_cache = ROOT / "output" / "oversold_bounce_v1" / "spy_cache.parquet"
    spy_cache = spy_bounce_cache if spy_bounce_cache.exists() else OUTPUT / "spy_cache.parquet"

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

    # ── Gap fill diagnostic ──
    print("\n[3/7] Running gap fill diagnostic...")
    fill_diag = gap_fill_analysis(all_data, spy_regime)

    # ── Run equity backtests ──
    print("\n[4/7] Running equity backtests (signal validation)...")
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

            all_results[label] = {
                'metrics': metrics,
                'trades': trades,
            }

            if metrics.get('sharpe', 0) > best_sharpe and metrics['n_trades'] >= 20:
                best_sharpe = metrics['sharpe']
                best_combo = label

    # ── Quality gates on ALL combos ──
    print("\n[5/7] Quality gates (HC #705 — ALL inline)...")
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
        print(f"  {label}: {status} (R1={r1['pass']}, perm_p={perm['p_value']:.3f}, adv={adv.get('overall_pass', False)})")

    # ── Options P&L estimation for top combos ──
    print("\n[6/7] Options P&L estimation (BS put/call spreads)...")
    options_results = {}

    ranked = sorted(all_results.items(),
                    key=lambda x: x[1]['metrics'].get('sharpe', -999), reverse=True)

    for label, res in ranked[:6]:
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
            'avg_debit': round(opt_df['debit_paid'].mean(), 4),
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
        'backtest': 'Gap-and-Fade (Contrarian) Backtest v1',
        'date_run': str(datetime.now()),
        'universe': f'{len(TICKERS)} large-cap stocks',
        'period': f"{all_data['date'].min().date()} to {all_data['date'].max().date()}",
        'starting_capital': STARTING_CAPITAL,
        'risk_per_trade': RISK_PER_TRADE,
        'max_concurrent': MAX_CONCURRENT,
        'concept': 'Fade gaps that occur on below-average volume (no institutional conviction)',
        'best_combo': best_combo,
        'best_sharpe': round(best_sharpe, 3) if best_sharpe > -999 else None,
        'gap_fill_diagnostic': fill_diag,

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
    }

    report_path = OUTPUT / "backtest_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved to {report_path}")

    # ═════════════════════════════════════════════════════════════════════════
    # PRINT SUMMARY
    # ═════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 80)
    print("GAP FILL DIAGNOSTIC")
    print("=" * 80)
    print("  How often does a gap-up (2-5%, below-avg vol) fill within N days?")
    for h, diag in fill_diag.items():
        bar = "#" * int(diag['fill_rate'] * 50)
        print(f"  {h:>2}d: {diag['fill_rate']:>5.1%} ({diag['fills']}/{diag['total']})  {bar}")

    print("\n" + "=" * 80)
    print("SUMMARY — EQUITY SIMULATION (Signal Validation)")
    print("=" * 80)

    print(f"\n{'Config':<35} {'N':>5} {'WR':>7} {'AvgRet':>8} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} {'MaxDD':>7}")
    print("-" * 92)
    for label, res in sorted(all_results.items(),
                             key=lambda x: x[1]['metrics'].get('sharpe', -999), reverse=True):
        m = res['metrics']
        if m['n_trades'] < 5:
            continue
        print(f"{label:<35} {m['n_trades']:>5} {m.get('win_rate',0):>6.1%} "
              f"{m.get('avg_return_pct',0):>7.2f}% {m.get('sharpe',0):>7.2f} "
              f"{m.get('sortino',0):>7.2f} {m.get('profit_factor',0):>6.2f} "
              f"{m.get('max_drawdown_pct',0):>6.1f}%")

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
    n_pass = 0
    n_fail = 0
    for label, qr in sorted(quality_results.items()):
        if 'skip' in qr:
            print(f"  {label:<35} SKIP ({qr['reason']})")
            continue
        status = "PASS" if qr['ALL_PASS'] else "FAIL"
        if qr['ALL_PASS']:
            n_pass += 1
        else:
            n_fail += 1
        r1_note = qr['regime_test'].get('note', '')
        perm_p = qr['permutation_test'].get('p_value', 1)
        adv_pass = qr['adversarial'].get('overall_pass', False)
        print(f"  {label:<35} {status}  R1: {r1_note}  perm_p={perm_p:.3f}  adv={adv_pass}")

    if options_results:
        print("\n" + "=" * 80)
        print("OPTIONS P&L ESTIMATION (BS Put/Call Spreads)")
        print("=" * 80)
        for label, opr in options_results.items():
            print(f"  {label}:")
            print(f"    Trades: {opr['n_trades']}, WR: {opr['win_rate']:.1%}, "
                  f"Total P&L: ${opr['total_pnl']:+,.0f}")
            print(f"    Avg P&L/trade: ${opr['avg_pnl_per_trade']:+,.0f}, "
                  f"Avg debit: ${opr['avg_debit']:.4f}, Avg contracts: {opr['avg_contracts']:.1f}")
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

    # ── WARNING BANNERS (HC #705) ──
    print("\n" + "!" * 80)
    print("!! HC #705 ADVERSARIAL CHECK WARNINGS !!")
    print("!" * 80)

    if n_pass == 0:
        print("!! WARNING: NO configs passed ALL quality gates.")
        print("!! Gap-and-Fade does NOT show a robust, regime-agnostic edge.")
    else:
        passing = [l for l, qr in quality_results.items() if qr.get('ALL_PASS', False)]
        print(f"!! {n_pass} config(s) PASSED all gates: {passing}")

    # Check if gap fill rate is too low to support the thesis
    gap_up_fill_5d = fill_diag.get(5, {}).get('fill_rate', 0)
    if gap_up_fill_5d < 0.50:
        print(f"!! WARNING: Gap fill rate at 5d is only {gap_up_fill_5d:.1%}.")
        print("!! Below 50% = the gap-fade thesis may be weak for this universe.")

    # Check for negative average returns across ALL signals
    all_neg = all(
        res['metrics'].get('avg_return_pct', 0) <= 0
        for res in all_results.values()
        if res['metrics']['n_trades'] >= 20
    )
    if all_neg:
        print("!! WARNING: ALL signals with 20+ trades have negative avg returns.")
        print("!! The contrarian gap-fade thesis is NOT supported by data.")

    # Check for regime dependency
    regime_fails = [l for l, qr in quality_results.items()
                    if not qr.get('skip') and not qr.get('regime_test', {}).get('pass', True)]
    if regime_fails:
        print(f"!! WARNING: {len(regime_fails)} configs failed regime test (regime-dependent, not real edge).")

    # Check permutation test failures
    perm_fails = [l for l, qr in quality_results.items()
                  if not qr.get('skip') and not qr.get('permutation_test', {}).get('pass', True)]
    if perm_fails:
        print(f"!! WARNING: {len(perm_fails)} configs failed permutation test (p >= 0.05, not stat. significant).")

    print("!" * 80)

    # ── Final verdict ──
    print("\n" + "=" * 80)
    if n_pass > 0:
        passing = [l for l, qr in quality_results.items() if qr.get('ALL_PASS', False)]
        print(f"VERDICT: {n_pass} config(s) passed ALL quality gates: {passing}")
        print("Consider options overlay on these configs for the $440 Robinhood account.")
    else:
        print("VERDICT: NO configs passed all quality gates.")
        print("Gap-and-Fade (contrarian on low-volume gaps) does NOT show a robust edge.")
        print("Do NOT trade this strategy without further refinement.")
    print("=" * 80)


if __name__ == '__main__':
    main()
