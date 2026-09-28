#!/usr/bin/env python3
"""
Oversold Bounce — Debit Call Spread Backtest v1
================================================
Buy call debit spreads on large-cap stocks after sharp selloffs,
betting on a mean-reversion bounce.

Entry signals tested:
  A) RSI(2) < 10
  B) RSI(5) < 20
  C) Single-day drop > 5%
  D) 5-day drop > 10%
  E) 3 consecutive down days

Trade setup:
  - Equity sim first (buy stock, sell N days later) to validate signal
  - Then BS-priced debit call spread (ATM / ATM+5%)

Hold periods: 3, 5, 10 trading days

Quality gates:
  - R1: Regime-agnostic (SPY green/red/flat, reject if |gap| > 0.50)
  - Permutation test: 100 shuffles, p < 0.05
  - Adversarial: sub-period, outlier removal, ticker concentration
  - Transaction costs: 0.1% slippage equity, $0.65/leg options

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
OUTPUT = ROOT / "output" / "oversold_bounce_v1"
OUTPUT.mkdir(parents=True, exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════════════════════

STARTING_CAPITAL       = 10_000
RISK_PER_TRADE         = 250       # $200-300 per trade
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
    'rsi2_lt10':       {'type': 'rsi',       'period': 2,  'threshold': 10},
    'rsi5_lt20':       {'type': 'rsi',       'period': 5,  'threshold': 20},
    'drop_1d_5pct':    {'type': 'daily_drop','pct': 0.05},
    'drop_5d_10pct':   {'type': 'multi_drop','days': 5,    'pct': 0.10},
    'down_3_consec':   {'type': 'consec_down','days': 3},
}

HOLD_PERIODS = [3, 5, 10]

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
                # Handle multi-level columns from yfinance
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]
                data['ticker'] = ticker
                data.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high',
                                     'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
                # Ensure lowercase columns
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
    df['ret_1d'] = df['close'].pct_change()
    df['ret_5d'] = df['close'].pct_change(5)
    df['rsi2'] = compute_rsi(df['close'], 2)
    df['rsi5'] = compute_rsi(df['close'], 5)

    # 3 consecutive down days
    df['down'] = (df['ret_1d'] < 0).astype(int)
    df['consec_down'] = df['down'].rolling(3).sum()

    return df


def check_entry(row, signal_cfg):
    """Check if a row triggers an entry signal."""
    stype = signal_cfg['type']
    if stype == 'rsi':
        val = row.get(f"rsi{signal_cfg['period']}", np.nan)
        return not np.isnan(val) and val < signal_cfg['threshold']
    elif stype == 'daily_drop':
        return row.get('ret_1d', 0) < -signal_cfg['pct']
    elif stype == 'multi_drop':
        return row.get(f"ret_{signal_cfg['days']}d", 0) < -signal_cfg['pct']
    elif stype == 'consec_down':
        return row.get('consec_down', 0) >= signal_cfg['days']
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
    """
    Estimate implied vol from realized vol with a premium.
    Uses 30-day realized vol * 1.3 as a proxy for IV.
    """
    hist = prices_df[(prices_df['ticker'] == ticker) & (prices_df['date'] <= date)]
    if len(hist) < 35:
        return 0.35  # default
    rets = hist['close'].pct_change().dropna().tail(30)
    rv = rets.std() * np.sqrt(252)
    # IV typically trades at premium to RV
    iv = max(rv * 1.3, 0.15)
    return min(iv, 1.5)  # cap at 150%


def price_debit_call_spread(S, spread_pct, T_entry, T_exit, S_exit, iv_entry, iv_exit_factor=0.9):
    """
    Price a bull call spread (buy ATM call, sell OTM call).
    Returns (debit_paid, exit_value, max_profit, max_loss).
    """
    K_long  = S                    # ATM
    K_short = S * (1 + spread_pct) # OTM by spread_pct

    # Entry: buy long call, sell short call
    long_entry  = bs_call_price(S, K_long,  T_entry, iv_entry)
    short_entry = bs_call_price(S, K_short, T_entry, iv_entry)
    debit_paid  = long_entry - short_entry

    # Exit: remaining time value + intrinsic
    iv_exit = iv_entry * iv_exit_factor  # IV typically drops after bounce
    T_remain = max(T_entry - T_exit, 1/252)  # remaining time
    long_exit  = bs_call_price(S_exit, K_long,  T_remain, iv_exit)
    short_exit = bs_call_price(S_exit, K_short, T_remain, iv_exit)
    exit_value = long_exit - short_exit

    spread_width = K_short - K_long
    max_profit = spread_width - debit_paid
    max_loss   = debit_paid

    return debit_paid, exit_value, max_profit, max_loss


# ═════════════════════════════════════════════════════════════════════════════
# EQUITY SIMULATION (validates the signal)
# ═════════════════════════════════════════════════════════════════════════════

def run_equity_backtest(all_data, signal_name, signal_cfg, hold_days, spy_regime):
    """
    Simulate buying the stock on signal, selling N days later.
    Returns list of trade dicts.
    """
    trades = []
    tickers = all_data['ticker'].unique()

    for ticker in tickers:
        df = all_data[all_data['ticker'] == ticker].sort_values('date').reset_index(drop=True)
        if len(df) < 30:
            continue

        df = compute_signals(df)
        dates = df['date'].values
        i = 0
        while i < len(df) - hold_days - 1:
            row = df.iloc[i]
            if check_entry(row, signal_cfg):
                entry_date = row['date']
                entry_price = df.iloc[i+1]['open']  # buy next day open
                exit_idx = min(i + 1 + hold_days, len(df) - 1)
                exit_price = df.iloc[exit_idx]['close']

                # Slippage
                entry_price *= (1 + EQUITY_SLIPPAGE_PCT)
                exit_price  *= (1 - EQUITY_SLIPPAGE_PCT)

                ret = (exit_price - entry_price) / entry_price
                entry_dt = pd.Timestamp(entry_date)

                # Regime at entry
                regime = spy_regime.get(entry_dt.normalize(), 'unknown')

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
    For each equity trade, estimate the corresponding debit call spread P&L.
    """
    options_trades = []
    for t in trades:
        ticker = t['ticker']
        entry_date = t['entry_date']
        S = t['entry_price']
        S_exit = t['exit_price']
        hold = t['hold_days']

        iv = estimate_iv(ticker, entry_date, all_data)
        spread_pct = 0.05  # 5% wide
        T_entry = 30 / 252  # ~30 DTE option
        T_exit  = hold / 252

        debit, exit_val, max_profit, max_loss = price_debit_call_spread(
            S, spread_pct, T_entry, T_exit, S_exit, iv
        )

        if debit <= 0:
            continue

        # Commission: 2 legs open + 2 legs close = 4 legs
        commission = 4 * OPTIONS_COMMISSION_LEG

        # Contract multiplier = 100, but we're sizing to risk $250
        spread_width_dollars = S * spread_pct * 100  # per contract
        n_contracts = max(1, int(RISK_PER_TRADE / (debit * 100)))
        n_contracts = min(n_contracts, 5)  # cap for small account

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
    """
    Simulate portfolio with max concurrent position limit.
    Returns equity curve and filtered trades.
    """
    if not trades_list:
        return [], []

    trades = sorted(trades_list, key=lambda t: t['entry_date'])
    capital = starting_capital
    open_positions = []
    executed = []
    equity_curve = [{'date': trades[0]['entry_date'], 'equity': capital}]

    for trade in trades:
        # Close expired positions
        still_open = []
        for pos in open_positions:
            if trade['entry_date'] >= pos['exit_date']:
                capital += pos.get('total_pnl', pos['return_pct'] * RISK_PER_TRADE)
            else:
                still_open.append(pos)
        open_positions = still_open

        # Check capacity
        if len(open_positions) >= MAX_CONCURRENT:
            continue

        # Check capital
        cost = trade.get('debit_paid', 0) * 100 * trade.get('n_contracts', 1) if 'debit_paid' in trade else RISK_PER_TRADE
        if capital < cost:
            continue

        open_positions.append(trade)
        executed.append(trade)
        equity_curve.append({'date': trade['entry_date'], 'equity': capital})

    # Close remaining
    for pos in open_positions:
        capital += pos.get('total_pnl', pos['return_pct'] * RISK_PER_TRADE)

    if executed:
        equity_curve.append({'date': executed[-1]['exit_date'], 'equity': capital})

    return equity_curve, executed


# ═════════════════════════════════════════════════════════════════════════════
# QUALITY GATES
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
    R1: Regime-agnostic OOT validation.
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
        results[regime] = {'sharpe': round(sharpe, 3), 'n': len(subset),
                          'wr': round((rets > 0).mean(), 3), 'avg_ret': round(rets.mean() * 100, 3)}

    sg = results.get('green', {}).get('sharpe', 0)
    sr = results.get('red', {}).get('sharpe', 0)
    denom = max(abs(sg), abs(sr), 0.001)
    gap = abs(sg - sr) / denom

    passed = gap <= 0.50
    return {
        'pass': passed,
        'gap': round(gap, 3),
        'regimes': results,
        'note': 'PASS' if passed else f'FAIL: gap={gap:.3f} > 0.50'
    }


def permutation_test(trades, n_perms=N_PERMUTATIONS):
    """
    Shuffle returns to get null distribution. p-value = fraction of shuffled
    Sharpes >= observed.
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
    HC #704 adversarial checks:
    1. Sub-period consistency (split in half)
    2. Outlier removal (drop top/bottom 5%)
    3. Ticker concentration (no single ticker > 30% of P&L)
    """
    if len(trades) < 20:
        return {'pass': False, 'reason': 'too few trades for adversarial'}

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
    # Both halves should be profitable
    sub_pass = avg1 > 0 and avg2 > 0
    results['sub_period'] = {
        'pass': sub_pass,
        'first_half': {'wr': round(wr1, 3), 'avg_ret': round(avg1*100, 3), 'n': len(first_half)},
        'second_half': {'wr': round(wr2, 3), 'avg_ret': round(avg2*100, 3), 'n': len(second_half)},
    }

    # 2. Outlier removal
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

    # 3. Ticker concentration
    ticker_pnl = df.groupby('ticker')['return_pct'].sum()
    total_pnl = ticker_pnl[ticker_pnl > 0].sum()
    if total_pnl > 0:
        max_conc = ticker_pnl.max() / total_pnl
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

    # Annualize using avg hold period
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

    return {
        'label': label,
        'n_trades': n,
        'win_rate': round(wr, 4),
        'avg_return_pct': round(avg_ret * 100, 3),
        'avg_bounce_pct': round(avg_ret * 100, 3),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'profit_factor': round(pf, 3),
        'avg_win_pct': round(avg_win * 100, 3),
        'avg_loss_pct': round(avg_loss * 100, 3),
        'max_drawdown_pct': round(rets.min() * 100, 2),
        'best_trade_pct': round(rets.max() * 100, 2),
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
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("OVERSOLD BOUNCE — DEBIT CALL SPREAD BACKTEST v1")
    print("=" * 80)

    # ── Load data ──
    print("\n[1/6] Loading price data...")
    prices_cache = OUTPUT / "prices_cache.parquet"
    spy_cache    = OUTPUT / "spy_cache.parquet"

    all_data = fetch_prices(TICKERS, prices_cache)
    spy_df   = fetch_spy(spy_cache)

    print(f"  Universe: {all_data['ticker'].nunique()} tickers, "
          f"{all_data['date'].min().date()} to {all_data['date'].max().date()}")

    # ── Classify SPY regime ──
    print("\n[2/6] Classifying SPY regime...")
    spy_regime = classify_spy_regime(spy_df)
    regime_counts = defaultdict(int)
    for v in spy_regime.values():
        regime_counts[v] += 1
    print(f"  Regime distribution: {dict(regime_counts)}")

    # ── Run equity backtests ──
    print("\n[3/6] Running equity backtests (signal validation)...")
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

    # ── Quality gates on all combos ──
    print("\n[4/6] Quality gates...")
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

    # ── Options P&L estimation for top combos ──
    print("\n[5/6] Options P&L estimation (BS debit call spreads)...")
    options_results = {}

    # Run options sim for top 5 combos by Sharpe
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

        # Portfolio simulation
        eq_curve, executed = simulate_portfolio(opt_trades, STARTING_CAPITAL)
        if executed:
            final_equity = STARTING_CAPITAL + sum(t['total_pnl'] for t in executed)
            options_results[label]['portfolio_final_equity'] = round(final_equity, 2)
            options_results[label]['portfolio_return_pct'] = round((final_equity / STARTING_CAPITAL - 1) * 100, 2)
            options_results[label]['n_executed'] = len(executed)

        print(f"  {label}: {n} trades, total P&L=${total_pnl:+,.0f}, WR={wr:.1%}, "
              f"avg=${avg_pnl:+,.0f}/trade")

    # ── Compile report ──
    print("\n[6/6] Compiling report...")

    # Per-signal comparison
    signal_comparison = {}
    for sig_name in ENTRY_SIGNALS:
        sig_trades = []
        for label, res in all_results.items():
            if label.startswith(sig_name):
                sig_trades.extend(res['trades'])
        if sig_trades:
            signal_comparison[sig_name] = compute_metrics(sig_trades, sig_name)

    # Per-ticker for best combo
    ticker_breakdown = {}
    if best_combo and all_results[best_combo]['trades']:
        ticker_breakdown = per_ticker_breakdown(all_results[best_combo]['trades'])

    # Build final report
    report = {
        'backtest': 'Oversold Bounce — Debit Call Spread v1',
        'date_run': str(datetime.now()),
        'universe': f'{len(TICKERS)} large-cap stocks',
        'period': f"{all_data['date'].min().date()} to {all_data['date'].max().date()}",
        'starting_capital': STARTING_CAPITAL,
        'risk_per_trade': RISK_PER_TRADE,
        'max_concurrent': MAX_CONCURRENT,
        'best_combo': best_combo,
        'best_sharpe': round(best_sharpe, 3),

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

    # Save
    report_path = OUTPUT / "backtest_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved to {report_path}")

    # ── Print summary ──
    print("\n" + "=" * 80)
    print("SUMMARY — EQUITY SIMULATION (Signal Validation)")
    print("=" * 80)

    print(f"\n{'Config':<30} {'N':>5} {'WR':>7} {'AvgRet':>8} {'Sharpe':>7} {'Sortino':>8} {'PF':>6}")
    print("-" * 80)
    for label, res in sorted(all_results.items(),
                             key=lambda x: x[1]['metrics'].get('sharpe', -999), reverse=True):
        m = res['metrics']
        if m['n_trades'] < 5:
            continue
        print(f"{label:<30} {m['n_trades']:>5} {m.get('win_rate',0):>6.1%} "
              f"{m.get('avg_return_pct',0):>7.2f}% {m.get('sharpe',0):>7.2f} "
              f"{m.get('sortino',0):>7.2f} {m.get('profit_factor',0):>6.2f}")

    print("\n" + "=" * 80)
    print("SIGNAL COMPARISON (aggregated across hold periods)")
    print("=" * 80)
    for sig, m in sorted(signal_comparison.items(), key=lambda x: x[1].get('sharpe', 0), reverse=True):
        print(f"  {sig:<20} n={m['n_trades']:>5}  WR={m.get('win_rate',0):.1%}  "
              f"Sharpe={m.get('sharpe',0):.2f}  PF={m.get('profit_factor',0):.2f}  "
              f"AvgBounce={m.get('avg_bounce_pct',0):.2f}%")

    print("\n" + "=" * 80)
    print("QUALITY GATES")
    print("=" * 80)
    for label, qr in quality_results.items():
        if 'skip' in qr:
            continue
        status = "PASS" if qr['ALL_PASS'] else "FAIL"
        r1_note = qr['regime_test'].get('note', '')
        perm_p = qr['permutation_test'].get('p_value', 1)
        print(f"  {label:<30} {status}  R1: {r1_note}  perm_p={perm_p:.3f}")

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
        print(f"\n{'─'*80}")
        print(f"TOP/BOTTOM TICKERS (best combo: {best_combo})")
        print(f"{'─'*80}")
        items = list(ticker_breakdown.items())
        for ticker, tb in items[:5]:
            print(f"  {ticker:<6} n={tb['n']:>3}  WR={tb['wr']:.1%}  avg={tb['avg_ret']:+.2f}%  total={tb['total_ret']:+.1f}%")
        if len(items) > 10:
            print("  ...")
            for ticker, tb in items[-3:]:
                print(f"  {ticker:<6} n={tb['n']:>3}  WR={tb['wr']:.1%}  avg={tb['avg_ret']:+.2f}%  total={tb['total_ret']:+.1f}%")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == '__main__':
    main()
