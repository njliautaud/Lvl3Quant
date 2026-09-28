#!/usr/bin/env python3
"""
Adversarial Validation: Combined Accumulation Signal (Insider Sentiment Variant F)

Strategy: Buy quality stocks when price >5% below 20-day high AND at least 2 of 3
accumulation conditions are met (OBV divergence, CMF positive, low-volume selloff).
Hold 10 days. Max $200/trade, max 3 concurrent, 2bps slippage.

6 adversarial tests:
1. Inverse Signal Test
2. Random Timing Test (1000 shuffles)
3. Sub-Period Stability (4 sub-periods)
4. Remove Top-3 Tickers
5. Parameter Sensitivity Grid
6. Cost Sensitivity
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from collections import defaultdict

warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────────────────────
UNIVERSE = [
    'AAPL', 'MSFT', 'AVGO', 'JPM', 'JNJ', 'PG', 'KO', 'PEP', 'HD', 'COST',
    'UNH', 'LLY', 'V', 'MA', 'ABBV', 'MRK', 'WMT', 'AMZN', 'GOOGL', 'META'
]
START = '2021-06-01'  # extra buffer for indicators
TRADE_START = '2022-01-01'
END = '2026-07-31'
MAX_PER_TRADE = 200.0
MAX_CONCURRENT = 3
DEFAULT_SLIPPAGE_BPS = 2
DEFAULT_DIP_PCT = 5
DEFAULT_N_CONDITIONS = 2
DEFAULT_CMF_LOOKBACK = 20
DEFAULT_HOLD_DAYS = 10
LOW_VOL_DROP_PCT = 3
LOW_VOL_RATIO = 0.7
N_SHUFFLES = 1000

# ── Data Download ───────────────────────────────────────────────────────────
print("Downloading price data...")
raw = yf.download(UNIVERSE, start=START, end=END, auto_adjust=True, progress=False)

# Handle multi-level columns from yfinance
price_data = {}
for ticker in UNIVERSE:
    try:
        df = pd.DataFrame()
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if isinstance(raw.columns, pd.MultiIndex):
                df[col] = raw[(col, ticker)]
            else:
                df[col] = raw[col]
        df = df.dropna()
        if len(df) > 50:
            price_data[ticker] = df
    except Exception as e:
        print(f"  Skipping {ticker}: {e}")

print(f"Loaded {len(price_data)} tickers")


# ── Indicator Functions ─────────────────────────────────────────────────────
def compute_obv(close, volume):
    """On-Balance Volume"""
    direction = np.sign(close.diff())
    direction.iloc[0] = 0
    return (direction * volume).cumsum()


def compute_cmf(high, low, close, volume, period=20):
    """Chaikin Money Flow"""
    mfm = ((close - low) - (high - close)) / (high - low)
    mfm = mfm.replace([np.inf, -np.inf], 0).fillna(0)
    mfv = mfm * volume
    cmf = mfv.rolling(period).sum() / volume.rolling(period).sum()
    return cmf.fillna(0)


def compute_signals(df, dip_pct=DEFAULT_DIP_PCT, n_conditions=DEFAULT_N_CONDITIONS,
                    cmf_lookback=DEFAULT_CMF_LOOKBACK):
    """Compute accumulation signals for a single ticker dataframe.
    Returns boolean Series of signal days."""
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']

    # 20-day rolling high of close
    high_20 = close.rolling(20).max()
    pct_below = (close - high_20) / high_20 * 100  # negative when below

    # Gate: price > dip_pct% below 20-day high
    dip_gate = pct_below <= -dip_pct

    # Condition A: OBV divergence - OBV new 20-day high while price dipped
    obv = compute_obv(close, volume)
    obv_high_20 = obv.rolling(20).max()
    cond_a = (obv >= obv_high_20) & dip_gate

    # Condition B: CMF positive despite price below high
    cmf = compute_cmf(high, low, close, volume, period=cmf_lookback)
    cond_b = (cmf > 0) & dip_gate

    # Condition C: Low-volume selloff - dropped >3% on volume < 0.7x avg
    ret_1d = close.pct_change() * 100
    vol_avg_20 = volume.rolling(20).mean()
    cond_c = (ret_1d <= -LOW_VOL_DROP_PCT) & (volume < LOW_VOL_RATIO * vol_avg_20)

    # Count conditions met
    conditions_met = cond_a.astype(int) + cond_b.astype(int) + cond_c.astype(int)

    # Final signal: dip gate AND at least n_conditions met
    signal = dip_gate & (conditions_met >= n_conditions)
    return signal


# ── Backtester ──────────────────────────────────────────────────────────────
def backtest(signals_by_ticker, price_data, hold_days=DEFAULT_HOLD_DAYS,
             slippage_bps=DEFAULT_SLIPPAGE_BPS, max_concurrent=MAX_CONCURRENT,
             max_per_trade=MAX_PER_TRADE, trade_start=TRADE_START,
             excluded_tickers=None):
    """
    Run backtest given precomputed signals.
    signals_by_ticker: dict of {ticker: boolean Series indexed by date}
    Returns dict with trades list and summary metrics.
    """
    trades = []
    active_trades = []  # list of (exit_date, ticker)

    # Collect all signal dates across tickers
    all_signal_events = []
    for ticker, sig in signals_by_ticker.items():
        if excluded_tickers and ticker in excluded_tickers:
            continue
        sig_dates = sig[sig].index
        sig_dates = sig_dates[sig_dates >= pd.Timestamp(trade_start)]
        for d in sig_dates:
            all_signal_events.append((d, ticker))

    all_signal_events.sort(key=lambda x: x[0])

    for entry_date, ticker in all_signal_events:
        # Remove expired trades
        active_trades = [(ed, t) for ed, t in active_trades if ed > entry_date]

        if len(active_trades) >= max_concurrent:
            continue

        df = price_data[ticker]
        if entry_date not in df.index:
            continue

        entry_idx = df.index.get_loc(entry_date)
        exit_idx = min(entry_idx + hold_days, len(df) - 1)
        exit_date = df.index[exit_idx]

        entry_price = float(df['Close'].iloc[entry_idx])
        exit_price = float(df['Close'].iloc[exit_idx])

        shares = max(1, int(max_per_trade / entry_price))
        cost = entry_price * shares
        slip = cost * slippage_bps / 10000 * 2  # entry + exit

        pnl = (exit_price - entry_price) * shares - slip
        ret = pnl / cost

        trades.append({
            'ticker': ticker,
            'entry_date': str(entry_date.date()),
            'exit_date': str(exit_date.date()),
            'entry_price': round(entry_price, 2),
            'exit_price': round(exit_price, 2),
            'shares': shares,
            'pnl': round(pnl, 2),
            'return': round(ret, 6),
        })

        active_trades.append((exit_date, ticker))

    return compute_metrics(trades)


def compute_metrics(trades):
    """Compute Sharpe, WR, PF from trade list."""
    if not trades:
        return {'sharpe': 0.0, 'win_rate': 0.0, 'profit_factor': 0.0,
                'n_trades': 0, 'total_pnl': 0.0, 'trades': []}

    returns = [t['return'] for t in trades]
    pnls = [t['pnl'] for t in trades]

    avg_ret = np.mean(returns)
    std_ret = np.std(returns) if len(returns) > 1 else 1e-9
    sharpe = (avg_ret / std_ret) * np.sqrt(252 / DEFAULT_HOLD_DAYS) if std_ret > 0 else 0.0

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    wr = len(wins) / len(pnls) if pnls else 0.0
    pf = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else (
        999.0 if wins else 0.0)

    return {
        'sharpe': round(sharpe, 3),
        'win_rate': round(wr, 4),
        'profit_factor': round(pf, 3),
        'n_trades': len(trades),
        'total_pnl': round(sum(pnls), 2),
        'trades': trades,
    }


# ── Compute Baseline Signals ───────────────────────────────────────────────
print("Computing baseline signals...")
baseline_signals = {}
for ticker, df in price_data.items():
    baseline_signals[ticker] = compute_signals(df)

baseline = backtest(baseline_signals, price_data)
print(f"Baseline: Sharpe={baseline['sharpe']}, WR={baseline['win_rate']:.1%}, "
      f"PF={baseline['profit_factor']}, N={baseline['n_trades']}, "
      f"PnL=${baseline['total_pnl']:.2f}")

results = {
    'strategy': 'Combined Accumulation Signal (Insider Sentiment Variant F)',
    'validation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'baseline': {
        'sharpe': baseline['sharpe'],
        'win_rate': baseline['win_rate'],
        'profit_factor': baseline['profit_factor'],
        'n_trades': baseline['n_trades'],
        'total_pnl': baseline['total_pnl'],
    },
    'tests': {},
}


# ── TEST 1: Inverse Signal ─────────────────────────────────────────────────
print("\n[1/6] Inverse Signal Test...")

inverse_signals = {}
for ticker, df in price_data.items():
    close = df['Close']
    high_20 = close.rolling(20).max()
    pct_below = (close - high_20) / high_20 * 100

    # Inverse: price ABOVE 20-day high OR no accumulation conditions met
    above_high = pct_below >= 0

    # Compute conditions to check none fire
    obv = compute_obv(close, df['Volume'])
    obv_high_20 = obv.rolling(20).max()
    dip_gate = pct_below <= -DEFAULT_DIP_PCT
    cond_a = (obv >= obv_high_20) & dip_gate
    cmf = compute_cmf(df['High'], df['Low'], close, df['Volume'], DEFAULT_CMF_LOOKBACK)
    cond_b = (cmf > 0) & dip_gate
    ret_1d = close.pct_change() * 100
    vol_avg_20 = df['Volume'].rolling(20).mean()
    cond_c = (ret_1d <= -LOW_VOL_DROP_PCT) & (df['Volume'] < LOW_VOL_RATIO * vol_avg_20)
    conditions_met = cond_a.astype(int) + cond_b.astype(int) + cond_c.astype(int)
    no_conditions = conditions_met == 0

    # Inverse signal: above high OR (dipped but no conditions)
    inv_sig = above_high | (dip_gate & no_conditions)

    # Subsample to ~similar trade count (random days matching baseline frequency)
    n_baseline = baseline_signals[ticker].sum()
    n_inv = inv_sig.sum()
    if n_inv > 0 and n_baseline > 0:
        # Sample at similar rate
        ratio = min(1.0, n_baseline / max(n_inv, 1) * 3)  # 3x to get enough
        np.random.seed(42)
        mask = np.random.random(len(inv_sig)) < ratio
        inv_sig = inv_sig & pd.Series(mask, index=inv_sig.index)

    inverse_signals[ticker] = inv_sig

inverse_result = backtest(inverse_signals, price_data)
inv_sharpe = inverse_result['sharpe']
threshold = 0.5 * baseline['sharpe']
t1_pass = inv_sharpe < threshold

print(f"  Inverse Sharpe: {inv_sharpe} (threshold: <{threshold:.3f})")
print(f"  PASS: {t1_pass}")

results['tests']['1_inverse_signal'] = {
    'inverse_sharpe': inv_sharpe,
    'baseline_sharpe': baseline['sharpe'],
    'threshold': round(threshold, 3),
    'inverse_n_trades': inverse_result['n_trades'],
    'pass': t1_pass,
}


# ── TEST 2: Random Timing (1000 shuffles) ──────────────────────────────────
print("\n[2/6] Random Timing Test (1000 shuffles)...")

baseline_sharpe = baseline['sharpe']
shuffle_sharpes = []

for i in range(N_SHUFFLES):
    shuffled_signals = {}
    rng = np.random.RandomState(i)
    for ticker, sig in baseline_signals.items():
        # Shuffle signal dates within the trading period
        sig_trading = sig[sig.index >= pd.Timestamp(TRADE_START)]
        n_signals = sig_trading.sum()
        if n_signals == 0:
            shuffled_signals[ticker] = sig  # no signals to shuffle
            continue
        # Create shuffled version: same number of True values, random positions
        new_sig = sig.copy()
        trading_idx = new_sig.index >= pd.Timestamp(TRADE_START)
        trading_vals = new_sig[trading_idx].values.copy()
        rng.shuffle(trading_vals)
        new_sig.loc[trading_idx] = trading_vals
        shuffled_signals[ticker] = new_sig

    shuf_result = backtest(shuffled_signals, price_data)
    shuffle_sharpes.append(shuf_result['sharpe'])

    if (i + 1) % 200 == 0:
        print(f"  Completed {i+1}/{N_SHUFFLES} shuffles...")

p_value = np.mean([s >= baseline_sharpe for s in shuffle_sharpes])
t2_pass = p_value < 0.05

print(f"  Baseline Sharpe: {baseline_sharpe}")
print(f"  Shuffle mean: {np.mean(shuffle_sharpes):.3f}, "
      f"median: {np.median(shuffle_sharpes):.3f}")
print(f"  p-value: {p_value:.4f}")
print(f"  PASS: {t2_pass}")

results['tests']['2_random_timing'] = {
    'baseline_sharpe': baseline_sharpe,
    'shuffle_mean': round(float(np.mean(shuffle_sharpes)), 3),
    'shuffle_median': round(float(np.median(shuffle_sharpes)), 3),
    'shuffle_std': round(float(np.std(shuffle_sharpes)), 3),
    'shuffle_p5': round(float(np.percentile(shuffle_sharpes, 5)), 3),
    'shuffle_p95': round(float(np.percentile(shuffle_sharpes, 95)), 3),
    'p_value': round(p_value, 4),
    'n_shuffles': N_SHUFFLES,
    'pass': t2_pass,
}


# ── TEST 3: Sub-Period Stability ────────────────────────────────────────────
print("\n[3/6] Sub-Period Stability...")

trade_dates = pd.date_range(TRADE_START, END, freq='B')
n_per_period = len(trade_dates) // 4
sub_periods = []
for i in range(4):
    start_d = trade_dates[i * n_per_period]
    end_d = trade_dates[min((i + 1) * n_per_period - 1, len(trade_dates) - 1)]
    sub_periods.append((str(start_d.date()), str(end_d.date())))

sub_sharpes = []
sub_details = []
for sp_start, sp_end in sub_periods:
    sp_result = backtest(baseline_signals, price_data, trade_start=sp_start)
    # Filter to trades within this sub-period
    sp_trades = [t for t in sp_result['trades']
                 if sp_start <= t['entry_date'] <= sp_end]
    sp_metrics = compute_metrics(sp_trades)
    sub_sharpes.append(sp_metrics['sharpe'])
    sub_details.append({
        'period': f"{sp_start} to {sp_end}",
        'sharpe': sp_metrics['sharpe'],
        'win_rate': sp_metrics['win_rate'],
        'n_trades': sp_metrics['n_trades'],
        'total_pnl': sp_metrics['total_pnl'],
    })
    print(f"  {sp_start} to {sp_end}: Sharpe={sp_metrics['sharpe']}, "
          f"N={sp_metrics['n_trades']}, PnL=${sp_metrics['total_pnl']:.2f}")

all_positive = all(s > 0 for s in sub_sharpes)
t3_pass = all_positive
print(f"  All positive Sharpe: {t3_pass}")

results['tests']['3_sub_period_stability'] = {
    'sub_periods': sub_details,
    'all_positive_sharpe': all_positive,
    'pass': t3_pass,
}


# ── TEST 4: Remove Top-3 Tickers ───────────────────────────────────────────
print("\n[4/6] Remove Top-3 Tickers...")

# Find PnL by ticker
ticker_pnl = defaultdict(float)
for t in baseline['trades']:
    ticker_pnl[t['ticker']] += t['pnl']

sorted_tickers = sorted(ticker_pnl.items(), key=lambda x: x[1], reverse=True)
top3 = [t[0] for t in sorted_tickers[:3]]
print(f"  Top 3 tickers by PnL: {[(t, round(p, 2)) for t, p in sorted_tickers[:3]]}")

reduced_result = backtest(baseline_signals, price_data, excluded_tickers=set(top3))
reduced_sharpe = reduced_result['sharpe']
drop_pct = (1 - reduced_sharpe / baseline['sharpe']) * 100 if baseline['sharpe'] != 0 else 100

t4_pass = drop_pct < 50
print(f"  Reduced Sharpe: {reduced_sharpe} (drop: {drop_pct:.1f}%)")
print(f"  PASS: {t4_pass}")

results['tests']['4_remove_top3'] = {
    'top3_tickers': [(t, round(p, 2)) for t, p in sorted_tickers[:3]],
    'full_sharpe': baseline['sharpe'],
    'reduced_sharpe': reduced_sharpe,
    'sharpe_drop_pct': round(drop_pct, 1),
    'reduced_n_trades': reduced_result['n_trades'],
    'reduced_pnl': reduced_result['total_pnl'],
    'pass': t4_pass,
}


# ── TEST 5: Parameter Sensitivity Grid ─────────────────────────────────────
print("\n[5/6] Parameter Sensitivity Grid...")

dip_thresholds = [3, 5, 7, 10]
n_conditions_list = [1, 2, 3]
cmf_lookbacks = [10, 14, 20, 30]
hold_periods = [5, 7, 10, 15]

total_combos = len(dip_thresholds) * len(n_conditions_list) * len(cmf_lookbacks) * len(hold_periods)
print(f"  Testing {total_combos} parameter combinations...")

grid_results = []
combo_count = 0
positive_count = 0

for dip in dip_thresholds:
    for n_cond in n_conditions_list:
        for cmf_lb in cmf_lookbacks:
            # Precompute signals for this param combo
            param_signals = {}
            for ticker, df in price_data.items():
                param_signals[ticker] = compute_signals(df, dip_pct=dip,
                                                        n_conditions=n_cond,
                                                        cmf_lookback=cmf_lb)
            for hold in hold_periods:
                combo_count += 1
                r = backtest(param_signals, price_data, hold_days=hold)
                if r['sharpe'] > 0.3:
                    positive_count += 1
                grid_results.append({
                    'dip_pct': dip,
                    'n_conditions': n_cond,
                    'cmf_lookback': cmf_lb,
                    'hold_days': hold,
                    'sharpe': r['sharpe'],
                    'n_trades': r['n_trades'],
                    'win_rate': r['win_rate'],
                })

pct_positive = positive_count / total_combos * 100
t5_pass = pct_positive > 30

# Find best and worst
grid_results.sort(key=lambda x: x['sharpe'], reverse=True)
print(f"  {positive_count}/{total_combos} combos ({pct_positive:.1f}%) have Sharpe > 0.3")
print(f"  Best: {grid_results[0]}")
print(f"  Worst: {grid_results[-1]}")
print(f"  PASS: {t5_pass}")

results['tests']['5_parameter_sensitivity'] = {
    'total_combos': total_combos,
    'combos_above_0_3': positive_count,
    'pct_above_0_3': round(pct_positive, 1),
    'best_combo': grid_results[0],
    'worst_combo': grid_results[-1],
    'top_5': grid_results[:5],
    'pass': t5_pass,
}


# ── TEST 6: Cost Sensitivity ───────────────────────────────────────────────
print("\n[6/6] Cost Sensitivity...")

cost_levels = [5, 10, 20, 50]
cost_results = []

for bps in cost_levels:
    r = backtest(baseline_signals, price_data, slippage_bps=bps)
    cost_results.append({
        'slippage_bps': bps,
        'sharpe': r['sharpe'],
        'total_pnl': r['total_pnl'],
        'win_rate': r['win_rate'],
        'n_trades': r['n_trades'],
    })
    print(f"  {bps} bps: Sharpe={r['sharpe']}, PnL=${r['total_pnl']:.2f}")

# Estimate breakeven via interpolation
sharpes_at_cost = [(c['slippage_bps'], c['sharpe']) for c in cost_results]
breakeven_bps = None
for i in range(len(sharpes_at_cost) - 1):
    bps1, s1 = sharpes_at_cost[i]
    bps2, s2 = sharpes_at_cost[i + 1]
    if s1 > 0 and s2 <= 0:
        # Linear interpolation
        breakeven_bps = bps1 + (bps2 - bps1) * s1 / (s1 - s2)
        break

if breakeven_bps is None:
    if all(c['sharpe'] > 0 for c in cost_results):
        breakeven_bps = cost_levels[-1] + 10  # beyond our test range
        print(f"  Breakeven: >{cost_levels[-1]} bps (still profitable at max tested)")
    else:
        breakeven_bps = cost_levels[0]  # already unprofitable
        print(f"  Breakeven: <{cost_levels[0]} bps")
else:
    print(f"  Breakeven: ~{breakeven_bps:.0f} bps")

t6_pass = (breakeven_bps is not None) and (breakeven_bps > 20)
print(f"  PASS: {t6_pass}")

results['tests']['6_cost_sensitivity'] = {
    'cost_levels': cost_results,
    'breakeven_bps': round(breakeven_bps, 1) if breakeven_bps else None,
    'pass': t6_pass,
}


# ── Summary ─────────────────────────────────────────────────────────────────
tests_passed = sum(1 for t in results['tests'].values() if t['pass'])
total_tests = len(results['tests'])

results['summary'] = {
    'tests_passed': tests_passed,
    'total_tests': total_tests,
    'pass_rate': f"{tests_passed}/{total_tests}",
    'overall_pass': tests_passed >= 4,  # majority pass
    'verdict': 'PASS' if tests_passed >= 4 else 'FAIL',
}

print(f"\n{'='*60}")
print(f"ADVERSARIAL VALIDATION SUMMARY")
print(f"{'='*60}")
for name, test in results['tests'].items():
    status = 'PASS' if test['pass'] else 'FAIL'
    print(f"  {name}: {status}")
print(f"\nOverall: {results['summary']['pass_rate']} tests passed "
      f"-> {results['summary']['verdict']}")

# ── Save Results ────────────────────────────────────────────────────────────
output_path = '/home/jupiter/Lvl3Quant/data/insider_sentiment_f_adversarial.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to {output_path}")
