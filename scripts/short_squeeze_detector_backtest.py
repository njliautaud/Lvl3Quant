#!/usr/bin/env python3
"""
SHORT INTEREST SQUEEZE DETECTOR BACKTEST
=========================================
Concept: Stocks with high short interest that show positive momentum reversal
+ volume surge are squeeze candidates. Since we can't get real-time short interest
history, we PROXY it using:
  - Price volatility (high vol = typically high short interest targets)
  - Recent drawdown from highs (shorts pile in after drops)
  - Volume spike detection (squeeze begins with volume surge)

Signal:
  1. Stock has dropped >15% from 20-day high (shorts piled in)
  2. 3-day price momentum turns positive (reversal starting)
  3. Volume > 2x 20-day average (institutional buying / short covering)
  4. Optional: VIX not extremely high (>35) — avoid catching falling knives in crashes

Trade: Buy shares (or simulate call equivalent), hold 5-10 trading days.
Exit: Fixed 7-day hold, OR +15% gain (take profit), OR -8% loss (stop loss).

Validation: 5 gates + adversarial checks.
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
INITIAL_CAPITAL = 645.0
MAX_POSITION_SIZE = 300.0  # max $300 per trade
COMMISSION_PER_TRADE = 0.0  # shares: $0 on Robinhood
# For options equivalent: $0.65/contract each way = $1.30 RT
# We'll model as shares for simplicity, then also show options version
OPTIONS_COMMISSION_RT = 1.30  # per contract round-trip
SPREAD_SLIPPAGE_PCT = 0.15  # 0.15% slippage on entry+exit combined

# Signal parameters
DRAWDOWN_LOOKBACK = 20  # days to measure drawdown from high
DRAWDOWN_THRESHOLD = -0.15  # must have dropped 15%+ from recent high
MOMENTUM_LOOKBACK = 3  # 3-day momentum for reversal
VOLUME_SURGE_MULT = 2.0  # volume must be 2x 20-day avg
VOLUME_AVG_LOOKBACK = 20

# Trade parameters
HOLD_DAYS = 7
TAKE_PROFIT_PCT = 0.15
STOP_LOSS_PCT = -0.08

# Walk-forward
WF_TRAIN_DAYS = 252  # 1 year training window
WF_TEST_DAYS = 21  # 1 month OOT

# Backtest period
BT_START = '2022-01-01'
BT_END = '2026-07-24'

EXCLUDE_TICKERS = ['SPY', '^VIX']  # Not squeeze candidates

np.random.seed(42)

# ============================================================
# LOAD DATA
# ============================================================
print("=" * 70)
print("SHORT INTEREST SQUEEZE DETECTOR BACKTEST")
print("=" * 70)

df = pd.read_parquet('/home/jupiter/Lvl3Quant/data/growth_stocks_prices.parquet')
df = df.reset_index()
df['date'] = pd.to_datetime(df['date'])
df = df[~df['ticker'].isin(EXCLUDE_TICKERS)]
df = df[(df['date'] >= BT_START) & (df['date'] <= BT_END)]

# Get VIX for regime classification
vix_df = pd.read_parquet('/home/jupiter/Lvl3Quant/data/growth_stocks_prices.parquet')
vix_df = vix_df.reset_index()
vix_df = vix_df[vix_df['ticker'] == '^VIX'][['date', 'close']].rename(columns={'close': 'vix'})
vix_df['date'] = pd.to_datetime(vix_df['date'])

# SPY for regime
spy_df = pd.read_parquet('/home/jupiter/Lvl3Quant/data/growth_stocks_prices.parquet')
spy_df = spy_df.reset_index()
spy_df = spy_df[spy_df['ticker'] == 'SPY'][['date', 'close']].rename(columns={'close': 'spy_close'})
spy_df['date'] = pd.to_datetime(spy_df['date'])
spy_df['spy_sma50'] = spy_df['spy_close'].rolling(50).mean()
spy_df['regime'] = np.where(spy_df['spy_close'] >= spy_df['spy_sma50'], 'bull', 'bear')

tickers = df['ticker'].unique()
print(f"Universe: {len(tickers)} tickers")
print(f"Period: {df['date'].min().date()} to {df['date'].max().date()}")

# ============================================================
# GENERATE SIGNALS
# ============================================================
def compute_signals(df):
    """Compute squeeze signals for all tickers."""
    signals = []

    for ticker in tickers:
        tdf = df[df['ticker'] == ticker].sort_values('date').copy()
        if len(tdf) < DRAWDOWN_LOOKBACK + 5:
            continue

        # Rolling 20-day high
        tdf['high_20d'] = tdf['high'].rolling(DRAWDOWN_LOOKBACK).max()

        # Drawdown from 20-day high
        tdf['drawdown'] = (tdf['close'] / tdf['high_20d']) - 1.0

        # 3-day momentum (price change)
        tdf['mom_3d'] = tdf['close'].pct_change(MOMENTUM_LOOKBACK)

        # Volume surge
        tdf['vol_avg_20d'] = tdf['volume'].rolling(VOLUME_AVG_LOOKBACK).mean()
        tdf['vol_ratio'] = tdf['volume'] / tdf['vol_avg_20d']

        # 5-day momentum (for confirmation)
        tdf['mom_5d'] = tdf['close'].pct_change(5)

        # Volatility (20-day realized vol as proxy for short interest)
        tdf['rvol_20d'] = tdf['close'].pct_change().rolling(20).std() * np.sqrt(252)

        # Signal: drawdown + momentum reversal + volume surge
        tdf['signal'] = (
            (tdf['drawdown'] <= DRAWDOWN_THRESHOLD) &  # dropped 15%+ from high
            (tdf['mom_3d'] > 0.02) &  # 3-day positive reversal >2%
            (tdf['vol_ratio'] >= VOLUME_SURGE_MULT) &  # volume surge
            (tdf['rvol_20d'] > 0.40)  # high realized vol (proxy for short interest)
        ).astype(int)

        # Forward returns for backtest
        for d in range(1, HOLD_DAYS + 3):
            tdf[f'fwd_ret_{d}d'] = tdf['close'].shift(-d) / tdf['close'] - 1.0

        tdf['ticker_col'] = ticker
        signals.append(tdf[tdf['signal'] == 1])

    if not signals:
        return pd.DataFrame()
    return pd.concat(signals, ignore_index=True)

sig_df = compute_signals(df)
print(f"\nTotal raw signals: {len(sig_df)}")

# Merge VIX filter
sig_df = sig_df.merge(vix_df, on='date', how='left')
# Filter out extreme VIX (>35) — crash mode, not squeeze mode
sig_df = sig_df[sig_df['vix'] <= 35]
print(f"After VIX filter (<=35): {len(sig_df)}")

# Merge regime
sig_df = sig_df.merge(spy_df[['date', 'regime']], on='date', how='left')

# ============================================================
# SIMULATE TRADES WITH POSITION SIZING
# ============================================================
def simulate_trades(sig_df, direction_override=None, exclude_ticker=None):
    """
    Simulate trades with realistic position sizing.
    direction_override: 'random' for random baseline, None for normal
    exclude_ticker: ticker to exclude for leave-one-out
    """
    trades = []
    equity = INITIAL_CAPITAL

    sdf = sig_df.copy()
    if exclude_ticker:
        sdf = sdf[sdf['ticker_col'] != exclude_ticker]

    # Sort by date, process one signal per day max per ticker
    sdf = sdf.sort_values('date')

    # Track open positions to avoid overlapping
    open_positions = {}  # ticker -> exit_date

    for _, row in sdf.iterrows():
        ticker = row['ticker_col']
        entry_date = row['date']

        # Skip if already in position for this ticker
        if ticker in open_positions and entry_date < open_positions[ticker]:
            continue

        # Position sizing
        pos_size = min(MAX_POSITION_SIZE, equity * 0.3)  # max 30% of equity or $300
        if pos_size < 50:  # minimum position
            continue

        # Simulate day-by-day P&L with TP/SL
        exit_ret = None
        exit_day = HOLD_DAYS

        for d in range(1, HOLD_DAYS + 1):
            col = f'fwd_ret_{d}d'
            if col not in row or pd.isna(row[col]):
                continue

            ret_d = row[col]

            if direction_override == 'random':
                # Random direction: flip sign 50% of the time
                if np.random.random() < 0.5:
                    ret_d = -ret_d

            if ret_d >= TAKE_PROFIT_PCT:
                exit_ret = TAKE_PROFIT_PCT
                exit_day = d
                break
            elif ret_d <= STOP_LOSS_PCT:
                exit_ret = STOP_LOSS_PCT
                exit_day = d
                break

        if exit_ret is None:
            # Hold to end
            col = f'fwd_ret_{HOLD_DAYS}d'
            if col in row and not pd.isna(row[col]):
                exit_ret = row[col]
                if direction_override == 'random' and np.random.random() < 0.5:
                    exit_ret = -exit_ret
            else:
                continue

        # Apply slippage
        exit_ret -= SPREAD_SLIPPAGE_PCT / 100.0

        # P&L
        pnl = pos_size * exit_ret
        equity += pnl

        open_positions[ticker] = entry_date + timedelta(days=exit_day * 1.5)  # approximate

        trades.append({
            'date': entry_date,
            'ticker': ticker,
            'entry_price': row['close'],
            'exit_ret': exit_ret,
            'pnl': pnl,
            'equity': equity,
            'hold_days': exit_day,
            'pos_size': pos_size,
            'regime': row.get('regime', 'unknown'),
        })

    return pd.DataFrame(trades)

# Main backtest
trades_df = simulate_trades(sig_df)
print(f"\nTotal trades: {len(trades_df)}")

if len(trades_df) == 0:
    print("NO TRADES GENERATED — strategy has no signals in this period.")
    print("RESULT: FAIL — insufficient signals")
    import sys; sys.exit(0)

# ============================================================
# COMPUTE METRICS
# ============================================================
def compute_metrics(trades_df, label="Main"):
    if len(trades_df) == 0:
        return None

    rets = trades_df['pnl'] / trades_df['pos_size']

    mean_ret = rets.mean()
    std_ret = rets.std()

    # Annualize (assume ~50 trades/year avg)
    trades_per_year = len(trades_df) / 4.5  # 4.5 year period
    ann_ret = mean_ret * trades_per_year
    ann_vol = std_ret * np.sqrt(trades_per_year)

    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    downside_rets = rets[rets < 0]
    downside_vol = downside_rets.std() * np.sqrt(trades_per_year) if len(downside_rets) > 0 else 1e-6
    sortino = ann_ret / downside_vol if downside_vol > 0 else 0

    wins = (rets > 0).sum()
    losses = (rets <= 0).sum()
    wr = wins / len(rets)

    avg_win = rets[rets > 0].mean() if wins > 0 else 0
    avg_loss = abs(rets[rets <= 0].mean()) if losses > 0 else 1e-6
    pf = (avg_win * wins) / (avg_loss * losses) if losses > 0 and avg_loss > 0 else float('inf')

    # Max drawdown from equity curve
    eq = trades_df['equity'].values
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    mdd = dd.min()

    final_equity = trades_df['equity'].iloc[-1]
    total_return = (final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL

    metrics = {
        'label': label,
        'n_trades': len(trades_df),
        'win_rate': wr,
        'sharpe': sharpe,
        'sortino': sortino,
        'profit_factor': pf,
        'max_drawdown': mdd,
        'final_equity': final_equity,
        'total_return': total_return,
        'avg_return_per_trade': mean_ret,
        'trades_per_year': trades_per_year,
    }

    print(f"\n--- {label} ---")
    print(f"  Trades: {metrics['n_trades']}")
    print(f"  Win Rate: {metrics['win_rate']:.1%}")
    print(f"  Sharpe: {metrics['sharpe']:.3f}")
    print(f"  Sortino: {metrics['sortino']:.3f}")
    print(f"  Profit Factor: {metrics['profit_factor']:.3f}")
    print(f"  Max Drawdown: {metrics['max_drawdown']:.1%}")
    print(f"  Final Equity: ${metrics['final_equity']:.2f} (from ${INITIAL_CAPITAL})")
    print(f"  Total Return: {metrics['total_return']:.1%}")
    print(f"  Avg Return/Trade: {metrics['avg_return_per_trade']:.2%}")

    return metrics

main_metrics = compute_metrics(trades_df, "Short Squeeze Detector")

# ============================================================
# REGIME ANALYSIS
# ============================================================
print("\n" + "=" * 50)
print("REGIME ANALYSIS")
print("=" * 50)

bull_trades = trades_df[trades_df['regime'] == 'bull']
bear_trades = trades_df[trades_df['regime'] == 'bear']

bull_m = compute_metrics(bull_trades, "BULL regime") if len(bull_trades) > 5 else None
bear_m = compute_metrics(bear_trades, "BEAR regime") if len(bear_trades) > 5 else None

if bull_m and bear_m:
    regime_gap = abs(bull_m['sharpe'] - bear_m['sharpe']) / max(abs(bull_m['sharpe']), abs(bear_m['sharpe']), 0.01)
    print(f"\n  Regime Sharpe Gap: {regime_gap:.3f} (threshold: < 0.50)")
else:
    regime_gap = 999
    print(f"\n  Regime gap: CANNOT COMPUTE (insufficient trades in one regime)")

# ============================================================
# VALIDATION GATE 1: PERMUTATION TEST
# ============================================================
print("\n" + "=" * 50)
print("PERMUTATION TEST (1000 shuffles)")
print("=" * 50)

actual_sharpe = main_metrics['sharpe']
n_perms = 1000
perm_sharpes = []

rets = (trades_df['pnl'] / trades_df['pos_size']).values
for i in range(n_perms):
    shuffled = np.random.permutation(rets)
    # Randomly flip signs
    signs = np.random.choice([-1, 1], size=len(shuffled))
    shuffled = shuffled * signs

    mean_s = shuffled.mean()
    std_s = shuffled.std()
    trades_per_year = len(trades_df) / 4.5
    if std_s > 0:
        s = (mean_s * trades_per_year) / (std_s * np.sqrt(trades_per_year))
    else:
        s = 0
    perm_sharpes.append(s)

perm_p = (np.array(perm_sharpes) >= actual_sharpe).mean()
print(f"  Actual Sharpe: {actual_sharpe:.3f}")
print(f"  Permutation p-value: {perm_p:.4f} (threshold: < 0.05)")

# ============================================================
# VALIDATION GATE 2: RANDOM DIRECTION BASELINE
# ============================================================
print("\n" + "=" * 50)
print("RANDOM DIRECTION BASELINE (100 runs)")
print("=" * 50)

random_sharpes = []
for i in range(100):
    np.random.seed(i + 1000)
    rand_trades = simulate_trades(sig_df, direction_override='random')
    if len(rand_trades) > 5:
        r_rets = rand_trades['pnl'] / rand_trades['pos_size']
        tpy = len(rand_trades) / 4.5
        r_sharpe = (r_rets.mean() * tpy) / (r_rets.std() * np.sqrt(tpy)) if r_rets.std() > 0 else 0
        random_sharpes.append(r_sharpe)

np.random.seed(42)  # reset
if random_sharpes:
    avg_random = np.mean(random_sharpes)
    beats_random = actual_sharpe > np.percentile(random_sharpes, 95)
    print(f"  Strategy Sharpe: {actual_sharpe:.3f}")
    print(f"  Random baseline avg Sharpe: {avg_random:.3f}")
    print(f"  Random 95th pctile: {np.percentile(random_sharpes, 95):.3f}")
    print(f"  Beats random (95th): {'YES' if beats_random else 'NO'}")
else:
    beats_random = False
    print("  Could not compute random baseline")

# ============================================================
# VALIDATION GATE 3: LEAVE-ONE-OUT TICKER TEST
# ============================================================
print("\n" + "=" * 50)
print("LEAVE-ONE-OUT (REMOVE TOP TICKER)")
print("=" * 50)

# Find top ticker by P&L
ticker_pnl = trades_df.groupby('ticker')['pnl'].sum().sort_values(ascending=False)
print(f"  Top 5 tickers by P&L:")
for t, p in ticker_pnl.head(5).items():
    print(f"    {t}: ${p:.2f}")

if len(ticker_pnl) > 1:
    top_ticker = ticker_pnl.index[0]
    loo_trades = simulate_trades(sig_df, exclude_ticker=top_ticker)
    loo_metrics = compute_metrics(loo_trades, f"Without {top_ticker}")

    robust_after_removal = loo_metrics['sharpe'] > 0 if loo_metrics else False
    print(f"\n  Sharpe without top ticker: {loo_metrics['sharpe']:.3f}" if loo_metrics else "  Cannot compute")
    print(f"  Still profitable: {'YES' if robust_after_removal else 'NO'}")
else:
    robust_after_removal = False

# ============================================================
# PER-TICKER BREAKDOWN
# ============================================================
print("\n" + "=" * 50)
print("PER-TICKER BREAKDOWN")
print("=" * 50)

for ticker in trades_df['ticker'].unique():
    t_trades = trades_df[trades_df['ticker'] == ticker]
    t_rets = t_trades['pnl'] / t_trades['pos_size']
    wr = (t_rets > 0).mean()
    print(f"  {ticker:6s}: {len(t_trades):3d} trades, WR={wr:.0%}, PnL=${t_trades['pnl'].sum():.0f}, avg={t_rets.mean():.2%}")

# ============================================================
# VALIDATION SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("VALIDATION SUMMARY")
print("=" * 70)

gates = {
    'Sharpe > 0.5': main_metrics['sharpe'] > 0.5,
    'Perm test p < 0.05': perm_p < 0.05,
    'Regime gap < 0.50': regime_gap < 0.50,
    'MDD > -50%': main_metrics['max_drawdown'] > -0.50,
    'Beats random (95th)': beats_random,
}

for gate, passed in gates.items():
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {gate}")

gates_passed = sum(gates.values())
print(f"\n  Gates passed: {gates_passed}/5")

adversarial = {
    'Beats random baseline': beats_random,
    'Survives top-ticker removal': robust_after_removal,
}

for check, passed in adversarial.items():
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] Adversarial: {check}")

overall = gates_passed >= 4 and all(adversarial.values())
print(f"\n  OVERALL: {'VALIDATED' if overall else 'REJECTED'}")

# ============================================================
# FINAL RESULTS
# ============================================================
print("\n" + "=" * 70)
print("FINAL RESULTS — SHORT SQUEEZE DETECTOR")
print("=" * 70)
print(f"  Sharpe:         {main_metrics['sharpe']:.3f}")
print(f"  Sortino:        {main_metrics['sortino']:.3f}")
print(f"  Win Rate:       {main_metrics['win_rate']:.1%}")
print(f"  Profit Factor:  {main_metrics['profit_factor']:.3f}")
print(f"  Max Drawdown:   {main_metrics['max_drawdown']:.1%}")
print(f"  Perm p-value:   {perm_p:.4f}")
print(f"  Regime Gap:     {regime_gap:.3f}")
print(f"  Total Trades:   {main_metrics['n_trades']}")
print(f"  Final Equity:   ${main_metrics['final_equity']:.2f} (from ${INITIAL_CAPITAL})")
print(f"  Verdict:        {'VALIDATED' if overall else 'REJECTED'}")

# Save results
import json
results = {
    'strategy': 'Short Squeeze Detector',
    'metrics': main_metrics,
    'perm_p_value': float(perm_p),
    'regime_gap': float(regime_gap),
    'gates_passed': gates_passed,
    'adversarial_passed': all(adversarial.values()),
    'overall': 'VALIDATED' if overall else 'REJECTED',
}
with open('/home/jupiter/Lvl3Quant/data/short_squeeze_backtest_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved to data/short_squeeze_backtest_results.json")
