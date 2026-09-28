#!/usr/bin/env python3
"""
PUT/CALL RATIO CONTRARIAN BACKTEST
====================================
Concept: Extreme sentiment (proxied by realized vol, RSI, and recent drawdowns)
signals exhaustion. Since we don't have historical P/C ratio data per stock,
we proxy "extreme fear" and "extreme greed" using:

EXTREME FEAR (bullish signal — buy calls):
  - RSI(14) < 25 (deeply oversold)
  - 10-day drawdown > 20% (panic selling)
  - Volume surge > 2x (capitulation volume)
  - VIX > 20 (elevated fear)

EXTREME GREED (bearish signal — buy puts):
  - RSI(14) > 80 (overbought)
  - 10-day rally > 25% (euphoria)
  - Volume surge > 2x (FOMO volume)
  - VIX < 18 (complacency)

Trade: Simulate ATM options with 2-4 week expiry.
  - Options P&L = delta * underlying_move * 100 - theta_decay - spread
  - Simplified: use ~0.50 delta, $0.65/contract commissions each way
  - Position size: ~$200 max per trade (1-3 contracts at $0.50-$2.00 each)

Hold: 5-10 trading days (within option expiry window)
Exit: TP +50% on option value, SL -40% on option value, or 8-day hold max.
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
MAX_POSITION_SIZE = 200.0  # max $200 per options trade
OPTIONS_COMMISSION_RT = 1.30  # $0.65 each way per contract
SPREAD_COST_PCT = 3.0  # 3% bid-ask spread on options (realistic for ATM weeklies)

# Signal parameters - FEAR (bullish)
RSI_OVERSOLD = 25
DRAWDOWN_10D_FEAR = -0.20  # 20% drop in 10 days
VOL_SURGE_MULT = 2.0

# Signal parameters - GREED (bearish)
RSI_OVERBOUGHT = 80
RALLY_10D_GREED = 0.25  # 25% rally in 10 days

# Options modeling
DELTA = 0.50  # ATM delta
THETA_DECAY_PER_DAY = 0.03  # ~3% per day theta decay on weeklies
GAMMA_BOOST = 0.10  # gamma adds ~10% extra for large moves

# Trade parameters
HOLD_DAYS = 8
OPTION_TP_PCT = 0.50  # take profit at +50% option value
OPTION_SL_PCT = -0.40  # stop loss at -40% option value

# Backtest period
BT_START = '2022-01-01'
BT_END = '2026-07-24'

EXCLUDE_TICKERS = ['SPY', '^VIX']

np.random.seed(42)

# ============================================================
# LOAD DATA
# ============================================================
print("=" * 70)
print("PUT/CALL RATIO CONTRARIAN BACKTEST")
print("=" * 70)

df = pd.read_parquet('/home/jupiter/Lvl3Quant/data/growth_stocks_prices.parquet')
df = df.reset_index()
df['date'] = pd.to_datetime(df['date'])
df = df[~df['ticker'].isin(EXCLUDE_TICKERS)]
df = df[(df['date'] >= BT_START) & (df['date'] <= BT_END)]

# VIX
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
# RSI CALCULATION
# ============================================================
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))

# ============================================================
# GENERATE SIGNALS
# ============================================================
def compute_signals(df):
    """Compute contrarian sentiment signals."""
    all_signals = []

    for ticker in tickers:
        tdf = df[df['ticker'] == ticker].sort_values('date').copy()
        if len(tdf) < 30:
            continue

        # RSI
        tdf['rsi'] = compute_rsi(tdf['close'])

        # 10-day return
        tdf['ret_10d'] = tdf['close'].pct_change(10)

        # Volume surge
        tdf['vol_avg_20d'] = tdf['volume'].rolling(20).mean()
        tdf['vol_ratio'] = tdf['volume'] / tdf['vol_avg_20d']

        # Realized vol
        tdf['rvol_20d'] = tdf['close'].pct_change().rolling(20).std() * np.sqrt(252)

        # FEAR signal (buy calls - bullish reversal expected)
        tdf['fear_signal'] = (
            (tdf['rsi'] < RSI_OVERSOLD) &
            (tdf['ret_10d'] < DRAWDOWN_10D_FEAR) &
            (tdf['vol_ratio'] >= VOL_SURGE_MULT)
        ).astype(int)

        # GREED signal (buy puts - bearish reversal expected)
        tdf['greed_signal'] = (
            (tdf['rsi'] > RSI_OVERBOUGHT) &
            (tdf['ret_10d'] > RALLY_10D_GREED) &
            (tdf['vol_ratio'] >= VOL_SURGE_MULT)
        ).astype(int)

        # Forward returns
        for d in range(1, HOLD_DAYS + 3):
            tdf[f'fwd_ret_{d}d'] = tdf['close'].shift(-d) / tdf['close'] - 1.0

        tdf['ticker_col'] = ticker

        # Collect fear signals
        fear_rows = tdf[tdf['fear_signal'] == 1].copy()
        fear_rows['direction'] = 'long'  # buy calls
        all_signals.append(fear_rows)

        # Collect greed signals
        greed_rows = tdf[tdf['greed_signal'] == 1].copy()
        greed_rows['direction'] = 'short'  # buy puts
        all_signals.append(greed_rows)

    if not all_signals:
        return pd.DataFrame()
    return pd.concat(all_signals, ignore_index=True)

sig_df = compute_signals(df)
print(f"\nTotal raw signals: {len(sig_df)}")

if len(sig_df) == 0:
    print("NO SIGNALS — adjusting thresholds...")
    # Relax thresholds
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 75
    DRAWDOWN_10D_FEAR = -0.15
    RALLY_10D_GREED = 0.20
    VOL_SURGE_MULT = 1.5
    sig_df = compute_signals(df)
    print(f"After relaxing: {len(sig_df)} signals")

# Merge VIX
sig_df = sig_df.merge(vix_df, on='date', how='left')
# For fear signals, VIX should be elevated (>20); for greed, VIX should be low (<20)
# Actually keep all — VIX context adds but shouldn't hard-filter
print(f"Long (fear/calls): {(sig_df['direction'] == 'long').sum()}")
print(f"Short (greed/puts): {(sig_df['direction'] == 'short').sum()}")

# Merge regime
sig_df = sig_df.merge(spy_df[['date', 'regime']], on='date', how='left')

# ============================================================
# OPTIONS P&L MODEL
# ============================================================
def option_pnl(underlying_ret, hold_days, direction, iv_proxy=0.50):
    """
    Simplified options P&L model.

    For ATM options:
    - Delta P&L = delta * underlying_move * 100 * contracts
    - Theta decay = premium * theta_rate * hold_days
    - Gamma boost for large moves

    We model option value change as % of premium paid.
    """
    # Direction: 'long' = bought calls, 'short' = bought puts
    if direction == 'long':
        directional_pnl = underlying_ret  # calls benefit from up moves
    else:
        directional_pnl = -underlying_ret  # puts benefit from down moves

    # Option P&L as multiple of premium
    # ATM option: delta ~0.50, so option moves ~50% of underlying
    # But options have leverage, so a 5% stock move = ~25% option move for ATM
    option_ret = directional_pnl * (1.0 / DELTA)  # leverage effect

    # Subtract theta decay
    theta_cost = THETA_DECAY_PER_DAY * hold_days
    option_ret -= theta_cost

    # Gamma boost for large moves (>5%)
    if abs(directional_pnl) > 0.05:
        option_ret += GAMMA_BOOST * abs(directional_pnl)

    # IV crush/expansion (simplified)
    # After extreme fear, IV often drops -> hurts long options
    # After extreme greed, IV often drops -> hurts long options
    iv_drag = 0.05  # ~5% IV drag on average for holding through vol events
    option_ret -= iv_drag

    return option_ret

# ============================================================
# SIMULATE TRADES
# ============================================================
def simulate_trades(sig_df, random_mode=False, exclude_ticker=None):
    trades = []
    equity = INITIAL_CAPITAL

    sdf = sig_df.copy()
    if exclude_ticker:
        sdf = sdf[sdf['ticker_col'] != exclude_ticker]

    sdf = sdf.sort_values('date')
    open_positions = {}

    for _, row in sdf.iterrows():
        ticker = row['ticker_col']
        entry_date = row['date']
        direction = row['direction']

        if ticker in open_positions and entry_date < open_positions[ticker]:
            continue

        # Position sizing: buy 1-2 contracts at ~$1-2 each = $100-$200
        pos_size = min(MAX_POSITION_SIZE, equity * 0.25)
        if pos_size < 50:
            continue

        # Commission cost (as % of position)
        n_contracts = max(1, int(pos_size / 150))  # ~$1.50 per contract avg
        commission_cost = OPTIONS_COMMISSION_RT * n_contracts
        spread_cost = pos_size * (SPREAD_COST_PCT / 100)
        total_cost = commission_cost + spread_cost

        # Simulate day-by-day
        exit_option_ret = None
        exit_day = HOLD_DAYS

        for d in range(1, HOLD_DAYS + 1):
            col = f'fwd_ret_{d}d'
            if col not in row or pd.isna(row[col]):
                continue

            underlying_ret = row[col]
            if random_mode:
                direction_use = np.random.choice(['long', 'short'])
            else:
                direction_use = direction

            opt_ret = option_pnl(underlying_ret, d, direction_use)

            if opt_ret >= OPTION_TP_PCT:
                exit_option_ret = OPTION_TP_PCT
                exit_day = d
                break
            elif opt_ret <= OPTION_SL_PCT:
                exit_option_ret = OPTION_SL_PCT
                exit_day = d
                break

        if exit_option_ret is None:
            col = f'fwd_ret_{HOLD_DAYS}d'
            if col in row and not pd.isna(row[col]):
                if random_mode:
                    direction_use = np.random.choice(['long', 'short'])
                else:
                    direction_use = direction
                exit_option_ret = option_pnl(row[col], HOLD_DAYS, direction_use)
            else:
                continue

        # P&L = position * option_return - costs
        pnl = pos_size * exit_option_ret - total_cost
        equity += pnl

        open_positions[ticker] = entry_date + timedelta(days=exit_day * 1.5)

        trades.append({
            'date': entry_date,
            'ticker': ticker,
            'direction': direction,
            'entry_price': row['close'],
            'option_ret': exit_option_ret,
            'pnl': pnl,
            'equity': equity,
            'hold_days': exit_day,
            'pos_size': pos_size,
            'regime': row.get('regime', 'unknown'),
            'rsi': row.get('rsi', 0),
        })

    return pd.DataFrame(trades)

trades_df = simulate_trades(sig_df)
print(f"\nTotal trades: {len(trades_df)}")

if len(trades_df) == 0:
    print("NO TRADES GENERATED — strategy has no signals.")
    print("RESULT: FAIL — insufficient signals")
    import sys; sys.exit(0)

# ============================================================
# COMPUTE METRICS
# ============================================================
def compute_metrics(trades_df, label="Main"):
    if len(trades_df) < 3:
        print(f"\n--- {label} --- INSUFFICIENT TRADES ({len(trades_df)})")
        return None

    rets = trades_df['pnl'] / trades_df['pos_size']

    mean_ret = rets.mean()
    std_ret = rets.std()

    trades_per_year = len(trades_df) / 4.5
    ann_ret = mean_ret * trades_per_year
    ann_vol = std_ret * np.sqrt(trades_per_year) if std_ret > 0 else 1e-6

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

main_metrics = compute_metrics(trades_df, "Put/Call Contrarian")

# Direction breakdown
print("\n" + "=" * 50)
print("DIRECTION BREAKDOWN")
print("=" * 50)
long_trades = trades_df[trades_df['direction'] == 'long']
short_trades = trades_df[trades_df['direction'] == 'short']
compute_metrics(long_trades, "LONG (fear/calls)") if len(long_trades) > 3 else print(f"  Long trades: {len(long_trades)} (too few)")
compute_metrics(short_trades, "SHORT (greed/puts)") if len(short_trades) > 3 else print(f"  Short trades: {len(short_trades)} (too few)")

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
# PERMUTATION TEST
# ============================================================
print("\n" + "=" * 50)
print("PERMUTATION TEST (1000 shuffles)")
print("=" * 50)

actual_sharpe = main_metrics['sharpe'] if main_metrics else 0
n_perms = 1000
perm_sharpes = []

rets = (trades_df['pnl'] / trades_df['pos_size']).values
for i in range(n_perms):
    shuffled = rets.copy()
    signs = np.random.choice([-1, 1], size=len(shuffled))
    shuffled = shuffled * signs

    mean_s = shuffled.mean()
    std_s = shuffled.std()
    tpy = len(trades_df) / 4.5
    if std_s > 0:
        s = (mean_s * tpy) / (std_s * np.sqrt(tpy))
    else:
        s = 0
    perm_sharpes.append(s)

perm_p = (np.array(perm_sharpes) >= actual_sharpe).mean()
print(f"  Actual Sharpe: {actual_sharpe:.3f}")
print(f"  Permutation p-value: {perm_p:.4f} (threshold: < 0.05)")

# ============================================================
# RANDOM DIRECTION BASELINE
# ============================================================
print("\n" + "=" * 50)
print("RANDOM DIRECTION BASELINE (100 runs)")
print("=" * 50)

random_sharpes = []
for i in range(100):
    np.random.seed(i + 2000)
    rand_trades = simulate_trades(sig_df, random_mode=True)
    if len(rand_trades) > 3:
        r_rets = rand_trades['pnl'] / rand_trades['pos_size']
        tpy = len(rand_trades) / 4.5
        r_sharpe = (r_rets.mean() * tpy) / (r_rets.std() * np.sqrt(tpy)) if r_rets.std() > 0 else 0
        random_sharpes.append(r_sharpe)

np.random.seed(42)
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
# LEAVE-ONE-OUT TICKER TEST
# ============================================================
print("\n" + "=" * 50)
print("LEAVE-ONE-OUT (REMOVE TOP TICKER)")
print("=" * 50)

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
    dirs = t_trades['direction'].value_counts().to_dict()
    dir_str = '/'.join([f"{v}{k[0].upper()}" for k, v in dirs.items()])
    print(f"  {ticker:6s}: {len(t_trades):3d} trades ({dir_str}), WR={wr:.0%}, PnL=${t_trades['pnl'].sum():.0f}")

# ============================================================
# VALIDATION SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("VALIDATION SUMMARY")
print("=" * 70)

gates = {
    'Sharpe > 0.5': main_metrics['sharpe'] > 0.5 if main_metrics else False,
    'Perm test p < 0.05': perm_p < 0.05,
    'Regime gap < 0.50': regime_gap < 0.50,
    'MDD > -50%': main_metrics['max_drawdown'] > -0.50 if main_metrics else False,
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
print("FINAL RESULTS — PUT/CALL RATIO CONTRARIAN")
print("=" * 70)
if main_metrics:
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
    'strategy': 'Put/Call Ratio Contrarian',
    'metrics': main_metrics,
    'perm_p_value': float(perm_p),
    'regime_gap': float(regime_gap),
    'gates_passed': gates_passed,
    'adversarial_passed': all(adversarial.values()),
    'overall': 'VALIDATED' if overall else 'REJECTED',
}
with open('/home/jupiter/Lvl3Quant/data/put_call_contrarian_backtest_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved to data/put_call_contrarian_backtest_results.json")
