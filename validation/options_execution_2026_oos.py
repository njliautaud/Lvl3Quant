"""
TRUE Out-of-Sample Validation: Evolved Options Execution Strategy on 2026 Data
==============================================================================
The strategy was evolved by AVO on 2022H1-2025H2 walk-forward folds.
This script tests on 2026 YTD data that evolution NEVER saw.

Strategy: Buy call options on sector ETFs when RSI < 35 (oversold dip-buy).
Black-Scholes pricing, 30-DTE, 0.35 delta target.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from datetime import datetime, timedelta
import csv
import sys
import os

# ===========================================================================
# STRATEGY PARAMETERS (exact copy from evolved strategy.py v25)
# ===========================================================================
ACCOUNT_SIZE = 650.0
MAX_CONCURRENT = 4
MAX_PER_TRADE_PCT = 0.22
SLIPPAGE_PCT = 0.001

RSI_PERIOD = 14
RSI_ENTRY_THRESHOLD = 35
RSI_DEEP_OVERSOLD = 30
MIN_VOLUME_RATIO = 0.8

PREFERRED_TICKERS = ['XLE', 'XLU', 'XLK', 'XLI', 'XLF', 'XLB', 'XLRE']

OPTION_DTE = 30
OPTION_DELTA_TARGET = 0.35
RISK_FREE_RATE = 0.05

HOLD_DAYS_MAX = 5
TP_PCT = 0.43
HIGHVOL_TP_PCT = 0.25
SL_PCT = -0.17
TRAILING_ACTIVATE_PCT = 0.08
TRAILING_GIVEBACK_PCT = 0.20
HIGHVOL_GIVEBACK_PCT = 0.35

SECTOR_HOLD_DAYS = {
    'XLK': 4, 'XLF': 4, 'XLE': 5, 'XLI': 5,
    'XLU': 10, 'XLB': 9, 'XLRE': 11,
}

HIGHVOL_VIX = 25
EXTREME_VIX = 35
HIGHVOL_SL_PCT = -0.12
HIGHVOL_HOLD_REDUCTION = 3
HIGHVOL_TRAILING_ACTIVATE_PCT = 0.10
HIGHVOL_IV_THRESHOLD = 0.35

REL_STRENGTH_LOOKBACK = 10
REL_STRENGTH_MIN = -0.04

WINNER_HOLD_EXTENSION = 2

IDLE_DAYS_THRESHOLD = 4
RSI_RELAXED_THRESHOLD = 38

HIGHVOL_EARLY_EXIT_DAY = 1
HIGHVOL_EARLY_EXIT_LOSS = -0.05


# ===========================================================================
# CORE FUNCTIONS (exact copy from strategy.py)
# ===========================================================================

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def bs_call_price(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def estimate_iv(prices, window=20):
    log_ret = np.log(prices / prices.shift(1))
    rv = log_ret.rolling(window).std() * np.sqrt(252)
    return rv * 1.15


def generate_signals(prices_df, spy_series, vix_series):
    """Generate buy signals — exact logic from strategy.py."""
    sectors = [c for c in prices_df.columns if c in PREFERRED_TICKERS]
    signals = pd.DataFrame(False, index=prices_df.index, columns=sectors)

    spy_ret_10d = spy_series.pct_change(REL_STRENGTH_LOOKBACK)

    ticker_data = {}
    for ticker in sectors:
        px = prices_df[ticker].dropna()
        if len(px) < RSI_PERIOD + 5:
            continue

        rsi = compute_rsi(px, RSI_PERIOD)
        vol = px.pct_change().abs()
        avg_vol = vol.rolling(20).mean()

        oversold = rsi < RSI_ENTRY_THRESHOLD
        relaxed_oversold = rsi < RSI_RELAXED_THRESHOLD
        deep_oversold = rsi < RSI_DEEP_OVERSOLD
        vol_ok = vol > avg_vol * MIN_VOLUME_RATIO

        bounce = px > px.shift(1)

        vix_aligned = vix_series.reindex(px.index, method='ffill')
        vix_prev = vix_series.shift(1).reindex(px.index, method='ffill')
        vix_prev2 = vix_series.shift(2).reindex(px.index, method='ffill')

        is_lowvol = vix_aligned < HIGHVOL_VIX
        is_highvol = (vix_aligned >= HIGHVOL_VIX) & (vix_aligned < EXTREME_VIX)
        vix_declining_2d = (vix_aligned < vix_prev) & (vix_prev < vix_prev2)

        sector_ret_10d = px.pct_change(REL_STRENGTH_LOOKBACK)
        spy_ret_aligned = spy_ret_10d.reindex(px.index, method='ffill')
        has_rel_strength = (sector_ret_10d - spy_ret_aligned) > REL_STRENGTH_MIN

        lowvol_signal = oversold & vol_ok & is_lowvol
        relaxed_lowvol_signal = relaxed_oversold & vol_ok & is_lowvol
        highvol_signal = (deep_oversold & vol_ok & bounce & vix_declining_2d
                          & is_highvol & has_rel_strength)

        standard_signal = lowvol_signal | highvol_signal
        relaxed_signal = relaxed_lowvol_signal | highvol_signal

        signals.loc[standard_signal.index, ticker] = standard_signal
        ticker_data[ticker] = {'standard': standard_signal, 'relaxed': relaxed_signal}

    # Idle capital relaxation pass
    any_signal = signals.any(axis=1)
    signal_count = any_signal.astype(int).rolling(IDLE_DAYS_THRESHOLD, min_periods=IDLE_DAYS_THRESHOLD).sum()
    is_idle = signal_count == 0

    for ticker, data in ticker_data.items():
        relaxed_only = data['relaxed'] & ~data['standard']
        idle_relaxed = relaxed_only & is_idle.reindex(relaxed_only.index, fill_value=False)
        combined = signals[ticker] | idle_relaxed.reindex(signals.index, fill_value=False)
        signals[ticker] = combined

    return signals


def should_exit(position, current_price, current_date):
    """Exit logic — exact copy from strategy.py."""
    days_held = np.busday_count(
        np.datetime64(position['entry_date'], 'D'),
        np.datetime64(current_date, 'D')
    )

    entry_underlying = position['entry_underlying']
    option_entry = position['option_entry_price']
    peak_option = position['peak_option_price']
    ticker = position['ticker']

    T_remaining = max((OPTION_DTE - days_held) / 365.0, 0.001)
    iv = position['iv']
    strike = position['strike']

    current_option = bs_call_price(current_price, strike, T_remaining,
                                   RISK_FREE_RATE, iv)
    option_return = ((current_option - option_entry) / option_entry
                     if option_entry > 0 else 0)

    if current_option > peak_option:
        position['peak_option_price'] = current_option
        peak_option = current_option

    peak_return = ((peak_option - option_entry) / option_entry
                   if option_entry > 0 else 0)

    is_highvol_entry = iv > HIGHVOL_IV_THRESHOLD

    sector_hold = SECTOR_HOLD_DAYS.get(ticker, HOLD_DAYS_MAX)
    if is_highvol_entry:
        sector_hold = max(2, sector_hold - HIGHVOL_HOLD_REDUCTION)

    effective_hold = sector_hold
    if option_return > 0:
        effective_hold = sector_hold + WINNER_HOLD_EXTENSION
    if days_held >= effective_hold:
        return True, current_option, 'time_stop'

    tp = HIGHVOL_TP_PCT if is_highvol_entry else TP_PCT
    if option_return >= tp:
        return True, current_option, 'take_profit'

    sl = HIGHVOL_SL_PCT if is_highvol_entry else SL_PCT
    if option_return <= sl:
        return True, current_option, 'stop_loss'

    # Failed bounce early exit — low vol
    if not is_highvol_entry:
        early_exit_day = 2 if sector_hold >= 7 else 1
        if days_held >= early_exit_day:
            if current_price < entry_underlying and option_return <= -0.12:
                return True, current_option, 'failed_bounce'

    # Failed bounce early exit — high vol
    if is_highvol_entry:
        if days_held >= HIGHVOL_EARLY_EXIT_DAY:
            if current_price < entry_underlying and option_return <= HIGHVOL_EARLY_EXIT_LOSS:
                return True, current_option, 'hv_failed_bounce'

    # Trailing stop
    trail_activate = (HIGHVOL_TRAILING_ACTIVATE_PCT if is_highvol_entry
                      else TRAILING_ACTIVATE_PCT)
    if peak_return >= trail_activate:
        giveback = peak_return - option_return
        gb_pct = HIGHVOL_GIVEBACK_PCT if is_highvol_entry else TRAILING_GIVEBACK_PCT
        max_giveback = peak_return * gb_pct
        if giveback >= max_giveback:
            return True, current_option, 'trailing_stop'

    return False, current_option, None


# ===========================================================================
# DOWNLOAD DATA
# ===========================================================================

def download_data():
    """Download data from 2025-09-01 (warmup) through today."""
    tickers = PREFERRED_TICKERS + ['SPY', '^VIX']
    start = '2025-09-01'
    end = datetime.now().strftime('%Y-%m-%d')

    print(f"Downloading {len(tickers)} tickers from {start} to {end}...")
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    # Handle multi-level columns
    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Close']
    else:
        prices = data

    # Make index tz-naive for clean date operations
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)

    spy = prices['SPY'].copy()
    vix = prices['^VIX'].copy()
    sector_prices = prices[PREFERRED_TICKERS].copy()

    print(f"  Data shape: {sector_prices.shape}")
    print(f"  Date range: {sector_prices.index[0].date()} to {sector_prices.index[-1].date()}")
    print(f"  2026 trading days: {(sector_prices.index >= '2026-01-01').sum()}")

    return sector_prices, spy, vix


# ===========================================================================
# BACKTEST ENGINE
# ===========================================================================

def run_backtest(sector_prices, spy, vix, benchmark_mode=False):
    """
    Run the full strategy simulation.
    benchmark_mode: if True, buy SPY calls on same RSI dips instead.
    """
    label = "SPY BENCHMARK" if benchmark_mode else "STRATEGY"

    # Generate signals
    if benchmark_mode:
        # For benchmark: generate signals on SPY using same RSI logic
        spy_df = pd.DataFrame({'SPY': spy})
        spy_rsi = compute_rsi(spy, RSI_PERIOD)
        spy_vol = spy.pct_change().abs()
        spy_avg_vol = spy_vol.rolling(20).mean()
        spy_signals = pd.DataFrame(False, index=spy.index, columns=['SPY'])
        spy_signals['SPY'] = (spy_rsi < RSI_ENTRY_THRESHOLD) & (spy_vol > spy_avg_vol * MIN_VOLUME_RATIO)
        signals = spy_signals
        prices_for_sim = spy_df
    else:
        signals = generate_signals(sector_prices, spy, vix)
        prices_for_sim = sector_prices

    # Compute IV for all tickers
    iv_data = {}
    for col in prices_for_sim.columns:
        iv_data[col] = estimate_iv(prices_for_sim[col])

    # OOS start date
    oos_start = pd.Timestamp('2026-01-02')

    account = ACCOUNT_SIZE
    positions = []  # active positions
    trade_log = []  # completed trades
    equity_curve = []

    all_dates = prices_for_sim.index.sort_values()

    for i, date in enumerate(all_dates):
        date_ts = pd.Timestamp(date)

        # Check exits first
        closed_today = []
        for pos in positions:
            ticker = pos['ticker']
            price_col = 'SPY' if benchmark_mode else ticker
            if price_col not in prices_for_sim.columns:
                continue
            current_price = prices_for_sim.loc[date, price_col]
            if pd.isna(current_price):
                continue

            do_exit, exit_option_price, exit_reason = should_exit(pos, current_price, date)
            if do_exit:
                contracts = pos['contracts']
                pnl = (exit_option_price - pos['option_entry_price']) * 100 * contracts
                pos['exit_date'] = date
                pos['exit_price'] = current_price
                pos['exit_option_price'] = exit_option_price
                pos['pnl'] = pnl
                pos['exit_reason'] = exit_reason
                pos['option_return'] = (exit_option_price - pos['option_entry_price']) / pos['option_entry_price'] if pos['option_entry_price'] > 0 else 0
                account += pos['capital_used'] + pnl  # return capital + P&L
                if date_ts >= oos_start:
                    trade_log.append(pos.copy())
                closed_today.append(id(pos))

        positions = [p for p in positions if id(p) not in closed_today]

        # Check entries
        if date_ts >= oos_start or True:  # always run signal generation, only log trades from OOS
            for ticker in signals.columns:
                if len(positions) >= MAX_CONCURRENT:
                    break

                # Check if we already have a position in this ticker
                if any(p['ticker'] == ticker for p in positions):
                    continue

                try:
                    sig = signals.loc[date, ticker]
                except (KeyError, IndexError):
                    continue

                if not sig:
                    continue

                price_col = 'SPY' if benchmark_mode else ticker
                current_price = prices_for_sim.loc[date, price_col]
                if pd.isna(current_price):
                    continue

                # Get IV
                try:
                    iv = iv_data[price_col].loc[date]
                except (KeyError, IndexError):
                    iv = 0.25
                if pd.isna(iv) or iv <= 0:
                    iv = 0.25

                # Option pricing
                strike = current_price * 1.03  # ~3% OTM for 0.35 delta
                T = OPTION_DTE / 365.0
                option_price = bs_call_price(current_price, strike, T, RISK_FREE_RATE, iv)

                if option_price <= 0.01:
                    continue

                # Apply slippage
                option_price_adj = option_price * (1 + SLIPPAGE_PCT)

                # Position sizing
                trade_budget = account * MAX_PER_TRADE_PCT
                cost_per_contract = option_price_adj * 100  # options are per 100 shares
                contracts = max(1, int(trade_budget / cost_per_contract))
                capital_used = contracts * cost_per_contract

                if capital_used > account * 0.95:  # leave some buffer
                    contracts = max(1, int((account * 0.95) / cost_per_contract))
                    capital_used = contracts * cost_per_contract

                if capital_used > account:
                    continue

                account -= capital_used

                vix_val = vix.reindex([date], method='ffill').iloc[0] if date in vix.index or True else 20
                try:
                    vix_val = vix.loc[:date].iloc[-1]
                except:
                    vix_val = 20

                pos = {
                    'ticker': ticker,
                    'entry_date': date,
                    'entry_underlying': current_price,
                    'strike': strike,
                    'iv': iv,
                    'option_entry_price': option_price_adj,
                    'peak_option_price': option_price_adj,
                    'contracts': contracts,
                    'capital_used': capital_used,
                    'vix_at_entry': vix_val,
                    'regime': 'high_vol' if vix_val >= HIGHVOL_VIX else 'low_vol',
                }
                positions.append(pos)

        # Track equity at end of day (account cash + mark-to-market of positions)
        if date_ts >= oos_start:
            mtm = account
            for pos in positions:
                ticker = pos['ticker']
                price_col = 'SPY' if benchmark_mode else ticker
                current_price = prices_for_sim.loc[date, price_col]
                if pd.isna(current_price):
                    mtm += pos['capital_used']
                    continue
                days_held = np.busday_count(
                    np.datetime64(pos['entry_date'], 'D'),
                    np.datetime64(date, 'D')
                )
                T_remaining = max((OPTION_DTE - days_held) / 365.0, 0.001)
                current_option = bs_call_price(current_price, pos['strike'], T_remaining,
                                               RISK_FREE_RATE, pos['iv'])
                pos_value = current_option * 100 * pos['contracts']
                mtm += pos_value

            equity_curve.append({'date': date, 'equity': mtm})

    # Force-close any remaining positions at last date
    last_date = all_dates[-1]
    for pos in positions:
        ticker = pos['ticker']
        price_col = 'SPY' if benchmark_mode else ticker
        current_price = prices_for_sim.loc[last_date, price_col]
        days_held = np.busday_count(
            np.datetime64(pos['entry_date'], 'D'),
            np.datetime64(last_date, 'D')
        )
        T_remaining = max((OPTION_DTE - days_held) / 365.0, 0.001)
        exit_option = bs_call_price(current_price, pos['strike'], T_remaining,
                                    RISK_FREE_RATE, pos['iv'])
        pnl = (exit_option - pos['option_entry_price']) * 100 * pos['contracts']
        pos['exit_date'] = last_date
        pos['exit_price'] = current_price
        pos['exit_option_price'] = exit_option
        pos['pnl'] = pnl
        pos['exit_reason'] = 'force_close'
        pos['option_return'] = (exit_option - pos['option_entry_price']) / pos['option_entry_price'] if pos['option_entry_price'] > 0 else 0
        if pd.Timestamp(last_date) >= oos_start:
            trade_log.append(pos.copy())

    return trade_log, equity_curve, label


# ===========================================================================
# METRICS & REPORTING
# ===========================================================================

def compute_metrics(trade_log, equity_curve, label):
    """Compute and print all required metrics."""
    print(f"\n{'='*70}")
    print(f"  {label} — 2026 Out-of-Sample Results")
    print(f"{'='*70}")

    if not trade_log:
        print("  NO TRADES TAKEN")
        return

    df = pd.DataFrame(trade_log)
    eq = pd.DataFrame(equity_curve)

    n_trades = len(df)
    winners = df[df['pnl'] > 0]
    losers = df[df['pnl'] <= 0]
    win_rate = len(winners) / n_trades * 100 if n_trades > 0 else 0

    total_pnl = df['pnl'].sum()
    final_equity = eq['equity'].iloc[-1] if len(eq) > 0 else ACCOUNT_SIZE + total_pnl
    compound_return = (final_equity / ACCOUNT_SIZE - 1) * 100

    # CAGR
    if len(eq) > 1:
        days = (eq['date'].iloc[-1] - eq['date'].iloc[0]).days
        years = max(days / 365.25, 0.01)
        cagr = ((final_equity / ACCOUNT_SIZE) ** (1 / years) - 1) * 100
    else:
        cagr = compound_return

    avg_profit = df['pnl'].mean()
    avg_profit_pct = df['option_return'].mean() * 100

    gross_profit = winners['pnl'].sum() if len(winners) > 0 else 0
    gross_loss = abs(losers['pnl'].sum()) if len(losers) > 0 else 0.01
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Sharpe from equity curve
    if len(eq) > 5:
        eq_series = eq.set_index('date')['equity']
        daily_returns = eq_series.pct_change().dropna()
        if daily_returns.std() > 0:
            sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        else:
            sharpe = 0
        sortino_downside = daily_returns[daily_returns < 0].std()
        if sortino_downside > 0:
            sortino = daily_returns.mean() / sortino_downside * np.sqrt(252)
        else:
            sortino = float('inf') if daily_returns.mean() > 0 else 0
    else:
        sharpe = 0
        sortino = 0

    # Max drawdown
    if len(eq) > 0:
        eq_arr = eq['equity'].values
        peak = np.maximum.accumulate(eq_arr)
        dd = (eq_arr - peak) / peak
        max_dd = dd.min() * 100
    else:
        max_dd = 0

    print(f"\n  HEADLINE METRICS:")
    print(f"  {'Compound Return:':<30} {compound_return:+.1f}%")
    print(f"  {'Final Equity:':<30} ${final_equity:,.2f} (from ${ACCOUNT_SIZE:.0f})")
    print(f"  {'CAGR:':<30} {cagr:+.1f}%")
    print(f"  {'Number of Trades:':<30} {n_trades}")
    print(f"  {'Win Rate:':<30} {win_rate:.1f}% ({len(winners)}W / {len(losers)}L)")
    print(f"  {'Avg Profit per Trade:':<30} ${avg_profit:+.2f} ({avg_profit_pct:+.1f}%)")
    print(f"  {'Profit Factor:':<30} {profit_factor:.2f}x (for every $1 lost, earned ${profit_factor:.2f})")
    print(f"  {'Sharpe Ratio:':<30} {sharpe:.2f}", end='')
    if sharpe > 2:
        print("  (excellent risk-adjusted returns)")
    elif sharpe > 1:
        print("  (good risk-adjusted returns)")
    elif sharpe > 0.5:
        print("  (moderate risk-adjusted returns)")
    elif sharpe > 0:
        print("  (weak risk-adjusted returns)")
    else:
        print("  (negative risk-adjusted returns)")
    print(f"  {'Sortino Ratio:':<30} {sortino:.2f}")
    print(f"  {'Max Drawdown:':<30} {max_dd:.1f}%")

    # --- Stratify by VIX Regime ---
    print(f"\n  BY VIX REGIME:")
    for regime in ['low_vol', 'high_vol']:
        sub = df[df['regime'] == regime]
        if len(sub) == 0:
            print(f"    {regime}: no trades")
            continue
        wr = len(sub[sub['pnl'] > 0]) / len(sub) * 100
        avg_pnl = sub['pnl'].mean()
        gp = sub[sub['pnl'] > 0]['pnl'].sum() if len(sub[sub['pnl'] > 0]) > 0 else 0
        gl = abs(sub[sub['pnl'] <= 0]['pnl'].sum()) if len(sub[sub['pnl'] <= 0]) > 0 else 0.01
        pf = gp / gl if gl > 0 else float('inf')
        print(f"    {regime}: {len(sub)} trades, WR {wr:.0f}%, avg ${avg_pnl:+.2f}, PF {pf:.2f}x")

    # --- Stratify by Sector ---
    print(f"\n  BY SECTOR:")
    for ticker in sorted(df['ticker'].unique()):
        sub = df[df['ticker'] == ticker]
        wr = len(sub[sub['pnl'] > 0]) / len(sub) * 100 if len(sub) > 0 else 0
        avg_pnl = sub['pnl'].mean()
        print(f"    {ticker}: {len(sub)} trades, WR {wr:.0f}%, avg ${avg_pnl:+.2f}, total ${sub['pnl'].sum():+.2f}")

    # --- Stratify by Month ---
    print(f"\n  BY MONTH:")
    df['month'] = pd.to_datetime(df['entry_date']).dt.to_period('M')
    for month in sorted(df['month'].unique()):
        sub = df[df['month'] == month]
        wr = len(sub[sub['pnl'] > 0]) / len(sub) * 100 if len(sub) > 0 else 0
        print(f"    {month}: {len(sub)} trades, WR {wr:.0f}%, total ${sub['pnl'].sum():+.2f}")

    # --- Exit Reason Breakdown ---
    print(f"\n  BY EXIT REASON:")
    for reason in df['exit_reason'].value_counts().index:
        sub = df[df['exit_reason'] == reason]
        wr = len(sub[sub['pnl'] > 0]) / len(sub) * 100
        print(f"    {reason}: {len(sub)} trades, WR {wr:.0f}%, avg ${sub['pnl'].mean():+.2f}")

    return {
        'compound_return': compound_return,
        'cagr': cagr,
        'n_trades': n_trades,
        'win_rate': win_rate,
        'avg_profit': avg_profit,
        'profit_factor': profit_factor,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_dd': max_dd,
        'final_equity': final_equity,
    }


def save_trade_log(trade_log, filename):
    """Save trade log to CSV."""
    if not trade_log:
        return
    df = pd.DataFrame(trade_log)
    cols_to_save = ['ticker', 'entry_date', 'exit_date', 'entry_underlying', 'strike',
                    'iv', 'option_entry_price', 'exit_option_price', 'contracts',
                    'capital_used', 'pnl', 'option_return', 'exit_reason',
                    'regime', 'vix_at_entry']
    cols_present = [c for c in cols_to_save if c in df.columns]
    df_out = df[cols_present].copy()
    # Format dates
    for dc in ['entry_date', 'exit_date']:
        if dc in df_out.columns:
            df_out[dc] = pd.to_datetime(df_out[dc]).dt.strftime('%Y-%m-%d')
    df_out.to_csv(filename, index=False)
    print(f"\n  Trade log saved to: {filename}")


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("  TRUE OUT-OF-SAMPLE VALIDATION: Evolved Options Strategy on 2026 Data")
    print("  Strategy evolved on 2022H1-2025H2. Testing on 2026 (unseen data).")
    print("=" * 70)

    # Download data
    sector_prices, spy, vix = download_data()

    # Run strategy backtest
    print("\n--- Running STRATEGY backtest ---")
    strat_trades, strat_equity, strat_label = run_backtest(sector_prices, spy, vix, benchmark_mode=False)
    strat_metrics = compute_metrics(strat_trades, strat_equity, strat_label)

    # Save trade log
    save_trade_log(strat_trades, '/home/jupiter/Lvl3Quant/validation/options_2026_oos_trades.csv')

    # Run SPY benchmark
    print("\n--- Running SPY BENCHMARK backtest ---")
    bench_trades, bench_equity, bench_label = run_backtest(sector_prices, spy, vix, benchmark_mode=True)
    bench_metrics = compute_metrics(bench_trades, bench_equity, bench_label)

    save_trade_log(bench_trades, '/home/jupiter/Lvl3Quant/validation/spy_benchmark_2026_trades.csv')

    # Comparison
    if strat_metrics and bench_metrics:
        print(f"\n{'='*70}")
        print(f"  STRATEGY vs SPY BENCHMARK COMPARISON")
        print(f"{'='*70}")
        print(f"  {'Metric':<25} {'Strategy':>15} {'SPY Bench':>15} {'Delta':>12}")
        print(f"  {'-'*67}")
        for key, label_str in [('compound_return', 'Compound Return %'),
                                ('n_trades', 'Trades'),
                                ('win_rate', 'Win Rate %'),
                                ('avg_profit', 'Avg Profit $'),
                                ('profit_factor', 'Profit Factor'),
                                ('sharpe', 'Sharpe'),
                                ('max_dd', 'Max DD %')]:
            sv = strat_metrics[key]
            bv = bench_metrics[key]
            delta = sv - bv
            fmt = '.1f' if key in ('compound_return', 'win_rate', 'max_dd') else '.2f'
            if key == 'n_trades':
                print(f"  {label_str:<25} {sv:>15.0f} {bv:>15.0f} {delta:>+12.0f}")
            else:
                print(f"  {label_str:<25} {sv:>15{fmt}} {bv:>15{fmt}} {delta:>+12{fmt}}")

    print(f"\n{'='*70}")
    print(f"  VALIDATION COMPLETE")
    print(f"{'='*70}")
