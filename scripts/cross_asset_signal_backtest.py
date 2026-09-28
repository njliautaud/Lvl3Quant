#!/usr/bin/env python3
"""
Cross-Asset Signals for Quality Stock Timing Backtest
Variants A-F using macro signals (VIX, bonds, credit, dollar, gold) to time quality stock entries.
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# ─── Parameters ───
CAPITAL = 669.0
MAX_PER_TRADE = 200.0
MAX_CONCURRENT = 3
SLIPPAGE_BPS = 2  # 2 basis points
HOLD_DAYS = 10
START = '2022-01-01'
END = '2026-07-31'

QUALITY_UNIVERSE = [
    'AAPL', 'MSFT', 'AVGO', 'JPM', 'JNJ', 'PG', 'KO', 'PEP', 'HD', 'COST',
    'UNH', 'LLY', 'V', 'MA', 'ABBV', 'MRK', 'WMT', 'AMZN', 'GOOGL', 'META'
]

MULTINATIONAL_STOCKS = ['AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', 'KO', 'PG', 'JNJ', 'MRK', 'ABBV']

MACRO_TICKERS = ['^VIX', '^TNX', 'HYG', 'UUP', 'GLD', 'SPY']

OUTPUT_PATH = '/home/jupiter/Lvl3Quant/data/cross_asset_signal_results.json'


def download_data():
    """Download all needed price data."""
    all_tickers = QUALITY_UNIVERSE + MACRO_TICKERS
    print(f"Downloading {len(all_tickers)} tickers...")
    data = yf.download(all_tickers, start=START, end=END, auto_adjust=True, progress=False)
    # yfinance returns MultiIndex columns (Price, Ticker) — extract Close
    if isinstance(data.columns, pd.MultiIndex):
        close = data['Close']
    else:
        close = data
    close = close.ffill().dropna(how='all')
    print(f"Data shape: {close.shape}, date range: {close.index[0].date()} to {close.index[-1].date()}")
    return close


def compute_rsi(series, period=14):
    """RSI calculation."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def compute_sma(series, period):
    return series.rolling(period, min_periods=period).mean()


def apply_slippage(price, direction='buy'):
    """Apply slippage to price."""
    if direction == 'buy':
        return price * (1 + SLIPPAGE_BPS / 10000)
    else:
        return price * (1 - SLIPPAGE_BPS / 10000)


def run_strategy(close, signal_func, variant_name):
    """
    Generic backtest engine.
    signal_func(date_idx, close) -> list of ticker strings to buy on that date.
    Returns trade list and equity curve.
    """
    dates = close.index
    trades = []
    positions = []  # list of {ticker, entry_date, entry_price, shares, exit_date_idx}
    equity_curve = []
    cash = CAPITAL

    for i in range(60, len(dates)):  # skip first 60 days for indicator warmup
        dt = dates[i]

        # Close expired positions
        new_positions = []
        for pos in positions:
            days_held = i - pos['entry_idx']
            if days_held >= HOLD_DAYS:
                exit_price = apply_slippage(close.loc[dates[i], pos['ticker']], 'sell')
                if np.isnan(exit_price):
                    new_positions.append(pos)
                    continue
                pnl = (exit_price - pos['entry_price']) * pos['shares']
                cash += exit_price * pos['shares']
                trades.append({
                    'ticker': pos['ticker'],
                    'entry_date': str(pos['entry_date'].date()),
                    'exit_date': str(dt.date()),
                    'entry_price': round(pos['entry_price'], 2),
                    'exit_price': round(exit_price, 2),
                    'shares': pos['shares'],
                    'pnl': round(pnl, 2),
                    'return_pct': round(pnl / (pos['entry_price'] * pos['shares']) * 100, 2)
                })
            else:
                new_positions.append(pos)
        positions = new_positions

        # Check for new signals if we have capacity
        if len(positions) < MAX_CONCURRENT:
            candidates = signal_func(i, close)
            slots = MAX_CONCURRENT - len(positions)
            # Don't buy something we already hold
            held_tickers = {p['ticker'] for p in positions}
            candidates = [t for t in candidates if t not in held_tickers]
            candidates = candidates[:slots]

            for ticker in candidates:
                price = close.loc[dt, ticker]
                if np.isnan(price) or price <= 0:
                    continue
                entry_price = apply_slippage(price, 'buy')
                alloc = min(MAX_PER_TRADE, cash)
                if alloc < 1:
                    continue
                shares = int(alloc / entry_price)
                if shares < 1:
                    continue
                cost = entry_price * shares
                cash -= cost
                positions.append({
                    'ticker': ticker,
                    'entry_date': dt,
                    'entry_idx': i,
                    'entry_price': entry_price,
                    'shares': shares
                })

        # Mark-to-market
        pos_value = 0
        for pos in positions:
            px = close.loc[dt, pos['ticker']]
            if not np.isnan(px):
                pos_value += px * pos['shares']
            else:
                pos_value += pos['entry_price'] * pos['shares']
        equity_curve.append({'date': str(dt.date()), 'equity': round(cash + pos_value, 2)})

    # Force close remaining positions
    last_dt = dates[-1]
    for pos in positions:
        exit_price = apply_slippage(close.loc[last_dt, pos['ticker']], 'sell')
        if np.isnan(exit_price):
            exit_price = pos['entry_price']
        pnl = (exit_price - pos['entry_price']) * pos['shares']
        cash += exit_price * pos['shares']
        trades.append({
            'ticker': pos['ticker'],
            'entry_date': str(pos['entry_date'].date()),
            'exit_date': str(last_dt.date()),
            'entry_price': round(pos['entry_price'], 2),
            'exit_price': round(exit_price, 2),
            'shares': pos['shares'],
            'pnl': round(pnl, 2),
            'return_pct': round(pnl / (pos['entry_price'] * pos['shares']) * 100, 2)
        })

    return trades, equity_curve


# ─── Signal Functions ───

def make_variant_a(close):
    """VIX mean reversion: VIX > 25 AND VIX drops >10% from 5-day high -> buy 3 most oversold quality stocks."""
    vix = close['^VIX']
    rsi_all = {t: compute_rsi(close[t]) for t in QUALITY_UNIVERSE}

    def signal(i, close_df):
        dates = close_df.index
        v = vix.iloc[max(0, i-5):i+1]
        if len(v) < 5:
            return []
        current_vix = vix.iloc[i]
        high_5d = v.max()
        if np.isnan(current_vix) or np.isnan(high_5d):
            return []
        if current_vix > 25 and (high_5d - current_vix) / high_5d > 0.10:
            # Get RSI for each stock, pick 3 lowest
            rsi_vals = []
            for t in QUALITY_UNIVERSE:
                r = rsi_all[t].iloc[i]
                if not np.isnan(r):
                    rsi_vals.append((t, r))
            rsi_vals.sort(key=lambda x: x[1])
            return [t for t, _ in rsi_vals[:3]]
        return []
    return signal


def make_variant_b(close):
    """Bond yield signal: 10Y yield drops >0.1% in 5 days -> buy stocks >5% below 20-SMA."""
    tnx = close['^TNX']
    sma20 = {t: compute_sma(close[t], 20) for t in QUALITY_UNIVERSE}

    def signal(i, close_df):
        dates = close_df.index
        if i < 5:
            return []
        yield_now = tnx.iloc[i]
        yield_5d_ago = tnx.iloc[i-5]
        if np.isnan(yield_now) or np.isnan(yield_5d_ago):
            return []
        # TNX is in percentage points (e.g., 4.5 = 4.5%), drop > 0.1 percentage points
        if (yield_5d_ago - yield_now) > 0.1:
            candidates = []
            for t in QUALITY_UNIVERSE:
                px = close_df.loc[dates[i], t]
                s = sma20[t].iloc[i]
                if not np.isnan(px) and not np.isnan(s) and s > 0:
                    pct_below = (s - px) / s
                    if pct_below > 0.05:
                        candidates.append((t, pct_below))
            candidates.sort(key=lambda x: x[1], reverse=True)
            return [t for t, _ in candidates[:3]]
        return []
    return signal


def make_variant_c(close):
    """Credit stress reversal: HYG up >1% over 3d after down >2% over prior 10d -> buy RSI<40 stocks."""
    hyg = close['HYG']
    rsi_all = {t: compute_rsi(close[t]) for t in QUALITY_UNIVERSE}

    def signal(i, close_df):
        if i < 13:
            return []
        # HYG 3-day return (current)
        hyg_3d = (hyg.iloc[i] / hyg.iloc[i-3] - 1) * 100
        # HYG 10-day return ending 3 days ago
        hyg_10d_prior = (hyg.iloc[i-3] / hyg.iloc[i-13] - 1) * 100
        if np.isnan(hyg_3d) or np.isnan(hyg_10d_prior):
            return []
        if hyg_3d > 1.0 and hyg_10d_prior < -2.0:
            candidates = []
            for t in QUALITY_UNIVERSE:
                r = rsi_all[t].iloc[i]
                if not np.isnan(r) and r < 40:
                    candidates.append((t, r))
            candidates.sort(key=lambda x: x[1])
            return [t for t, _ in candidates[:3]]
        return []
    return signal


def make_variant_d(close):
    """Dollar weakness: UUP falls >1% over 5 days -> buy multinational stocks."""
    uup = close['UUP']
    rsi_all = {t: compute_rsi(close[t]) for t in MULTINATIONAL_STOCKS}

    def signal(i, close_df):
        if i < 5:
            return []
        uup_5d = (uup.iloc[i] / uup.iloc[i-5] - 1) * 100
        if np.isnan(uup_5d):
            return []
        if uup_5d < -1.0:
            # Buy most oversold multinationals
            candidates = []
            for t in MULTINATIONAL_STOCKS:
                r = rsi_all[t].iloc[i]
                if not np.isnan(r):
                    candidates.append((t, r))
            candidates.sort(key=lambda x: x[1])
            return [t for t, _ in candidates[:3]]
        return []
    return signal


def make_variant_e(close):
    """Gold divergence: GLD up >2% over 10d + quality stocks haven't fallen -> avoid.
       GLD down >2% over 10d + quality avg dipped >3% -> buy dippers."""
    gld = close['GLD']
    rsi_all = {t: compute_rsi(close[t]) for t in QUALITY_UNIVERSE}

    def signal(i, close_df):
        dates = close_df.index
        if i < 10:
            return []
        gld_10d = (gld.iloc[i] / gld.iloc[i-10] - 1) * 100
        if np.isnan(gld_10d):
            return []

        # Compute quality stock avg return over 10d
        q_rets = []
        for t in QUALITY_UNIVERSE:
            px_now = close_df.loc[dates[i], t]
            px_10d = close_df.loc[dates[i-10], t]
            if not np.isnan(px_now) and not np.isnan(px_10d) and px_10d > 0:
                q_rets.append((px_now / px_10d - 1) * 100)
        if not q_rets:
            return []
        avg_q_ret = np.mean(q_rets)

        # Buy signal: GLD falling, quality dipped
        if gld_10d < -2.0 and avg_q_ret < -3.0:
            candidates = []
            for t in QUALITY_UNIVERSE:
                px_now = close_df.loc[dates[i], t]
                px_10d = close_df.loc[dates[i-10], t]
                if not np.isnan(px_now) and not np.isnan(px_10d) and px_10d > 0:
                    ret = (px_now / px_10d - 1) * 100
                    if ret < -3.0:
                        candidates.append((t, ret))
            candidates.sort(key=lambda x: x[1])
            return [t for t, _ in candidates[:3]]
        return []
    return signal


def make_variant_f(close):
    """Composite: require 2 of 3 (A, B, C) macro signals active -> buy dipped quality stocks."""
    vix = close['^VIX']
    tnx = close['^TNX']
    hyg = close['HYG']
    rsi_all = {t: compute_rsi(close[t]) for t in QUALITY_UNIVERSE}
    sma20 = {t: compute_sma(close[t], 20) for t in QUALITY_UNIVERSE}

    def signal(i, close_df):
        dates = close_df.index
        if i < 13:
            return []

        score = 0

        # A: VIX mean reversion
        v = vix.iloc[max(0, i-5):i+1]
        current_vix = vix.iloc[i]
        high_5d = v.max()
        if not np.isnan(current_vix) and not np.isnan(high_5d) and current_vix > 25:
            if high_5d > 0 and (high_5d - current_vix) / high_5d > 0.10:
                score += 1

        # B: Bond yield drop
        yield_now = tnx.iloc[i]
        yield_5d = tnx.iloc[i-5]
        if not np.isnan(yield_now) and not np.isnan(yield_5d):
            if (yield_5d - yield_now) > 0.1:
                score += 1

        # C: Credit stress reversal
        hyg_3d = (hyg.iloc[i] / hyg.iloc[i-3] - 1) * 100 if i >= 3 else 0
        hyg_10d_prior = (hyg.iloc[i-3] / hyg.iloc[i-13] - 1) * 100 if i >= 13 else 0
        if not np.isnan(hyg_3d) and not np.isnan(hyg_10d_prior):
            if hyg_3d > 1.0 and hyg_10d_prior < -2.0:
                score += 1

        if score >= 2:
            candidates = []
            for t in QUALITY_UNIVERSE:
                r = rsi_all[t].iloc[i]
                px = close_df.loc[dates[i], t]
                s = sma20[t].iloc[i]
                if not np.isnan(r) and not np.isnan(px) and not np.isnan(s) and s > 0:
                    below_sma = (s - px) / s
                    if r < 40 or below_sma > 0.05:
                        candidates.append((t, r))
            candidates.sort(key=lambda x: x[1])
            return [t for t, _ in candidates[:3]]
        return []
    return signal


# ─── Evaluation ───

def compute_metrics(trades, equity_curve, close):
    """Compute all required metrics."""
    if len(trades) == 0:
        return {
            'sharpe': 0, 'sortino': 0, 'win_rate': 0, 'profit_factor': 0,
            'max_drawdown': 0, 'total_return': 0, 'num_trades': 0,
            'bull_sharpe': 0, 'bear_sharpe': 0, 'regime_gap': 0,
            'permutation_p': 1.0, 'five_gate': 'FAIL'
        }

    # Basic trade stats
    pnls = [t['pnl'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_rate = len(wins) / len(pnls) if pnls else 0
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.001
    profit_factor = gross_profit / gross_loss

    # Returns from equity curve
    equities = [e['equity'] for e in equity_curve]
    eq_series = pd.Series(equities)
    daily_returns = eq_series.pct_change().dropna()

    if len(daily_returns) < 2:
        sharpe = sortino = 0
    else:
        ann_factor = np.sqrt(252)
        mean_ret = daily_returns.mean()
        std_ret = daily_returns.std()
        sharpe = (mean_ret / std_ret * ann_factor) if std_ret > 0 else 0

        downside = daily_returns[daily_returns < 0]
        down_std = downside.std() if len(downside) > 1 else std_ret
        sortino = (mean_ret / down_std * ann_factor) if down_std > 0 else 0

    # Drawdown
    eq_arr = np.array(equities)
    peak = np.maximum.accumulate(eq_arr)
    dd = (eq_arr - peak) / peak
    max_drawdown = float(dd.min())

    # Total return
    total_return = (equities[-1] / CAPITAL - 1) * 100 if equities else 0

    # ─── Regime analysis ───
    spy = close['SPY']
    spy_sma200 = compute_sma(spy, 200)
    eq_dates = [e['date'] for e in equity_curve]
    eq_df = pd.DataFrame({'date': eq_dates, 'equity': equities})
    eq_df['date'] = pd.to_datetime(eq_df['date'])
    eq_df = eq_df.set_index('date')
    eq_df['return'] = eq_df['equity'].pct_change()

    bull_returns = []
    bear_returns = []
    for dt in eq_df.index:
        if dt in spy.index and dt in spy_sma200.index:
            s = spy.loc[dt]
            sma = spy_sma200.loc[dt]
            if not np.isnan(s) and not np.isnan(sma):
                ret = eq_df.loc[dt, 'return']
                if not np.isnan(ret):
                    if s > sma:
                        bull_returns.append(ret)
                    else:
                        bear_returns.append(ret)

    ann = np.sqrt(252)
    if len(bull_returns) > 5:
        br = np.array(bull_returns)
        bull_sharpe = float(br.mean() / br.std() * ann) if br.std() > 0 else 0
    else:
        bull_sharpe = 0
    if len(bear_returns) > 5:
        br2 = np.array(bear_returns)
        bear_sharpe = float(br2.mean() / br2.std() * ann) if br2.std() > 0 else 0
    else:
        bear_sharpe = 0

    max_abs = max(abs(bull_sharpe), abs(bear_sharpe), 0.001)
    regime_gap = abs(bull_sharpe - bear_sharpe) / max_abs

    # ─── Permutation test ───
    actual_total = sum(pnls)
    n_perms = 1000
    count_better = 0
    trade_dates_idx = list(range(60, len(close.index) - HOLD_DAYS))
    rng = np.random.RandomState(42)
    for _ in range(n_perms):
        shuffled_pnl = 0
        for t in trades:
            rand_idx = rng.choice(trade_dates_idx)
            rand_date = close.index[rand_idx]
            ticker = t['ticker']
            if ticker in close.columns:
                entry_px = close.loc[rand_date, ticker]
                exit_idx = min(rand_idx + HOLD_DAYS, len(close.index) - 1)
                exit_px = close.iloc[exit_idx][ticker] if ticker in close.columns else entry_px
                if not np.isnan(entry_px) and not np.isnan(exit_px) and entry_px > 0:
                    shares = int(MAX_PER_TRADE / entry_px)
                    if shares > 0:
                        shuffled_pnl += (exit_px - entry_px) * shares
        if shuffled_pnl >= actual_total:
            count_better += 1
    permutation_p = count_better / n_perms

    # ─── 5-Gate Validation ───
    gate_1 = sharpe > 0.5
    gate_2 = permutation_p < 0.05
    gate_3 = regime_gap < 0.5
    gate_4 = max_drawdown > -0.50
    gate_5 = len(trades) >= 20

    all_pass = gate_1 and gate_2 and gate_3 and gate_4 and gate_5
    five_gate = 'PASS' if all_pass else 'FAIL'
    gate_details = {
        'sharpe_gt_0.5': bool(gate_1),
        'perm_p_lt_0.05': bool(gate_2),
        'regime_gap_lt_0.5': bool(gate_3),
        'max_dd_gt_neg50pct': bool(gate_4),
        'min_20_trades': bool(gate_5)
    }

    return {
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'win_rate': round(win_rate, 3),
        'profit_factor': round(profit_factor, 3),
        'max_drawdown': round(max_drawdown, 4),
        'total_return': round(total_return, 2),
        'num_trades': len(trades),
        'total_pnl': round(sum(pnls), 2),
        'bull_sharpe': round(bull_sharpe, 3),
        'bear_sharpe': round(bear_sharpe, 3),
        'regime_gap': round(regime_gap, 3),
        'permutation_p': round(permutation_p, 3),
        'five_gate': five_gate,
        'gate_details': gate_details
    }


def main():
    close = download_data()

    variants = {
        'A_vix_mean_reversion': make_variant_a(close),
        'B_bond_yield_signal': make_variant_b(close),
        'C_credit_stress_reversal': make_variant_c(close),
        'D_dollar_weakness': make_variant_d(close),
        'E_gold_divergence': make_variant_e(close),
        'F_composite_macro': make_variant_f(close),
    }

    results = {}
    for name, sig_func in variants.items():
        print(f"\n{'='*60}")
        print(f"Running variant: {name}")
        print(f"{'='*60}")
        trades, equity_curve = run_strategy(close, sig_func, name)
        metrics = compute_metrics(trades, equity_curve, close)
        results[name] = metrics

        print(f"  Trades: {metrics['num_trades']}")
        print(f"  Total PnL: ${metrics['total_pnl']}")
        print(f"  Total Return: {metrics['total_return']}%")
        print(f"  Sharpe: {metrics['sharpe']}")
        print(f"  Sortino: {metrics['sortino']}")
        print(f"  Win Rate: {metrics['win_rate']}")
        print(f"  Profit Factor: {metrics['profit_factor']}")
        print(f"  Max DD: {metrics['max_drawdown']}")
        print(f"  Bull Sharpe: {metrics['bull_sharpe']} | Bear Sharpe: {metrics['bear_sharpe']}")
        print(f"  Regime Gap: {metrics['regime_gap']}")
        print(f"  Permutation p: {metrics['permutation_p']}")
        print(f"  5-Gate: {metrics['five_gate']}")
        for gate, val in metrics['gate_details'].items():
            status = 'PASS' if val else 'FAIL'
            print(f"    {gate}: {status}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    passed = [k for k, v in results.items() if v['five_gate'] == 'PASS']
    failed = [k for k, v in results.items() if v['five_gate'] == 'FAIL']
    print(f"PASSED 5-Gate: {passed if passed else 'None'}")
    print(f"FAILED 5-Gate: {failed if failed else 'None'}")

    # Best variant by Sharpe
    best = max(results.items(), key=lambda x: x[1]['sharpe'])
    print(f"Best Sharpe: {best[0]} ({best[1]['sharpe']})")

    # Save results
    output = {
        'metadata': {
            'strategy': 'Cross-Asset Signals for Quality Stock Timing',
            'period': f'{START} to {END}',
            'capital': CAPITAL,
            'max_per_trade': MAX_PER_TRADE,
            'max_concurrent': MAX_CONCURRENT,
            'slippage_bps': SLIPPAGE_BPS,
            'hold_days': HOLD_DAYS,
            'universe': QUALITY_UNIVERSE,
            'run_date': str(datetime.now().date())
        },
        'variants': results
    }

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
