#!/usr/bin/env python3
"""
Multi-Day Streak Mean Reversion Options v1
============================================
HYPOTHESIS: Growth stocks that drop 3+ consecutive days tend to bounce.
Buy short-dated ATM calls (7-14 DTE) on the bounce setup.

Entry: After 3+ consecutive down days, if RSI(5) < 30
Exit: 5-7 day hold or at 50-100% gain on the call
Position: ATM or slightly OTM call, max $200

This exploits the behavioral tendency of growth stock investors to
overreact to short-term selloffs, creating mean reversion opportunities.

Uses Black-Scholes with realized vol for call pricing.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
import mlflow
import json
import warnings
warnings.filterwarnings('ignore')

# === UNIVERSE ===
GROWTH_UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'CRM', 'NFLX',
    'ADBE', 'PYPL', 'SQ', 'SHOP', 'SNOW', 'PLTR', 'COIN', 'HOOD', 'SOFI', 'SNAP',
    'PINS', 'TTD', 'NET', 'DDOG', 'ZS', 'CRWD', 'PANW', 'MDB', 'RBLX', 'U',
    'ENPH', 'SEDG', 'RIVN', 'LCID', 'NIO', 'MARA', 'RIOT', 'UPST', 'AFRM', 'ABNB',
    'UBER', 'LYFT', 'DIS', 'BA', 'JPM', 'GS', 'V', 'MA'
]

CAPITAL = 645.0
MAX_POSITION = 200.0
COMMISSION = 1.30  # $0.65 x 2 legs (open + close)
START_DATE = '2022-01-01'
END_DATE = '2026-07-25'


def bs_call_price(S, K, T, r, sigma):
    """Black-Scholes call price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def find_streak_signals(price_data, min_streak=3, max_rsi=30):
    """Find consecutive down-day streaks with RSI filter."""
    df = price_data.copy()
    df['ret'] = df['Close'].pct_change()
    df['down'] = df['ret'] < 0

    # Count consecutive down days
    df['streak'] = 0
    streak = 0
    for i in range(len(df)):
        if df.iloc[i]['down']:
            streak += 1
        else:
            streak = 0
        df.iloc[i, df.columns.get_loc('streak')] = streak

    # RSI(5)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(5).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(5).mean()
    rs = gain / loss
    df['rsi5'] = 100 - (100 / (1 + rs))

    # 5-day cumulative return
    df['ret_5d'] = df['Close'].pct_change(5)

    # Signal: streak >= min_streak AND RSI < max_rsi
    signals = df[(df['streak'] >= min_streak) & (df['rsi5'] < max_rsi)].copy()

    return signals


def simulate_call_trade(price_data, signal_date, signal_row, variant_params):
    """Simulate buying a call option on a streak reversion signal."""
    hold_days = variant_params.get('hold_days', 7)
    dte = variant_params.get('dte', 14)
    strike_offset = variant_params.get('strike_offset', 0.0)  # 0 = ATM, 0.02 = 2% OTM
    profit_target = variant_params.get('profit_target', 1.0)  # 100% gain
    stop_loss = variant_params.get('stop_loss', 0.50)  # 50% loss
    min_streak = variant_params.get('min_streak', 3)
    volume_filter = variant_params.get('volume_filter', False)

    signal_idx = price_data.index.get_loc(signal_date)

    # Entry next day
    entry_idx = signal_idx + 1
    if entry_idx >= len(price_data) - hold_days:
        return None

    S = price_data.iloc[entry_idx]['Close']

    # Volume filter: entry day volume > 1.5x 20-day average
    if volume_filter:
        if 'Volume' in price_data.columns:
            vol_20d = price_data.iloc[max(0,entry_idx-20):entry_idx]['Volume'].mean()
            entry_vol = price_data.iloc[entry_idx]['Volume']
            if entry_vol < vol_20d * 1.5:
                return None

    # Calculate IV from recent realized vol (elevated due to selloff)
    lookback = min(30, entry_idx)
    hist_prices = price_data.iloc[entry_idx-lookback:entry_idx]['Close']
    if len(hist_prices) < 10:
        return None
    log_returns = np.log(hist_prices / hist_prices.shift(1)).dropna()
    realized_vol = log_returns.std() * np.sqrt(252)
    if realized_vol < 0.15:
        realized_vol = 0.30

    # IV is typically 1.2-1.5x realized vol, higher after selloff
    iv = realized_vol * 1.3

    r = 0.05
    K = S * (1 + strike_offset)
    K = round(K)  # round to nearest dollar
    T_entry = dte / 252

    call_price = bs_call_price(S, K, T_entry, r, iv)

    if call_price < 0.10:
        return None

    # How many contracts can we buy?
    cost_per_contract = call_price * 100
    if cost_per_contract > MAX_POSITION:
        return None
    n_contracts = max(1, int(MAX_POSITION / cost_per_contract))
    total_cost = cost_per_contract * n_contracts

    # Simulate hold period
    exit_pnl = None
    exit_day = hold_days

    for day in range(1, hold_days + 1):
        current_idx = entry_idx + day
        if current_idx >= len(price_data):
            break

        current_price = price_data.iloc[current_idx]['Close']
        remaining_T = max((dte - day) / 252, 1/252)

        # IV decays slightly as stock recovers (vol decreases)
        # But theta is working against us
        bounce_pct = (current_price - S) / S
        # If stock recovers, IV drops; if continues falling, IV stays high
        current_iv = iv * (1 - bounce_pct * 0.5)  # Rough approximation
        current_iv = max(current_iv, realized_vol * 0.8)

        call_now = bs_call_price(current_price, K, remaining_T, r, current_iv)
        day_pnl = (call_now - call_price) * 100 * n_contracts

        day_return = day_pnl / total_cost

        # Check profit target
        if day_return >= profit_target:
            exit_pnl = day_pnl
            exit_day = day
            break

        # Check stop loss
        if day_return <= -stop_loss:
            exit_pnl = day_pnl
            exit_day = day
            break

    if exit_pnl is None:
        # Exit at end of hold period
        final_idx = min(entry_idx + hold_days, len(price_data) - 1)
        final_price = price_data.iloc[final_idx]['Close']
        remaining_T = max((dte - hold_days) / 252, 1/252)

        # Final IV
        bounce_pct = (final_price - S) / S
        final_iv = iv * (1 - bounce_pct * 0.5)
        final_iv = max(final_iv, realized_vol * 0.8)

        call_final = bs_call_price(final_price, K, remaining_T, r, final_iv)
        exit_pnl = (call_final - call_price) * 100 * n_contracts

    net_pnl = exit_pnl - COMMISSION * n_contracts

    return {
        'ticker': None,
        'signal_date': signal_date,
        'entry_date': price_data.index[entry_idx],
        'entry_price': S,
        'streak': signal_row['streak'],
        'rsi5': signal_row['rsi5'],
        'ret_5d': signal_row.get('ret_5d', 0),
        'strike': K,
        'call_price': call_price,
        'iv': iv,
        'n_contracts': n_contracts,
        'total_cost': total_cost,
        'pnl': net_pnl,
        'return_pct': net_pnl / total_cost * 100,
        'hold_days': exit_day
    }


def run_variant(variant_name, params, all_prices):
    """Run a single variant across all tickers."""
    trades = []

    for ticker in GROWTH_UNIVERSE:
        if ticker not in all_prices:
            continue

        price_data = all_prices[ticker]
        signals = find_streak_signals(
            price_data,
            min_streak=params.get('min_streak', 3),
            max_rsi=params.get('max_rsi', 30)
        )

        for date, row in signals.iterrows():
            result = simulate_call_trade(price_data, date, row, params)
            if result is not None:
                result['ticker'] = ticker
                trades.append(result)

    if len(trades) < 10:
        return {
            'variant': variant_name,
            'trades': len(trades),
            'sharpe': -999,
            'sortino': -999,
            'win_rate': 0,
            'pf': 0,
            'total_return': 0,
            'mdd': -100,
            'msg': f'Too few trades ({len(trades)})'
        }

    df = pd.DataFrame(trades).sort_values('entry_date')

    # Remove overlapping trades (max 1 position per ticker at a time)
    clean_trades = []
    ticker_exit = {}
    for _, trade in df.iterrows():
        ticker = trade['ticker']
        entry = trade['entry_date']
        if ticker in ticker_exit and entry < ticker_exit[ticker]:
            continue  # Skip overlapping
        ticker_exit[ticker] = entry + pd.Timedelta(days=trade['hold_days'])
        clean_trades.append(trade)

    df = pd.DataFrame(clean_trades)

    # Walk-forward equity curve
    equity = CAPITAL
    equity_curve = [CAPITAL]
    returns = []

    for _, trade in df.iterrows():
        # Don't risk more than available
        if trade['total_cost'] > equity:
            continue
        pnl = trade['pnl']
        ret = pnl / equity
        equity += pnl
        equity_curve.append(equity)
        returns.append(ret)

    if len(returns) < 10:
        return {
            'variant': variant_name,
            'trades': len(returns),
            'sharpe': -999,
            'sortino': -999,
            'win_rate': 0,
            'pf': 0,
            'total_return': 0,
            'mdd': -100,
            'msg': f'Too few executed trades ({len(returns)})'
        }

    returns = np.array(returns)
    equity_curve = np.array(equity_curve)

    # Metrics
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(len(returns)) if np.std(returns) > 0 else 0
    downside = returns[returns < 0]
    sortino = np.mean(returns) / np.std(downside) * np.sqrt(len(returns)) if len(downside) > 0 and np.std(downside) > 0 else 0

    wins = returns[returns > 0]
    losses = returns[returns < 0]
    win_rate = len(wins) / len(returns) * 100
    pf = abs(wins.sum() / losses.sum()) if len(losses) > 0 and losses.sum() != 0 else 999

    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peak) / peak * 100
    mdd = dd.min()

    total_return = (equity - CAPITAL) / CAPITAL * 100

    # Ticker concentration
    ticker_counts = df['ticker'].value_counts()
    top_ticker_pct = ticker_counts.iloc[0] / len(df) * 100

    return {
        'variant': variant_name,
        'trades': len(returns),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'win_rate': round(win_rate, 1),
        'pf': round(pf, 2),
        'total_return': round(total_return, 1),
        'final_equity': round(equity, 2),
        'mdd': round(mdd, 1),
        'avg_pnl': round(df['pnl'].mean(), 2),
        'top_ticker': f"{ticker_counts.index[0]} ({top_ticker_pct:.0f}%)",
        'avg_streak': round(df['streak'].mean(), 1),
        'avg_rsi': round(df['rsi5'].mean(), 1),
        'trades_df': df,
        'returns': returns
    }


def permutation_test_sharpe(returns, n_perms=1000):
    """Sign-flip permutation test."""
    real_sharpe = np.mean(returns) / np.std(returns) * np.sqrt(len(returns)) if np.std(returns) > 0 else 0
    count = sum(1 for _ in range(n_perms)
                if (lambda s: np.mean(s) / np.std(s) * np.sqrt(len(s)) if np.std(s) > 0 else 0)(
                    returns * np.random.choice([-1, 1], size=len(returns))) >= real_sharpe)
    return count / n_perms


def main():
    print("=" * 70)
    print("STREAK REVERSION OPTIONS v1")
    print("Buy calls after 3+ consecutive down days on growth stocks")
    print("=" * 70)

    # Download data
    print("\nDownloading price data...")
    all_prices = {}
    for ticker in GROWTH_UNIVERSE:
        try:
            data = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
            if len(data) > 100:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                all_prices[ticker] = data
        except:
            pass
    print(f"  Loaded {len(all_prices)} tickers")

    # Variants
    variants = {
        'A_Baseline': {
            'min_streak': 3, 'max_rsi': 30, 'hold_days': 7, 'dte': 14,
            'strike_offset': 0.0, 'profit_target': 1.0, 'stop_loss': 0.50
        },
        'B_DeepOversold': {
            'min_streak': 4, 'max_rsi': 25, 'hold_days': 7, 'dte': 14,
            'strike_offset': 0.0, 'profit_target': 1.0, 'stop_loss': 0.50
        },
        'C_QuickFlip': {
            'min_streak': 3, 'max_rsi': 30, 'hold_days': 3, 'dte': 7,
            'strike_offset': 0.0, 'profit_target': 0.50, 'stop_loss': 0.30
        },
        'D_OTM': {
            'min_streak': 3, 'max_rsi': 30, 'hold_days': 10, 'dte': 21,
            'strike_offset': 0.03, 'profit_target': 1.5, 'stop_loss': 0.50
        },
        'E_VolFilter': {
            'min_streak': 3, 'max_rsi': 30, 'hold_days': 7, 'dte': 14,
            'strike_offset': 0.0, 'profit_target': 1.0, 'stop_loss': 0.50,
            'volume_filter': True
        },
        'F_LongStreak': {
            'min_streak': 5, 'max_rsi': 35, 'hold_days': 10, 'dte': 21,
            'strike_offset': 0.0, 'profit_target': 1.5, 'stop_loss': 0.40
        }
    }

    results = []
    for name, params in variants.items():
        print(f"\nRunning {name}...")
        result = run_variant(name, params, all_prices)
        results.append(result)

        if result['sharpe'] > -900:
            print(f"  Trades: {result['trades']}, Sharpe: {result['sharpe']}, "
                  f"WR: {result['win_rate']}%, PF: {result['pf']}, "
                  f"Return: {result['total_return']}%, MDD: {result['mdd']}%")
        else:
            print(f"  {result.get('msg', 'FAILED')}")

    # Gate checks
    print("\n" + "=" * 70)
    print("GATE CHECKS")
    print("=" * 70)

    for result in results:
        if result['sharpe'] < -900:
            print(f"\n{result['variant']}: SKIP — {result.get('msg', 'insufficient trades')}")
            result['gates_passed'] = 0
            continue

        gates = 0
        details = []

        g1 = result['sharpe'] > 0.5
        gates += g1
        details.append(f"G1 Sharpe>0.5: {'PASS' if g1 else 'FAIL'} ({result['sharpe']})")

        g2 = result['win_rate'] > 40  # Lower bar for options (winners should be big)
        gates += g2
        details.append(f"G2 WR>40%: {'PASS' if g2 else 'FAIL'} ({result['win_rate']}%)")

        g3 = result['pf'] > 1.2
        gates += g3
        details.append(f"G3 PF>1.2: {'PASS' if g3 else 'FAIL'} ({result['pf']})")

        g4 = result['mdd'] > -50
        gates += g4
        details.append(f"G4 MDD>-50%: {'PASS' if g4 else 'FAIL'} ({result['mdd']}%)")

        # Permutation test
        returns = result.get('returns')
        if returns is not None and len(returns) >= 10:
            p_val = permutation_test_sharpe(returns)
            g5 = p_val < 0.05
            gates += g5
            details.append(f"G5 Perm p<0.05: {'PASS' if g5 else 'FAIL'} (p={p_val:.3f})")
        else:
            details.append("G5 Perm: SKIP")

        result['gates_passed'] = gates
        status = "PASS" if gates >= 4 else "FAIL"

        print(f"\n{result['variant']}: {gates}/5 gates — {status}")
        for d in details:
            print(f"  {d}")
        if 'top_ticker' in result:
            print(f"  Top ticker: {result['top_ticker']}")

    # MLflow
    try:
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("streak_reversion_options_v1")
        with mlflow.start_run(run_name="streak_rev_6variants"):
            for result in results:
                name = result['variant']
                mlflow.log_metric(f"{name}_sharpe", result['sharpe'])
                mlflow.log_metric(f"{name}_trades", result['trades'])
                mlflow.log_metric(f"{name}_wr", result['win_rate'])
                mlflow.log_metric(f"{name}_gates", result.get('gates_passed', 0))

            summary = {r['variant']: {k: v for k, v in r.items() if k not in ['trades_df', 'returns']}
                       for r in results}
            mlflow.log_dict(summary, "streak_reversion_v1_results.json")
    except Exception as e:
        print(f"\nMLflow error: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    valid = [r for r in results if r['sharpe'] > -900]
    if valid:
        best = max(valid, key=lambda x: x['sharpe'])
        print(f"\nBest: {best['variant']} — Sharpe {best['sharpe']}, WR {best['win_rate']}%, "
              f"${CAPITAL}→${best.get('final_equity', 'N/A')}")

    passed = [r for r in results if r.get('gates_passed', 0) >= 4]
    print(f"\n{len(passed)}/{len(results)} variants pass 4+/5 gates")

    if not passed:
        print("\nVERDICT: Streak reversion with options does NOT generate reliable alpha")
        print("Likely reasons: theta decay eats the small bounces, timing too imprecise")
    else:
        print(f"\nVERDICT: {len(passed)} variants PASS — worth adversarial testing")


if __name__ == '__main__':
    main()
