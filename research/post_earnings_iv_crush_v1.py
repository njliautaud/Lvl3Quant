#!/usr/bin/env python3
"""
Post-Earnings IV Crush Income Strategy v1
==========================================
HYPOTHESIS: After earnings, IV is still elevated (IV crush hasn't fully normalized).
Sell put credit spreads or iron condors AFTER earnings on stocks that reported well.
Capture the IV normalization over 7-14 days post-earnings.

Entry: 1 day after earnings report (if stock didn't gap down >5%)
Exit: 7-14 days later or at 50% profit target
Position: Bull put spread (OTM puts) — bullish bias on stocks that beat earnings

This is the REVERSE of IV run-up (which buys pre-earnings).
Here we SELL post-earnings when IV is still high.

Uses Black-Scholes pricing with realistic IV crush modeling.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm, permutation_test
from datetime import datetime, timedelta
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
COMMISSION_PER_LEG = 0.65  # Robinhood options commission per contract per leg
START_DATE = '2022-01-01'
END_DATE = '2026-07-25'


def bs_put_price(S, K, T, r, sigma):
    """Black-Scholes put price."""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def get_earnings_dates(ticker, start, end):
    """Get historical earnings dates from yfinance."""
    try:
        stock = yf.Ticker(ticker)
        # Try to get earnings dates
        earnings = stock.earnings_dates
        if earnings is not None and len(earnings) > 0:
            dates = earnings.index.tz_localize(None) if earnings.index.tz else earnings.index
            mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
            return sorted(dates[mask].tolist())
    except:
        pass

    # Fallback: quarterly approximation (every ~90 days aligned to typical quarters)
    # This is a rough proxy — real strategy would use actual dates
    quarterly_months = [1, 4, 7, 10]  # Approximate earnings months
    dates = []
    current = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    while current <= end_ts:
        for m in quarterly_months:
            # Typical earnings: last week of Jan/Apr/Jul/Oct
            d = pd.Timestamp(year=current.year, month=m, day=25)
            if start <= str(d.date()) <= end:
                dates.append(d)
        current += pd.DateOffset(years=1)
    return sorted(list(set(dates)))


def simulate_iv_crush_trade(price_data, earnings_date, variant_params):
    """
    Simulate a post-earnings IV crush trade.

    Strategy: Sell bull put spread after earnings.
    - Short put at delta ~0.20 (OTM)
    - Long put at delta ~0.05 (further OTM, for protection)
    - Collect credit from IV being elevated post-earnings
    - IV normalizes over hold period, spread decays
    """
    entry_offset = variant_params.get('entry_offset', 1)  # days after earnings
    hold_days = variant_params.get('hold_days', 10)
    short_delta = variant_params.get('short_delta', 0.20)
    spread_width_pct = variant_params.get('spread_width_pct', 0.05)
    profit_target_pct = variant_params.get('profit_target_pct', 0.50)
    min_gap_pct = variant_params.get('min_gap_pct', -5.0)  # min gap to enter (reject big drops)
    iv_premium = variant_params.get('iv_premium', 1.5)  # IV is 1.5x normal post-earnings

    # Find entry date (1 day after earnings)
    earnings_idx = price_data.index.get_indexer([earnings_date], method='ffill')[0]
    if earnings_idx < 1 or earnings_idx + entry_offset >= len(price_data):
        return None

    entry_idx = earnings_idx + entry_offset
    if entry_idx >= len(price_data):
        return None

    # Check earnings gap
    pre_price = price_data.iloc[earnings_idx - 1]['Close']
    post_price = price_data.iloc[entry_idx]['Close']
    gap_pct = (post_price - pre_price) / pre_price * 100

    # Skip if gapped down too much (we want stable/up stocks)
    if gap_pct < min_gap_pct:
        return None

    S = post_price
    r = 0.05  # risk-free rate

    # Calculate historical volatility as base
    lookback = min(60, entry_idx)
    hist_prices = price_data.iloc[entry_idx-lookback:entry_idx]['Close']
    if len(hist_prices) < 20:
        return None
    log_returns = np.log(hist_prices / hist_prices.shift(1)).dropna()
    base_iv = log_returns.std() * np.sqrt(252)
    if base_iv < 0.1:
        base_iv = 0.3

    # Post-earnings IV is elevated (IV crush hasn't fully happened)
    entry_iv = base_iv * iv_premium

    # Strike prices
    T_entry = hold_days / 252
    # Short put: ~delta 0.20 OTM
    short_strike = S * (1 - short_delta * entry_iv * np.sqrt(T_entry))
    # Long put: further OTM
    long_strike = short_strike * (1 - spread_width_pct)

    # Round strikes to nearest $1
    short_strike = round(short_strike)
    long_strike = round(long_strike)

    if long_strike <= 0 or short_strike <= long_strike:
        return None

    # Calculate entry prices
    short_put_entry = bs_put_price(S, short_strike, T_entry, r, entry_iv)
    long_put_entry = bs_put_price(S, long_strike, T_entry, r, entry_iv)

    credit_received = short_put_entry - long_put_entry
    if credit_received < 0.05:  # minimum credit
        return None

    max_loss = (short_strike - long_strike) - credit_received
    if max_loss <= 0:
        return None

    # Check if trade fits budget
    collateral = (short_strike - long_strike) * 100  # per contract
    if collateral > MAX_POSITION:
        return None

    # Simulate daily mark-to-market
    exit_idx = min(entry_idx + hold_days, len(price_data) - 1)

    best_pnl = 0
    exit_pnl = None

    for day in range(1, exit_idx - entry_idx + 1):
        current_idx = entry_idx + day
        current_price = price_data.iloc[current_idx]['Close']
        remaining_T = max((hold_days - day) / 252, 1/252)

        # IV decays back toward base over the hold period
        decay_factor = iv_premium - (iv_premium - 1.0) * (day / hold_days)
        current_iv = base_iv * decay_factor

        short_put_now = bs_put_price(current_price, short_strike, remaining_T, r, current_iv)
        long_put_now = bs_put_price(current_price, long_strike, remaining_T, r, current_iv)

        spread_cost_now = short_put_now - long_put_now
        day_pnl = (credit_received - spread_cost_now) * 100  # per contract

        if day_pnl > best_pnl:
            best_pnl = day_pnl

        # Early exit at profit target
        if day_pnl >= credit_received * 100 * profit_target_pct:
            exit_pnl = day_pnl
            break

    if exit_pnl is None:
        # Hold to expiry — check if ITM
        final_price = price_data.iloc[exit_idx]['Close']
        if final_price >= short_strike:
            # Both puts expire worthless, keep full credit
            exit_pnl = credit_received * 100
        elif final_price >= long_strike:
            # Short put ITM, long put OTM
            exit_pnl = (credit_received - (short_strike - final_price)) * 100
        else:
            # Both ITM — max loss
            exit_pnl = -max_loss * 100

    # Subtract commissions (4 legs: open short, open long, close short, close long)
    total_commission = COMMISSION_PER_LEG * 4
    net_pnl = exit_pnl - total_commission

    return {
        'ticker': None,  # filled by caller
        'earnings_date': earnings_date,
        'entry_date': price_data.index[entry_idx],
        'entry_price': S,
        'gap_pct': gap_pct,
        'short_strike': short_strike,
        'long_strike': long_strike,
        'credit': credit_received,
        'entry_iv': entry_iv,
        'base_iv': base_iv,
        'pnl': net_pnl,
        'collateral': collateral,
        'return_pct': net_pnl / collateral * 100
    }


def run_variant(variant_name, params, all_prices, all_earnings):
    """Run a single variant across all tickers."""
    trades = []

    for ticker in GROWTH_UNIVERSE:
        if ticker not in all_prices or ticker not in all_earnings:
            continue

        price_data = all_prices[ticker]
        earnings_dates = all_earnings[ticker]

        for ed in earnings_dates:
            result = simulate_iv_crush_trade(price_data, ed, params)
            if result is not None:
                result['ticker'] = ticker
                trades.append(result)

    if len(trades) < 5:
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

    # Walk-forward equity curve
    equity = CAPITAL
    equity_curve = [CAPITAL]
    returns = []

    for _, trade in df.iterrows():
        pnl = trade['pnl']
        ret = pnl / equity
        equity += pnl
        equity_curve.append(equity)
        returns.append(ret)

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

    # Max drawdown
    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peak) / peak * 100
    mdd = dd.min()

    total_return = (equity - CAPITAL) / CAPITAL * 100

    # Regime analysis (simplified — use SPY as proxy)
    # Will be done in adversarial if variant passes

    return {
        'variant': variant_name,
        'trades': len(trades),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'win_rate': round(win_rate, 1),
        'pf': round(pf, 2),
        'total_return': round(total_return, 1),
        'final_equity': round(equity, 2),
        'mdd': round(mdd, 1),
        'avg_pnl': round(df['pnl'].mean(), 2),
        'avg_credit': round(df['credit'].mean(), 4),
        'avg_gap': round(df['gap_pct'].mean(), 1),
        'trades_df': df
    }


def permutation_test_sharpe(returns, n_perms=1000):
    """Test if Sharpe is significantly different from random direction."""
    real_sharpe = np.mean(returns) / np.std(returns) * np.sqrt(len(returns)) if np.std(returns) > 0 else 0

    count_better = 0
    for _ in range(n_perms):
        # Random sign flip
        signs = np.random.choice([-1, 1], size=len(returns))
        shuffled = returns * signs
        rand_sharpe = np.mean(shuffled) / np.std(shuffled) * np.sqrt(len(shuffled)) if np.std(shuffled) > 0 else 0
        if rand_sharpe >= real_sharpe:
            count_better += 1

    return count_better / n_perms


def regime_analysis(trades_df, spy_data):
    """Analyze performance in different market regimes."""
    results = {}
    for _, trade in trades_df.iterrows():
        entry_date = trade['entry_date']
        # Find SPY return over 20 days before entry
        spy_idx = spy_data.index.get_indexer([entry_date], method='ffill')[0]
        if spy_idx < 20:
            regime = 'unknown'
        else:
            spy_ret_20d = (spy_data.iloc[spy_idx]['Close'] / spy_data.iloc[spy_idx-20]['Close'] - 1) * 100
            if spy_ret_20d > 2:
                regime = 'bull'
            elif spy_ret_20d < -2:
                regime = 'bear'
            else:
                regime = 'flat'

        if regime not in results:
            results[regime] = []
        results[regime].append(trade['pnl'])

    regime_stats = {}
    for regime, pnls in results.items():
        pnls = np.array(pnls)
        regime_stats[regime] = {
            'count': len(pnls),
            'avg_pnl': round(np.mean(pnls), 2),
            'win_rate': round(len(pnls[pnls > 0]) / len(pnls) * 100, 1) if len(pnls) > 0 else 0
        }

    return regime_stats


def main():
    print("=" * 70)
    print("POST-EARNINGS IV CRUSH INCOME v1")
    print("Sell put spreads after earnings when IV is still elevated")
    print("=" * 70)

    # Download price data
    print("\nDownloading price data...")
    all_prices = {}
    all_earnings = {}

    for ticker in GROWTH_UNIVERSE:
        try:
            data = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
            if len(data) > 100:
                # Flatten multi-level columns if present
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                all_prices[ticker] = data
                all_earnings[ticker] = get_earnings_dates(ticker, START_DATE, END_DATE)
        except Exception as e:
            print(f"  {ticker}: error - {e}")

    print(f"  Loaded {len(all_prices)} tickers with price data")

    # Also get SPY for regime analysis
    spy_data = yf.download('SPY', start=START_DATE, end=END_DATE, progress=False)
    if isinstance(spy_data.columns, pd.MultiIndex):
        spy_data.columns = spy_data.columns.get_level_values(0)

    # Define variants
    variants = {
        'A_Baseline': {
            'entry_offset': 1, 'hold_days': 10, 'short_delta': 0.20,
            'spread_width_pct': 0.05, 'profit_target_pct': 0.50,
            'min_gap_pct': -5.0, 'iv_premium': 1.5
        },
        'B_Conservative': {
            'entry_offset': 1, 'hold_days': 14, 'short_delta': 0.15,
            'spread_width_pct': 0.03, 'profit_target_pct': 0.40,
            'min_gap_pct': -3.0, 'iv_premium': 1.5
        },
        'C_Aggressive': {
            'entry_offset': 1, 'hold_days': 7, 'short_delta': 0.25,
            'spread_width_pct': 0.07, 'profit_target_pct': 0.60,
            'min_gap_pct': -8.0, 'iv_premium': 1.8
        },
        'D_BeatOnly': {
            'entry_offset': 1, 'hold_days': 10, 'short_delta': 0.20,
            'spread_width_pct': 0.05, 'profit_target_pct': 0.50,
            'min_gap_pct': 0.0, 'iv_premium': 1.5  # Only enter on positive gap (beat)
        },
        'E_HighIV': {
            'entry_offset': 1, 'hold_days': 10, 'short_delta': 0.20,
            'spread_width_pct': 0.05, 'profit_target_pct': 0.50,
            'min_gap_pct': -5.0, 'iv_premium': 2.0  # Higher IV = more credit
        },
        'F_Delayed': {
            'entry_offset': 3, 'hold_days': 14, 'short_delta': 0.20,
            'spread_width_pct': 0.05, 'profit_target_pct': 0.50,
            'min_gap_pct': -5.0, 'iv_premium': 1.3  # Enter 3 days after, IV partially normalized
        }
    }

    # Run all variants
    results = []
    for name, params in variants.items():
        print(f"\nRunning {name}...")
        result = run_variant(name, params, all_prices, all_earnings)
        results.append(result)

        print(f"  Trades: {result['trades']}, Sharpe: {result['sharpe']}, "
              f"WR: {result['win_rate']}%, PF: {result['pf']}, "
              f"Return: {result['total_return']}%, MDD: {result['mdd']}%")

    # Gate checks for each variant
    print("\n" + "=" * 70)
    print("GATE CHECKS")
    print("=" * 70)

    for result in results:
        name = result['variant']
        trades_df = result.get('trades_df')

        gates_passed = 0
        total_gates = 5
        gate_details = []

        # Gate 1: Sharpe > 0.5
        g1 = result['sharpe'] > 0.5
        gates_passed += g1
        gate_details.append(f"G1 Sharpe>0.5: {'PASS' if g1 else 'FAIL'} ({result['sharpe']})")

        # Gate 2: Win Rate > 50%
        g2 = result['win_rate'] > 50
        gates_passed += g2
        gate_details.append(f"G2 WR>50%: {'PASS' if g2 else 'FAIL'} ({result['win_rate']}%)")

        # Gate 3: Profit Factor > 1.2
        g3 = result['pf'] > 1.2
        gates_passed += g3
        gate_details.append(f"G3 PF>1.2: {'PASS' if g3 else 'FAIL'} ({result['pf']})")

        # Gate 4: MDD > -30%
        g4 = result['mdd'] > -30
        gates_passed += g4
        gate_details.append(f"G4 MDD>-30%: {'PASS' if g4 else 'FAIL'} ({result['mdd']}%)")

        # Gate 5: Permutation test p < 0.05
        if trades_df is not None and len(trades_df) >= 5:
            returns = trades_df['return_pct'].values / 100
            p_val = permutation_test_sharpe(returns)
            g5 = p_val < 0.05
            gates_passed += g5
            gate_details.append(f"G5 Perm p<0.05: {'PASS' if g5 else 'FAIL'} (p={p_val:.3f})")
        else:
            gate_details.append("G5 Perm: SKIP (too few trades)")

        # Regime analysis
        regime_str = ""
        if trades_df is not None and len(trades_df) >= 5:
            regime_stats = regime_analysis(trades_df, spy_data)
            regime_str = f" | Regimes: {json.dumps(regime_stats)}"

        status = "PASS" if gates_passed >= 4 else "FAIL"
        print(f"\n{name}: {gates_passed}/{total_gates} gates — {status}")
        for g in gate_details:
            print(f"  {g}")
        if regime_str:
            print(f"  {regime_str}")

        result['gates_passed'] = gates_passed
        result['status'] = status

    # Log to MLflow
    try:
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("post_earnings_iv_crush_v1")

        with mlflow.start_run(run_name="iv_crush_6variants"):
            for result in results:
                name = result['variant']
                mlflow.log_metric(f"{name}_sharpe", result['sharpe'])
                mlflow.log_metric(f"{name}_win_rate", result['win_rate'])
                mlflow.log_metric(f"{name}_pf", result['pf'])
                mlflow.log_metric(f"{name}_mdd", result['mdd'])
                mlflow.log_metric(f"{name}_trades", result['trades'])
                mlflow.log_metric(f"{name}_gates", result.get('gates_passed', 0))

            # Save summary
            summary = {name: {k: v for k, v in r.items() if k != 'trades_df'}
                       for r, name in zip(results, [r['variant'] for r in results])}
            mlflow.log_dict(summary, "iv_crush_v1_results.json")
    except Exception as e:
        print(f"\nMLflow logging error: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    best = max(results, key=lambda x: x['sharpe'])
    print(f"\nBest variant: {best['variant']}")
    print(f"  Sharpe: {best['sharpe']}, Sortino: {best['sortino']}")
    print(f"  Trades: {best['trades']}, WR: {best['win_rate']}%")
    print(f"  PF: {best['pf']}, MDD: {best['mdd']}%")
    print(f"  ${CAPITAL} -> ${best.get('final_equity', 'N/A')}")

    passed = [r for r in results if r.get('gates_passed', 0) >= 4]
    print(f"\n{len(passed)}/{len(results)} variants pass 4+/5 gates")

    if len(passed) == 0:
        print("\nVERDICT: Post-earnings IV crush does NOT work at $645 scale")
        print("Likely reasons: credits too small, commissions eat profit, BS pricing unrealistic")
    else:
        print(f"\nVERDICT: {len(passed)} variants PASS — worth adversarial testing")


if __name__ == '__main__':
    main()
