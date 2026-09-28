#!/usr/bin/env python3
"""
Dispersion / Correlation Trading Backtest
==========================================
6 variants trading on market-wide realized correlation dynamics.
Walk-forward OOT: Jan 2022 - Jul 2026.
5-gate validation: Sharpe>0.5, perm p<0.05, regime gap<0.5, MaxDD>-50%, >=20 trades.

$645 Robinhood account, $0 commission, 0.02% slippage.
"""

import json
import warnings
import datetime as dt
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings('ignore')

# ─── CONFIG ────────────────────────────────────────────────────────────────────
ACCOUNT = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%
OOT_START = '2022-01-01'
OOT_END = '2026-07-30'
LOOKBACK_START = '2020-01-01'  # extra history for rolling calcs
N_PERM = 1000

BASKET = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'JNJ', 'XOM', 'PG', 'HD']
SECTOR_ETFS = ['XLK', 'XLE', 'XLF', 'XLV', 'XLC', 'XLY', 'XLI']
TRADE_TICKERS = ['SPY', 'QQQ', 'TLT']
ALL_TICKERS = list(set(BASKET + SECTOR_ETFS + TRADE_TICKERS + ['^VIX']))

RESULTS_PATH = Path('/home/jupiter/Lvl3Quant/data/dispersion_trading_results.json')


def download_data():
    """Download all required price data."""
    print("Downloading market data...")
    data = {}
    # Download in batches to avoid rate limits
    tickers_no_vix = [t for t in ALL_TICKERS if t != '^VIX']

    df = yf.download(tickers_no_vix, start=LOOKBACK_START, end=OOT_END, auto_adjust=True, progress=False)

    # Handle multi-level columns
    if isinstance(df.columns, pd.MultiIndex):
        close = df['Close']
    else:
        close = df

    # Download VIX separately
    vix = yf.download('^VIX', start=LOOKBACK_START, end=OOT_END, auto_adjust=True, progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        close['VIX'] = vix['Close'].iloc[:, 0] if isinstance(vix['Close'], pd.DataFrame) else vix['Close']
    else:
        close['VIX'] = vix['Close']

    close = close.ffill().dropna(how='all')
    print(f"  Data: {close.index[0].strftime('%Y-%m-%d')} to {close.index[-1].strftime('%Y-%m-%d')}, {len(close)} days")
    return close


def compute_correlation_index(close, basket=BASKET, window=20):
    """Compute rolling average pairwise correlation of the basket."""
    returns = close[basket].pct_change()

    # Rolling pairwise correlation - vectorized approach
    corr_series = []
    pairs = list(combinations(basket, 2))

    for t1, t2 in pairs:
        rc = returns[t1].rolling(window).corr(returns[t2])
        corr_series.append(rc)

    avg_corr = pd.concat(corr_series, axis=1).mean(axis=1)
    avg_corr.name = 'avg_corr'
    return avg_corr


def compute_dispersion(close, sectors=SECTOR_ETFS, window=20):
    """Compute cross-sector return dispersion (std of sector returns)."""
    sector_ret = close[sectors].pct_change().rolling(window).mean()
    dispersion = sector_ret.std(axis=1)
    dispersion.name = 'dispersion'
    return dispersion


def compute_features(close):
    """Build all features needed for the 6 variants."""
    avg_corr = compute_correlation_index(close)
    dispersion = compute_dispersion(close)

    # Rolling percentiles (252-day history)
    corr_pctile = avg_corr.rolling(252).rank(pct=True)
    disp_pctile = dispersion.rolling(252).rank(pct=True)

    # VIX percentile
    vix = close['VIX']
    vix_pctile = vix.rolling(252).rank(pct=True)

    # 5-day changes
    corr_5d_chg = avg_corr.diff(5)
    vix_5d_chg = vix.diff(5)

    # SPY 200-SMA for regime
    spy_sma200 = close['SPY'].rolling(200).mean()
    regime = (close['SPY'] > spy_sma200).astype(int)  # 1=bull, 0=bear

    # Sector momentum (20-day returns)
    sector_mom = close[SECTOR_ETFS].pct_change(20)

    features = pd.DataFrame({
        'avg_corr': avg_corr,
        'corr_pctile': corr_pctile,
        'disp_pctile': disp_pctile,
        'vix': vix,
        'vix_pctile': vix_pctile,
        'corr_5d_chg': corr_5d_chg,
        'vix_5d_chg': vix_5d_chg,
        'regime': regime,
        'spy_sma200': spy_sma200,
    }, index=close.index)

    # Add sector momentum columns
    for s in SECTOR_ETFS:
        features[f'mom_{s}'] = sector_mom[s]

    return features


def apply_slippage(price, direction='buy'):
    """Apply slippage to trade price."""
    if direction == 'buy':
        return price * (1 + SLIPPAGE_PCT)
    else:
        return price * (1 - SLIPPAGE_PCT)


def simulate_trades(close, features, variant):
    """
    Simulate trades for a given variant. Returns a DataFrame of trades.
    Each trade: entry_date, exit_date, ticker, entry_price, exit_price, return, regime
    """
    oot_mask = close.index >= OOT_START
    dates = close.index[oot_mask]
    trades = []

    in_position = False
    position_ticker = None
    entry_date = None
    entry_price = None
    hold_days = 0
    max_hold = 10
    cash_mode = False  # for variant B

    for i, date in enumerate(dates):
        feat = features.loc[date]

        # Skip if features not available
        if pd.isna(feat['corr_pctile']) or pd.isna(feat['avg_corr']):
            continue

        # ── EXIT LOGIC ──
        if in_position:
            hold_days += 1
            should_exit = False

            if variant in ['A', 'C', 'E']:
                # Fixed 10-day hold
                if hold_days >= max_hold:
                    should_exit = True
            elif variant == 'B':
                # Exit when correlation drops below 20th pctile (go to cash)
                if feat['corr_pctile'] < 0.20:
                    should_exit = True
                    cash_mode = True
            elif variant == 'D':
                # Weekly rebalance (5 days)
                if hold_days >= 5:
                    should_exit = True
            elif variant == 'F':
                # Exit when correlation direction reverses
                if position_ticker == 'TLT' and feat['corr_5d_chg'] < -0.05:
                    should_exit = True
                elif position_ticker == 'QQQ' and feat['corr_5d_chg'] > 0.05:
                    should_exit = True
                elif hold_days >= 20:  # max hold safety
                    should_exit = True

            if should_exit:
                exit_price = apply_slippage(close.loc[date, position_ticker], 'sell')
                trade_ret = (exit_price / entry_price) - 1
                trades.append({
                    'entry_date': entry_date.strftime('%Y-%m-%d'),
                    'exit_date': date.strftime('%Y-%m-%d'),
                    'ticker': position_ticker,
                    'entry_price': float(entry_price),
                    'exit_price': float(exit_price),
                    'return': float(trade_ret),
                    'regime': 'bull' if feat['regime'] == 1 else 'bear',
                    'hold_days': hold_days,
                })
                in_position = False
                position_ticker = None
                hold_days = 0
            continue  # don't enter same day as exit

        # ── ENTRY LOGIC ──
        if variant == 'A':
            # Correlation Spike Fade: corr > 80th pctile → buy QQQ
            if feat['corr_pctile'] > 0.80:
                ticker = 'QQQ'
                entry_price = apply_slippage(close.loc[date, ticker], 'buy')
                entry_date = date
                position_ticker = ticker
                in_position = True
                max_hold = 10

        elif variant == 'B':
            # Low Correlation Warning: cash when corr < 20th, re-enter when > 40th
            if cash_mode:
                if feat['corr_pctile'] > 0.40:
                    cash_mode = False
                    ticker = 'QQQ'
                    entry_price = apply_slippage(close.loc[date, ticker], 'buy')
                    entry_date = date
                    position_ticker = ticker
                    in_position = True
            else:
                if feat['corr_pctile'] >= 0.20:
                    # Stay in QQQ (enter if not already)
                    ticker = 'QQQ'
                    entry_price = apply_slippage(close.loc[date, ticker], 'buy')
                    entry_date = date
                    position_ticker = ticker
                    in_position = True

        elif variant == 'C':
            # Dispersion Play: high dispersion → buy best sector
            if not pd.isna(feat['disp_pctile']) and feat['disp_pctile'] > 0.80:
                # Find best-performing sector ETF
                mom_cols = [f'mom_{s}' for s in SECTOR_ETFS]
                mom_vals = {s: feat[f'mom_{s}'] for s in SECTOR_ETFS if not pd.isna(feat[f'mom_{s}'])}
                if mom_vals:
                    best_sector = max(mom_vals, key=mom_vals.get)
                    entry_price = apply_slippage(close.loc[date, best_sector], 'buy')
                    entry_date = date
                    position_ticker = best_sector
                    in_position = True
                    max_hold = 10

        elif variant == 'D':
            # Correlation Regime Switch: high corr → SPY, low corr → best sector
            if feat['corr_pctile'] > 0.60:
                ticker = 'SPY'
                entry_price = apply_slippage(close.loc[date, ticker], 'buy')
                entry_date = date
                position_ticker = ticker
                in_position = True
            elif feat['corr_pctile'] < 0.40:
                mom_cols = [f'mom_{s}' for s in SECTOR_ETFS]
                mom_vals = {s: feat[f'mom_{s}'] for s in SECTOR_ETFS if not pd.isna(feat[f'mom_{s}'])}
                if mom_vals:
                    best_sector = max(mom_vals, key=mom_vals.get)
                    entry_price = apply_slippage(close.loc[date, best_sector], 'buy')
                    entry_date = date
                    position_ticker = best_sector
                    in_position = True

        elif variant == 'E':
            # Correlation + VIX Combo: both > 70th AND both declining
            if (feat['corr_pctile'] > 0.70 and feat['vix_pctile'] > 0.70
                and feat['corr_5d_chg'] < 0 and feat['vix_5d_chg'] < 0):
                ticker = 'QQQ'
                entry_price = apply_slippage(close.loc[date, ticker], 'buy')
                entry_date = date
                position_ticker = ticker
                in_position = True
                max_hold = 10

        elif variant == 'F':
            # Anti-Correlation rotation
            if not pd.isna(feat['corr_5d_chg']):
                if feat['corr_5d_chg'] > 0.05:
                    ticker = 'TLT'
                    entry_price = apply_slippage(close.loc[date, ticker], 'buy')
                    entry_date = date
                    position_ticker = ticker
                    in_position = True
                elif feat['corr_5d_chg'] < -0.05:
                    ticker = 'QQQ'
                    entry_price = apply_slippage(close.loc[date, ticker], 'buy')
                    entry_date = date
                    position_ticker = ticker
                    in_position = True

    # Close any open position at end
    if in_position and position_ticker:
        last_date = dates[-1]
        exit_price = apply_slippage(close.loc[last_date, position_ticker], 'sell')
        trade_ret = (exit_price / entry_price) - 1
        trades.append({
            'entry_date': entry_date.strftime('%Y-%m-%d'),
            'exit_date': last_date.strftime('%Y-%m-%d'),
            'ticker': position_ticker,
            'entry_price': float(entry_price),
            'exit_price': float(exit_price),
            'return': float(trade_ret),
            'regime': 'bull' if features.loc[last_date, 'regime'] == 1 else 'bear',
            'hold_days': hold_days,
        })

    return pd.DataFrame(trades)


def compute_metrics(trades_df, close):
    """Compute performance metrics from trades DataFrame."""
    if len(trades_df) == 0:
        return None

    returns = trades_df['return'].values
    n_trades = len(returns)

    # Build daily equity curve for proper Sharpe/Sortino
    equity = ACCOUNT
    equity_curve = [ACCOUNT]
    daily_returns = []

    for _, t in trades_df.iterrows():
        pnl = equity * t['return']
        equity += pnl
        equity_curve.append(equity)
        # Approximate daily return from trade return / hold days
        hold = max(t['hold_days'], 1)
        daily_ret = (1 + t['return']) ** (1/hold) - 1
        for _ in range(hold):
            daily_returns.append(daily_ret)

    daily_returns = np.array(daily_returns)
    equity_arr = np.array(equity_curve)

    # Sharpe (annualized, 252 trading days)
    if len(daily_returns) > 1 and np.std(daily_returns) > 0:
        sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Sortino
    downside = daily_returns[daily_returns < 0]
    if len(downside) > 1 and np.std(downside) > 0:
        sortino = np.mean(daily_returns) / np.std(downside) * np.sqrt(252)
    else:
        sortino = sharpe

    # Win rate
    wr = np.mean(returns > 0)

    # Profit factor
    gross_profit = returns[returns > 0].sum() if (returns > 0).any() else 0
    gross_loss = abs(returns[returns < 0].sum()) if (returns < 0).any() else 1e-9
    pf = gross_profit / gross_loss if gross_loss > 0 else 999.0

    # Max drawdown
    peak = np.maximum.accumulate(equity_arr)
    dd = (equity_arr - peak) / peak
    max_dd = dd.min()

    # Total return
    total_ret = (equity_arr[-1] / ACCOUNT) - 1

    # CAGR
    n_years = len(daily_returns) / 252
    if n_years > 0 and equity_arr[-1] > 0:
        cagr = (equity_arr[-1] / ACCOUNT) ** (1/n_years) - 1
    else:
        cagr = 0.0

    # QQQ correlation
    qqq_daily = close['QQQ'].pct_change().dropna()
    # Align trade returns to dates
    trade_daily_rets = pd.Series(0.0, index=close.index[close.index >= OOT_START])
    for _, t in trades_df.iterrows():
        entry = pd.Timestamp(t['entry_date'])
        exit_ = pd.Timestamp(t['exit_date'])
        hold = max(t['hold_days'], 1)
        daily_ret = (1 + t['return']) ** (1/hold) - 1
        mask = (trade_daily_rets.index >= entry) & (trade_daily_rets.index <= exit_)
        trade_daily_rets.loc[mask] = daily_ret

    # Compute correlation
    aligned = pd.DataFrame({'strat': trade_daily_rets, 'qqq': qqq_daily}).dropna()
    if len(aligned) > 20:
        qqq_corr = aligned['strat'].corr(aligned['qqq'])
    else:
        qqq_corr = np.nan

    return {
        'n_trades': int(n_trades),
        'total_return_pct': round(total_ret * 100, 2),
        'cagr_pct': round(cagr * 100, 2),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'win_rate': round(wr, 4),
        'profit_factor': round(pf, 3),
        'max_drawdown_pct': round(max_dd * 100, 2),
        'final_equity': round(equity_arr[-1], 2),
        'avg_trade_return_pct': round(np.mean(returns) * 100, 3),
        'avg_hold_days': round(trades_df['hold_days'].mean(), 1),
        'qqq_correlation': round(qqq_corr, 4) if not np.isnan(qqq_corr) else None,
    }


def permutation_test(trades_df, close, n_perm=N_PERM):
    """
    Proper permutation test: randomly sample entry dates from the OOT period,
    compute forward returns over the same hold period and ticker as each trade.
    Tests whether the TIMING of entries adds value vs random timing.
    """
    if len(trades_df) < 5:
        return 1.0

    actual_mean = trades_df['return'].mean()
    n_trades = len(trades_df)

    # Pre-compute all possible forward returns for each ticker used
    oot_dates = close.index[close.index >= OOT_START]
    ticker_fwd_rets = {}
    for ticker in trades_df['ticker'].unique():
        prices = close[ticker].reindex(oot_dates).ffill()
        ticker_fwd_rets[ticker] = prices

    # For each trade, record ticker and hold_days
    trade_specs = trades_df[['ticker', 'hold_days']].values.tolist()

    count_ge = 0
    rng = np.random.default_rng(42)

    for _ in range(n_perm):
        perm_returns = []
        for ticker, hold in trade_specs:
            hold = int(max(hold, 1))
            prices = ticker_fwd_rets[ticker]
            max_idx = len(prices) - hold - 1
            if max_idx < 1:
                continue
            rand_idx = rng.integers(0, max_idx)
            entry_p = prices.iloc[rand_idx] * (1 + SLIPPAGE_PCT)
            exit_p = prices.iloc[rand_idx + hold] * (1 - SLIPPAGE_PCT)
            perm_returns.append(exit_p / entry_p - 1)

        if perm_returns:
            perm_mean = np.mean(perm_returns)
            if perm_mean >= actual_mean:
                count_ge += 1

    p_value = (count_ge + 1) / (n_perm + 1)  # continuity correction
    return p_value


def regime_analysis(trades_df):
    """Compute Sharpe per regime and regime gap."""
    if len(trades_df) == 0:
        return {'bull_sharpe': 0, 'bear_sharpe': 0, 'regime_gap': 1.0}

    results = {}
    for regime in ['bull', 'bear']:
        subset = trades_df[trades_df['regime'] == regime]
        if len(subset) >= 3:
            rets = subset['return'].values
            if np.std(rets) > 0:
                s = np.mean(rets) / np.std(rets) * np.sqrt(252 / max(subset['hold_days'].mean(), 1))
            else:
                s = 0.0
            results[f'{regime}_sharpe'] = round(s, 3)
        else:
            results[f'{regime}_sharpe'] = 0.0

    s_bull = abs(results['bull_sharpe'])
    s_bear = abs(results['bear_sharpe'])
    denom = max(s_bull, s_bear, 0.001)
    results['regime_gap'] = round(abs(results['bull_sharpe'] - results['bear_sharpe']) / denom, 4)

    return results


def validate_5gate(metrics, perm_p, regime_info):
    """Apply 5-gate validation."""
    gates = {
        'sharpe_gt_0.5': metrics['sharpe'] > 0.5,
        'perm_p_lt_0.05': perm_p < 0.05,
        'regime_gap_lt_0.5': regime_info['regime_gap'] < 0.5,
        'maxdd_gt_neg50': metrics['max_drawdown_pct'] > -50.0,
        'trades_gte_20': metrics['n_trades'] >= 20,
    }
    gates['all_pass'] = all(gates.values())
    return gates


VARIANT_NAMES = {
    'A': 'Correlation Spike Fade',
    'B': 'Low Correlation Warning',
    'C': 'Dispersion Play (Best Sector)',
    'D': 'Correlation Regime Switch',
    'E': 'Correlation + VIX Combo',
    'F': 'Anti-Correlation Rotation',
}


def main():
    print("=" * 70)
    print("DISPERSION / CORRELATION TRADING BACKTEST")
    print(f"OOT: {OOT_START} to {OOT_END}  |  Account: ${ACCOUNT}")
    print("=" * 70)

    close = download_data()
    print("\nComputing correlation features...")
    features = compute_features(close)

    all_results = {}

    for variant in ['A', 'B', 'C', 'D', 'E', 'F']:
        name = VARIANT_NAMES[variant]
        print(f"\n{'─' * 60}")
        print(f"Variant {variant}: {name}")
        print(f"{'─' * 60}")

        trades_df = simulate_trades(close, features, variant)

        if len(trades_df) == 0:
            print(f"  NO TRADES generated.")
            all_results[variant] = {
                'name': name,
                'metrics': None,
                'validation': {'all_pass': False, 'reason': 'no trades'},
            }
            continue

        print(f"  Trades: {len(trades_df)}")
        metrics = compute_metrics(trades_df, close)

        if metrics is None:
            all_results[variant] = {
                'name': name,
                'metrics': None,
                'validation': {'all_pass': False, 'reason': 'compute failed'},
            }
            continue

        # Permutation test
        print(f"  Running {N_PERM}-iteration permutation test...")
        perm_p = permutation_test(trades_df, close)

        # Regime analysis
        regime_info = regime_analysis(trades_df)

        # 5-gate validation
        gates = validate_5gate(metrics, perm_p, regime_info)

        # Print results
        print(f"  Sharpe: {metrics['sharpe']:.3f}  |  Sortino: {metrics['sortino']:.3f}")
        print(f"  Win Rate: {metrics['win_rate']:.1%}  |  PF: {metrics['profit_factor']:.2f}")
        print(f"  Total Return: {metrics['total_return_pct']:.1f}%  |  MaxDD: {metrics['max_drawdown_pct']:.1f}%")
        print(f"  Final Equity: ${metrics['final_equity']:.2f}  |  QQQ Corr: {metrics['qqq_correlation']}")
        print(f"  Avg Hold: {metrics['avg_hold_days']:.1f} days  |  Avg Trade: {metrics['avg_trade_return_pct']:.3f}%")
        print(f"  Perm p-value: {perm_p:.4f}")
        print(f"  Regime — Bull Sharpe: {regime_info['bull_sharpe']:.3f}, Bear Sharpe: {regime_info['bear_sharpe']:.3f}, Gap: {regime_info['regime_gap']:.4f}")
        print(f"  5-Gate: {'PASS' if gates['all_pass'] else 'FAIL'}")
        for g, v in gates.items():
            if g != 'all_pass':
                status = 'OK' if v else 'FAIL'
                print(f"    {g}: {status}")

        # Ticker breakdown
        if 'ticker' in trades_df.columns:
            print(f"  Ticker breakdown:")
            for ticker, grp in trades_df.groupby('ticker'):
                avg_r = grp['return'].mean() * 100
                print(f"    {ticker}: {len(grp)} trades, avg {avg_r:.3f}%")

        all_results[variant] = {
            'name': name,
            'metrics': metrics,
            'perm_p_value': round(perm_p, 4),
            'regime': regime_info,
            'validation': gates,
            'sample_trades': trades_df.head(5).to_dict('records'),
        }

    # ── SUMMARY ──
    print(f"\n{'=' * 70}")
    print("SUMMARY — 5-GATE VALIDATION")
    print(f"{'=' * 70}")
    print(f"{'Variant':<35} {'Sharpe':>7} {'QQQ Corr':>9} {'Trades':>7} {'Pass':>5}")
    print(f"{'─' * 70}")

    for v in ['A', 'B', 'C', 'D', 'E', 'F']:
        r = all_results[v]
        if r['metrics']:
            m = r['metrics']
            passed = 'YES' if r['validation']['all_pass'] else 'NO'
            qqq_c = f"{m['qqq_correlation']:.3f}" if m['qqq_correlation'] is not None else 'N/A'
            print(f"  {v}: {r['name']:<30} {m['sharpe']:>7.3f} {qqq_c:>9} {m['n_trades']:>7} {passed:>5}")
        else:
            print(f"  {v}: {r['name']:<30} {'N/A':>7} {'N/A':>9} {'0':>7} {'NO':>5}")

    # Save results
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump({
            'meta': {
                'strategy_class': 'dispersion_correlation_trading',
                'oot_period': f'{OOT_START} to {OOT_END}',
                'account_size': ACCOUNT,
                'slippage_pct': SLIPPAGE_PCT,
                'n_permutations': N_PERM,
                'basket': BASKET,
                'sector_etfs': SECTOR_ETFS,
                'generated': dt.datetime.now().isoformat(),
            },
            'variants': all_results,
        }, f, indent=2, default=str)

    print(f"\nResults saved to {RESULTS_PATH}")
    return all_results


if __name__ == '__main__':
    main()
