"""
Pairs Trading / Mean-Reversion Strategy Research
Walk-forward backtest with rolling cointegration on ETF pairs.

Author: Claude Opus 4.6
Date: 2026-07-11
"""

import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import warnings
import json
import os
from datetime import datetime

warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/pairs_trading_research'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. DATA DOWNLOAD
# ============================================================

PAIRS = {
    # Sector pairs
    'XLK_XLC': ('XLK', 'XLC'),
    'XLF_XLI': ('XLF', 'XLI'),
    'XLE_XLB': ('XLE', 'XLB'),
    # Cross-asset
    'SPY_QQQ': ('SPY', 'QQQ'),
    'GLD_GDX': ('GLD', 'GDX'),
    'TLT_IEF': ('TLT', 'IEF'),
    # Additional interesting pairs
    'XLV_XBI': ('XLV', 'XBI'),   # health vs biotech
    'XLU_XLP': ('XLU', 'XLP'),   # utilities vs staples (defensive)
    'EEM_EFA': ('EEM', 'EFA'),   # EM vs developed intl
    'IWM_SPY': ('IWM', 'SPY'),   # small vs large cap
}

# Beta-neutral residual pairs (sector vs SPY)
BETA_NEUTRAL_SECTORS = ['XLK', 'XLF', 'XLE', 'XLI', 'XLC', 'XLV', 'XLU', 'XLB', 'XLP', 'XLY']

def download_data(start='2015-01-01', end='2026-07-10'):
    """Download all required ETF data."""
    all_tickers = set()
    for a, b in PAIRS.values():
        all_tickers.add(a)
        all_tickers.add(b)
    all_tickers.add('SPY')
    for s in BETA_NEUTRAL_SECTORS:
        all_tickers.add(s)

    tickers = sorted(all_tickers)
    print(f"Downloading {len(tickers)} tickers: {tickers}")

    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = data['Close'].dropna(how='all')

    # Forward fill small gaps, then drop remaining NaN
    prices = prices.ffill().dropna()

    print(f"Data: {prices.index[0].date()} to {prices.index[-1].date()}, {len(prices)} trading days")
    return prices


# ============================================================
# 2. PAIR ANALYSIS FUNCTIONS
# ============================================================

def compute_hedge_ratio(y, x):
    """OLS hedge ratio: y = beta * x + alpha."""
    X = add_constant(x)
    model = OLS(y, X).fit()
    return model.params[1], model.params[0]  # beta, alpha

def compute_spread(y, x, beta, alpha):
    """Compute spread = y - beta*x - alpha."""
    return y - beta * x - alpha

def compute_zscore(spread, lookback=20):
    """Rolling z-score of spread."""
    mean = spread.rolling(lookback).mean()
    std = spread.rolling(lookback).std()
    return (spread - mean) / std

def half_life(spread):
    """Ornstein-Uhlenbeck half-life of mean reversion."""
    spread_lag = spread.shift(1).dropna()
    spread_diff = spread.diff().dropna()

    # Align
    idx = spread_lag.index.intersection(spread_diff.index)
    spread_lag = spread_lag.loc[idx]
    spread_diff = spread_diff.loc[idx]

    X = add_constant(spread_lag)
    model = OLS(spread_diff, X).fit()

    theta = model.params.iloc[1] if hasattr(model.params, 'iloc') else model.params[1]
    if theta >= 0:
        return np.inf  # not mean-reverting
    return -np.log(2) / theta

def engle_granger_test(y, x):
    """Engle-Granger cointegration test. Returns (t-stat, p-value, coint_flag)."""
    score, pvalue, _ = coint(y, x)
    return score, pvalue, pvalue < 0.05

def analyze_pair_rolling(prices_y, prices_x, window=90):
    """Rolling cointegration analysis over training window."""
    n = len(prices_y)
    if n < window:
        return None

    # Use the full window for cointegration test
    _, pval, is_coint = engle_granger_test(prices_y[-window:], prices_x[-window:])

    # Hedge ratio from training window
    beta, alpha = compute_hedge_ratio(prices_y[-window:], prices_x[-window:])

    # Spread and half-life
    spread = compute_spread(prices_y[-window:], prices_x[-window:], beta, alpha)
    hl = half_life(spread)

    return {
        'coint_pval': pval,
        'is_coint': is_coint,
        'beta': beta,
        'alpha': alpha,
        'half_life': hl,
        'spread_mean': spread.mean(),
        'spread_std': spread.std(),
    }


# ============================================================
# 3. WALK-FORWARD BACKTEST ENGINE
# ============================================================

COST_BPS = 5  # 5 bps per side = 10 bps round trip

def walk_forward_backtest(prices, pair_name, ticker_a, ticker_b,
                          train_window=252, oos_step=21,
                          z_entry=2.0, z_exit=0.0, z_stop=3.5,
                          zscore_lookback=20, coint_window=90,
                          require_coint=True):
    """
    Walk-forward pairs trading backtest.

    - Train on `train_window` days to estimate hedge ratio and spread params
    - Trade for `oos_step` days OOS
    - Slide forward
    """
    pa = prices[ticker_a]
    pb = prices[ticker_b]

    n = len(pa)
    results = []
    trades = []

    # Track open position
    position = 0  # +1 = long spread, -1 = short spread, 0 = flat
    entry_zscore = 0
    entry_idx = None
    entry_spread = 0
    entry_prices = (0, 0)

    # Walk-forward loop
    start_idx = train_window

    all_oos_rets = []  # daily returns for the strategy
    spy_aligned_dates = []

    while start_idx < n:
        end_idx = min(start_idx + oos_step, n)

        # Training data
        train_y = pa.iloc[start_idx - train_window:start_idx]
        train_x = pb.iloc[start_idx - train_window:start_idx]

        # Analyze pair on training data
        analysis = analyze_pair_rolling(train_y, train_x, window=coint_window)
        if analysis is None:
            start_idx += oos_step
            continue

        # Skip if not cointegrated (optional filter)
        if require_coint and not analysis['is_coint']:
            # Still advance, but don't trade
            # Close any open position if cointegration breaks
            if position != 0:
                # Force exit
                exit_date = pa.index[start_idx]
                exit_pa = pa.iloc[start_idx]
                exit_pb = pb.iloc[start_idx]

                pnl_a = position * (exit_pa - entry_prices[0]) / entry_prices[0]
                pnl_b = -position * (exit_pb - entry_prices[1]) / entry_prices[1]
                gross_pnl = pnl_a + pnl_b
                cost = 2 * COST_BPS / 10000  # exit cost (entry cost already counted)
                net_pnl = gross_pnl - cost

                trades.append({
                    'entry_date': pa.index[entry_idx],
                    'exit_date': exit_date,
                    'direction': position,
                    'entry_z': entry_zscore,
                    'exit_reason': 'coint_break',
                    'gross_pnl': gross_pnl,
                    'net_pnl': net_pnl,
                    'hold_days': start_idx - entry_idx,
                })
                position = 0

            start_idx += oos_step
            continue

        beta = analysis['beta']
        alpha = analysis['alpha']
        spread_mean = analysis['spread_mean']
        spread_std = analysis['spread_std']

        if spread_std < 1e-8:
            start_idx += oos_step
            continue

        # OOS trading
        for i in range(start_idx, end_idx):
            date = pa.index[i]
            price_a = pa.iloc[i]
            price_b = pb.iloc[i]

            # Compute OOS spread using training params
            spread = price_a - beta * price_b - alpha
            zscore = (spread - spread_mean) / spread_std

            daily_ret = 0.0

            if position == 0:
                # Check for entry
                if zscore > z_entry:
                    # Spread is rich -> short spread (short A, long B)
                    position = -1
                    entry_zscore = zscore
                    entry_idx = i
                    entry_spread = spread
                    entry_prices = (price_a, price_b)
                    cost = 2 * COST_BPS / 10000  # entry cost both legs
                    daily_ret = -cost
                elif zscore < -z_entry:
                    # Spread is cheap -> long spread (long A, short B)
                    position = 1
                    entry_zscore = zscore
                    entry_idx = i
                    entry_spread = spread
                    entry_prices = (price_a, price_b)
                    cost = 2 * COST_BPS / 10000
                    daily_ret = -cost
            else:
                # Mark to market daily return (dollar neutral)
                prev_pa = pa.iloc[i-1]
                prev_pb = pb.iloc[i-1]
                ret_a = (price_a - prev_pa) / prev_pa
                ret_b = (price_b - prev_pb) / prev_pb
                daily_ret = position * (ret_a - beta * ret_b)

                # Check for exit
                exit_reason = None
                if position == 1 and zscore >= z_exit:
                    exit_reason = 'mean_revert'
                elif position == -1 and zscore <= z_exit:
                    exit_reason = 'mean_revert'
                elif abs(zscore) > z_stop:
                    exit_reason = 'stop'

                if exit_reason:
                    # Compute trade P&L
                    pnl_a = position * (price_a - entry_prices[0]) / entry_prices[0]
                    pnl_b = -position * beta * (price_b - entry_prices[1]) / entry_prices[1]
                    gross_pnl = pnl_a + pnl_b
                    cost = 2 * COST_BPS / 10000  # exit cost
                    net_pnl = gross_pnl - cost

                    trades.append({
                        'entry_date': pa.index[entry_idx],
                        'exit_date': date,
                        'direction': position,
                        'entry_z': entry_zscore,
                        'exit_reason': exit_reason,
                        'gross_pnl': gross_pnl,
                        'net_pnl': net_pnl,
                        'hold_days': i - entry_idx,
                    })

                    daily_ret -= 2 * COST_BPS / 10000  # exit transaction cost
                    position = 0

            all_oos_rets.append(daily_ret)
            spy_aligned_dates.append(date)

        start_idx += oos_step

    # Close any remaining position at end
    if position != 0 and len(pa) > 0:
        exit_pa = pa.iloc[-1]
        exit_pb = pb.iloc[-1]
        pnl_a = position * (exit_pa - entry_prices[0]) / entry_prices[0]
        pnl_b = -position * beta * (exit_pb - entry_prices[1]) / entry_prices[1]
        gross_pnl = pnl_a + pnl_b
        cost = 2 * COST_BPS / 10000
        net_pnl = gross_pnl - cost
        trades.append({
            'entry_date': pa.index[entry_idx],
            'exit_date': pa.index[-1],
            'direction': position,
            'entry_z': entry_zscore,
            'exit_reason': 'end_of_data',
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'hold_days': len(pa) - 1 - entry_idx,
        })

    return {
        'pair': pair_name,
        'daily_returns': pd.Series(all_oos_rets, index=spy_aligned_dates),
        'trades': trades,
    }


# ============================================================
# 4. PERFORMANCE METRICS
# ============================================================

def compute_metrics(daily_rets, trades, spy_rets=None):
    """Compute performance metrics from daily returns and trades."""
    if len(daily_rets) == 0 or daily_rets.std() == 0:
        return None

    # Annualized metrics
    ann_ret = daily_rets.mean() * 252
    ann_vol = daily_rets.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    # Sortino
    downside = daily_rets[daily_rets < 0]
    downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else 1e-8
    sortino = ann_ret / downside_vol if downside_vol > 0 else 0

    # Drawdown
    cum_rets = (1 + daily_rets).cumprod()
    rolling_max = cum_rets.cummax()
    drawdowns = (cum_rets - rolling_max) / rolling_max
    max_dd = drawdowns.min()

    # Trade stats
    if len(trades) > 0:
        net_pnls = [t['net_pnl'] for t in trades]
        winners = [p for p in net_pnls if p > 0]
        losers = [p for p in net_pnls if p <= 0]
        win_rate = len(winners) / len(net_pnls) if net_pnls else 0
        avg_win = np.mean(winners) if winners else 0
        avg_loss = np.mean(losers) if losers else 0
        profit_factor = (sum(winners) / abs(sum(losers))) if losers and sum(losers) != 0 else np.inf
        avg_hold = np.mean([t['hold_days'] for t in trades])

        # Exit reason breakdown
        exit_reasons = {}
        for t in trades:
            r = t['exit_reason']
            exit_reasons[r] = exit_reasons.get(r, 0) + 1
    else:
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0
        avg_hold = 0
        exit_reasons = {}
        net_pnls = []

    # Regime analysis (compare up vs down market periods)
    regime_gap = None
    if spy_rets is not None and len(spy_rets) > 0:
        common_idx = daily_rets.index.intersection(spy_rets.index)
        if len(common_idx) > 50:
            dr = daily_rets.loc[common_idx]
            sr = spy_rets.loc[common_idx]

            # 60-day rolling SPY return to classify regime
            spy_rolling = sr.rolling(60).mean()
            up_mask = spy_rolling > 0
            down_mask = spy_rolling <= 0

            up_rets = dr[up_mask].dropna()
            down_rets = dr[down_mask].dropna()

            if len(up_rets) > 20 and len(down_rets) > 20:
                sharpe_up = (up_rets.mean() * 252) / (up_rets.std() * np.sqrt(252)) if up_rets.std() > 0 else 0
                sharpe_down = (down_rets.mean() * 252) / (down_rets.std() * np.sqrt(252)) if down_rets.std() > 0 else 0
                max_abs = max(abs(sharpe_up), abs(sharpe_down))
                regime_gap = abs(sharpe_up - sharpe_down) / max_abs if max_abs > 0 else 0

    total_return = cum_rets.iloc[-1] - 1 if len(cum_rets) > 0 else 0

    return {
        'total_return': total_return,
        'ann_return': ann_ret,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_dd': max_dd,
        'num_trades': len(trades),
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'avg_hold_days': avg_hold,
        'exit_reasons': exit_reasons,
        'regime_gap': regime_gap,
    }


# ============================================================
# 5. BETA-NEUTRAL RESIDUAL PAIRS
# ============================================================

def create_beta_neutral_pairs(prices, sectors, benchmark='SPY', window=252):
    """
    For each sector ETF, compute residual vs SPY over rolling window.
    Then test cointegration between sector residual pairs.
    """
    spy = prices[benchmark]
    residuals = {}

    for sector in sectors:
        if sector not in prices.columns:
            continue
        sec = prices[sector]

        # Rolling beta and residual
        res = pd.Series(index=prices.index, dtype=float)
        for i in range(window, len(prices)):
            y = sec.iloc[i-window:i].pct_change().dropna()
            x = spy.iloc[i-window:i].pct_change().dropna()
            idx = y.index.intersection(x.index)
            if len(idx) < 50:
                continue
            X = add_constant(x.loc[idx])
            model = OLS(y.loc[idx], X).fit()
            beta = model.params.iloc[1] if hasattr(model.params, 'iloc') else model.params[1]
            # Residual = sector return - beta * SPY return (cumulative)
            res.iloc[i] = np.log(sec.iloc[i]) - beta * np.log(spy.iloc[i])

        residuals[sector] = res.dropna()

    # Test cointegration between residual pairs
    beta_neutral_results = []
    sector_list = list(residuals.keys())
    for i in range(len(sector_list)):
        for j in range(i+1, len(sector_list)):
            s1, s2 = sector_list[i], sector_list[j]
            common = residuals[s1].index.intersection(residuals[s2].index)
            if len(common) < 120:
                continue
            r1 = residuals[s1].loc[common]
            r2 = residuals[s2].loc[common]

            # Test cointegration on last 252 days
            test_window = min(252, len(common))
            _, pval, is_coint = engle_granger_test(
                r1.iloc[-test_window:], r2.iloc[-test_window:]
            )
            hl = half_life(r1.iloc[-test_window:] - r2.iloc[-test_window:])

            beta_neutral_results.append({
                'pair': f'{s1}_res_{s2}_res',
                'coint_pval': pval,
                'is_coint': is_coint,
                'half_life': hl,
            })

    return beta_neutral_results


# ============================================================
# 6. MAIN RESEARCH
# ============================================================

def _fmt_rg(val):
    return 'N/A' if val is None else f'{val:.2f}'

def main():
    print("=" * 70)
    print("PAIRS TRADING RESEARCH - Walk-Forward Backtest")
    print("=" * 70)

    # Download data
    prices = download_data()
    prices.to_csv(os.path.join(OUTPUT_DIR, 'etf_prices.csv'))

    # SPY returns for benchmark comparison
    spy_rets = prices['SPY'].pct_change().dropna()

    # ---- PHASE 1: Static cointegration scan ----
    print("\n" + "=" * 70)
    print("PHASE 1: Cointegration Scan (full sample)")
    print("=" * 70)

    scan_results = []
    for pair_name, (ta, tb) in PAIRS.items():
        if ta not in prices.columns or tb not in prices.columns:
            print(f"  {pair_name}: SKIPPED (missing data)")
            continue

        pa = prices[ta].dropna()
        pb = prices[tb].dropna()
        common = pa.index.intersection(pb.index)
        pa = pa.loc[common]
        pb = pb.loc[common]

        # Full-sample cointegration
        score, pval, is_coint = engle_granger_test(pa, pb)

        # Hedge ratio and half-life
        beta, alpha = compute_hedge_ratio(pa, pb)
        spread = compute_spread(pa, pb, beta, alpha)
        hl = half_life(spread)

        # Rolling cointegration (what % of 120d windows are cointegrated?)
        roll_coint_pct = 0
        roll_window = 120
        n_tests = 0
        n_coint = 0
        for i in range(roll_window, len(pa), 21):  # test every 21 days
            _, p, c = engle_granger_test(pa.iloc[i-roll_window:i], pb.iloc[i-roll_window:i])
            n_tests += 1
            if c:
                n_coint += 1
        roll_coint_pct = n_coint / n_tests if n_tests > 0 else 0

        result = {
            'pair': pair_name,
            'coint_pval': pval,
            'is_coint': is_coint,
            'half_life': min(hl, 999),
            'beta': beta,
            'roll_coint_pct': roll_coint_pct,
        }
        scan_results.append(result)

        flag = "COINT" if is_coint else "---"
        print(f"  {pair_name:12s} | p={pval:.4f} {flag:6s} | HL={min(hl,999):6.1f}d | beta={beta:.3f} | roll_coint={roll_coint_pct:.1%}")

    # ---- PHASE 1b: Beta-neutral residual scan ----
    print("\n" + "=" * 70)
    print("PHASE 1b: Beta-Neutral Residual Pairs")
    print("=" * 70)

    bn_results = create_beta_neutral_pairs(prices, BETA_NEUTRAL_SECTORS)
    for r in sorted(bn_results, key=lambda x: x['coint_pval']):
        flag = "COINT" if r['is_coint'] else "---"
        print(f"  {r['pair']:25s} | p={r['coint_pval']:.4f} {flag:6s} | HL={min(r['half_life'],999):6.1f}d")

    # ---- PHASE 2: Walk-forward backtests ----
    print("\n" + "=" * 70)
    print("PHASE 2: Walk-Forward Backtests (OOS only)")
    print("=" * 70)

    # Test with and without cointegration filter
    configs = [
        {'z_entry': 2.0, 'z_exit': 0.0, 'z_stop': 3.5, 'require_coint': True, 'label': 'z2.0_coint'},
        {'z_entry': 2.0, 'z_exit': 0.0, 'z_stop': 3.5, 'require_coint': False, 'label': 'z2.0_nocoint'},
        {'z_entry': 1.5, 'z_exit': 0.5, 'z_stop': 3.0, 'require_coint': True, 'label': 'z1.5_coint'},
        {'z_entry': 2.5, 'z_exit': 0.0, 'z_stop': 4.0, 'require_coint': True, 'label': 'z2.5_coint'},
    ]

    all_backtest_results = []

    for config in configs:
        label = config.pop('label')
        print(f"\n--- Config: {label} ---")

        for pair_name, (ta, tb) in PAIRS.items():
            if ta not in prices.columns or tb not in prices.columns:
                continue

            bt = walk_forward_backtest(
                prices, pair_name, ta, tb,
                train_window=252, oos_step=21,
                **config
            )

            if len(bt['daily_returns']) == 0:
                print(f"  {pair_name:12s} | NO TRADES")
                continue

            metrics = compute_metrics(bt['daily_returns'], bt['trades'], spy_rets)
            if metrics is None:
                print(f"  {pair_name:12s} | INSUFFICIENT DATA")
                continue

            metrics['pair'] = pair_name
            metrics['config'] = label
            all_backtest_results.append(metrics)

            print(f"  {pair_name:12s} | Sharpe={metrics['sharpe']:+.2f} | Sortino={metrics['sortino']:+.2f} | "
                  f"MaxDD={metrics['max_dd']:.1%} | WR={metrics['win_rate']:.1%} | "
                  f"N={metrics['num_trades']:3d} | AvgHold={metrics['avg_hold_days']:.1f}d | "
                  f"PF={metrics['profit_factor']:.2f} | "
                  f"RegimeGap={_fmt_rg(metrics['regime_gap'])}")

        # Restore label
        config['label'] = label

    # ---- PHASE 3: Portfolio-level analysis ----
    print("\n" + "=" * 70)
    print("PHASE 3: Equal-Weight Portfolio of Best Pairs")
    print("=" * 70)

    # Filter to best config (z2.0_coint) and pairs with positive Sharpe
    best_config = 'z2.0_coint'
    best_pairs = [r for r in all_backtest_results
                  if r['config'] == best_config and r['sharpe'] > 0]

    if best_pairs:
        print(f"\nPairs with positive Sharpe in {best_config}:")
        for r in sorted(best_pairs, key=lambda x: -x['sharpe']):
            print(f"  {r['pair']:12s} | Sharpe={r['sharpe']:+.2f} | Sortino={r['sortino']:+.2f} | "
                  f"WR={r['win_rate']:.1%} | PF={r['profit_factor']:.2f}")

        # Build equal-weight portfolio from all pairs (including negative Sharpe for diversification check)
        all_config_results = [r for r in all_backtest_results if r['config'] == best_config]

        # Re-run to get daily returns for portfolio construction
        portfolio_rets = {}
        for pair_name, (ta, tb) in PAIRS.items():
            if ta not in prices.columns or tb not in prices.columns:
                continue
            bt = walk_forward_backtest(
                prices, pair_name, ta, tb,
                train_window=252, oos_step=21,
                z_entry=2.0, z_exit=0.0, z_stop=3.5, require_coint=True
            )
            if len(bt['daily_returns']) > 0:
                portfolio_rets[pair_name] = bt['daily_returns']

        if portfolio_rets:
            # Combine into equal-weight portfolio
            all_dr = pd.DataFrame(portfolio_rets)
            all_dr = all_dr.fillna(0)
            portfolio_daily = all_dr.mean(axis=1)

            port_metrics = compute_metrics(portfolio_daily, [], spy_rets)
            if port_metrics:
                print(f"\nEqual-Weight Portfolio (all {len(portfolio_rets)} pairs):")
                print(f"  Sharpe:     {port_metrics['sharpe']:+.3f}")
                print(f"  Sortino:    {port_metrics['sortino']:+.3f}")
                print(f"  Ann Return: {port_metrics['ann_return']:.2%}")
                print(f"  Ann Vol:    {port_metrics['ann_vol']:.2%}")
                print(f"  Max DD:     {port_metrics['max_dd']:.2%}")
                print(f"  Total Ret:  {port_metrics['total_return']:.2%}")
                rg = port_metrics['regime_gap']
                print(f"  Regime Gap: {'N/A' if rg is None else f'{rg:.2f}'}")
    else:
        print("  No pairs with positive Sharpe after costs.")

    # ---- PHASE 4: SPY Buy & Hold Comparison ----
    print("\n" + "=" * 70)
    print("PHASE 4: SPY Buy & Hold Comparison")
    print("=" * 70)

    # Use same OOS period as our backtests (after 252-day warmup)
    spy_oos = spy_rets.iloc[252:]
    spy_cum = (1 + spy_oos).cumprod()
    spy_ann_ret = spy_oos.mean() * 252
    spy_ann_vol = spy_oos.std() * np.sqrt(252)
    spy_sharpe = spy_ann_ret / spy_ann_vol
    spy_dd = ((spy_cum - spy_cum.cummax()) / spy_cum.cummax()).min()
    spy_downside = spy_oos[spy_oos < 0].std() * np.sqrt(252)
    spy_sortino = spy_ann_ret / spy_downside if spy_downside > 0 else 0

    print(f"  SPY B&H:     Sharpe={spy_sharpe:+.3f} | Sortino={spy_sortino:+.3f} | "
          f"Ann Ret={spy_ann_ret:.2%} | MaxDD={spy_dd:.2%} | Total={spy_cum.iloc[-1]-1:.2%}")

    # ---- PHASE 5: Summary & Recommendations ----
    print("\n" + "=" * 70)
    print("PHASE 5: Summary & Recommendations")
    print("=" * 70)

    # Sort all results by Sharpe
    all_backtest_results.sort(key=lambda x: -x['sharpe'])

    viable = [r for r in all_backtest_results if r['sharpe'] > 0.3 and r['num_trades'] >= 10]
    marginal = [r for r in all_backtest_results if 0 < r['sharpe'] <= 0.3 and r['num_trades'] >= 10]

    print(f"\nViable pairs (Sharpe > 0.3, N >= 10): {len(viable)}")
    for r in viable:
        rg = r['regime_gap']
        rg_str = 'N/A' if rg is None else f'{rg:.2f}'
        regime_flag = ' REGIME-DEPENDENT' if rg is not None and rg > 0.50 else ''
        print(f"  {r['pair']:12s} [{r['config']:15s}] Sharpe={r['sharpe']:+.2f} Sortino={r['sortino']:+.2f} "
              f"WR={r['win_rate']:.0%} PF={r['profit_factor']:.2f} N={r['num_trades']} "
              f"RegGap={rg_str}{regime_flag}")

    print(f"\nMarginal pairs (0 < Sharpe <= 0.3, N >= 10): {len(marginal)}")
    for r in marginal:
        rg = r['regime_gap']
        rg_str = 'N/A' if rg is None else f'{rg:.2f}'
        print(f"  {r['pair']:12s} [{r['config']:15s}] Sharpe={r['sharpe']:+.2f} N={r['num_trades']} RegGap={rg_str}")

    unprofitable = [r for r in all_backtest_results if r['sharpe'] <= 0]
    print(f"\nUnprofitable (Sharpe <= 0): {len(unprofitable)} pair-config combos")

    # Honest assessment
    print("\n--- HONEST ASSESSMENT ---")
    if len(viable) == 0:
        print("NO pairs show robust edge after costs in walk-forward OOS testing.")
        print("Pairs trading on daily ETFs with liquid sector ETFs is heavily arbitraged.")
        print("Recommendation: Do NOT deploy. The edge does not survive transaction costs.")
    elif all(r.get('regime_gap', 0) and r['regime_gap'] > 0.5 for r in viable if r.get('regime_gap') is not None):
        print("All viable pairs are regime-dependent (gap > 0.50). This is NOT robust edge.")
        print("Recommendation: Do NOT deploy without further regime-agnostic validation.")
    else:
        print("Some pairs show potential edge. Review regime gaps and trade counts carefully.")
        print("Consider paper trading the top pair(s) for 3+ months before live deployment.")

    # Save results
    results_summary = {
        'scan_results': scan_results,
        'backtest_results': [{k: v for k, v in r.items() if k != 'exit_reasons' or isinstance(v, (str, int, float))}
                             for r in all_backtest_results],
        'spy_benchmark': {
            'sharpe': spy_sharpe,
            'sortino': spy_sortino,
            'ann_return': spy_ann_ret,
            'max_dd': spy_dd,
        },
        'run_date': datetime.now().isoformat(),
        'parameters': {
            'cost_bps_per_side': COST_BPS,
            'train_window': 252,
            'oos_step': 21,
            'configs_tested': [c['label'] for c in configs],
        }
    }

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    def deep_convert(obj):
        if isinstance(obj, dict):
            return {k: deep_convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [deep_convert(v) for v in obj]
        return convert(obj)

    with open(os.path.join(OUTPUT_DIR, 'results_summary.json'), 'w') as f:
        json.dump(deep_convert(results_summary), f, indent=2, default=str)

    print(f"\nResults saved to {OUTPUT_DIR}/results_summary.json")
    print("=" * 70)


if __name__ == '__main__':
    main()
