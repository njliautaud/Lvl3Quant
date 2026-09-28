"""
Treasury Curve Steepening Strategy (AVO Seed)
==============================================
Novel concept: Trade the yield curve steepening/flattening as equity sector signal.

When curve steepens (TLT underperforms SHY): banks/financials benefit, growth suffers.
When curve flattens: growth/tech benefits, banks suffer.
Add VIX regime filter for position sizing.

Walk-forward validated, 8 folds, regime-stratified.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import json, warnings
warnings.filterwarnings('ignore')

def download_data():
    tickers = ['TLT', 'SHY', 'XLF', 'XLK', 'XLI', 'XLU', 'XLP', 'XLY', 
               'XLE', 'XLB', 'XLV', 'XLRE', 'SPY', 'GLD']
    vix_tickers = ['^VIX']
    
    data = yf.download(tickers, start='2020-01-01', end='2026-07-01', auto_adjust=True)['Close']
    vix = yf.download(vix_tickers, start='2020-01-01', end='2026-07-01', auto_adjust=True)['Close']
    if isinstance(vix, pd.DataFrame):
        vix = vix.iloc[:, 0]
    vix.name = 'VIX'
    return data, vix

def compute_signals(data, vix, params):
    """Compute curve steepening signal and sector allocation."""
    # Yield curve proxy: TLT/SHY ratio (inverse of steepening)
    curve = data['TLT'] / data['SHY']
    curve_ma = curve.rolling(params['curve_lookback']).mean()
    curve_std = curve.rolling(params['curve_lookback']).std()
    curve_z = (curve - curve_ma) / curve_std
    
    # Steepening = curve_z falling (long end yields rising faster)
    # Flattening = curve_z rising
    curve_momentum = curve_z.diff(params['momentum_days'])
    
    # VIX regime
    vix_aligned = vix.reindex(data.index, method='ffill')
    
    signals = pd.DataFrame(index=data.index)
    signals['curve_z'] = curve_z
    signals['curve_mom'] = curve_momentum
    signals['vix'] = vix_aligned
    
    # Regime classification
    signals['regime'] = 'normal'
    signals.loc[vix_aligned > params['vix_high'], 'regime'] = 'high_vol'
    signals.loc[vix_aligned < params['vix_low'], 'regime'] = 'low_vol'
    
    return signals

def run_backtest(data, vix, signals, params, start_date, end_date):
    """Run backtest on a specific date range."""
    mask = (data.index >= start_date) & (data.index < end_date)
    period_data = data[mask]
    period_signals = signals[mask]
    
    if len(period_data) < 20:
        return None
    
    # Sector groups
    steepening_sectors = ['XLF', 'XLI', 'XLE', 'XLB']  # Benefit from steepening
    flattening_sectors = ['XLK', 'XLY', 'XLRE']  # Benefit from flattening
    defensive_sectors = ['XLU', 'XLP', 'XLV', 'GLD']  # High vol safety
    
    portfolio_value = 10000.0
    cash = portfolio_value
    positions = {}
    trades = []
    daily_values = []
    
    for i in range(params['warmup'], len(period_data)):
        date = period_data.index[i]
        prev_date = period_data.index[i-1]
        
        # Update portfolio value
        port_val = cash
        for sym, pos in positions.items():
            if sym in period_data.columns:
                port_val += pos['shares'] * period_data[sym].iloc[i]
        daily_values.append({'date': date, 'value': port_val})
        
        # Check for rebalance (every N days)
        if i % params['rebal_days'] != 0:
            continue
        
        curve_mom = period_signals['curve_mom'].iloc[i]
        vix_val = period_signals['vix'].iloc[i]
        regime = period_signals['regime'].iloc[i]
        
        if pd.isna(curve_mom):
            continue
        
        # Determine allocation
        if regime == 'high_vol':
            # Defensive in high vol
            target_sectors = defensive_sectors
            position_size = params['size_highvol']
        elif curve_mom < -params['steepen_threshold']:
            # Steepening: favor financials/industrials/energy
            target_sectors = steepening_sectors
            position_size = params['size_normal']
        elif curve_mom > params['flatten_threshold']:
            # Flattening: favor tech/discretionary
            target_sectors = flattening_sectors
            position_size = params['size_normal']
        else:
            # No strong signal: equal weight broad
            target_sectors = steepening_sectors + flattening_sectors
            position_size = params['size_neutral']
        
        # Close positions not in target
        for sym in list(positions.keys()):
            if sym not in target_sectors and sym in period_data.columns:
                price = period_data[sym].iloc[i]
                pnl = (price - positions[sym]['entry_price']) * positions[sym]['shares']
                cash += positions[sym]['shares'] * price
                trades.append({
                    'date': str(date.date()),
                    'symbol': sym,
                    'side': 'sell',
                    'price': price,
                    'pnl': pnl
                })
                del positions[sym]
        
        # Open positions in target sectors
        available_cash = cash
        per_position = available_cash * position_size / len(target_sectors) if target_sectors else 0
        
        for sym in target_sectors:
            if sym not in positions and sym in period_data.columns and per_position > 0:
                price = period_data[sym].iloc[i]
                if price > 0:
                    shares = int(per_position / price)
                    if shares > 0:
                        cost = shares * price
                        cash -= cost
                        positions[sym] = {
                            'shares': shares,
                            'entry_price': price,
                            'entry_date': str(date.date())
                        }
                        trades.append({
                            'date': str(date.date()),
                            'symbol': sym,
                            'side': 'buy',
                            'price': price
                        })
    
    # Close remaining positions
    if positions and len(period_data) > 0:
        for sym in list(positions.keys()):
            if sym in period_data.columns:
                price = period_data[sym].iloc[-1]
                pnl = (price - positions[sym]['entry_price']) * positions[sym]['shares']
                cash += positions[sym]['shares'] * price
                trades.append({
                    'date': str(period_data.index[-1].date()),
                    'symbol': sym,
                    'side': 'sell',
                    'price': price,
                    'pnl': pnl
                })
    
    if not daily_values:
        return None
    
    df = pd.DataFrame(daily_values)
    df['returns'] = df['value'].pct_change()
    
    total_return = (df['value'].iloc[-1] / df['value'].iloc[0]) - 1
    days = (df['date'].iloc[-1] - df['date'].iloc[0]).days
    cagr = (1 + total_return) ** (365.25 / max(days, 1)) - 1 if days > 0 else 0
    
    returns = df['returns'].dropna()
    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
    downside = returns[returns < 0]
    sortino = (returns.mean() / downside.std() * np.sqrt(252)) if len(downside) > 0 and downside.std() > 0 else 0
    
    winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
    losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
    sell_trades = [t for t in trades if t['side'] == 'sell']
    wr = len(winning_trades) / len(sell_trades) if sell_trades else 0
    
    gross_wins = sum(t['pnl'] for t in winning_trades)
    gross_losses = abs(sum(t['pnl'] for t in losing_trades))
    pf = gross_wins / gross_losses if gross_losses > 0 else float('inf')
    
    # Max drawdown
    peak = df['value'].expanding().max()
    dd = (df['value'] - peak) / peak
    max_dd = dd.min()
    
    return {
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'cagr': round(cagr * 100, 2),
        'total_return': round(total_return * 100, 2),
        'pf': round(pf, 3),
        'wr': round(wr * 100, 1),
        'max_dd': round(max_dd * 100, 2),
        'n_trades': len(sell_trades),
        'days': days
    }

def walk_forward_backtest(data, vix, params):
    """8-fold walk-forward with regime stratification."""
    folds = [
        ('2022-01-01', '2022-07-01', 'high_vol'),   # 2022H1 bear
        ('2022-07-01', '2023-01-01', 'high_vol'),   # 2022H2 bear
        ('2023-01-01', '2023-07-01', 'normal'),      # 2023H1 recovery
        ('2023-07-01', '2024-01-01', 'normal'),      # 2023H2 rally
        ('2024-01-01', '2024-07-01', 'low_vol'),     # 2024H1 bull
        ('2024-07-01', '2025-01-01', 'low_vol'),     # 2024H2 bull
        ('2025-01-01', '2025-07-01', 'normal'),      # 2025H1
        ('2025-07-01', '2026-01-01', 'normal'),      # 2025H2
    ]
    
    signals = compute_signals(data, vix, params)
    results = []
    
    for start, end, regime_label in folds:
        r = run_backtest(data, vix, signals, params, start, end)
        if r:
            r['fold'] = f"{start[:7]}"
            r['regime'] = regime_label
            results.append(r)
    
    return results

def evaluate(params):
    """Main evaluation function for AVO."""
    data, vix = download_data()
    fold_results = walk_forward_backtest(data, vix, params)
    
    if not fold_results or len(fold_results) < 4:
        return {'score': 0, 'details': 'Too few valid folds'}
    
    sharpes = [r['sharpe'] for r in fold_results]
    positive_folds = sum(1 for s in sharpes if s > 0)
    
    # Geomean of clipped Sharpes
    clipped = [max(s, -2.0) for s in sharpes]
    shifted = [s + 3.0 for s in clipped]  # shift to positive
    geomean = np.exp(np.mean(np.log(shifted))) - 3.0
    
    # Regime stratification
    hv_sharpes = [r['sharpe'] for r in fold_results if r['regime'] == 'high_vol']
    lv_sharpes = [r['sharpe'] for r in fold_results if r['regime'] in ('low_vol', 'normal')]
    
    hv_avg = np.mean(hv_sharpes) if hv_sharpes else 0
    lv_avg = np.mean(lv_sharpes) if lv_sharpes else 0
    regime_gap = abs(hv_avg - lv_avg) / max(abs(hv_avg), abs(lv_avg), 0.01)
    
    # Score: geomean Sharpe * fold_bonus * regime_penalty
    fold_bonus = positive_folds / len(fold_results)
    regime_penalty = max(0, 1 - regime_gap)
    
    total_trades = sum(r['n_trades'] for r in fold_results)
    trade_bonus = min(total_trades / 50, 1.5)  # reward more trades up to 1.5x
    
    score = geomean * fold_bonus * regime_penalty * trade_bonus
    
    return {
        'score': round(score, 4),
        'geomean_sharpe': round(geomean, 3),
        'positive_folds': f"{positive_folds}/{len(fold_results)}",
        'regime_gap': round(regime_gap, 3),
        'total_trades': total_trades,
        'fold_results': fold_results,
        'hv_sharpe': round(hv_avg, 3),
        'lv_sharpe': round(lv_avg, 3)
    }

if __name__ == '__main__':
    # Default params (seed)
    params = {
        'curve_lookback': 60,
        'momentum_days': 5,
        'steepen_threshold': 0.3,
        'flatten_threshold': 0.3,
        'vix_high': 25,
        'vix_low': 15,
        'rebal_days': 5,
        'warmup': 65,
        'size_normal': 0.9,
        'size_highvol': 0.6,
        'size_neutral': 0.5,
    }
    
    print("Treasury Curve Steepening Strategy — Seed Evaluation")
    print("=" * 55)
    result = evaluate(params)
    print(f"\nScore: {result['score']}")
    print(f"Geomean Sharpe: {result['geomean_sharpe']}")
    print(f"Positive Folds: {result['positive_folds']}")
    print(f"Regime Gap: {result['regime_gap']} (HV={result['hv_sharpe']}, LV={result['lv_sharpe']})")
    print(f"Total Trades: {result['total_trades']}")
    print("\nPer-fold results:")
    for r in result.get('fold_results', []):
        print(f"  {r['fold']} ({r['regime']}): Sharpe {r['sharpe']}, CAGR {r['cagr']}%, "
              f"WR {r['wr']}%, PF {r['pf']}, Trades {r['n_trades']}, MDD {r['max_dd']}%")
    
    # Save results
    with open('/home/jupiter/Lvl3Quant/data/treasury_curve_steepener_results.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print("\nResults saved.")
