#!/usr/bin/env python3
"""
Portfolio Vol-Targeting v1 — Equalize Daily Volatility Across Regimes

Key insight from hedged v1: signal-level regime gap ≠ portfolio MTM regime gap.
Signals work in both regimes (entry-to-exit), but daily MTM swings are bigger
in bears. The fix isn't hedging (which fights the alpha) — it's VOL-TARGETING:
scale position count/size by inverse realized volatility.

When vol is high (bear): fewer, smaller positions → lower daily P&L swings
When vol is low (bull): more, larger positions → capture more of the calm edge

This should equalize the Sharpe ratio across regimes without fighting direction.

Also fixes the 2020 COVID compounding distortion by capping position sizes.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import json, os, sys, warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/portfolio_voltarget_v1'
os.makedirs(OUTPUT_DIR, exist_ok=True)

COMMISSION_PER_SHARE = 0.005
SLIPPAGE_PCT = 0.0005


def load_stock_data():
    cache = '/home/jupiter/Lvl3Quant/output/flow_enhanced_signals_v1/prices_cache.parquet'
    df = pd.read_parquet(cache)
    col_map = {c: c.title() for c in df.columns if c.lower() in ['close','open','high','low','volume']}
    df = df.rename(columns=col_map)
    df['date'] = pd.to_datetime(df['date'])
    return df


def load_spy():
    cache = '/home/jupiter/Lvl3Quant/output/flow_enhanced_signals_v1/spy_cache.parquet'
    spy = pd.read_parquet(cache)
    col_map = {c: c.title() for c in spy.columns if c.lower() == 'close'}
    spy = spy.rename(columns=col_map)
    spy['date'] = pd.to_datetime(spy['date'])
    spy = spy.sort_values('date')
    spy['sma_200'] = spy['Close'].rolling(200).mean()
    spy['regime'] = np.where(spy['Close'] > spy['sma_200'], 'bull', 'bear')
    spy['spy_ret'] = spy['Close'].pct_change()
    # Realized vol (20d annualized)
    spy['realized_vol'] = spy['spy_ret'].rolling(20).std() * np.sqrt(252)
    # Long-term median vol for scaling
    spy['vol_median'] = spy['realized_vol'].expanding(min_periods=252).median()
    # Vol ratio: current / median (>1 = high vol, <1 = low vol)
    spy['vol_ratio'] = spy['realized_vol'] / spy['vol_median'].replace(0, np.nan)
    return spy.set_index('date')


def compute_features(df):
    results = []
    for ticker, gdf in df.groupby('ticker'):
        g = gdf.sort_values('date').copy()
        g['ret_1d'] = g['Close'].pct_change()
        delta = g['Close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        g['rsi'] = 100 - (100 / (1 + rs))
        g['vol_20d'] = g['ret_1d'].rolling(20).std()
        g['vol_pctile'] = g['vol_20d'].rolling(252).rank(pct=True)
        g['vol_avg_20d'] = g['Volume'].rolling(20).mean()
        g['vol_ratio'] = g['Volume'] / g['vol_avg_20d'].replace(0, np.nan)
        tp = (g['High'] + g['Low'] + g['Close']) / 3
        rmf = tp * g['Volume']
        pos = rmf.where(tp > tp.shift(1), 0).rolling(14).sum()
        neg = rmf.where(tp < tp.shift(1), 0).rolling(14).sum()
        g['mfi'] = 100 - (100 / (1 + pos / neg.replace(0, np.nan)))
        results.append(g)
    return pd.concat(results, ignore_index=True)


def detect_signals(df):
    signals = []
    configs = [
        ('oversold_mfi', (df['rsi'] < 20) & (df['mfi'] < 20), 10),
        ('drop3_mfi', (df['ret_1d'] < -0.03) & (df['mfi'] < 20), 21),
        ('confluence_drop_vc', (df['ret_1d'] < -0.03) & (df['vol_pctile'] < 0.10), 10),
        ('vol_climax', (df['vol_ratio'] >= 2.0) & (df['ret_1d'] < -0.02) & (df['vol_pctile'] < 0.10), 5),
        ('volcomp_mfi', (df['vol_pctile'] < 0.10) & (df['mfi'] < 20), 10),
        ('drop3_highvol', (df['ret_1d'] < -0.03) & (df['vol_ratio'] > 1.5), 10),
        ('oversold_highvol', (df['rsi'] < 20) & (df['vol_ratio'] > 1.5), 21),
    ]
    for name, mask, hold in configs:
        s = df[mask][['date', 'ticker']].copy()
        s['strategy'] = name
        s['hold_days'] = hold
        signals.append(s)
    all_sigs = pd.concat(signals, ignore_index=True)
    all_sigs['date'] = pd.to_datetime(all_sigs['date'])
    return all_sigs


def run_voltarget_portfolio(signals_df, price_data, spy_df,
                            base_max_positions=15, capital=100000,
                            vol_target_mode='none', target_vol=0.15,
                            max_position_cap=None):
    """
    Vol-targeted portfolio.

    vol_target_mode:
    - 'none': Fixed position count (baseline)
    - 'inverse_vol': Scale max positions by median_vol/current_vol
    - 'target_vol': Scale total exposure to achieve target portfolio vol
    - 'capped_inverse': inverse_vol but cap position size at initial capital/max_pos
    """
    price_pivot = price_data.pivot_table(index='date', columns='ticker', values='Close').sort_index()
    trading_dates = sorted(price_pivot.index)

    active_positions = []
    closed_trades = []
    cash = float(capital)
    daily_records = []

    # Position size cap to prevent COVID-style compounding
    max_pos_size = capital / base_max_positions * 1.5 if max_position_cap else float('inf')

    for day_idx, date in enumerate(trading_dates):
        if date < pd.Timestamp('2014-01-01'):
            continue

        # Get vol scaling factor
        vol_ratio = spy_df.loc[date, 'vol_ratio'] if date in spy_df.index else 1.0
        regime = spy_df.loc[date, 'regime'] if date in spy_df.index else 'bull'

        if np.isnan(vol_ratio) or vol_ratio <= 0:
            vol_ratio = 1.0

        # Determine effective max positions based on vol
        if vol_target_mode == 'none':
            effective_max = base_max_positions
            size_scalar = 1.0
        elif vol_target_mode == 'inverse_vol':
            # When vol is 2x median → half the positions
            # When vol is 0.5x median → double positions (capped at 2x base)
            effective_max = max(3, min(int(base_max_positions / vol_ratio), base_max_positions * 2))
            size_scalar = 1.0 / max(vol_ratio, 0.5)  # Also scale position size
        elif vol_target_mode == 'target_vol':
            # Target specific portfolio vol
            current_vol = spy_df.loc[date, 'realized_vol'] if date in spy_df.index else 0.15
            if np.isnan(current_vol) or current_vol <= 0:
                current_vol = 0.15
            vol_scalar = target_vol / max(current_vol, 0.05)
            effective_max = max(3, min(int(base_max_positions * vol_scalar), base_max_positions * 2))
            size_scalar = min(vol_scalar, 2.0)
        elif vol_target_mode == 'capped_inverse':
            effective_max = max(3, min(int(base_max_positions / vol_ratio), base_max_positions * 2))
            size_scalar = 1.0  # Don't scale size, only count
        else:
            effective_max = base_max_positions
            size_scalar = 1.0

        # Exit positions
        remaining = []
        for pos in active_positions:
            if date >= pos['exit_date']:
                exit_price = price_pivot.loc[date, pos['ticker']] if pos['ticker'] in price_pivot.columns else None
                if exit_price is not None and not np.isnan(exit_price):
                    exit_cost = pos['shares'] * COMMISSION_PER_SHARE + exit_price * pos['shares'] * SLIPPAGE_PCT
                    proceeds = pos['shares'] * exit_price - exit_cost
                    cash += proceeds
                    pos['exit_price'] = exit_price
                    pos['pnl'] = proceeds - (pos['shares'] * pos['entry_price'] + pos['entry_cost'])
                    closed_trades.append(pos)
                else:
                    remaining.append(pos)
            else:
                remaining.append(pos)
        active_positions = remaining

        # Enter new positions
        today_signals = signals_df[signals_df['date'] == date]
        slots = effective_max - len(active_positions)

        if slots > 0 and len(today_signals) > 0 and cash > 0:
            active_tickers = {p['ticker'] for p in active_positions}
            new_sigs = today_signals[~today_signals['ticker'].isin(active_tickers)]
            strat_counts = {}
            for p in active_positions:
                strat_counts[p['strategy']] = strat_counts.get(p['strategy'], 0) + 1
            new_sigs = new_sigs.copy()
            new_sigs['sc'] = new_sigs['strategy'].map(lambda s: strat_counts.get(s, 0))
            new_sigs = new_sigs.sort_values('sc')

            base_size = capital / base_max_positions
            position_size = min(base_size * size_scalar, max_pos_size, cash / max(slots, 1))

            for _, sig in new_sigs.head(slots).iterrows():
                if cash < position_size * 0.3:
                    break
                entry_price = price_pivot.loc[date, sig['ticker']] if sig['ticker'] in price_pivot.columns else None
                if entry_price is None or np.isnan(entry_price) or entry_price <= 0:
                    continue
                shares = int(min(position_size, cash * 0.90) / entry_price)
                if shares <= 0:
                    continue
                entry_cost = shares * COMMISSION_PER_SHARE + entry_price * shares * SLIPPAGE_PCT
                total_entry = shares * entry_price + entry_cost
                if total_entry > cash:
                    continue
                cash -= total_entry
                hold = int(sig['hold_days'])
                exit_idx = day_idx + hold
                exit_date = trading_dates[exit_idx] if exit_idx < len(trading_dates) else trading_dates[-1]
                active_positions.append({
                    'ticker': sig['ticker'], 'strategy': sig['strategy'],
                    'entry_date': date, 'exit_date': exit_date,
                    'entry_price': entry_price, 'shares': shares,
                    'entry_cost': entry_cost,
                })

        # Mark to market
        positions_value = sum(
            price_pivot.loc[date, p['ticker']] * p['shares']
            if p['ticker'] in price_pivot.columns and not np.isnan(price_pivot.loc[date, p['ticker']])
            else p['entry_price'] * p['shares']
            for p in active_positions
        )
        total_value = cash + positions_value

        daily_records.append({
            'date': date, 'total_value': total_value, 'cash': cash,
            'positions_value': positions_value, 'n_positions': len(active_positions),
            'regime': regime, 'vol_ratio': vol_ratio,
            'effective_max': effective_max,
        })

    daily_df = pd.DataFrame(daily_records)
    daily_df['daily_return'] = daily_df['total_value'].pct_change()
    daily_df['cum_return'] = daily_df['total_value'] / daily_df['total_value'].iloc[0]
    daily_df['drawdown'] = daily_df['cum_return'] / daily_df['cum_return'].cummax() - 1
    trades_df = pd.DataFrame(closed_trades) if closed_trades else pd.DataFrame()
    return daily_df, trades_df


def compute_metrics(daily_df):
    rets = daily_df['daily_return'].dropna()
    ann = 252
    total_ret = daily_df['cum_return'].iloc[-1] - 1
    years = len(rets) / ann
    cagr = (1 + total_ret) ** (1 / max(years, 0.1)) - 1
    sharpe = rets.mean() / rets.std() * np.sqrt(ann) if rets.std() > 0 else 0
    downside = rets[rets < 0].std()
    sortino = rets.mean() / downside * np.sqrt(ann) if downside > 0 else 0
    max_dd = daily_df['drawdown'].min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    return {
        'cagr': f"{cagr:.1%}", 'sharpe': round(sharpe, 3), 'sortino': round(sortino, 3),
        'calmar': round(calmar, 3), 'max_drawdown': f"{max_dd:.1%}",
        'final_value': round(daily_df['total_value'].iloc[-1], 0),
    }


def regime_analysis(daily_df):
    results = {}
    for regime in ['bull', 'bear']:
        rdf = daily_df[daily_df['regime'] == regime]
        if len(rdf) < 20:
            continue
        rets = rdf['daily_return'].dropna()
        if rets.std() == 0:
            continue
        sharpe = rets.mean() / rets.std() * np.sqrt(252)
        vol = rets.std() * np.sqrt(252)
        results[regime] = {'sharpe': round(sharpe, 3), 'ann_vol': round(vol, 3), 'days': len(rdf)}
    if 'bull' in results and 'bear' in results:
        sg = results['bull']['sharpe']
        sr = results['bear']['sharpe']
        gap = abs(sg - sr) / max(abs(sg), abs(sr), 0.001)
        results['regime_gap'] = round(gap, 3)
        results['regime_gap_pass'] = gap <= 0.50
        # Vol ratio between regimes (ideal = 1.0)
        results['vol_ratio_regimes'] = round(results['bear']['ann_vol'] / max(results['bull']['ann_vol'], 0.001), 2)
    return results


def yearly_perf(daily_df):
    daily_df2 = daily_df.copy()
    daily_df2['year'] = daily_df2['date'].dt.year
    rows = []
    for yr, ydf in daily_df2.groupby('year'):
        rets = ydf['daily_return'].dropna()
        if len(rets) == 0: continue
        ret = (1 + rets).prod() - 1
        sh = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
        rows.append({'year': yr, 'return': f"{ret:.1%}", 'sharpe': round(sh, 2)})
    return rows


def main():
    print("=" * 70)
    print("PORTFOLIO VOL-TARGETING v1")
    print("Scale positions by inverse volatility to equalize regime performance")
    print("=" * 70)

    print("\n[1] Loading data...")
    stock_df = load_stock_data()
    spy_df = load_spy()
    print(f"  {stock_df['ticker'].nunique()} stocks")
    print(f"  SPY vol range: {spy_df['realized_vol'].min():.2f} - {spy_df['realized_vol'].max():.2f}")
    print(f"  SPY vol median: {spy_df['vol_median'].dropna().iloc[-1]:.2f}")

    print("\n[2] Computing features...")
    stock_df = compute_features(stock_df)

    print("\n[3] Detecting signals...")
    signals = detect_signals(stock_df)
    print(f"  {len(signals)} signals")

    scenarios = {
        'baseline': {'vol_target_mode': 'none', 'max_position_cap': False},
        'baseline_capped': {'vol_target_mode': 'none', 'max_position_cap': True},
        'inverse_vol': {'vol_target_mode': 'inverse_vol', 'max_position_cap': True},
        'capped_inverse': {'vol_target_mode': 'capped_inverse', 'max_position_cap': True},
        'target_10pct': {'vol_target_mode': 'target_vol', 'target_vol': 0.10, 'max_position_cap': True},
        'target_15pct': {'vol_target_mode': 'target_vol', 'target_vol': 0.15, 'max_position_cap': True},
        'target_20pct': {'vol_target_mode': 'target_vol', 'target_vol': 0.20, 'max_position_cap': True},
    }

    all_results = {}

    for name, params in scenarios.items():
        print(f"\n  --- {name} ---")
        daily_df, trades_df = run_voltarget_portfolio(
            signals, stock_df, spy_df,
            vol_target_mode=params.get('vol_target_mode', 'none'),
            target_vol=params.get('target_vol', 0.15),
            max_position_cap=params.get('max_position_cap', False),
        )

        metrics = compute_metrics(daily_df)
        regime = regime_analysis(daily_df)
        yearly = yearly_perf(daily_df)

        gap = regime.get('regime_gap', 999)
        gap_pass = regime.get('regime_gap_pass', False)
        vol_ratio = regime.get('vol_ratio_regimes', '?')

        print(f"  Sharpe {metrics['sharpe']}, CAGR {metrics['cagr']}, MaxDD {metrics['max_drawdown']}, "
              f"gap {gap:.3f} {'✅' if gap_pass else '❌'}, "
              f"vol_ratio {vol_ratio}, final ${metrics['final_value']:.0f}")

        bull_s = regime.get('bull', {}).get('sharpe', '?')
        bear_s = regime.get('bear', {}).get('sharpe', '?')
        print(f"  Bull Sharpe {bull_s}, Bear Sharpe {bear_s}")

        if len(trades_df) > 0 and 'pnl' in trades_df.columns:
            print(f"  Trades: {len(trades_df)}, WR {(trades_df['pnl']>0).mean():.1%}")

        # Avg positions by regime
        bull_pos = daily_df[daily_df['regime']=='bull']['n_positions'].mean()
        bear_pos = daily_df[daily_df['regime']=='bear']['n_positions'].mean()
        print(f"  Avg positions: bull {bull_pos:.1f}, bear {bear_pos:.1f}")

        all_results[name] = {
            'metrics': metrics, 'regime': regime, 'yearly': yearly,
            'avg_bull_pos': round(bull_pos, 1), 'avg_bear_pos': round(bear_pos, 1),
        }

        daily_df.to_parquet(os.path.join(OUTPUT_DIR, f'{name}_daily.parquet'))

    # Summary table
    print(f"\n{'='*90}")
    print(f"{'Scenario':<20} {'Sharpe':>7} {'CAGR':>7} {'MaxDD':>7} {'Gap':>7} {'Pass':>5} {'VolRat':>7} {'BullS':>7} {'BearS':>7}")
    print(f"{'-'*90}")
    for name, r in all_results.items():
        gap = r['regime'].get('regime_gap', 999)
        gap_pass = r['regime'].get('regime_gap_pass', False)
        vr = r['regime'].get('vol_ratio_regimes', '?')
        bs = r['regime'].get('bull', {}).get('sharpe', '?')
        brs = r['regime'].get('bear', {}).get('sharpe', '?')
        print(f"{name:<20} {r['metrics']['sharpe']:>7.3f} {r['metrics']['cagr']:>7} {r['metrics']['max_drawdown']:>7} "
              f"{gap:>7.3f} {'✅' if gap_pass else '❌':>5} {vr:>7} {bs:>7} {brs:>7}")

    # Find best passing scenario
    passing = {k: v for k, v in all_results.items() if v['regime'].get('regime_gap_pass', False)}
    if passing:
        best = max(passing.items(), key=lambda x: x[1]['metrics']['sharpe'])
        print(f"\n  ✅ BEST PASSING: {best[0]} — Sharpe {best[1]['metrics']['sharpe']}, CAGR {best[1]['metrics']['cagr']}, "
              f"gap {best[1]['regime']['regime_gap']}")
    else:
        best = min(all_results.items(), key=lambda x: x[1]['regime'].get('regime_gap', 999))
        print(f"\n  ❌ NO SCENARIO PASSES. Closest: {best[0]} — gap {best[1]['regime'].get('regime_gap', 999):.3f}")

    with open(os.path.join(OUTPUT_DIR, 'report.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
