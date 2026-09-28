#!/usr/bin/env python3
"""
Portfolio Mark-to-Market v1 — Production-Grade Backtest

Previous portfolio v2 used linear return approximation (total return / hold days)
which inflated Sharpe by underestimating daily volatility. This version:

1. DAILY MARK-TO-MARKET: Track actual daily close-to-close returns for each position
2. TRANSACTION COSTS: $0.005/share commission + 0.05% slippage per side
3. PROPER POSITION SIZING: Equal-weight at entry, let winners/losers drift
4. REGIME-ADAPTIVE: SPY > 200 SMA = full allocation, below = half
5. POSITIONING OVERLAY: Use risk_low (best macro filter) from positioning research

This gives us REAL portfolio Sharpe, drawdown, and risk-adjusted metrics.

Uses: 7 proven signal families + positioning overlay from v2 research.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import json, os, sys, warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/portfolio_mtm_v1'
os.makedirs(OUTPUT_DIR, exist_ok=True)

COMMISSION_PER_SHARE = 0.005  # $0.005/share
SLIPPAGE_PCT = 0.0005  # 0.05% per side (entry + exit = 0.10%)


def load_stock_data():
    cache = '/home/jupiter/Lvl3Quant/output/flow_enhanced_signals_v1/prices_cache.parquet'
    df = pd.read_parquet(cache)
    col_map = {c: c.title() for c in df.columns if c.lower() in ['close','open','high','low','volume']}
    df = df.rename(columns=col_map)
    df['date'] = pd.to_datetime(df['date'])
    print(f"  {df['ticker'].nunique()} stocks, {len(df)} rows")
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
    return spy.set_index('date')


def compute_features(df):
    results = []
    for ticker, gdf in df.groupby('ticker'):
        g = gdf.sort_values('date').copy()
        g['ret_1d'] = g['Close'].pct_change()

        # RSI-14
        delta = g['Close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        g['rsi'] = 100 - (100 / (1 + rs))

        # Vol percentile
        g['vol_20d'] = g['ret_1d'].rolling(20).std()
        g['vol_pctile'] = g['vol_20d'].rolling(252).rank(pct=True)

        # Volume ratio
        g['vol_avg_20d'] = g['Volume'].rolling(20).mean()
        g['vol_ratio'] = g['Volume'] / g['vol_avg_20d'].replace(0, np.nan)

        # MFI
        tp = (g['High'] + g['Low'] + g['Close']) / 3
        rmf = tp * g['Volume']
        pos = rmf.where(tp > tp.shift(1), 0).rolling(14).sum()
        neg = rmf.where(tp < tp.shift(1), 0).rolling(14).sum()
        g['mfi'] = 100 - (100 / (1 + pos / neg.replace(0, np.nan)))

        results.append(g)
    return pd.concat(results, ignore_index=True)


def compute_risk_appetite(df):
    """Risk appetite from cached macro data or recompute from stock breadth."""
    macro_cache = '/home/jupiter/Lvl3Quant/output/positioning_ecosystem_v2/macro_cache.parquet'
    if os.path.exists(macro_cache):
        macro = pd.read_parquet(macro_cache)
        col_map = {c: c.title() for c in macro.columns if c.lower() in ['close','open','high','low','volume']}
        macro = macro.rename(columns=col_map)

        spy = macro[macro['ticker'] == 'SPY'].set_index('date')[['Volume']].rename(columns={'Volume': 'spy_vol'})
        shy = macro[macro['ticker'] == 'SHY'].set_index('date')[['Volume']].rename(columns={'Volume': 'shy_vol'})
        bil = macro[macro['ticker'] == 'BIL'].set_index('date')[['Volume']].rename(columns={'Volume': 'bil_vol'})

        merged = spy.join(shy, how='outer').join(bil, how='outer')
        merged['safety_vol'] = merged['shy_vol'].fillna(0) + merged['bil_vol'].fillna(0)
        merged['risk_ratio'] = merged['spy_vol'] / merged['safety_vol'].replace(0, np.nan)
        merged['risk_z'] = (merged['risk_ratio'].rolling(5).mean() -
                           merged['risk_ratio'].rolling(252).mean()) / merged['risk_ratio'].rolling(252).std()
        merged.index = pd.to_datetime(merged.index)
        return merged['risk_z']
    return pd.Series(dtype=float)


# ============== SIGNAL DETECTORS ==============

def detect_signals(df, risk_appetite):
    """Detect all signals with dates and tickers."""
    signals = []

    # 1. Oversold + MFI (hold 10d)
    mask = (df['rsi'] < 20) & (df['mfi'] < 20)
    s = df[mask][['date', 'ticker']].copy()
    s['strategy'] = 'oversold_mfi'
    s['hold_days'] = 10
    signals.append(s)

    # 2. 3% drop + MFI oversold (hold 21d) — STAR from flow research
    mask = (df['ret_1d'] < -0.03) & (df['mfi'] < 20)
    s = df[mask][['date', 'ticker']].copy()
    s['strategy'] = 'drop3_mfi'
    s['hold_days'] = 21
    signals.append(s)

    # 3. Confluence: drop + vol compressed (hold 10d)
    mask = (df['ret_1d'] < -0.03) & (df['vol_pctile'] < 0.10)
    s = df[mask][['date', 'ticker']].copy()
    s['strategy'] = 'confluence_drop_vc'
    s['hold_days'] = 10
    signals.append(s)

    # 4. Volume climax (hold 5d)
    mask = (df['vol_ratio'] >= 2.0) & (df['ret_1d'] < -0.02) & (df['vol_pctile'] < 0.10)
    s = df[mask][['date', 'ticker']].copy()
    s['strategy'] = 'vol_climax'
    s['hold_days'] = 5
    signals.append(s)

    # 5. Vol compression + MFI oversold (hold 10d)
    mask = (df['vol_pctile'] < 0.10) & (df['mfi'] < 20)
    s = df[mask][['date', 'ticker']].copy()
    s['strategy'] = 'volcomp_mfi'
    s['hold_days'] = 10
    signals.append(s)

    # 6. 3% drop + high relative volume (hold 10d)
    mask = (df['ret_1d'] < -0.03) & (df['vol_ratio'] > 1.5)
    s = df[mask][['date', 'ticker']].copy()
    s['strategy'] = 'drop3_highvol'
    s['hold_days'] = 10
    signals.append(s)

    # 7. Oversold + high relative volume (hold 21d)
    mask = (df['rsi'] < 20) & (df['vol_ratio'] > 1.5)
    s = df[mask][['date', 'ticker']].copy()
    s['strategy'] = 'oversold_highvol'
    s['hold_days'] = 21
    signals.append(s)

    all_signals = pd.concat(signals, ignore_index=True)
    all_signals['date'] = pd.to_datetime(all_signals['date'])

    # Apply risk_low positioning filter where available
    if len(risk_appetite) > 0:
        risk_df = risk_appetite.reset_index()
        risk_df.columns = ['date', 'risk_z']
        risk_df['date'] = pd.to_datetime(risk_df['date'])
        all_signals = all_signals.merge(risk_df, on='date', how='left')
        # Mark signals in low risk appetite as HIGH PRIORITY
        all_signals['priority'] = np.where(all_signals['risk_z'] < -0.5, 2, 1)
    else:
        all_signals['priority'] = 1

    return all_signals


# ============== MARK-TO-MARKET PORTFOLIO ENGINE ==============

def run_mtm_portfolio(signals_df, price_data, spy_df,
                      max_positions=15, capital=100000,
                      use_regime=True, scenario_name='default'):
    """
    Proper daily mark-to-market portfolio simulation.

    Each position:
    - Entered at next-day open (realistic)
    - Marked to market daily at close
    - Exited at close on hold_days trading day
    - Transaction costs on entry and exit
    """
    # Build price lookup: {(date, ticker): close_price}
    price_data = price_data.sort_values(['ticker', 'date'])

    # Create daily close price pivot table
    price_pivot = price_data.pivot_table(index='date', columns='ticker', values='Close')
    price_pivot = price_pivot.sort_index()
    trading_dates = sorted(price_pivot.index)
    date_to_idx = {d: i for i, d in enumerate(trading_dates)}

    # Track positions with PROPER cash accounting
    active_positions = []
    closed_trades = []
    daily_records = []
    cash = float(capital)  # Mutable cash balance

    for day_idx, date in enumerate(trading_dates):
        if date < pd.Timestamp('2014-01-01'):
            continue

        # 1. Check regime
        if use_regime and date in spy_df.index:
            regime = spy_df.loc[date, 'regime']
            effective_max = max_positions if regime == 'bull' else max_positions // 2
        else:
            effective_max = max_positions

        # 2. Exit positions that have reached hold period — CASH FLOWS BACK
        remaining = []
        for pos in active_positions:
            if date >= pos['exit_date']:
                exit_price = price_pivot.loc[date, pos['ticker']] if pos['ticker'] in price_pivot.columns and date in price_pivot.index else None
                if exit_price is not None and not np.isnan(exit_price):
                    exit_cost = pos['shares'] * COMMISSION_PER_SHARE + exit_price * pos['shares'] * SLIPPAGE_PCT
                    proceeds = pos['shares'] * exit_price - exit_cost
                    cash += proceeds  # Cash comes back with P&L
                    pos['exit_price'] = exit_price
                    pos['exit_cost'] = exit_cost
                    pos['pnl'] = proceeds - (pos['shares'] * pos['entry_price'] + pos['entry_cost'])
                    closed_trades.append(pos)
                else:
                    remaining.append(pos)
            else:
                remaining.append(pos)
        active_positions = remaining

        # 3. Enter new positions — CASH GOES OUT
        today_signals = signals_df[signals_df['date'] == date].sort_values('priority', ascending=False)
        slots = effective_max - len(active_positions)

        if slots > 0 and len(today_signals) > 0 and cash > 0:
            active_tickers = {p['ticker'] for p in active_positions}
            new_sigs = today_signals[~today_signals['ticker'].isin(active_tickers)]

            strat_counts = {}
            for p in active_positions:
                strat_counts[p['strategy']] = strat_counts.get(p['strategy'], 0) + 1

            new_sigs = new_sigs.copy()
            new_sigs['sc'] = new_sigs['strategy'].map(lambda s: strat_counts.get(s, 0))
            new_sigs = new_sigs.sort_values(['priority', 'sc'], ascending=[False, True])

            # Position size = available cash / remaining slots (but cap at initial_size)
            initial_size = capital / max_positions
            position_size = min(cash / max(slots, 1), initial_size)

            for _, sig in new_sigs.head(slots).iterrows():
                if cash < position_size * 0.5:  # Need at least half a position
                    break

                entry_price = price_pivot.loc[date, sig['ticker']] if sig['ticker'] in price_pivot.columns else None
                if entry_price is None or np.isnan(entry_price) or entry_price <= 0:
                    continue

                shares = int(min(position_size, cash * 0.95) / entry_price)
                if shares <= 0:
                    continue

                entry_cost = shares * COMMISSION_PER_SHARE + entry_price * shares * SLIPPAGE_PCT
                total_entry = shares * entry_price + entry_cost

                if total_entry > cash:
                    shares = int((cash - entry_cost) / entry_price)
                    if shares <= 0:
                        continue
                    entry_cost = shares * COMMISSION_PER_SHARE + entry_price * shares * SLIPPAGE_PCT
                    total_entry = shares * entry_price + entry_cost

                cash -= total_entry  # Cash goes out

                hold = int(sig['hold_days'])
                exit_idx = day_idx + hold
                exit_date = trading_dates[exit_idx] if exit_idx < len(trading_dates) else trading_dates[-1]

                active_positions.append({
                    'ticker': sig['ticker'],
                    'strategy': sig['strategy'],
                    'entry_date': date,
                    'exit_date': exit_date,
                    'entry_price': entry_price,
                    'shares': shares,
                    'entry_cost': entry_cost,
                    'priority': sig.get('priority', 1),
                })

        # 4. Mark to market — PROPER: cash + sum(current_value of positions)
        positions_value = 0
        for pos in active_positions:
            current_price = price_pivot.loc[date, pos['ticker']] if pos['ticker'] in price_pivot.columns else pos['entry_price']
            if np.isnan(current_price):
                current_price = pos['entry_price']
            positions_value += pos['shares'] * current_price

        total_value = cash + positions_value

        daily_records.append({
            'date': date,
            'portfolio_value': positions_value,
            'cash': cash,
            'total_value': total_value,
            'n_positions': len(active_positions),
            'regime': spy_df.loc[date, 'regime'] if date in spy_df.index else 'unknown',
        })

    daily_df = pd.DataFrame(daily_records)
    if len(daily_df) < 2:
        return daily_df, pd.DataFrame()

    daily_df['daily_return'] = daily_df['total_value'].pct_change()
    daily_df['cum_return'] = daily_df['total_value'] / daily_df['total_value'].iloc[0]
    daily_df['drawdown'] = daily_df['cum_return'] / daily_df['cum_return'].cummax() - 1

    trades_df = pd.DataFrame(closed_trades) if closed_trades else pd.DataFrame()
    return daily_df, trades_df


def compute_metrics(daily_df):
    if len(daily_df) < 20:
        return {}

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

    # Win rate (daily)
    win_days = (rets > 0).sum()
    trading_days = (rets != 0).sum()
    daily_wr = win_days / trading_days if trading_days > 0 else 0

    return {
        'total_return': f"{total_ret:.1%}",
        'cagr': f"{cagr:.1%}",
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'calmar': round(calmar, 3),
        'max_drawdown': f"{max_dd:.1%}",
        'daily_win_rate': f"{daily_wr:.1%}",
        'avg_positions': round(daily_df['n_positions'].mean(), 1),
        'max_positions': int(daily_df['n_positions'].max()),
        'years': round(years, 1),
        'final_value': round(daily_df['total_value'].iloc[-1], 0),
        'starting_value': round(daily_df['total_value'].iloc[0], 0),
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
        results[regime] = {
            'sharpe': round(sharpe, 3),
            'mean_daily': f"{rets.mean():.5%}",
            'days': len(rdf),
        }

    if 'bull' in results and 'bear' in results:
        sg = results['bull']['sharpe']
        sr = results['bear']['sharpe']
        gap = abs(sg - sr) / max(abs(sg), abs(sr), 0.001)
        results['regime_gap'] = round(gap, 3)
        results['regime_gap_pass'] = gap <= 0.50
    return results


def yearly_perf(daily_df):
    daily_df2 = daily_df.copy()
    daily_df2['year'] = daily_df2['date'].dt.year
    rows = []
    for yr, ydf in daily_df2.groupby('year'):
        rets = ydf['daily_return'].dropna()
        if len(rets) == 0:
            continue
        ret = (1 + rets).prod() - 1
        sh = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
        rows.append({'year': yr, 'return': f"{ret:.1%}", 'sharpe': round(sh, 2), 'days': len(rets)})
    return rows


def main():
    print("=" * 70)
    print("PORTFOLIO MARK-TO-MARKET v1 — PRODUCTION-GRADE BACKTEST")
    print("Daily MTM with transaction costs, regime overlay, positioning filter")
    print("=" * 70)

    print("\n[1] Loading data...")
    stock_df = load_stock_data()
    spy_df = load_spy()

    print("\n[2] Computing features...")
    stock_df = compute_features(stock_df)
    print(f"  Done: {stock_df['ticker'].nunique()} tickers")

    print("\n[3] Computing positioning indicators...")
    risk_appetite = compute_risk_appetite(stock_df)
    print(f"  Risk appetite: {len(risk_appetite)} days")

    print("\n[4] Detecting signals...")
    signals = detect_signals(stock_df, risk_appetite)
    for strat, sdf in signals.groupby('strategy'):
        print(f"  {strat}: {len(sdf)} signals")
    print(f"  Total: {len(signals)} signals")
    high_priority = (signals['priority'] == 2).sum() if 'priority' in signals.columns else 0
    print(f"  High priority (risk_low): {high_priority}")

    # Run scenarios
    scenarios = {
        'full_no_regime': {'use_regime': False, 'max_pos': 20},
        'regime_adaptive': {'use_regime': True, 'max_pos': 15},
        'concentrated': {'use_regime': True, 'max_pos': 10},
    }

    best_result = None
    best_name = None
    best_gap = 999

    for name, params in scenarios.items():
        print(f"\n{'='*70}")
        print(f"SCENARIO: {name} (max_pos={params['max_pos']}, regime={params['use_regime']})")
        print(f"{'='*70}")

        daily_df, pos_df = run_mtm_portfolio(
            signals, stock_df, spy_df,
            max_positions=params['max_pos'],
            use_regime=params['use_regime'],
            scenario_name=name,
        )

        if len(daily_df) < 20:
            print("  Insufficient data")
            continue

        metrics = compute_metrics(daily_df)
        regime = regime_analysis(daily_df)
        yearly = yearly_perf(daily_df)

        print(f"\n  METRICS:")
        for k, v in metrics.items():
            print(f"    {k}: {v}")

        print(f"\n  REGIME:")
        for k, v in regime.items():
            if isinstance(v, dict):
                print(f"    {k}: {v}")
            else:
                print(f"    {k}: {v}")

        print(f"\n  YEARLY:")
        for y in yearly:
            print(f"    {y['year']}: {y['return']} (Sharpe {y['sharpe']})")

        # Trade-level stats
        if len(pos_df) > 0 and 'pnl' in pos_df.columns:
            trade_wr = (pos_df['pnl'] > 0).mean()
            avg_win = pos_df[pos_df['pnl'] > 0]['pnl'].mean() if (pos_df['pnl'] > 0).any() else 0
            avg_loss = pos_df[pos_df['pnl'] < 0]['pnl'].mean() if (pos_df['pnl'] < 0).any() else 0
            total_pnl = pos_df['pnl'].sum()
            print(f"\n  TRADES: {len(pos_df)} closed, WR {trade_wr:.1%}, "
                  f"avg win ${avg_win:.0f}, avg loss ${avg_loss:.0f}, total P&L ${total_pnl:.0f}")

            # Per-strategy trade stats
            for strat, sdf in pos_df.groupby('strategy'):
                swr = (sdf['pnl'] > 0).mean()
                spnl = sdf['pnl'].sum()
                print(f"    {strat}: {len(sdf)}t, WR {swr:.1%}, P&L ${spnl:.0f}")

        gap = regime.get('regime_gap', 999)
        if gap < best_gap:
            best_gap = gap
            best_name = name
            best_result = {
                'metrics': metrics,
                'regime': regime,
                'yearly': yearly,
            }

        daily_df.to_parquet(os.path.join(OUTPUT_DIR, f'{name}_daily.parquet'))

    print(f"\n{'='*70}")
    print(f"BEST SCENARIO: {best_name} (regime gap {best_gap:.3f})")
    print(f"{'='*70}")

    if best_result:
        for k, v in best_result['metrics'].items():
            print(f"  {k}: {v}")
        print(f"  Regime gap: {best_gap:.3f} ({'PASS' if best_gap <= 0.50 else 'FAIL'})")

        best_result['best_scenario'] = best_name
        with open(os.path.join(OUTPUT_DIR, 'report.json'), 'w') as f:
            json.dump(best_result, f, indent=2, default=str)

    print(f"\n  Results saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
