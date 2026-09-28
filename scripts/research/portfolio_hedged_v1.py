#!/usr/bin/env python3
"""
Portfolio Hedged v1 — Bear Regime Protection

Portfolio MTM v1 showed: Sharpe 0.805, CAGR 10.3%, but regime gap 1.06 (FAIL).
The edge is real but directional (long equity). In bear regimes, it loses.

This version tests hedging overlays to bring regime gap under 0.50:

1. SPY SHORT OVERLAY: Short SPY (or buy SH) when SPY < 200 SMA, proportional to bear severity
2. CASH BUFFER: Hold more cash in bear regimes (already in MTM v1, but expand)
3. VIX LONG OVERLAY: Buy VIXY when VIX spikes (tail hedge)
4. REDUCED SIZE + HEDGE: Smaller positions + partial hedge

Proper daily MTM with transaction costs throughout.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import json, os, sys, warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/portfolio_hedged_v1'
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


def load_spy_full():
    """Load full SPY OHLCV (need more than just close for hedging)."""
    cache = os.path.join(OUTPUT_DIR, 'spy_full_cache.parquet')
    if os.path.exists(cache):
        spy = pd.read_parquet(cache)
    else:
        spy = yf.download('SPY', start='2013-01-01', end='2026-07-22', progress=False)
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)
        spy.index.name = 'date'
        spy = spy.reset_index()
        spy.to_parquet(cache)

    col_map = {c: c.title() for c in spy.columns if c.lower() in ['close','open','high','low','volume']}
    spy = spy.rename(columns=col_map)
    spy['date'] = pd.to_datetime(spy['date'])
    spy = spy.sort_values('date')
    spy['sma_200'] = spy['Close'].rolling(200).mean()
    spy['sma_50'] = spy['Close'].rolling(50).mean()
    spy['regime'] = np.where(spy['Close'] > spy['sma_200'], 'bull', 'bear')
    # Bear severity: how far below 200 SMA (0-1 scale, capped at 20% below)
    spy['bear_severity'] = np.clip((spy['sma_200'] - spy['Close']) / spy['sma_200'], 0, 0.20) / 0.20
    spy['spy_ret'] = spy['Close'].pct_change()
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

    # Core proven signals
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


def run_hedged_portfolio(signals_df, price_data, spy_df,
                         max_positions=15, capital=100000,
                         hedge_mode='none', hedge_pct=0.15):
    """
    Portfolio with hedging overlay.

    hedge_mode options:
    - 'none': No hedging (baseline)
    - 'spy_short': Short SPY proportional to bear severity
    - 'cash_heavy': Hold 50% cash in bear regime (vs normal 0-20%)
    - 'dynamic_hedge': Short SPY scaled by bear_severity (0% bull, up to hedge_pct% bear)
    - 'always_hedge': Always hold hedge_pct in SPY short (permanent tail hedge)
    """
    price_pivot = price_data.pivot_table(index='date', columns='ticker', values='Close').sort_index()
    trading_dates = sorted(price_pivot.index)

    active_positions = []
    closed_trades = []
    cash = float(capital)
    hedge_value = 0.0  # Value of hedge position (negative for short)
    hedge_entry_price = None
    daily_records = []

    for day_idx, date in enumerate(trading_dates):
        if date < pd.Timestamp('2014-01-01'):
            continue

        # Get regime
        regime = spy_df.loc[date, 'regime'] if date in spy_df.index else 'bull'
        bear_severity = spy_df.loc[date, 'bear_severity'] if date in spy_df.index else 0
        spy_price = spy_df.loc[date, 'Close'] if date in spy_df.index else None
        spy_ret = spy_df.loc[date, 'spy_ret'] if date in spy_df.index else 0

        # Determine hedge allocation
        if hedge_mode == 'none':
            target_hedge_frac = 0
        elif hedge_mode == 'spy_short':
            target_hedge_frac = hedge_pct if regime == 'bear' else 0
        elif hedge_mode == 'cash_heavy':
            target_hedge_frac = 0  # Cash handled separately
        elif hedge_mode == 'dynamic_hedge':
            target_hedge_frac = hedge_pct * bear_severity if regime == 'bear' else 0
        elif hedge_mode == 'always_hedge':
            target_hedge_frac = hedge_pct * 0.5  # Half-size permanent hedge
        else:
            target_hedge_frac = 0

        # Update hedge position mark-to-market
        if hedge_value != 0 and spy_ret != 0 and not np.isnan(spy_ret):
            # Short SPY: gains when SPY falls
            hedge_value *= (1 - spy_ret)  # Inverse return

        # Adjust hedge size
        total_nav = cash + sum(
            price_pivot.loc[date, p['ticker']] * p['shares']
            if p['ticker'] in price_pivot.columns and date in price_pivot.index
            and not np.isnan(price_pivot.loc[date, p['ticker']])
            else p['entry_price'] * p['shares']
            for p in active_positions
        ) + abs(hedge_value)

        target_hedge = total_nav * target_hedge_frac
        hedge_diff = target_hedge - abs(hedge_value)

        if abs(hedge_diff) > total_nav * 0.02:  # Rebalance if >2% off target
            if hedge_diff > 0:
                # Increase hedge (costs cash)
                cost = hedge_diff * SLIPPAGE_PCT * 2  # Entry cost
                if cash > hedge_diff + cost:
                    cash -= (hedge_diff + cost)
                    hedge_value += hedge_diff  # Positive = short position notional
            elif hedge_diff < 0 and abs(hedge_value) > 0:
                # Decrease hedge (returns cash)
                reduce = min(abs(hedge_diff), abs(hedge_value))
                cost = reduce * SLIPPAGE_PCT * 2
                cash += (reduce - cost)
                hedge_value -= reduce

        # Determine max positions based on regime + hedge mode
        if hedge_mode == 'cash_heavy' and regime == 'bear':
            effective_max = max_positions // 3  # Very few positions in bear
        elif regime == 'bear':
            effective_max = max_positions // 2
        else:
            effective_max = max_positions

        # Exit positions at hold period
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

            initial_size = capital / max_positions
            position_size = min(cash / max(slots, 1), initial_size)

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
                    'ticker': sig['ticker'],
                    'strategy': sig['strategy'],
                    'entry_date': date,
                    'exit_date': exit_date,
                    'entry_price': entry_price,
                    'shares': shares,
                    'entry_cost': entry_cost,
                })

        # Mark to market
        positions_value = sum(
            price_pivot.loc[date, p['ticker']] * p['shares']
            if p['ticker'] in price_pivot.columns and not np.isnan(price_pivot.loc[date, p['ticker']])
            else p['entry_price'] * p['shares']
            for p in active_positions
        )
        total_value = cash + positions_value + abs(hedge_value)

        daily_records.append({
            'date': date,
            'total_value': total_value,
            'cash': cash,
            'positions_value': positions_value,
            'hedge_value': abs(hedge_value),
            'n_positions': len(active_positions),
            'regime': regime,
            'bear_severity': bear_severity,
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
        'total_return': f"{total_ret:.1%}",
        'cagr': f"{cagr:.1%}",
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'calmar': round(calmar, 3),
        'max_drawdown': f"{max_dd:.1%}",
        'years': round(years, 1),
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
        results[regime] = {'sharpe': round(sharpe, 3), 'days': len(rdf)}

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
        rows.append({'year': yr, 'return': f"{ret:.1%}", 'sharpe': round(sh, 2)})
    return rows


def main():
    print("=" * 70)
    print("PORTFOLIO HEDGED v1 — BEAR REGIME PROTECTION")
    print("Testing hedging overlays to reduce regime gap while preserving edge")
    print("=" * 70)

    print("\n[1] Loading data...")
    stock_df = load_stock_data()
    spy_df = load_spy_full()
    print(f"  {stock_df['ticker'].nunique()} stocks, SPY {len(spy_df)} days")

    print("\n[2] Computing features...")
    stock_df = compute_features(stock_df)

    print("\n[3] Detecting signals...")
    signals = detect_signals(stock_df)
    print(f"  Total: {len(signals)} signals across {signals['strategy'].nunique()} strategies")

    # Test scenarios
    scenarios = {
        'baseline_no_hedge': {'hedge_mode': 'none', 'hedge_pct': 0, 'max_pos': 15},
        'spy_short_15pct': {'hedge_mode': 'spy_short', 'hedge_pct': 0.15, 'max_pos': 15},
        'spy_short_25pct': {'hedge_mode': 'spy_short', 'hedge_pct': 0.25, 'max_pos': 15},
        'spy_short_35pct': {'hedge_mode': 'spy_short', 'hedge_pct': 0.35, 'max_pos': 15},
        'dynamic_hedge_20pct': {'hedge_mode': 'dynamic_hedge', 'hedge_pct': 0.20, 'max_pos': 15},
        'dynamic_hedge_30pct': {'hedge_mode': 'dynamic_hedge', 'hedge_pct': 0.30, 'max_pos': 15},
        'cash_heavy': {'hedge_mode': 'cash_heavy', 'hedge_pct': 0, 'max_pos': 15},
        'always_hedge_10pct': {'hedge_mode': 'always_hedge', 'hedge_pct': 0.10, 'max_pos': 15},
    }

    all_results = {}
    best_name = None
    best_gap = 999
    best_sharpe = -999

    for name, params in scenarios.items():
        print(f"\n{'='*60}")
        print(f"  {name}: hedge={params['hedge_mode']}, pct={params['hedge_pct']}, max={params['max_pos']}")
        print(f"{'='*60}")

        daily_df, trades_df = run_hedged_portfolio(
            signals, stock_df, spy_df,
            max_positions=params['max_pos'],
            hedge_mode=params['hedge_mode'],
            hedge_pct=params['hedge_pct'],
        )

        metrics = compute_metrics(daily_df)
        regime = regime_analysis(daily_df)
        yearly = yearly_perf(daily_df)

        print(f"  Sharpe: {metrics['sharpe']}, CAGR: {metrics['cagr']}, MaxDD: {metrics['max_drawdown']}")
        gap = regime.get('regime_gap', 999)
        gap_pass = regime.get('regime_gap_pass', False)
        print(f"  Regime: bull {regime.get('bull',{}).get('sharpe','?')}, bear {regime.get('bear',{}).get('sharpe','?')}, gap {gap} {'✅' if gap_pass else '❌'}")
        print(f"  Yearly: ", end='')
        for y in yearly:
            print(f"{y['year']}:{y['return']}", end=' ')
        print()

        if len(trades_df) > 0 and 'pnl' in trades_df.columns:
            wr = (trades_df['pnl'] > 0).mean()
            print(f"  Trades: {len(trades_df)}, WR {wr:.1%}, total P&L ${trades_df['pnl'].sum():.0f}")

        all_results[name] = {
            'metrics': metrics,
            'regime': regime,
            'yearly': yearly,
        }

        # Best = lowest gap that still has positive Sharpe, then highest Sharpe
        if gap_pass and metrics['sharpe'] > best_sharpe:
            best_sharpe = metrics['sharpe']
            best_name = name
            best_gap = gap
        elif not best_name and gap < best_gap:
            best_gap = gap
            best_name = name

        daily_df.to_parquet(os.path.join(OUTPUT_DIR, f'{name}_daily.parquet'))

    print(f"\n{'='*70}")
    print(f"COMPARISON TABLE")
    print(f"{'='*70}")
    print(f"{'Scenario':<25} {'Sharpe':>7} {'CAGR':>7} {'MaxDD':>7} {'Gap':>7} {'Pass':>5}")
    print(f"{'-'*60}")
    for name, r in all_results.items():
        gap = r['regime'].get('regime_gap', 'N/A')
        gap_pass = r['regime'].get('regime_gap_pass', False)
        print(f"{name:<25} {r['metrics']['sharpe']:>7.3f} {r['metrics']['cagr']:>7} {r['metrics']['max_drawdown']:>7} {gap if isinstance(gap, str) else f'{gap:.3f}':>7} {'✅' if gap_pass else '❌':>5}")

    print(f"\n  BEST: {best_name} (gap {best_gap:.3f}, Sharpe {best_sharpe:.3f})")

    with open(os.path.join(OUTPUT_DIR, 'report.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n  Results saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
