"""
Earnings Overreaction Reversal Backtest v1
============================================
Hypothesis: stocks that gap down on earnings often overreact. 
Buying 1-3 days after the gap down catches the mean reversion.
This should be regime-neutral because overreactions happen in all markets.

Variants:
A) Gap down >5%, buy day after, hold 10d
B) Gap down >8%, buy day after, hold 10d (bigger overreaction)
C) Gap down >5%, wait for RSI<30 confirmation within 3d, hold 10d
D) Gap down >5%, buy only if stock still above 200SMA (quality filter)
E) Gap down >5%, buy only if prior 3 earnings were beats (one-time miss)
F) Adaptive: gap>5% + vol crush (volume day 2 < day 1 = panic fading)
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

UNIVERSE = ['AAPL','MSFT','GOOGL','AMZN','META','NVDA','TSLA','AMD',
            'NFLX','CRM','PLTR','SOFI','HOOD','SNAP','PINS','COIN','RBLX','UBER']

ACCOUNT_SIZE = 669.0
MAX_POSITION_PCT = 0.90
OOT_START = '2022-01-01'
OOT_END = '2026-07-30'
HOLD_DAYS = 10
N_PERMS = 1000
COMMISSION_PCT = 0.001

def download_data():
    print("Downloading price data...")
    data = {}
    for ticker in UNIVERSE:
        try:
            df = yf.download(ticker, start='2021-01-01', end='2026-08-15', progress=False)
            if len(df) > 200:
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                data[ticker] = df
        except:
            pass
    print(f"  Loaded {len(data)} tickers")
    return data

def find_earnings_gaps(data):
    """Find days with large gap downs (>5%) on high volume — earnings proxy."""
    signals = []
    
    for ticker, df in data.items():
        df = df.copy()
        df['gap'] = df['Open'] / df['Close'].shift(1) - 1
        df['ret_day'] = df['Close'] / df['Open'] - 1  # intraday return
        df['vol_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi_5'] = 100 - (100 / (1 + delta.where(delta > 0, 0).rolling(5).mean() / (-delta.where(delta < 0, 0)).rolling(5).mean().replace(0, np.nan)))
        
        df['sma_200'] = df['Close'].rolling(200).mean()
        df['above_200sma'] = (df['Close'] > df['sma_200']).astype(int)
        
        # Forward returns at various offsets
        for d in [5, 10, 15]:
            df[f'fwd_ret_{d}d'] = df['Close'].shift(-d) / df['Close'] - 1
        
        # Next day volume ratio
        df['next_vol_ratio'] = df['vol_ratio'].shift(-1)
        
        # Prior returns
        df['ret_30d_prior'] = df['Close'] / df['Close'].shift(30) - 1
        df['ret_60d_prior'] = df['Close'] / df['Close'].shift(60) - 1
        
        df_oot = df.loc[OOT_START:OOT_END]
        
        for i in range(len(df_oot)):
            row = df_oot.iloc[i]
            if pd.isna(row['gap']) or pd.isna(row.get('fwd_ret_10d')):
                continue
            
            # Large gap down on high volume = earnings event
            if row['gap'] < -0.05 and row['vol_ratio'] > 2.0:
                sig = {
                    'date': df_oot.index[i],
                    'ticker': ticker,
                    'close': row['Close'],
                    'gap_pct': row['gap'],
                    'intraday_ret': row['ret_day'],
                    'vol_ratio': row['vol_ratio'],
                    'rsi': row['rsi'],
                    'rsi_5': row['rsi_5'] if not pd.isna(row['rsi_5']) else 50,
                    'above_200sma': row['above_200sma'],
                    'next_vol_ratio': row['next_vol_ratio'] if not pd.isna(row['next_vol_ratio']) else 1.0,
                    'ret_30d_prior': row['ret_30d_prior'] if not pd.isna(row['ret_30d_prior']) else 0,
                    'fwd_ret_5d': row['fwd_ret_5d'],
                    'fwd_ret_10d': row['fwd_ret_10d'],
                    'fwd_ret_15d': row['fwd_ret_15d'] if not pd.isna(row['fwd_ret_15d']) else np.nan,
                }
                signals.append(sig)
    
    return pd.DataFrame(signals)

def variant_A_simple(sdf):
    """Gap down >5%, buy at close same day, hold 10d."""
    return sdf[sdf['gap_pct'] < -0.05].copy()

def variant_B_big_gap(sdf):
    """Gap down >8%, bigger overreaction."""
    return sdf[sdf['gap_pct'] < -0.08].copy()

def variant_C_rsi_confirm(sdf):
    """Gap >5% + RSI<35 (oversold confirmation)."""
    return sdf[(sdf['gap_pct'] < -0.05) & (sdf['rsi'] < 35)].copy()

def variant_D_quality_filter(sdf):
    """Gap >5% but still above 200SMA — quality stock having bad day."""
    return sdf[(sdf['gap_pct'] < -0.05) & (sdf['above_200sma'] == 1)].copy()

def variant_E_prior_uptrend(sdf):
    """Gap >5% + was up >10% in prior 60d — temporary setback in uptrend."""
    return sdf[(sdf['gap_pct'] < -0.05) & (sdf['ret_30d_prior'] > 0.05)].copy()

def variant_F_vol_crush(sdf):
    """Gap >5% + next day volume drops (panic fading, not new info)."""
    return sdf[(sdf['gap_pct'] < -0.05) & (sdf['next_vol_ratio'] < sdf['vol_ratio'] * 0.7)].copy()

def get_spy_regime(end=None):
    spy = yf.download('SPY', start='2021-01-01', end=end, progress=False)
    spy.columns = [c[0] if isinstance(c, tuple) else c for c in spy.columns]
    spy['sma200'] = spy['Close'].rolling(200).mean()
    spy['bull'] = spy['Close'] > spy['sma200']
    return spy['bull'].to_dict()

def backtest_variant(trades_df, fwd_col='fwd_ret_10d'):
    if trades_df is None or len(trades_df) == 0:
        return None
    
    trades = trades_df.dropna(subset=[fwd_col]).sort_values('date')
    
    pnls = []
    last_entry = None
    
    for _, t in trades.iterrows():
        # Non-overlapping trades
        if last_entry is not None and (t['date'] - last_entry).days < HOLD_DAYS:
            continue
        
        max_cost = ACCOUNT_SIZE * MAX_POSITION_PCT
        shares = int(max_cost / t['close'])
        if shares < 1:
            continue
        
        cost = shares * t['close']
        gross_pnl = cost * t[fwd_col]
        commission = cost * COMMISSION_PCT * 2
        net_pnl = gross_pnl - commission
        
        pnls.append({
            'date': t['date'],
            'ticker': t['ticker'],
            'gap': t['gap_pct'],
            'pnl': net_pnl,
            'ret': t[fwd_col] - COMMISSION_PCT * 2
        })
        last_entry = t['date']
    
    if len(pnls) < 5:
        return None
    
    pdf = pd.DataFrame(pnls)
    
    total_pnl = pdf['pnl'].sum()
    rets = pdf['ret'].values
    sharpe = np.mean(rets) / np.std(rets) * np.sqrt(len(rets) / 4.5) if np.std(rets) > 0 else 0
    
    downside = rets[rets < 0]
    down_std = np.std(downside) if len(downside) > 1 else np.std(rets)
    sortino = np.mean(rets) / down_std * np.sqrt(len(rets) / 4.5) if down_std > 0 else 0
    
    gross_profit = pdf[pdf['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(pdf[pdf['pnl'] < 0]['pnl'].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else 999
    
    cum = pdf['pnl'].cumsum()
    dd = cum - cum.cummax()
    mdd = dd.min()
    
    return {
        'n_trades': len(pdf),
        'total_pnl': round(total_pnl, 2),
        'total_return_pct': round(total_pnl / ACCOUNT_SIZE * 100, 2),
        'avg_pnl': round(pdf['pnl'].mean(), 2),
        'median_pnl': round(pdf['pnl'].median(), 2),
        'win_rate': round((pdf['pnl'] > 0).mean() * 100, 1),
        'profit_factor': round(pf, 2),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'max_drawdown_dollar': round(mdd, 2),
        'max_drawdown_pct': round(mdd / ACCOUNT_SIZE * 100, 1),
        'dates': [str(d)[:10] for d in pdf['date'].values],
        'returns': rets.tolist()
    }

def regime_analysis(trades_df, spy_regime, fwd_col='fwd_ret_10d'):
    if trades_df is None or len(trades_df) == 0:
        return 0, 0, 99
    
    trades = trades_df.dropna(subset=[fwd_col]).sort_values('date')
    bull_r, bear_r = [], []
    
    for _, t in trades.iterrows():
        d = t['date']
        is_bull = True
        for sd, bull in spy_regime.items():
            if isinstance(sd, str):
                sd = pd.Timestamp(sd)
            if sd <= d:
                is_bull = bull
        
        r = t[fwd_col] - COMMISSION_PCT * 2
        if is_bull:
            bull_r.append(r)
        else:
            bear_r.append(r)
    
    def s(arr):
        a = np.array(arr)
        if len(a) < 3 or np.std(a) == 0:
            return 0.0
        return float(np.mean(a) / np.std(a) * np.sqrt(len(a) / 4.5))
    
    sb, sr = s(bull_r), s(bear_r)
    gap = abs(sb - sr) / max(abs(sb), abs(sr), 0.001)
    return round(sb, 3), round(sr, 3), round(gap, 3)

def perm_test(trades_df, all_signals, obs_sharpe, fwd_col='fwd_ret_10d', n=N_PERMS):
    if trades_df is None or len(trades_df) < 5:
        return 1.0
    
    n_t = len(trades_df)
    better = 0
    valid_signals = all_signals.dropna(subset=[fwd_col])
    
    for _ in range(n):
        sample = valid_signals.sample(n=min(n_t, len(valid_signals)), replace=False)
        r = backtest_variant(sample, fwd_col)
        if r and r['sharpe'] >= obs_sharpe:
            better += 1
    
    return round(better / n, 3)

def main():
    print("=" * 60)
    print("EARNINGS OVERREACTION REVERSAL BACKTEST v1")
    print("=" * 60)
    
    data = download_data()
    all_gaps = find_earnings_gaps(data)
    print(f"Total gap-down events (>5%, >2x vol): {len(all_gaps)}")
    
    if len(all_gaps) == 0:
        print("NO GAP EVENTS FOUND. Exiting.")
        return
    
    spy_regime = get_spy_regime(end=OOT_END)
    
    variants = {
        'A_Simple_Gap': variant_A_simple,
        'B_Big_Gap': variant_B_big_gap,
        'C_RSI_Confirm': variant_C_rsi_confirm,
        'D_Quality_Filter': variant_D_quality_filter,
        'E_Prior_Uptrend': variant_E_prior_uptrend,
        'F_Vol_Crush': variant_F_vol_crush,
    }
    
    results = {}
    
    for name, func in variants.items():
        print(f"\n{'='*50}")
        print(f"Variant: {name}")
        trades = func(all_gaps)
        print(f"  Raw signals: {len(trades)}")
        
        bt = backtest_variant(trades)
        if bt is None:
            print(f"  SKIP — too few non-overlapping trades")
            results[name] = {'variant': name, 'status': 'TOO_FEW', 'raw_signals': len(trades)}
            continue
        
        sb, sr, gap = regime_analysis(trades, spy_regime)
        
        print(f"  Running {N_PERMS} permutations...")
        pp = perm_test(trades, all_gaps, bt['sharpe'])
        
        g1 = bt['sharpe'] > 0.5
        g2 = pp < 0.05
        g3 = gap < 0.5
        g4 = bt['max_drawdown_pct'] > -50
        g5 = bt['n_trades'] >= 20
        passed = sum([g1, g2, g3, g4, g5])
        
        gate_str = f"{'✓' if g1 else '✗'} sharpe>0.5 | {'✓' if g2 else '✗'} perm<0.05 | {'✓' if g3 else '✗'} gap<0.5 | {'✓' if g4 else '✗'} mdd>-50% | {'✓' if g5 else '✗'} trades≥20"
        
        result = {
            'variant': name, 'n_trades': bt['n_trades'],
            'total_pnl': bt['total_pnl'], 'total_return_pct': bt['total_return_pct'],
            'avg_pnl': bt['avg_pnl'], 'median_pnl': bt['median_pnl'],
            'win_rate': bt['win_rate'], 'profit_factor': bt['profit_factor'],
            'sharpe': bt['sharpe'], 'sortino': bt['sortino'],
            'max_drawdown_dollar': bt['max_drawdown_dollar'],
            'max_drawdown_pct': bt['max_drawdown_pct'],
            'sharpe_bull': sb, 'sharpe_bear': sr, 'regime_gap': gap,
            'perm_p_value': pp, 'gates_passed': str(passed), 'gate_detail': gate_str,
        }
        results[name] = result
        
        print(f"  Trades: {bt['n_trades']}, PnL: ${bt['total_pnl']}, Sharpe: {bt['sharpe']}")
        print(f"  WR: {bt['win_rate']}%, PF: {bt['profit_factor']}, MDD: {bt['max_drawdown_pct']}%")
        print(f"  Bull: {sb}, Bear: {sr}, Gap: {gap}, Perm: {pp}")
        print(f"  Gates: {passed}/5 — {gate_str}")
    
    valid = {k: v for k, v in results.items() if 'sharpe' in v}
    champion = max(valid, key=lambda k: int(valid[k].get('gates_passed', '0')) * 100 + valid[k].get('sharpe', -99)) if valid else "NONE"
    
    output = {
        'metadata': {
            'script': 'earnings_overreaction_backtest.py',
            'run_date': datetime.now().isoformat(),
            'account_size': ACCOUNT_SIZE,
            'oot_period': f"{OOT_START} to {OOT_END}",
            'hold_days': HOLD_DAYS,
            'n_permutations': N_PERMS,
        },
        'variant_results': results,
        'champion': champion,
        'five_gate_summary': {
            k: {'passed': v.get('gates_passed', '0'), 'sharpe': v.get('sharpe', 0), 'total_pnl': v.get('total_pnl', 0)}
            for k, v in results.items() if 'sharpe' in v
        }
    }
    
    with open('/home/jupiter/Lvl3Quant/data/earnings_overreaction_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print(f"CHAMPION: {champion}")
    print("=" * 60)

if __name__ == '__main__':
    main()
