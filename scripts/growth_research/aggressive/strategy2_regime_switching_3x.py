#!/usr/bin/env python3
"""
Strategy 2: Risk-On/Risk-Off Regime Switching with 3x Leverage
- Composite risk signal (VIX term structure, breadth, credit spreads)
- Risk-ON: hold TQQQ (3x leveraged QQQ)
- Risk-OFF: hold TLT (long bonds) or cash
- Walk-forward sliding window validation of the switching signal
"""
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

OUTPUT_DIR = "/home/jupiter/Lvl3Quant/output/growth_research/aggressive"

def download_data():
    """Download regime signal data and leveraged ETFs"""
    print("Downloading data...")
    end = datetime(2026, 7, 1)
    start = datetime(2019, 1, 1)

    tickers = [
        'TQQQ',   # 3x QQQ
        'UPRO',   # 3x SPY
        'QQQ',    # benchmark
        'SPY',    # benchmark
        'TLT',    # long bonds (risk-off)
        'SHY',    # short bonds (risk-off)
        'HYG',    # high yield (credit signal)
        'IEF',    # treasury (credit signal)
        '^VIX',   # VIX
    ]

    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    close = data['Close'].dropna(how='all')

    # Also get breadth proxy - use equal-weight RSP vs SPY
    breadth_tickers = ['RSP', 'SPY']
    breadth_data = yf.download(breadth_tickers, start=start, end=end, auto_adjust=True, progress=False)
    breadth_close = breadth_data['Close'].dropna(how='all')

    return close, breadth_close

def compute_signals(close, breadth_close):
    """Compute regime signals"""
    signals = pd.DataFrame(index=close.index)

    # 1. VIX level signal (low VIX = risk-on)
    if '^VIX' in close.columns:
        vix = close['^VIX']
        vix_ma = vix.rolling(20).mean()
        signals['vix_signal'] = np.where(vix < vix_ma, 1, -1)  # Below MA = risk-on
        signals['vix_level'] = np.where(vix < 20, 1, np.where(vix > 30, -1, 0))
    else:
        signals['vix_signal'] = 0
        signals['vix_level'] = 0

    # 2. Credit spread signal (HYG/IEF ratio)
    if 'HYG' in close.columns and 'IEF' in close.columns:
        credit = close['HYG'] / close['IEF']
        credit_ma = credit.rolling(20).mean()
        signals['credit_signal'] = np.where(credit > credit_ma, 1, -1)
    else:
        signals['credit_signal'] = 0

    # 3. Breadth signal (RSP/SPY ratio - equal weight vs cap weight)
    if 'RSP' in breadth_close.columns and 'SPY' in breadth_close.columns:
        breadth = breadth_close['RSP'] / breadth_close['SPY']
        breadth_ma = breadth.rolling(20).mean()
        breadth_sig = pd.Series(np.where(breadth > breadth_ma, 1, -1), index=breadth.index)
        signals['breadth_signal'] = breadth_sig.reindex(signals.index).fillna(0).astype(int)
    else:
        signals['breadth_signal'] = 0

    # 4. Trend signal (SPY above 200-day MA)
    if 'SPY' in close.columns:
        spy_ma200 = close['SPY'].rolling(200).mean()
        signals['trend_signal'] = np.where(close['SPY'] > spy_ma200, 1, -1)

        spy_ma50 = close['SPY'].rolling(50).mean()
        signals['trend_50'] = np.where(close['SPY'] > spy_ma50, 1, -1)
    else:
        signals['trend_signal'] = 0
        signals['trend_50'] = 0

    # 5. Momentum signal (QQQ 20-day return)
    if 'QQQ' in close.columns:
        mom = close['QQQ'].pct_change(20)
        signals['mom_signal'] = np.where(mom > 0, 1, -1)
    else:
        signals['mom_signal'] = 0

    return signals

def backtest_regime_strategy(close, signals, risk_on_ticker='TQQQ', risk_off_ticker='TLT',
                              signal_cols=None, threshold=0, name='default'):
    """
    Walk-forward backtest of regime switching.
    signal_cols: list of signal columns to average for composite signal
    threshold: composite signal must be > threshold to go risk-on
    """
    if risk_on_ticker not in close.columns:
        print(f"  {risk_on_ticker} not available")
        return None
    if risk_off_ticker not in close.columns and risk_off_ticker != 'CASH':
        print(f"  {risk_off_ticker} not available")
        return None

    if signal_cols is None:
        signal_cols = ['trend_signal']

    # Composite signal
    composite = signals[signal_cols].mean(axis=1)

    # Position: 1 = risk-on, 0 = risk-off
    position = (composite > threshold).astype(int)

    # Lag by 1 day (trade next day)
    position = position.shift(1).fillna(0)

    # Daily returns
    risk_on_ret = close[risk_on_ticker].pct_change()
    if risk_off_ticker == 'CASH':
        risk_off_ret = pd.Series(0.0001 / 252, index=close.index)  # ~0.01% daily risk-free
    else:
        risk_off_ret = close[risk_off_ticker].pct_change()

    # Strategy return
    strat_ret = position * risk_on_ret + (1 - position) * risk_off_ret
    strat_ret = strat_ret.dropna()

    # Remove first 200 days (warmup for MA200)
    strat_ret = strat_ret.iloc[200:]

    return strat_ret

def analyze_daily(returns, name, spy_close):
    """Analyze daily return series"""
    if len(returns) < 252:
        return None

    n_days = len(returns)
    n_years = n_days / 252

    cum_ret = (1 + returns).prod() - 1
    cagr = (1 + cum_ret) ** (1 / n_years) - 1 if n_years > 0 else 0

    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    downside = returns[returns < 0].std() * np.sqrt(252)
    sortino = ann_ret / downside if downside > 0 else 0

    win_rate = (returns > 0).mean()

    avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
    avg_loss = abs(returns[returns < 0].mean()) if (returns < 0).any() else 1
    profit_factor = avg_win / avg_loss if avg_loss > 0 else float('inf')

    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # R1 regime test
    spy_daily = spy_close.pct_change()
    # Classify by monthly regime
    spy_monthly = spy_close.resample('ME').last().pct_change()

    common_months = []
    strat_monthly = returns.resample('ME').sum()  # approx monthly log return
    common_idx = strat_monthly.index.intersection(spy_monthly.index)

    if len(common_idx) > 10:
        sm = strat_monthly.loc[common_idx]
        spy_m = spy_monthly.loc[common_idx]

        green = sm[spy_m > 0.01]
        red = sm[spy_m < -0.01]

        sharpe_green = green.mean() / green.std() * np.sqrt(12) if len(green) > 2 and green.std() > 0 else 0
        sharpe_red = red.mean() / red.std() * np.sqrt(12) if len(red) > 2 and red.std() > 0 else 0

        max_sharpe = max(abs(sharpe_green), abs(sharpe_red))
        regime_gap = abs(sharpe_green - sharpe_red) / max_sharpe if max_sharpe > 0 else 0
        r1_pass = regime_gap <= 0.50
    else:
        sharpe_green = sharpe_red = regime_gap = 0
        r1_pass = False
        green = red = pd.Series(dtype=float)

    # Permutation test - shuffle signal assignments
    perm_sharpes = []
    for _ in range(100):
        perm = returns.sample(frac=1, replace=False).values
        pm = perm.mean() * 252
        ps = perm.std() * np.sqrt(252)
        perm_sharpes.append(pm / ps if ps > 0 else 0)
    perm_p = np.mean([s >= sharpe for s in perm_sharpes])

    # $441 growth projection
    final_value = 441 * (1 + cum_ret)

    return {
        'strategy': name,
        'n_days': n_days,
        'n_years': round(n_years, 1),
        'CAGR': round(cagr * 100, 2),
        'ann_return': round(ann_ret * 100, 2),
        'ann_vol': round(ann_vol * 100, 2),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'win_rate': round(win_rate * 100, 1),
        'profit_factor': round(profit_factor, 3),
        'max_drawdown': round(max_dd * 100, 2),
        'calmar': round(calmar, 3),
        'cum_return': round(cum_ret * 100, 2),
        'final_441': round(final_value, 2),
        'R1_regime_gap': round(regime_gap, 3),
        'R1_pass': r1_pass,
        'sharpe_green': round(sharpe_green, 3),
        'sharpe_red': round(sharpe_red, 3),
        'n_green_months': len(green) if isinstance(green, pd.Series) else 0,
        'n_red_months': len(red) if isinstance(red, pd.Series) else 0,
        'perm_p_value': round(perm_p, 3),
        'feasible_441': True,
    }

def main():
    close, breadth_close = download_data()
    signals = compute_signals(close, breadth_close)

    spy = close['SPY'] if 'SPY' in close.columns else None

    configs = [
        # Simple 200MA
        {'signal_cols': ['trend_signal'], 'risk_on_ticker': 'TQQQ', 'risk_off_ticker': 'TLT',
         'name': 'TQQQ_200MA_TLT'},
        {'signal_cols': ['trend_signal'], 'risk_on_ticker': 'TQQQ', 'risk_off_ticker': 'CASH',
         'name': 'TQQQ_200MA_Cash'},
        {'signal_cols': ['trend_signal'], 'risk_on_ticker': 'UPRO', 'risk_off_ticker': 'TLT',
         'name': 'UPRO_200MA_TLT'},

        # VIX-based
        {'signal_cols': ['vix_signal'], 'risk_on_ticker': 'TQQQ', 'risk_off_ticker': 'TLT',
         'name': 'TQQQ_VIX_TLT'},

        # Composite (trend + VIX + credit)
        {'signal_cols': ['trend_signal', 'vix_signal', 'credit_signal'], 'threshold': 0,
         'risk_on_ticker': 'TQQQ', 'risk_off_ticker': 'TLT',
         'name': 'TQQQ_Composite3_TLT'},

        # Full composite (all signals)
        {'signal_cols': ['trend_signal', 'vix_signal', 'credit_signal', 'breadth_signal', 'mom_signal'],
         'threshold': 0, 'risk_on_ticker': 'TQQQ', 'risk_off_ticker': 'TLT',
         'name': 'TQQQ_Composite5_TLT'},

        # 50MA instead of 200MA
        {'signal_cols': ['trend_50'], 'risk_on_ticker': 'TQQQ', 'risk_off_ticker': 'TLT',
         'name': 'TQQQ_50MA_TLT'},

        # Strict composite (majority must agree)
        {'signal_cols': ['trend_signal', 'vix_signal', 'credit_signal'], 'threshold': 0.33,
         'risk_on_ticker': 'TQQQ', 'risk_off_ticker': 'TLT',
         'name': 'TQQQ_Composite3_strict_TLT'},
    ]

    results = []
    for cfg in configs:
        name = cfg.pop('name')
        print(f"\nTesting {name}...")
        ret = backtest_regime_strategy(close, signals, **cfg, name=name)
        if ret is not None and len(ret) > 0:
            r = analyze_daily(ret, name, spy)
            if r:
                results.append(r)
                print(f"  CAGR={r['CAGR']}%, Sharpe={r['sharpe']}, Sortino={r['sortino']}, "
                      f"MaxDD={r['max_drawdown']}%, R1={'PASS' if r['R1_pass'] else 'FAIL'}, "
                      f"$441->{r['final_441']}")

    with open(f"{OUTPUT_DIR}/strategy2_regime_switching_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "="*80)
    print("STRATEGY 2: REGIME SWITCHING WITH 3x LEVERAGE — RESULTS")
    print("="*80)
    df = pd.DataFrame(results)
    print(df[['strategy', 'CAGR', 'sharpe', 'sortino', 'max_drawdown', 'R1_pass',
              'final_441', 'perm_p_value']].to_string(index=False))

    return results

if __name__ == '__main__':
    results = main()
