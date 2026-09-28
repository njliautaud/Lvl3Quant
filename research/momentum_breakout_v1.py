#!/usr/bin/env python3
"""
Momentum Breakout v1 — Buy on 20-day high breakout with volume confirmation
=============================================================================
Hypothesis: Stocks breaking to new 20-day highs on 2x+ volume tend to
continue 5-10 more days. This is a classic CTA/trend-following signal
applied to individual stocks with options leverage.

Tested on growth/volatile universe where we can afford options.

Variants:
A) Basic breakout: new 20d high + 2x volume -> buy shares, 10d hold
B) Tight breakout: new 10d high + 1.5x volume -> buy shares, 5d hold
C) Breakout + momentum: 20d high + vol + 5d RSI > 70 -> momentum confirmed
D) Breakout + relative strength: stock outperforming SPY by 5%+ over 20d
E) Sector leader breakout: best-performing stock in sector breaks out
F) Multi-timeframe: 20d AND 50d high simultaneously -> strongest signal
"""

import json
import warnings
from datetime import date
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

ACCOUNT_SIZE = 645
MAX_POSITION = 200
OOT_START = "2022-01-01"
OOT_END = "2026-07-25"

UNIVERSE = [
    # Growth/volatile — affordable for options
    "SOFI", "SNAP", "LYFT", "LCID", "RIOT", "MARA", "PLUG", "PINS",
    "DKNG", "UPST", "PLTR", "HOOD", "RBLX", "NIO", "RIVN", "COIN",
    "HIMS", "AFRM", "RDDT", "CAVA",
    # Liquid large-caps (shares only)
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AMD",
    "NFLX", "CRM", "BA", "DIS", "INTC",
    # Sector ETFs
    "XLE", "XLF", "XLK", "XLY", "XLC", "XLV", "XLI", "XLB", "XLU",
    # Commodity/bond
    "GLD", "SLV", "USO", "TLT",
]


def load_data():
    data = {}
    spy = yf.Ticker("SPY").history(start="2021-06-01", end=OOT_END)
    data['SPY'] = spy
    for sym in UNIVERSE:
        try:
            hist = yf.Ticker(sym).history(start="2021-06-01", end=OOT_END)
            if len(hist) > 100:
                data[sym] = hist
        except:
            pass
    print(f"  Loaded {len(data)} symbols")
    return data


def add_indicators(df):
    c = df['Close']
    df['high_20d'] = c.rolling(20).max()
    df['high_10d'] = c.rolling(10).max()
    df['high_50d'] = c.rolling(50).max()
    df['vol_avg_20d'] = df['Volume'].rolling(20).mean()
    df['vol_ratio'] = df['Volume'] / (df['vol_avg_20d'] + 1)
    # RSI(5)
    delta = c.diff()
    gain = delta.where(delta > 0, 0.0).ewm(alpha=1/5, min_periods=5).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/5, min_periods=5).mean()
    df['rsi5'] = 100 - 100 / (1 + gain / (loss + 1e-10))
    # 20d return
    df['ret_20d'] = c.pct_change(20)
    df['ret_5d'] = c.pct_change(5)
    return df


def scan_breakouts(data, lookback=20, vol_mult=2.0, hold_days=10, extra_filter=None):
    trades = []
    spy_df = data.get('SPY')

    for sym in UNIVERSE:
        if sym not in data:
            continue
        df = add_indicators(data[sym].copy())
        df = df.loc[OOT_START:OOT_END]
        if len(df) < 30:
            continue

        cooldown = 0
        for i in range(lookback + 5, len(df) - hold_days - 1):
            if cooldown > 0:
                cooldown -= 1
                continue

            row = df.iloc[i]
            prev = df.iloc[i - 1]

            # Basic breakout: close above prior N-day high
            if lookback == 20:
                breakout = row['Close'] >= prev['high_20d'] and prev['Close'] < prev['high_20d']
            elif lookback == 10:
                breakout = row['Close'] >= prev['high_10d'] and prev['Close'] < prev['high_10d']
            else:
                breakout = False

            if not breakout:
                continue
            if row['vol_ratio'] < vol_mult:
                continue

            # Extra filters
            if extra_filter == 'momentum':
                if row['rsi5'] < 65:
                    continue
            elif extra_filter == 'relative_strength':
                if spy_df is not None:
                    spy_ret = spy_df.loc[:df.index[i]]['Close'].pct_change(20).iloc[-1] if len(spy_df.loc[:df.index[i]]) > 20 else 0
                    stock_ret = row.get('ret_20d', 0)
                    if pd.isna(stock_ret) or pd.isna(spy_ret):
                        continue
                    if stock_ret - spy_ret < 0.05:
                        continue
            elif extra_filter == 'multi_tf':
                if row['Close'] < prev.get('high_50d', float('inf')):
                    continue

            entry_price = float(row['Close'])
            entry_date = df.index[i].date()

            # Track forward
            exit_idx = min(i + hold_days, len(df) - 1)
            best_ret = 0
            exit_price = entry_price
            exit_reason = 'TIME'

            for j in range(1, hold_days + 1):
                if i + j >= len(df):
                    break
                fwd = df.iloc[i + j]
                ret = (fwd['Close'] - entry_price) / entry_price

                best_ret = max(best_ret, ret)

                # Trailing stop: if drops 3% from peak
                if best_ret > 0.03 and ret < best_ret - 0.03:
                    exit_price = float(fwd['Close'])
                    exit_reason = 'TRAIL'
                    exit_idx = i + j
                    break

                # Stop loss
                if ret < -0.05:
                    exit_price = float(fwd['Close'])
                    exit_reason = 'SL'
                    exit_idx = i + j
                    break
            else:
                exit_price = float(df.iloc[exit_idx]['Close'])

            stock_ret = (exit_price - entry_price) / entry_price
            exit_date = df.index[exit_idx].date()

            # Position sizing: shares for expensive, calls for cheap
            if entry_price <= 30:
                # Estimate call P&L
                call_cost = min(entry_price * 0.05 * 100, MAX_POSITION)
                delta_effect = stock_ret * 0.55
                gamma = max(0, abs(stock_ret) - 0.02) * 0.15
                theta = (exit_idx - i) * 0.004
                call_ret = delta_effect + (gamma if stock_ret > 0 else -gamma) - theta
                pnl = call_cost * call_ret / 0.05
                instrument = 'CALL'
            else:
                shares = MAX_POSITION / entry_price
                pnl = float(shares * (exit_price - entry_price))
                instrument = 'SHARES'

            trades.append({
                'symbol': sym,
                'entry_date': str(entry_date),
                'exit_date': str(exit_date),
                'entry_price': round(entry_price, 2),
                'exit_price': round(exit_price, 2),
                'stock_ret_pct': round(stock_ret * 100, 2),
                'mfe_pct': round(best_ret * 100, 2),
                'exit_reason': exit_reason,
                'hold_days': exit_idx - i,
                'vol_ratio': round(float(row['vol_ratio']), 1),
                'pnl': round(float(pnl), 2),
                'instrument': instrument,
            })
            cooldown = hold_days  # Don't re-enter same stock immediately

    trades.sort(key=lambda x: x['entry_date'])
    return trades


def evaluate(trades, label, account=ACCOUNT_SIZE):
    if not trades:
        return {'name': label, 'n_trades': 0, 'sharpe': 0, 'wr': 0, 'pf': 0,
                'total_pnl': 0, 'avg_pnl': 0, 'final_equity': account, 'mdd_pct': 0,
                'perm_p': 1.0, 'regime_gap': 0, 'sortino': 0,
                'gates_passed': 0, 'gates_total': 5, 'gate_results': {}}

    pnls = [float(t['pnl']) for t in trades]
    n = len(pnls)
    wins = sum(1 for p in pnls if p > 0)
    eq = [account]
    for p in pnls:
        eq.append(eq[-1] + p)
    eq = np.array(eq)
    mdd = float(np.min(eq / np.maximum.accumulate(eq) - 1) * 100)
    mean_p = np.mean(pnls)
    std_p = np.std(pnls) if n > 1 else 1
    sharpe = float(mean_p / std_p * np.sqrt(252 / max(1, n))) if std_p > 0 else 0
    neg = [p for p in pnls if p < 0]
    ds = np.std(neg) if neg else 1
    sortino = float(mean_p / ds * np.sqrt(252 / max(1, n))) if ds > 0 else 0
    gp = sum(p for p in pnls if p > 0)
    gl = abs(sum(p for p in pnls if p < 0))
    pf = float(gp / gl) if gl > 0 else float('inf')

    obs = mean_p / std_p if std_p > 0 else 0
    ct = sum(1 for _ in range(5000)
             if np.mean(np.random.choice([-1,1], n) * np.abs(pnls)) / (np.std(np.random.choice([-1,1], n) * np.abs(pnls)) + 1e-10) >= obs)
    perm_p = ct / 5000

    mid = n // 2
    rg = 0
    if mid > 2:
        h1, h2 = np.mean(pnls[:mid]), np.mean(pnls[mid:])
        rg = abs(h1 - h2) / (abs(max(h1, h2)) + 1e-10)

    gates = {'sharpe': sharpe >= 0.5, 'perm_test': perm_p < 0.05,
             'regime': rg < 0.50, 'mdd': mdd > -50, 'trades': n >= 10}
    return {'name': label, 'n_trades': n, 'wins': int(wins),
            'wr': round(wins/n*100, 1), 'sharpe': round(sharpe, 3),
            'sortino': round(sortino, 3), 'pf': round(pf, 2),
            'total_pnl': round(float(sum(pnls)), 2),
            'avg_pnl': round(float(mean_p), 2),
            'final_equity': round(float(eq[-1]), 2),
            'mdd_pct': round(mdd, 1), 'perm_p': round(perm_p, 3),
            'regime_gap': round(float(rg), 2),
            'gates_passed': sum(gates.values()), 'gates_total': len(gates),
            'gate_results': {k: bool(v) for k, v in gates.items()}}


def main():
    print("=" * 70)
    print("MOMENTUM BREAKOUT v1 — Volume-confirmed breakout strategies")
    print(f"OOT: {OOT_START} to {OOT_END}")
    print("=" * 70)
    print("\nLoading data...")
    data = load_data()

    configs = [
        ("A_Basic_20d", 20, 2.0, 10, None),
        ("B_Tight_10d", 10, 1.5, 5, None),
        ("C_Momentum", 20, 2.0, 10, 'momentum'),
        ("D_RelStrength", 20, 2.0, 10, 'relative_strength'),
        ("E_MultiTF", 20, 2.0, 10, 'multi_tf'),
        ("F_LoVol_20d", 20, 1.3, 10, None),  # Lower vol threshold
    ]

    results = []
    for name, lb, vm, hd, ef in configs:
        print(f"\n── {name} ──")
        trades = scan_breakouts(data, lookback=lb, vol_mult=vm, hold_days=hd, extra_filter=ef)
        res = evaluate(trades, name)
        results.append(res)
        print(f"  {res['n_trades']} trades, Sharpe {res['sharpe']}, WR {res['wr']}%, "
              f"PF {res['pf']}, MDD {res['mdd_pct']}%, p={res['perm_p']}")
        if trades:
            tc = Counter(t['symbol'] for t in trades)
            print(f"  Top: {tc.most_common(5)}")
            er = Counter(t['exit_reason'] for t in trades)
            print(f"  Exits: {dict(er)}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for r in results:
        g = r['gates_passed']
        st = "✅ PASS" if g >= 4 else ("⚠️ PARTIAL" if g >= 3 else "❌ FAIL")
        print(f"  {r['name']:20s} | Sharpe {r['sharpe']:6.3f} | WR {r['wr']:5.1f}% | "
              f"PF {r['pf']:5.2f} | MDD {r['mdd_pct']:6.1f}% | p={r['perm_p']:.3f} | "
              f"{g}/{r['gates_total']} | {st}")

    out = Path("/home/jupiter/Lvl3Quant/research/findings/momentum_breakout_v1.json")
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("\nResults saved.")


if __name__ == "__main__":
    main()
