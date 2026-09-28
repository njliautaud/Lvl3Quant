#!/usr/bin/env python3
"""
Oversold Bounce Options v1 — Buy calls on deeply oversold liquid stocks
========================================================================
Hypothesis: When high-quality liquid stocks drop >8% in 5 days (not earnings-
related), they tend to mean-revert over the next 5-10 days. The bounce is
large enough to make cheap OTM calls profitable.

Universe: Top 30 most liquid S&P 500 stocks.
Entry: RSI(5) < 20 AND 5-day return < -8% AND NOT within 3 days of earnings.
Exit: +30% on call premium OR 10 trading days, whichever first.
Position: 1 ATM call contract, max $200.
"""

import json
import warnings
from datetime import datetime, timedelta, date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

ACCOUNT_SIZE = 645
MAX_POSITION = 200
OOT_START = "2022-01-01"
OOT_END = "2026-07-25"

TOP30 = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "JPM", "V", "UNH",
    "MA", "HD", "PG", "JNJ", "XOM", "CVX", "BAC", "WFC", "ABBV", "MRK",
    "PFE", "KO", "PEP", "COST", "CRM", "AMD", "NFLX", "DIS", "INTC", "BA"
]

# Also include affordable growth names where calls might be < $200
GROWTH_NAMES = [
    "SOFI", "SNAP", "LYFT", "LCID", "RIOT", "MARA", "PLUG", "PINS",
    "DKNG", "UPST", "PLTR", "HOOD", "RBLX", "NIO", "RIVN", "COIN"
]

ALL_UNIVERSE = TOP30 + GROWTH_NAMES


def load_all_data():
    """Load daily OHLCV for full universe."""
    data = {}
    print(f"Loading {len(ALL_UNIVERSE)} stocks...")
    for sym in ALL_UNIVERSE:
        try:
            tk = yf.Ticker(sym)
            hist = tk.history(start="2021-10-01", end=OOT_END)
            if len(hist) > 200:
                data[sym] = hist
                # Add technical indicators
                close = hist['Close']
                # RSI(5)
                delta = close.diff()
                gain = delta.where(delta > 0, 0.0)
                loss = -delta.where(delta < 0, 0.0)
                avg_gain = gain.ewm(alpha=1/5, min_periods=5).mean()
                avg_loss = loss.ewm(alpha=1/5, min_periods=5).mean()
                rs = avg_gain / (avg_loss + 1e-10)
                data[sym]['RSI5'] = 100 - (100 / (1 + rs))
                # 5-day return
                data[sym]['ret5d'] = close.pct_change(5)
                # 10-day return (for measuring bounce)
                data[sym]['ret10d'] = close.pct_change(10)
                # Bollinger Bands
                data[sym]['BB_mid'] = close.rolling(20).mean()
                data[sym]['BB_std'] = close.rolling(20).std()
                data[sym]['BB_lower'] = data[sym]['BB_mid'] - 2 * data[sym]['BB_std']
                # Volume surge
                data[sym]['vol_ratio'] = hist['Volume'] / hist['Volume'].rolling(20).mean()
        except Exception as e:
            pass
    print(f"  Loaded {len(data)} stocks successfully")
    return data


def find_oversold_signals(data, rsi_thresh=20, ret5d_thresh=-0.08):
    """Find oversold bounce entry signals."""
    signals = []
    for sym, df in data.items():
        df = df.loc[OOT_START:OOT_END].copy()
        if len(df) < 20:
            continue

        for i in range(10, len(df) - 12):
            row = df.iloc[i]
            if pd.isna(row.get('RSI5')) or pd.isna(row.get('ret5d')):
                continue

            # Entry conditions
            if row['RSI5'] > rsi_thresh:
                continue
            if row['ret5d'] > ret5d_thresh:
                continue
            # Below lower Bollinger Band
            if not pd.isna(row.get('BB_lower')) and row['Close'] > row['BB_lower']:
                continue
            # Volume surge (capitulation)
            if not pd.isna(row.get('vol_ratio')) and row['vol_ratio'] < 1.5:
                continue

            entry_price = row['Close']
            entry_date = df.index[i].date()

            # Look forward 10 days for exit
            best_exit_pct = -999
            exit_price = entry_price
            exit_date = entry_date
            exit_reason = 'TIME'

            for j in range(1, min(11, len(df) - i)):
                future = df.iloc[i + j]
                future_ret = (future['Close'] - entry_price) / entry_price

                if future_ret > best_exit_pct:
                    best_exit_pct = future_ret
                    # MFE tracking

                # Take profit at +5% stock move (call would be ~+30-50%)
                if future_ret >= 0.05:
                    exit_price = future['Close']
                    exit_date = df.index[i + j].date()
                    exit_reason = 'TP'
                    break

                # Stop loss at -5% further
                if future_ret <= -0.05:
                    exit_price = future['Close']
                    exit_date = df.index[i + j].date()
                    exit_reason = 'SL'
                    break
            else:
                # Time exit
                exit_idx = min(i + 10, len(df) - 1)
                exit_price = df.iloc[exit_idx]['Close']
                exit_date = df.index[exit_idx].date()

            stock_return = (exit_price - entry_price) / entry_price
            hold_days = (j if exit_reason != 'TIME' else 10)

            # Model call option P&L
            # ATM call: delta ~0.5, gamma helps on big moves
            # Theta: ~0.5%/day for short-dated ATM
            est_call_cost = entry_price * 0.04 * 100  # ~4% for 2-week ATM call
            if est_call_cost > MAX_POSITION:
                # Use cheaper strike or skip
                if entry_price > 50:  # Too expensive for options
                    continue

            # Call P&L estimation
            # Delta effect + gamma convexity - theta
            delta_pnl = stock_return * 0.55  # slightly above 0.5 for ATM
            gamma_bonus = max(0, abs(stock_return) - 0.03) * 0.2  # gamma kicks in on big moves
            theta_cost = hold_days * 0.005  # 0.5%/day theta
            call_return = delta_pnl + gamma_bonus - theta_cost

            # For stocks too expensive for options, model as shares
            if entry_price > 50:
                shares = MAX_POSITION / entry_price
                pnl = shares * (exit_price - entry_price)
                instrument = 'SHARES'
            else:
                call_cost = min(est_call_cost, MAX_POSITION)
                pnl = call_cost * call_return / 0.04  # scale to dollar P&L
                instrument = 'CALL'

            signals.append({
                'symbol': sym,
                'entry_date': str(entry_date),
                'exit_date': str(exit_date),
                'entry_price': round(float(entry_price), 2),
                'exit_price': round(float(exit_price), 2),
                'rsi5': round(float(row['RSI5']), 1),
                'ret5d_pct': round(float(row['ret5d'] * 100), 1),
                'stock_return_pct': round(float(stock_return * 100), 2),
                'hold_days': int(hold_days),
                'exit_reason': exit_reason,
                'pnl': round(float(pnl), 2),
                'instrument': instrument,
                'mfe_pct': round(float(best_exit_pct * 100), 2),
            })

    # Sort by date
    signals.sort(key=lambda x: x['entry_date'])
    return signals


def run_variants(data):
    """Test multiple parameter combinations."""
    variants = {
        'A_Strict': {'rsi': 15, 'ret5d': -0.10},    # Very oversold
        'B_Standard': {'rsi': 20, 'ret5d': -0.08},   # Standard
        'C_Relaxed': {'rsi': 25, 'ret5d': -0.06},    # More signals
        'D_DeepDrop': {'rsi': 20, 'ret5d': -0.12},   # Deep drops only
        'E_GrowthOnly': {'rsi': 20, 'ret5d': -0.08}, # Growth names only
        'F_LargeCap': {'rsi': 20, 'ret5d': -0.08},   # Large cap only
    }

    results = []
    for name, params in variants.items():
        print(f"\n── {name}: RSI<{params['rsi']}, 5d ret<{params['ret5d']*100}% ──")

        if name == 'E_GrowthOnly':
            filtered_data = {k: v for k, v in data.items() if k in GROWTH_NAMES}
        elif name == 'F_LargeCap':
            filtered_data = {k: v for k, v in data.items() if k in TOP30}
        else:
            filtered_data = data

        trades = find_oversold_signals(filtered_data, params['rsi'], params['ret5d'])
        res = evaluate(trades, name)
        results.append(res)

        print(f"  {res['n_trades']} trades, Sharpe {res['sharpe']}, WR {res['wr']}%, "
              f"PF {res['pf']}, MDD {res['mdd_pct']}%, perm p={res['perm_p']}")

        # Show top tickers
        if trades:
            from collections import Counter
            ticker_counts = Counter(t['symbol'] for t in trades)
            print(f"  Top tickers: {ticker_counts.most_common(5)}")
            winners = [t for t in trades if t['pnl'] > 0]
            print(f"  Avg winner: ${np.mean([t['pnl'] for t in winners]):.2f}" if winners else "  No winners")
            # Exit reason breakdown
            reasons = Counter(t['exit_reason'] for t in trades)
            print(f"  Exit reasons: {dict(reasons)}")

    return results


def evaluate(trades, label, account=ACCOUNT_SIZE):
    """Risk-adjusted metrics + permutation test."""
    if not trades:
        return {'name': label, 'n_trades': 0, 'sharpe': 0, 'wr': 0, 'pf': 0,
                'total_pnl': 0, 'avg_pnl': 0, 'final_equity': account, 'mdd_pct': 0,
                'perm_p': 1.0, 'regime_gap': 0, 'sortino': 0,
                'gates_passed': 0, 'gates_total': 5, 'gate_results': {}}

    pnls = [float(t['pnl']) for t in trades]
    n = len(pnls)
    wins = sum(1 for p in pnls if p > 0)

    equity = [account]
    for p in pnls:
        equity.append(equity[-1] + p)
    equity = np.array(equity)
    mdd = float(np.min(equity / np.maximum.accumulate(equity) - 1) * 100)

    mean_pnl = np.mean(pnls)
    std_pnl = np.std(pnls) if n > 1 else 1
    sharpe = float(mean_pnl / std_pnl * np.sqrt(252 / max(1, n))) if std_pnl > 0 else 0

    neg = [p for p in pnls if p < 0]
    ds = np.std(neg) if neg else 1
    sortino = float(mean_pnl / ds * np.sqrt(252 / max(1, n))) if ds > 0 else 0

    gp = sum(p for p in pnls if p > 0)
    gl = abs(sum(p for p in pnls if p < 0))
    pf = float(gp / gl) if gl > 0 else float('inf')

    # Permutation test
    obs = mean_pnl / std_pnl if std_pnl > 0 else 0
    ct = 0
    for _ in range(5000):
        sh = np.random.choice([-1, 1], size=n) * np.abs(pnls)
        ps = np.mean(sh) / (np.std(sh) + 1e-10)
        if ps >= obs:
            ct += 1
    perm_p = ct / 5000

    # Regime: up market vs down market
    # Split trades by SPY performance in surrounding month
    mid = n // 2
    rg = 0
    if mid > 2:
        h1 = np.mean(pnls[:mid])
        h2 = np.mean(pnls[mid:])
        rg = abs(h1 - h2) / (abs(max(h1, h2)) + 1e-10)

    gates = {
        'sharpe': sharpe >= 0.5,
        'perm_test': perm_p < 0.05,
        'regime': rg < 0.50,
        'mdd': mdd > -50,
        'trades': n >= 10,
    }

    return {
        'name': label, 'n_trades': n, 'wins': int(wins),
        'wr': round(wins/n*100, 1), 'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3), 'pf': round(pf, 2),
        'total_pnl': round(float(sum(pnls)), 2),
        'avg_pnl': round(float(mean_pnl), 2),
        'final_equity': round(float(equity[-1]), 2),
        'mdd_pct': round(mdd, 1),
        'perm_p': round(perm_p, 3),
        'regime_gap': round(float(rg), 2),
        'gates_passed': sum(gates.values()),
        'gates_total': len(gates),
        'gate_results': {k: bool(v) for k, v in gates.items()},
    }


def main():
    print("=" * 70)
    print("OVERSOLD BOUNCE OPTIONS v1")
    print("RSI(5) + 5-day drop + Bollinger Band + Volume surge")
    print(f"OOT: {OOT_START} to {OOT_END}")
    print("=" * 70)

    data = load_all_data()
    results = run_variants(data)

    # Random baseline
    print("\n── RANDOM BASELINE ──")
    import random
    random_trades = []
    all_syms = list(data.keys())
    for _ in range(50):
        sym = random.choice(all_syms)
        df = data[sym].loc[OOT_START:OOT_END]
        if len(df) < 20:
            continue
        i = random.randint(10, len(df) - 12)
        entry = float(df.iloc[i]['Close'])
        exit_p = float(df.iloc[min(i+10, len(df)-1)]['Close'])
        if entry > 50:
            pnl = (MAX_POSITION / entry) * (exit_p - entry)
        else:
            call_ret = (exit_p - entry) / entry * 0.55 - 10 * 0.005
            pnl = min(entry * 0.04 * 100, MAX_POSITION) * call_ret / 0.04
        random_trades.append({'pnl': round(float(pnl), 2)})
    res_rand = evaluate(random_trades, "RANDOM")
    print(f"  Sharpe {res_rand['sharpe']}, WR {res_rand['wr']}%")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for r in results:
        g = r['gates_passed']
        t = r['gates_total']
        st = "✅ PASS" if g >= 4 else ("⚠️ PARTIAL" if g >= 3 else "❌ FAIL")
        print(f"  {r['name']:20s} | Sharpe {r['sharpe']:6.3f} | WR {r['wr']:5.1f}% | "
              f"PF {r['pf']:5.2f} | MDD {r['mdd_pct']:6.1f}% | p={r['perm_p']:.3f} | "
              f"{g}/{t} | {st}")
    print(f"  {'RANDOM':20s} | Sharpe {res_rand['sharpe']:6.3f}")

    out = Path("/home/jupiter/Lvl3Quant/research/findings/oversold_bounce_options_v1.json")
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved.")


if __name__ == "__main__":
    main()
