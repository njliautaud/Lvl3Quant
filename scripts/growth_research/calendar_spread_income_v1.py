#!/usr/bin/env python3
"""
Calendar Spread Income v1 — Time Decay Differential Strategy
=============================================================

THESIS: Sell near-term options, buy longer-term options at SAME strike.
Front-month decays faster (theta) → spread widens → profit from time decay differential.

ADVANTAGES:
- Capital efficient: max loss = net debit (good for $645 account)
- Non-directional: profits from stock staying near strike
- Theta positive: time works in your favor
- Lower risk than naked selling

STRATEGY:
- Sell 30-DTE option, buy 60-DTE option at same ATM strike
- Entry when IV rank > 30% (want elevated IV for better credits)
- Exit at 25-50% profit or 100% loss
- Roll front month at 7 DTE if still profitable

TICKERS: Liquid sector ETFs where we have momentum edge
Variants test ATM vs OTM, different DTEs, VIX filters, ETF selection

Capital levels tested: $645 (agentic) and $10K (portfolio allocation)
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

def fprint(*args, **kwargs):
    print(*args, **kwargs, flush=True)

RESULTS_DIR = Path('/home/jupiter/Lvl3Quant/research/findings')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = RESULTS_DIR / 'calendar_spread_income_v1_results.json'

MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen('http://jupiter:5000/', timeout=2)
    import mlflow
    mlflow.set_tracking_uri('http://jupiter:5000')
    MLFLOW_OK = True
except:
    pass


def bs_price(S, K, T, sigma, r=0.04, opt='call'):
    if T <= 0 or sigma <= 0:
        return max(0, S - K) if opt == 'call' else max(0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_theta(S, K, T, sigma, r=0.04, opt='call'):
    """Daily theta (time decay) in dollars per share."""
    if T <= 0 or sigma <= 0:
        return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    common = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    if opt == 'call':
        theta = common - r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        theta = common + r * K * np.exp(-r * T) * norm.cdf(-d2)
    return theta / 365  # per day


def simulate_calendar(ticker_close, vix_close, spy_close,
                      front_dte=30, back_dte=60, strike_type='atm',
                      opt_type='call', iv_rank_min=30, profit_target=0.50,
                      stop_loss=1.0, max_concurrent=3,
                      capital=10000, name='base', ticker_name='SPY'):
    """
    Simulate calendar spread strategy.

    Buy back-month option, sell front-month option at same strike.
    Profit from front-month decaying faster than back-month.
    """
    fprint(f"\n--- {name} ---")

    spy_sma200 = spy_close.rolling(200).mean()
    vix_rank = vix_close.rolling(252).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10) * 100
    )

    dates = ticker_close.index[252:]
    trades = []
    open_positions = []
    equity = capital

    for i_idx in range(len(dates) - back_dte - 1):
        date = dates[i_idx]
        S = float(ticker_close.loc[date])
        vix = float(vix_close.loc[date]) if date in vix_close.index else 15
        sigma = vix / 100  # Simplified: use VIX as proxy for all ETF IV
        spy_val = float(spy_close.loc[date]) if date in spy_close.index else S
        sma_val = float(spy_sma200.loc[date]) if date in spy_sma200.index else spy_val
        regime = 'bear' if spy_val < sma_val else 'bull'

        # === Check existing positions ===
        new_open = []
        for pos in open_positions:
            days_held = (date - pos['entry_date']).days
            front_dte_remain = max(pos['front_dte'] - days_held, 0)
            back_dte_remain = max(pos['back_dte'] - days_held, 0)

            S_now = float(ticker_close.loc[date])
            sigma_now = float(vix_close.loc[date]) / 100 if date in vix_close.index else sigma

            if front_dte_remain <= 0:
                # Front expired — close entire spread
                # Front settles at intrinsic, back still has time value
                front_val = max(0, S_now - pos['strike']) if opt_type == 'call' else max(0, pos['strike'] - S_now)
                T_back = max(back_dte_remain / 365, 0.001)
                back_val = bs_price(S_now, pos['strike'], T_back, sigma_now, opt=opt_type)

                # We're short front, long back
                pnl = (-front_val + back_val - pos['net_debit']) * 100 * pos['n_contracts'] - 4
                equity += pnl
                trades.append({
                    'entry': str(pos['entry_date']), 'exit': str(date),
                    'pnl': round(pnl, 2), 'win': pnl > 0, 'regime': regime,
                    'exit_reason': 'front_expiry',
                    'underlying_move': round((S_now / pos['entry_px'] - 1) * 100, 1),
                })
                continue

            # Mark to market
            T_front = max(front_dte_remain / 365, 0.001)
            T_back = max(back_dte_remain / 365, 0.001)

            front_now = bs_price(S_now, pos['strike'], T_front, sigma_now, opt=opt_type)
            back_now = bs_price(S_now, pos['strike'], T_back, sigma_now, opt=opt_type)
            spread_now = back_now - front_now  # Long back, short front

            unrealized = (spread_now - pos['net_debit']) * 100 * pos['n_contracts']

            # Profit target
            if unrealized >= pos['net_debit'] * 100 * pos['n_contracts'] * profit_target:
                pnl = unrealized - 4  # Commission
                equity += pnl
                trades.append({
                    'entry': str(pos['entry_date']), 'exit': str(date),
                    'pnl': round(pnl, 2), 'win': True, 'regime': regime,
                    'exit_reason': 'profit_target',
                    'underlying_move': round((S_now / pos['entry_px'] - 1) * 100, 1),
                })
                continue

            # Stop loss (max loss = net debit)
            if unrealized < -(pos['net_debit'] * 100 * pos['n_contracts'] * stop_loss):
                pnl = unrealized - 4
                equity += pnl
                trades.append({
                    'entry': str(pos['entry_date']), 'exit': str(date),
                    'pnl': round(pnl, 2), 'win': False, 'regime': regime,
                    'exit_reason': 'stop_loss',
                    'underlying_move': round((S_now / pos['entry_px'] - 1) * 100, 1),
                })
                continue

            new_open.append(pos)

        open_positions = new_open

        # Skip if at max concurrent
        if len(open_positions) >= max_concurrent:
            continue

        # Entry conditions
        if date.weekday() not in [0, 4]:  # Mon/Fri only
            continue

        # IV rank filter
        if date in vix_rank.index:
            ivr = vix_rank.loc[date]
            if pd.isna(ivr) or ivr < iv_rank_min:
                continue

        # Strike selection
        if strike_type == 'atm':
            K = round(S)
        elif strike_type == 'otm_call':
            K = round(S * 1.02)  # 2% OTM call
        elif strike_type == 'otm_put':
            K = round(S * 0.98)  # 2% OTM put
        else:
            K = round(S)

        T_front = front_dte / 365
        T_back = back_dte / 365

        # Price the calendar spread
        front_price = bs_price(S, K, T_front, sigma, opt=opt_type)
        back_price = bs_price(S, K, T_back, sigma, opt=opt_type)
        net_debit = back_price - front_price  # Pay for back, receive for front

        if net_debit <= 0.10:  # Spread too cheap
            continue
        if net_debit > S * 0.05:  # Spread too expensive (>5% of underlying)
            continue

        # Position sizing
        cost_per_contract = net_debit * 100
        max_alloc = equity * 0.15  # Max 15% per position
        n_contracts = max(1, int(max_alloc / cost_per_contract))
        n_contracts = min(n_contracts, 5)

        if cost_per_contract * n_contracts > equity * 0.4:
            n_contracts = max(1, int(equity * 0.4 / cost_per_contract))

        if cost_per_contract > equity:
            continue

        equity -= 4  # Commission

        open_positions.append({
            'entry_date': date,
            'strike': K,
            'front_dte': front_dte,
            'back_dte': back_dte,
            'net_debit': net_debit,
            'n_contracts': n_contracts,
            'entry_px': S,
        })

    # Close remaining
    for pos in open_positions:
        S_now = float(ticker_close.iloc[-1])
        sigma_now = float(vix_close.iloc[-1]) / 100 if len(vix_close) > 0 else 0.15
        days_held = (dates[-1] - pos['entry_date']).days
        front_r = max(pos['front_dte'] - days_held, 1)
        back_r = max(pos['back_dte'] - days_held, 1)

        front_now = bs_price(S_now, pos['strike'], front_r/365, sigma_now, opt=opt_type)
        back_now = bs_price(S_now, pos['strike'], back_r/365, sigma_now, opt=opt_type)
        spread_now = back_now - front_now

        pnl = (spread_now - pos['net_debit']) * 100 * pos['n_contracts'] - 4
        equity += pnl
        trades.append({
            'entry': str(pos['entry_date']), 'exit': str(dates[-1]),
            'pnl': round(pnl, 2), 'win': pnl > 0, 'regime': 'bull',
        })

    if not trades:
        fprint("  No trades!")
        return None

    # Metrics
    n_trades = len(trades)
    wins = sum(1 for t in trades if t['win'])
    wr = wins / n_trades * 100

    total_pnl = sum(t['pnl'] for t in trades)

    tdf = pd.DataFrame(trades)
    tdf['month'] = pd.to_datetime(tdf['entry']).dt.to_period('M')
    monthly = tdf.groupby('month')['pnl'].sum() / capital
    n_years = len(monthly) / 12

    sharpe = (monthly.mean() * 12) / (monthly.std() * np.sqrt(12) + 1e-10) if len(monthly) > 3 else 0
    cagr = (1 + total_pnl / capital) ** (1 / max(n_years, 0.5)) - 1

    down = monthly[monthly < 0]
    sortino = (monthly.mean() * 12) / (down.std() * np.sqrt(12) + 1e-10) if len(down) > 0 else 999

    cum = np.cumsum([t['pnl'] for t in trades])
    peak = np.maximum.accumulate(cum + capital)
    dd = (cum + capital - peak) / peak
    maxdd = dd.min()

    pf = abs(sum(t['pnl'] for t in trades if t['win']) / (sum(t['pnl'] for t in trades if not t['win']) + 1e-10))

    bull_t = [t for t in trades if t.get('regime') == 'bull']
    bear_t = [t for t in trades if t.get('regime') == 'bear']
    bull_wr = sum(1 for t in bull_t if t['win']) / max(len(bull_t), 1) * 100
    bear_wr = sum(1 for t in bear_t if t['win']) / max(len(bear_t), 1) * 100
    r1_gap = abs(bull_wr - bear_wr) / max(bull_wr, bear_wr, 1)

    # Underlying moves on trades
    moves = [t.get('underlying_move', 0) for t in trades if 'underlying_move' in t]
    avg_move = np.mean(moves) if moves else 0

    result = {
        'name': name, 'ticker': ticker_name,
        'n_trades': n_trades, 'win_rate': round(wr, 1),
        'total_pnl': round(total_pnl, 2), 'final_equity': round(equity, 2),
        'cagr_pct': round(cagr * 100, 1), 'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2), 'maxdd_pct': round(maxdd * 100, 1),
        'pf': round(pf, 2),
        'r1_gap': round(r1_gap, 3), 'r1_pass': r1_gap <= 0.50,
        'bull_wr': round(bull_wr, 1), 'bear_wr': round(bear_wr, 1),
        'bull_trades': len(bull_t), 'bear_trades': len(bear_t),
        'avg_underlying_move': round(avg_move, 1),
        'monthly_returns': monthly.tolist(),
    }

    fprint(f"  {name}: {n_trades} trades, WR {wr:.1f}%, Sharpe {sharpe:.2f}, "
           f"CAGR {cagr*100:.1f}%, MaxDD {maxdd*100:.1f}%, PF {pf:.2f}")
    fprint(f"    ${capital:,} → ${equity:,.0f} | Avg underlying move: {avg_move:.1f}%")
    fprint(f"    Bull WR {bull_wr:.0f}% ({len(bull_t)}) | Bear WR {bear_wr:.0f}% ({len(bear_t)}) | R1 gap {r1_gap:.3f}")

    return result


def permutation_test(returns, n_perms=1000):
    if len(returns) < 5:
        return 1.0
    real = np.mean(returns) / (np.std(returns) + 1e-10)
    count = sum(1 for _ in range(n_perms)
                if np.mean(returns * np.random.choice([-1, 1], len(returns))) /
                (np.std(returns) + 1e-10) >= real)
    return count / n_perms


def main():
    import yfinance as yf

    fprint(f"Calendar Spread Income v1 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 70)

    # Test on SPY (most liquid) and sector ETFs
    tickers = ['SPY', 'QQQ', 'XLE', 'XLK', 'XLF']

    raw = yf.download(tickers + ['^VIX'], start='2008-01-01', end='2026-07-25', progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw['Close']
    else:
        close = raw

    vix_col = '^VIX' if '^VIX' in close.columns else 'VIX'
    vix = close[vix_col].dropna()
    spy = close['SPY'].dropna()

    common = spy.index.intersection(vix.index)
    vix = vix.loc[common]
    spy = spy.loc[common]

    fprint(f"Data: {len(spy)} days")

    # Variants
    variants = [
        # (name, ticker, front_dte, back_dte, strike_type, opt_type, ivr_min, profit_target, stop_loss, capital)
        ('A_SPY_ATM_30_60', 'SPY', 30, 60, 'atm', 'call', 30, 0.50, 1.0, 10000),
        ('B_SPY_ATM_21_45', 'SPY', 21, 45, 'atm', 'call', 30, 0.40, 1.0, 10000),
        ('C_SPY_OTM_30_60', 'SPY', 30, 60, 'otm_call', 'call', 30, 0.50, 1.0, 10000),
        ('D_SPY_Put_30_60', 'SPY', 30, 60, 'atm', 'put', 30, 0.50, 1.0, 10000),
        ('E_SPY_NoFilter', 'SPY', 30, 60, 'atm', 'call', 0, 0.50, 1.0, 10000),
        ('F_XLE_ATM_30_60', 'XLE', 30, 60, 'atm', 'call', 30, 0.50, 1.0, 10000),
        ('G_QQQ_ATM_30_60', 'QQQ', 30, 60, 'atm', 'call', 30, 0.50, 1.0, 10000),
        ('H_SPY_ATM_645', 'SPY', 30, 60, 'atm', 'call', 30, 0.50, 1.0, 645),
        ('I_XLE_ATM_645', 'XLE', 30, 60, 'atm', 'call', 30, 0.50, 1.0, 645),
        ('J_SPY_30_90', 'SPY', 30, 90, 'atm', 'call', 30, 0.50, 1.0, 10000),
    ]

    results = []

    if MLFLOW_OK:
        exp_name = 'calendar_spread_income_v1'
        try:
            if not mlflow.get_experiment_by_name(exp_name):
                mlflow.create_experiment(exp_name)
        except:
            pass
        mlflow.set_experiment(exp_name)

    for vname, ticker, f_dte, b_dte, stype, otype, ivr, pt, sl, cap in variants:
        try:
            ticker_close = close[ticker].dropna() if ticker in close.columns else spy
            tc = ticker_close.loc[ticker_close.index.intersection(common)]
            vc = vix.loc[vix.index.intersection(tc.index)]
            sc = spy.loc[spy.index.intersection(tc.index)]

            r = simulate_calendar(
                tc, vc, sc,
                front_dte=f_dte, back_dte=b_dte, strike_type=stype,
                opt_type=otype, iv_rank_min=ivr, profit_target=pt,
                stop_loss=sl, capital=cap, name=vname, ticker_name=ticker
            )
            if r:
                if MLFLOW_OK:
                    with mlflow.start_run(run_name=vname):
                        mlflow.log_params({
                            'ticker': ticker, 'front_dte': f_dte, 'back_dte': b_dte,
                            'strike_type': stype, 'opt_type': otype,
                            'ivr_min': ivr, 'capital': cap
                        })
                        mlflow.log_metrics({k: v for k, v in r.items()
                                           if isinstance(v, (int, float)) and not np.isnan(v) and not np.isinf(v)})
                results.append(r)
        except Exception as e:
            fprint(f"  ERROR {vname}: {e}")
            import traceback
            traceback.print_exc()

    if not results:
        fprint("No results!")
        return

    # Adversarial
    fprint("\n" + "=" * 70)
    fprint("ADVERSARIAL VALIDATION")
    fprint("=" * 70)

    for r in results:
        rets = np.array(r['monthly_returns'])
        if len(rets) < 5:
            r.update({'perm_p': 1.0, 'g1_pass': False, 'g2_pass': r['r1_pass'],
                      'g3_pass': False, 'g4_pass': False, 'gates_passed': 0})
            continue

        r['perm_p'] = round(permutation_test(rets), 3)
        r['g1_pass'] = r['perm_p'] < 0.05
        r['g2_pass'] = r['r1_pass']

        n = len(rets)
        chunk = max(n // 3, 1)
        subs = []
        for j in range(3):
            sub = rets[j * chunk:(j + 1) * chunk]
            subs.append(np.mean(sub) * 12 / (np.std(sub) * np.sqrt(12) + 1e-10) if len(sub) > 1 else 0)
        r['g3_pass'] = all(s > 0 for s in subs)
        r['sub_sharpes'] = [round(s, 2) for s in subs]

        if n > 5:
            nt = max(1, int(n * 0.05))
            tr = np.sort(rets)[nt:-nt] if nt < n // 2 else rets
            trimmed = np.mean(tr) / (np.std(tr) + 1e-10)
            orig = np.mean(rets) / (np.std(rets) + 1e-10)
            r['g4_pass'] = trimmed > 0 and trimmed / (orig + 1e-10) > 0.5
        else:
            r['g4_pass'] = False

        r['gates_passed'] = sum([r['g1_pass'], r['g2_pass'], r['g3_pass'], r['g4_pass']])

        fprint(f"\n{r['name']}: G1={'PASS' if r['g1_pass'] else 'FAIL'}(p={r['perm_p']}), "
               f"G2={'PASS' if r['g2_pass'] else 'FAIL'}(gap={r['r1_gap']}), "
               f"G3={'PASS' if r['g3_pass'] else 'FAIL'}, "
               f"G4={'PASS' if r['g4_pass'] else 'FAIL'} → {r['gates_passed']}/4")

    # Summary
    fprint("\n" + "=" * 70)
    fprint("SUMMARY — Calendar Spread Income v1")
    fprint("=" * 70)
    fprint(f"{'Name':<22} {'Ticker':<5} {'Trades':>6} {'WR':>6} {'Sharpe':>7} {'CAGR':>7} "
           f"{'MaxDD':>7} {'PF':>5} {'R1gap':>6} {'Gates':>6}")
    fprint("-" * 90)
    for r in sorted(results, key=lambda x: x['sharpe'], reverse=True):
        fprint(f"{r['name']:<22} {r['ticker']:<5} {r['n_trades']:>6} {r['win_rate']:>5.1f}% "
               f"{r['sharpe']:>7.2f} {r['cagr_pct']:>6.1f}% {r['maxdd_pct']:>6.1f}% "
               f"{r['pf']:>5.2f} {r['r1_gap']:>6.3f} {r['gates_passed']:>4}/4")

    # Key insights
    fprint("\n--- KEY INSIGHTS ---")
    passing = [r for r in results if r['gates_passed'] >= 3]
    if passing:
        fprint(f"  {len(passing)} variants pass 3+ gates")
        best = max(passing, key=lambda x: x['sharpe'])
        fprint(f"  Best: {best['name']} — Sharpe {best['sharpe']}, CAGR {best['cagr_pct']}%")
    else:
        fprint("  Calendar spreads may not have edge — check if any pass perm test")

    # $645 account viability
    small = [r for r in results if 'H_' in r['name'] or 'I_' in r['name']]
    if small:
        fprint("\n  $645 Account viability:")
        for r in small:
            fprint(f"    {r['name']}: {r['n_trades']} trades, WR {r['win_rate']:.0f}%, "
                   f"Sharpe {r['sharpe']}, {r['gates_passed']}/4 gates")

    save = [{k: v for k, v in r.items() if k != 'monthly_returns'} for r in results]
    with open(RESULTS_PATH, 'w') as f:
        json.dump(save, f, indent=2, default=str)
    fprint(f"\nDone — {datetime.now().strftime('%H:%M:%S')}")


if __name__ == '__main__':
    main()
