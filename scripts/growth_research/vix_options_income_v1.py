#!/usr/bin/env python3
"""
VIX Options Income & Hedge v1 — Trading VIX Options for Income + Portfolio Insurance
=====================================================================================

HC #747: VIX options are in scope for agentic account and portfolio strategies.

STRATEGIES:
1. VIX MEAN REVERSION: When VIX spikes >25, sell VIX call spreads (mean-reversion).
   VIX reliably reverts from spikes. Sell call spreads for defined-risk income.

2. VIX CRASH HEDGE: When VIX is low (<15), buy cheap VIX calls as portfolio insurance.
   VIX calls gain 200-500% in crashes. Cheap insurance that pays off asymmetrically.

3. VIX PUT SPREADS (VRP HARVESTING): Sell VIX puts to harvest volatility risk premium.
   VIX term structure is typically in contango → VIX overpriced → selling is profitable.

4. VIX IRON CONDORS: Non-directional premium selling on VIX itself. Range-bound thesis.

5. COMBINED: Best combo of above with dynamic regime switching.

Note: VIX options settle on VIX settlement value (AM of expiration day).
VIX options are European-style, cash-settled. No early assignment.
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
RESULTS_PATH = RESULTS_DIR / 'vix_options_income_v1_results.json'

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


def simulate_vix_strategy(vix_close, spy_close, strategy='mean_rev',
                          capital=10000, name='base',
                          entry_threshold=25, exit_threshold=18,
                          dte=30, delta_short=0.30, spread_width=5,
                          max_concurrent=3, profit_target=0.50,
                          stop_loss=2.0):
    """
    Simulate VIX options strategies.

    strategy types:
    - 'mean_rev': Sell call spreads when VIX > entry_threshold
    - 'crash_hedge': Buy calls when VIX < 15
    - 'put_spread': Sell put spreads (VRP harvest)
    - 'iron_condor': Sell IC on VIX
    - 'combined': Dynamic mix
    """
    fprint(f"\n--- {name} ---")

    spy_sma200 = spy_close.rolling(200).mean()
    vix_rank = vix_close.rolling(252).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10) * 100
    )
    vix_sma20 = vix_close.rolling(20).mean()

    dates = vix_close.index[252:]
    trades = []
    open_positions = []
    equity = capital

    for i_idx in range(len(dates) - dte - 1):
        date = dates[i_idx]
        vix = float(vix_close.loc[date])
        spy = float(spy_close.loc[date]) if date in spy_close.index else 0
        sma = float(spy_sma200.loc[date]) if date in spy_sma200.index else spy
        regime = 'bear' if spy < sma else 'bull'

        # VIX vol-of-vol for pricing VIX options
        # VIX options have high IV — typically 80-120%
        vix_vol = 0.80  # Conservative VIX option IV assumption

        # === Check and close existing positions ===
        new_open = []
        for pos in open_positions:
            days_held = (date - pos['entry_date']).days
            dte_remain = max(pos['initial_dte'] - days_held, 0)
            T_now = max(dte_remain / 365, 0.001)

            vix_now = vix

            if dte_remain <= 0:
                # Expired — cash settlement
                if pos['type'] == 'call_spread_short':
                    # Short call spread: profit if VIX < short strike
                    short_val = max(0, vix_now - pos['short_strike'])
                    long_val = max(0, vix_now - pos['long_strike'])
                    settlement = (short_val - long_val) * 100  # Cash settlement
                    pnl = pos['credit'] * 100 - settlement - 3  # Commission
                elif pos['type'] == 'put_spread_short':
                    short_val = max(0, pos['short_strike'] - vix_now)
                    long_val = max(0, pos['long_strike'] - vix_now)
                    settlement = (short_val - long_val) * 100
                    pnl = pos['credit'] * 100 - settlement - 3
                elif pos['type'] == 'call_long':
                    intrinsic = max(0, vix_now - pos['strike']) * 100
                    pnl = intrinsic - pos['cost'] * 100 - 3
                elif pos['type'] == 'iron_condor':
                    call_short_val = max(0, vix_now - pos['call_short'])
                    call_long_val = max(0, vix_now - pos['call_long'])
                    put_short_val = max(0, pos['put_short'] - vix_now)
                    put_long_val = max(0, pos['put_long'] - vix_now)
                    settlement = (call_short_val - call_long_val + put_short_val - put_long_val) * 100
                    pnl = pos['credit'] * 100 - settlement - 3
                else:
                    pnl = 0

                equity += pnl
                trades.append({
                    'entry': str(pos['entry_date']), 'exit': str(date),
                    'type': pos['type'], 'pnl': round(pnl, 2),
                    'win': pnl > 0, 'regime': regime,
                    'vix_entry': pos['vix_entry'], 'vix_exit': round(vix_now, 1),
                })
                continue

            # Early exit check (for short positions)
            if pos['type'] in ['call_spread_short', 'put_spread_short', 'iron_condor']:
                # Mark to market
                if pos['type'] == 'call_spread_short':
                    short_px = bs_price(vix_now, pos['short_strike'], T_now, vix_vol, opt='call')
                    long_px = bs_price(vix_now, pos['long_strike'], T_now, vix_vol, opt='call')
                    current_cost = short_px - long_px
                elif pos['type'] == 'put_spread_short':
                    short_px = bs_price(vix_now, pos['short_strike'], T_now, vix_vol, opt='put')
                    long_px = bs_price(vix_now, pos['long_strike'], T_now, vix_vol, opt='put')
                    current_cost = short_px - long_px
                elif pos['type'] == 'iron_condor':
                    cs = bs_price(vix_now, pos['call_short'], T_now, vix_vol, opt='call')
                    cl = bs_price(vix_now, pos['call_long'], T_now, vix_vol, opt='call')
                    ps = bs_price(vix_now, pos['put_short'], T_now, vix_vol, opt='put')
                    pl = bs_price(vix_now, pos['put_long'], T_now, vix_vol, opt='put')
                    current_cost = (cs - cl) + (ps - pl)

                unrealized = pos['credit'] - current_cost

                # Profit target
                if unrealized >= pos['credit'] * profit_target:
                    pnl = unrealized * 100 - 3
                    equity += pnl
                    trades.append({
                        'entry': str(pos['entry_date']), 'exit': str(date),
                        'type': pos['type'], 'pnl': round(pnl, 2),
                        'win': True, 'regime': regime,
                        'vix_entry': pos['vix_entry'], 'vix_exit': round(vix_now, 1),
                        'exit_reason': 'profit_target'
                    })
                    continue

                # Stop loss
                if unrealized < -pos['credit'] * stop_loss:
                    pnl = unrealized * 100 - 3
                    equity += pnl
                    trades.append({
                        'entry': str(pos['entry_date']), 'exit': str(date),
                        'type': pos['type'], 'pnl': round(pnl, 2),
                        'win': False, 'regime': regime,
                        'vix_entry': pos['vix_entry'], 'vix_exit': round(vix_now, 1),
                        'exit_reason': 'stop_loss'
                    })
                    continue

            new_open.append(pos)

        open_positions = new_open

        # Skip if at max concurrent
        if len(open_positions) >= max_concurrent:
            continue

        # Only enter on Mon/Fri
        if date.weekday() not in [0, 4]:
            continue

        T = dte / 365

        # === ENTRY LOGIC BY STRATEGY ===

        if strategy == 'mean_rev':
            # Sell call spreads when VIX spikes
            if vix < entry_threshold:
                continue

            short_strike = round(vix + 2)  # Slightly OTM
            long_strike = short_strike + spread_width

            short_px = bs_price(vix, short_strike, T, vix_vol, opt='call')
            long_px = bs_price(vix, long_strike, T, vix_vol, opt='call')
            credit = short_px - long_px

            if credit < 0.30:
                continue

            max_loss = spread_width - credit
            if max_loss * 100 > equity * 0.15:
                continue

            equity -= 3  # Commission
            open_positions.append({
                'type': 'call_spread_short',
                'entry_date': date, 'initial_dte': dte,
                'short_strike': short_strike, 'long_strike': long_strike,
                'credit': credit, 'vix_entry': round(vix, 1),
            })

        elif strategy == 'crash_hedge':
            # Buy cheap VIX calls when VIX is low
            if vix > 16:
                continue

            strike = round(vix * 1.5)  # 50% OTM — cheap lottery tickets
            call_price = bs_price(vix, strike, T, vix_vol, opt='call')

            if call_price < 0.10 or call_price > 2.0:
                continue

            # Spend 1% of equity on hedge
            budget = equity * 0.01
            n_contracts = max(1, int(budget / (call_price * 100)))
            n_contracts = min(n_contracts, 3)

            cost = call_price * 100 * n_contracts + 3
            if cost > equity * 0.05:
                continue

            equity -= cost
            open_positions.append({
                'type': 'call_long',
                'entry_date': date, 'initial_dte': dte,
                'strike': strike, 'cost': call_price,
                'n_contracts': n_contracts, 'vix_entry': round(vix, 1),
            })

        elif strategy == 'put_spread':
            # Sell put spreads (VRP harvest) — VIX usually overpriced
            if vix < 14:  # Don't sell when VIX is already very low
                continue

            short_strike = max(10, round(vix - 3))  # Slightly OTM put
            long_strike = max(8, short_strike - spread_width)

            short_px = bs_price(vix, short_strike, T, vix_vol, opt='put')
            long_px = bs_price(vix, long_strike, T, vix_vol, opt='put')
            credit = short_px - long_px

            if credit < 0.20:
                continue

            max_loss = spread_width - credit
            if max_loss * 100 > equity * 0.15:
                continue

            equity -= 3
            open_positions.append({
                'type': 'put_spread_short',
                'entry_date': date, 'initial_dte': dte,
                'short_strike': short_strike, 'long_strike': long_strike,
                'credit': credit, 'vix_entry': round(vix, 1),
            })

        elif strategy == 'iron_condor':
            # Iron condor on VIX — range-bound thesis
            if vix < 14 or vix > 35:  # Only when VIX in normal range
                continue

            call_short = round(vix + 5)
            call_long = call_short + spread_width
            put_short = max(10, round(vix - 5))
            put_long = max(8, put_short - spread_width)

            cs_px = bs_price(vix, call_short, T, vix_vol, opt='call')
            cl_px = bs_price(vix, call_long, T, vix_vol, opt='call')
            ps_px = bs_price(vix, put_short, T, vix_vol, opt='put')
            pl_px = bs_price(vix, put_long, T, vix_vol, opt='put')

            credit = (cs_px - cl_px) + (ps_px - pl_px)
            if credit < 0.40:
                continue

            max_loss = spread_width - credit
            if max_loss * 100 > equity * 0.15:
                continue

            equity -= 3
            open_positions.append({
                'type': 'iron_condor',
                'entry_date': date, 'initial_dte': dte,
                'call_short': call_short, 'call_long': call_long,
                'put_short': put_short, 'put_long': put_long,
                'credit': credit, 'vix_entry': round(vix, 1),
            })

        elif strategy == 'combined':
            # Dynamic: spike → sell call spreads, calm → sell put spreads
            if vix > entry_threshold:
                # Mean reversion: sell call spreads
                short_strike = round(vix + 2)
                long_strike = short_strike + spread_width

                short_px = bs_price(vix, short_strike, T, vix_vol, opt='call')
                long_px = bs_price(vix, long_strike, T, vix_vol, opt='call')
                credit = short_px - long_px

                if credit >= 0.30:
                    max_loss = spread_width - credit
                    if max_loss * 100 <= equity * 0.15:
                        equity -= 3
                        open_positions.append({
                            'type': 'call_spread_short',
                            'entry_date': date, 'initial_dte': dte,
                            'short_strike': short_strike, 'long_strike': long_strike,
                            'credit': credit, 'vix_entry': round(vix, 1),
                        })

            elif vix >= 14:
                # Normal: sell put spreads (VRP)
                short_strike = max(10, round(vix - 3))
                long_strike = max(8, short_strike - spread_width)

                short_px = bs_price(vix, short_strike, T, vix_vol, opt='put')
                long_px = bs_price(vix, long_strike, T, vix_vol, opt='put')
                credit = short_px - long_px

                if credit >= 0.20:
                    max_loss = spread_width - credit
                    if max_loss * 100 <= equity * 0.15:
                        equity -= 3
                        open_positions.append({
                            'type': 'put_spread_short',
                            'entry_date': date, 'initial_dte': dte,
                            'short_strike': short_strike, 'long_strike': long_strike,
                            'credit': credit, 'vix_entry': round(vix, 1),
                        })

    # Close remaining
    last_date = dates[-1]
    for pos in open_positions:
        vix_now = float(vix_close.iloc[-1])
        if pos['type'] == 'call_spread_short':
            short_val = max(0, vix_now - pos['short_strike'])
            long_val = max(0, vix_now - pos['long_strike'])
            pnl = pos['credit'] * 100 - (short_val - long_val) * 100 - 3
        elif pos['type'] == 'put_spread_short':
            short_val = max(0, pos['short_strike'] - vix_now)
            long_val = max(0, pos['long_strike'] - vix_now)
            pnl = pos['credit'] * 100 - (short_val - long_val) * 100 - 3
        elif pos['type'] == 'call_long':
            intrinsic = max(0, vix_now - pos['strike']) * 100
            pnl = intrinsic - pos['cost'] * 100 - 3
        elif pos['type'] == 'iron_condor':
            call_stl = max(0, vix_now - pos['call_short']) - max(0, vix_now - pos['call_long'])
            put_stl = max(0, pos['put_short'] - vix_now) - max(0, pos['put_long'] - vix_now)
            pnl = pos['credit'] * 100 - (call_stl + put_stl) * 100 - 3
        else:
            pnl = 0

        equity += pnl
        trades.append({
            'entry': str(pos['entry_date']), 'exit': str(last_date),
            'type': pos['type'], 'pnl': round(pnl, 2),
            'win': pnl > 0, 'regime': 'bull',
            'vix_entry': pos['vix_entry'], 'vix_exit': round(vix_now, 1),
        })

    if not trades:
        fprint("  No trades!")
        return None

    # === Metrics ===
    n_trades = len(trades)
    wins = sum(1 for t in trades if t['win'])
    wr = wins / n_trades * 100

    total_pnl = sum(t['pnl'] for t in trades)

    # Monthly P&L
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
    calmar = cagr / abs(maxdd) if maxdd < 0 else 999

    pf = abs(sum(t['pnl'] for t in trades if t['win']) / (sum(t['pnl'] for t in trades if not t['win']) + 1e-10))

    bull_t = [t for t in trades if t['regime'] == 'bull']
    bear_t = [t for t in trades if t['regime'] == 'bear']
    bull_wr = sum(1 for t in bull_t if t['win']) / max(len(bull_t), 1) * 100
    bear_wr = sum(1 for t in bear_t if t['win']) / max(len(bear_t), 1) * 100
    r1_gap = abs(bull_wr - bear_wr) / max(bull_wr, bear_wr, 1)

    result = {
        'name': name, 'strategy': strategy,
        'n_trades': n_trades, 'win_rate': round(wr, 1),
        'total_pnl': round(total_pnl, 2), 'final_equity': round(equity, 2),
        'cagr_pct': round(cagr * 100, 1), 'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2), 'maxdd_pct': round(maxdd * 100, 1),
        'calmar': round(calmar, 2), 'pf': round(pf, 2),
        'r1_gap': round(r1_gap, 3), 'r1_pass': r1_gap <= 0.50,
        'bull_wr': round(bull_wr, 1), 'bear_wr': round(bear_wr, 1),
        'bull_trades': len(bull_t), 'bear_trades': len(bear_t),
        'monthly_returns': monthly.tolist(),
    }

    fprint(f"  {name}: {n_trades} trades, WR {wr:.1f}%, Sharpe {sharpe:.2f}, "
           f"CAGR {cagr*100:.1f}%, MaxDD {maxdd*100:.1f}%, PF {pf:.2f}")
    fprint(f"    ${capital:,} → ${equity:,.0f} | Bull WR {bull_wr:.0f}% ({len(bull_t)}) Bear WR {bear_wr:.0f}% ({len(bear_t)})")

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

    fprint(f"VIX Options Income & Hedge v1 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 70)

    raw = yf.download(['SPY', '^VIX'], start='2008-01-01', end='2026-07-25', progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw['Close']
    else:
        close = raw

    spy = close['SPY'].dropna()
    vix_col = '^VIX' if '^VIX' in close.columns else 'VIX'
    vix = close[vix_col].dropna()
    common = spy.index.intersection(vix.index)
    spy, vix = spy.loc[common], vix.loc[common]

    fprint(f"Data: {len(spy)} days, {spy.index[0].strftime('%Y-%m-%d')} to {spy.index[-1].strftime('%Y-%m-%d')}")

    # === Variants ===
    variants = [
        # (name, strategy, entry_thresh, exit_thresh, dte, spread_width, profit_target, stop_loss, capital)
        ('A_MeanRev_VIX25', 'mean_rev', 25, 18, 30, 5, 0.50, 2.0, 10000),
        ('B_MeanRev_VIX30', 'mean_rev', 30, 20, 30, 5, 0.50, 2.0, 10000),
        ('C_MeanRev_45DTE', 'mean_rev', 25, 18, 45, 5, 0.50, 2.0, 10000),
        ('D_CrashHedge', 'crash_hedge', 16, 12, 60, 5, 0.50, 2.0, 10000),
        ('E_PutSpread_VRP', 'put_spread', 14, 10, 30, 5, 0.50, 2.0, 10000),
        ('F_IronCondor', 'iron_condor', 14, 10, 30, 5, 0.50, 2.0, 10000),
        ('G_Combined', 'combined', 25, 18, 30, 5, 0.50, 2.0, 10000),
        ('H_Combined_645', 'combined', 25, 18, 30, 3, 0.50, 2.0, 645),  # Agentic account
    ]

    results = []

    if MLFLOW_OK:
        exp_name = 'vix_options_income_v1'
        try:
            if not mlflow.get_experiment_by_name(exp_name):
                mlflow.create_experiment(exp_name)
        except:
            pass
        mlflow.set_experiment(exp_name)

    for vname, strat, entry_t, exit_t, dte, sw, pt, sl, cap in variants:
        try:
            r = simulate_vix_strategy(
                vix, spy, strategy=strat,
                capital=cap, name=vname,
                entry_threshold=entry_t, exit_threshold=exit_t,
                dte=dte, spread_width=sw,
                profit_target=pt, stop_loss=sl
            )
            if r:
                if MLFLOW_OK:
                    with mlflow.start_run(run_name=vname):
                        mlflow.log_params({
                            'strategy': strat, 'entry_threshold': entry_t,
                            'dte': dte, 'spread_width': sw,
                            'profit_target': pt, 'stop_loss': sl, 'capital': cap
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

    # === Adversarial Validation ===
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
    fprint("SUMMARY — VIX Options Income & Hedge v1")
    fprint("=" * 70)
    fprint(f"{'Name':<25} {'Type':<12} {'Trades':>6} {'WR':>6} {'Sharpe':>7} {'CAGR':>7} {'MaxDD':>7} {'PF':>5} {'Gates':>6}")
    fprint("-" * 90)
    for r in sorted(results, key=lambda x: x['sharpe'], reverse=True):
        fprint(f"{r['name']:<25} {r['strategy']:<12} {r['n_trades']:>6} {r['win_rate']:>5.1f}% "
               f"{r['sharpe']:>7.2f} {r['cagr_pct']:>6.1f}% {r['maxdd_pct']:>6.1f}% {r['pf']:>5.2f} {r['gates_passed']:>4}/4")

    # Correlation with SPY
    fprint("\n--- VIX STRATEGY CHARACTERISTICS ---")
    fprint("VIX options are NATURALLY DECORRELATED from equity:")
    fprint("  - Mean reversion strategies PROFIT from vol spikes (bear market = more entries)")
    fprint("  - Crash hedges are ANTI-correlated with SPY (gain when market drops)")
    fprint("  - VRP harvesting profits from vol premium (works in most regimes)")
    fprint("  - Iron condors profit from VIX mean reversion to normal range")

    save = [{k: v for k, v in r.items() if k != 'monthly_returns'} for r in results]
    with open(RESULTS_PATH, 'w') as f:
        json.dump(save, f, indent=2, default=str)
    fprint(f"\nDone — {datetime.now().strftime('%H:%M:%S')}")


if __name__ == '__main__':
    main()
