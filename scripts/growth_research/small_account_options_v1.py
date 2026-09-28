#!/usr/bin/env python3
"""
Small Account Options Playbook v1 — $645 Account Strategies
=============================================================

HC #749: Agentic account ($645) is options-only, max $200-300/trade.

This tests EVERY options strategy type we've validated at $10K+ but
specifically sized for $645:
1. Bull call spreads on top momentum ETFs (our best edge)
2. Narrow iron condors on SPY (income, small capital)
3. Put credit spreads on momentum (bullish + premium)
4. Cheap LEAPS on sector ETFs
5. VIX call spreads after spikes (mean-reversion)
6. Combined: momentum spreads + IC income

KEY CONSTRAINT: Max $200 per trade, max 3 concurrent positions.
Commission: $0.65/contract (Robinhood-level).

Focus on ETFs with cheap options: XLE (~$60), XLF (~$45), XLY (~$200),
XLP (~$85), XLU (~$85), IWM (~$215), EEM (~$43).
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
RESULTS_PATH = RESULTS_DIR / 'small_account_options_v1_results.json'

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


# ====== MOMENTUM RANKING (simplified, fast) ======

def momentum_rank(close_df, date, top_n=3):
    """Simple 12-1 momentum ranking."""
    scores = {}
    for col in close_df.columns:
        px = close_df[col].loc[:date].dropna()
        if len(px) > 252:
            ret_12m = px.iloc[-1] / px.iloc[-252] - 1
            ret_1m = px.iloc[-1] / px.iloc[-21] - 1
            scores[col] = ret_12m - ret_1m
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_n]


# ====== STRATEGY A: MOMENTUM BULL CALL SPREADS ======

def simulate_momentum_spreads(close_df, vix_close, spy_close,
                              top_n=3, spread_pct=3.0, dte=45,
                              capital=645, max_per_trade=200,
                              max_concurrent=3, name='base'):
    """Buy bull call spreads on top momentum ETFs. ~$100-200 per spread."""
    fprint(f"\n--- {name} ---")

    spy_sma200 = spy_close.rolling(200).mean()
    month_ends = close_df.index.to_series().resample('M').last().dropna()

    trades = []
    equity = capital
    open_pos = []

    for i in range(13, len(month_ends) - 2):
        date = month_ends.iloc[i]
        next_date = month_ends.iloc[i + 1]

        spy_val = float(spy_close.loc[date]) if date in spy_close.index else 0
        sma_val = float(spy_sma200.loc[date]) if date in spy_sma200.index else spy_val
        regime = 'bear' if spy_val < sma_val else 'bull'
        vix = float(vix_close.loc[date]) if date in vix_close.index else 15
        sigma = vix / 100

        # Close existing positions (at next month end = simplified expiry)
        for pos in open_pos:
            S_now = float(close_df[pos['ticker']].loc[next_date]) if next_date in close_df[pos['ticker']].index else pos['entry_px']

            # Spread value at expiry (simplified to intrinsic)
            long_val = max(0, S_now - pos['long_strike'])
            short_val = max(0, S_now - pos['short_strike'])
            spread_val = (long_val - short_val) * 100

            pnl = spread_val - pos['cost'] - 1.30  # Commission both legs
            equity += pnl
            trades.append({
                'entry': str(pos['entry_date']), 'exit': str(next_date),
                'ticker': pos['ticker'], 'pnl': round(pnl, 2),
                'win': pnl > 0, 'regime': regime,
                'underlying_ret': round((S_now / pos['entry_px'] - 1) * 100, 1),
            })

        open_pos = []

        # New positions
        if equity < 50:  # Busted
            continue

        top_picks = momentum_rank(close_df, date, top_n)

        for ticker in top_picks[:max_concurrent]:
            if ticker not in close_df.columns:
                continue

            S = float(close_df[ticker].loc[date])
            T = dte / 365

            # ATM call spread: buy ATM, sell OTM
            K_long = round(S)
            K_short = round(S * (1 + spread_pct / 100))

            long_price = bs_price(S, K_long, T, sigma, opt='call')
            short_price = bs_price(S, K_short, T, sigma, opt='call')
            net_debit = long_price - short_price

            cost = net_debit * 100  # Per contract
            if cost <= 0 or cost > max_per_trade or cost > equity * 0.40:
                continue

            equity -= 1.30  # Commission
            open_pos.append({
                'ticker': ticker,
                'entry_date': date,
                'long_strike': K_long,
                'short_strike': K_short,
                'cost': cost,
                'entry_px': S,
            })

    # Close remaining
    if open_pos:
        last_date = month_ends.iloc[-1]
        for pos in open_pos:
            S_now = float(close_df[pos['ticker']].iloc[-1])
            long_val = max(0, S_now - pos['long_strike'])
            short_val = max(0, S_now - pos['short_strike'])
            spread_val = (long_val - short_val) * 100
            pnl = spread_val - pos['cost'] - 1.30
            equity += pnl
            trades.append({
                'entry': str(pos['entry_date']), 'exit': str(last_date),
                'pnl': round(pnl, 2), 'win': pnl > 0, 'regime': 'bull',
            })

    return compute_results(trades, equity, capital, name)


# ====== STRATEGY B: NARROW IRON CONDORS ======

def simulate_narrow_ic(spy_close, vix_close, capital=645,
                       delta=0.20, width=2, dte=30,
                       max_per_trade=200, name='base'):
    """Narrow iron condors on SPY — small enough for $645."""
    fprint(f"\n--- {name} ---")

    spy_sma200 = spy_close.rolling(200).mean()
    vix_rank = vix_close.rolling(252).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10) * 100
    )

    dates = spy_close.index[252:]
    trades = []
    equity = capital
    open_trades = []

    for i in range(len(dates) - dte - 1):
        date = dates[i]
        S = float(spy_close.loc[date])
        vix = float(vix_close.loc[date]) if date in vix_close.index else 15
        sigma = vix / 100
        spy_val = S
        sma_val = float(spy_sma200.loc[date]) if date in spy_sma200.index else S
        regime = 'bear' if spy_val < sma_val else 'bull'

        # Close expired/exited
        new_open = []
        for ot in open_trades:
            days_held = (date - ot['entry_date']).days
            if days_held >= dte:
                # Expired
                call_val = max(0, S - ot['call_short']) - max(0, S - ot['call_long'])
                put_val = max(0, ot['put_short'] - S) - max(0, ot['put_long'] - S)
                settlement = (call_val + put_val) * 100
                pnl = ot['credit'] * 100 - settlement - 2.60
                equity += pnl
                trades.append({
                    'entry': str(ot['entry_date']), 'exit': str(date),
                    'pnl': round(pnl, 2), 'win': pnl > 0, 'regime': regime,
                })
            else:
                new_open.append(ot)
        open_trades = new_open

        if len(open_trades) >= 2:
            continue

        if date.weekday() not in [0, 4]:
            continue

        T = dte / 365

        # Strikes
        d1_call = norm.ppf(1 - delta)
        d1_put = norm.ppf(delta)
        call_short = S * np.exp(d1_call * sigma * np.sqrt(T))
        put_short = S * np.exp(d1_put * sigma * np.sqrt(T))
        call_long = call_short + width
        put_long = put_short - width

        # Price
        sc = bs_price(S, call_short, T, sigma, opt='call')
        lc = bs_price(S, call_long, T, sigma, opt='call')
        sp = bs_price(S, put_short, T, sigma, opt='put')
        lp = bs_price(S, put_long, T, sigma, opt='put')
        credit = (sc - lc) + (sp - lp)

        max_loss = width - credit
        margin = max_loss * 100

        if margin > max_per_trade or margin > equity * 0.35:
            continue
        if credit < 0.10:
            continue

        equity -= 2.60  # Commission (4 legs)
        open_trades.append({
            'entry_date': date,
            'call_short': call_short, 'call_long': call_long,
            'put_short': put_short, 'put_long': put_long,
            'credit': credit,
        })

    # Close remaining
    for ot in open_trades:
        S = float(spy_close.iloc[-1])
        pnl = ot['credit'] * 100 - 2.60
        equity += pnl
        trades.append({'entry': str(ot['entry_date']), 'exit': str(dates[-1]),
                       'pnl': round(pnl, 2), 'win': pnl > 0, 'regime': 'bull'})

    return compute_results(trades, equity, capital, name)


# ====== STRATEGY C: PUT CREDIT SPREADS ON MOMENTUM ======

def simulate_put_credit_spreads(close_df, vix_close, spy_close,
                                top_n=2, spread_pct=3.0, dte=30,
                                capital=645, max_per_trade=200,
                                name='base'):
    """Sell put spreads on top momentum ETFs — bullish + premium."""
    fprint(f"\n--- {name} ---")

    spy_sma200 = spy_close.rolling(200).mean()
    month_ends = close_df.index.to_series().resample('M').last().dropna()

    trades = []
    equity = capital

    for i in range(13, len(month_ends) - 2):
        date = month_ends.iloc[i]
        next_date = month_ends.iloc[i + 1]

        spy_val = float(spy_close.loc[date]) if date in spy_close.index else 0
        sma_val = float(spy_sma200.loc[date]) if date in spy_sma200.index else spy_val
        regime = 'bear' if spy_val < sma_val else 'bull'
        vix = float(vix_close.loc[date]) if date in vix_close.index else 15
        sigma = vix / 100

        top_picks = momentum_rank(close_df, date, top_n)

        for ticker in top_picks:
            if ticker not in close_df.columns:
                continue

            S = float(close_df[ticker].loc[date])
            T = dte / 365

            # Put credit spread: sell slightly OTM put, buy further OTM put
            K_short = round(S * (1 - spread_pct / 100))  # OTM put
            K_long = round(S * (1 - 2 * spread_pct / 100))  # Further OTM put

            short_put = bs_price(S, K_short, T, sigma, opt='put')
            long_put = bs_price(S, K_long, T, sigma, opt='put')
            credit = short_put - long_put

            if credit < 0.10:
                continue

            max_loss = (K_short - K_long) - credit
            margin = max_loss * 100

            if margin > max_per_trade or margin > equity * 0.35:
                continue

            # At expiry
            S_next = float(close_df[ticker].loc[next_date]) if next_date in close_df[ticker].index else S
            short_val = max(0, K_short - S_next)
            long_val = max(0, K_long - S_next)
            settlement = (short_val - long_val) * 100

            pnl = credit * 100 - settlement - 1.30
            equity += pnl
            trades.append({
                'entry': str(date), 'exit': str(next_date),
                'ticker': ticker, 'pnl': round(pnl, 2),
                'win': pnl > 0, 'regime': regime,
            })

    return compute_results(trades, equity, capital, name)


# ====== STRATEGY D: CHEAP ETF CALLS (LOTTERY APPROACH) ======

def simulate_cheap_calls(close_df, vix_close, spy_close,
                         top_n=2, otm_pct=5.0, dte=45,
                         capital=645, max_per_trade=150,
                         name='base'):
    """Buy cheap OTM calls on top momentum ETFs. Asymmetric payoff."""
    fprint(f"\n--- {name} ---")

    spy_sma200 = spy_close.rolling(200).mean()
    month_ends = close_df.index.to_series().resample('M').last().dropna()

    trades = []
    equity = capital

    for i in range(13, len(month_ends) - 2):
        date = month_ends.iloc[i]
        next_date = month_ends.iloc[i + 1]

        regime = 'bear' if float(spy_close.loc[date]) < float(spy_sma200.loc[date]) else 'bull'
        vix = float(vix_close.loc[date]) if date in vix_close.index else 15
        sigma = vix / 100

        top_picks = momentum_rank(close_df, date, top_n)

        for ticker in top_picks:
            if ticker not in close_df.columns:
                continue

            S = float(close_df[ticker].loc[date])
            T = dte / 365

            K = round(S * (1 + otm_pct / 100))
            call_price = bs_price(S, K, T, sigma, opt='call')
            cost = call_price * 100

            if cost < 5 or cost > max_per_trade or cost > equity * 0.25:
                continue

            # Budget: spend max 15% of equity per call
            equity -= cost + 0.65

            # At expiry
            S_next = float(close_df[ticker].loc[next_date]) if next_date in close_df[ticker].index else S
            intrinsic = max(0, S_next - K) * 100
            pnl = intrinsic - cost - 0.65

            equity += intrinsic
            trades.append({
                'entry': str(date), 'exit': str(next_date),
                'ticker': ticker, 'pnl': round(pnl, 2),
                'win': pnl > 0, 'regime': regime,
            })

    return compute_results(trades, equity, capital, name)


# ====== COMMON METRICS ======

def compute_results(trades, equity, capital, name):
    if not trades:
        fprint("  No trades!")
        return None

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

    result = {
        'name': name, 'n_trades': n_trades, 'win_rate': round(wr, 1),
        'total_pnl': round(total_pnl, 2), 'final_equity': round(equity, 2),
        'cagr_pct': round(cagr * 100, 1), 'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2), 'maxdd_pct': round(maxdd * 100, 1),
        'pf': round(pf, 2),
        'r1_gap': round(r1_gap, 3), 'r1_pass': r1_gap <= 0.50,
        'bull_wr': round(bull_wr, 1), 'bear_wr': round(bear_wr, 1),
        'bull_trades': len(bull_t), 'bear_trades': len(bear_t),
        'monthly_returns': monthly.tolist(),
    }

    fprint(f"  {name}: {n_trades} trades, WR {wr:.1f}%, Sharpe {sharpe:.2f}, "
           f"CAGR {cagr*100:.1f}%, MaxDD {maxdd*100:.1f}%, PF {pf:.2f}")
    fprint(f"    ${capital} → ${equity:.0f} | Bull WR {bull_wr:.0f}% ({len(bull_t)}) Bear WR {bear_wr:.0f}% ({len(bear_t)})")

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

    fprint(f"Small Account Options Playbook v1 ($645) — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 70)

    # Cheap ETFs suitable for $645 account
    tickers = ['XLE', 'XLF', 'XLY', 'XLP', 'XLU', 'XLB', 'XLI', 'XLK',
               'IWM', 'EEM', 'EFA', 'GLD', 'SPY', 'QQQ']

    raw = yf.download(tickers + ['^VIX'], start='2008-01-01', end='2026-07-25', progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw['Close']
    else:
        close = raw

    vix_col = '^VIX' if '^VIX' in close.columns else 'VIX'
    vix = close[vix_col].dropna()
    spy = close['SPY'].dropna()

    etf_close = close[[c for c in tickers if c in close.columns]].dropna(how='all')
    common = etf_close.index.intersection(vix.index).intersection(spy.index)
    etf_close = etf_close.loc[common]
    vix = vix.loc[common]
    spy = spy.loc[common]

    # Only use cheap ETFs for the playbook (price < $100)
    cheap_tickers = [c for c in etf_close.columns if float(etf_close[c].iloc[-1]) < 120]
    cheap_close = etf_close[cheap_tickers]

    fprint(f"Data: {len(common)} days, {len(cheap_tickers)} cheap ETFs (<$120)")
    fprint(f"Cheap ETFs: {cheap_tickers}")

    results = []

    if MLFLOW_OK:
        exp_name = 'small_account_options_v1'
        try:
            if not mlflow.get_experiment_by_name(exp_name):
                mlflow.create_experiment(exp_name)
        except:
            pass
        mlflow.set_experiment(exp_name)

    # === A: Momentum Bull Call Spreads ===
    for spread_pct, dte, n, vname in [
        (3, 45, 2, 'A1_BullSpread_3pct_45d'),
        (5, 45, 2, 'A2_BullSpread_5pct_45d'),
        (3, 30, 3, 'A3_BullSpread_3pct_30d_T3'),
        (3, 60, 2, 'A4_BullSpread_3pct_60d'),
    ]:
        r = simulate_momentum_spreads(
            cheap_close, vix, spy,
            top_n=n, spread_pct=spread_pct, dte=dte,
            capital=645, max_per_trade=200, name=vname
        )
        if r:
            results.append(r)

    # === B: Narrow Iron Condors ===
    for width, delta, vname in [
        (2, 0.20, 'B1_NarrowIC_2wide_20d'),
        (3, 0.16, 'B2_NarrowIC_3wide_16d'),
        (2, 0.10, 'B3_NarrowIC_2wide_10d'),
    ]:
        r = simulate_narrow_ic(spy, vix, capital=645, delta=delta, width=width, name=vname)
        if r:
            results.append(r)

    # === C: Put Credit Spreads ===
    for spread_pct, n, vname in [
        (3, 2, 'C1_PutCredit_3pct_T2'),
        (5, 2, 'C2_PutCredit_5pct_T2'),
        (3, 1, 'C3_PutCredit_3pct_T1'),
    ]:
        r = simulate_put_credit_spreads(
            cheap_close, vix, spy,
            top_n=n, spread_pct=spread_pct, dte=30,
            capital=645, max_per_trade=200, name=vname
        )
        if r:
            results.append(r)

    # === D: Cheap OTM Calls ===
    for otm_pct, n, vname in [
        (5, 2, 'D1_CheapCalls_5pctOTM'),
        (3, 2, 'D2_CheapCalls_3pctOTM'),
        (5, 1, 'D3_CheapCalls_5pct_T1'),
    ]:
        r = simulate_cheap_calls(
            cheap_close, vix, spy,
            top_n=n, otm_pct=otm_pct, dte=45,
            capital=645, max_per_trade=150, name=vname
        )
        if r:
            results.append(r)

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
        subs = [np.mean(rets[j*chunk:(j+1)*chunk])*12 / (np.std(rets[j*chunk:(j+1)*chunk])*np.sqrt(12)+1e-10)
                if len(rets[j*chunk:(j+1)*chunk]) > 1 else 0 for j in range(3)]
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
    fprint("SUMMARY — Small Account Options Playbook v1 ($645)")
    fprint("=" * 70)
    fprint(f"{'Name':<30} {'Trades':>6} {'WR':>6} {'Sharpe':>7} {'CAGR':>7} "
           f"{'MaxDD':>7} {'PF':>5} {'Final$':>7} {'Gates':>6}")
    fprint("-" * 95)
    for r in sorted(results, key=lambda x: x['sharpe'], reverse=True):
        fprint(f"{r['name']:<30} {r['n_trades']:>6} {r['win_rate']:>5.1f}% {r['sharpe']:>7.2f} "
               f"{r['cagr_pct']:>6.1f}% {r['maxdd_pct']:>6.1f}% {r['pf']:>5.2f} "
               f"${r['final_equity']:>6.0f} {r['gates_passed']:>4}/4")

    # Verdict
    fprint("\n--- VERDICT FOR $645 ACCOUNT ---")
    winners = [r for r in results if r['gates_passed'] >= 3]
    if winners:
        best = max(winners, key=lambda x: x['sharpe'])
        fprint(f"  BEST: {best['name']} — Sharpe {best['sharpe']}, CAGR {best['cagr_pct']}%, "
               f"MaxDD {best['maxdd_pct']}%, ${645} → ${best['final_equity']:.0f}")
    else:
        fprint("  NO strategy passes 3+ gates at $645. Account is too small for systematic options.")
        fprint("  RECOMMENDATION: Focus on 1-2 high-conviction discretionary plays per month")
        fprint("  using the play scanner (momentum + flow + earnings signals)")

    save = [{k: v for k, v in r.items() if k != 'monthly_returns'} for r in results]
    with open(RESULTS_PATH, 'w') as f:
        json.dump(save, f, indent=2, default=str)
    fprint(f"\nDone — {datetime.now().strftime('%H:%M:%S')}")


if __name__ == '__main__':
    main()
