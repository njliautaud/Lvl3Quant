#!/usr/bin/env python3
"""
SPY Iron Condor Income Strategy v1
====================================

Prior findings: ALL directional options strategies on momentum ETFs failed (0/4 gates).
Momentum signal doesn't transfer to options.

NEW APPROACH: Non-directional premium selling. Iron condors on SPY/QQQ collect theta
decay regardless of direction. Edge comes from:
1. Selling options when IV rank is high (mean-reverting vol)
2. Wide enough wings that SPY stays inside 68-80% of the time
3. Active management (close at 50% profit, roll if tested)

Strategy:
- Sell monthly iron condors on SPY (most liquid, tightest spreads)
- Short strikes at ~16-delta (1 SD), width 5-10 points
- Entry: when IV rank > 30% (avoid selling cheap vol)
- Exit: 50% profit target, or 21 DTE (whichever first)
- Stop: close if short strike breached

Variants:
A. 16-delta, $5 wide, IV rank > 30%
B. 16-delta, $10 wide, IV rank > 30%
C. 10-delta, $5 wide, IV rank > 30% (wider, fewer tests)
D. 16-delta, $5 wide, IV rank > 50% (more selective)
E. 20-delta, $5 wide, IV rank > 30% (tighter, more premium)
F. 16-delta, $5 wide, NO IV filter (always sell)

Simplified model: Use historical SPY returns + VIX as IV proxy.
$645 account → 1 contract per trade (~$100-300 margin requirement for defined risk).
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

def fprint(*args, **kwargs):
    print(*args, **kwargs, flush=True)

RESULTS_DIR = Path('/home/jupiter/Lvl3Quant/research/findings')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = RESULTS_DIR / 'spy_iron_condor_income_v1_results.json'

# MLflow
MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen('http://jupiter:5000/', timeout=2)
    import mlflow
    mlflow.set_tracking_uri('http://jupiter:5000')
    MLFLOW_OK = True
except:
    pass


def estimate_option_price(S, K, T, sigma, r=0.04, option_type='call'):
    """Black-Scholes option price estimate."""
    if T <= 0 or sigma <= 0:
        return max(0, S - K) if option_type == 'call' else max(0, K - S)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def get_strike_at_delta(S, T, sigma, target_delta, r=0.04, option_type='call'):
    """Find strike price for a given delta using bisection."""
    if T <= 0:
        return S

    if option_type == 'call':
        lo, hi = S * 0.8, S * 1.5
    else:
        lo, hi = S * 0.5, S * 1.2

    for _ in range(50):
        mid = (lo + hi) / 2
        d1 = (np.log(S / mid) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

        if option_type == 'call':
            delta = norm.cdf(d1)
            if delta > target_delta:
                lo = mid
            else:
                hi = mid
        else:
            delta = -norm.cdf(-d1)
            if abs(delta) > target_delta:
                hi = mid
            else:
                lo = mid

    return round((lo + hi) / 2)


def simulate_iron_condor(spy_close, vix_close, delta=0.16, width=5, iv_rank_min=30,
                          dte=30, profit_target=0.50, name='base', capital=645):
    """
    Simulate iron condor strategy on SPY.

    Iron condor = sell call spread + sell put spread.
    Credit received = premium of short strikes - premium of long strikes.
    Max loss = width - credit.
    Max profit = credit received.
    """
    fprint(f"\n--- {name} ---")

    # Calculate IV rank (VIX percentile over trailing 252 days)
    vix_rank = vix_close.rolling(252).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10) * 100
    )

    # Trading days
    dates = spy_close.index[252:]  # Need 252 days of VIX history

    trades = []
    all_returns = []
    all_dates = []
    all_regimes = []

    spy_sma200 = spy_close.rolling(200).mean()

    i = 0
    while i < len(dates) - dte - 1:
        date = dates[i]

        # Check IV rank filter
        if date not in vix_rank.index:
            i += 1
            continue

        ivr = vix_rank.loc[date]
        if pd.isna(ivr) or ivr < iv_rank_min:
            i += 1
            continue

        S = float(spy_close.loc[date])
        vix = float(vix_close.loc[date])
        sigma = vix / 100  # VIX is annualized vol in %
        T = dte / 252

        # Find short strikes at target delta
        call_strike = get_strike_at_delta(S, T, sigma, delta, option_type='call')
        put_strike = get_strike_at_delta(S, T, sigma, delta, option_type='put')

        # Long strikes (wings)
        call_long_strike = call_strike + width
        put_long_strike = put_strike - width

        # Calculate premiums
        short_call_prem = estimate_option_price(S, call_strike, T, sigma, option_type='call')
        long_call_prem = estimate_option_price(S, call_long_strike, T, sigma, option_type='call')
        short_put_prem = estimate_option_price(S, put_strike, T, sigma, option_type='put')
        long_put_prem = estimate_option_price(S, put_long_strike, T, sigma, option_type='put')

        call_spread_credit = short_call_prem - long_call_prem
        put_spread_credit = short_put_prem - long_put_prem
        total_credit = call_spread_credit + put_spread_credit

        max_loss = width - total_credit

        if total_credit <= 0 or max_loss <= 0:
            i += 1
            continue

        # Check if trade fits account size (margin = max_loss * 100)
        margin_req = max_loss * 100
        if margin_req > capital * 0.5:  # Don't risk more than 50% on one trade
            i += 1
            continue

        # Simulate daily P&L until exit
        entry_credit = total_credit
        exit_date = None
        exit_pnl = None

        for j in range(1, dte + 1):
            if i + j >= len(dates):
                break

            check_date = dates[i + j]
            S_now = float(spy_close.loc[check_date])
            days_left = dte - j
            T_now = max(days_left / 252, 0.001)

            # Use current VIX for pricing (IV changes)
            vix_now = float(vix_close.loc[check_date]) if check_date in vix_close.index else vix
            sigma_now = vix_now / 100

            # Current value of position
            sc = estimate_option_price(S_now, call_strike, T_now, sigma_now, option_type='call')
            lc = estimate_option_price(S_now, call_long_strike, T_now, sigma_now, option_type='call')
            sp = estimate_option_price(S_now, put_strike, T_now, sigma_now, option_type='put')
            lp = estimate_option_price(S_now, put_long_strike, T_now, sigma_now, option_type='put')

            current_cost = (sc - lc) + (sp - lp)  # Cost to close
            current_pnl = entry_credit - current_cost

            # Exit conditions
            # 1. Profit target hit
            if current_pnl >= entry_credit * profit_target:
                exit_pnl = current_pnl
                exit_date = check_date
                break

            # 2. Short strike breached (stop loss)
            if S_now >= call_strike or S_now <= put_strike:
                exit_pnl = current_pnl  # Could be negative
                exit_date = check_date
                break

            # 3. 7 DTE - close remaining
            if days_left <= 7:
                exit_pnl = current_pnl
                exit_date = check_date
                break

        if exit_pnl is None:
            # Expired - check final payoff
            S_exp = float(spy_close.iloc[min(i + dte, len(spy_close) - 1)])

            # Payoff at expiry
            call_spread_loss = max(0, S_exp - call_strike) - max(0, S_exp - call_long_strike)
            put_spread_loss = max(0, put_strike - S_exp) - max(0, put_long_strike - S_exp)

            exit_pnl = entry_credit - call_spread_loss - put_spread_loss
            exit_date = dates[min(i + dte, len(dates) - 1)]

        # Per-contract P&L ($100 multiplier)
        dollar_pnl = exit_pnl * 100

        # Commission cost (~$5 round trip for 4-leg)
        dollar_pnl -= 5.0

        pct_return = dollar_pnl / (max_loss * 100)  # Return on risk

        # Regime
        sma = spy_sma200.loc[date] if date in spy_sma200.index else S
        regime = 'bear' if S < sma else 'bull'

        trade = {
            'entry_date': str(date),
            'exit_date': str(exit_date),
            'spy_entry': round(S, 2),
            'call_strike': call_strike,
            'put_strike': put_strike,
            'credit': round(entry_credit, 2),
            'pnl': round(exit_pnl, 2),
            'dollar_pnl': round(dollar_pnl, 2),
            'pct_return': round(pct_return * 100, 1),
            'iv_rank': round(ivr, 1),
            'regime': regime,
            'win': exit_pnl > 0,
        }
        trades.append(trade)

        all_returns.append(pct_return)
        all_dates.append(date)
        all_regimes.append(regime)

        # Skip forward past this trade
        if exit_date and exit_date in dates:
            i = list(dates).index(exit_date) + 5  # Wait 5 days before next trade
        else:
            i += dte + 5

        continue

    if not trades:
        fprint(f"  No trades generated!")
        return None

    returns = np.array(all_returns)
    regimes = np.array(all_regimes)

    # Metrics
    n_trades = len(trades)
    wins = sum(1 for t in trades if t['win'])
    wr = wins / n_trades * 100

    avg_win = np.mean([t['dollar_pnl'] for t in trades if t['win']]) if wins > 0 else 0
    avg_loss = np.mean([t['dollar_pnl'] for t in trades if not t['win']]) if wins < n_trades else 0

    total_pnl = sum(t['dollar_pnl'] for t in trades)

    # Monthly returns for Sharpe calc
    # Group trades by month
    trade_df = pd.DataFrame(trades)
    trade_df['month'] = pd.to_datetime(trade_df['entry_date']).dt.to_period('M')
    monthly_pnl = trade_df.groupby('month')['dollar_pnl'].sum()

    # Convert to returns on capital
    monthly_rets = monthly_pnl / capital
    n_months = len(monthly_rets)
    n_years = n_months / 12

    if n_months < 6:
        fprint(f"  Only {n_months} months of data, skipping")
        return None

    ann_ret = monthly_rets.mean() * 12
    ann_vol = monthly_rets.std() * np.sqrt(12)
    sharpe = ann_ret / (ann_vol + 1e-10)

    downside = monthly_rets[monthly_rets < 0]
    sortino = ann_ret / (downside.std() * np.sqrt(12) + 1e-10) if len(downside) > 0 else 999

    cagr = (1 + total_pnl / capital) ** (1 / max(n_years, 0.01)) - 1

    # Cumulative returns for MaxDD
    cum_pnl = np.cumsum([t['dollar_pnl'] for t in trades])
    peak = np.maximum.accumulate(cum_pnl + capital)
    dd = (cum_pnl + capital - peak) / peak
    maxdd = dd.min()

    pf = abs(sum(t['dollar_pnl'] for t in trades if t['win']) /
             (sum(t['dollar_pnl'] for t in trades if not t['win']) + 1e-10))

    # R1 regime
    bull_trades = [t for t in trades if t['regime'] == 'bull']
    bear_trades = [t for t in trades if t['regime'] == 'bear']
    bull_wr = sum(1 for t in bull_trades if t['win']) / max(len(bull_trades), 1) * 100
    bear_wr = sum(1 for t in bear_trades if t['win']) / max(len(bear_trades), 1) * 100

    bull_avg = np.mean([t['dollar_pnl'] for t in bull_trades]) if bull_trades else 0
    bear_avg = np.mean([t['dollar_pnl'] for t in bear_trades]) if bear_trades else 0

    # Regime Sharpe from monthly returns
    monthly_df = trade_df.copy()
    monthly_df['date'] = pd.to_datetime(monthly_df['entry_date'])

    r1_gap = abs(bull_wr - bear_wr) / max(bull_wr, bear_wr, 1)

    result = {
        'name': name,
        'n_trades': n_trades,
        'win_rate': round(wr, 1),
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'total_pnl': round(total_pnl, 2),
        'cagr_pct': round(cagr * 100, 1),
        'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2),
        'maxdd_pct': round(maxdd * 100, 1),
        'pf': round(pf, 2),
        'r1_gap': round(r1_gap, 3),
        'r1_pass': r1_gap <= 0.50,
        'bull_wr': round(bull_wr, 1),
        'bear_wr': round(bear_wr, 1),
        'bull_trades': len(bull_trades),
        'bear_trades': len(bear_trades),
        'avg_credit': round(np.mean([t['credit'] for t in trades]), 2),
        'avg_days_held': round(np.mean([(pd.Timestamp(t['exit_date']) - pd.Timestamp(t['entry_date'])).days
                                        for t in trades]), 1),
        'monthly_returns': monthly_rets.tolist(),
        'delta': delta,
        'width': width,
        'iv_rank_min': iv_rank_min,
    }

    fprint(f"  {name}: {n_trades} trades, WR {wr:.1f}%, Sharpe {sharpe:.2f}, "
           f"CAGR {cagr*100:.1f}%, MaxDD {maxdd*100:.1f}%, PF {pf:.2f}")
    fprint(f"    Avg credit: ${np.mean([t['credit'] for t in trades]):.2f}/share, "
           f"Avg win: ${avg_win:.0f}, Avg loss: ${avg_loss:.0f}")
    fprint(f"    Bull WR: {bull_wr:.0f}% ({len(bull_trades)} trades), "
           f"Bear WR: {bear_wr:.0f}% ({len(bear_trades)} trades)")
    fprint(f"    Total P&L: ${total_pnl:.0f} on ${capital} capital over {n_years:.1f}y")

    return result


def permutation_test(returns, n_perms=1000):
    real_sharpe = np.mean(returns) / (np.std(returns) + 1e-10)
    n = len(returns)
    count = 0
    for _ in range(n_perms):
        perm = returns.copy()
        # Random sign flip (since timing is the edge, not direction)
        signs = np.random.choice([-1, 1], size=n)
        perm = perm * signs
        if np.mean(perm) / (np.std(perm) + 1e-10) >= real_sharpe:
            count += 1
    return count / n_perms


def main():
    import yfinance as yf

    fprint(f"SPY Iron Condor Income v1 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 70)

    fprint("Downloading data...")
    raw = yf.download(['SPY', '^VIX'], start='2008-01-01', end='2026-07-25', progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw['Close']
    else:
        close = raw

    spy_close = close['SPY'].dropna()
    vix_close = close['^VIX'].dropna() if '^VIX' in close.columns else close['VIX'].dropna()

    # Align
    common = spy_close.index.intersection(vix_close.index)
    spy_close = spy_close.loc[common]
    vix_close = vix_close.loc[common]

    fprint(f"Data: {len(spy_close)} days, {spy_close.index[0].strftime('%Y-%m-%d')} to {spy_close.index[-1].strftime('%Y-%m-%d')}")
    fprint(f"VIX range: {vix_close.min():.1f} to {vix_close.max():.1f}")

    variants = [
        ('A_16d_5w_ivr30', 0.16, 5, 30),
        ('B_16d_10w_ivr30', 0.16, 10, 30),
        ('C_10d_5w_ivr30', 0.10, 5, 30),
        ('D_16d_5w_ivr50', 0.16, 5, 50),
        ('E_20d_5w_ivr30', 0.20, 5, 30),
        ('F_16d_5w_nofilter', 0.16, 5, 0),
    ]

    results = []

    if MLFLOW_OK:
        exp_name = 'spy_iron_condor_income_v1'
        try:
            exp = mlflow.get_experiment_by_name(exp_name)
            if exp is None:
                mlflow.create_experiment(exp_name)
        except:
            pass
        mlflow.set_experiment(exp_name)

    for vname, delta, width, ivr_min in variants:
        try:
            if MLFLOW_OK:
                with mlflow.start_run(run_name=vname):
                    r = simulate_iron_condor(spy_close, vix_close, delta=delta, width=width,
                                            iv_rank_min=ivr_min, name=vname)
                    if r:
                        mlflow.log_params({'delta': delta, 'width': width, 'iv_rank_min': ivr_min})
                        mlflow.log_metrics({k: v for k, v in r.items()
                                          if isinstance(v, (int, float)) and k not in ['monthly_returns']})
                        results.append(r)
            else:
                r = simulate_iron_condor(spy_close, vix_close, delta=delta, width=width,
                                        iv_rank_min=ivr_min, name=vname)
                if r:
                    results.append(r)
        except Exception as e:
            fprint(f"  ERROR {vname}: {e}")
            import traceback
            traceback.print_exc()

    if not results:
        fprint("No results!")
        return

    # Adversarial validation
    fprint("\n" + "=" * 70)
    fprint("ADVERSARIAL VALIDATION")
    fprint("=" * 70)

    for r in results:
        rets = np.array(r['monthly_returns'])

        perm_p = permutation_test(rets)
        r['perm_p'] = round(perm_p, 3)
        r['g1_pass'] = perm_p < 0.05

        r['g2_pass'] = r['r1_pass']

        # Sub-period
        n = len(rets)
        chunk = max(n // 3, 1)
        subs = []
        for i in range(3):
            sub = rets[i*chunk:(i+1)*chunk]
            if len(sub) > 1:
                subs.append(np.mean(sub) * 12 / (np.std(sub) * np.sqrt(12) + 1e-10))
            else:
                subs.append(0)
        r['g3_pass'] = all(s > 0 for s in subs)
        r['sub_sharpes'] = [round(s, 2) for s in subs]

        # Outlier
        if n > 5:
            n_trim = max(1, int(n * 0.05))
            trimmed = np.sort(rets)[n_trim:-n_trim] if n_trim < n // 2 else rets
            orig = np.mean(rets) / (np.std(rets) + 1e-10)
            trim = np.mean(trimmed) / (np.std(trimmed) + 1e-10)
            r['g4_pass'] = trim > 0 and (trim / (orig + 1e-10)) > 0.5
        else:
            r['g4_pass'] = False

        gates = sum([r['g1_pass'], r['g2_pass'], r['g3_pass'], r['g4_pass']])
        r['gates_passed'] = gates

        fprint(f"\n{r['name']}:")
        fprint(f"  G1 Perm: {'PASS' if r['g1_pass'] else 'FAIL'} (p={r['perm_p']})")
        fprint(f"  G2 R1:   {'PASS' if r['g2_pass'] else 'FAIL'} (gap={r['r1_gap']})")
        fprint(f"  G3 Sub:  {'PASS' if r['g3_pass'] else 'FAIL'} (sharpes={r['sub_sharpes']})")
        fprint(f"  G4 Out:  {'PASS' if r.get('g4_pass') else 'FAIL'}")
        fprint(f"  GATES: {gates}/4")

    # Summary
    fprint("\n" + "=" * 70)
    fprint("SUMMARY — SPY IRON CONDOR INCOME")
    fprint("=" * 70)
    fprint(f"{'Name':<25} {'Trades':>6} {'WR':>6} {'Sharpe':>7} {'CAGR':>7} {'MaxDD':>7} {'PF':>5} {'Gates':>6}")
    fprint("-" * 75)
    for r in sorted(results, key=lambda x: x['sharpe'], reverse=True):
        fprint(f"{r['name']:<25} {r['n_trades']:>6} {r['win_rate']:>5.1f}% {r['sharpe']:>7.2f} "
               f"{r['cagr_pct']:>6.1f}% {r['maxdd_pct']:>6.1f}% {r['pf']:>5.2f} {r['gates_passed']:>4}/4")

    # Save
    save = [{k: v for k, v in r.items() if k != 'monthly_returns'} for r in results]
    with open(RESULTS_PATH, 'w') as f:
        json.dump(save, f, indent=2, default=str)
    fprint(f"\nDone — {datetime.now().strftime('%H:%M:%S')}")


if __name__ == '__main__':
    main()
