#!/usr/bin/env python3
"""
Momentum Cash-Secured Put (CSP) Writing on Sector ETFs v1
==========================================================

Sell puts on ETFs WITH strong momentum (wind at your back).
The validated momentum signal (perm p=0.000) identifies which ETFs will likely
continue rising — selling puts on those ETFs captures premium while having
low probability of assignment.

Key insight: Momentum predicts DIRECTION, selling puts in that direction
captures theta decay + directional edge. If assigned, you own an ETF with
strong momentum (likely to recover).

Unlike put credit SPREADS (which failed on ETFs due to thin premiums),
cash-secured puts on momentum ETFs:
- Collect full premium (not just spread premium)
- If assigned at a discount, own a trending ETF
- Premium is larger on individual sector ETFs vs index

Variants:
1. Top 2 momentum, sell 5-delta OTM put, 21d DTE
2. Top 2 momentum, sell 10-delta OTM put, 21d DTE
3. Top 3 momentum, sell 5-delta OTM put, 21d DTE
4. Top 2 momentum, sell 5-delta OTM put, 14d DTE (faster theta)
5. Top 2 momentum, IV rank >30 filter (sell when premium is decent)
6. Top 3 momentum, skip bear regime
7. Top 2 momentum, 10-delta, vol-scaled sizing
8. Aggressive: Top 2, ATM put, 21d DTE

Universe: 22 sector/factor ETFs (survivorship-free).
Walk-forward: 252d lookback for momentum, 21d rebalance, sliding.
Costs: Commission + realistic BS pricing with bid-ask spread.
Capital: $10,000 (need cash to secure puts) — report per $10K and scaled to $100K.
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
    print(*args, **kwargs)
    sys.stdout.flush()

RESULTS_DIR = Path('/home/jupiter/Lvl3Quant/research/findings')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = RESULTS_DIR / 'momentum_csp_etf_v1_results.json'

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

UNIVERSE = [
    'XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU', 'XLRE',
    'XLC', 'QQQ', 'IWM', 'MDY', 'EFA', 'EEM', 'GLD', 'TLT', 'HYG', 'IYR',
    'VNQ', 'DBC',
]


def bs_put_price(S, K, T, r, sigma):
    """Black-Scholes put price."""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_delta_put(S, K, T, r, sigma):
    """Black-Scholes put delta (negative for puts)."""
    if T <= 0 or sigma <= 0:
        return -1.0 if S < K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) - 1


def find_strike_by_delta(S, target_delta, T, r, sigma, n_steps=50):
    """Find strike price that gives target put delta."""
    # Put delta is negative, target_delta should be negative (e.g., -0.05 for 5-delta)
    best_K = S * 0.95  # default
    best_err = float('inf')

    for pct in np.linspace(0.80, 1.02, n_steps):
        K = S * pct
        delta = bs_delta_put(S, K, T, r, sigma)
        err = abs(delta - target_delta)
        if err < best_err:
            best_err = err
            best_K = K

    return best_K


def download_data():
    """Download ETF price data."""
    import yfinance as yf

    tickers = UNIVERSE + ['SPY']
    fprint(f"Downloading {len(tickers)} tickers...")

    data = yf.download(tickers, start='2008-01-01', auto_adjust=True, progress=False)
    close = data['Close'].dropna(how='all')

    for col in close.columns:
        if close[col].isna().mean() > 0.20:
            close = close.drop(columns=[col])
    close = close.ffill().dropna()

    fprint(f"  Loaded {len([c for c in close.columns if c in UNIVERSE])} ETFs + SPY ({len(close)} days)")
    return close


def compute_momentum(close_df, date, lookback=252, skip=21):
    """12-1 momentum score for all ETFs."""
    idx = close_df.index.get_loc(date)
    if idx < lookback:
        return None

    scores = {}
    for etf in [c for c in close_df.columns if c in UNIVERSE]:
        prices = close_df[etf].iloc[idx - lookback:idx + 1]
        if len(prices) < lookback:
            continue
        ret_12m = prices.iloc[-1] / prices.iloc[0] - 1
        ret_1m = prices.iloc[-1] / prices.iloc[-skip] - 1
        scores[etf] = ret_12m - ret_1m

    return scores


def compute_vol(close_df, etf, date, window=63):
    """Compute annualized vol."""
    idx = close_df.index.get_loc(date)
    if idx < window:
        return 0.20
    prices = close_df[etf].iloc[idx - window:idx + 1]
    lr = np.log(prices / prices.shift(1)).dropna()
    return float(lr.std() * np.sqrt(252))


def run_csp_backtest(close_df, top_k=2, delta_target=-0.05, dte_days=21,
                      iv_filter=None, regime_filter=False, vol_sizing=False,
                      atm=False, starting_capital=10000, name='variant'):
    """
    Run CSP backtest.

    Sell cash-secured puts on top momentum ETFs.
    Cash requirement = strike * 100 per contract.
    """
    spy = close_df['SPY']
    spy_sma200 = spy.rolling(200).mean()
    etf_cols = [c for c in close_df.columns if c in UNIVERSE]

    # Monthly rebalance dates
    dates = close_df.index[252:]
    rebal_dates = []
    last_rebal = None
    for d in dates:
        if last_rebal is None or (close_df.index.get_loc(d) - close_df.index.get_loc(last_rebal)) >= dte_days:
            rebal_dates.append(d)
            last_rebal = d

    capital = starting_capital
    capital_history = [(rebal_dates[0], capital)]
    trades = []
    monthly_returns = []

    # Vol history for IV rank
    vol_history = {}
    for etf in etf_cols:
        lr = np.log(close_df[etf] / close_df[etf].shift(1)).dropna()
        vol_history[etf] = lr.rolling(63).std() * np.sqrt(252)

    for i in range(len(rebal_dates) - 1):
        entry_date = rebal_dates[i]
        exit_date = rebal_dates[i + 1]
        entry_idx = close_df.index.get_loc(entry_date)
        exit_idx = close_df.index.get_loc(exit_date)

        # Regime filter
        if regime_filter and spy.iloc[entry_idx] < spy_sma200.iloc[entry_idx]:
            monthly_returns.append(0.0)
            capital_history.append((exit_date, capital))
            continue

        # Score ETFs
        scores = compute_momentum(close_df, entry_date)
        if not scores:
            monthly_returns.append(0.0)
            capital_history.append((exit_date, capital))
            continue

        # IV filter
        if iv_filter is not None:
            filtered = {}
            for etf, score in scores.items():
                vol = compute_vol(close_df, etf, entry_date)
                vhist = vol_history.get(etf)
                if vhist is not None:
                    vs = vhist.iloc[:entry_idx+1].dropna()
                    if len(vs) >= 252:
                        iv_rank = (vol - vs.iloc[-252:].min()) / (vs.iloc[-252:].max() - vs.iloc[-252:].min()) * 100
                        iv_rank = max(0, min(100, iv_rank))
                        if iv_rank >= iv_filter:
                            filtered[etf] = score
                    else:
                        filtered[etf] = score
                else:
                    filtered[etf] = score
            scores = filtered

        if len(scores) < top_k:
            monthly_returns.append(0.0)
            capital_history.append((exit_date, capital))
            continue

        # Select top K by momentum
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        period_pnl = 0
        for etf, mom in ranked:
            entry_spot = float(close_df[etf].iloc[entry_idx])
            exit_spot = float(close_df[etf].iloc[exit_idx])
            vol = compute_vol(close_df, etf, entry_date)

            T = dte_days / 252
            r = 0.05

            # Find strike
            if atm:
                strike = entry_spot
            else:
                strike = find_strike_by_delta(entry_spot, delta_target, T, r, vol)

            # Round strike to nearest $0.50
            strike = round(strike * 2) / 2

            # Cash required per contract
            cash_req = strike * 100

            # How many contracts can we sell?
            available_cash = capital / top_k
            n_contracts = int(available_cash / cash_req)

            if vol_sizing and vol > 0:
                target_vol = 0.20
                vol_adj = target_vol / vol
                n_contracts = max(1, int(n_contracts * vol_adj))

            if n_contracts < 1:
                continue

            # Premium received (sell at bid = theoretical - spread)
            put_price = bs_put_price(entry_spot, strike, T, r, vol)
            bid_ask_spread = 0.003  # 30bps bid-ask on ETF options
            premium_received = put_price * (1 - bid_ask_spread) * 100 * n_contracts

            # At expiry: was the put assigned?
            if exit_spot < strike:
                # Assigned: loss = (strike - exit_spot) * 100 * n - premium
                assignment_loss = (strike - exit_spot) * 100 * n_contracts
                pnl = premium_received - assignment_loss
            else:
                # Expired worthless: keep full premium
                pnl = premium_received

            # Commission: $0.65 per contract per leg
            # Sell to open = 1 leg, if assigned/expire = 1 more
            commission = 0.65 * 2 * n_contracts
            pnl -= commission

            period_pnl += pnl

            trades.append({
                'entry_date': str(entry_date.date()),
                'exit_date': str(exit_date.date()),
                'etf': etf,
                'momentum': round(mom, 4),
                'entry_price': round(entry_spot, 2),
                'exit_price': round(exit_spot, 2),
                'strike': round(strike, 2),
                'premium': round(premium_received, 2),
                'n_contracts': n_contracts,
                'assigned': exit_spot < strike,
                'pnl': round(pnl, 2),
            })

        capital += period_pnl
        capital = max(0, capital)

        ret = period_pnl / max(capital - period_pnl, 1) if capital > 0 else -1
        monthly_returns.append(ret)
        capital_history.append((exit_date, capital))

        if capital <= 0:
            for j in range(i + 2, len(rebal_dates)):
                monthly_returns.append(0)
                capital_history.append((rebal_dates[j], 0))
            break

    # Metrics
    r = np.array(monthly_returns)
    r_nz = r[r != 0] if np.any(r != 0) else r

    sharpe = np.mean(r_nz) / np.std(r_nz) * np.sqrt(12) if np.std(r_nz) > 0 else 0
    downside = r_nz[r_nz < 0]
    ds_vol = np.std(downside) * np.sqrt(12) if len(downside) > 0 else 1e-6
    sortino = np.mean(r_nz) * 12 / ds_vol if ds_vol > 0 else 0

    if len(capital_history) > 1:
        years = (capital_history[-1][0] - capital_history[0][0]).days / 365.25
        if years > 0 and capital_history[-1][1] > 0:
            cagr = (capital_history[-1][1] / starting_capital) ** (1/years) - 1
        else:
            cagr = -1
    else:
        years = 0
        cagr = 0

    peak = starting_capital
    maxdd = 0
    for _, val in capital_history:
        peak = max(peak, val)
        dd = (val - peak) / peak if peak > 0 else 0
        maxdd = min(maxdd, dd)

    tpnls = [t['pnl'] for t in trades]
    wr = sum(1 for p in tpnls if p > 0) / len(tpnls) * 100 if tpnls else 0
    gp = sum(p for p in tpnls if p > 0)
    gl = abs(sum(p for p in tpnls if p < 0))
    pf = gp / gl if gl > 0 else float('inf')
    calmar = abs(cagr / maxdd) if maxdd < 0 else 0

    # Assignment rate
    n_assigned = sum(1 for t in trades if t.get('assigned'))
    assign_rate = n_assigned / len(trades) * 100 if trades else 0

    metrics = {
        'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2),
        'cagr': round(cagr * 100, 1),
        'maxdd': round(maxdd * 100, 1),
        'wr': round(wr, 1),
        'pf': round(pf, 2),
        'calmar': round(calmar, 2),
        'final_capital': round(capital_history[-1][1], 2),
        'n_trades': len(trades),
        'assignment_rate': round(assign_rate, 1),
        'avg_premium': round(np.mean([t['premium'] for t in trades]), 2) if trades else 0,
        'years': round(years, 1),
    }

    return metrics, trades, monthly_returns, capital_history


def adversarial_gates(monthly_returns, trades, close_df, name):
    """Run 4-gate adversarial validation."""
    r = np.array(monthly_returns)
    r_nz = r[r != 0] if np.any(r != 0) else r
    gates = {}

    fprint(f"  Adversarial gates for {name}:")

    # 1. Permutation test
    actual_sharpe = np.mean(r_nz) / np.std(r_nz) * np.sqrt(12) if np.std(r_nz) > 0 else 0

    n_perm = 1000
    perm_sharpes = []
    for _ in range(n_perm):
        perm = np.random.permutation(r_nz)
        ps = np.mean(perm) / np.std(perm) * np.sqrt(12) if np.std(perm) > 0 else 0
        perm_sharpes.append(ps)
    p_value = np.mean([ps >= actual_sharpe for ps in perm_sharpes])
    gates['permutation'] = {
        'actual_sharpe': round(float(actual_sharpe), 3),
        'p_value': round(float(p_value), 3),
        'pass': p_value < 0.05,
    }
    fprint(f"    Perm: Sharpe={actual_sharpe:.2f}, p={p_value:.3f} {'PASS' if p_value < 0.05 else 'FAIL'}")

    # 2. R1 Regime test
    spy = close_df['SPY']
    spy_monthly = spy.resample('ME').last().pct_change().dropna()

    if len(r_nz) > 6:
        bull_mask = np.zeros(len(r_nz), dtype=bool)
        bear_mask = np.zeros(len(r_nz), dtype=bool)

        for i in range(len(r_nz)):
            offset = len(r_nz) - i
            if offset <= len(spy_monthly):
                spy_ret = float(spy_monthly.iloc[-offset])
                if spy_ret >= 0:
                    bull_mask[i] = True
                else:
                    bear_mask[i] = True
            else:
                bull_mask[i] = True

        if bull_mask.sum() >= 3 and bear_mask.sum() >= 3:
            bull_r = r_nz[bull_mask]
            bear_r = r_nz[bear_mask]
            bs = np.mean(bull_r) / np.std(bull_r) * np.sqrt(12) if np.std(bull_r) > 0 else 0
            brs = np.mean(bear_r) / np.std(bear_r) * np.sqrt(12) if np.std(bear_r) > 0 else 0
            ms = max(abs(bs), abs(brs))
            gap = abs(bs - brs) / ms if ms > 0 else 0
            gates['regime_r1'] = {
                'bull_sharpe': round(float(bs), 2),
                'bear_sharpe': round(float(brs), 2),
                'gap': round(float(gap), 3),
                'pass': gap < 0.50,
            }
            fprint(f"    R1: bull={bs:.2f}, bear={brs:.2f}, gap={gap:.3f} {'PASS' if gap < 0.50 else 'FAIL'}")
        else:
            gates['regime_r1'] = {'pass': None, 'note': 'Insufficient data'}
    else:
        gates['regime_r1'] = {'pass': None, 'note': 'Too few months'}

    # 3. Sub-period
    mid = len(r_nz) // 2
    if mid >= 3:
        h1 = r_nz[:mid]
        h2 = r_nz[mid:]
        h1s = np.mean(h1) / np.std(h1) * np.sqrt(12) if np.std(h1) > 0 else 0
        h2s = np.mean(h2) / np.std(h2) * np.sqrt(12) if np.std(h2) > 0 else 0
        sp = h1s > 0 and h2s > 0
        gates['sub_period'] = {'h1_sharpe': round(float(h1s), 2), 'h2_sharpe': round(float(h2s), 2), 'pass': sp}
        fprint(f"    Sub: H1={h1s:.2f}, H2={h2s:.2f} {'PASS' if sp else 'FAIL'}")
    else:
        gates['sub_period'] = {'pass': None}

    # 4. Outlier
    if len(r_nz) >= 10:
        nr = max(1, int(len(r_nz) * 0.05))
        si = np.argsort(r_nz)[::-1]
        trimmed = np.delete(r_nz, si[:nr])
        ts = np.mean(trimmed) / np.std(trimmed) * np.sqrt(12) if np.std(trimmed) > 0 else 0
        op = ts > 0
        gates['outlier'] = {'trimmed_sharpe': round(float(ts), 2), 'n_removed': nr, 'pass': op}
        fprint(f"    Outlier: trimmed={ts:.2f} {'PASS' if op else 'FAIL'}")
    else:
        gates['outlier'] = {'pass': None}

    n_pass = sum(1 for k in ['permutation', 'regime_r1', 'sub_period', 'outlier']
                 if gates.get(k, {}).get('pass') is True)
    n_total = sum(1 for k in ['permutation', 'regime_r1', 'sub_period', 'outlier']
                  if gates.get(k, {}).get('pass') is not None)
    fprint(f"    GATES: {n_pass}/{n_total}")

    return gates


def main():
    fprint("=" * 70)
    fprint("MOMENTUM CASH-SECURED PUT WRITING ON ETFs v1")
    fprint("=" * 70)
    fprint(f"Universe: {len(UNIVERSE)} ETFs | Starting capital: $10,000")
    fprint(f"Strategy: Sell puts on top momentum ETFs, keep premium if OTM at expiry")
    fprint()

    close = download_data()

    variants = [
        {'name': 'A_Top2_5d_21d', 'top_k': 2, 'delta_target': -0.05, 'dte_days': 21,
         'desc': 'Top 2, 5-delta put, 21d'},
        {'name': 'B_Top2_10d_21d', 'top_k': 2, 'delta_target': -0.10, 'dte_days': 21,
         'desc': 'Top 2, 10-delta put, 21d'},
        {'name': 'C_Top3_5d_21d', 'top_k': 3, 'delta_target': -0.05, 'dte_days': 21,
         'desc': 'Top 3, 5-delta put, 21d'},
        {'name': 'D_Top2_5d_14d', 'top_k': 2, 'delta_target': -0.05, 'dte_days': 14,
         'desc': 'Top 2, 5-delta put, 14d (faster theta)'},
        {'name': 'E_Top2_IVfilt', 'top_k': 2, 'delta_target': -0.05, 'dte_days': 21,
         'iv_filter': 30, 'desc': 'Top 2, 5-delta, IV rank>30'},
        {'name': 'F_Top3_regime', 'top_k': 3, 'delta_target': -0.05, 'dte_days': 21,
         'regime_filter': True, 'desc': 'Top 3, skip bear'},
        {'name': 'G_Top2_volsize', 'top_k': 2, 'delta_target': -0.10, 'dte_days': 21,
         'vol_sizing': True, 'desc': 'Top 2, 10-delta, vol-sized'},
        {'name': 'H_Top2_ATM', 'top_k': 2, 'delta_target': -0.50, 'dte_days': 21,
         'atm': True, 'desc': 'Top 2, ATM put (aggressive)'},
    ]

    results = {}
    gate_results = {}

    for v in variants:
        vname = v['name']
        fprint(f"\n--- {vname} ({v['desc']}) ---")

        kwargs = {
            'top_k': v['top_k'],
            'delta_target': v['delta_target'],
            'dte_days': v['dte_days'],
            'iv_filter': v.get('iv_filter'),
            'regime_filter': v.get('regime_filter', False),
            'vol_sizing': v.get('vol_sizing', False),
            'atm': v.get('atm', False),
            'name': vname,
        }

        metrics, trades, monthly_returns, cap_hist = run_csp_backtest(close, **kwargs)

        fprint(f"  Sharpe: {metrics['sharpe']}, Sortino: {metrics['sortino']}, "
               f"CAGR: {metrics['cagr']}%, MaxDD: {metrics['maxdd']}%, "
               f"WR: {metrics['wr']}%, PF: {metrics['pf']}, "
               f"Final: ${metrics['final_capital']}, Trades: {metrics['n_trades']}, "
               f"Assignment: {metrics['assignment_rate']}%")

        gates = adversarial_gates(monthly_returns, trades, close, vname)

        results[vname] = {
            'desc': v['desc'],
            'metrics': metrics,
        }
        gate_results[vname] = gates

    # Summary
    fprint(f"\n{'='*70}")
    fprint("SUMMARY")
    fprint(f"{'='*70}")

    valid = [(n, r) for n, r in results.items()
             if r['metrics']['sharpe'] > 0 and r['metrics']['n_trades'] > 10]

    if valid:
        def sort_key(item):
            n, r = item
            g = gate_results[n]
            np_ = sum(1 for k in ['permutation', 'regime_r1', 'sub_period', 'outlier']
                      if g.get(k, {}).get('pass') is True)
            return (np_, r['metrics']['sharpe'])

        valid.sort(key=sort_key, reverse=True)

        for n, r in valid:
            m = r['metrics']
            g = gate_results[n]
            np_ = sum(1 for k in ['permutation', 'regime_r1', 'sub_period', 'outlier']
                      if g.get(k, {}).get('pass') is True)
            nt = sum(1 for k in ['permutation', 'regime_r1', 'sub_period', 'outlier']
                     if g.get(k, {}).get('pass') is not None)
            fprint(f"  {n}: Sharpe {m['sharpe']}, CAGR {m['cagr']}%, MaxDD {m['maxdd']}%, "
                   f"WR {m['wr']}%, PF {m['pf']}, Assign {m['assignment_rate']}% | "
                   f"Gates {np_}/{nt}")

        winner_name = valid[0][0]
        winner = results[winner_name]
        wm = winner['metrics']
        fprint(f"\n  WINNER: {winner_name}")
        fprint(f"    {winner['desc']}")
        fprint(f"    Sharpe {wm['sharpe']}, Sortino {wm['sortino']}, CAGR {wm['cagr']}%, "
               f"MaxDD {wm['maxdd']}%, WR {wm['wr']}%, PF {wm['pf']}")
        fprint(f"    $10K → ${wm['final_capital']} over {wm['years']}y, {wm['n_trades']} trades")
        fprint(f"    Assignment rate: {wm['assignment_rate']}%, Avg premium: ${wm['avg_premium']}")

        # Scale to $100K
        scale = 100000 / 10000
        fprint(f"    Scaled to $100K: ~${wm['final_capital'] * scale:,.0f}")
    else:
        fprint("  NO VALID VARIANTS")
        winner_name = None

    # Save results
    output = {
        'strategy': 'Momentum CSP on ETFs v1',
        'timestamp': datetime.now().isoformat(),
        'universe': UNIVERSE,
        'starting_capital': 10000,
        'results': results,
        'gates': gate_results,
        'winner': winner_name,
    }

    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    fprint(f"\nResults saved to {RESULTS_PATH}")

    # MLflow
    if MLFLOW_OK and winner_name:
        try:
            exp_name = 'momentum_csp_etf_v1'
            try:
                mlflow.create_experiment(exp_name)
            except:
                pass
            mlflow.set_experiment(exp_name)

            with mlflow.start_run(run_name=f'v1_{winner_name}'):
                wm = results[winner_name]['metrics']
                wg = gate_results[winner_name]
                mlflow.log_metrics({
                    'sharpe': wm.get('sharpe', 0),
                    'sortino': wm.get('sortino', 0),
                    'cagr': wm.get('cagr', 0),
                    'maxdd': wm.get('maxdd', 0),
                    'wr': wm.get('wr', 0),
                    'pf': wm.get('pf', 0),
                    'assignment_rate': wm.get('assignment_rate', 0),
                    'perm_p': wg.get('permutation', {}).get('p_value', -1),
                    'r1_gap': wg.get('regime_r1', {}).get('gap', -1),
                    'n_pass': sum(1 for k in ['permutation', 'regime_r1', 'sub_period', 'outlier']
                                  if wg.get(k, {}).get('pass') is True),
                })
                mlflow.log_params({
                    'winner': winner_name,
                    'universe_size': len(UNIVERSE),
                    'starting_capital': 10000,
                    'n_variants': len(variants),
                })
                fprint(f"MLflow logged (exp: {exp_name})")
        except Exception as e:
            fprint(f"MLflow error: {e}")

    fprint(f"\n{'='*70}")
    fprint("DONE")
    fprint(f"{'='*70}")

    return output


if __name__ == '__main__':
    main()
