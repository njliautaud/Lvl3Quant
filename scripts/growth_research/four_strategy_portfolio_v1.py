#!/usr/bin/env python3
"""
Four-Strategy Ultimate Portfolio v1 — Best-of-Each Combined
=============================================================

Combines our 4 VALIDATED strategy families into one portfolio:
1. Sector ETF Momentum (LightGBM) — Growth engine, Sharpe ~0.84
2. Alt Trend Following (Bonds/Commodities) — Decorrelator, Corr(SPY) ~0.12
3. SPY Iron Condor Income — Non-directional income, Sharpe ~3.5
4. VIX Mean-Reversion — Spike-timing income, R1 gap 0.035 (near-perfect)

From prior research:
- Eq↔Alt correlation: 0.23 (good)
- Eq↔IC correlation: 0.01 (perfect)
- Alt↔IC correlation: -0.10 (negative!)
- VIX MR is anti-correlated with equity in bear markets

Portfolio allocation schemes tested:
A: Equal weight (25% each)
B: Risk Parity (inverse-vol weighted)
C: Growth tilt (40% equity, 20% each other)
D: Income tilt (20% equity, 20% alt, 30% IC, 30% VIX)
E: Regime adaptive (increase equity in bull, increase VIX/alt in bear)
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
RESULTS_PATH = RESULTS_DIR / 'four_strategy_portfolio_v1_results.json'

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


# ====== STRATEGY 1: SECTOR ETF MOMENTUM (simplified LightGBM) ======

def etf_momentum_returns(close_df, spy_close):
    """Monthly returns from top-3 ETF momentum with defensive shift."""
    spy_sma200 = spy_close.rolling(200).mean()
    month_ends = close_df.index.to_series().resample('M').last().dropna()

    monthly_rets = []
    for i in range(13, len(month_ends) - 1):
        date = month_ends.iloc[i]
        next_date = month_ends.iloc[i + 1]

        # Simple 12-1 momentum ranking
        scores = {}
        for col in close_df.columns:
            px = close_df[col].loc[:date].dropna()
            if len(px) > 252:
                ret_12m = px.iloc[-1] / px.iloc[-252] - 1
                ret_1m = px.iloc[-1] / px.iloc[-21] - 1
                scores[col] = ret_12m - ret_1m

        if not scores:
            monthly_rets.append({'date': date, 'ret': 0})
            continue

        # Defensive shift: reduce exposure in bear
        spy_val = float(spy_close.loc[date]) if date in spy_close.index else 0
        sma_val = float(spy_sma200.loc[date]) if date in spy_sma200.index else spy_val
        in_bear = spy_val < sma_val

        top3 = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:3]

        # Forward returns (equal weight top 3)
        fwd_rets = []
        for t in top3:
            px_now = close_df[t].loc[:date].dropna()
            px_next = close_df[t].loc[:next_date].dropna()
            if len(px_now) > 0 and len(px_next) > 0:
                fwd_rets.append(px_next.iloc[-1] / px_now.iloc[-1] - 1)

        portfolio_ret = np.mean(fwd_rets) if fwd_rets else 0

        # Defensive shift: 50% cash in bear
        if in_bear:
            portfolio_ret *= 0.5

        monthly_rets.append({'date': date, 'ret': portfolio_ret})

    return pd.DataFrame(monthly_rets).set_index('date')['ret']


# ====== STRATEGY 2: ALT TREND FOLLOWING ======

def alt_trend_returns(alt_close):
    """Monthly returns from SMA200 trend following on non-equity assets."""
    month_ends = alt_close.index.to_series().resample('M').last().dropna()

    monthly_rets = []
    for i in range(12, len(month_ends) - 1):
        date = month_ends.iloc[i]
        next_date = month_ends.iloc[i + 1]

        holding = []
        for col in alt_close.columns:
            px = alt_close[col].loc[:date].dropna()
            if len(px) > 200:
                sma200 = px.iloc[-200:].mean()
                if px.iloc[-1] > sma200:  # Above SMA200 = hold
                    px_next = alt_close[col].loc[:next_date].dropna()
                    if len(px_next) > 0:
                        holding.append(px_next.iloc[-1] / px.iloc[-1] - 1)

        ret = np.mean(holding) if holding else 0
        monthly_rets.append({'date': date, 'ret': ret})

    return pd.DataFrame(monthly_rets).set_index('date')['ret']


# ====== STRATEGY 3: IRON CONDOR INCOME ======

def iron_condor_monthly_returns(spy_close, vix_close, capital=10000):
    """Monthly returns from SPY iron condor strategy (simplified)."""
    vix_rank = vix_close.rolling(252).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10) * 100
    )

    month_ends = spy_close.index.to_series().resample('M').last().dropna()
    monthly_rets = []

    for i in range(12, len(month_ends) - 1):
        date = month_ends.iloc[i]
        next_date = month_ends.iloc[i + 1]

        S = float(spy_close.loc[date])
        vix = float(vix_close.loc[date]) if date in vix_close.index else 15
        sigma = vix / 100
        T = 30 / 365

        # 16-delta iron condor, $5 wide
        from scipy.stats import norm as norm_dist
        d1_call = norm_dist.ppf(0.84)  # 16-delta OTM call
        d1_put = norm_dist.ppf(0.16)   # 16-delta OTM put
        call_short = S * np.exp(d1_call * sigma * np.sqrt(T))
        put_short = S * np.exp(d1_put * sigma * np.sqrt(T))

        # Price IC
        sc = bs_price(S, call_short, T, sigma, opt='call')
        lc = bs_price(S, call_short + 5, T, sigma, opt='call')
        sp = bs_price(S, put_short, T, sigma, opt='put')
        lp = bs_price(S, put_short - 5, T, sigma, opt='put')
        credit = (sc - lc) + (sp - lp)

        # After 1 month, what happened?
        S_next = float(spy_close.loc[next_date]) if next_date in spy_close.index else S
        vix_next = float(vix_close.loc[next_date]) if next_date in vix_close.index else vix

        # Intrinsic at expiry (simplified)
        call_loss = max(0, S_next - call_short) - max(0, S_next - (call_short + 5))
        put_loss = max(0, put_short - S_next) - max(0, (put_short - 5) - S_next)
        settlement = call_loss + put_loss

        pnl = (credit - settlement) * 100 - 5  # Per contract
        ret = pnl / capital

        monthly_rets.append({'date': date, 'ret': ret})

    return pd.DataFrame(monthly_rets).set_index('date')['ret']


# ====== STRATEGY 4: VIX MEAN REVERSION ======

def vix_meanrev_monthly_returns(vix_close, capital=10000):
    """Monthly returns from VIX call spread mean-reversion."""
    month_ends = vix_close.index.to_series().resample('M').last().dropna()
    monthly_rets = []

    for i in range(12, len(month_ends) - 1):
        date = month_ends.iloc[i]
        next_date = month_ends.iloc[i + 1]
        vix = float(vix_close.loc[date])

        if vix < 25:
            # No trade when VIX is calm
            monthly_rets.append({'date': date, 'ret': 0})
            continue

        # Sell call spread: short VIX+2, long VIX+7
        T = 30 / 365
        vix_vol = 0.80
        short_strike = vix + 2
        long_strike = short_strike + 5

        credit = bs_price(vix, short_strike, T, vix_vol, opt='call') - \
                 bs_price(vix, long_strike, T, vix_vol, opt='call')

        if credit <= 0:
            monthly_rets.append({'date': date, 'ret': 0})
            continue

        # At expiry
        vix_next = float(vix_close.loc[next_date]) if next_date in vix_close.index else vix
        short_val = max(0, vix_next - short_strike)
        long_val = max(0, vix_next - long_strike)
        settlement = short_val - long_val

        pnl = (credit - settlement) * 100 - 3
        ret = pnl / capital

        monthly_rets.append({'date': date, 'ret': ret})

    return pd.DataFrame(monthly_rets).set_index('date')['ret']


# ====== PORTFOLIO COMBINATION ======

def combine_portfolio(strat_rets, weights, name='portfolio'):
    """Combine strategy returns with given weights."""
    # Align all on common dates
    aligned = pd.DataFrame(strat_rets)
    aligned = aligned.dropna(how='all').fillna(0)

    portfolio_ret = sum(aligned[col] * w for col, w in zip(aligned.columns, weights))

    return portfolio_ret


def permutation_test(returns, n_perms=1000):
    if len(returns) < 5:
        return 1.0
    real = np.mean(returns) / (np.std(returns) + 1e-10)
    count = sum(1 for _ in range(n_perms)
                if np.mean(returns * np.random.choice([-1, 1], len(returns))) /
                (np.std(returns) + 1e-10) >= real)
    return count / n_perms


def compute_metrics(monthly_ret, name, spy_sma200_monthly=None, spy_monthly=None, capital=100000):
    """Compute full metrics from monthly return series."""
    n_months = len(monthly_ret)
    if n_months < 6:
        return None

    n_years = n_months / 12

    sharpe = (monthly_ret.mean() * 12) / (monthly_ret.std() * np.sqrt(12) + 1e-10)
    cagr = (1 + monthly_ret.sum()) ** (1 / max(n_years, 0.5)) - 1

    down = monthly_ret[monthly_ret < 0]
    sortino = (monthly_ret.mean() * 12) / (down.std() * np.sqrt(12) + 1e-10) if len(down) > 0 else 999

    cum = (1 + monthly_ret).cumprod()
    peak = cum.expanding().max()
    dd = (cum - peak) / peak
    maxdd = dd.min()
    calmar = cagr / abs(maxdd) if maxdd < 0 else 999

    win_months = (monthly_ret > 0).sum()
    wr = win_months / n_months * 100

    gross_win = monthly_ret[monthly_ret > 0].sum()
    gross_loss = abs(monthly_ret[monthly_ret < 0].sum())
    pf = gross_win / (gross_loss + 1e-10)

    # Regime analysis using SPY SMA200
    bull_rets = []
    bear_rets = []
    if spy_sma200_monthly is not None and spy_monthly is not None:
        for date in monthly_ret.index:
            if date in spy_monthly.index and date in spy_sma200_monthly.index:
                if spy_monthly.loc[date] > spy_sma200_monthly.loc[date]:
                    bull_rets.append(monthly_ret.loc[date])
                else:
                    bear_rets.append(monthly_ret.loc[date])

    bull_sharpe = np.mean(bull_rets) * 12 / (np.std(bull_rets) * np.sqrt(12) + 1e-10) if len(bull_rets) > 3 else 0
    bear_sharpe = np.mean(bear_rets) * 12 / (np.std(bear_rets) * np.sqrt(12) + 1e-10) if len(bear_rets) > 3 else 0

    r1_gap = abs(bull_sharpe - bear_sharpe) / max(abs(bull_sharpe), abs(bear_sharpe), 0.01)

    result = {
        'name': name, 'n_months': n_months,
        'sharpe': round(sharpe, 2), 'sortino': round(sortino, 2),
        'cagr_pct': round(cagr * 100, 1), 'maxdd_pct': round(maxdd * 100, 1),
        'calmar': round(calmar, 2), 'pf': round(pf, 2),
        'win_rate': round(wr, 1),
        'bull_sharpe': round(bull_sharpe, 2), 'bear_sharpe': round(bear_sharpe, 2),
        'r1_gap': round(r1_gap, 3), 'r1_pass': r1_gap <= 0.50,
        'monthly_returns': monthly_ret.tolist(),
    }

    fprint(f"\n  {name}: Sharpe {sharpe:.2f}, Sortino {sortino:.2f}, "
           f"CAGR {cagr*100:.1f}%, MaxDD {maxdd*100:.1f}%, Calmar {calmar:.2f}, "
           f"WR {wr:.0f}%, PF {pf:.2f}")
    fprint(f"    Bull Sharpe: {bull_sharpe:.2f} | Bear Sharpe: {bear_sharpe:.2f} | "
           f"R1 gap: {r1_gap:.3f} {'PASS' if r1_gap <= 0.50 else 'FAIL'}")

    return result


def main():
    import yfinance as yf

    fprint(f"Four-Strategy Ultimate Portfolio v1 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 70)

    # Download equity ETFs
    equity_tickers = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB', 'XLRE',
                      'XLC', 'SMH', 'XBI', 'XHB', 'XRT', 'SPY', 'QQQ', 'IWM', 'GLD', 'TLT', 'EEM', 'EFA']
    alt_tickers = ['TLT', 'IEF', 'GLD', 'DBC', 'USO', 'HYG']  # Non-equity

    all_tickers = list(set(equity_tickers + alt_tickers + ['SPY', '^VIX']))

    fprint("Downloading data...")
    raw = yf.download(all_tickers, start='2008-01-01', end='2026-07-25', progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw['Close']
    else:
        close = raw

    vix_col = '^VIX' if '^VIX' in close.columns else 'VIX'
    vix = close[vix_col].dropna()
    spy = close['SPY'].dropna()

    equity_close = close[[c for c in equity_tickers if c in close.columns]].dropna(how='all')
    alt_close = close[[c for c in alt_tickers if c in close.columns]].dropna(how='all')

    # Align
    common = equity_close.index.intersection(vix.index).intersection(spy.index).intersection(alt_close.index)
    equity_close = equity_close.loc[common]
    alt_close = alt_close.loc[common]
    spy = spy.loc[common]
    vix = vix.loc[common]

    fprint(f"Data: {len(common)} days, {common[0].strftime('%Y-%m-%d')} to {common[-1].strftime('%Y-%m-%d')}")

    # Compute individual strategy returns
    fprint("\nComputing individual strategy returns...")

    fprint("  1. Sector ETF Momentum...")
    eq_ret = etf_momentum_returns(equity_close, spy)

    fprint("  2. Alt Trend Following...")
    alt_ret = alt_trend_returns(alt_close)

    fprint("  3. Iron Condor Income...")
    ic_ret = iron_condor_monthly_returns(spy, vix)

    fprint("  4. VIX Mean Reversion...")
    vix_ret = vix_meanrev_monthly_returns(vix)

    # Align all
    strats = pd.DataFrame({
        'equity_mom': eq_ret,
        'alt_trend': alt_ret,
        'iron_condor': ic_ret,
        'vix_meanrev': vix_ret,
    }).dropna(how='all').fillna(0)

    fprint(f"\nAligned: {len(strats)} months of data")

    # Correlation matrix
    corr = strats.corr()
    fprint("\nCorrelation Matrix:")
    fprint(corr.round(3).to_string())

    # SPY regime data for R1
    spy_sma200 = spy.rolling(200).mean()
    spy_monthly = spy.resample('M').last()
    spy_sma200_monthly = spy_sma200.resample('M').last()

    # === Portfolio Variants ===
    variants = [
        ('A_EqualWeight', [0.25, 0.25, 0.25, 0.25]),
        ('B_RiskParity', None),  # Computed from inverse vol
        ('C_GrowthTilt', [0.40, 0.20, 0.20, 0.20]),
        ('D_IncomeTilt', [0.20, 0.20, 0.30, 0.30]),
        ('E_RegimeAdaptive', None),  # Dynamic
    ]

    results = []

    if MLFLOW_OK:
        exp_name = 'four_strategy_portfolio_v1'
        try:
            if not mlflow.get_experiment_by_name(exp_name):
                mlflow.create_experiment(exp_name)
        except:
            pass
        mlflow.set_experiment(exp_name)

    for vname, weights in variants:
        fprint(f"\n--- {vname} ---")

        if vname == 'B_RiskParity':
            # Inverse-vol risk parity
            vols = strats.rolling(12).std().iloc[-1]
            inv_vol = 1.0 / (vols + 1e-10)
            weights = (inv_vol / inv_vol.sum()).values
            fprint(f"  Risk Parity weights: {dict(zip(strats.columns, np.round(weights, 3)))}")

            port_ret = sum(strats[col] * w for col, w in zip(strats.columns, weights))

        elif vname == 'E_RegimeAdaptive':
            # Dynamic weights based on regime
            port_ret = pd.Series(0.0, index=strats.index)
            for date in strats.index:
                if date in spy_monthly.index and date in spy_sma200_monthly.index:
                    in_bear = spy_monthly.loc[date] < spy_sma200_monthly.loc[date]
                else:
                    in_bear = False

                if in_bear:
                    # Bear: reduce equity, increase defensive
                    w = [0.10, 0.30, 0.25, 0.35]  # Less equity, more alt/VIX
                else:
                    # Bull: tilt to growth
                    w = [0.35, 0.20, 0.25, 0.20]  # More equity

                port_ret.loc[date] = sum(strats[col].loc[date] * ww
                                         for col, ww in zip(strats.columns, w))
            weights = 'dynamic'
        else:
            port_ret = sum(strats[col] * w for col, w in zip(strats.columns, weights))

        r = compute_metrics(port_ret, vname, spy_sma200_monthly, spy_monthly)
        if r:
            if isinstance(weights, (list, np.ndarray)):
                r['weights'] = {col: round(w, 3) for col, w in zip(strats.columns, weights)}
            else:
                r['weights'] = 'dynamic'
            if MLFLOW_OK:
                with mlflow.start_run(run_name=vname):
                    w_str = json.dumps(r['weights']) if isinstance(r['weights'], dict) else 'dynamic'
                    mlflow.log_params({'allocation': vname, 'weights': w_str})
                    mlflow.log_metrics({k: v for k, v in r.items()
                                       if isinstance(v, (int, float)) and not np.isnan(v) and not np.isinf(v)})
            results.append(r)

    # Also compute individual strategy metrics for comparison
    fprint("\n--- Individual Strategy Metrics ---")
    for col in strats.columns:
        compute_metrics(strats[col], f"Solo_{col}", spy_sma200_monthly, spy_monthly)

    if not results:
        fprint("No results!")
        return

    # === Adversarial ===
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
               f"G3={'PASS' if r['g3_pass'] else 'FAIL'}(subs={r['sub_sharpes']}), "
               f"G4={'PASS' if r['g4_pass'] else 'FAIL'} → {r['gates_passed']}/4")

    # Summary
    fprint("\n" + "=" * 70)
    fprint("SUMMARY — Four-Strategy Ultimate Portfolio v1")
    fprint("=" * 70)
    fprint(f"{'Name':<25} {'Sharpe':>7} {'Sortino':>8} {'CAGR':>7} {'MaxDD':>7} {'Calmar':>7} {'WR':>5} {'PF':>5} {'R1gap':>6} {'Gates':>6}")
    fprint("-" * 100)
    for r in sorted(results, key=lambda x: x['sharpe'], reverse=True):
        fprint(f"{r['name']:<25} {r['sharpe']:>7.2f} {r['sortino']:>8.2f} "
               f"{r['cagr_pct']:>6.1f}% {r['maxdd_pct']:>6.1f}% {r['calmar']:>7.2f} "
               f"{r['win_rate']:>4.0f}% {r['pf']:>5.2f} {r['r1_gap']:>6.3f} {r['gates_passed']:>4}/4")

    save = [{k: v for k, v in r.items() if k != 'monthly_returns'} for r in results]
    with open(RESULTS_PATH, 'w') as f:
        json.dump(save, f, indent=2, default=str)
    fprint(f"\nResults saved. Done — {datetime.now().strftime('%H:%M:%S')}")


if __name__ == '__main__':
    main()
