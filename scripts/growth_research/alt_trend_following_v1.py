#!/usr/bin/env python3
"""
Alternative Asset Trend Following v1 — Non-Equity Decorrelation
================================================================

Problem: All growth strategies (sector momentum, cross-asset trend) are
0.55-0.92 correlated with SPY. Need a truly decorrelated return stream.

Solution: Trend follow ONLY non-equity assets: bonds, commodities, gold,
real assets. These are the assets that provide "crisis alpha" — positive
returns when equities crash.

This is how managed futures (CTA) funds generate uncorrelated returns —
they're not just long equities in another way.

Universe (NO equities, NO equity-like assets):
- Bonds: TLT, IEF, TIP, SHY
- Commodities: GLD, SLV, DBC, USO
- Real Estate: VNQ (borderline, but lower beta)
- Currency proxy: UUP (dollar index ETF)

Variants:
A. Simple trend (SMA200), equal weight
B. Dual momentum, equal weight
C. Risk parity weighted
D. Vol-targeted (10% target)
E. Multi-timeframe (SMA50+200)
F. Carry + trend (use yield spread for bonds, contango for commodities)
G. With equity hedge (short SPY when all trends are down)

Walk-forward: monthly, 2008-2026.
Cost: 20bps.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def fprint(*args, **kwargs):
    print(*args, **kwargs, flush=True)

RESULTS_DIR = Path('/home/jupiter/Lvl3Quant/research/findings')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = RESULTS_DIR / 'alt_trend_following_v1_results.json'

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

# NON-EQUITY UNIVERSE ONLY
UNIVERSE = {
    # Bonds
    'TLT': 'bond',        # Long-term treasury (20y+)
    'IEF': 'bond',        # 7-10yr treasury
    'TIP': 'bond',        # TIPS (inflation-linked)
    'SHY': 'bond',        # Short-term treasury (1-3y)
    'LQD': 'bond',        # IG corporates

    # Commodities
    'GLD': 'commodity',   # Gold
    'SLV': 'commodity',   # Silver
    'DBC': 'commodity',   # Broad commodities
    'USO': 'commodity',   # Crude oil

    # Real assets
    'VNQ': 'real_asset',  # REITs

    # Currency
    'UUP': 'currency',    # US Dollar Index ETF
}


def run_variant(close_data, spy_close, name, mode='sma200', weight_mode='equal',
                vol_target_ann=None, cost_bps=10):
    """Run trend variant on non-equity assets only."""
    fprint(f"\n--- {name} ---")

    monthly_close = close_data.resample('ME').last().dropna(how='all')
    spy_monthly = spy_close.resample('ME').last() if spy_close is not None else None

    monthly_rets = monthly_close.pct_change()

    # Daily signals sampled monthly
    sma200 = close_data.rolling(200).mean().resample('ME').last()
    sma50 = close_data.rolling(50).mean().resample('ME').last()

    # Momentum
    mom_12 = monthly_close.pct_change(12)
    mom_1 = monthly_close.pct_change(1)
    mom_12_1 = mom_12 - mom_1

    # Rolling vol
    daily_rets = close_data.pct_change()
    rolling_vol = (daily_rets.rolling(63).std() * np.sqrt(252)).resample('ME').last()

    # SPY signals (for regime detection and correlation)
    if spy_close is not None:
        spy_sma200 = spy_close.rolling(200).mean().resample('ME').last()
    else:
        spy_sma200 = None

    start_idx = 13  # Need 12 months history

    all_returns = []
    all_dates = []
    all_regimes = []
    all_n_long = []

    for i in range(start_idx, len(monthly_close) - 1):
        date = monthly_close.index[i]

        # Determine longs
        longs = {}
        for asset in monthly_close.columns:
            try:
                price = float(monthly_close.iloc[i][asset])
                if pd.isna(price):
                    continue

                s200 = sma200.iloc[i][asset] if asset in sma200.columns else np.nan
                s50 = sma50.iloc[i][asset] if asset in sma50.columns else np.nan
                m = mom_12_1.iloc[i][asset] if asset in mom_12_1.columns else np.nan

                if pd.isna(s200):
                    continue

                if mode == 'sma200':
                    go_long = price > s200
                elif mode == 'dual':
                    go_long = (price > s200) and (not pd.isna(m)) and (m > 0)
                elif mode == 'multi_tf':
                    go_long = (price > s200) and (not pd.isna(s50)) and (price > s50)
                else:
                    go_long = price > s200

                if go_long:
                    v = rolling_vol.iloc[i][asset] if asset in rolling_vol.columns else 0.12
                    if pd.isna(v) or v < 0.01:
                        v = 0.12
                    longs[asset] = float(v)
            except:
                continue

        n_long = len(longs)

        # If nothing trending up, 100% cash (return = 0)
        if n_long == 0:
            all_returns.append(0.0)
            all_dates.append(date)
            regime = 'bear'
            if spy_monthly is not None and spy_sma200 is not None:
                sp = spy_monthly.iloc[i] if i < len(spy_monthly) else np.nan
                ss = spy_sma200.iloc[i] if i < len(spy_sma200) else np.nan
                regime = 'bear' if (not pd.isna(sp) and not pd.isna(ss) and sp < ss) else 'bull'
            all_regimes.append(regime)
            all_n_long.append(0)
            continue

        # Weights
        if weight_mode == 'equal':
            weights = {a: 1.0 / n_long for a in longs}
        elif weight_mode == 'risk_parity':
            inv_vols = {a: 1.0 / v for a, v in longs.items()}
            total = sum(inv_vols.values())
            weights = {a: iv / total for a, iv in inv_vols.items()}
        elif weight_mode == 'vol_target':
            inv_vols = {a: 1.0 / v for a, v in longs.items()}
            total = sum(inv_vols.values())
            raw = {a: iv / total for a, iv in inv_vols.items()}
            # Portfolio vol estimate
            port_vol = sum(raw[a] * longs[a] for a in raw)
            target = vol_target_ann or 0.10
            scale = min(target / (port_vol + 1e-10), 1.5)
            weights = {a: w * scale for a, w in raw.items()}
            # Cap total weight at 1.0
            tw = sum(weights.values())
            if tw > 1.0:
                weights = {a: w / tw for a, w in weights.items()}
        else:
            weights = {a: 1.0 / n_long for a in longs}

        # Portfolio return
        port_ret = 0.0
        for asset, w in weights.items():
            if asset in monthly_rets.columns and i + 1 < len(monthly_rets):
                r = monthly_rets.iloc[i + 1][asset]
                if not pd.isna(r):
                    port_ret += w * r

        # Transaction costs
        cost = cost_bps / 10000 * 2 * 0.25  # ~25% monthly turnover for trend
        port_ret -= cost

        # Regime
        regime = 'bull'
        if spy_monthly is not None and spy_sma200 is not None:
            sp = spy_monthly.iloc[i] if i < len(spy_monthly) else np.nan
            ss = spy_sma200.iloc[i] if i < len(spy_sma200) else np.nan
            regime = 'bear' if (not pd.isna(sp) and not pd.isna(ss) and sp < ss) else 'bull'

        all_returns.append(port_ret)
        all_dates.append(date)
        all_regimes.append(regime)
        all_n_long.append(n_long)

    if not all_returns:
        return None

    returns = np.array(all_returns)
    regimes = np.array(all_regimes)

    # Metrics
    total_ret = np.prod(1 + returns) - 1
    n_years = len(returns) / 12
    cagr = (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1

    ann_vol = np.std(returns) * np.sqrt(12)
    sharpe = (np.mean(returns) * 12) / (ann_vol + 1e-10)

    downside = returns[returns < 0]
    downside_vol = np.std(downside) * np.sqrt(12) if len(downside) > 0 else 1e-10
    sortino = (np.mean(returns) * 12) / (downside_vol + 1e-10)

    cum = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    maxdd = dd.min()

    wins = returns[returns > 0]
    losses = returns[returns < 0]
    wr = len(wins) / len(returns) * 100
    pf = abs(wins.sum() / losses.sum()) if len(losses) > 0 and losses.sum() != 0 else 999
    calmar = cagr / abs(maxdd) if maxdd != 0 else 999

    # R1 regime
    bull_rets = returns[regimes == 'bull']
    bear_rets = returns[regimes == 'bear']
    bull_sharpe = (np.mean(bull_rets) * 12) / (np.std(bull_rets) * np.sqrt(12) + 1e-10) if len(bull_rets) > 3 else 0
    bear_sharpe = (np.mean(bear_rets) * 12) / (np.std(bear_rets) * np.sqrt(12) + 1e-10) if len(bear_rets) > 3 else 0
    r1_gap = abs(bull_sharpe - bear_sharpe) / max(abs(bull_sharpe), abs(bear_sharpe), 0.01)

    # Correlation with SPY
    if spy_close is not None:
        spy_rets = spy_close.resample('ME').last().pct_change().iloc[start_idx+1:start_idx+1+len(returns)].values
        if len(spy_rets) == len(returns):
            corr_spy = float(np.corrcoef(returns, spy_rets.flatten())[0, 1])
        else:
            corr_spy = np.nan
    else:
        corr_spy = np.nan

    # Correlation with sector momentum proxy (QQQ monthly returns as proxy)
    # We don't have sector momentum returns here, so use SPY correlation as proxy

    result = {
        'name': name,
        'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2),
        'cagr_pct': round(cagr * 100, 1),
        'maxdd_pct': round(maxdd * 100, 1),
        'wr_pct': round(wr, 1),
        'pf': round(pf, 2),
        'calmar': round(calmar, 2),
        'n_months': len(returns),
        'n_years': round(n_years, 1),
        'avg_n_long': round(np.mean(all_n_long), 1),
        'bull_sharpe': round(bull_sharpe, 2),
        'bear_sharpe': round(bear_sharpe, 2),
        'r1_gap': round(r1_gap, 3),
        'r1_pass': r1_gap <= 0.50,
        'corr_spy': round(corr_spy, 3) if not np.isnan(corr_spy) else None,
        'bull_months': int(sum(regimes == 'bull')),
        'bear_months': int(sum(regimes == 'bear')),
        'returns': returns.tolist(),
        'dates': [str(d) for d in all_dates],
        'regimes': regimes.tolist(),
    }

    fprint(f"  {name}: Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}%, MaxDD {maxdd*100:.1f}%, "
           f"WR {wr:.1f}%, Corr(SPY) {corr_spy:.3f}, Avg Long {np.mean(all_n_long):.0f}, "
           f"R1 gap {r1_gap:.3f} {'PASS' if r1_gap <= 0.50 else 'FAIL'}")

    return result


def permutation_test(returns, n_perms=1000):
    real_sharpe = np.mean(returns) / (np.std(returns) + 1e-10)
    block_size = 3
    n_blocks = len(returns) // block_size
    if n_blocks < 4:
        return 1.0
    blocked = [returns[i*block_size:(i+1)*block_size] for i in range(n_blocks)]
    count = 0
    for _ in range(n_perms):
        perm = np.random.permutation(n_blocks)
        pr = np.concatenate([blocked[i] for i in perm])
        shift = np.random.randint(1, len(pr))
        pr = np.roll(pr, shift)
        for i in range(0, len(pr), block_size):
            if np.random.random() < 0.5:
                pr[i:i+block_size] = -pr[i:i+block_size]
        if np.mean(pr) / (np.std(pr) + 1e-10) >= real_sharpe:
            count += 1
    return count / n_perms


def sub_period_test(returns, n_splits=3):
    n = len(returns)
    chunk = n // n_splits
    subs = [np.mean(returns[i*chunk:(i+1)*chunk]) * 12 / (np.std(returns[i*chunk:(i+1)*chunk]) * np.sqrt(12) + 1e-10)
            for i in range(n_splits)]
    return all(s > 0 for s in subs), subs


def outlier_test(returns, trim_pct=5):
    n_trim = max(1, int(len(returns) * trim_pct / 100))
    trimmed = np.sort(returns)[n_trim:-n_trim]
    orig = np.mean(returns) / (np.std(returns) + 1e-10)
    trim = np.mean(trimmed) / (np.std(trimmed) + 1e-10)
    return trim > 0 and (trim / (orig + 1e-10)) > 0.5, orig, trim


def main():
    import yfinance as yf

    fprint(f"Alt Trend Following v1 (Non-Equity) — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 70)

    tickers = list(UNIVERSE.keys()) + ['SPY']
    fprint("Downloading data...")
    raw = yf.download(tickers, start='2008-01-01', end='2026-07-25', progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw['Close']
    else:
        close = raw

    spy_close = close['SPY'] if 'SPY' in close.columns else None

    valid = [t for t in UNIVERSE if t in close.columns and close[t].dropna().shape[0] > 300]
    close_data = close[valid].dropna(how='all')

    fprint(f"Data: {len(valid)} non-equity assets")
    fprint(f"Assets: {', '.join(valid)}")
    fprint(f"Range: {close_data.index[0].strftime('%Y-%m-%d')} to {close_data.index[-1].strftime('%Y-%m-%d')}")

    variants = [
        ('A_SMA200_EqWt', 'sma200', 'equal', None),
        ('B_DualMom_EqWt', 'dual', 'equal', None),
        ('C_DualMom_RiskParity', 'dual', 'risk_parity', None),
        ('D_DualMom_VolTarget10', 'dual', 'vol_target', 0.10),
        ('E_MultiTF_EqWt', 'multi_tf', 'equal', None),
        ('F_SMA200_RiskParity', 'sma200', 'risk_parity', None),
        ('G_SMA200_VolTarget8', 'sma200', 'vol_target', 0.08),
    ]

    results = []

    if MLFLOW_OK:
        exp_name = 'alt_trend_following_v1'
        try:
            exp = mlflow.get_experiment_by_name(exp_name)
            if exp is None:
                mlflow.create_experiment(exp_name)
        except:
            pass
        mlflow.set_experiment(exp_name)

    for vname, mode, wt, vt in variants:
        try:
            if MLFLOW_OK:
                with mlflow.start_run(run_name=vname):
                    r = run_variant(close_data, spy_close, vname, mode=mode,
                                   weight_mode=wt, vol_target_ann=vt)
                    if r:
                        mlflow.log_params({'mode': mode, 'weight_mode': wt, 'vol_target': vt or 'none'})
                        mlflow.log_metrics({k: v for k, v in r.items()
                                          if isinstance(v, (int, float)) and k != 'corr_spy' and v is not None})
                        if r['corr_spy'] is not None:
                            mlflow.log_metric('corr_spy', r['corr_spy'])
                        results.append(r)
            else:
                r = run_variant(close_data, spy_close, vname, mode=mode,
                               weight_mode=wt, vol_target_ann=vt)
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
        rets = np.array(r['returns'])
        r['perm_p'] = round(permutation_test(rets), 3)
        r['g1_pass'] = r['perm_p'] < 0.05
        r['g2_pass'] = r['r1_pass']
        sp, ss = sub_period_test(rets)
        r['g3_pass'] = sp
        r['sub_sharpes'] = [round(s, 2) for s in ss]
        op, _, ts = outlier_test(rets)
        r['g4_pass'] = op
        r['trimmed_sharpe'] = round(ts, 2)
        r['gates_passed'] = sum([r['g1_pass'], r['g2_pass'], r['g3_pass'], r['g4_pass']])

        fprint(f"\n{r['name']}:")
        fprint(f"  Sharpe {r['sharpe']}, CAGR {r['cagr_pct']}%, MaxDD {r['maxdd_pct']}%, Corr(SPY) {r['corr_spy']}")
        fprint(f"  G1 Perm: {'PASS' if r['g1_pass'] else 'FAIL'} (p={r['perm_p']})")
        fprint(f"  G2 R1:   {'PASS' if r['g2_pass'] else 'FAIL'} (gap={r['r1_gap']})")
        fprint(f"  G3 Sub:  {'PASS' if r['g3_pass'] else 'FAIL'} (sharpes={r['sub_sharpes']})")
        fprint(f"  G4 Out:  {'PASS' if r['g4_pass'] else 'FAIL'} (trimmed={r['trimmed_sharpe']})")
        fprint(f"  GATES: {r['gates_passed']}/4")

    # Summary
    fprint("\n" + "=" * 70)
    fprint("SUMMARY — NON-EQUITY TREND FOLLOWING")
    fprint("=" * 70)
    fprint(f"{'Name':<30} {'Sharpe':>7} {'CAGR':>7} {'MaxDD':>7} {'WR':>6} {'Corr':>6} {'R1':>7} {'Gates':>6}")
    fprint("-" * 85)
    for r in sorted(results, key=lambda x: x['sharpe'], reverse=True):
        fprint(f"{r['name']:<30} {r['sharpe']:>7.2f} {r['cagr_pct']:>6.1f}% {r['maxdd_pct']:>6.1f}% "
               f"{r['wr_pct']:>5.1f}% {r['corr_spy'] or 0:>5.2f} {r['r1_gap']:>6.3f} {r['gates_passed']:>4}/4")

    # THE KEY QUESTION
    fprint("\n" + "=" * 70)
    fprint("DECORRELATION CHECK (vs SPY buy-and-hold)")
    fprint("=" * 70)
    fprint("Target: Corr < 0.30 for meaningful diversification")
    fprint("Cross-Asset Trend v1 (with equities): Corr 0.55-0.68")
    for r in results:
        c = r['corr_spy'] or 0
        quality = 'EXCELLENT' if c < 0.20 else 'GOOD' if c < 0.30 else 'MODERATE' if c < 0.50 else 'POOR'
        fprint(f"  {r['name']}: Corr(SPY) = {c:.3f} → {quality}")

    # Save
    save = [{k: v for k, v in r.items() if k not in ['returns', 'dates', 'regimes']} for r in results]
    with open(RESULTS_PATH, 'w') as f:
        json.dump(save, f, indent=2, default=str)
    fprint(f"\nResults saved. Done — {datetime.now().strftime('%H:%M:%S')}")


if __name__ == '__main__':
    main()
