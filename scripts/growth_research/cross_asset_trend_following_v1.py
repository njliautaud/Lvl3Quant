#!/usr/bin/env python3
"""
Cross-Asset Trend Following v1 — CTA-Style Diversified Trend
=============================================================

Problem: All growth strategies are 92% correlated (sector ETF momentum).
Need a DECORRELATED return stream.

Trend following (managed futures / CTA style) historically has LOW correlation
with equity momentum (often negative in crises — "crisis alpha").

Strategy:
- Trade across asset classes: equities, bonds, commodities, currencies, REITs
- Use dual momentum (absolute + relative) for each asset
- Go LONG if above SMA and positive momentum, FLAT otherwise (long-only for simplicity)
- Rebalance monthly, risk-parity weighted

This is fundamentally different from sector ETF momentum:
- Sector momentum: rank sectors, buy top N
- Trend following: go long ANY asset showing uptrend, go flat if downtrend

Variants:
A. Simple trend (above SMA200 = long, else flat), equal weight
B. Dual momentum (above SMA200 + positive 12-1 mom), equal weight
C. Risk parity weighted trend
D. Multi-timeframe trend (SMA50 + SMA200 both agree)
E. Trend with volatility targeting (scale position by inverse vol)
F. Carry + trend (add carry signal for bonds/commodities)

Universe: broad cross-asset
Cost: 20bps round-trip.
Walk-forward: monthly, 2008-2026.
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
    print(*args, **kwargs)
    sys.stdout.flush()

RESULTS_DIR = Path('/home/jupiter/Lvl3Quant/research/findings')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = RESULTS_DIR / 'cross_asset_trend_following_v1_results.json'

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

# Broad cross-asset universe
# Key: different asset classes for decorrelation
UNIVERSE = {
    # US Equities (broad)
    'SPY': 'equity',      # S&P 500
    'QQQ': 'equity',      # Nasdaq
    'IWM': 'equity',      # Small cap
    'EFA': 'equity',      # International developed
    'EEM': 'equity',      # Emerging markets

    # Fixed Income
    'TLT': 'bond',        # Long-term treasury
    'IEF': 'bond',        # 7-10yr treasury
    'HYG': 'bond',        # High yield
    'LQD': 'bond',        # Investment grade corporate
    'TIP': 'bond',        # TIPS

    # Commodities
    'GLD': 'commodity',   # Gold
    'SLV': 'commodity',   # Silver
    'DBC': 'commodity',   # Broad commodities
    'USO': 'commodity',   # Oil

    # Real Estate
    'VNQ': 'reit',        # US REITs
    'IYR': 'reit',        # US REITs

    # Alternatives
    'XLE': 'sector',      # Energy (acts like commodity)
    'XLU': 'sector',      # Utilities (acts like bond)
}


def run_variant(close_data, spy_close, name, mode='sma200', weight_mode='equal',
                vol_target=False, multi_tf=False, cost_bps=10):
    """
    Run a trend following variant.

    Modes:
    - sma200: long if price > SMA200, else flat
    - dual: long if price > SMA200 AND 12-1 mom > 0, else flat
    - multi_tf: long if BOTH SMA50 and SMA200 agree
    """
    fprint(f"\n--- {name} ---")

    # Monthly rebalancing
    monthly_close = close_data.resample('ME').last().dropna(how='all')
    spy_monthly = spy_close.resample('ME').last()

    # Calculate signals for each asset
    n_assets = len(monthly_close.columns)

    # Compute monthly returns
    monthly_rets = monthly_close.pct_change()

    # Compute signals
    # For SMA, we use daily data then sample monthly
    sma200 = close_data.rolling(200).mean().resample('ME').last()
    sma50 = close_data.rolling(50).mean().resample('ME').last()

    # 12-1 momentum (skip last month)
    mom_12_1 = monthly_close.pct_change(12).shift(0) - monthly_close.pct_change(1).shift(0)
    # Actually: 12m return minus 1m return
    mom_12 = monthly_close.pct_change(12)
    mom_1 = monthly_close.pct_change(1)
    mom_12_1 = mom_12 - mom_1

    # Rolling volatility (daily, sampled monthly)
    daily_rets = close_data.pct_change()
    rolling_vol = (daily_rets.rolling(63).std() * np.sqrt(252)).resample('ME').last()

    # Start after enough data
    start_idx = max(13, 1)  # Need 12 months of history minimum

    all_returns = []
    all_dates = []
    all_regimes = []
    all_n_long = []

    for i in range(start_idx, len(monthly_close) - 1):
        date = monthly_close.index[i]
        next_date = monthly_close.index[i + 1]

        # Determine which assets to hold
        signals = {}
        for asset in monthly_close.columns:
            try:
                price = monthly_close.iloc[i][asset]
                if pd.isna(price):
                    continue

                # SMA200 trend
                s200 = sma200.iloc[i][asset] if asset in sma200.columns else np.nan
                s50 = sma50.iloc[i][asset] if asset in sma50.columns else np.nan
                m12_1 = mom_12_1.iloc[i][asset] if asset in mom_12_1.columns else np.nan

                if pd.isna(s200):
                    continue

                if mode == 'sma200':
                    long = price > s200
                elif mode == 'dual':
                    long = (price > s200) and (not pd.isna(m12_1)) and (m12_1 > 0)
                elif mode == 'multi_tf':
                    if pd.isna(s50):
                        continue
                    long = (price > s200) and (price > s50)
                else:
                    long = price > s200

                if long:
                    signals[asset] = 1.0
            except:
                continue

        n_long = len(signals)

        if n_long == 0:
            all_returns.append(0.0)
            all_dates.append(date)
            # Regime
            spy_price = spy_monthly.iloc[i] if i < len(spy_monthly) else np.nan
            spy_sma = sma200.iloc[i]['SPY'] if 'SPY' in sma200.columns and i < len(sma200) else np.nan
            regime = 'bear' if (not pd.isna(spy_price) and not pd.isna(spy_sma) and spy_price < spy_sma) else 'bull'
            all_regimes.append(regime)
            all_n_long.append(0)
            continue

        # Weight allocation
        if weight_mode == 'equal':
            weights = {a: 1.0 / n_long for a in signals}
        elif weight_mode == 'risk_parity':
            # Inverse volatility weighting
            vols = {}
            for a in signals:
                v = rolling_vol.iloc[i][a] if a in rolling_vol.columns else 0.15
                if pd.isna(v) or v < 0.01:
                    v = 0.15
                vols[a] = v
            inv_vols = {a: 1.0 / v for a, v in vols.items()}
            total_inv = sum(inv_vols.values())
            weights = {a: iv / total_inv for a, iv in inv_vols.items()}
        elif weight_mode == 'vol_target':
            # Target 10% annual portfolio vol
            target_vol = 0.10
            vols = {}
            for a in signals:
                v = rolling_vol.iloc[i][a] if a in rolling_vol.columns else 0.15
                if pd.isna(v) or v < 0.01:
                    v = 0.15
                vols[a] = v
            # Equal risk contribution, then scale to target
            inv_vols = {a: 1.0 / v for a, v in vols.items()}
            total_inv = sum(inv_vols.values())
            raw_weights = {a: iv / total_inv for a, iv in inv_vols.items()}
            # Estimate portfolio vol
            port_vol = sum(raw_weights[a] * vols[a] for a in raw_weights)
            scale = min(target_vol / (port_vol + 1e-10), 1.5)  # cap leverage at 1.5x
            weights = {a: w * scale for a, w in raw_weights.items()}
        else:
            weights = {a: 1.0 / n_long for a in signals}

        # Calculate portfolio return
        port_ret = 0.0
        for asset, w in weights.items():
            if asset in monthly_rets.columns and i + 1 < len(monthly_rets):
                r = monthly_rets.iloc[i + 1][asset]
                if not pd.isna(r):
                    port_ret += w * r

        # Transaction costs (approximate turnover)
        cost = cost_bps / 10000 * 2 * 0.3  # ~30% monthly turnover average
        port_ret -= cost

        # Regime
        spy_price = spy_monthly.iloc[i] if i < len(spy_monthly) else np.nan
        spy_sma = sma200.iloc[i]['SPY'] if 'SPY' in sma200.columns and i < len(sma200) else np.nan
        regime = 'bear' if (not pd.isna(spy_price) and not pd.isna(spy_sma) and spy_price < spy_sma) else 'bull'

        all_returns.append(port_ret)
        all_dates.append(date)
        all_regimes.append(regime)
        all_n_long.append(n_long)

    if not all_returns:
        return None

    returns = np.array(all_returns)
    regimes = np.array(all_regimes)
    n_long_arr = np.array(all_n_long)

    # Metrics (monthly → annual)
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
    r1_pass = r1_gap <= 0.50

    # Correlation with SPY buy-and-hold
    spy_rets = spy_monthly.pct_change().iloc[start_idx+1:start_idx+1+len(returns)].values
    if len(spy_rets) == len(returns):
        corr_spy = np.corrcoef(returns, spy_rets.flatten())[0, 1]
    else:
        corr_spy = np.nan

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
        'avg_n_long': round(np.mean(n_long_arr), 1),
        'bull_sharpe': round(bull_sharpe, 2),
        'bear_sharpe': round(bear_sharpe, 2),
        'r1_gap': round(r1_gap, 3),
        'r1_pass': r1_pass,
        'corr_spy': round(corr_spy, 3) if not np.isnan(corr_spy) else None,
        'bull_months': int(sum(regimes == 'bull')),
        'bear_months': int(sum(regimes == 'bear')),
        'returns': returns.tolist(),
        'dates': [str(d) for d in all_dates],
        'regimes': regimes.tolist(),
    }

    fprint(f"  {name}: Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}%, MaxDD {maxdd*100:.1f}%, "
           f"WR {wr:.1f}%, Corr(SPY) {corr_spy:.2f}, Avg Long {np.mean(n_long_arr):.0f}, "
           f"R1 gap {r1_gap:.3f} {'PASS' if r1_pass else 'FAIL'}")

    return result


def permutation_test(returns, n_perms=1000):
    """Block permutation test — shuffle returns in 3-month blocks."""
    real_sharpe = np.mean(returns) / (np.std(returns) + 1e-10)
    block_size = 3
    n_blocks = len(returns) // block_size

    if n_blocks < 4:
        return 1.0

    blocked = [returns[i*block_size:(i+1)*block_size] for i in range(n_blocks)]

    count_better = 0
    for _ in range(n_perms):
        perm_idx = np.random.permutation(n_blocks)
        perm_returns = np.concatenate([blocked[i] for i in perm_idx])
        shift = np.random.randint(1, len(perm_returns))
        perm_returns = np.roll(perm_returns, shift)
        for i in range(0, len(perm_returns), block_size):
            if np.random.random() < 0.5:
                perm_returns[i:i+block_size] = -perm_returns[i:i+block_size]

        perm_sharpe = np.mean(perm_returns) / (np.std(perm_returns) + 1e-10)
        if perm_sharpe >= real_sharpe:
            count_better += 1

    return count_better / n_perms


def sub_period_test(returns, n_splits=3):
    n = len(returns)
    chunk = n // n_splits
    sub_sharpes = []
    for i in range(n_splits):
        sub = returns[i*chunk:(i+1)*chunk]
        s = np.mean(sub) * 12 / (np.std(sub) * np.sqrt(12) + 1e-10)
        sub_sharpes.append(s)
    all_positive = all(s > 0 for s in sub_sharpes)
    return all_positive, sub_sharpes


def outlier_test(returns, trim_pct=5):
    n_trim = max(1, int(len(returns) * trim_pct / 100))
    sorted_rets = np.sort(returns)
    trimmed = sorted_rets[n_trim:-n_trim]

    orig_sharpe = np.mean(returns) / (np.std(returns) + 1e-10)
    trim_sharpe = np.mean(trimmed) / (np.std(trimmed) + 1e-10)
    passes = trim_sharpe > 0 and (trim_sharpe / (orig_sharpe + 1e-10)) > 0.5
    return passes, orig_sharpe, trim_sharpe


def main():
    import yfinance as yf

    fprint(f"Cross-Asset Trend Following v1 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 70)

    # Download data
    fprint("Downloading data...")
    tickers = list(UNIVERSE.keys()) + ['SPY']
    raw = yf.download(tickers, start='2008-01-01', end='2026-07-25', progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw['Close']
    else:
        close = raw

    spy_close = close['SPY'] if 'SPY' in close.columns else None

    # Filter to assets with enough data
    valid_assets = []
    for asset in UNIVERSE:
        if asset in close.columns:
            s = close[asset].dropna()
            if len(s) > 500:
                valid_assets.append(asset)

    close_data = close[valid_assets].dropna(how='all')
    fprint(f"Data: {len(valid_assets)} assets loaded across {len(set(UNIVERSE[a] for a in valid_assets))} asset classes")
    fprint(f"Range: {close_data.index[0].strftime('%Y-%m-%d')} to {close_data.index[-1].strftime('%Y-%m-%d')}")
    fprint(f"Assets: {', '.join(valid_assets)}")

    # Run variants
    variants = [
        ('A_SimpleTrend_EqWt', 'sma200', 'equal', False, False),
        ('B_DualMom_EqWt', 'dual', 'equal', False, False),
        ('C_DualMom_RiskParity', 'dual', 'risk_parity', False, False),
        ('D_MultiTF_EqWt', 'multi_tf', 'equal', False, False),
        ('E_DualMom_VolTarget', 'dual', 'vol_target', True, False),
        ('F_SimpleTrend_RiskParity', 'sma200', 'risk_parity', False, False),
    ]

    results = []

    if MLFLOW_OK:
        exp_name = 'cross_asset_trend_v1'
        try:
            exp = mlflow.get_experiment_by_name(exp_name)
            if exp is None:
                mlflow.create_experiment(exp_name)
        except:
            pass
        mlflow.set_experiment(exp_name)

    for vname, mode, wt_mode, vol_tgt, multi_tf in variants:
        try:
            if MLFLOW_OK:
                with mlflow.start_run(run_name=vname):
                    r = run_variant(close_data, spy_close, vname,
                                   mode=mode, weight_mode=wt_mode,
                                   vol_target=vol_tgt, multi_tf=multi_tf)
                    if r:
                        mlflow.log_params({
                            'mode': mode,
                            'weight_mode': wt_mode,
                            'vol_target': vol_tgt,
                        })
                        mlflow.log_metrics({
                            'sharpe': r['sharpe'],
                            'sortino': r['sortino'],
                            'cagr_pct': r['cagr_pct'],
                            'maxdd_pct': r['maxdd_pct'],
                            'wr_pct': r['wr_pct'],
                            'pf': r['pf'],
                            'r1_gap': r['r1_gap'],
                            'corr_spy': r['corr_spy'] or 0,
                        })
                        results.append(r)
            else:
                r = run_variant(close_data, spy_close, vname,
                               mode=mode, weight_mode=wt_mode,
                               vol_target=vol_tgt, multi_tf=multi_tf)
                if r:
                    results.append(r)
        except Exception as e:
            fprint(f"  ERROR {vname}: {e}")
            import traceback
            traceback.print_exc()

    if not results:
        fprint("\nNo results!")
        return

    # Adversarial validation
    fprint("\n" + "=" * 70)
    fprint("ADVERSARIAL VALIDATION")
    fprint("=" * 70)

    for r in results:
        rets = np.array(r['returns'])

        perm_p = permutation_test(rets, n_perms=1000)
        r['perm_p'] = round(perm_p, 3)
        r['g1_pass'] = perm_p < 0.05

        r['g2_pass'] = r['r1_pass']

        sub_pass, sub_sharpes = sub_period_test(rets)
        r['g3_pass'] = sub_pass
        r['sub_sharpes'] = [round(s, 2) for s in sub_sharpes]

        out_pass, orig_s, trim_s = outlier_test(rets)
        r['g4_pass'] = out_pass
        r['trimmed_sharpe'] = round(trim_s, 2)

        gates = sum([r['g1_pass'], r['g2_pass'], r['g3_pass'], r['g4_pass']])
        r['gates_passed'] = gates

        fprint(f"\n{r['name']}:")
        fprint(f"  Sharpe {r['sharpe']}, CAGR {r['cagr_pct']}%, MaxDD {r['maxdd_pct']}%, Corr(SPY) {r['corr_spy']}")
        fprint(f"  G1 Perm: {'PASS' if r['g1_pass'] else 'FAIL'} (p={r['perm_p']})")
        fprint(f"  G2 R1:   {'PASS' if r['g2_pass'] else 'FAIL'} (gap={r['r1_gap']})")
        fprint(f"  G3 Sub:  {'PASS' if r['g3_pass'] else 'FAIL'} (sharpes={r['sub_sharpes']})")
        fprint(f"  G4 Out:  {'PASS' if r['g4_pass'] else 'FAIL'} (trimmed={r['trimmed_sharpe']})")
        fprint(f"  GATES: {gates}/4")

    # Summary
    fprint("\n" + "=" * 70)
    fprint("SUMMARY TABLE")
    fprint("=" * 70)
    fprint(f"{'Name':<30} {'Sharpe':>7} {'CAGR':>7} {'MaxDD':>7} {'WR':>6} {'Corr':>6} {'R1':>7} {'Gates':>6}")
    fprint("-" * 85)
    for r in sorted(results, key=lambda x: x['sharpe'], reverse=True):
        fprint(f"{r['name']:<30} {r['sharpe']:>7.2f} {r['cagr_pct']:>6.1f}% {r['maxdd_pct']:>6.1f}% "
               f"{r['wr_pct']:>5.1f}% {r['corr_spy'] or 0:>5.2f} {r['r1_gap']:>6.3f} {r['gates_passed']:>4}/4")

    # Key question: decorrelation
    fprint("\n" + "=" * 70)
    fprint("DECORRELATION ANALYSIS")
    fprint("=" * 70)
    fprint("Sector ETF Momentum strategies have 0.92 correlation with each other.")
    fprint("For trend following to be useful in a portfolio, we need Corr < 0.50")
    for r in results:
        useful = (r['corr_spy'] or 0) < 0.50 if r['corr_spy'] is not None else False
        fprint(f"  {r['name']}: Corr(SPY) = {r['corr_spy']} → {'USEFUL for diversification' if useful else 'Too correlated'}")

    # Save
    save_results = []
    for r in results:
        r_save = {k: v for k, v in r.items() if k not in ['returns', 'dates', 'regimes']}
        save_results.append(r_save)

    with open(RESULTS_PATH, 'w') as f:
        json.dump(save_results, f, indent=2, default=str)
    fprint(f"\nResults saved to {RESULTS_PATH}")
    fprint(f"\nDone — {datetime.now().strftime('%H:%M:%S')}")


if __name__ == '__main__':
    main()
