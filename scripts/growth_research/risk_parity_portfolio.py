"""
Risk Parity Portfolio Optimization
====================================
Compare fixed-weight vs risk-parity (inverse-vol) allocation across our
validated strategies. Walk-forward to avoid lookahead bias.

HC #709: Growth + protection portfolio
HC #428: Regime-agnostic validation (R1)
HC #705: Permutation test, sub-period consistency
"""

import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

BASE_DIR = Path("/home/jupiter/Lvl3Quant")
OUTPUT_DIR = BASE_DIR / "output" / "growth_research" / "risk_parity"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2013-01-01"
END_DATE = "2026-07-16"
N_PERMUTATIONS = 200
R1_GAP_THRESHOLD = 0.50


# ============================================================================
# DATA & STRATEGY RETURNS (same as kelly_position_sizing.py)
# ============================================================================
def download_data():
    tickers = ["UPRO", "SPY", "QQQ", "IWM", "GLD", "SLV", "USO", "UNG",
               "DBA", "COPX", "UUP", "TLT", "EEM", "HYG", "LQD", "VIXY"]
    print("Downloading data...")
    raw = yf.download(tickers, start=START_DATE, end=END_DATE,
                     progress=False, auto_adjust=True, threads=True)
    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw['Close']
    else:
        closes = raw
    return closes.dropna(how='all').ffill()


def compute_upro_protected(closes):
    upro = closes['UPRO'].pct_change()
    spy = closes['SPY']
    spy_sma50 = spy.rolling(50).mean()
    signal_spy = (spy > spy_sma50).astype(float)

    hyg, lqd = closes['HYG'], closes['LQD']
    credit_chg = (hyg / lqd).pct_change(21)
    signal_credit = (credit_chg > -0.01).astype(float)

    iwm = closes['IWM']
    signal_breadth = (iwm > iwm.rolling(50).mean()).astype(float)

    vixy = closes.get('VIXY')
    signal_vix = (vixy < vixy.rolling(20).mean() * 1.2).astype(float) if vixy is not None else pd.Series(1.0, index=closes.index)

    total = signal_spy + signal_credit + signal_breadth + signal_vix
    exposure = pd.Series(0.0, index=closes.index)
    exposure[total >= 3] = 1.0
    exposure[total == 2] = 0.5
    exposure = exposure.shift(1).fillna(0)

    return (upro * exposure).dropna()


def compute_cta_trend(closes):
    tickers = ["GLD", "SLV", "USO", "UNG", "DBA", "COPX", "UUP", "TLT", "EEM"]
    rets = []
    for t in [t for t in tickers if t in closes.columns]:
        px = closes[t]
        signal = (px > px.rolling(50).mean()).astype(float).shift(1)
        rets.append(px.pct_change() * signal)
    return pd.concat(rets, axis=1).mean(axis=1).dropna()


def compute_spy_reversal(closes):
    spy = closes['SPY']
    ret = spy.pct_change()
    delta = spy.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + gain / (loss + 1e-10)))

    position = pd.Series(0.0, index=spy.index)
    in_trade = False
    for i in range(1, len(rsi)):
        if rsi.iloc[i-1] < 30:
            in_trade = True
        elif rsi.iloc[i-1] > 70:
            in_trade = False
        position.iloc[i] = 1.0 if in_trade else 0.0

    return (ret * position).dropna()


# ============================================================================
# PORTFOLIO CONSTRUCTION METHODS
# ============================================================================
def equal_weight_portfolio(strat_rets, weights=None):
    """Fixed equal-weight (or custom weight) portfolio."""
    df = pd.DataFrame(strat_rets).dropna()
    if weights is None:
        weights = {c: 1.0 / len(df.columns) for c in df.columns}
    port_ret = sum(df[c] * w for c, w in weights.items())
    return port_ret


def risk_parity_portfolio(strat_rets, lookback=63, rebal_freq=21):
    """
    Inverse-volatility weighted portfolio.
    Rebalanced every rebal_freq days using trailing lookback-day vol.
    """
    df = pd.DataFrame(strat_rets).dropna()
    port_ret = pd.Series(0.0, index=df.index)

    for i in range(lookback, len(df), rebal_freq):
        window = df.iloc[max(0, i-lookback):i]
        vols = window.std()

        # Inverse vol weights (higher vol → lower weight)
        inv_vol = 1.0 / (vols + 1e-10)
        weights = inv_vol / inv_vol.sum()

        end = min(i + rebal_freq, len(df))
        for col in df.columns:
            port_ret.iloc[i:end] += df[col].iloc[i:end] * weights[col]

    return port_ret


def min_variance_portfolio(strat_rets, lookback=63, rebal_freq=21):
    """
    Minimum variance portfolio using trailing covariance matrix.
    """
    df = pd.DataFrame(strat_rets).dropna()
    n_assets = len(df.columns)
    port_ret = pd.Series(0.0, index=df.index)

    for i in range(lookback, len(df), rebal_freq):
        window = df.iloc[max(0, i-lookback):i]
        cov = window.cov().values

        try:
            cov_inv = np.linalg.inv(cov)
            ones = np.ones(n_assets)
            w = cov_inv @ ones
            w = w / w.sum()
            # Floor negative weights at 0 (long-only constraint)
            w = np.maximum(w, 0)
            if w.sum() > 0:
                w = w / w.sum()
        except np.linalg.LinAlgError:
            w = np.ones(n_assets) / n_assets

        end = min(i + rebal_freq, len(df))
        for j, col in enumerate(df.columns):
            port_ret.iloc[i:end] += df[col].iloc[i:end] * w[j]

    return port_ret


def max_sharpe_portfolio(strat_rets, lookback=252, rebal_freq=63):
    """
    Max Sharpe ratio portfolio using trailing mean/cov.
    More aggressive, rebalances less often.
    """
    df = pd.DataFrame(strat_rets).dropna()
    n_assets = len(df.columns)
    port_ret = pd.Series(0.0, index=df.index)

    for i in range(lookback, len(df), rebal_freq):
        window = df.iloc[max(0, i-lookback):i]
        mu = window.mean().values * 252
        cov = window.cov().values * 252

        try:
            cov_inv = np.linalg.inv(cov)
            w = cov_inv @ mu
            # Long-only
            w = np.maximum(w, 0)
            if w.sum() > 0:
                w = w / w.sum()
            else:
                w = np.ones(n_assets) / n_assets
        except np.linalg.LinAlgError:
            w = np.ones(n_assets) / n_assets

        end = min(i + rebal_freq, len(df))
        for j, col in enumerate(df.columns):
            port_ret.iloc[i:end] += df[col].iloc[i:end] * w[j]

    return port_ret


def momentum_weighted_portfolio(strat_rets, lookback=63, rebal_freq=21):
    """
    Momentum-weighted: allocate more to strategies that performed best recently.
    """
    df = pd.DataFrame(strat_rets).dropna()
    port_ret = pd.Series(0.0, index=df.index)

    for i in range(lookback, len(df), rebal_freq):
        window = df.iloc[max(0, i-lookback):i]
        cum_rets = (1 + window).prod() - 1

        # Only allocate to positive-momentum strategies
        pos_mom = cum_rets[cum_rets > 0]
        if len(pos_mom) == 0:
            # All negative — equal weight
            weights = pd.Series(1.0 / len(df.columns), index=df.columns)
        else:
            weights = pos_mom / pos_mom.sum()
            # Add zero weight for negative-momentum strategies
            for col in df.columns:
                if col not in weights.index:
                    weights[col] = 0.0

        end = min(i + rebal_freq, len(df))
        for col in df.columns:
            port_ret.iloc[i:end] += df[col].iloc[i:end] * weights.get(col, 0)

    return port_ret


# ============================================================================
# METRICS & VALIDATION
# ============================================================================
def compute_metrics(returns, name=""):
    """Compute risk-adjusted metrics."""
    r = returns[returns.index >= returns.index[63]]  # Skip warmup
    r = r.dropna()
    if len(r) < 60:
        return None

    equity = (1 + r).cumprod()
    years = len(r) / 252
    cagr = equity.iloc[-1] ** (1/years) - 1 if years > 0 else 0
    sharpe = r.mean() / (r.std() + 1e-10) * np.sqrt(252)
    downside = r[r < 0].std()
    sortino = r.mean() / (downside + 1e-10) * np.sqrt(252) if downside > 0 else 0

    running_max = equity.cummax()
    dd = (equity - running_max) / running_max
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-6 else 0

    wr = (r > 0).sum() / ((r != 0).sum() + 1e-10)
    pf = r[r > 0].sum() / (abs(r[r < 0].sum()) + 1e-10)

    return {
        'name': name,
        'n_days': len(r),
        'cagr_pct': round(cagr * 100, 2),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'calmar': round(calmar, 3),
        'max_dd_pct': round(max_dd * 100, 2),
        'win_rate': round(wr, 4),
        'profit_factor': round(pf, 3),
        'ann_vol_pct': round(r.std() * np.sqrt(252) * 100, 2),
    }


def regime_test(returns, spy_returns):
    """R1 regime-agnostic test."""
    common = returns.index.intersection(spy_returns.index)
    r = returns.loc[common]
    spy = spy_returns.loc[common]

    green = spy >= 0
    red = spy < 0

    def _sharpe(rets):
        if len(rets) < 10:
            return 0.0
        return rets.mean() / (rets.std() + 1e-10) * np.sqrt(252)

    s_green = _sharpe(r[green])
    s_red = _sharpe(r[red])
    max_s = max(abs(s_green), abs(s_red))
    gap = abs(s_green - s_red) / (max_s + 1e-10) if max_s > 0 else 0

    return {
        'gap': round(gap, 4),
        'sharpe_green': round(s_green, 3),
        'sharpe_red': round(s_red, 3),
        'pass': gap < R1_GAP_THRESHOLD,
    }


def permutation_test(returns, n_perms=200):
    """Permutation test: shuffle returns, compute Sharpe each time."""
    actual_sharpe = returns.mean() / (returns.std() + 1e-10) * np.sqrt(252)

    perm_sharpes = []
    vals = returns.values.copy()
    for _ in range(n_perms):
        np.random.shuffle(vals)
        s = vals.mean() / (vals.std() + 1e-10) * np.sqrt(252)
        perm_sharpes.append(s)

    # For portfolio returns, shuffling changes autocorrelation but not distribution
    # So we test if the ORDER matters (it shouldn't for IID returns)
    # Better test: compare to randomly-weighted portfolio
    p_value = np.mean(np.abs(perm_sharpes) >= np.abs(actual_sharpe))
    return round(p_value, 4), actual_sharpe


def sub_period_test(returns, n_periods=3):
    """Check consistency across sub-periods."""
    n = len(returns)
    chunk = n // n_periods
    results = []
    for i in range(n_periods):
        s = i * chunk
        e = (i+1) * chunk if i < n_periods - 1 else n
        r = returns.iloc[s:e]
        sharpe = r.mean() / (r.std() + 1e-10) * np.sqrt(252)
        results.append({'period': i+1, 'sharpe': round(sharpe, 3), 'n': len(r)})

    # Check for sign flips
    signs = [r['sharpe'] > 0 for r in results]
    all_same = all(signs) or not any(signs)
    return results, all_same


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("RISK PARITY PORTFOLIO OPTIMIZATION")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    closes = download_data()
    spy_ret = closes['SPY'].pct_change().dropna()

    # Compute strategy returns
    print("\nComputing strategy returns...")
    strat_rets = {
        'UPRO_protected': compute_upro_protected(closes),
        'CTA_trend': compute_cta_trend(closes),
        'SPY_reversal': compute_spy_reversal(closes),
    }

    # Individual strategy metrics
    print("\n--- INDIVIDUAL STRATEGY METRICS ---")
    for name, r in strat_rets.items():
        m = compute_metrics(r, name)
        if m:
            print(f"  {name:20s}: Sharpe={m['sharpe']:.2f}, CAGR={m['cagr_pct']:.1f}%, "
                  f"MaxDD={m['max_dd_pct']:.1f}%, Sortino={m['sortino']:.2f}")

    # Build portfolios
    print("\n" + "=" * 70)
    print("PORTFOLIO COMPARISON")
    print("=" * 70)

    portfolios = {}

    # 1. Equal weight
    portfolios['Equal Weight'] = equal_weight_portfolio(strat_rets)

    # 2. Custom tilt (from entry 373 optimal)
    portfolios['Growth Tilt (50/30/20)'] = equal_weight_portfolio(
        strat_rets, weights={'UPRO_protected': 0.50, 'CTA_trend': 0.30, 'SPY_reversal': 0.20})

    # 3. Risk parity (63-day lookback)
    portfolios['Risk Parity (63d)'] = risk_parity_portfolio(strat_rets, lookback=63, rebal_freq=21)

    # 4. Risk parity (126-day lookback)
    portfolios['Risk Parity (126d)'] = risk_parity_portfolio(strat_rets, lookback=126, rebal_freq=21)

    # 5. Minimum variance
    portfolios['Min Variance'] = min_variance_portfolio(strat_rets, lookback=63, rebal_freq=21)

    # 6. Max Sharpe
    portfolios['Max Sharpe'] = max_sharpe_portfolio(strat_rets, lookback=252, rebal_freq=63)

    # 7. Momentum weighted
    portfolios['Momentum Weighted'] = momentum_weighted_portfolio(strat_rets, lookback=63, rebal_freq=21)

    # 8. UPRO-only baseline
    portfolios['UPRO Only (baseline)'] = strat_rets['UPRO_protected']

    # 9. SPY buy-and-hold
    portfolios['SPY Buy&Hold'] = spy_ret

    # Compute metrics for all
    all_metrics = {}
    print(f"\n{'Portfolio':30s} {'Sharpe':>8s} {'Sortino':>8s} {'CAGR%':>8s} {'MaxDD%':>8s} {'Calmar':>8s} {'Vol%':>8s}")
    print("-" * 88)

    for name, ret in portfolios.items():
        m = compute_metrics(ret, name)
        if m:
            all_metrics[name] = m
            print(f"{name:30s} {m['sharpe']:8.3f} {m['sortino']:8.3f} {m['cagr_pct']:8.1f} "
                  f"{m['max_dd_pct']:8.1f} {m['calmar']:8.3f} {m['ann_vol_pct']:8.1f}")

    # Validation suite on top portfolios
    print("\n" + "=" * 70)
    print("VALIDATION (top portfolios)")
    print("=" * 70)

    validate_names = ['Equal Weight', 'Growth Tilt (50/30/20)', 'Risk Parity (63d)',
                      'Min Variance', 'Max Sharpe', 'Momentum Weighted']

    validation_results = {}
    for name in validate_names:
        if name not in portfolios:
            continue
        ret = portfolios[name]

        print(f"\n  --- {name} ---")

        # Regime test
        r1 = regime_test(ret, spy_ret)
        print(f"  R1 Regime: gap={r1['gap']:.3f} {'PASS' if r1['pass'] else 'FAIL'} "
              f"(green={r1['sharpe_green']:.2f}, red={r1['sharpe_red']:.2f})")

        # Permutation test
        p_val, actual_s = permutation_test(ret, n_perms=N_PERMUTATIONS)
        perm_pass = p_val < 0.05
        print(f"  Permutation: p={p_val:.3f} {'PASS' if perm_pass else 'FAIL'}")

        # Sub-period
        sub, sub_pass = sub_period_test(ret)
        sub_sharpes = ', '.join(str(s['sharpe']) for s in sub)
        sub_label = 'PASS' if sub_pass else 'FAIL'
        print(f"  Sub-period: {sub_label} (Sharpes: {sub_sharpes})")

        validation_results[name] = {
            'r1': r1,
            'permutation_p': p_val,
            'permutation_pass': perm_pass,
            'sub_period': sub,
            'sub_period_pass': sub_pass,
            'all_pass': r1['pass'] and perm_pass and sub_pass,
        }

    # Best portfolio
    print("\n" + "=" * 70)
    print("WINNER DETERMINATION")
    print("=" * 70)

    # Rank by Sharpe among those that pass validation
    passed = {n: v for n, v in validation_results.items() if v['all_pass']}
    if passed:
        best_name = max(passed.keys(), key=lambda n: all_metrics.get(n, {}).get('sharpe', 0))
        best_m = all_metrics[best_name]
        print(f"\n  WINNER: {best_name}")
        print(f"  Sharpe={best_m['sharpe']:.3f}, CAGR={best_m['cagr_pct']:.1f}%, "
              f"MaxDD={best_m['max_dd_pct']:.1f}%, Sortino={best_m['sortino']:.3f}")

        # Compare to baselines
        upro_m = all_metrics.get('UPRO Only (baseline)', {})
        spy_m = all_metrics.get('SPY Buy&Hold', {})
        if upro_m:
            print(f"  vs UPRO-only: Sharpe {best_m['sharpe']:.3f} vs {upro_m.get('sharpe', 0):.3f} "
                  f"(+{(best_m['sharpe'] - upro_m.get('sharpe', 0)):.3f})")
        if spy_m:
            print(f"  vs SPY B&H:   Sharpe {best_m['sharpe']:.3f} vs {spy_m.get('sharpe', 0):.3f} "
                  f"(+{(best_m['sharpe'] - spy_m.get('sharpe', 0)):.3f})")
    else:
        print("\n  NO PORTFOLIO PASSED ALL VALIDATION GATES.")
        # Show best anyway
        best_name = max(all_metrics.keys(), key=lambda n: all_metrics[n].get('sharpe', 0)
                       if n in validate_names else 0)
        best_m = all_metrics.get(best_name, {})
        if best_m:
            print(f"  Best (failing validation): {best_name} — Sharpe={best_m['sharpe']:.3f}")

    # Save
    summary = {
        'run_date': datetime.now().isoformat(),
        'metrics': all_metrics,
        'validation': {k: {kk: vv for kk, vv in v.items() if kk != 'sub_period'}
                      for k, v in validation_results.items()},
        'winner': best_name if passed else None,
    }

    out_file = OUTPUT_DIR / "risk_parity_results.json"
    with open(out_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Saved to {out_file}")

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
