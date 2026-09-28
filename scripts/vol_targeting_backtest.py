#!/usr/bin/env python3
"""
Volatility-Targeting / Risk-Parity Strategy Backtest
6 variants: GLD Vol-Target, Multi-Asset Vol-Target, Inverse VIX Sizing,
Risk Parity, Momentum+Vol Filter, Managed Volatility Portfolio.

OOT: 2022-01-01 to 2026-07-29, capital $645, slippage 0.02%, commission $0.
Validation: 5-gate framework + QQQ correlation + permutation test (1000) + regime split.
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from scipy import stats

warnings.filterwarnings('ignore')
np.random.seed(42)

# ─── Config ───
CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%
COMMISSION = 0.0
OOT_START = '2022-01-01'
OOT_END = '2026-07-29'
TICKERS = ['SPY', 'QQQ', 'GLD', 'TLT', 'UUP', 'IWM', '^VIX', 'RSP']
N_PERM = 1000

# ─── Data Download ───
print("Downloading data...")
# Need extra history for lookback
download_start = '2021-06-01'
raw = yf.download(TICKERS, start=download_start, end=OOT_END, auto_adjust=True, progress=False)

# Handle multi-level columns
if isinstance(raw.columns, pd.MultiIndex):
    close = raw['Close']
else:
    close = raw

# Flatten column names if needed
close.columns = [c[0] if isinstance(c, tuple) else c for c in close.columns]
close = close.ffill().dropna(how='all')

# Rename ^VIX
if '^VIX' in close.columns:
    close = close.rename(columns={'^VIX': 'VIX'})

print(f"Data shape: {close.shape}, range: {close.index[0].date()} to {close.index[-1].date()}")

returns = close.pct_change()

# OOT mask
oot_mask = (close.index >= OOT_START) & (close.index <= OOT_END)
oot_dates = close.index[oot_mask]
print(f"OOT dates: {len(oot_dates)} ({oot_dates[0].date()} to {oot_dates[-1].date()})")


def apply_slippage(ret_series):
    """Apply slippage on rebalance days (subtract from return)."""
    return ret_series  # slippage applied in strategy logic


def calc_metrics(equity_curve, name, qqq_rets):
    """Calculate comprehensive metrics for an equity curve."""
    rets = equity_curve.pct_change().dropna()
    if len(rets) < 20:
        return {'strategy': name, 'error': 'insufficient data'}

    total_ret = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    n_years = len(rets) / 252
    cagr = (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1
    ann_vol = rets.std() * np.sqrt(252)
    sharpe = (rets.mean() * 252) / (rets.std() * np.sqrt(252)) if rets.std() > 0 else 0
    downside = rets[rets < 0].std() * np.sqrt(252) if len(rets[rets < 0]) > 0 else 1e-6
    sortino = (rets.mean() * 252) / downside

    # Max drawdown
    cummax = equity_curve.cummax()
    dd = (equity_curve - cummax) / cummax
    max_dd = dd.min()

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Win rate
    wr = (rets > 0).mean()

    # Profit factor
    gains = rets[rets > 0].sum()
    losses = abs(rets[rets < 0].sum())
    pf = gains / losses if losses > 0 else float('inf')

    # QQQ correlation
    aligned = pd.DataFrame({'strat': rets, 'qqq': qqq_rets}).dropna()
    qqq_corr = aligned['strat'].corr(aligned['qqq']) if len(aligned) > 20 else np.nan

    # Monthly returns
    monthly = rets.resample('ME').apply(lambda x: (1+x).prod()-1)
    monthly_wr = (monthly > 0).mean()

    return {
        'strategy': name,
        'total_return_pct': round(total_ret * 100, 2),
        'cagr_pct': round(cagr * 100, 2),
        'ann_vol_pct': round(ann_vol * 100, 2),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'max_dd_pct': round(max_dd * 100, 2),
        'calmar': round(calmar, 3),
        'win_rate_daily': round(wr, 4),
        'profit_factor': round(pf, 3),
        'qqq_correlation': round(qqq_corr, 4) if not np.isnan(qqq_corr) else None,
        'monthly_win_rate': round(monthly_wr, 4),
        'n_days': len(rets),
        'final_equity': round(equity_curve.iloc[-1], 2),
    }


def regime_split(equity_curve, spy_rets):
    """Split performance by SPY regime (up/down/flat days)."""
    rets = equity_curve.pct_change().dropna()
    aligned = pd.DataFrame({'strat': rets, 'spy': spy_rets}).dropna()

    regimes = {}
    for label, mask in [
        ('bull', aligned['spy'] > 0.005),
        ('bear', aligned['spy'] < -0.005),
        ('flat', (aligned['spy'] >= -0.005) & (aligned['spy'] <= 0.005)),
    ]:
        sub = aligned.loc[mask, 'strat']
        if len(sub) > 5:
            regimes[label] = {
                'n_days': int(len(sub)),
                'mean_ret_bps': round(sub.mean() * 10000, 2),
                'sharpe': round(sub.mean() / sub.std() * np.sqrt(252), 3) if sub.std() > 0 else 0,
                'win_rate': round((sub > 0).mean(), 4),
            }
    return regimes


def permutation_test(equity_curve, n_perm=N_PERM):
    """Permutation test: shuffle daily returns, compute Sharpe distribution."""
    rets = equity_curve.pct_change().dropna().values
    actual_sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0

    perm_sharpes = []
    for _ in range(n_perm):
        shuffled = np.random.permutation(rets)
        s = shuffled.mean() / shuffled.std() * np.sqrt(252) if shuffled.std() > 0 else 0
        perm_sharpes.append(s)

    perm_sharpes = np.array(perm_sharpes)
    p_value = (perm_sharpes >= actual_sharpe).mean()
    return {
        'actual_sharpe': round(actual_sharpe, 4),
        'perm_mean_sharpe': round(perm_sharpes.mean(), 4),
        'perm_std_sharpe': round(perm_sharpes.std(), 4),
        'p_value': round(p_value, 4),
        'significant_5pct': bool(p_value < 0.05),
    }


def five_gate_validation(metrics, perm_result, regime_data):
    """5-gate framework validation."""
    gates = {}

    # Gate 1: Positive Sharpe
    gates['G1_positive_sharpe'] = {
        'pass': metrics.get('sharpe', 0) > 0,
        'value': metrics.get('sharpe', 0),
        'threshold': '> 0',
    }

    # Gate 2: Max DD < 25%
    gates['G2_max_dd'] = {
        'pass': metrics.get('max_dd_pct', -100) > -25,
        'value': metrics.get('max_dd_pct', -100),
        'threshold': '> -25%',
    }

    # Gate 3: Statistical significance (permutation p < 0.10)
    gates['G3_perm_test'] = {
        'pass': perm_result.get('p_value', 1) < 0.10,
        'value': perm_result.get('p_value', 1),
        'threshold': '< 0.10',
    }

    # Gate 4: Regime robustness (positive mean return in at least 2/3 regimes)
    pos_regimes = sum(1 for r in regime_data.values() if r.get('mean_ret_bps', 0) > 0)
    gates['G4_regime_robust'] = {
        'pass': pos_regimes >= 2,
        'value': f"{pos_regimes}/3 regimes positive",
        'threshold': '>= 2/3',
    }

    # Gate 5: QQQ correlation < 0.60 (diversification value)
    qqq_corr = metrics.get('qqq_correlation')
    if qqq_corr is not None:
        gates['G5_low_qqq_corr'] = {
            'pass': abs(qqq_corr) < 0.60,
            'value': qqq_corr,
            'threshold': '|corr| < 0.60',
        }
    else:
        gates['G5_low_qqq_corr'] = {'pass': False, 'value': None, 'threshold': '|corr| < 0.60'}

    gates['all_passed'] = all(g['pass'] for g in gates.values() if isinstance(g, dict) and 'pass' in g)
    gates['gates_passed'] = sum(1 for g in gates.values() if isinstance(g, dict) and g.get('pass', False))
    gates['gates_total'] = 5
    return gates


# ──────────────────────────────────────────────
# Strategy Implementations
# ──────────────────────────────────────────────

def strategy_A_gld_vol_target(close, returns, oot_dates):
    """A: GLD Vol-Target 10%. Always long GLD, size to target 10% annualized vol."""
    target_vol = 0.10
    lookback = 20
    gld_ret = returns['GLD']
    equity = pd.Series(index=oot_dates, dtype=float)
    cash = CAPITAL
    position_pct = 1.0  # start at 100%

    rebal_dates = oot_dates[::5]  # weekly

    for i, date in enumerate(oot_dates):
        if i == 0:
            equity.iloc[i] = CAPITAL
            continue

        # Daily return
        r = gld_ret.loc[date] if date in gld_ret.index and not np.isnan(gld_ret.loc[date]) else 0.0
        daily_pnl = equity.iloc[i-1] * position_pct * (r - SLIPPAGE_PCT * (1 if date in rebal_dates else 0))
        equity.iloc[i] = equity.iloc[i-1] + daily_pnl

        # Rebalance weekly
        if date in rebal_dates:
            hist = gld_ret.loc[:date].tail(lookback)
            if len(hist) >= lookback:
                realized_vol = hist.std() * np.sqrt(252)
                if realized_vol > 0:
                    position_pct = target_vol / realized_vol
                    position_pct = np.clip(position_pct, 0.30, 1.50)

    return equity.dropna()


def strategy_B_multi_asset_vol_target(close, returns, oot_dates):
    """B: Multi-Asset Vol-Target. GLD+TLT+UUP equal weight, scale to 8% portfolio vol."""
    target_vol = 0.08
    lookback = 20
    assets = ['GLD', 'TLT', 'UUP']
    port_ret = returns[assets].mean(axis=1)  # equal weight

    equity = pd.Series(index=oot_dates, dtype=float)
    position_pct = 1.0
    rebal_dates = oot_dates[::5]

    for i, date in enumerate(oot_dates):
        if i == 0:
            equity.iloc[i] = CAPITAL
            continue

        r = port_ret.loc[date] if date in port_ret.index and not np.isnan(port_ret.loc[date]) else 0.0
        daily_pnl = equity.iloc[i-1] * position_pct * (r - SLIPPAGE_PCT * (1 if date in rebal_dates else 0))
        equity.iloc[i] = equity.iloc[i-1] + daily_pnl

        if date in rebal_dates:
            hist = port_ret.loc[:date].tail(lookback)
            if len(hist) >= lookback:
                realized_vol = hist.std() * np.sqrt(252)
                if realized_vol > 0:
                    position_pct = target_vol / realized_vol
                    position_pct = np.clip(position_pct, 0.30, 1.50)

    return equity.dropna()


def strategy_C_inverse_vix_gld(close, returns, oot_dates):
    """C: Inverse VIX Sizing (GLD). Size GLD inversely proportional to VIX level."""
    gld_ret = returns['GLD']
    vix = close['VIX']
    equity = pd.Series(index=oot_dates, dtype=float)

    def vix_alloc(v):
        if v < 15: return 1.50
        elif v < 20: return 1.00
        elif v < 30: return 0.60
        else: return 0.30

    prev_alloc = 1.0

    for i, date in enumerate(oot_dates):
        if i == 0:
            equity.iloc[i] = CAPITAL
            if date in vix.index and not np.isnan(vix.loc[date]):
                prev_alloc = vix_alloc(vix.loc[date])
            continue

        r = gld_ret.loc[date] if date in gld_ret.index and not np.isnan(gld_ret.loc[date]) else 0.0

        # Check VIX for threshold crossings
        if date in vix.index and not np.isnan(vix.loc[date]):
            new_alloc = vix_alloc(vix.loc[date])
            slippage = SLIPPAGE_PCT if new_alloc != prev_alloc else 0.0
            prev_alloc = new_alloc
        else:
            slippage = 0.0

        daily_pnl = equity.iloc[i-1] * prev_alloc * (r - slippage)
        equity.iloc[i] = equity.iloc[i-1] + daily_pnl

    return equity.dropna()


def strategy_D_risk_parity(close, returns, oot_dates):
    """D: Risk Parity (GLD/TLT/UUP Equal Risk). Volatility-weighted allocation."""
    assets = ['GLD', 'TLT', 'UUP']
    lookback = 60
    equity = pd.Series(index=oot_dates, dtype=float)
    weights = {a: 1.0/3 for a in assets}

    # Monthly rebalance
    rebal_months = set()
    rebal_dates = []
    for d in oot_dates:
        key = (d.year, d.month)
        if key not in rebal_months:
            rebal_months.add(key)
            rebal_dates.append(d)

    for i, date in enumerate(oot_dates):
        if i == 0:
            equity.iloc[i] = CAPITAL
            continue

        port_ret = sum(weights[a] * (returns[a].loc[date] if date in returns[a].index and not np.isnan(returns[a].loc[date]) else 0.0) for a in assets)
        slippage = SLIPPAGE_PCT if date in rebal_dates else 0.0
        daily_pnl = equity.iloc[i-1] * (port_ret - slippage)
        equity.iloc[i] = equity.iloc[i-1] + daily_pnl

        if date in rebal_dates:
            vols = {}
            for a in assets:
                hist = returns[a].loc[:date].tail(lookback)
                vols[a] = hist.std() * np.sqrt(252) if len(hist) >= lookback else 0.10

            inv_vol = {a: 1.0/max(v, 0.01) for a, v in vols.items()}
            total_inv = sum(inv_vol.values())
            weights = {a: inv_vol[a] / total_inv for a in assets}

    return equity.dropna()


def strategy_E_momentum_vol_filter(close, returns, oot_dates):
    """E: Momentum + Vol Filter (GLD). Long when 3-month momentum > 0, vol-targeted sizing."""
    target_vol = 0.10
    lookback_vol = 20
    lookback_mom = 63  # ~3 months
    gld_ret = returns['GLD']
    gld_close = close['GLD']
    equity = pd.Series(index=oot_dates, dtype=float)
    position_pct = 0.0
    rebal_dates = oot_dates[::5]

    for i, date in enumerate(oot_dates):
        if i == 0:
            equity.iloc[i] = CAPITAL
            continue

        r = gld_ret.loc[date] if date in gld_ret.index and not np.isnan(gld_ret.loc[date]) else 0.0
        daily_pnl = equity.iloc[i-1] * position_pct * (r - SLIPPAGE_PCT * (1 if date in rebal_dates else 0))
        equity.iloc[i] = equity.iloc[i-1] + daily_pnl

        if date in rebal_dates:
            # Check momentum
            hist_close = gld_close.loc[:date].tail(lookback_mom + 1)
            if len(hist_close) >= lookback_mom + 1:
                mom = (hist_close.iloc[-1] / hist_close.iloc[0]) - 1
            else:
                mom = 0

            if mom > 0:
                hist_ret = gld_ret.loc[:date].tail(lookback_vol)
                if len(hist_ret) >= lookback_vol:
                    realized_vol = hist_ret.std() * np.sqrt(252)
                    if realized_vol > 0:
                        position_pct = target_vol / realized_vol
                        position_pct = np.clip(position_pct, 0.30, 1.50)
                    else:
                        position_pct = 1.0
                else:
                    position_pct = 1.0
            else:
                position_pct = 0.0  # Cash

    return equity.dropna()


def strategy_F_managed_vol_portfolio(close, returns, oot_dates):
    """F: Managed Volatility Portfolio. Equal-weight GLD+UUP with vol-based scaling."""
    assets = ['GLD', 'UUP']
    lookback = 20
    port_ret = returns[assets].mean(axis=1)
    equity = pd.Series(index=oot_dates, dtype=float)
    position_pct = 1.0
    rebal_dates = oot_dates[::5]

    for i, date in enumerate(oot_dates):
        if i == 0:
            equity.iloc[i] = CAPITAL
            continue

        r = port_ret.loc[date] if date in port_ret.index and not np.isnan(port_ret.loc[date]) else 0.0
        daily_pnl = equity.iloc[i-1] * position_pct * (r - SLIPPAGE_PCT * (1 if date in rebal_dates else 0))
        equity.iloc[i] = equity.iloc[i-1] + daily_pnl

        if date in rebal_dates:
            hist = port_ret.loc[:date].tail(lookback)
            if len(hist) >= lookback:
                realized_vol = hist.std() * np.sqrt(252)
                if realized_vol > 0.18:
                    position_pct = 0.25
                elif realized_vol > 0.12:
                    position_pct = 0.50
                elif realized_vol < 0.08:
                    position_pct = 1.25
                else:
                    position_pct = 1.0

    return equity.dropna()


# ──────────────────────────────────────────────
# Run All Strategies
# ──────────────────────────────────────────────

strategies = {
    'A_GLD_VolTarget_10pct': strategy_A_gld_vol_target,
    'B_MultiAsset_VolTarget_8pct': strategy_B_multi_asset_vol_target,
    'C_InverseVIX_GLD': strategy_C_inverse_vix_gld,
    'D_RiskParity_GLD_TLT_UUP': strategy_D_risk_parity,
    'E_Momentum_VolFilter_GLD': strategy_E_momentum_vol_filter,
    'F_ManagedVol_GLD_UUP': strategy_F_managed_vol_portfolio,
}

# QQQ and SPY returns for correlation / regime analysis
qqq_rets = returns['QQQ'].loc[oot_dates]
spy_rets = returns['SPY'].loc[oot_dates]

results = {
    'metadata': {
        'run_timestamp': datetime.now().isoformat(),
        'oot_start': OOT_START,
        'oot_end': OOT_END,
        'capital': CAPITAL,
        'slippage_pct': SLIPPAGE_PCT,
        'commission': COMMISSION,
        'n_permutations': N_PERM,
        'tickers': TICKERS,
        'n_oot_days': len(oot_dates),
    },
    'strategies': {},
    'benchmarks': {},
    'ranking': [],
}

# Benchmarks: buy-and-hold for each key asset
for ticker in ['SPY', 'QQQ', 'GLD', 'TLT', 'UUP', 'IWM', 'RSP']:
    if ticker in returns.columns:
        bench_rets = returns[ticker].loc[oot_dates].fillna(0)
        bench_eq = CAPITAL * (1 + bench_rets).cumprod()
        bench_eq.iloc[0] = CAPITAL
        m = calc_metrics(bench_eq, f'BH_{ticker}', qqq_rets)
        results['benchmarks'][ticker] = m
        print(f"  Benchmark {ticker}: Return={m.get('total_return_pct',0):.1f}%, Sharpe={m.get('sharpe',0):.3f}")

print("\n" + "="*70)
print("Running 6 volatility-targeting strategies...")
print("="*70)

for name, func in strategies.items():
    print(f"\n--- {name} ---")
    equity = func(close, returns, oot_dates)

    metrics = calc_metrics(equity, name, qqq_rets)
    regime = regime_split(equity, spy_rets)
    perm = permutation_test(equity)
    gates = five_gate_validation(metrics, perm, regime)

    results['strategies'][name] = {
        'metrics': metrics,
        'regime_split': regime,
        'permutation_test': perm,
        'five_gate': gates,
    }

    print(f"  Return: {metrics.get('total_return_pct',0):.1f}% | Sharpe: {metrics.get('sharpe',0):.3f} | "
          f"Sortino: {metrics.get('sortino',0):.3f} | MaxDD: {metrics.get('max_dd_pct',0):.1f}%")
    print(f"  QQQ Corr: {metrics.get('qqq_correlation','N/A')} | PF: {metrics.get('profit_factor',0):.3f} | "
          f"WR: {metrics.get('win_rate_daily',0):.4f}")
    print(f"  Perm p-val: {perm.get('p_value',1):.4f} | Gates: {gates.get('gates_passed',0)}/{gates.get('gates_total',5)}")
    print(f"  Regimes: {regime}")

# Ranking by Sharpe
ranking = []
for name, data in results['strategies'].items():
    m = data['metrics']
    ranking.append({
        'strategy': name,
        'sharpe': m.get('sharpe', 0),
        'sortino': m.get('sortino', 0),
        'total_return_pct': m.get('total_return_pct', 0),
        'max_dd_pct': m.get('max_dd_pct', 0),
        'qqq_correlation': m.get('qqq_correlation'),
        'gates_passed': data['five_gate'].get('gates_passed', 0),
        'perm_p_value': data['permutation_test'].get('p_value', 1),
    })
ranking.sort(key=lambda x: x['sharpe'], reverse=True)
results['ranking'] = ranking

# Summary
print("\n" + "="*70)
print("RANKING BY SHARPE")
print("="*70)
for i, r in enumerate(ranking):
    print(f"  {i+1}. {r['strategy']}: Sharpe={r['sharpe']:.3f}, Sortino={r['sortino']:.3f}, "
          f"Ret={r['total_return_pct']:.1f}%, DD={r['max_dd_pct']:.1f}%, "
          f"QQQ_Corr={r['qqq_correlation']}, Gates={r['gates_passed']}/5, p={r['perm_p_value']:.3f}")

# Verdict
best = ranking[0] if ranking else None
results['verdict'] = {
    'best_strategy': best['strategy'] if best else None,
    'best_sharpe': best['sharpe'] if best else None,
    'any_pass_all_gates': any(d['five_gate'].get('all_passed', False) for d in results['strategies'].values()),
    'strategies_passing_all_gates': [name for name, d in results['strategies'].items() if d['five_gate'].get('all_passed', False)],
    'conclusion': '',
}

passing = results['verdict']['strategies_passing_all_gates']
if passing:
    results['verdict']['conclusion'] = f"Vol-targeting shows promise: {', '.join(passing)} passed all 5 gates. Best diversified return stream with low QQQ correlation."
else:
    best_gates = max(ranking, key=lambda x: x['gates_passed']) if ranking else None
    results['verdict']['conclusion'] = f"No strategy passed all 5 gates. Best was {best_gates['strategy']} with {best_gates['gates_passed']}/5 gates. Vol-targeting alone doesn't produce alpha — it's a sizing mechanism, not a signal."

# Save
output_path = '/home/jupiter/Lvl3Quant/data/vol_targeting_results.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved to {output_path}")
print(f"\nVERDICT: {results['verdict']['conclusion']}")
