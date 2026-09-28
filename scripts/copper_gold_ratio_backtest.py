#!/usr/bin/env python3
"""
Copper-Gold Ratio Sector ETF Rotation Backtest
===============================================
Tests whether changes in the Cu/Au ratio predict sector ETF rotation
between cyclicals and defensives.

Hypotheses:
- Rising Cu/Au → cyclicals outperform (XLI, XLB, XLE, XLF, XLK, XLY)
- Falling Cu/Au → defensives outperform (XLU, XLP, XLV, XLRE)

5-Gate Validation:
  G1: Sharpe > 0.5
  G2: Permutation p < 0.05
  G3: Regime gap < 0.50
  G4: Trade count > 50
  G5: Max drawdown < 40%
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────────────────
CYCLICALS = ['XLI', 'XLB', 'XLE', 'XLF', 'XLK', 'XLY']
DEFENSIVES = ['XLU', 'XLP', 'XLV', 'XLRE']
ALL_SECTORS = CYCLICALS + DEFENSIVES + ['XLC']  # XLC is ambiguous
HOLD_DAYS = 5
COST_BPS = 10  # 0.10% round-trip
LOOKBACK = 252  # sliding window
ZSCORE_WINDOW = 60
ROC_WINDOWS = [5, 10]
FWD_HORIZONS = [1, 3, 5]
N_PERMUTATIONS = 1000
QUINTILES = 5
REGIME_GAP_THRESHOLD = 0.50
OUTPUT_DIR = Path('/home/jupiter/Lvl3Quant/output')
OUTPUT_FILE = OUTPUT_DIR / 'copper_gold_ratio_results.json'

np.random.seed(42)


def download_data():
    """Download 5 years of daily data via yfinance."""
    import yfinance as yf

    tickers = ALL_SECTORS + ['SPY', 'GLD']
    copper_tickers = ['HG=F', 'CPER']  # try futures first, fallback to ETF

    end = datetime.now()
    start = end.replace(year=end.year - 5)

    print(f"Downloading data from {start.date()} to {end.date()}...")

    # Download copper
    copper = None
    for ct in copper_tickers:
        try:
            d = yf.download(ct, start=start, end=end, progress=False)
            if d is not None and len(d) > 200:
                copper = d['Close'].squeeze()
                copper.name = 'Copper'
                print(f"  Copper: {ct} -> {len(copper)} days")
                break
        except Exception as e:
            print(f"  {ct} failed: {e}")

    if copper is None:
        raise RuntimeError("Could not download copper data")

    # Download everything else
    data = yf.download(tickers, start=start, end=end, progress=False)
    closes = data['Close']
    print(f"  Sector ETFs + SPY + GLD: {len(closes)} days, {len(closes.columns)} tickers")

    # Align dates
    df = closes.copy()
    df['Copper'] = copper
    df = df.dropna(how='all')
    df = df.ffill().dropna()
    print(f"  After alignment: {len(df)} days")
    return df


def compute_signals(df):
    """Compute Cu/Au ratio and its derivatives."""
    signals = pd.DataFrame(index=df.index)

    # Cu/Au ratio normalized to 100
    raw_ratio = df['Copper'] / df['GLD']
    signals['cu_au_ratio'] = raw_ratio / raw_ratio.iloc[0] * 100

    # Rate of change
    for w in ROC_WINDOWS:
        signals[f'roc_{w}d'] = signals['cu_au_ratio'].pct_change(w)

    # Z-score of 5d ROC (rolling 60d)
    roc5 = signals['roc_5d']
    signals['zscore_roc5'] = (
        (roc5 - roc5.rolling(ZSCORE_WINDOW).mean()) /
        roc5.rolling(ZSCORE_WINDOW).std()
    )

    # Z-score of 10d ROC
    roc10 = signals['roc_10d']
    signals['zscore_roc10'] = (
        (roc10 - roc10.rolling(ZSCORE_WINDOW).mean()) /
        roc10.rolling(ZSCORE_WINDOW).std()
    )

    return signals


def compute_forward_returns(df, sectors):
    """Compute forward returns for each sector at multiple horizons."""
    fwd = {}
    for h in FWD_HORIZONS:
        fwd_h = pd.DataFrame(index=df.index)
        for s in sectors:
            if s in df.columns:
                fwd_h[s] = df[s].pct_change(h).shift(-h)
        fwd[h] = fwd_h
    return fwd


def compute_momentum(df, sectors, window=20):
    """Compute momentum for each sector (for redundancy test)."""
    mom = pd.DataFrame(index=df.index)
    for s in sectors:
        if s in df.columns:
            mom[s] = df[s].pct_change(window)
    return mom


def information_coefficient(signal, returns):
    """Rank IC between signal and returns."""
    mask = signal.notna() & returns.notna()
    if mask.sum() < 30:
        return np.nan
    return signal[mask].rank().corr(returns[mask].rank(), method='spearman')


def permutation_test_ic(signal, returns, n_perm=N_PERMUTATIONS):
    """Permutation test for IC significance."""
    actual_ic = information_coefficient(signal, returns)
    if np.isnan(actual_ic):
        return actual_ic, 1.0

    mask = signal.notna() & returns.notna()
    sig_vals = signal[mask].values
    ret_vals = returns[mask].values
    n = len(sig_vals)

    count_extreme = 0
    for _ in range(n_perm):
        perm_idx = np.random.permutation(n)
        perm_sig = pd.Series(sig_vals[perm_idx])
        perm_ret = pd.Series(ret_vals)
        perm_ic = perm_sig.rank().corr(perm_ret.rank(), method='spearman')
        if abs(perm_ic) >= abs(actual_ic):
            count_extreme += 1

    p_value = count_extreme / n_perm
    return actual_ic, p_value


def quintile_analysis(signal, returns, label=""):
    """Sort signal into quintiles, measure forward returns per quintile."""
    mask = signal.notna() & returns.notna()
    s = signal[mask]
    r = returns[mask]
    if len(s) < 50:
        return None

    quintile_labels = pd.qcut(s, QUINTILES, labels=False, duplicates='drop')
    results = {}
    for q in sorted(quintile_labels.unique()):
        q_mask = quintile_labels == q
        q_rets = r[q_mask]
        results[int(q)] = {
            'mean_return_bps': round(q_rets.mean() * 10000, 2),
            'median_return_bps': round(q_rets.median() * 10000, 2),
            'count': int(q_mask.sum()),
            'hit_rate': round((q_rets > 0).mean() * 100, 1),
        }

    # Monotonicity: Q5 - Q1 spread
    q_keys = sorted(results.keys())
    if len(q_keys) >= 2:
        spread = results[q_keys[-1]]['mean_return_bps'] - results[q_keys[0]]['mean_return_bps']
        results['q5_q1_spread_bps'] = round(spread, 2)

    return results


def backtest_strategy(signal, sector_returns, direction, cost_bps=COST_BPS, hold=HOLD_DAYS):
    """
    Backtest a simple strategy:
    - direction='long_rising': go long when signal > 0
    - direction='long_falling': go long when signal < 0
    - direction='long_short': long when signal > 0, short when signal < 0

    Returns daily strategy returns series.
    """
    mask = signal.notna() & sector_returns.notna()
    sig = signal[mask]
    ret = sector_returns[mask]

    if len(sig) < 50:
        return None, {}

    # Position: +1 long, -1 short, 0 flat
    if direction == 'long_rising':
        pos = (sig > 0).astype(float)
    elif direction == 'long_falling':
        pos = (sig < 0).astype(float)
    elif direction == 'long_short':
        pos = np.sign(sig).astype(float)
    else:
        raise ValueError(f"Unknown direction: {direction}")

    # Costs on position changes
    pos_change = pos.diff().abs()
    costs = pos_change * (cost_bps / 10000)

    strat_returns = pos * ret - costs
    strat_returns = strat_returns.dropna()

    if len(strat_returns) < 50:
        return None, {}

    # Metrics
    ann_factor = 252 / hold
    mean_ret = strat_returns.mean()
    std_ret = strat_returns.std()
    sharpe = mean_ret / std_ret * np.sqrt(ann_factor) if std_ret > 0 else 0

    # Sortino
    downside = strat_returns[strat_returns < 0].std()
    sortino = mean_ret / downside * np.sqrt(ann_factor) if downside > 0 else 0

    # Drawdown
    cum = (1 + strat_returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = abs(dd.min()) * 100

    # Win rate
    trades = strat_returns[pos != 0]
    wr = (trades > 0).mean() * 100 if len(trades) > 0 else 0

    # Profit factor
    gross_profit = trades[trades > 0].sum()
    gross_loss = abs(trades[trades < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Trade count (position changes)
    n_trades = int(pos_change.sum())

    # Annual return
    total_days = len(strat_returns)
    total_return = cum.iloc[-1] - 1 if len(cum) > 0 else 0
    ann_return = (1 + total_return) ** (252 / total_days) - 1 if total_days > 0 else 0

    metrics = {
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'profit_factor': round(min(pf, 99), 2),
        'win_rate_pct': round(wr, 1),
        'max_drawdown_pct': round(max_dd, 1),
        'n_trades': n_trades,
        'ann_return_pct': round(ann_return * 100, 2),
        'total_return_pct': round(total_return * 100, 2),
        'n_days': total_days,
    }

    return strat_returns, metrics


def regime_stratify(signal, sector_returns, spy_returns, direction):
    """
    Stratify backtest by green/red SPY days.
    Returns per-regime Sharpe and the regime gap.
    """
    mask = signal.notna() & sector_returns.notna() & spy_returns.notna()
    sig = signal[mask]
    ret = sector_returns[mask]
    spy = spy_returns[mask]

    green_mask = spy > 0
    red_mask = spy <= 0

    results = {}
    for regime, rmask in [('green', green_mask), ('red', red_mask)]:
        if rmask.sum() < 20:
            results[regime] = {'sharpe': 0, 'n_days': 0}
            continue

        _, metrics = backtest_strategy(sig[rmask], ret[rmask], direction)
        if metrics:
            results[regime] = {
                'sharpe': metrics.get('sharpe', 0),
                'n_days': metrics.get('n_days', 0),
            }
        else:
            results[regime] = {'sharpe': 0, 'n_days': 0}

    # Regime gap
    sg = abs(results.get('green', {}).get('sharpe', 0))
    sr = abs(results.get('red', {}).get('sharpe', 0))
    denom = max(sg, sr)
    gap = abs(sg - sr) / denom if denom > 0 else 0
    results['regime_gap'] = round(gap, 3)
    results['pass'] = gap < REGIME_GAP_THRESHOLD

    return results


def five_gate_validation(metrics, perm_pvalue, regime_result):
    """Apply 5-gate validation."""
    gates = {
        'G1_sharpe_gt_0.5': metrics.get('sharpe', 0) > 0.5,
        'G2_perm_p_lt_0.05': perm_pvalue < 0.05,
        'G3_regime_gap_lt_0.50': regime_result.get('pass', False),
        'G4_trades_gt_50': metrics.get('n_trades', 0) > 50,
        'G5_max_dd_lt_40': metrics.get('max_drawdown_pct', 100) < 40,
    }
    gates['all_pass'] = all(gates.values())
    return gates


def partial_correlation_ic(cu_au_signal, momentum_signal, returns):
    """
    Compute IC of Cu/Au controlling for momentum.
    Uses residual method: regress Cu/Au on momentum, take residual, compute IC of residual vs returns.
    """
    mask = cu_au_signal.notna() & momentum_signal.notna() & returns.notna()
    if mask.sum() < 50:
        return np.nan

    cu = cu_au_signal[mask].values
    mom = momentum_signal[mask].values
    ret = returns[mask].values

    # Regress cu_au on momentum
    from numpy.linalg import lstsq
    A = np.column_stack([mom, np.ones(len(mom))])
    coef, _, _, _ = lstsq(A, cu, rcond=None)
    residual = cu - A @ coef

    # IC of residual vs returns
    res_rank = pd.Series(residual).rank()
    ret_rank = pd.Series(ret).rank()
    return round(res_rank.corr(ret_rank, method='spearman'), 4)


def main():
    print("=" * 70)
    print("COPPER-GOLD RATIO SECTOR ROTATION BACKTEST")
    print("=" * 70)

    # ── Download data ─────────────────────────────────────────────────
    df = download_data()

    # ── Compute signals ───────────────────────────────────────────────
    signals = compute_signals(df)
    print(f"\nSignals computed. Cu/Au ratio range: {signals['cu_au_ratio'].min():.1f} - {signals['cu_au_ratio'].max():.1f}")
    print(f"5d ROC range: {signals['roc_5d'].dropna().quantile(0.05):.4f} to {signals['roc_5d'].dropna().quantile(0.95):.4f}")

    # ── Forward returns ───────────────────────────────────────────────
    sectors_in_data = [s for s in ALL_SECTORS if s in df.columns]
    fwd_returns = compute_forward_returns(df, sectors_in_data)
    spy_daily_ret = df['SPY'].pct_change()
    momentum = compute_momentum(df, sectors_in_data, window=20)

    results = {
        'metadata': {
            'start_date': str(df.index[0].date()),
            'end_date': str(df.index[-1].date()),
            'n_days': len(df),
            'sectors': sectors_in_data,
            'cyclicals': [s for s in CYCLICALS if s in sectors_in_data],
            'defensives': [s for s in DEFENSIVES if s in sectors_in_data],
        },
        'per_sector_ic': {},
        'quintile_analysis': {},
        'strategy_variants': {},
        'redundancy_test': {},
        'gate_results': {},
    }

    # ── 1. Per-Sector IC Analysis ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("1. INFORMATION COEFFICIENT: Cu/Au 5d ROC vs Sector Forward Returns")
    print("=" * 70)

    signal_col = 'roc_5d'
    signal = signals[signal_col].copy()

    for horizon in FWD_HORIZONS:
        print(f"\n  Forward {horizon}d returns:")
        horizon_results = {}
        for sector in sectors_in_data:
            if sector not in fwd_returns[horizon].columns:
                continue
            ret = fwd_returns[horizon][sector]
            ic, p_val = permutation_test_ic(signal, ret)
            cat = 'CYCLICAL' if sector in CYCLICALS else ('DEFENSIVE' if sector in DEFENSIVES else 'AMBIG')
            sig_marker = '***' if p_val < 0.01 else ('**' if p_val < 0.05 else ('*' if p_val < 0.10 else ''))
            print(f"    {sector:5s} ({cat:9s}): IC={ic:+.4f}  p={p_val:.3f} {sig_marker}")
            horizon_results[sector] = {
                'ic': round(ic, 4) if not np.isnan(ic) else None,
                'p_value': round(p_val, 4),
                'category': cat,
                'significant_5pct': p_val < 0.05,
            }
        results['per_sector_ic'][f'fwd_{horizon}d'] = horizon_results

    # ── 2. Quintile Analysis ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("2. QUINTILE SORT: 5d Cu/Au ROC quintiles → Forward 5d Sector Returns")
    print("=" * 70)

    for sector in sectors_in_data:
        if sector not in fwd_returns[5].columns:
            continue
        ret = fwd_returns[5][sector]
        qa = quintile_analysis(signal, ret, label=sector)
        if qa:
            cat = 'CYCLICAL' if sector in CYCLICALS else ('DEFENSIVE' if sector in DEFENSIVES else 'AMBIG')
            spread = qa.get('q5_q1_spread_bps', 0)
            print(f"\n  {sector} ({cat}):")
            for q in range(QUINTILES):
                if q in qa:
                    d = qa[q]
                    print(f"    Q{q+1}: mean={d['mean_return_bps']:+6.1f}bps  hit={d['hit_rate']:.0f}%  n={d['count']}")
            print(f"    Q5-Q1 spread: {spread:+.1f} bps")
            results['quintile_analysis'][sector] = qa

    # ── 3. Strategy Variants ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("3. STRATEGY BACKTESTS")
    print("=" * 70)

    # Basket returns
    cyc_sectors = [s for s in CYCLICALS if s in df.columns]
    def_sectors = [s for s in DEFENSIVES if s in df.columns]

    basket_cyc_ret = df[cyc_sectors].pct_change().mean(axis=1)  # equal-weight basket
    basket_def_ret = df[def_sectors].pct_change().mean(axis=1)

    # Use 5d forward basket returns
    basket_cyc_fwd5 = basket_cyc_ret.rolling(5).sum().shift(-5)
    basket_def_fwd5 = basket_def_ret.rolling(5).sum().shift(-5)

    # Long-short: cyclicals - defensives
    ls_fwd5 = basket_cyc_fwd5 - basket_def_fwd5

    spy_fwd1 = df['SPY'].pct_change().shift(-1)

    variants = {
        'A_long_cyclicals_rising': (signal, basket_cyc_fwd5, 'long_rising',
                                     'Long cyclical basket when Cu/Au 5d ROC > 0'),
        'B_long_defensives_falling': (signal, basket_def_fwd5, 'long_falling',
                                       'Long defensive basket when Cu/Au 5d ROC < 0'),
        'C_long_short_rotation': (signal, ls_fwd5, 'long_short',
                                   'Long cyclicals vs short defensives based on Cu/Au direction'),
    }

    for variant_name, (sig, ret, direction, desc) in variants.items():
        print(f"\n  Variant {variant_name}: {desc}")

        strat_ret, metrics = backtest_strategy(sig, ret, direction)
        if strat_ret is None:
            print("    SKIP: insufficient data")
            continue

        # Permutation test on the strategy Sharpe
        ic, p_val = permutation_test_ic(sig, ret)

        # Regime stratification
        regime = regime_stratify(sig, ret, spy_daily_ret, direction)

        # 5-gate
        gates = five_gate_validation(metrics, p_val, regime)

        print(f"    Sharpe: {metrics['sharpe']:.3f}  Sortino: {metrics['sortino']:.3f}  "
              f"PF: {metrics['profit_factor']:.2f}  WR: {metrics['win_rate_pct']:.1f}%")
        print(f"    MaxDD: {metrics['max_drawdown_pct']:.1f}%  Trades: {metrics['n_trades']}  "
              f"Ann Ret: {metrics['ann_return_pct']:.2f}%")
        print(f"    IC: {ic:+.4f}  p={p_val:.3f}")
        print(f"    Regime: green Sharpe={regime.get('green',{}).get('sharpe',0):.3f}  "
              f"red Sharpe={regime.get('red',{}).get('sharpe',0):.3f}  "
              f"gap={regime.get('regime_gap',0):.3f}")

        gate_pass = "PASS" if gates['all_pass'] else "FAIL"
        failed = [g for g, v in gates.items() if not v and g != 'all_pass']
        print(f"    5-Gate: {gate_pass}  {'(failed: ' + ', '.join(failed) + ')' if failed else ''}")

        results['strategy_variants'][variant_name] = {
            'description': desc,
            'metrics': metrics,
            'ic': round(ic, 4) if not np.isnan(ic) else None,
            'p_value': round(p_val, 4),
            'regime': {k: v for k, v in regime.items() if k != 'pass'},
            'gates': gates,
        }

    # ── 4. Per-Sector Best Response ───────────────────────────────────
    print("\n" + "=" * 70)
    print("4. PER-SECTOR: Which sectors respond most to Cu/Au changes?")
    print("=" * 70)

    sector_response = []
    for sector in sectors_in_data:
        if sector not in fwd_returns[5].columns:
            continue
        ret = fwd_returns[5][sector]
        ic = information_coefficient(signal, ret)
        if np.isnan(ic):
            continue
        cat = 'CYCLICAL' if sector in CYCLICALS else ('DEFENSIVE' if sector in DEFENSIVES else 'AMBIG')
        sector_response.append((sector, cat, ic))

    sector_response.sort(key=lambda x: abs(x[2]), reverse=True)
    print(f"\n  Ranked by |IC| with Cu/Au 5d ROC → 5d forward return:")
    per_sector_rank = {}
    for sector, cat, ic in sector_response:
        direction = 'same' if ic > 0 else 'opposite'
        print(f"    {sector:5s} ({cat:9s}): IC={ic:+.4f}  (Cu/Au↑ → sector {direction})")
        per_sector_rank[sector] = {'ic': round(ic, 4), 'category': cat, 'direction': direction}

    results['per_sector_ranking'] = per_sector_rank

    # ── 5. Individual Sector Backtests ────────────────────────────────
    print("\n" + "=" * 70)
    print("5. INDIVIDUAL SECTOR BACKTESTS (Long when Cu/Au rising)")
    print("=" * 70)

    individual_results = {}
    for sector in sectors_in_data:
        if sector not in fwd_returns[5].columns:
            continue
        ret = fwd_returns[5][sector]
        cat = 'CYCLICAL' if sector in CYCLICALS else ('DEFENSIVE' if sector in DEFENSIVES else 'AMBIG')

        # For cyclicals: long when signal > 0; for defensives: long when signal < 0
        if sector in CYCLICALS:
            direction = 'long_rising'
        elif sector in DEFENSIVES:
            direction = 'long_falling'
        else:
            direction = 'long_rising'  # default

        strat_ret, metrics = backtest_strategy(signal, ret, direction)
        if strat_ret is None:
            continue

        ic, p_val = permutation_test_ic(signal, ret)
        regime = regime_stratify(signal, ret, spy_daily_ret, direction)
        gates = five_gate_validation(metrics, p_val, regime)

        gate_str = "PASS" if gates['all_pass'] else "FAIL"
        print(f"  {sector:5s} ({cat:9s}): Sharpe={metrics['sharpe']:+.3f}  "
              f"PF={metrics['profit_factor']:.2f}  WR={metrics['win_rate_pct']:.0f}%  "
              f"MaxDD={metrics['max_drawdown_pct']:.1f}%  IC={ic:+.4f}  p={p_val:.3f}  [{gate_str}]")

        individual_results[sector] = {
            'category': cat,
            'direction': direction,
            'metrics': metrics,
            'ic': round(ic, 4) if not np.isnan(ic) else None,
            'p_value': round(p_val, 4),
            'regime': {k: v for k, v in regime.items() if k != 'pass'},
            'gates': gates,
        }

    results['individual_sector_backtests'] = individual_results

    # ── 6. Redundancy Test: Cu/Au vs Momentum ─────────────────────────
    print("\n" + "=" * 70)
    print("6. REDUNDANCY TEST: Does Cu/Au add value beyond momentum?")
    print("=" * 70)

    for sector in sectors_in_data:
        if sector not in fwd_returns[5].columns or sector not in momentum.columns:
            continue
        ret = fwd_returns[5][sector]
        mom = momentum[sector]

        raw_ic = information_coefficient(signal, ret)
        mom_ic = information_coefficient(mom, ret)
        partial_ic = partial_correlation_ic(signal, mom, ret)

        cat = 'CYCLICAL' if sector in CYCLICALS else ('DEFENSIVE' if sector in DEFENSIVES else 'AMBIG')

        # Correlation between Cu/Au signal and momentum
        mask = signal.notna() & mom.notna()
        cu_mom_corr = signal[mask].corr(mom[mask]) if mask.sum() > 30 else np.nan

        print(f"  {sector:5s} ({cat:9s}): raw_IC={raw_ic:+.4f}  mom_IC={mom_ic:+.4f}  "
              f"partial_IC={partial_ic:+.4f}  Cu/Mom_corr={cu_mom_corr:+.3f}")

        results['redundancy_test'][sector] = {
            'raw_ic': round(raw_ic, 4) if not np.isnan(raw_ic) else None,
            'momentum_ic': round(mom_ic, 4) if not np.isnan(mom_ic) else None,
            'partial_ic_controlling_momentum': round(partial_ic, 4) if not np.isnan(partial_ic) else None,
            'cu_au_momentum_correlation': round(cu_mom_corr, 3) if not np.isnan(cu_mom_corr) else None,
            'adds_value': abs(partial_ic) > 0.02 if not np.isnan(partial_ic) else False,
        }

    # ── 7. Z-score Signal Variant ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("7. Z-SCORE SIGNAL VARIANT (zscore of 5d ROC)")
    print("=" * 70)

    zscore_signal = signals['zscore_roc5']

    for sector in sectors_in_data[:6]:  # top 6 for brevity
        if sector not in fwd_returns[5].columns:
            continue
        ret = fwd_returns[5][sector]
        ic = information_coefficient(zscore_signal, ret)
        cat = 'CYCLICAL' if sector in CYCLICALS else ('DEFENSIVE' if sector in DEFENSIVES else 'AMBIG')
        print(f"  {sector:5s} ({cat:9s}): zscore IC={ic:+.4f}  (vs raw ROC IC={information_coefficient(signal, ret):+.4f})")

    # ── 8. 10d ROC variant ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("8. 10d ROC VARIANT")
    print("=" * 70)

    roc10_signal = signals['roc_10d']
    for sector in sectors_in_data[:6]:
        if sector not in fwd_returns[5].columns:
            continue
        ret = fwd_returns[5][sector]
        ic_5d = information_coefficient(signal, ret)
        ic_10d = information_coefficient(roc10_signal, ret)
        cat = 'CYCLICAL' if sector in CYCLICALS else ('DEFENSIVE' if sector in DEFENSIVES else 'AMBIG')
        print(f"  {sector:5s} ({cat:9s}): 5d_ROC IC={ic_5d:+.4f}  10d_ROC IC={ic_10d:+.4f}")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Count passes
    variant_passes = sum(
        1 for v in results['strategy_variants'].values()
        if v.get('gates', {}).get('all_pass', False)
    )
    sector_passes = sum(
        1 for v in results['individual_sector_backtests'].values()
        if v.get('gates', {}).get('all_pass', False)
    )
    redundancy_adds = sum(
        1 for v in results['redundancy_test'].values()
        if v.get('adds_value', False)
    )

    summary = {
        'variant_strategies_tested': len(results['strategy_variants']),
        'variant_strategies_passed_all_gates': variant_passes,
        'individual_sectors_tested': len(results['individual_sector_backtests']),
        'individual_sectors_passed_all_gates': sector_passes,
        'sectors_where_cu_au_adds_value_beyond_momentum': redundancy_adds,
        'total_sectors_tested_for_redundancy': len(results['redundancy_test']),
    }
    results['summary'] = summary

    print(f"\n  Strategy variants: {variant_passes}/{len(results['strategy_variants'])} passed all 5 gates")
    print(f"  Individual sectors: {sector_passes}/{len(results['individual_sector_backtests'])} passed all 5 gates")
    print(f"  Cu/Au adds value beyond momentum: {redundancy_adds}/{len(results['redundancy_test'])} sectors")

    # Best strategy
    best_variant = None
    best_sharpe = -999
    for name, v in results['strategy_variants'].items():
        s = v.get('metrics', {}).get('sharpe', -999)
        if s > best_sharpe:
            best_sharpe = s
            best_variant = name

    if best_variant:
        bv = results['strategy_variants'][best_variant]
        print(f"\n  Best variant: {best_variant}")
        print(f"    Sharpe={bv['metrics']['sharpe']:.3f}  Sortino={bv['metrics']['sortino']:.3f}  "
              f"PF={bv['metrics']['profit_factor']:.2f}  WR={bv['metrics']['win_rate_pct']:.1f}%")
        print(f"    MaxDD={bv['metrics']['max_drawdown_pct']:.1f}%  AnnRet={bv['metrics']['ann_return_pct']:.2f}%")
        gate_status = "ALL GATES PASSED" if bv['gates']['all_pass'] else "FAILED GATES"
        print(f"    {gate_status}")

    # Overall verdict
    if variant_passes > 0:
        verdict = "SIGNAL HAS POTENTIAL - at least one variant passed all 5 gates"
    elif best_sharpe > 0.3:
        verdict = "MARGINAL - some predictive power but fails validation gates"
    else:
        verdict = "WEAK/NO SIGNAL - Cu/Au ratio does not reliably predict sector rotation"

    results['verdict'] = verdict
    print(f"\n  VERDICT: {verdict}")

    # ── Save ──────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Convert any numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return obj

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            r = convert(obj)
            if r is not obj:
                return r
            return super().default(obj)

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    print(f"\n  Results saved to {OUTPUT_FILE}")
    print("=" * 70)


if __name__ == '__main__':
    main()
