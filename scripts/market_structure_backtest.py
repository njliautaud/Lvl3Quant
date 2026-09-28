#!/usr/bin/env python3
"""
Market Structure / Options Sentiment Signal Backtest
6 variants testing structural (non-directional) signals for QQQ decorrelation.
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────────────────────
CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%
COMMISSION = 0.0
OOT_START = '2022-01-01'
OOT_END = '2026-07-29'
DATA_START = '2021-01-01'  # extra history for warmup
PERM_ITERS = 1000

TICKERS = ['SPY', 'QQQ', '^VIX', '^VIX3M', 'RSP', 'IWM', 'GLD', 'TLT', 'UUP',
           'XLE', 'XLF', 'XLV', 'XLK', 'XLU', 'XLP']

RESULTS_PATH = '/home/jupiter/Lvl3Quant/data/market_structure_results.json'


def download_data():
    """Download all needed data via yfinance."""
    print("Downloading data...")
    data = {}
    for t in TICKERS:
        try:
            df = yf.download(t, start=DATA_START, end='2026-07-30', progress=False, auto_adjust=True)
            if len(df) > 50:
                data[t] = df['Close'].squeeze()
                print(f"  {t}: {len(df)} rows")
            else:
                print(f"  {t}: insufficient data ({len(df)} rows)")
        except Exception as e:
            print(f"  {t}: FAILED - {e}")

    # If ^VIX3M unavailable, approximate from ^VIX
    if '^VIX3M' not in data and '^VIX' in data:
        print("  ^VIX3M unavailable, approximating from ^VIX (0.9x smoothed)")
        data['^VIX3M'] = data['^VIX'].rolling(5).mean() * 0.9

    prices = pd.DataFrame(data)
    prices = prices.ffill().dropna(how='all')
    return prices


def apply_costs(returns_series):
    """Apply slippage to trade-day returns."""
    # Slippage on rebalance days (when position changes)
    trades = returns_series.copy()
    return trades  # costs applied per-strategy at rebalance


def calc_sharpe(returns, ann=252):
    if len(returns) < 10 or returns.std() == 0:
        return 0.0
    return float(np.sqrt(ann) * returns.mean() / returns.std())


def calc_sortino(returns, ann=252):
    if len(returns) < 10:
        return 0.0
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 10.0  # all positive
    return float(np.sqrt(ann) * returns.mean() / downside.std())


def calc_max_dd(equity_curve):
    peak = equity_curve.cummax()
    dd = (equity_curve - peak) / peak
    return float(dd.min())


def calc_profit_factor(returns):
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    if losses == 0:
        return 10.0
    return float(gains / losses)


def calc_win_rate(returns):
    if len(returns) == 0:
        return 0.0
    return float((returns > 0).sum() / len(returns))


def permutation_test(returns, n_iter=PERM_ITERS):
    """Sign-flip permutation test: p-value for Sharpe ratio.
    Under H0 (no skill), the sign of each return is random.
    We randomly flip signs and compute Sharpe to build null distribution."""
    obs_sharpe = calc_sharpe(returns)
    if obs_sharpe <= 0:
        return 1.0
    count = 0
    r = returns.values if hasattr(returns, 'values') else np.array(returns)
    for _ in range(n_iter):
        signs = np.random.choice([-1, 1], size=len(r))
        perm = r * signs
        perm_sharpe = np.sqrt(252) * perm.mean() / (perm.std() + 1e-10)
        if perm_sharpe >= obs_sharpe:
            count += 1
    return count / n_iter


def regime_split(returns, spy_close):
    """Split returns by regime: SPY above/below 200-SMA."""
    sma200 = spy_close.rolling(200).mean()
    aligned = pd.DataFrame({'ret': returns, 'spy': spy_close, 'sma': sma200}).dropna()
    bull = aligned[aligned['spy'] > aligned['sma']]['ret']
    bear = aligned[aligned['spy'] <= aligned['sma']]['ret']
    return bull, bear


def count_trades(positions):
    """Count number of position changes."""
    changes = positions.diff().fillna(0)
    return int((changes != 0).sum())


def strategy_A(prices, oot_mask):
    """Equal-Weight vs Cap-Weight Signal."""
    if 'RSP' not in prices.columns or 'SPY' not in prices.columns or 'GLD' not in prices.columns:
        return None, "Missing RSP/SPY/GLD"

    ratio = prices['RSP'] / prices['SPY']
    ratio_20d = ratio.pct_change(20)

    # Build positions
    pos = pd.Series(index=prices.index, dtype=float)
    pos[:] = np.nan
    asset = pd.Series(index=prices.index, dtype=str)
    asset[:] = ''

    rebal_days = prices.index[prices.index >= oot_mask.index[oot_mask].min()]
    last_rebal = None
    for d in rebal_days:
        if d not in ratio_20d.index or pd.isna(ratio_20d.loc[d]):
            continue
        if last_rebal is not None and (d - last_rebal).days < 10:
            continue
        last_rebal = d
        if ratio_20d.loc[d] > 0:
            asset.loc[d] = 'RSP'
        else:
            asset.loc[d] = 'GLD'

    asset = asset.replace('', np.nan).ffill()

    # Build returns
    ret_rsp = prices['RSP'].pct_change()
    ret_gld = prices['GLD'].pct_change()
    rets = pd.Series(0.0, index=prices.index)
    for d in prices.index:
        if pd.isna(asset.loc[d]):
            continue
        if asset.loc[d] == 'RSP':
            rets.loc[d] = ret_rsp.loc[d] if not pd.isna(ret_rsp.loc[d]) else 0
        else:
            rets.loc[d] = ret_gld.loc[d] if not pd.isna(ret_gld.loc[d]) else 0

    # Apply slippage on rebalance days
    rebal_mask = asset.ne(asset.shift())
    rets[rebal_mask] -= SLIPPAGE_PCT

    return rets[oot_mask], "RSP/SPY ratio → RSP or GLD"


def strategy_B(prices, oot_mask):
    """Sector Breadth Signal."""
    sectors = ['XLE', 'XLF', 'XLV', 'XLK', 'XLU', 'XLP']
    available = [s for s in sectors if s in prices.columns]
    if len(available) < 4:
        return None, f"Only {len(available)} sectors available"

    # Count sectors above 50d SMA
    above_count = pd.Series(0, index=prices.index)
    for s in available:
        sma50 = prices[s].rolling(50).mean()
        above_count += (prices[s] > sma50).astype(int)

    # Determine position
    asset = pd.Series('', index=prices.index)
    # Weekly rebalance
    week_num = pd.Series(prices.index.isocalendar().week.values.astype(int), index=prices.index)
    rebal = (week_num != week_num.shift()).fillna(True)

    for d in prices.index:
        if not rebal.loc[d] and d != prices.index[0]:
            continue
        if above_count.loc[d] >= 5:
            asset.loc[d] = 'RSP'
        elif above_count.loc[d] <= 2:
            asset.loc[d] = 'GLD'
        else:
            asset.loc[d] = 'CASH'

    asset = asset.replace('', np.nan).ffill()

    ret_rsp = prices['RSP'].pct_change() if 'RSP' in prices.columns else pd.Series(0, index=prices.index)
    ret_gld = prices['GLD'].pct_change() if 'GLD' in prices.columns else pd.Series(0, index=prices.index)

    rets = pd.Series(0.0, index=prices.index)
    for d in prices.index:
        if pd.isna(asset.loc[d]) or asset.loc[d] == 'CASH':
            continue
        if asset.loc[d] == 'RSP':
            rets.loc[d] = ret_rsp.loc[d] if not pd.isna(ret_rsp.loc[d]) else 0
        elif asset.loc[d] == 'GLD':
            rets.loc[d] = ret_gld.loc[d] if not pd.isna(ret_gld.loc[d]) else 0

    rebal_mask = asset.ne(asset.shift())
    rets[rebal_mask] -= SLIPPAGE_PCT

    return rets[oot_mask], "Sector breadth → RSP/GLD/Cash"


def strategy_C(prices, oot_mask):
    """VIX Term Structure."""
    if '^VIX' not in prices.columns:
        return None, "Missing VIX"

    vix = prices['^VIX']
    if '^VIX3M' in prices.columns:
        vix3m = prices['^VIX3M']
    else:
        vix3m = vix.rolling(5).mean() * 0.9

    asset = pd.Series('', index=prices.index)
    for d in prices.index:
        if pd.isna(vix.loc[d]) or pd.isna(vix3m.loc[d]):
            continue
        if vix.loc[d] < vix3m.loc[d]:
            asset.loc[d] = 'SPY'  # contango → risk-on
        else:
            asset.loc[d] = 'TLT'  # backwardation → risk-off

    asset = asset.replace('', np.nan).ffill()

    ret_spy = prices['SPY'].pct_change()
    ret_tlt = prices['TLT'].pct_change() if 'TLT' in prices.columns else pd.Series(0, index=prices.index)

    rets = pd.Series(0.0, index=prices.index)
    for d in prices.index:
        if pd.isna(asset.loc[d]):
            continue
        if asset.loc[d] == 'SPY':
            rets.loc[d] = ret_spy.loc[d] if not pd.isna(ret_spy.loc[d]) else 0
        else:
            rets.loc[d] = ret_tlt.loc[d] if not pd.isna(ret_tlt.loc[d]) else 0

    rebal_mask = asset.ne(asset.shift())
    rets[rebal_mask] -= SLIPPAGE_PCT

    return rets[oot_mask], "VIX term structure → SPY or TLT"


def strategy_D(prices, oot_mask):
    """Small-Large Cap Spread."""
    if 'IWM' not in prices.columns or 'SPY' not in prices.columns or 'GLD' not in prices.columns:
        return None, "Missing IWM/SPY/GLD"

    ratio = prices['IWM'] / prices['SPY']
    ratio_20d = ratio.pct_change(20)

    asset = pd.Series('', index=prices.index)
    last_rebal = None
    for d in prices.index:
        if d not in ratio_20d.index or pd.isna(ratio_20d.loc[d]):
            continue
        if last_rebal is not None and (d - last_rebal).days < 10:
            continue
        last_rebal = d
        if ratio_20d.loc[d] > 0:
            asset.loc[d] = 'IWM'
        else:
            asset.loc[d] = 'GLD'

    asset = asset.replace('', np.nan).ffill()

    ret_iwm = prices['IWM'].pct_change()
    ret_gld = prices['GLD'].pct_change()

    rets = pd.Series(0.0, index=prices.index)
    for d in prices.index:
        if pd.isna(asset.loc[d]):
            continue
        if asset.loc[d] == 'IWM':
            rets.loc[d] = ret_iwm.loc[d] if not pd.isna(ret_iwm.loc[d]) else 0
        else:
            rets.loc[d] = ret_gld.loc[d] if not pd.isna(ret_gld.loc[d]) else 0

    rebal_mask = asset.ne(asset.shift())
    rets[rebal_mask] -= SLIPPAGE_PCT

    return rets[oot_mask], "IWM/SPY ratio → IWM or GLD"


def strategy_E(prices, oot_mask):
    """Multi-Asset Relative Strength (monthly rotation)."""
    assets = ['SPY', 'GLD', 'TLT', 'UUP']
    available = [a for a in assets if a in prices.columns]
    if len(available) < 3:
        return None, f"Only {len(available)} assets available"

    # 3-month momentum
    mom_3m = pd.DataFrame({a: prices[a].pct_change(63) for a in available})

    # Monthly rebalance
    month_num = prices.index.month
    month_change = pd.Series(month_num, index=prices.index) != pd.Series(month_num, index=prices.index).shift()

    chosen = pd.Series('', index=prices.index)
    for d in prices.index:
        if not month_change.loc[d]:
            continue
        if d not in mom_3m.index:
            continue
        row = mom_3m.loc[d].dropna()
        if len(row) == 0:
            continue
        best = row.idxmax()
        chosen.loc[d] = best

    chosen = chosen.replace('', np.nan).ffill()

    # Build returns
    daily_rets = {a: prices[a].pct_change() for a in available}
    rets = pd.Series(0.0, index=prices.index)
    for d in prices.index:
        if pd.isna(chosen.loc[d]):
            continue
        a = chosen.loc[d]
        if a in daily_rets:
            r = daily_rets[a].loc[d]
            rets.loc[d] = r if not pd.isna(r) else 0

    rebal_mask = chosen.ne(chosen.shift())
    rets[rebal_mask] -= SLIPPAGE_PCT

    return rets[oot_mask], "Monthly top-1 momentum from SPY/GLD/TLT/UUP"


def strategy_F(prices, oot_mask):
    """Structural Risk Score."""
    # Components
    rsp_spy_20d = (prices['RSP'] / prices['SPY']).pct_change(20) if 'RSP' in prices.columns and 'SPY' in prices.columns else None
    iwm_spy_20d = (prices['IWM'] / prices['SPY']).pct_change(20) if 'IWM' in prices.columns and 'SPY' in prices.columns else None
    vix = prices['^VIX'] if '^VIX' in prices.columns else None

    sectors = ['XLE', 'XLF', 'XLV', 'XLK', 'XLU', 'XLP']
    available_sectors = [s for s in sectors if s in prices.columns]

    above_count = pd.Series(0, index=prices.index)
    for s in available_sectors:
        sma50 = prices[s].rolling(50).mean()
        above_count += (prices[s] > sma50).astype(int)

    score = pd.Series(0, index=prices.index)
    if rsp_spy_20d is not None:
        score += (rsp_spy_20d > 0).astype(int)
    if iwm_spy_20d is not None:
        score += (iwm_spy_20d > 0).astype(int)
    if vix is not None:
        score += (vix < 20).astype(int)
    score += (above_count >= 4).astype(int)

    # Rebalance every 5 days
    asset = pd.Series('', index=prices.index)
    last_rebal = None
    for d in prices.index:
        if last_rebal is not None and (d - last_rebal).days < 5:
            continue
        last_rebal = d
        s = score.loc[d]
        if pd.isna(s):
            continue
        if s >= 3:
            asset.loc[d] = 'RSP'
        elif s == 0:
            asset.loc[d] = 'GLD'
        else:
            asset.loc[d] = 'UUP'

    asset = asset.replace('', np.nan).ffill()

    ret_rsp = prices['RSP'].pct_change() if 'RSP' in prices.columns else pd.Series(0, index=prices.index)
    ret_gld = prices['GLD'].pct_change() if 'GLD' in prices.columns else pd.Series(0, index=prices.index)
    ret_uup = prices['UUP'].pct_change() if 'UUP' in prices.columns else pd.Series(0, index=prices.index)

    rets = pd.Series(0.0, index=prices.index)
    for d in prices.index:
        if pd.isna(asset.loc[d]):
            continue
        if asset.loc[d] == 'RSP':
            rets.loc[d] = ret_rsp.loc[d] if not pd.isna(ret_rsp.loc[d]) else 0
        elif asset.loc[d] == 'GLD':
            rets.loc[d] = ret_gld.loc[d] if not pd.isna(ret_gld.loc[d]) else 0
        elif asset.loc[d] == 'UUP':
            rets.loc[d] = ret_uup.loc[d] if not pd.isna(ret_uup.loc[d]) else 0

    rebal_mask = asset.ne(asset.shift())
    rets[rebal_mask] -= SLIPPAGE_PCT

    return rets[oot_mask], "Structural risk score → RSP/GLD/UUP"


def evaluate_strategy(name, rets, prices, oot_mask):
    """Full evaluation with 5-gate framework."""
    if rets is None:
        return None

    rets = rets.dropna()
    if len(rets) < 20:
        return {'name': name, 'error': f'Only {len(rets)} returns', 'pass_all_gates': False}

    equity = (1 + rets).cumprod() * CAPITAL
    total_ret = float((equity.iloc[-1] / CAPITAL) - 1)
    ann_ret = float((1 + total_ret) ** (252 / len(rets)) - 1)

    sharpe = calc_sharpe(rets)
    sortino = calc_sortino(rets)
    max_dd = calc_max_dd(equity)
    pf = calc_profit_factor(rets)
    wr = calc_win_rate(rets)

    # Permutation test
    perm_p = permutation_test(rets)

    # Regime split
    spy_close = prices['SPY'].reindex(rets.index)
    bull_rets, bear_rets = regime_split(rets, spy_close)
    sharpe_bull = calc_sharpe(bull_rets) if len(bull_rets) > 20 else 0
    sharpe_bear = calc_sharpe(bear_rets) if len(bear_rets) > 20 else 0

    max_sharpe = max(abs(sharpe_bull), abs(sharpe_bear), 1e-10)
    regime_gap = abs(sharpe_bull - sharpe_bear) / max_sharpe

    # QQQ correlation
    qqq_rets = prices['QQQ'].pct_change().reindex(rets.index).dropna()
    common_idx = rets.index.intersection(qqq_rets.index)
    if len(common_idx) > 20:
        qqq_corr = float(rets.loc[common_idx].corr(qqq_rets.loc[common_idx]))
    else:
        qqq_corr = 0.0

    # Trade count
    n_trades = int((rets != 0).sum())

    # 5-gate validation
    g1 = sharpe > 0.5
    g2 = perm_p < 0.05
    g3 = regime_gap < 0.5
    g4 = max_dd > -0.50
    g5 = n_trades >= 20

    result = {
        'name': name,
        'total_return_pct': round(total_ret * 100, 2),
        'annual_return_pct': round(ann_ret * 100, 2),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'max_drawdown_pct': round(max_dd * 100, 2),
        'profit_factor': round(pf, 3),
        'win_rate': round(wr, 3),
        'num_trading_days': len(rets),
        'num_active_days': n_trades,
        'final_equity': round(float(equity.iloc[-1]), 2),
        'qqq_correlation': round(qqq_corr, 3),
        'permutation_p_value': round(perm_p, 4),
        'sharpe_bull_regime': round(sharpe_bull, 3),
        'sharpe_bear_regime': round(sharpe_bear, 3),
        'regime_gap': round(regime_gap, 3),
        'gates': {
            'sharpe_gt_0.5': g1,
            'perm_p_lt_0.05': g2,
            'regime_gap_lt_0.5': g3,
            'mdd_gt_neg50': g4,
            'trades_gte_20': g5,
        },
        'pass_all_gates': all([g1, g2, g3, g4, g5]),
        'gates_passed': sum([g1, g2, g3, g4, g5]),
    }

    return result


def main():
    np.random.seed(42)
    prices = download_data()

    oot_mask = pd.Series((prices.index >= OOT_START) & (prices.index <= OOT_END), index=prices.index)
    print(f"\nOOT period: {OOT_START} to {OOT_END}")
    print(f"OOT trading days: {oot_mask.sum()}")

    strategies = {
        'A_EqualWeight_vs_CapWeight': strategy_A,
        'B_Sector_Breadth': strategy_B,
        'C_VIX_Term_Structure': strategy_C,
        'D_SmallLarge_Cap_Spread': strategy_D,
        'E_MultiAsset_RelStrength': strategy_E,
        'F_Structural_Risk_Score': strategy_F,
    }

    # QQQ benchmark
    qqq_rets = prices['QQQ'].pct_change()[oot_mask].dropna()
    qqq_result = evaluate_strategy('QQQ_Benchmark', qqq_rets, prices, oot_mask)

    all_results = {'benchmark': qqq_result, 'strategies': {}, 'meta': {
        'capital': CAPITAL,
        'slippage_pct': SLIPPAGE_PCT,
        'commission': COMMISSION,
        'oot_start': OOT_START,
        'oot_end': OOT_END,
        'run_timestamp': datetime.now().isoformat(),
        'permutation_iterations': PERM_ITERS,
    }}

    for name, fn in strategies.items():
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        rets, desc = fn(prices, oot_mask)
        if rets is None:
            print(f"  SKIPPED: {desc}")
            all_results['strategies'][name] = {'error': desc, 'pass_all_gates': False}
            continue

        result = evaluate_strategy(name, rets, prices, oot_mask)
        result['description'] = desc
        all_results['strategies'][name] = result

        gates_str = f"{result['gates_passed']}/5"
        print(f"  Sharpe: {result['sharpe']:.3f} | Sortino: {result['sortino']:.3f} | "
              f"MDD: {result['max_drawdown_pct']:.1f}% | QQQ corr: {result['qqq_correlation']:.3f}")
        print(f"  Perm p: {result['permutation_p_value']:.4f} | Regime gap: {result['regime_gap']:.3f} | "
              f"Gates: {gates_str} | PASS: {result['pass_all_gates']}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    passing = [n for n, r in all_results['strategies'].items() if r.get('pass_all_gates')]
    print(f"Strategies passing all 5 gates: {len(passing)} / {len(strategies)}")
    for n in passing:
        r = all_results['strategies'][n]
        print(f"  ✓ {n}: Sharpe={r['sharpe']:.3f}, QQQ_corr={r['qqq_correlation']:.3f}")

    # Low QQQ correlation highlights
    low_corr = [(n, r) for n, r in all_results['strategies'].items()
                if isinstance(r.get('qqq_correlation'), (int, float)) and abs(r['qqq_correlation']) < 0.3]
    if low_corr:
        print(f"\nLow QQQ correlation (<0.3):")
        for n, r in sorted(low_corr, key=lambda x: abs(x[1]['qqq_correlation'])):
            print(f"  {n}: corr={r['qqq_correlation']:.3f}, Sharpe={r['sharpe']:.3f}")

    # Save
    with open(RESULTS_PATH, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")

    return all_results


if __name__ == '__main__':
    main()
