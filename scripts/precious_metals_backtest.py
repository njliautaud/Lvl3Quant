#!/usr/bin/env python3
"""
Precious Metals Timing/Ratio Backtest
Goal: Find strategies UNCORRELATED to QQQ (main signal Sharpe 2.38)
Gold/silver driven by real rates, USD, safe haven — different from equities.

6 Variants: A-F (see docstrings below)
OOT: 2022-01-01 to 2026-07-29
Capital: $645, Slippage: 0.02%, Commission: $0

5-Gate Validation:
  1. Sharpe > 0.5
  2. Permutation p-value < 0.05
  3. Regime gap < 0.5
  4. Max drawdown > -50%
  5. Trades >= 20
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

# ─── Config ───
CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%
COMMISSION = 0.0
OOT_START = '2022-01-01'
OOT_END = '2026-07-29'
TICKERS = ['GLD', 'SLV', 'GDX', 'GDXJ', 'IAU', 'SPY', 'QQQ', '^VIX', 'TLT', 'UUP']
N_PERMUTATIONS = 1000
RESULTS_PATH = Path('/home/jupiter/Lvl3Quant/data/precious_metals_results.json')


def download_data():
    """Download all required ETF data from yfinance."""
    print(f"Downloading {len(TICKERS)} tickers...")
    # Download with extra buffer for lookback
    start = '2021-01-01'
    data = {}
    for t in TICKERS:
        try:
            df = yf.download(t, start=start, end='2026-07-30', progress=False, auto_adjust=True)
            if len(df) > 50:
                data[t] = df['Close'].squeeze()
                print(f"  {t}: {len(df)} rows")
            else:
                print(f"  {t}: INSUFFICIENT DATA ({len(df)} rows)")
        except Exception as e:
            print(f"  {t}: FAILED - {e}")
    prices = pd.DataFrame(data)
    prices.index = pd.to_datetime(prices.index)
    # Remove timezone if present
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    prices = prices.ffill()
    return prices


def compute_returns(prices, ticker):
    """Daily returns for a ticker."""
    return prices[ticker].pct_change()


def backtest_strategy(prices, signals, trade_ticker, hold_days, name):
    """
    Generic backtest engine.
    signals: Series of 1 (long) or 0 (cash), indexed by date.
    trade_ticker: which ETF to go long when signal=1.
    hold_days: minimum hold period.
    Returns dict of metrics.
    """
    oot_mask = (prices.index >= OOT_START) & (prices.index <= OOT_END)
    px = prices.loc[oot_mask, trade_ticker].copy()
    sig = signals.reindex(px.index).fillna(0).astype(int)

    daily_ret = px.pct_change().fillna(0)

    # Build position series with hold period enforcement
    position = pd.Series(0, index=px.index)
    hold_until = pd.Timestamp('1900-01-01')

    for i, dt in enumerate(px.index):
        if dt >= hold_until:
            if sig.iloc[i] == 1:
                position.iloc[i] = 1
                hold_until = dt + pd.Timedelta(days=hold_days)
            else:
                position.iloc[i] = 0
        else:
            position.iloc[i] = position.iloc[i-1] if i > 0 else 0

    # Strategy returns (apply slippage on position changes)
    pos_changes = position.diff().fillna(0).abs()
    strat_ret = position.shift(1).fillna(0) * daily_ret - pos_changes * SLIPPAGE_PCT

    # Equity curve
    equity = CAPITAL * (1 + strat_ret).cumprod()

    # Metrics
    total_return = (equity.iloc[-1] / CAPITAL) - 1
    n_years = len(px) / 252
    ann_return = (1 + total_return) ** (1 / max(n_years, 0.5)) - 1

    daily_std = strat_ret.std()
    ann_vol = daily_std * np.sqrt(252) if daily_std > 0 else 0.001
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    # Sortino
    downside = strat_ret[strat_ret < 0].std() * np.sqrt(252) if (strat_ret < 0).any() else 0.001
    sortino = ann_return / downside if downside > 0 else 0

    # Max drawdown
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_dd = drawdown.min()

    # Win rate and trade count
    trade_starts = (position.diff().fillna(0) == 1)
    n_trades = trade_starts.sum()

    # Per-trade P&L
    trade_pnls = []
    in_trade = False
    entry_equity = 0
    for i in range(len(position)):
        if not in_trade and position.iloc[i] == 1:
            in_trade = True
            entry_equity = equity.iloc[i-1] if i > 0 else CAPITAL
        elif in_trade and position.iloc[i] == 0:
            in_trade = False
            exit_equity = equity.iloc[i]
            trade_pnls.append(exit_equity - entry_equity)
    if in_trade:
        trade_pnls.append(equity.iloc[-1] - entry_equity)

    win_rate = sum(1 for p in trade_pnls if p > 0) / max(len(trade_pnls), 1)
    profit_factor = (
        sum(p for p in trade_pnls if p > 0) / abs(sum(p for p in trade_pnls if p < 0))
        if any(p < 0 for p in trade_pnls) and any(p > 0 for p in trade_pnls)
        else float('inf') if all(p >= 0 for p in trade_pnls) and any(p > 0 for p in trade_pnls)
        else 0
    )

    # QQQ correlation
    qqq_ret = compute_returns(prices, 'QQQ').reindex(strat_ret.index).fillna(0)
    qqq_corr = strat_ret.corr(qqq_ret) if strat_ret.std() > 0 else 0

    # SPY correlation
    spy_ret = compute_returns(prices, 'SPY').reindex(strat_ret.index).fillna(0)
    spy_corr = strat_ret.corr(spy_ret) if strat_ret.std() > 0 else 0

    return {
        'position': position,  # for permutation test
        'underlying_returns': daily_ret,  # for permutation test
        'name': name,
        'trade_ticker': trade_ticker,
        'total_return_pct': round(total_return * 100, 2),
        'ann_return_pct': round(ann_return * 100, 2),
        'ann_vol_pct': round(ann_vol * 100, 2),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'max_drawdown_pct': round(max_dd * 100, 2),
        'n_trades': int(n_trades),
        'win_rate': round(win_rate, 3),
        'profit_factor': round(profit_factor, 3) if profit_factor != float('inf') else 999.0,
        'qqq_correlation': round(qqq_corr, 3),
        'spy_correlation': round(spy_corr, 3),
        'final_equity': round(float(equity.iloc[-1]), 2),
        'equity_curve': {str(d.date()): round(float(v), 2) for d, v in equity.iloc[::20].items()},  # sampled
        'strat_returns': strat_ret,  # kept for permutation test, removed before JSON
    }


def permutation_test(position_series, underlying_returns, n_perms=N_PERMUTATIONS):
    """
    Permutation test: shuffle the position (signal) assignments across dates.
    This tests whether the TIMING of the signal matters, not just being long.
    Returns p-value: fraction of random timings that achieve >= actual Sharpe.
    """
    # Actual strategy returns
    actual_ret = position_series.shift(1).fillna(0) * underlying_returns
    actual_sharpe = actual_ret.mean() / actual_ret.std() * np.sqrt(252) if actual_ret.std() > 0 else 0

    pos_arr = position_series.values.copy()
    ret_arr = underlying_returns.values
    count_better = 0

    for _ in range(n_perms):
        shuffled_pos = np.random.permutation(pos_arr)
        perm_ret = np.roll(shuffled_pos, 1) * ret_arr  # shift(1) equivalent
        perm_ret[0] = 0
        std = perm_ret.std()
        s = perm_ret.mean() / std * np.sqrt(252) if std > 0 else 0
        if s >= actual_sharpe:
            count_better += 1
    return round(count_better / n_perms, 4)


def regime_gap(strat_returns, qqq_returns):
    """
    Compute regime gap: |Sharpe_up - Sharpe_down| / max(|Sharpe_up|, |Sharpe_down|).
    Regimes: QQQ 20d rolling return > 0 = up, else down.
    """
    qqq_20d = qqq_returns.rolling(20).sum()
    up_mask = qqq_20d > 0
    down_mask = qqq_20d <= 0

    sr_up = strat_returns[up_mask]
    sr_down = strat_returns[down_mask]

    sharpe_up = sr_up.mean() / sr_up.std() * np.sqrt(252) if len(sr_up) > 20 and sr_up.std() > 0 else 0
    sharpe_down = sr_down.mean() / sr_down.std() * np.sqrt(252) if len(sr_down) > 20 and sr_down.std() > 0 else 0

    denom = max(abs(sharpe_up), abs(sharpe_down), 0.001)
    gap = abs(sharpe_up - sharpe_down) / denom

    return round(gap, 3), round(sharpe_up, 3), round(sharpe_down, 3)


# ─── Strategy Signal Generators ───

def strategy_A_gold_silver_ratio(prices):
    """A: Gold-Silver Ratio Mean Reversion.
    Uses z-score of GLD/SLV ratio (rolling 60d).
    Z > 1.0 (silver cheap) → long SLV (reversion).
    Z < -1.0 (gold cheap) → long GLD.
    Maps user's 80/70 concept to ETF price ratio space.
    """
    oot_mask = (prices.index >= OOT_START) & (prices.index <= OOT_END)
    ratio = prices['GLD'] / prices['SLV']

    # Use z-score for adaptive thresholds
    ratio_mean = ratio.rolling(60).mean()
    ratio_std = ratio.rolling(60).std()
    z = (ratio - ratio_mean) / ratio_std.replace(0, np.nan)

    # Two sub-signals
    sig_slv = (z > 1.0).astype(int)   # silver cheap → long SLV
    sig_gld = (z < -1.0).astype(int)  # gold cheap → long GLD

    # When ratio > 80, trade SLV. When < 70, trade GLD. 70-80 = cash.
    # We'll run both and combine returns manually
    return sig_slv, sig_gld, ratio


def strategy_B_gold_momentum_vix(prices):
    """B: Gold Momentum + VIX. Long GLD when 20d ret > 0 AND (VIX>20 OR UUP down)."""
    gld_20d = prices['GLD'].pct_change(20)
    vix_high = prices['^VIX'] > 20
    uup_down = prices['UUP'].pct_change(20) < 0

    signal = ((gld_20d > 0) & (vix_high | uup_down)).astype(int)
    return signal


def strategy_C_miner_leverage(prices):
    """C: Miner Leverage. Long GDX when GLD 10d return > 1%."""
    gld_10d = prices['GLD'].pct_change(10)
    signal = (gld_10d > 0.01).astype(int)
    return signal


def strategy_D_gold_safe_haven(prices):
    """D: Gold Safe Haven. SPY drops >3% in 10d → long GLD. SPY up >3% → cash."""
    spy_10d = prices['SPY'].pct_change(10)
    signal = (spy_10d < -0.03).astype(int)
    return signal


def strategy_E_real_rate_proxy(prices):
    """E: Real Rate Proxy. TLT/UUP rising → long GLD."""
    ratio = prices['TLT'] / prices['UUP']
    ratio_mom = ratio.pct_change(20)
    signal = (ratio_mom > 0).astype(int)
    return signal


def strategy_F_multi_metal_score(prices):
    """F: Multi-Metal Score. Composite of 4 signals, score >= 2 → long GLD."""
    # Component 1: GLD 1m momentum > 0
    gld_mom = (prices['GLD'].pct_change(21) > 0).astype(int)

    # Component 2: GLD/SLV ratio z-score > 1 (silver underperforming → gold strength)
    _ratio = prices['GLD'] / prices['SLV']
    _rmean = _ratio.rolling(60).mean()
    _rstd = _ratio.rolling(60).std().replace(0, np.nan)
    ratio_sig = ((_ratio - _rmean) / _rstd > 1.0).astype(int)

    # Component 3: VIX > 20
    vix_flag = (prices['^VIX'] > 20).astype(int)

    # Component 4: UUP downtrend (20d return < 0)
    uup_down = (prices['UUP'].pct_change(20) < 0).astype(int)

    score = gld_mom + ratio_sig + vix_flag + uup_down

    # Score >= 2 → long, score <= 0 → cash, otherwise hold previous
    signal = pd.Series(0, index=prices.index)
    prev = 0
    for i in range(len(score)):
        s = score.iloc[i]
        if s >= 2:
            prev = 1
        elif s <= 0:
            prev = 0
        signal.iloc[i] = prev

    return signal


def validate_5gate(result, perm_p, r_gap):
    """5-gate validation framework."""
    gates = {
        'sharpe_gt_0.5': bool(result['sharpe'] > 0.5),
        'perm_p_lt_0.05': bool(perm_p < 0.05),
        'regime_gap_lt_0.5': bool(r_gap < 0.5),
        'mdd_gt_neg50': bool(result['max_drawdown_pct'] > -50),
        'trades_gte_20': bool(result['n_trades'] >= 20),
    }
    gates['gates_passed'] = sum(1 for k, v in gates.items() if v)
    gates['all_passed'] = (gates['gates_passed'] == 5)
    return gates


def run_all():
    """Run all 6 strategy variants and validate."""
    np.random.seed(42)
    prices = download_data()

    oot_mask = (prices.index >= OOT_START) & (prices.index <= OOT_END)
    qqq_ret = compute_returns(prices, 'QQQ').loc[oot_mask].fillna(0)

    results = {}

    # ─── A: Gold-Silver Ratio (composite of SLV and GLD sub-signals) ───
    print("\n=== Strategy A: Gold-Silver Ratio ===")
    sig_slv, sig_gld, ratio = strategy_A_gold_silver_ratio(prices)

    # Run SLV leg
    res_slv = backtest_strategy(prices, sig_slv, 'SLV', hold_days=20, name='A_GoldSilverRatio_SLV')
    # Run GLD leg
    res_gld = backtest_strategy(prices, sig_gld, 'GLD', hold_days=20, name='A_GoldSilverRatio_GLD')

    # Combine: use the SLV leg returns when SLV signal on, GLD leg when GLD signal on
    combined_ret = res_slv['strat_returns'].copy()
    # Where GLD signal is on and SLV is off, use GLD returns
    gld_only = (sig_gld.reindex(combined_ret.index).fillna(0) == 1) & (sig_slv.reindex(combined_ret.index).fillna(0) == 0)
    gld_strat_ret = res_gld['strat_returns']
    combined_ret[gld_only] = gld_strat_ret[gld_only]

    equity_a = CAPITAL * (1 + combined_ret).cumprod()
    total_ret_a = (equity_a.iloc[-1] / CAPITAL) - 1
    n_years = len(equity_a) / 252
    ann_ret_a = (1 + total_ret_a) ** (1/max(n_years,0.5)) - 1
    vol_a = combined_ret.std() * np.sqrt(252)
    sharpe_a = ann_ret_a / vol_a if vol_a > 0 else 0
    down_a = combined_ret[combined_ret<0].std() * np.sqrt(252) if (combined_ret<0).any() else 0.001
    sortino_a = ann_ret_a / down_a
    dd_a = ((equity_a - equity_a.cummax()) / equity_a.cummax()).min()
    qqq_corr_a = combined_ret.corr(qqq_ret)
    spy_ret_a = compute_returns(prices, 'SPY').reindex(combined_ret.index).fillna(0)
    spy_corr_a = combined_ret.corr(spy_ret_a)
    n_trades_a = res_slv['n_trades'] + res_gld['n_trades']

    result_a = {
        'name': 'A_GoldSilverRatio_Combined',
        'trade_ticker': 'SLV+GLD',
        'total_return_pct': round(total_ret_a*100, 2),
        'ann_return_pct': round(ann_ret_a*100, 2),
        'ann_vol_pct': round(vol_a*100, 2),
        'sharpe': round(sharpe_a, 3),
        'sortino': round(sortino_a, 3),
        'max_drawdown_pct': round(dd_a*100, 2),
        'n_trades': int(n_trades_a),
        'win_rate': round((res_slv['win_rate'] + res_gld['win_rate'])/2, 3),
        'profit_factor': round((res_slv['profit_factor'] + res_gld['profit_factor'])/2, 3),
        'qqq_correlation': round(float(qqq_corr_a), 3),
        'spy_correlation': round(float(spy_corr_a), 3),
        'final_equity': round(float(equity_a.iloc[-1]), 2),
        'ratio_stats': {
            'mean': round(float(ratio.loc[oot_mask].mean()), 2),
            'min': round(float(ratio.loc[oot_mask].min()), 2),
            'max': round(float(ratio.loc[oot_mask].max()), 2),
        }
    }
    # Build combined position for permutation test
    combined_pos = res_slv['position'].copy()
    combined_pos[gld_only] = res_gld['position'][gld_only]
    # Use GLD returns as proxy underlying (dominant leg)
    combined_underlying = compute_returns(prices, 'GLD').reindex(combined_ret.index).fillna(0)
    perm_p_a = permutation_test(combined_pos, combined_underlying)
    gap_a, su_a, sd_a = regime_gap(combined_ret, qqq_ret)
    result_a['perm_p_value'] = perm_p_a
    result_a['regime_gap'] = gap_a
    result_a['sharpe_up_regime'] = su_a
    result_a['sharpe_down_regime'] = sd_a
    result_a['validation'] = validate_5gate(result_a, perm_p_a, gap_a)
    results['A'] = result_a
    print(f"  Sharpe={result_a['sharpe']}, QQQ_corr={result_a['qqq_correlation']}, Trades={result_a['n_trades']}")

    # ─── B: Gold Momentum + VIX ───
    print("\n=== Strategy B: Gold Momentum + VIX ===")
    sig_b = strategy_B_gold_momentum_vix(prices)
    res_b = backtest_strategy(prices, sig_b, 'GLD', hold_days=15, name='B_GoldMomentumVIX')
    perm_p_b = permutation_test(res_b['position'], res_b['underlying_returns'])
    gap_b, su_b, sd_b = regime_gap(res_b['strat_returns'], qqq_ret)
    for k in ['strat_returns', 'position', 'underlying_returns']: res_b.pop(k, None)
    res_b['perm_p_value'] = perm_p_b
    res_b['regime_gap'] = gap_b
    res_b['sharpe_up_regime'] = su_b
    res_b['sharpe_down_regime'] = sd_b
    res_b['validation'] = validate_5gate(res_b, perm_p_b, gap_b)
    results['B'] = res_b
    print(f"  Sharpe={res_b['sharpe']}, QQQ_corr={res_b['qqq_correlation']}, Trades={res_b['n_trades']}")

    # ─── C: Miner Leverage ───
    print("\n=== Strategy C: Miner Leverage ===")
    sig_c = strategy_C_miner_leverage(prices)
    res_c = backtest_strategy(prices, sig_c, 'GDX', hold_days=10, name='C_MinerLeverage')
    perm_p_c = permutation_test(res_c['position'], res_c['underlying_returns'])
    gap_c, su_c, sd_c = regime_gap(res_c['strat_returns'], qqq_ret)
    for k in ['strat_returns', 'position', 'underlying_returns']: res_c.pop(k, None)
    res_c['perm_p_value'] = perm_p_c
    res_c['regime_gap'] = gap_c
    res_c['sharpe_up_regime'] = su_c
    res_c['sharpe_down_regime'] = sd_c
    res_c['validation'] = validate_5gate(res_c, perm_p_c, gap_c)
    results['C'] = res_c
    print(f"  Sharpe={res_c['sharpe']}, QQQ_corr={res_c['qqq_correlation']}, Trades={res_c['n_trades']}")

    # ─── D: Gold Safe Haven ───
    print("\n=== Strategy D: Gold Safe Haven ===")
    sig_d = strategy_D_gold_safe_haven(prices)
    res_d = backtest_strategy(prices, sig_d, 'GLD', hold_days=10, name='D_GoldSafeHaven')
    perm_p_d = permutation_test(res_d['position'], res_d['underlying_returns'])
    gap_d, su_d, sd_d = regime_gap(res_d['strat_returns'], qqq_ret)
    for k in ['strat_returns', 'position', 'underlying_returns']: res_d.pop(k, None)
    res_d['perm_p_value'] = perm_p_d
    res_d['regime_gap'] = gap_d
    res_d['sharpe_up_regime'] = su_d
    res_d['sharpe_down_regime'] = sd_d
    res_d['validation'] = validate_5gate(res_d, perm_p_d, gap_d)
    results['D'] = res_d
    print(f"  Sharpe={res_d['sharpe']}, QQQ_corr={res_d['qqq_correlation']}, Trades={res_d['n_trades']}")

    # ─── E: Real Rate Proxy ───
    print("\n=== Strategy E: Real Rate Proxy ===")
    sig_e = strategy_E_real_rate_proxy(prices)
    res_e = backtest_strategy(prices, sig_e, 'GLD', hold_days=15, name='E_RealRateProxy')
    perm_p_e = permutation_test(res_e['position'], res_e['underlying_returns'])
    gap_e, su_e, sd_e = regime_gap(res_e['strat_returns'], qqq_ret)
    for k in ['strat_returns', 'position', 'underlying_returns']: res_e.pop(k, None)
    res_e['perm_p_value'] = perm_p_e
    res_e['regime_gap'] = gap_e
    res_e['sharpe_up_regime'] = su_e
    res_e['sharpe_down_regime'] = sd_e
    res_e['validation'] = validate_5gate(res_e, perm_p_e, gap_e)
    results['E'] = res_e
    print(f"  Sharpe={res_e['sharpe']}, QQQ_corr={res_e['qqq_correlation']}, Trades={res_e['n_trades']}")

    # ─── F: Multi-Metal Score ───
    print("\n=== Strategy F: Multi-Metal Score ===")
    sig_f = strategy_F_multi_metal_score(prices)
    res_f = backtest_strategy(prices, sig_f, 'GLD', hold_days=10, name='F_MultiMetalScore')
    perm_p_f = permutation_test(res_f['position'], res_f['underlying_returns'])
    gap_f, su_f, sd_f = regime_gap(res_f['strat_returns'], qqq_ret)
    for k in ['strat_returns', 'position', 'underlying_returns']: res_f.pop(k, None)
    res_f['perm_p_value'] = perm_p_f
    res_f['regime_gap'] = gap_f
    res_f['sharpe_up_regime'] = su_f
    res_f['sharpe_down_regime'] = sd_f
    res_f['validation'] = validate_5gate(res_f, perm_p_f, gap_f)
    results['F'] = res_f
    print(f"  Sharpe={res_f['sharpe']}, QQQ_corr={res_f['qqq_correlation']}, Trades={res_f['n_trades']}")

    # ─── Summary ───
    print("\n" + "="*80)
    print("PRECIOUS METALS BACKTEST SUMMARY")
    print("="*80)
    print(f"{'Variant':<35} {'Sharpe':>7} {'Sortino':>8} {'QQQ_r':>7} {'MDD%':>7} {'Trades':>7} {'Gates':>6}")
    print("-"*80)
    for k in sorted(results.keys()):
        r = results[k]
        v = r['validation']
        print(f"{r['name']:<35} {r['sharpe']:>7.3f} {r['sortino']:>8.3f} {r['qqq_correlation']:>7.3f} "
              f"{r['max_drawdown_pct']:>7.2f} {r['n_trades']:>7d} {v['gates_passed']:>4d}/5")

    # ─── Rank by uncorrelated Sharpe ───
    print("\n--- RANKED BY |QQQ_corr| (lower = more uncorrelated) ---")
    ranked = sorted(results.items(), key=lambda x: abs(x[1]['qqq_correlation']))
    for k, r in ranked:
        marker = " *** PASS ***" if r['validation']['all_passed'] else ""
        print(f"  {r['name']}: QQQ_corr={r['qqq_correlation']:+.3f}, Sharpe={r['sharpe']:.3f}{marker}")

    # ─── Build output ───
    output = {
        'metadata': {
            'run_date': datetime.now().isoformat(),
            'oot_start': OOT_START,
            'oot_end': OOT_END,
            'capital': CAPITAL,
            'slippage_pct': SLIPPAGE_PCT,
            'commission': COMMISSION,
            'n_permutations': N_PERMUTATIONS,
            'purpose': 'Find precious metals strategies uncorrelated to QQQ (Sharpe 2.38)',
        },
        'strategies': results,
        'summary': {
            'best_sharpe': max(results.items(), key=lambda x: x[1]['sharpe'])[1]['name'],
            'best_sharpe_value': max(r['sharpe'] for r in results.values()),
            'lowest_qqq_corr': min(results.items(), key=lambda x: abs(x[1]['qqq_correlation']))[1]['name'],
            'lowest_qqq_corr_value': min(abs(r['qqq_correlation']) for r in results.values()),
            'passed_5gate': [r['name'] for r in results.values() if r['validation']['all_passed']],
            'n_passed': sum(1 for r in results.values() if r['validation']['all_passed']),
        }
    }

    # Save
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")

    return output


if __name__ == '__main__':
    run_all()
