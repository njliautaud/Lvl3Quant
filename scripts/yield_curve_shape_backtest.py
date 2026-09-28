#!/usr/bin/env python3
"""
Yield Curve Shape Changes -> Sector ETF Rotation Backtest
=========================================================
Signal: 5-day rate of change in 2s10s and 3m10s yield curve spreads
        predicts 1d/3d/5d forward returns in sector ETFs.

Hypotheses:
  - 2s10s steepening -> XLF, XLRE outperform (banks benefit)
  - 2s10s flattening -> XLU, XLP outperform (defensive rotation)
  - 3m10s changes capture Fed policy expectations

Validation gates:
  1. Sharpe > 0.5
  2. Permutation p-value < 0.05
  3. Regime gap < 0.50
  4. Trade count > 50
  5. Max drawdown < 40%

Window: 252d sliding (never expanding -- HC #0)
Cost:   0.10% round-trip
"""

import json
import warnings
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings('ignore')

# -- CONFIG ---------------------------------------------------------------
SECTOR_ETFS = ['XLE', 'XLU', 'XLP', 'XLF', 'XLK', 'XLC', 'XLV', 'XLY', 'XLRE', 'XLB', 'XLI']
CYCLICAL_SECTORS = ['XLF', 'XLRE', 'XLI', 'XLE', 'XLB']
DEFENSIVE_SECTORS = ['XLU', 'XLP', 'XLV']
LOOKBACK = 252
CURVE_CHANGE_DAYS = 5
HOLD_PERIODS = [1, 3, 5]
COST_RT_PCT = 0.10 / 100  # 0.10% round-trip
N_PERMUTATIONS = 1000
START_DATE = '2019-01-01'
END_DATE = '2026-08-15'
REGIME_GAP_THRESHOLD = 0.50
OUTPUT_PATH = Path('/home/jupiter/Lvl3Quant/output/yield_curve_shape_results.json')


def download_treasury_data():
    """Download treasury yield data. Try pandas_datareader/FRED first, fall back to yfinance."""
    print("Downloading treasury yield data...")
    try:
        import pandas_datareader.data as web
        series = {'DGS2': 'DGS2', 'DGS10': 'DGS10', 'DGS3MO': 'DGS3MO'}
        dfs = {}
        for name, code in series.items():
            df = web.DataReader(code, 'fred', START_DATE, END_DATE)
            dfs[name] = df[code]
        yields_df = pd.DataFrame(dfs)
        yields_df = yields_df.apply(pd.to_numeric, errors='coerce')
        yields_df = yields_df.dropna()
        if len(yields_df) > 500:
            print(f"  FRED data: {len(yields_df)} rows ({yields_df.index[0].date()} to {yields_df.index[-1].date()})")
            return yields_df
        print(f"  FRED returned only {len(yields_df)} rows, falling back to yfinance...")
    except Exception as e:
        print(f"  FRED failed: {e}, falling back to yfinance...")

    print("  Using yfinance treasury proxies...")
    tnx = yf.download('^TNX', start=START_DATE, end=END_DATE, progress=False)['Close']
    irx = yf.download('^IRX', start=START_DATE, end=END_DATE, progress=False)['Close']
    fvx = yf.download('^FVX', start=START_DATE, end=END_DATE, progress=False)['Close']

    # Flatten multi-level columns
    for s in [tnx, irx, fvx]:
        if isinstance(s.index, pd.MultiIndex):
            pass
        if hasattr(s, 'columns'):
            s = s.squeeze()

    yields_df = pd.DataFrame({
        'DGS10': tnx.squeeze() / 10.0,
        'DGS3MO': irx.squeeze() / 100.0,
        'DGS2': (irx.squeeze() / 100.0 + fvx.squeeze() / 10.0) / 2
    }).dropna()

    print(f"  yfinance data: {len(yields_df)} rows ({yields_df.index[0].date()} to {yields_df.index[-1].date()})")
    return yields_df


def download_etf_data():
    """Download sector ETF + SPY data."""
    print("Downloading sector ETF data...")
    tickers = SECTOR_ETFS + ['SPY']
    data = yf.download(tickers, start=START_DATE, end=END_DATE, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        close = data['Close']
    else:
        close = data

    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(0)

    close = close.dropna()
    print(f"  ETF data: {len(close)} rows, {len(close.columns)} tickers")
    return close


def compute_curve_signals(yields_df):
    """Compute yield curve spread changes."""
    print("Computing yield curve signals...")
    spread_2s10s = (yields_df['DGS10'] - yields_df['DGS2']) * 100  # bps
    spread_3m10s = (yields_df['DGS10'] - yields_df['DGS3MO']) * 100

    chg_2s10s_5d = spread_2s10s.diff(CURVE_CHANGE_DAYS)
    chg_3m10s_5d = spread_3m10s.diff(CURVE_CHANGE_DAYS)

    signals = pd.DataFrame({
        'spread_2s10s': spread_2s10s,
        'spread_3m10s': spread_3m10s,
        'chg_2s10s_5d': chg_2s10s_5d,
        'chg_3m10s_5d': chg_3m10s_5d,
    })

    print(f"  2s10s spread: mean={spread_2s10s.mean():.1f}bps, std={spread_2s10s.std():.1f}bps")
    print(f"  3m10s spread: mean={spread_3m10s.mean():.1f}bps, std={spread_3m10s.std():.1f}bps")
    return signals


def rolling_percentile(series, window=252):
    """Compute rolling percentile rank of each value within its trailing window."""
    arr = series.values.astype(float)
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(window, n):
        w = arr[i-window:i+1]
        valid = w[~np.isnan(w)]
        if len(valid) < 50:
            continue
        result[i] = stats.percentileofscore(valid, arr[i], kind='rank') / 100.0
    return pd.Series(result, index=series.index)


def vectorized_backtest(signal_pctile, etf_fwd_returns, hold_period):
    """
    Vectorized quintile-sort backtest.
    Top quintile signal (steepening) -> long cyclicals, short defensives.
    Bottom quintile signal (flattening) -> long defensives, short cyclicals.
    """
    # Identify trade days
    top_q = signal_pctile >= 0.80
    bot_q = signal_pctile <= 0.20

    cyclical_cols = [c for c in CYCLICAL_SECTORS if c in etf_fwd_returns.columns]
    defensive_cols = [c for c in DEFENSIVE_SECTORS if c in etf_fwd_returns.columns]

    cyclical_ret = etf_fwd_returns[cyclical_cols].mean(axis=1)
    defensive_ret = etf_fwd_returns[defensive_cols].mean(axis=1)

    # Top quintile: long cyclicals, short defensives
    top_pnl = (cyclical_ret - defensive_ret) / 2 - COST_RT_PCT

    # Bottom quintile: long defensives, short cyclicals
    bot_pnl = (defensive_ret - cyclical_ret) / 2 - COST_RT_PCT

    # Combine: take signals from both tails
    strat = pd.Series(0.0, index=signal_pctile.index)
    strat[top_q] = top_pnl[top_q]
    strat[bot_q] = bot_pnl[bot_q]

    # Only keep days with actual trades
    trade_mask = top_q | bot_q
    strat_active = strat[trade_mask].dropna()

    return strat_active, int(trade_mask.sum())


def compute_sharpe(returns, annualize=True):
    if len(returns) < 10 or returns.std() == 0:
        return 0.0
    sr = returns.mean() / returns.std()
    if annualize:
        sr *= np.sqrt(252)
    return sr


def compute_sortino(returns, annualize=True):
    if len(returns) < 10:
        return 0.0
    downside = returns[returns < 0]
    if len(downside) == 0:
        return compute_sharpe(returns, annualize)
    downside_std = np.sqrt((downside ** 2).mean())
    if downside_std == 0:
        return 0.0
    s = returns.mean() / downside_std
    if annualize:
        s *= np.sqrt(252)
    return s


def compute_max_drawdown(returns):
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return dd.min()


def permutation_test_fast(signal_pctile, etf_fwd_returns, hold_period, observed_sharpe, n_perms=N_PERMUTATIONS):
    """
    Fast permutation test: shuffle the percentile-trade mapping.
    Instead of recomputing rolling percentile (slow), we shuffle which days get
    top/bottom quintile classification. This tests whether the timing of trades matters.
    """
    print(f"    Running {n_perms} permutations...")
    rng = np.random.RandomState(42)
    perm_sharpes = np.zeros(n_perms)

    # Pre-compute the L/S returns for all days (not just quintile days)
    cyclical_cols = [c for c in CYCLICAL_SECTORS if c in etf_fwd_returns.columns]
    defensive_cols = [c for c in DEFENSIVE_SECTORS if c in etf_fwd_returns.columns]
    cyclical_ret = etf_fwd_returns[cyclical_cols].mean(axis=1)
    defensive_ret = etf_fwd_returns[defensive_cols].mean(axis=1)
    ls_ret = ((cyclical_ret - defensive_ret) / 2).dropna()

    # Get the actual trade signals: +1 for top quintile, -1 for bottom, 0 for no trade
    valid_pctile = signal_pctile.reindex(ls_ret.index).dropna()
    common = valid_pctile.index.intersection(ls_ret.index)
    valid_pctile = valid_pctile.reindex(common)
    ls_aligned = ls_ret.reindex(common)

    positions = np.zeros(len(common))
    positions[valid_pctile.values >= 0.80] = 1.0   # steepening -> long cyclicals
    positions[valid_pctile.values <= 0.20] = -1.0   # flattening -> long defensives

    trade_mask = positions != 0
    n_trade_days = trade_mask.sum()
    if n_trade_days < 20:
        return 1.0

    for p in range(n_perms):
        # Shuffle positions (which days are top/bottom quintile)
        shuf_pos = positions.copy()
        rng.shuffle(shuf_pos)
        shuf_mask = shuf_pos != 0

        pnl = shuf_pos[shuf_mask] * ls_aligned.values[shuf_mask] - COST_RT_PCT
        if len(pnl) > 10 and pnl.std() > 0:
            perm_sharpes[p] = (pnl.mean() / pnl.std()) * np.sqrt(252)

        if (p + 1) % 250 == 0:
            print(f"      {p+1}/{n_perms} done")

    p_value = np.mean(perm_sharpes >= observed_sharpe)
    return float(p_value)


def regime_stratify(strategy_returns, spy_close):
    spy_daily = spy_close.pct_change()
    common = strategy_returns.index.intersection(spy_daily.index)
    if len(common) < 20:
        return None, None, None

    strat = strategy_returns.reindex(common).dropna()
    spy = spy_daily.reindex(strat.index)

    green = strat[spy > 0]
    red = strat[spy <= 0]

    sharpe_green = compute_sharpe(green) if len(green) > 10 else 0.0
    sharpe_red = compute_sharpe(red) if len(red) > 10 else 0.0

    denom = max(abs(sharpe_green), abs(sharpe_red), 1e-6)
    gap = abs(sharpe_green - sharpe_red) / denom

    return sharpe_green, sharpe_red, gap


def per_sector_analysis(signals, etf_close, signal_col_name, hold_period):
    """Compute IC between signal and forward return for each sector."""
    fwd_rets = etf_close[SECTOR_ETFS].pct_change(hold_period).shift(-hold_period)
    sig = signals[signal_col_name].dropna()

    results = {}
    for sector in SECTOR_ETFS:
        if sector not in fwd_rets.columns:
            continue
        common = sig.index.intersection(fwd_rets[sector].dropna().index)
        if len(common) < 50:
            results[sector] = {'ic': 0, 'pval': 1, 'n': len(common)}
            continue

        ic, pval = stats.spearmanr(sig.reindex(common), fwd_rets[sector].reindex(common))
        results[sector] = {
            'ic': round(float(ic), 4),
            'pval': round(float(pval), 4),
            'n': int(len(common))
        }
    return results


def main():
    print("=" * 70)
    print("YIELD CURVE SHAPE CHANGES -> SECTOR ETF ROTATION BACKTEST")
    print("=" * 70)
    print()

    # -- DATA --
    yields_df = download_treasury_data()
    etf_close = download_etf_data()

    common_idx = yields_df.index.intersection(etf_close.index)
    yields_df = yields_df.reindex(common_idx)
    etf_close = etf_close.reindex(common_idx)
    spy_close = etf_close['SPY'] if 'SPY' in etf_close.columns else None

    print(f"\nAligned data: {len(common_idx)} trading days "
          f"({common_idx[0].date()} to {common_idx[-1].date()})")

    # -- SIGNALS --
    signals = compute_curve_signals(yields_df)

    # -- PER-SECTOR IC --
    print("\n" + "=" * 70)
    print("PER-SECTOR INFORMATION COEFFICIENT (Spearman IC)")
    print("=" * 70)

    all_results = {}

    for signal_name in ['chg_2s10s_5d', 'chg_3m10s_5d']:
        print(f"\n-- Signal: {signal_name} --")
        for hp in HOLD_PERIODS:
            sector_ics = per_sector_analysis(signals, etf_close, signal_name, hp)
            print(f"\n  Forward {hp}d returns:")
            for sector in SECTOR_ETFS:
                if sector in sector_ics:
                    r = sector_ics[sector]
                    sig_str = "***" if r['pval'] < 0.01 else "** " if r['pval'] < 0.05 else "*  " if r['pval'] < 0.10 else "   "
                    print(f"    {sector:5s}: IC={r['ic']:+.4f}  p={r['pval']:.4f} {sig_str}  n={r['n']}")

            key = f"{signal_name}_fwd{hp}d"
            all_results[key] = sector_ics

    # -- PORTFOLIO BACKTEST --
    print("\n" + "=" * 70)
    print("PORTFOLIO BACKTEST: Long/Short Quintile Sort")
    print("=" * 70)

    backtest_results = {}

    for signal_name in ['chg_2s10s_5d', 'chg_3m10s_5d']:
        sig = signals[signal_name]

        # Pre-compute rolling percentile once per signal
        print(f"\n  Computing rolling percentile for {signal_name}...")
        sig_pctile = rolling_percentile(sig, LOOKBACK)

        for hp in HOLD_PERIODS:
            print(f"\n-- Signal: {signal_name}, Hold: {hp}d --")

            etf_fwd = etf_close[SECTOR_ETFS].pct_change(hp).shift(-hp)
            strat_returns, n_trades = vectorized_backtest(sig_pctile, etf_fwd, hp)

            if len(strat_returns) < 20:
                print(f"  Too few trades ({n_trades}), skipping.")
                continue

            sharpe = compute_sharpe(strat_returns)
            sortino_val = compute_sortino(strat_returns)
            max_dd = compute_max_drawdown(strat_returns)
            cum_ret = (1 + strat_returns).prod() - 1
            win_rate = (strat_returns > 0).mean()
            neg_sum = abs(strat_returns[strat_returns < 0].sum())
            pf = strat_returns[strat_returns > 0].sum() / neg_sum if neg_sum > 0 else 999.0

            print(f"  Trades: {n_trades}")
            print(f"  Sharpe: {sharpe:.3f}")
            print(f"  Sortino: {sortino_val:.3f}")
            print(f"  Cum Return: {cum_ret:.2%}")
            print(f"  Max DD: {max_dd:.2%}")
            print(f"  Win Rate: {win_rate:.1%}")
            print(f"  Profit Factor: {pf:.2f}")

            # -- GATE CHECKS --
            print(f"\n  5-GATE VALIDATION:")

            g1 = sharpe > 0.5
            print(f"    Gate 1 (Sharpe > 0.5):       {sharpe:.3f} -> {'PASS' if g1 else 'FAIL'}")

            p_value = permutation_test_fast(sig_pctile, etf_fwd, hp, sharpe)
            g2 = p_value < 0.05
            print(f"    Gate 2 (Perm p < 0.05):      p={p_value:.4f} -> {'PASS' if g2 else 'FAIL'}")

            if spy_close is not None:
                sharpe_green, sharpe_red, regime_gap = regime_stratify(strat_returns, spy_close)
                g3 = regime_gap is not None and regime_gap < REGIME_GAP_THRESHOLD
                if regime_gap is not None:
                    print(f"    Gate 3 (Regime gap < 0.50):  gap={regime_gap:.3f} "
                          f"(green={sharpe_green:.2f}, red={sharpe_red:.2f}) -> {'PASS' if g3 else 'FAIL'}")
                else:
                    print(f"    Gate 3 (Regime gap < 0.50):  INSUFFICIENT DATA -> FAIL")
                    g3 = False
            else:
                sharpe_green, sharpe_red, regime_gap = None, None, None
                g3 = False
                print(f"    Gate 3 (Regime gap < 0.50):  NO SPY DATA -> FAIL")

            g4 = n_trades > 50
            print(f"    Gate 4 (Trades > 50):        {n_trades} -> {'PASS' if g4 else 'FAIL'}")

            g5 = max_dd > -0.40
            print(f"    Gate 5 (MaxDD < 40%):        {max_dd:.2%} -> {'PASS' if g5 else 'FAIL'}")

            all_pass = g1 and g2 and g3 and g4 and g5
            gates_passed = sum([g1, g2, g3, g4, g5])
            verdict = "PASS ALL GATES" if all_pass else f"FAIL ({gates_passed}/5 passed)"
            print(f"\n  == VERDICT: {verdict} ==")

            key = f"{signal_name}_hold{hp}d"
            backtest_results[key] = {
                'signal': signal_name,
                'hold_period': hp,
                'n_trades': n_trades,
                'sharpe': round(float(sharpe), 4),
                'sortino': round(float(sortino_val), 4),
                'cum_return': round(float(cum_ret), 4),
                'max_drawdown': round(float(max_dd), 4),
                'win_rate': round(float(win_rate), 4),
                'profit_factor': round(float(pf), 4) if pf != float('inf') else 999.0,
                'perm_p_value': round(float(p_value), 4),
                'regime_gap': round(float(regime_gap), 4) if regime_gap is not None else None,
                'sharpe_green': round(float(sharpe_green), 4) if sharpe_green is not None else None,
                'sharpe_red': round(float(sharpe_red), 4) if sharpe_red is not None else None,
                'gate_1_sharpe': g1,
                'gate_2_permutation': g2,
                'gate_3_regime': g3,
                'gate_4_trades': g4,
                'gate_5_drawdown': g5,
                'all_gates_pass': all_pass,
                'gates_passed': gates_passed,
            }

    # -- SUMMARY --
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)

    passing = [k for k, v in backtest_results.items() if v.get('all_gates_pass')]
    if passing:
        print(f"\nPASSING CONFIGS ({len(passing)}):")
        for k in passing:
            r = backtest_results[k]
            print(f"  {k}: Sharpe={r['sharpe']:.3f}, Sortino={r['sortino']:.3f}, "
                  f"WR={r['win_rate']:.1%}, Trades={r['n_trades']}")
    else:
        print("\nNO CONFIGS PASSED ALL 5 GATES.")
        best = sorted(backtest_results.items(), key=lambda x: x[1].get('gates_passed', 0), reverse=True)[:3]
        print("\nBest attempts:")
        for k, r in best:
            print(f"  {k}: {r['gates_passed']}/5 gates, Sharpe={r['sharpe']:.3f}, "
                  f"p={r['perm_p_value']:.3f}, gap={r.get('regime_gap', 'N/A')}")

    # -- HYPOTHESIS TEST SUMMARY --
    print("\n" + "=" * 70)
    print("HYPOTHESIS ASSESSMENT")
    print("=" * 70)

    # Check if cyclicals vs defensives IC pattern matches hypothesis
    for sig_name, label in [('chg_2s10s_5d', '2s10s'), ('chg_3m10s_5d', '3m10s')]:
        print(f"\n  {label} steepening -> cyclicals outperform?")
        key_5d = f"{sig_name}_fwd5d"
        if key_5d in all_results:
            cyc_ics = [all_results[key_5d].get(s, {}).get('ic', 0) for s in CYCLICAL_SECTORS]
            def_ics = [all_results[key_5d].get(s, {}).get('ic', 0) for s in DEFENSIVE_SECTORS]
            avg_cyc = np.mean(cyc_ics)
            avg_def = np.mean(def_ics)
            print(f"    Avg cyclical IC: {avg_cyc:+.4f}")
            print(f"    Avg defensive IC: {avg_def:+.4f}")
            if avg_cyc > avg_def and avg_cyc > 0:
                print(f"    -> SUPPORTED (cyclicals have higher positive IC)")
            elif avg_def > avg_cyc and avg_def > 0:
                print(f"    -> REVERSED (defensives benefit from steepening)")
            else:
                print(f"    -> INCONCLUSIVE")

    # -- SAVE --
    output = {
        'metadata': {
            'run_date': datetime.now().isoformat(),
            'start_date': START_DATE,
            'end_date': END_DATE,
            'lookback': LOOKBACK,
            'curve_change_days': CURVE_CHANGE_DAYS,
            'cost_rt_pct': COST_RT_PCT * 100,
            'n_permutations': N_PERMUTATIONS,
        },
        'per_sector_ic': all_results,
        'backtest_results': backtest_results,
        'passing_configs': passing,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_PATH}")

    return output


if __name__ == '__main__':
    main()
