#!/usr/bin/env python3
"""
Multi-Signal Aggregation Backtest
Combines 6 individually weak signals into composite scores.
Tests 6 variant strategies: A-F.
Walk-forward OOT: Jan 2022 - Jul 2026. $645 start. 5-gate validation.
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────────
INITIAL_CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%
START_DATE = '2022-01-01'
END_DATE = '2026-07-30'
PERM_ITERATIONS = 500

TICKERS = ['SPY', 'QQQ', 'IWM', 'GLD', 'TLT',
           'XLK', 'XLF', 'XLE', 'XLV', 'XLI',
           'XLP', 'XLU', 'XLB', 'XLRE', 'XLC']

SECTOR_ETFS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI',
               'XLP', 'XLU', 'XLB', 'XLRE', 'XLC']

GATES = {
    'sharpe_min': 0.5,
    'perm_p_max': 0.05,
    'regime_gap_max': 0.5,
    'maxdd_min': -0.50,
    'min_trades': 20,
}

OUTPUT_PATH = Path('/home/jupiter/Lvl3Quant/data/signal_aggregation_results.json')


# ── Data Download ──────────────────────────────────────────────────────────
def download_data():
    """Download OHLCV data for all tickers."""
    print("Downloading data...")
    # Add buffer for lookback
    buf_start = '2021-06-01'
    data = {}
    for t in TICKERS:
        try:
            df = yf.download(t, start=buf_start, end=END_DATE, progress=False, auto_adjust=True)
            if len(df) > 100:
                data[t] = df
                print(f"  {t}: {len(df)} bars")
            else:
                print(f"  {t}: SKIPPED (only {len(df)} bars)")
        except Exception as e:
            print(f"  {t}: FAILED ({e})")
    return data


# ── Signal Generators ──────────────────────────────────────────────────────
def calc_momentum(close, period=20):
    """Signal 1: 20-day return, scaled to [-1, +1]."""
    ret = close.pct_change(period)
    # Scale: clip at ±20% and normalize
    scaled = (ret / 0.20).clip(-1, 1)
    return scaled


def calc_rsi(close, period=5):
    """Signal 2: 5-day RSI. <30 → +1, >70 → -1, else 0."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    signal = pd.Series(0.0, index=close.index)
    signal[rsi < 30] = 1.0
    signal[rsi > 70] = -1.0
    return signal


def calc_volume_signal(close, volume, period=20):
    """Signal 3: Volume spike on directional move."""
    vol_ratio = volume / volume.rolling(period).mean()
    daily_ret = close.pct_change()
    signal = pd.Series(0.0, index=close.index)
    signal[(vol_ratio > 1.5) & (daily_ret > 0)] = 1.0
    signal[(vol_ratio > 1.5) & (daily_ret < 0)] = -1.0
    return signal


def calc_trend(close, period=50):
    """Signal 4: Price vs 50-SMA. Above → +1, Below → -1."""
    sma = close.rolling(period).mean()
    signal = pd.Series(0.0, index=close.index)
    signal[close > sma] = 1.0
    signal[close < sma] = -1.0
    return signal


def calc_breadth(sector_closes):
    """Signal 5: % of sector ETFs above their own 20-SMA."""
    above_sma = pd.DataFrame()
    for t, close in sector_closes.items():
        sma20 = close.rolling(20).mean()
        above_sma[t] = (close > sma20).astype(float)
    pct_above = above_sma.mean(axis=1)
    signal = pd.Series(0.0, index=pct_above.index)
    signal[pct_above > 0.70] = 1.0
    signal[pct_above < 0.30] = -1.0
    return signal


def calc_vol_regime(close):
    """Signal 6: 20-day vol vs 60-day vol. Contracting → +1, Expanding → -1."""
    ret = close.pct_change()
    vol20 = ret.rolling(20).std()
    vol60 = ret.rolling(60).std()
    signal = pd.Series(0.0, index=close.index)
    signal[vol20 < vol60] = 1.0
    signal[vol20 > vol60] = -1.0
    return signal


def build_all_signals(data):
    """Build signal DataFrame for each ticker and the breadth signal."""
    # Common date index (intersection)
    common_idx = None
    for t in TICKERS:
        if t in data:
            idx = data[t].index
            common_idx = idx if common_idx is None else common_idx.intersection(idx)

    signals = {}  # signals[ticker] = DataFrame with columns: mom, rsi, vol_sig, trend, breadth, vol_regime
    sector_closes = {}
    for t in SECTOR_ETFS:
        if t in data:
            sector_closes[t] = data[t]['Close'].reindex(common_idx)

    breadth_signal = calc_breadth(sector_closes)

    for t in TICKERS:
        if t not in data:
            continue
        df = data[t].reindex(common_idx)
        close = df['Close']
        volume = df['Volume']

        sig_df = pd.DataFrame(index=common_idx)
        sig_df['momentum'] = calc_momentum(close)
        sig_df['rsi'] = calc_rsi(close)
        sig_df['volume'] = calc_volume_signal(close, volume)
        sig_df['trend'] = calc_trend(close)
        sig_df['breadth'] = breadth_signal
        sig_df['vol_regime'] = calc_vol_regime(close)
        sig_df['composite'] = sig_df.mean(axis=1)
        signals[t] = sig_df

    return signals, common_idx


# ── Performance Metrics ────────────────────────────────────────────────────
def calc_metrics(equity_curve, trades_count, spy_returns):
    """Calculate all performance metrics from an equity curve Series."""
    returns = equity_curve.pct_change().dropna()
    returns = returns.replace([np.inf, -np.inf], 0).fillna(0)

    total_ret = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    n_years = len(returns) / 252
    ann_ret = (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1
    ann_vol = returns.std() * np.sqrt(252) if returns.std() > 0 else 1e-6

    sharpe = ann_ret / ann_vol if ann_vol > 1e-8 else 0.0

    downside = returns[returns < 0]
    downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 and downside.std() > 0 else 1e-6
    sortino = ann_ret / downside_vol

    # Profit factor
    pos_ret = returns[returns > 0].sum()
    neg_ret = abs(returns[returns < 0].sum())
    pf = pos_ret / neg_ret if neg_ret > 0 else 99.9

    # Win rate (daily)
    wr = (returns > 0).sum() / max(len(returns[returns != 0]), 1)

    # Max drawdown
    peak = equity_curve.cummax()
    dd = (equity_curve - peak) / peak
    maxdd = dd.min()

    # QQQ correlation
    common = returns.index.intersection(spy_returns.index)
    if len(common) > 20:
        qqq_corr = returns.reindex(common).corr(spy_returns.reindex(common))
    else:
        qqq_corr = 0.0

    return {
        'sharpe': round(float(sharpe), 3),
        'sortino': round(float(sortino), 3),
        'pf': round(float(min(pf, 99.9)), 3),
        'wr': round(float(wr), 3),
        'maxdd': round(float(maxdd), 4),
        'total_return': round(float(total_ret), 4),
        'n_trades': int(trades_count),
        'ann_return': round(float(ann_ret), 4),
        'ann_vol': round(float(ann_vol), 4),
        'qqq_corr': round(float(qqq_corr), 3),
    }


def calc_regime_metrics(equity_curve, spy_close):
    """Calculate bull/bear Sharpe and regime gap."""
    sma200 = spy_close.rolling(200).mean()
    returns = equity_curve.pct_change().dropna()
    common = returns.index.intersection(sma200.dropna().index)
    returns = returns.reindex(common)
    spy_c = spy_close.reindex(common)
    sma_c = sma200.reindex(common)

    bull_mask = spy_c > sma_c
    bear_mask = ~bull_mask

    def regime_sharpe(r):
        if len(r) < 10 or r.std() < 1e-8:
            return 0.0
        return float((r.mean() * 252) / (r.std() * np.sqrt(252)))

    bull_sharpe = regime_sharpe(returns[bull_mask])
    bear_sharpe = regime_sharpe(returns[bear_mask])

    denom = max(abs(bull_sharpe), abs(bear_sharpe), 1e-6)
    regime_gap = abs(bull_sharpe - bear_sharpe) / denom

    return {
        'bull_sharpe': round(bull_sharpe, 3),
        'bear_sharpe': round(bear_sharpe, 3),
        'regime_gap': round(regime_gap, 3),
    }


# ── Strategy Simulation Engine ─────────────────────────────────────────────
def simulate_allocation(alloc_series, data, oot_start):
    """
    alloc_series: dict of {date: {ticker: weight}} daily allocations.
    Returns equity curve and trade count.
    """
    dates = sorted(alloc_series.keys())
    dates = [d for d in dates if d >= pd.Timestamp(oot_start)]
    if not dates:
        return pd.Series(dtype=float), 0

    equity = INITIAL_CAPITAL
    equity_curve = {}
    prev_alloc = {}
    trades = 0

    for d in dates:
        target = alloc_series[d]
        # Count trades (allocation changes)
        for t, w in target.items():
            old_w = prev_alloc.get(t, 0.0)
            if abs(w - old_w) > 0.01:
                trades += 1

        # Calculate daily return based on target allocation
        # (assume rebalance at open, earn close-to-close return with slippage on trades)
        day_ret = 0.0
        for t, w in target.items():
            if t in data and d in data[t].index:
                idx = data[t].index.get_loc(d)
                if idx > 0:
                    r = (data[t]['Close'].iloc[idx] / data[t]['Close'].iloc[idx - 1]) - 1
                    day_ret += w * r
                    # Slippage on allocation changes
                    old_w = prev_alloc.get(t, 0.0)
                    if abs(w - old_w) > 0.01:
                        day_ret -= abs(w - old_w) * SLIPPAGE_PCT

        equity *= (1 + day_ret)
        equity_curve[d] = equity
        prev_alloc = target.copy()

    return pd.Series(equity_curve), trades


# ── Variant Strategies ─────────────────────────────────────────────────────
def variant_a(signals, data, common_idx, oot_start):
    """Threshold Long-Only: SPY when composite > +0.3, GLD when < -0.3, cash between."""
    alloc = {}
    spy_sig = signals.get('SPY')
    if spy_sig is None:
        return pd.Series(dtype=float), 0
    for d in common_idx:
        if d < pd.Timestamp(oot_start):
            continue
        comp = spy_sig.loc[d, 'composite'] if d in spy_sig.index else 0
        if np.isnan(comp):
            alloc[d] = {}
        elif comp > 0.3:
            alloc[d] = {'SPY': 1.0}
        elif comp < -0.3:
            alloc[d] = {'GLD': 1.0}
        else:
            alloc[d] = {}  # cash
    return simulate_allocation(alloc, data, oot_start)


def variant_b(signals, data, common_idx, oot_start):
    """Sector Selection: 100% into highest composite sector ETF. Weekly rebalance."""
    alloc = {}
    day_count = 0
    current_pick = None
    for d in common_idx:
        if d < pd.Timestamp(oot_start):
            continue
        day_count += 1
        if day_count % 5 == 1 or current_pick is None:
            best_score = -999
            best_ticker = SECTOR_ETFS[0]
            for t in SECTOR_ETFS:
                if t in signals and d in signals[t].index:
                    sc = signals[t].loc[d, 'composite']
                    if not np.isnan(sc) and sc > best_score:
                        best_score = sc
                        best_ticker = t
            current_pick = best_ticker
        alloc[d] = {current_pick: 1.0}
    return simulate_allocation(alloc, data, oot_start)


def variant_c(signals, data, common_idx, oot_start):
    """Conviction Sizing: SPY/GLD blend based on composite score."""
    alloc = {}
    spy_sig = signals.get('SPY')
    if spy_sig is None:
        return pd.Series(dtype=float), 0
    for d in common_idx:
        if d < pd.Timestamp(oot_start):
            continue
        comp = spy_sig.loc[d, 'composite'] if d in spy_sig.index else 0
        if np.isnan(comp):
            comp = 0
        if comp > 0.5:
            alloc[d] = {'SPY': 1.0}
        elif comp > 0.2:
            alloc[d] = {'SPY': 0.7, 'GLD': 0.3}
        elif comp > -0.2:
            alloc[d] = {'SPY': 0.5, 'GLD': 0.5}
        else:
            alloc[d] = {'GLD': 1.0}
    return simulate_allocation(alloc, data, oot_start)


def variant_d(signals, data, common_idx, oot_start):
    """Momentum + Mean Reversion Only: QQQ when both bullish, TLT when both bearish, 50/50 when disagree. Weekly."""
    alloc = {}
    spy_sig = signals.get('SPY')
    if spy_sig is None:
        return pd.Series(dtype=float), 0
    day_count = 0
    current_alloc = {'QQQ': 0.5, 'TLT': 0.5}
    for d in common_idx:
        if d < pd.Timestamp(oot_start):
            continue
        day_count += 1
        if day_count % 5 == 1 or day_count == 1:
            mom = spy_sig.loc[d, 'momentum'] if d in spy_sig.index else 0
            rsi = spy_sig.loc[d, 'rsi'] if d in spy_sig.index else 0
            if np.isnan(mom):
                mom = 0
            if np.isnan(rsi):
                rsi = 0
            mom_bull = mom > 0
            rsi_bull = rsi > 0
            if mom_bull and rsi_bull:
                current_alloc = {'QQQ': 1.0}
            elif (not mom_bull) and (not rsi_bull):
                current_alloc = {'TLT': 1.0}
            else:
                current_alloc = {'QQQ': 0.5, 'TLT': 0.5}
        alloc[d] = current_alloc.copy()
    return simulate_allocation(alloc, data, oot_start)


def variant_e(signals, data, common_idx, oot_start):
    """Breadth-Vol Filter: QQQ when breadth bullish AND vol contracting, GLD otherwise. Weekly."""
    alloc = {}
    spy_sig = signals.get('SPY')
    if spy_sig is None:
        return pd.Series(dtype=float), 0
    day_count = 0
    current_alloc = {'GLD': 1.0}
    for d in common_idx:
        if d < pd.Timestamp(oot_start):
            continue
        day_count += 1
        if day_count % 5 == 1 or day_count == 1:
            breadth = spy_sig.loc[d, 'breadth'] if d in spy_sig.index else 0
            vol_reg = spy_sig.loc[d, 'vol_regime'] if d in spy_sig.index else 0
            if np.isnan(breadth):
                breadth = 0
            if np.isnan(vol_reg):
                vol_reg = 0
            if breadth > 0 and vol_reg > 0:
                current_alloc = {'QQQ': 1.0}
            else:
                current_alloc = {'GLD': 1.0}
        alloc[d] = current_alloc.copy()
    return simulate_allocation(alloc, data, oot_start)


def variant_f(signals, data, common_idx, oot_start):
    """Adaptive Weighting: Weight signals by 60-day correlation with forward 5-day SPY returns. Weekly."""
    alloc = {}
    spy_sig = signals.get('SPY')
    if spy_sig is None or 'SPY' not in data:
        return pd.Series(dtype=float), 0

    spy_close = data['SPY']['Close'].reindex(common_idx)
    fwd_5d_ret = spy_close.pct_change(5).shift(-5)  # forward 5-day return

    signal_cols = ['momentum', 'rsi', 'volume', 'trend', 'breadth', 'vol_regime']
    day_count = 0
    current_alloc = {'SPY': 0.5, 'GLD': 0.5}

    for d in common_idx:
        if d < pd.Timestamp(oot_start):
            continue
        day_count += 1
        if day_count % 5 == 1 or day_count == 1:
            # Get trailing 60-day window (before d) for weight calculation
            loc = common_idx.get_loc(d)
            if loc >= 65:
                window = common_idx[loc - 60:loc]
                weights = {}
                for col in signal_cols:
                    sig_vals = spy_sig[col].reindex(window)
                    fwd_vals = fwd_5d_ret.reindex(window)
                    valid = sig_vals.notna() & fwd_vals.notna()
                    if valid.sum() > 10:
                        corr = sig_vals[valid].corr(fwd_vals[valid])
                        weights[col] = corr if not np.isnan(corr) else 0.0
                    else:
                        weights[col] = 0.0

                # Normalize weights (absolute values sum to 1, keep signs)
                total_abs = sum(abs(v) for v in weights.values())
                if total_abs > 1e-8:
                    weights = {k: v / total_abs for k, v in weights.items()}
                else:
                    weights = {k: 1.0 / len(signal_cols) for k in signal_cols}

                # Weighted composite
                comp = 0.0
                for col in signal_cols:
                    val = spy_sig.loc[d, col] if d in spy_sig.index else 0
                    if np.isnan(val):
                        val = 0
                    comp += weights[col] * val

                if comp > 0:
                    current_alloc = {'SPY': 1.0}
                else:
                    current_alloc = {'GLD': 1.0}

        alloc[d] = current_alloc.copy()
    return simulate_allocation(alloc, data, oot_start)


# ── Permutation Test ───────────────────────────────────────────────────────
def permutation_test(variant_func, signals, data, common_idx, oot_start,
                     actual_sharpe, n_iter=PERM_ITERATIONS):
    """Shuffle signal dates to test significance."""
    rng = np.random.RandomState(42)
    count_better = 0

    # Create shuffled signals
    signal_cols = ['momentum', 'rsi', 'volume', 'trend', 'breadth', 'vol_regime']
    oot_idx = [d for d in common_idx if d >= pd.Timestamp(oot_start)]
    n_oot = len(oot_idx)

    for i in range(n_iter):
        shuffled_signals = {}
        for t, sig_df in signals.items():
            s = sig_df.copy()
            # Shuffle each signal column independently within OOT period
            for col in signal_cols:
                if col in s.columns:
                    oot_vals = s.loc[s.index.isin(oot_idx), col].values.copy()
                    rng.shuffle(oot_vals)
                    s.loc[s.index.isin(oot_idx), col] = oot_vals
            s['composite'] = s[signal_cols].mean(axis=1)
            shuffled_signals[t] = s

        eq, _ = variant_func(shuffled_signals, data, common_idx, oot_start)
        if len(eq) > 20:
            ret = eq.pct_change().dropna().replace([np.inf, -np.inf], 0).fillna(0)
            vol = ret.std() * np.sqrt(252)
            ann_r = (1 + (eq.iloc[-1] / eq.iloc[0] - 1)) ** (1 / max(len(ret) / 252, 0.01)) - 1
            perm_sharpe = ann_r / vol if vol > 1e-8 else 0
            if perm_sharpe >= actual_sharpe:
                count_better += 1

    return round((count_better + 1) / (n_iter + 1), 4)


# ── Gate Check ─────────────────────────────────────────────────────────────
def check_gates(metrics, regime_metrics, perm_p):
    """Check if strategy passes all 5 gates."""
    results = {
        'sharpe': metrics['sharpe'] >= GATES['sharpe_min'],
        'perm_p': perm_p <= GATES['perm_p_max'],
        'regime_gap': regime_metrics['regime_gap'] <= GATES['regime_gap_max'],
        'maxdd': metrics['maxdd'] >= GATES['maxdd_min'],
        'n_trades': metrics['n_trades'] >= GATES['min_trades'],
    }
    results['pass_all'] = all(results.values())
    return results


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("MULTI-SIGNAL AGGREGATION BACKTEST")
    print(f"OOT: {START_DATE} to {END_DATE} | Capital: ${INITIAL_CAPITAL}")
    print("=" * 80)

    data = download_data()
    if 'SPY' not in data:
        print("FATAL: SPY data missing")
        return

    # Flatten multi-index columns if needed
    for t in list(data.keys()):
        if isinstance(data[t].columns, pd.MultiIndex):
            data[t].columns = data[t].columns.get_level_values(0)

    signals, common_idx = build_all_signals(data)
    print(f"\nSignals built. Common dates: {len(common_idx)}")
    oot_dates = [d for d in common_idx if d >= pd.Timestamp(START_DATE)]
    print(f"OOT dates: {len(oot_dates)} ({oot_dates[0].date()} to {oot_dates[-1].date()})")

    spy_returns = data['SPY']['Close'].reindex(common_idx).pct_change().dropna()
    spy_close = data['SPY']['Close'].reindex(common_idx)

    # Also get QQQ returns for correlation
    qqq_returns = data['QQQ']['Close'].reindex(common_idx).pct_change().dropna() if 'QQQ' in data else spy_returns

    variants = {
        'A_threshold_long_only': variant_a,
        'B_sector_selection': variant_b,
        'C_conviction_sizing': variant_c,
        'D_mom_meanrev': variant_d,
        'E_breadth_vol_filter': variant_e,
        'F_adaptive_weighting': variant_f,
    }

    all_results = {}

    for name, func in variants.items():
        print(f"\n{'─' * 60}")
        print(f"Running Variant {name}...")
        eq, trades = func(signals, data, common_idx, START_DATE)

        if len(eq) < 20:
            print(f"  SKIP: only {len(eq)} equity points")
            continue

        metrics = calc_metrics(eq, trades, qqq_returns)
        regime = calc_regime_metrics(eq, spy_close)

        print(f"  Sharpe={metrics['sharpe']:.3f}  Sortino={metrics['sortino']:.3f}  "
              f"PF={metrics['pf']:.2f}  WR={metrics['wr']:.1%}  MaxDD={metrics['maxdd']:.1%}  "
              f"Return={metrics['total_return']:.1%}  Trades={metrics['n_trades']}")
        print(f"  Bull Sharpe={regime['bull_sharpe']:.3f}  Bear Sharpe={regime['bear_sharpe']:.3f}  "
              f"Regime Gap={regime['regime_gap']:.3f}  QQQ Corr={metrics['qqq_corr']:.3f}")

        # Check non-perm gates first; skip expensive perm test if already failing
        non_perm_pass = (
            metrics['sharpe'] >= GATES['sharpe_min'] and
            regime['regime_gap'] <= GATES['regime_gap_max'] and
            metrics['maxdd'] >= GATES['maxdd_min'] and
            metrics['n_trades'] >= GATES['min_trades']
        )
        if non_perm_pass:
            print(f"  Running {PERM_ITERATIONS}-iter permutation test...")
            perm_p = permutation_test(func, signals, data, common_idx, START_DATE, metrics['sharpe'])
        else:
            # Run quick 50-iter perm test just for reporting
            print(f"  Non-perm gate(s) failed. Running quick 50-iter perm test...")
            perm_p = permutation_test(func, signals, data, common_idx, START_DATE, metrics['sharpe'], n_iter=50)
        print(f"  Perm p-value: {perm_p}")

        gates = check_gates(metrics, regime, perm_p)
        gate_str = " | ".join([f"{k}:{'PASS' if v else 'FAIL'}" for k, v in gates.items()])
        print(f"  Gates: {gate_str}")
        verdict = "PASS ALL GATES" if gates['pass_all'] else "REJECTED"
        print(f"  >>> VERDICT: {verdict}")

        all_results[name] = {
            **metrics,
            **regime,
            'perm_p': perm_p,
            'gates': {k: bool(v) for k, v in gates.items()},
            'verdict': verdict,
            'final_equity': round(float(eq.iloc[-1]), 2),
        }

    # ── Buy & Hold Benchmarks ──────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("Benchmarks (Buy & Hold):")
    for bench in ['SPY', 'QQQ', 'GLD']:
        if bench not in data:
            continue
        bench_alloc = {d: {bench: 1.0} for d in common_idx if d >= pd.Timestamp(START_DATE)}
        eq_b, tr_b = simulate_allocation(bench_alloc, data, START_DATE)
        if len(eq_b) > 20:
            m = calc_metrics(eq_b, 1, qqq_returns)
            r = calc_regime_metrics(eq_b, spy_close)
            print(f"  {bench}: Sharpe={m['sharpe']:.3f}  Return={m['total_return']:.1%}  "
                  f"MaxDD={m['maxdd']:.1%}  Bull={r['bull_sharpe']:.3f}  Bear={r['bear_sharpe']:.3f}")
            all_results[f'BH_{bench}'] = {**m, **r, 'perm_p': None, 'gates': None, 'verdict': 'BENCHMARK',
                                          'final_equity': round(float(eq_b.iloc[-1]), 2)}

    # ── Summary Table ──────────────────────────────────────────────────────
    print(f"\n{'=' * 100}")
    print(f"{'SUMMARY TABLE':^100}")
    print(f"{'=' * 100}")
    header = f"{'Variant':<28} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} {'WR':>6} {'MaxDD':>7} {'Return':>8} {'Trades':>7} {'PermP':>7} {'RGap':>6} {'Verdict':>12}"
    print(header)
    print("-" * 100)
    for name, r in all_results.items():
        perm_str = f"{r['perm_p']:.3f}" if r['perm_p'] is not None else "  N/A"
        rgap_str = f"{r['regime_gap']:.3f}" if 'regime_gap' in r else "  N/A"
        print(f"{name:<28} {r['sharpe']:>7.3f} {r['sortino']:>8.3f} {r['pf']:>6.2f} "
              f"{r['wr']:>5.1%} {r['maxdd']:>7.1%} {r['total_return']:>7.1%} "
              f"{r['n_trades']:>7d} {perm_str:>7} {rgap_str:>6} {r['verdict']:>12}")

    # Count passes
    passes = [n for n, r in all_results.items() if r.get('verdict') == 'PASS ALL GATES']
    print(f"\n{'=' * 100}")
    print(f"GATE PASSES: {len(passes)}/{sum(1 for n in all_results if not n.startswith('BH_'))} variants")
    if passes:
        print(f"  Passing: {', '.join(passes)}")
    else:
        print("  No variants passed all 5 gates.")

    # ── Save Results ───────────────────────────────────────────────────────
    output = {
        'meta': {
            'backtest': 'multi_signal_aggregation',
            'oot_period': f'{START_DATE} to {END_DATE}',
            'initial_capital': INITIAL_CAPITAL,
            'slippage_pct': SLIPPAGE_PCT,
            'perm_iterations': PERM_ITERATIONS,
            'n_signals': 6,
            'signals': ['momentum_20d', 'rsi_5d', 'volume_spike', 'trend_50sma',
                        'breadth_sector', 'vol_regime_20v60'],
            'run_timestamp': datetime.now().isoformat(),
            'gates': GATES,
        },
        'results': all_results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
