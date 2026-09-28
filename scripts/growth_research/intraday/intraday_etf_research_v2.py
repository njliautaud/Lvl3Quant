#!/usr/bin/env python3
"""
Intraday ETF Growth Strategy Research v2 — Optimized
=====================================================
Faster version: vectorized where possible, smaller param grids,
focus on SPY/QQQ first (most liquid), expand if promising.

Data: yfinance 1h bars (2 years)
Strategies: ORB, Mean Reversion, Gap Fill, VWAP Reversion
Validation: Sliding window walk-forward (60d train, 1d OOT)
"""

import os, json, warnings
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings('ignore')

DATA_DIR = Path('/home/jupiter/Lvl3Quant/output/growth_research/intraday')
DATA_DIR.mkdir(parents=True, exist_ok=True)

SLIPPAGE_BPS = 1.0
SLIPPAGE_FRAC = SLIPPAGE_BPS / 10000.0
TRAIN_DAYS = 60
N_PERMUTATIONS = 100

# ── Load data (already downloaded by v1) ──────────────────────────────────

def load_data():
    all_data = {}
    for f in DATA_DIR.glob('*.parquet'):
        all_data[f.stem] = pd.read_parquet(f)
    return all_data

def classify_regimes(daily_df):
    daily = daily_df.copy()
    ret = daily['Close'].pct_change()
    regime = {}
    for dt, r in zip(daily.index, ret):
        d = dt.date() if hasattr(dt, 'date') else dt
        if pd.isna(r): regime[d] = 'flat'
        elif r > 0.001: regime[d] = 'green'
        elif r < -0.001: regime[d] = 'red'
        else: regime[d] = 'flat'
    return regime

# ── Strategy A: ORB (vectorized per day) ──────────────────────────────────

def run_orb_all_days(h1_df, rr_ratio):
    """Run ORB on all days, return dict of date -> pnl_pct."""
    df = h1_df.copy()
    dates_arr = np.array([d.date() for d in df.index])
    unique_dates = sorted(set(dates_arr))

    results = {}
    for date in unique_dates:
        mask = dates_arr == date
        bars = df[mask]
        if len(bars) < 3:
            results[date] = 0.0
            continue

        or_high = bars['High'].iloc[0]
        or_low = bars['Low'].iloc[0]
        or_range = or_high - or_low
        if or_range <= 0:
            results[date] = 0.0
            continue

        traded = False
        for i in range(1, len(bars)):
            h, l, c = bars['High'].iloc[i], bars['Low'].iloc[i], bars['Close'].iloc[i]

            if h > or_high and not traded:
                entry = or_high * (1 + SLIPPAGE_FRAC)
                stop = or_low
                target = entry + rr_ratio * or_range
                pnl = _sim_from_bar(bars.iloc[i:], entry, stop, target, 'long')
                results[date] = pnl
                traded = True
                break
            elif l < or_low and not traded:
                entry = or_low * (1 - SLIPPAGE_FRAC)
                stop = or_high
                target = entry - rr_ratio * or_range
                pnl = _sim_from_bar(bars.iloc[i:], entry, stop, target, 'short')
                results[date] = pnl
                traded = True
                break

        if not traded:
            results[date] = 0.0

    return results

def _sim_from_bar(bars, entry, stop, target, direction):
    for i in range(len(bars)):
        h, l = bars['High'].iloc[i], bars['Low'].iloc[i]
        if direction == 'long':
            if l <= stop: return (stop * (1 - SLIPPAGE_FRAC) - entry) / entry
            if h >= target: return (target * (1 - SLIPPAGE_FRAC) - entry) / entry
        else:
            if h >= stop: return (entry - stop * (1 + SLIPPAGE_FRAC)) / entry
            if l <= target: return (entry - target * (1 + SLIPPAGE_FRAC)) / entry
    exit_p = bars['Close'].iloc[-1]
    if direction == 'long':
        return (exit_p * (1 - SLIPPAGE_FRAC) - entry) / entry
    else:
        return (entry - exit_p * (1 + SLIPPAGE_FRAC)) / entry

# ── Strategy B: Mean Reversion RSI ────────────────────────────────────────

def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def run_meanrev_all(h1_df, rsi_period, rsi_low, rsi_high):
    """Run mean reversion on hourly bars. Return dict of date -> pnl_pct."""
    df = h1_df.copy()
    df['rsi'] = compute_rsi(df['Close'], rsi_period)
    dates_arr = np.array([d.date() for d in df.index])

    results = defaultdict(float)
    in_trade = False
    entry_price = None
    direction = None
    entry_date = None
    bars_held = 0

    for i in range(len(df)):
        rsi_val = df['rsi'].iloc[i]
        close = df['Close'].iloc[i]
        date = dates_arr[i]

        if in_trade:
            bars_held += 1
            exit_now = False
            if direction == 'long' and (rsi_val > 50 or bars_held >= 6):
                exit_now = True
            elif direction == 'short' and (rsi_val < 50 or bars_held >= 6):
                exit_now = True

            if exit_now:
                if direction == 'long':
                    pnl = (close * (1 - SLIPPAGE_FRAC) - entry_price) / entry_price
                else:
                    pnl = (entry_price - close * (1 + SLIPPAGE_FRAC)) / entry_price
                results[entry_date] += pnl
                in_trade = False

        if not in_trade and not pd.isna(rsi_val):
            if rsi_val < rsi_low:
                entry_price = close * (1 + SLIPPAGE_FRAC)
                direction = 'long'
                entry_date = date
                in_trade = True
                bars_held = 0
            elif rsi_val > rsi_high:
                entry_price = close * (1 - SLIPPAGE_FRAC)
                direction = 'short'
                entry_date = date
                in_trade = True
                bars_held = 0

    # Fill all dates
    all_dates = sorted(set(dates_arr))
    full_results = {d: results.get(d, 0.0) for d in all_dates}
    return full_results

# ── Strategy C: Gap Fill ──────────────────────────────────────────────────

def run_gapfill_all(h1_df, gap_threshold):
    """Run gap fill on all days. Return dict of date -> pnl_pct."""
    dates_arr = np.array([d.date() for d in h1_df.index])
    unique_dates = sorted(set(dates_arr))

    results = {}
    prev_close = None

    for date in unique_dates:
        mask = dates_arr == date
        bars = h1_df[mask]

        if prev_close is None or len(bars) < 2:
            results[date] = 0.0
            prev_close = bars['Close'].iloc[-1] if len(bars) > 0 else prev_close
            continue

        today_open = bars['Open'].iloc[0]
        gap_pct = (today_open - prev_close) / prev_close

        if abs(gap_pct) < gap_threshold:
            results[date] = 0.0
            prev_close = bars['Close'].iloc[-1]
            continue

        if gap_pct > 0:
            entry = today_open * (1 - SLIPPAGE_FRAC)
            target = prev_close
            stop = today_open * (1 + abs(gap_pct))
            direction = 'short'
        else:
            entry = today_open * (1 + SLIPPAGE_FRAC)
            target = prev_close
            stop = today_open * (1 - abs(gap_pct))
            direction = 'long'

        max_bars = min(4, len(bars))
        pnl = 0.0
        for i in range(max_bars):
            h, l = bars['High'].iloc[i], bars['Low'].iloc[i]
            if direction == 'long':
                if l <= stop:
                    pnl = (stop * (1 - SLIPPAGE_FRAC) - entry) / entry; break
                if h >= target:
                    pnl = (target * (1 - SLIPPAGE_FRAC) - entry) / entry; break
            else:
                if h >= stop:
                    pnl = (entry - stop * (1 + SLIPPAGE_FRAC)) / entry; break
                if l <= target:
                    pnl = (entry - target * (1 + SLIPPAGE_FRAC)) / entry; break
        else:
            exit_p = bars['Close'].iloc[max_bars - 1]
            if direction == 'long':
                pnl = (exit_p * (1 - SLIPPAGE_FRAC) - entry) / entry
            else:
                pnl = (entry - exit_p * (1 + SLIPPAGE_FRAC)) / entry

        results[date] = pnl
        prev_close = bars['Close'].iloc[-1]

    return results

# ── Strategy D: VWAP Reversion ────────────────────────────────────────────

def run_vwap_all(h1_df, std_threshold):
    """VWAP reversion on all days. Return dict of date -> pnl_pct."""
    dates_arr = np.array([d.date() for d in h1_df.index])
    unique_dates = sorted(set(dates_arr))

    results = {}
    for date in unique_dates:
        mask = dates_arr == date
        bars = h1_df[mask]
        if len(bars) < 4:
            results[date] = 0.0
            continue

        cum_vol = bars['Volume'].cumsum().values
        cum_vp = (bars['Close'] * bars['Volume']).cumsum().values
        vwap = cum_vp / np.where(cum_vol == 0, np.nan, cum_vol)

        deviation = bars['Close'].values - vwap
        # Rolling std of deviation
        dev_std = pd.Series(deviation).rolling(3, min_periods=2).std().values

        day_pnl = 0.0
        in_trade = False
        entry = None
        direction = None

        for i in range(3, len(bars)):
            if np.isnan(dev_std[i]) or dev_std[i] == 0:
                continue
            z = deviation[i] / dev_std[i]

            if not in_trade:
                if z > std_threshold:
                    entry = bars['Close'].iloc[i] * (1 - SLIPPAGE_FRAC)
                    direction = 'short'
                    in_trade = True
                elif z < -std_threshold:
                    entry = bars['Close'].iloc[i] * (1 + SLIPPAGE_FRAC)
                    direction = 'long'
                    in_trade = True
            elif in_trade:
                if abs(z) < 0.5 or i == len(bars) - 1:
                    exit_p = bars['Close'].iloc[i]
                    if direction == 'long':
                        day_pnl += (exit_p * (1 - SLIPPAGE_FRAC) - entry) / entry
                    else:
                        day_pnl += (entry - exit_p * (1 + SLIPPAGE_FRAC)) / entry
                    in_trade = False

        results[date] = day_pnl

    return results

# ── Walk-Forward Engine ───────────────────────────────────────────────────

def walkforward_from_daily_pnls(daily_pnls_by_param, dates, regimes, train_window=60):
    """
    Walk-forward optimization given pre-computed daily PnLs for each param.

    daily_pnls_by_param: list of (param_name, {date: pnl_pct})
    Returns: list of OOT results
    """
    oot_results = []

    for i in range(train_window, len(dates)):
        train_dates = dates[i - train_window:i]
        oot_date = dates[i]

        # Find best param on train window
        best_param = None
        best_sharpe = -np.inf

        for param_name, pnl_dict in daily_pnls_by_param:
            train_pnls = [pnl_dict.get(d, 0.0) for d in train_dates]
            train_pnls = np.array(train_pnls)
            trade_pnls = train_pnls[train_pnls != 0]

            if len(trade_pnls) > 3:
                sharpe = np.mean(train_pnls) / (np.std(train_pnls) + 1e-10) * np.sqrt(252)
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_param = (param_name, pnl_dict)

        if best_param is None:
            best_param = daily_pnls_by_param[0]

        param_name, pnl_dict = best_param
        pnl = pnl_dict.get(oot_date, 0.0)
        regime = regimes.get(oot_date, 'flat')

        oot_results.append({
            'date': oot_date,
            'regime': regime,
            'pnl_pct': pnl,
            'params': param_name,
            'traded': pnl != 0,
        })

    return oot_results

# ── Metrics ───────────────────────────────────────────────────────────────

def compute_metrics(oot_results):
    if not oot_results:
        return {}

    pnls = np.array([r['pnl_pct'] for r in oot_results])
    trade_pnls = pnls[pnls != 0]

    n_days = len(pnls)
    n_trades = len(trade_pnls)

    if n_trades == 0:
        return {'n_days': n_days, 'n_trades': 0, 'total_return': 0}

    mean_d = np.mean(pnls)
    std_d = np.std(pnls) + 1e-10

    sharpe = mean_d / std_d * np.sqrt(252)

    downside = pnls[pnls < 0]
    ds_std = np.std(downside) if len(downside) > 0 else 1e-10
    sortino = mean_d / (ds_std + 1e-10) * np.sqrt(252)

    wr = np.mean(trade_pnls > 0)

    gp = np.sum(trade_pnls[trade_pnls > 0])
    gl = abs(np.sum(trade_pnls[trade_pnls < 0]))
    pf = gp / (gl + 1e-10)

    cum = np.cumsum(pnls)
    rmax = np.maximum.accumulate(cum)
    max_dd = np.min(cum - rmax)

    total_ret = np.sum(pnls)
    years = n_days / 252
    cagr = (1 + total_ret) ** (1 / max(years, 0.01)) - 1 if total_ret > -1 else -1
    calmar = cagr / (abs(max_dd) + 1e-10)

    return {
        'n_days': n_days, 'n_trades': n_trades,
        'trade_freq': n_trades / n_days,
        'total_return_pct': total_ret * 100,
        'cagr_pct': cagr * 100,
        'sharpe': sharpe, 'sortino': sortino,
        'win_rate': wr, 'profit_factor': pf,
        'max_dd_pct': max_dd * 100, 'calmar': calmar,
        'avg_trade_pct': np.mean(trade_pnls) * 100,
    }

def regime_analysis(oot_results):
    by_regime = defaultdict(list)
    for r in oot_results:
        by_regime[r['regime']].append(r['pnl_pct'])

    regime_metrics = {}
    for regime, pnls in by_regime.items():
        pnls = np.array(pnls)
        tp = pnls[pnls != 0]
        mean_d = np.mean(pnls)
        std_d = np.std(pnls) + 1e-10
        regime_metrics[regime] = {
            'n_days': len(pnls), 'n_trades': len(tp),
            'sharpe': mean_d / std_d * np.sqrt(252),
            'win_rate': np.mean(tp > 0) if len(tp) > 0 else 0,
            'total_return_pct': np.sum(pnls) * 100,
        }

    s_g = regime_metrics.get('green', {}).get('sharpe', 0)
    s_r = regime_metrics.get('red', {}).get('sharpe', 0)
    max_s = max(abs(s_g), abs(s_r))
    r1_gap = abs(s_g - s_r) / (max_s + 1e-10) if max_s > 0 else 0
    return regime_metrics, r1_gap, r1_gap <= 0.50

def permutation_test(oot_results, n=100):
    pnls = np.array([r['pnl_pct'] for r in oot_results])
    actual_sharpe = np.mean(pnls) / (np.std(pnls) + 1e-10) * np.sqrt(252)
    rng = np.random.RandomState(42)
    count = sum(1 for _ in range(n)
                if np.mean(pnls * rng.choice([-1, 1], len(pnls))) /
                   (np.std(pnls * rng.choice([-1, 1], len(pnls))) + 1e-10) * np.sqrt(252)
                >= actual_sharpe)
    return count / n

# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("INTRADAY ETF GROWTH STRATEGY RESEARCH v2")
    print("=" * 70)

    all_data = load_data()
    if not all_data:
        print("ERROR: No data found. Run v1 first to download.")
        return

    # Use SPY daily for regime classification
    spy_daily = all_data.get('SPY_daily')
    if spy_daily is None:
        print("ERROR: No SPY daily data")
        return
    regimes = classify_regimes(spy_daily)

    TICKERS = ['SPY', 'QQQ', 'IWM', 'XLK', 'XLE', 'XLF']
    all_results = {}

    for ticker in TICKERS:
        h1_key = f'{ticker}_1h'
        if h1_key not in all_data:
            continue

        h1 = all_data[h1_key]
        dates_arr = np.array([d.date() for d in h1.index])
        all_dates = sorted(set(dates_arr))

        print(f"\n{'─' * 50}")
        print(f"{ticker}: {len(all_dates)} trading days")

        # ── Strategy A: ORB ──
        print(f"  [A] ORB...", end=' ')
        orb_params = []
        for rr in [0.75, 1.0, 1.5, 2.0, 3.0]:
            pnl_dict = run_orb_all_days(h1, rr)
            orb_params.append((f'rr={rr}', pnl_dict))

        orb_oot = walkforward_from_daily_pnls(orb_params, all_dates, regimes, TRAIN_DAYS)
        orb_m = compute_metrics(orb_oot)
        all_results[f'{ticker}_ORB'] = {'metrics': orb_m, 'oot': orb_oot}
        print(f"CAGR={orb_m.get('cagr_pct',0):.1f}% Sharpe={orb_m.get('sharpe',0):.2f} WR={orb_m.get('win_rate',0):.0%} PF={orb_m.get('profit_factor',0):.2f} DD={orb_m.get('max_dd_pct',0):.1f}% T={orb_m.get('n_trades',0)}")

        # ── Strategy B: Mean Reversion ──
        print(f"  [B] MeanRev...", end=' ')
        mr_params = []
        for p, lo, hi in [(7,25,75), (7,30,70), (14,25,75), (14,30,70), (14,35,65), (21,30,70)]:
            pnl_dict = run_meanrev_all(h1, p, lo, hi)
            mr_params.append((f'p={p}_lo={lo}_hi={hi}', pnl_dict))

        mr_oot = walkforward_from_daily_pnls(mr_params, all_dates, regimes, TRAIN_DAYS)
        mr_m = compute_metrics(mr_oot)
        all_results[f'{ticker}_MeanRev'] = {'metrics': mr_m, 'oot': mr_oot}
        print(f"CAGR={mr_m.get('cagr_pct',0):.1f}% Sharpe={mr_m.get('sharpe',0):.2f} WR={mr_m.get('win_rate',0):.0%} PF={mr_m.get('profit_factor',0):.2f} DD={mr_m.get('max_dd_pct',0):.1f}% T={mr_m.get('n_trades',0)}")

        # ── Strategy C: Gap Fill ──
        print(f"  [C] GapFill...", end=' ')
        gf_params = []
        for thr in [0.002, 0.003, 0.005, 0.007, 0.01]:
            pnl_dict = run_gapfill_all(h1, thr)
            gf_params.append((f'thr={thr}', pnl_dict))

        gf_oot = walkforward_from_daily_pnls(gf_params, all_dates, regimes, TRAIN_DAYS)
        gf_m = compute_metrics(gf_oot)
        all_results[f'{ticker}_GapFill'] = {'metrics': gf_m, 'oot': gf_oot}
        print(f"CAGR={gf_m.get('cagr_pct',0):.1f}% Sharpe={gf_m.get('sharpe',0):.2f} WR={gf_m.get('win_rate',0):.0%} PF={gf_m.get('profit_factor',0):.2f} DD={gf_m.get('max_dd_pct',0):.1f}% T={gf_m.get('n_trades',0)}")

        # ── Strategy D: VWAP Reversion ──
        print(f"  [D] VWAP...", end=' ')
        vwap_params = []
        for thr in [1.0, 1.5, 2.0, 2.5, 3.0]:
            pnl_dict = run_vwap_all(h1, thr)
            vwap_params.append((f'std={thr}', pnl_dict))

        vwap_oot = walkforward_from_daily_pnls(vwap_params, all_dates, regimes, TRAIN_DAYS)
        vwap_m = compute_metrics(vwap_oot)
        all_results[f'{ticker}_VWAP'] = {'metrics': vwap_m, 'oot': vwap_oot}
        print(f"CAGR={vwap_m.get('cagr_pct',0):.1f}% Sharpe={vwap_m.get('sharpe',0):.2f} WR={vwap_m.get('win_rate',0):.0%} PF={vwap_m.get('profit_factor',0):.2f} DD={vwap_m.get('max_dd_pct',0):.1f}% T={vwap_m.get('n_trades',0)}")

    # ── RANKING ──
    print("\n" + "=" * 70)
    print("STRATEGY RANKING (Walk-Forward OOT, by Sharpe, min 10 trades)")
    print("=" * 70)

    ranked = [(k, v['metrics']) for k, v in all_results.items()
              if v['metrics'].get('n_trades', 0) > 10]
    ranked.sort(key=lambda x: x[1].get('sharpe', -99), reverse=True)

    print(f"\n{'Strategy':<25} {'CAGR%':>8} {'Sharpe':>8} {'Sortino':>8} {'WR':>6} {'PF':>6} {'MaxDD%':>8} {'Trades':>7}")
    print("─" * 80)
    for k, m in ranked:
        print(f"{k:<25} {m['cagr_pct']:>7.1f}% {m['sharpe']:>8.2f} {m['sortino']:>8.2f} "
              f"{m['win_rate']:>5.0%} {m['profit_factor']:>6.2f} {m['max_dd_pct']:>7.1f}% {m['n_trades']:>7}")

    # ── DEEP ANALYSIS on top 3 ──
    positive = [x for x in ranked if x[1].get('sharpe', 0) > 0]
    analyze = positive[:3] if positive else ranked[:3]

    print("\n" + "=" * 70)
    print("DEEP ANALYSIS")
    print("=" * 70)

    for key, m in analyze:
        oot = all_results[key]['oot']
        print(f"\n{'─' * 60}")
        print(f"STRATEGY: {key}")
        print(f"{'─' * 60}")

        # Metrics
        for k2, v in sorted(m.items()):
            print(f"  {k2}: {v:.4f}" if isinstance(v, float) else f"  {k2}: {v}")

        # Regime
        rm, r1_gap, r1_pass = regime_analysis(oot)
        print(f"\n  Regime Stratification:")
        for regime in ['green', 'red', 'flat']:
            if regime in rm:
                r = rm[regime]
                print(f"    {regime:>6}: Sharpe={r['sharpe']:.2f}  WR={r['win_rate']:.0%}  "
                      f"Return={r['total_return_pct']:.2f}%  Days={r['n_days']}  Trades={r['n_trades']}")
        print(f"  R1 Gap: {r1_gap:.3f} ({'PASS' if r1_pass else 'FAIL — regime-dependent'})")

        # Permutation
        p_val = permutation_test(oot, N_PERMUTATIONS)
        print(f"  Permutation p-value: {p_val:.3f} ({'SIGNIFICANT' if p_val < 0.05 else 'NOT SIGNIFICANT'})")

        # Slippage sensitivity
        print(f"\n  Slippage Sensitivity:")
        pnls_raw = np.array([r['pnl_pct'] for r in oot])
        for slip in [0, 1, 2, 5, 10]:
            # Each trade has 2 * slippage cost
            n_traded = np.sum(pnls_raw != 0)
            adj = pnls_raw.copy()
            traded_mask = adj != 0
            adj[traded_mask] = adj[traded_mask] + 2*SLIPPAGE_FRAC - 2*slip/10000
            ret = np.sum(adj) * 100
            sh = np.mean(adj) / (np.std(adj) + 1e-10) * np.sqrt(252)
            print(f"    {slip:>2} bps: Total={ret:>7.2f}%  Sharpe={sh:>6.2f}")

        # Capacity
        print(f"\n  $441 Account:")
        print(f"    Est annual return: ${441 * m.get('cagr_pct',0) / 100:.2f}")
        ann_trades = m.get('n_trades',0) * 252 / max(m.get('n_days',1), 1)
        print(f"    Est annual trades: {ann_trades:.0f}")

        # Per-day breakdown (sample)
        print(f"\n  OOT daily PnL distribution:")
        pnls = np.array([r['pnl_pct'] for r in oot])
        for pct in [5, 25, 50, 75, 95]:
            print(f"    p{pct}: {np.percentile(pnls, pct)*100:.4f}%")

    # ── OVERALL VERDICT ──
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    above_25 = [x for x in ranked if x[1].get('cagr_pct', 0) >= 25]
    profitable = [x for x in ranked if x[1].get('sharpe', 0) > 0.5 and x[1].get('cagr_pct', 0) > 0]

    if above_25:
        print(f"\n  {len(above_25)} strategy(ies) achieved >= 25% CAGR in walk-forward OOT:")
        for k, m in above_25:
            print(f"    {k}: CAGR={m['cagr_pct']:.1f}% Sharpe={m['sharpe']:.2f}")
        print("\n  HOWEVER: verify R1 regime test and permutation significance above.")
    elif profitable:
        print(f"\n  {len(profitable)} strategy(ies) are profitable (Sharpe > 0.5) but none hit 25% CAGR.")
        for k, m in profitable:
            print(f"    {k}: CAGR={m['cagr_pct']:.1f}% Sharpe={m['sharpe']:.2f}")
        print("\n  Options: combine strategies, increase leverage, or try different timeframe.")
    else:
        print("\n  HONEST RESULT: No intraday strategy on 1h ETF bars produced reliable edge")
        print("  in walk-forward out-of-sample testing.")
        print("\n  This is expected — hourly bars on liquid ETFs are well-arbitraged.")
        print("  Possible next steps:")
        print("    1. Try shorter timeframes (5m bars, but only 60 days of data)")
        print("    2. Try less efficient instruments (small-cap ETFs, sector rotation)")
        print("    3. Try event-driven overlays (earnings, FOMC) on these strategies")
        print("    4. Accept that TQQQ+200MA at ~28% CAGR may be near-optimal for passive approaches")

    # Save
    summary = {k: v['metrics'] for k, v in all_results.items()}
    with open(DATA_DIR / 'strategy_summary_v2.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    for k, v in all_results.items():
        oot_df = pd.DataFrame(v['oot'])
        oot_df['date'] = oot_df['date'].astype(str)
        oot_df.to_csv(DATA_DIR / f'{k}_oot_v2.csv', index=False)

    print(f"\nResults saved to {DATA_DIR}")

if __name__ == '__main__':
    main()
