#!/usr/bin/env python3
"""
Stat Arb Baseline — Adversarial Validation (Direction + Timing Permutations)
============================================================================
Tests whether the pure z-score pairs trading baseline (Sharpe ~0.807) is genuine alpha.

Two separate permutation tests:
  1. DIRECTION SHUFFLE: Keep same z-score entry timing, randomize long/short direction.
     Tests: does FADING the z-score matter, or is any direction profitable?
  2. TIMING SHUFFLE: Randomize which days get entries (same frequency).
     Tests: does z-score TIMING matter, or is any timing profitable?

Also: sub-period stability, outlier robustness, R1 regime check.

Uses the EXACT same backtest logic as ml_stat_arb.py run_backtest(use_ml=False).

HC #420: Authorized trading research
HC #665: Adversarial validation mandatory
"""
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from itertools import combinations
import json, os, sys, time, warnings
warnings.filterwarnings('ignore')

OUTPUT = '/home/jupiter/Lvl3Quant/output/stat_arb_baseline'
os.makedirs(OUTPUT, exist_ok=True)

# Parameters — IDENTICAL to ml_stat_arb.py
INITIAL_CAPITAL = 100_000
TRAIN_WINDOW = 252
COINT_LOOKBACK = 126
ENTRY_Z = 1.5
EXIT_Z = 0.3
STOP_Z = 4.0
MAX_HOLD = 42
MAX_PAIRS = 8
POS_SIZE = 1.0 / MAX_PAIRS
COST_BPS = 10
N_PERM = 30  # Reduced from 100 — still enough for p-value significance


def download_data():
    """Download sector ETFs + cross-asset universe — SAME as ml_stat_arb.py."""
    sectors = ['XLK','XLF','XLE','XLV','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']
    cross = ['GLD','GDX','TLT','IEF','HYG','LQD','SPY','QQQ','IWM','EEM','DIA']
    tickers = list(set(sectors + cross))
    print(f"Downloading {len(tickers)} assets...")
    sys.stdout.flush()
    df = yf.download(tickers, start='2005-01-01', progress=False)
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    close = df['Close'] if isinstance(df.columns, pd.MultiIndex) else df
    close = close.ffill()
    valid = close.columns[close.notna().sum() > 2000]
    close = close[valid].dropna()
    print(f"  {len(close)} days, {len(close.columns)} assets, {close.index[0].date()} to {close.index[-1].date()}")
    sys.stdout.flush()
    return close


def rolling_cointegration(price_a, price_b, window=COINT_LOOKBACK):
    """Test if two series are cointegrated over rolling window — SAME as ml_stat_arb.py."""
    if len(price_a) < window:
        return np.nan, np.nan, np.nan
    a = price_a.values[-window:]
    b = price_b.values[-window:]
    a_norm = a / a[0]
    b_norm = b / b[0]
    X = np.column_stack([b_norm, np.ones(window)])
    try:
        beta, alpha = np.linalg.lstsq(X, a_norm, rcond=None)[0]
    except:
        return np.nan, np.nan, np.nan
    residual = a_norm - beta * b_norm - alpha
    if np.std(residual) < 1e-10:
        return np.nan, np.nan, np.nan
    residual_lag = residual[:-1]
    residual_diff = np.diff(residual)
    if np.std(residual_lag) < 1e-10:
        return np.nan, np.nan, np.nan
    slope = np.polyfit(residual_lag, residual_diff, 1)[0]
    if slope >= 0:
        return np.nan, np.nan, np.nan
    half_life = -np.log(2) / slope
    lags = range(2, min(20, window // 5))
    tau = []
    for lag in lags:
        tau.append(np.std(np.subtract(residual[lag:], residual[:-lag])))
    if len(tau) < 2 or any(t <= 0 for t in tau):
        return np.nan, beta, half_life
    try:
        hurst = np.polyfit(np.log(list(lags)), np.log(tau), 1)[0]
    except:
        hurst = 0.5
    return hurst, beta, half_life


def run_backtest(close, shuffle_direction=False, shuffle_timing=False, rng=None):
    """
    Run the EXACT same baseline backtest as ml_stat_arb.py run_backtest(use_ml=False).

    shuffle_direction: keep same entry timing + pairs, randomize long/short direction
    shuffle_timing: randomize which scan days produce entries (same frequency)
    """
    if rng is None:
        rng = np.random.RandomState(42)

    returns = close.pct_change()
    n_days = len(close)
    assets = list(close.columns)
    all_pairs = list(combinations(assets, 2))

    daily_returns = np.zeros(n_days)
    positions = {}  # {pair_tuple: {'direction': 1/-1, 'entry_day': int, 'entry_z': float}}
    trade_log = []

    start_day = max(TRAIN_WINDOW + COINT_LOOKBACK, 504)

    # For timing shuffle: pre-compute which scan days will produce entries
    # We'll first count how many scan days produce entries in the real run,
    # then in the shuffle run, randomly select that many scan days.
    # But we can't know the count without running... so instead, on each scan day,
    # we flip a coin weighted by the historical entry rate.
    # Simpler approach: on each scan day, shuffle_timing replaces the z-score filter
    # with a random decision at the same average rate.
    # We estimate ~30% of scan days produce entries based on the original script behavior.
    # Actually, better: we keep all the cointegration filtering but randomize WHICH
    # qualifying pairs we enter (shuffle the pair order randomly instead of sorting by hurst).
    # Even better for a clean test: on days where the real logic WOULD enter,
    # shuffle_timing skips with 50% probability, and on days where real logic would NOT enter,
    # it enters with some probability. This is too complicated.
    #
    # Cleanest approach: shuffle_timing = on each scan day (day%5==0), instead of
    # checking z-scores, just randomly decide to enter random qualifying pairs.
    # The entry RATE is controlled by randomly sampling from all cointegrated pairs
    # regardless of z-score threshold.

    for day in range(start_day, n_days):
        # Daily P&L from existing positions — IDENTICAL to ml_stat_arb.py
        day_pnl = 0
        closed_pairs = []

        for pair, pos in positions.items():
            a, b = pair
            ret_a = returns[a].iloc[day] if not np.isnan(returns[a].iloc[day]) else 0
            ret_b = returns[b].iloc[day] if not np.isnan(returns[b].iloc[day]) else 0
            pair_ret = pos['direction'] * (ret_a - ret_b) * POS_SIZE
            day_pnl += pair_ret

            # Check exit conditions
            pa = close[a].iloc[max(0, day - 63):day + 1]
            pb = close[b].iloc[max(0, day - 63):day + 1]
            if len(pa) > 10:
                hurst, beta, hl = rolling_cointegration(pa, pb, min(63, len(pa)))
                if not np.isnan(beta):
                    spread = pa.iloc[-1] / pa.iloc[0] - beta * (pb.iloc[-1] / pb.iloc[0])
                    sp_mean = (pa / pa.iloc[0] - beta * (pb / pb.iloc[0])).mean()
                    sp_std = (pa / pa.iloc[0] - beta * (pb / pb.iloc[0])).std()
                    if sp_std > 1e-8:
                        current_z = (spread - sp_mean) / sp_std
                    else:
                        current_z = 0
                else:
                    current_z = 0
            else:
                current_z = 0

            days_held = day - pos['entry_day']

            exit_signal = (
                abs(current_z) < EXIT_Z or
                (pos['direction'] * current_z > STOP_Z) or
                days_held >= MAX_HOLD
            )

            if exit_signal:
                closed_pairs.append(pair)
                cum_ret = 0
                for d in range(pos['entry_day'] + 1, day + 1):
                    ra = returns[a].iloc[d] if not np.isnan(returns[a].iloc[d]) else 0
                    rb = returns[b].iloc[d] if not np.isnan(returns[b].iloc[d]) else 0
                    cum_ret += pos['direction'] * (ra - rb)

                trade_log.append({
                    'pair': f"{a}/{b}",
                    'direction': pos['direction'],
                    'entry_day': pos['entry_day'],
                    'exit_day': day,
                    'days_held': days_held,
                    'return': cum_ret - COST_BPS / 10000 * 2,
                    'exit_reason': 'revert' if abs(current_z) < EXIT_Z else ('stop' if pos['direction'] * current_z > STOP_Z else 'time')
                })

        for pair in closed_pairs:
            del positions[pair]

        daily_returns[day] = day_pnl

        # Every 5 days: scan for new entries — IDENTICAL to ml_stat_arb.py
        if day % 5 != 0:
            continue

        if len(positions) >= MAX_PAIRS:
            continue

        # For timing shuffle: randomly decide whether this scan day produces entries
        if shuffle_timing:
            # ~60% of scan days in the original produce no entries anyway
            # We match that by skipping ~60% randomly
            if rng.random() > 0.4:
                continue

        # Score all pairs by cointegration quality
        pair_scores = []
        for pair in all_pairs:
            if pair in positions:
                continue
            a, b = pair
            pa = close[a].iloc[max(0, day - COINT_LOOKBACK):day + 1]
            pb = close[b].iloc[max(0, day - COINT_LOOKBACK):day + 1]

            if len(pa) < 63:
                continue

            hurst, beta, half_life = rolling_cointegration(pa, pb)
            if np.isnan(hurst) or hurst > 0.45 or np.isnan(half_life) or half_life > 42 or half_life < 2:
                continue

            # Compute current z-score
            spread = pa / pa.iloc[0] - beta * (pb / pb.iloc[0])
            sp_mean = spread.rolling(63).mean().iloc[-1]
            sp_std = spread.rolling(63).std().iloc[-1]
            if sp_std < 1e-8:
                continue
            z = (spread.iloc[-1] - sp_mean) / sp_std

            if shuffle_timing:
                # For timing shuffle: enter ANY cointegrated pair regardless of z threshold
                # This tests whether z-score timing matters
                pair_scores.append((pair, z, beta, hurst, half_life))
            else:
                if abs(z) < ENTRY_Z:
                    continue
                pair_scores.append((pair, z, beta, hurst, half_life))

        if not pair_scores:
            continue

        # Sort by Hurst (lower = more mean-reverting) — SAME as ml_stat_arb.py baseline
        pair_scores.sort(key=lambda x: x[3])

        for pair, z, beta, hurst, half_life in pair_scores[:MAX_PAIRS - len(positions)]:
            if shuffle_direction:
                # RANDOM direction instead of fading z-score
                direction = rng.choice([-1, 1])
            else:
                direction = -1 if z > 0 else 1  # Fade the z-score
            positions[pair] = {
                'direction': direction,
                'entry_day': day,
                'entry_z': z
            }

    return daily_returns[start_day:], trade_log, start_day


def compute_metrics(daily_returns):
    """Compute risk-adjusted metrics — SAME as ml_stat_arb.py."""
    if len(daily_returns) == 0 or np.std(daily_returns) == 0:
        return {'sharpe': 0, 'sortino': 0, 'cagr': 0, 'maxdd': 0, 'wr': 0, 'pf': 0, 'n_days': 0, 'n_years': 0, 'ann_vol': 0}

    equity = (1 + pd.Series(daily_returns)).cumprod()
    n_years = len(daily_returns) / 252

    ann_ret = equity.iloc[-1] ** (1 / n_years) - 1 if n_years > 0 else 0
    ann_vol = np.std(daily_returns) * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    downside = daily_returns[daily_returns < 0]
    down_vol = np.std(downside) * np.sqrt(252) if len(downside) > 0 else 1
    sortino = ann_ret / down_vol if down_vol > 0 else 0

    peak = equity.cummax()
    dd = (equity - peak) / peak
    maxdd = dd.min()

    wins = daily_returns[daily_returns > 0]
    losses = daily_returns[daily_returns < 0]
    wr = len(wins) / max(len(wins) + len(losses), 1)
    pf = abs(wins.sum() / losses.sum()) if len(losses) > 0 and losses.sum() != 0 else 0

    return {
        'sharpe': sharpe,
        'sortino': sortino,
        'cagr': ann_ret,
        'maxdd': maxdd,
        'wr': wr,
        'pf': pf,
        'n_days': len(daily_returns),
        'n_years': n_years,
        'ann_vol': ann_vol
    }


def main():
    t0 = time.time()
    print("=" * 70)
    print("STAT ARB BASELINE — ADVERSARIAL VALIDATION")
    print("Pure z-score pairs trading: is the Sharpe ~0.807 genuine alpha?")
    print("=" * 70)
    sys.stdout.flush()

    # ── Download data ──
    close = download_data()

    # ── 1. Reproduce baseline ──
    print("\n[1/7] Running baseline (pure z-score, no ML)...")
    sys.stdout.flush()
    baseline_rets, baseline_trades, start_day = run_backtest(close)
    bm = compute_metrics(baseline_rets)
    real_sharpe = bm['sharpe']

    print(f"\n  Baseline Results:")
    print(f"    Sharpe:  {bm['sharpe']:.3f}")
    print(f"    Sortino: {bm['sortino']:.3f}")
    print(f"    CAGR:    {bm['cagr']*100:.1f}%")
    print(f"    MaxDD:   {bm['maxdd']*100:.1f}%")
    print(f"    WR:      {bm['wr']*100:.1f}%")
    print(f"    PF:      {bm['pf']:.3f}")
    print(f"    Trades:  {len(baseline_trades)}")
    print(f"    Years:   {bm['n_years']:.1f}")
    sys.stdout.flush()

    if baseline_trades:
        trade_df = pd.DataFrame(baseline_trades)
        print(f"    Avg hold:      {trade_df['days_held'].mean():.1f} days")
        print(f"    WR (trades):   {(trade_df['return'] > 0).mean()*100:.1f}%")
        print(f"    Exit reasons:  {trade_df['exit_reason'].value_counts().to_dict()}")
        sys.stdout.flush()

    # Market correlation
    spy_rets_full = close['SPY'].pct_change().iloc[start_day:start_day + len(baseline_rets)].values
    if len(spy_rets_full) == len(baseline_rets):
        mkt_corr = np.corrcoef(baseline_rets, spy_rets_full)[0, 1]
        print(f"    SPY correlation: {mkt_corr:.3f}")
    else:
        mkt_corr = np.nan
    sys.stdout.flush()

    # ── 2. Direction permutation test ──
    print(f"\n[2/7] Direction permutation test ({N_PERM} shuffles)...")
    print("  Keeps same entry timing + pairs, randomizes long/short direction")
    sys.stdout.flush()
    dir_perm_sharpes = []
    for i in range(N_PERM):
        if i % 10 == 0:
            print(f"    Perm {i}/{N_PERM}...")
            sys.stdout.flush()
        rng = np.random.RandomState(1000 + i)
        perm_rets, _, _ = run_backtest(close, shuffle_direction=True, rng=rng)
        pm = compute_metrics(perm_rets)
        dir_perm_sharpes.append(pm['sharpe'])

    dir_perm_mean = np.mean(dir_perm_sharpes)
    dir_perm_std = np.std(dir_perm_sharpes) if np.std(dir_perm_sharpes) > 0 else 1
    dir_p_value = np.mean([s >= real_sharpe for s in dir_perm_sharpes])
    dir_pass = dir_p_value < 0.05

    print(f"\n  Direction Permutation Result:")
    print(f"    Real Sharpe:   {real_sharpe:.3f}")
    print(f"    Random Sharpe: {dir_perm_mean:.3f} +/- {dir_perm_std:.3f}")
    print(f"    p-value:       {dir_p_value:.3f}")
    print(f"    VERDICT:       {'PASS — fading z-score MATTERS' if dir_pass else 'FAIL — direction is IRRELEVANT'}")
    sys.stdout.flush()

    # ── 3. Timing permutation test ──
    print(f"\n[3/7] Timing permutation test ({N_PERM} shuffles)...")
    print("  Randomizes which days get entries, keeps cointegration filter")
    sys.stdout.flush()
    time_perm_sharpes = []
    for i in range(N_PERM):
        if i % 10 == 0:
            print(f"    Perm {i}/{N_PERM}...")
            sys.stdout.flush()
        rng = np.random.RandomState(2000 + i)
        perm_rets, _, _ = run_backtest(close, shuffle_timing=True, rng=rng)
        pm = compute_metrics(perm_rets)
        time_perm_sharpes.append(pm['sharpe'])

    time_perm_mean = np.mean(time_perm_sharpes)
    time_perm_std = np.std(time_perm_sharpes) if np.std(time_perm_sharpes) > 0 else 1
    time_p_value = np.mean([s >= real_sharpe for s in time_perm_sharpes])
    time_pass = time_p_value < 0.05

    print(f"\n  Timing Permutation Result:")
    print(f"    Real Sharpe:   {real_sharpe:.3f}")
    print(f"    Random Sharpe: {time_perm_mean:.3f} +/- {time_perm_std:.3f}")
    print(f"    p-value:       {time_p_value:.3f}")
    print(f"    VERDICT:       {'PASS — z-score TIMING matters' if time_pass else 'FAIL — any timing works equally well'}")
    sys.stdout.flush()

    # ── 4. Sub-period stability (4 quarters) ──
    print("\n[4/7] Sub-period stability (4 equal quarters)...")
    sys.stdout.flush()
    n = len(baseline_rets)
    quarters = np.array_split(baseline_rets, 4)
    q_sharpes = []
    for qi, q in enumerate(quarters):
        qm = compute_metrics(q)
        q_sharpes.append(qm['sharpe'])
        print(f"    Q{qi+1}: Sharpe {qm['sharpe']:.3f}, CAGR {qm['cagr']*100:.1f}%, WR {qm['wr']*100:.1f}%")
    cv = np.std(q_sharpes) / max(abs(np.mean(q_sharpes)), 0.01)
    subp_pass = cv < 1.0
    pos_quarters = sum(1 for s in q_sharpes if s > 0)
    print(f"    CV: {cv:.3f}, Positive quarters: {pos_quarters}/4")
    print(f"    VERDICT: {'PASS' if subp_pass else 'FAIL'} (CV < 1.0)")
    sys.stdout.flush()

    # ── 5. Outlier robustness (trim top/bottom 5%) ──
    print("\n[5/7] Outlier robustness (trim top/bottom 5% of daily returns)...")
    sys.stdout.flush()
    sorted_rets = np.sort(baseline_rets)
    trim_n = max(1, int(len(sorted_rets) * 0.05))
    trimmed = sorted_rets[trim_n:-trim_n]
    tm = compute_metrics(trimmed)
    trimmed_sharpe = tm['sharpe']
    degradation = (real_sharpe - trimmed_sharpe) / max(abs(real_sharpe), 0.01)

    print(f"    Full Sharpe:    {real_sharpe:.3f}")
    print(f"    Trimmed Sharpe: {trimmed_sharpe:.3f}")
    print(f"    Degradation:    {degradation*100:.1f}%")
    outlier_pass = abs(degradation) < 0.50
    print(f"    VERDICT: {'PASS' if outlier_pass else 'FAIL'} (|degradation| < 50%)")
    sys.stdout.flush()

    # ── 6. R1 Regime check (green vs red days based on SPY) ──
    print("\n[6/7] R1 Regime check (green/red SPY days)...")
    sys.stdout.flush()
    spy_data = close['SPY'].iloc[start_day:start_day + len(baseline_rets)]
    spy_daily = spy_data.pct_change()

    green_mask = spy_daily > 0
    red_mask = spy_daily < 0

    min_len = min(len(green_mask), len(baseline_rets))
    green_rets = baseline_rets[:min_len][green_mask.values[:min_len]]
    red_rets = baseline_rets[:min_len][red_mask.values[:min_len]]

    green_sharpe = compute_metrics(green_rets)['sharpe'] if len(green_rets) > 10 else 0
    red_sharpe = compute_metrics(red_rets)['sharpe'] if len(red_rets) > 10 else 0

    gap = abs(green_sharpe - red_sharpe) / max(abs(green_sharpe), abs(red_sharpe), 0.01)
    r1_pass = gap < 0.50

    print(f"    Green day Sharpe: {green_sharpe:.3f} ({green_mask.sum()} days)")
    print(f"    Red day Sharpe:   {red_sharpe:.3f} ({red_mask.sum()} days)")
    print(f"    Gap:              {gap:.3f}")
    print(f"    VERDICT: {'PASS — regime-agnostic' if r1_pass else 'FAIL — regime-dependent'}")
    sys.stdout.flush()

    # ── 7. Combined direction + timing permutation (double shuffle) ──
    print(f"\n[7/7] Double shuffle (direction + timing, 50 perms)...")
    sys.stdout.flush()
    double_perm_sharpes = []
    for i in range(50):
        if i % 10 == 0:
            print(f"    Perm {i}/50...")
            sys.stdout.flush()
        rng = np.random.RandomState(3000 + i)
        perm_rets, _, _ = run_backtest(close, shuffle_direction=True, shuffle_timing=True, rng=rng)
        pm = compute_metrics(perm_rets)
        double_perm_sharpes.append(pm['sharpe'])

    double_perm_mean = np.mean(double_perm_sharpes)
    double_p_value = np.mean([s >= real_sharpe for s in double_perm_sharpes])

    print(f"    Real Sharpe:   {real_sharpe:.3f}")
    print(f"    Random Sharpe: {double_perm_mean:.3f}")
    print(f"    p-value:       {double_p_value:.3f}")
    sys.stdout.flush()

    # ── Final Verdict ──
    elapsed = time.time() - t0
    gates = {
        'direction_perm': dir_pass,
        'timing_perm': time_pass,
        'subperiod': subp_pass,
        'outlier': outlier_pass,
        'r1_regime': r1_pass,
    }
    gates_passed = sum(gates.values())
    total_gates = len(gates)

    print(f"\n{'='*70}")
    print(f"FINAL VERDICT: {gates_passed}/{total_gates} adversarial gates")
    print(f"  Direction perm: {'PASS' if dir_pass else 'FAIL'} (p={dir_p_value:.3f}) — does fading z-score matter?")
    print(f"  Timing perm:    {'PASS' if time_pass else 'FAIL'} (p={time_p_value:.3f}) — does z-score timing matter?")
    print(f"  Sub-period:     {'PASS' if subp_pass else 'FAIL'} (CV={cv:.3f})")
    print(f"  Outlier:        {'PASS' if outlier_pass else 'FAIL'} (deg={degradation*100:.1f}%)")
    print(f"  R1 Regime:      {'PASS' if r1_pass else 'FAIL'} (gap={gap:.3f})")
    print(f"{'='*70}")
    print(f"\nINTERPRETATION:")
    if dir_pass and time_pass:
        print("  STRONG: Both direction and timing matter — genuine mean-reversion alpha")
    elif dir_pass and not time_pass:
        print("  MODERATE: Direction matters but timing doesn't — z-score direction has edge,")
        print("  but you could enter at any time with cointegrated pairs and still profit")
    elif not dir_pass and time_pass:
        print("  WEAK: Timing matters but direction doesn't — the z-score identifies good")
        print("  entry points, but the spread would revert regardless of your direction")
    else:
        print("  NONE: Neither direction nor timing matters — likely market exposure artifact")

    if not r1_pass:
        print("  WARNING: Regime-dependent — edge may only work in one market regime")
    if not outlier_pass:
        print("  WARNING: Outlier-dependent — a few extreme days drive the returns")
    if not subp_pass:
        print("  WARNING: Unstable across sub-periods — may have structural breaks")

    print(f"\nRuntime: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    sys.stdout.flush()

    # ── Save results ──
    results = {
        'baseline': {k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v
                     for k, v in bm.items()},
        'market_correlation': float(mkt_corr) if not np.isnan(mkt_corr) else None,
        'n_trades': len(baseline_trades),
        'direction_perm': {
            'real_sharpe': float(real_sharpe),
            'perm_mean': float(dir_perm_mean),
            'perm_std': float(dir_perm_std),
            'p_value': float(dir_p_value),
            'pass': bool(dir_pass),
            'all_sharpes': [float(s) for s in dir_perm_sharpes],
        },
        'timing_perm': {
            'real_sharpe': float(real_sharpe),
            'perm_mean': float(time_perm_mean),
            'perm_std': float(time_perm_std),
            'p_value': float(time_p_value),
            'pass': bool(time_pass),
            'all_sharpes': [float(s) for s in time_perm_sharpes],
        },
        'double_perm': {
            'perm_mean': float(double_perm_mean),
            'p_value': float(double_p_value),
        },
        'subperiod': {
            'quarter_sharpes': [float(s) for s in q_sharpes],
            'cv': float(cv),
            'pass': bool(subp_pass),
        },
        'outlier': {
            'full_sharpe': float(real_sharpe),
            'trimmed_sharpe': float(trimmed_sharpe),
            'degradation_pct': float(degradation * 100),
            'pass': bool(outlier_pass),
        },
        'r1_regime': {
            'green_sharpe': float(green_sharpe),
            'red_sharpe': float(red_sharpe),
            'gap': float(gap),
            'pass': bool(r1_pass),
        },
        'gates_passed': gates_passed,
        'total_gates': total_gates,
        'runtime_sec': elapsed,
        'timestamp': datetime.now().isoformat(),
    }

    with open(f'{OUTPUT}/adversarial_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save daily returns
    np.save(f'{OUTPUT}/baseline_daily_returns.npy', baseline_rets)

    # Save permutation distributions
    np.save(f'{OUTPUT}/direction_perm_sharpes.npy', np.array(dir_perm_sharpes))
    np.save(f'{OUTPUT}/timing_perm_sharpes.npy', np.array(time_perm_sharpes))

    print(f"\nResults saved to {OUTPUT}/")
    sys.stdout.flush()
    return results


if __name__ == '__main__':
    main()
