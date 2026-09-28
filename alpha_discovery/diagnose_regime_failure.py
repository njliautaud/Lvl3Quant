"""
Regime Failure Diagnosis: WHY does the composite signal fail in Nov 2025?
=========================================================================
Key question: If November is a down month, why can't the signal just go SHORT?

Three hypotheses:
  A) BIAS: Signal is still predicting LONG in Nov (directional bias from training)
  B) NOISE: Signal predicts both directions but both are wrong (decorrelation)
  C) COST: Signal predicts SHORT correctly but wins don't cover spread+commission

This script diagnoses which failure mode is occurring.
"""

import sys
import time
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.stats import spearmanr

TICK = 0.25
TICK_VAL = 12.50
COMM_TICKS = 3.00 / TICK_VAL  # commission in ticks
BARS_PER_SEC = 10  # 100ms bars -> 10 bars per second

LVL3_ROOT = Path('C:/Users/Footb/Documents/Github/Lvl3Quant')
FEAT_CACHE = LVL3_ROOT / 'data' / 'processed' / 'mbo_features_cache'
SIG_DIR = LVL3_ROOT / 'data' / 'processed' / 'signal_predictions'

# Forward return horizon: 600 bars = 60 seconds (1 min real time)
# But with 100ms bars, 600 bars = 60s. The holdout script uses hold=600s = 6000 bars.
# Let's use 6000 bars (= 600s = 10 min) to match the hold time used in sim.
FWD_BARS = 6000  # 10 minutes forward (600 seconds * 10 bars/sec)
Z_THRESHOLD = 2.0  # Minimum z-score to consider a "signal"


def load_data():
    """Load MBO mid-price data for all 100 days."""
    files = sorted(FEAT_CACHE.glob('*_mbo_features.npz'))[:100]
    print(f"Loading {len(files)} days of MBO data...")

    days = {}
    for f in files:
        date = f.stem.replace('_mbo_features', '').replace('mbo_features_', '')
        try:
            data = np.load(str(f))
            feats = data['mbo_features']
            days[date] = {
                'mid': feats[:, 0].astype(np.float32).copy(),
                'spread': feats[:, 1].astype(np.float32).copy(),
                'n_bars': len(feats),
            }
            del feats, data
        except Exception as e:
            print(f"  Skip {date}: {e}")

    print(f"Loaded {len(days)} days")
    return days


def load_composite_signals(days):
    """Build composite signals (mean z-score) exactly as in true_oos_holdout_local.py."""
    sig_names_map = defaultdict(set)
    for f in SIG_DIR.glob('*.npz'):
        parts = f.stem.split('_')
        for i in range(len(parts)):
            if parts[i].startswith('2025-'):
                sig_name = '_'.join(parts[:i])
                date = '_'.join(parts[i:])
                sig_names_map[sig_name].add(date)
                break

    available_sigs = {k: v for k, v in sig_names_map.items() if len(v) >= 80}
    print(f"\nSignals with >=80 days: {len(available_sigs)}")
    for name, dates in sorted(available_sigs.items()):
        print(f"  {name}: {len(dates)} days")

    all_dates = sorted(days.keys())
    composite = {}
    for date in all_dates:
        n_bars = days[date]['n_bars']
        sig_sum = np.zeros(n_bars, dtype=np.float64)
        n_sigs = 0

        for sig_name in available_sigs:
            path = SIG_DIR / f'{sig_name}_{date}.npz'
            if not path.exists():
                continue
            try:
                preds = np.load(str(path))['predictions']
            except Exception:
                continue
            if len(preds) != n_bars:
                preds = preds[:n_bars] if len(preds) > n_bars else np.pad(preds, (0, n_bars - len(preds)))
            sig_sum += preds.astype(np.float64)
            n_sigs += 1

        if n_sigs > 0:
            composite[date] = (sig_sum / n_sigs).astype(np.float32)

    print(f"Composite signals built for {len(composite)} days")
    return composite, available_sigs


def get_month(date_str):
    """Extract month label from date string like 2025-07-14."""
    parts = date_str.split('-')
    month_names = {'07': 'Jul', '08': 'Aug', '09': 'Sep', '10': 'Oct', '11': 'Nov'}
    return month_names.get(parts[1], parts[1])


def compute_forward_returns(mid, fwd_bars):
    """Compute forward returns (in ticks) for each bar."""
    n = len(mid)
    fwd_ret = np.full(n, np.nan, dtype=np.float32)
    valid_end = n - fwd_bars
    if valid_end > 0:
        fwd_ret[:valid_end] = (mid[fwd_bars:] - mid[:valid_end]) / TICK
    return fwd_ret


def main():
    t0 = time.time()
    print("=" * 70)
    print("REGIME FAILURE DIAGNOSIS")
    print("Why can't the composite signal SHORT in down months?")
    print("=" * 70)

    days = load_data()
    composite, available_sigs = load_composite_signals(days)
    all_dates = sorted(days.keys())

    # ---- Per-month containers ----
    months_order = ['Jul', 'Aug', 'Sep', 'Oct', 'Nov']
    monthly = {m: {
        'dates': [],
        'long_signals': 0, 'short_signals': 0,
        'long_correct': 0, 'short_correct': 0,
        'long_pnl_ticks': [], 'short_pnl_ticks': [],
        'all_z': [], 'all_fwd': [],
        'z_abs': [],
        'daily_open_close_ret': [],
        'daily_vol': [],
        'signal_count': 0,
    } for m in months_order}

    print(f"\nAnalyzing {len(all_dates)} days with fwd horizon = {FWD_BARS} bars ({FWD_BARS/BARS_PER_SEC:.0f}s)...")
    print(f"Signal threshold: |z| > {Z_THRESHOLD}")

    for date in all_dates:
        if date not in composite:
            continue

        month = get_month(date)
        if month not in monthly:
            continue

        mid = days[date]['mid']
        spread = days[date]['spread']
        sig = composite[date]
        n = min(len(mid), len(sig))
        mid = mid[:n]
        spread = spread[:n]
        sig = sig[:n]

        monthly[month]['dates'].append(date)

        # Daily open-to-close return (in ticks)
        daily_ret = (mid[-1] - mid[0]) / TICK
        monthly[month]['daily_open_close_ret'].append(daily_ret)

        # Daily realized volatility (std of 1-second returns in ticks)
        sec_bars = BARS_PER_SEC  # 10 bars = 1 second
        n_secs = n // sec_bars
        if n_secs > 1:
            sec_prices = mid[::sec_bars][:n_secs]
            sec_rets = np.diff(sec_prices) / TICK
            daily_rvol = np.std(sec_rets)
            monthly[month]['daily_vol'].append(daily_rvol)

        # Forward returns
        fwd_ret = compute_forward_returns(mid, FWD_BARS)

        # Only look at bars where we have valid forward returns AND strong signal
        valid_mask = ~np.isnan(fwd_ret)
        strong_long = (sig > Z_THRESHOLD) & valid_mask
        strong_short = (sig < -Z_THRESHOLD) & valid_mask
        any_strong = strong_long | strong_short

        # Collect z-scores and forward returns for IC calculation
        if valid_mask.sum() > 100:
            # Subsample to avoid memory issues (every 100th bar)
            subsample = np.arange(0, n, 100)
            subsample = subsample[valid_mask[subsample]]
            if len(subsample) > 0:
                monthly[month]['all_z'].extend(sig[subsample].tolist())
                monthly[month]['all_fwd'].extend(fwd_ret[subsample].tolist())

        # Average absolute z-score
        monthly[month]['z_abs'].extend(np.abs(sig[valid_mask]).tolist()[:1000])  # cap for memory

        # Long signals
        n_long = strong_long.sum()
        if n_long > 0:
            monthly[month]['long_signals'] += n_long
            long_fwd = fwd_ret[strong_long]
            long_correct = (long_fwd > 0).sum()
            monthly[month]['long_correct'] += long_correct
            # PnL accounting: forward return minus spread and commission
            long_z = sig[strong_long]
            long_spread = spread[strong_long]
            long_pnl = long_fwd - long_spread / TICK - COMM_TICKS
            monthly[month]['long_pnl_ticks'].extend(long_pnl.tolist()[:500])

        # Short signals
        n_short = strong_short.sum()
        if n_short > 0:
            monthly[month]['short_signals'] += n_short
            short_fwd = fwd_ret[strong_short]
            short_correct = (short_fwd < 0).sum()  # correct short = price went down
            monthly[month]['short_correct'] += short_correct
            short_z = sig[strong_short]
            short_spread = spread[strong_short]
            # Short PnL: negative forward return (we sold) minus costs
            short_pnl = -short_fwd - short_spread / TICK - COMM_TICKS
            monthly[month]['short_pnl_ticks'].extend(short_pnl.tolist()[:500])

        monthly[month]['signal_count'] += n_long + n_short

    # ---- Print Results ----
    print("\n" + "=" * 70)
    print("MONTHLY MARKET REGIME")
    print("=" * 70)

    for m in months_order:
        d = monthly[m]
        n_days = len(d['dates'])
        if n_days == 0:
            continue
        total_ret = sum(d['daily_open_close_ret'])
        avg_ret = np.mean(d['daily_open_close_ret'])
        avg_vol = np.mean(d['daily_vol']) if d['daily_vol'] else 0
        pct_up = np.mean([r > 0 for r in d['daily_open_close_ret']]) * 100
        print(f"  {m} 2025: {n_days} days | Total return: {total_ret:+.0f} ticks | "
              f"Avg daily: {avg_ret:+.1f} ticks | "
              f"Avg 1s rvol: {avg_vol:.3f} ticks | {pct_up:.0f}% up days")

    print("\n" + "=" * 70)
    print("SIGNAL DIRECTION BREAKDOWN (|z| > {:.1f})".format(Z_THRESHOLD))
    print("=" * 70)
    print(f"  {'Month':<6} {'Long Sigs':>10} {'Short Sigs':>11} {'L/(L+S)':>8} {'Long Bias?':>11}")
    print(f"  {'-'*6} {'-'*10} {'-'*11} {'-'*8} {'-'*11}")

    for m in months_order:
        d = monthly[m]
        total = d['long_signals'] + d['short_signals']
        if total == 0:
            continue
        l_pct = d['long_signals'] / total * 100
        bias = "YES - LONG" if l_pct > 60 else ("YES - SHORT" if l_pct < 40 else "balanced")
        print(f"  {m:<6} {d['long_signals']:>10,} {d['short_signals']:>11,} {l_pct:>7.1f}% {bias:>11}")

    print("\n" + "=" * 70)
    print("DIRECTIONAL ACCURACY (does signal predict direction correctly?)")
    print("=" * 70)
    print(f"  {'Month':<6} {'Long WR':>8} {'Short WR':>9} {'Combined':>9} {'vs 50%':>7}")
    print(f"  {'-'*6} {'-'*8} {'-'*9} {'-'*9} {'-'*7}")

    for m in months_order:
        d = monthly[m]
        long_wr = d['long_correct'] / d['long_signals'] * 100 if d['long_signals'] > 0 else 0
        short_wr = d['short_correct'] / d['short_signals'] * 100 if d['short_signals'] > 0 else 0
        total_correct = d['long_correct'] + d['short_correct']
        total_sigs = d['long_signals'] + d['short_signals']
        combined_wr = total_correct / total_sigs * 100 if total_sigs > 0 else 0
        edge = combined_wr - 50
        print(f"  {m:<6} {long_wr:>7.1f}% {short_wr:>8.1f}% {combined_wr:>8.1f}% {edge:>+6.1f}%")

    print("\n" + "=" * 70)
    print("INFORMATION COEFFICIENT (rank correlation: signal z vs fwd return)")
    print("=" * 70)
    print(f"  {'Month':<6} {'IC':>7} {'t-stat':>8} {'N samples':>10} {'Interpretation':>20}")
    print(f"  {'-'*6} {'-'*7} {'-'*8} {'-'*10} {'-'*20}")

    for m in months_order:
        d = monthly[m]
        if len(d['all_z']) < 100:
            print(f"  {m:<6} {'N/A':>7} {'N/A':>8} {len(d['all_z']):>10}")
            continue
        z_arr = np.array(d['all_z'])
        fwd_arr = np.array(d['all_fwd'])
        ic, pval = spearmanr(z_arr, fwd_arr)
        n = len(z_arr)
        t_stat = ic * np.sqrt(n - 2) / np.sqrt(1 - ic**2) if abs(ic) < 1 else 0
        if abs(ic) < 0.005:
            interp = "ZERO (noise)"
        elif ic > 0.02:
            interp = "POSITIVE (signal works)"
        elif ic > 0:
            interp = "WEAK POSITIVE"
        elif ic > -0.02:
            interp = "WEAK NEGATIVE"
        else:
            interp = "NEGATIVE (inverted!)"
        print(f"  {m:<6} {ic:>+7.4f} {t_stat:>+8.1f} {n:>10,} {interp:>20}")

    print("\n" + "=" * 70)
    print("AVERAGE SIGNAL STRENGTH (mean |z-score| of all bars)")
    print("=" * 70)

    for m in months_order:
        d = monthly[m]
        if not d['z_abs']:
            continue
        avg_z = np.mean(d['z_abs'])
        pct_strong = np.mean([z > Z_THRESHOLD for z in d['z_abs']]) * 100
        print(f"  {m}: avg |z| = {avg_z:.3f} | {pct_strong:.1f}% of bars have |z| > {Z_THRESHOLD}")

    print("\n" + "=" * 70)
    print("PNL BREAKDOWN: LONG vs SHORT (after costs, in ticks)")
    print("=" * 70)
    print(f"  {'Month':<6} {'Long avg':>9} {'Long med':>9} {'Short avg':>10} {'Short med':>10} {'Long N':>7} {'Short N':>8}")
    print(f"  {'-'*6} {'-'*9} {'-'*9} {'-'*10} {'-'*10} {'-'*7} {'-'*8}")

    for m in months_order:
        d = monthly[m]
        l_avg = np.mean(d['long_pnl_ticks']) if d['long_pnl_ticks'] else 0
        l_med = np.median(d['long_pnl_ticks']) if d['long_pnl_ticks'] else 0
        s_avg = np.mean(d['short_pnl_ticks']) if d['short_pnl_ticks'] else 0
        s_med = np.median(d['short_pnl_ticks']) if d['short_pnl_ticks'] else 0
        print(f"  {m:<6} {l_avg:>+9.2f} {l_med:>+9.2f} {s_avg:>+10.2f} {s_med:>+10.2f} "
              f"{len(d['long_pnl_ticks']):>7,} {len(d['short_pnl_ticks']):>8,}")

    print("\n" + "=" * 70)
    print("REALIZED VOLATILITY vs SIGNAL EDGE")
    print("=" * 70)
    print(f"  {'Month':<6} {'1s RVol':>8} {'Avg |z|':>8} {'IC':>7} {'Edge/Vol':>9}")
    print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*7} {'-'*9}")

    for m in months_order:
        d = monthly[m]
        avg_vol = np.mean(d['daily_vol']) if d['daily_vol'] else 0
        avg_z = np.mean(d['z_abs']) if d['z_abs'] else 0
        if len(d['all_z']) >= 100:
            ic, _ = spearmanr(d['all_z'], d['all_fwd'])
        else:
            ic = 0
        ratio = ic / avg_vol if avg_vol > 0 else 0
        print(f"  {m:<6} {avg_vol:>8.4f} {avg_z:>8.3f} {ic:>+7.4f} {ratio:>+9.4f}")

    # ---- THE DIAGNOSIS ----
    print("\n" + "=" * 70)
    print("DIAGNOSIS: WHY CAN'T THE SIGNAL SHORT IN DOWN MONTHS?")
    print("=" * 70)

    # Check hypothesis A: Long bias
    nov = monthly.get('Nov', None)
    oct = monthly.get('Oct', None)

    if nov and (nov['long_signals'] + nov['short_signals']) > 0:
        nov_long_pct = nov['long_signals'] / (nov['long_signals'] + nov['short_signals']) * 100
        if nov_long_pct > 60:
            print("\n  [A] LONG BIAS: YES - Signal is {:.0f}% long in Nov despite down market".format(nov_long_pct))
            print("      -> The model was trained on predominantly up-market data")
            print("      -> It hasn't learned to predict downward moves")
        elif nov_long_pct < 40:
            print("\n  [A] LONG BIAS: NO - Signal is actually more SHORT in Nov ({:.0f}% long)".format(nov_long_pct))
        else:
            print("\n  [A] LONG BIAS: NO - Signal is balanced in Nov ({:.0f}% long)".format(nov_long_pct))

    # Check hypothesis B: Noise / decorrelation
    if nov and len(nov['all_z']) >= 100:
        nov_ic, _ = spearmanr(nov['all_z'], nov['all_fwd'])
        jul_ic, _ = spearmanr(monthly['Jul']['all_z'], monthly['Jul']['all_fwd']) if len(monthly['Jul']['all_z']) >= 100 else (0, 0)

        if abs(nov_ic) < 0.005:
            print(f"\n  [B] NOISE/DECORRELATION: YES - Nov IC = {nov_ic:+.4f} (essentially zero)")
            print(f"      -> Signal predictions are RANDOM in Nov")
            print(f"      -> Compare to Jul IC = {jul_ic:+.4f}")
            print(f"      -> The microstructure patterns learned in Jul-Sep don't exist in Oct-Nov")
        elif nov_ic < -0.005:
            print(f"\n  [B] SIGNAL INVERSION: YES - Nov IC = {nov_ic:+.4f} (NEGATIVE)")
            print(f"      -> Signal is ANTI-predictive in Nov: it predicts the WRONG direction")
            print(f"      -> Compare to Jul IC = {jul_ic:+.4f}")
        else:
            print(f"\n  [B] NOISE: NO - Nov IC = {nov_ic:+.4f} still positive")
            print(f"      -> Signal still has some predictive power in Nov")

    # Check hypothesis C: Cost problem
    if nov:
        nov_short_pnl = nov['short_pnl_ticks']
        nov_long_pnl = nov['long_pnl_ticks']

        # Gross (before costs) vs net
        if nov_short_pnl:
            avg_spread = 1.0  # typical spread in ticks
            gross_short = np.mean(nov_short_pnl) + COMM_TICKS + avg_spread
            net_short = np.mean(nov_short_pnl)
            print(f"\n  [C] COST PROBLEM (Nov shorts):")
            print(f"      Gross avg short PnL: {gross_short:+.2f} ticks")
            print(f"      Net avg short PnL:   {net_short:+.2f} ticks")
            print(f"      Costs eat: {COMM_TICKS + avg_spread:.2f} ticks per trade")
            if gross_short > 0 and net_short < 0:
                print(f"      -> YES: Shorts are gross-positive but costs kill them")
            elif gross_short < 0:
                print(f"      -> NO: Shorts are gross-NEGATIVE even before costs")
            else:
                print(f"      -> NO: Shorts are net-positive")

    # Final verdict
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    # Compare ICs across months
    ics = {}
    for m in months_order:
        d = monthly[m]
        if len(d['all_z']) >= 100:
            ic, _ = spearmanr(d['all_z'], d['all_fwd'])
            ics[m] = ic

    if ics:
        good_months = ['Jul', 'Aug', 'Sep']
        bad_months = ['Oct', 'Nov']
        good_ic = np.mean([ics.get(m, 0) for m in good_months if m in ics])
        bad_ic = np.mean([ics.get(m, 0) for m in bad_months if m in ics])

        print(f"\n  Good months (Jul-Sep) avg IC: {good_ic:+.4f}")
        print(f"  Bad months (Oct-Nov) avg IC:  {bad_ic:+.4f}")
        print(f"  IC degradation: {good_ic - bad_ic:+.4f}")

        if bad_ic < 0:
            print("\n  --> REGIME SHIFT: Signal is INVERTED in Oct-Nov.")
            print("      The microstructure patterns that predict UP in Jul-Sep")
            print("      actually predict DOWN in Oct-Nov (or vice versa).")
            print("      This is classic non-stationarity / regime dependence.")
            print("      Even if you flip the signal, you'd need to KNOW you're in")
            print("      a different regime -- which is the real unsolved problem.")
        elif abs(bad_ic) < 0.005:
            print("\n  --> DECORRELATION: Signal becomes pure noise in Oct-Nov.")
            print("      The microstructure features the model learned are")
            print("      regime-specific and simply don't exist in the new regime.")
            print("      No amount of long/short flipping can fix noise.")
        else:
            print("\n  --> PARTIAL DEGRADATION: Signal weakens but doesn't die.")
            print("      There may be a cost/volatility issue rather than pure noise.")

    # Additional: per-month win rates after costs
    print("\n" + "=" * 70)
    print("TRADE-LEVEL WIN RATES (after costs)")
    print("=" * 70)
    print(f"  {'Month':<6} {'Long WR':>8} {'Short WR':>9} {'All WR':>7}")
    print(f"  {'-'*6} {'-'*8} {'-'*9} {'-'*7}")

    for m in months_order:
        d = monthly[m]
        l_wr = np.mean([p > 0 for p in d['long_pnl_ticks']]) * 100 if d['long_pnl_ticks'] else 0
        s_wr = np.mean([p > 0 for p in d['short_pnl_ticks']]) * 100 if d['short_pnl_ticks'] else 0
        all_pnl = d['long_pnl_ticks'] + d['short_pnl_ticks']
        a_wr = np.mean([p > 0 for p in all_pnl]) * 100 if all_pnl else 0
        print(f"  {m:<6} {l_wr:>7.1f}% {s_wr:>8.1f}% {a_wr:>6.1f}%")

    elapsed = time.time() - t0
    print(f"\n  Elapsed: {elapsed:.0f}s")


if __name__ == '__main__':
    main()
