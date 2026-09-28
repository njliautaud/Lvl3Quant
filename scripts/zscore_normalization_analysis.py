"""
Z-Score Normalization Analysis
===============================
Tests 5 z-score normalization methods on existing CNN WF stacked predictions
to find the best approach for consistent, high-quality signal generation.

Methods:
  1. INTRADAY rolling (current): z = (pred - rolling_mean_today) / rolling_std_today, window=1000
  2. HISTORICAL: z = (pred - OOT_mean) / OOT_std  (prior OOT days as reference)
  3. EXPANDING: z = (pred - lifetime_mean) / lifetime_std  (grows over time)
  4. MULTI-SCALE: 0.5 * z_fast(1000 bars) + 0.5 * z_slow(10000 bars)
  5. QUANTILE: rolling percentile rank, mapped to [-3, +3]

Usage (on Jupiter):
  python3 zscore_normalization_analysis.py

Data notes:
  - NPZ files: YYYY-MM-DD_<variant>.npz, key='predictions', shape=(234000,)
  - 106 variants per date; we use the canonical conv2.0_vol70 (closest to live card config)
  - 234000 = full walk-forward window; non-zero bars are the OOT bars for that day
  - Mid prices: synthesised from prediction autocorrelation (MBO .dbn.zst not decoded here)
  - OOT range: 2025-12-01 to 2026-03-06 (68 trading days)
"""

import os
import sys
import glob
import json
import re
import numpy as np
from datetime import datetime, date

# ── Config ──────────────────────────────────────────────────────────────────
PREDS_DIR  = "/home/jupiter/Lvl3Quant/data/processed/cnn_wf_stacked_predictions"

# Canonical variant to use (one file per date).
# conv2.0_vol70 = Card4-style config; change to conv1.5_vol0 for wider/looser.
VARIANT     = "book_predstdExit_conv2.0_vol70"

OOT_START   = date(2025, 12, 1)
OOT_END     = date(2026, 3, 6)

# Trading sim params
Z_THRESHOLD  = 1.5        # |z| > this to enter
HOLD_BARS    = 100        # bars to hold after entry (10s bars → ~16 min)
COST_TICKS   = 0.5        # round-trip cost in ES ticks
MIN_STD      = 1e-6       # avoid division by zero

# Rolling windows
FAST_WINDOW  = 1000       # ~2.8 hr of RTH
SLOW_WINDOW  = 10_000     # ~28 hr (~1.2 RTH days)
QUANT_WIN    = 2000       # for quantile z-score

METHODS = ["INTRADAY", "HISTORICAL", "EXPANDING", "MULTISCALE", "QUANTILE"]

# ── Z-score helpers ──────────────────────────────────────────────────────────

def rolling_mean_std_fast(arr, window):
    """Vectorised rolling mean/std via cumsum. O(n)."""
    n      = len(arr)
    arr    = arr.astype(np.float64)
    cum    = np.cumsum(arr)
    cum_sq = np.cumsum(arr ** 2)

    eff = np.minimum(np.arange(1, n + 1), window).astype(np.float64)

    sum_w  = cum.copy()
    sum_sq = cum_sq.copy()
    sum_w[window:]  = cum[window:]  - cum[:-window]
    sum_sq[window:] = cum_sq[window:] - cum_sq[:-window]

    means = sum_w / eff
    var   = np.maximum(sum_sq / eff - means ** 2, 0.0)
    stds  = np.maximum(np.sqrt(var), MIN_STD)
    return means.astype(np.float32), stds.astype(np.float32)


def _ppf_approx(p):
    """Rational approximation to normal PPF (Abramowitz & Stegun 26.2.17)."""
    p = np.clip(p, 0.001, 0.999)
    mask = p < 0.5
    q    = np.where(mask, p, 1.0 - p)
    t    = np.sqrt(-2.0 * np.log(q))
    c    = np.array([2.515517, 0.802853, 0.010328])
    d    = np.array([1.432788, 0.189269, 0.001308])
    num  = c[0] + c[1]*t + c[2]*t*t
    den  = 1.0 + d[0]*t + d[1]*t*t + d[2]*t*t*t
    z    = t - num / den
    return np.where(mask, -z, z)


def quantile_zscore_fast(arr, window):
    """Rolling percentile rank → normal z-score. Approx O(n*window) but vectorised by chunk."""
    n = len(arr)
    z = np.zeros(n, dtype=np.float32)
    for i in range(n):
        start = max(0, i - window + 1)
        chunk = arr[start:i + 1]
        pct   = np.mean(chunk < arr[i])
        z[i]  = float(_ppf_approx(np.array([pct]))[0])
    return np.clip(z, -4.0, 4.0)


def quantile_zscore_vectorised(arr, window, step=50):
    """Faster vectorised approximation: compute every `step` bars, interpolate."""
    n         = len(arr)
    anchors   = list(range(0, n, step)) + [n - 1]
    z_anchors = np.zeros(len(anchors), dtype=np.float32)
    for j, i in enumerate(anchors):
        start = max(0, i - window + 1)
        chunk = arr[start:i + 1]
        pct   = np.mean(chunk < arr[i])
        z_anchors[j] = float(_ppf_approx(np.array([pct]))[0])
    # Interpolate to full resolution
    z_full = np.interp(np.arange(n), anchors, z_anchors).astype(np.float32)
    return np.clip(z_full, -4.0, 4.0)


# ── Trade simulation ─────────────────────────────────────────────────────────

def simulate_trades(z_scores, mid_prices, threshold=Z_THRESHOLD, hold_bars=HOLD_BARS):
    """
    Entry: |z| > threshold, not already in trade.
    Direction: long if z > 0, short if z < 0.
    Exit: after exactly hold_bars bars.
    PnL in ticks (1 ES micro tick = 0.25 index points).
    """
    n      = len(z_scores)
    pnls   = []
    in_trade   = False
    entry_idx  = -1
    direction  = 0

    for i in range(n):
        # Check exit first
        if in_trade and (i - entry_idx) >= hold_bars:
            price_chg = mid_prices[i] - mid_prices[entry_idx]
            pnl = direction * price_chg - COST_TICKS
            pnls.append(float(pnl))
            in_trade = False

        # Check entry
        if not in_trade and abs(z_scores[i]) > threshold and i + hold_bars < n:
            direction  = 1 if z_scores[i] > 0 else -1
            entry_idx  = i
            in_trade   = True

    return pnls


def trade_stats(pnls, trading_days):
    if not pnls:
        return dict(trades=0, trades_per_day=0.0, win_rate=0.0,
                    avg_pnl=0.0, sharpe=0.0, total_pnl=0.0,
                    pnl_std=0.0, max_dd=0.0)
    arr = np.array(pnls, dtype=np.float64)
    wins = float(np.sum(arr > 0))
    # Sharpe: annualise assuming all bars are 10s RTH bars
    bars_per_year = 252 * 23400 / HOLD_BARS   # trades per year at this hold
    sharpe_ann    = (np.mean(arr) / max(np.std(arr), MIN_STD)) * np.sqrt(bars_per_year)
    # Max drawdown on cumulative pnl
    cum  = np.cumsum(arr)
    peak = np.maximum.accumulate(cum)
    dd   = float(np.min(cum - peak))
    return dict(
        trades        = len(arr),
        trades_per_day= len(arr) / max(trading_days, 1),
        win_rate      = wins / len(arr),
        avg_pnl       = float(np.mean(arr)),
        pnl_std       = float(np.std(arr)),
        sharpe        = float(sharpe_ann),
        total_pnl     = float(np.sum(arr)),
        max_dd        = dd,
    )


# ── Synthetic mid prices ──────────────────────────────────────────────────────

def synthetic_mid_from_predictions(preds, scale=0.25, noise_frac=0.1):
    """
    Create plausible mid-price series from predictions.
    Assumes predictions are directional signals (pos=up, neg=down).
    We integrate them with added noise to get a random-walk-like price.
    scale: ticks per unit of prediction
    """
    rng    = np.random.default_rng(42)
    noise  = rng.normal(0, scale * noise_frac, len(preds)).astype(np.float32)
    # Smooth predictions to get a drifting mid price
    returns = preds * scale + noise
    return np.cumsum(returns).astype(np.float32)


# ── Load data ────────────────────────────────────────────────────────────────

def load_daily_predictions(verbose=True):
    """
    For each OOT date, load the canonical variant NPZ.
    The NPZ has shape (234000,) = walk-forward window.
    We extract the non-zero segment as the OOT bars for that day.
    Returns: list of (date_str, preds_array) sorted by date.
    """
    all_files = sorted(glob.glob(os.path.join(PREDS_DIR, f"*_{VARIANT}.npz")))
    if not all_files:
        # Fallback: any book_predstd variant
        all_files = sorted(glob.glob(os.path.join(PREDS_DIR, "*book_predstd*.npz")))
        if not all_files:
            raise FileNotFoundError(f"No matching NPZ files in {PREDS_DIR}")
        print(f"  WARNING: Canonical variant '{VARIANT}' not found. Using first available.")

    if verbose:
        print(f"Found {len(all_files)} files matching variant '{VARIANT}'")

    day_data = []
    for fp in all_files:
        m = re.match(r'(\d{4}-\d{2}-\d{2})', os.path.basename(fp))
        if not m:
            continue
        date_str = m.group(1)
        try:
            d = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            continue
        if d < OOT_START or d > OOT_END:
            continue

        try:
            npz   = np.load(fp)
            preds = npz['predictions'].astype(np.float32)
        except Exception as e:
            print(f"  Error loading {os.path.basename(fp)}: {e}")
            continue

        # Extract non-zero segment (the OOT bars for this day)
        nz = np.where(preds != 0)[0]
        if len(nz) < 100:
            if verbose:
                print(f"  {date_str}: too few non-zero bars ({len(nz)}), skipping")
            continue

        oot_preds = preds[nz[0]:nz[-1] + 1]
        day_data.append((date_str, oot_preds))
        if verbose:
            print(f"  {date_str}: {len(oot_preds)} OOT bars  "
                  f"(mean={np.mean(oot_preds):.4f}, std={np.std(oot_preds):.4f})")

    day_data.sort(key=lambda x: x[0])
    return day_data


# ── Z-score computation ───────────────────────────────────────────────────────

def compute_all_zscores(preds_concat, day_boundaries):
    """
    preds_concat  : 1-D float32 array, all OOT bars concatenated in date order
    day_boundaries: list of (start_idx, end_idx) per day

    Returns: dict method_name → z_score_array (same length as preds_concat)
    """
    n   = len(preds_concat)
    res = {}

    # 1. INTRADAY: rolling mean/std reset each day
    print("  [1/5] INTRADAY rolling z-scores...")
    z_intraday = np.zeros(n, dtype=np.float32)
    for start, end in day_boundaries:
        day = preds_concat[start:end]
        m, s = rolling_mean_std_fast(day, FAST_WINDOW)
        z_intraday[start:end] = (day - m) / s
    res["INTRADAY"] = z_intraday

    # 2. HISTORICAL: use all prior OOT days' stats
    print("  [2/5] HISTORICAL z-scores (prior-OOT reference)...")
    z_hist = np.zeros(n, dtype=np.float32)
    prior_preds = []
    for start, end in day_boundaries:
        if prior_preds:
            hist = np.concatenate(prior_preds)
            mu   = np.mean(hist)
            sig  = max(np.std(hist), MIN_STD)
        else:
            mu   = np.mean(preds_concat[start:end])
            sig  = max(np.std(preds_concat[start:end]), MIN_STD)
        z_hist[start:end] = (preds_concat[start:end] - mu) / sig
        prior_preds.append(preds_concat[start:end])
    res["HISTORICAL"] = z_hist

    # 3. EXPANDING: growing lifetime window
    print("  [3/5] EXPANDING z-scores (lifetime window)...")
    z_exp  = np.zeros(n, dtype=np.float32)
    cum    = 0.0
    cum_sq = 0.0
    for i in range(n):
        if i > 0:
            mu  = cum / i
            var = max(cum_sq / i - mu**2, 0.0)
            sig = max(np.sqrt(var), MIN_STD)
        else:
            mu, sig = 0.0, MIN_STD
        z_exp[i] = (preds_concat[i] - mu) / sig
        cum    += float(preds_concat[i])
        cum_sq += float(preds_concat[i])**2
    res["EXPANDING"] = z_exp

    # 4. MULTI-SCALE: 0.5*z_fast + 0.5*z_slow (cross-day rolling)
    print("  [4/5] MULTI-SCALE z-scores (fast+slow rolling, cross-day)...")
    m_fast, s_fast = rolling_mean_std_fast(preds_concat, FAST_WINDOW)
    m_slow, s_slow = rolling_mean_std_fast(preds_concat, SLOW_WINDOW)
    z_fast = (preds_concat - m_fast) / s_fast
    z_slow = (preds_concat - m_slow) / s_slow
    res["MULTISCALE"] = (0.5 * z_fast + 0.5 * z_slow).astype(np.float32)

    # 5. QUANTILE: rolling percentile rank → normal z-score
    print("  [5/5] QUANTILE z-scores (rolling percentile, interpolated)...")
    res["QUANTILE"] = quantile_zscore_vectorised(preds_concat, QUANT_WIN, step=30)

    return res


# ── Main ─────────────────────────────────────────────────────────────────────

def run_analysis():
    print("=" * 68)
    print("Z-SCORE NORMALIZATION ANALYSIS")
    print("=" * 68)
    print(f"OOT range   : {OOT_START} → {OOT_END}")
    print(f"Variant     : {VARIANT}")
    print(f"Threshold   : |z| > {Z_THRESHOLD}")
    print(f"Hold bars   : {HOLD_BARS}  (~{HOLD_BARS*10//60} min at 10s/bar)")
    print(f"Fast window : {FAST_WINDOW} bars (~{FAST_WINDOW*10//3600:.1f} hr)")
    print(f"Slow window : {SLOW_WINDOW} bars (~{SLOW_WINDOW*10//3600:.1f} hr)")
    print()

    # Load predictions
    print("Loading predictions...")
    day_data = load_daily_predictions(verbose=True)
    if not day_data:
        print("ERROR: No data loaded. Check PREDS_DIR and VARIANT.")
        sys.exit(1)

    # Build concatenated arrays + day boundaries
    preds_list  = []
    midpx_list  = []
    day_bounds  = []
    idx         = 0

    for date_str, preds in day_data:
        mid = synthetic_mid_from_predictions(preds)
        preds_list.append(preds)
        midpx_list.append(mid)
        day_bounds.append((idx, idx + len(preds)))
        idx += len(preds)

    preds_all    = np.concatenate(preds_list).astype(np.float32)
    midpx_all    = np.concatenate(midpx_list).astype(np.float32)
    trading_days = len(day_data)

    print(f"\nUsable days : {trading_days}")
    print(f"Total bars  : {len(preds_all):,}")
    print(f"\nPrediction stats (OOT segments):")
    print(f"  mean={np.mean(preds_all):.6f}  std={np.std(preds_all):.6f}")
    print(f"  min={np.min(preds_all):.6f}  max={np.max(preds_all):.6f}")
    print()

    # Compute all z-scores
    print("Computing z-scores...")
    zscore_arrays = compute_all_zscores(preds_all, day_bounds)

    # Simulate trades per method
    print("\nSimulating trades...")
    summary = {}
    for method in METHODS:
        z    = zscore_arrays[method]
        pnls = simulate_trades(z, midpx_all)
        stats = trade_stats(pnls, trading_days)
        # Z-score distribution stats
        stats["z_mean"]     = float(np.mean(z))
        stats["z_std"]      = float(np.std(z))
        stats["z_skew"]     = float(_skewness(z))
        stats["z_frac_sig"] = float(np.mean(np.abs(z) > Z_THRESHOLD))
        summary[method] = stats

    # Print results table
    print()
    print("=" * 68)
    print(f"RESULTS  (threshold={Z_THRESHOLD}, hold={HOLD_BARS} bars, cost={COST_TICKS} tick/rt)")
    print("=" * 68)
    header = (f"{'Method':<12} {'Trades':>7} {'T/Day':>6} {'WinRate':>8} "
              f"{'AvgPnL':>8} {'Sharpe':>8} {'MaxDD':>9} {'FracSig':>8}")
    print(header)
    print("-" * 68)

    ranked = sorted(summary.items(), key=lambda x: x[1]["sharpe"], reverse=True)
    for method, s in ranked:
        print(f"{method:<12} {s['trades']:>7d} {s['trades_per_day']:>6.1f} "
              f"{s['win_rate']:>8.1%} {s['avg_pnl']:>8.3f} "
              f"{s['sharpe']:>8.2f} {s['max_dd']:>9.2f} {s['z_frac_sig']:>8.2%}")

    print()
    print("Z-Score Distribution Quality:")
    print(f"{'Method':<12} {'Z_mean':>8} {'Z_std':>8} {'Z_skew':>8} {'Note'}")
    print("-" * 55)
    for method, s in summary.items():
        note = []
        if abs(s["z_std"] - 1.0) > 0.3:
            note.append(f"std≠1 ({s['z_std']:.2f})")
        if abs(s["z_mean"]) > 0.2:
            note.append(f"biased mean ({s['z_mean']:.3f})")
        if abs(s["z_skew"]) > 1.0:
            note.append(f"skewed ({s['z_skew']:.2f})")
        note_str = ", ".join(note) if note else "well-calibrated"
        print(f"{method:<12} {s['z_mean']:>8.4f} {s['z_std']:>8.4f} "
              f"{s['z_skew']:>8.4f}  {note_str}")

    # Winner
    best_method  = ranked[0][0]
    best_stats   = ranked[0][1]
    print()
    print("=" * 68)
    print(f"WINNER: {best_method}  "
          f"(Sharpe={best_stats['sharpe']:.2f}, "
          f"{best_stats['trades_per_day']:.1f} trades/day, "
          f"WR={best_stats['win_rate']:.0%})")
    print("=" * 68)

    print()
    print("INTERPRETATION:")
    for method, s in ranked:
        notes = []
        if s["trades_per_day"] < 0.5:
            notes.append("too sparse — signal almost never fires")
        elif s["trades_per_day"] > 80:
            notes.append("very frequent — may be over-fitted to noise")
        if s["win_rate"] < 0.42:
            notes.append("low win rate — signals may be wrong-way")
        elif s["win_rate"] > 0.65:
            notes.append("very high win rate — check for look-ahead bias")
        if s["z_frac_sig"] < 0.01:
            notes.append("under 1% bars trigger — threshold may be too high")
        if not notes:
            notes.append("signal profile looks reasonable")
        note_str = "; ".join(notes)
        rank_num = next(i+1 for i, (m, _) in enumerate(ranked) if m == method)
        print(f"  #{rank_num} {method:<12}: Sharpe={s['sharpe']:5.2f}, "
              f"{s['trades_per_day']:.1f} t/day, WR={s['win_rate']:.0%}  — {note_str}")

    # Sensitivity sweep for winner
    print()
    print("=" * 68)
    print(f"SENSITIVITY SWEEP — {best_method}")
    print("=" * 68)
    z_best = zscore_arrays[best_method]
    thresholds = [1.0, 1.5, 2.0, 2.5, 3.0]
    holds      = [50, 100, 200, 390]
    print(f"{'Thresh':>8} {'Hold':>6} {'Trades':>7} {'T/Day':>6} {'WinRate':>8} {'Sharpe':>8}")
    print("-" * 55)
    best_sweep_sh  = -999
    best_sweep_row = None
    for th in thresholds:
        for h in holds:
            pnls  = simulate_trades(z_best, midpx_all, threshold=th, hold_bars=h)
            stats = trade_stats(pnls, trading_days)
            sh    = stats["sharpe"]
            print(f"{th:>8.1f} {h:>6d} {stats['trades']:>7d} "
                  f"{stats['trades_per_day']:>6.1f} {stats['win_rate']:>8.1%} {sh:>8.2f}")
            if sh > best_sweep_sh:
                best_sweep_sh  = sh
                best_sweep_row = (th, h, stats)
    if best_sweep_row:
        th, h, s = best_sweep_row
        print(f"\nBest params: threshold={th}, hold={h} bars → "
              f"Sharpe={best_sweep_sh:.2f}, {s['trades_per_day']:.1f} t/day")

    # Save JSON
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "zscore_analysis_results.json")
    out = {
        "run_date"       : str(date.today()),
        "oot_range"      : f"{OOT_START} to {OOT_END}",
        "variant"        : VARIANT,
        "threshold"      : Z_THRESHOLD,
        "hold_bars"      : HOLD_BARS,
        "trading_days"   : trading_days,
        "total_bars"     : int(len(preds_all)),
        "results"        : summary,
        "winner"         : best_method,
        "sensitivity_best": {"threshold": best_sweep_row[0],
                             "hold_bars": best_sweep_row[1],
                             "sharpe"   : best_sweep_sh} if best_sweep_row else None,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved: {out_path}")

    print()
    print("RECOMMENDATION:")
    print(f"  Use {best_method} for z-score normalisation.")
    if best_sweep_row:
        th, h, _ = best_sweep_row
        print(f"  Optimal params: threshold={th}, hold={h} bars (~{h*10//60} min)")
    print()
    print("NEXT STEPS:")
    print("  1. Verify with real mid-prices from decoded MBO .dbn.zst data")
    print("  2. Run Rust fill_sim on top method with realistic fill rates")
    print("  3. Update paper engine signal normalisation code")
    print("  4. Monitor live z-score calibration for first week")

    return summary, best_method, zscore_arrays


def _skewness(arr):
    n = len(arr)
    if n < 3:
        return 0.0
    mu  = np.mean(arr)
    sig = np.std(arr)
    if sig < 1e-10:
        return 0.0
    return float(np.mean(((arr - mu) / sig) ** 3))


if __name__ == "__main__":
    run_analysis()
