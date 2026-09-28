#!/usr/bin/env python3
"""
Trade Tape Imbalance v1 - novel microstructure signal independent of v3.4.2.

Hypothesis: aggressor-side trade-tape flow (sweeps vs absorptive trades) is a
distinct signal not captured by BBO-level features. Tests whether tape
imbalance predicts realized forward 5s/10s mid-price moves on ES futures.

Data: processed MBO event NPZs in data/processed/mbo_events/. Trades are
filtered by event_type_id == 3. side_id 0 = bid lift (buy aggressor),
side_id 1 = ask hit (sell aggressor). qty_log = log(1+size) of the trade.
Realized forward returns come from labels_10s / labels_5s (signed integer
tick moves to mid h-seconds forward).

OOT range: 2026-02-23 .. 2026-04-29 (weekday trading days).

Deploy gates (HC #428):
  ACCEPT  if pooled_net >= +0.10t and profit_days >= 0.60 and Sharpe >= 0.30
          and day_conc <= 0.70 and regime_imbalance <= 0.50
  PARTIAL if pooled_net >= +0.05t and profit_days >= 0.55
  REJECT  otherwise
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT = Path('/home/jupiter/Lvl3Quant')
MBO_DIR = ROOT / 'data' / 'processed' / 'mbo_events'
OUT_DIR = ROOT / 'output' / 'trade_tape_imbalance_v1'
OUT_DIR.mkdir(parents=True, exist_ok=True)

OOT_START = '20260223'
OOT_END = '20260429'

HORIZONS_NS = {
    '1s':  1_000_000_000,
    '5s':  5_000_000_000,
    '10s': 10_000_000_000,
}

# Cost model (HC #428 / cost constants)
PASSIVE_COST_TICKS = 0.376   # commission only - passive limit
MARKET_COST_TICKS  = 1.376   # commission + 1 tick spread crossing

# Sample policy: trim per-date number of evaluated events for tractability.
MAX_EVENTS_PER_DAY_FOR_IC = 200_000   # used for Spearman per-day
TOP_FRACTION_FOR_PNL = 0.05            # top 5 % by feature magnitude


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    ts = datetime.now().strftime('%H:%M:%S')
    print(f'[{ts}] {msg}', flush=True)


def list_oot_dates() -> list[str]:
    dates = []
    for p in sorted(MBO_DIR.glob('*_mbo_events.npz')):
        d = p.name[:8]
        if OOT_START <= d <= OOT_END:
            dt = datetime.strptime(d, '%Y%m%d')
            if dt.weekday() < 5:
                dates.append(d)
    return dates


def rolling_sum_lookback(times_ns: np.ndarray, values: np.ndarray,
                         horizon_ns: int) -> np.ndarray:
    """For each event i, sum of values[j] for j where
    times_ns[j] in (times_ns[i] - horizon_ns, times_ns[i]] (inclusive of i).
    Fully vectorised: cumulative sum + searchsorted for the lower bound."""
    n = len(times_ns)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    csum = np.concatenate(([0.0], np.cumsum(values.astype(np.float64))))
    cutoffs = times_ns - horizon_ns
    # We want lo = first index where times_ns[lo] > cutoff.
    lo = np.searchsorted(times_ns, cutoffs, side='right')
    # Sum over [lo, i] inclusive = csum[i+1] - csum[lo]
    idx_hi = np.arange(1, n + 1)
    return csum[idx_hi] - csum[lo]


def rolling_max_streak_lookback(times_ns: np.ndarray, sides: np.ndarray,
                                horizon_ns: int) -> np.ndarray:
    """Trailing same-side run length ending at event i, truncated to the
    past `horizon_ns`. Vectorised: run_start[i] is the index of the first
    trade in the current run; the window cutoff trims it via searchsorted."""
    n = len(times_ns)
    if n == 0:
        return np.zeros(0, dtype=np.int32)
    # run_start[i] = index where the current run began
    changes = np.concatenate(([True], sides[1:] != sides[:-1]))
    run_start = np.where(changes,
                         np.arange(n, dtype=np.int64),
                         -1)
    # forward-fill
    mask = run_start >= 0
    last = np.maximum.accumulate(np.where(mask, np.arange(n), -1))
    run_start = run_start[last]
    cutoffs = times_ns - horizon_ns
    # First index where ts > cutoff
    window_start = np.searchsorted(times_ns, cutoffs, side='right')
    effective_start = np.maximum(run_start, window_start)
    out = (np.arange(n) - effective_start + 1).astype(np.int32)
    out[out < 0] = 0
    return out


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 50:
        return np.nan
    mx = np.isfinite(x) & np.isfinite(y)
    if mx.sum() < 50:
        return np.nan
    # Spearman is insensitive to scale; if x is constant return nan
    if np.std(x[mx]) == 0 or np.std(y[mx]) == 0:
        return np.nan
    r, _ = spearmanr(x[mx], y[mx])
    return float(r)


# ---------------------------------------------------------------------------
# Per-date feature & target builder
# ---------------------------------------------------------------------------
@dataclass
class DayFeatures:
    date: str
    n_trades: int
    feature_names: list
    features: np.ndarray         # shape (n_trades, n_features)
    target_5s: np.ndarray        # ticks
    target_10s: np.ndarray       # ticks


def build_day_features(npz_path: Path) -> DayFeatures | None:
    d = np.load(npz_path, allow_pickle=True)
    events = d['events']                  # (N, 6) float32
    ts = d['timestamps']                  # (N,) int64 ns
    l5 = d['labels_5s']
    l10 = d['labels_10s']

    # Filter trades
    tmask = events[:, 1] == 3.0
    if tmask.sum() < 10_000:
        return None
    t_ts = ts[tmask]
    t_side = events[tmask, 2].astype(np.int8)     # 0=B (buy agg), 1=A (sell agg)
    t_qty_log = events[tmask, 4].astype(np.float32)
    # Recover raw size from log1p inverse: qty_log = log(size+1)
    t_size = np.expm1(t_qty_log).astype(np.float64)

    # Filter out side==2 (none/N) if any
    keep = (t_side == 0) | (t_side == 1)
    t_ts = t_ts[keep]
    t_side = t_side[keep]
    t_size = t_size[keep]

    # Aggressor sign: +1 for buy aggressor, -1 for sell aggressor
    sign = np.where(t_side == 0, 1.0, -1.0)
    signed_size = sign * t_size

    # Normalize size by daily median (kept for reference / reporting)
    med_size = float(np.median(t_size))
    if med_size <= 0:
        med_size = 1.0
    size_z = t_size / med_size

    # Buckets: percentile-based (ES is dominated by 1-lot trades so the
    # original z<1 / z>=3 thresholds collapse). Use:
    #   SMALL  = size <= p50 (absorptive 1-lots and tiny prints)
    #   LARGE  = size >= p90 (sweep candidates - top decile by contract size)
    #   MEDIUM = everything else
    p50 = float(np.percentile(t_size, 50))
    p90 = float(np.percentile(t_size, 90))
    large_mask = t_size >= max(p90, p50 + 1.0)   # avoid degenerate equal-percentile collapse
    small_mask = t_size <= p50

    signed_large = signed_size * large_mask
    signed_small = signed_size * small_mask
    ones = np.ones_like(t_size)
    large_ones = large_mask.astype(np.float64)

    # Sort by timestamp (should already be sorted but guarantee)
    order = np.argsort(t_ts, kind='stable')
    t_ts = t_ts[order]
    sign = sign[order]
    signed_size = signed_size[order]
    signed_large = signed_large[order]
    signed_small = signed_small[order]
    ones = ones[order]
    large_ones = large_ones[order]
    t_size_sorted = t_size[order]

    # Map trade event indices back to absolute label arrays
    abs_idx = np.where(tmask)[0]
    abs_idx = abs_idx[keep][order]
    y5 = l5[abs_idx].astype(np.float64)
    y10 = l10[abs_idx].astype(np.float64)

    feats = {}
    names = []
    for h_name, h_ns in HORIZONS_NS.items():
        buy_minus_sell = rolling_sum_lookback(t_ts, signed_size, h_ns)
        total_size = rolling_sum_lookback(t_ts, t_size_sorted.astype(np.float64), h_ns)
        with np.errstate(invalid='ignore', divide='ignore'):
            tape_imb = np.where(total_size > 0, buy_minus_sell / total_size, 0.0)
        feats[f'tape_imbalance_{h_name}'] = tape_imb
        names.append(f'tape_imbalance_{h_name}')

        large_flow = rolling_sum_lookback(t_ts, signed_large, h_ns)
        feats[f'large_aggressor_flow_{h_name}'] = large_flow
        names.append(f'large_aggressor_flow_{h_name}')

        small_flow = rolling_sum_lookback(t_ts, signed_small, h_ns)
        feats[f'small_aggressor_flow_{h_name}'] = small_flow
        names.append(f'small_aggressor_flow_{h_name}')

        intensity = rolling_sum_lookback(t_ts, ones, h_ns) / (h_ns / 1e9)
        feats[f'trade_intensity_{h_name}'] = intensity
        names.append(f'trade_intensity_{h_name}')

        sweep_cnt = rolling_sum_lookback(t_ts, large_ones, h_ns)
        feats[f'sweep_count_{h_name}'] = sweep_cnt
        names.append(f'sweep_count_{h_name}')

    # Aggressor streak only at 10s (most expensive)
    streak_10s = rolling_max_streak_lookback(t_ts, sign.astype(np.int8),
                                             HORIZONS_NS['10s'])
    feats['mean_aggressor_streak_10s'] = streak_10s.astype(np.float64) * sign
    names.append('mean_aggressor_streak_10s')

    fmat = np.column_stack([feats[n] for n in names]).astype(np.float32)
    return DayFeatures(date=npz_path.name[:8], n_trades=len(t_ts),
                       feature_names=names, features=fmat,
                       target_5s=y5, target_10s=y10)


# ---------------------------------------------------------------------------
# Per-date evaluation
# ---------------------------------------------------------------------------
def evaluate_day(day: DayFeatures) -> dict:
    """Compute per-feature Spearman vs realized 5s/10s, plus top-decile PnL
    for the headline tape_imbalance_10s feature."""
    out = {'date': day.date, 'n_trades': day.n_trades, 'ic': {}}
    # Limit number of points used for Spearman (per-day cap)
    n = day.n_trades
    if n > MAX_EVENTS_PER_DAY_FOR_IC:
        idx = np.linspace(0, n - 1, MAX_EVENTS_PER_DAY_FOR_IC).astype(np.int64)
    else:
        idx = np.arange(n)

    y5 = day.target_5s[idx]
    y10 = day.target_10s[idx]

    for j, name in enumerate(day.feature_names):
        x = day.features[idx, j]
        out['ic'][f'{name}__y5'] = safe_spearman(x, y5)
        out['ic'][f'{name}__y10'] = safe_spearman(x, y10)

    # Top-decile / extreme-tail P&L test for tape_imbalance_10s
    # Long when tape_imb_10s >= p95, short when <= p5
    ti10_idx = day.feature_names.index('tape_imbalance_10s')
    ti10 = day.features[:, ti10_idx].astype(np.float64)
    y10_full = day.target_10s
    valid = np.isfinite(ti10) & np.isfinite(y10_full)
    ti10v = ti10[valid]
    y10v = y10_full[valid]
    if len(ti10v) >= 1000:
        p95 = np.percentile(ti10v, 95)
        p5  = np.percentile(ti10v,  5)
        long_mask  = ti10v >= p95
        short_mask = ti10v <= p5
        # Gross tick move; cost = passive limit on both legs (commission only)
        long_net  = (y10v[long_mask]  - PASSIVE_COST_TICKS).mean() if long_mask.any() else np.nan
        short_net = (-y10v[short_mask] - PASSIVE_COST_TICKS).mean() if short_mask.any() else np.nan
        out['top5pct'] = {
            'n_long': int(long_mask.sum()),
            'n_short': int(short_mask.sum()),
            'long_gross_t': float(y10v[long_mask].mean()) if long_mask.any() else np.nan,
            'short_gross_t': float(-y10v[short_mask].mean()) if short_mask.any() else np.nan,
            'long_net_t': float(long_net),
            'short_net_t': float(short_net),
        }
        # Combined: take both sides
        combined = np.concatenate([y10v[long_mask] - PASSIVE_COST_TICKS,
                                   -y10v[short_mask] - PASSIVE_COST_TICKS])
        out['top5pct']['combined_net_t'] = float(combined.mean()) if len(combined) else np.nan
        out['top5pct']['combined_n'] = int(len(combined))
        out['top5pct']['combined_std_t'] = float(combined.std()) if len(combined) else np.nan
    else:
        out['top5pct'] = None
    return out


# ---------------------------------------------------------------------------
# ES regime classifier (green / red / flat day) using daily mid-price proxy.
# Without external close-to-close, use sum(labels_10s)/N as a coarse proxy:
# instead, use cumulative integer "drift" of mid via sequential delta of an
# integer cumulator: trade-time mid changes. Cleaner: derive day return from
# first and last valid labels using a synthetic mid_t reconstruction.
# Simplest robust proxy: read book_normalized first/last bid+ask if present;
# fall back to label-based directional vote.
# ---------------------------------------------------------------------------
def day_return_proxy(date: str, l10: np.ndarray) -> float:
    """Rough directional proxy: mean of all valid labels_10s. >0 = up, <0=down."""
    v = l10[np.isfinite(l10)]
    if len(v) == 0:
        return 0.0
    return float(np.mean(v))


def classify_regime(day_ret: float) -> str:
    # Use coarse threshold on mean-of-10s-labels (in ticks).
    # Practical: |mean| > 0.05 ticks = directional day.
    if day_ret > 0.05:
        return 'green'
    if day_ret < -0.05:
        return 'red'
    return 'flat'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    t0 = time.time()
    dates = list_oot_dates()
    log(f'Processing {len(dates)} OOT weekday dates: {dates[0]}..{dates[-1]}')

    per_date_records = []
    feature_names: list[str] | None = None
    daily_regimes = {}

    for i, date in enumerate(dates):
        path = MBO_DIR / f'{date}_mbo_events.npz'
        if not path.exists():
            log(f'  skip {date} (missing)')
            continue
        td0 = time.time()
        try:
            day = build_day_features(path)
        except Exception as e:
            log(f'  ERR {date}: {e}')
            continue
        if day is None:
            log(f'  skip {date} (too few trades)')
            continue
        if feature_names is None:
            feature_names = day.feature_names
        ev = evaluate_day(day)
        # Regime
        npz = np.load(path, allow_pickle=True)
        dr = day_return_proxy(date, npz['labels_10s'])
        regime = classify_regime(dr)
        ev['day_return_proxy'] = dr
        ev['regime'] = regime
        daily_regimes[date] = regime
        per_date_records.append(ev)
        log(f'  [{i+1:2d}/{len(dates)}] {date} trades={day.n_trades:>7,d} '
            f'regime={regime} ret_proxy={dr:+.3f}  '
            f'IC10(tape_imb_10s)={ev["ic"].get("tape_imbalance_10s__y10", np.nan):+.3f}  '
            f'top5%comb={ev["top5pct"]["combined_net_t"] if ev["top5pct"] else float("nan"):+.3f}t  '
            f'({time.time()-td0:.1f}s)')

    # ------------- Aggregation ----------------
    log('Aggregating cross-date statistics...')

    # Per-feature, per-horizon Spearman summary
    ic_rows = []
    for rec in per_date_records:
        row = {'date': rec['date']}
        row.update(rec['ic'])
        ic_rows.append(row)
    ic_df = pd.DataFrame(ic_rows).set_index('date')
    ic_summary = pd.DataFrame({
        'mean': ic_df.mean(),
        'std':  ic_df.std(),
        'median': ic_df.median(),
        'n':    ic_df.count(),
        'frac_pos': (ic_df > 0).sum() / ic_df.count(),
    })
    ic_summary.to_csv(OUT_DIR / 'feature_ic_per_date.csv')
    log(f'IC summary written -> feature_ic_per_date.csv  rows={len(ic_summary)}')

    # Best feature by |median IC| at 10s
    y10_cols = [c for c in ic_df.columns if c.endswith('__y10')]
    best_by_abs_med = ic_summary.loc[y10_cols, 'median'].abs().sort_values(ascending=False)
    log('Top 5 features by |median Spearman vs y10|:')
    for fname, val in best_by_abs_med.head(5).items():
        med = ic_summary.loc[fname, 'median']
        log(f'   {fname:40s}  median IC = {med:+.4f}')

    # Per-day pnl table for tape_imbalance_10s top-decile combined strategy
    pnl_rows = []
    for rec in per_date_records:
        t5 = rec.get('top5pct') or {}
        pnl_rows.append({
            'date': rec['date'],
            'regime': rec['regime'],
            'n_trades_day': rec['n_trades'],
            'n_long_signals': t5.get('n_long', 0),
            'n_short_signals': t5.get('n_short', 0),
            'long_gross_t': t5.get('long_gross_t', np.nan),
            'short_gross_t': t5.get('short_gross_t', np.nan),
            'long_net_t': t5.get('long_net_t', np.nan),
            'short_net_t': t5.get('short_net_t', np.nan),
            'combined_net_t': t5.get('combined_net_t', np.nan),
            'combined_n': t5.get('combined_n', 0),
            'combined_std_t': t5.get('combined_std_t', np.nan),
        })
    pnl_df = pd.DataFrame(pnl_rows)
    pnl_df.to_csv(OUT_DIR / 'best_cell_per_day_pnl.csv', index=False)

    # Deploy-gate stats based on combined_net_t
    valid_days = pnl_df.dropna(subset=['combined_net_t'])
    pooled_net = float(np.nansum(valid_days['combined_net_t'] * valid_days['combined_n']) /
                       max(int(valid_days['combined_n'].sum()), 1))
    profit_days = (valid_days['combined_net_t'] > 0).sum()
    profit_days_ratio = float(profit_days / max(len(valid_days), 1))

    # Day Sharpe: per-day mean of combined_net_t (treat each day as 1 obs)
    day_means = valid_days['combined_net_t'].to_numpy()
    if len(day_means) > 1 and day_means.std() > 0:
        day_sharpe = float(day_means.mean() / day_means.std() * np.sqrt(252))
    else:
        day_sharpe = float('nan')

    # Day concentration: largest single-day |abs net contribution| / total
    abs_contribs = np.abs(valid_days['combined_net_t'].fillna(0).to_numpy() *
                          valid_days['combined_n'].fillna(0).to_numpy())
    total_abs = abs_contribs.sum()
    day_conc = float(abs_contribs.max() / total_abs) if total_abs > 0 else float('nan')

    # Regime imbalance
    reg_sharpe = {}
    for r in ('green', 'red', 'flat'):
        sub = valid_days[valid_days['regime'] == r]['combined_net_t'].to_numpy()
        if len(sub) > 1 and sub.std() > 0:
            reg_sharpe[r] = float(sub.mean() / sub.std() * np.sqrt(252))
        else:
            reg_sharpe[r] = float('nan')
    sg, sr = reg_sharpe.get('green', np.nan), reg_sharpe.get('red', np.nan)
    if np.isfinite(sg) and np.isfinite(sr):
        denom = max(abs(sg), abs(sr))
        regime_imbalance = float(abs(sg - sr) / denom) if denom > 0 else float('nan')
    else:
        regime_imbalance = float('nan')

    # Verdict
    if (pooled_net >= 0.10 and profit_days_ratio >= 0.60 and
            (day_sharpe is not None and day_sharpe >= 0.30) and
            (day_conc is not None and day_conc <= 0.70) and
            (regime_imbalance is not None and regime_imbalance <= 0.50)):
        verdict = 'ACCEPT'
    elif pooled_net >= 0.05 and profit_days_ratio >= 0.55:
        verdict = 'PARTIAL'
    else:
        verdict = 'REJECT'

    # Leave-one-day-out stability for the best feature
    best_feature_y10 = best_by_abs_med.index[0]
    base_med = float(ic_summary.loc[best_feature_y10, 'median'])
    loo = []
    series = ic_df[best_feature_y10].dropna()
    for d in series.index:
        loo.append(series.drop(d).median())
    loo = np.array(loo)
    if len(loo) > 1:
        loo_std = float(loo.std())
        loo_min = float(loo.min())
        loo_max = float(loo.max())
    else:
        loo_std = loo_min = loo_max = float('nan')

    # SWEEP vs ABSORPTION comparison: large_aggressor_flow_10s vs
    # small_aggressor_flow_10s median IC at y10.
    large_med = ic_summary.loc['large_aggressor_flow_10s__y10', 'median'] \
        if 'large_aggressor_flow_10s__y10' in ic_summary.index else np.nan
    small_med = ic_summary.loc['small_aggressor_flow_10s__y10', 'median'] \
        if 'small_aggressor_flow_10s__y10' in ic_summary.index else np.nan
    if abs(large_med) > abs(small_med):
        flow_winner = f'SWEEP (large) — |median IC|={abs(large_med):.4f} vs absorption {abs(small_med):.4f}'
    else:
        flow_winner = f'ABSORPTION (small) — |median IC|={abs(small_med):.4f} vs sweep {abs(large_med):.4f}'

    # Best side
    long_pool = float(np.nansum(valid_days['long_net_t'] * valid_days['n_long_signals']) /
                      max(int(valid_days['n_long_signals'].sum()), 1)) \
        if 'n_long_signals' in valid_days else float('nan')
    short_pool = float(np.nansum(valid_days['short_net_t'] * valid_days['n_short_signals']) /
                       max(int(valid_days['n_short_signals'].sum()), 1)) \
        if 'n_short_signals' in valid_days else float('nan')
    best_side = 'long' if (np.isfinite(long_pool) and (not np.isfinite(short_pool) or long_pool > short_pool)) else 'short'

    # Report
    elapsed = time.time() - t0
    report_lines = []
    rl = report_lines.append
    rl('# Trade Tape Imbalance v1 — Report')
    rl('')
    rl(f'**Generated:** {datetime.now().isoformat(timespec="seconds")}')
    rl(f'**OOT range:** {OOT_START} .. {OOT_END}  (weekday trading days)')
    rl(f'**N dates evaluated:** {len(per_date_records)}')
    rl(f'**Wall time:** {elapsed:.1f}s')
    rl('')
    rl('## Hypothesis')
    rl('Aggressor-side trade-tape imbalance (large sweeps vs small absorptive')
    rl('trades) is predictive of forward 5s/10s mid-price moves and is')
    rl('independent of BBO-level queue/OFI features in v3.4.2.')
    rl('')
    rl('## Top features by |median per-date Spearman| vs realized 10s tick move')
    rl('| Feature | Median IC | Mean IC | Std | Frac days >0 |')
    rl('|---|---:|---:|---:|---:|')
    for fname, _ in best_by_abs_med.head(10).items():
        med = ic_summary.loc[fname, 'median']
        mn  = ic_summary.loc[fname, 'mean']
        sd  = ic_summary.loc[fname, 'std']
        fp  = ic_summary.loc[fname, 'frac_pos']
        rl(f'| `{fname}` | {med:+.4f} | {mn:+.4f} | {sd:.4f} | {fp:.2f} |')
    rl('')
    rl('## Headline strategy: top/bottom-5% tape_imbalance_10s combined long+short, passive cost 0.376t')
    rl('')
    rl(f'- Pooled net (per-event mean):  **{pooled_net:+.4f} ticks**')
    rl(f'- Profit days ratio: **{profit_days_ratio:.2f}** ({int(profit_days)} / {len(valid_days)})')
    rl(f'- Day Sharpe (annualized): **{day_sharpe:.2f}**')
    rl(f'- Day concentration: **{day_conc:.2f}**  (cap 0.70)')
    rl(f'- Long pooled net:  {long_pool:+.4f} ticks')
    rl(f'- Short pooled net: {short_pool:+.4f} ticks')
    rl(f'- Best side: **{best_side}**')
    rl('')
    rl('## Regime stratification (annualized day-Sharpe)')
    rl(f'- green-day Sharpe: {reg_sharpe.get("green", float("nan")):.2f}')
    rl(f'- red-day   Sharpe: {reg_sharpe.get("red",   float("nan")):.2f}')
    rl(f'- flat-day  Sharpe: {reg_sharpe.get("flat",  float("nan")):.2f}')
    rl(f'- regime imbalance (|green-red|/max): **{regime_imbalance:.2f}**  (cap 0.50)')
    rl('')
    rl('## Sweep vs Absorption (large vs small aggressor flow)')
    rl(f'- {flow_winner}')
    rl('')
    rl('## Leave-one-day-out stability — best feature ({})'.format(best_feature_y10))
    rl(f'- Base median IC: {base_med:+.4f}')
    rl(f'- LOO median IC range: [{loo_min:+.4f}, {loo_max:+.4f}],  std = {loo_std:.4f}')
    rl('')
    rl('## Deploy gates (HC #428)')
    rl(f'- ACCEPT requires: pooled_net >= +0.10 AND profit_days >= 0.60 AND Sharpe >= 0.30 AND day_conc <= 0.70 AND regime_imbalance <= 0.50')
    rl(f'- PARTIAL requires: pooled_net >= +0.05 AND profit_days >= 0.55')
    rl('')
    rl(f'## VERDICT: **{verdict}**')
    rl('')
    rl('## Honest caveats')
    rl('- This is **label-level (mid-price) P&L**, not FIFO market replay. HC #74 says FIFO is canonical for any production decision. If verdict is PASS/PARTIAL the next step is a FIFO fill-sim run.')
    rl('- Sample is {} trading days. Statistical robustness modest.'.format(len(per_date_records)))
    rl('- Regime classifier uses mean(labels_10s) as proxy for ES close-to-close direction (no external SPX close fetched here).')
    rl('- Passive cost (0.376 ticks) assumes both legs fill at the resting price without queue jumping; in reality top-decile tape-imbalance moments are exactly when the queue runs you over. Real FIFO net likely worse.')

    (OUT_DIR / 'REPORT.md').write_text('\n'.join(report_lines))

    # regen complete marker
    regen = {
        'completed_at': datetime.now().isoformat(timespec='seconds'),
        'n_dates': len(per_date_records),
        'verdict': verdict,
        'best_feature_y10': best_feature_y10,
        'best_feature_median_ic_y10': base_med,
        'pooled_net_t': pooled_net,
        'profit_days_ratio': profit_days_ratio,
        'day_sharpe': day_sharpe,
        'day_conc': day_conc,
        'regime_imbalance': regime_imbalance,
        'flow_winner': flow_winner,
        'best_side': best_side,
        'long_pool_net_t': long_pool,
        'short_pool_net_t': short_pool,
        'elapsed_s': elapsed,
    }
    with open(OUT_DIR / '.regen_complete.json', 'w') as f:
        json.dump(regen, f, indent=2, default=float)

    # Final summary to stdout
    print('')
    print('=' * 78)
    print('TRADE-TAPE IMBALANCE v1 — FINAL SUMMARY')
    print('=' * 78)
    print(f'Best tape feature (by |median IC| vs y10): {best_feature_y10}')
    print(f'   median per-date Spearman: {base_med:+.4f}')
    print(f'   LOO range: [{loo_min:+.4f}, {loo_max:+.4f}], std={loo_std:.4f}')
    print(f'Top-5% combined strategy net (passive 0.376t): {pooled_net:+.4f} ticks')
    print(f'Profit-days ratio: {profit_days_ratio:.2f} ({int(profit_days)}/{len(valid_days)})')
    print(f'Day Sharpe (annualized): {day_sharpe:.2f}')
    print(f'Day concentration: {day_conc:.2f}')
    print(f'Regime imbalance: {regime_imbalance:.2f}')
    print(f'Best side: {best_side} (long_pool={long_pool:+.4f}t, short_pool={short_pool:+.4f}t)')
    print(f'Flow type winner: {flow_winner}')
    print('-' * 78)
    print(f'VERDICT: {verdict}')
    print('=' * 78)
    print(f'Outputs -> {OUT_DIR}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
