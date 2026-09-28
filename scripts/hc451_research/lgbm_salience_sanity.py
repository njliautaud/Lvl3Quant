#!/usr/bin/env python3
"""HC #451 — LightGBM sanity sweep: do salience tags lift 5s direction prediction?

Bounded ~30-45 min CPU. Trains two LightGBM regressors on per-trade features
predicting signed 5s forward mid-price move (in ticks):
  Model A: baseline trade-derived features only
  Model B: baseline + sweep_tag + large_print_tag

Reports rank IC + top-1% precision on a contiguous OOT window.

CPU only, ≤6 threads, single seed. No GPU, no dispatch.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import polars as pl
import pandas as pd
import databento as db
from scipy.stats import spearmanr
import lightgbm as lgb


RAW_DIR = Path("/home/jupiter/Lvl3Quant/data/raw/mbo")
SAL_DIR = Path("/home/jupiter/Lvl3Quant/output/hc451_salience_tags/per_day")
CACHE_DIR = Path("/home/jupiter/Lvl3Quant/data/derived/mid_price_cache_hc439")
OUT_CSV = Path("/home/jupiter/Lvl3Quant/output/hc451_salience_tags/lgbm_sanity.csv")

A_TRADE = ord('T')
A_ASK = ord('A')
A_BID = ord('B')

# Spec follows precompute_salience_tags.py
def pick_front_month(arr):
    actions = arr['action'].view(np.uint8)
    t_mask = actions == A_TRADE
    px = arr['price'][t_mask].astype(np.int64)
    iid = arr['instrument_id'][t_mask]
    es_mask = (px > 5_000_000_000_000) & (px < 8_000_000_000_000)
    iid = iid[es_mask]
    if len(iid) == 0:
        return None
    uniq, cnt = np.unique(iid, return_counts=True)
    return int(uniq[np.argmax(cnt)])


def load_day_trades(date_str):
    """Return DataFrame of front-month trades with ts, px (ticks), side, size,
    bid_px, ask_px (tracked from updates so we can compute mid).
    For sanity-check speed we use last-trade-price as mid proxy and compute
    a *signed* move via lookahead on the trade-price series (instead of true
    L1 mid). This is the standard quick proxy used in HC #451 R2 validation."""
    dbn_path = RAW_DIR / f"glbx-mdp3-{date_str}.mbo.dbn.zst"
    if not dbn_path.exists():
        return None
    store = db.DBNStore.from_file(str(dbn_path))
    arr = store.to_ndarray()
    front_id = pick_front_month(arr)
    if front_id is None:
        return None
    actions = arr['action'].view(np.uint8)
    sides = arr['side'].view(np.uint8)
    iids = arr['instrument_id']
    t_mask = (actions == A_TRADE) & (iids == front_id)
    tr_ts = arr['ts_event'][t_mask].astype(np.int64)
    tr_px = arr['price'][t_mask].astype(np.int64)   # raw fixed-point /1e9
    tr_side = sides[t_mask]
    tr_size = arr['size'][t_mask].astype(np.int64)

    # ensure sorted
    if not np.all(np.diff(tr_ts) >= 0):
        order = np.argsort(tr_ts, kind='stable')
        tr_ts = tr_ts[order]; tr_px = tr_px[order]
        tr_side = tr_side[order]; tr_size = tr_size[order]

    # Convert price to ticks (1 tick = 0.25 = 250_000_000 fixed-point)
    TICK_RAW = 250_000_000
    px_ticks = (tr_px // TICK_RAW).astype(np.int64)

    # Aggressor side: 'A' = trade hit ask → buyer aggressor → +1
    #                 'B' = trade hit bid → seller aggressor → -1
    sign = np.zeros(len(tr_side), dtype=np.int8)
    sign[tr_side == A_ASK] = 1
    sign[tr_side == A_BID] = -1

    # Load salience tags for this date and join by ts
    sal_path = SAL_DIR / f"{date_str}_salience.parquet"
    sal = pl.read_parquet(sal_path).to_pandas()
    # The salience parquet has one row per source event. We need per-trade tags.
    # The trade ts in `tr_ts` are a subset. Use a hash-set join via merge.
    # But parquet has many rows per ts. Since sweep/large flags are only true on
    # trades, and on non-trade rows they're False, we can collapse: groupby ts max.
    sal_agg = sal.groupby('ts_ns', sort=True).agg(
        sweep_tag=('sweep_tag', 'max'),
        large_print_tag=('large_print_tag', 'max')
    ).reset_index()
    # Merge onto trades
    tr_df = pd.DataFrame({
        'ts_ns': tr_ts,
        'px_ticks': px_ticks,
        'sign': sign,
        'size': tr_size,
    })
    tr_df = tr_df.merge(sal_agg, on='ts_ns', how='left')
    tr_df['sweep_tag'] = tr_df['sweep_tag'].fillna(False).astype(bool)
    tr_df['large_print_tag'] = tr_df['large_print_tag'].fillna(False).astype(bool)
    return tr_df


def add_features_and_label(df):
    """Build minimal rolling features + 5s forward signed move label (in ticks)."""
    ts = df['ts_ns'].to_numpy()
    px = df['px_ticks'].to_numpy().astype(np.float64)
    sign = df['sign'].to_numpy().astype(np.float64)
    size = df['size'].to_numpy().astype(np.float64)
    n = len(df)

    out = {}
    # ---- rolling signed volume over last K trades (event-based, fast) ----
    for K in [20, 50, 200]:
        signed_size = sign * size
        # cumulative trick
        csum = np.concatenate([[0.0], np.cumsum(signed_size)])
        roll = csum[1:] - csum[np.maximum(0, np.arange(1, n+1) - K)]
        out[f'sv_{K}'] = roll
        # trade count is just min(K, i+1) — use scaled
        out[f'tcnt_{K}'] = np.minimum(np.arange(1, n+1), K).astype(np.float64)
        # mean trade size in window
        csum_sz = np.concatenate([[0.0], np.cumsum(size)])
        roll_sz = csum_sz[1:] - csum_sz[np.maximum(0, np.arange(1, n+1) - K)]
        out[f'avgsz_{K}'] = roll_sz / out[f'tcnt_{K}']
        # signed-volume / total-volume = trade imbalance ratio
        out[f'imb_{K}'] = out[f'sv_{K}'] / (roll_sz + 1e-9)

    # ---- price momentum: change in last-trade px over last K trades ----
    for K in [10, 50, 200]:
        out[f'mom_{K}'] = px - np.concatenate([np.full(K, px[0]), px[:-K]])

    # ---- realized vol (std of px changes) over last K trades ----
    dpx = np.diff(px, prepend=px[0])
    # rolling std via cumulative second-moment
    csum2 = np.concatenate([[0.0], np.cumsum(dpx*dpx)])
    csum1 = np.concatenate([[0.0], np.cumsum(dpx)])
    for K in [50, 200]:
        lo = np.maximum(0, np.arange(1, n+1) - K)
        s2 = csum2[1:] - csum2[lo]
        s1 = csum1[1:] - csum1[lo]
        cnt = np.minimum(np.arange(1, n+1), K).astype(np.float64)
        var = s2 / cnt - (s1/cnt)**2
        out[f'rv_{K}'] = np.sqrt(np.clip(var, 0, None))

    # ---- time since last trade (microseconds) ----
    dt_us = np.concatenate([[0], np.diff(ts)]) / 1000.0
    out['dt_us'] = dt_us

    # ---- inter-trade rate over last K ----
    for K in [50, 200]:
        lo = np.maximum(0, np.arange(1, n+1) - K)
        span_ns = ts - ts[lo]
        cnt = np.minimum(np.arange(1, n+1), K).astype(np.float64)
        rate = cnt / np.maximum(span_ns / 1e9, 1e-3)  # trades/sec
        out[f'rate_{K}'] = rate

    # ---- LABEL: signed 5s forward price move in ticks (last px within window) ----
    HORIZON_NS = 5_000_000_000  # 5s
    # For each i, find largest j s.t. ts[j] <= ts[i] + HORIZON_NS, use px[j] - px[i]
    j_target = ts + HORIZON_NS
    # searchsorted on ts (sorted)
    j_idx = np.searchsorted(ts, j_target, side='right') - 1
    j_idx = np.clip(j_idx, 0, n-1)
    label = px[j_idx] - px
    # Drop last 5s of session (label leaks past end) — mark NaN where future window truncated
    last_ts = ts[-1]
    valid = (ts + HORIZON_NS) <= last_ts
    label = np.where(valid, label, np.nan)

    feat_df = pd.DataFrame(out)
    feat_df['sweep_tag'] = df['sweep_tag'].astype(np.int8).values
    feat_df['large_print_tag'] = df['large_print_tag'].astype(np.int8).values
    feat_df['_label'] = label
    feat_df['_valid'] = valid
    return feat_df


def build_dataset(dates, label_tag):
    """Concatenate per-day feature frames."""
    frames = []
    for d in dates:
        t0 = time.time()
        tr = load_day_trades(d)
        if tr is None or len(tr) < 1000:
            print(f"[{label_tag}] {d}: skip (no trades)")
            continue
        feats = add_features_and_label(tr)
        feats['_date'] = d
        # Drop rows where label is invalid
        feats = feats[feats['_valid']].copy()
        # Drop first 500 rows (rolling features warmup)
        if len(feats) > 1000:
            feats = feats.iloc[500:].copy()
        frames.append(feats)
        print(f"[{label_tag}] {d}: loaded {len(tr):,} trades, kept {len(feats):,} rows in {time.time()-t0:.1f}s")
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True)
    return out


def top_k_precision(pred, label, frac=0.01):
    """Of top-frac highest pred (long signals), what fraction had label > 0?
    Also same for bottom-frac (short signals) with label < 0. Avg the two."""
    n = len(pred)
    k = max(1, int(n * frac))
    order = np.argsort(pred)
    short_idx = order[:k]
    long_idx = order[-k:]
    long_prec = float((label[long_idx] > 0).mean())
    short_prec = float((label[short_idx] < 0).mean())
    return long_prec, short_prec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-train', type=int, default=30)
    ap.add_argument('--n-oot', type=int, default=7)
    args = ap.parse_args()

    # Build date list from cache directory
    all_dates = sorted([f.replace('_trades.npz','')
                        for f in os.listdir(CACHE_DIR) if f.endswith('_trades.npz')])
    print(f"Total available dates: {len(all_dates)} ({all_dates[0]} -> {all_dates[-1]})")
    # Train = first n_train, OOT = next n_oot (contiguous)
    train_dates = all_dates[:args.n_train]
    oot_dates = all_dates[args.n_train:args.n_train + args.n_oot]
    print(f"Train: {len(train_dates)} days ({train_dates[0]} -> {train_dates[-1]})")
    print(f"OOT:   {len(oot_dates)} days ({oot_dates[0]} -> {oot_dates[-1]})")

    t0 = time.time()
    train_df = build_dataset(train_dates, 'TRAIN')
    print(f"Train build: {time.time()-t0:.1f}s, rows={len(train_df):,}")
    t0 = time.time()
    oot_df = build_dataset(oot_dates, 'OOT')
    print(f"OOT build: {time.time()-t0:.1f}s, rows={len(oot_df):,}")

    BASE_FEATS = [c for c in train_df.columns
                  if c not in ('_label','_valid','_date','sweep_tag','large_print_tag')]
    AUG_FEATS  = BASE_FEATS + ['sweep_tag','large_print_tag']
    print(f"baseline features ({len(BASE_FEATS)}): {BASE_FEATS}")

    y_train = train_df['_label'].astype(np.float32).values
    y_oot   = oot_df['_label'].astype(np.float32).values

    lgbm_params = dict(
        objective='regression',
        metric='rmse',
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_data_in_leaf=200,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=5,
        verbose=-1,
        num_threads=6,
        seed=42,
    )
    NUM_BOOST = 600

    results = []
    importances = {}
    for name, feats in [('baseline', BASE_FEATS), ('salience', AUG_FEATS)]:
        print(f"\n=== Training model: {name} ({len(feats)} feats) ===")
        Xtr = train_df[feats].astype(np.float32).values
        Xoot = oot_df[feats].astype(np.float32).values
        ds = lgb.Dataset(Xtr, label=y_train, feature_name=feats)
        t0 = time.time()
        model = lgb.train(lgbm_params, ds, num_boost_round=NUM_BOOST)
        print(f"  fit: {time.time()-t0:.1f}s")
        pred = model.predict(Xoot)
        # rank IC
        rho, _ = spearmanr(pred, y_oot)
        # top-1% precision
        long_p, short_p = top_k_precision(pred, y_oot, frac=0.01)
        # gain importance
        imp = dict(zip(feats, model.feature_importance(importance_type='gain')))
        importances[name] = imp
        print(f"  rank IC = {rho:.5f}")
        print(f"  top-1% long prec = {long_p:.4f}   top-1% short prec = {short_p:.4f}")
        results.append({
            'model': name,
            'n_train_rows': len(y_train),
            'n_oot_rows': len(y_oot),
            'rank_ic': round(rho, 5),
            'top1pct_long_prec': round(long_p, 4),
            'top1pct_short_prec': round(short_p, 4),
            'top1pct_avg_prec': round((long_p + short_p) / 2, 4),
        })

    # Compute & display importances of salience features in the augmented model
    aug = importances['salience']
    total_gain = sum(aug.values())
    sorted_feats = sorted(aug.items(), key=lambda kv: kv[1], reverse=True)
    rank_sweep = next((i+1 for i,(k,_) in enumerate(sorted_feats) if k=='sweep_tag'), None)
    rank_large = next((i+1 for i,(k,_) in enumerate(sorted_feats) if k=='large_print_tag'), None)
    sweep_gain_pct = 100 * aug.get('sweep_tag', 0) / total_gain if total_gain else 0
    large_gain_pct = 100 * aug.get('large_print_tag', 0) / total_gain if total_gain else 0
    print(f"\n=== Salience model feature ranks (out of {len(aug)}) ===")
    print(f"sweep_tag        rank #{rank_sweep}  gain%={sweep_gain_pct:.3f}")
    print(f"large_print_tag  rank #{rank_large}  gain%={large_gain_pct:.3f}")
    print(f"\ntop 10 feats by gain:")
    for k, v in sorted_feats[:10]:
        print(f"  {k:20s} {v:.1f}  ({100*v/total_gain:.2f}%)")

    # Append rank info to results
    for r in results:
        if r['model'] == 'salience':
            r['sweep_tag_rank'] = rank_sweep
            r['sweep_tag_gain_pct'] = round(sweep_gain_pct, 3)
            r['large_print_tag_rank'] = rank_large
            r['large_print_tag_gain_pct'] = round(large_gain_pct, 3)

    out_df = pd.DataFrame(results)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}")
    print(out_df.to_string(index=False))

    # Summary delta
    base = next(r for r in results if r['model'] == 'baseline')
    sal_r = next(r for r in results if r['model'] == 'salience')
    print(f"\n=== DELTA (salience − baseline) ===")
    print(f"rank IC delta:      {sal_r['rank_ic'] - base['rank_ic']:+.5f}")
    print(f"top-1% long  delta: {sal_r['top1pct_long_prec'] - base['top1pct_long_prec']:+.4f}")
    print(f"top-1% short delta: {sal_r['top1pct_short_prec'] - base['top1pct_short_prec']:+.4f}")


if __name__ == '__main__':
    main()
