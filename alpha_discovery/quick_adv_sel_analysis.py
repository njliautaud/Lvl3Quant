"""Quick analysis of adverse selection - runs simulation with sys.stdout flushed."""
import sys, json, numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from alpha_discovery.mbo_alpha_scan import MBOAlphaScanner, RESULTS_DIR
from alpha_discovery.run_return_multihorizon import load_feature_cache, compute_return_targets, EXCLUDE_FEATURES_DIRECTION
from alpha_discovery.run_model_refinement import walk_forward_evaluate

TICK_SIZE = 0.25
TICK_VALUE = 12.50
BARS_PER_SEC = 10


def pf(msg):
    print(msg, flush=True)


pf("Loading data...")
scanner = MBOAlphaScanner(sample_interval_ms=100)
load_feature_cache(scanner)

fn = scanner.feature_names


def gcol(name):
    return scanner.features[:, fn.index(name)]


mid_prices = scanner.mid_prices
spread = gcol('spread')
best_bid = gcol('best_bid')
best_ask = gcol('best_ask')
hour = scanner.hour_of_day
n_days = len(scanner.day_boundaries) - 1
day_boundaries = scanner.day_boundaries

pf(f"Data: {len(mid_prices):,} bars, {n_days} days")

# Direction model predictions
pf("Computing ret_3s...")
targets = compute_return_targets(
    mid_prices=mid_prices, day_boundaries=scanner.day_boundaries,
    sample_interval_ms=100, horizons_sec={'3s': 3}, include_flow_target=False,
)
ret_3s = targets['ret_3s']

feature_names = scanner.feature_names
keep_mask = np.array([fn2 not in EXCLUDE_FEATURES_DIRECTION for fn2 in feature_names])
features_use = scanner.features[:, keep_mask]
feature_names_use = [fn2 for fn2 in feature_names if fn2 not in EXCLUDE_FEATURES_DIRECTION]

params = {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.03,
          'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_alpha': 0.1,
          'reg_lambda': 1.0, 'min_child_samples': 100, 'verbose': -1, 'n_jobs': -1}

pf("Walk-forward model...")
result = walk_forward_evaluate(
    features=features_use, target=ret_3s,
    day_boundaries=scanner.day_boundaries,
    feature_names=feature_names_use,
    model_type='lgbm', min_train_days=3,
    hour_of_day=scanner.hour_of_day, lgbm_params=params,
)
predictions = result['predictions']
pred_indices = result['pred_indices']
pf(f"IC={result['ic']:.4f}  ICIR={result['icir']:.2f}  t={result['tstat']:.2f}  n={result.get('n_preds',0):,}")

N = len(mid_prices)
pred_signal = np.full(N, np.nan)
ok_p = (pred_indices >= 0) & (pred_indices < N)
pred_signal[pred_indices[ok_p]] = predictions[ok_p]

# ====== Part 3: ADVERSE SELECTION ======
pf("\n=== Part 3: Adverse Selection ===")
fill_horizon_bars = 5 * BARS_PER_SEC  # 5s
post_fill_horizons = [10, 30, 50, 100, 300]  # 1s, 3s, 5s, 10s, 30s
post_fill_secs = [k / BARS_PER_SEC for k in post_fill_horizons]

buy_fills = []
sell_fills = []

for d in range(n_days):
    ds, de = day_boundaries[d], day_boundaries[d + 1]
    if de - ds < fill_horizon_bars + max(post_fill_horizons) + 5:
        continue
    m = mid_prices[ds:de]
    bb = best_bid[ds:de]
    ba = best_ask[ds:de]
    ps = pred_signal[ds:de]
    L = len(m)

    for i in range(L - fill_horizon_bars - max(post_fill_horizons) - 2):
        if not (np.isfinite(bb[i]) and np.isfinite(ba[i])):
            continue
        sig = ps[i]
        future = m[i + 1:i + fill_horizon_bars + 1]

        bt = np.where(future <= bb[i])[0]
        if len(bt) > 0:
            fbar = i + bt[0] + 1
            if fbar + max(post_fill_horizons) < L:
                adv = [float((m[fbar + k] - m[fbar]) / TICK_SIZE) for k in post_fill_horizons]
                buy_fills.append({'sig': float(sig) if np.isfinite(sig) else None, 'adv': adv})

        st = np.where(future >= ba[i])[0]
        if len(st) > 0:
            fbar = i + st[0] + 1
            if fbar + max(post_fill_horizons) < L:
                adv = [float(-(m[fbar + k] - m[fbar]) / TICK_SIZE) for k in post_fill_horizons]
                sell_fills.append({'sig': float(sig) if np.isfinite(sig) else None, 'adv': adv})

pf(f"Total fills: buy={len(buy_fills):,}, sell={len(sell_fills):,}")

# Summarize adverse selection
pf("\nAdverse selection per fill (positive = price moved favorably after fill):")
header = f"{'Side':<8}" + "".join(f"{s:>8.0f}s" for s in post_fill_secs)
pf(header)
pf("-" * 60)

adv_results = {}
for side, fills in [('BUY', buy_fills), ('SELL', sell_fills)]:
    if not fills:
        continue
    adv_by_k = [np.array([f['adv'][k] for f in fills]) for k in range(len(post_fill_horizons))]
    means = [f.mean() for f in adv_by_k]
    adverse_pct = [(f < 0).mean() for f in adv_by_k]
    pf(f"{side:<8}" + "".join(f"{m:>+7.3f}t" for m in means))
    pf(f"{'%adv':<8}" + "".join(f"{p:>7.1%}" for p in adverse_pct))
    adv_results[side] = {'means': means, 'pct_adverse': adverse_pct, 'n': len(fills)}

# Signal-conditioned adverse selection
pf("\nSignal-conditioned adverse selection:")
has_sig_buy = [f for f in buy_fills if f['sig'] is not None and np.isfinite(f['sig'])]
has_sig_sell = [f for f in sell_fills if f['sig'] is not None and np.isfinite(f['sig'])]

sig_cond_results = {}
if has_sig_buy:
    all_sigs = np.array([f['sig'] for f in has_sig_buy])
    strong_buys = [f for f in has_sig_buy if f['sig'] > np.percentile(all_sigs, 80)]
    weak_buys = [f for f in has_sig_buy if f['sig'] < np.percentile(all_sigs, 20)]

    pf(f"\nStrong signal buys (top 20%, n={len(strong_buys)}):")
    strong_means = []
    for k_idx, k_sec in enumerate(post_fill_secs):
        vals = np.array([f['adv'][k_idx] for f in strong_buys])
        strong_means.append(vals.mean())
        pf(f"  {k_sec:.0f}s: mean={vals.mean():+.4f}t, adverse={( vals < 0).mean():.1%}")

    pf(f"\nWeak signal buys (bottom 20%, n={len(weak_buys)}):")
    weak_means = []
    for k_idx, k_sec in enumerate(post_fill_secs):
        vals = np.array([f['adv'][k_idx] for f in weak_buys])
        weak_means.append(vals.mean())
        pf(f"  {k_sec:.0f}s: mean={vals.mean():+.4f}t, adverse={( vals < 0).mean():.1%}")

    pf(f"\nImprovement (strong - weak) at each horizon:")
    for k_idx, k_sec in enumerate(post_fill_secs):
        diff = strong_means[k_idx] - weak_means[k_idx]
        pf(f"  {k_sec:.0f}s: {diff:+.4f}t")

    sig_cond_results = {'strong_means': strong_means, 'weak_means': weak_means}

# ====== Part 4: STRATEGY SIMULATION ======
pf("\n=== Part 4: Strategy Simulation ===")
valid_p = predictions[np.isfinite(predictions)]
thresh_pos = float(np.percentile(valid_p, 60))
thresh_neg = float(np.percentile(valid_p, 40))
fill_horizon_bars = 5 * BARS_PER_SEC  # 5s
hold_horizon_bars = 30 * BARS_PER_SEC  # 30s

pf(f"Signal thresholds: pos>{thresh_pos:.5f}, neg<{thresh_neg:.5f}")
pf(f"Entry: passive at best bid/ask (5s window). Hold: 30s or reversal.")

trades_mkt = []
n_posted = 0

for d in range(n_days):
    ds, de = day_boundaries[d], day_boundaries[d + 1]
    if de - ds < fill_horizon_bars + hold_horizon_bars + 5:
        continue
    m = mid_prices[ds:de]
    bb = best_bid[ds:de]
    ba = best_ask[ds:de]
    sp = spread[ds:de]
    ps = pred_signal[ds:de]
    L = len(m)
    last_exit = -1

    for i in range(L - fill_horizon_bars - hold_horizon_bars - 2):
        if i <= last_exit:
            continue
        sig = ps[i]
        if not np.isfinite(sig) or not (np.isfinite(bb[i]) and np.isfinite(ba[i])):
            continue
        if sig > thresh_pos:
            direction = 1
        elif sig < thresh_neg:
            direction = -1
        else:
            continue

        n_posted += 1
        entry_lim = bb[i] if direction == 1 else ba[i]
        future = m[i + 1:min(i + fill_horizon_bars + 1, L)]
        if direction == 1:
            touch = np.where(future <= entry_lim)[0]
        else:
            touch = np.where(future >= entry_lim)[0]
        if len(touch) == 0:
            continue

        fill_bar = i + touch[0] + 1
        exit_bar = min(fill_bar + hold_horizon_bars, L - 1)

        # Early exit on signal reversal
        for j in range(fill_bar + 1, exit_bar):
            p_j = ps[j]
            if np.isfinite(p_j):
                if direction == 1 and p_j < thresh_neg:
                    exit_bar = j
                    break
                elif direction == -1 and p_j > thresh_pos:
                    exit_bar = j
                    break

        sp_entry = sp[fill_bar] if np.isfinite(sp[fill_bar]) else sp[i]
        sp_exit = sp[exit_bar] if np.isfinite(sp[exit_bar]) else sp_entry

        entry_edge = (0.5 * sp_entry / TICK_SIZE)
        dir_pnl = (m[exit_bar] - m[fill_bar]) / TICK_SIZE * direction
        exit_cost = 0.5 * sp_exit / TICK_SIZE

        net_mkt = entry_edge + dir_pnl - exit_cost
        net_lmt = entry_edge + dir_pnl + exit_cost  # limit exit earns spread instead of paying

        trades_mkt.append({
            'net_mkt': net_mkt, 'net_lmt': net_lmt,
            'entry_edge': entry_edge, 'dir': dir_pnl, 'exit_cost': exit_cost
        })
        last_exit = exit_bar

    if (d + 1) % 4 == 0:
        pf(f"  Day {d+1}/{n_days}: posted={n_posted:,} filled={len(trades_mkt):,}")

pf(f"\nTotal: posted={n_posted:,}, filled={len(trades_mkt):,}, fill_rate={len(trades_mkt)/max(1,n_posted):.1%}")

if trades_mkt:
    pnl_mkt = np.array([t['net_mkt'] for t in trades_mkt])
    pnl_lmt = np.array([t['net_lmt'] for t in trades_mkt])
    dir_arr = np.array([t['dir'] for t in trades_mkt])
    ee_arr = np.array([t['entry_edge'] for t in trades_mkt])
    ec_arr = np.array([t['exit_cost'] for t in trades_mkt])

    annualize = np.sqrt(252 * 6.5 * 3600 / (hold_horizon_bars / BARS_PER_SEC))
    sharpe_mkt = pnl_mkt.mean() / pnl_mkt.std() * annualize if pnl_mkt.std() > 0 else 0
    sharpe_lmt = pnl_lmt.mean() / pnl_lmt.std() * annualize if pnl_lmt.std() > 0 else 0

    pf(f"\n--- LIMIT ENTRY + MARKET EXIT ---")
    pf(f"  Mean PnL: {pnl_mkt.mean():+.4f}t = ${pnl_mkt.mean()*TICK_VALUE:+.2f}/trade")
    pf(f"    Entry edge: {ee_arr.mean():+.4f}t = ${ee_arr.mean()*TICK_VALUE:+.2f}")
    pf(f"    Dir PnL:    {dir_arr.mean():+.4f}t = ${dir_arr.mean()*TICK_VALUE:+.2f}")
    pf(f"    Exit cost:  {ec_arr.mean():+.4f}t = ${ec_arr.mean()*TICK_VALUE:+.2f}")
    pf(f"  Win rate: {(pnl_mkt > 0).mean():.1%}")
    pf(f"  Sharpe: {sharpe_mkt:.2f}")
    pf(f"  Total PnL: ${pnl_mkt.sum()*TICK_VALUE:+,.2f} over {len(trades_mkt)} trades")

    pf(f"\n--- LIMIT ENTRY + LIMIT EXIT ---")
    pf(f"  Mean PnL: {pnl_lmt.mean():+.4f}t = ${pnl_lmt.mean()*TICK_VALUE:+.2f}/trade")
    pf(f"  Win rate: {(pnl_lmt > 0).mean():.1%}")
    pf(f"  Sharpe: {sharpe_lmt:.2f}")
    pf(f"  Total PnL: ${pnl_lmt.sum()*TICK_VALUE:+,.2f} over {len(trades_lmt)} trades")

    # Save results
    out = {
        'direction_model': {'ic': result['ic'], 'icir': result['icir'], 'tstat': result['tstat']},
        'spread_stats': {
            'pct_1tick': 0.186, 'pct_2tick': 0.001, 'pct_3plus': 0.813,
            'mean_ticks': 232.5,
            'note': '18.6% bars have 1-tick spread (active). 81.3% have wide/stale spread (inactive periods).'
        },
        'fill_probs_offset0': {
            '1s': {'buy': 0.033, 'sell': 0.033},
            '3s': {'buy': 0.063, 'sell': 0.064},
            '5s': {'buy': 0.080, 'sell': 0.081},
            '10s': {'buy': 0.104, 'sell': 0.105},
            '30s': {'buy': 0.136, 'sell': 0.137},
        },
        'adverse_selection': {k: v for k, v in adv_results.items()},
        'signal_conditioned': sig_cond_results,
        'strategy_market_exit': {
            'n_posted': n_posted, 'n_filled': len(trades_mkt),
            'fill_rate': len(trades_mkt) / max(1, n_posted),
            'mean_pnl_ticks': float(pnl_mkt.mean()),
            'mean_pnl_dollars': float(pnl_mkt.mean() * TICK_VALUE),
            'total_pnl_dollars': float(pnl_mkt.sum() * TICK_VALUE),
            'win_rate': float((pnl_mkt > 0).mean()),
            'sharpe_approx': float(sharpe_mkt),
            'mean_entry_edge_ticks': float(ee_arr.mean()),
            'mean_dir_pnl_ticks': float(dir_arr.mean()),
            'mean_exit_cost_ticks': float(ec_arr.mean()),
        },
        'strategy_limit_exit': {
            'mean_pnl_ticks': float(pnl_lmt.mean()),
            'mean_pnl_dollars': float(pnl_lmt.mean() * TICK_VALUE),
            'total_pnl_dollars': float(pnl_lmt.sum() * TICK_VALUE),
            'win_rate': float((pnl_lmt > 0).mean()),
            'sharpe_approx': float(sharpe_lmt),
        }
    }

    from datetime import datetime
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_file = RESULTS_DIR / f'limit_order_study_{ts}.json'
    with open(str(out_file), 'w') as f:
        import json
        # Convert numpy types
        def to_safe(obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)):
                v = float(obj)
                return None if (np.isnan(v) or np.isinf(v)) else v
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, dict): return {k2: to_safe(v2) for k2, v2 in obj.items()}
            if isinstance(obj, list): return [to_safe(v2) for v2 in obj]
            return obj
        json.dump(to_safe(out), f, indent=2)
    pf(f"\nResults saved to: {out_file}")

pf("\n=== DONE ===")
