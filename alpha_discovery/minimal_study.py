"""Minimal limit order study - just the key metrics we need."""
import sys, numpy as np, gc
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
stats = load_feature_cache(scanner)
if stats is None:
    pf("Cache miss - loading from raw...")
    stats = scanner.load_from_cache()

fn = scanner.feature_names
mid_prices = scanner.mid_prices
spread = scanner.features[:, fn.index('spread')]
best_bid = scanner.features[:, fn.index('best_bid')]
best_ask = scanner.features[:, fn.index('best_ask')]
hour = scanner.hour_of_day
n_days = len(scanner.day_boundaries) - 1
day_boundaries = scanner.day_boundaries

pf(f"Data: {len(mid_prices):,} bars, {n_days} days")

# Direction model
pf("Computing targets + walk-forward model...")
targets = compute_return_targets(
    mid_prices=mid_prices, day_boundaries=scanner.day_boundaries,
    sample_interval_ms=100, horizons_sec={'3s': 3}, include_flow_target=False,
)
ret_3s = targets['ret_3s']
gc.collect()

keep_mask = np.array([f not in EXCLUDE_FEATURES_DIRECTION for f in fn])
features_use = scanner.features[:, keep_mask]
feature_names_use = [f for f in fn if f not in EXCLUDE_FEATURES_DIRECTION]

params = {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.03,
          'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_alpha': 0.1,
          'reg_lambda': 1.0, 'min_child_samples': 100, 'verbose': -1, 'n_jobs': -1}

result = walk_forward_evaluate(
    features=features_use, target=ret_3s,
    day_boundaries=scanner.day_boundaries,
    feature_names=feature_names_use,
    model_type='lgbm', min_train_days=3,
    hour_of_day=scanner.hour_of_day, lgbm_params=params,
)
pf(f"IC={result['ic']:.4f}  ICIR={result['icir']:.2f}  t={result['tstat']:.2f}")

predictions = result['predictions']
pred_indices = result['pred_indices']
N = len(mid_prices)
pred_signal = np.full(N, np.nan)
ok_p = (pred_indices >= 0) & (pred_indices < N)
pred_signal[pred_indices[ok_p]] = predictions[ok_p]

del features_use, ret_3s
gc.collect()

pf("\n=== ADVERSE SELECTION (vectorized per-day) ===")
fill_horizon_bars = 5 * BARS_PER_SEC
post_fill_horizons = [10, 30, 50, 100, 300]
pfh_sec = [k / BARS_PER_SEC for k in post_fill_horizons]

# Store only numpy arrays, not Python dicts
n_buy = 0
n_sell = 0
buy_adv_sum = np.zeros(len(post_fill_horizons))
sell_adv_sum = np.zeros(len(post_fill_horizons))
buy_adv_neg = np.zeros(len(post_fill_horizons))  # count of adverse fills
sell_adv_neg = np.zeros(len(post_fill_horizons))

# For signal-conditioned: track by signal strength
# Use 5 quantile bins
valid_preds = predictions[np.isfinite(predictions)]
q_edges = np.percentile(np.abs(valid_preds), np.linspace(0, 100, 6))

qbuy_n = np.zeros((5, len(post_fill_horizons)))
qbuy_sum = np.zeros((5, len(post_fill_horizons)))
qsell_n = np.zeros((5, len(post_fill_horizons)))
qsell_sum = np.zeros((5, len(post_fill_horizons)))

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

        # BUY fill
        bt = np.where(future <= bb[i])[0]
        if len(bt) > 0:
            fbar = i + bt[0] + 1
            if fbar + max(post_fill_horizons) < L:
                n_buy += 1
                for k_idx, k in enumerate(post_fill_horizons):
                    adv = (m[fbar + k] - m[fbar]) / TICK_SIZE
                    buy_adv_sum[k_idx] += adv
                    if adv < 0:
                        buy_adv_neg[k_idx] += 1
                if np.isfinite(sig):
                    q = min(int(np.searchsorted(q_edges[1:], abs(sig), side='right')), 4)
                    for k_idx, k in enumerate(post_fill_horizons):
                        adv = (m[fbar + k] - m[fbar]) / TICK_SIZE
                        qbuy_n[q, k_idx] += 1
                        qbuy_sum[q, k_idx] += adv

        # SELL fill
        st = np.where(future >= ba[i])[0]
        if len(st) > 0:
            fbar = i + st[0] + 1
            if fbar + max(post_fill_horizons) < L:
                n_sell += 1
                for k_idx, k in enumerate(post_fill_horizons):
                    adv = -(m[fbar + k] - m[fbar]) / TICK_SIZE  # negative = good for seller
                    sell_adv_sum[k_idx] += adv
                    if adv < 0:
                        sell_adv_neg[k_idx] += 1
                if np.isfinite(sig):
                    q = min(int(np.searchsorted(q_edges[1:], abs(sig), side='right')), 4)
                    for k_idx, k in enumerate(post_fill_horizons):
                        adv = -(m[fbar + k] - m[fbar]) / TICK_SIZE
                        qsell_n[q, k_idx] += 1
                        qsell_sum[q, k_idx] += adv

    if (d + 1) % 4 == 0:
        pf(f"  Day {d+1}/{n_days}: buy_fills={n_buy:,} sell_fills={n_sell:,}")

gc.collect()

pf(f"\nTotal fills: buy={n_buy:,}, sell={n_sell:,}")
pf("\nAdverse selection per fill (positive = price moved favorably after fill):")
pf(f"{'Side':<8}" + "".join(f"{s:>8.0f}s" for s in pfh_sec))
pf("-" * 50)
pf(f"{'BUY':<8}" + "".join(f"{buy_adv_sum[k]/max(1,n_buy):>+7.3f}t" for k in range(len(post_fill_horizons))))
pf(f"{'%adv':<8}" + "".join(f"{buy_adv_neg[k]/max(1,n_buy):>7.1%}" for k in range(len(post_fill_horizons))))
pf(f"{'SELL':<8}" + "".join(f"{sell_adv_sum[k]/max(1,n_sell):>+7.3f}t" for k in range(len(post_fill_horizons))))
pf(f"{'%adv':<8}" + "".join(f"{sell_adv_neg[k]/max(1,n_sell):>7.1%}" for k in range(len(post_fill_horizons))))

pf("\nSignal-conditioned: Fill outcomes by signal strength (Q1=weak, Q5=strong):")
pf(f"{'Quintile':<12}" + "".join(f"{s:>8.0f}s" for s in pfh_sec))
for q in range(5):
    buy_means = [qbuy_sum[q, k] / max(1, qbuy_n[q, k]) for k in range(len(post_fill_horizons))]
    pf(f"Q{q+1}(BUY){'':4}" + "".join(f"{v:>+7.3f}t" for v in buy_means))
    sell_means = [qsell_sum[q, k] / max(1, qsell_n[q, k]) for k in range(len(post_fill_horizons))]
    pf(f"Q{q+1}(SELL){'':3}" + "".join(f"{v:>+7.3f}t" for v in sell_means))

pf("\n=== STRATEGY SIMULATION ===")
valid_p = predictions[np.isfinite(predictions)]
thresh_pos = float(np.percentile(valid_p, 60))
thresh_neg = float(np.percentile(valid_p, 40))
fill_hz = 5 * BARS_PER_SEC
hold_hz = 30 * BARS_PER_SEC

n_posted = 0
n_filled = 0
pnl_mkt_sum = 0.0
pnl_lmt_sum = 0.0
pnl_mkt_sq = 0.0
entry_edge_sum = 0.0
dir_pnl_sum = 0.0
exit_cost_sum = 0.0
n_win_mkt = 0
n_win_lmt = 0

for d in range(n_days):
    ds, de = day_boundaries[d], day_boundaries[d + 1]
    if de - ds < fill_hz + hold_hz + 5:
        continue
    m = mid_prices[ds:de]
    bb = best_bid[ds:de]
    ba = best_ask[ds:de]
    sp = spread[ds:de]
    ps = pred_signal[ds:de]
    L = len(m)
    last_exit = -1

    for i in range(L - fill_hz - hold_hz - 2):
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
        future = m[i + 1:min(i + fill_hz + 1, L)]
        if direction == 1:
            touch = np.where(future <= entry_lim)[0]
        else:
            touch = np.where(future >= entry_lim)[0]
        if len(touch) == 0:
            continue

        fill_bar = i + touch[0] + 1
        exit_bar = min(fill_bar + hold_hz, L - 1)

        for j in range(fill_bar + 1, exit_bar):
            p_j = ps[j]
            if np.isfinite(p_j):
                if direction == 1 and p_j < thresh_neg:
                    exit_bar = j; break
                elif direction == -1 and p_j > thresh_pos:
                    exit_bar = j; break

        sp_entry = sp[fill_bar] if np.isfinite(sp[fill_bar]) else sp[i]
        sp_exit = sp[exit_bar] if np.isfinite(sp[exit_bar]) else sp_entry

        entry_edge = 0.5 * sp_entry / TICK_SIZE
        dir_pnl = (m[exit_bar] - m[fill_bar]) / TICK_SIZE * direction
        exit_cost = 0.5 * sp_exit / TICK_SIZE

        net_mkt = entry_edge + dir_pnl - exit_cost
        net_lmt = entry_edge + dir_pnl + exit_cost

        n_filled += 1
        pnl_mkt_sum += net_mkt
        pnl_lmt_sum += net_lmt
        pnl_mkt_sq += net_mkt ** 2
        entry_edge_sum += entry_edge
        dir_pnl_sum += dir_pnl
        exit_cost_sum += exit_cost
        if net_mkt > 0: n_win_mkt += 1
        if net_lmt > 0: n_win_lmt += 1

        last_exit = exit_bar

    if (d + 1) % 4 == 0:
        pf(f"  Sim day {d+1}/{n_days}: posted={n_posted:,} filled={n_filled:,}")

gc.collect()

if n_filled > 0:
    mean_mkt = pnl_mkt_sum / n_filled
    mean_lmt = pnl_lmt_sum / n_filled
    std_mkt = np.sqrt(pnl_mkt_sq / n_filled - mean_mkt ** 2)
    annualize = np.sqrt(252 * 6.5 * 3600 / (hold_hz / BARS_PER_SEC))
    sharpe_mkt = mean_mkt / std_mkt * annualize if std_mkt > 0 else 0

    pf(f"\nResults: posted={n_posted:,}, filled={n_filled:,}, fill_rate={n_filled/n_posted:.1%}")
    pf(f"\n--- LIMIT ENTRY + MARKET EXIT ---")
    pf(f"  Mean PnL: {mean_mkt:+.4f}t = ${mean_mkt*TICK_VALUE:+.2f}/trade")
    pf(f"    Entry edge: {entry_edge_sum/n_filled:+.4f}t = ${entry_edge_sum/n_filled*TICK_VALUE:+.2f}")
    pf(f"    Dir PnL:    {dir_pnl_sum/n_filled:+.4f}t = ${dir_pnl_sum/n_filled*TICK_VALUE:+.2f}")
    pf(f"    Exit cost:  {exit_cost_sum/n_filled:+.4f}t = ${exit_cost_sum/n_filled*TICK_VALUE:+.2f}")
    pf(f"  Win rate: {n_win_mkt/n_filled:.1%}")
    pf(f"  Sharpe: {sharpe_mkt:.2f}")
    pf(f"  Total PnL: ${pnl_mkt_sum*TICK_VALUE:+,.2f} over {n_filled} trades")

    pf(f"\n--- LIMIT ENTRY + LIMIT EXIT ---")
    pf(f"  Mean PnL: {mean_lmt:+.4f}t = ${mean_lmt*TICK_VALUE:+.2f}/trade")
    pf(f"  Win rate: {n_win_lmt/n_filled:.1%}")
    pf(f"  Total PnL: ${pnl_lmt_sum*TICK_VALUE:+,.2f} over {n_filled} trades")

    # Save
    import json
    from datetime import datetime
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = {
        'timestamp': ts,
        'direction_model': {'ic': float(result['ic']), 'icir': float(result['icir']), 'tstat': float(result['tstat'])},
        'n_bars': int(N), 'n_days': int(n_days),
        'spread_stats': {
            'pct_1tick': 0.186, 'pct_2tick': 0.001, 'pct_3plus': 0.813,
            'mean_ticks_all_bars': 232.5,
            'note': '18.6% of 100ms bars have 1-tick spread (active market). Others have stale/wide quotes.'
        },
        'fill_probs_offset0': {
            '1s': {'buy': 0.033, 'sell': 0.033},
            '3s': {'buy': 0.063, 'sell': 0.064},
            '5s': {'buy': 0.080, 'sell': 0.081},
            '10s': {'buy': 0.104, 'sell': 0.105},
            '30s': {'buy': 0.136, 'sell': 0.137},
        },
        'adverse_selection': {
            'n_buy': n_buy, 'n_sell': n_sell,
            'buy_mean_by_horizon': {f'{pfh_sec[k]:.0f}s': float(buy_adv_sum[k]/max(1,n_buy)) for k in range(len(post_fill_horizons))},
            'sell_mean_by_horizon': {f'{pfh_sec[k]:.0f}s': float(sell_adv_sum[k]/max(1,n_sell)) for k in range(len(post_fill_horizons))},
            'buy_pct_adverse': {f'{pfh_sec[k]:.0f}s': float(buy_adv_neg[k]/max(1,n_buy)) for k in range(len(post_fill_horizons))},
            'sell_pct_adverse': {f'{pfh_sec[k]:.0f}s': float(sell_adv_neg[k]/max(1,n_sell)) for k in range(len(post_fill_horizons))},
        },
        'signal_conditioned': {
            f'q{q+1}': {
                'buy_mean': {f'{pfh_sec[k]:.0f}s': float(qbuy_sum[q,k]/max(1,qbuy_n[q,k])) for k in range(len(post_fill_horizons))},
                'sell_mean': {f'{pfh_sec[k]:.0f}s': float(qsell_sum[q,k]/max(1,qsell_n[q,k])) for k in range(len(post_fill_horizons))},
            } for q in range(5)
        },
        'strategy_market_exit': {
            'n_posted': n_posted, 'n_filled': n_filled,
            'fill_rate': n_filled/n_posted if n_posted > 0 else 0,
            'mean_pnl_ticks': float(mean_mkt),
            'mean_pnl_dollars': float(mean_mkt*TICK_VALUE),
            'total_pnl_dollars': float(pnl_mkt_sum*TICK_VALUE),
            'win_rate': float(n_win_mkt/n_filled),
            'sharpe_approx': float(sharpe_mkt),
            'mean_entry_edge_ticks': float(entry_edge_sum/n_filled),
            'mean_dir_pnl_ticks': float(dir_pnl_sum/n_filled),
            'mean_exit_cost_ticks': float(exit_cost_sum/n_filled),
        },
        'strategy_limit_exit': {
            'mean_pnl_ticks': float(mean_lmt),
            'mean_pnl_dollars': float(mean_lmt*TICK_VALUE),
            'total_pnl_dollars': float(pnl_lmt_sum*TICK_VALUE),
            'win_rate': float(n_win_lmt/n_filled),
        }
    }
    out_file = RESULTS_DIR / f'limit_order_study_{ts}.json'
    with open(str(out_file), 'w') as f:
        json.dump(out, f, indent=2)
    pf(f"\nResults saved: {out_file}")

pf("\n=== DONE ===")
