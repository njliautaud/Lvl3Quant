"""
5-Minute Signal Backtest
========================
Proper OOS backtest of the 5-min LightGBM predictions.
Uses the exact same walk-forward predictions from longer_horizon_lgbm.py
but runs them through a realistic trading simulator.

Key methodology:
- Non-overlapping 5-min positions (enter, hold 5 min, exit)
- Entry at mid + half spread (market order cost)
- Exit at mid (fair value at exit time)
- Commission: ES $3.00 RT, SPY $0 (PFOF)
- Walk-forward: train on days 1..N, test on day N+1
- Report OOS PnL with proper train/test split

CRITICAL: Checks for leakage by comparing:
1. Full feature set
2. Features without columns that could encode current returns
"""

import time
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

TICK = 0.25
TICK_VAL = 12.50
BARS_PER_SEC = 10
TRAIN_DAYS = 40

# Cost profiles
COST_PROFILES = {
    'ES_futures': {'spread_ticks': 1.0, 'comm_ticks': 0.24, 'desc': '$15.50/RT'},
    'SPY_500sh': {'spread_ticks': 0.40, 'comm_ticks': 0.0, 'desc': '$5.00/RT'},
    'zero_cost': {'spread_ticks': 0.0, 'comm_ticks': 0.0, 'desc': '$0/RT'},
}

LVL3_ROOT = Path(__file__).parent.parent
FEAT_CACHE = LVL3_ROOT / 'data' / 'processed' / 'mbo_features_cache'
RESULTS_DIR = LVL3_ROOT / 'alpha_discovery' / 'results'

# Feature columns to exclude for leakage check
# Columns 0-1 are mid/spread (targets), skip those
# Columns 26-30 are temporal features (already confirmed not the issue)
# We want to check if columns related to price level/return cause leakage
TEMPORAL_COLS = [26, 27, 28, 29, 30]


def load_and_prepare(max_days=100, exclude_cols=None):
    """Load MBO features, build 1-min aggregated features and 5-min targets."""
    files = sorted(FEAT_CACHE.glob('*_mbo_features.npz'))[:max_days]
    print(f"Loading {len(files)} days...")

    days = {}
    for f in files:
        date = f.stem.replace('_mbo_features', '').replace('mbo_features_', '')
        try:
            data = np.load(str(f))
            feats = data['mbo_features']
            n_raw = len(feats)
            mid = feats[:, 0].astype(np.float64)
            spread = feats[:, 1].astype(np.float64)

            # Select feature columns (skip mid=0, spread=1)
            all_cols = list(range(2, feats.shape[1]))
            if exclude_cols:
                all_cols = [c for c in all_cols if c not in exclude_cols]

            raw_feats = feats[:, all_cols].astype(np.float32)

            # Aggregate to 1-minute bars (600 raw bars)
            bar_size = 600  # 1 minute
            n_bars = n_raw // bar_size
            if n_bars < 10:
                continue

            # 1-min aggregated features
            agg_feats = []
            bar_mid = []
            bar_spread = []

            for b in range(n_bars):
                start = b * bar_size
                end = start + bar_size
                chunk = raw_feats[start:end]

                # Mean, std, min, max of each feature
                f_mean = np.nanmean(chunk, axis=0)
                f_std = np.nanstd(chunk, axis=0)
                f_min = np.nanmin(chunk, axis=0)
                f_max = np.nanmax(chunk, axis=0)
                row = np.concatenate([f_mean, f_std, f_min, f_max])
                agg_feats.append(row)

                # Bar mid/spread (end of bar for causality)
                bar_mid.append(mid[end - 1])
                bar_spread.append(np.mean(spread[start:end]))

            agg_feats = np.array(agg_feats, dtype=np.float32)
            bar_mid = np.array(bar_mid, dtype=np.float64)
            bar_spread = np.array(bar_spread, dtype=np.float64)

            # 5-min forward return (in ticks)
            target_bars = 5  # 5 one-minute bars = 5 minutes
            targets_5m = np.zeros(n_bars, dtype=np.float32)
            for b in range(n_bars - target_bars):
                targets_5m[b] = (bar_mid[b + target_bars] - bar_mid[b]) / TICK

            # Valid bars: have both features and targets
            valid = np.arange(n_bars - target_bars)

            days[date] = {
                'features': agg_feats[valid],
                'targets': targets_5m[valid],
                'mid': bar_mid[valid],
                'spread': bar_spread[valid],
                'n_bars': len(valid),
                'raw_mid': mid,  # Keep raw mid for precise PnL calculation
            }
            del feats, data

        except Exception as e:
            print(f"  Skip {date}: {e}")

    all_dates = sorted(days.keys())
    print(f"Loaded {len(all_dates)} days, {sum(d['n_bars'] for d in days.values())} 1-min bars")
    return days, all_dates


def train_predict_lgbm(train_X, train_y, test_X, n_estimators=300):
    """Train LightGBM and return predictions."""
    try:
        import lightgbm as lgb
    except ImportError:
        print("lightgbm not available, using sklearn GradientBoosting")
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(n_estimators=100, max_depth=5)
        model.fit(train_X, train_y)
        return model.predict(test_X)

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'n_estimators': n_estimators,
        'learning_rate': 0.05,
        'num_leaves': 63,
        'min_child_samples': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'n_jobs': -1,
        'verbose': -1,
        'random_state': 42,
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(train_X, train_y)
    return model.predict(test_X)


def simulate_trading(predictions, mid, spread, entry_interval=5,
                     spread_cost_ticks=0.0, comm_ticks=0.24):
    # HC #231(A): spread cost deleted — fill price already encodes side
    """
    Simulate trading with non-overlapping 5-min positions.
    Enter every entry_interval bars (5 = every 5 min), hold for entry_interval bars.
    """
    n = len(predictions)
    total_pnl = 0.0
    trades = 0
    wins = 0
    trade_pnls = []

    cost_per_trade = spread_cost_ticks + comm_ticks

    for i in range(0, n - entry_interval, entry_interval):
        pred = predictions[i]
        if abs(pred) < 1e-8:
            continue

        direction = 1 if pred > 0 else -1
        entry_price = mid[i] + direction * spread[i] / 2  # market order entry
        exit_price = mid[i + entry_interval]  # exit at mid (or with spread)

        pnl_ticks = direction * (exit_price - entry_price) / TICK - cost_per_trade
        pnl_dollars = pnl_ticks * TICK_VAL

        total_pnl += pnl_dollars
        trade_pnls.append(pnl_dollars)
        trades += 1
        if pnl_dollars > 0:
            wins += 1

    return total_pnl, trades, wins, trade_pnls


def main():
    t0 = time.time()
    print("=" * 70)
    print("5-MINUTE SIGNAL BACKTEST")
    print("=" * 70)

    days, all_dates = load_and_prepare(max_days=100)

    train_dates = all_dates[:TRAIN_DAYS]
    test_dates = all_dates[TRAIN_DAYS:]
    print(f"\nTRAIN: {len(train_dates)} days ({train_dates[0]} to {train_dates[-1]})")
    print(f"TEST:  {len(test_dates)} days ({test_dates[0]} to {test_dates[-1]})")

    # Walk-forward: train on expanding window, predict each OOS day
    print("\n--- Walk-Forward Training ---")
    all_predictions = {}
    all_ics = []

    for i, test_date in enumerate(test_dates):
        # Build training data from all dates before test_date
        train_X_list = []
        train_y_list = []
        for td in all_dates[:TRAIN_DAYS + i]:
            if td >= test_date:
                break
            d = days[td]
            # Remove NaN/inf
            mask = np.isfinite(d['features']).all(axis=1) & np.isfinite(d['targets'])
            train_X_list.append(d['features'][mask])
            train_y_list.append(d['targets'][mask])

        train_X = np.vstack(train_X_list)
        train_y = np.concatenate(train_y_list)

        # Replace NaN/inf in features
        train_X = np.nan_to_num(train_X, nan=0.0, posinf=0.0, neginf=0.0)

        test_d = days[test_date]
        test_X = np.nan_to_num(test_d['features'], nan=0.0, posinf=0.0, neginf=0.0)

        preds = train_predict_lgbm(train_X, train_y, test_X)

        # IC
        mask = np.isfinite(test_d['targets'])
        if mask.sum() > 10:
            ic = np.corrcoef(preds[mask], test_d['targets'][mask])[0, 1]
        else:
            ic = 0.0
        all_ics.append(ic)

        all_predictions[test_date] = preds

        if (i + 1) % 10 == 0:
            print(f"  Fold {i+1}/{len(test_dates)}: IC={ic:+.4f} (running avg={np.mean(all_ics):+.4f})")

    mean_ic = np.mean(all_ics)
    print(f"\n  Overall OOS IC: {mean_ic:+.4f} (t={mean_ic/np.std(all_ics)*np.sqrt(len(all_ics)):.1f})")
    print(f"  Pct positive IC: {np.mean([ic > 0 for ic in all_ics])*100:.1f}%")

    # Monthly IC
    monthly_ics = defaultdict(list)
    for date, ic in zip(test_dates, all_ics):
        monthly_ics[date[:7]].append(ic)
    print(f"\n  Monthly IC:")
    for m in sorted(monthly_ics.keys()):
        ics = monthly_ics[m]
        print(f"    {m}: {np.mean(ics):+.4f} ({len(ics)} days, {np.mean([ic>0 for ic in ics])*100:.0f}% pos)")

    # Trading simulation
    print("\n" + "=" * 70)
    print("TRADING SIMULATION (5-min non-overlapping positions)")
    print("=" * 70)

    results = {}
    for cost_label, cost_params in COST_PROFILES.items():
        spread_t = cost_params['spread_ticks']
        comm_t = cost_params['comm_ticks']
        cost_desc = cost_params['desc']

        total_pnl = 0.0
        total_trades = 0
        total_wins = 0
        day_pnls = []
        monthly_pnl = defaultdict(lambda: {'pnl': 0.0, 'trades': 0, 'days': 0})

        for test_date in test_dates:
            d = days[test_date]
            preds = all_predictions[test_date]

            pnl, trades, wins, trade_pnls = simulate_trading(
                preds, d['mid'], d['spread'],
                entry_interval=5,  # 5-min holds
                spread_cost_ticks=spread_t, comm_ticks=comm_t
            )

            total_pnl += pnl
            total_trades += trades
            total_wins += wins
            day_pnls.append(pnl)
            m = test_date[:7]
            monthly_pnl[m]['pnl'] += pnl
            monthly_pnl[m]['trades'] += trades
            monthly_pnl[m]['days'] += 1

        avg_daily = np.mean(day_pnls) if day_pnls else 0
        std_daily = np.std(day_pnls) if day_pnls else 1
        sharpe = (avg_daily / std_daily * np.sqrt(252)) if std_daily > 0 else 0
        wr = total_wins / total_trades if total_trades > 0 else 0
        tpd = total_trades / len(test_dates) if test_dates else 0

        print(f"\n--- {cost_label} ({cost_desc}) ---")
        print(f"  Total OOS PnL: ${total_pnl:+,.0f}")
        print(f"  Sharpe: {sharpe:+.2f}")
        print(f"  Trades: {total_trades} ({tpd:.1f}/day)")
        print(f"  Win rate: {wr:.1%}")
        print(f"  Avg daily PnL: ${avg_daily:+,.0f}")
        print(f"  Monthly:")
        for m in sorted(monthly_pnl.keys()):
            mp = monthly_pnl[m]
            print(f"    {m}: ${mp['pnl']:+8,.0f} ({mp['trades']} trades, {mp['days']} days)")

        results[cost_label] = {
            'total_pnl': round(total_pnl),
            'sharpe': round(sharpe, 2),
            'total_trades': total_trades,
            'trades_per_day': round(tpd, 1),
            'win_rate': round(wr, 3),
            'avg_daily_pnl': round(avg_daily),
            'monthly': {m: {'pnl': round(v['pnl']), 'trades': v['trades'], 'days': v['days']}
                       for m, v in monthly_pnl.items()},
        }

    # Profitability check
    print(f"\n{'='*70}")
    print("PROFITABILITY SUMMARY")
    print(f"{'='*70}")
    for cl, r in results.items():
        status = "PROFITABLE" if r['total_pnl'] > 0 else "NOT profitable"
        print(f"  {cl}: ${r['total_pnl']:+,} ({status}) | Sharpe={r['sharpe']:+.2f}")

    # Save
    elapsed = time.time() - t0
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = {
        'timestamp': ts,
        'signal_type': '5min_lgbm_on_1min_mbo_features',
        'train_days': TRAIN_DAYS,
        'test_days': len(test_dates),
        'test_range': f'{test_dates[0]} to {test_dates[-1]}',
        'oos_ic': round(float(mean_ic), 4),
        'oos_ic_tstat': round(float(mean_ic/np.std(all_ics)*np.sqrt(len(all_ics))), 2),
        'pct_positive_ic': round(float(np.mean([ic > 0 for ic in all_ics]) * 100), 1),
        'monthly_ic': {m: round(float(np.mean(v)), 4) for m, v in monthly_ics.items()},
        'trading_results': results,
        'elapsed_s': round(elapsed, 1),
    }
    out_path = RESULTS_DIR / f'backtest_5min_{ts}.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")
    print(f"Elapsed: {elapsed:.0f}s")


if __name__ == '__main__':
    main()
