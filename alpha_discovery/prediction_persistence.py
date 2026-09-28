"""
Prediction Persistence Analysis
================================
Analyzes how quickly model predictions change direction.
The ET's predictions flip every ~1 second — this killed the hold-until-flip strategy.
Run this on both ET and CNN predictions to compare persistence.

Key metrics:
- Mean sign duration (bars between sign changes)
- Autocorrelation at various lags
- Prediction "momentum" (correlation of pred[t] with pred[t+lag])
- Trade viability: at what hold period does the signal maintain direction?
"""

import numpy as np
import json
from pathlib import Path
from collections import defaultdict

BARS_PER_SEC = 10
TICK = 0.25
TICK_VAL = 12.50

LVL3_ROOT = Path(__file__).parent.parent
FEAT_CACHE = LVL3_ROOT / 'data' / 'processed' / 'mbo_features_cache'
RESULTS_DIR = LVL3_ROOT / 'alpha_discovery' / 'results'


def analyze_persistence(preds, label=''):
    """Analyze prediction sign persistence and autocorrelation."""
    n = len(preds)
    signs = np.sign(preds)

    # 1. Sign duration: how many bars before prediction flips
    durations = []
    current_sign = signs[0]
    current_duration = 1
    for i in range(1, n):
        if signs[i] == current_sign or signs[i] == 0:
            current_duration += 1
        else:
            durations.append(current_duration)
            current_sign = signs[i]
            current_duration = 1
    durations.append(current_duration)
    durations = np.array(durations)

    # 2. Autocorrelation at various lags
    centered = preds - np.mean(preds)
    var = np.var(preds)
    autocorrs = {}
    for lag_s in [0.1, 0.5, 1, 2, 5, 10, 30, 60, 300]:
        lag_bars = int(lag_s * BARS_PER_SEC)
        if lag_bars >= n:
            continue
        ac = np.mean(centered[:n-lag_bars] * centered[lag_bars:]) / var if var > 0 else 0
        autocorrs[lag_s] = float(ac)

    # 3. Direction agreement: P(sign[t+lag] == sign[t])
    dir_agreement = {}
    for lag_s in [1, 2, 5, 10, 30, 60, 300]:
        lag_bars = int(lag_s * BARS_PER_SEC)
        if lag_bars >= n:
            continue
        agree = np.mean(signs[:n-lag_bars] == signs[lag_bars:])
        dir_agreement[lag_s] = float(agree)

    # 4. Prediction momentum: correlation of pred magnitude at different lags
    abs_preds = np.abs(preds)
    momentum = {}
    for lag_s in [1, 5, 10, 30, 60]:
        lag_bars = int(lag_s * BARS_PER_SEC)
        if lag_bars >= n:
            continue
        c = np.corrcoef(abs_preds[:n-lag_bars], abs_preds[lag_bars:])[0, 1]
        momentum[lag_s] = float(c)

    return {
        'n_bars': n,
        'n_sign_changes': len(durations),
        'mean_sign_duration_bars': float(np.mean(durations)),
        'mean_sign_duration_sec': float(np.mean(durations) / BARS_PER_SEC),
        'median_sign_duration_sec': float(np.median(durations) / BARS_PER_SEC),
        'p10_duration_sec': float(np.percentile(durations, 10) / BARS_PER_SEC),
        'p90_duration_sec': float(np.percentile(durations, 90) / BARS_PER_SEC),
        'autocorrelation': autocorrs,
        'direction_agreement': dir_agreement,
        'prediction_magnitude_momentum': momentum,
    }


def analyze_tradability(mid, preds, z_preds):
    """Test if holding in the predicted direction for N seconds is profitable."""
    n = min(len(mid), len(preds))
    results = {}

    for hold_s in [1, 2, 5, 10, 30, 60, 300, 600]:
        hold_bars = hold_s * BARS_PER_SEC
        if hold_bars >= n:
            continue

        # For each bar with |z|>1, compute return over hold period
        abs_z = np.abs(z_preds[:n])
        returns = []
        directions = []

        for i in range(0, n - hold_bars, hold_bars):  # non-overlapping
            if abs_z[i] <= 1.0:
                continue
            direction = 1 if preds[i] > 0 else -1
            ret = direction * (mid[i + hold_bars] - mid[i]) / TICK
            returns.append(ret)
            directions.append(direction)

        if not returns:
            continue

        returns = np.array(returns)
        results[f'{hold_s}s'] = {
            'n_trades': len(returns),
            'mean_return_ticks': float(np.mean(returns)),
            'std_return_ticks': float(np.std(returns)),
            'sharpe_per_trade': float(np.mean(returns) / np.std(returns)) if np.std(returns) > 0 else 0,
            'win_rate': float(np.mean(returns > 0)),
            'pct_long': float(np.mean(np.array(directions) == 1)),
            'profitable_at_ES': np.mean(returns) > 1.24,  # ES cost in ticks
            'profitable_at_SPY': np.mean(returns) > 0.40,  # SPY cost in ticks
            'edge_minus_ES_cost': float(np.mean(returns) - 1.24),
            'edge_minus_SPY_cost': float(np.mean(returns) - 0.40),
        }

    return results


def main():
    print("=" * 70)
    print("PREDICTION PERSISTENCE ANALYSIS")
    print("=" * 70)

    # Load ET predictions
    et_pred_file = LVL3_ROOT / 'alpha_discovery' / 'deep_models' / 'results' / 'oos_predictions_event_20260228_123832.npz'
    cnn_pred_file = LVL3_ROOT / 'alpha_discovery' / 'deep_models' / 'results' / 'oos_predictions_book_20260301_093742.npz'

    models = {}
    if et_pred_file.exists():
        models['EventTransformer'] = et_pred_file
    if cnn_pred_file.exists():
        models['BookSpatialCNN'] = cnn_pred_file

    if not models:
        print("No prediction files found!")
        return

    all_results = {}

    for model_name, pred_file in models.items():
        print(f"\n{'='*70}")
        print(f"MODEL: {model_name}")
        print(f"{'='*70}")

        pred_data = np.load(str(pred_file), allow_pickle=True)
        pred_dates = sorted(set(
            k.replace('_preds', '').replace('_targets', '')
            for k in pred_data.keys()
        ))

        day_persistence = []
        day_tradability = []

        for date in pred_dates:
            preds_key = f'{date}_preds'
            if preds_key not in pred_data:
                continue

            preds = pred_data[preds_key]

            # Load MBO for mid prices
            mbo_path = FEAT_CACHE / f'{date}_mbo_features.npz'
            if not mbo_path.exists():
                continue

            mbo = np.load(str(mbo_path))
            mid = mbo['mbo_features'][:, 0].astype(np.float32)

            n = min(len(mid), len(preds))
            preds = preds[:n].copy()
            mid = mid[:n].copy()

            # Z-score
            pred_std = np.std(preds)
            pred_mean = np.mean(preds)
            z_preds = (preds - pred_mean) / pred_std if pred_std > 0 else np.zeros_like(preds)

            p = analyze_persistence(preds, date)
            t = analyze_tradability(mid, preds, z_preds)

            day_persistence.append(p)
            day_tradability.append(t)

        # Aggregate
        mean_sign_dur = np.mean([d['mean_sign_duration_sec'] for d in day_persistence])
        med_sign_dur = np.mean([d['median_sign_duration_sec'] for d in day_persistence])

        print(f"\n--- PERSISTENCE ---")
        print(f"  Mean sign duration: {mean_sign_dur:.2f}s")
        print(f"  Median sign duration: {med_sign_dur:.2f}s")
        print(f"  P10/P90 duration: {np.mean([d['p10_duration_sec'] for d in day_persistence]):.2f}s / "
              f"{np.mean([d['p90_duration_sec'] for d in day_persistence]):.2f}s")

        print(f"\n  Autocorrelation (averaged across days):")
        for lag_s in [0.1, 0.5, 1, 2, 5, 10, 30, 60, 300]:
            acs = [d['autocorrelation'].get(lag_s, None) for d in day_persistence]
            acs = [a for a in acs if a is not None]
            if acs:
                print(f"    {lag_s:>6.1f}s: {np.mean(acs):+.4f}")

        print(f"\n  Direction agreement P(sign[t+lag]==sign[t]):")
        for lag_s in [1, 2, 5, 10, 30, 60, 300]:
            das = [d['direction_agreement'].get(lag_s, None) for d in day_persistence]
            das = [a for a in das if a is not None]
            if das:
                excess = np.mean(das) - 0.5
                print(f"    {lag_s:>5}s: {np.mean(das):.4f} (excess: {excess:+.4f})")

        print(f"\n--- TRADABILITY (hold for N seconds, entry on |z|>1) ---")
        for hold_s_str in ['1s', '2s', '5s', '10s', '30s', '60s', '300s', '600s']:
            edges = [d.get(hold_s_str, {}).get('mean_return_ticks', None) for d in day_tradability]
            edges = [e for e in edges if e is not None]
            trades = [d.get(hold_s_str, {}).get('n_trades', 0) for d in day_tradability]
            wrs = [d.get(hold_s_str, {}).get('win_rate', None) for d in day_tradability]
            wrs = [w for w in wrs if w is not None]

            if edges:
                mean_edge = np.mean(edges)
                total_trades = sum(trades)
                mean_wr = np.mean(wrs)
                es_viable = "YES" if mean_edge > 1.24 else "no"
                spy_viable = "YES" if mean_edge > 0.40 else "no"
                print(f"    {hold_s_str:>5}: edge={mean_edge:+.3f} ticks, WR={mean_wr:.1%}, "
                      f"trades={total_trades}, ES={es_viable}, SPY={spy_viable}")

        all_results[model_name] = {
            'n_days': len(day_persistence),
            'mean_sign_duration_sec': mean_sign_dur,
            'median_sign_duration_sec': med_sign_dur,
            'persistence': {
                lag: float(np.mean([d['autocorrelation'].get(lag, 0) for d in day_persistence]))
                for lag in [0.1, 0.5, 1, 2, 5, 10, 30, 60, 300]
            },
            'direction_agreement': {
                lag: float(np.mean([d['direction_agreement'].get(lag, 0.5) for d in day_persistence]))
                for lag in [1, 2, 5, 10, 30, 60, 300]
            },
            'tradability': {}
        }

        for hold_s_str in ['1s', '2s', '5s', '10s', '30s', '60s', '300s', '600s']:
            edges = [d.get(hold_s_str, {}).get('mean_return_ticks', None) for d in day_tradability]
            edges = [e for e in edges if e is not None]
            if edges:
                all_results[model_name]['tradability'][hold_s_str] = {
                    'mean_edge_ticks': float(np.mean(edges)),
                    'profitable_ES': float(np.mean(edges)) > 1.24,
                    'profitable_SPY': float(np.mean(edges)) > 0.40,
                }

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = RESULTS_DIR / f'prediction_persistence_{ts}.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")

    # Side-by-side comparison if both models
    if len(all_results) >= 2:
        print(f"\n{'='*70}")
        print("SIDE-BY-SIDE COMPARISON")
        print(f"{'='*70}")
        names = list(all_results.keys())
        print(f"{'Metric':<35} {names[0]:<20} {names[1]:<20}")
        print("-" * 75)
        print(f"{'Mean sign duration (s)':<35} {all_results[names[0]]['mean_sign_duration_sec']:<20.2f} {all_results[names[1]]['mean_sign_duration_sec']:<20.2f}")

        for lag in [1, 5, 10, 30, 60]:
            a1 = all_results[names[0]]['direction_agreement'].get(lag, 0.5)
            a2 = all_results[names[1]]['direction_agreement'].get(lag, 0.5)
            print(f"{'Dir agree @' + str(lag) + 's':<35} {a1:<20.4f} {a2:<20.4f}")


if __name__ == '__main__':
    main()
