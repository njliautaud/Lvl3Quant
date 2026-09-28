#!/usr/bin/env python3
"""
LGBM Freshness Validation
==========================

Tests LGBM fold 35 on RECENT data (April 2026) to verify performance hasn't decayed.

Critical Check: If IC on recent data is significantly lower than training IC,
the model has decayed and should be retrained before deployment.
"""

import pickle
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from datetime import datetime
import sys

# Import feature extraction from training script
sys.path.insert(0, str(Path(__file__).parent.parent / 'alpha_discovery'))
from train_lgbm_sliding_60d import events_to_bars, bar_features, conf_ic

def load_lgbm_model(horizon='1s'):
    """Load trained LGBM model"""
    model_path = Path('/home/jupiter/Lvl3Quant/models/lgbm_60_5_fold35')
    model_file = model_path / f'labels_{horizon}_lgbm.pkl'

    print(f"Loading model: {model_file}")
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    print(f"✓ Model loaded")
    return model

def load_recent_data(days=7):
    """Load most recent N days of MBO data"""
    data_dir = Path('/home/jupiter/Lvl3Quant/data/processed/mbo_events')
    files = sorted(data_dir.glob('*_mbo_events.npz'))

    if not files:
        print("❌ No MBO data files found")
        return []

    # Get most recent files
    recent_files = files[-days:]
    print(f"\nLoading {len(recent_files)} recent days of data:")
    for f in recent_files:
        date_str = f.stem[:8]
        print(f"  - {date_str}")

    return recent_files

def validate_freshness(horizon='1s', recent_days=5):
    """
    Validate model freshness by testing on recent data

    Returns:
        (ic, is_fresh, message)
    """
    print("="*70)
    print(f"LGBM FRESHNESS VALIDATION - Horizon: {horizon}")
    print("="*70)

    # Load model
    model = load_lgbm_model(horizon)

    # Load recent data
    recent_files = load_recent_data(recent_days)
    if not recent_files:
        return None, False, "No recent data available"

    # Process files and extract features
    print("\nExtracting features from recent data...")
    all_preds = []
    all_labels = []

    for i, file_path in enumerate(recent_files):
        print(f"  Processing {file_path.stem[:8]}... ", end='', flush=True)

        try:
            # Load and process
            d = np.load(file_path, allow_pickle=True)
            events = d['events'].astype('f4')[:2_000_000]  # Limit to 2M events
            labels_raw = d[f'labels_{horizon}'].astype('f4')[:2_000_000]

            # Convert to bars and extract features
            bars = events_to_bars(events, labels_raw, bar_ms=100)
            if bars is None:
                print("skipped (insufficient data)")
                continue

            features, labels = bar_features(bars, lookback=30)
            if features is None or len(features) == 0:
                print("skipped (no features)")
                continue

            # Generate predictions
            preds = model.predict(features)

            all_preds.append(preds)
            all_labels.append(labels)

            print(f"✓ {len(preds):,} predictions")

        except Exception as e:
            print(f"error: {e}")
            continue

    if not all_preds:
        return None, False, "No predictions generated from recent data"

    # Concatenate all predictions and labels
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    print(f"\nTotal predictions: {len(all_preds):,}")

    # Calculate IC
    print("\n" + "="*70)
    print("PERFORMANCE ON RECENT DATA (April 2026)")
    print("="*70)

    # Overall IC
    ic = float(spearmanr(all_preds, all_labels).correlation)
    print(f"\n📊 Overall IC: {ic:.4f}")

    # Confidence-stratified IC
    print("\n📊 Stratified IC:")
    stratified = conf_ic(all_preds, all_labels)

    # Historical benchmark (from fold 35)
    historical_ic = {
        '1s': 0.122,
        '5s': 0.055,
        '10s': 0.041,
        '30s': 0.018
    }
    benchmark = historical_ic.get(horizon.replace('labels_', ''), 0.10)

    # Freshness assessment
    print("\n" + "="*70)
    print("FRESHNESS ASSESSMENT")
    print("="*70)

    print(f"Historical IC (March test): {benchmark:.4f}")
    print(f"Recent IC (April data): {ic:.4f}")
    decay_pct = ((ic - benchmark) / benchmark) * 100
    print(f"Decay: {decay_pct:+.1f}%")

    # Thresholds
    is_fresh = True
    message = ""

    if ic >= benchmark * 0.8:  # Within 20% of historical
        is_fresh = True
        message = f"✅ FRESH - Model performance maintained ({ic:.4f} vs {benchmark:.4f})"
        print(f"\n✅ VERDICT: MODEL IS FRESH")
        print(f"   Performance within 20% of historical benchmark")
        print(f"   Safe to deploy for Monday")
    elif ic >= benchmark * 0.5:  # 50-80% of historical
        is_fresh = False
        message = f"⚠️  DEGRADED - Performance declined by {abs(decay_pct):.0f}% ({ic:.4f} vs {benchmark:.4f})"
        print(f"\n⚠️  VERDICT: MODEL DEGRADED")
        print(f"   Performance declined by {abs(decay_pct):.0f}%")
        print(f"   Consider retraining before deployment")
    else:  # <50% of historical
        is_fresh = False
        message = f"❌ DECAYED - Severe performance loss ({ic:.4f} vs {benchmark:.4f})"
        print(f"\n❌ VERDICT: MODEL SEVERELY DECAYED")
        print(f"   Performance <50% of historical benchmark")
        print(f"   MUST retrain before deployment")

    return ic, is_fresh, message

if __name__ == "__main__":
    # Validate all horizons
    horizons = ['1s', '5s', '10s']

    results = {}
    for horizon in horizons:
        print("\n\n")
        ic, is_fresh, msg = validate_freshness(f'labels_{horizon}', recent_days=5)
        results[horizon] = {'ic': ic, 'fresh': is_fresh, 'message': msg}

    # Summary
    print("\n\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    for horizon, result in results.items():
        if result['ic'] is not None:
            status = "✅" if result['fresh'] else "❌"
            print(f"{status} {horizon}: {result['message']}")
        else:
            print(f"❌ {horizon}: Validation failed")

    # Overall verdict
    print("\n" + "="*70)
    any_fresh = any(r['fresh'] for r in results.values() if r['ic'] is not None)

    if any_fresh:
        print("✅ DEPLOYMENT RECOMMENDATION: Model is fresh enough for Monday")
        print("   Monitor closely during first hour of live trading")
        sys.exit(0)
    else:
        print("❌ DEPLOYMENT RECOMMENDATION: Retrain before Monday")
        print("   Model has decayed too much for reliable trading")
        sys.exit(1)
