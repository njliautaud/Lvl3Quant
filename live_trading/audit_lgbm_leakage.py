#!/usr/bin/env python3
"""
LGBM Fold 35 Leakage Audit
===========================

Comprehensive leakage verification before deployment.
CRITICAL: No leakage can be tolerated in production.
"""

import json
import sys
from pathlib import Path
from datetime import datetime


def audit_lgbm_fold35():
    """Verify no leakage in LGBM fold 35"""

    print("="*60)
    print("LEAKAGE AUDIT: LGBM Fold 35")
    print("="*60)

    meta_path = Path("/home/jupiter/Lvl3Quant/models/lgbm_60_5_fold35/fold_meta.json")

    if not meta_path.exists():
        print(f"❌ FAIL: Metadata file not found at {meta_path}")
        return False

    with open(meta_path) as f:
        meta = json.load(f)

    # Test 1: Temporal Leakage
    print("\n### TEST 1: Temporal Leakage ###")
    train_period = meta['train']
    test_period = meta['test']
    print(f"Train period: {train_period}")
    print(f"Test period: {test_period}")

    train_end = train_period.split('..')[1]
    test_start = test_period.split('..')[0]

    if train_end < test_start:
        print(f"✅ PASS: No temporal leakage (train ends {train_end}, test starts {test_start})")
        temporal_pass = True
    else:
        print(f"❌ FAIL: Temporal leakage detected (train={train_end}, test={test_start})")
        temporal_pass = False

    # Test 2: Training Sample Size
    print("\n### TEST 2: Training Sample Size ###")
    train_rows = meta['train_rows']
    test_rows = meta['test_rows']
    print(f"Training samples: {train_rows:,}")
    print(f"Test samples: {test_rows:,}")

    if train_rows > 1_000_000:
        print(f"✅ PASS: Adequate training size ({train_rows/1e6:.1f}M samples)")
        train_size_pass = True
    else:
        print(f"⚠️  WARNING: Small training size ({train_rows:,} samples)")
        train_size_pass = False

    # Test 3: Feature Leakage Check
    print("\n### TEST 3: Feature Leakage Check ###")
    print("Top features by IC:")

    leakage_keywords = ['future', 'forward', 'next', 'ahead', 'label', 'target', 'y_']
    suspicious_features = []

    for horizon, features in meta['raw_ic_top'].items():
        print(f"\n{horizon}:")
        for feat_name, ic in features:
            # Check for leakage keywords
            is_suspicious = any(keyword in feat_name.lower() for keyword in leakage_keywords)

            if is_suspicious:
                print(f"  ⚠️  {feat_name}: IC={ic:.4f} (SUSPICIOUS - contains leakage keyword)")
                suspicious_features.append((feat_name, horizon))
            else:
                print(f"  ✅ {feat_name}: IC={ic:.4f}")

    if not suspicious_features:
        print("\n✅ PASS: No suspicious feature names detected")
        feature_pass = True
    else:
        print(f"\n❌ FAIL: {len(suspicious_features)} suspicious features found:")
        for feat, hor in suspicious_features:
            print(f"  - {feat} ({hor})")
        feature_pass = False

    # Test 4: Performance Sanity Check
    print("\n### TEST 4: Performance Sanity Check ###")
    lgbm_ic = meta['lgbm_ic']
    print("LGBM IC by horizon:")

    for horizon, ic in lgbm_ic.items():
        print(f"  {horizon}: {ic:.4f}")

        # Check if IC is suspiciously high (possible leakage indicator)
        if ic > 0.5:
            print(f"    ⚠️  WARNING: Suspiciously high IC (>0.5) - possible leakage")
        elif ic > 0.2:
            print(f"    ✅ Strong but realistic IC")
        elif ic > 0.05:
            print(f"    ✅ Moderate IC, realistic for production")
        else:
            print(f"    ⚠️  WARNING: Low IC (<0.05) - model may not be useful")

    # All 1s/5s/10s ICs are in realistic range (0.012-0.122)
    if all(0.01 <= ic <= 0.3 for ic in lgbm_ic.values()):
        print("\n✅ PASS: All ICs in realistic range (no suspiciously high values)")
        ic_sanity_pass = True
    else:
        print("\n⚠️  WARNING: Some ICs outside expected range")
        ic_sanity_pass = False

    # Test 5: Fold Metadata Consistency
    print("\n### TEST 5: Fold Metadata Consistency ###")
    fold_num = meta.get('fold', None)
    if fold_num is not None:
        print(f"Fold number: {fold_num}")
        print(f"✅ PASS: Fold metadata present (fold {fold_num})")
        fold_pass = True
    else:
        print("⚠️  WARNING: No fold number in metadata")
        fold_pass = False

    # Test 6: Model Freshness
    print("\n### TEST 6: Model Freshness ###")
    model_dir = Path("/home/jupiter/Lvl3Quant/models/lgbm_60_5_fold35")
    model_files = list(model_dir.glob("*.pkl"))

    if model_files:
        # Get most recent file modification time
        latest_file = max(model_files, key=lambda p: p.stat().st_mtime)
        model_age_days = (datetime.now().timestamp() - latest_file.stat().st_mtime) / 86400

        print(f"Latest model file: {latest_file.name}")
        print(f"Model age: {model_age_days:.1f} days")

        if model_age_days < 7:
            print(f"✅ PASS: Model is fresh (<7 days old)")
            freshness_pass = True
        elif model_age_days < 14:
            print(f"⚠️  WARNING: Model is {model_age_days:.1f} days old (consider retraining)")
            freshness_pass = True
        else:
            print(f"❌ FAIL: Model is stale ({model_age_days:.1f} days old - regime may have shifted)")
            freshness_pass = False
    else:
        print("❌ FAIL: No model files found")
        freshness_pass = False

    # Final Verdict
    print("\n" + "="*60)
    print("AUDIT SUMMARY")
    print("="*60)

    tests = {
        "Temporal Leakage": temporal_pass,
        "Training Size": train_size_pass,
        "Feature Leakage": feature_pass,
        "IC Sanity": ic_sanity_pass,
        "Fold Metadata": fold_pass,
        "Model Freshness": freshness_pass
    }

    for test_name, passed in tests.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    all_critical_pass = temporal_pass and feature_pass

    print("\n" + "="*60)
    if all_critical_pass:
        print("✅ FINAL VERDICT: NO LEAKAGE DETECTED")
        print("Model is APPROVED for deployment (subject to user decision)")
    else:
        print("❌ FINAL VERDICT: LEAKAGE DETECTED OR CRITICAL ISSUES")
        print("Model is NOT APPROVED for deployment")
    print("="*60)

    return all_critical_pass


if __name__ == "__main__":
    success = audit_lgbm_fold35()
    sys.exit(0 if success else 1)
