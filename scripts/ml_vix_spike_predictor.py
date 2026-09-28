#!/usr/bin/env python3
"""
ML VIX Spike Predictor — HC #714 R2+R4
Predicts VIX>30 events 1-5 days in advance using cross-asset signals.
If we can forecast spikes even 1 day early, we can position for the
"buy UPRO at VIX>30" play (#479: avg +7% in 1mo, 71% WR).

Features (HC #710 cross-asset):
- VIX term structure (VIX vs VIX3M, contango/backwardation)
- Credit spreads (HYG-IEF)
- Gold/Silver/Copper momentum
- Oil momentum
- Dollar strength (UUP)
- Bond trend (TLT)
- SPY momentum + volatility
- Put/call ratio proxy (via VIX shape)

Adversarial validation built-in (HC #705):
- Permutation test (100 shuffles)
- Sub-period consistency
- Walk-forward validation
- No DCA (HC #713)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
import warnings
import json
import os
from datetime import datetime

warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/ml_vix_spike_predictor'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("ML VIX SPIKE PREDICTOR — HC #714 R2+R4")
print("=" * 70)

# ── 1. Data Download ──────────────────────────────────────────────────
print("\n[1/6] Downloading cross-asset data...")

tickers = {
    'VIX': '^VIX',
    'VIX3M': '^VIX3M',  # 3-month VIX for term structure
    'SPY': 'SPY',
    'GLD': 'GLD',
    'SLV': 'SLV',
    'USO': 'USO',       # Oil
    'UUP': 'UUP',       # Dollar
    'TLT': 'TLT',       # Long bonds
    'HYG': 'HYG',       # High yield
    'IEF': 'IEF',       # Investment grade
    'CPER': 'CPER',     # Copper
    'BTC-USD': 'BTC-USD',
}

data = {}
for name, ticker in tickers.items():
    try:
        df = yf.download(ticker, start='2010-01-01', end='2026-07-17', progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        data[name] = df['Close'].rename(name)
        print(f"  {name}: {len(df)} bars")
    except Exception as e:
        print(f"  {name}: FAILED ({e})")

prices = pd.DataFrame(data)
prices = prices.ffill().dropna(how='all')
print(f"\nCombined: {len(prices)} rows, {prices.shape[1]} assets")
print(f"Date range: {prices.index[0]} to {prices.index[-1]}")

# ── 2. Feature Engineering ────────────────────────────────────────────
print("\n[2/6] Engineering features...")

feat = pd.DataFrame(index=prices.index)

# VIX features
feat['vix'] = prices['VIX']
feat['vix_5d_chg'] = prices['VIX'].pct_change(5)
feat['vix_10d_chg'] = prices['VIX'].pct_change(10)
feat['vix_20d_chg'] = prices['VIX'].pct_change(20)
feat['vix_pctile_63d'] = prices['VIX'].rolling(63).apply(lambda x: (x[-1] - x.min()) / (x.max() - x.min() + 1e-8), raw=True)
feat['vix_pctile_252d'] = prices['VIX'].rolling(252).apply(lambda x: (x[-1] - x.min()) / (x.max() - x.min() + 1e-8), raw=True)
feat['vix_zscore_20d'] = (prices['VIX'] - prices['VIX'].rolling(20).mean()) / prices['VIX'].rolling(20).std()

# VIX term structure
if 'VIX3M' in prices.columns:
    feat['vix_term_ratio'] = prices['VIX'] / prices['VIX3M']
    feat['vix_term_5d_chg'] = feat['vix_term_ratio'].pct_change(5)
    feat['vix_backwardation'] = (feat['vix_term_ratio'] > 1.0).astype(float)

# Credit spread (HYG-IEF)
if 'HYG' in prices.columns and 'IEF' in prices.columns:
    credit_spread = np.log(prices['HYG']) - np.log(prices['IEF'])
    feat['credit_spread'] = credit_spread
    feat['credit_spread_5d_chg'] = credit_spread.diff(5)
    feat['credit_spread_20d_chg'] = credit_spread.diff(20)
    feat['credit_zscore_20d'] = (credit_spread - credit_spread.rolling(20).mean()) / credit_spread.rolling(20).std()

# SPY features
feat['spy_ret_5d'] = prices['SPY'].pct_change(5)
feat['spy_ret_10d'] = prices['SPY'].pct_change(10)
feat['spy_ret_20d'] = prices['SPY'].pct_change(20)
feat['spy_vol_10d'] = prices['SPY'].pct_change().rolling(10).std() * np.sqrt(252)
feat['spy_vol_20d'] = prices['SPY'].pct_change().rolling(20).std() * np.sqrt(252)
feat['spy_vol_ratio'] = feat['spy_vol_10d'] / (feat['spy_vol_20d'] + 1e-8)
feat['spy_ma_ratio_50_200'] = prices['SPY'].rolling(50).mean() / prices['SPY'].rolling(200).mean()
feat['spy_drawdown'] = prices['SPY'] / prices['SPY'].rolling(252).max() - 1

# Gold features
if 'GLD' in prices.columns:
    feat['gold_ret_5d'] = prices['GLD'].pct_change(5)
    feat['gold_ret_20d'] = prices['GLD'].pct_change(20)
    feat['gold_vol_10d'] = prices['GLD'].pct_change().rolling(10).std() * np.sqrt(252)

# Silver features
if 'SLV' in prices.columns:
    feat['silver_ret_5d'] = prices['SLV'].pct_change(5)
    feat['gold_silver_ratio'] = prices['GLD'] / prices['SLV'] if 'GLD' in prices.columns else np.nan

# Oil features
if 'USO' in prices.columns:
    feat['oil_ret_5d'] = prices['USO'].pct_change(5)
    feat['oil_ret_20d'] = prices['USO'].pct_change(20)

# Dollar features
if 'UUP' in prices.columns:
    feat['dollar_ret_5d'] = prices['UUP'].pct_change(5)
    feat['dollar_ret_20d'] = prices['UUP'].pct_change(20)

# Bond features
if 'TLT' in prices.columns:
    feat['bond_ret_5d'] = prices['TLT'].pct_change(5)
    feat['bond_ret_20d'] = prices['TLT'].pct_change(20)
    feat['bond_vol_10d'] = prices['TLT'].pct_change().rolling(10).std() * np.sqrt(252)

# Copper features
if 'CPER' in prices.columns:
    feat['copper_ret_5d'] = prices['CPER'].pct_change(5)
    feat['copper_ret_20d'] = prices['CPER'].pct_change(20)

# Bitcoin features (if available)
if 'BTC-USD' in prices.columns:
    feat['btc_ret_5d'] = prices['BTC-USD'].pct_change(5)
    feat['btc_ret_20d'] = prices['BTC-USD'].pct_change(20)
    feat['btc_vol_10d'] = prices['BTC-USD'].pct_change().rolling(10).std() * np.sqrt(252)

# Cross-asset: rate of change divergences
feat['spy_vix_diverge'] = feat.get('spy_ret_5d', 0) + feat.get('vix_5d_chg', 0)  # normally inverse
feat['gold_spy_diverge'] = feat.get('gold_ret_5d', 0) - feat.get('spy_ret_5d', 0)

# Drop rows with NaN
feat = feat.dropna()
print(f"Features: {feat.shape[1]} columns, {len(feat)} rows")

# ── 3. Target: VIX crosses above 30 within N days ─────────────────────
print("\n[3/6] Building targets...")

vix_aligned = prices['VIX'].reindex(feat.index)

# Multiple horizons
targets = {}
for horizon in [1, 3, 5, 10]:
    # Will VIX be above 30 at any point in the next N days?
    future_max_vix = vix_aligned.rolling(horizon, min_periods=1).max().shift(-horizon)
    targets[f'spike_{horizon}d'] = (future_max_vix >= 30).astype(int)
    spike_rate = targets[f'spike_{horizon}d'].mean()
    spike_count = targets[f'spike_{horizon}d'].sum()
    print(f"  VIX>30 within {horizon}d: {spike_count:.0f} events ({spike_rate:.1%} of days)")

# Primary target: 5-day horizon (gives time to position)
target_col = 'spike_5d'
y = targets[target_col].reindex(feat.index).dropna()
X = feat.loc[y.index]

# Remove last N rows (no future data)
valid = ~y.isna()
X = X[valid]
y = y[valid]

print(f"\nPrimary target (5d): {len(X)} samples, {y.sum():.0f} positive ({y.mean():.1%})")

# ── 4. Walk-Forward Training ──────────────────────────────────────────
print("\n[4/6] Walk-forward training...")

# Time-series split: train on 3+ years, test on next year
min_train = 756  # ~3 years
test_size = 252  # ~1 year
n_splits = (len(X) - min_train) // test_size

print(f"  Samples: {len(X)}, Splits: {n_splits}, Train≥{min_train}, Test={test_size}")

all_preds = []
all_true = []
all_dates = []
fold_metrics = []

for i in range(n_splits):
    train_end = min_train + i * test_size
    test_end = min(train_end + test_size, len(X))

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    X_test = X.iloc[train_end:test_end]
    y_test = y.iloc[train_end:test_end]

    if len(X_test) == 0:
        break

    # GBM with conservative params to avoid overfitting
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=20,
        random_state=42
    )

    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    auc = roc_auc_score(y_test, proba) if y_test.nunique() > 1 else 0.5
    pred_binary = (proba > 0.5).astype(int)
    prec = precision_score(y_test, pred_binary, zero_division=0)
    rec = recall_score(y_test, pred_binary, zero_division=0)

    fold_dates = X_test.index
    print(f"  Fold {i+1}: {fold_dates[0].strftime('%Y-%m')} to {fold_dates[-1].strftime('%Y-%m')} | "
          f"AUC={auc:.3f} Prec={prec:.3f} Rec={rec:.3f} | "
          f"Spikes={y_test.sum():.0f}/{len(y_test)}")

    fold_metrics.append({
        'fold': i + 1,
        'start': str(fold_dates[0].date()),
        'end': str(fold_dates[-1].date()),
        'auc': auc,
        'precision': prec,
        'recall': rec,
        'n_spikes': int(y_test.sum()),
        'n_samples': len(y_test)
    })

    all_preds.extend(proba)
    all_true.extend(y_test.values)
    all_dates.extend(fold_dates)

# Overall OOS metrics
all_preds = np.array(all_preds)
all_true = np.array(all_true)

overall_auc = roc_auc_score(all_true, all_preds) if len(np.unique(all_true)) > 1 else 0.5
overall_pred = (all_preds > 0.5).astype(int)
overall_prec = precision_score(all_true, overall_pred, zero_division=0)
overall_rec = recall_score(all_true, overall_pred, zero_division=0)
overall_f1 = f1_score(all_true, overall_pred, zero_division=0)

print(f"\n  OVERALL OOS: AUC={overall_auc:.3f} Prec={overall_prec:.3f} "
      f"Rec={overall_rec:.3f} F1={overall_f1:.3f}")

# ── 5. Feature Importance ─────────────────────────────────────────────
print("\n[5/6] Feature importance (last fold)...")

importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 15 features:")
for feat_name, imp in importances.head(15).items():
    print(f"  {feat_name:30s} {imp:.4f}")

# ── 6. Adversarial Validation (HC #705) ───────────────────────────────
print("\n[6/6] Adversarial validation...")

# 6a. Permutation test (100 shuffles)
print("\n  [6a] Permutation test (100 shuffles)...")
perm_aucs = []
np.random.seed(42)
for s in range(100):
    y_perm = np.random.permutation(all_true)
    perm_auc = roc_auc_score(y_perm, all_preds) if len(np.unique(y_perm)) > 1 else 0.5
    perm_aucs.append(perm_auc)

perm_p = np.mean([pa >= overall_auc for pa in perm_aucs])
print(f"  Real AUC: {overall_auc:.3f} | Perm mean: {np.mean(perm_aucs):.3f} | p-value: {perm_p:.3f}")
perm_pass = perm_p < 0.05

# 6b. Sub-period consistency
print("\n  [6b] Sub-period consistency...")
n_periods = 3
period_size = len(all_preds) // n_periods
sub_aucs = []
for p in range(n_periods):
    start = p * period_size
    end = (p + 1) * period_size if p < n_periods - 1 else len(all_preds)
    sub_true = all_true[start:end]
    sub_preds = all_preds[start:end]
    if len(np.unique(sub_true)) > 1:
        sub_auc = roc_auc_score(sub_true, sub_preds)
    else:
        sub_auc = 0.5
    sub_aucs.append(sub_auc)
    print(f"  Period {p+1}: AUC={sub_auc:.3f}")

sub_cv = np.std(sub_aucs) / (np.mean(sub_aucs) + 1e-8)
sub_pass = all(a > 0.5 for a in sub_aucs) and sub_cv < 0.5
print(f"  CV: {sub_cv:.3f} | All > 0.5: {all(a > 0.5 for a in sub_aucs)} | PASS: {sub_pass}")

# 6c. Practical value: what happens if we buy UPRO when model predicts spike?
print("\n  [6c] Trading value analysis...")
# When model says "spike coming in 5d" with high confidence, what happens?
spy_rets = prices['SPY'].pct_change().reindex(pd.DatetimeIndex(all_dates))

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
for thresh in thresholds:
    signal = all_preds >= thresh
    n_signals = signal.sum()
    if n_signals > 0:
        # Count how many signals actually preceded a spike
        signal_true = all_true[signal]
        hit_rate = signal_true.mean()

        # Average forward 5d SPY return when signal fires (we want to SELL/hedge)
        # Or buy VIX products
        print(f"  Threshold {thresh:.1f}: {n_signals:4d} signals | "
              f"Hit rate: {hit_rate:.1%} | "
              f"True spikes caught: {int(signal_true.sum())}/{int(all_true.sum())}")

# ── Summary ───────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

gates_passed = sum([perm_pass, sub_pass, overall_auc > 0.65])
total_gates = 3

verdict = "PASS" if gates_passed >= 3 else ("MARGINAL" if gates_passed >= 2 else "FAIL")

results = {
    'model': 'GBM VIX Spike Predictor (5d horizon)',
    'date': datetime.now().isoformat(),
    'overall_auc': round(overall_auc, 4),
    'overall_precision': round(overall_prec, 4),
    'overall_recall': round(overall_rec, 4),
    'overall_f1': round(overall_f1, 4),
    'perm_p_value': round(perm_p, 4),
    'perm_pass': perm_pass,
    'sub_period_cv': round(sub_cv, 4),
    'sub_period_pass': sub_pass,
    'auc_above_065': overall_auc > 0.65,
    'gates_passed': f"{gates_passed}/{total_gates}",
    'verdict': verdict,
    'top_features': {k: round(v, 4) for k, v in importances.head(10).items()},
    'fold_metrics': fold_metrics,
}

print(f"\nOOS AUC: {overall_auc:.3f}")
print(f"Permutation: {'PASS' if perm_pass else 'FAIL'} (p={perm_p:.3f})")
print(f"Sub-period: {'PASS' if sub_pass else 'FAIL'} (CV={sub_cv:.3f})")
print(f"AUC > 0.65: {'PASS' if overall_auc > 0.65 else 'FAIL'}")
print(f"\nVERDICT: {verdict} ({gates_passed}/{total_gates} gates)")

if overall_auc > 0.65 and perm_pass:
    print("\n→ Model has genuine predictive power for VIX spikes.")
    print("→ Next step: build real-time signal watcher (HC #712 R3).")
elif overall_auc > 0.55:
    print("\n→ Model has marginal skill. Worth investigating feature subsets.")
    print("→ Cross-asset signals may add value as confirming indicators.")
else:
    print("\n→ VIX spikes are not reliably predictable with these features.")
    print("→ Stick to reactive strategy (buy AFTER VIX>30, per playbook #479).")

# Save results — convert numpy bools to Python bools for JSON
def make_serializable(obj):
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_serializable(i) for i in obj]
    return obj

with open(f'{OUTPUT_DIR}/results.json', 'w') as f:
    json.dump(make_serializable(results), f, indent=2)

# Save predictions
pred_df = pd.DataFrame({
    'date': all_dates,
    'true': all_true,
    'predicted_proba': all_preds,
    'predicted': overall_pred
})
pred_df.to_parquet(f'{OUTPUT_DIR}/predictions.parquet', index=False)

# === CRITICAL: Check if model predicts BEFORE spikes, not during ===
print("\n" + "="*70)
print("LEAKAGE CHECK: Does the model predict spikes when VIX is still LOW?")
print("="*70)

# Filter predictions where VIX < 25 at prediction time
feat_aligned = feat.loc[pred_df['date'].values]
low_vix_mask = feat_aligned['vix'].values < 25
high_vix_mask = feat_aligned['vix'].values >= 25

pred_low = pred_df[low_vix_mask]
pred_high = pred_df[high_vix_mask]

print(f"\nPredictions when VIX < 25: {len(pred_low)} days")
if pred_low['true'].sum() > 0:
    low_auc = roc_auc_score(pred_low['true'], pred_low['predicted_proba'])
    low_signals = (pred_low['predicted_proba'] > 0.5).sum()
    low_hits = ((pred_low['predicted_proba'] > 0.5) & pred_low['true']).sum()
    low_precision = low_hits / low_signals if low_signals > 0 else 0
    print(f"  AUC (low-VIX only): {low_auc:.3f}")
    print(f"  Signals at p>0.5: {low_signals}, True positives: {low_hits}, Precision: {low_precision:.1%}")
    if low_auc > 0.65:
        print("  → GENUINE EARLY WARNING: Model detects pre-spike conditions")
    else:
        print("  → MODEL LEAKS: Only works when VIX already elevated")
else:
    print("  No true spikes from low-VIX regime — can't test")

print(f"\nPredictions when VIX >= 25: {len(pred_high)} days")
if pred_high['true'].sum() > 0:
    high_auc = roc_auc_score(pred_high['true'], pred_high['predicted_proba'])
    print(f"  AUC (high-VIX only): {high_auc:.3f}")
else:
    print("  No true spikes from high-VIX regime — can't test")

# === Retrain WITHOUT vix level as feature (ablation) ===
print("\n" + "="*70)
print("ABLATION: Retrain WITHOUT raw VIX (remove circularity)")
print("="*70)

# Remove vix and vix_pctile features
ablation_cols = [c for c in X.columns if c not in ['vix', 'vix_pctile_252d', 'vix_pctile_63d']]
X_ablation = X[ablation_cols]

ab_preds, ab_true = [], []
for i in range(n_splits):
    train_end = min_train + i * test_size
    test_end = min(train_end + test_size, len(X_ablation))

    X_tr = X_ablation.iloc[:train_end]
    y_tr = y.iloc[:train_end]
    X_te = X_ablation.iloc[train_end:test_end]
    y_te = y.iloc[train_end:test_end]

    if len(X_te) == 0:
        break

    model_ab = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=20, random_state=42
    )
    model_ab.fit(X_tr, y_tr)
    proba = model_ab.predict_proba(X_te)[:, 1]
    ab_preds.extend(proba)
    ab_true.extend(y_te.values)

ab_true_arr = np.array(ab_true)
ab_preds_arr = np.array(ab_preds)
ab_auc = roc_auc_score(ab_true_arr, ab_preds_arr) if len(np.unique(ab_true_arr)) > 1 else 0.5
ab_pred_binary = (ab_preds_arr > 0.5).astype(int)
ab_prec = precision_score(ab_true_arr, ab_pred_binary, zero_division=0)
ab_rec = recall_score(ab_true_arr, ab_pred_binary, zero_division=0)

print(f"  Ablation AUC (no raw VIX): {ab_auc:.3f} (was {overall_auc:.3f})")
print(f"  Ablation Precision: {ab_prec:.3f} Recall: {ab_rec:.3f}")
if ab_auc > 0.65:
    print("  → Cross-asset signals have GENUINE predictive power beyond VIX level")
else:
    print("  → Model relies on VIX level — cross-asset signals don't add much")

# Feature importance for ablation model
ab_imp = pd.Series(model_ab.feature_importances_, index=ablation_cols).sort_values(ascending=False)
print(f"\n  Top 10 features (ablation):")
for feat, imp in ab_imp.head(10).items():
    print(f"    {feat:35s} {imp:.4f}")

print(f"\nResults saved to {OUTPUT_DIR}/")
print("DONE.")
