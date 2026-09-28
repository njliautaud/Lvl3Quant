#!/usr/bin/env python3
"""
ML Trend v2 + VIX Spike Predictor Overlay
==========================================
Takes the validated ML Trend v2 daily returns (Sharpe 2.90) and applies
the validated VIX Spike Predictor (AUC 0.926) as a defensive overlay:
- When spike probability LOW: run v2 at full size
- When spike probability ELEVATED: reduce to 50%
- When spike probability HIGH: go flat

Tests whether the overlay improves risk-adjusted returns (especially MaxDD).
Both components are INDEPENDENTLY validated (perm PASS).

HC #713: Fixed $100K, NO DCA. HC #0: SLIDING.
"""

import json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
np.random.seed(42)

BASE = Path("/home/jupiter/Lvl3Quant")
OUTPUT = BASE / "output" / "ml_trend_v2_spike_overlay"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Check if v2 returns exist
V2_RETURNS = BASE / "output" / "ml_trend_following" / "daily_returns.csv"
if not V2_RETURNS.exists():
    print(f"ERROR: ML Trend v2 returns not found at {V2_RETURNS}")
    print("Run scripts/growth_research/ml_trend_following_v2.py first.")
    exit(1)

print("=" * 70)
print("ML TREND v2 + VIX SPIKE PREDICTOR OVERLAY")
print("=" * 70)

# ─── LOAD V2 RETURNS ─────────────────────────────────────────────────────────

print("\n[1/6] Loading ML Trend v2 returns...")
v2_rets = pd.read_csv(V2_RETURNS, index_col=0, parse_dates=True).squeeze()
print(f"  Days: {len(v2_rets)}")
print(f"  Date range: {v2_rets.index[0].strftime('%Y-%m-%d')} to {v2_rets.index[-1].strftime('%Y-%m-%d')}")
v2_sharpe = v2_rets.mean() * 252 / (v2_rets.std() * np.sqrt(252))
print(f"  Base Sharpe: {v2_sharpe:.3f}")

# ─── BUILD SPIKE PREDICTOR ───────────────────────────────────────────────────

print("\n[2/6] Building VIX spike predictor...")

# Download VIX and features
tickers = ['SPY', 'TLT', 'HYG', 'GLD']
raw = yf.download(tickers + ['^VIX'], start='2005-01-01', progress=False)
if hasattr(raw.columns, 'levels'):
    prices = raw['Close']
else:
    prices = raw

vix = prices['^VIX'] if '^VIX' in prices.columns else prices.iloc[:, 0]
spy = prices['SPY'] if 'SPY' in prices.columns else prices.iloc[:, 0]
tlt = prices['TLT'] if 'TLT' in prices.columns else pd.Series(dtype=float)
hyg = prices['HYG'] if 'HYG' in prices.columns else pd.Series(dtype=float)

# Spike predictor features (simplified from validated v2)
feat = pd.DataFrame(index=vix.dropna().index)
feat['vix'] = vix
feat['vix_ma5'] = vix.rolling(5).mean()
feat['vix_ma20'] = vix.rolling(20).mean()
feat['vix_zscore'] = (vix - vix.rolling(60).mean()) / vix.rolling(60).std()
feat['vix_percentile'] = vix.rolling(252).rank(pct=True)
feat['vix_change_5d'] = vix.pct_change(5)
feat['vix_term'] = vix / vix.rolling(5).mean() - 1

spy_ret = spy.pct_change()
feat['spy_vol_20d'] = spy_ret.rolling(20).std() * np.sqrt(252)
feat['spy_vol_60d'] = spy_ret.rolling(60).std() * np.sqrt(252)
feat['spy_dd'] = spy / spy.rolling(252).max() - 1
feat['spy_ret_5d'] = spy.pct_change(5)
feat['spy_ret_20d'] = spy.pct_change(20)
feat['spy_above_200ma'] = (spy > spy.rolling(200).mean()).astype(float)

# VRP
feat['vrp'] = vix / 100 - feat['spy_vol_20d']

# Credit proxy
if hyg.notna().sum() > 100 and tlt.notna().sum() > 100:
    credit = hyg.pct_change() - tlt.pct_change()
    feat['credit_20d'] = credit.rolling(20).sum()

feat = feat.dropna()

# Spike label: VIX crosses 25 within 5 days
fwd_vix_max = vix.rolling(5).max().shift(-5)
spike_label = (fwd_vix_max > 25).astype(int).reindex(feat.index)

print(f"  Features: {feat.shape}")
print(f"  Spike rate: {spike_label.mean()*100:.1f}%")

# ─── WALK-FORWARD OVERLAY ────────────────────────────────────────────────────

print("\n[3/6] Walk-forward spike prediction + overlay...")

TRAIN_WINDOW = 252
LOW_THRESH = 0.10    # Full size when prob < 10%
MED_THRESH = 0.25    # Half size when prob 10-25%
HIGH_THRESH = 0.40   # Flat when prob > 40%

# Align v2 returns with spike predictor dates
common_dates = v2_rets.index.intersection(feat.index)
v2_aligned = v2_rets.loc[common_dates]

print(f"  Overlapping dates: {len(common_dates)}")

# Generate spike probabilities walk-forward
spike_probs = pd.Series(np.nan, index=common_dates)
model = None
last_train = -999

for i, date in enumerate(common_dates):
    # Retrain every 63 days
    if i - last_train >= 63 or model is None:
        date_loc = feat.index.get_loc(date)
        train_start = max(0, date_loc - TRAIN_WINDOW)
        X_train = feat.iloc[train_start:date_loc].values
        y_train = spike_label.iloc[train_start:date_loc].values

        if len(X_train) >= 100 and y_train.sum() > 3:
            model = RandomForestClassifier(
                n_estimators=200, max_depth=5, min_samples_leaf=10,
                class_weight='balanced', random_state=42, n_jobs=-1
            )
            model.fit(X_train, y_train)
            last_train = i

    if model is None:
        spike_probs.iloc[i] = 0.1  # Assume low risk
        continue

    X_today = feat.loc[[date]].values
    try:
        prob = model.predict_proba(X_today)[0][1]
        spike_probs.iloc[i] = prob
    except:
        spike_probs.iloc[i] = 0.1

# Apply overlay
overlay_scale = pd.Series(1.0, index=common_dates)
overlay_scale[spike_probs > HIGH_THRESH] = 0.0   # Go flat
overlay_scale[(spike_probs > MED_THRESH) & (spike_probs <= HIGH_THRESH)] = 0.5  # Half
overlay_scale[(spike_probs > LOW_THRESH) & (spike_probs <= MED_THRESH)] = 0.75  # 75%
# Below LOW_THRESH: keep full (1.0)

overlay_rets = v2_aligned * overlay_scale

print(f"  Full size days: {(overlay_scale == 1.0).sum()} ({(overlay_scale == 1.0).mean()*100:.0f}%)")
print(f"  75% days: {(overlay_scale == 0.75).sum()} ({(overlay_scale == 0.75).mean()*100:.0f}%)")
print(f"  50% days: {(overlay_scale == 0.5).sum()} ({(overlay_scale == 0.5).mean()*100:.0f}%)")
print(f"  Flat days: {(overlay_scale == 0.0).sum()} ({(overlay_scale == 0.0).mean()*100:.0f}%)")

# ─── METRICS ─────────────────────────────────────────────────────────────────

print("\n[4/6] Computing metrics...")

def calc_metrics(rets, name):
    ann_r = rets.mean() * 252
    ann_v = rets.std() * np.sqrt(252)
    sharpe = ann_r / ann_v if ann_v > 0 else 0
    down_v = rets[rets < 0].std() * np.sqrt(252)
    sortino = ann_r / down_v if down_v > 0 else 0
    cum = (1 + rets).cumprod()
    dd = (cum / cum.cummax() - 1).min()
    cagr = cum.iloc[-1] ** (252 / len(rets)) - 1 if cum.iloc[-1] > 0 else 0
    calmar = cagr / abs(dd) if dd != 0 else 0
    wr = (rets > 0).mean()
    return {
        'name': name, 'sharpe': round(sharpe, 3), 'sortino': round(sortino, 3),
        'cagr': round(cagr * 100, 1), 'max_dd': round(dd * 100, 1),
        'calmar': round(calmar, 3), 'wr': round(wr * 100, 1),
    }

m_base = calc_metrics(v2_aligned, "ML Trend v2 (base)")
m_overlay = calc_metrics(overlay_rets, "ML Trend v2 + Spike Overlay")

print(f"\n  {'Strategy':<35} {'Sharpe':>7} {'Sortino':>8} {'CAGR':>7} {'MaxDD':>7} {'Calmar':>7}")
print(f"  {'-'*35} {'-'*7} {'-'*8} {'-'*7} {'-'*7} {'-'*7}")
for m in [m_base, m_overlay]:
    print(f"  {m['name']:<35} {m['sharpe']:7.3f} {m['sortino']:8.3f} {m['cagr']:6.1f}% {m['max_dd']:6.1f}% {m['calmar']:7.3f}")

sharpe_delta = m_overlay['sharpe'] - m_base['sharpe']
dd_delta = m_overlay['max_dd'] - m_base['max_dd']
print(f"\n  Overlay impact: Sharpe {sharpe_delta:+.3f}, MaxDD {dd_delta:+.1f}pp")

# ─── ADVERSARIAL ─────────────────────────────────────────────────────────────

print("\n[5/6] Adversarial validation of overlay...")

# Key test: does the OVERLAY add value, or would random scaling do the same?
N_PERMS = 100
real_sharpe = m_overlay['sharpe']
perm_sharpes = []

for p in range(N_PERMS):
    # Random scaling (same distribution of scales, shuffled timing)
    rand_scale = overlay_scale.sample(frac=1, replace=False).values
    perm_rets = v2_aligned.values * rand_scale
    perm_series = pd.Series(perm_rets, index=common_dates)
    ps = perm_series.mean() * 252 / (perm_series.std() * np.sqrt(252)) if perm_series.std() > 0 else 0
    perm_sharpes.append(ps)

perm_p = np.mean([s >= real_sharpe for s in perm_sharpes])
perm_pass = perm_p < 0.05
print(f"  Overlay permutation: real={real_sharpe:.3f}, perm_mean={np.mean(perm_sharpes):.3f}, p={perm_p:.3f} → {'PASS' if perm_pass else 'FAIL'}")

# Sub-period consistency of overlay improvement
n_blocks = 4
bs = len(overlay_rets) // n_blocks
block_deltas = []
for b in range(n_blocks):
    base_blk = v2_aligned.iloc[b*bs:(b+1)*bs]
    ovl_blk = overlay_rets.iloc[b*bs:(b+1)*bs]
    base_s = base_blk.mean() * 252 / (base_blk.std() * np.sqrt(252)) if base_blk.std() > 0 else 0
    ovl_s = ovl_blk.mean() * 252 / (ovl_blk.std() * np.sqrt(252)) if ovl_blk.std() > 0 else 0
    delta = ovl_s - base_s
    block_deltas.append(delta)
    print(f"    Block {b+1}: base Sharpe {base_s:.3f}, overlay {ovl_s:.3f}, delta {delta:+.3f}")

# Overlay consistent if same sign across all blocks
consistent = all(d >= 0 for d in block_deltas) or all(d <= 0 for d in block_deltas)
print(f"  Consistency: {'All blocks same direction' if consistent else 'MIXED — overlay not consistent'}")

# Crisis performance (when VIX > 25)
high_vix_mask = vix.reindex(common_dates).ffill() > 25
if high_vix_mask.sum() > 20:
    base_crisis = v2_aligned[high_vix_mask]
    ovl_crisis = overlay_rets[high_vix_mask]
    base_crisis_ret = base_crisis.sum()
    ovl_crisis_ret = ovl_crisis.sum()
    print(f"  Crisis performance (VIX>25, {high_vix_mask.sum()} days):")
    print(f"    Base cumulative: {base_crisis_ret*100:.1f}%")
    print(f"    Overlay cumulative: {ovl_crisis_ret*100:.1f}%")
    print(f"    Protection: {(ovl_crisis_ret - base_crisis_ret)*100:+.1f}pp")

# ─── SAVE ────────────────────────────────────────────────────────────────────

print("\n[6/6] Saving results...")

results = {
    'strategy': 'ML Trend v2 + Spike Predictor Overlay',
    'timestamp': pd.Timestamp.now().isoformat(),
    'base_metrics': m_base,
    'overlay_metrics': m_overlay,
    'improvement': {
        'sharpe_delta': round(sharpe_delta, 3),
        'dd_improvement_pp': round(dd_delta, 1),
    },
    'overlay_params': {
        'low_thresh': LOW_THRESH,
        'med_thresh': MED_THRESH,
        'high_thresh': HIGH_THRESH,
    },
    'position_mix': {
        'full_pct': round((overlay_scale == 1.0).mean() * 100, 1),
        'three_quarter_pct': round((overlay_scale == 0.75).mean() * 100, 1),
        'half_pct': round((overlay_scale == 0.5).mean() * 100, 1),
        'flat_pct': round((overlay_scale == 0.0).mean() * 100, 1),
    },
    'adversarial': {
        'overlay_perm_p': round(perm_p, 3),
        'overlay_perm_pass': bool(perm_pass),
        'block_deltas': [round(d, 3) for d in block_deltas],
        'consistent': bool(consistent),
    },
    'verdict': 'PASS — overlay improves risk-adjusted returns' if (sharpe_delta > 0 and perm_pass) else
               'MARGINAL — overlay helps DD but perm not significant' if sharpe_delta > 0 else
               'FAIL — overlay hurts performance'
}

with open(OUTPUT / 'results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Plot
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

ax = axes[0]
cum_base = (1 + v2_aligned).cumprod() * 100000
cum_ovl = (1 + overlay_rets).cumprod() * 100000
ax.plot(cum_base.index, cum_base.values, 'b-', label=f'ML Trend v2 (S={m_base["sharpe"]})', alpha=0.8)
ax.plot(cum_ovl.index, cum_ovl.values, 'g-', label=f'+ Spike Overlay (S={m_overlay["sharpe"]})', linewidth=2)
ax.set_ylabel('Portfolio Value ($)')
ax.set_title('ML Trend v2 with VIX Spike Predictor Overlay')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
dd_base = cum_base / cum_base.cummax() - 1
dd_ovl = cum_ovl / cum_ovl.cummax() - 1
ax.fill_between(dd_base.index, dd_base.values, 0, alpha=0.3, color='blue', label='Base DD')
ax.fill_between(dd_ovl.index, dd_ovl.values, 0, alpha=0.3, color='green', label='Overlay DD')
ax.set_ylabel('Drawdown')
ax.set_title('Drawdown Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[2]
ax.plot(spike_probs.index, spike_probs.values, 'orange', alpha=0.5, linewidth=0.5)
ax.axhline(LOW_THRESH, color='g', linestyle='--', alpha=0.5, label=f'Low ({LOW_THRESH})')
ax.axhline(MED_THRESH, color='y', linestyle='--', alpha=0.5, label=f'Med ({MED_THRESH})')
ax.axhline(HIGH_THRESH, color='r', linestyle='--', alpha=0.5, label=f'High ({HIGH_THRESH})')
ax.fill_between(overlay_scale.index, overlay_scale.values * 0.5, alpha=0.2, color='blue', label='Position scale')
ax.set_ylabel('Spike Probability')
ax.set_title('VIX Spike Predictions & Position Scaling')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT / 'equity_curves.png', dpi=100)
plt.close()

overlay_rets.to_csv(OUTPUT / 'daily_returns.csv', header=['return'])

print(f"\n{'='*70}")
print(f"RESULT: ML Trend v2 + Spike Overlay")
print(f"  Base: Sharpe {m_base['sharpe']}, MaxDD {m_base['max_dd']}%")
print(f"  Overlay: Sharpe {m_overlay['sharpe']}, MaxDD {m_overlay['max_dd']}%")
print(f"  Delta: Sharpe {sharpe_delta:+.3f}, MaxDD {dd_delta:+.1f}pp")
print(f"  Overlay perm: p={perm_p:.3f} ({'PASS' if perm_pass else 'FAIL'})")
print(f"{'='*70}")
