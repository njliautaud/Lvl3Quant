#!/usr/bin/env python3
"""
Portfolio Combination: ML Trend v2 (8 CTA) + ML Sector Rotation (11 sectors)
=============================================================================
Our two validated regime-agnostic strategies. Test:
1. Correlation of daily returns between strategies
2. Optimal blend weights (50/50, risk-parity, mean-variance)
3. Combined portfolio metrics (Sharpe, Sortino, MaxDD, Calmar)
4. Adversarial validation of combination
5. Does combining reduce outlier dependence (the one weakness)?

HC #713: Fixed $100K, NO DCA. HC #0: SLIDING window.
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier

warnings.filterwarnings('ignore')
np.random.seed(42)

BASE = Path("/home/jupiter/Lvl3Quant")
OUTPUT = BASE / "output" / "portfolio_combination"
OUTPUT.mkdir(parents=True, exist_ok=True)

INITIAL_CAPITAL = 100_000
TRAIN_WINDOW = 252
MA_SHORT = 20
MA_LONG = 100
ML_THRESHOLD = 0.55
TARGET_VOL = 0.10
REBAL_COST_BPS = 10
N_PERMUTATIONS = 100

print("=" * 70)
print("PORTFOLIO COMBINATION: ML TREND v2 + ML SECTOR ROTATION")
print("=" * 70)

# ─── DATA DOWNLOAD ──────────────────────────────────────────────────────────

# CTA Universe (ML Trend v2)
CTA_ASSETS = ['SPY', 'TLT', 'GLD', 'UUP', 'USO', 'EEM', 'VNQ', 'HYG']
# Sector Universe (ML Sector Rotation)
SECTOR_ASSETS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU', 'XLRE', 'XLC']

ALL_TICKERS = list(set(CTA_ASSETS + SECTOR_ASSETS + ['SPY', 'VIX']))  # VIX for features

print("\n[1/8] Downloading data...")
t0 = time.time()

# Try cache first
cache_file = BASE / "data" / "portfolio_combo_cache.parquet"
if cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < 86400:
    prices = pd.read_parquet(cache_file)
    print(f"  Loaded from cache: {prices.shape}")
else:
    raw = yf.download(ALL_TICKERS, start='2000-01-01', progress=False)
    if hasattr(raw.columns, 'levels'):
        prices = raw['Close']
    else:
        prices = raw
    prices.to_parquet(cache_file)
    print(f"  Downloaded: {prices.shape}")

# VIX for regime features
try:
    vix_raw = yf.download('^VIX', start='2000-01-01', progress=False)
    if hasattr(vix_raw.columns, 'levels'):
        vix = vix_raw['Close'].squeeze()
    else:
        vix = vix_raw['Close']
    vix.name = 'VIX'
except:
    vix = pd.Series(dtype=float)

print(f"  Time: {time.time()-t0:.1f}s")

# ─── HELPER FUNCTIONS ────────────────────────────────────────────────────────

def compute_features(prices_subset, vix_series, asset):
    """ML features for trend filtering"""
    p = prices_subset[asset].dropna()
    ret = p.pct_change()
    
    features = pd.DataFrame(index=p.index)
    # Trend features
    ma_s = p.rolling(MA_SHORT).mean()
    ma_l = p.rolling(MA_LONG).mean()
    features['trend_strength'] = (ma_s - ma_l) / ma_l
    features['momentum_20d'] = p.pct_change(20)
    features['momentum_60d'] = p.pct_change(60)
    features['vol_20d'] = ret.rolling(20).std() * np.sqrt(252)
    features['vol_ratio'] = ret.rolling(20).std() / ret.rolling(60).std()
    features['drawdown'] = p / p.rolling(252).max() - 1
    features['rsi'] = compute_rsi(p, 14)
    
    # Cross-asset alignment
    n_trending = 0
    for other in prices_subset.columns:
        if other != asset:
            op = prices_subset[other].dropna().reindex(p.index)
            if len(op.dropna()) > MA_LONG:
                oma_s = op.rolling(MA_SHORT).mean()
                oma_l = op.rolling(MA_LONG).mean()
                n_trending += ((oma_s > oma_l).astype(int) * 2 - 1)
    features['cross_asset_trend'] = n_trending / max(1, len(prices_subset.columns) - 1)
    
    # VIX features
    if len(vix_series) > 0:
        vix_aligned = vix_series.reindex(p.index).ffill()
        features['vix_level'] = vix_aligned
        features['vix_ma20'] = vix_aligned.rolling(20).mean()
        features['vix_zscore'] = (vix_aligned - vix_aligned.rolling(60).mean()) / vix_aligned.rolling(60).std()
    
    return features.dropna()

def compute_rsi(prices, window=14):
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def run_ml_trend_strategy(prices_subset, vix_series, universe, label="Strategy"):
    """Run ML trend strategy on a given universe. Returns daily returns series."""
    print(f"  Running {label} ({len(universe)} assets)...")
    
    # Get common date range
    available = [a for a in universe if a in prices_subset.columns]
    if len(available) < 3:
        print(f"    Only {len(available)} assets available, skipping")
        return pd.Series(dtype=float)
    
    px = prices_subset[available].dropna(how='all')
    start_date = px.dropna(thresh=3).index[TRAIN_WINDOW + MA_LONG + 10]
    
    all_returns = []
    dates = px.index[px.index >= start_date]
    
    # Walk-forward
    daily_rets = pd.Series(0.0, index=dates)
    
    for i, date in enumerate(dates):
        if i % TRAIN_WINDOW == 0 or i == 0:
            # Retrain ML model
            train_end_idx = px.index.get_loc(date)
            train_start_idx = max(0, train_end_idx - TRAIN_WINDOW - MA_LONG)
            train_data = px.iloc[train_start_idx:train_end_idx]
            
            # Build training features and labels
            X_train_all, y_train_all = [], []
            for asset in available:
                if asset not in train_data.columns:
                    continue
                feat = compute_features(train_data, vix_series, asset)
                if len(feat) < 50:
                    continue
                
                # Label: was trend continuation profitable? (next 20d return in trend direction)
                asset_ret = train_data[asset].pct_change(20).shift(-20)
                ma_s = train_data[asset].rolling(MA_SHORT).mean()
                ma_l = train_data[asset].rolling(MA_LONG).mean()
                trend_dir = (ma_s > ma_l).astype(int) * 2 - 1
                label_vals = (asset_ret * trend_dir > 0).astype(int)
                
                common = feat.index.intersection(label_vals.dropna().index)
                if len(common) < 30:
                    continue
                X_train_all.append(feat.loc[common].values)
                y_train_all.append(label_vals.loc[common].values)
            
            if len(X_train_all) == 0:
                continue
            
            X_train = np.vstack(X_train_all)
            y_train = np.concatenate(y_train_all)
            
            # Train GBM
            model = GradientBoostingClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                subsample=0.8, random_state=42
            )
            model.fit(X_train, y_train)
        
        # Generate positions for today
        positions = {}
        total_risk = 0
        
        for asset in available:
            if asset not in px.columns:
                continue
            
            loc = px.index.get_loc(date)
            if loc < MA_LONG + 20:
                continue
            
            # Check trend
            recent = px[asset].iloc[max(0,loc-MA_LONG-5):loc+1]
            if len(recent) < MA_LONG:
                continue
            
            ma_s = recent.iloc[-MA_SHORT:].mean()
            ma_l = recent.iloc[-MA_LONG:].mean()
            
            if ma_s == ma_l:
                continue
            
            trend_dir = 1 if ma_s > ma_l else -1
            
            # ML filter
            try:
                window_data = px.iloc[max(0,loc-MA_LONG-20):loc+1]
                feat = compute_features(window_data, vix_series, asset)
                if len(feat) == 0:
                    continue
                last_feat = feat.iloc[-1:].values
                prob = model.predict_proba(last_feat)[0][1]
            except:
                continue
            
            if prob < ML_THRESHOLD:
                continue
            
            # Vol targeting
            asset_ret = px[asset].pct_change().iloc[max(0,loc-20):loc+1]
            vol = asset_ret.std() * np.sqrt(252)
            if vol < 0.01:
                continue
            
            weight = min(TARGET_VOL / vol, 0.3)  # cap at 30% per position
            positions[asset] = trend_dir * weight
        
        # Calculate portfolio return
        if positions:
            port_ret = 0
            for asset, weight in positions.items():
                loc = px.index.get_loc(date)
                if loc + 1 < len(px):
                    asset_ret = px[asset].iloc[loc+1] / px[asset].iloc[loc] - 1
                    port_ret += weight * asset_ret
            
            # Transaction costs (approximate)
            daily_rets.iloc[i] = port_ret - abs(port_ret) * REBAL_COST_BPS / 10000
        
    return daily_rets

# ─── RUN BOTH STRATEGIES ─────────────────────────────────────────────────────

print("\n[2/8] Running ML Trend v2 (CTA 8-asset)...")
cta_prices = prices[[c for c in CTA_ASSETS if c in prices.columns]].copy()
cta_returns = run_ml_trend_strategy(cta_prices, vix, CTA_ASSETS, "CTA Trend")

print("\n[3/8] Running ML Sector Rotation (11 sectors)...")
sector_prices = prices[[c for c in SECTOR_ASSETS if c in prices.columns]].copy()
sector_returns = run_ml_trend_strategy(sector_prices, vix, SECTOR_ASSETS, "Sector Rotation")

# ─── COMBINE AND ANALYZE ─────────────────────────────────────────────────────

print("\n[4/8] Combining strategies...")

# Align dates
common_dates = cta_returns.index.intersection(sector_returns.index)
cta_aligned = cta_returns.loc[common_dates]
sec_aligned = sector_returns.loc[common_dates]

# Correlation
corr = cta_aligned.corr(sec_aligned)
print(f"  Daily return correlation: {corr:.4f}")
print(f"  Common trading days: {len(common_dates)}")

# Portfolio blends
blends = {
    '100% CTA': (1.0, 0.0),
    '75/25 CTA/Sector': (0.75, 0.25),
    '50/50': (0.50, 0.50),
    '25/75 CTA/Sector': (0.25, 0.75),
    '100% Sector': (0.0, 1.0),
}

# Risk-parity weights
cta_vol = cta_aligned.std() * np.sqrt(252)
sec_vol = sec_aligned.std() * np.sqrt(252)
if cta_vol > 0 and sec_vol > 0:
    rp_cta = (1/cta_vol) / (1/cta_vol + 1/sec_vol)
    rp_sec = 1 - rp_cta
    blends['Risk Parity'] = (rp_cta, rp_sec)
    print(f"  Risk Parity weights: CTA={rp_cta:.1%}, Sector={rp_sec:.1%}")

print("\n[5/8] Computing metrics for all blends...")

def compute_metrics(returns, label):
    """Compute strategy metrics"""
    if len(returns) == 0 or returns.std() == 0:
        return None
    
    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    
    downside = returns[returns < 0].std() * np.sqrt(252)
    sortino = ann_ret / downside if downside > 0 else 0
    
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()
    
    cagr = (cum.iloc[-1] ** (252 / len(returns))) - 1
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    
    # Win rate
    wr = (returns > 0).mean()
    
    # Profit factor
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    return {
        'label': label,
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'cagr': round(cagr * 100, 1),
        'max_dd': round(max_dd * 100, 1),
        'calmar': round(calmar, 3),
        'annual_vol': round(ann_vol * 100, 1),
        'win_rate': round(wr * 100, 1),
        'profit_factor': round(pf, 3),
        'n_days': len(returns),
        'years': round(len(returns) / 252, 1),
    }

results = {}
blend_returns = {}

for name, (w_cta, w_sec) in blends.items():
    combo = w_cta * cta_aligned + w_sec * sec_aligned
    metrics = compute_metrics(combo, name)
    if metrics:
        results[name] = metrics
        blend_returns[name] = combo
        print(f"  {name:25s} | Sharpe {metrics['sharpe']:5.2f} | Sortino {metrics['sortino']:5.2f} | "
              f"CAGR {metrics['cagr']:5.1f}% | MaxDD {metrics['max_dd']:6.1f}% | Calmar {metrics['calmar']:5.2f}")

# ─── ADVERSARIAL VALIDATION ON BEST BLEND ─────────────────────────────────────

print("\n[6/8] Adversarial validation on best blend...")

# Find best Sharpe
best_name = max(results.keys(), key=lambda k: results[k]['sharpe'])
best_returns = blend_returns[best_name]
print(f"  Best blend: {best_name} (Sharpe {results[best_name]['sharpe']})")

# Permutation test: shuffle CTA/Sector alignment
print(f"  Running {N_PERMUTATIONS} permutations...")
real_sharpe = results[best_name]['sharpe']
perm_sharpes = []

for p in range(N_PERMUTATIONS):
    # Shuffle one strategy's returns (break temporal alignment)
    shuffled_sec = sec_aligned.sample(frac=1, replace=False).values
    w_cta, w_sec = blends[best_name]
    perm_combo = w_cta * cta_aligned.values + w_sec * shuffled_sec
    perm_ret = pd.Series(perm_combo, index=common_dates)
    perm_sharpe = perm_ret.mean() * 252 / (perm_ret.std() * np.sqrt(252))
    perm_sharpes.append(perm_sharpe)

perm_p = np.mean([s >= real_sharpe for s in perm_sharpes])
perm_pass = perm_p < 0.05
print(f"  Permutation: real={real_sharpe:.3f}, perm_mean={np.mean(perm_sharpes):.3f}, p={perm_p:.3f} → {'PASS' if perm_pass else 'FAIL'}")

# Sub-period consistency
n_blocks = 4
block_size = len(best_returns) // n_blocks
block_sharpes = []
for b in range(n_blocks):
    block = best_returns.iloc[b*block_size:(b+1)*block_size]
    bs = block.mean() * 252 / (block.std() * np.sqrt(252)) if block.std() > 0 else 0
    block_sharpes.append(bs)
    print(f"    Block {b+1}: Sharpe {bs:.3f}")

sub_cv = np.std(block_sharpes) / max(abs(np.mean(block_sharpes)), 0.01)
sub_pass = sub_cv < 0.50
print(f"  Sub-period CV: {sub_cv:.3f} → {'PASS' if sub_pass else 'FAIL'}")

# Outlier robustness (remove top/bottom 1% of days)
trimmed = best_returns[(best_returns > best_returns.quantile(0.01)) & 
                        (best_returns < best_returns.quantile(0.99))]
trim_sharpe = trimmed.mean() * 252 / (trimmed.std() * np.sqrt(252))
outlier_deg = (trim_sharpe - real_sharpe) / abs(real_sharpe) if real_sharpe != 0 else 0
outlier_pass = abs(outlier_deg) < 0.50
print(f"  Outlier robustness: trimmed Sharpe {trim_sharpe:.3f}, degradation {outlier_deg:.1%} → {'PASS' if outlier_pass else 'FAIL'}")

# R1 regime test
spy_ret = prices['SPY'].pct_change().reindex(common_dates)
green_days = spy_ret > 0
red_days = spy_ret < 0

green_sharpe = best_returns[green_days].mean() * 252 / (best_returns[green_days].std() * np.sqrt(252)) if best_returns[green_days].std() > 0 else 0
red_sharpe = best_returns[red_days].mean() * 252 / (best_returns[red_days].std() * np.sqrt(252)) if best_returns[red_days].std() > 0 else 0
r1_gap = abs(green_sharpe - red_sharpe) / max(abs(green_sharpe), abs(red_sharpe), 0.01)
r1_pass = r1_gap < 0.50
print(f"  R1 regime: green={green_sharpe:.3f}, red={red_sharpe:.3f}, gap={r1_gap:.3f} → {'PASS' if r1_pass else 'FAIL'}")

gates_passed = sum([perm_pass, sub_pass, outlier_pass, r1_pass])
print(f"\n  ADVERSARIAL SUMMARY: {gates_passed}/4 gates → {'PASS' if gates_passed >= 3 else 'FAIL'}")

# ─── DIVERSIFICATION BENEFIT ──────────────────────────────────────────────────

print("\n[7/8] Diversification analysis...")

# Compare 50/50 vs individual strategies
combo_50 = 0.5 * cta_aligned + 0.5 * sec_aligned
combo_metrics = compute_metrics(combo_50, "50/50 Blend")
cta_metrics = compute_metrics(cta_aligned, "CTA Only")
sec_metrics = compute_metrics(sec_aligned, "Sector Only")

if combo_metrics and cta_metrics and sec_metrics:
    avg_sharpe = (cta_metrics['sharpe'] + sec_metrics['sharpe']) / 2
    div_benefit = combo_metrics['sharpe'] / avg_sharpe - 1 if avg_sharpe > 0 else 0
    dd_improvement = combo_metrics['max_dd'] / min(cta_metrics['max_dd'], sec_metrics['max_dd']) - 1
    print(f"  Diversification Sharpe uplift: {div_benefit:+.1%}")
    print(f"  MaxDD improvement vs worst individual: {dd_improvement:+.1%}")
    print(f"  Correlation: {corr:.4f}")

# ─── SAVE RESULTS ────────────────────────────────────────────────────────────

print("\n[8/8] Saving results...")

final_results = {
    'analysis': 'Portfolio Combination: ML Trend v2 + ML Sector Rotation',
    'date': pd.Timestamp.now().isoformat(),
    'correlation': round(corr, 4),
    'n_common_days': len(common_dates),
    'date_range': f"{common_dates[0].strftime('%Y-%m-%d')} to {common_dates[-1].strftime('%Y-%m-%d')}",
    'blends': results,
    'best_blend': best_name,
    'adversarial': {
        'permutation': {'p': round(perm_p, 3), 'pass': perm_pass},
        'sub_period': {'cv': round(sub_cv, 3), 'pass': sub_pass},
        'outlier': {'degradation': round(outlier_deg, 3), 'pass': outlier_pass},
        'r1_regime': {'gap': round(r1_gap, 3), 'green': round(green_sharpe, 3), 'red': round(red_sharpe, 3), 'pass': r1_pass},
        'gates_passed': gates_passed,
        'verdict': 'PASS' if gates_passed >= 3 else 'FAIL'
    },
    'diversification': {
        'sharpe_uplift_pct': round(div_benefit * 100, 1) if combo_metrics else None,
        'dd_improvement_pct': round(dd_improvement * 100, 1) if combo_metrics else None,
    }
}

with open(OUTPUT / 'results.json', 'w') as f:
    json.dump(final_results, f, indent=2, default=str)

# Equity curve plot
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Top: Equity curves
ax = axes[0]
for name, rets in blend_returns.items():
    cum = (1 + rets).cumprod() * INITIAL_CAPITAL
    ax.plot(cum.index, cum.values, label=f"{name} (S={results[name]['sharpe']:.2f})", alpha=0.8)

spy_cum = (1 + prices['SPY'].pct_change().reindex(common_dates).fillna(0)).cumprod() * INITIAL_CAPITAL
ax.plot(spy_cum.index, spy_cum.values, 'k--', label='SPY B&H', alpha=0.5)
ax.set_ylabel('Portfolio Value ($)')
ax.set_title('ML Trend v2 + ML Sector Rotation: Portfolio Combinations')
ax.legend(fontsize=8, loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

# Bottom: Rolling correlation
ax = axes[1]
rolling_corr = cta_aligned.rolling(63).corr(sec_aligned)
ax.plot(rolling_corr.index, rolling_corr.values, 'b-', alpha=0.7)
ax.axhline(corr, color='r', linestyle='--', label=f'Full-period corr: {corr:.3f}')
ax.set_ylabel('63-day Rolling Correlation')
ax.set_xlabel('Date')
ax.set_title('Strategy Return Correlation Over Time')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(-1, 1)

plt.tight_layout()
plt.savefig(OUTPUT / 'equity_curves.png', dpi=100)
plt.close()

print(f"\n{'='*70}")
print(f"RESULTS SUMMARY")
print(f"{'='*70}")
print(f"  Strategy correlation: {corr:.4f}")
print(f"  Best blend: {best_name}")
print(f"  Best Sharpe: {results[best_name]['sharpe']}")
print(f"  Adversarial: {gates_passed}/4 gates {'PASS' if gates_passed >= 3 else 'FAIL'}")
if combo_metrics:
    print(f"  Diversification benefit: {div_benefit:+.1%} Sharpe uplift")
print(f"  Output: {OUTPUT}")
print(f"{'='*70}")

