#!/usr/bin/env python3
"""
Portfolio Combination: ML Trend v2 (8 CTA) + ML Sector Rotation (11 sectors)
=============================================================================
Two validated regime-agnostic strategies. Tests diversification benefit.
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

# ─── DATA ────────────────────────────────────────────────────────────────────

CTA_ASSETS = ['SPY', 'TLT', 'GLD', 'UUP', 'USO', 'EEM', 'VNQ', 'HYG']
SECTOR_ASSETS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU', 'XLRE', 'XLC']
ALL_TICKERS = list(set(CTA_ASSETS + SECTOR_ASSETS))

print("\n[1/8] Downloading data...")
t0 = time.time()

raw = yf.download(ALL_TICKERS, start='2000-01-01', progress=False)
if hasattr(raw.columns, 'levels'):
    prices = raw['Close']
else:
    prices = raw

# VIX
vix_raw = yf.download('^VIX', start='2000-01-01', progress=False)
if hasattr(vix_raw.columns, 'levels'):
    vix = vix_raw['Close'].squeeze()
else:
    vix = vix_raw['Close'] if 'Close' in vix_raw.columns else vix_raw.iloc[:, 0]
vix.name = 'VIX'

print(f"  Prices shape: {prices.shape}, VIX: {len(vix)} days")
print(f"  Available CTA: {[a for a in CTA_ASSETS if a in prices.columns]}")
print(f"  Available Sector: {[a for a in SECTOR_ASSETS if a in prices.columns]}")
print(f"  Time: {time.time()-t0:.1f}s")

# ─── HELPERS ─────────────────────────────────────────────────────────────────

def compute_rsi(p, window=14):
    delta = p.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def build_features(px_df, vix_s, asset):
    """Build ML features for one asset"""
    p = px_df[asset].dropna()
    if len(p) < MA_LONG + 50:
        return pd.DataFrame()

    ret = p.pct_change()
    ma_s = p.rolling(MA_SHORT).mean()
    ma_l = p.rolling(MA_LONG).mean()

    feat = pd.DataFrame(index=p.index)
    feat['trend_str'] = (ma_s - ma_l) / ma_l
    feat['mom_20'] = p.pct_change(20)
    feat['mom_60'] = p.pct_change(60)
    feat['vol_20'] = ret.rolling(20).std() * np.sqrt(252)
    feat['vol_ratio'] = ret.rolling(20).std() / ret.rolling(60).std()
    feat['dd'] = p / p.rolling(252).max() - 1
    feat['rsi'] = compute_rsi(p, 14)

    # Cross-asset trend alignment
    n_up = pd.Series(0.0, index=p.index)
    count = 0
    for other in px_df.columns:
        if other == asset:
            continue
        op = px_df[other].reindex(p.index).ffill()
        if op.notna().sum() > MA_LONG:
            oma_s = op.rolling(MA_SHORT).mean()
            oma_l = op.rolling(MA_LONG).mean()
            n_up += (oma_s > oma_l).astype(float)
            count += 1
    feat['cross_trend'] = n_up / max(count, 1)

    # VIX
    vix_a = vix_s.reindex(p.index).ffill()
    if vix_a.notna().sum() > 60:
        feat['vix'] = vix_a
        feat['vix_z'] = (vix_a - vix_a.rolling(60).mean()) / vix_a.rolling(60).std().replace(0, 1)

    return feat.dropna()

def run_strategy(px_df, vix_s, universe, label):
    """Run ML-filtered trend on a universe. Returns daily portfolio returns."""
    print(f"  Running {label}...")
    available = [a for a in universe if a in px_df.columns and px_df[a].notna().sum() > MA_LONG + TRAIN_WINDOW]
    print(f"    Available assets: {len(available)}/{len(universe)}")

    if len(available) < 3:
        return pd.Series(dtype=float)

    px = px_df[available].copy()

    # Find start date (need enough history)
    valid_start = px.dropna(thresh=3).index[0]
    all_dates = px.index[px.index >= valid_start]

    # Skip initial warmup
    warmup = TRAIN_WINDOW + MA_LONG + 20
    if len(all_dates) <= warmup:
        return pd.Series(dtype=float)

    trade_dates = all_dates[warmup:]
    daily_rets = pd.Series(0.0, index=trade_dates)

    model = None
    last_train = -999

    for i, date in enumerate(trade_dates):
        date_loc = px.index.get_loc(date)

        # Retrain every TRAIN_WINDOW days
        if i - last_train >= TRAIN_WINDOW or model is None:
            train_slice = px.iloc[max(0, date_loc - TRAIN_WINDOW - MA_LONG):date_loc]

            X_all, y_all = [], []
            for asset in available:
                feat = build_features(train_slice, vix_s, asset)
                if len(feat) < 50:
                    continue

                # Label: trend continuation profitable in next 20d
                fwd_ret = train_slice[asset].pct_change(20).shift(-20)
                ma_s = train_slice[asset].rolling(MA_SHORT).mean()
                ma_l = train_slice[asset].rolling(MA_LONG).mean()
                direction = ((ma_s > ma_l).astype(int) * 2 - 1)
                label_s = (fwd_ret * direction > 0).astype(int)

                common = feat.index.intersection(label_s.dropna().index)
                if len(common) < 20:
                    continue
                X_all.append(feat.loc[common].values)
                y_all.append(label_s.loc[common].values)

            if X_all:
                X_train = np.vstack(X_all)
                y_train = np.concatenate(y_all)
                model = GradientBoostingClassifier(
                    n_estimators=100, max_depth=3, learning_rate=0.1,
                    subsample=0.8, random_state=42
                )
                model.fit(X_train, y_train)
                last_train = i

        if model is None:
            continue

        # Generate positions
        port_ret = 0.0
        total_weight = 0.0

        for asset in available:
            if date_loc < MA_LONG + 20:
                continue

            recent = px[asset].iloc[max(0, date_loc - MA_LONG - 5):date_loc + 1]
            if recent.isna().sum() > 10 or len(recent) < MA_LONG:
                continue

            ma_s_val = recent.iloc[-MA_SHORT:].mean()
            ma_l_val = recent.iloc[-MA_LONG:].mean()
            if ma_l_val == 0:
                continue

            trend_dir = 1 if ma_s_val > ma_l_val else -1

            # ML filter
            try:
                window_data = px.iloc[max(0, date_loc - MA_LONG - 20):date_loc + 1]
                feat = build_features(window_data, vix_s, asset)
                if len(feat) == 0:
                    continue
                prob = model.predict_proba(feat.iloc[-1:].values)[0][1]
            except:
                continue

            if prob < ML_THRESHOLD:
                continue

            # Vol target sizing
            asset_vol = px[asset].pct_change().iloc[max(0, date_loc - 20):date_loc + 1].std() * np.sqrt(252)
            if asset_vol < 0.01:
                continue

            weight = min(TARGET_VOL / asset_vol, 0.25)

            # Get next day return
            if date_loc + 1 < len(px):
                nxt = px[asset].iloc[date_loc + 1]
                cur = px[asset].iloc[date_loc]
                if pd.notna(nxt) and pd.notna(cur) and cur > 0:
                    asset_ret = nxt / cur - 1
                    port_ret += trend_dir * weight * asset_ret
                    total_weight += abs(weight)

        # Transaction cost approximation
        daily_rets.iloc[i] = port_ret - total_weight * REBAL_COST_BPS / 10000 / 252

    print(f"    Done: {len(daily_rets)} days, mean={daily_rets.mean()*252:.4f}")
    return daily_rets

# ─── RUN STRATEGIES ──────────────────────────────────────────────────────────

print("\n[2/8] Running ML Trend v2 (CTA)...")
cta_returns = run_strategy(prices, vix, CTA_ASSETS, "CTA Trend")

print("\n[3/8] Running ML Sector Rotation...")
sector_returns = run_strategy(prices, vix, SECTOR_ASSETS, "Sector Rotation")

# ─── COMBINE ─────────────────────────────────────────────────────────────────

print("\n[4/8] Combining strategies...")

common_dates = cta_returns.index.intersection(sector_returns.index)
cta_a = cta_returns.loc[common_dates]
sec_a = sector_returns.loc[common_dates]

corr = cta_a.corr(sec_a)
print(f"  Correlation: {corr:.4f}")
print(f"  Common days: {len(common_dates)}")

if len(common_dates) < 252:
    print("ERROR: Not enough common days for analysis")
    exit(1)

# ─── METRICS ─────────────────────────────────────────────────────────────────

def calc_metrics(rets, name):
    if len(rets) == 0 or rets.std() == 0:
        return None
    ann_r = rets.mean() * 252
    ann_v = rets.std() * np.sqrt(252)
    sharpe = ann_r / ann_v
    down_v = rets[rets < 0].std() * np.sqrt(252)
    sortino = ann_r / down_v if down_v > 0 else 0
    cum = (1 + rets).cumprod()
    dd = (cum / cum.cummax() - 1).min()
    cagr = cum.iloc[-1] ** (252 / len(rets)) - 1
    calmar = cagr / abs(dd) if dd != 0 else 0
    pf = rets[rets > 0].sum() / abs(rets[rets < 0].sum()) if rets[rets < 0].sum() != 0 else 0
    return {
        'label': name, 'sharpe': round(sharpe, 3), 'sortino': round(sortino, 3),
        'cagr': round(cagr * 100, 1), 'max_dd': round(dd * 100, 1),
        'calmar': round(calmar, 3), 'vol': round(ann_v * 100, 1),
        'pf': round(pf, 3), 'wr': round((rets > 0).mean() * 100, 1),
        'n_days': len(rets), 'years': round(len(rets) / 252, 1)
    }

print("\n[5/8] Portfolio blends...")

# Risk parity weights
cta_vol = cta_a.std() * np.sqrt(252)
sec_vol = sec_a.std() * np.sqrt(252)
rp_cta = (1/cta_vol) / (1/cta_vol + 1/sec_vol) if cta_vol > 0 and sec_vol > 0 else 0.5
rp_sec = 1 - rp_cta

blends = {
    'CTA Only': (1.0, 0.0),
    'Sector Only': (0.0, 1.0),
    '50/50': (0.5, 0.5),
    '70/30 CTA/Sec': (0.7, 0.3),
    '30/70 CTA/Sec': (0.3, 0.7),
    f'Risk Parity ({rp_cta:.0%}/{rp_sec:.0%})': (rp_cta, rp_sec),
}

results = {}
blend_rets = {}

print(f"\n  {'Blend':<25} {'Sharpe':>7} {'Sortino':>8} {'CAGR':>7} {'MaxDD':>7} {'Calmar':>7}")
print(f"  {'-'*25} {'-'*7} {'-'*8} {'-'*7} {'-'*7} {'-'*7}")

for name, (wc, ws) in blends.items():
    combo = wc * cta_a + ws * sec_a
    m = calc_metrics(combo, name)
    if m:
        results[name] = m
        blend_rets[name] = combo
        print(f"  {name:<25} {m['sharpe']:7.2f} {m['sortino']:8.2f} {m['cagr']:6.1f}% {m['max_dd']:6.1f}% {m['calmar']:7.2f}")

# SPY benchmark
spy_ret = prices['SPY'].pct_change().reindex(common_dates).fillna(0)
spy_m = calc_metrics(spy_ret, 'SPY B&H')
if spy_m:
    print(f"  {'SPY B&H':<25} {spy_m['sharpe']:7.2f} {spy_m['sortino']:8.2f} {spy_m['cagr']:6.1f}% {spy_m['max_dd']:6.1f}% {spy_m['calmar']:7.2f}")

# ─── ADVERSARIAL ─────────────────────────────────────────────────────────────

print("\n[6/8] Adversarial validation...")

best_name = max(results.keys(), key=lambda k: results[k]['sharpe'])
best_ret = blend_rets[best_name]
real_sharpe = results[best_name]['sharpe']
print(f"  Testing: {best_name} (Sharpe {real_sharpe})")

# Perm test: shuffle temporal alignment
perm_sharpes = []
for _ in range(N_PERMUTATIONS):
    shuf = sec_a.sample(frac=1, replace=False).values
    wc, ws = blends[best_name] if best_name in blends else (0.5, 0.5)
    combo_perm = wc * cta_a.values + ws * shuf
    ps = np.mean(combo_perm) * 252 / (np.std(combo_perm) * np.sqrt(252))
    perm_sharpes.append(ps)

perm_p = np.mean([s >= real_sharpe for s in perm_sharpes])
perm_pass = perm_p < 0.05
print(f"  Perm: real={real_sharpe:.3f}, mean_perm={np.mean(perm_sharpes):.3f}, p={perm_p:.3f} → {'PASS' if perm_pass else 'FAIL'}")

# Sub-period
n_blocks = 4
bs = len(best_ret) // n_blocks
block_sharpes = []
for b in range(n_blocks):
    blk = best_ret.iloc[b*bs:(b+1)*bs]
    s = blk.mean() * 252 / (blk.std() * np.sqrt(252)) if blk.std() > 0 else 0
    block_sharpes.append(s)
sub_cv = np.std(block_sharpes) / max(abs(np.mean(block_sharpes)), 0.01)
sub_pass = sub_cv < 0.50
print(f"  Sub-period: blocks={[f'{s:.2f}' for s in block_sharpes]}, CV={sub_cv:.3f} → {'PASS' if sub_pass else 'FAIL'}")

# Outlier
trimmed = best_ret[(best_ret > best_ret.quantile(0.01)) & (best_ret < best_ret.quantile(0.99))]
trim_s = trimmed.mean() * 252 / (trimmed.std() * np.sqrt(252)) if trimmed.std() > 0 else 0
outlier_deg = (trim_s - real_sharpe) / abs(real_sharpe) if real_sharpe != 0 else 0
outlier_pass = abs(outlier_deg) < 0.50
print(f"  Outlier: trimmed Sharpe {trim_s:.3f}, deg={outlier_deg:.1%} → {'PASS' if outlier_pass else 'FAIL'}")

# R1 regime
green = spy_ret > 0
red = spy_ret < 0
g_s = best_ret[green].mean() * 252 / (best_ret[green].std() * np.sqrt(252)) if best_ret[green].std() > 0 else 0
r_s = best_ret[red].mean() * 252 / (best_ret[red].std() * np.sqrt(252)) if best_ret[red].std() > 0 else 0
r1_gap = abs(g_s - r_s) / max(abs(g_s), abs(r_s), 0.01)
r1_pass = r1_gap < 0.50
print(f"  R1: green={g_s:.3f}, red={r_s:.3f}, gap={r1_gap:.3f} → {'PASS' if r1_pass else 'FAIL'}")

gates = sum([perm_pass, sub_pass, outlier_pass, r1_pass])
print(f"\n  ADVERSARIAL: {gates}/4 → {'PASS' if gates >= 3 else 'FAIL'}")

# ─── DIVERSIFICATION BENEFIT ─────────────────────────────────────────────────

print("\n[7/8] Diversification benefit...")
combo_50 = calc_metrics(0.5*cta_a + 0.5*sec_a, "50/50")
cta_m = calc_metrics(cta_a, "CTA")
sec_m = calc_metrics(sec_a, "Sector")

if combo_50 and cta_m and sec_m:
    avg_s = (cta_m['sharpe'] + sec_m['sharpe']) / 2
    div_uplift = (combo_50['sharpe'] / avg_s - 1) if avg_s > 0 else 0
    print(f"  CTA Sharpe: {cta_m['sharpe']}, Sector Sharpe: {sec_m['sharpe']}")
    print(f"  50/50 Sharpe: {combo_50['sharpe']} (expected avg: {avg_s:.3f})")
    print(f"  Diversification uplift: {div_uplift:+.1%}")
    print(f"  MaxDD: CTA {cta_m['max_dd']}%, Sector {sec_m['max_dd']}%, 50/50 {combo_50['max_dd']}%")

# ─── SAVE ────────────────────────────────────────────────────────────────────

print("\n[8/8] Saving...")

final = {
    'analysis': 'Portfolio Combination: ML Trend v2 + ML Sector Rotation',
    'timestamp': pd.Timestamp.now().isoformat(),
    'correlation': round(corr, 4),
    'n_days': len(common_dates),
    'date_range': f"{common_dates[0].strftime('%Y-%m-%d')} to {common_dates[-1].strftime('%Y-%m-%d')}",
    'blends': results,
    'best_blend': best_name,
    'adversarial': {
        'permutation': {'p': round(perm_p, 3), 'pass': bool(perm_pass)},
        'sub_period': {'cv': round(sub_cv, 3), 'pass': bool(sub_pass)},
        'outlier': {'deg': round(outlier_deg, 3), 'pass': bool(outlier_pass)},
        'r1_regime': {'gap': round(r1_gap, 3), 'g': round(g_s, 3), 'r': round(r_s, 3), 'pass': bool(r1_pass)},
        'gates_passed': gates,
        'verdict': 'PASS' if gates >= 3 else 'FAIL'
    },
    'diversification': {
        'uplift_pct': round(div_uplift * 100, 1) if combo_50 else None,
    }
}

with open(OUTPUT / 'results.json', 'w') as f:
    json.dump(final, f, indent=2, default=str)

# Plot
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
ax = axes[0]
for name, rets in blend_rets.items():
    cum = (1 + rets).cumprod() * INITIAL_CAPITAL
    ax.plot(cum.index, cum.values, label=f"{name} (S={results[name]['sharpe']:.2f})", alpha=0.8)
spy_cum = (1 + spy_ret).cumprod() * INITIAL_CAPITAL
ax.plot(spy_cum.index, spy_cum.values, 'k--', label='SPY B&H', alpha=0.5)
ax.set_ylabel('Portfolio Value ($)')
ax.set_title('ML Trend v2 + ML Sector Rotation: Portfolio Combinations')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

ax = axes[1]
rc = cta_a.rolling(63).corr(sec_a)
ax.plot(rc.index, rc.values, 'b-', alpha=0.7)
ax.axhline(corr, color='r', linestyle='--', label=f'Full corr: {corr:.3f}')
ax.set_ylabel('63d Rolling Correlation')
ax.set_title('CTA vs Sector Strategy Correlation')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(-1, 1)

plt.tight_layout()
plt.savefig(OUTPUT / 'equity_curves.png', dpi=100)
plt.close()

print(f"\n{'='*70}")
print(f"FINAL RESULT: {best_name} | Sharpe {results[best_name]['sharpe']} | {gates}/4 adversarial")
print(f"Correlation: {corr:.4f} | Diversification: {div_uplift:+.1%} Sharpe uplift")
print(f"{'='*70}")
