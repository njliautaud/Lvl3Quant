#!/usr/bin/env python3
"""
ML Mean Reversion Strategy
===========================
INSIGHT: Mean reversion is the OPPOSITE of trend following. If we can find
a validated mean-reversion strategy, combining it with ML Trend v2 (corr~0)
could significantly reduce portfolio drawdowns.

Strategy:
  1. Identify oversold/overbought conditions (RSI, Bollinger bands, distance from MA)
  2. Use ML to predict which "extreme" moves will revert vs which continue
  3. Trade the reversions with tight stops (mean reversion fails badly in trends)
  4. Key: only trade when ML says "revert likely" — avoid catching falling knives

Universe: Liquid ETFs with mean-reverting properties (sector ETFs, indices)
Timeframe: Daily (hold 2-10 days)

HC #713: Fixed $100K, NO DCA. HC #0: SLIDING 252d. HC #714: ML exploration.
"""

import json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
np.random.seed(42)

BASE = Path("/home/jupiter/Lvl3Quant")
OUTPUT = BASE / "output" / "ml_mean_reversion"
OUTPUT.mkdir(parents=True, exist_ok=True)

INITIAL_CAPITAL = 100_000
TRAIN_WINDOW = 252
N_PERMS = 100
REBAL_COST_BPS = 5  # Tighter spreads on liquid ETFs

# Universe: liquid sector + asset class ETFs
UNIVERSE = ['SPY', 'QQQ', 'IWM', 'XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP',
            'XLI', 'TLT', 'GLD', 'EEM', 'VNQ', 'HYG']

# Mean reversion parameters
RSI_WINDOW = 5       # Short RSI for oversold/overbought
BB_WINDOW = 20       # Bollinger band window
BB_STD = 2.0         # Bollinger band std
REVERT_HORIZON = 5   # Hold for 5 days (mean reversion is short-lived)
ML_THRESHOLD = 0.55  # ML confidence to trade
MAX_POSITIONS = 5    # Max concurrent positions

print("=" * 70)
print("ML MEAN REVERSION STRATEGY")
print("=" * 70)

# ─── DATA ────────────────────────────────────────────────────────────────────

print("\n[1/7] Downloading data...")
t0 = time.time()

raw = yf.download(UNIVERSE + ['^VIX'], start='2006-01-01', progress=False)
if hasattr(raw.columns, 'levels'):
    prices = raw['Close']
    volume = raw['Volume']
else:
    prices = raw
    volume = pd.DataFrame()

vix = prices['^VIX'] if '^VIX' in prices.columns else pd.Series(dtype=float)
prices = prices.drop('^VIX', axis=1, errors='ignore')
if '^VIX' in volume.columns:
    volume = volume.drop('^VIX', axis=1, errors='ignore')

available = [a for a in UNIVERSE if a in prices.columns and prices[a].notna().sum() > 500]
prices = prices[available]
print(f"  Assets: {len(available)}/{len(UNIVERSE)}")
print(f"  Days: {len(prices)}")
print(f"  Time: {time.time()-t0:.1f}s")

# ─── FEATURE ENGINEERING ─────────────────────────────────────────────────────

print("\n[2/7] Building features...")

def build_mr_features(px, vol_df, vix_s, asset):
    """Mean reversion features for one asset"""
    p = px[asset].dropna()
    if len(p) < 260:
        return pd.DataFrame(), pd.Series(dtype=float)

    ret = p.pct_change()

    f = pd.DataFrame(index=p.index)

    # RSI (short-term for MR)
    delta = p.diff()
    gain = delta.clip(lower=0).rolling(RSI_WINDOW).mean()
    loss = (-delta.clip(upper=0)).rolling(RSI_WINDOW).mean()
    rs = gain / loss.replace(0, np.nan)
    f['rsi_5'] = 100 - (100 / (1 + rs))

    # Bollinger band position
    bb_mid = p.rolling(BB_WINDOW).mean()
    bb_std = p.rolling(BB_WINDOW).std()
    f['bb_pos'] = (p - bb_mid) / (bb_std * BB_STD)  # -1 = lower band, +1 = upper band

    # Distance from moving averages
    f['dist_20ma'] = (p - p.rolling(20).mean()) / p.rolling(20).mean()
    f['dist_50ma'] = (p - p.rolling(50).mean()) / p.rolling(50).mean()

    # Recent performance (looking for extremes)
    f['ret_1d'] = ret
    f['ret_3d'] = p.pct_change(3)
    f['ret_5d'] = p.pct_change(5)
    f['ret_10d'] = p.pct_change(10)

    # Volatility context
    f['vol_10d'] = ret.rolling(10).std() * np.sqrt(252)
    f['vol_ratio'] = ret.rolling(10).std() / ret.rolling(60).std()

    # Volume surge (if available)
    if asset in vol_df.columns and vol_df[asset].notna().sum() > 100:
        v = vol_df[asset].reindex(p.index)
        f['vol_surge'] = v / v.rolling(20).mean()

    # VIX context
    va = vix_s.reindex(p.index).ffill()
    if va.notna().sum() > 60:
        f['vix'] = va
        f['vix_pctile'] = va.rolling(252).rank(pct=True)

    # Consecutive direction
    f['consec_down'] = (ret < 0).astype(int).rolling(5).sum()
    f['consec_up'] = (ret > 0).astype(int).rolling(5).sum()

    # Label: does the asset revert in next REVERT_HORIZON days?
    # Revert = if currently below MA, goes back up; if above MA, goes back down
    fwd_ret = p.pct_change(REVERT_HORIZON).shift(-REVERT_HORIZON)
    current_extreme = f['bb_pos']

    # Oversold (bb_pos < -0.5) that reverts UP = positive label
    # Overbought (bb_pos > 0.5) that reverts DOWN = positive label
    label = pd.Series(np.nan, index=p.index)
    oversold = current_extreme < -0.5
    overbought = current_extreme > 0.5
    label[oversold] = (fwd_ret[oversold] > 0.005).astype(float)  # Reverts up
    label[overbought] = (fwd_ret[overbought] < -0.005).astype(float)  # Reverts down

    f = f.dropna()
    return f, label

# Build all features
print("  Building features for each asset...")
all_features = {}
all_labels = {}

for asset in available:
    feat, label = build_mr_features(prices, volume, vix, asset)
    if len(feat) > 200:
        all_features[asset] = feat
        all_labels[asset] = label
        labeled_count = label.dropna().sum()
        print(f"    {asset}: {len(feat)} days, {int(labeled_count)} labeled extreme events")

print(f"  Total assets with features: {len(all_features)}")

# ─── WALK-FORWARD SIMULATION ─────────────────────────────────────────────────

print("\n[3/7] Walk-forward simulation...")

# Common date range
all_dates = prices.index
start_idx = TRAIN_WINDOW + 260  # Need extra warmup for features
trade_dates = all_dates[start_idx:]

daily_returns = pd.Series(0.0, index=trade_dates)
n_trades = 0
positions_held = {}  # {asset: {'direction': 1/-1, 'entry_date': date, 'entry_price': p}}

model = None
last_train = -999

for i, date in enumerate(trade_dates):
    loc = all_dates.get_loc(date)

    # Retrain model quarterly
    if i - last_train >= 63 or model is None:
        X_all, y_all = [], []

        for asset in all_features:
            feat = all_features[asset]
            label = all_labels[asset]

            # Get training window
            train_mask = (feat.index < date) & (feat.index >= all_dates[max(0, loc - TRAIN_WINDOW)])
            feat_train = feat[train_mask]
            label_train = label.reindex(feat_train.index).dropna()

            common = feat_train.index.intersection(label_train.index)
            if len(common) < 20:
                continue

            X_all.append(feat_train.loc[common].values)
            y_all.append(label_train.loc[common].values)

        if X_all:
            X = np.vstack(X_all)
            y = np.concatenate(y_all)

            if len(y) > 100 and y.sum() > 10:
                model = GradientBoostingClassifier(
                    n_estimators=100, max_depth=3, learning_rate=0.1,
                    subsample=0.8, min_samples_leaf=20, random_state=42
                )
                model.fit(X, y)
                last_train = i

    if model is None:
        continue

    # Exit positions that have been held for REVERT_HORIZON days
    to_remove = []
    exit_pnl = 0.0
    for asset, pos in positions_held.items():
        days_held = (date - pos['entry_date']).days
        if days_held >= REVERT_HORIZON:
            # Exit
            if date in prices.index and asset in prices.columns:
                exit_price = prices[asset].loc[date]
                entry_price = pos['entry_price']
                if pd.notna(exit_price) and entry_price > 0:
                    pnl = pos['direction'] * (exit_price / entry_price - 1)
                    exit_pnl += pnl / MAX_POSITIONS  # Equal weight
                    n_trades += 1
            to_remove.append(asset)

    for asset in to_remove:
        del positions_held[asset]

    # Find new entry signals
    if len(positions_held) < MAX_POSITIONS:
        candidates = []

        for asset in all_features:
            if asset in positions_held:
                continue

            feat = all_features[asset]
            if date not in feat.index:
                continue

            # Check if extreme (only trade extremes)
            bb_pos = feat.loc[date, 'bb_pos'] if 'bb_pos' in feat.columns else 0
            if abs(bb_pos) < 0.5:  # Not extreme enough
                continue

            # ML prediction
            today_feat = feat.loc[[date]].values
            if today_feat.shape[1] != model.n_features_in_:
                continue

            try:
                prob = model.predict_proba(today_feat)[0][1]
            except:
                continue

            if prob >= ML_THRESHOLD:
                direction = -1 if bb_pos > 0.5 else 1  # Fade the move
                candidates.append((asset, prob, direction))

        # Take top candidates by probability
        candidates.sort(key=lambda x: -x[1])
        slots = MAX_POSITIONS - len(positions_held)

        for asset, prob, direction in candidates[:slots]:
            entry_price = prices[asset].loc[date] if date in prices.index else None
            if entry_price is not None and pd.notna(entry_price) and entry_price > 0:
                positions_held[asset] = {
                    'direction': direction,
                    'entry_date': date,
                    'entry_price': entry_price,
                    'prob': prob
                }

    # Daily return from exits
    daily_returns.iloc[i] = exit_pnl - abs(exit_pnl) * REBAL_COST_BPS / 10000

# ─── METRICS ─────────────────────────────────────────────────────────────────

print("\n[4/7] Computing metrics...")

rets = daily_returns
nonzero = rets[rets != 0]
ann_ret = rets.mean() * 252
ann_vol = rets.std() * np.sqrt(252)
sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
downside = rets[rets < 0].std() * np.sqrt(252)
sortino = ann_ret / downside if downside > 0 else 0
cum = (1 + rets).cumprod()
max_dd = (cum / cum.cummax() - 1).min()
cagr = cum.iloc[-1] ** (252 / len(rets)) - 1 if cum.iloc[-1] > 0 else 0
calmar = cagr / abs(max_dd) if max_dd != 0 else 0
wr = (nonzero > 0).mean() if len(nonzero) > 0 else 0
pf = nonzero[nonzero > 0].sum() / abs(nonzero[nonzero < 0].sum()) if nonzero[nonzero < 0].sum() != 0 else 0

print(f"  Sharpe: {sharpe:.3f}")
print(f"  Sortino: {sortino:.3f}")
print(f"  CAGR: {cagr*100:.1f}%")
print(f"  MaxDD: {max_dd*100:.1f}%")
print(f"  Calmar: {calmar:.3f}")
print(f"  Win Rate: {wr*100:.1f}%")
print(f"  Profit Factor: {pf:.3f}")
print(f"  Total trades: {n_trades}")
print(f"  Avg trades/year: {n_trades/(len(rets)/252):.0f}")

# Correlation with trend following (proxy: SPY momentum)
spy_mom = prices['SPY'].pct_change(20)
mr_monthly = rets.resample('ME').sum()
spy_mom_monthly = spy_mom.resample('ME').last()
common_m = mr_monthly.index.intersection(spy_mom_monthly.dropna().index)
if len(common_m) > 12:
    corr_trend = mr_monthly.loc[common_m].corr(spy_mom_monthly.loc[common_m])
    print(f"  Correlation with SPY momentum: {corr_trend:.3f}")

# ─── ADVERSARIAL ─────────────────────────────────────────────────────────────

print("\n[5/7] Adversarial validation...")

# Permutation: randomly assign long/short to extreme events
perm_sharpes = []
for p in range(N_PERMS):
    perm_rets = daily_returns.copy()
    # Flip all trade directions randomly
    mask = daily_returns != 0
    signs = np.random.choice([-1, 1], size=mask.sum())
    perm_rets[mask] = daily_returns[mask].abs().values * signs
    ps = perm_rets.mean() * 252 / (perm_rets.std() * np.sqrt(252)) if perm_rets.std() > 0 else 0
    perm_sharpes.append(ps)

perm_p = np.mean([s >= sharpe for s in perm_sharpes])
perm_pass = perm_p < 0.05
print(f"  Permutation: real={sharpe:.3f}, perm_mean={np.mean(perm_sharpes):.3f}, p={perm_p:.3f} → {'PASS' if perm_pass else 'FAIL'}")

# Sub-period
n_blocks = 4
bs = len(rets) // n_blocks
block_sharpes = []
for b in range(n_blocks):
    blk = rets.iloc[b*bs:(b+1)*bs]
    s = blk.mean() * 252 / (blk.std() * np.sqrt(252)) if blk.std() > 0 else 0
    block_sharpes.append(s)
sub_cv = np.std(block_sharpes) / max(abs(np.mean(block_sharpes)), 0.01)
sub_pass = sub_cv < 0.50
print(f"  Sub-period: {[f'{s:.2f}' for s in block_sharpes]}, CV={sub_cv:.3f} → {'PASS' if sub_pass else 'FAIL'}")

# Outlier robustness
trimmed = rets[(rets > rets.quantile(0.01)) & (rets < rets.quantile(0.99))]
trim_s = trimmed.mean() * 252 / (trimmed.std() * np.sqrt(252)) if trimmed.std() > 0 else 0
outlier_deg = (trim_s - sharpe) / abs(sharpe) if sharpe != 0 else 0
outlier_pass = abs(outlier_deg) < 0.50
print(f"  Outlier: trimmed={trim_s:.3f}, deg={outlier_deg:.1%} → {'PASS' if outlier_pass else 'FAIL'}")

# R1 regime
spy_daily = prices['SPY'].pct_change().reindex(rets.index)
green = spy_daily > 0
red = spy_daily < 0
if rets[green].std() > 0 and rets[red].std() > 0:
    g_s = rets[green].mean() * 252 / (rets[green].std() * np.sqrt(252))
    r_s = rets[red].mean() * 252 / (rets[red].std() * np.sqrt(252))
    r1_gap = abs(g_s - r_s) / max(abs(g_s), abs(r_s), 0.01)
    r1_pass = r1_gap < 0.50
    print(f"  R1: green={g_s:.3f}, red={r_s:.3f}, gap={r1_gap:.3f} → {'PASS' if r1_pass else 'FAIL'}")
else:
    r1_pass = False
    r1_gap = 999

gates = sum([perm_pass, sub_pass, outlier_pass, r1_pass])
print(f"\n  ADVERSARIAL: {gates}/4 → {'PASS' if gates >= 3 else 'FAIL'}")

# ─── SAVE ────────────────────────────────────────────────────────────────────

print("\n[6/7] Saving results...")

results = {
    'strategy': 'ML Mean Reversion',
    'timestamp': pd.Timestamp.now().isoformat(),
    'universe': available,
    'metrics': {
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'cagr': round(cagr * 100, 1),
        'max_dd': round(max_dd * 100, 1),
        'calmar': round(calmar, 3),
        'win_rate': round(wr * 100, 1),
        'profit_factor': round(pf, 3),
        'total_trades': n_trades,
    },
    'adversarial': {
        'permutation': {'p': round(perm_p, 3), 'pass': bool(perm_pass)},
        'sub_period': {'cv': round(sub_cv, 3), 'pass': bool(sub_pass)},
        'outlier': {'deg': round(outlier_deg, 3), 'pass': bool(outlier_pass)},
        'r1_regime': {'gap': round(float(r1_gap), 3), 'pass': bool(r1_pass)},
        'gates_passed': gates,
        'verdict': 'PASS' if gates >= 3 else 'FAIL'
    },
    'parameters': {
        'rsi_window': RSI_WINDOW,
        'bb_window': BB_WINDOW,
        'bb_std': BB_STD,
        'revert_horizon': REVERT_HORIZON,
        'ml_threshold': ML_THRESHOLD,
        'max_positions': MAX_POSITIONS,
        'train_window': TRAIN_WINDOW,
    }
}

with open(OUTPUT / 'results.json', 'w') as f:
    json.dump(results, f, indent=2, default=lambda x: int(x) if hasattr(x, 'item') else str(x))

# Equity curve
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
ax = axes[0]
cum_strat = (1 + rets).cumprod() * INITIAL_CAPITAL
spy_cum = (1 + prices['SPY'].pct_change().reindex(rets.index).fillna(0)).cumprod() * INITIAL_CAPITAL
ax.plot(cum_strat.index, cum_strat.values, 'b-', label=f'ML Mean Reversion (S={sharpe:.2f})')
ax.plot(spy_cum.index, spy_cum.values, 'k--', label='SPY B&H', alpha=0.5)
ax.set_ylabel('Portfolio Value ($)')
ax.set_title('ML Mean Reversion Strategy')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
monthly = rets.resample('ME').sum()
colors = ['g' if x > 0 else 'r' for x in monthly.values]
ax.bar(monthly.index, monthly.values * 100, width=20, color=colors, alpha=0.7)
ax.set_ylabel('Monthly Return (%)')
ax.set_title(f'Monthly Returns (WR={wr*100:.0f}%, {n_trades} total trades)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT / 'equity_curve.png', dpi=100)
plt.close()

# Save daily returns for combination analysis
rets.to_csv(OUTPUT / 'daily_returns.csv')

print(f"\n{'='*70}")
print(f"RESULT: ML Mean Reversion")
print(f"  Sharpe {sharpe:.3f} | Sortino {sortino:.3f} | CAGR {cagr*100:.1f}%")
print(f"  MaxDD {max_dd*100:.1f}% | WR {wr*100:.0f}% | {n_trades} trades")
print(f"  Adversarial: {gates}/4 gates → {'PASS' if gates >= 3 else 'FAIL'}")
print(f"{'='*70}")
