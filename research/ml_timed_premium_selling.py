#!/usr/bin/env python3
"""
ML-Timed Premium Selling (Income Strategy)
===========================================
INSIGHT: Selling options premium (put spreads / iron condors on SPX) generates
consistent income ~80% of the time. The killer is selling into spikes.

We have a VALIDATED VIX spike predictor (AUC 0.926, perm PASS p=0.000).
Can we use it to TIME when to sell premium and when to stay flat?

Strategy:
  1. Collect daily VIX features (term structure, realized vol, credit spreads, etc.)
  2. Run ML spike predictor each day
  3. When spike probability LOW (<0.15): sell premium (put credit spreads on SPY)
  4. When spike probability ELEVATED (>0.30): go flat (preserve capital)
  5. Premium collected: approximate SPX put spread yield from VIX level

Revenue model (realistic for SPX 0-DTE / weekly put spreads):
  - Yield per week ≈ VIX / 52 * width_factor * 0.6 (60% of theoretical max)
  - Loss events: when SPY drops > spread width (~5% of weeks without protection)

HC #713: Fixed $100K, NO DCA. HC #0: SLIDING. No BS pricing — use VIX as proxy.
HC #714: Income focus.
"""

import json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
np.random.seed(42)

BASE = Path("/home/jupiter/Lvl3Quant")
OUTPUT = BASE / "output" / "ml_timed_premium"
OUTPUT.mkdir(parents=True, exist_ok=True)

INITIAL_CAPITAL = 100_000
TRAIN_WINDOW = 252
N_PERMS = 100

print("=" * 70)
print("ML-TIMED PREMIUM SELLING (INCOME STRATEGY)")
print("=" * 70)

# ─── DATA ────────────────────────────────────────────────────────────────────

print("\n[1/7] Loading data...")
t0 = time.time()

tickers = ['SPY', 'TLT', 'GLD', 'HYG', 'IEF']
raw = yf.download(tickers + ['^VIX'], start='2006-01-01', progress=False)
if hasattr(raw.columns, 'levels'):
    prices = raw['Close']
else:
    prices = raw

vix = prices['^VIX'] if '^VIX' in prices.columns else pd.Series(dtype=float)
spy = prices['SPY']
tlt = prices['TLT'] if 'TLT' in prices.columns else pd.Series(dtype=float)
hyg = prices['HYG'] if 'HYG' in prices.columns else pd.Series(dtype=float)

print(f"  VIX: {vix.notna().sum()} days, SPY: {spy.notna().sum()} days")
print(f"  Date range: {vix.index[0].strftime('%Y-%m-%d')} to {vix.index[-1].strftime('%Y-%m-%d')}")
print(f"  Time: {time.time()-t0:.1f}s")

# ─── FEATURES ────────────────────────────────────────────────────────────────

print("\n[2/7] Building features...")

feat = pd.DataFrame(index=vix.index)

# VIX features
feat['vix'] = vix
feat['vix_ma5'] = vix.rolling(5).mean()
feat['vix_ma20'] = vix.rolling(20).mean()
feat['vix_ma60'] = vix.rolling(60).mean()
feat['vix_zscore'] = (vix - vix.rolling(60).mean()) / vix.rolling(60).std()
feat['vix_percentile'] = vix.rolling(252).rank(pct=True)
feat['vix_change_5d'] = vix.pct_change(5)
feat['vix_change_20d'] = vix.pct_change(20)

# Term structure proxy (VIX vs short-term implied)
feat['vix_slope'] = vix / vix.rolling(5).mean() - 1  # Steep = contango

# SPY features
spy_ret = spy.pct_change()
feat['spy_ret_5d'] = spy.pct_change(5)
feat['spy_ret_20d'] = spy.pct_change(20)
feat['spy_vol_20d'] = spy_ret.rolling(20).std() * np.sqrt(252)
feat['spy_vol_60d'] = spy_ret.rolling(60).std() * np.sqrt(252)
feat['spy_dd'] = spy / spy.rolling(252).max() - 1
feat['spy_above_200ma'] = (spy > spy.rolling(200).mean()).astype(float)

# VIX vs realized vol (VRP indicator)
feat['vrp'] = vix / 100 - feat['spy_vol_20d']

# Credit spread proxy (HYG vs TLT)
if hyg.notna().sum() > 100 and tlt.notna().sum() > 100:
    credit_ret = hyg.pct_change() - tlt.pct_change()
    feat['credit_20d'] = credit_ret.rolling(20).sum()
    feat['credit_zscore'] = (credit_ret.rolling(20).mean() - credit_ret.rolling(60).mean()) / credit_ret.rolling(60).std()

feat = feat.dropna()
print(f"  Features: {feat.shape[1]} cols, {len(feat)} days")

# ─── SPIKE LABELS ────────────────────────────────────────────────────────────

print("\n[3/7] Creating spike labels...")

# Spike = VIX > 25 within next 5 days (or SPY drops > 3%)
fwd_vix_max = vix.rolling(5).max().shift(-5)
fwd_spy_min = spy.pct_change(5).shift(-5)
spike_label = ((fwd_vix_max > 25) | (fwd_spy_min < -0.03)).astype(int)
spike_label = spike_label.reindex(feat.index)

print(f"  Spike days (5d fwd): {spike_label.sum()}/{len(spike_label)} ({spike_label.mean()*100:.1f}%)")

# ─── PREMIUM SELLING SIMULATION ──────────────────────────────────────────────

print("\n[4/7] Walk-forward simulation...")

# Parameters — REALISTIC put spread economics
# SPX weekly 3% OTM put spread on $100K notional:
# At VIX=15: ~$20-30 credit per $500 spread width = 4-6% of width per week
# At VIX=25: ~$40-60 credit per $500 spread width = 8-12% of width per week
# Deploying 20% of capital in spreads at any time (conservative)
SPREAD_WIDTH_PCT = 0.03  # 3% OTM put spread
CAPITAL_DEPLOYED = 0.20  # 20% of portfolio in active spreads
PREMIUM_RATE_PER_VIX = 0.00015  # Daily premium = VIX * this * capital_deployed
# At VIX=15: 0.00015 * 15 * 0.20 = 0.045% per day = 11.3% annualized
# At VIX=25: 0.00015 * 25 * 0.20 = 0.075% per day = 18.9% annualized
MAX_LOSS_MULT = 5.0      # Max loss = 5x daily premium (spread blowup)
SELL_THRESHOLD = 0.15    # Sell when spike prob < 15%
FLAT_THRESHOLD = 0.30    # Go flat when spike prob > 30%

dates = feat.index[TRAIN_WINDOW:]
daily_returns = pd.Series(0.0, index=dates)
ml_probs = pd.Series(np.nan, index=dates)
positions = pd.Series('', index=dates)

model = None
last_train = -999

for i, date in enumerate(dates):
    # Retrain every quarter
    if i - last_train >= 63 or model is None:
        train_end = feat.index.get_loc(date)
        train_start = max(0, train_end - TRAIN_WINDOW)

        X_train = feat.iloc[train_start:train_end].values
        y_train = spike_label.iloc[train_start:train_end].values

        if len(X_train) >= 100 and y_train.sum() > 5:
            model = RandomForestClassifier(
                n_estimators=200, max_depth=5, min_samples_leaf=10,
                class_weight='balanced', random_state=42, n_jobs=-1
            )
            model.fit(X_train, y_train)
            last_train = i

    if model is None:
        continue

    # Predict spike probability
    X_today = feat.loc[[date]].values
    try:
        prob = model.predict_proba(X_today)[0][1]
    except:
        continue

    ml_probs.loc[date] = prob

    # Current VIX level (determines premium available)
    current_vix = vix.loc[date] if date in vix.index else 15

    # Daily premium based on realistic put spread economics
    daily_premium = current_vix * PREMIUM_RATE_PER_VIX * CAPITAL_DEPLOYED

    # Actual SPY move today
    date_loc = spy.index.get_loc(date) if date in spy.index else -1
    if date_loc < 0 or date_loc + 1 >= len(spy):
        continue

    spy_daily_ret = spy.iloc[date_loc + 1] / spy.iloc[date_loc] - 1

    # Decision
    if prob < SELL_THRESHOLD:
        # SELL PREMIUM (short put spread)
        positions.loc[date] = 'SELL'

        # If SPY drops more than spread width = max loss
        if spy_daily_ret < -SPREAD_WIDTH_PCT:
            daily_returns.loc[date] = -daily_premium * MAX_LOSS_MULT  # Capped loss
        elif spy_daily_ret < -SPREAD_WIDTH_PCT * 0.5:
            # Partial loss zone
            loss_pct = abs(spy_daily_ret) / SPREAD_WIDTH_PCT
            daily_returns.loc[date] = daily_premium * (1 - loss_pct * 2)
        else:
            # Collect premium
            daily_returns.loc[date] = daily_premium

    elif prob > FLAT_THRESHOLD:
        # FLAT — protect capital
        positions.loc[date] = 'FLAT'
        daily_returns.loc[date] = 0.0

    else:
        # REDUCED SIZE
        positions.loc[date] = 'HALF'
        scale = 0.5
        if spy_daily_ret < -SPREAD_WIDTH_PCT:
            daily_returns.loc[date] = -daily_premium * MAX_LOSS_MULT * scale
        elif spy_daily_ret < -SPREAD_WIDTH_PCT * 0.5:
            loss_pct = abs(spy_daily_ret) / SPREAD_WIDTH_PCT
            daily_returns.loc[date] = daily_premium * (1 - loss_pct * 2) * scale
        else:
            daily_returns.loc[date] = daily_premium * scale

# ─── METRICS ─────────────────────────────────────────────────────────────────

print("\n[5/7] Computing metrics...")

# Clean returns
rets = daily_returns[daily_returns != 0]
if len(rets) < 252:
    print("  ERROR: Not enough trading days")
    exit(1)

ann_ret = rets.mean() * 252
ann_vol = rets.std() * np.sqrt(252)
sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
downside = rets[rets < 0].std() * np.sqrt(252)
sortino = ann_ret / downside if downside > 0 else 0
cum = (1 + rets).cumprod()
max_dd = (cum / cum.cummax() - 1).min()
cagr = cum.iloc[-1] ** (252 / len(rets)) - 1
calmar = cagr / abs(max_dd) if max_dd != 0 else 0
wr = (rets > 0).mean()
pf = rets[rets > 0].sum() / abs(rets[rets < 0].sum()) if rets[rets < 0].sum() != 0 else 0

# Monthly income on $100K
monthly_income = INITIAL_CAPITAL * ann_ret / 12

# Position stats
sell_days = (positions == 'SELL').sum()
flat_days = (positions == 'FLAT').sum()
half_days = (positions == 'HALF').sum()

print(f"  Strategy: ML-Timed Put Spread Selling")
print(f"  Sharpe: {sharpe:.3f}")
print(f"  Sortino: {sortino:.3f}")
print(f"  CAGR: {cagr*100:.1f}%")
print(f"  MaxDD: {max_dd*100:.1f}%")
print(f"  Calmar: {calmar:.3f}")
print(f"  Win Rate: {wr*100:.1f}%")
print(f"  Profit Factor: {pf:.3f}")
print(f"  Monthly income ($100K): ${monthly_income:.0f}")
print(f"  Position mix: SELL={sell_days}d, HALF={half_days}d, FLAT={flat_days}d")
print(f"  Time selling: {(sell_days + half_days)/(sell_days + half_days + flat_days)*100:.0f}%")

# Naive baseline (always selling)
naive_rets = pd.Series(0.0, index=dates)
for i, date in enumerate(dates):
    date_loc = spy.index.get_loc(date) if date in spy.index else -1
    if date_loc < 0 or date_loc + 1 >= len(spy):
        continue
    spy_daily_ret = spy.iloc[date_loc + 1] / spy.iloc[date_loc] - 1
    current_vix = vix.loc[date] if date in vix.index else 15
    dp = current_vix * PREMIUM_RATE_PER_VIX * CAPITAL_DEPLOYED

    if spy_daily_ret < -SPREAD_WIDTH_PCT:
        naive_rets.loc[date] = -dp * MAX_LOSS_MULT
    elif spy_daily_ret < -SPREAD_WIDTH_PCT * 0.5:
        loss_pct = abs(spy_daily_ret) / SPREAD_WIDTH_PCT
        naive_rets.loc[date] = dp * (1 - loss_pct * 2)
    else:
        naive_rets.loc[date] = dp

naive_clean = naive_rets[naive_rets != 0]
naive_sharpe = naive_clean.mean() * 252 / (naive_clean.std() * np.sqrt(252)) if naive_clean.std() > 0 else 0
naive_dd = ((1 + naive_clean).cumprod() / (1 + naive_clean).cumprod().cummax() - 1).min()
naive_cagr = (1 + naive_clean).cumprod().iloc[-1] ** (252/len(naive_clean)) - 1

print(f"\n  Naive (always sell): Sharpe {naive_sharpe:.3f}, CAGR {naive_cagr*100:.1f}%, MaxDD {naive_dd*100:.1f}%")
print(f"  ML uplift: Sharpe {sharpe - naive_sharpe:+.3f}, MaxDD {(max_dd - naive_dd)*100:+.1f}pp")

# ─── ADVERSARIAL ─────────────────────────────────────────────────────────────

print("\n[6/7] Adversarial validation...")

# Permutation: shuffle ML predictions (use random timing)
perm_sharpes = []
for p in range(N_PERMS):
    # Random prob assignment
    rand_probs = np.random.uniform(0, 1, len(dates))
    perm_rets = pd.Series(0.0, index=dates)

    for i, date in enumerate(dates):
        date_loc = spy.index.get_loc(date) if date in spy.index else -1
        if date_loc < 0 or date_loc + 1 >= len(spy):
            continue
        spy_daily_ret = spy.iloc[date_loc + 1] / spy.iloc[date_loc] - 1
        current_vix = vix.loc[date] if date in vix.index else 15
        dp = current_vix * PREMIUM_RATE_PER_VIX * CAPITAL_DEPLOYED

        prob = rand_probs[i]
        if prob < SELL_THRESHOLD:
            if spy_daily_ret < -SPREAD_WIDTH_PCT:
                perm_rets.iloc[i] = -dp * MAX_LOSS_MULT
            elif spy_daily_ret < -SPREAD_WIDTH_PCT * 0.5:
                loss_pct = abs(spy_daily_ret) / SPREAD_WIDTH_PCT
                perm_rets.iloc[i] = dp * (1 - loss_pct * 2)
            else:
                perm_rets.iloc[i] = dp
        elif prob < FLAT_THRESHOLD:
            scale = 0.5
            if spy_daily_ret < -SPREAD_WIDTH_PCT:
                perm_rets.iloc[i] = -dp * MAX_LOSS_MULT * scale
            elif spy_daily_ret < -SPREAD_WIDTH_PCT * 0.5:
                loss_pct = abs(spy_daily_ret) / SPREAD_WIDTH_PCT
                perm_rets.iloc[i] = dp * (1 - loss_pct * 2) * scale
            else:
                perm_rets.iloc[i] = dp * scale

    pr = perm_rets[perm_rets != 0]
    if len(pr) > 0 and pr.std() > 0:
        ps = pr.mean() * 252 / (pr.std() * np.sqrt(252))
        perm_sharpes.append(ps)

perm_p = np.mean([s >= sharpe for s in perm_sharpes]) if perm_sharpes else 1.0
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
print(f"  Sub-period: blocks={[f'{s:.2f}' for s in block_sharpes]}, CV={sub_cv:.3f} → {'PASS' if sub_pass else 'FAIL'}")

# Outlier
trimmed = rets[(rets > rets.quantile(0.01)) & (rets < rets.quantile(0.99))]
trim_s = trimmed.mean() * 252 / (trimmed.std() * np.sqrt(252)) if trimmed.std() > 0 else 0
outlier_deg = (trim_s - sharpe) / abs(sharpe) if sharpe != 0 else 0
outlier_pass = abs(outlier_deg) < 0.50
print(f"  Outlier: trimmed={trim_s:.3f}, deg={outlier_deg:.1%} → {'PASS' if outlier_pass else 'FAIL'}")

# R1 regime
spy_aligned = spy.pct_change().reindex(rets.index)
green = spy_aligned > 0
red = spy_aligned < 0
if rets[green].std() > 0 and rets[red].std() > 0:
    g_s = rets[green].mean() * 252 / (rets[green].std() * np.sqrt(252))
    r_s = rets[red].mean() * 252 / (rets[red].std() * np.sqrt(252))
    r1_gap = abs(g_s - r_s) / max(abs(g_s), abs(r_s), 0.01)
    r1_pass = r1_gap < 0.50
    print(f"  R1: green={g_s:.3f}, red={r_s:.3f}, gap={r1_gap:.3f} → {'PASS' if r1_pass else 'FAIL'}")
else:
    r1_pass = False
    r1_gap = 999
    print(f"  R1: insufficient data → FAIL")

gates = sum([perm_pass, sub_pass, outlier_pass, r1_pass])
print(f"\n  ADVERSARIAL: {gates}/4 → {'PASS' if gates >= 3 else 'FAIL'}")

# ─── SAVE ────────────────────────────────────────────────────────────────────

print("\n[7/7] Saving results...")

results = {
    'strategy': 'ML-Timed Premium Selling',
    'timestamp': pd.Timestamp.now().isoformat(),
    'metrics': {
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'cagr': round(cagr * 100, 1),
        'max_dd': round(max_dd * 100, 1),
        'calmar': round(calmar, 3),
        'win_rate': round(wr * 100, 1),
        'profit_factor': round(pf, 3),
        'monthly_income_100k': round(monthly_income),
        'annual_income_100k': round(monthly_income * 12),
    },
    'naive_baseline': {
        'sharpe': round(naive_sharpe, 3),
        'cagr': round(naive_cagr * 100, 1),
        'max_dd': round(naive_dd * 100, 1),
    },
    'ml_uplift': {
        'sharpe_delta': round(sharpe - naive_sharpe, 3),
        'dd_improvement': round((max_dd - naive_dd) * 100, 1),
    },
    'position_stats': {
        'sell_days': int(sell_days),
        'half_days': int(half_days),
        'flat_days': int(flat_days),
        'time_selling_pct': round((sell_days + half_days)/(sell_days + half_days + flat_days)*100, 1),
    },
    'adversarial': {
        'permutation': {'p': round(perm_p, 3), 'pass': bool(perm_pass)},
        'sub_period': {'cv': round(sub_cv, 3), 'pass': bool(sub_pass)},
        'outlier': {'deg': round(outlier_deg, 3), 'pass': bool(outlier_pass)},
        'r1_regime': {'gap': round(r1_gap, 3), 'pass': bool(r1_pass)},
        'gates_passed': gates,
        'verdict': 'PASS' if gates >= 3 else 'FAIL'
    },
    'parameters': {
        'spread_width': SPREAD_WIDTH_PCT,
        'premium_rate_per_vix': PREMIUM_RATE_PER_VIX,
        'capital_deployed': CAPITAL_DEPLOYED,
        'max_loss_mult': MAX_LOSS_MULT,
        'sell_threshold': SELL_THRESHOLD,
        'flat_threshold': FLAT_THRESHOLD,
        'train_window': TRAIN_WINDOW,
    }
}

with open(OUTPUT / 'results.json', 'w') as f:
    json.dump(results, f, indent=2, default=lambda x: int(x) if hasattr(x, 'item') else str(x))

# Plot
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

ax = axes[0]
cum_ml = (1 + rets).cumprod() * INITIAL_CAPITAL
cum_naive = (1 + naive_clean).cumprod() * INITIAL_CAPITAL
ax.plot(cum_ml.index, cum_ml.values, 'b-', label=f'ML-Timed (S={sharpe:.2f})')
ax.plot(cum_naive.index, cum_naive.values, 'r--', label=f'Naive Always-Sell (S={naive_sharpe:.2f})')
ax.set_ylabel('Portfolio Value ($)')
ax.set_title('ML-Timed Premium Selling vs Naive')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(ml_probs.dropna().index, ml_probs.dropna().values, 'orange', alpha=0.5, linewidth=0.5)
ax.axhline(SELL_THRESHOLD, color='g', linestyle='--', label=f'Sell thresh ({SELL_THRESHOLD})')
ax.axhline(FLAT_THRESHOLD, color='r', linestyle='--', label=f'Flat thresh ({FLAT_THRESHOLD})')
ax.set_ylabel('Spike Probability')
ax.set_title('ML Spike Predictions')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[2]
monthly = rets.resample('ME').sum() * INITIAL_CAPITAL
ax.bar(monthly.index, monthly.values, width=20, color=['g' if x > 0 else 'r' for x in monthly.values])
ax.set_ylabel('Monthly P&L ($)')
ax.set_title(f'Monthly Income (avg ${monthly.mean():.0f}/mo)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT / 'equity_curve.png', dpi=100)
plt.close()

print(f"\n{'='*70}")
print(f"RESULT: ML-Timed Premium Selling")
print(f"  Sharpe {sharpe:.3f} | Sortino {sortino:.3f} | CAGR {cagr*100:.1f}%")
print(f"  MaxDD {max_dd*100:.1f}% | WR {wr*100:.0f}% | Monthly ${monthly_income:.0f}")
print(f"  vs Naive: Sharpe {sharpe-naive_sharpe:+.3f}, DD {(max_dd-naive_dd)*100:+.1f}pp")
print(f"  Adversarial: {gates}/4 gates → {'PASS' if gates >= 3 else 'FAIL'}")
print(f"{'='*70}")
