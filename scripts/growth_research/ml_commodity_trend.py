#!/usr/bin/env python3
"""
ML Commodity Trend Following — v2 Framework Applied to Commodities
===================================================================
Takes the proven ML trend v2 framework (Sharpe 2.90 on 8-asset CTA,
Sharpe 1.99 on 11-sector ETFs) and applies it to commodity ETFs.

Universe: GLD (gold), SLV (silver), USO (oil), UNG (nat gas),
          DBA (agriculture), CPER (copper), PDBC (commodities broad),
          DBC (commodities), WEAT (wheat), CORN (corn).

HC compliance:
  - HC #0: Sliding walk-forward (252d train, 21d advance, oldest-drop)
  - HC #713: Fixed $100K capital, no DCA
  - HC #714: Growth + ML/AI for exploration
  - HC #710: Broad commodity signals
  - HC #428 R1: Regime-agnostic validation (40+ OOT days, gap < 0.50)
  - Permutation test: signal-shuffle (not return-shuffle)
  - Sub-period consistency, outlier robustness
"""

import json
import warnings
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')

# ─── Config ───
TICKERS = ['GLD', 'SLV', 'USO', 'UNG', 'DBA', 'CPER', 'DBC', 'PDBC']
BENCHMARK = 'SPY'
TRAIN_DAYS = 252
ADVANCE_DAYS = 21  # monthly rebalance
LABEL_HORIZON = 21  # 21-day forward return
START_YEAR = 2007
INITIAL_CAPITAL = 100_000
N_PERMS = 200
OUTPUT_DIR = Path('/home/jupiter/Lvl3Quant/output/ml_commodity_trend')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("ML COMMODITY TREND FOLLOWING — v2 FRAMEWORK")
print("=" * 70)

# ─── Download data ───
print(f"\nDownloading {len(TICKERS)} commodity ETFs + {BENCHMARK}...")
all_tickers = TICKERS + [BENCHMARK, 'TLT', 'VIX']  # VIX for regime classification

# Download with extra history for features
data = {}
for t in all_tickers + ['^VIX']:
    try:
        df = yf.download(t, start=f'{START_YEAR-1}-01-01', end='2026-07-19',
                         progress=False, auto_adjust=True)
        if len(df) > 100:
            # Handle multi-level columns from newer yfinance
            if isinstance(df.columns, pd.MultiIndex):
                close = df[('Close', t)].copy()
            else:
                close = df['Close'].copy()
            clean_name = t.replace('^', '')
            close.name = clean_name
            data[clean_name] = close
            print(f"  {t}: {len(df)} days ({df.index[0].date()} to {df.index[-1].date()})")
        else:
            print(f"  {t}: SKIP (only {len(df)} days)")
    except Exception as e:
        print(f"  {t}: FAILED ({e})")

# Align all data
prices = pd.DataFrame(data)
prices = prices.dropna(how='all').ffill().dropna()
print(f"\nAligned: {len(prices)} days, {prices.shape[1]} assets")
print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

# Filter to only assets with enough data
min_days = TRAIN_DAYS + 252  # need at least 1yr train + 1yr buffer
valid_tickers = [t for t in TICKERS if t in prices.columns and prices[t].notna().sum() >= min_days]
print(f"Valid tickers ({len(valid_tickers)}): {valid_tickers}")

if len(valid_tickers) < 3:
    print("ERROR: Need at least 3 valid commodity ETFs")
    exit(1)

# ─── Feature engineering (same as v2 framework) ───
def build_features(prices_df, ticker, ref_date_idx):
    """Build features for a single asset at a point in time."""
    p = prices_df[ticker].iloc[:ref_date_idx+1]
    spy = prices_df['SPY'].iloc[:ref_date_idx+1] if 'SPY' in prices_df.columns else None
    vix = prices_df['VIX'].iloc[:ref_date_idx+1] if 'VIX' in prices_df.columns else None

    if len(p) < 252:
        return None

    feats = {}

    # Price momentum features (multiple windows)
    for w in [5, 10, 21, 42, 63, 126, 252]:
        if len(p) > w:
            feats[f'ret_{w}d'] = (p.iloc[-1] / p.iloc[-w] - 1) if p.iloc[-w] > 0 else 0

    # Moving average features
    for w in [10, 21, 50, 100, 200]:
        if len(p) > w:
            ma = p.iloc[-w:].mean()
            feats[f'price_vs_ma{w}'] = (p.iloc[-1] / ma - 1) if ma > 0 else 0

    # Volatility features
    rets = p.pct_change().dropna()
    for w in [10, 21, 63]:
        if len(rets) > w:
            feats[f'vol_{w}d'] = rets.iloc[-w:].std() * np.sqrt(252)

    # Vol ratio (short/long)
    if len(rets) > 63:
        vol_21 = rets.iloc[-21:].std()
        vol_63 = rets.iloc[-63:].std()
        feats['vol_ratio_21_63'] = vol_21 / vol_63 if vol_63 > 0 else 1

    # Drawdown from peak
    if len(p) > 252:
        peak_252 = p.iloc[-252:].max()
        feats['dd_from_252d_peak'] = (p.iloc[-1] / peak_252 - 1) if peak_252 > 0 else 0

    # RSI (14-day)
    if len(rets) > 14:
        gains = rets.iloc[-14:].clip(lower=0).mean()
        losses = (-rets.iloc[-14:].clip(upper=0)).mean()
        feats['rsi_14'] = 100 * gains / (gains + losses) if (gains + losses) > 0 else 50

    # Cross-asset features
    if spy is not None and len(spy) > 21:
        spy_ret_21 = (spy.iloc[-1] / spy.iloc[-21] - 1) if spy.iloc[-21] > 0 else 0
        asset_ret_21 = feats.get('ret_21d', 0)
        feats['rel_strength_vs_spy'] = asset_ret_21 - spy_ret_21

        # Correlation with SPY (rolling 63d)
        if len(rets) > 63:
            spy_rets = spy.pct_change().dropna()
            if len(spy_rets) > 63:
                common_idx = rets.index.intersection(spy_rets.index)[-63:]
                if len(common_idx) > 30:
                    feats['corr_spy_63d'] = rets.loc[common_idx].corr(spy_rets.loc[common_idx])

    # VIX features
    if vix is not None and len(vix) > 21:
        feats['vix_level'] = vix.iloc[-1]
        feats['vix_ret_21d'] = (vix.iloc[-1] / vix.iloc[-21] - 1) if vix.iloc[-21] > 0 else 0
        if len(vix) > 63:
            feats['vix_percentile_63d'] = (vix.iloc[-1] - vix.iloc[-63:].min()) / \
                                           (vix.iloc[-63:].max() - vix.iloc[-63:].min() + 1e-8)

    # Mean reversion features
    if len(p) > 21:
        z_21 = (p.iloc[-1] - p.iloc[-21:].mean()) / (p.iloc[-21:].std() + 1e-8)
        feats['zscore_21d'] = z_21

    # Commodity-specific: cross-commodity momentum
    # (average momentum across all commodity ETFs vs this one)

    return feats


# ─── Walk-forward backtest ───
print("\n" + "=" * 70)
print("WALK-FORWARD BACKTEST")
print("=" * 70)

try:
    import lightgbm as lgb
    USE_LGB = True
    print("Using LightGBM")
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    USE_LGB = False
    print("LightGBM not available, using sklearn GBM")

# Build all features and labels
print("\nBuilding features for all assets and dates...")
all_features = []
all_labels = []
all_meta = []

dates = prices.index.tolist()

for i in range(TRAIN_DAYS, len(dates) - LABEL_HORIZON):
    date = dates[i]

    for ticker in valid_tickers:
        feats = build_features(prices, ticker, i)
        if feats is None:
            continue

        # Label: will this asset outperform equal-weight basket over next 21 days?
        future_ret = (prices[ticker].iloc[i + LABEL_HORIZON] / prices[ticker].iloc[i]) - 1
        basket_ret = np.mean([(prices[t].iloc[i + LABEL_HORIZON] / prices[t].iloc[i]) - 1
                              for t in valid_tickers if not np.isnan(prices[t].iloc[i])])

        label = 1 if future_ret > basket_ret else 0  # outperformer

        feats['_ticker'] = ticker
        feats['_date'] = date
        feats['_label'] = label
        feats['_future_ret'] = future_ret

        all_features.append(feats)

df_all = pd.DataFrame(all_features)
print(f"Total observations: {len(df_all)}")
print(f"Date range: {df_all['_date'].min().date()} to {df_all['_date'].max().date()}")
print(f"Label balance: {df_all['_label'].mean():.1%} positive")

feature_cols = [c for c in df_all.columns if not c.startswith('_')]
print(f"Features: {len(feature_cols)}")

# ─── Walk-forward with sliding window ───
# Collect predictions for each rebalance date
results_by_date = {}
fold_count = 0
rebalance_dates = df_all['_date'].unique()[TRAIN_DAYS::ADVANCE_DAYS]

print(f"\nRunning walk-forward: {len(rebalance_dates)} rebalance periods")

all_unique_dates = df_all['_date'].unique()
for reb_idx, reb_date in enumerate(rebalance_dates):
    # HC #718: label gap = LABEL_HORIZON to prevent look-ahead
    # Exclude last LABEL_HORIZON dates before reb_date so no training label overlaps test
    all_prior_dates = sorted(all_unique_dates[all_unique_dates < reb_date])
    if len(all_prior_dates) > LABEL_HORIZON:
        train_cutoff_date = all_prior_dates[-LABEL_HORIZON]
    else:
        continue
    train_mask = (df_all['_date'] < train_cutoff_date)
    train_dates = df_all[train_mask]['_date'].unique()
    if len(train_dates) < TRAIN_DAYS // 2:
        continue

    # Use only last TRAIN_DAYS of unique dates (sliding window)
    cutoff_dates = sorted(train_dates)[-TRAIN_DAYS:]
    train_df = df_all[(df_all['_date'].isin(cutoff_dates))]

    # Test data: this rebalance date
    test_df = df_all[df_all['_date'] == reb_date]

    if len(test_df) == 0 or len(train_df) < 50:
        continue

    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df['_label'].values
    X_test = test_df[feature_cols].fillna(0).values

    # Train model
    if USE_LGB:
        model = lgb.LGBMClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_samples=20, verbose=-1, n_jobs=4
        )
    else:
        model = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=20
        )

    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]

    # Store predictions for this date
    preds = []
    for j, (_, row) in enumerate(test_df.iterrows()):
        preds.append({
            'ticker': row['_ticker'],
            'prob': probs[j],
            'future_ret': row['_future_ret']
        })

    results_by_date[reb_date] = preds
    fold_count += 1

    if fold_count % 50 == 0:
        print(f"  Fold {fold_count}/{len(rebalance_dates)}...")

print(f"Completed {fold_count} folds")

# ─── Portfolio construction ───
# Each rebalance: go long top-ranked assets, short bottom-ranked (or cash)
print("\n" + "=" * 70)
print("PORTFOLIO CONSTRUCTION")
print("=" * 70)

# Strategy: long top 3 predicted outperformers, equal weight
# Alternative: long assets with P > 0.5, cash otherwise
portfolio_returns = []
portfolio_dates = []
positions_log = []

# HC #718 R3: Transaction costs — 5 bps per leg on turnover
COST_BPS = 5
prev_top_tickers = set()

for reb_date in sorted(results_by_date.keys()):
    preds = results_by_date[reb_date]
    if not preds:
        continue

    # Sort by probability
    preds_sorted = sorted(preds, key=lambda x: x['prob'], reverse=True)

    # Strategy 1: Top-K long (K = min(3, n_assets/2))
    k = min(3, len(preds_sorted) // 2)
    if k == 0:
        k = 1

    top_k = preds_sorted[:k]
    current_tickers = set(p['ticker'] for p in top_k)

    # Equal weight among selected
    period_ret = np.mean([p['future_ret'] for p in top_k])

    # HC #718 R3: transaction costs — deduct on turnover (changed positions)
    if prev_top_tickers:
        turnover_frac = len(current_tickers.symmetric_difference(prev_top_tickers)) / max(len(current_tickers), len(prev_top_tickers))
        period_ret -= turnover_frac * 2 * COST_BPS / 10000  # sell old + buy new per changed leg
    else:
        period_ret -= COST_BPS / 10000  # initial buy
    prev_top_tickers = current_tickers

    # Also compute equal-weight benchmark
    equal_weight_ret = np.mean([p['future_ret'] for p in preds_sorted])

    portfolio_returns.append({
        'date': reb_date,
        'ml_return': period_ret,
        'equal_weight_return': equal_weight_ret,
        'n_long': k,
        'top_tickers': [p['ticker'] for p in top_k],
        'top_probs': [p['prob'] for p in top_k]
    })

if not portfolio_returns:
    print("ERROR: No portfolio returns generated")
    exit(1)

df_port = pd.DataFrame(portfolio_returns)
print(f"Portfolio periods: {len(df_port)}")
print(f"Date range: {df_port['date'].min().date()} to {df_port['date'].max().date()}")

# ─── Performance metrics ───
def calc_metrics(returns, label="Strategy"):
    """Calculate risk-adjusted metrics from period returns."""
    r = np.array(returns)
    n_periods = len(r)
    periods_per_year = 252 / ADVANCE_DAYS  # ~12 periods/year

    # Annualize
    mean_ret = np.mean(r) * periods_per_year
    std_ret = np.std(r, ddof=1) * np.sqrt(periods_per_year)

    sharpe = mean_ret / std_ret if std_ret > 0 else 0

    # Sortino
    downside = r[r < 0]
    down_std = np.std(downside, ddof=1) * np.sqrt(periods_per_year) if len(downside) > 1 else std_ret
    sortino = mean_ret / down_std if down_std > 0 else 0

    # CAGR (compound)
    cum = np.cumprod(1 + r)
    n_years = n_periods / periods_per_year
    cagr = (cum[-1] ** (1 / n_years) - 1) * 100 if n_years > 0 and cum[-1] > 0 else 0

    # MaxDD
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    max_dd = dd.min() * 100

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Win rate
    wr = np.mean(r > 0) * 100

    # Profit factor
    gains = r[r > 0].sum()
    losses = abs(r[r < 0].sum())
    pf = gains / losses if losses > 0 else float('inf')

    print(f"\n{label}:")
    print(f"  Sharpe:  {sharpe:.3f}")
    print(f"  Sortino: {sortino:.3f}")
    print(f"  CAGR:    {cagr:.1f}%")
    print(f"  MaxDD:   {max_dd:.1f}%")
    print(f"  Calmar:  {calmar:.2f}")
    print(f"  WR:      {wr:.1f}%")
    print(f"  PF:      {pf:.2f}")
    print(f"  Periods: {n_periods}")

    return {
        'sharpe': round(sharpe, 3), 'sortino': round(sortino, 3),
        'cagr': round(cagr, 1), 'max_dd': round(max_dd, 1),
        'calmar': round(calmar, 2), 'wr': round(wr, 1),
        'pf': round(pf, 2), 'n_periods': n_periods,
        'cum_return': round((cum[-1] - 1) * 100, 1)
    }

ml_metrics = calc_metrics(df_port['ml_return'].values, "ML Top-K Commodity Trend")
ew_metrics = calc_metrics(df_port['equal_weight_return'].values, "Equal-Weight Commodity")

# SPY benchmark over same periods
spy_returns = []
for _, row in df_port.iterrows():
    d = row['date']
    d_idx = prices.index.get_loc(d)
    if d_idx + LABEL_HORIZON < len(prices):
        spy_ret = (prices['SPY'].iloc[d_idx + LABEL_HORIZON] / prices['SPY'].iloc[d_idx]) - 1
        spy_returns.append(spy_ret)
    else:
        spy_returns.append(0)

spy_metrics = calc_metrics(spy_returns, "SPY Buy & Hold (same periods)")

# ─── Year-by-year breakdown (HC #713 — proper CAGR) ───
print("\n" + "=" * 70)
print("YEAR-BY-YEAR RETURNS")
print("=" * 70)

df_port['year'] = df_port['date'].apply(lambda d: d.year)
yearly = df_port.groupby('year').agg({
    'ml_return': lambda x: (np.prod(1 + x) - 1) * 100,
    'equal_weight_return': lambda x: (np.prod(1 + x) - 1) * 100
}).rename(columns={'ml_return': 'ML_Return_%', 'equal_weight_return': 'EW_Return_%'})

spy_yearly = pd.DataFrame(spy_returns, index=df_port['date'].values)
spy_yearly['year'] = pd.DatetimeIndex(spy_yearly.index).year
spy_yr = spy_yearly.groupby('year')[0].apply(lambda x: (np.prod(1 + x) - 1) * 100).rename('SPY_%')
yearly = yearly.join(spy_yr)

print(yearly.round(1).to_string())

negative_years = (yearly['ML_Return_%'] < 0).sum()
print(f"\nNegative years: {negative_years}/{len(yearly)}")

# ─── ADVERSARIAL VALIDATION ───
print("\n" + "=" * 70)
print("ADVERSARIAL VALIDATION")
print("=" * 70)

# 1. Permutation test (signal shuffle — HC #428)
print("\n--- PERMUTATION TEST (n=200, signal shuffle) ---")
ml_sharpe = ml_metrics['sharpe']
perm_sharpes = []

for perm_i in range(N_PERMS):
    perm_returns = []
    for reb_date in sorted(results_by_date.keys()):
        preds = results_by_date[reb_date]
        if not preds:
            continue

        # Shuffle probabilities (break signal-to-asset mapping)
        shuffled_probs = np.random.permutation([p['prob'] for p in preds])
        preds_shuffled = sorted(zip(shuffled_probs, [p['future_ret'] for p in preds]),
                                 key=lambda x: x[0], reverse=True)

        k = min(3, len(preds_shuffled) // 2)
        if k == 0:
            k = 1

        period_ret = np.mean([p[1] for p in preds_shuffled[:k]])
        perm_returns.append(period_ret)

    if perm_returns:
        r = np.array(perm_returns)
        periods_per_year = 252 / ADVANCE_DAYS
        mean_r = np.mean(r) * periods_per_year
        std_r = np.std(r, ddof=1) * np.sqrt(periods_per_year)
        perm_sharpe = mean_r / std_r if std_r > 0 else 0
        perm_sharpes.append(perm_sharpe)

    if (perm_i + 1) % 50 == 0:
        print(f"  Perm {perm_i + 1}/{N_PERMS}...")

p_value = np.mean([ps >= ml_sharpe for ps in perm_sharpes])
print(f"  Observed Sharpe: {ml_sharpe:.3f}")
print(f"  Perm mean Sharpe: {np.mean(perm_sharpes):.3f}")
print(f"  Perm p95 Sharpe:  {np.percentile(perm_sharpes, 95):.3f}")
print(f"  p-value: {p_value:.3f}")
perm_verdict = "PASS" if p_value < 0.05 else "FAIL"
print(f"  Verdict: {perm_verdict}")

# 2. Sub-period consistency
print("\n--- SUB-PERIOD CONSISTENCY ---")
n_periods_total = len(df_port)
quarter_size = n_periods_total // 4
sub_sharpes = []
for q in range(4):
    start = q * quarter_size
    end = (q + 1) * quarter_size if q < 3 else n_periods_total
    sub_rets = df_port['ml_return'].iloc[start:end].values
    periods_per_year = 252 / ADVANCE_DAYS
    mean_r = np.mean(sub_rets) * periods_per_year
    std_r = np.std(sub_rets, ddof=1) * np.sqrt(periods_per_year)
    sub_sharpe = mean_r / std_r if std_r > 0 else 0
    sub_sharpes.append(sub_sharpe)
    dates_range = f"{df_port['date'].iloc[start].strftime('%Y-%m')} to {df_port['date'].iloc[end-1].strftime('%Y-%m')}"
    print(f"  Q{q+1} ({dates_range}): Sharpe {sub_sharpe:.3f}")

if np.mean(sub_sharpes) > 0:
    cv = np.std(sub_sharpes) / abs(np.mean(sub_sharpes))
else:
    cv = 999
sub_verdict = "PASS" if cv < 0.75 and all(s > -0.5 for s in sub_sharpes) else "FAIL"
print(f"  CV: {cv:.3f}")
print(f"  Verdict: {sub_verdict}")

# 3. Outlier robustness (remove top 5% returns)
print("\n--- OUTLIER ROBUSTNESS ---")
ml_rets = df_port['ml_return'].values
cutoff = np.percentile(abs(ml_rets), 95)
trimmed = ml_rets[abs(ml_rets) <= cutoff]
if len(trimmed) > 5:
    periods_per_year = 252 / ADVANCE_DAYS
    trim_mean = np.mean(trimmed) * periods_per_year
    trim_std = np.std(trimmed, ddof=1) * np.sqrt(periods_per_year)
    trimmed_sharpe = trim_mean / trim_std if trim_std > 0 else 0
    degradation = (1 - trimmed_sharpe / ml_sharpe) * 100 if ml_sharpe != 0 else 0
    print(f"  Full Sharpe: {ml_sharpe:.3f}")
    print(f"  Trimmed Sharpe: {trimmed_sharpe:.3f}")
    print(f"  Degradation: {degradation:.1f}%")
    outlier_verdict = "PASS" if degradation < 50 else "FAIL"
    print(f"  Verdict: {outlier_verdict}")
else:
    outlier_verdict = "FAIL"
    trimmed_sharpe = 0
    degradation = 100

# 4. R1 Regime test (HC #428)
print("\n--- R1 REGIME TEST ---")
if 'VIX' in prices.columns:
    spy_close = prices['SPY']
    spy_daily_ret = spy_close.pct_change()

    green_dates = set()
    red_dates = set()

    # Classify each rebalance period by SPY performance
    for _, row in df_port.iterrows():
        d = row['date']
        d_idx = prices.index.get_loc(d)
        if d_idx + LABEL_HORIZON < len(prices):
            spy_period_ret = (prices['SPY'].iloc[d_idx + LABEL_HORIZON] / prices['SPY'].iloc[d_idx]) - 1
            if spy_period_ret >= 0:
                green_dates.add(d)
            else:
                red_dates.add(d)

    green_rets = df_port[df_port['date'].isin(green_dates)]['ml_return'].values
    red_rets = df_port[df_port['date'].isin(red_dates)]['ml_return'].values

    print(f"  Green periods: {len(green_rets)}, Red periods: {len(red_rets)}")

    if len(green_rets) > 5 and len(red_rets) > 5:
        periods_per_year = 252 / ADVANCE_DAYS
        green_sharpe = (np.mean(green_rets) * periods_per_year) / \
                       (np.std(green_rets, ddof=1) * np.sqrt(periods_per_year)) if np.std(green_rets) > 0 else 0
        red_sharpe = (np.mean(red_rets) * periods_per_year) / \
                     (np.std(red_rets, ddof=1) * np.sqrt(periods_per_year)) if np.std(red_rets) > 0 else 0

        gap = abs(green_sharpe - red_sharpe) / max(abs(green_sharpe), abs(red_sharpe), 0.01)

        print(f"  Green Sharpe: {green_sharpe:.3f}")
        print(f"  Red Sharpe:   {red_sharpe:.3f}")
        print(f"  Gap:          {gap:.3f}")
        r1_verdict = "PASS" if gap < 0.50 else "FAIL"
        print(f"  Verdict: {r1_verdict}")
    else:
        r1_verdict = "FAIL"
        green_sharpe = red_sharpe = gap = 0
        print("  Insufficient periods for regime test")
else:
    r1_verdict = "FAIL"
    green_sharpe = red_sharpe = gap = 0
    print("  VIX data not available")

# ─── Correlation with existing strategies ───
print("\n" + "=" * 70)
print("CORRELATION WITH EXISTING STRATEGIES")
print("=" * 70)

# Approximate existing strategy returns using their asset universes
# CTA trend uses SPY, QQQ, GLD, TLT, EEM, VNQ, HYG, XLE
# Sector rotation uses XLK, XLF, etc.
# This strategy uses commodities — inherently different
print("  This strategy trades COMMODITY ETFs only")
print("  Expected low correlation with equity-based CTA and sector rotation")
print("  (Formal cross-strategy correlation requires daily returns from all strategies)")

# SPY correlation
spy_corr = np.corrcoef(df_port['ml_return'].values, spy_returns)[0, 1]
print(f"  SPY correlation: {spy_corr:.3f}")

# ─── Save results ───
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

gates_passed = sum([
    perm_verdict == "PASS",
    sub_verdict == "PASS",
    outlier_verdict == "PASS",
    r1_verdict == "PASS"
])

results = {
    'strategy': 'ML Commodity Trend Following',
    'framework': 'v2 (proven on CTA + sectors)',
    'universe': valid_tickers,
    'n_tickers': len(valid_tickers),
    'n_folds': fold_count,
    'n_periods': len(df_port),
    'date_range': f"{df_port['date'].min().date()} to {df_port['date'].max().date()}",
    'ml_metrics': ml_metrics,
    'ew_metrics': ew_metrics,
    'spy_metrics': spy_metrics,
    'yearly_returns': yearly.to_dict(),
    'negative_years': int(negative_years),
    'adversarial': {
        'permutation': {
            'observed_sharpe': ml_sharpe,
            'perm_mean_sharpe': round(np.mean(perm_sharpes), 3),
            'p_value': round(p_value, 3),
            'verdict': perm_verdict
        },
        'sub_period': {
            'sharpes': [round(s, 3) for s in sub_sharpes],
            'cv': round(cv, 3),
            'verdict': sub_verdict
        },
        'outlier': {
            'full_sharpe': ml_sharpe,
            'trimmed_sharpe': round(trimmed_sharpe, 3),
            'degradation_pct': round(degradation, 1),
            'verdict': outlier_verdict
        },
        'r1_regime': {
            'green_sharpe': round(green_sharpe, 3),
            'red_sharpe': round(red_sharpe, 3),
            'gap': round(gap, 3),
            'verdict': r1_verdict
        },
        'gates_passed': f"{gates_passed}/4"
    },
    'spy_correlation': round(spy_corr, 3),
    'timestamp': dt.datetime.now().isoformat()
}

with open(OUTPUT_DIR / 'results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

# Save daily returns for portfolio optimization
df_port.to_parquet(OUTPUT_DIR / 'portfolio_returns.parquet', index=False)

print(f"\nResults saved to {OUTPUT_DIR}")

# ─── Final verdict ───
print("\n" + "=" * 70)
verdict = "VALIDATED" if gates_passed >= 3 else ("PARTIAL" if gates_passed >= 2 else "REJECTED")
print(f"FINAL VERDICT: {verdict} ({gates_passed}/4 gates)")
print("=" * 70)
print(f"\nML Commodity Trend: Sharpe {ml_sharpe:.3f}, CAGR {ml_metrics['cagr']}%, MaxDD {ml_metrics['max_dd']}%")
print(f"Equal-Weight Commodities: Sharpe {ew_metrics['sharpe']:.3f}")
print(f"SPY: Sharpe {spy_metrics['sharpe']:.3f}")
print(f"Gates: Perm={perm_verdict}, SubP={sub_verdict}, Outlier={outlier_verdict}, R1={r1_verdict}")
