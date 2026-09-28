#!/usr/bin/env python3
"""
ML Capitulation Recovery Strategy
===================================
Asymmetric upside: detect market capitulation (panic selling exhaustion)
and position for recovery with leveraged exposure.

Concept:
- Capitulation = extreme fear + volume spike + VIX inversion + credit stress
- After capitulation, markets typically snap back 5-15% in 20-60 days
- ML learns which capitulation signals produce the strongest recoveries
- Default position: SPY (passive). On capitulation signal: UPRO (3x SPY)
- This is structurally DIFFERENT from prior work:
  - Not drawdown prediction (predicting when DD happens) — this predicts RECOVERY
  - Not VIX spike buying (fixed threshold) — ML learns multi-factor capitulation
  - Not sector rotation — single-asset timing (SPY vs UPRO vs SHY)

Universe: SPY (passive), UPRO (recovery bet), SHY (defensive)
Signal: LightGBM predicts probability of >5% SPY recovery in next 42 days
        after cross-asset stress signals fire
Rebalance: Weekly (every 5 trading days)

HC compliance:
  - HC #0: Sliding walk-forward (252d train, 5d advance, oldest-drop)
  - HC #713: Fixed $100K capital, no DCA
  - HC #428 R1: Regime-agnostic validation
  - Full 4-gate adversarial
"""

import json
import warnings
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb

warnings.filterwarnings('ignore')

# --- Config ---
ASSETS = {
    'core': ['SPY', 'UPRO', 'SHY'],
    'features': ['SPY', 'TLT', 'GLD', 'HYG', 'IEF', 'EEM', 'IWM', 'QQQ',
                 'XLF', 'XLU', 'DBC', 'UUP', 'LQD', 'VXX'],
}
BENCHMARK = 'SPY'
TRAIN_DAYS = 252  # 1 year sliding window
ADVANCE_DAYS = 5  # weekly rebalance
RECOVERY_HORIZON = 42  # 2-month recovery window
RECOVERY_THRESHOLD = 0.05  # 5% recovery = capitulation signal target
INITIAL_CAPITAL = 100_000
N_PERMS = 200
OUTPUT_DIR = Path('/home/jupiter/Lvl3Quant/output/ml_capitulation_recovery')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("ML CAPITULATION RECOVERY STRATEGY")
print("=" * 70)

# --- Download data ---
all_tickers = list(set(ASSETS['core'] + ASSETS['features']))
print(f"\nDownloading {len(all_tickers)} tickers...")

data = {}
for t in all_tickers + ['^VIX', '^VIX3M']:
    try:
        df = yf.download(t, start='2006-01-01', end='2026-07-21',
                         progress=False, auto_adjust=True)
        if len(df) > 100:
            if isinstance(df.columns, pd.MultiIndex):
                close = df[('Close', t)].copy()
                vol = df[('Volume', t)].copy() if ('Volume', t) in df.columns else None
            else:
                close = df['Close'].copy()
                vol = df['Volume'].copy() if 'Volume' in df.columns else None
            clean_name = t.replace('^', '')
            data[clean_name] = close
            if vol is not None:
                data[f'{clean_name}_vol'] = vol
            print(f"  {t}: {len(df)} days")
        else:
            print(f"  {t}: SKIP ({len(df)} days)")
    except Exception as e:
        print(f"  {t}: FAILED ({e})")

prices = pd.DataFrame({k: v for k, v in data.items() if '_vol' not in k})
volumes = pd.DataFrame({k.replace('_vol', ''): v for k, v in data.items() if '_vol' in k})
prices = prices.dropna(how='all').ffill().dropna()
volumes = volumes.reindex(prices.index).ffill().fillna(0)

print(f"\nAligned: {len(prices)} days, {prices.shape[1]} assets")
print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

# Check UPRO availability
has_upro = 'UPRO' in prices.columns
upro_start = prices['UPRO'].first_valid_index() if has_upro else None
print(f"UPRO available: {has_upro} (from {upro_start.date() if upro_start else 'N/A'})")

# --- Feature engineering ---
def build_capitulation_features(prices_df, volumes_df, idx):
    """Build features that detect capitulation/extreme stress."""
    feats = {}

    if idx < 252:
        return None

    spy = prices_df['SPY'].iloc[:idx+1]
    spy_rets = spy.pct_change().dropna()

    if len(spy_rets) < 252:
        return None

    # === DRAWDOWN FEATURES ===
    # Current drawdown from recent highs
    for w in [21, 42, 63, 126, 252]:
        if len(spy) > w:
            peak = spy.iloc[-w:].max()
            feats[f'dd_from_{w}d_high'] = float(spy.iloc[-1] / peak - 1)

    # Speed of drawdown (how fast did we get here?)
    for w in [5, 10, 21]:
        if len(spy_rets) > w:
            feats[f'ret_{w}d'] = float(spy.iloc[-1] / spy.iloc[-w] - 1)

    # === VOLATILITY / FEAR FEATURES ===
    # Realized vol at multiple windows
    for w in [5, 10, 21, 63]:
        if len(spy_rets) > w:
            feats[f'rvol_{w}d'] = float(spy_rets.iloc[-w:].std() * np.sqrt(252))

    # Vol ratio (short-term panic vs normal)
    if len(spy_rets) > 63:
        vol_5 = spy_rets.iloc[-5:].std()
        vol_63 = spy_rets.iloc[-63:].std()
        feats['vol_ratio_5_63'] = float(vol_5 / max(vol_63, 1e-8))

    # VIX features
    if 'VIX' in prices_df.columns:
        vix = prices_df['VIX'].iloc[:idx+1]
        if len(vix) > 252:
            feats['vix'] = float(vix.iloc[-1])
            feats['vix_5d_chg'] = float(vix.iloc[-1] / vix.iloc[-5] - 1) if vix.iloc[-5] > 0 else 0
            feats['vix_zscore_63'] = float((vix.iloc[-1] - vix.iloc[-63:].mean()) / max(vix.iloc[-63:].std(), 0.1))
            feats['vix_pctile_252'] = float((vix.iloc[-252:] < vix.iloc[-1]).mean())
            # VIX spike (how far above its MA)
            feats['vix_vs_ma21'] = float(vix.iloc[-1] / vix.iloc[-21:].mean() - 1)
            feats['vix_vs_ma63'] = float(vix.iloc[-1] / vix.iloc[-63:].mean() - 1)

    # VIX term structure (inversion = panic)
    if 'VIX' in prices_df.columns and 'VIX3M' in prices_df.columns:
        vix_val = prices_df['VIX'].iloc[idx]
        vix3m_val = prices_df['VIX3M'].iloc[idx]
        if vix3m_val > 0:
            feats['vix_term_ratio'] = float(vix_val / vix3m_val)
            feats['vix_inverted'] = float(1 if vix_val > vix3m_val else 0)

    # === VOLUME FEATURES (capitulation signature) ===
    if 'SPY' in volumes_df.columns:
        sv = volumes_df['SPY'].iloc[:idx+1]
        if len(sv) > 63 and sv.iloc[-63:].mean() > 0:
            feats['vol_ratio_5d'] = float(sv.iloc[-5:].mean() / sv.iloc[-63:].mean())
            feats['vol_ratio_1d'] = float(sv.iloc[-1] / sv.iloc[-63:].mean())
            feats['vol_spike'] = float(1 if sv.iloc[-1] > sv.iloc[-63:].quantile(0.95) else 0)

    # === CROSS-ASSET STRESS FEATURES ===
    # Credit spread proxy (HYG vs IEF)
    if 'HYG' in prices_df.columns and 'IEF' in prices_df.columns:
        hyg = prices_df['HYG'].iloc[:idx+1]
        ief = prices_df['IEF'].iloc[:idx+1]
        if len(hyg) > 63 and len(ief) > 63:
            spread = (hyg / ief).pct_change().dropna()
            if len(spread) > 63:
                feats['credit_spread_5d'] = float(spread.iloc[-5:].sum())
                feats['credit_spread_21d'] = float(spread.iloc[-21:].sum())
                feats['credit_zscore'] = float(
                    (spread.iloc[-5:].mean() - spread.iloc[-63:].mean()) / max(spread.iloc[-63:].std(), 1e-8)
                )

    # Flight to quality (TLT relative strength)
    if 'TLT' in prices_df.columns:
        tlt = prices_df['TLT'].iloc[:idx+1]
        if len(tlt) > 21:
            feats['tlt_ret_5d'] = float(tlt.iloc[-1] / tlt.iloc[-5] - 1)
            feats['tlt_ret_21d'] = float(tlt.iloc[-1] / tlt.iloc[-21] - 1)
            # SPY/TLT ratio change (risk-off indicator)
            ratio = spy / tlt
            if len(ratio) > 21:
                feats['spy_tlt_ratio_chg_21d'] = float(ratio.iloc[-1] / ratio.iloc[-21] - 1)

    # Gold as fear gauge
    if 'GLD' in prices_df.columns:
        gld = prices_df['GLD'].iloc[:idx+1]
        if len(gld) > 21:
            feats['gld_ret_21d'] = float(gld.iloc[-1] / gld.iloc[-21] - 1)
            feats['spy_gld_ratio_chg'] = float(
                (spy.iloc[-1] / gld.iloc[-1]) / (spy.iloc[-21] / gld.iloc[-21]) - 1
            ) if gld.iloc[-21] > 0 else 0

    # EM stress
    if 'EEM' in prices_df.columns:
        eem = prices_df['EEM'].iloc[:idx+1]
        if len(eem) > 21:
            feats['eem_ret_21d'] = float(eem.iloc[-1] / eem.iloc[-21] - 1)

    # Small cap stress (IWM underperformance = risk-off)
    if 'IWM' in prices_df.columns:
        iwm = prices_df['IWM'].iloc[:idx+1]
        if len(iwm) > 21:
            feats['iwm_ret_21d'] = float(iwm.iloc[-1] / iwm.iloc[-21] - 1)
            feats['iwm_spy_spread_21d'] = float(
                (iwm.iloc[-1] / iwm.iloc[-21] - 1) - (spy.iloc[-1] / spy.iloc[-21] - 1)
            )

    # === TECHNICAL EXHAUSTION FEATURES ===
    # RSI (oversold = potential capitulation)
    if len(spy_rets) > 14:
        gains = spy_rets.iloc[-14:].clip(lower=0).mean()
        losses = (-spy_rets.iloc[-14:].clip(upper=0)).mean()
        rs = gains / max(losses, 1e-8)
        feats['rsi_14'] = float(100 - 100 / (1 + rs))

    # Distance from 200-day MA (deep below = potential capitulation)
    if len(spy) > 200:
        feats['spy_vs_ma200'] = float(spy.iloc[-1] / spy.iloc[-200:].mean() - 1)
        feats['spy_vs_ma50'] = float(spy.iloc[-1] / spy.iloc[-50:].mean() - 1)

    # Consecutive down days
    recent_rets = spy_rets.iloc[-10:]
    feats['consec_down'] = float(max(0, sum(1 for r in recent_rets if r < 0)))
    feats['down_pct_10d'] = float((recent_rets < 0).mean())

    # Worst single-day return in recent window
    feats['worst_1d_10d'] = float(spy_rets.iloc[-10:].min())
    feats['worst_1d_21d'] = float(spy_rets.iloc[-21:].min())

    return feats


# --- Build dataset ---
print("\nBuilding walk-forward dataset...")

features_list = []
labels_list = []
dates_list = []
spy_fwd_rets = []

dates = prices.index
n = len(dates)

for i in range(TRAIN_DAYS, n - RECOVERY_HORIZON, ADVANCE_DAYS):
    feats = build_capitulation_features(prices, volumes, i)
    if feats is None:
        continue

    # Label: does SPY recover >5% from this point within RECOVERY_HORIZON?
    spy_now = prices['SPY'].iloc[i]
    spy_future = prices['SPY'].iloc[i+1:i+RECOVERY_HORIZON+1]
    max_future_ret = (spy_future.max() / spy_now - 1)

    # Binary: 1 = strong recovery ahead (>5% upside within 42 days)
    label = 1 if max_future_ret > RECOVERY_THRESHOLD else 0

    # Also store the actual forward return for analysis
    spy_fwd = prices['SPY'].iloc[min(i + ADVANCE_DAYS, n-1)] / spy_now - 1

    features_list.append(feats)
    labels_list.append(label)
    dates_list.append(dates[i])
    spy_fwd_rets.append(spy_fwd)

X = pd.DataFrame(features_list, index=dates_list)
y = np.array(labels_list)

print(f"Dataset: {len(X)} observations, {X.shape[1]} features")
print(f"Label distribution: {y.mean():.1%} recovery-ahead")
print(f"Date range: {X.index[0].date()} to {X.index[-1].date()}")

# --- Walk-forward backtest ---
print(f"\n{'WALK-FORWARD BACKTEST (Sliding Window)':=^70}")

# Walk-forward: ~50 weekly observations = ~1 year train
WF_TRAIN = 50  # ~1 year of weekly observations
WF_MIN_TRAIN = 30

results = []
all_probs = []

for test_start in range(WF_TRAIN, len(X)):
    train_start = max(0, test_start - WF_TRAIN)  # sliding window

    X_train = X.iloc[train_start:test_start]
    y_train = y[train_start:test_start]
    X_test = X.iloc[test_start:test_start+1]
    test_date = X.index[test_start]

    if len(X_train) < WF_MIN_TRAIN:
        continue

    # Clean data
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

    common_cols = X_train.columns.intersection(X_test.columns)
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    if len(common_cols) < 5:
        continue

    # Train LightGBM
    try:
        # Use class weights since recovery events might be imbalanced
        pos_rate = y_train.mean()
        scale = (1 - pos_rate) / max(pos_rate, 0.01)

        model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            min_child_samples=5,
            subsample=0.8,
            colsample_bytree=0.7,
            scale_pos_weight=scale,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1,
            n_jobs=1
        )
        model.fit(X_train.values, y_train)
        prob = model.predict_proba(X_test.values)[0, 1]
    except Exception:
        prob = 0.5

    # Decision logic:
    # High prob of recovery (>0.6) → go UPRO (leveraged long)
    # Medium prob (0.4-0.6) → hold SPY (neutral)
    # Low prob (<0.4) → hold SHY (defensive)
    test_idx = prices.index.get_indexer([test_date], method='nearest')[0]
    end_idx = min(test_idx + ADVANCE_DAYS, len(prices) - 1)

    spy_ret = prices['SPY'].iloc[end_idx] / prices['SPY'].iloc[test_idx] - 1

    if prob > 0.6 and has_upro and test_idx >= prices.index.get_indexer([upro_start], method='nearest')[0]:
        # Leveraged long for recovery
        upro_ret = prices['UPRO'].iloc[end_idx] / prices['UPRO'].iloc[test_idx] - 1
        port_ret = upro_ret
        position = 'UPRO'
    elif prob < 0.4:
        # Defensive
        shy_ret = prices['SHY'].iloc[end_idx] / prices['SHY'].iloc[test_idx] - 1
        port_ret = shy_ret
        position = 'SHY'
    else:
        # Neutral
        port_ret = spy_ret
        position = 'SPY'

    results.append({
        'date': test_date,
        'prob_recovery': prob,
        'position': position,
        'port_ret': port_ret,
        'spy_ret': spy_ret,
        'excess': port_ret - spy_ret,
    })
    all_probs.append(prob)

results_df = pd.DataFrame(results)

# Filter to period where UPRO exists for fair comparison
if has_upro:
    results_df = results_df[results_df['date'] >= upro_start].reset_index(drop=True)

print(f"\nTotal periods: {len(results_df)}")
pos_counts = results_df['position'].value_counts()
for pos in ['UPRO', 'SPY', 'SHY']:
    if pos in pos_counts:
        print(f"  {pos}: {pos_counts[pos]} ({pos_counts[pos]/len(results_df):.1%})")

period_returns = results_df['port_ret'].values
spy_returns = results_df['spy_ret'].values

# Annualize (5-day periods, ~50 per year)
periods_per_year = 252 / ADVANCE_DAYS
n_periods = len(period_returns)
n_years = n_periods / periods_per_year

# Strategy stats
strat_mean = np.mean(period_returns) * periods_per_year
strat_std = np.std(period_returns) * np.sqrt(periods_per_year)
strat_sharpe = strat_mean / strat_std if strat_std > 0 else 0

spy_mean = np.mean(spy_returns) * periods_per_year
spy_std = np.std(spy_returns) * np.sqrt(periods_per_year)
spy_sharpe = spy_mean / spy_std if spy_std > 0 else 0

# Cumulative
cum_strat = np.cumprod(1 + period_returns)
cum_spy = np.cumprod(1 + spy_returns)

max_dd_strat = np.min(cum_strat / np.maximum.accumulate(cum_strat) - 1)
max_dd_spy = np.min(cum_spy / np.maximum.accumulate(cum_spy) - 1)

# CAGR
total_ret_strat = cum_strat[-1] - 1
total_ret_spy = cum_spy[-1] - 1
cagr_strat = (1 + total_ret_strat) ** (1 / n_years) - 1 if n_years > 0 else 0
cagr_spy = (1 + total_ret_spy) ** (1 / n_years) - 1 if n_years > 0 else 0

# Win rate
excess = period_returns - spy_returns
wr_vs_spy = np.mean(excess > 0)

# Sortino
downside = period_returns[period_returns < 0]
downside_std = np.std(downside) * np.sqrt(periods_per_year) if len(downside) > 0 else strat_std
sortino = strat_mean / downside_std if downside_std > 0 else 0

# PF
gains = period_returns[period_returns > 0].sum()
losses = abs(period_returns[period_returns < 0].sum())
pf = gains / losses if losses > 0 else float('inf')

# Correlation
spy_corr = np.corrcoef(period_returns, spy_returns)[0, 1] if len(period_returns) > 2 else 1.0

# Analyze UPRO-period returns specifically
upro_mask = results_df['position'] == 'UPRO'
if upro_mask.sum() > 0:
    upro_periods = results_df[upro_mask]
    upro_avg_ret = upro_periods['port_ret'].mean()
    upro_wr = (upro_periods['port_ret'] > 0).mean()
    upro_avg_excess = upro_periods['excess'].mean()
    print(f"\n  UPRO periods analysis:")
    print(f"    Count: {upro_mask.sum()}")
    print(f"    Avg return: {upro_avg_ret:.2%} per week")
    print(f"    Win rate: {upro_wr:.1%}")
    print(f"    Avg excess vs SPY: {upro_avg_excess:.2%}")

print(f"\n{'STRATEGY RESULTS':=^70}")
print(f"  Sharpe:    {strat_sharpe:.3f}  (SPY: {spy_sharpe:.3f})")
print(f"  Sortino:   {sortino:.3f}")
print(f"  CAGR:      {cagr_strat:.1%}  (SPY: {cagr_spy:.1%})")
print(f"  MaxDD:     {max_dd_strat:.1%}  (SPY: {max_dd_spy:.1%})")
print(f"  WR vs SPY: {wr_vs_spy:.1%}")
print(f"  PF:        {pf:.2f}")
print(f"  SPY corr:  {spy_corr:.3f}")
print(f"  Periods:   {n_periods} ({n_years:.1f} years)")

# --- Feature importance ---
print(f"\n{'TOP 15 FEATURES':=^70}")
try:
    # Retrain on full data for feature importance
    X_full = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    model_full = lgb.LGBMClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        min_child_samples=5, verbose=-1, random_state=42
    )
    model_full.fit(X_full.values, y)
    importances = pd.Series(model_full.feature_importances_, index=X_full.columns)
    importances = importances.sort_values(ascending=False)
    for feat, imp in importances.head(15).items():
        print(f"  {feat:30s} {imp:6.0f}")
except Exception as e:
    print(f"  Feature importance failed: {e}")

# --- Adversarial Gate 1: Permutation Test ---
print(f"\n{'ADVERSARIAL GATE 1: PERMUTATION TEST':=^70}")
print(f"Running {N_PERMS} permutations (shuffling ML signals)...")

actual_sharpe = strat_sharpe
perm_sharpes = []

for perm_i in range(N_PERMS):
    # Shuffle the position assignments
    shuffled_positions = np.random.permutation(results_df['position'].values)

    perm_returns = []
    for j, row in enumerate(results_df.itertuples()):
        test_idx = prices.index.get_indexer([row.date], method='nearest')[0]
        end_idx = min(test_idx + ADVANCE_DAYS, len(prices) - 1)

        pos = shuffled_positions[j]
        if pos == 'UPRO' and has_upro:
            r = prices['UPRO'].iloc[end_idx] / prices['UPRO'].iloc[test_idx] - 1
        elif pos == 'SHY':
            r = prices['SHY'].iloc[end_idx] / prices['SHY'].iloc[test_idx] - 1
        else:
            r = prices['SPY'].iloc[end_idx] / prices['SPY'].iloc[test_idx] - 1
        perm_returns.append(r)

    pr = np.array(perm_returns)
    pm = np.mean(pr) * periods_per_year
    ps = np.std(pr) * np.sqrt(periods_per_year)
    perm_sharpes.append(pm / ps if ps > 0 else 0)

    if (perm_i + 1) % 50 == 0:
        print(f"  Completed {perm_i+1}/{N_PERMS} permutations")

perm_p = np.mean(np.array(perm_sharpes) >= actual_sharpe)
perm_mean = np.mean(perm_sharpes)
perm_std = np.std(perm_sharpes)
perm_z = (actual_sharpe - perm_mean) / perm_std if perm_std > 0 else 0

PERM_PASS = perm_p < 0.05
print(f"\n  Actual Sharpe:  {actual_sharpe:.3f}")
print(f"  Perm mean:      {perm_mean:.3f} +/- {perm_std:.3f}")
print(f"  p-value:        {perm_p:.3f}")
print(f"  z-score:        {perm_z:.2f}")
print(f"  GATE 1:         {'PASS' if PERM_PASS else 'FAIL'}")

# --- Adversarial Gate 2: Sub-Period Consistency ---
print(f"\n{'ADVERSARIAL GATE 2: SUB-PERIOD CONSISTENCY':=^70}")

n_blocks = 4
block_size = len(results_df) // n_blocks
block_sharpes = []

for b in range(n_blocks):
    start = b * block_size
    end = start + block_size if b < n_blocks - 1 else len(results_df)
    block_rets = period_returns[start:end]

    bm = np.mean(block_rets) * periods_per_year
    bs = np.std(block_rets) * np.sqrt(periods_per_year)
    block_sharpe = bm / bs if bs > 0 else 0
    block_sharpes.append(block_sharpe)

    date_range = f"{results_df.iloc[start]['date'].date()} to {results_df.iloc[min(end-1, len(results_df)-1)]['date'].date()}"
    print(f"  Block {b+1}: Sharpe {block_sharpe:.3f} ({date_range})")

sub_mean = np.mean(block_sharpes)
sub_cv = np.std(block_sharpes) / abs(sub_mean) if abs(sub_mean) > 0.01 else 99
SUB_PASS = sub_cv < 0.50
print(f"\n  Block Sharpes:  {[f'{s:.3f}' for s in block_sharpes]}")
print(f"  CV:             {sub_cv:.3f}")
print(f"  GATE 2:         {'PASS' if SUB_PASS else 'FAIL'}")

# --- Adversarial Gate 3: Outlier Robustness ---
print(f"\n{'ADVERSARIAL GATE 3: OUTLIER ROBUSTNESS':=^70}")

n_remove = max(1, int(len(period_returns) * 0.05))
sorted_rets = np.sort(period_returns)
trimmed = sorted_rets[n_remove:-n_remove]

tm = np.mean(trimmed) * periods_per_year
ts = np.std(trimmed) * np.sqrt(periods_per_year)
trimmed_sharpe = tm / ts if ts > 0 else 0

robustness_ratio = trimmed_sharpe / actual_sharpe if actual_sharpe != 0 else 0

OUTLIER_PASS = robustness_ratio > 0.50
print(f"  Full Sharpe:     {actual_sharpe:.3f}")
print(f"  Trimmed Sharpe:  {trimmed_sharpe:.3f}")
print(f"  Robustness ratio:{robustness_ratio:.3f}")
print(f"  GATE 3:          {'PASS' if OUTLIER_PASS else 'FAIL'}")

# --- Adversarial Gate 4: R1 Regime-Agnostic ---
print(f"\n{'ADVERSARIAL GATE 4: R1 REGIME-AGNOSTIC':=^70}")

green_mask = spy_returns > 0
red_mask = spy_returns <= 0

green_rets = period_returns[green_mask]
red_rets = period_returns[red_mask]

if len(green_rets) > 5 and len(red_rets) > 5:
    green_sharpe = (np.mean(green_rets) * periods_per_year) / (np.std(green_rets) * np.sqrt(periods_per_year)) if np.std(green_rets) > 0 else 0
    red_sharpe = (np.mean(red_rets) * periods_per_year) / (np.std(red_rets) * np.sqrt(periods_per_year)) if np.std(red_rets) > 0 else 0

    regime_gap = abs(green_sharpe - red_sharpe) / max(abs(green_sharpe), abs(red_sharpe), 0.01)
    R1_PASS = regime_gap < 0.50

    print(f"  Green periods:   {green_mask.sum()} - Sharpe {green_sharpe:.3f}")
    print(f"  Red periods:     {red_mask.sum()} - Sharpe {red_sharpe:.3f}")
    print(f"  Regime gap:      {regime_gap:.3f}")
    print(f"  GATE 4:          {'PASS' if R1_PASS else 'FAIL'}")
else:
    R1_PASS = False
    regime_gap = 99
    green_sharpe = red_sharpe = 0
    print("  Insufficient data for regime analysis")
    print(f"  GATE 4:          FAIL")

# --- Summary ---
gates_passed = sum([PERM_PASS, SUB_PASS, OUTLIER_PASS, R1_PASS])

print(f"\n{'FINAL SUMMARY':=^70}")
print(f"  Strategy:        ML Capitulation Recovery")
print(f"  Sharpe:          {strat_sharpe:.3f} (SPY: {spy_sharpe:.3f})")
print(f"  Sortino:         {sortino:.3f}")
print(f"  CAGR:            {cagr_strat:.1%} (SPY: {cagr_spy:.1%})")
print(f"  MaxDD:           {max_dd_strat:.1%} (SPY: {max_dd_spy:.1%})")
print(f"  WR vs SPY:       {wr_vs_spy:.1%}")
print(f"  PF:              {pf:.2f}")
print(f"  SPY corr:        {spy_corr:.3f}")
print(f"  Periods:         {n_periods} ({n_years:.1f} years)")
print(f"")
print(f"  ADVERSARIAL GATES: {gates_passed}/4")
print(f"    Gate 1 (Perm):    {'PASS' if PERM_PASS else 'FAIL'} (p={perm_p:.3f})")
print(f"    Gate 2 (SubP):    {'PASS' if SUB_PASS else 'FAIL'} (CV={sub_cv:.3f})")
print(f"    Gate 3 (Outlier): {'PASS' if OUTLIER_PASS else 'FAIL'} (ratio={robustness_ratio:.3f})")
print(f"    Gate 4 (R1):      {'PASS' if R1_PASS else 'FAIL'} (gap={regime_gap:.3f})")

verdict = "VALIDATED" if gates_passed == 4 else "PARTIAL" if gates_passed >= 3 else "REJECTED"
print(f"\n  VERDICT: {verdict}")

# --- Save results ---
output = {
    'strategy': 'ML Capitulation Recovery',
    'timestamp': str(dt.datetime.now()),
    'performance': {
        'sharpe': round(strat_sharpe, 3),
        'sortino': round(sortino, 3),
        'cagr': round(cagr_strat, 4),
        'max_dd': round(max_dd_strat, 4),
        'wr_vs_spy': round(wr_vs_spy, 4),
        'pf': round(pf, 2),
        'spy_corr': round(spy_corr, 3),
        'n_periods': n_periods,
        'n_years': round(n_years, 1),
    },
    'position_mix': {pos: int(cnt) for pos, cnt in pos_counts.items()},
    'benchmark': {
        'spy_sharpe': round(spy_sharpe, 3),
        'spy_cagr': round(cagr_spy, 4),
        'spy_max_dd': round(max_dd_spy, 4),
    },
    'adversarial': {
        'gates_passed': gates_passed,
        'gate1_perm': {'pass': PERM_PASS, 'p_value': round(perm_p, 3), 'z_score': round(perm_z, 2)},
        'gate2_subperiod': {'pass': SUB_PASS, 'cv': round(sub_cv, 3), 'block_sharpes': [round(s, 3) for s in block_sharpes]},
        'gate3_outlier': {'pass': OUTLIER_PASS, 'robustness_ratio': round(robustness_ratio, 3)},
        'gate4_r1': {'pass': R1_PASS, 'regime_gap': round(regime_gap, 3), 'green_sharpe': round(green_sharpe, 3), 'red_sharpe': round(red_sharpe, 3)},
    },
    'verdict': verdict,
}

with open(OUTPUT_DIR / 'results.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)

results_df.to_csv(OUTPUT_DIR / 'trades.csv', index=False)

print(f"\nResults saved to {OUTPUT_DIR}")
print("DONE.")
