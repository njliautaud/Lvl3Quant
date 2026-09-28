#!/usr/bin/env python3
"""
ML Sector Dispersion Timing Strategy
======================================
Novel concept: Cross-sectional return dispersion among S&P sectors as a
meta-signal for WHEN active sector rotation adds alpha vs passive indexing.

Hypothesis:
- High dispersion = sectors diverging = rotation opportunities
- Low dispersion = everything correlated = just hold SPY
- ML learns which dispersion regimes predict profitable active rotation

This is DIFFERENT from prior sector rotation (which always rotated) and
from cross-sectional momentum (which was pure beta). Here we use dispersion
as a gating signal — only rotate when dispersion is elevated.

Universe: 11 SPDR sectors + SPY (passive fallback)
Signal: LightGBM predicts whether active top-K rotation will outperform SPY
        over the next 21 trading days, using dispersion + momentum features
Action: High-confidence active → hold top-K sectors by ML score
        Low-confidence → hold SPY (passive)

HC compliance:
  - HC #0: Sliding walk-forward (252d train, 21d advance, oldest-drop)
  - HC #713: Fixed $100K capital, no DCA
  - HC #428 R1: Regime-agnostic validation (40+ OOT days, gap < 0.50)
  - Full 4-gate adversarial (perm, sub-period, outlier, R1)
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
SECTORS = ['XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY']
BENCHMARK = 'SPY'
SAFETY = 'SHY'  # risk-off asset
TRAIN_DAYS = 252
ADVANCE_DAYS = 21  # monthly rebalance
LABEL_HORIZON = 21  # 21-day forward return
START_YEAR = 2007
INITIAL_CAPITAL = 100_000
TOP_K = 3  # hold top-K sectors when active
N_PERMS = 200
OUTPUT_DIR = Path('/home/jupiter/Lvl3Quant/output/ml_dispersion_timing')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("ML SECTOR DISPERSION TIMING STRATEGY")
print("=" * 70)

# --- Download data ---
all_tickers = SECTORS + [BENCHMARK, SAFETY]
print(f"\nDownloading {len(all_tickers)} tickers...")

data = {}
for t in all_tickers + ['^VIX']:
    try:
        df = yf.download(t, start=f'{START_YEAR-1}-01-01', end='2026-07-21',
                         progress=False, auto_adjust=True)
        if len(df) > 100:
            if isinstance(df.columns, pd.MultiIndex):
                close = df[('Close', t)].copy()
            else:
                close = df['Close'].copy()
            clean_name = t.replace('^', '')
            close.name = clean_name
            data[clean_name] = close
            print(f"  {t}: {len(df)} days")
        else:
            print(f"  {t}: SKIP (only {len(df)} days)")
    except Exception as e:
        print(f"  {t}: FAILED ({e})")

prices = pd.DataFrame(data)
prices = prices.dropna(how='all').ffill().dropna()
print(f"\nAligned: {len(prices)} days, {prices.shape[1]} assets")
print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

# Filter sectors with enough data
valid_sectors = [t for t in SECTORS if t in prices.columns and prices[t].notna().sum() >= TRAIN_DAYS + 252]
print(f"Valid sectors ({len(valid_sectors)}): {valid_sectors}")

if len(valid_sectors) < 5:
    print("ERROR: Need at least 5 valid sectors")
    exit(1)

# --- Feature engineering ---
def compute_dispersion_features(prices_df, sectors, idx):
    """Compute cross-sectional dispersion and related features at time idx."""
    feats = {}

    # Need enough history
    if idx < 252:
        return None

    # Cross-sectional return dispersion at multiple horizons
    for w in [5, 10, 21, 42, 63]:
        if idx >= w:
            sector_rets = []
            for s in sectors:
                p = prices_df[s].iloc[idx-w:idx+1]
                if len(p) > 1 and p.iloc[0] > 0:
                    sector_rets.append(p.iloc[-1] / p.iloc[0] - 1)
            if len(sector_rets) >= 5:
                sr = np.array(sector_rets)
                feats[f'disp_{w}d'] = np.std(sr)  # cross-sectional std
                feats[f'disp_range_{w}d'] = np.max(sr) - np.min(sr)  # max spread
                feats[f'disp_iqr_{w}d'] = np.percentile(sr, 75) - np.percentile(sr, 25)
                feats[f'disp_skew_{w}d'] = float(pd.Series(sr).skew())
                feats[f'disp_kurt_{w}d'] = float(pd.Series(sr).kurtosis())

    # Dispersion trend (is dispersion increasing or decreasing?)
    if idx >= 63:
        disp_series = []
        for j in range(63):
            lookback = 21
            if idx - j >= lookback:
                sr = []
                for s in sectors:
                    p = prices_df[s].iloc[idx-j-lookback:idx-j+1]
                    if len(p) > 1 and p.iloc[0] > 0:
                        sr.append(p.iloc[-1] / p.iloc[0] - 1)
                if len(sr) >= 5:
                    disp_series.append(np.std(sr))
        if len(disp_series) >= 21:
            ds = np.array(disp_series)
            feats['disp_trend_21d'] = ds[:21].mean() - ds[21:42].mean() if len(ds) >= 42 else 0
            feats['disp_ma_ratio'] = ds[:10].mean() / max(ds[10:30].mean(), 1e-8)

    # Market-level features
    spy = prices_df[BENCHMARK].iloc[:idx+1]
    spy_rets = spy.pct_change().dropna()

    for w in [5, 10, 21, 42, 63, 126, 252]:
        if len(spy_rets) > w:
            feats[f'spy_ret_{w}d'] = float(spy.iloc[-1] / spy.iloc[-w] - 1) if spy.iloc[-w] > 0 else 0
            feats[f'spy_vol_{w}d'] = float(spy_rets.iloc[-w:].std() * np.sqrt(252))

    # SPY vs MA
    for w in [50, 100, 200]:
        if len(spy) > w:
            feats[f'spy_vs_ma{w}'] = float(spy.iloc[-1] / spy.iloc[-w:].mean() - 1)

    # VIX features
    if 'VIX' in prices_df.columns:
        vix = prices_df['VIX'].iloc[:idx+1]
        if len(vix) > 63:
            feats['vix_level'] = float(vix.iloc[-1])
            feats['vix_ma21'] = float(vix.iloc[-21:].mean())
            feats['vix_pctile_63'] = float((vix.iloc[-63:] < vix.iloc[-1]).mean())
            feats['vix_pctile_252'] = float((vix.iloc[-252:] < vix.iloc[-1]).mean()) if len(vix) > 252 else 0.5
            feats['vix_5d_chg'] = float(vix.iloc[-1] / vix.iloc[-5] - 1) if vix.iloc[-5] > 0 else 0

    # Sector momentum features (which sectors are leading?)
    for w in [21, 63]:
        if idx >= w:
            sector_rets_dict = {}
            for s in sectors:
                p = prices_df[s].iloc[idx-w:idx+1]
                if len(p) > 1 and p.iloc[0] > 0:
                    sector_rets_dict[s] = p.iloc[-1] / p.iloc[0] - 1
            if len(sector_rets_dict) >= 5:
                sorted_rets = sorted(sector_rets_dict.values())
                feats[f'top3_avg_{w}d'] = np.mean(sorted_rets[-3:])
                feats[f'bot3_avg_{w}d'] = np.mean(sorted_rets[:3])
                feats[f'ls_spread_{w}d'] = np.mean(sorted_rets[-3:]) - np.mean(sorted_rets[:3])

    # Correlation features (average pairwise correlation among sectors)
    if idx >= 63:
        rets_mat = pd.DataFrame()
        for s in sectors:
            if s in prices_df.columns:
                r = prices_df[s].iloc[idx-62:idx+1].pct_change().dropna()
                if len(r) >= 60:
                    rets_mat[s] = r.values[:60]
        if rets_mat.shape[1] >= 5:
            corr = rets_mat.corr()
            # Average pairwise correlation (exclude diagonal)
            mask = np.ones_like(corr, dtype=bool)
            np.fill_diagonal(mask, False)
            feats['avg_corr_63d'] = float(corr.values[mask].mean())
            feats['min_corr_63d'] = float(corr.values[mask].min())
            feats['max_corr_63d'] = float(corr.values[mask].max())

    return feats


def compute_sector_scores(prices_df, sectors, idx):
    """Score each sector for the active rotation portfolio."""
    scores = {}
    for s in sectors:
        p = prices_df[s].iloc[:idx+1]
        if len(p) < 252:
            continue
        rets = p.pct_change().dropna()

        # Composite: momentum + risk-adjusted
        ret_21 = p.iloc[-1] / p.iloc[-21] - 1 if p.iloc[-21] > 0 else 0
        ret_63 = p.iloc[-1] / p.iloc[-63] - 1 if p.iloc[-63] > 0 else 0
        vol = rets.iloc[-63:].std() * np.sqrt(252) if len(rets) >= 63 else 0.2

        # Risk-adjusted momentum (Sharpe-like)
        scores[s] = (0.5 * ret_21 + 0.5 * ret_63) / max(vol, 0.05)

    return scores


# --- Build dataset ---
print("\nBuilding walk-forward dataset...")

# Compute features and labels for each rebalance point
dates = prices.index
n = len(dates)

features_list = []
labels_list = []
dates_list = []

for i in range(TRAIN_DAYS, n - LABEL_HORIZON, ADVANCE_DAYS):
    feats = compute_dispersion_features(prices, valid_sectors, i)
    if feats is None:
        continue

    # Label: did active top-K rotation beat SPY over next 21 days?
    spy_fwd = prices[BENCHMARK].iloc[i+LABEL_HORIZON] / prices[BENCHMARK].iloc[i] - 1

    # Compute what top-K sectors by current momentum would have returned
    scores = compute_sector_scores(prices, valid_sectors, i)
    if len(scores) < TOP_K:
        continue

    top_k = sorted(scores, key=scores.get, reverse=True)[:TOP_K]
    top_k_ret = np.mean([
        prices[s].iloc[i+LABEL_HORIZON] / prices[s].iloc[i] - 1
        for s in top_k if prices[s].iloc[i] > 0
    ])

    # Binary label: 1 if active rotation beats SPY by >0.5% (transaction cost buffer)
    label = 1 if top_k_ret > spy_fwd + 0.005 else 0

    features_list.append(feats)
    labels_list.append(label)
    dates_list.append(dates[i])

X = pd.DataFrame(features_list, index=dates_list)
y = np.array(labels_list)

print(f"Dataset: {len(X)} observations, {X.shape[1]} features")
print(f"Label distribution: {y.mean():.1%} active-beats-passive")

# --- Walk-forward backtest ---
print("\n" + "=" * 70)
print("WALK-FORWARD BACKTEST (Sliding Window)")
print("=" * 70)

# Walk-forward: 12 months train, advance 1 month
WF_TRAIN = 12  # 12 observation periods (~12 months)
WF_MIN_TRAIN = 8

results = []
all_predictions = []

for test_start in range(WF_TRAIN, len(X)):
    train_start = max(0, test_start - WF_TRAIN)  # sliding window

    X_train = X.iloc[train_start:test_start]
    y_train = y[train_start:test_start]
    X_test = X.iloc[test_start:test_start+1]
    test_date = X.index[test_start]

    if len(X_train) < WF_MIN_TRAIN:
        continue

    # Handle any NaN/inf
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Ensure columns match
    common_cols = X_train.columns.intersection(X_test.columns)
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    if len(common_cols) < 5:
        continue

    # Train LightGBM
    try:
        model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            min_child_samples=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbose=-1,
            n_jobs=1
        )
        model.fit(X_train.values, y_train)
        prob = model.predict_proba(X_test.values)[0, 1]
    except Exception:
        prob = 0.5

    # Decision: active rotation if ML says >0.5, else passive SPY
    active = prob > 0.5

    # Compute actual return for this period
    test_idx = prices.index.get_indexer([test_date], method='nearest')[0]
    end_idx = min(test_idx + LABEL_HORIZON, len(prices) - 1)

    spy_ret = prices[BENCHMARK].iloc[end_idx] / prices[BENCHMARK].iloc[test_idx] - 1

    if active:
        # Active: hold top-K sectors
        scores = compute_sector_scores(prices, valid_sectors, test_idx)
        if len(scores) >= TOP_K:
            top_k = sorted(scores, key=scores.get, reverse=True)[:TOP_K]
            port_ret = np.mean([
                prices[s].iloc[end_idx] / prices[s].iloc[test_idx] - 1
                for s in top_k if prices[s].iloc[test_idx] > 0
            ])
        else:
            port_ret = spy_ret
            active = False
    else:
        # Passive: hold SPY
        port_ret = spy_ret

    results.append({
        'date': test_date,
        'prob_active': prob,
        'active': active,
        'port_ret': port_ret,
        'spy_ret': spy_ret,
        'excess': port_ret - spy_ret,
    })
    all_predictions.append(prob)

results_df = pd.DataFrame(results)
print(f"\nTotal periods: {len(results_df)}")
print(f"Active periods: {results_df['active'].sum()} ({results_df['active'].mean():.1%})")
print(f"Passive periods: {(~results_df['active']).sum()} ({(~results_df['active']).mean():.1%})")

# --- Performance calculation (fixed capital) ---
# Each period's return is on fixed $100K capital
period_returns = results_df['port_ret'].values
spy_returns = results_df['spy_ret'].values

# Annualize (21-day periods, ~12 per year)
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

# Cumulative returns for drawdown
cum_strat = np.cumprod(1 + period_returns)
cum_spy = np.cumprod(1 + spy_returns)

max_dd_strat = np.min(cum_strat / np.maximum.accumulate(cum_strat) - 1)
max_dd_spy = np.min(cum_spy / np.maximum.accumulate(cum_spy) - 1)

# CAGR
total_ret_strat = cum_strat[-1] - 1
total_ret_spy = cum_spy[-1] - 1
cagr_strat = (1 + total_ret_strat) ** (1 / n_years) - 1 if n_years > 0 else 0
cagr_spy = (1 + total_ret_spy) ** (1 / n_years) - 1 if n_years > 0 else 0

# Win rate (vs SPY)
excess = period_returns - spy_returns
wr_vs_spy = np.mean(excess > 0)

# Sortino
downside = period_returns[period_returns < 0]
downside_std = np.std(downside) * np.sqrt(periods_per_year) if len(downside) > 0 else strat_std
sortino = strat_mean / downside_std if downside_std > 0 else 0

# Profit factor
gains = period_returns[period_returns > 0].sum()
losses = abs(period_returns[period_returns < 0].sum())
pf = gains / losses if losses > 0 else float('inf')

# SPY correlation
spy_corr = np.corrcoef(period_returns, spy_returns)[0, 1]

print(f"\n{'STRATEGY RESULTS':=^70}")
print(f"  Sharpe:    {strat_sharpe:.3f}  (SPY: {spy_sharpe:.3f})")
print(f"  Sortino:   {sortino:.3f}")
print(f"  CAGR:      {cagr_strat:.1%}  (SPY: {cagr_spy:.1%})")
print(f"  MaxDD:     {max_dd_strat:.1%}  (SPY: {max_dd_spy:.1%})")
print(f"  WR vs SPY: {wr_vs_spy:.1%}")
print(f"  PF:        {pf:.2f}")
print(f"  SPY corr:  {spy_corr:.3f}")
print(f"  Periods:   {n_periods} ({n_years:.1f} years)")

# --- Adversarial Gate 1: Permutation Test ---
print(f"\n{'ADVERSARIAL GATE 1: PERMUTATION TEST':=^70}")
print(f"Running {N_PERMS} permutations (shuffling ML signals)...")

actual_sharpe = strat_sharpe
perm_sharpes = []

for perm_i in range(N_PERMS):
    # Shuffle the active/passive decisions
    shuffled_active = np.random.permutation(results_df['active'].values)

    perm_returns = []
    for j, row in enumerate(results_df.itertuples()):
        test_idx = prices.index.get_indexer([row.date], method='nearest')[0]
        end_idx = min(test_idx + LABEL_HORIZON, len(prices) - 1)
        spy_r = prices[BENCHMARK].iloc[end_idx] / prices[BENCHMARK].iloc[test_idx] - 1

        if shuffled_active[j]:
            scores = compute_sector_scores(prices, valid_sectors, test_idx)
            if len(scores) >= TOP_K:
                top_k = sorted(scores, key=scores.get, reverse=True)[:TOP_K]
                port_r = np.mean([
                    prices[s].iloc[end_idx] / prices[s].iloc[test_idx] - 1
                    for s in top_k if prices[s].iloc[test_idx] > 0
                ])
            else:
                port_r = spy_r
        else:
            port_r = spy_r
        perm_returns.append(port_r)

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

sub_cv = np.std(block_sharpes) / abs(np.mean(block_sharpes)) if abs(np.mean(block_sharpes)) > 0.01 else 99
SUB_PASS = sub_cv < 0.50
print(f"\n  Block Sharpes:  {[f'{s:.3f}' for s in block_sharpes]}")
print(f"  CV:             {sub_cv:.3f}")
print(f"  GATE 2:         {'PASS' if SUB_PASS else 'FAIL'}")

# --- Adversarial Gate 3: Outlier Robustness ---
print(f"\n{'ADVERSARIAL GATE 3: OUTLIER ROBUSTNESS':=^70}")

# Remove top/bottom 5% of returns
n_remove = max(1, int(len(period_returns) * 0.05))
sorted_rets = np.sort(period_returns)
trimmed = sorted_rets[n_remove:-n_remove]

tm = np.mean(trimmed) * periods_per_year
ts = np.std(trimmed) * np.sqrt(periods_per_year)
trimmed_sharpe = tm / ts if ts > 0 else 0

degradation = 1 - (trimmed_sharpe / actual_sharpe) if actual_sharpe != 0 else 0
robustness_ratio = trimmed_sharpe / actual_sharpe if actual_sharpe != 0 else 0

OUTLIER_PASS = robustness_ratio > 0.50
print(f"  Full Sharpe:     {actual_sharpe:.3f}")
print(f"  Trimmed Sharpe:  {trimmed_sharpe:.3f}")
print(f"  Degradation:     {degradation:.1%}")
print(f"  Robustness ratio:{robustness_ratio:.3f}")
print(f"  GATE 3:          {'PASS' if OUTLIER_PASS else 'FAIL'}")

# --- Adversarial Gate 4: R1 Regime-Agnostic (HC #428) ---
print(f"\n{'ADVERSARIAL GATE 4: R1 REGIME-AGNOSTIC':=^70}")

# Classify each period as green/red based on SPY close-to-close
green_mask = spy_returns > 0
red_mask = spy_returns <= 0

green_rets = period_returns[green_mask]
red_rets = period_returns[red_mask]

if len(green_rets) > 2 and len(red_rets) > 2:
    green_sharpe = (np.mean(green_rets) * periods_per_year) / (np.std(green_rets) * np.sqrt(periods_per_year)) if np.std(green_rets) > 0 else 0
    red_sharpe = (np.mean(red_rets) * periods_per_year) / (np.std(red_rets) * np.sqrt(periods_per_year)) if np.std(red_rets) > 0 else 0

    regime_gap = abs(green_sharpe - red_sharpe) / max(abs(green_sharpe), abs(red_sharpe), 0.01)
    R1_PASS = regime_gap < 0.50

    print(f"  Green periods:   {green_mask.sum()} — Sharpe {green_sharpe:.3f}")
    print(f"  Red periods:     {red_mask.sum()} — Sharpe {red_sharpe:.3f}")
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
print(f"  Strategy:        ML Sector Dispersion Timing")
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
    'strategy': 'ML Sector Dispersion Timing',
    'timestamp': str(dt.datetime.now()),
    'params': {
        'sectors': valid_sectors,
        'train_days': TRAIN_DAYS,
        'advance_days': ADVANCE_DAYS,
        'label_horizon': LABEL_HORIZON,
        'top_k': TOP_K,
        'initial_capital': INITIAL_CAPITAL,
    },
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
        'active_pct': round(results_df['active'].mean(), 3),
    },
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
