#!/usr/bin/env python3
"""
Overnight Gap Prediction for Sector ETFs
=========================================
Hypothesis: Close-to-close moves in macro assets (VIX, TLT, UUP, GLD)
plus intraday momentum and technical signals predict next-day open gaps
for sector ETFs.

Strategy: Each day, predict gap direction for 11 sector ETFs.
Go long the top 3 with highest predicted gap-up probability.
Measure actual gap return (open/prev_close - 1).

Rules:
- Sliding 60-day train window (never expanding)
- Cost: 0.10% round-trip (shares/limit orders)
- All available OOT days tested
- LightGBM classifier
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import json
import sys

# ============================================================
# CONFIG
# ============================================================
SECTOR_ETFS = ['XLK', 'XLF', 'XLE', 'XLU', 'XLP', 'XLY', 'XLV', 'XLI', 'XLB', 'XLC', 'XLRE']
MACRO_TICKERS = ['SPY', '^VIX', 'TLT', 'UUP', 'GLD']
ALL_TICKERS = SECTOR_ETFS + MACRO_TICKERS

TRAIN_WINDOW = 60  # sliding window days
TOP_N = 3          # go long top N predicted gap-up sectors
COST_PCT = 0.0010  # 10 bps round-trip cost
START_DATE = '2019-06-01'  # extra for warmup
END_DATE = '2026-08-20'

# ============================================================
# DATA DOWNLOAD
# ============================================================
print("=" * 70)
print("OVERNIGHT GAP PREDICTION — SECTOR ETFs")
print("=" * 70)
print(f"\nDownloading data for {len(ALL_TICKERS)} tickers...")

data = {}
for ticker in ALL_TICKERS:
    try:
        df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if len(df) > 100:
            data[ticker] = df
            print(f"  {ticker}: {len(df)} days ({df.index[0].date()} to {df.index[-1].date()})")
        else:
            print(f"  {ticker}: SKIPPED (only {len(df)} days)")
    except Exception as e:
        print(f"  {ticker}: FAILED ({e})")

# Check we have what we need
missing = [t for t in ALL_TICKERS if t not in data]
if missing:
    print(f"\nWARNING: Missing tickers: {missing}")
    # Remove missing sector ETFs
    active_sectors = [s for s in SECTOR_ETFS if s in data]
else:
    active_sectors = SECTOR_ETFS

print(f"\nActive sector ETFs: {len(active_sectors)}")

# ============================================================
# FEATURE ENGINEERING
# ============================================================
print("\nEngineering features...")

# Build aligned daily dataframe
# Use SPY dates as reference (most liquid, fewest gaps)
ref_dates = data['SPY'].index

# Close prices for all tickers
closes = pd.DataFrame({t: data[t]['Close'].reindex(ref_dates) for t in data.keys()})
opens = pd.DataFrame({t: data[t]['Open'].reindex(ref_dates) for t in data.keys()})
highs = pd.DataFrame({t: data[t]['High'].reindex(ref_dates) for t in data.keys()})
lows = pd.DataFrame({t: data[t]['Low'].reindex(ref_dates) for t in data.keys()})

# Forward fill any gaps (holidays differ slightly)
closes = closes.ffill()
opens = opens.ffill()

# ---------- Features computed at close of day t, predicting gap on day t+1 ----------

features_per_sector = {}

for sector in active_sectors:
    feats = pd.DataFrame(index=ref_dates)

    # --- Macro features (same for all sectors) ---
    # VIX close-to-close change
    if '^VIX' in closes.columns:
        feats['vix_chg'] = closes['^VIX'].pct_change()
        feats['vix_level'] = closes['^VIX']
        feats['vix_5d_chg'] = closes['^VIX'].pct_change(5)

    # TLT close-to-close (bond move → rate sensitivity)
    if 'TLT' in closes.columns:
        feats['tlt_chg'] = closes['TLT'].pct_change()
        feats['tlt_5d_chg'] = closes['TLT'].pct_change(5)

    # UUP close-to-close (dollar strength)
    if 'UUP' in closes.columns:
        feats['uup_chg'] = closes['UUP'].pct_change()
        feats['uup_5d_chg'] = closes['UUP'].pct_change(5)

    # GLD close-to-close (risk-off flow)
    if 'GLD' in closes.columns:
        feats['gld_chg'] = closes['GLD'].pct_change()
        feats['gld_5d_chg'] = closes['GLD'].pct_change(5)

    # SPY daily return (proxy for last-hour momentum — we only have daily data)
    if 'SPY' in closes.columns:
        feats['spy_ret'] = closes['SPY'].pct_change()
        feats['spy_5d_ret'] = closes['SPY'].pct_change(5)
        # SPY intraday range as volatility proxy
        if 'SPY' in highs.columns and 'SPY' in lows.columns:
            feats['spy_intraday_range'] = (highs['SPY'] - lows['SPY']) / closes['SPY']

    # --- Sector-specific features ---
    sec_close = closes[sector]

    # RSI(5) at close
    delta = sec_close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(5).mean()
    avg_loss = loss.rolling(5).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    feats['rsi_5'] = 100 - (100 / (1 + rs))

    # RSI(14) at close
    avg_gain14 = gain.rolling(14).mean()
    avg_loss14 = loss.rolling(14).mean()
    rs14 = avg_gain14 / avg_loss14.replace(0, np.nan)
    feats['rsi_14'] = 100 - (100 / (1 + rs14))

    # Distance from 20-day SMA
    sma20 = sec_close.rolling(20).mean()
    feats['dist_sma20'] = (sec_close - sma20) / sma20

    # Distance from 50-day SMA
    sma50 = sec_close.rolling(50).mean()
    feats['dist_sma50'] = (sec_close - sma50) / sma50

    # Sector daily return
    feats['sector_ret'] = sec_close.pct_change()
    feats['sector_5d_ret'] = sec_close.pct_change(5)

    # Sector vs SPY relative strength (1d and 5d)
    if 'SPY' in closes.columns:
        feats['rel_str_1d'] = sec_close.pct_change() - closes['SPY'].pct_change()
        feats['rel_str_5d'] = sec_close.pct_change(5) - closes['SPY'].pct_change(5)

    # Sector realized volatility (5d)
    feats['rvol_5d'] = sec_close.pct_change().rolling(5).std()

    # Previous gap (today's gap as feature for tomorrow's gap — momentum/mean-reversion)
    feats['prev_gap'] = opens[sector] / sec_close.shift(1) - 1

    # Day of week
    feats['dow'] = pd.Series(ref_dates.dayofweek, index=ref_dates)

    # --- Target: next-day gap direction ---
    # Gap = open[t+1] / close[t] - 1
    gap_return = opens[sector].shift(-1) / sec_close - 1
    feats['gap_return'] = gap_return
    feats['gap_direction'] = (gap_return > 0).astype(int)

    features_per_sector[sector] = feats

# Identify common valid dates (after warmup)
sample_feats = features_per_sector[active_sectors[0]]
feature_cols = [c for c in sample_feats.columns if c not in ['gap_return', 'gap_direction']]
print(f"Features per sector: {len(feature_cols)}")
print(f"Feature list: {feature_cols}")

# ============================================================
# SLIDING WINDOW BACKTEST
# ============================================================
print(f"\nRunning sliding-window backtest (train={TRAIN_WINDOW}d, top-{TOP_N} sectors)...")

# Find first valid date (need 60 warmup for features + 60 for first train window)
warmup = 80  # enough for 50d SMA + some buffer
start_idx = warmup + TRAIN_WINDOW

# Results storage
daily_results = []

# Get dates from reference
all_dates = ref_dates.tolist()
n_dates = len(all_dates)

print(f"Total dates: {n_dates}, OOT starts at index {start_idx}")
print(f"OOT period: {all_dates[start_idx].date()} to {all_dates[-2].date()}")
print(f"Expected OOT days: ~{n_dates - start_idx - 1}")

n_oot = 0
n_skipped = 0

for t in range(start_idx, n_dates - 1):  # -1 because we need next-day open
    test_date = all_dates[t]
    train_start = t - TRAIN_WINDOW
    train_end = t  # exclusive

    # Build training data across all sectors (pooled)
    X_train_list = []
    y_train_list = []

    for sector in active_sectors:
        sf = features_per_sector[sector]
        train_slice = sf.iloc[train_start:train_end]

        # Drop rows with NaN in features or target
        valid = train_slice[feature_cols + ['gap_direction']].dropna()
        if len(valid) < 10:
            continue

        X_train_list.append(valid[feature_cols].values)
        y_train_list.append(valid['gap_direction'].values)

    if not X_train_list:
        n_skipped += 1
        continue

    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)

    # Check class balance
    pos_rate = y_train.mean()
    if pos_rate < 0.05 or pos_rate > 0.95:
        n_skipped += 1
        continue

    # Train LightGBM
    try:
        model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            verbose=-1,
            random_state=42,
        )
        model.fit(X_train, y_train)
    except Exception as e:
        n_skipped += 1
        continue

    # Predict for each sector on test date
    sector_predictions = {}
    sector_gap_returns = {}

    for sector in active_sectors:
        sf = features_per_sector[sector]
        test_row = sf.iloc[t:t+1]

        # Check for valid features and target
        feat_vals = test_row[feature_cols].values
        gap_ret = test_row['gap_return'].values[0]

        if np.any(np.isnan(feat_vals)) or np.isnan(gap_ret):
            continue

        prob = model.predict_proba(feat_vals)[0][1]  # prob of gap up
        sector_predictions[sector] = prob
        sector_gap_returns[sector] = gap_ret

    if len(sector_predictions) < TOP_N:
        n_skipped += 1
        continue

    # Rank sectors by predicted gap-up probability, go long top N
    ranked = sorted(sector_predictions.items(), key=lambda x: x[1], reverse=True)
    top_sectors = [s for s, _ in ranked[:TOP_N]]
    bottom_sectors = [s for s, _ in ranked[-TOP_N:]]

    # Long-only: equal-weight top N
    long_return = np.mean([sector_gap_returns[s] for s in top_sectors])
    long_return_net = long_return - COST_PCT

    # Long-short: long top N, short bottom N
    short_return = np.mean([sector_gap_returns[s] for s in bottom_sectors])
    ls_return = long_return - short_return
    ls_return_net = ls_return - 2 * COST_PCT  # cost on both legs

    # Equal-weight all sectors (benchmark)
    all_gap_ret = np.mean([sector_gap_returns[s] for s in sector_gap_returns])

    # SPY gap as another benchmark
    spy_gap = np.nan
    if 'SPY' in features_per_sector:
        spy_sf = features_per_sector.get('SPY', None)
    # Use opens/closes directly
    if t + 1 < n_dates:
        spy_gap = opens['SPY'].iloc[t+1] / closes['SPY'].iloc[t] - 1 if not np.isnan(opens['SPY'].iloc[t+1]) else np.nan

    daily_results.append({
        'date': test_date,
        'long_return': long_return,
        'long_return_net': long_return_net,
        'ls_return': ls_return,
        'ls_return_net': ls_return_net,
        'all_sectors_gap': all_gap_ret,
        'spy_gap': spy_gap,
        'top_sectors': top_sectors,
        'top_probs': [sector_predictions[s] for s in top_sectors],
        'n_sectors_predicted': len(sector_predictions),
    })
    n_oot += 1

print(f"\nOOT days: {n_oot}, Skipped: {n_skipped}")

# ============================================================
# RESULTS ANALYSIS
# ============================================================
if n_oot < 30:
    print("\nFATAL: Too few OOT days for meaningful analysis.")
    sys.exit(1)

results = pd.DataFrame(daily_results)
results.set_index('date', inplace=True)

print(f"\n{'='*70}")
print("RESULTS: OVERNIGHT GAP PREDICTION — SECTOR ETFs")
print(f"{'='*70}")
print(f"OOT Period: {results.index[0].date()} to {results.index[-1].date()}")
print(f"Total OOT Days: {len(results)}")

def compute_metrics(returns, name):
    """Compute risk-adjusted metrics for a return series."""
    r = returns.dropna()
    n = len(r)
    if n < 10:
        return None

    total_ret = (1 + r).prod() - 1
    ann_factor = 252  # daily

    mean_daily = r.mean()
    std_daily = r.std()

    sharpe = mean_daily / std_daily * np.sqrt(ann_factor) if std_daily > 0 else 0

    downside = r[r < 0].std()
    sortino = mean_daily / downside * np.sqrt(ann_factor) if downside > 0 else 0

    wins = r[r > 0]
    losses = r[r < 0]
    wr = len(wins) / n * 100

    gross_profit = wins.sum() if len(wins) > 0 else 0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 1e-9
    pf = gross_profit / gross_loss

    # Max drawdown
    cum = (1 + r).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()

    # Avg daily return in bps
    avg_bps = mean_daily * 10000

    print(f"\n--- {name} ---")
    print(f"  Total return:    {total_ret*100:.2f}%")
    print(f"  Avg daily:       {avg_bps:.2f} bps")
    print(f"  Sharpe:          {sharpe:.3f}")
    print(f"  Sortino:         {sortino:.3f}")
    print(f"  Win Rate:        {wr:.1f}%")
    print(f"  Profit Factor:   {pf:.3f}")
    print(f"  Max Drawdown:    {max_dd*100:.2f}%")
    print(f"  Days:            {n}")

    return {
        'name': name, 'sharpe': sharpe, 'sortino': sortino,
        'wr': wr, 'pf': pf, 'max_dd': max_dd, 'total_ret': total_ret,
        'avg_bps': avg_bps, 'n_days': n
    }

# Compute metrics for all strategies
m_long_gross = compute_metrics(results['long_return'], "LONG TOP-3 (Gross)")
m_long_net = compute_metrics(results['long_return_net'], "LONG TOP-3 (Net of 10bps)")
m_ls_gross = compute_metrics(results['ls_return'], "LONG/SHORT (Gross)")
m_ls_net = compute_metrics(results['ls_return_net'], "LONG/SHORT (Net of 20bps)")
m_benchmark = compute_metrics(results['all_sectors_gap'], "BENCHMARK: Equal-Weight All Sectors")
m_spy = compute_metrics(results['spy_gap'], "BENCHMARK: SPY Gap")

# ============================================================
# REGIME STRATIFICATION
# ============================================================
print(f"\n{'='*70}")
print("REGIME STRATIFICATION")
print(f"{'='*70}")

# Classify regime by SPY daily close-to-close
spy_daily_ret = closes['SPY'].pct_change().reindex(results.index)

# Green day: SPY close > prev close, Red day: SPY close < prev close
results['regime'] = 'flat'
results.loc[spy_daily_ret > 0.001, 'regime'] = 'green'
results.loc[spy_daily_ret < -0.001, 'regime'] = 'red'
results.loc[(spy_daily_ret >= -0.001) & (spy_daily_ret <= 0.001), 'regime'] = 'flat'

for regime in ['green', 'red', 'flat']:
    mask = results['regime'] == regime
    n_days = mask.sum()
    if n_days < 10:
        print(f"\n{regime.upper()} regime: only {n_days} days, skipping")
        continue

    r = results.loc[mask, 'long_return_net']
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
    sortino_down = r[r < 0].std()
    sortino = r.mean() / sortino_down * np.sqrt(252) if sortino_down > 0 else 0
    wr = (r > 0).sum() / len(r) * 100

    print(f"\n{regime.upper()} regime ({n_days} days):")
    print(f"  Sharpe:    {sharpe:.3f}")
    print(f"  Sortino:   {sortino:.3f}")
    print(f"  Win Rate:  {wr:.1f}%")
    print(f"  Avg bps:   {r.mean()*10000:.2f}")

# Regime asymmetry check
green_r = results.loc[results['regime'] == 'green', 'long_return_net']
red_r = results.loc[results['regime'] == 'red', 'long_return_net']
if len(green_r) > 10 and len(red_r) > 10:
    sharpe_green = green_r.mean() / green_r.std() * np.sqrt(252) if green_r.std() > 0 else 0
    sharpe_red = red_r.mean() / red_r.std() * np.sqrt(252) if red_r.std() > 0 else 0
    asymmetry = abs(sharpe_green - sharpe_red) / max(abs(sharpe_green), abs(sharpe_red), 0.001)

    print(f"\nRegime Asymmetry: |{sharpe_green:.3f} - {sharpe_red:.3f}| / max = {asymmetry:.3f}")
    if asymmetry > 0.50:
        print("  *** FAIL: Regime asymmetry > 0.50 — strategy is regime-tailored, not edge ***")
    else:
        print("  PASS: Regime asymmetry within tolerance")

# ============================================================
# FEATURE IMPORTANCE
# ============================================================
print(f"\n{'='*70}")
print("FEATURE IMPORTANCE (from last model)")
print(f"{'='*70}")

try:
    importances = model.feature_importances_
    feat_imp = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
    for fname, imp in feat_imp[:15]:
        bar = '#' * int(imp / max(importances) * 30)
        print(f"  {fname:25s} {imp:6.0f}  {bar}")
except:
    print("  (feature importance unavailable)")

# ============================================================
# YEARLY BREAKDOWN
# ============================================================
print(f"\n{'='*70}")
print("YEARLY BREAKDOWN (Long Top-3, Net)")
print(f"{'='*70}")

results['year'] = results.index.year
for year, grp in results.groupby('year'):
    r = grp['long_return_net']
    n = len(r)
    if n < 5:
        continue
    total = (1 + r).prod() - 1
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
    wr = (r > 0).sum() / n * 100
    print(f"  {year}: {n:4d} days | Return: {total*100:+7.2f}% | Sharpe: {sharpe:+.3f} | WR: {wr:.1f}%")

# ============================================================
# SECTOR HIT RATE
# ============================================================
print(f"\n{'='*70}")
print("SECTOR SELECTION FREQUENCY (how often each is in top-3)")
print(f"{'='*70}")

from collections import Counter
sector_counts = Counter()
for _, row in results.iterrows():
    for s in row['top_sectors']:
        sector_counts[s] += 1

total_picks = len(results) * TOP_N
for sector, count in sorted(sector_counts.items(), key=lambda x: x[1], reverse=True):
    pct = count / len(results) * 100
    print(f"  {sector:6s}: {count:5d} picks ({pct:.1f}% of days)")

# ============================================================
# DAY CONCENTRATION CHECK
# ============================================================
print(f"\n{'='*70}")
print("DAY CONCENTRATION CHECK (HC #344: cap <= 0.70)")
print(f"{'='*70}")

# Max single-day contribution to total P&L
cum_pnl = results['long_return_net'].cumsum()
total_pnl = cum_pnl.iloc[-1]
if abs(total_pnl) > 0:
    max_day_contrib = results['long_return_net'].abs().max() / abs(total_pnl)
    print(f"  Max single-day |return| / |total P&L|: {max_day_contrib:.3f}")
    if max_day_contrib > 0.70:
        print("  *** FAIL: Single day drives >70% of total P&L ***")
    else:
        print("  PASS")
else:
    print("  Total P&L is ~0, concentration check N/A")

# ============================================================
# FINAL VERDICT
# ============================================================
print(f"\n{'='*70}")
print("VERDICT")
print(f"{'='*70}")

if m_long_net:
    sharpe = m_long_net['sharpe']
    if sharpe >= 1.0:
        verdict = "STRONG — Sharpe >= 1.0, worth further investigation"
    elif sharpe >= 0.5:
        verdict = "MARGINAL — Sharpe 0.5-1.0, needs refinement"
    else:
        verdict = "DEAD — Sharpe < 0.5, insufficient edge"

    print(f"\n  Long Top-3 (Net) Sharpe: {sharpe:.3f}")
    print(f"  Verdict: {verdict}")

if m_ls_net:
    sharpe_ls = m_ls_net['sharpe']
    if sharpe_ls >= 1.0:
        verdict_ls = "STRONG"
    elif sharpe_ls >= 0.5:
        verdict_ls = "MARGINAL"
    else:
        verdict_ls = "DEAD"
    print(f"\n  Long/Short (Net) Sharpe: {sharpe_ls:.3f}")
    print(f"  Verdict: {verdict_ls}")

print(f"\n{'='*70}")
print("DONE")
print(f"{'='*70}")
