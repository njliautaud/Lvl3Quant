#!/usr/bin/env python3
"""
ML Crypto-Equity Momentum Spillover
=====================================
Bitcoin and crypto markets trade 24/7 and often lead equity market moves,
especially during risk-on/risk-off transitions. Weekend/overnight crypto moves
can predict Monday/next-day equity behavior.

Features: BTC weekend return, BTC/ETH momentum (1/3/5d), BTC-ETH spread change,
BTC vol regime (20d realized vol percentile), BTC drawdown from 30d high,
BTC-SPY rolling correlation (60d), ETH/BTC ratio trend.

Target: Next 5-day SPY return direction/magnitude.
Walk-forward LightGBM (252d sliding, 21d step).
Trade: top quintile -> UPRO, bottom -> SHY, middle -> SPY.
Full adversarial: permutation test, sub-period, outlier robustness, R1 regime.

MANDATORY: 5-day label gap, signal-date permutation test, 10bps costs.
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

# ─── Config ───
CRYPTO_TICKERS = ['BTC-USD', 'ETH-USD']
EQUITY_TICKERS = ['SPY', 'QQQ', 'ARKK']
TRADE_TICKERS = ['UPRO', 'SHY', 'SPY']
VIX_TICKER = '^VIX'

TRAIN_DAYS = 252
ADVANCE_DAYS = 21
LABEL_HORIZON = 5       # 5-day forward return
LABEL_GAP = 5           # mandatory gap to prevent leakage
INITIAL_CAPITAL = 100_000
COST_BPS = 10           # 10bps round-trip
N_PERMS = 200
OUTPUT_DIR = Path('/home/jupiter/Lvl3Quant/output/ml_crypto_equity_spillover')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("ML CRYPTO-EQUITY MOMENTUM SPILLOVER")
print("=" * 70)

# ─── Download data ───
print(f"\nDownloading data...")
all_tickers = CRYPTO_TICKERS + EQUITY_TICKERS + TRADE_TICKERS + [VIX_TICKER]
all_tickers = list(set(all_tickers))

data = {}
for t in all_tickers:
    try:
        df = yf.download(t, start='2016-01-01', end='2026-07-20',
                         progress=False, auto_adjust=True)
        if len(df) > 100:
            if isinstance(df.columns, pd.MultiIndex):
                close = df[('Close', t)].copy()
            else:
                close = df['Close'].copy()
            clean = t.replace('^', '').replace('-', '_')
            close.name = clean
            data[clean] = close
            print(f"  {t}: {len(df)} days")
        else:
            print(f"  {t}: SKIP ({len(df)} days)")
    except Exception as e:
        print(f"  {t}: FAILED ({e})")

prices = pd.DataFrame(data).dropna(how='all').ffill().dropna()
# Filter to 2017+ (when crypto became institutional)
prices = prices[prices.index >= '2017-01-01']
print(f"\nAligned: {len(prices)} days, {prices.shape[1]} assets")
print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")


# ─── Feature engineering ───
def build_features(prices_df, date_idx):
    """Build crypto-equity spillover features at a given date index."""
    feats = {}
    p = prices_df.iloc[:date_idx+1]
    if len(p) < 252:
        return None

    btc = p.get('BTC_USD')
    eth = p.get('ETH_USD')
    spy = p.get('SPY')
    qqq = p.get('QQQ')
    arkk = p.get('ARKK')
    vix = p.get('VIX')

    if btc is None or eth is None or spy is None:
        return None
    if len(btc.dropna()) < 252:
        return None

    # --- BTC/ETH Momentum (1d, 3d, 5d, 10d, 21d) ---
    for asset_name, asset in [('btc', btc), ('eth', eth)]:
        for w in [1, 3, 5, 10, 21]:
            if len(asset) > w and asset.iloc[-w] > 0:
                feats[f'{asset_name}_mom_{w}d'] = asset.iloc[-1] / asset.iloc[-w] - 1
            else:
                feats[f'{asset_name}_mom_{w}d'] = 0.0

    # --- BTC Weekend Return (Fri close -> Mon proxy) ---
    # Use dayofweek: check if today is Monday (0) or Tuesday (1)
    current_date = p.index[-1]
    dow = current_date.dayofweek
    if dow <= 1 and len(btc) > 3:
        # Look back to find last Friday's close
        lookback = p.index[-min(5, len(p)):]
        fri_mask = lookback.dayofweek == 4
        if fri_mask.any():
            fri_date = lookback[fri_mask][-1]
            fri_close = btc.loc[fri_date]
            if fri_close > 0:
                feats['btc_weekend_ret'] = btc.iloc[-1] / fri_close - 1
            else:
                feats['btc_weekend_ret'] = 0.0
        else:
            feats['btc_weekend_ret'] = 0.0
    else:
        feats['btc_weekend_ret'] = 0.0

    # --- BTC-ETH Spread Change (risk appetite proxy) ---
    if len(btc) > 5 and len(eth) > 5:
        ratio_now = eth.iloc[-1] / btc.iloc[-1] if btc.iloc[-1] > 0 else 0
        ratio_5d = eth.iloc[-5] / btc.iloc[-5] if btc.iloc[-5] > 0 else 0
        feats['eth_btc_spread_chg_5d'] = ratio_now - ratio_5d
        ratio_1d = eth.iloc[-2] / btc.iloc[-2] if btc.iloc[-2] > 0 else 0
        feats['eth_btc_spread_chg_1d'] = ratio_now - ratio_1d
    else:
        feats['eth_btc_spread_chg_5d'] = 0.0
        feats['eth_btc_spread_chg_1d'] = 0.0

    # --- ETH/BTC Ratio Trend (rotation within crypto = risk appetite) ---
    if len(btc) > 21:
        eth_btc = (eth / btc).dropna()
        if len(eth_btc) > 21:
            feats['eth_btc_ratio'] = eth_btc.iloc[-1]
            feats['eth_btc_ratio_ma10'] = eth_btc.iloc[-10:].mean()
            feats['eth_btc_ratio_ma21'] = eth_btc.iloc[-21:].mean()
            feats['eth_btc_ratio_vs_ma21'] = eth_btc.iloc[-1] / eth_btc.iloc[-21:].mean() - 1
        else:
            feats['eth_btc_ratio'] = 0.0
            feats['eth_btc_ratio_ma10'] = 0.0
            feats['eth_btc_ratio_ma21'] = 0.0
            feats['eth_btc_ratio_vs_ma21'] = 0.0
    else:
        feats['eth_btc_ratio'] = 0.0
        feats['eth_btc_ratio_ma10'] = 0.0
        feats['eth_btc_ratio_ma21'] = 0.0
        feats['eth_btc_ratio_vs_ma21'] = 0.0

    # --- BTC Volatility Regime (20d realized vol percentile) ---
    btc_rets = btc.pct_change().dropna()
    if len(btc_rets) > 252:
        vol_20d = btc_rets.iloc[-20:].std() * np.sqrt(252)
        vol_history = btc_rets.rolling(20).std().dropna() * np.sqrt(252)
        if len(vol_history) > 20:
            feats['btc_vol_20d'] = vol_20d
            feats['btc_vol_pctile'] = (vol_history < vol_20d).mean()
        else:
            feats['btc_vol_20d'] = 0.0
            feats['btc_vol_pctile'] = 0.5
    else:
        feats['btc_vol_20d'] = 0.0
        feats['btc_vol_pctile'] = 0.5

    # --- BTC Drawdown from 30d High ---
    if len(btc) > 30:
        high_30d = btc.iloc[-30:].max()
        feats['btc_dd_30d'] = btc.iloc[-1] / high_30d - 1 if high_30d > 0 else 0
    else:
        feats['btc_dd_30d'] = 0.0

    # --- BTC-SPY Rolling Correlation (60d) ---
    if len(btc) > 60 and len(spy) > 60:
        btc_r = btc.pct_change().iloc[-60:]
        spy_r = spy.pct_change().iloc[-60:]
        aligned = pd.DataFrame({'btc': btc_r, 'spy': spy_r}).dropna()
        if len(aligned) > 30:
            feats['btc_spy_corr_60d'] = aligned['btc'].corr(aligned['spy'])
        else:
            feats['btc_spy_corr_60d'] = 0.0
    else:
        feats['btc_spy_corr_60d'] = 0.0

    # --- VIX features ---
    if vix is not None and len(vix.dropna()) > 21:
        v = vix.dropna()
        feats['vix_level'] = v.iloc[-1]
        feats['vix_chg_1d'] = v.iloc[-1] - v.iloc[-2] if len(v) > 1 else 0
        feats['vix_chg_5d'] = v.iloc[-1] - v.iloc[-5] if len(v) > 5 else 0
        feats['vix_ma21'] = v.iloc[-21:].mean()
        feats['vix_vs_ma21'] = v.iloc[-1] / v.iloc[-21:].mean() - 1
    else:
        feats['vix_level'] = 20.0
        feats['vix_chg_1d'] = 0.0
        feats['vix_chg_5d'] = 0.0
        feats['vix_ma21'] = 20.0
        feats['vix_vs_ma21'] = 0.0

    # --- SPY/QQQ context ---
    for asset_name, asset in [('spy', spy), ('qqq', qqq)]:
        if asset is not None and len(asset.dropna()) > 21:
            a = asset.dropna()
            for w in [1, 5, 10, 21]:
                if len(a) > w and a.iloc[-w] > 0:
                    feats[f'{asset_name}_ret_{w}d'] = a.iloc[-1] / a.iloc[-w] - 1
                else:
                    feats[f'{asset_name}_ret_{w}d'] = 0.0

    # --- ARKK (speculative proxy) ---
    if arkk is not None and len(arkk.dropna()) > 21:
        a = arkk.dropna()
        for w in [1, 5, 10]:
            if len(a) > w and a.iloc[-w] > 0:
                feats[f'arkk_ret_{w}d'] = a.iloc[-1] / a.iloc[-w] - 1
            else:
                feats[f'arkk_ret_{w}d'] = 0.0
        # ARKK vs SPY relative strength
        if spy is not None and len(spy.dropna()) > 10:
            spy_5d = spy.iloc[-1] / spy.iloc[-5] - 1 if spy.iloc[-5] > 0 else 0
            arkk_5d = a.iloc[-1] / a.iloc[-5] - 1 if a.iloc[-5] > 0 else 0
            feats['arkk_vs_spy_5d'] = arkk_5d - spy_5d

    # --- BTC dominance proxy (BTC mom vs ETH mom) ---
    if 'btc_mom_5d' in feats and 'eth_mom_5d' in feats:
        feats['btc_dom_5d'] = feats['btc_mom_5d'] - feats['eth_mom_5d']
    if 'btc_mom_1d' in feats and 'eth_mom_1d' in feats:
        feats['btc_dom_1d'] = feats['btc_mom_1d'] - feats['eth_mom_1d']

    return feats


# ─── Build feature matrix ───
print("\nBuilding feature matrix...")
feature_rows = []
labels = []
dates = []

spy_series = prices['SPY']

for i in range(252, len(prices) - LABEL_HORIZON - LABEL_GAP):
    f = build_features(prices, i)
    if f is None:
        continue
    # Label: 5-day forward SPY return, with 5-day gap
    future_idx = i + LABEL_GAP + LABEL_HORIZON
    gap_idx = i + LABEL_GAP
    if future_idx >= len(spy_series) or gap_idx >= len(spy_series):
        continue
    fwd_ret = spy_series.iloc[future_idx] / spy_series.iloc[gap_idx] - 1
    feature_rows.append(f)
    labels.append(fwd_ret)
    dates.append(prices.index[i])

X = pd.DataFrame(feature_rows, index=dates)
y = np.array(labels)
print(f"Feature matrix: {X.shape[0]} samples x {X.shape[1]} features")
print(f"Label (5d SPY fwd ret): mean={y.mean()*100:.3f}%, std={y.std()*100:.3f}%")
print(f"Date range: {X.index[0].date()} to {X.index[-1].date()}")

# Fill any NaN features
X = X.fillna(0)


# ─── Walk-forward LightGBM ───
print("\n" + "=" * 70)
print("WALK-FORWARD LIGHTGBM (252d sliding, 21d step)")
print("=" * 70)

lgb_params = {
    'objective': 'regression',
    'metric': 'mae',
    'n_estimators': 200,
    'max_depth': 4,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'verbose': -1,
    'n_jobs': 2,
    'random_state': 42,
}

predictions = []
actuals = []
pred_dates = []
fold_metrics = []
feature_importance_accum = np.zeros(X.shape[1])

fold = 0
i = TRAIN_DAYS
while i < len(X):
    train_end = i
    train_start = max(0, i - TRAIN_DAYS)
    test_end = min(i + ADVANCE_DAYS, len(X))

    X_train = X.iloc[train_start:train_end]
    y_train = y[train_start:train_end]
    X_test = X.iloc[i:test_end]
    y_test = y[i:test_end]

    if len(X_test) == 0:
        break

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    predictions.extend(preds)
    actuals.extend(y_test)
    pred_dates.extend(X.index[i:test_end])

    # Feature importance
    feature_importance_accum += model.feature_importances_

    # Fold IC
    if len(preds) > 5:
        corr = np.corrcoef(preds, y_test)[0, 1] if np.std(preds) > 0 else 0
        fold_metrics.append({
            'fold': fold,
            'train_start': X.index[train_start].date().isoformat(),
            'test_start': X.index[i].date().isoformat(),
            'test_end': X.index[min(test_end-1, len(X)-1)].date().isoformat(),
            'n_test': len(X_test),
            'ic': round(corr, 4),
            'mean_pred': round(np.mean(preds), 6),
            'mean_actual': round(np.mean(y_test), 6),
        })

    fold += 1
    i += ADVANCE_DAYS

predictions = np.array(predictions)
actuals = np.array(actuals)
pred_dates = pd.DatetimeIndex(pred_dates)

print(f"\nTotal OOT predictions: {len(predictions)}")
concat_ic = np.corrcoef(predictions, actuals)[0, 1] if np.std(predictions) > 0 else 0
print(f"Concat IC: {concat_ic:.4f}")

# Feature importance
fi = pd.DataFrame({
    'feature': X.columns,
    'importance': feature_importance_accum / max(fold, 1)
}).sort_values('importance', ascending=False)
print(f"\nTop 15 features:")
for _, row in fi.head(15).iterrows():
    print(f"  {row['feature']:30s}  {row['importance']:.1f}")


# ─── Build backtest ───
print("\n" + "=" * 70)
print("BACKTEST: Top quintile -> UPRO, Bottom -> SHY, Middle -> SPY")
print("=" * 70)

# Get trade instrument prices aligned to prediction dates
trade_prices = {}
for t in ['UPRO', 'SHY', 'SPY']:
    clean = t.replace('^', '').replace('-', '_')
    if clean in prices.columns:
        trade_prices[t] = prices[clean]

if 'UPRO' not in trade_prices or 'SHY' not in trade_prices:
    print("ERROR: Missing UPRO or SHY data")
    # Fall back to SPY for all
    trade_prices['UPRO'] = prices['SPY'] * 3  # rough approximation
    trade_prices['SHY'] = prices['SPY'] * 0.01 + 100  # rough approximation

# Build prediction DataFrame
bt = pd.DataFrame({
    'prediction': predictions,
    'actual_5d_ret': actuals,
    'date': pred_dates,
}).set_index('date')

# Rolling quintile assignment (expanding to avoid look-ahead)
bt['signal'] = np.nan
WARMUP = 63  # need 63 days before assigning quintiles

for i in range(WARMUP, len(bt)):
    hist_preds = bt['prediction'].iloc[:i+1]
    current_pred = bt['prediction'].iloc[i]
    pctile = (hist_preds < current_pred).mean()
    bt.iloc[i, bt.columns.get_loc('signal')] = pctile

bt = bt.dropna(subset=['signal'])

# Assign instruments: top quintile -> UPRO, bottom -> SHY, middle -> SPY
bt['instrument'] = 'SPY'
bt.loc[bt['signal'] >= 0.8, 'instrument'] = 'UPRO'
bt.loc[bt['signal'] <= 0.2, 'instrument'] = 'SHY'

# Calculate returns using LABEL_HORIZON forward returns of the chosen instrument
bt['trade_ret'] = 0.0
cost_factor = COST_BPS / 10000.0

for idx in bt.index:
    inst = bt.loc[idx, 'instrument']
    if inst in trade_prices:
        p_series = trade_prices[inst]
        if idx in p_series.index:
            loc = p_series.index.get_loc(idx)
            fwd_loc = loc + LABEL_HORIZON + LABEL_GAP
            gap_loc = loc + LABEL_GAP
            if fwd_loc < len(p_series) and gap_loc < len(p_series):
                ret = p_series.iloc[fwd_loc] / p_series.iloc[gap_loc] - 1
                bt.loc[idx, 'trade_ret'] = ret - cost_factor  # deduct costs

# Benchmark: buy-and-hold SPY
bt['spy_ret'] = 0.0
spy_price = trade_prices.get('SPY', prices['SPY'])
for idx in bt.index:
    if idx in spy_price.index:
        loc = spy_price.index.get_loc(idx)
        fwd_loc = loc + LABEL_HORIZON + LABEL_GAP
        gap_loc = loc + LABEL_GAP
        if fwd_loc < len(spy_price) and gap_loc < len(spy_price):
            bt.loc[idx, 'spy_ret'] = spy_price.iloc[fwd_loc] / spy_price.iloc[gap_loc] - 1

# Note: overlapping 5-day windows; we average returns per calendar period
# For clean metrics, compute cumulative
bt['cum_strat'] = (1 + bt['trade_ret']).cumprod()
bt['cum_spy'] = (1 + bt['spy_ret']).cumprod()

# Annualized metrics (approximate: ~252 observations/year, each is 5-day return)
periods_per_year = 252 / LABEL_HORIZON  # ~50.4
mean_ret = bt['trade_ret'].mean()
std_ret = bt['trade_ret'].std()
sharpe = mean_ret / std_ret * np.sqrt(periods_per_year) if std_ret > 0 else 0

downside = bt['trade_ret'][bt['trade_ret'] < 0].std()
sortino = mean_ret / downside * np.sqrt(periods_per_year) if downside > 0 else 0

wins = (bt['trade_ret'] > 0).sum()
total = len(bt['trade_ret'])
wr = wins / total if total > 0 else 0

gross_profit = bt['trade_ret'][bt['trade_ret'] > 0].sum()
gross_loss = abs(bt['trade_ret'][bt['trade_ret'] < 0].sum())
pf = gross_profit / gross_loss if gross_loss > 0 else 999

spy_mean = bt['spy_ret'].mean()
spy_std = bt['spy_ret'].std()
spy_sharpe = spy_mean / spy_std * np.sqrt(periods_per_year) if spy_std > 0 else 0

final_strat = bt['cum_strat'].iloc[-1]
final_spy = bt['cum_spy'].iloc[-1]

# Max drawdown
cum = bt['cum_strat']
peak = cum.expanding().max()
dd = (cum - peak) / peak
max_dd = dd.min()

print(f"\n--- Strategy Performance (after {COST_BPS}bps costs) ---")
print(f"Total observations:   {total}")
print(f"Signal dates:         {bt.index[0].date()} to {bt.index[-1].date()}")
print(f"Cumulative return:    {(final_strat - 1)*100:.1f}%")
print(f"SPY B&H return:       {(final_spy - 1)*100:.1f}%")
print(f"Sharpe ratio:         {sharpe:.3f}")
print(f"Sortino ratio:        {sortino:.3f}")
print(f"Win rate:             {wr*100:.1f}%")
print(f"Profit factor:        {pf:.2f}")
print(f"Max drawdown:         {max_dd*100:.1f}%")
print(f"SPY Sharpe:           {spy_sharpe:.3f}")
print(f"Concat IC:            {concat_ic:.4f}")

# Instrument allocation
print(f"\nAllocation breakdown:")
for inst in ['UPRO', 'SPY', 'SHY']:
    mask = bt['instrument'] == inst
    n = mask.sum()
    pct = n / len(bt) * 100
    avg_ret = bt.loc[mask, 'trade_ret'].mean() * 100 if n > 0 else 0
    print(f"  {inst}: {n} ({pct:.1f}%) signals, avg ret: {avg_ret:.3f}%")


# ─── R1 Regime-Agnostic Validation ───
print("\n" + "=" * 70)
print("R1: REGIME-AGNOSTIC VALIDATION")
print("=" * 70)

# Classify days by SPY close-to-close (green/red/flat)
spy_daily_ret = prices['SPY'].pct_change()
bt['regime'] = 'flat'
for idx in bt.index:
    if idx in spy_daily_ret.index:
        r = spy_daily_ret.loc[idx]
        if r > 0.002:
            bt.loc[idx, 'regime'] = 'green'
        elif r < -0.002:
            bt.loc[idx, 'regime'] = 'red'

regime_stats = {}
for regime in ['green', 'red', 'flat']:
    mask = bt['regime'] == regime
    if mask.sum() > 10:
        rets = bt.loc[mask, 'trade_ret']
        std = rets.std()
        regime_sharpe = rets.mean() / std * np.sqrt(periods_per_year) if std > 0 else 0
        regime_stats[regime] = {
            'n': int(mask.sum()),
            'sharpe': round(regime_sharpe, 3),
            'mean_ret': round(rets.mean() * 100, 4),
            'wr': round((rets > 0).mean() * 100, 1),
        }
        print(f"  {regime:6s}: n={mask.sum():4d}, Sharpe={regime_sharpe:.3f}, "
              f"mean={rets.mean()*100:.3f}%, WR={((rets>0).mean())*100:.1f}%")

# R1 check: |Sharpe_green - Sharpe_red| / max(|Sharpe_green|, |Sharpe_red|) <= 0.50
if 'green' in regime_stats and 'red' in regime_stats:
    sg = regime_stats['green']['sharpe']
    sr = regime_stats['red']['sharpe']
    denom = max(abs(sg), abs(sr))
    regime_divergence = abs(sg - sr) / denom if denom > 0 else 0
    r1_pass = regime_divergence <= 0.50
    print(f"\n  Regime divergence: {regime_divergence:.3f} (threshold: 0.50)")
    print(f"  R1 PASS: {r1_pass}")
else:
    r1_pass = False
    regime_divergence = None
    print("  WARN: insufficient regime data for R1 test")


# ─── Permutation Test (Signal-Date Shuffle) ───
print("\n" + "=" * 70)
print(f"PERMUTATION TEST ({N_PERMS} shuffles)")
print("=" * 70)

np.random.seed(42)
perm_sharpes = []
for p_i in range(N_PERMS):
    shuffled_preds = predictions.copy()
    np.random.shuffle(shuffled_preds)

    # Re-assign quintiles with shuffled predictions
    perm_bt = bt[['trade_ret', 'spy_ret']].copy()
    # Build percentile from shuffled
    perm_signals = np.full(len(shuffled_preds), np.nan)
    for j in range(WARMUP, len(shuffled_preds)):
        hist = shuffled_preds[:j+1]
        perm_signals[j] = (hist < shuffled_preds[j]).mean()

    # We need to align — only use the subset that matches bt
    # bt was built from predictions after WARMUP, so perm_signals[WARMUP:] maps to bt
    valid_mask = ~np.isnan(perm_signals)
    valid_indices = np.where(valid_mask)[0]

    # Map back to bt indices
    if len(valid_indices) > 0:
        # perm_signals corresponds to the full predictions array
        # bt starts at index WARMUP of predictions array
        perm_rets = []
        for j in valid_indices:
            if j >= WARMUP and (j - WARMUP) < len(bt):
                bt_idx = j - WARMUP
                sig = perm_signals[j]
                if sig >= 0.8:
                    inst = 'UPRO'
                elif sig <= 0.2:
                    inst = 'SHY'
                else:
                    inst = 'SPY'

                date = bt.index[bt_idx]
                if inst in trade_prices and date in trade_prices[inst].index:
                    p_series = trade_prices[inst]
                    loc = p_series.index.get_loc(date)
                    fwd_loc = loc + LABEL_HORIZON + LABEL_GAP
                    gap_loc = loc + LABEL_GAP
                    if fwd_loc < len(p_series) and gap_loc < len(p_series):
                        ret = p_series.iloc[fwd_loc] / p_series.iloc[gap_loc] - 1 - cost_factor
                        perm_rets.append(ret)

        if len(perm_rets) > 10:
            perm_rets = np.array(perm_rets)
            perm_std = perm_rets.std()
            ps = perm_rets.mean() / perm_std * np.sqrt(periods_per_year) if perm_std > 0 else 0
            perm_sharpes.append(ps)

    if (p_i + 1) % 50 == 0:
        print(f"  Completed {p_i + 1}/{N_PERMS} permutations...")

perm_sharpes = np.array(perm_sharpes)
if len(perm_sharpes) > 0:
    perm_pval = (perm_sharpes >= sharpe).mean()
    print(f"\n  Real Sharpe:     {sharpe:.3f}")
    print(f"  Perm mean:       {perm_sharpes.mean():.3f}")
    print(f"  Perm std:        {perm_sharpes.std():.3f}")
    print(f"  Perm p-value:    {perm_pval:.4f}")
    print(f"  Perm 95th pctile: {np.percentile(perm_sharpes, 95):.3f}")
    perm_pass = perm_pval < 0.05
    print(f"  PERM PASS (p<0.05): {perm_pass}")
else:
    perm_pval = 1.0
    perm_pass = False
    print("  ERROR: No valid permutation results")


# ─── Sub-Period Analysis (4 blocks) ───
print("\n" + "=" * 70)
print("SUB-PERIOD ANALYSIS (4 blocks)")
print("=" * 70)

n_blocks = 4
block_size = len(bt) // n_blocks
sub_period_results = []

for b in range(n_blocks):
    start = b * block_size
    end = (b + 1) * block_size if b < n_blocks - 1 else len(bt)
    block = bt.iloc[start:end]

    if len(block) > 10:
        block_ret = block['trade_ret']
        block_std = block_ret.std()
        block_sharpe = block_ret.mean() / block_std * np.sqrt(periods_per_year) if block_std > 0 else 0
        block_wr = (block_ret > 0).mean()

        sub_period_results.append({
            'block': b + 1,
            'start': block.index[0].date().isoformat(),
            'end': block.index[-1].date().isoformat(),
            'n': len(block),
            'sharpe': round(block_sharpe, 3),
            'mean_ret_pct': round(block_ret.mean() * 100, 4),
            'wr': round(block_wr * 100, 1),
        })
        print(f"  Block {b+1}: {block.index[0].date()} to {block.index[-1].date()} | "
              f"n={len(block):4d} | Sharpe={block_sharpe:.3f} | WR={block_wr*100:.1f}%")

# Check if all sub-periods have same-sign Sharpe
sub_sharpes = [sp['sharpe'] for sp in sub_period_results]
all_positive = all(s > 0 for s in sub_sharpes)
all_negative = all(s < 0 for s in sub_sharpes)
consistent = all_positive or all_negative
print(f"\n  Sub-period consistency: {consistent} (all same-sign Sharpe)")


# ─── Outlier Robustness ───
print("\n" + "=" * 70)
print("OUTLIER ROBUSTNESS (winsorize 1%/5%)")
print("=" * 70)

for clip_pct in [0.01, 0.05]:
    lower = bt['trade_ret'].quantile(clip_pct)
    upper = bt['trade_ret'].quantile(1 - clip_pct)
    clipped = bt['trade_ret'].clip(lower, upper)
    clip_std = clipped.std()
    clip_sharpe = clipped.mean() / clip_std * np.sqrt(periods_per_year) if clip_std > 0 else 0
    clip_wr = (clipped > 0).mean()
    print(f"  Winsorize {clip_pct*100:.0f}%: Sharpe={clip_sharpe:.3f}, WR={clip_wr*100:.1f}%")


# ─── Day Concentration Check ───
print("\n" + "=" * 70)
print("DAY CONCENTRATION CHECK")
print("=" * 70)

bt['dow'] = bt.index.dayofweek
dow_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri'}
for d in range(5):
    mask = bt['dow'] == d
    if mask.sum() > 10:
        rets = bt.loc[mask, 'trade_ret']
        day_std = rets.std()
        day_sharpe = rets.mean() / day_std * np.sqrt(periods_per_year) if day_std > 0 else 0
        print(f"  {dow_map.get(d, str(d))}: n={mask.sum():4d}, Sharpe={day_sharpe:.3f}, "
              f"mean={rets.mean()*100:.3f}%, WR={(rets>0).mean()*100:.1f}%")

# Day concentration: max fraction of total P&L from single day
day_pnl = bt.groupby('dow')['trade_ret'].sum()
total_pnl = day_pnl.sum()
if total_pnl != 0:
    max_day_conc = day_pnl.abs().max() / abs(total_pnl)
    print(f"\n  Max day concentration: {max_day_conc:.2f} (cap: 0.70)")
    day_conc_pass = max_day_conc <= 0.70
    print(f"  Day-conc PASS: {day_conc_pass}")
else:
    max_day_conc = 0
    day_conc_pass = True


# ─── Save results ───
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

results = {
    'strategy': 'ML Crypto-Equity Momentum Spillover',
    'run_date': dt.datetime.now().isoformat(),
    'data_range': f"{prices.index[0].date()} to {prices.index[-1].date()}",
    'n_samples': int(X.shape[0]),
    'n_features': int(X.shape[1]),
    'n_oot_predictions': int(len(predictions)),
    'concat_ic': round(float(concat_ic), 4),
    'performance': {
        'sharpe': round(float(sharpe), 3),
        'sortino': round(float(sortino), 3),
        'win_rate': round(float(wr * 100), 1),
        'profit_factor': round(float(pf), 2),
        'max_drawdown_pct': round(float(max_dd * 100), 1),
        'cumulative_return_pct': round(float((final_strat - 1) * 100), 1),
        'spy_bh_return_pct': round(float((final_spy - 1) * 100), 1),
        'spy_sharpe': round(float(spy_sharpe), 3),
        'cost_bps': COST_BPS,
    },
    'adversarial': {
        'perm_test_pvalue': round(float(perm_pval), 4),
        'perm_test_pass': bool(perm_pass),
        'perm_real_sharpe': round(float(sharpe), 3),
        'perm_mean_sharpe': round(float(perm_sharpes.mean()), 3) if len(perm_sharpes) > 0 else None,
        'r1_regime_divergence': round(float(regime_divergence), 3) if regime_divergence is not None else None,
        'r1_pass': bool(r1_pass),
        'sub_period_consistent': bool(consistent),
        'sub_periods': sub_period_results,
        'day_conc_pass': bool(day_conc_pass),
    },
    'regime_stats': regime_stats,
    'fold_metrics': fold_metrics,
    'top_features': fi.head(20).to_dict('records'),
    'allocation': {
        inst: {
            'n': int((bt['instrument'] == inst).sum()),
            'pct': round(float((bt['instrument'] == inst).mean() * 100), 1),
        } for inst in ['UPRO', 'SPY', 'SHY']
    },
}

with open(OUTPUT_DIR / 'results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

bt.to_csv(OUTPUT_DIR / 'backtest.csv')
fi.to_csv(OUTPUT_DIR / 'feature_importance.csv', index=False)
print(f"Saved results to {OUTPUT_DIR}")


# ─── Final Verdict ───
print("\n" + "=" * 70)
print("FINAL VERDICT")
print("=" * 70)

tests = {
    'Sharpe > 0': sharpe > 0,
    'Sharpe > SPY Sharpe': sharpe > spy_sharpe,
    'Perm test p<0.05': perm_pass,
    'R1 regime-agnostic': r1_pass,
    'Sub-period consistent': consistent,
    'Day-conc < 0.70': day_conc_pass,
    'Concat IC > 0': concat_ic > 0,
}

all_pass = all(tests.values())
for test, passed in tests.items():
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {test}")

print(f"\n  OVERALL: {'PASS - Strategy has edge' if all_pass else 'FAIL - Strategy does NOT pass all tests'}")
print(f"  Sharpe: {sharpe:.3f} | Sortino: {sortino:.3f} | IC: {concat_ic:.4f}")
print(f"  Perm p-value: {perm_pval:.4f} | Regime div: {regime_divergence:.3f}" if regime_divergence else "")
print("=" * 70)
