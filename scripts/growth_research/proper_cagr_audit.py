#!/usr/bin/env python3
"""
PROPER CAGR AUDIT — Genuine Year-by-Year Returns
==================================================
Validates our top 3 strategies with HONEST yearly CAGR:
1. ML Trend Following v2 (8-asset CTA)
2. ML Sector Rotation (11 sector ETFs)
3. ML Credit Timing (bond rotation)

Calculates:
- True geometric CAGR (not annualized arithmetic mean)
- Year-by-year returns (every single calendar year)
- Worst year, best year, median year
- Rolling 3-year CAGR
- Comparison to SPY each year
"""
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import GradientBoostingClassifier
from datetime import datetime
import os, json, warnings
warnings.filterwarnings('ignore')

OUTPUT = '/home/jupiter/Lvl3Quant/output/proper_cagr_audit'
os.makedirs(OUTPUT, exist_ok=True)
np.random.seed(42)

print("=" * 70)
print("PROPER CAGR AUDIT — YEAR-BY-YEAR VERIFICATION")
print("Genuine geometric returns, no shortcuts")
print("=" * 70)

# ============================================================
# STRATEGY 1: ML TREND FOLLOWING v2 (8-asset CTA)
# ============================================================
print(f"\n{'='*70}")
print("STRATEGY 1: ML TREND FOLLOWING v2")
print(f"{'='*70}")

cta_tickers = ['SPY', 'QQQ', 'GLD', 'TLT', 'EEM', 'VNQ', 'HYG', 'XLE']
print(f"Downloading {cta_tickers}...")
cta_data = yf.download(cta_tickers + ['^VIX'], start='2007-01-01', progress=False)
if hasattr(cta_data.index, 'tz') and cta_data.index.tz is not None:
    cta_data.index = cta_data.index.tz_localize(None)

cta_close = cta_data['Close'] if isinstance(cta_data.columns, pd.MultiIndex) else cta_data
cta_close = cta_close.ffill().dropna()
vix = cta_close['^VIX'] if '^VIX' in cta_close.columns else None
cta_close = cta_close.drop('^VIX', axis=1, errors='ignore')

print(f"  {len(cta_close)} days, {cta_close.index[0].date()} to {cta_close.index[-1].date()}")

def build_trend_features(prices):
    """Build trend features for one asset."""
    ret = prices.pct_change()
    feats = pd.DataFrame(index=prices.index)
    for h in [5, 10, 21, 42, 63, 126, 252]:
        feats[f'mom_{h}d'] = prices.pct_change(h)
    for fast, slow in [(10, 50), (20, 100), (50, 200)]:
        ma_fast = prices.rolling(fast).mean()
        ma_slow = prices.rolling(slow).mean()
        feats[f'ma_{fast}_{slow}'] = (ma_fast - ma_slow) / ma_slow
    for h in [10, 21, 63]:
        feats[f'vol_{h}d'] = ret.rolling(h).std() * np.sqrt(252)
    feats['vol_ratio'] = feats['vol_10d'] / (feats['vol_63d'] + 1e-8)
    rolling_max = prices.rolling(252).max()
    feats['dd_from_peak'] = prices / rolling_max - 1
    delta = ret.copy()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    feats['rsi_14'] = 100 - 100 / (1 + rs)
    feats['breakout_63'] = (prices - prices.rolling(63).min()) / (prices.rolling(63).max() - prices.rolling(63).min() + 1e-8)
    return feats

def run_ml_trend_strategy(close_prices, train_window=252, target_hold=21, threshold=0.55, target_vol=0.10):
    """Run ML trend following with vol targeting."""
    all_asset_signals = {}

    for ticker in close_prices.columns:
        prices = close_prices[ticker]
        ret = prices.pct_change()
        feats = build_trend_features(prices)

        # Target: positive 21-day forward return
        fwd_ret = prices.pct_change(target_hold).shift(-target_hold)
        target = (fwd_ret > 0).astype(int)

        combined = pd.concat([feats, target.rename('target')], axis=1).dropna()
        X = combined.drop('target', axis=1)
        y = combined['target']

        if len(X) < train_window + 252:
            continue

        # Walk-forward (monthly steps)
        step = 21
        signals = pd.Series(0.0, index=X.index)

        for i in range(train_window, len(X) - step, step):
            X_train = X.iloc[i-train_window:i]
            y_train = y.iloc[i-train_window:i]
            X_oot = X.iloc[i:i+step]

            model = GradientBoostingClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                subsample=0.8, random_state=42
            )
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_oot)[:, 1]

            # Signal: 1 if prob > threshold, else 0
            for j, idx in enumerate(X_oot.index):
                signals[idx] = 1.0 if probs[j] > threshold else 0.0

        all_asset_signals[ticker] = signals

    # Portfolio: equal weight across active positions, vol-targeted
    signal_df = pd.DataFrame(all_asset_signals)
    ret_df = close_prices.pct_change()

    # Align
    common = signal_df.index.intersection(ret_df.index)
    signal_df = signal_df.reindex(common)
    ret_df = ret_df.reindex(common)

    # Equal weight among active signals
    n_active = signal_df.sum(axis=1).replace(0, np.nan)
    weights = signal_df.div(n_active, axis=0).fillna(0)

    # Raw portfolio return
    port_ret = (weights.shift(1) * ret_df).sum(axis=1)

    # Vol targeting
    rolling_vol = port_ret.rolling(21).std() * np.sqrt(252)
    vol_scalar = target_vol / rolling_vol.clip(lower=0.01)
    vol_scalar = vol_scalar.clip(upper=2.0)  # Cap at 2x leverage

    port_ret_targeted = port_ret * vol_scalar.shift(1)
    port_ret_targeted = port_ret_targeted.dropna()

    return port_ret_targeted

print("\nRunning ML Trend walk-forward...")
cta_returns = run_ml_trend_strategy(cta_close)
print(f"  Strategy returns: {len(cta_returns)} days")

# ============================================================
# STRATEGY 2: ML SECTOR ROTATION (11 sectors)
# ============================================================
print(f"\n{'='*70}")
print("STRATEGY 2: ML SECTOR ROTATION")
print(f"{'='*70}")

sector_tickers = ['XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU', 'XLRE', 'XLC']
print(f"Downloading {sector_tickers}...")
sector_data = yf.download(sector_tickers, start='2000-01-01', progress=False)
if hasattr(sector_data.index, 'tz') and sector_data.index.tz is not None:
    sector_data.index = sector_data.index.tz_localize(None)

sector_close = sector_data['Close'] if isinstance(sector_data.columns, pd.MultiIndex) else sector_data
sector_close = sector_close.ffill()
# Drop assets without enough history
valid_sectors = sector_close.columns[sector_close.notna().sum() > 3000]
sector_close = sector_close[valid_sectors].dropna()
print(f"  {len(sector_close)} days, {len(valid_sectors)} sectors with sufficient data")

print("\nRunning ML Sector walk-forward...")
sector_returns = run_ml_trend_strategy(sector_close)
print(f"  Strategy returns: {len(sector_returns)} days")

# ============================================================
# STRATEGY 3: ML CREDIT TIMING (bond rotation)
# ============================================================
print(f"\n{'='*70}")
print("STRATEGY 3: ML CREDIT TIMING")
print(f"{'='*70}")

credit_tickers = ['HYG', 'LQD', 'TLT', 'IEF', 'SPY']
print(f"Downloading {credit_tickers}...")
credit_data = yf.download(credit_tickers + ['^VIX'], start='2007-01-01', progress=False)
if hasattr(credit_data.index, 'tz') and credit_data.index.tz is not None:
    credit_data.index = credit_data.index.tz_localize(None)

credit_close = credit_data['Close'] if isinstance(credit_data.columns, pd.MultiIndex) else credit_data
credit_close = credit_close.ffill().dropna()
credit_vix = credit_close['^VIX'] if '^VIX' in credit_close.columns else None
credit_close = credit_close.drop('^VIX', axis=1, errors='ignore')

print(f"  {len(credit_close)} days")

# Credit timing: choose best bond ETF each 21 days
def run_credit_timing(close_prices, vix_data, train_periods=24, rebal=21):
    """ML picks HYG vs TLT vs LQD each month."""
    assets = ['HYG', 'TLT', 'LQD']
    available = [a for a in assets if a in close_prices.columns]

    spy_ret = close_prices['SPY'].pct_change()

    # Build features
    features = pd.DataFrame(index=close_prices.index)
    for asset in available:
        ret = close_prices[asset].pct_change()
        for h in [5, 10, 21, 42, 63]:
            features[f'{asset}_mom_{h}d'] = close_prices[asset].pct_change(h)
        features[f'{asset}_vol_21d'] = ret.rolling(21).std() * np.sqrt(252)
        features[f'{asset}_vs_spy'] = close_prices[asset].pct_change(21) - close_prices['SPY'].pct_change(21)

    if vix_data is not None:
        features['vix'] = vix_data
        features['vix_ma21'] = vix_data.rolling(21).mean()
        features['vix_change'] = vix_data.pct_change(5)

    features['spy_mom_21d'] = close_prices['SPY'].pct_change(21)
    features['spy_vol_21d'] = spy_ret.rolling(21).std() * np.sqrt(252)

    # Non-overlapping periods
    features = features.dropna()
    period_starts = list(range(0, len(features) - rebal, rebal))

    period_data = []
    for start in period_starts:
        end = start + rebal
        if end + rebal > len(features):
            break

        feat_row = features.iloc[end]
        fwd_rets = {}
        for asset in available:
            fwd_rets[asset] = (close_prices[asset].reindex(features.index).iloc[end + rebal] /
                              close_prices[asset].reindex(features.index).iloc[end]) - 1

        best = max(fwd_rets, key=fwd_rets.get)
        period_data.append({
            'feat_idx': end,
            'best_asset': best,
            **{f'{a}_ret': fwd_rets[a] for a in available}
        })

    if len(period_data) < train_periods + 10:
        return pd.Series(dtype=float)

    period_df = pd.DataFrame(period_data)
    asset_map = {a: i for i, a in enumerate(available)}
    period_df['target'] = period_df['best_asset'].map(asset_map)

    # Walk-forward
    strategy_rets = []
    strategy_dates = []

    for i in range(train_periods, len(period_df)):
        train_slice = period_df.iloc[i-train_periods:i]
        test_row = period_df.iloc[i]

        train_feats = features.iloc[[r['feat_idx'] for _, r in train_slice.iterrows()]]
        test_feat = features.iloc[test_row['feat_idx']:test_row['feat_idx']+1]

        valid_cols = train_feats.columns[train_feats.notna().all()]
        if len(valid_cols) < 5:
            continue

        X_train = train_feats[valid_cols].values
        y_train = train_slice['target'].values
        X_test = test_feat[valid_cols].values

        if np.isnan(X_train).any() or np.isnan(X_test).any():
            continue

        try:
            model = GradientBoostingClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                subsample=0.8, random_state=42
            )
            model.fit(X_train, y_train)
            pred_idx = np.argmax(model.predict_proba(X_test)[0])
            pred_asset = available[pred_idx]

            ret = test_row[f'{pred_asset}_ret']
            strategy_rets.append(ret)
            strategy_dates.append(features.index[test_row['feat_idx']])
        except:
            continue

    # Convert period returns to daily-equivalent series
    # Each return covers 21 days, spread it as geometric daily
    daily_rets = []
    daily_dates = []
    for ret, date in zip(strategy_rets, strategy_dates):
        # This is a 21-day period return
        daily_equiv = (1 + ret) ** (1/rebal) - 1
        for d in range(rebal):
            daily_rets.append(daily_equiv)
            daily_dates.append(date + pd.Timedelta(days=d))

    return pd.Series(strategy_rets, index=pd.DatetimeIndex(strategy_dates))

print("\nRunning ML Credit Timing walk-forward...")
credit_period_returns = run_credit_timing(credit_close, credit_vix)
print(f"  Strategy periods: {len(credit_period_returns)}")

# ============================================================
# PROPER CAGR CALCULATION
# ============================================================
print(f"\n{'='*70}")
print("YEAR-BY-YEAR RETURNS (GENUINE GEOMETRIC CAGR)")
print(f"{'='*70}")

# SPY benchmark
spy_bench = yf.download('SPY', start='2007-01-01', progress=False)['Close']
if hasattr(spy_bench.index, 'tz') and spy_bench.index.tz is not None:
    spy_bench.index = spy_bench.index.tz_localize(None)
spy_daily_ret = spy_bench.pct_change()

def yearly_analysis(daily_returns, name, is_period=False, period_days=21):
    """Calculate proper year-by-year CAGR."""
    print(f"\n--- {name} ---")

    if is_period:
        # Period returns (each return = 21 days of performance)
        # Group by year
        yearly = {}
        for date, ret in daily_returns.items():
            year = date.year
            if year not in yearly:
                yearly[year] = []
            yearly[year].append(ret)

        year_rets = {}
        for year, rets in sorted(yearly.items()):
            # Compound the period returns
            cum = np.prod([1 + r for r in rets]) - 1
            year_rets[year] = cum
    else:
        # Daily returns - group by year
        yearly_grouped = daily_returns.groupby(daily_returns.index.year)
        year_rets = {}
        for year, group in yearly_grouped:
            cum = (1 + group).prod() - 1
            year_rets[year] = cum

    # Print year by year
    print(f"  {'Year':<6} {'Return':>8} {'vs SPY':>8}")
    print(f"  {'-'*24}")

    all_years = sorted(year_rets.keys())
    spy_yearly = spy_daily_ret.groupby(spy_daily_ret.index.year).apply(lambda x: (1+x).prod() - 1)

    for year in all_years:
        ret = year_rets[year]
        spy_y = spy_yearly.get(year, 0) if year in spy_yearly.index else 0
        alpha = ret - spy_y
        print(f"  {year:<6} {ret:>+7.1%} {alpha:>+7.1%}")

    # Proper geometric CAGR
    total_compound = np.prod([1 + r for r in year_rets.values()])
    n_years = len(year_rets)
    geometric_cagr = total_compound ** (1/n_years) - 1

    # Also calculate from cumulative
    if not is_period:
        cum = (1 + daily_returns).cumprod()
        total_return = cum.iloc[-1] - 1
        exact_years = len(daily_returns) / 252
        exact_cagr = (1 + total_return) ** (1/exact_years) - 1
    else:
        total_return = total_compound - 1
        # Each period = 21 days
        exact_years = len(daily_returns) * period_days / 252
        exact_cagr = (1 + total_return) ** (1/exact_years) - 1

    rets_list = list(year_rets.values())

    print(f"\n  SUMMARY:")
    print(f"  Geometric CAGR (year-by-year):  {geometric_cagr:+.1%}")
    print(f"  Geometric CAGR (exact days):    {exact_cagr:+.1%}")
    print(f"  Total return:                   {total_return:+.1%}")
    print(f"  Years covered:                  {n_years} ({exact_years:.1f} exact)")
    print(f"  Best year:                      {max(rets_list):+.1%} ({all_years[rets_list.index(max(rets_list))]})")
    print(f"  Worst year:                     {min(rets_list):+.1%} ({all_years[rets_list.index(min(rets_list))]})")
    print(f"  Median year:                    {np.median(rets_list):+.1%}")
    print(f"  Positive years:                 {sum(1 for r in rets_list if r > 0)}/{n_years} ({sum(1 for r in rets_list if r > 0)/n_years:.0%})")

    # Volatility and Sharpe from yearly returns
    yearly_vol = np.std(rets_list)
    yearly_sharpe = geometric_cagr / yearly_vol if yearly_vol > 0 else 0

    # Max drawdown from cumulative
    if not is_period:
        cum = (1 + daily_returns).cumprod()
        max_dd = (cum / cum.cummax() - 1).min()
    else:
        cum_periods = np.cumprod([1 + r for r in daily_returns.values])
        peak = np.maximum.accumulate(cum_periods)
        max_dd = np.min(cum_periods / peak - 1)

    print(f"  Annual volatility (from yearly): {yearly_vol:.1%}")
    print(f"  Sharpe (yearly basis):           {yearly_sharpe:.2f}")
    print(f"  Max drawdown:                    {max_dd:.1%}")

    return {
        'name': name,
        'geometric_cagr': float(geometric_cagr),
        'exact_cagr': float(exact_cagr),
        'total_return': float(total_return),
        'n_years': n_years,
        'exact_years': float(exact_years),
        'best_year': float(max(rets_list)),
        'worst_year': float(min(rets_list)),
        'median_year': float(np.median(rets_list)),
        'pct_positive_years': float(sum(1 for r in rets_list if r > 0)/n_years),
        'yearly_vol': float(yearly_vol),
        'yearly_sharpe': float(yearly_sharpe),
        'max_dd': float(max_dd),
        'year_by_year': {str(y): float(r) for y, r in year_rets.items()}
    }

# Run analysis
results = {}
results['ml_trend'] = yearly_analysis(cta_returns, "ML TREND FOLLOWING v2 (8-asset CTA)")
results['ml_sectors'] = yearly_analysis(sector_returns, "ML SECTOR ROTATION (11 sectors)")
results['ml_credit'] = yearly_analysis(credit_period_returns, "ML CREDIT TIMING (bond rotation)",
                                        is_period=True, period_days=21)

# Combined portfolio (70/30 CTA/Sectors)
print(f"\n--- COMBINED PORTFOLIO (70% CTA + 30% Sectors) ---")
common_idx = cta_returns.index.intersection(sector_returns.index)
combined_ret = 0.7 * cta_returns.reindex(common_idx) + 0.3 * sector_returns.reindex(common_idx)
combined_ret = combined_ret.dropna()
results['combined_70_30'] = yearly_analysis(combined_ret, "COMBINED (70% CTA + 30% Sectors)")

# SPY benchmark
print(f"\n--- SPY BENCHMARK ---")
spy_subset = spy_daily_ret.reindex(cta_returns.index).dropna()
results['spy_benchmark'] = yearly_analysis(spy_subset, "SPY Buy & Hold")

# ============================================================
# FINAL COMPARISON TABLE
# ============================================================
print(f"\n\n{'='*70}")
print("FINAL COMPARISON — RELIABLE CAGR NUMBERS")
print(f"{'='*70}")

print(f"\n{'Strategy':<35} {'CAGR':>7} {'Best Yr':>8} {'Worst Yr':>9} {'Med Yr':>7} {'+Yrs':>5} {'MaxDD':>7} {'Sharpe':>7}")
print("-" * 90)
for key in ['ml_trend', 'ml_sectors', 'ml_credit', 'combined_70_30', 'spy_benchmark']:
    r = results[key]
    print(f"{r['name']:<35} {r['exact_cagr']:>+6.1%} {r['best_year']:>+7.1%} "
          f"{r['worst_year']:>+8.1%} {r['median_year']:>+6.1%} "
          f"{r['pct_positive_years']:>4.0%} {r['max_dd']:>6.1%} {r['yearly_sharpe']:>7.2f}")

# Save
with open(f'{OUTPUT}/results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n\nResults saved to {OUTPUT}/results.json")
print("DONE")
