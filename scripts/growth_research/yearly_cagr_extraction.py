#!/usr/bin/env python3
"""
Yearly CAGR Extraction: ML Trend (CTA) + ML Sector Rotation
=============================================================
Runs the EXACT same logic as ml_portfolio_combo.py but:
  - Saves daily portfolio returns to CSV
  - Computes year-by-year returns, Sharpe, MaxDD for each strategy + 50/50 combo

Walk-forward: SLIDING 252d (HC #0). Fixed $100K, NO DCA (HC #713).
"""

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
from sklearn.ensemble import GradientBoostingClassifier

warnings.filterwarnings('ignore')
np.random.seed(42)

sys.stdout.reconfigure(line_buffering=True)

# ─────────────────────────────────────────────────────────────────────────────
INITIAL_CAPITAL  = 100_000
TRAIN_WINDOW     = 252
REBAL_COST_BPS   = 10
MA_SHORT         = 20
MA_LONG          = 100
ML_THRESHOLD     = 0.55
TARGET_VOL       = 0.10

BASE   = Path("/home/jupiter/Lvl3Quant")
OUTPUT = BASE / "output" / "ml_portfolio_combo"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Strategy 1: CTA Trend Following universe
UNIVERSE_CTA = {
    'SPY': 'US Equities', 'TLT': 'Long Bonds', 'GLD': 'Gold',
    'UUP': 'US Dollar', 'EEM': 'Emerging Mkts', 'VNQ': 'Real Estate',
    'HYG': 'High Yield', 'XLE': 'Energy',
}

# Strategy 2: Sector Rotation universe
UNIVERSE_SECTORS = {
    'XLK': 'Technology', 'XLF': 'Financials', 'XLE': 'Energy',
    'XLV': 'Health Care', 'XLY': 'Consumer Disc', 'XLP': 'Consumer Staples',
    'XLI': 'Industrials', 'XLB': 'Materials', 'XLU': 'Utilities',
    'XLRE': 'Real Estate', 'XLC': 'Communication',
}


def download_data():
    print("=" * 80)
    print("STEP 1: DOWNLOADING DATA")
    print("=" * 80)

    all_tickers = sorted(set(
        list(UNIVERSE_CTA.keys()) + list(UNIVERSE_SECTORS.keys()) + ['^VIX', 'SPY']
    ))

    cache = BASE / "data" / "cache" / "portfolio_combo_data.parquet"
    if cache.exists() and (time.time() - cache.stat().st_mtime) < 7200:
        df = pd.read_parquet(cache)
        print(f"  Cached: {df.shape}, {df.index[0].date()} -> {df.index[-1].date()}")
        return df

    print(f"  Downloading {len(all_tickers)} tickers...")
    raw = yf.download(all_tickers, start='2008-01-01', auto_adjust=True, progress=False)
    closes = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw
    if '^VIX' in closes.columns:
        closes = closes.rename(columns={'^VIX': 'VIX'})
    closes = closes.ffill().dropna(thresh=len(closes.columns) - 3)
    cache.parent.mkdir(parents=True, exist_ok=True)
    closes.to_parquet(cache)
    print(f"  Shape: {closes.shape}, {closes.index[0].date()} -> {closes.index[-1].date()}")
    return closes


def build_features_vectorized(df, universe):
    """VECTORIZED feature building — exact same as ml_portfolio_combo.py."""
    print(f"  Building features for {len(universe)} assets...")
    t0 = time.time()

    all_features = []

    for ticker in universe:
        if ticker not in df.columns:
            continue

        price = df[ticker]
        ret = price.pct_change()
        ma_short = price.rolling(MA_SHORT).mean()
        ma_long = price.rolling(MA_LONG).mean()

        # Trend signal
        trend = pd.Series(0.0, index=df.index)
        trend[ma_short > ma_long] = 1.0
        trend[ma_short < ma_long] = -1.0

        # Only keep days with a signal and enough history
        valid_start = MA_LONG + 60
        valid_mask = (trend != 0) & (pd.Series(range(len(df)), index=df.index) >= valid_start)
        valid_idx = df.index[valid_mask]

        if len(valid_idx) == 0:
            continue

        # Vectorized features
        feat_df = pd.DataFrame(index=valid_idx)
        feat_df['ticker'] = ticker
        feat_df['direction'] = trend.loc[valid_idx]
        feat_df['ma_dist'] = ((ma_short - ma_long) / (price + 1e-8)).loc[valid_idx]

        # Trend duration
        trend_change = (trend != trend.shift(1)).astype(int)
        trend_duration = trend_change.groupby(trend_change.cumsum()).cumcount()
        feat_df['trend_duration'] = trend_duration.loc[valid_idx]

        # Momentum
        feat_df['mom_5d'] = price.pct_change(5).loc[valid_idx]
        feat_df['mom_20d'] = price.pct_change(20).loc[valid_idx]
        feat_df['mom_60d'] = price.pct_change(60).loc[valid_idx]

        # Volatility
        feat_df['vol_20d'] = ret.rolling(20).std().loc[valid_idx] * np.sqrt(252)
        feat_df['vol_60d'] = ret.rolling(60).std().loc[valid_idx] * np.sqrt(252)

        # Drawdown from 252d peak
        rolling_max = price.rolling(252, min_periods=1).max()
        feat_df['drawdown'] = (price / rolling_max - 1).loc[valid_idx]

        # Skew/kurt
        feat_df['skew_20d'] = ret.rolling(20).skew().loc[valid_idx]
        feat_df['kurt_20d'] = ret.rolling(20).kurt().loc[valid_idx]

        # VIX
        if 'VIX' in df.columns:
            feat_df['vix'] = df['VIX'].loc[valid_idx]
            feat_df['vix_ma20'] = df['VIX'].rolling(20).mean().loc[valid_idx]

        # Target: does trend continue profitably over next 20 days?
        future_price = price.shift(-20)
        future_ret = (future_price / price - 1) * trend
        feat_df['future_ret'] = future_ret.loc[valid_idx]
        feat_df['target'] = (future_ret.loc[valid_idx] > 0).astype(float)
        feat_df.loc[future_price.loc[valid_idx].isna(), 'target'] = np.nan
        feat_df.loc[future_price.loc[valid_idx].isna(), 'future_ret'] = np.nan

        feat_df['date'] = feat_df.index
        feat_df = feat_df.reset_index(drop=True)
        all_features.append(feat_df)

    features_df = pd.concat(all_features, ignore_index=True)

    # Cross-asset alignment
    date_dir_counts = features_df.groupby(['date', 'direction']).size().reset_index(name='count')
    features_df = features_df.merge(
        date_dir_counts, on=['date', 'direction'], how='left'
    )
    n_assets = features_df.groupby('date')['ticker'].transform('count')
    features_df['cross_align'] = (features_df['count'] - 1) / (n_assets - 1).clip(lower=1)
    features_df.drop('count', axis=1, inplace=True)

    elapsed = time.time() - t0
    print(f"  Done: {len(features_df)} observations in {elapsed:.1f}s, "
          f"positive target rate: {features_df['target'].mean():.1%}")
    return features_df


def train_ml_filter(features_df, universe_name=""):
    """Walk-forward GBM filter — exact same as ml_portfolio_combo.py."""
    print(f"  [{universe_name}] Training ML walk-forward...")
    t0 = time.time()

    feature_cols = [c for c in features_df.columns
                   if c not in ['ticker', 'date', 'direction', 'target', 'future_ret']]

    features_df = features_df.sort_values('date').reset_index(drop=True)
    features_df['ml_prob'] = np.nan
    dates = sorted(features_df['date'].unique())

    n_folds = 0
    for i in range(TRAIN_WINDOW, len(dates)):
        train_end = dates[i]
        train_start = dates[max(0, i - TRAIN_WINDOW)]

        train_mask = (features_df['date'] >= train_start) & (features_df['date'] < train_end)
        test_mask = features_df['date'] == train_end

        X_train = features_df.loc[train_mask, feature_cols].fillna(0)
        y_train = features_df.loc[train_mask, 'target']
        X_test = features_df.loc[test_mask, feature_cols].fillna(0)

        valid = ~y_train.isna()
        X_train = X_train[valid]
        y_train = y_train[valid]

        if len(X_train) < 50 or len(X_test) == 0:
            continue

        model = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=42
        )
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        features_df.loc[test_mask, 'ml_prob'] = probs
        n_folds += 1

        if n_folds % 500 == 0:
            print(f"    [{universe_name}] Fold {n_folds}/{len(dates)-TRAIN_WINDOW}...")

    elapsed = time.time() - t0
    print(f"  [{universe_name}] {n_folds} folds in {elapsed:.0f}s")

    valid_preds = features_df.dropna(subset=['ml_prob', 'target'])
    if len(valid_preds) > 0:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(valid_preds['target'], valid_preds['ml_prob'])
        print(f"  [{universe_name}] OOS AUC: {auc:.3f}")

    return features_df


def backtest_single_strategy(features_df, df, universe_name=""):
    """Backtest one strategy, return daily return series — exact same as ml_portfolio_combo.py."""
    pred_df = features_df.dropna(subset=['ml_prob']).copy()
    dates = sorted(pred_df['date'].unique())

    portfolio_returns = []
    for date in dates:
        day_signals = pred_df[pred_df['date'] == date]
        high_conf = day_signals[day_signals['ml_prob'] > ML_THRESHOLD]

        if len(high_conf) == 0:
            portfolio_returns.append({'date': date, 'return': 0.0, 'n_pos': 0})
            continue

        positions = []
        for _, row in high_conf.iterrows():
            ticker = row['ticker']
            direction = row['direction']
            vol = max(row['vol_20d'], 0.05)
            weight = (TARGET_VOL / vol) / len(high_conf)
            weight = min(weight, 0.5)

            if ticker in df.columns:
                ticker_dates = df.index
                date_loc = ticker_dates.get_loc(date) if date in ticker_dates else None
                if date_loc is not None and date_loc + 1 < len(ticker_dates):
                    next_ret = df[ticker].iloc[date_loc + 1] / df[ticker].iloc[date_loc] - 1
                    positions.append({
                        'ticker': ticker, 'direction': direction,
                        'weight': weight, 'return': next_ret * direction * weight,
                    })

        if positions:
            day_ret = sum(p['return'] for p in positions)
            turnover = sum(p['weight'] for p in positions)
            cost = turnover * REBAL_COST_BPS / 10000
            day_ret -= cost * 0.1
            portfolio_returns.append({'date': date, 'return': day_ret, 'n_pos': len(positions)})
        else:
            portfolio_returns.append({'date': date, 'return': 0.0, 'n_pos': 0})

    ret_df = pd.DataFrame(portfolio_returns).set_index('date')
    ret_df.index = pd.to_datetime(ret_df.index)
    ret_series = ret_df['return']
    print(f"  [{universe_name}] Trading days: {len(ret_series)}, "
          f"Mean positions/day: {ret_df['n_pos'].mean():.1f}")
    return ret_series


def compute_metrics(returns, name="Strategy"):
    """Compute risk-adjusted metrics — exact same as ml_portfolio_combo.py."""
    r = returns.dropna()
    if len(r) < 30:
        return {}

    mu = r.mean() * 252
    sigma = r.std() * np.sqrt(252)
    sharpe = mu / (sigma + 1e-8)

    downside = r[r < 0].std() * np.sqrt(252)
    sortino = mu / (downside + 1e-8)

    cumret = (1 + r).cumprod()
    total_return = cumret.iloc[-1] - 1
    years = len(r) / 252
    cagr = (cumret.iloc[-1]) ** (1 / years) - 1 if years > 0 else 0

    running_max = cumret.cummax()
    drawdown = cumret / running_max - 1
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    gains = r[r > 0].sum()
    losses = abs(r[r < 0].sum())
    pf = gains / (losses + 1e-8)
    wr = (r > 0).mean()

    return {
        'name': name, 'sharpe': round(sharpe, 3), 'sortino': round(sortino, 3),
        'cagr': round(cagr * 100, 1), 'max_dd': round(max_dd * 100, 1),
        'calmar': round(calmar, 3), 'total_return': round(total_return * 100, 1),
        'profit_factor': round(pf, 3), 'win_rate': round(wr * 100, 1),
        'annual_vol': round(sigma * 100, 1), 'years': round(years, 1),
    }


def compute_yearly_metrics(ret_series, name="Strategy"):
    """Compute year-by-year return, Sharpe, MaxDD."""
    ret_series = ret_series.copy()
    ret_series.index = pd.to_datetime(ret_series.index)

    yearly = {}
    for year in sorted(ret_series.index.year.unique()):
        yr_ret = ret_series[ret_series.index.year == year]
        if len(yr_ret) < 20:
            continue

        # Annual return (compounded daily)
        cumret = (1 + yr_ret).prod() - 1

        # Sharpe (annualized from daily)
        mu = yr_ret.mean() * 252
        sigma = yr_ret.std() * np.sqrt(252)
        sharpe = mu / (sigma + 1e-8)

        # Max drawdown within year
        cum = (1 + yr_ret).cumprod()
        dd = cum / cum.cummax() - 1
        max_dd = dd.min()

        # Win rate
        wr = (yr_ret > 0).mean()

        yearly[str(year)] = {
            'return_pct': round(cumret * 100, 2),
            'sharpe': round(sharpe, 3),
            'max_dd_pct': round(max_dd * 100, 2),
            'win_rate': round(wr * 100, 1),
            'trading_days': len(yr_ret),
        }

    return yearly


def main():
    t0 = time.time()
    print("=" * 80)
    print("YEARLY CAGR EXTRACTION: ML CTA + ML SECTOR ROTATION")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # 1. Data
    df = download_data()

    # 2. Strategy 1: CTA
    print("\n" + "=" * 80)
    print("STRATEGY 1: CTA TREND FOLLOWING")
    print("=" * 80)
    features_cta = build_features_vectorized(df, UNIVERSE_CTA)
    features_cta = train_ml_filter(features_cta, "CTA")
    ret_cta = backtest_single_strategy(features_cta, df, "CTA")

    # 3. Strategy 2: Sectors
    print("\n" + "=" * 80)
    print("STRATEGY 2: SECTOR ROTATION")
    print("=" * 80)
    features_sectors = build_features_vectorized(df, UNIVERSE_SECTORS)
    features_sectors = train_ml_filter(features_sectors, "Sectors")
    ret_sectors = backtest_single_strategy(features_sectors, df, "Sectors")

    # 4. Save daily returns to CSV
    print("\n" + "=" * 80)
    print("SAVING DAILY RETURNS")
    print("=" * 80)

    cta_csv = OUTPUT / "cta_daily_returns.csv"
    ret_cta.to_csv(cta_csv, header=['return'])
    print(f"  CTA daily returns: {len(ret_cta)} days -> {cta_csv}")

    sector_csv = OUTPUT / "sector_daily_returns.csv"
    ret_sectors.to_csv(sector_csv, header=['return'])
    print(f"  Sector daily returns: {len(ret_sectors)} days -> {sector_csv}")

    # 5. Overall metrics (sanity check — should match results.json)
    print("\n" + "=" * 80)
    print("OVERALL METRICS (SANITY CHECK)")
    print("=" * 80)
    m_cta = compute_metrics(ret_cta, "ML CTA Trend")
    m_sectors = compute_metrics(ret_sectors, "ML Sector Rotation")

    # 50/50 combo on common dates
    common = ret_cta.index.intersection(ret_sectors.index)
    combo_50 = 0.5 * ret_cta.loc[common] + 0.5 * ret_sectors.loc[common]
    m_combo = compute_metrics(combo_50, "EW Combo (50/50)")

    print(f"\n  {'Strategy':<25} {'Sharpe':>8} {'Sortino':>8} {'CAGR':>8} {'MaxDD':>8}")
    print(f"  {'-'*55}")
    for m in [m_cta, m_sectors, m_combo]:
        if m:
            print(f"  {m['name']:<25} {m['sharpe']:>8.3f} {m['sortino']:>8.3f} "
                  f"{m['cagr']:>7.1f}% {m['max_dd']:>7.1f}%")

    # Save combo daily returns too
    combo_csv = OUTPUT / "combo_daily_returns.csv"
    combo_50.to_csv(combo_csv, header=['return'])
    print(f"\n  Combo daily returns: {len(combo_50)} days -> {combo_csv}")

    # 6. Year-by-year breakdown
    print("\n" + "=" * 80)
    print("YEAR-BY-YEAR BREAKDOWN")
    print("=" * 80)

    yearly_cta = compute_yearly_metrics(ret_cta, "CTA")
    yearly_sectors = compute_yearly_metrics(ret_sectors, "Sectors")
    yearly_combo = compute_yearly_metrics(combo_50, "Combo")

    # Print table
    all_years = sorted(set(list(yearly_cta.keys()) + list(yearly_sectors.keys()) + list(yearly_combo.keys())))

    print(f"\n  {'Year':<6} | {'CTA Return':>11} {'Sharpe':>7} {'MaxDD':>7} | "
          f"{'Sector Return':>13} {'Sharpe':>7} {'MaxDD':>7} | "
          f"{'Combo Return':>13} {'Sharpe':>7} {'MaxDD':>7}")
    print(f"  {'-'*6}-+-{'-'*27}-+-{'-'*29}-+-{'-'*29}")

    for year in all_years:
        c = yearly_cta.get(year, {})
        s = yearly_sectors.get(year, {})
        cb = yearly_combo.get(year, {})

        c_ret = f"{c.get('return_pct', 0):+.1f}%" if c else "  N/A  "
        c_sh = f"{c.get('sharpe', 0):.2f}" if c else " N/A "
        c_dd = f"{c.get('max_dd_pct', 0):.1f}%" if c else " N/A  "

        s_ret = f"{s.get('return_pct', 0):+.1f}%" if s else "  N/A  "
        s_sh = f"{s.get('sharpe', 0):.2f}" if s else " N/A "
        s_dd = f"{s.get('max_dd_pct', 0):.1f}%" if s else " N/A  "

        cb_ret = f"{cb.get('return_pct', 0):+.1f}%" if cb else "  N/A  "
        cb_sh = f"{cb.get('sharpe', 0):.2f}" if cb else " N/A "
        cb_dd = f"{cb.get('max_dd_pct', 0):.1f}%" if cb else " N/A  "

        print(f"  {year:<6} | {c_ret:>11} {c_sh:>7} {c_dd:>7} | "
              f"{s_ret:>13} {s_sh:>7} {s_dd:>7} | "
              f"{cb_ret:>13} {cb_sh:>7} {cb_dd:>7}")

    # Negative year count
    cta_neg = sum(1 for y in yearly_cta.values() if y['return_pct'] < 0)
    sec_neg = sum(1 for y in yearly_sectors.values() if y['return_pct'] < 0)
    combo_neg = sum(1 for y in yearly_combo.values() if y['return_pct'] < 0)
    print(f"\n  Negative years: CTA={cta_neg}/{len(yearly_cta)}, "
          f"Sectors={sec_neg}/{len(yearly_sectors)}, "
          f"Combo={combo_neg}/{len(yearly_combo)}")

    # Best/worst years
    if yearly_combo:
        best_yr = max(yearly_combo.items(), key=lambda x: x[1]['return_pct'])
        worst_yr = min(yearly_combo.items(), key=lambda x: x[1]['return_pct'])
        print(f"  Combo best year: {best_yr[0]} ({best_yr[1]['return_pct']:+.1f}%)")
        print(f"  Combo worst year: {worst_yr[0]} ({worst_yr[1]['return_pct']:+.1f}%)")

    # 7. SPY benchmark yearly
    spy_ret = df['SPY'].pct_change().reindex(common).fillna(0)
    yearly_spy = compute_yearly_metrics(spy_ret, "SPY")

    print(f"\n  SPY B&H Year-by-Year:")
    for year in all_years:
        sp = yearly_spy.get(year, {})
        if sp:
            print(f"    {year}: {sp['return_pct']:+.1f}%, Sharpe {sp['sharpe']:.2f}, MaxDD {sp['max_dd_pct']:.1f}%")

    # 8. Save to JSON
    output = {
        'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
        'overall_metrics': {
            'cta': m_cta,
            'sectors': m_sectors,
            'combo_50_50': m_combo,
        },
        'yearly': {
            'cta': yearly_cta,
            'sectors': yearly_sectors,
            'combo_50_50': yearly_combo,
            'spy_bh': yearly_spy,
        },
        'negative_years': {
            'cta': cta_neg,
            'sectors': sec_neg,
            'combo': combo_neg,
        },
        'parameters': {
            'ma_short': MA_SHORT, 'ma_long': MA_LONG,
            'ml_threshold': ML_THRESHOLD, 'target_vol': TARGET_VOL,
            'train_window': TRAIN_WINDOW,
            'universe_cta': list(UNIVERSE_CTA.keys()),
            'universe_sectors': list(UNIVERSE_SECTORS.keys()),
        },
        'runtime_seconds': round(time.time() - t0, 1),
    }

    json_out = OUTPUT / 'year_by_year.json'
    with open(json_out, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved: {json_out}")

    elapsed = time.time() - t0
    print(f"\n  Runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == '__main__':
    main()
