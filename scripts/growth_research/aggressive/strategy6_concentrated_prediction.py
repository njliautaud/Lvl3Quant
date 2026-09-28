#!/usr/bin/env python3
"""
Strategy 6: Concentrated Best-Ideas (Stock Prediction Enhanced)
- Build a simple cross-sectional relative return predictor
- Only trade at highest confidence levels (>=70%)
- Concentrated: 1-2 stocks at a time
- Walk-forward sliding window
- This simulates what our stock prediction v2 model does
"""
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = "/home/jupiter/Lvl3Quant/output/growth_research/aggressive"

UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'INTC', 'CRM',
    'ADBE', 'NFLX', 'AVGO', 'QCOM', 'JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'LLY',
    'JPM', 'BAC', 'WFC', 'GS', 'WMT', 'PG', 'KO', 'PEP', 'COST', 'MCD',
    'HD', 'CAT', 'BA', 'HON', 'XOM', 'CVX', 'COP', 'V', 'MA', 'DIS',
    'ORCL', 'ACN', 'NOW', 'ISRG', 'REGN', 'LIN', 'NEE', 'UNP', 'RTX', 'DE',
]

def download_data():
    print("Downloading data for concentrated prediction strategy...")
    end = datetime(2026, 7, 1)
    start = datetime(2019, 1, 1)
    data = yf.download(UNIVERSE + ['SPY'], start=start, end=end, auto_adjust=True, progress=False)
    close = data['Close'].dropna(how='all')
    volume = data['Volume'].dropna(how='all')
    high = data['High'].dropna(how='all')
    low = data['Low'].dropna(how='all')
    return close, volume, high, low

def compute_features(close, volume, high, low, ticker):
    """Compute features for a single stock"""
    c = close[ticker].dropna()
    spy = close['SPY'].dropna()

    common = c.index.intersection(spy.index)
    if len(common) < 252:
        return None

    c = c.loc[common]
    spy_c = spy.loc[common]

    features = pd.DataFrame(index=common)

    # Momentum features (relative to SPY)
    for w in [5, 10, 20, 60]:
        stock_ret = c.pct_change(w)
        spy_ret = spy_c.pct_change(w)
        features[f'rel_mom_{w}d'] = stock_ret - spy_ret

    # Mean reversion
    for w in [5, 10, 20]:
        ma = c.rolling(w).mean()
        features[f'dist_ma_{w}d'] = (c - ma) / ma

    # Volatility features
    log_ret = np.log(c / c.shift(1))
    for w in [10, 20]:
        features[f'vol_{w}d'] = log_ret.rolling(w).std()

    # Volume features
    if ticker in volume.columns:
        v = volume[ticker].loc[common]
        v_ma = v.rolling(20).mean()
        features['vol_ratio'] = v / v_ma

    # RSI
    delta = c.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    features['rsi_14'] = 100 - (100 / (1 + rs))

    # ATR ratio (if high/low available)
    if ticker in high.columns and ticker in low.columns:
        h = high[ticker].loc[common]
        l = low[ticker].loc[common]
        tr = pd.concat([h - l, abs(h - c.shift(1)), abs(l - c.shift(1))], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        features['atr_pct'] = atr / c

    # Target: relative outperformance over next 20 days
    fwd_ret = c.pct_change(20).shift(-20)
    spy_fwd = spy_c.pct_change(20).shift(-20)
    rel_fwd = fwd_ret - spy_fwd

    # Binary: outperforms SPY by >5%
    features['target'] = (rel_fwd > 0.05).astype(int)
    features['fwd_abs_ret'] = fwd_ret  # For PnL calculation

    features['ticker'] = ticker

    return features.dropna()

def walk_forward_predict(close, volume, high, low,
                          train_months=12, hold_days=20, min_confidence=0.70, max_positions=1):
    """
    Walk-forward prediction with sliding window.
    Train on train_months, predict next month, slide.
    """
    # Build feature matrix for all stocks
    print("  Building features...")
    all_features = []
    for ticker in UNIVERSE:
        if ticker not in close.columns:
            continue
        feat = compute_features(close, volume, high, low, ticker)
        if feat is not None:
            all_features.append(feat)

    if not all_features:
        return None, None

    full_df = pd.concat(all_features)
    feature_cols = [c for c in full_df.columns if c not in ['target', 'fwd_abs_ret', 'ticker']]

    # Monthly walk-forward
    months = pd.date_range(full_df.index.min(), full_df.index.max(), freq='ME')
    train_window = train_months

    trades = []
    portfolio_returns = []
    dates = []

    for i in range(train_window, len(months) - 1):
        train_start = months[i - train_window]
        train_end = months[i]
        test_month = months[i + 1] if i + 1 < len(months) else months[i] + pd.DateOffset(months=1)

        # Training data
        train_mask = (full_df.index >= train_start) & (full_df.index < train_end)
        train_data = full_df.loc[train_mask]

        if len(train_data) < 100:
            continue

        X_train = train_data[feature_cols].values
        y_train = train_data['target'].values

        if y_train.sum() < 10 or (1 - y_train).sum() < 10:
            continue

        # Scale
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)

        # Train
        model = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            min_samples_leaf=20, random_state=42
        )
        model.fit(X_train_s, y_train)

        # Predict on test month (use last available day of the month)
        test_mask = (full_df.index >= train_end) & (full_df.index < test_month)
        test_data = full_df.loc[test_mask]

        if len(test_data) == 0:
            continue

        # Get predictions for the last day of the test window for each stock
        last_day_data = test_data.groupby('ticker').last()
        if len(last_day_data) == 0:
            continue

        X_test = last_day_data[feature_cols].values
        X_test_s = scaler.transform(X_test)

        proba = model.predict_proba(X_test_s)[:, 1]
        last_day_data['confidence'] = proba

        # Filter by confidence threshold
        high_conf = last_day_data[last_day_data['confidence'] >= min_confidence]

        if len(high_conf) == 0:
            continue

        # Pick top max_positions
        top_picks = high_conf.nlargest(max_positions, 'confidence')

        for ticker, row in top_picks.iterrows():
            trades.append({
                'ticker': ticker,
                'month': str(train_end.date()),
                'confidence': float(row['confidence']),
                'fwd_return': float(row['fwd_abs_ret']) if pd.notna(row['fwd_abs_ret']) else 0,
            })

        # Average return of picks
        avg_ret = top_picks['fwd_abs_ret'].mean()
        if pd.notna(avg_ret):
            portfolio_returns.append(avg_ret)
            dates.append(test_month)

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    returns = pd.Series(portfolio_returns, index=dates) if portfolio_returns else pd.Series(dtype=float)

    return returns, trades_df

def analyze(returns, trades_df, name, spy_close):
    if returns is None or len(returns) < 12:
        return None

    n_periods = len(returns)
    n_years = n_periods / 12  # monthly periods

    cum_ret = (1 + returns).prod() - 1
    cagr = (1 + cum_ret) ** (1 / n_years) - 1 if n_years > 0 else 0

    ann_ret = returns.mean() * 12
    ann_vol = returns.std() * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    downside = returns[returns < 0].std() * np.sqrt(12)
    sortino = ann_ret / downside if downside > 0 else 0

    wr = (returns > 0).mean()
    avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
    avg_loss = abs(returns[returns < 0].mean()) if (returns < 0).any() else 1
    pf = avg_win / avg_loss if avg_loss > 0 else float('inf')

    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Trade-level stats
    n_trades = len(trades_df)
    trade_wr = (trades_df['fwd_return'] > 0).mean() if n_trades > 0 else 0
    avg_confidence = trades_df['confidence'].mean() if n_trades > 0 else 0

    # R1 regime test
    spy_monthly = spy_close.resample('ME').last().pct_change()
    common_idx = returns.index.intersection(spy_monthly.index)
    r1_pass = False
    sharpe_green = sharpe_red = regime_gap = 0

    if len(common_idx) > 10:
        sm = returns.loc[common_idx]
        spy_m = spy_monthly.loc[common_idx]
        green = sm[spy_m > 0.01]
        red = sm[spy_m < -0.01]
        sharpe_green = green.mean() / green.std() * np.sqrt(12) if len(green) > 2 and green.std() > 0 else 0
        sharpe_red = red.mean() / red.std() * np.sqrt(12) if len(red) > 2 and red.std() > 0 else 0
        max_s = max(abs(sharpe_green), abs(sharpe_red))
        regime_gap = abs(sharpe_green - sharpe_red) / max_s if max_s > 0 else 0
        r1_pass = regime_gap <= 0.50

    # Permutation test
    perm_sharpes = []
    for _ in range(100):
        perm = returns.sample(frac=1, replace=False).values
        pm = perm.mean() * 12
        ps = perm.std() * np.sqrt(12)
        perm_sharpes.append(pm / ps if ps > 0 else 0)
    perm_p = np.mean([s >= sharpe for s in perm_sharpes])

    final_441 = 441 * (1 + cum_ret)

    return {
        'strategy': name,
        'n_trades': n_trades,
        'n_periods': n_periods,
        'n_years': round(n_years, 1),
        'CAGR': round(cagr * 100, 2),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'win_rate': round(wr * 100, 1),
        'trade_win_rate': round(trade_wr * 100, 1),
        'avg_confidence': round(avg_confidence, 3),
        'profit_factor': round(pf, 3),
        'max_drawdown': round(max_dd * 100, 2),
        'calmar': round(calmar, 3),
        'cum_return': round(cum_ret * 100, 2),
        'final_441': round(final_441, 2),
        'R1_regime_gap': round(regime_gap, 3),
        'R1_pass': r1_pass,
        'sharpe_green': round(sharpe_green, 3),
        'sharpe_red': round(sharpe_red, 3),
        'perm_p_value': round(perm_p, 3),
        'feasible_441': True,
    }

def main():
    close, volume, high, low = download_data()
    spy = close['SPY']

    configs = [
        {'train_months': 12, 'min_confidence': 0.70, 'max_positions': 1, 'name': 'Concentrated_70conf_1pos'},
        {'train_months': 12, 'min_confidence': 0.70, 'max_positions': 2, 'name': 'Concentrated_70conf_2pos'},
        {'train_months': 12, 'min_confidence': 0.60, 'max_positions': 2, 'name': 'Concentrated_60conf_2pos'},
        {'train_months': 12, 'min_confidence': 0.80, 'max_positions': 1, 'name': 'Concentrated_80conf_1pos'},
        {'train_months': 6, 'min_confidence': 0.70, 'max_positions': 1, 'name': 'Concentrated_70conf_6mtrain'},
        {'train_months': 24, 'min_confidence': 0.70, 'max_positions': 1, 'name': 'Concentrated_70conf_24mtrain'},
    ]

    results = []
    for cfg in configs:
        name = cfg.pop('name')
        print(f"\nTesting {name}...")
        ret, trades = walk_forward_predict(close, volume, high, low, **cfg)
        if ret is not None and len(ret) > 0:
            r = analyze(ret, trades, name, spy)
            if r:
                results.append(r)
                print(f"  CAGR={r['CAGR']}%, Sharpe={r['sharpe']}, Sortino={r['sortino']}, "
                      f"WR={r['win_rate']}%, Trades={r['n_trades']}, "
                      f"R1={'PASS' if r['R1_pass'] else 'FAIL'}, $441->{r['final_441']}")

    with open(f"{OUTPUT_DIR}/strategy6_concentrated_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "="*80)
    print("STRATEGY 6: CONCENTRATED PREDICTION — RESULTS")
    print("="*80)
    if results:
        df = pd.DataFrame(results)
        print(df[['strategy', 'CAGR', 'sharpe', 'sortino', 'n_trades', 'win_rate',
                  'max_drawdown', 'R1_pass', 'final_441', 'perm_p_value']].to_string(index=False))

    return results

if __name__ == '__main__':
    results = main()
