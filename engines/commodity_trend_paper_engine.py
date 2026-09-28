#!/usr/bin/env python3
"""
ML Commodity Trend Paper Engine
================================
Generates daily signals for the ML Commodity Trend strategy.
Runs as PM2 cron at 16:35 ET weekdays.

Uses walk-forward GBM to rank commodity ETFs and select top 3.
"""

import json
import warnings
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')

OUTPUT_DIR = Path('/home/jupiter/Lvl3Quant/output/ml_commodity_trend_paper')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = ['GLD', 'SLV', 'USO', 'UNG', 'DBA', 'CPER', 'DBC', 'PDBC']
TRAIN_DAYS = 252

def download_prices():
    """Download latest prices."""
    data = {}
    for t in TICKERS + ['SPY', '^VIX']:
        try:
            df = yf.download(t, period='3y', progress=False, auto_adjust=True)
            if len(df) > 100:
                if isinstance(df.columns, pd.MultiIndex):
                    close = df[('Close', t)].copy()
                else:
                    close = df['Close'].copy()
                clean_name = t.replace('^', '')
                close.name = clean_name
                data[clean_name] = close
        except:
            pass
    return pd.DataFrame(data).dropna(how='all').ffill().dropna()


def build_features(prices_df, ticker):
    """Build features for latest observation."""
    p = prices_df[ticker]
    if len(p) < 252:
        return None

    feats = {}
    for w in [5, 10, 21, 42, 63, 126, 252]:
        if len(p) > w:
            feats[f'ret_{w}d'] = (p.iloc[-1] / p.iloc[-w] - 1) if p.iloc[-w] > 0 else 0

    for w in [10, 21, 50, 100, 200]:
        if len(p) > w:
            ma = p.iloc[-w:].mean()
            feats[f'price_vs_ma{w}'] = (p.iloc[-1] / ma - 1) if ma > 0 else 0

    rets = p.pct_change().dropna()
    for w in [10, 21, 63]:
        if len(rets) > w:
            feats[f'vol_{w}d'] = rets.iloc[-w:].std() * np.sqrt(252)

    if len(rets) > 63:
        vol_21 = rets.iloc[-21:].std()
        vol_63 = rets.iloc[-63:].std()
        feats['vol_ratio_21_63'] = vol_21 / vol_63 if vol_63 > 0 else 1

    if len(p) > 252:
        peak_252 = p.iloc[-252:].max()
        feats['dd_from_252d_peak'] = (p.iloc[-1] / peak_252 - 1) if peak_252 > 0 else 0

    if len(rets) > 14:
        gains = rets.iloc[-14:].clip(lower=0).mean()
        losses = (-rets.iloc[-14:].clip(upper=0)).mean()
        feats['rsi_14'] = 100 * gains / (gains + losses) if (gains + losses) > 0 else 50

    spy = prices_df.get('SPY')
    if spy is not None and len(spy) > 21:
        spy_ret_21 = (spy.iloc[-1] / spy.iloc[-21] - 1) if spy.iloc[-21] > 0 else 0
        feats['rel_strength_vs_spy'] = feats.get('ret_21d', 0) - spy_ret_21
        if len(rets) > 63:
            spy_rets = spy.pct_change().dropna()
            common_idx = rets.index.intersection(spy_rets.index)[-63:]
            if len(common_idx) > 30:
                feats['corr_spy_63d'] = rets.loc[common_idx].corr(spy_rets.loc[common_idx])

    vix = prices_df.get('VIX')
    if vix is not None and len(vix) > 21:
        feats['vix_level'] = vix.iloc[-1]
        feats['vix_ret_21d'] = (vix.iloc[-1] / vix.iloc[-21] - 1) if vix.iloc[-21] > 0 else 0
        if len(vix) > 63:
            feats['vix_percentile_63d'] = (vix.iloc[-1] - vix.iloc[-63:].min()) / \
                                           (vix.iloc[-63:].max() - vix.iloc[-63:].min() + 1e-8)

    if len(p) > 21:
        feats['zscore_21d'] = (p.iloc[-1] - p.iloc[-21:].mean()) / (p.iloc[-21:].std() + 1e-8)

    return feats


def main():
    today = dt.date.today()
    print(f"\n{'='*50}")
    print(f"ML COMMODITY TREND — Paper Signal {today}")
    print(f"{'='*50}")

    prices = download_prices()
    valid = [t for t in TICKERS if t in prices.columns]
    print(f"Valid tickers: {valid}")

    if len(valid) < 3:
        print("ERROR: Not enough tickers")
        return

    try:
        import lightgbm as lgb
        USE_LGB = True
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        USE_LGB = False

    # Build training data from historical features + labels
    all_feats = []
    dates = prices.index.tolist()

    for i in range(TRAIN_DAYS, len(dates) - 21):
        for ticker in valid:
            feats = build_features(prices.iloc[:i+1], ticker)
            if feats is None:
                continue
            future_ret = (prices[ticker].iloc[i + 21] / prices[ticker].iloc[i]) - 1
            basket_ret = np.mean([(prices[t].iloc[i + 21] / prices[t].iloc[i]) - 1
                                  for t in valid])
            feats['_label'] = 1 if future_ret > basket_ret else 0
            feats['_ticker'] = ticker
            all_feats.append(feats)

    df = pd.DataFrame(all_feats)
    feature_cols = [c for c in df.columns if not c.startswith('_')]

    # Train on all available data
    X = df[feature_cols].fillna(0).values
    y = df['_label'].values

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

    model.fit(X, y)

    # Generate today's signal
    signals = []
    for ticker in valid:
        feats = build_features(prices, ticker)
        if feats is None:
            continue
        X_today = pd.DataFrame([feats])[feature_cols].fillna(0).values
        prob = model.predict_proba(X_today)[0, 1]
        signals.append({
            'ticker': ticker,
            'outperform_prob': round(float(prob), 3),
            'price': round(float(prices[ticker].iloc[-1]), 2),
            'ret_21d': round(float((prices[ticker].iloc[-1] / prices[ticker].iloc[-21] - 1) * 100), 1)
        })

    signals.sort(key=lambda x: x['outperform_prob'], reverse=True)

    # Top 3 = positions
    top_3 = signals[:3]
    print(f"\nTOP 3 COMMODITY PICKS:")
    for s in top_3:
        print(f"  {s['ticker']}: {s['outperform_prob']:.0%} prob, ${s['price']}, {s['ret_21d']}% 21d ret")

    print(f"\nAll signals:")
    for s in signals:
        marker = "✅" if s in top_3 else "  "
        print(f"  {marker} {s['ticker']}: {s['outperform_prob']:.0%}")

    # Save signal
    result = {
        'date': str(today),
        'signals': signals,
        'positions': [s['ticker'] for s in top_3],
        'timestamp': dt.datetime.now().isoformat()
    }

    signal_file = OUTPUT_DIR / f'signal_{today}.json'
    with open(signal_file, 'w') as f:
        json.dump(result, f, indent=2)

    # Append to history
    history_file = OUTPUT_DIR / 'signal_history.jsonl'
    with open(history_file, 'a') as f:
        f.write(json.dumps(result) + '\n')

    print(f"\nSignal saved to {signal_file}")


if __name__ == '__main__':
    main()
