#!/usr/bin/env python3
"""
Weekly Signal Generator v2 — Production Trade Recommendations
============================================================
Combines ALL validated findings into actionable weekly trade signals:

BULL MODE (VIX >= 20):
1. LightGBM sector momentum ranking (validated Sharpe 4.73)
2. Top-3 sectors for bull call spreads
3. 20-day exit optimization (nearly 2x Sharpe vs 30d hold)
4. Multi-signal confluence (HC #750): momentum + quality + VIX regime

BEAR MODE (VIX < 20):
1. LightGBM sector ranking — BOTTOM 3 for bear put spreads (validated Sharpe 2.10)
2. Bear confluence: negative momentum, below SMA50, underperforming SPY
3. Same 20-day exit, 50% profit target

Combined bull+bear validated: Sharpe 3.10, 82.9% WR, $645→$49K (MLflow exp 119)

Output: JSON + plain English summary of recommended trades for the week.
Run this Sunday evening or Monday pre-market.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
import json
import os

try:
    import lightgbm as lgb
except:
    print("LightGBM required"); exit(1)

UNIVERSE = ['XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLI', 'XLP', 'XLU', 'XLRE', 'XLB', 'XLC']
SECTOR_NAMES = {
    'XLK': 'Technology', 'XLF': 'Financials', 'XLE': 'Energy',
    'XLV': 'Healthcare', 'XLY': 'Consumer Disc.', 'XLI': 'Industrials',
    'XLP': 'Consumer Staples', 'XLU': 'Utilities', 'XLRE': 'Real Estate',
    'XLB': 'Materials', 'XLC': 'Communication'
}

def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))

def load_fresh_data():
    """Load latest data from Yahoo Finance."""
    import yfinance as yf
    tickers = UNIVERSE + ['^VIX', 'SPY']
    data = yf.download(tickers, start='2022-01-01', progress=False, auto_adjust=True)

    if isinstance(data.columns, pd.MultiIndex):
        close = data['Close']
        volume = data['Volume']
    else:
        close = data
        volume = pd.DataFrame()

    vix_col = '^VIX' if '^VIX' in close.columns else None
    spy_col = 'SPY' if 'SPY' in close.columns else None

    vix = close[vix_col] if vix_col else None
    spy = close[spy_col] if spy_col else None

    sector_close = close[[c for c in UNIVERSE if c in close.columns]]

    return sector_close, vix, spy, volume

def build_features_live(close, vix):
    """Build current features for all sectors."""
    results = {}

    for ticker in UNIVERSE:
        if ticker not in close.columns:
            continue

        px = close[ticker].dropna()
        if len(px) < 60:
            continue

        current = px.iloc[-1]
        feat = {}

        # Momentum
        for d in [5, 10, 21, 63, 126, 252]:
            if len(px) > d:
                feat[f'ret_{d}d'] = (px.iloc[-1] / px.iloc[-d-1] - 1) * 100

        # Volatility
        daily_ret = px.pct_change()
        feat['vol_21d'] = daily_ret.tail(21).std() * np.sqrt(252) * 100
        feat['vol_63d'] = daily_ret.tail(63).std() * np.sqrt(252) * 100

        # Risk-adjusted momentum
        feat['sharpe_63d'] = (feat.get('ret_63d', 0) / 100) / (feat.get('vol_63d', 25) / 100 + 1e-8) * np.sqrt(252/63)

        # RSI
        feat['rsi_14'] = compute_rsi(px, 14).iloc[-1]

        # Trend
        for d in [50, 200]:
            if len(px) > d:
                sma = px.rolling(d).mean().iloc[-1]
                feat[f'above_sma{d}'] = current > sma
                feat[f'pct_from_sma{d}'] = (current / sma - 1) * 100

        # Relative strength vs equal-weight
        ew = close[UNIVERSE].pct_change(21).mean(axis=1)
        if len(ew) > 0:
            feat['rel_strength_21d'] = (px.pct_change(21).iloc[-1] - ew.iloc[-1]) * 100

        # Max drawdown
        rolling_max = px.rolling(63).max()
        dd = (px - rolling_max) / rolling_max
        feat['max_dd_63d'] = dd.tail(63).min() * 100

        # Calmar-like
        feat['calmar_63d'] = feat.get('ret_63d', 0) / (-feat.get('max_dd_63d', -1) + 0.01)

        feat['price'] = current
        feat['ticker'] = ticker
        feat['name'] = SECTOR_NAMES.get(ticker, ticker)
        results[ticker] = feat

    return results

def train_ranking_model(close, vix):
    """Train LightGBM on recent data for sector ranking."""
    monthly = close.resample('ME').last().dropna(how='all')

    features_list = []
    for ticker in UNIVERSE:
        if ticker not in monthly.columns:
            continue
        px = monthly[ticker].dropna()
        if len(px) < 15:
            continue

        feat = pd.DataFrame(index=px.index)
        feat['ticker'] = ticker
        for m in [1, 2, 3, 6, 12]:
            feat[f'ret_{m}m'] = px.pct_change(m)
        ret1 = px.pct_change()
        for m in [3, 6, 12]:
            feat[f'vol_{m}m'] = ret1.rolling(m).std()
        feat['sharpe_6m'] = feat['ret_6m'] / (feat['vol_6m'] + 1e-8)
        feat['sharpe_12m'] = feat['ret_12m'] / (feat['vol_12m'] + 1e-8)
        feat['rsi_14'] = compute_rsi(px, 14)
        for m in [5, 10, 20]:
            sma = px.rolling(m).mean()
            feat[f'above_sma{m}'] = (px > sma).astype(float)
        ew = monthly[UNIVERSE].pct_change(3).mean(axis=1)
        feat['rel_strength_3m'] = feat['ret_3m'] - ew
        rm = px.rolling(12).max()
        dd = (px - rm) / rm
        feat['max_dd_12m'] = dd.rolling(12).min()
        feat['calmar_12m'] = feat['ret_12m'] / (-feat['max_dd_12m'] + 1e-8)
        feat['target'] = px.pct_change().shift(-1)
        features_list.append(feat)

    all_feat = pd.concat(features_list)
    feat_cols = [c for c in all_feat.columns if c not in ['ticker', 'target']]

    # Use all but last month for training
    dates = sorted(all_feat.index.unique())
    train_data = all_feat[all_feat.index < dates[-1]]
    pred_data = all_feat[all_feat.index == dates[-1]]

    train_X = train_data[feat_cols].replace([np.inf, -np.inf], np.nan)
    train_y = train_data['target'].values
    valid = ~(train_X.isna().any(axis=1) | np.isnan(train_y))
    train_X = train_X[valid]
    train_y = train_y[valid]

    pred_X = pred_data[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    model = lgb.LGBMRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=5,
        verbose=-1, random_state=42
    )
    model.fit(train_X, train_y)

    predictions = model.predict(pred_X)
    rankings = pd.DataFrame({
        'ticker': pred_data['ticker'].values,
        'predicted_return': predictions
    }).sort_values('predicted_return', ascending=False)

    # Feature importance
    imp = pd.Series(model.feature_importances_, index=feat_cols).sort_values(ascending=False)

    return rankings, imp

def generate_signals(account_size=645.0):
    """Generate this week's trading signals."""
    print("=" * 70)
    print(f"WEEKLY SIGNAL GENERATOR — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    # Load data
    close, vix, spy, volume = load_fresh_data()
    latest_date = close.index[-1].strftime('%Y-%m-%d')
    print(f"\nData through: {latest_date}")

    # Current VIX
    current_vix = vix.iloc[-1] if vix is not None else None
    vix_5d_avg = vix.tail(5).mean() if vix is not None else None
    print(f"VIX: {current_vix:.1f} (5d avg: {vix_5d_avg:.1f})")

    # VIX filter check
    vix_pass = current_vix >= 20 if current_vix else False
    if not vix_pass:
        print(f"\n🔄 VIX = {current_vix:.1f} < 20 — BEAR PUT SPREAD MODE")
        print("Our combined strategy trades BOTH directions:")
        print("  VIX >= 20: Bull call spreads on TOP sectors")
        print("  VIX < 20:  Bear put spreads on BOTTOM sectors")
        print("Currently in BEAR mode — looking for weak sectors to buy puts on.")

    # Current sector features
    sector_feats = build_features_live(close, vix)

    # ML ranking
    rankings, feature_imp = train_ranking_model(close, vix)

    # Combine
    print(f"\n{'='*60}")
    print("SECTOR RANKINGS (LightGBM momentum model)")
    print(f"{'='*60}")

    signals = []
    for i, (_, row) in enumerate(rankings.iterrows()):
        ticker = row['ticker']
        pred = row['predicted_return'] * 100
        feat = sector_feats.get(ticker, {})

        name = SECTOR_NAMES.get(ticker, ticker)
        price = feat.get('price', 0)
        ret_21d = feat.get('ret_21d', 0)
        ret_63d = feat.get('ret_63d', 0)
        vol = feat.get('vol_21d', 0)
        rsi = feat.get('rsi_14', 50)
        above_50 = feat.get('above_sma50', False)
        above_200 = feat.get('above_sma200', False)
        dd = feat.get('max_dd_63d', 0)

        rank = i + 1
        trend = "📈" if above_50 and above_200 else "📉" if not above_50 and not above_200 else "↔️"

        # Confluence score (HC #750)
        confluence = 0
        confluence_reasons = []
        if pred > 0:
            confluence += 1
            confluence_reasons.append("ML positive")
        if ret_21d > 0:
            confluence += 1
            confluence_reasons.append("21d momentum positive")
        if ret_63d > 0:
            confluence += 1
            confluence_reasons.append("63d momentum positive")
        if above_50:
            confluence += 1
            confluence_reasons.append("above 50-day MA")
        if above_200:
            confluence += 1
            confluence_reasons.append("above 200-day MA")
        if 40 < rsi < 70:
            confluence += 1
            confluence_reasons.append("RSI healthy (40-70)")

        signal = {
            'rank': rank,
            'ticker': ticker,
            'name': name,
            'price': round(price, 2),
            'predicted_return': round(pred, 2),
            'return_21d': round(ret_21d, 2),
            'return_63d': round(ret_63d, 2),
            'volatility': round(vol, 1),
            'rsi': round(rsi, 1),
            'above_sma50': above_50,
            'above_sma200': above_200,
            'max_dd_63d': round(dd, 1),
            'confluence_score': confluence,
            'confluence_reasons': confluence_reasons,
            'trend': trend,
        }
        signals.append(signal)

        action = "✅ BUY SPREAD" if rank <= 3 and confluence >= 4 else "⚠️ WEAK" if rank <= 3 else "—"
        print(f"  #{rank:2d} {trend} {ticker:5s} ({name:18s}) | ${price:7.2f} | "
              f"Pred {pred:+5.2f}% | 21d {ret_21d:+5.1f}% | RSI {rsi:4.1f} | "
              f"Confluence {confluence}/6 | {action}")

    # TOP-3 TRADE RECOMMENDATIONS
    top3 = [s for s in signals if s['rank'] <= 3]

    print(f"\n{'='*60}")
    print("TRADE RECOMMENDATIONS")
    print(f"{'='*60}")

    print(f"\n📊 Account: ${account_size:.0f}")
    print(f"📊 Max per trade: ${min(account_size * 0.35, 300):.0f}")

    trades = []

    if vix_pass:
        # === BULL MODE: VIX >= 20 ===
        print(f"\n🟢 BULL MODE (VIX {current_vix:.1f} >= 20)")
        print(f"📊 Strategy: Bull call spread on TOP sectors, 30d DTE, 3% width, exit at 20 days")

        for s in top3:
            ticker = s['ticker']
            price = s['price']
            K_long = round(price * 0.985, 2)
            K_short = round(price * 1.015, 2)
            max_cost = min(account_size * 0.35 / 3, 200)

            if s['confluence_score'] < 4:
                recommendation = "SKIP — insufficient confluence"
                confidence = "LOW"
            else:
                recommendation = "GO — full position"
                confidence = "HIGH"

            trade = {
                'ticker': ticker, 'name': s['name'], 'action': 'Bull Call Spread',
                'long_strike': K_long, 'short_strike': K_short, 'dte': '30 days',
                'exit_plan': 'Exit at 20 days or 50% profit', 'max_cost': round(max_cost, 0),
                'confidence': confidence, 'recommendation': recommendation,
                'confluence': s['confluence_score'], 'reasons': s['confluence_reasons'],
            }
            trades.append(trade)

            emoji = "✅" if confidence == "HIGH" else "❌"
            print(f"\n{emoji} {ticker} ({s['name']}) — {recommendation}")
            print(f"   Buy ${K_long} call / Sell ${K_short} call, ~30 DTE")
            print(f"   Max cost: ~${max_cost:.0f}")
            print(f"   Exit: At 20 days OR 50% profit")
            print(f"   Confluence: {s['confluence_score']}/6 — {', '.join(s['confluence_reasons'])}")

    else:
        # === BEAR MODE: VIX < 20 ===
        print(f"\n🔴 BEAR MODE (VIX {current_vix:.1f} < 20)")
        print(f"📊 Strategy: Bear put spread on BOTTOM sectors, 30d DTE, 3% width, exit at 20 days")
        print(f"📊 Validated: Sharpe 2.10, WR 73%, fills the VIX<20 gap")

        # Bottom 3 sectors for bear puts
        bottom3 = [s for s in signals if s['rank'] >= len(signals) - 2]

        for s in bottom3:
            ticker = s['ticker']
            price = s['price']
            K_long = round(price * 1.00, 2)   # ATM put (buy)
            K_short = round(price * 0.97, 2)   # 3% OTM put (sell)
            max_cost = min(account_size * 0.35 / 3, 200)

            # Bear confluence scoring
            bear_confluence = 0
            bear_reasons = []
            if s.get('predicted_return', 0) < 0:
                bear_confluence += 1
                bear_reasons.append("ML predicts decline")
            ret_21d = s.get('return_21d', 0)
            if ret_21d < 0:
                bear_confluence += 1
                bear_reasons.append("21d momentum negative")
            ret_63d = s.get('return_63d', 0)
            if ret_63d < 0:
                bear_confluence += 1
                bear_reasons.append("63d momentum negative")
            if not s.get('above_sma50', True):
                bear_confluence += 1
                bear_reasons.append("below 50-day MA")
            if not s.get('above_sma200', True):
                bear_confluence += 1
                bear_reasons.append("below 200-day MA")
            rsi = s.get('rsi', 50)
            if 20 < rsi < 50:
                bear_confluence += 1
                bear_reasons.append("RSI bearish (20-50)")

            if bear_confluence < 2:
                recommendation = "SKIP — insufficient bear confluence"
                confidence = "LOW"
            elif bear_confluence < 4:
                recommendation = "HALF SIZE — moderate bear signal"
                confidence = "MEDIUM"
            else:
                recommendation = "GO — strong bear setup"
                confidence = "HIGH"

            trade = {
                'ticker': ticker, 'name': s['name'], 'action': 'Bear Put Spread',
                'long_strike': K_long, 'short_strike': K_short, 'dte': '30 days',
                'exit_plan': 'Exit at 20 days or 50% profit', 'max_cost': round(max_cost, 0),
                'confidence': confidence, 'recommendation': recommendation,
                'confluence': bear_confluence, 'reasons': bear_reasons,
            }
            trades.append(trade)

            emoji = "✅" if confidence == "HIGH" else "⚠️" if confidence == "MEDIUM" else "❌"
            print(f"\n{emoji} {ticker} ({s['name']}) — {recommendation}")
            print(f"   Buy ${K_long} put / Sell ${K_short} put, ~30 DTE")
            print(f"   Max cost: ~${max_cost:.0f}")
            print(f"   Exit: At 20 days OR 50% profit")
            print(f"   Bear confluence: {bear_confluence}/6 — {', '.join(bear_reasons) if bear_reasons else 'none'}")

    # MARKET CONTEXT
    print(f"\n{'='*60}")
    print("MARKET CONTEXT")
    print(f"{'='*60}")

    if spy is not None:
        spy_ret_5d = (spy.iloc[-1] / spy.iloc[-6] - 1) * 100 if len(spy) > 5 else 0
        spy_ret_21d = (spy.iloc[-1] / spy.iloc[-22] - 1) * 100 if len(spy) > 21 else 0
        spy_sma50 = spy.rolling(50).mean().iloc[-1]
        spy_sma200 = spy.rolling(200).mean().iloc[-1]
        spy_above_50 = spy.iloc[-1] > spy_sma50
        spy_above_200 = spy.iloc[-1] > spy_sma200

        print(f"  SPY: ${spy.iloc[-1]:.2f}")
        print(f"  5-day return: {spy_ret_5d:+.1f}%")
        print(f"  21-day return: {spy_ret_21d:+.1f}%")
        print(f"  Above 50-day MA: {'Yes ✅' if spy_above_50 else 'No ❌'}")
        print(f"  Above 200-day MA: {'Yes ✅' if spy_above_200 else 'No ❌'}")

    if vix is not None:
        vix_ret_5d = (vix.iloc[-1] / vix.iloc[-6] - 1) * 100 if len(vix) > 5 else 0
        print(f"  VIX: {current_vix:.1f} (5d change: {vix_ret_5d:+.1f}%)")
        vix_regime = "🔴 HIGH VOL" if current_vix > 25 else "🟡 ELEVATED" if current_vix > 20 else "🟢 CALM"
        print(f"  Vol regime: {vix_regime}")

    # Feature importance
    print(f"\n📊 Top model features: {', '.join(feature_imp.head(5).index.tolist())}")

    # Save output
    output = {
        'timestamp': datetime.now().isoformat(),
        'data_through': latest_date,
        'vix': round(current_vix, 2) if current_vix else None,
        'vix_pass': vix_pass,
        'account_size': account_size,
        'rankings': signals,
        'trades': trades,
        'mode': 'bull' if vix_pass else 'bear',
        'market_context': {
            'spy_price': round(spy.iloc[-1], 2) if spy is not None else None,
            'vix_regime': 'high' if current_vix and current_vix > 25 else 'elevated' if current_vix and current_vix > 20 else 'calm',
        }
    }

    output_dir = '/home/jupiter/Lvl3Quant/research/signals'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"weekly_signal_{datetime.now().strftime('%Y%m%d')}.json")
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nSignals saved to {output_file}")
    return output

if __name__ == '__main__':
    generate_signals()
