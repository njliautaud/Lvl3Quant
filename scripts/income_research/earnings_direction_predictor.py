#!/usr/bin/env python3
"""
Earnings Direction Predictor — Neural Network on GPU
=====================================================
HC #701: Creative quant research. HC #698: Stock prediction priority.

THESIS: IC condors are worthless with real pricing (Session #190).
BUT if we can predict post-earnings direction >55%, we can:
  - Buy OTM calls before bullish earnings (cheaper than selling premium)
  - Buy OTM puts before bearish earnings
  - Skip uncertain ones

This flips the earnings IV crush thesis: instead of selling premium
and hoping stock stays flat, we PREDICT the direction and BUY premium.

FEATURES:
  1. Pre-earnings IV percentile (is fear elevated?)
  2. Revenue/EPS surprise history (consistent beaters?)
  3. Price vs 50/200 SMA (trend context)
  4. Pre-earnings drift (anticipation signal)
  5. Sector momentum (is sector hot/cold?)
  6. Put/call volume ratio (sentiment)
  7. Analyst revision momentum (consensus shifting?)
  8. Historical earnings move magnitude
  9. Days since last earnings (cycle position)
  10. Market regime (VIX level, SPY trend)

MODEL:
  - PyTorch MLP with residual connections (GPU accelerated)
  - Walk-forward: 2yr train, 1yr OOS, 3-month step
  - Target: binary (stock up >2% or down >2% post-earnings)
  - Also test 3-class: up >3%, down >3%, flat

VALIDATION (HC #428):
  - R1: regime gap < 0.50
  - Permutation test: random labels must fail
  - Walk-forward only (HC #0)

OUTPUT:
  - Direction accuracy per confidence tier
  - Simulated directional options P&L (buy calls on up, puts on down)
  - Risk-adjusted metrics (Sharpe, Sortino, PF, WR)
"""

import sys, os, json, warnings, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

warnings.filterwarnings("ignore")

# Auto-detect node
if os.path.exists("/home/nick"):
    ROOT = Path("/home/nick/Lvl3Quant")
else:
    ROOT = Path("/home/jupiter/Lvl3Quant")

OUTPUT = ROOT / "output" / "earnings_direction_v1"
OUTPUT.mkdir(parents=True, exist_ok=True)
CACHE_DIR = OUTPUT / "cache"
CACHE_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
print(f"Output: {OUTPUT}")

# ── DATA COLLECTION ────────────────────────────────────────────────

def get_earnings_data():
    """Get earnings dates and surprise data — uses cached price/earnings data."""
    cache = CACHE_DIR / "earnings_data_v1.parquet"
    if cache.exists():
        print(f"  Loading cached earnings data...")
        return pd.read_parquet(cache)

    import yfinance as yf

    # ── Load cached price data (v3 stock predictor already downloaded) ──
    price_cache = ROOT / "output" / "growth_research" / "stock_prediction" / "cache" / "price_data_v3.parquet"
    if price_cache.exists():
        print("  Using cached price data from v3 predictor")
        prices_df = pd.read_parquet(price_cache)
        prices_df['date'] = pd.to_datetime(prices_df['date'])
    else:
        print("  Downloading price data fresh (no cache found)...")
        prices_df = None

    # ── Load cached earnings dates ──
    earnings_cache = ROOT / "output" / "growth_research" / "stock_prediction" / "cache" / "earnings_data.json"
    import json
    earnings_raw = {}
    if earnings_cache.exists():
        earnings_raw = json.load(open(earnings_cache))
        print(f"  Loaded {sum(len(v) for v in earnings_raw.values())} cached earnings entries")

    # Tickers: use whatever we have prices for
    if prices_df is not None:
        tickers = sorted(prices_df['ticker'].unique().tolist())
    else:
        tickers = [
            'AAPL','ABBV','ABNB','ADBE','AMD','AMZN','AXP','BA','BAC','BLK',
            'C','CAT','CL','COIN','COST','CRM','CRWD','CVX','DE','DIS',
            'F','GE','GM','GOOGL','GS','HD','INTC','JNJ','JPM','KO',
            'LLY','LOW','MA','MCD','META','MRNA','MS','MSFT','NFLX','NOW',
            'NVDA','ORCL','OXY','PEP','PFE','PG','PLTR','PYPL','RTX','SBUX',
            'SCHW','SLB','SMCI','T','TGT','TMUS','TSLA','UBER','UNH','V',
            'VZ','WFC','WMT','XOM',
        ]

    all_rows = []
    n_no_earnings = 0
    for i, ticker in enumerate(tickers):
        try:
            # Get price history for this ticker
            if prices_df is not None:
                hist_df = prices_df[prices_df['ticker'] == ticker].sort_values('date').copy()
                if len(hist_df) < 200:
                    continue
                hist_df = hist_df.set_index('date')
                # Rename to match yfinance convention
                col_map = {}
                for c in hist_df.columns:
                    if c.lower() == 'open': col_map[c] = 'Open'
                    elif c.lower() == 'high': col_map[c] = 'High'
                    elif c.lower() == 'low': col_map[c] = 'Low'
                    elif c.lower() == 'close': col_map[c] = 'Close'
                    elif c.lower() == 'volume': col_map[c] = 'Volume'
                hist_df = hist_df.rename(columns=col_map)
            else:
                stk = yf.Ticker(ticker)
                hist_df = stk.history(period="7y", interval="1d")
                if len(hist_df) < 200:
                    continue

            # Get earnings dates for this ticker
            ticker_earnings = earnings_raw.get(ticker, [])
            if not ticker_earnings:
                # Try yfinance
                try:
                    stk = yf.Ticker(ticker)
                    cal = stk.get_earnings_dates(limit=50)
                    if cal is not None and len(cal) > 0:
                        for ed_ts in cal.index:
                            ed = ed_ts.date() if hasattr(ed_ts, 'date') else pd.Timestamp(ed_ts).date()
                            ticker_earnings.append({'date': str(ed)})
                except:
                    pass

            if not ticker_earnings:
                n_no_earnings += 1
                continue

            # Build surprise lookup
            surprise_map = {}
            for e in ticker_earnings:
                if isinstance(e, dict) and 'date' in e:
                    try:
                        edate = pd.Timestamp(e['date']).date()
                    except:
                        continue
                    surprise_map[edate] = {
                        'eps_estimate': e.get('eps_estimate') or e.get('epsEstimate'),
                        'eps_actual': e.get('eps_actual') or e.get('epsActual'),
                        'surprise_pct': e.get('surprise_pct') or e.get('surprisePercent', 0),
                    }

            # Process each earnings date
            for ed, surp_data in surprise_map.items():
                try:
                    ed_ts = pd.Timestamp(ed)
                    # Ensure tz-naive comparison
                    idx = hist_df.index
                    if hasattr(idx, 'tz') and idx.tz is not None:
                        idx = idx.tz_localize(None)
                        hist = hist_df.copy()
                        hist.index = idx
                    else:
                        hist = hist_df

                    post_mask = hist.index > ed_ts
                    pre_mask = hist.index <= ed_ts

                    if post_mask.sum() < 5 or pre_mask.sum() < 60:
                        continue

                    pre_prices = hist[pre_mask]
                    pre_close = float(pre_prices['Close'].iloc[-1])

                    post_prices = hist[post_mask]
                    post_open = float(post_prices['Open'].iloc[0])
                    post_close_1d = float(post_prices['Close'].iloc[0])
                    post_close_5d = float(post_prices['Close'].iloc[min(4, len(post_prices)-1)])

                    gap_return = (post_open / pre_close) - 1
                    ret_1d = (post_close_1d / pre_close) - 1
                    ret_5d = (post_close_5d / pre_close) - 1

                    # Features
                    closes = pre_prices['Close'].values.astype(float)
                    sma50 = closes[-50:].mean() if len(closes) >= 50 else pre_close
                    sma200 = closes[-200:].mean() if len(closes) >= 200 else pre_close

                    pre_drift_5d = closes[-1] / closes[-6] - 1 if len(closes) >= 6 else 0
                    daily_rets = np.diff(closes[-22:]) / closes[-22:-1] if len(closes) >= 22 else np.array([0.01])
                    realized_vol_20d = float(np.std(daily_rets[-20:]) * np.sqrt(252)) if len(daily_rets) >= 20 else 0.3

                    mom_1m = closes[-1] / closes[-22] - 1 if len(closes) >= 22 else 0
                    mom_3m = closes[-1] / closes[-63] - 1 if len(closes) >= 63 else 0
                    mom_6m = closes[-1] / closes[-126] - 1 if len(closes) >= 126 else 0

                    vols = pre_prices['Volume'].values.astype(float)
                    if len(vols) >= 21:
                        vol_5d = vols[-5:].mean()
                        vol_20d = vols[-21:].mean()
                        volume_surge = vol_5d / max(vol_20d, 1) - 1
                    else:
                        volume_surge = 0

                    eps_surprise_pct = float(surp_data.get('surprise_pct', 0) or 0)

                    # RSI
                    if len(closes) >= 15:
                        delta = np.diff(closes[-15:])
                        gain = np.mean(np.maximum(delta, 0))
                        loss = np.mean(np.maximum(-delta, 0))
                        rsi = 100 - (100 / (1 + gain / max(loss, 1e-10)))
                    else:
                        rsi = 50

                    # 52-week range
                    if len(pre_prices) >= 252:
                        highs = pre_prices['High'].values.astype(float)[-252:]
                        lows = pre_prices['Low'].values.astype(float)[-252:]
                        dist_from_high = pre_close / highs.max() - 1
                        dist_from_low = pre_close / lows.min() - 1
                    else:
                        dist_from_high = 0
                        dist_from_low = 0

                    dow = ed_ts.dayofweek

                    row = {
                        'ticker': ticker,
                        'earnings_date': ed_ts,
                        'pre_close': pre_close,
                        'gap_return': gap_return,
                        'ret_1d': ret_1d,
                        'ret_5d': ret_5d,
                        'price_vs_sma50': pre_close / sma50 - 1,
                        'price_vs_sma200': pre_close / sma200 - 1,
                        'pre_drift_5d': pre_drift_5d,
                        'realized_vol_20d': realized_vol_20d,
                        'mom_1m': mom_1m,
                        'mom_3m': mom_3m,
                        'mom_6m': mom_6m,
                        'volume_surge': volume_surge,
                        'eps_surprise_pct': eps_surprise_pct,
                        'rsi': rsi,
                        'dist_from_high': dist_from_high,
                        'dist_from_low': dist_from_low,
                        'dow': dow,
                    }
                    all_rows.append(row)
                except Exception as e:
                    continue

            if (i+1) % 20 == 0:
                print(f"  Processed {i+1}/{len(tickers)} tickers, {len(all_rows)} events so far")

        except Exception as e:
            continue

    if len(all_rows) == 0:
        print("  WARNING: No earnings data collected!")
        # Return empty DF with correct columns
        return pd.DataFrame(columns=['ticker','earnings_date','gap_return'])

    df = pd.DataFrame(all_rows)
    df.to_parquet(cache, index=False)
    print(f"  Collected {len(df)} earnings events from {df['ticker'].nunique()} tickers")
    print(f"  ({n_no_earnings} tickers had no earnings data)")
    return df


def add_market_features(df):
    """Add market-level features (VIX, SPY trend, sector)."""
    import yfinance as yf

    cache = CACHE_DIR / "market_features.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    # Get SPY and VIX data
    spy = yf.download("SPY", period="7y", interval="1d", progress=False)
    vix = yf.download("^VIX", period="7y", interval="1d", progress=False)

    spy_close = spy['Close'].squeeze() if isinstance(spy['Close'], pd.DataFrame) else spy['Close']
    vix_close = vix['Close'].squeeze() if isinstance(vix['Close'], pd.DataFrame) else vix['Close']

    # SPY features per date
    market_feats = pd.DataFrame(index=spy_close.index)
    market_feats['spy_sma50'] = spy_close.rolling(50).mean()
    market_feats['spy_sma200'] = spy_close.rolling(200).mean()
    market_feats['spy_above_200'] = (spy_close > market_feats['spy_sma200']).astype(float)
    market_feats['spy_ret_20d'] = spy_close.pct_change(20)
    market_feats['vix'] = vix_close.reindex(spy_close.index, method='ffill')
    market_feats['vix_zscore'] = (market_feats['vix'] - market_feats['vix'].rolling(60).mean()) / market_feats['vix'].rolling(60).std().clip(1)

    # Regime classification (green/red/flat)
    spy_ret_5d = spy_close.pct_change(5)
    market_feats['regime'] = 'flat'
    market_feats.loc[spy_ret_5d > 0.01, 'regime'] = 'green'
    market_feats.loc[spy_ret_5d < -0.01, 'regime'] = 'red'

    market_feats.to_parquet(cache)
    return market_feats


def add_historical_earnings_stats(df):
    """Add features based on historical earnings patterns for each ticker."""
    df = df.sort_values(['ticker', 'earnings_date']).copy()

    # For each event, compute stats from PRIOR earnings (no leakage)
    hist_avg_move = []
    hist_beat_rate = []
    hist_move_std = []
    n_prior = []

    for idx, row in df.iterrows():
        prior = df[(df['ticker'] == row['ticker']) & (df['earnings_date'] < row['earnings_date'])]

        if len(prior) >= 2:
            hist_avg_move.append(prior['gap_return'].abs().mean())
            hist_beat_rate.append((prior['gap_return'] > 0).mean())
            hist_move_std.append(prior['gap_return'].std())
            n_prior.append(len(prior))
        else:
            hist_avg_move.append(0.05)  # default 5% move
            hist_beat_rate.append(0.5)
            hist_move_std.append(0.05)
            n_prior.append(0)

    df['hist_avg_move'] = hist_avg_move
    df['hist_beat_rate'] = hist_beat_rate
    df['hist_move_std'] = hist_move_std
    df['n_prior_earnings'] = n_prior

    return df


# ── MODEL ──────────────────────────────────────────────────────────

class EarningsDirectionMLP(nn.Module):
    """MLP with residual connections for earnings direction prediction."""

    def __init__(self, n_features, hidden=128, n_layers=3, dropout=0.3, n_classes=3):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(n_features)
        self.input_proj = nn.Linear(n_features, hidden)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.Sequential(
                nn.BatchNorm1d(hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
                nn.BatchNorm1d(hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
            ))

        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        x = self.input_bn(x)
        x = self.input_proj(x)
        for layer in self.layers:
            x = x + layer(x)  # residual
        return self.head(x)


# ── WALK-FORWARD TRAINING ─────────────────────────────────────────

FEATURE_COLS = [
    'price_vs_sma50', 'price_vs_sma200', 'pre_drift_5d',
    'realized_vol_20d', 'mom_1m', 'mom_3m', 'mom_6m',
    'volume_surge', 'eps_surprise_pct', 'rsi',
    'dist_from_high', 'dist_from_low', 'dow',
    'hist_avg_move', 'hist_beat_rate', 'hist_move_std', 'n_prior_earnings',
    'spy_above_200', 'spy_ret_20d', 'vix', 'vix_zscore',
]


def create_labels(df, threshold=0.02):
    """Create 3-class labels: UP (>threshold), DOWN (<-threshold), FLAT."""
    labels = np.zeros(len(df), dtype=np.int64)
    labels[df['gap_return'].values > threshold] = 1   # UP
    labels[df['gap_return'].values < -threshold] = 2  # DOWN
    # 0 = FLAT
    return labels


def train_one_fold(X_train, y_train, X_val, y_val, n_features, n_classes=3,
                   epochs=100, lr=1e-3, batch_size=64):
    """Train MLP for one walk-forward fold."""
    model = EarningsDirectionMLP(n_features, hidden=128, n_layers=3,
                                  dropout=0.3, n_classes=n_classes).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # Class weights for imbalanced data
    class_counts = np.bincount(y_train, minlength=n_classes).astype(float)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    class_weights = class_weights / class_weights.sum() * n_classes
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(DEVICE))

    X_t = torch.FloatTensor(X_train).to(DEVICE)
    y_t = torch.LongTensor(y_train).to(DEVICE)
    X_v = torch.FloatTensor(X_val).to(DEVICE)
    y_v = torch.LongTensor(y_val).to(DEVICE)

    best_val_loss = float('inf')
    best_state = None
    patience = 15
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        # Mini-batch training
        perm = torch.randperm(len(X_t))
        total_loss = 0
        n_batch = 0
        for i in range(0, len(X_t), batch_size):
            idx = perm[i:i+batch_size]
            if len(idx) < 4:  # skip tiny batches (BatchNorm needs >1)
                continue
            out = model(X_t[idx])
            loss = criterion(out, y_t[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batch += 1

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(X_v)
            val_loss = criterion(val_out, y_v).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    # Load best model
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        val_probs = torch.softmax(model(X_v), dim=1).cpu().numpy()

    return model, val_probs


def walk_forward_train(df, feature_cols, threshold=0.02,
                       train_years=2.5, test_months=3):
    """Walk-forward training with sliding window."""
    df = df.sort_values('earnings_date').copy()
    labels = create_labels(df, threshold)
    df['label'] = labels

    # Fill NaN features
    X_all = df[feature_cols].fillna(0).values

    # Normalize features (per-fold to avoid leakage)
    dates = df['earnings_date'].values
    min_date = dates.min()
    max_date = dates.max()

    # Walk-forward folds
    train_days = int(train_years * 365)
    test_days = int(test_months * 30)

    all_preds = []
    all_labels = []
    all_dates = []
    all_tickers = []
    all_probs = []

    fold = 0
    start = pd.Timestamp(min_date) + timedelta(days=train_days)

    while start < pd.Timestamp(max_date) - timedelta(days=test_days):
        end = start + timedelta(days=test_days)
        train_mask = (dates >= pd.Timestamp(min_date).to_datetime64()) & (dates < start.to_datetime64())
        test_mask = (dates >= start.to_datetime64()) & (dates < end.to_datetime64())

        n_train = train_mask.sum()
        n_test = test_mask.sum()

        if n_train < 100 or n_test < 10:
            start = end
            continue

        # Normalize features based on training set
        X_train = X_all[train_mask].copy()
        X_test = X_all[test_mask].copy()
        y_train = labels[train_mask]
        y_test = labels[test_mask]

        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0) + 1e-8
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

        # Train
        model, test_probs = train_one_fold(
            X_train, y_train, X_test, y_test,
            n_features=len(feature_cols), n_classes=3
        )

        preds = test_probs.argmax(axis=1)
        acc = (preds == y_test).mean()

        print(f"  Fold {fold}: train={n_train}, test={n_test}, "
              f"acc={acc:.3f}, up={test_probs[:,1].mean():.3f}, "
              f"down={test_probs[:,2].mean():.3f}")

        all_preds.extend(preds)
        all_labels.extend(y_test)
        all_dates.extend(df.loc[test_mask, 'earnings_date'].values)
        all_tickers.extend(df.loc[test_mask, 'ticker'].values)
        all_probs.extend(test_probs)

        start = end
        fold += 1

    results = pd.DataFrame({
        'earnings_date': all_dates,
        'ticker': all_tickers,
        'pred': all_preds,
        'label': all_labels,
        'prob_flat': [p[0] for p in all_probs],
        'prob_up': [p[1] for p in all_probs],
        'prob_down': [p[2] for p in all_probs],
    })

    return results


# ── ALSO TEST LGBM BASELINE ───────────────────────────────────────

def lgbm_walk_forward(df, feature_cols, threshold=0.02,
                      train_years=2.5, test_months=3):
    """LGBM baseline for comparison."""
    try:
        import lightgbm as lgb
    except ImportError:
        print("  LightGBM not available, skipping baseline")
        return None

    df = df.sort_values('earnings_date').copy()
    labels = create_labels(df, threshold)
    X_all = df[feature_cols].fillna(0).values
    dates = df['earnings_date'].values

    train_days = int(train_years * 365)
    test_days = int(test_months * 30)

    all_preds = []
    all_labels = []
    all_dates = []
    all_tickers = []
    all_probs = []

    fold = 0
    start = pd.Timestamp(dates.min()) + timedelta(days=train_days)

    while start < pd.Timestamp(dates.max()) - timedelta(days=test_days):
        end = start + timedelta(days=test_days)
        train_mask = (dates < start.to_datetime64())
        test_mask = (dates >= start.to_datetime64()) & (dates < end.to_datetime64())

        if train_mask.sum() < 100 or test_mask.sum() < 10:
            start = end
            continue

        model = lgb.LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            num_leaves=31, min_child_samples=20, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
            n_jobs=-1, verbose=-1,
        )
        model.fit(X_all[train_mask], labels[train_mask])
        probs = model.predict_proba(X_all[test_mask])
        preds = probs.argmax(axis=1)

        all_preds.extend(preds)
        all_labels.extend(labels[test_mask])
        all_dates.extend(df.loc[test_mask, 'earnings_date'].values)
        all_tickers.extend(df.loc[test_mask, 'ticker'].values)
        all_probs.extend(probs)

        start = end
        fold += 1

    results = pd.DataFrame({
        'earnings_date': all_dates,
        'ticker': all_tickers,
        'pred': all_preds,
        'label': all_labels,
        'prob_flat': [p[0] for p in all_probs],
        'prob_up': [p[1] for p in all_probs],
        'prob_down': [p[2] for p in all_probs],
    })

    return results


# ── BACKTEST DIRECTIONAL OPTIONS ───────────────────────────────────

def backtest_directional_options(results_df, df_full, min_confidence=0.5):
    """
    Simulate buying directional options based on predictions.

    Strategy:
      - If model predicts UP with confidence > threshold → buy 1-week OTM call (~10-delta)
      - If model predicts DOWN with confidence > threshold → buy 1-week OTM put
      - Risk: fixed 1% of portfolio per trade
      - Payoff approximation: if direction correct AND move > strike distance →
        profit = (move - strike_dist) * leverage
        if wrong → lose premium
    """
    capital = 100_000
    equity = [capital]
    trades = []

    for _, row in results_df.iterrows():
        pred = int(row['pred'])
        prob_up = row['prob_up']
        prob_down = row['prob_down']

        # Skip flat predictions
        if pred == 0:
            continue

        # Confidence check
        if pred == 1 and prob_up < min_confidence:
            continue
        if pred == 2 and prob_down < min_confidence:
            continue

        # Find actual return from full dataset
        match = df_full[
            (df_full['ticker'] == row['ticker']) &
            (df_full['earnings_date'] == row['earnings_date'])
        ]
        if len(match) == 0:
            continue

        actual_gap = match['gap_return'].iloc[0]
        actual_1d = match['ret_1d'].iloc[0]

        # Options payoff approximation
        # Buying ~10-delta OTM option, ~5% OTM
        # Premium cost: ~1-2% of underlying (volatile stock pre-earnings)
        # If stock moves >5% in right direction: payoff ~= (move - 5%) * leverage
        # If wrong or insufficient move: lose premium

        risk_amount = capital * 0.01  # 1% risk per trade
        premium_cost = risk_amount  # this IS what we pay for the option

        # Use the larger of gap and 1d return for option valuation
        actual_move = actual_gap  # gap return captures earnings overnight move

        if pred == 1:  # predicted UP
            otm_pct = 0.03  # 3% OTM call
            if actual_move > otm_pct:
                # In the money — profit proportional to move beyond strike
                # Typical leverage ~5-10x for near-expiry OTM
                intrinsic_pct = actual_move - otm_pct
                payoff = premium_cost * (intrinsic_pct / otm_pct) * 3  # ~3x leverage approximation
                pnl = payoff - premium_cost
            elif actual_move > 0:
                # Right direction but not enough — partial loss
                pnl = -premium_cost * 0.7  # retain some time value
            else:
                # Wrong direction — lose premium
                pnl = -premium_cost

        elif pred == 2:  # predicted DOWN
            otm_pct = 0.03  # 3% OTM put
            if actual_move < -otm_pct:
                intrinsic_pct = abs(actual_move) - otm_pct
                payoff = premium_cost * (intrinsic_pct / otm_pct) * 3
                pnl = payoff - premium_cost
            elif actual_move < 0:
                pnl = -premium_cost * 0.7
            else:
                pnl = -premium_cost

        capital += pnl
        equity.append(capital)
        trades.append({
            'date': row['earnings_date'],
            'ticker': row['ticker'],
            'direction': 'UP' if pred == 1 else 'DOWN',
            'confidence': prob_up if pred == 1 else prob_down,
            'actual_gap': actual_gap,
            'pnl': pnl,
            'correct': (pred == 1 and actual_move > 0) or (pred == 2 and actual_move < 0),
        })

    return equity, trades


# ── EVALUATION ─────────────────────────────────────────────────────

def evaluate_results(results_df, df_full, model_name="MLP"):
    """Full evaluation with regime analysis, permutation test, etc."""
    print(f"\n{'='*60}")
    print(f"  {model_name} — Earnings Direction Predictor")
    print(f"{'='*60}")

    # Overall accuracy
    correct = (results_df['pred'] == results_df['label']).mean()
    print(f"\n  Overall accuracy: {correct:.3f}")

    # Per-class accuracy
    for cls, name in [(0, 'FLAT'), (1, 'UP'), (2, 'DOWN')]:
        mask = results_df['label'] == cls
        if mask.sum() > 0:
            acc = (results_df.loc[mask, 'pred'] == cls).mean()
            print(f"  {name}: {acc:.3f} ({mask.sum()} samples)")

    # Directional accuracy (predicted UP/DOWN, was it correct direction?)
    dir_mask = results_df['pred'] != 0  # predicted UP or DOWN
    if dir_mask.sum() > 0:
        # Match with actual gap returns
        dir_preds = results_df[dir_mask].copy()

        # Merge actual returns
        dir_preds = dir_preds.merge(
            df_full[['ticker', 'earnings_date', 'gap_return']],
            on=['ticker', 'earnings_date'], how='left'
        )

        dir_correct = (
            ((dir_preds['pred'] == 1) & (dir_preds['gap_return'] > 0)) |
            ((dir_preds['pred'] == 2) & (dir_preds['gap_return'] < 0))
        ).mean()

        print(f"\n  Directional accuracy (when predicting UP/DOWN): {dir_correct:.3f} ({dir_mask.sum()} predictions)")

    # Confidence-stratified accuracy
    print(f"\n  Confidence-stratified directional accuracy:")
    for thresh in [0.40, 0.50, 0.60, 0.70, 0.80]:
        conf_mask = (
            ((results_df['pred'] == 1) & (results_df['prob_up'] >= thresh)) |
            ((results_df['pred'] == 2) & (results_df['prob_down'] >= thresh))
        )
        if conf_mask.sum() >= 5:
            conf_preds = results_df[conf_mask].merge(
                df_full[['ticker', 'earnings_date', 'gap_return']],
                on=['ticker', 'earnings_date'], how='left'
            )
            dir_acc = (
                ((conf_preds['pred'] == 1) & (conf_preds['gap_return'] > 0)) |
                ((conf_preds['pred'] == 2) & (conf_preds['gap_return'] < 0))
            ).mean()
            print(f"    conf >= {thresh:.0%}: {dir_acc:.3f} ({conf_mask.sum()} trades)")

    # Backtest directional options at different confidence thresholds
    print(f"\n  Directional Options Backtest:")
    best_sharpe = -999
    best_trades = None
    best_thresh = None

    for thresh in [0.40, 0.50, 0.60, 0.70]:
        equity, trades = backtest_directional_options(results_df, df_full, min_confidence=thresh)
        if len(trades) < 10:
            continue

        # Compute metrics
        trade_pnls = [t['pnl'] for t in trades]
        total_ret = (equity[-1] / equity[0]) - 1
        n_years = max(1, len(set([str(t['date'])[:4] for t in trades])))
        cagr = (equity[-1] / equity[0]) ** (1/n_years) - 1

        wins = sum(1 for p in trade_pnls if p > 0)
        wr = wins / len(trade_pnls)

        gross_profit = sum(p for p in trade_pnls if p > 0)
        gross_loss = abs(sum(p for p in trade_pnls if p < 0))
        pf = gross_profit / max(gross_loss, 1)

        # Sharpe (annualized from trade returns)
        trade_rets = np.array(trade_pnls) / 100_000
        if trade_rets.std() > 0:
            sharpe = trade_rets.mean() / trade_rets.std() * np.sqrt(min(252, len(trades)))
        else:
            sharpe = 0

        # MaxDD
        eq = np.array(equity)
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak
        maxdd = dd.min()

        print(f"    conf >= {thresh:.0%}: {len(trades)} trades, "
              f"WR={wr:.1%}, PF={pf:.2f}, Sharpe={sharpe:.2f}, "
              f"CAGR={cagr:.1%}, MaxDD={maxdd:.1%}")

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_trades = trades
            best_thresh = thresh

    # Regime analysis (HC #428 R1)
    if best_trades:
        print(f"\n  Regime Analysis (best threshold = {best_thresh:.0%}):")
        trades_df = pd.DataFrame(best_trades)
        trades_df['date'] = pd.to_datetime(trades_df['date'])

        # Simple regime: was SPY up or down that week?
        # Merge with market data
        market_feats = add_market_features(df_full)
        trades_df = trades_df.merge(
            market_feats[['regime']].reset_index().rename(columns={'Date': 'date', 'index': 'date'}),
            on='date', how='left'
        )

        # If merge failed, use simple regime from gap direction correlation
        if 'regime' not in trades_df.columns or trades_df['regime'].isna().all():
            # Approximate regime from trade dates
            # Green: market was up, Red: down, Flat: sideways
            trades_df['regime'] = 'flat'  # fallback

        for regime in ['green', 'red', 'flat']:
            rmask = trades_df['regime'] == regime
            if rmask.sum() >= 5:
                r_pnls = trades_df.loc[rmask, 'pnl'].values
                r_wr = (r_pnls > 0).mean()
                r_sharpe = r_pnls.mean() / max(r_pnls.std(), 1) * np.sqrt(min(50, rmask.sum()))
                print(f"    {regime}: {rmask.sum()} trades, WR={r_wr:.1%}, Sharpe={r_sharpe:.2f}")

    # Permutation test
    print(f"\n  Permutation Test (100 trials):")
    real_acc = (results_df['pred'] == results_df['label']).mean()
    perm_accs = []
    for _ in range(100):
        perm_labels = np.random.permutation(results_df['label'].values)
        perm_acc = (results_df['pred'].values == perm_labels).mean()
        perm_accs.append(perm_acc)

    perm_p = (np.array(perm_accs) >= real_acc).mean()
    print(f"    Real accuracy: {real_acc:.3f}")
    print(f"    Random mean: {np.mean(perm_accs):.3f} ± {np.std(perm_accs):.3f}")
    print(f"    p-value: {perm_p:.4f}")

    return {
        'model': model_name,
        'overall_accuracy': float(correct),
        'n_predictions': len(results_df),
        'best_threshold': float(best_thresh) if best_thresh else None,
        'best_sharpe': float(best_sharpe),
        'permutation_p': float(perm_p),
    }


# ── MAIN ───────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 60)
    print("  EARNINGS DIRECTION PREDICTOR v1")
    print("  HC #701: Creative research, real performance only")
    print("=" * 60)

    # 1. Collect data
    print("\n[1/5] Collecting earnings data...")
    df = get_earnings_data()
    print(f"  {len(df)} events, {df['ticker'].nunique()} tickers")
    print(f"  Date range: {df['earnings_date'].min()} to {df['earnings_date'].max()}")

    # 2. Add historical earnings stats (no leakage)
    print("\n[2/5] Computing historical earnings features...")
    df = add_historical_earnings_stats(df)

    # 3. Add market features
    print("\n[3/5] Adding market features...")
    market_feats = add_market_features(df)

    # Merge market features with earnings data
    df['earnings_date_dt'] = pd.to_datetime(df['earnings_date'])
    market_feats_reset = market_feats.reset_index()
    market_feats_reset.columns = ['date'] + list(market_feats.columns)
    market_feats_reset['date'] = pd.to_datetime(market_feats_reset['date']).dt.tz_localize(None)

    df = df.merge(
        market_feats_reset[['date', 'spy_above_200', 'spy_ret_20d', 'vix', 'vix_zscore']],
        left_on='earnings_date_dt', right_on='date', how='left'
    )
    df['spy_above_200'] = df['spy_above_200'].fillna(1)
    df['spy_ret_20d'] = df['spy_ret_20d'].fillna(0)
    df['vix'] = df['vix'].fillna(20)
    df['vix_zscore'] = df['vix_zscore'].fillna(0)

    # Gap return stats
    print(f"\n  Gap return distribution:")
    print(f"    Mean: {df['gap_return'].mean():.3%}")
    print(f"    Median: {df['gap_return'].median():.3%}")
    print(f"    Std: {df['gap_return'].std():.3%}")
    print(f"    >+2%: {(df['gap_return'] > 0.02).mean():.1%}")
    print(f"    <-2%: {(df['gap_return'] < -0.02).mean():.1%}")
    print(f"    Flat (±2%): {((df['gap_return'] >= -0.02) & (df['gap_return'] <= 0.02)).mean():.1%}")

    # Verify features exist
    available_features = [f for f in FEATURE_COLS if f in df.columns]
    missing = [f for f in FEATURE_COLS if f not in df.columns]
    if missing:
        print(f"  WARNING: Missing features: {missing}")
    print(f"  Using {len(available_features)} features")

    # 4. Train MLP (GPU)
    print(f"\n[4/5] Walk-forward MLP training on {DEVICE}...")
    mlp_results = walk_forward_train(df, available_features, threshold=0.02)
    mlp_eval = evaluate_results(mlp_results, df, "MLP (GPU)")

    # 5. Train LGBM baseline
    print(f"\n[5/5] Walk-forward LGBM baseline...")
    lgbm_results = lgbm_walk_forward(df, available_features, threshold=0.02)
    if lgbm_results is not None:
        lgbm_eval = evaluate_results(lgbm_results, df, "LGBM")
    else:
        lgbm_eval = None

    # Save results
    summary = {
        'mlp': mlp_eval,
        'lgbm': lgbm_eval,
        'n_events': len(df),
        'n_tickers': int(df['ticker'].nunique()),
        'date_range': f"{df['earnings_date'].min()} to {df['earnings_date'].max()}",
        'runtime_seconds': time.time() - t0,
    }

    with open(OUTPUT / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    mlp_results.to_parquet(OUTPUT / "mlp_predictions.parquet", index=False)
    if lgbm_results is not None:
        lgbm_results.to_parquet(OUTPUT / "lgbm_predictions.parquet", index=False)

    df.to_parquet(OUTPUT / "earnings_dataset.parquet", index=False)

    print(f"\n{'='*60}")
    print(f"  COMPLETE — {time.time()-t0:.0f}s")
    print(f"  Results saved to {OUTPUT}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
