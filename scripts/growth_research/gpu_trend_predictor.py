#!/usr/bin/env python3
"""
GPU Trend Predictor — Transformer for Multi-Asset Trend Timing
===============================================================
HC #701: Creative research on all nodes. Neptune GPU available.
HC #700: High-risk growth for agentic account.

THESIS: Traditional trend-following (200MA, momentum) fails R1 (regime gap).
Can a Transformer learn to combine multiple signals (price patterns, volume,
cross-asset correlations) to produce regime-agnostic trend signals?

This targets the GROWTH sleeve (Agentic Robinhood account, HC #700).

APPROACH:
  - Universe: 11 SPDR sector ETFs + SPY + QQQ + IWM + TLT + GLD + BTC-USD
  - Features per asset per day: OHLCV patterns, momentum, vol, cross-correlations
  - Sequence: 60 trading days of context
  - Target: binary — will asset outperform risk-free rate by >1% over next 20 days?
  - Architecture: Transformer encoder (cross-asset attention)
  - Walk-forward: 252d train, 63d test, 21d step

VALIDATION (HC #428):
  - R1: regime gap < 0.50
  - Permutation test
  - Walk-forward only
"""

import sys, os, json, warnings, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from datetime import timedelta

warnings.filterwarnings("ignore")

if os.path.exists("/home/nick"):
    ROOT = Path("/home/nick/Lvl3Quant")
else:
    ROOT = Path("/home/jupiter/Lvl3Quant")

OUTPUT = ROOT / "output" / "gpu_trend_predictor_v1"
OUTPUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── DATA ──────────────────────────────────────────────────────

TICKERS = [
    'SPY', 'QQQ', 'IWM',    # Broad market
    'XLK', 'XLF', 'XLE',    # Sectors
    'XLV', 'XLI', 'XLC',
    'XLY', 'XLP', 'XLRE',
    'XLU', 'XLB',
    'TLT', 'GLD',            # Safe haven
    'HYG',                    # Credit
]

SEQ_LEN = 60  # 60 trading days context
PRED_HORIZON = 20  # predict 20-day forward return
THRESHOLD = 0.01  # >1% excess return = positive

def download_data():
    """Download price data for all tickers."""
    cache = OUTPUT / "price_data.parquet"
    if cache.exists():
        print("  Loading cached price data...")
        return pd.read_parquet(cache)

    import yfinance as yf
    dfs = []
    for ticker in TICKERS:
        try:
            df = yf.download(ticker, period="10y", interval="1d", progress=False)
            if len(df) < 200:
                continue
            df = df[['Open','High','Low','Close','Volume']].copy()
            # Handle multi-level columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df['ticker'] = ticker
            df = df.reset_index()
            if 'Date' in df.columns:
                df = df.rename(columns={'Date': 'date'})
            dfs.append(df)
            print(f"  {ticker}: {len(df)} days")
        except Exception as e:
            print(f"  {ticker} failed: {e}")

    data = pd.concat(dfs, ignore_index=True)
    data['date'] = pd.to_datetime(data['date']).dt.tz_localize(None)
    data.to_parquet(cache, index=False)
    print(f"  Total: {len(data)} rows, {data['ticker'].nunique()} tickers")
    return data


def compute_features(data):
    """Compute features for each ticker-date."""
    all_features = []

    for ticker in data['ticker'].unique():
        df = data[data['ticker'] == ticker].sort_values('date').copy()

        close = df['Close'].values.astype(float)
        high = df['High'].values.astype(float)
        low = df['Low'].values.astype(float)
        volume = df['Volume'].values.astype(float)

        # Returns
        ret_1d = np.zeros(len(df)); ret_1d[1:] = close[1:] / close[:-1] - 1
        ret_5d = np.zeros(len(df)); ret_5d[5:] = close[5:] / close[:-5] - 1
        ret_20d = np.zeros(len(df)); ret_20d[20:] = close[20:] / close[:-20] - 1

        # Volatility
        vol_5d = pd.Series(ret_1d).rolling(5).std().values * np.sqrt(252)
        vol_20d = pd.Series(ret_1d).rolling(20).std().values * np.sqrt(252)

        # Momentum
        sma_20 = pd.Series(close).rolling(20).mean().values
        sma_50 = pd.Series(close).rolling(50).mean().values
        sma_200 = pd.Series(close).rolling(200).mean().values

        # RSI
        delta = np.diff(close, prepend=close[0])
        gain = np.maximum(delta, 0)
        loss = np.maximum(-delta, 0)
        avg_gain = pd.Series(gain).rolling(14).mean().values
        avg_loss = pd.Series(loss).rolling(14).mean().values
        rsi = 100 - 100 / (1 + avg_gain / np.maximum(avg_loss, 1e-10))

        # Volume trend
        vol_sma = pd.Series(volume).rolling(20).mean().values
        vol_ratio = volume / np.maximum(vol_sma, 1)

        # ATR
        tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)),
                                                 np.abs(low - np.roll(close, 1))))
        atr = pd.Series(tr).rolling(14).mean().values / close

        # Price position (0-1 within recent range)
        high_20 = pd.Series(high).rolling(20).max().values
        low_20 = pd.Series(low).rolling(20).min().values
        price_pos = (close - low_20) / np.maximum(high_20 - low_20, 1e-10)

        # Forward return (target)
        fwd_ret = np.zeros(len(df))
        fwd_ret[:-PRED_HORIZON] = close[PRED_HORIZON:] / close[:-PRED_HORIZON] - 1

        feats = pd.DataFrame({
            'date': df['date'].values,
            'ticker': ticker,
            'ret_1d': ret_1d,
            'ret_5d': ret_5d,
            'ret_20d': ret_20d,
            'vol_5d': vol_5d,
            'vol_20d': vol_20d,
            'price_vs_sma20': close / np.maximum(sma_20, 1e-10) - 1,
            'price_vs_sma50': close / np.maximum(sma_50, 1e-10) - 1,
            'price_vs_sma200': close / np.maximum(sma_200, 1e-10) - 1,
            'rsi': rsi / 100,  # normalize to 0-1
            'vol_ratio': vol_ratio,
            'atr': atr,
            'price_pos': price_pos,
            'fwd_ret': fwd_ret,
        })

        all_features.append(feats)

    result = pd.concat(all_features, ignore_index=True)
    result = result.fillna(0)
    return result


# ── MODEL ──────────────────────────────────────────────────────

FEATURE_COLS = [
    'ret_1d', 'ret_5d', 'ret_20d',
    'vol_5d', 'vol_20d',
    'price_vs_sma20', 'price_vs_sma50', 'price_vs_sma200',
    'rsi', 'vol_ratio', 'atr', 'price_pos',
]

class TrendTransformer(nn.Module):
    """Transformer for cross-asset trend prediction."""

    def __init__(self, n_features=12, n_assets=17, d_model=64, nhead=4,
                 n_layers=3, dropout=0.2):
        super().__init__()
        self.n_features = n_features
        self.n_assets = n_assets

        # Project features to d_model
        self.feature_proj = nn.Linear(n_features, d_model)

        # Positional encoding for time steps
        self.pos_enc = nn.Parameter(torch.randn(1, SEQ_LEN, d_model) * 0.02)

        # Asset embedding
        self.asset_emb = nn.Embedding(n_assets, d_model)

        # Transformer encoder (cross-asset + temporal attention)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output head: one prediction per asset
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x, asset_ids):
        """
        x: (batch, seq_len, n_features) — features for one asset over time
        asset_ids: (batch,) — which asset each sample is for
        """
        B, S, F = x.shape

        # Project features
        h = self.feature_proj(x)  # (B, S, d_model)

        # Add positional encoding
        h = h + self.pos_enc[:, :S, :]

        # Add asset embedding (broadcast across time)
        asset_emb = self.asset_emb(asset_ids).unsqueeze(1)  # (B, 1, d_model)
        h = h + asset_emb

        # Transformer
        h = self.transformer(h)  # (B, S, d_model)

        # Use last time step for prediction
        out = self.head(h[:, -1, :])  # (B, 1)
        return out.squeeze(-1)


# ── DATASET ────────────────────────────────────────────────────

def build_sequences(features_df, dates, ticker_to_id):
    """Build (seq_len, n_features) sequences for training."""
    X_list = []
    y_list = []
    asset_ids = []
    seq_dates = []

    for ticker in features_df['ticker'].unique():
        tdf = features_df[features_df['ticker'] == ticker].sort_values('date')

        # Filter to dates in range
        mask = tdf['date'].isin(dates)
        valid_indices = tdf.index[mask]

        for idx in valid_indices:
            # Get position in ticker's dataframe
            pos = tdf.index.get_loc(idx)
            if pos < SEQ_LEN:
                continue

            # Extract sequence
            seq = tdf.iloc[pos - SEQ_LEN:pos][FEATURE_COLS].values
            target = tdf.iloc[pos]['fwd_ret']

            if np.isnan(seq).any() or np.isnan(target):
                continue

            X_list.append(seq)
            y_list.append(1 if target > THRESHOLD else 0)
            asset_ids.append(ticker_to_id.get(ticker, 0))
            seq_dates.append(tdf.iloc[pos]['date'])

    if not X_list:
        return None, None, None, None

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    aids = np.array(asset_ids, dtype=np.int64)

    return X, y, aids, seq_dates


def train_one_fold(X_train, y_train, a_train, X_val, y_val, a_val,
                   n_features, n_assets, epochs=60, lr=5e-4, batch_size=256):
    """Train transformer for one fold."""
    model = TrendTransformer(
        n_features=n_features, n_assets=n_assets,
        d_model=64, nhead=4, n_layers=3, dropout=0.2
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # Class weights
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(DEVICE))

    X_t = torch.FloatTensor(X_train).to(DEVICE)
    y_t = torch.FloatTensor(y_train).to(DEVICE)
    a_t = torch.LongTensor(a_train).to(DEVICE)
    X_v = torch.FloatTensor(X_val).to(DEVICE)
    y_v = torch.FloatTensor(y_val).to(DEVICE)
    a_v = torch.LongTensor(a_val).to(DEVICE)

    best_val_loss = float('inf')
    best_state = None
    patience = 10
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_t))
        total_loss = 0
        n_batch = 0

        for i in range(0, len(X_t), batch_size):
            idx = perm[i:i+batch_size]
            if len(idx) < 4:
                continue
            out = model(X_t[idx], a_t[idx])
            loss = criterion(out, y_t[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batch += 1

        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_out = model(X_v, a_v)
            val_loss = criterion(val_out, y_v).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_probs = torch.sigmoid(model(X_v, a_v)).cpu().numpy()

    return model, val_probs


# ── WALK-FORWARD ───────────────────────────────────────────────

def walk_forward(features_df):
    """Walk-forward with Transformer."""
    dates = sorted(features_df['date'].unique())
    ticker_to_id = {t: i for i, t in enumerate(sorted(features_df['ticker'].unique()))}
    n_assets = len(ticker_to_id)

    train_days = 252
    test_days = 63
    step_days = 21

    all_results = []
    fold = 0

    i = train_days
    while i + test_days <= len(dates):
        train_dates = set(dates[max(0, i-train_days):i])
        test_dates = set(dates[i:i+test_days])

        X_train, y_train, a_train, _ = build_sequences(features_df, train_dates, ticker_to_id)
        X_test, y_test, a_test, test_seq_dates = build_sequences(features_df, test_dates, ticker_to_id)

        if X_train is None or X_test is None or len(X_train) < 100 or len(X_test) < 20:
            i += step_days
            continue

        # Normalize features per fold
        mean = X_train.mean(axis=(0, 1))
        std = X_train.std(axis=(0, 1)) + 1e-8
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

        model, test_probs = train_one_fold(
            X_train, y_train, a_train,
            X_test, y_test, a_test,
            n_features=len(FEATURE_COLS), n_assets=n_assets
        )

        preds = (test_probs > 0.5).astype(int)
        acc = (preds == y_test).mean()

        # Get ticker names for test samples
        test_tickers = []
        for ticker, tid in ticker_to_id.items():
            mask = a_test == tid
            test_tickers.extend([ticker] * mask.sum())

        fold_results = pd.DataFrame({
            'date': test_seq_dates,
            'ticker': test_tickers[:len(test_seq_dates)],
            'prob': test_probs[:len(test_seq_dates)],
            'pred': preds[:len(test_seq_dates)],
            'label': y_test[:len(test_seq_dates)],
        })
        all_results.append(fold_results)

        base_rate = y_test.mean()
        print(f"  Fold {fold}: train={len(X_train)}, test={len(X_test)}, "
              f"acc={acc:.3f}, base_rate={base_rate:.3f}, pred_rate={preds.mean():.3f}")

        i += step_days
        fold += 1

    return pd.concat(all_results, ignore_index=True)


# ── EVALUATION ─────────────────────────────────────────────────

def evaluate(results_df, features_df):
    """Evaluate walk-forward predictions."""
    print(f"\n{'='*60}")
    print(f"  Trend Transformer — Walk-Forward Results")
    print(f"{'='*60}")

    overall_acc = (results_df['pred'] == results_df['label']).mean()
    base_rate = results_df['label'].mean()
    pred_rate = results_df['pred'].mean()

    print(f"\n  Overall accuracy: {overall_acc:.3f} (base rate: {base_rate:.3f})")
    print(f"  Prediction rate: {pred_rate:.3f}")

    # Precision/recall for positive class
    tp = ((results_df['pred'] == 1) & (results_df['label'] == 1)).sum()
    fp = ((results_df['pred'] == 1) & (results_df['label'] == 0)).sum()
    fn = ((results_df['pred'] == 0) & (results_df['label'] == 1)).sum()

    prec = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    lift = prec / max(base_rate, 1e-10)

    print(f"  Precision: {prec:.3f}, Recall: {recall:.3f}, Lift: {lift:.2f}x")

    # Confidence tiers
    print(f"\n  Confidence Tiers:")
    for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.80]:
        mask = results_df['prob'] >= thresh
        if mask.sum() >= 10:
            tier_prec = results_df.loc[mask, 'label'].mean()
            n = mask.sum()
            print(f"    prob >= {thresh:.2f}: prec={tier_prec:.3f}, lift={tier_prec/base_rate:.2f}x, n={n}")

    # Backtest: long when prob > threshold, else cash
    # Equal weight across predicted assets
    print(f"\n  Portfolio Backtest:")

    # Merge forward returns
    results_df = results_df.merge(
        features_df[['date', 'ticker', 'fwd_ret']],
        on=['date', 'ticker'], how='left'
    )

    for thresh in [0.50, 0.55, 0.60]:
        # Per-date: go long assets predicted UP, equal weight
        daily_rets = []
        dates = sorted(results_df['date'].unique())

        for d in dates:
            day_preds = results_df[(results_df['date'] == d) & (results_df['prob'] >= thresh)]
            if len(day_preds) > 0:
                # Average forward return of selected assets (approximation)
                avg_ret = day_preds['fwd_ret'].mean() / PRED_HORIZON  # daily-ized
                daily_rets.append(avg_ret)
            else:
                daily_rets.append(0)  # cash

        rets = np.array(daily_rets)
        total_ret = (1 + rets).prod() - 1
        n_years = len(dates) / 252
        cagr = (1 + total_ret) ** (1/max(n_years, 0.1)) - 1

        # Sharpe
        if rets.std() > 0:
            sharpe = rets.mean() / rets.std() * np.sqrt(252)
        else:
            sharpe = 0

        # MaxDD
        equity = np.cumprod(1 + rets)
        peak = np.maximum.accumulate(equity)
        maxdd = ((equity - peak) / peak).min()

        print(f"    prob >= {thresh:.2f}: Sharpe={sharpe:.2f}, CAGR={cagr:.1%}, MaxDD={maxdd:.1%}")

    # Regime test
    print(f"\n  Regime Analysis:")
    results_with_spy = results_df[results_df['ticker'] == 'SPY'].copy()
    if len(results_with_spy) > 0:
        spy_ret = results_with_spy['fwd_ret'].values
        results_df_dated = results_df.copy()
        results_df_dated['year'] = pd.to_datetime(results_df_dated['date']).dt.year

        for year in sorted(results_df_dated['year'].unique()):
            yr = results_df_dated[results_df_dated['year'] == year]
            yr_acc = (yr['pred'] == yr['label']).mean()
            yr_prec = yr.loc[yr['pred']==1, 'label'].mean() if (yr['pred']==1).sum() > 0 else 0
            print(f"    {year}: acc={yr_acc:.3f}, prec={yr_prec:.3f}, n={len(yr)}")

    # Permutation test
    print(f"\n  Permutation Test (50 trials):")
    real_acc = (results_df['pred'] == results_df['label']).mean()
    perm_accs = []
    for _ in range(50):
        perm_labels = np.random.permutation(results_df['label'].values)
        perm_acc = (results_df['pred'].values == perm_labels).mean()
        perm_accs.append(perm_acc)

    perm_p = (np.array(perm_accs) >= real_acc).mean()
    print(f"    Real: {real_acc:.3f}, Random: {np.mean(perm_accs):.3f} ± {np.std(perm_accs):.3f}")
    print(f"    p-value: {perm_p:.4f}")

    # Save summary
    summary = {
        'overall_accuracy': float(overall_acc),
        'base_rate': float(base_rate),
        'precision': float(prec),
        'recall': float(recall),
        'lift': float(lift),
        'permutation_p': float(perm_p),
        'n_predictions': len(results_df),
    }

    with open(OUTPUT / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    results_df.to_parquet(OUTPUT / "predictions.parquet", index=False)

    return summary


# ── ALSO TEST LGBM BASELINE ───────────────────────────────────

def lgbm_baseline(features_df):
    """LGBM baseline for comparison."""
    try:
        import lightgbm as lgb
    except:
        print("  LightGBM not available")
        return None

    dates = sorted(features_df['date'].unique())

    train_days = 252
    test_days = 63
    step_days = 21

    all_results = []
    fold = 0

    i = train_days
    while i + test_days <= len(dates):
        train_mask = features_df['date'].isin(dates[max(0, i-train_days):i])
        test_mask = features_df['date'].isin(dates[i:i+test_days])

        train_df = features_df[train_mask]
        test_df = features_df[test_mask]

        if len(train_df) < 500 or len(test_df) < 50:
            i += step_days
            continue

        X_train = train_df[FEATURE_COLS].values
        y_train = (train_df['fwd_ret'].values > THRESHOLD).astype(int)
        X_test = test_df[FEATURE_COLS].values
        y_test = (test_df['fwd_ret'].values > THRESHOLD).astype(int)

        model = lgb.LGBMClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.05,
            num_leaves=31, min_child_samples=20, subsample=0.8,
            colsample_bytree=0.8, n_jobs=-1, verbose=-1,
        )
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs > 0.5).astype(int)

        acc = (preds == y_test).mean()

        fold_results = pd.DataFrame({
            'date': test_df['date'].values,
            'ticker': test_df['ticker'].values,
            'prob': probs,
            'pred': preds,
            'label': y_test,
        })
        all_results.append(fold_results)

        if fold % 10 == 0:
            print(f"  LGBM Fold {fold}: acc={acc:.3f}")

        i += step_days
        fold += 1

    return pd.concat(all_results, ignore_index=True)


# ── MAIN ───────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("="*60)
    print("  GPU TREND PREDICTOR v1")
    print("  Transformer for Multi-Asset Trend Timing")
    print("="*60)

    print("\n[1/4] Downloading data...")
    data = download_data()

    print("\n[2/4] Computing features...")
    features = compute_features(data)
    print(f"  {len(features)} rows, {features['ticker'].nunique()} tickers")
    print(f"  Date range: {features['date'].min()} to {features['date'].max()}")

    base_rate = (features['fwd_ret'] > THRESHOLD).mean()
    print(f"  Base rate (fwd_ret > {THRESHOLD:.0%}): {base_rate:.3f}")

    print(f"\n[3/4] Walk-forward Transformer training on {DEVICE}...")
    results = walk_forward(features)
    summary_tf = evaluate(results, features)

    print(f"\n[4/4] LGBM baseline...")
    lgbm_results = lgbm_baseline(features)
    if lgbm_results is not None:
        summary_lgbm = evaluate(lgbm_results, features)

    print(f"\n{'='*60}")
    print(f"  COMPLETE — {(time.time()-t0)/60:.1f} minutes")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
