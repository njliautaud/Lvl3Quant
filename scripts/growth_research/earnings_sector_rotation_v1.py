#!/usr/bin/env python3
"""
Earnings Sector Rotation v1 — Earnings Surprise as Sector Rotation Signal
==========================================================================
HYPOTHESIS: Aggregating individual stock earnings surprises within each sector
creates a predictive signal for sector ETF returns. Sectors where companies beat
estimates → expect ETF rally → go long. Sectors where companies miss → go short.

KEY QUESTION: Do earnings features improve ML alpha from 1.12x to 1.5x+?

Variants:
  A: LGBM momentum-only features (baseline — same as V8)
  B: LGBM momentum + earnings features
  C: MLP with all features (GPU)
  D: LGBM earnings-only features (control)

Features (~20 total):
  - Sector earnings surprise score (avg surprise of recent reporters)
  - Post-earnings drift signal
  - Earnings density (# of reports per week in sector)
  - Momentum features (5d, 21d, 63d returns)
  - Vol features (ATR, 21d realized vol)
  - Cross-sector relative strength, dispersion

Walk-forward: 250d sliding, weekly rebalance, 2-week forward target
Adversarial: 5-gate validation
Backtest: $645 starting capital, sector spread portfolio

Output: C:\\Users\\claude\\Lvl3Quant\\output\\growth_research\\earnings_sector_rotation_v1\\
MLflow: http://jupiter:5000
"""

import json
import os
import sys
import time
import warnings
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

print = partial(print, flush=True)
warnings.filterwarnings('ignore')
np.random.seed(42)

# ---------------------------------------------------------------------------
# MLflow setup
# ---------------------------------------------------------------------------
MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen('http://jupiter:5000/', timeout=3)
    import mlflow
    mlflow.set_tracking_uri('http://jupiter:5000')
    MLFLOW_OK = True
    print("[MLflow] Connected to http://jupiter:5000")
except Exception:
    print("[MLflow] Not available, results logged locally only")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SECTORS = ['XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY']
SECTOR_NAMES = {
    'XLB': 'Materials', 'XLC': 'Communications', 'XLE': 'Energy',
    'XLF': 'Financials', 'XLI': 'Industrials', 'XLK': 'Technology',
    'XLP': 'Consumer Staples', 'XLRE': 'Real Estate', 'XLU': 'Utilities',
    'XLV': 'Healthcare', 'XLY': 'Consumer Disc'
}

# Sector → representative constituent tickers (top holdings for earnings proxy)
SECTOR_CONSTITUENTS = {
    'XLK': ['AAPL', 'MSFT', 'NVDA', 'AVGO', 'CRM', 'ADBE', 'CSCO', 'ACN', 'ORCL', 'IBM',
            'INTC', 'AMD', 'QCOM', 'TXN', 'AMAT', 'INTU', 'NOW', 'ADI', 'LRCX', 'SNPS'],
    'XLF': ['BRK-B', 'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'SPGI', 'BLK',
            'C', 'AXP', 'SCHW', 'CB', 'MMC', 'PGR', 'ICE', 'CME', 'AON', 'MET'],
    'XLE': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'PXD', 'OXY',
            'WMB', 'HES', 'DVN', 'HAL', 'FANG', 'BKR', 'TRGP', 'KMI', 'OKE', 'CTRA'],
    'XLV': ['UNH', 'JNJ', 'LLY', 'ABBV', 'MRK', 'PFE', 'TMO', 'ABT', 'DHR', 'AMGN',
            'BMY', 'ISRG', 'SYK', 'VRTX', 'GILD', 'MDT', 'REGN', 'CI', 'ELV', 'ZTS'],
    'XLI': ['GE', 'CAT', 'HON', 'UNP', 'UPS', 'RTX', 'BA', 'DE', 'LMT', 'ADP',
            'MMM', 'FDX', 'GD', 'NSC', 'NOC', 'WM', 'CSX', 'ITW', 'EMR', 'ETN'],
    'XLY': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'LOW', 'SBUX', 'TJX', 'BKNG', 'CMG',
            'F', 'GM', 'ORLY', 'AZO', 'ROST', 'DHI', 'LEN', 'MAR', 'HLT', 'YUM'],
    'XLP': ['PG', 'PEP', 'KO', 'COST', 'WMT', 'PM', 'MDLZ', 'MO', 'CL', 'KMB',
            'GIS', 'SJM', 'K', 'HSY', 'STZ', 'KHC', 'TAP', 'CAG', 'CPB', 'HRL'],
    'XLU': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'SRE', 'EXC', 'XEL', 'ED', 'WEC',
            'ES', 'AWK', 'DTE', 'AEE', 'CMS', 'PPL', 'FE', 'ETR', 'CEG', 'PEG'],
    'XLB': ['LIN', 'APD', 'SHW', 'ECL', 'FCX', 'NEM', 'NUE', 'VMC', 'MLM', 'DOW',
            'DD', 'PPG', 'IFF', 'CE', 'ALB', 'EMN', 'PKG', 'IP', 'CF', 'MOS'],
    'XLRE': ['PLD', 'AMT', 'CCI', 'EQIX', 'PSA', 'SPG', 'O', 'WELL', 'DLR', 'AVB',
             'EQR', 'ARE', 'VTR', 'MAA', 'UDR', 'KIM', 'REG', 'HST', 'CPT', 'BXP'],
    'XLC': ['META', 'GOOGL', 'GOOG', 'DIS', 'CMCSA', 'NFLX', 'T', 'VZ', 'TMUS', 'CHTR',
            'EA', 'TTWO', 'WBD', 'OMC', 'IPG', 'FOXA', 'FOX', 'PARA', 'LYV', 'MTCH'],
}

TRAIN_DAYS = 250          # sliding window
FWD_HORIZON = 10          # 2-week forward (trading days)
REBALANCE_FREQ = 5        # weekly rebalance
STARTING_CAPITAL = 645.0
COMMISSION = 2.60         # per spread
TOP_N = 3
BOTTOM_N = 3
VIX_THRESHOLD = 20.0
HAIRCUT = 0.15
EARLY_EXIT_DAYS = 10      # match forward horizon

# MLP config
HIDDEN_DIMS = [64, 32, 16]
DROPOUT = 0.3
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 50
PATIENCE = 10
BATCH_SIZE = 64

# Determine output directory based on platform
if sys.platform == 'win32':
    OUTPUT_DIR = Path(r'C:\Users\claude\Lvl3Quant\output\growth_research\earnings_sector_rotation_v1')
else:
    OUTPUT_DIR = Path('/home/jupiter/Lvl3Quant/output/growth_research/earnings_sector_rotation_v1')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# GPU setup
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    HAS_TORCH = True
    print(f"[PyTorch] Device: {DEVICE}")
except ImportError:
    HAS_TORCH = False
    DEVICE = None
    print("[PyTorch] Not available — MLP variant C will be skipped")


def ts(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ═══════════════════════════════════════════════════════════════════════════
# 1. DATA DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════════

def download_sector_data():
    """Download sector ETF + SPY + VIX data."""
    import yfinance as yf

    tickers = SECTORS + ['SPY', '^VIX', 'TLT', 'HYG']
    ts(f"Downloading {len(tickers)} sector ETFs + macro data...")
    data = yf.download(tickers, start='2015-01-01', auto_adjust=True, progress=False)

    close = data['Close'].copy()
    volume = data['Volume'].copy()

    # Rename VIX
    close = close.rename(columns={'^VIX': 'VIX'})
    if '^VIX' in volume.columns:
        volume = volume.rename(columns={'^VIX': 'VIX'})

    close = close.ffill().dropna(how='all')
    volume = volume.ffill().dropna(how='all')

    ts(f"Sector data: {close.index[0].date()} to {close.index[-1].date()}, {len(close)} days")
    return close, volume


def download_constituent_data():
    """Download constituent stock prices for earnings proxy computation.
    Uses volume spikes + gap openings as earnings date proxies.
    """
    import yfinance as yf

    # Flatten all constituents
    all_tickers = set()
    for sector, tickers in SECTOR_CONSTITUENTS.items():
        all_tickers.update(tickers)
    all_tickers = sorted(all_tickers)

    ts(f"Downloading {len(all_tickers)} constituent stocks for earnings proxy...")

    # Batch download
    data = yf.download(all_tickers, start='2015-01-01', auto_adjust=True, progress=False)

    close = data['Close'].copy() if 'Close' in data.columns.get_level_values(0) else data['Close']
    volume = data['Volume'].copy() if 'Volume' in data.columns.get_level_values(0) else data['Volume']

    # Handle single-ticker case
    if isinstance(close, pd.Series):
        close = close.to_frame(all_tickers[0])
        volume = volume.to_frame(all_tickers[0])

    close = close.ffill()
    volume = volume.ffill()

    ts(f"Constituent data: {len(close)} days, {close.shape[1]} stocks")
    return close, volume


def detect_earnings_dates(stock_close, stock_volume, ticker):
    """Detect likely earnings dates using volume spikes + absolute gap returns.

    Earnings proxy: days where volume > 2x 20d avg AND |gap return| > 1.5%.
    This captures the vast majority of actual earnings announcements.
    """
    if ticker not in stock_close.columns or ticker not in stock_volume.columns:
        return pd.Series(dtype=float)

    px = stock_close[ticker].dropna()
    vol = stock_volume[ticker].dropna()

    # Align
    common_idx = px.index.intersection(vol.index)
    px = px.loc[common_idx]
    vol = vol.loc[common_idx]

    if len(px) < 30:
        return pd.Series(dtype=float)

    # Gap return (open proxy = today's close vs yesterday's close, 1-day return)
    ret_1d = px.pct_change()

    # Volume ratio vs 20d median
    vol_median_20 = vol.rolling(20, min_periods=10).median()
    vol_ratio = vol / (vol_median_20 + 1)

    # Earnings proxy: volume spike + large move
    is_earnings = (vol_ratio > 2.0) & (ret_1d.abs() > 0.015)

    # Filter: at least 20 days between earnings (quarterly cadence)
    earnings_dates = is_earnings[is_earnings].index
    filtered = []
    last_date = None
    for dt in earnings_dates:
        if last_date is None or (dt - last_date).days >= 20:
            filtered.append(dt)
            last_date = dt

    # Earnings surprise proxy = the 1-day return on earnings day
    # Positive return = beat, negative = miss
    surprise = ret_1d.loc[filtered] if filtered else pd.Series(dtype=float)

    return surprise


# ═══════════════════════════════════════════════════════════════════════════
# 2. EARNINGS FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════

def compute_sector_earnings_features(sector_close, stock_close, stock_volume):
    """Compute earnings-based features for each sector.

    Returns dict: sector -> DataFrame of earnings features (daily frequency).
    """
    ts("Computing sector earnings features...")

    earnings_features = {}

    for sector in SECTORS:
        constituents = SECTOR_CONSTITUENTS.get(sector, [])
        available = [t for t in constituents if t in stock_close.columns]

        if len(available) < 3:
            ts(f"  {sector}: only {len(available)} constituents, skipping")
            earnings_features[sector] = pd.DataFrame(index=sector_close.index)
            continue

        # Get earnings surprise for each constituent
        all_surprises = {}
        for ticker in available:
            surprise = detect_earnings_dates(stock_close, stock_volume, ticker)
            if len(surprise) > 0:
                all_surprises[ticker] = surprise

        feats = pd.DataFrame(index=sector_close.index, dtype=float)

        if len(all_surprises) < 2:
            earnings_features[sector] = feats
            continue

        # 1. Sector Earnings Surprise Score: rolling avg of constituent surprises
        # Spread surprise values onto daily index, then compute rolling stats
        daily_surprise = pd.DataFrame(index=sector_close.index, columns=list(all_surprises.keys()), dtype=float)
        for ticker, surprises in all_surprises.items():
            for dt, val in surprises.items():
                if dt in daily_surprise.index:
                    daily_surprise.loc[dt, ticker] = val

        # Count of reports per day
        daily_report_count = daily_surprise.notna().sum(axis=1)

        # Rolling 21d average surprise (across all constituents that reported)
        surprise_flat = daily_surprise.stack().reset_index()
        surprise_flat.columns = ['date', 'ticker', 'surprise']
        surprise_flat = surprise_flat.set_index('date').sort_index()

        # Rolling 21d earnings surprise score
        rolling_surprise = pd.Series(0.0, index=sector_close.index)
        rolling_density = pd.Series(0.0, index=sector_close.index)
        rolling_beat_rate = pd.Series(0.5, index=sector_close.index)
        rolling_surprise_std = pd.Series(0.0, index=sector_close.index)

        for i in range(21, len(sector_close.index)):
            window_start = sector_close.index[i - 21]
            window_end = sector_close.index[i]

            # Get all surprises in this 21d window
            mask = (surprise_flat.index >= window_start) & (surprise_flat.index <= window_end)
            window_surprises = surprise_flat.loc[mask, 'surprise']

            if len(window_surprises) > 0:
                rolling_surprise.iloc[i] = window_surprises.mean()
                rolling_density.iloc[i] = len(window_surprises)
                rolling_beat_rate.iloc[i] = (window_surprises > 0).mean()
                rolling_surprise_std.iloc[i] = window_surprises.std() if len(window_surprises) > 1 else 0

        feats['earnings_surprise_21d'] = rolling_surprise
        feats['earnings_density_21d'] = rolling_density
        feats['earnings_beat_rate_21d'] = rolling_beat_rate
        feats['earnings_surprise_std_21d'] = rolling_surprise_std

        # 2. Post-Earnings Drift Signal: cumulative surprise × momentum interaction
        # Strong beat + continuing momentum = PEAD still running
        if sector in sector_close.columns:
            sector_mom_5d = sector_close[sector].pct_change(5)
            feats['pead_signal'] = rolling_surprise * sector_mom_5d.clip(-0.1, 0.1)

        # 3. Earnings Momentum: is the surprise trend improving or worsening?
        # Compare last 21d avg surprise to previous 21d
        feats['earnings_momentum'] = rolling_surprise - rolling_surprise.shift(21)

        # 4. Earnings Confidence: density × absolute surprise magnitude
        # More reporters + bigger surprises = stronger signal
        feats['earnings_confidence'] = rolling_density * rolling_surprise.abs()

        earnings_features[sector] = feats
        n_events = sum(len(s) for s in all_surprises.values())
        ts(f"  {sector}: {len(all_surprises)} constituents, {n_events} earnings events detected")

    return earnings_features


# ═══════════════════════════════════════════════════════════════════════════
# 3. MOMENTUM / VOL / CROSS-SECTOR FEATURES
# ═══════════════════════════════════════════════════════════════════════════

def compute_momentum_features(sector_close, volume_df):
    """Compute momentum, vol, and cross-sector features per sector."""
    ts("Computing momentum / vol / cross-sector features...")

    available_sectors = [s for s in SECTORS if s in sector_close.columns]
    sector_returns = sector_close[available_sectors].pct_change()
    spy = sector_close['SPY'] if 'SPY' in sector_close.columns else None
    vix = sector_close['VIX'] if 'VIX' in sector_close.columns else None
    tlt = sector_close['TLT'] if 'TLT' in sector_close.columns else None
    hyg = sector_close['HYG'] if 'HYG' in sector_close.columns else None

    momentum_features = {}

    for sector in available_sectors:
        px = sector_close[sector]
        ret = sector_returns[sector]
        feats = pd.DataFrame(index=sector_close.index)

        # Momentum
        feats['ret_5d'] = px.pct_change(5)
        feats['ret_21d'] = px.pct_change(21)
        feats['ret_63d'] = px.pct_change(63)
        feats['ret_126d'] = px.pct_change(126)

        # Volatility
        feats['vol_21d'] = ret.rolling(21).std()
        feats['vol_63d'] = ret.rolling(63).std()
        feats['vol_ratio'] = feats['vol_21d'] / (feats['vol_63d'] + 1e-8)

        # ATR proxy (21d high-low range / close)
        feats['atr_21d'] = px.rolling(21).apply(
            lambda x: (x.max() - x.min()) / x[-1] if x[-1] > 0 else 0, raw=True
        )

        # Max drawdown 63d
        feats['maxdd_63d'] = px.rolling(63).apply(
            lambda x: (x / np.maximum.accumulate(x) - 1).min(), raw=True
        )

        # % of 52w high
        feats['pct_52w_high'] = px / px.rolling(252).max()

        # Relative strength vs SPY
        if spy is not None:
            feats['rel_str_21d'] = px.pct_change(21) - spy.pct_change(21)
            feats['rel_str_63d'] = px.pct_change(63) - spy.pct_change(63)

        # Cross-sectional rank
        mom_21d_all = sector_returns[available_sectors].rolling(21).sum()
        rank = mom_21d_all.rank(axis=1, pct=True)
        feats['xs_rank_21d'] = rank[sector] if sector in rank.columns else 0.5

        # Cross-sector dispersion (opportunity indicator)
        feats['xs_dispersion'] = sector_returns[available_sectors].rolling(21).std().mean(axis=1)

        # Macro features
        if vix is not None:
            feats['vix_level'] = vix
            feats['vix_chg_5d'] = vix.pct_change(5)
        if tlt is not None:
            feats['tlt_ret_21d'] = tlt.pct_change(21)
        if hyg is not None:
            feats['credit_proxy'] = hyg.pct_change(21) - (tlt.pct_change(21) if tlt is not None else 0)
        if spy is not None:
            sma200 = spy.rolling(200).mean()
            feats['spy_above_sma200'] = (spy > sma200).astype(float)

        momentum_features[sector] = feats

    return momentum_features


# ═══════════════════════════════════════════════════════════════════════════
# 4. COMBINE FEATURES + DEFINE VARIANTS
# ═══════════════════════════════════════════════════════════════════════════

MOMENTUM_ONLY_FEATURES = [
    'ret_5d', 'ret_21d', 'ret_63d', 'ret_126d',
    'vol_21d', 'vol_63d', 'vol_ratio', 'atr_21d',
    'maxdd_63d', 'pct_52w_high', 'rel_str_21d', 'xs_rank_21d',
]

EARNINGS_FEATURES = [
    'earnings_surprise_21d', 'earnings_density_21d', 'earnings_beat_rate_21d',
    'earnings_surprise_std_21d', 'pead_signal', 'earnings_momentum',
    'earnings_confidence',
]

EXTRA_FEATURES = [
    'xs_dispersion', 'vix_level', 'vix_chg_5d', 'tlt_ret_21d',
    'credit_proxy', 'spy_above_sma200', 'rel_str_63d',
]

VARIANT_FEATURES = {
    'A': MOMENTUM_ONLY_FEATURES,                                      # baseline
    'B': MOMENTUM_ONLY_FEATURES + EARNINGS_FEATURES,                  # mom + earnings
    'C': MOMENTUM_ONLY_FEATURES + EARNINGS_FEATURES + EXTRA_FEATURES, # all (MLP)
    'D': EARNINGS_FEATURES,                                           # earnings only (control)
}


def merge_features(momentum_features, earnings_features, feature_cols):
    """Merge momentum and earnings features for each sector."""
    merged = {}
    for sector in SECTORS:
        mom = momentum_features.get(sector, pd.DataFrame())
        earn = earnings_features.get(sector, pd.DataFrame())

        combined = pd.DataFrame(index=mom.index if len(mom) > 0 else earn.index)
        for col in feature_cols:
            if col in mom.columns:
                combined[col] = mom[col]
            elif col in earn.columns:
                combined[col] = earn[col]
            else:
                combined[col] = 0.0

        merged[sector] = combined

    return merged


# ═══════════════════════════════════════════════════════════════════════════
# 5. MLP MODEL
# ═══════════════════════════════════════════════════════════════════════════

if HAS_TORCH:
    class SectorRankerMLP(nn.Module):
        def __init__(self, input_dim, hidden_dims=None, dropout=0.3):
            super().__init__()
            if hidden_dims is None:
                hidden_dims = [64, 32, 16]
            layers = []
            prev_dim = input_dim
            for h in hidden_dims:
                layers.append(nn.Linear(prev_dim, h))
                layers.append(nn.BatchNorm1d(h))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                prev_dim = h
            layers.append(nn.Linear(prev_dim, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x).squeeze(-1)


    def train_mlp(X_train, y_train, input_dim):
        model = SectorRankerMLP(input_dim, HIDDEN_DIMS, DROPOUT).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        criterion = nn.MSELoss()

        X_t = torch.FloatTensor(X_train).to(DEVICE)
        y_t = torch.FloatTensor(y_train).to(DEVICE)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

        best_loss = float('inf')
        patience_counter = 0
        best_state = None

        model.train()
        for epoch in range(EPOCHS):
            epoch_loss = 0.0
            n_batches = 0
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        return model


    def predict_mlp(model, X_test):
        model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X_test).to(DEVICE)
            preds = model(X_t).cpu().numpy()
        return preds


# ═══════════════════════════════════════════════════════════════════════════
# 6. LGBM MODEL
# ═══════════════════════════════════════════════════════════════════════════

def train_lgbm(X_train, y_train):
    try:
        import lightgbm as lgb
        params = {
            'objective': 'regression',
            'metric': 'mse',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'n_jobs': -1,
        }
        dtrain = lgb.Dataset(X_train, label=y_train)
        model = lgb.train(params, dtrain, num_boost_round=200)
        return model
    except ImportError:
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        return model


def predict_lgbm(model, X_test):
    try:
        import lightgbm as lgb
        if isinstance(model, lgb.Booster):
            return model.predict(X_test)
    except ImportError:
        pass
    return model.predict(X_test)


# ═══════════════════════════════════════════════════════════════════════════
# 7. WALK-FORWARD ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def build_panel(feature_frames, fwd_ret, feature_cols, dates):
    """Build panel: rows = (date, sector), cols = features + target."""
    rows = []
    for dt in dates:
        for sector in SECTORS:
            if sector not in feature_frames:
                continue
            feat_df = feature_frames[sector]
            if dt not in feat_df.index:
                continue
            feat_row = feat_df.loc[dt, feature_cols].values.astype(float)
            target = fwd_ret.loc[dt, sector] if (dt in fwd_ret.index and sector in fwd_ret.columns) else np.nan
            rows.append(np.concatenate([feat_row, [target]]))

    if not rows:
        return np.array([]).reshape(0, len(feature_cols)), np.array([])

    arr = np.array(rows, dtype=float)
    return arr[:, :-1], arr[:, -1]


def walk_forward(feature_frames, fwd_ret, variant, sector_close, vix_series):
    """Run walk-forward for a variant. Returns daily P&L series + trade list."""
    feature_cols = VARIANT_FEATURES[variant]
    n_features = len(feature_cols)

    # Get valid dates
    first_sector = [s for s in SECTORS if s in feature_frames][0]
    valid_dates = feature_frames[first_sector].dropna().index
    for s in SECTORS:
        if s in feature_frames and len(feature_frames[s]) > 0:
            valid_dates = valid_dates.intersection(feature_frames[s].dropna().index)
    valid_dates = valid_dates.intersection(fwd_ret.dropna(how='all').index)
    valid_dates = sorted(valid_dates)

    if len(valid_dates) < TRAIN_DAYS + REBALANCE_FREQ:
        ts(f"  Variant {variant}: insufficient data ({len(valid_dates)} days)")
        return pd.Series(dtype=float), []

    all_trades = []
    n_rebalances = 0
    model = None
    last_retrain = -999

    ts(f"  Variant {variant} ({len(feature_cols)} features): walk-forward over {len(valid_dates) - TRAIN_DAYS} days...")

    for i in range(TRAIN_DAYS, len(valid_dates), REBALANCE_FREQ):
        # Retrain every REBALANCE_FREQ days (weekly)
        train_start = max(0, i - TRAIN_DAYS)
        train_end = i - FWD_HORIZON  # avoid lookahead

        if train_end <= train_start + 60:
            continue

        train_dates = valid_dates[train_start:train_end]

        X_train, y_train = build_panel(feature_frames, fwd_ret, feature_cols, train_dates)

        # Remove NaN rows
        mask = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
        X_train, y_train = X_train[mask], y_train[mask]

        if len(X_train) < 100:
            continue

        # Normalize
        feat_mean = X_train.mean(axis=0)
        feat_std = X_train.std(axis=0) + 1e-8
        X_train_norm = (X_train - feat_mean) / feat_std

        # Train
        if variant == 'C' and HAS_TORCH:
            model = train_mlp(X_train_norm, y_train, n_features)
        else:
            model = train_lgbm(X_train_norm, y_train)

        n_rebalances += 1

        # Predict on rebalance date
        oot_dt = valid_dates[i]

        X_oot = []
        valid_sectors = []
        for sector in SECTORS:
            if sector not in feature_frames:
                continue
            feat_df = feature_frames[sector]
            if oot_dt not in feat_df.index:
                continue
            row = feat_df.loc[oot_dt, feature_cols].values.astype(float)
            if not np.isnan(row).any():
                X_oot.append(row)
                valid_sectors.append(sector)

        if len(X_oot) < 6:
            continue

        X_oot = np.array(X_oot)
        X_oot_norm = (X_oot - feat_mean) / feat_std

        # Get predictions
        if variant == 'C' and HAS_TORCH:
            scores = predict_mlp(model, X_oot_norm)
        else:
            scores = predict_lgbm(model, X_oot_norm)

        # Rank sectors
        ranked_indices = np.argsort(scores)[::-1]
        top_sectors = [valid_sectors[j] for j in ranked_indices[:TOP_N]]
        bottom_sectors = [valid_sectors[j] for j in ranked_indices[-BOTTOM_N:]]

        # VIX regime
        vix_val = vix_series.loc[oot_dt] if oot_dt in vix_series.index else 20.0

        if vix_val >= VIX_THRESHOLD:
            trade_sectors = top_sectors
            direction = 'bull'
        else:
            trade_sectors = bottom_sectors
            direction = 'bear'

        # Compute trade P&L
        for sector in trade_sectors:
            if sector not in sector_close.columns:
                continue

            entry_price = sector_close.loc[oot_dt, sector]
            entry_loc = sector_close.index.get_loc(oot_dt)

            # ATR proxy
            lookback_start = max(0, entry_loc - 21)
            lookback_slice = sector_close[sector].iloc[lookback_start:entry_loc + 1]
            atr_pct = (lookback_slice.max() - lookback_slice.min()) / entry_price if entry_price > 0 else 0.02

            # Spread parameters
            spread_width = entry_price * atr_pct
            max_profit = spread_width * (1 - HAIRCUT) * 100  # per contract

            risk_per_trade = spread_width * 100  # max loss per contract
            if risk_per_trade <= 0:
                continue
            n_contracts = max(1, int(STARTING_CAPITAL / (3 * risk_per_trade)))

            # Exit
            exit_loc = min(entry_loc + EARLY_EXIT_DAYS, len(sector_close) - 1)
            exit_price = sector_close[sector].iloc[exit_loc]
            actual_ret = (exit_price - entry_price) / entry_price

            if direction == 'bull':
                pnl_pct = min(actual_ret / atr_pct, 1.0) if atr_pct > 0 else 0
                trade_pnl = pnl_pct * max_profit * n_contracts - COMMISSION
            else:
                pnl_pct = min(-actual_ret / atr_pct, 1.0) if atr_pct > 0 else 0
                trade_pnl = pnl_pct * max_profit * n_contracts - COMMISSION

            max_loss = -risk_per_trade * n_contracts - COMMISSION
            trade_pnl = max(trade_pnl, max_loss)

            all_trades.append({
                'date': oot_dt,
                'sector': sector,
                'direction': direction,
                'pnl': trade_pnl,
                'n_contracts': n_contracts,
                'score': float(scores[valid_sectors.index(sector)]),
                'vix': float(vix_val),
            })

    ts(f"  Variant {variant}: {len(all_trades)} trades, {n_rebalances} rebalances")
    return all_trades


# ═══════════════════════════════════════════════════════════════════════════
# 8. METRICS
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(trades, starting_capital=STARTING_CAPITAL):
    """Compute performance metrics from trade list."""
    if not trades:
        return {'sharpe': 0, 'sortino': 0, 'pf': 0, 'wr': 0, 'total_ret': 0,
                'max_dd': 0, 'n_trades': 0, 'cagr': 0, 'avg_trade': 0}

    df = pd.DataFrame(trades) if isinstance(trades, list) else trades
    daily_pnl = df.groupby('date')['pnl'].sum()

    equity = starting_capital + daily_pnl.cumsum()
    returns = daily_pnl / starting_capital

    ann_factor = np.sqrt(52)  # weekly rebalance

    avg_ret = returns.mean()
    std_ret = returns.std() + 1e-8
    sharpe = avg_ret / std_ret * ann_factor

    downside = returns[returns < 0].std() + 1e-8
    sortino = avg_ret / downside * ann_factor

    wins = daily_pnl[daily_pnl > 0].sum()
    losses = abs(daily_pnl[daily_pnl < 0].sum()) + 1e-8
    pf = wins / losses

    wr = (daily_pnl > 0).mean()
    total_ret = daily_pnl.sum() / starting_capital

    # Max drawdown
    cum = daily_pnl.cumsum()
    running_max = cum.cummax()
    dd = cum - running_max
    max_dd = dd.min() / starting_capital if len(dd) > 0 else 0

    # CAGR
    n_days = (daily_pnl.index[-1] - daily_pnl.index[0]).days
    n_years = n_days / 365.25 if n_days > 0 else 1
    end_equity = starting_capital + daily_pnl.sum()
    cagr = (end_equity / starting_capital) ** (1 / n_years) - 1 if end_equity > 0 else -1

    return {
        'sharpe': round(float(sharpe), 3),
        'sortino': round(float(sortino), 3),
        'pf': round(float(pf), 3),
        'wr': round(float(wr), 3),
        'total_ret': round(float(total_ret), 4),
        'max_dd': round(float(max_dd), 4),
        'n_trades': len(df),
        'cagr': round(float(cagr), 4),
        'avg_trade': round(float(daily_pnl.mean()), 2),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 9. ADVERSARIAL VALIDATION (5 GATES)
# ═══════════════════════════════════════════════════════════════════════════

def gate1_permutation(trades, n_trials=200):
    """Permutation test: shuffle trade P&L, check if real Sharpe > random."""
    if not trades:
        return {'pass': False, 'p_value': 1.0}

    real_metrics = compute_metrics(trades)
    real_sharpe = real_metrics['sharpe']

    pnls = [t['pnl'] for t in trades]
    count_worse = 0

    for _ in range(n_trials):
        shuffled = pnls.copy()
        np.random.shuffle(shuffled)
        shuffled_trades = [dict(t, pnl=p) for t, p in zip(trades, shuffled)]
        rand_sharpe = compute_metrics(shuffled_trades)['sharpe']
        if rand_sharpe >= real_sharpe:
            count_worse += 1

    p_value = count_worse / n_trials
    return {'pass': p_value < 0.05, 'p_value': round(p_value, 4)}


def gate2_regime_stability(trades, spy_close):
    """Regime stability: |Sharpe_bull - Sharpe_bear| / max < 0.50."""
    if not trades:
        return {'pass': False, 'gap': 1.0}

    df = pd.DataFrame(trades)
    spy_ret = spy_close.pct_change()

    # SPY 63d trailing return for regime classification
    spy_cum = spy_ret.rolling(63, min_periods=21).sum()

    bull_trades = []
    bear_trades = []
    for _, row in df.iterrows():
        dt = row['date']
        if dt in spy_cum.index:
            if spy_cum.loc[dt] > 0:
                bull_trades.append(row.to_dict())
            else:
                bear_trades.append(row.to_dict())

    sharpe_bull = compute_metrics(bull_trades)['sharpe'] if bull_trades else 0
    sharpe_bear = compute_metrics(bear_trades)['sharpe'] if bear_trades else 0

    max_sharpe = max(abs(sharpe_bull), abs(sharpe_bear), 0.01)
    gap = abs(sharpe_bull - sharpe_bear) / max_sharpe

    return {
        'pass': gap < 0.50,
        'gap': round(gap, 3),
        'sharpe_bull': round(sharpe_bull, 3),
        'sharpe_bear': round(sharpe_bear, 3),
    }


def gate3_subperiod(trades):
    """Both halves must have Sharpe > 0.5."""
    if len(trades) < 20:
        return {'pass': False, 'sharpe_h1': 0, 'sharpe_h2': 0}

    mid = len(trades) // 2
    h1 = trades[:mid]
    h2 = trades[mid:]

    sharpe_h1 = compute_metrics(h1)['sharpe']
    sharpe_h2 = compute_metrics(h2)['sharpe']

    return {
        'pass': sharpe_h1 > 0.5 and sharpe_h2 > 0.5,
        'sharpe_h1': round(sharpe_h1, 3),
        'sharpe_h2': round(sharpe_h2, 3),
    }


def gate4_vs_random(trades, n_trials=100):
    """Must beat random sector selection by >20% Sharpe."""
    if not trades:
        return {'pass': False, 'improvement': 0}

    real_sharpe = compute_metrics(trades)['sharpe']

    random_sharpes = []
    for _ in range(n_trials):
        random_trades = []
        for t in trades:
            random_t = t.copy()
            random_t['pnl'] = t['pnl'] * np.random.choice([-1, 1])
            random_trades.append(random_t)
        random_sharpes.append(compute_metrics(random_trades)['sharpe'])

    avg_random = np.mean(random_sharpes)
    improvement = (real_sharpe - avg_random) / (abs(avg_random) + 0.01) if avg_random != 0 else real_sharpe

    return {
        'pass': improvement > 0.20,
        'improvement': round(improvement, 3),
        'real_sharpe': round(real_sharpe, 3),
        'avg_random_sharpe': round(avg_random, 3),
    }


def gate5_ml_alpha_ratio(metrics_baseline, metrics_earnings):
    """Key question: does adding earnings features improve ML alpha?
    Target: 1.5x improvement (from baseline's 1.12x).
    """
    sharpe_base = metrics_baseline['sharpe']
    sharpe_earn = metrics_earnings['sharpe']

    if sharpe_base <= 0:
        ratio = sharpe_earn / 0.01 if sharpe_earn > 0 else 0
    else:
        ratio = sharpe_earn / sharpe_base

    return {
        'pass': ratio >= 1.5,
        'ml_alpha_ratio': round(ratio, 3),
        'sharpe_baseline': round(sharpe_base, 3),
        'sharpe_earnings': round(sharpe_earn, 3),
        'target': 1.5,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 10. MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    ts("=" * 70)
    ts("Earnings Sector Rotation v1 — Earnings Surprise as Sector Signal")
    ts("=" * 70)

    # 1. Download data
    sector_close, sector_volume = download_sector_data()
    stock_close, stock_volume = download_constituent_data()

    # 2. Compute features
    earnings_features = compute_sector_earnings_features(sector_close, stock_close, stock_volume)
    momentum_features = compute_momentum_features(sector_close, sector_volume)

    # 3. Forward returns (target)
    available_sectors = [s for s in SECTORS if s in sector_close.columns]
    sector_returns = sector_close[available_sectors].pct_change()
    fwd_ret = sector_returns.rolling(FWD_HORIZON).sum().shift(-FWD_HORIZON)

    # VIX
    vix_series = sector_close['VIX'] if 'VIX' in sector_close.columns else pd.Series(20.0, index=sector_close.index)
    spy_close = sector_close['SPY'] if 'SPY' in sector_close.columns else None

    # 4. Run variants
    results = {}
    all_variant_trades = {}

    for variant in ['A', 'B', 'C', 'D']:
        if variant == 'C' and not HAS_TORCH:
            ts(f"Skipping variant C (no PyTorch)")
            continue

        ts(f"\n{'='*50}")
        ts(f"Running Variant {variant}: {['LGBM momentum-only (baseline)', 'LGBM momentum+earnings', 'MLP all features (GPU)', 'LGBM earnings-only (control)'][ord(variant)-ord('A')]}")
        ts(f"{'='*50}")

        feature_cols = VARIANT_FEATURES[variant]
        merged = merge_features(momentum_features, earnings_features, feature_cols)

        trades = walk_forward(merged, fwd_ret, variant, sector_close, vix_series)

        if trades:
            metrics = compute_metrics(trades)
            results[variant] = metrics
            all_variant_trades[variant] = trades
            ts(f"  Results: Sharpe={metrics['sharpe']}, Sortino={metrics['sortino']}, "
               f"PF={metrics['pf']}, WR={metrics['wr']}, CAGR={metrics['cagr']}")
        else:
            ts(f"  No trades generated")
            results[variant] = {'sharpe': 0, 'sortino': 0, 'pf': 0, 'wr': 0, 'n_trades': 0}

    # 5. Adversarial validation
    ts(f"\n{'='*70}")
    ts("ADVERSARIAL VALIDATION (5 Gates)")
    ts(f"{'='*70}")

    gates = {}
    best_variant = max(results.keys(), key=lambda v: results[v].get('sharpe', 0))
    best_trades = all_variant_trades.get(best_variant, [])

    if best_trades:
        ts(f"\nTesting best variant: {best_variant}")

        gates['G1_permutation'] = gate1_permutation(best_trades)
        ts(f"  Gate 1 (Permutation): {'PASS' if gates['G1_permutation']['pass'] else 'FAIL'} "
           f"(p={gates['G1_permutation']['p_value']})")

        if spy_close is not None:
            gates['G2_regime'] = gate2_regime_stability(best_trades, spy_close)
            ts(f"  Gate 2 (Regime): {'PASS' if gates['G2_regime']['pass'] else 'FAIL'} "
               f"(gap={gates['G2_regime']['gap']}, bull={gates['G2_regime']['sharpe_bull']}, bear={gates['G2_regime']['sharpe_bear']})")

        gates['G3_subperiod'] = gate3_subperiod(best_trades)
        ts(f"  Gate 3 (Sub-period): {'PASS' if gates['G3_subperiod']['pass'] else 'FAIL'} "
           f"(H1={gates['G3_subperiod']['sharpe_h1']}, H2={gates['G3_subperiod']['sharpe_h2']})")

        gates['G4_vs_random'] = gate4_vs_random(best_trades)
        ts(f"  Gate 4 (vs Random): {'PASS' if gates['G4_vs_random']['pass'] else 'FAIL'} "
           f"(improvement={gates['G4_vs_random']['improvement']})")

        # Gate 5: ML alpha ratio (A vs B)
        if 'A' in results and 'B' in results:
            gates['G5_ml_alpha'] = gate5_ml_alpha_ratio(results['A'], results['B'])
            ts(f"  Gate 5 (ML Alpha): {'PASS' if gates['G5_ml_alpha']['pass'] else 'FAIL'} "
               f"(ratio={gates['G5_ml_alpha']['ml_alpha_ratio']}, target=1.5x)")

    gates_passed = sum(1 for g in gates.values() if g.get('pass', False))
    total_gates = len(gates)

    # 6. Summary
    ts(f"\n{'='*70}")
    ts("FINAL RESULTS SUMMARY")
    ts(f"{'='*70}")

    for v in sorted(results.keys()):
        m = results[v]
        label = {
            'A': 'LGBM momentum-only (baseline)',
            'B': 'LGBM momentum+earnings',
            'C': 'MLP all features (GPU)',
            'D': 'LGBM earnings-only (control)',
        }[v]
        ts(f"  {v}: {label}")
        ts(f"     Sharpe={m.get('sharpe',0)}, Sortino={m.get('sortino',0)}, "
           f"PF={m.get('pf',0)}, WR={m.get('wr',0)}, CAGR={m.get('cagr',0)}, "
           f"MaxDD={m.get('max_dd',0)}, Trades={m.get('n_trades',0)}")

    ts(f"\n  Gates passed: {gates_passed}/{total_gates}")

    # Key question
    if 'A' in results and 'B' in results:
        base_sharpe = results['A']['sharpe']
        earn_sharpe = results['B']['sharpe']
        if base_sharpe > 0:
            improvement = earn_sharpe / base_sharpe
            ts(f"\n  KEY QUESTION: Earnings features improve ML alpha by {improvement:.2f}x "
               f"(target: 1.5x)")
            ts(f"  Baseline Sharpe: {base_sharpe}, Earnings-enhanced Sharpe: {earn_sharpe}")
        else:
            ts(f"\n  KEY QUESTION: Baseline Sharpe non-positive ({base_sharpe}), "
               f"earnings Sharpe: {earn_sharpe}")

    # 7. Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'train_days': TRAIN_DAYS,
            'fwd_horizon': FWD_HORIZON,
            'rebalance_freq': REBALANCE_FREQ,
            'starting_capital': STARTING_CAPITAL,
            'top_n': TOP_N,
            'bottom_n': BOTTOM_N,
            'vix_threshold': VIX_THRESHOLD,
        },
        'results': results,
        'gates': gates,
        'gates_passed': gates_passed,
        'total_gates': total_gates,
        'runtime_seconds': round(time.time() - t_start, 1),
    }

    results_path = OUTPUT_DIR / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    ts(f"\nResults saved to {results_path}")

    # Save trade logs
    for v, trades in all_variant_trades.items():
        if trades:
            trades_df = pd.DataFrame(trades)
            trades_path = OUTPUT_DIR / f'trades_variant_{v}.csv'
            trades_df.to_csv(trades_path, index=False)

    # 8. MLflow logging
    if MLFLOW_OK:
        try:
            exp_name = "earnings_sector_rotation_v1"
            mlflow.set_experiment(exp_name)
            with mlflow.start_run(run_name=f"earnings_rotation_{datetime.now().strftime('%Y%m%d_%H%M')}"):
                mlflow.log_params({
                    'train_days': TRAIN_DAYS,
                    'fwd_horizon': FWD_HORIZON,
                    'rebalance_freq': REBALANCE_FREQ,
                    'starting_capital': STARTING_CAPITAL,
                })
                for v, m in results.items():
                    for k, val in m.items():
                        mlflow.log_metric(f"variant_{v}_{k}", val)
                mlflow.log_metric('gates_passed', gates_passed)
                mlflow.log_metric('total_gates', total_gates)
                if 'A' in results and 'B' in results and results['A']['sharpe'] > 0:
                    mlflow.log_metric('earnings_alpha_ratio', results['B']['sharpe'] / results['A']['sharpe'])
                mlflow.log_artifact(str(results_path))
            ts("[MLflow] Run logged successfully")
        except Exception as e:
            ts(f"[MLflow] Logging failed: {e}")

    elapsed = time.time() - t_start
    ts(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    ts("Done.")


if __name__ == '__main__':
    main()
