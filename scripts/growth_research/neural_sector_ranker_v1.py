#!/usr/bin/env python3
"""
Neural Sector Ranker v1 — MLP vs LightGBM for sector ranking
==============================================================
Tests whether a small neural network can outperform LightGBM for
ranking 11 sector ETFs in a bull-call-spread options strategy.

Variants:
  A: LGBM Baseline (100 trees, depth 4, lr 0.05)
  B: MLP Small (2-layer 64-32, dropout 0.2, 50 epochs)
  C: MLP Large (3-layer 128-64-32, dropout 0.3, 100 epochs)
  D: MLP + Attention (self-attention on features before MLP)
  E: Ensemble (average ranking from LGBM + MLP Small)

Walk-forward: 500d train, 250d test, biweekly rebalance, sliding window.
Options: bull call spread, 3% width, 21 DTE, hold-to-expiry (intrinsic only).
Capital: $645, $200 max/trade, 15% entry haircut, $2.60 commission.
VIX > 20 regime filter.

5-gate adversarial validation on each variant.
MLflow tracking: experiment "neural_sector_ranker_v1".

Cross-platform (pathlib), GPU-accelerated (torch cuda if available).
"""

import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# MLflow setup (non-blocking — runs without MLflow)
# ---------------------------------------------------------------------------
MLFLOW_OK = False
try:
    import urllib.request

    urllib.request.urlopen("http://jupiter:5000/", timeout=3)
    import mlflow

    mlflow.set_tracking_uri("http://jupiter:5000")
    MLFLOW_OK = True
except Exception:
    pass

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
BENCHMARK = "SPY"
VIX_TICKER = "^VIX"
CROSS_ASSETS = ["TLT", "SHY", "HYG", "GLD"]
ALL_TICKERS = SECTORS + [BENCHMARK, VIX_TICKER] + CROSS_ASSETS

TRAIN_DAYS = 500
TEST_DAYS = 250
REBAL_FREQ = 10  # biweekly (every 10 trading days)
FORWARD_DAYS = 21
TOP_N = 3

# Options parameters
SPREAD_WIDTH_PCT = 0.03  # 3% width
DTE = 21
STARTING_CAPITAL = 645.0
MAX_PER_TRADE = 200.0
HAIRCUT = 0.15  # 15% entry haircut on debit
COMMISSION = 2.60  # per spread
IV_MULTIPLIER = 1.2
VIX_THRESHOLD = 20.0

# Feature list (21 features)
FEATURE_NAMES = [
    "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "vol_21d", "vol_63d", "sharpe_63d", "maxdd_63d", "pct_52w_high",
    "mom_accel", "pct_pos_months_12m", "sortino_63d", "calmar_1y",
    "up_capture", "dn_capture", "trend_r2_63d", "trend_slope_63d",
    "rel_vol_21d", "sector_spy_beta_63d",
]
# Additional features computed cross-sectionally
CROSS_FEATURES = ["sector_relative_vol_21d", "cross_sector_dispersion"]

ALL_FEATURES = FEATURE_NAMES + CROSS_FEATURES  # 23 total

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "output" / "neural_sector_ranker_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def ts(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------
def download_data() -> pd.DataFrame:
    """Download sector ETFs + VIX + cross-asset data via yfinance."""
    import yfinance as yf

    ts(f"Downloading {len(ALL_TICKERS)} tickers...")
    data = yf.download(ALL_TICKERS, start="2012-01-01", auto_adjust=True, progress=False)
    close = data["Close"].copy()
    close = close.rename(columns={"^VIX": "VIX"})
    close = close.ffill().dropna(how="all")

    # Drop rows where any sector is missing
    for s in SECTORS:
        close = close.dropna(subset=[s])
    close = close.dropna(subset=["VIX", "SPY"])
    ts(f"Data: {close.index[0].date()} to {close.index[-1].date()}, {len(close)} days")
    return close


# ---------------------------------------------------------------------------
# Feature engineering (21 features + 2 cross-sectional = 23)
# ---------------------------------------------------------------------------
def compute_features(close_df: pd.DataFrame) -> tuple:
    """Compute all 23 features for each sector. Returns feature_frames dict + forward returns."""
    spy = close_df["SPY"]
    spy_ret = spy.pct_change()

    feature_frames = {}

    for sector in SECTORS:
        px = close_df[sector]
        ret_daily = px.pct_change()
        feats = pd.DataFrame(index=close_df.index)

        # --- Momentum features ---
        feats["ret_5d"] = px.pct_change(5)
        feats["ret_10d"] = px.pct_change(10)
        feats["ret_21d"] = px.pct_change(21)
        feats["ret_63d"] = px.pct_change(63)
        feats["ret_126d"] = px.pct_change(126)
        feats["ret_252d"] = px.pct_change(252)

        # --- Volatility features ---
        feats["vol_21d"] = ret_daily.rolling(21).std() * np.sqrt(252)
        feats["vol_63d"] = ret_daily.rolling(63).std() * np.sqrt(252)

        # --- Risk-adjusted ---
        mean_63d = ret_daily.rolling(63).mean() * 252
        std_63d = feats["vol_63d"]
        feats["sharpe_63d"] = mean_63d / (std_63d + 1e-8)

        # Max drawdown 63d
        feats["maxdd_63d"] = px.rolling(63).apply(
            lambda x: (x / np.maximum.accumulate(x) - 1).min(), raw=True
        )

        # Pct of 52-week high
        feats["pct_52w_high"] = px / px.rolling(252).max()

        # Momentum acceleration
        feats["mom_accel"] = px.pct_change(252) - px.pct_change(126)

        # Pct positive months in trailing 12 months
        monthly_ret = ret_daily.rolling(21).sum()
        feats["pct_pos_months_12m"] = monthly_ret.rolling(252).apply(
            lambda x: (x[::21] > 0).mean() if len(x[::21]) > 0 else 0.5, raw=True
        )

        # Sortino 63d
        downside_std = ret_daily.rolling(63).apply(
            lambda x: np.sqrt(np.mean(np.minimum(x, 0) ** 2)) * np.sqrt(252), raw=True
        )
        feats["sortino_63d"] = mean_63d / (downside_std + 1e-8)

        # Calmar 1y: annualized return / abs(max drawdown)
        ann_ret_1y = px.pct_change(252)
        maxdd_1y = px.rolling(252).apply(
            lambda x: (x / np.maximum.accumulate(x) - 1).min(), raw=True
        )
        feats["calmar_1y"] = ann_ret_1y / (np.abs(maxdd_1y) + 1e-8)

        # Up/down capture vs SPY (63d rolling)
        spy_up_mask = spy_ret.rolling(63).apply(lambda x: x[x > 0].sum(), raw=True)
        spy_dn_mask = spy_ret.rolling(63).apply(lambda x: x[x < 0].sum(), raw=True)
        # Simplified up/down capture using rolling correlation approach
        roll_corr = ret_daily.rolling(63).corr(spy_ret)
        vol_ratio = feats["vol_63d"] / (spy_ret.rolling(63).std() * np.sqrt(252) + 1e-8)
        feats["up_capture"] = roll_corr * vol_ratio
        feats["dn_capture"] = feats["up_capture"]  # symmetric approximation

        # Trend strength: R² and slope of 63d linear regression on log prices
        log_px = np.log(px)
        feats["trend_r2_63d"] = log_px.rolling(63).apply(
            lambda y: np.corrcoef(np.arange(len(y)), y)[0, 1] ** 2 if len(y) == 63 else 0,
            raw=True,
        )
        feats["trend_slope_63d"] = log_px.rolling(63).apply(
            lambda y: np.polyfit(np.arange(len(y)), y, 1)[0] if len(y) == 63 else 0,
            raw=True,
        )

        # Relative volatility: sector vol / SPY vol
        spy_vol_21d = spy_ret.rolling(21).std() * np.sqrt(252)
        feats["rel_vol_21d"] = feats["vol_21d"] / (spy_vol_21d + 1e-8)

        # Beta to SPY (63d rolling)
        cov_63 = ret_daily.rolling(63).cov(spy_ret)
        var_spy_63 = spy_ret.rolling(63).var()
        feats["sector_spy_beta_63d"] = cov_63 / (var_spy_63 + 1e-8)

        feature_frames[sector] = feats

    # --- Cross-sectional features (computed across all sectors at each date) ---
    vol_21d_all = pd.DataFrame({s: feature_frames[s]["vol_21d"] for s in SECTORS})
    mean_vol = vol_21d_all.mean(axis=1)
    std_vol = vol_21d_all.std(axis=1)

    ret_5d_all = pd.DataFrame({s: feature_frames[s]["ret_5d"] for s in SECTORS})
    dispersion = ret_5d_all.std(axis=1)

    for sector in SECTORS:
        feature_frames[sector]["sector_relative_vol_21d"] = (
            feature_frames[sector]["vol_21d"] - mean_vol
        ) / (std_vol + 1e-8)
        feature_frames[sector]["cross_sector_dispersion"] = dispersion

    # Forward 21d return (target for ranking)
    sectors_close = close_df[SECTORS]
    fwd_ret = sectors_close.pct_change(FORWARD_DAYS).shift(-FORWARD_DAYS)

    return feature_frames, fwd_ret


# ---------------------------------------------------------------------------
# Neural network architectures
# ---------------------------------------------------------------------------
class MLPSmall(nn.Module):
    """Variant B: 2-layer MLP (64-32), ReLU, dropout 0.2."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MLPLarge(nn.Module):
    """Variant C: 3-layer MLP (128-64-32), ReLU, dropout 0.3."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class FeatureAttention(nn.Module):
    """Simple self-attention over features (treating each feature as a token)."""

    def __init__(self, n_features: int, embed_dim: int = 32, n_heads: int = 2):
        super().__init__()
        self.n_features = n_features
        self.embed_dim = embed_dim
        # Project each scalar feature to embed_dim
        self.feature_embed = nn.Linear(1, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (batch, n_features)
        # Reshape to (batch, n_features, 1) then project
        x_emb = self.feature_embed(x.unsqueeze(-1))  # (batch, n_features, embed_dim)
        attn_out, _ = self.attn(x_emb, x_emb, x_emb)
        attn_out = self.norm(attn_out + x_emb)  # residual
        # Pool: flatten
        return attn_out.reshape(x.shape[0], -1)  # (batch, n_features * embed_dim)


class MLPWithAttention(nn.Module):
    """Variant D: Self-attention on features then MLP (64-32)."""

    def __init__(self, input_dim: int, embed_dim: int = 32, n_heads: int = 2):
        super().__init__()
        self.attention = FeatureAttention(input_dim, embed_dim, n_heads)
        attn_out_dim = input_dim * embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(attn_out_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        attn_feats = self.attention(x)
        return self.mlp(attn_feats).squeeze(-1)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------
def train_torch_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 64,
    patience: int = 10,
) -> nn.Module:
    """Train a PyTorch model with early stopping on training loss."""
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    X_t = torch.FloatTensor(X_train).to(DEVICE)
    y_t = torch.FloatTensor(y_train).to(DEVICE)
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    best_loss = float("inf")
    patience_counter = 0
    best_state = None

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        if avg_loss < best_loss - 1e-6:
            best_loss = avg_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def predict_torch(model: nn.Module, X_test: np.ndarray) -> np.ndarray:
    """Get predictions from a PyTorch model."""
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X_test).to(DEVICE)
        preds = model(X_t).cpu().numpy()
    return preds


def train_lgbm(X_train: np.ndarray, y_train: np.ndarray):
    """Train production LGBM: 100 trees, depth 4, lr 0.05."""
    import lightgbm as lgb

    params = {
        "objective": "regression",
        "metric": "mse",
        "num_leaves": 15,  # depth ~4
        "max_depth": 4,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "n_jobs": -1,
    }
    dtrain = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, dtrain, num_boost_round=100)
    return model


def predict_lgbm(model, X_test: np.ndarray) -> np.ndarray:
    """Get LGBM predictions."""
    return model.predict(X_test)


# ---------------------------------------------------------------------------
# ATR-based BS pricing for bull call spreads
# ---------------------------------------------------------------------------
def compute_bs_price(spot: float, strike_low: float, strike_high: float,
                     atr_pct: float, dte_days: int = 21) -> dict:
    """
    ATR-based Black-Scholes proxy for bull call spread pricing.
    Returns debit, max_profit, max_loss.
    """
    width = strike_high - strike_low
    # Implied vol from ATR (ATR ≈ 1.25 * daily_vol * sqrt(period))
    daily_vol = atr_pct / (1.25 * np.sqrt(21))
    iv = daily_vol * np.sqrt(252) * IV_MULTIPLIER

    # BS delta-approximation for ATM-ish call spread
    t = dte_days / 365.0
    vol_t = iv * np.sqrt(t)

    # Simplified: debit ≈ width * N(d1_low) - width * N(d1_high)
    # For near-ATM with small width, debit ≈ width * n(0) * vol_t ≈ width * 0.4 * vol_t
    # More accurate: use the spread width relative to vol
    d1_low = (np.log(spot / strike_low) + 0.5 * iv**2 * t) / (vol_t + 1e-8)
    d1_high = (np.log(spot / strike_high) + 0.5 * iv**2 * t) / (vol_t + 1e-8)

    from scipy.stats import norm as sp_norm

    call_low = spot * sp_norm.cdf(d1_low) - strike_low * sp_norm.cdf(d1_low - vol_t)
    call_high = spot * sp_norm.cdf(d1_high) - strike_high * sp_norm.cdf(d1_high - vol_t)

    debit = max(call_low - call_high, 0.01)
    debit *= (1 + HAIRCUT)  # 15% entry haircut

    max_profit = width - debit
    max_loss = debit

    return {"debit": debit, "max_profit": max_profit, "max_loss": max_loss, "width": width}


# ---------------------------------------------------------------------------
# Walk-forward engine
# ---------------------------------------------------------------------------
def build_panel(feature_frames: dict, fwd_ret: pd.DataFrame,
                dates: list) -> tuple:
    """Build panel: rows = (date, sector), with features + target + metadata."""
    rows = []
    meta = []
    for dt in dates:
        for sector in SECTORS:
            if sector not in feature_frames:
                continue
            if dt not in feature_frames[sector].index:
                continue
            feat_row = feature_frames[sector].loc[dt, ALL_FEATURES].values
            target = fwd_ret.loc[dt, sector] if dt in fwd_ret.index else np.nan
            rows.append(np.concatenate([feat_row, [target]]))
            meta.append({"date": dt, "sector": sector})
    arr = np.array(rows, dtype=np.float64)
    return arr[:, :-1], arr[:, -1], meta


def walk_forward_variant(
    feature_frames: dict,
    fwd_ret: pd.DataFrame,
    vix_series: pd.Series,
    close_df: pd.DataFrame,
    variant: str,
) -> tuple:
    """
    Run sliding walk-forward for a single variant.
    Returns: (trades_list, ranking_correlations, top3_overlaps)
    """
    # Get dates where all features + target are valid
    valid_dates = feature_frames[SECTORS[0]].dropna(subset=ALL_FEATURES).index
    valid_dates = valid_dates.intersection(fwd_ret.dropna(how="all").index)
    valid_dates = sorted(valid_dates)

    if len(valid_dates) < TRAIN_DAYS + TEST_DAYS:
        ts(f"  Variant {variant}: insufficient data ({len(valid_dates)} days)")
        return [], [], []

    all_trades = []
    rank_corrs = []
    top3_overlaps = []

    # Sliding walk-forward
    fold_idx = 0
    pos = TRAIN_DAYS  # start of first test window

    while pos + REBAL_FREQ <= len(valid_dates):
        train_start = pos - TRAIN_DAYS
        train_dates = valid_dates[train_start:pos]
        # Test dates: next REBAL_FREQ days (biweekly rebalance)
        test_end = min(pos + REBAL_FREQ, len(valid_dates))
        test_dates = valid_dates[pos:test_end]

        if len(test_dates) == 0:
            pos += REBAL_FREQ
            continue

        # Build training panel
        X_train, y_train, _ = build_panel(feature_frames, fwd_ret, train_dates)

        # Remove NaN rows
        mask = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
        X_train, y_train = X_train[mask], y_train[mask]

        if len(X_train) < 100:
            pos += REBAL_FREQ
            continue

        # Normalize features (fit on train)
        scaler = StandardScaler()
        X_train_norm = scaler.fit_transform(X_train)

        # Train model based on variant
        n_feat = len(ALL_FEATURES)

        if variant == "A":
            model = train_lgbm(X_train_norm, y_train)
        elif variant == "B":
            model = train_torch_model(MLPSmall(n_feat), X_train_norm, y_train,
                                       epochs=50, lr=1e-3, batch_size=64, patience=10)
        elif variant == "C":
            model = train_torch_model(MLPLarge(n_feat), X_train_norm, y_train,
                                       epochs=100, lr=5e-4, batch_size=64, patience=15)
        elif variant == "D":
            model = train_torch_model(MLPWithAttention(n_feat), X_train_norm, y_train,
                                       epochs=50, lr=1e-3, batch_size=64, patience=10)
        elif variant == "E":
            # Ensemble: train both LGBM and MLP Small
            model_lgbm = train_lgbm(X_train_norm, y_train)
            model_mlp = train_torch_model(MLPSmall(n_feat), X_train_norm, y_train,
                                           epochs=50, lr=1e-3, batch_size=64, patience=10)
            model = (model_lgbm, model_mlp)

        # Predict on each rebalance date in the test window
        rebal_date = test_dates[0]  # rebalance at start of window

        # Get features for all sectors on rebalance date
        X_oot = []
        valid_sectors = []
        for sector in SECTORS:
            if rebal_date not in feature_frames[sector].index:
                continue
            row = feature_frames[sector].loc[rebal_date, ALL_FEATURES].values
            if not np.isnan(row).any():
                X_oot.append(row)
                valid_sectors.append(sector)

        if len(X_oot) < 6:
            pos += REBAL_FREQ
            continue

        X_oot = np.array(X_oot)
        X_oot_norm = scaler.transform(X_oot)

        # Get predictions / rankings
        if variant == "E":
            scores_lgbm = predict_lgbm(model[0], X_oot_norm)
            scores_mlp = predict_torch(model[1], X_oot_norm)
            # Average rank
            rank_lgbm = stats.rankdata(-scores_lgbm)
            rank_mlp = stats.rankdata(-scores_mlp)
            avg_rank = (rank_lgbm + rank_mlp) / 2
            scores = -avg_rank  # negate so higher = better
        elif variant == "A":
            scores = predict_lgbm(model, X_oot_norm)
        else:
            scores = predict_torch(model, X_oot_norm)

        # Rank sectors (highest score = best predicted)
        ranked_indices = np.argsort(scores)[::-1]
        top_sectors = [valid_sectors[i] for i in ranked_indices[:TOP_N]]

        # --- Ranking correlation: Spearman between predicted and actual forward returns ---
        actual_fwd = np.array([
            fwd_ret.loc[rebal_date, s] if rebal_date in fwd_ret.index and s in fwd_ret.columns else np.nan
            for s in valid_sectors
        ])
        valid_mask = ~np.isnan(actual_fwd)
        if valid_mask.sum() >= 5:
            rho, _ = stats.spearmanr(scores[valid_mask], actual_fwd[valid_mask])
            rank_corrs.append(rho)

            # Top-3 overlap with hindsight-optimal
            actual_ranked = np.argsort(actual_fwd[valid_mask])[::-1]
            actual_top3 = set(np.array(np.array(valid_sectors)[valid_mask])[actual_ranked[:TOP_N]])
            pred_top3 = set(top_sectors)
            overlap = len(pred_top3 & actual_top3) / TOP_N
            top3_overlaps.append(overlap)

        # --- VIX regime filter ---
        vix_val = vix_series.loc[rebal_date] if rebal_date in vix_series.index else 15.0
        if vix_val <= VIX_THRESHOLD:
            # Skip trading when VIX <= 20
            pos += REBAL_FREQ
            fold_idx += 1
            continue

        # --- Execute bull call spreads on top-3 sectors ---
        for sector in top_sectors:
            if sector not in close_df.columns:
                continue

            entry_price = close_df.loc[rebal_date, sector]
            entry_idx = close_df.index.get_loc(rebal_date)

            # ATR proxy (21d high-low range / close)
            lb_start = max(0, entry_idx - 21)
            lb_slice = close_df[sector].iloc[lb_start: entry_idx + 1]
            atr_pct = (lb_slice.max() - lb_slice.min()) / entry_price
            atr_pct = max(atr_pct, 0.005)  # floor at 0.5%

            # Bull call spread: buy call at spot, sell call at spot + 3% width
            strike_low = entry_price
            strike_high = entry_price * (1 + SPREAD_WIDTH_PCT)

            # Price the spread
            pricing = compute_bs_price(entry_price, strike_low, strike_high, atr_pct, DTE)

            debit_per_share = pricing["debit"]
            debit_per_contract = debit_per_share * 100  # options = 100 shares

            if debit_per_contract <= 0:
                continue

            # Position sizing: scale with equity, max $200 per trade
            # Current equity will be computed from cumulative P&L later, use starting for now
            max_contracts = max(1, int(MAX_PER_TRADE / debit_per_contract))
            n_contracts = max_contracts

            # Hold to expiry: intrinsic value only
            exit_idx = min(entry_idx + DTE, len(close_df) - 1)
            exit_price = close_df[sector].iloc[exit_idx]

            # Intrinsic value at expiry
            # Bull call spread: max(0, S - K_low) - max(0, S - K_high)
            intrinsic_low = max(0, exit_price - strike_low)
            intrinsic_high = max(0, exit_price - strike_high)
            spread_value_at_expiry = intrinsic_low - intrinsic_high
            spread_value_at_expiry = max(0, min(spread_value_at_expiry, pricing["width"]))

            # P&L per contract
            pnl_per_contract = (spread_value_at_expiry - debit_per_share) * 100
            total_pnl = pnl_per_contract * n_contracts - COMMISSION

            # Floor at max loss
            max_loss = -(debit_per_contract * n_contracts + COMMISSION)
            total_pnl = max(total_pnl, max_loss)

            all_trades.append({
                "date": rebal_date,
                "exit_date": close_df.index[exit_idx],
                "sector": sector,
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "debit": float(debit_per_contract),
                "n_contracts": n_contracts,
                "pnl": float(total_pnl),
                "vix": float(vix_val),
                "spread_width": float(pricing["width"]),
            })

        pos += REBAL_FREQ
        fold_idx += 1

    return all_trades, rank_corrs, top3_overlaps


# ---------------------------------------------------------------------------
# Metrics (calendar month Sharpe)
# ---------------------------------------------------------------------------
def compute_metrics(trades: list, starting_capital: float = STARTING_CAPITAL) -> dict:
    """Compute performance metrics from trade list. Calendar month Sharpe."""
    if not trades:
        return {
            "sharpe": 0, "sortino": 0, "pf": 0, "wr": 0,
            "total_ret": 0, "max_dd": 0, "n_trades": 0,
        }

    trades_df = pd.DataFrame(trades)
    trades_df["date"] = pd.to_datetime(trades_df["date"])

    # Daily P&L series (aggregate trades by date)
    daily_pnl = trades_df.groupby("date")["pnl"].sum()

    # Equity curve
    equity = starting_capital + daily_pnl.cumsum()

    # Calendar month returns for Sharpe/Sortino
    equity_series = pd.Series(starting_capital, index=[daily_pnl.index[0] - pd.Timedelta(days=1)])
    equity_series = pd.concat([equity_series, equity])
    monthly_equity = equity_series.resample("ME").last().ffill()
    monthly_ret = monthly_equity.pct_change().dropna()

    if len(monthly_ret) < 3:
        # Fallback to trade-level metrics
        trade_rets = trades_df["pnl"] / starting_capital
        sharpe = trade_rets.mean() / (trade_rets.std() + 1e-8) * np.sqrt(12)
        downside = trade_rets[trade_rets < 0].std() + 1e-8
        sortino = trade_rets.mean() / downside * np.sqrt(12)
    else:
        sharpe = monthly_ret.mean() / (monthly_ret.std() + 1e-8) * np.sqrt(12)
        downside = monthly_ret[monthly_ret < 0].std() + 1e-8
        sortino = monthly_ret.mean() / downside * np.sqrt(12)

    # Profit factor
    wins = trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum()
    losses = abs(trades_df.loc[trades_df["pnl"] < 0, "pnl"].sum()) + 1e-8
    pf = wins / losses

    # Win rate
    wr = (trades_df["pnl"] > 0).mean()

    # Total return
    total_pnl = trades_df["pnl"].sum()
    total_ret = total_pnl / starting_capital

    # Max drawdown
    cum_pnl = daily_pnl.cumsum()
    running_max = cum_pnl.cummax()
    dd = cum_pnl - running_max
    max_dd = dd.min() / starting_capital if len(dd) > 0 else 0

    return {
        "sharpe": round(float(sharpe), 3),
        "sortino": round(float(sortino), 3),
        "pf": round(float(pf), 3),
        "wr": round(float(wr), 3),
        "total_ret": round(float(total_ret), 4),
        "max_dd": round(float(max_dd), 4),
        "n_trades": len(trades_df),
    }


# ---------------------------------------------------------------------------
# 5-Gate adversarial validation
# ---------------------------------------------------------------------------
def gate1_permutation(trades: list, n_trials: int = 300) -> dict:
    """Shuffle trade P&L to check if real Sharpe exceeds random."""
    real_metrics = compute_metrics(trades)
    real_sharpe = real_metrics["sharpe"]

    pnls = np.array([t["pnl"] for t in trades])
    count_worse = 0
    for _ in range(n_trials):
        np.random.shuffle(pnls)
        shuffled_trades = [dict(t, pnl=float(p)) for t, p in zip(trades, pnls)]
        rand_sharpe = compute_metrics(shuffled_trades)["sharpe"]
        if rand_sharpe >= real_sharpe:
            count_worse += 1

    p_value = count_worse / n_trials
    return {"pass": p_value < 0.05, "p_value": round(p_value, 4)}


def gate2_regime_stability(trades: list, spy_returns: pd.Series) -> dict:
    """R1: |Sharpe_bull - Sharpe_bear| / max < 0.50."""
    if not trades:
        return {"pass": False, "gap": 1.0}

    trades_df = pd.DataFrame(trades)
    trades_df["date"] = pd.to_datetime(trades_df["date"])

    # Classify each trade date as bull or bear
    spy_cum_63 = spy_returns.rolling(63, min_periods=21).sum()
    bull_trades = []
    bear_trades = []

    for _, row in trades_df.iterrows():
        dt = row["date"]
        if dt in spy_cum_63.index:
            if spy_cum_63.loc[dt] > 0:
                bull_trades.append(row.to_dict())
            else:
                bear_trades.append(row.to_dict())
        else:
            bull_trades.append(row.to_dict())

    sharpe_bull = compute_metrics(bull_trades)["sharpe"] if len(bull_trades) > 5 else 0
    sharpe_bear = compute_metrics(bear_trades)["sharpe"] if len(bear_trades) > 5 else 0

    max_sharpe = max(abs(sharpe_bull), abs(sharpe_bear), 0.01)
    gap = abs(sharpe_bull - sharpe_bear) / max_sharpe

    return {
        "pass": gap < 0.50,
        "gap": round(gap, 3),
        "sharpe_bull": round(sharpe_bull, 3),
        "sharpe_bear": round(sharpe_bear, 3),
    }


def gate3_subperiod(trades: list) -> dict:
    """Both halves of the trade history must have Sharpe > 0.5."""
    if len(trades) < 20:
        return {"pass": False, "sharpe_h1": 0, "sharpe_h2": 0}

    mid = len(trades) // 2
    h1 = trades[:mid]
    h2 = trades[mid:]

    sharpe_h1 = compute_metrics(h1)["sharpe"]
    sharpe_h2 = compute_metrics(h2)["sharpe"]

    return {
        "pass": sharpe_h1 > 0.5 and sharpe_h2 > 0.5,
        "sharpe_h1": round(sharpe_h1, 3),
        "sharpe_h2": round(sharpe_h2, 3),
    }


def gate4_outlier_removal(trades: list) -> dict:
    """Remove top/bottom 5% of trades; Sharpe must remain > 0.5."""
    if len(trades) < 20:
        return {"pass": False, "sharpe_trimmed": 0}

    pnls = sorted([t["pnl"] for t in trades])
    n = len(pnls)
    lo = int(n * 0.05)
    hi = int(n * 0.95)
    trimmed_pnl_set = set(range(lo, hi))

    # Sort trades by PnL and keep middle 90%
    sorted_trades = sorted(trades, key=lambda t: t["pnl"])
    trimmed = sorted_trades[lo:hi]

    sharpe_trimmed = compute_metrics(trimmed)["sharpe"]

    return {"pass": sharpe_trimmed > 0.5, "sharpe_trimmed": round(sharpe_trimmed, 3)}


def gate5_yearly_consistency(trades: list) -> dict:
    """At least 60% of calendar years must be profitable."""
    if not trades:
        return {"pass": False, "pct_profitable_years": 0}

    trades_df = pd.DataFrame(trades)
    trades_df["date"] = pd.to_datetime(trades_df["date"])
    trades_df["year"] = trades_df["date"].dt.year

    yearly_pnl = trades_df.groupby("year")["pnl"].sum()
    n_years = len(yearly_pnl)
    if n_years < 2:
        return {"pass": False, "pct_profitable_years": 0, "yearly_pnl": {}}

    pct_profitable = (yearly_pnl > 0).mean()
    return {
        "pass": pct_profitable >= 0.60,
        "pct_profitable_years": round(float(pct_profitable), 3),
        "n_years": n_years,
        "yearly_pnl": {str(y): round(float(v), 2) for y, v in yearly_pnl.items()},
    }


def run_all_gates(trades: list, spy_returns: pd.Series, variant: str) -> dict:
    """Run all 5 adversarial gates."""
    ts(f"  Running 5-gate validation for variant {variant}...")

    g1 = gate1_permutation(trades)
    g2 = gate2_regime_stability(trades, spy_returns)
    g3 = gate3_subperiod(trades)
    g4 = gate4_outlier_removal(trades)
    g5 = gate5_yearly_consistency(trades)

    n_pass = sum([g1["pass"], g2["pass"], g3["pass"], g4["pass"], g5["pass"]])
    ts(
        f"    Gates: {n_pass}/5 | Perm p={g1['p_value']:.3f} | "
        f"Regime gap={g2['gap']:.2f} | SubP: H1={g3['sharpe_h1']:.2f} H2={g3['sharpe_h2']:.2f} | "
        f"Trimmed Sharpe={g4['sharpe_trimmed']:.2f} | "
        f"Yearly: {g5.get('pct_profitable_years', 0):.0%}"
    )

    return {
        "gate1_permutation": g1,
        "gate2_regime": g2,
        "gate3_subperiod": g3,
        "gate4_outlier_removal": g4,
        "gate5_yearly_consistency": g5,
        "gates_passed": n_pass,
        "all_pass": n_pass == 5,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    start_time = time.time()
    ts(f"Neural Sector Ranker v1 — Device: {DEVICE}")
    ts(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        ts(f"GPU: {torch.cuda.get_device_name(0)}")
    ts(f"Output: {OUTPUT_DIR}")

    # Download data
    close_df = download_data()
    vix_series = close_df["VIX"]
    spy_returns = close_df["SPY"].pct_change()

    # Compute features
    ts("Computing 23 features for 11 sectors...")
    feature_frames, fwd_ret = compute_features(close_df)
    ts("Features computed.")

    # Start MLflow experiment
    mlflow_run = None
    if MLFLOW_OK:
        try:
            mlflow.set_experiment("neural_sector_ranker_v1")
            mlflow_run = mlflow.start_run(run_name=f"nsrv1_{datetime.now().strftime('%Y%m%d_%H%M')}")
            mlflow.log_params({
                "train_days": TRAIN_DAYS,
                "test_days": TEST_DAYS,
                "rebal_freq": REBAL_FREQ,
                "top_n": TOP_N,
                "spread_width_pct": SPREAD_WIDTH_PCT,
                "dte": DTE,
                "starting_capital": STARTING_CAPITAL,
                "max_per_trade": MAX_PER_TRADE,
                "haircut": HAIRCUT,
                "commission": COMMISSION,
                "iv_multiplier": IV_MULTIPLIER,
                "vix_threshold": VIX_THRESHOLD,
                "n_features": len(ALL_FEATURES),
                "device": str(DEVICE),
            })
        except Exception as e:
            ts(f"MLflow start failed: {e}")

    # Run all 5 variants
    variant_names = {
        "A": "LGBM Baseline (100 trees, depth 4)",
        "B": "MLP Small (64-32, dropout 0.2, 50ep)",
        "C": "MLP Large (128-64-32, dropout 0.3, 100ep)",
        "D": "MLP + Attention (self-attn → 64-32)",
        "E": "Ensemble (LGBM + MLP Small avg rank)",
    }

    results = {}
    all_variant_trades = {}

    for variant in ["A", "B", "C", "D", "E"]:
        ts(f"\n{'='*60}")
        ts(f"Variant {variant}: {variant_names[variant]}")
        ts(f"{'='*60}")

        t0 = time.time()
        try:
            trades, rank_corrs, top3_overlaps = walk_forward_variant(
                feature_frames, fwd_ret, vix_series, close_df, variant
            )
        except Exception as e:
            ts(f"  ERROR in variant {variant}: {e}")
            import traceback
            traceback.print_exc()
            results[variant] = {
                "name": variant_names[variant],
                "metrics": compute_metrics([]),
                "gates": {},
                "rank_corr_mean": 0,
                "top3_overlap_mean": 0,
                "elapsed_s": round(time.time() - t0, 1),
                "error": str(e),
            }
            continue
        elapsed = time.time() - t0

        if not trades:
            ts(f"  No trades generated for variant {variant}")
            results[variant] = {
                "name": variant_names[variant],
                "metrics": compute_metrics([]),
                "gates": {},
                "rank_corr_mean": 0,
                "top3_overlap_mean": 0,
                "elapsed_s": round(elapsed, 1),
            }
            continue

        all_variant_trades[variant] = trades
        metrics = compute_metrics(trades)
        avg_rank_corr = np.mean(rank_corrs) if rank_corrs else 0
        avg_top3_overlap = np.mean(top3_overlaps) if top3_overlaps else 0

        ts(
            f"  Sharpe={metrics['sharpe']:.3f}  Sortino={metrics['sortino']:.3f}  "
            f"PF={metrics['pf']:.2f}  WR={metrics['wr']:.1%}  "
            f"TotalRet={metrics['total_ret']:.1%}  MaxDD={metrics['max_dd']:.1%}  "
            f"N={metrics['n_trades']}  ({elapsed:.1f}s)"
        )
        ts(
            f"  Avg Rank Corr (Spearman): {avg_rank_corr:.4f}  |  "
            f"Avg Top-3 Overlap: {avg_top3_overlap:.1%}"
        )

        # Run 5-gate adversarial validation
        gates = run_all_gates(trades, spy_returns, variant)

        results[variant] = {
            "name": variant_names[variant],
            "metrics": metrics,
            "gates": gates,
            "rank_corr_mean": round(float(avg_rank_corr), 4),
            "top3_overlap_mean": round(float(avg_top3_overlap), 4),
            "elapsed_s": round(elapsed, 1),
        }

        # Log to MLflow
        if MLFLOW_OK:
            try:
                prefix = f"v{variant}"
                mlflow.log_metrics({
                    f"{prefix}_sharpe": metrics["sharpe"],
                    f"{prefix}_sortino": metrics["sortino"],
                    f"{prefix}_pf": metrics["pf"],
                    f"{prefix}_wr": metrics["wr"],
                    f"{prefix}_total_ret": metrics["total_ret"],
                    f"{prefix}_max_dd": metrics["max_dd"],
                    f"{prefix}_n_trades": metrics["n_trades"],
                    f"{prefix}_rank_corr": avg_rank_corr,
                    f"{prefix}_top3_overlap": avg_top3_overlap,
                    f"{prefix}_gates_passed": gates["gates_passed"],
                })
            except Exception:
                pass

    # --- Summary table sorted by Sharpe ---
    ts(f"\n{'='*80}")
    ts("RESULTS SUMMARY (sorted by Sharpe)")
    ts(f"{'='*80}")
    ts(f"{'Var':<4} {'Name':<40} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} {'WR':>6} "
       f"{'TotRet':>8} {'MaxDD':>8} {'#Trades':>8} {'RankCorr':>9} {'Top3Ovl':>8} {'Gates':>6}")
    ts("-" * 130)

    sorted_variants = sorted(
        results.items(),
        key=lambda x: x[1]["metrics"].get("sharpe", 0),
        reverse=True,
    )

    for v, r in sorted_variants:
        m = r["metrics"]
        g = r.get("gates", {})
        n_gates = g.get("gates_passed", 0) if g else 0
        ts(
            f"  {v:<3} {r['name']:<40} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} "
            f"{m['pf']:>6.2f} {m['wr']:>5.1%} {m['total_ret']:>7.1%} "
            f"{m['max_dd']:>7.1%} {m['n_trades']:>8d} {r['rank_corr_mean']:>9.4f} "
            f"{r['top3_overlap_mean']:>7.1%} {n_gates:>3}/5"
        )

    ts("-" * 130)

    # Best variant
    best_v = sorted_variants[0][0] if sorted_variants else None
    best_sharpe = sorted_variants[0][1]["metrics"]["sharpe"] if sorted_variants else 0
    best_gates = sorted_variants[0][1].get("gates", {}).get("all_pass", False) if sorted_variants else False

    ts(f"\nBest variant: {best_v} (Sharpe={best_sharpe:.3f}, "
       f"{'ALL GATES PASS' if best_gates else 'NOT all gates pass'})")

    # Save results JSON
    output = {
        "timestamp": datetime.now().isoformat(),
        "device": str(DEVICE),
        "config": {
            "sectors": SECTORS,
            "train_days": TRAIN_DAYS,
            "test_days": TEST_DAYS,
            "rebal_freq": REBAL_FREQ,
            "forward_days": FORWARD_DAYS,
            "top_n": TOP_N,
            "spread_width_pct": SPREAD_WIDTH_PCT,
            "dte": DTE,
            "starting_capital": STARTING_CAPITAL,
            "max_per_trade": MAX_PER_TRADE,
            "haircut": HAIRCUT,
            "commission": COMMISSION,
            "iv_multiplier": IV_MULTIPLIER,
            "vix_threshold": VIX_THRESHOLD,
            "features": ALL_FEATURES,
            "n_features": len(ALL_FEATURES),
        },
        "variants": results,
        "best_variant": best_v,
        "best_sharpe": best_sharpe,
        "total_elapsed_s": round(time.time() - start_time, 1),
    }

    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    ts(f"Results saved to {results_path}")

    # Log artifact to MLflow
    if MLFLOW_OK:
        try:
            mlflow.log_artifact(str(results_path))
            mlflow.log_metrics({
                "best_sharpe": best_sharpe,
                "total_elapsed_s": time.time() - start_time,
            })
            mlflow.end_run()
        except Exception:
            pass

    ts(f"\nTotal elapsed: {time.time() - start_time:.0f}s")
    return output


if __name__ == "__main__":
    main()
