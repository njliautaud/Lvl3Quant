#!/usr/bin/env python3
"""
Options Volatility Predictor — Growth Research
================================================
Predicts next-21-day realized volatility for SPY, compares to implied vol (VIX),
and evaluates whether the IV-RV spread signal improves premium-selling timing.

Walk-forward SLIDING window: 252d train, 63d test (quarterly steps).
Models: LSTM (GPU), MLP (GPU), LGBM (CPU) — ensemble.
Regime-agnostic validation per HC #428.

Usage:
    python options_vol_predictor.py                    # full run
    python options_vol_predictor.py --quick-test       # small subset for testing
    python options_vol_predictor.py --no-gpu           # force CPU mode
"""

import argparse
import json
import logging
import os
import platform
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Platform-aware paths
# ---------------------------------------------------------------------------
IS_WINDOWS = platform.system() == "Windows"

if IS_WINDOWS:
    OUTPUT_DIR = Path(r"C:\Users\claude\Lvl3Quant\output\growth_research\vol_predictor")
else:
    OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research/vol_predictor")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(OUTPUT_DIR / "run.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRAIN_DAYS = 252
TEST_DAYS = 63
LOOKBACK = 20          # feature lookback window
RV_WINDOW = 21         # target: 21-day realized vol
ANNUALIZE = np.sqrt(252)
TICKERS = ["SPY", "^VIX", "^VIX3M", "TLT", "GLD", "HYG"]

# Regime classification thresholds (SPY 21d return)
REGIME_GREEN_THRESH = 0.01   # >1% = green
REGIME_RED_THRESH = -0.01    # <-1% = red


# ============================================================================
# DATA
# ============================================================================

def download_data(start: str = "2010-01-01", end: str = "2026-07-15") -> pd.DataFrame:
    """Download OHLCV for all tickers via yfinance."""
    import yfinance as yf

    log.info(f"Downloading data for {TICKERS} from {start} to {end}")
    frames = {}
    for tkr in TICKERS:
        try:
            df = yf.download(tkr, start=start, end=end, progress=False, auto_adjust=True)
            if df.empty:
                log.warning(f"No data for {tkr}")
                continue
            # Flatten multi-level columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            frames[tkr] = df
        except Exception as e:
            log.warning(f"Failed to download {tkr}: {e}")

    if "SPY" not in frames:
        raise RuntimeError("Cannot proceed without SPY data")

    # Cache to disk
    cache_path = OUTPUT_DIR / "raw_data.parquet"
    spy = frames["SPY"].copy()
    for tkr, df in frames.items():
        if tkr == "SPY":
            continue
        prefix = tkr.replace("^", "")
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in df.columns:
                spy[f"{prefix}_{col}"] = df[col]

    spy.to_parquet(cache_path)
    log.info(f"Cached {len(spy)} rows to {cache_path}")
    return spy


def load_or_download(quick_test: bool = False) -> pd.DataFrame:
    """Load cached data or download fresh."""
    cache_path = OUTPUT_DIR / "raw_data.parquet"
    if cache_path.exists():
        age_hours = (datetime.now().timestamp() - cache_path.stat().st_mtime) / 3600
        if age_hours < 24:
            log.info("Loading cached data")
            df = pd.read_parquet(cache_path)
            if quick_test:
                df = df.iloc[-800:]  # ~3 years for quick test
            return df

    start = "2018-01-01" if quick_test else "2010-01-01"
    df = download_data(start=start)
    if quick_test:
        df = df.iloc[-800:]
    return df


# ============================================================================
# FEATURES
# ============================================================================

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features from raw OHLCV data."""
    feat = pd.DataFrame(index=df.index)

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].astype(float)
    log_ret = np.log(close / close.shift(1))

    # --- Realized vol at multiple windows ---
    for w in [5, 10, 21, 63]:
        feat[f"rv_{w}d"] = log_ret.rolling(w).std() * ANNUALIZE

    # --- Realized vol ratios (vol-of-vol, term structure) ---
    feat["rv_ratio_5_21"] = feat["rv_5d"] / feat["rv_21d"]
    feat["rv_ratio_21_63"] = feat["rv_21d"] / feat["rv_63d"]

    # --- Intraday range ---
    feat["intraday_range"] = (high - low) / close
    feat["intraday_range_ma5"] = feat["intraday_range"].rolling(5).mean()
    feat["intraday_range_ma21"] = feat["intraday_range"].rolling(21).mean()

    # --- Price momentum ---
    for w in [5, 10, 21]:
        feat[f"mom_{w}d"] = close.pct_change(w)

    # --- Volume features ---
    feat["vol_ma20"] = volume.rolling(20).mean()
    feat["vol_ratio"] = volume / feat["vol_ma20"]
    feat["vol_ratio_ma5"] = feat["vol_ratio"].rolling(5).mean()

    # --- VIX features ---
    if "VIX_Close" in df.columns:
        vix = df["VIX_Close"].astype(float)
        feat["vix"] = vix
        feat["vix_ma5"] = vix.rolling(5).mean()
        feat["vix_ma21"] = vix.rolling(21).mean()
        feat["vix_pctile_63"] = vix.rolling(63).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
        # IV-RV spread (VIX is annualized implied vol for SPY)
        feat["iv_rv_spread"] = vix / 100 - feat["rv_21d"]

    # --- VIX term structure ---
    if "VIX3M_Close" in df.columns and "VIX_Close" in df.columns:
        vix3m = df["VIX3M_Close"].astype(float)
        vix = df["VIX_Close"].astype(float)
        feat["vix_term_ratio"] = vix / vix3m  # <1 = contango (normal), >1 = backwardation (fear)
        feat["vix_term_ratio_ma5"] = feat["vix_term_ratio"].rolling(5).mean()

    # --- Cross-asset momentum ---
    for asset in ["TLT", "GLD", "HYG"]:
        col = f"{asset}_Close"
        if col in df.columns:
            px = df[col].astype(float)
            for w in [5, 10, 21]:
                feat[f"{asset}_mom_{w}d"] = px.pct_change(w)
            # Correlation with SPY
            feat[f"{asset}_corr_21d"] = log_ret.rolling(21).corr(
                np.log(px / px.shift(1))
            )

    # --- Calendar effects ---
    feat["dow"] = pd.to_datetime(feat.index).dayofweek  # 0=Mon, 4=Fri
    feat["month"] = pd.to_datetime(feat.index).month
    feat["is_opex_week"] = (pd.to_datetime(feat.index).day >= 15) & (
        pd.to_datetime(feat.index).day <= 21
    )
    feat["is_opex_week"] = feat["is_opex_week"].astype(float)

    # --- Lagged target (autoregressive) ---
    feat["rv_21d_lag1"] = feat["rv_21d"].shift(1)
    feat["rv_21d_lag5"] = feat["rv_21d"].shift(5)

    return feat


def build_target(df: pd.DataFrame) -> pd.Series:
    """Next 21-day realized volatility (forward-looking, annualized)."""
    close = df["Close"].astype(float)
    log_ret = np.log(close / close.shift(1))
    # Forward rolling std (shift by -RV_WINDOW to get next-21d rv)
    fwd_rv = log_ret.shift(-1).rolling(RV_WINDOW).std().shift(-(RV_WINDOW - 1)) * ANNUALIZE
    return fwd_rv


def classify_regime(df: pd.DataFrame) -> pd.Series:
    """Classify each day's regime based on trailing 21d SPY return."""
    close = df["Close"].astype(float)
    ret_21d = close.pct_change(21)
    regime = pd.Series("flat", index=df.index)
    regime[ret_21d > REGIME_GREEN_THRESH] = "green"
    regime[ret_21d < REGIME_RED_THRESH] = "red"
    return regime


# ============================================================================
# MODELS
# ============================================================================

def get_device(force_cpu: bool = False):
    """Get PyTorch device."""
    import torch
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        log.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return dev
    log.info("No GPU available, using CPU")
    return torch.device("cpu")


class LSTMModel:
    """LSTM for vol prediction (PyTorch)."""

    def __init__(self, n_features: int, device, hidden_size: int = 64,
                 n_layers: int = 2, dropout: float = 0.2, lr: float = 1e-3,
                 epochs: int = 100, seq_len: int = 20):
        import torch
        import torch.nn as nn

        self.device = device
        self.seq_len = seq_len
        self.epochs = epochs
        self.lr = lr
        self.n_features = n_features

        class _LSTM(nn.Module):
            def __init__(self, inp, hid, nlayers, drop):
                super().__init__()
                self.lstm = nn.LSTM(inp, hid, nlayers, batch_first=True,
                                    dropout=drop if nlayers > 1 else 0)
                self.head = nn.Sequential(
                    nn.Linear(hid, 32),
                    nn.ReLU(),
                    nn.Dropout(drop),
                    nn.Linear(32, 1),
                )

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :]).squeeze(-1)

        self.model = _LSTM(n_features, hidden_size, n_layers, dropout).to(device)
        self.scaler_X = None
        self.scaler_y = None

    def _make_sequences(self, X: np.ndarray, y: np.ndarray = None):
        import torch
        seqs, targets = [], []
        for i in range(self.seq_len, len(X)):
            seqs.append(X[i - self.seq_len:i])
            if y is not None:
                targets.append(y[i])
        seqs = np.array(seqs, dtype=np.float32)
        t_seqs = torch.tensor(seqs, dtype=torch.float32, device=self.device)
        if y is not None:
            targets = np.array(targets, dtype=np.float32)
            t_targets = torch.tensor(targets, dtype=torch.float32, device=self.device)
            return t_seqs, t_targets
        return t_seqs, None

    def fit(self, X: np.ndarray, y: np.ndarray):
        import torch
        import torch.nn as nn
        from sklearn.preprocessing import StandardScaler

        self.scaler_X = StandardScaler().fit(X)
        self.scaler_y = StandardScaler().fit(y.reshape(-1, 1))

        X_sc = self.scaler_X.transform(X)
        y_sc = self.scaler_y.transform(y.reshape(-1, 1)).ravel()

        t_X, t_y = self._make_sequences(X_sc, y_sc)
        if len(t_X) < 10:
            return self

        dataset = torch.utils.data.TensorDataset(t_X, t_y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                                      weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5
        )
        criterion = nn.MSELoss()

        self.model.train()
        best_loss = float("inf")
        patience_counter = 0
        for epoch in range(self.epochs):
            epoch_loss = 0
            for bx, by in loader:
                optimizer.zero_grad()
                pred = self.model(bx)
                loss = criterion(pred, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(loader)
            scheduler.step(avg_loss)
            if avg_loss < best_loss - 1e-6:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 20:
                    break
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch
        self.model.eval()
        X_sc = self.scaler_X.transform(X)
        t_X, _ = self._make_sequences(X_sc)
        if len(t_X) == 0:
            return np.full(len(X), np.nan)
        with torch.no_grad():
            pred_sc = self.model(t_X).cpu().numpy()
        pred = self.scaler_y.inverse_transform(pred_sc.reshape(-1, 1)).ravel()
        # Pad beginning (no sequences available)
        result = np.full(len(X), np.nan)
        result[self.seq_len:] = pred
        return result


class MLPModel:
    """Simple MLP for vol prediction (PyTorch)."""

    def __init__(self, n_features: int, device, hidden: int = 128,
                 dropout: float = 0.3, lr: float = 1e-3, epochs: int = 150):
        import torch
        import torch.nn as nn

        self.device = device
        self.epochs = epochs
        self.lr = lr

        class _MLP(nn.Module):
            def __init__(self, inp, hid, drop):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(inp, hid),
                    nn.ReLU(),
                    nn.BatchNorm1d(hid),
                    nn.Dropout(drop),
                    nn.Linear(hid, hid // 2),
                    nn.ReLU(),
                    nn.BatchNorm1d(hid // 2),
                    nn.Dropout(drop),
                    nn.Linear(hid // 2, 1),
                )

            def forward(self, x):
                return self.net(x).squeeze(-1)

        self.model = _MLP(n_features, hidden, dropout).to(device)
        self.scaler_X = None
        self.scaler_y = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        import torch
        import torch.nn as nn
        from sklearn.preprocessing import StandardScaler

        self.scaler_X = StandardScaler().fit(X)
        self.scaler_y = StandardScaler().fit(y.reshape(-1, 1))

        X_sc = torch.tensor(self.scaler_X.transform(X), dtype=torch.float32,
                            device=self.device)
        y_sc = torch.tensor(self.scaler_y.transform(y.reshape(-1, 1)).ravel(),
                            dtype=torch.float32, device=self.device)

        dataset = torch.utils.data.TensorDataset(X_sc, y_sc)
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                                      weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5
        )
        criterion = nn.MSELoss()

        self.model.train()
        best_loss = float("inf")
        patience_counter = 0
        for epoch in range(self.epochs):
            epoch_loss = 0
            for bx, by in loader:
                optimizer.zero_grad()
                pred = self.model(bx)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(loader)
            scheduler.step(avg_loss)
            if avg_loss < best_loss - 1e-6:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 20:
                    break
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch
        self.model.eval()
        X_sc = torch.tensor(self.scaler_X.transform(X), dtype=torch.float32,
                            device=self.device)
        with torch.no_grad():
            pred_sc = self.model(X_sc).cpu().numpy()
        return self.scaler_y.inverse_transform(pred_sc.reshape(-1, 1)).ravel()


class LGBMModel:
    """LightGBM for vol prediction (CPU)."""

    def __init__(self):
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        import lightgbm as lgb
        self.model = lgb.LGBMRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            min_child_samples=20,
            verbose=-1,
        )
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


# ============================================================================
# WALK-FORWARD ENGINE
# ============================================================================

def walk_forward(features: pd.DataFrame, target: pd.Series, regime: pd.Series,
                 force_cpu: bool = False, quick_test: bool = False) -> pd.DataFrame:
    """Sliding window walk-forward with LSTM, MLP, LGBM ensemble."""
    device = get_device(force_cpu)

    # Align features and target, drop NaN
    common_idx = features.dropna().index.intersection(target.dropna().index)
    features = features.loc[common_idx]
    target = target.loc[common_idx]
    regime = regime.loc[common_idx]

    feature_cols = [c for c in features.columns if features[c].dtype in [np.float64, np.float32, np.int64, float, int]]
    X_all = features[feature_cols].values.astype(np.float32)
    y_all = target.values.astype(np.float32)

    n = len(X_all)
    n_features = X_all.shape[1]
    log.info(f"Total samples: {n}, features: {n_features}")
    log.info(f"Feature columns: {feature_cols}")

    results = []
    fold = 0
    start = 0

    # Reduce epochs for quick test
    lstm_epochs = 30 if quick_test else 100
    mlp_epochs = 40 if quick_test else 150

    while start + TRAIN_DAYS + TEST_DAYS <= n:
        train_end = start + TRAIN_DAYS
        test_end = min(train_end + TEST_DAYS, n)

        X_train = X_all[start:train_end]
        y_train = y_all[start:train_end]
        X_test = X_all[train_end:test_end]
        y_test = y_all[train_end:test_end]
        test_dates = features.index[train_end:test_end]
        test_regime = regime.iloc[train_end:test_end]

        log.info(f"Fold {fold}: train {start}-{train_end} ({TRAIN_DAYS}d), "
                 f"test {train_end}-{test_end} ({test_end - train_end}d)")

        # --- Train models ---
        # LGBM
        lgbm = LGBMModel().fit(X_train, y_train)
        pred_lgbm = lgbm.predict(X_test)

        # MLP
        mlp = MLPModel(n_features, device, epochs=mlp_epochs).fit(X_train, y_train)
        pred_mlp = mlp.predict(X_test)

        # LSTM
        lstm = LSTMModel(n_features, device, epochs=lstm_epochs, seq_len=min(20, TRAIN_DAYS // 10))
        lstm.fit(X_train, y_train)
        pred_lstm = lstm.predict(X_test)

        # --- Ensemble (equal weight, NaN-tolerant) ---
        preds = np.column_stack([pred_lgbm, pred_mlp, pred_lstm])
        pred_ensemble = np.nanmean(preds, axis=1)

        for i in range(len(X_test)):
            results.append({
                "date": test_dates[i],
                "actual_rv": y_test[i],
                "pred_lgbm": pred_lgbm[i],
                "pred_mlp": pred_mlp[i],
                "pred_lstm": pred_lstm[i],
                "pred_ensemble": pred_ensemble[i],
                "regime": test_regime.iloc[i],
                "fold": fold,
            })

        fold += 1
        start += TEST_DAYS  # slide by test window size

    results_df = pd.DataFrame(results)
    log.info(f"Walk-forward complete: {fold} folds, {len(results_df)} predictions")
    return results_df


# ============================================================================
# STRATEGY SIMULATION
# ============================================================================

def simulate_strangle_strategy(results: pd.DataFrame, spy_df: pd.DataFrame) -> dict:
    """
    Simulate premium-selling strategy timing:
    - When predicted RV < IV (VIX): sell premium (IV overpriced, good for sellers)
    - When predicted RV >= IV: stay cash (IV fair/underpriced, bad for sellers)

    Benchmark: always-sell (every day is a selling day).
    """
    res = results.copy()
    res["date"] = pd.to_datetime(res["date"])
    res = res.set_index("date").sort_index()

    # Get VIX as proxy for implied vol
    if "VIX_Close" in spy_df.columns:
        vix = spy_df["VIX_Close"].astype(float) / 100  # Convert to decimal
        vix.index = pd.to_datetime(vix.index)
        res["iv"] = vix.reindex(res.index, method="ffill")
    else:
        log.warning("No VIX data, using rv_21d * 1.15 as IV proxy")
        res["iv"] = res["actual_rv"] * 1.15

    # Signal: predicted RV < IV → sell premium (IV overpriced)
    res["iv_rv_spread"] = res["iv"] - res["pred_ensemble"]
    res["signal"] = (res["iv_rv_spread"] > 0).astype(float)  # 1 = sell premium, 0 = cash

    # Daily P&L proxy: short vol daily return
    # When selling premium, you profit when realized vol < implied vol
    # Simplified: daily PnL proportional to (IV - realized_vol_that_day)
    spy_close = spy_df["Close"].astype(float)
    spy_close.index = pd.to_datetime(spy_close.index)
    daily_ret = spy_close.pct_change().reindex(res.index, method="ffill")
    daily_abs_ret = daily_ret.abs()

    # Strangle P&L proxy: theta收益 - gamma loss
    # Simplified: +theta (IV/sqrt(252)) - abs(daily_move) * sqrt(252)
    # This captures the essence: you earn IV-based theta, lose on actual moves
    theta_daily = res["iv"] / np.sqrt(252)
    gamma_loss = daily_abs_ret * np.sqrt(252) * 0.5  # rough gamma scaling

    res["pnl_per_day"] = theta_daily - gamma_loss

    # Strategy PnL: only trade on signal days
    res["strategy_pnl"] = res["pnl_per_day"] * res["signal"]
    # Benchmark: always sell
    res["benchmark_pnl"] = res["pnl_per_day"]

    return res


def compute_metrics(pnl_series: pd.Series, label: str) -> dict:
    """Compute risk-adjusted metrics for a PnL series."""
    pnl = pnl_series.dropna()
    if len(pnl) < 10:
        return {"label": label, "error": "insufficient data"}

    cum_pnl = pnl.cumsum()
    total_ret = cum_pnl.iloc[-1]
    n_days = len(pnl)

    mean_daily = pnl.mean()
    std_daily = pnl.std()
    downside_std = pnl[pnl < 0].std() if (pnl < 0).any() else 1e-9

    sharpe = mean_daily / std_daily * np.sqrt(252) if std_daily > 0 else 0
    sortino = mean_daily / downside_std * np.sqrt(252) if downside_std > 0 else 0

    wins = (pnl > 0).sum()
    losses = (pnl < 0).sum()
    wr = wins / (wins + losses) if (wins + losses) > 0 else 0

    avg_win = pnl[pnl > 0].mean() if wins > 0 else 0
    avg_loss = abs(pnl[pnl < 0].mean()) if losses > 0 else 1e-9
    pf = avg_win * wins / (avg_loss * losses) if losses > 0 and avg_loss > 0 else float("inf")

    # Max drawdown
    running_max = cum_pnl.cummax()
    dd = cum_pnl - running_max
    max_dd = dd.min()

    return {
        "label": label,
        "n_days": n_days,
        "total_return": round(float(total_ret), 4),
        "sharpe": round(float(sharpe), 3),
        "sortino": round(float(sortino), 3),
        "profit_factor": round(float(min(pf, 99.9)), 3),
        "win_rate": round(float(wr), 3),
        "max_drawdown": round(float(max_dd), 4),
        "avg_daily_pnl": round(float(mean_daily), 6),
        "std_daily_pnl": round(float(std_daily), 6),
    }


# ============================================================================
# REGIME-AGNOSTIC VALIDATION (HC #428 R1)
# ============================================================================

def regime_validation(results: pd.DataFrame) -> dict:
    """
    R1: Per-regime Sharpe. Reject if |Sharpe_green - Sharpe_red| / max > 0.50.
    """
    report = {}
    for regime_name in ["green", "red", "flat"]:
        mask = results["regime"] == regime_name
        subset = results.loc[mask]
        if len(subset) < 10:
            report[regime_name] = {"n_days": int(mask.sum()), "sharpe": None}
            continue

        pnl = subset["strategy_pnl"]
        metrics = compute_metrics(pnl, f"regime_{regime_name}")
        report[regime_name] = metrics

    # Regime asymmetry check
    green_sharpe = report.get("green", {}).get("sharpe")
    red_sharpe = report.get("red", {}).get("sharpe")

    if green_sharpe is not None and red_sharpe is not None:
        max_sharpe = max(abs(green_sharpe), abs(red_sharpe))
        if max_sharpe > 0:
            asymmetry = abs(green_sharpe - red_sharpe) / max_sharpe
        else:
            asymmetry = 0
        report["asymmetry"] = round(float(asymmetry), 3)
        report["regime_agnostic_pass"] = asymmetry <= 0.50
    else:
        report["asymmetry"] = None
        report["regime_agnostic_pass"] = None

    return report


# ============================================================================
# MODEL ACCURACY METRICS
# ============================================================================

def prediction_accuracy(results: pd.DataFrame) -> dict:
    """Evaluate prediction accuracy of each model and ensemble."""
    res = results.dropna(subset=["actual_rv", "pred_ensemble"])
    report = {}

    for model in ["lgbm", "mlp", "lstm", "ensemble"]:
        col = f"pred_{model}"
        valid = res.dropna(subset=[col])
        if len(valid) < 10:
            report[model] = {"error": "insufficient data"}
            continue

        actual = valid["actual_rv"].values
        pred = valid[col].values

        mae = np.mean(np.abs(actual - pred))
        rmse = np.sqrt(np.mean((actual - pred) ** 2))
        corr = np.corrcoef(actual, pred)[0, 1] if len(actual) > 1 else 0

        # Directional accuracy: did we predict vol going up/down correctly?
        actual_diff = np.diff(actual)
        pred_diff = np.diff(pred[:len(actual)])
        if len(actual_diff) > 0:
            dir_acc = np.mean(np.sign(actual_diff) == np.sign(pred_diff))
        else:
            dir_acc = 0

        report[model] = {
            "mae": round(float(mae), 4),
            "rmse": round(float(rmse), 4),
            "correlation": round(float(corr), 4),
            "directional_accuracy": round(float(dir_acc), 4),
            "n_predictions": len(valid),
        }

    return report


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Options Vol Predictor — Growth Research")
    parser.add_argument("--quick-test", action="store_true",
                        help="Run on small data subset for testing")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Force CPU mode")
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("OPTIONS VOLATILITY PREDICTOR — GROWTH RESEARCH")
    log.info(f"Platform: {platform.system()} | Quick test: {args.quick_test}")
    log.info(f"Output: {OUTPUT_DIR}")
    log.info("=" * 70)

    # --- Load data ---
    spy_df = load_or_download(quick_test=args.quick_test)
    log.info(f"Data shape: {spy_df.shape}, date range: {spy_df.index[0]} to {spy_df.index[-1]}")

    # --- Build features and target ---
    features = build_features(spy_df)
    target = build_target(spy_df)
    regime = classify_regime(spy_df)

    # Drop rows with NaN in features or target
    valid_mask = features.notna().all(axis=1) & target.notna()
    features = features[valid_mask]
    target = target[valid_mask]
    regime = regime[valid_mask]

    log.info(f"After NaN cleanup: {len(features)} samples, {features.shape[1]} features")

    if len(features) < TRAIN_DAYS + TEST_DAYS + 50:
        log.error(f"Insufficient data: need {TRAIN_DAYS + TEST_DAYS + 50}, have {len(features)}")
        sys.exit(1)

    # --- Walk-forward ---
    results = walk_forward(features, target, regime,
                           force_cpu=args.no_gpu, quick_test=args.quick_test)

    # Save raw predictions
    results.to_parquet(OUTPUT_DIR / "predictions.parquet", index=False)
    log.info(f"Saved predictions to {OUTPUT_DIR / 'predictions.parquet'}")

    # --- Prediction accuracy ---
    acc_report = prediction_accuracy(results)
    log.info("\n=== PREDICTION ACCURACY ===")
    for model, metrics in acc_report.items():
        log.info(f"  {model}: {metrics}")

    # --- Strategy simulation ---
    strat_results = simulate_strangle_strategy(results, spy_df)

    # Overall metrics
    strategy_metrics = compute_metrics(strat_results["strategy_pnl"], "Signal-Timed Strategy")
    benchmark_metrics = compute_metrics(strat_results["benchmark_pnl"], "Always-Sell Benchmark")

    log.info("\n=== STRATEGY PERFORMANCE ===")
    log.info(f"  Signal-Timed: {strategy_metrics}")
    log.info(f"  Always-Sell:  {benchmark_metrics}")

    # Signal statistics
    n_signal = int(strat_results["signal"].sum())
    n_total = len(strat_results)
    log.info(f"\n  Signal active: {n_signal}/{n_total} days ({100*n_signal/n_total:.1f}%)")

    # --- Regime validation (R1) ---
    regime_report = regime_validation(strat_results)
    log.info("\n=== REGIME VALIDATION (R1) ===")
    for k, v in regime_report.items():
        log.info(f"  {k}: {v}")

    # --- Day concentration check (HC #344: cap <= 0.70) ---
    if "fold" in strat_results.columns:
        fold_counts = strat_results.groupby("fold").size()
        max_fold_frac = fold_counts.max() / fold_counts.sum()
        day_conc_pass = max_fold_frac <= 0.70
        log.info(f"\n  Day concentration: {max_fold_frac:.3f} (pass={day_conc_pass})")
    else:
        day_conc_pass = True

    # --- Compile final report ---
    final_report = {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.system(),
        "quick_test": args.quick_test,
        "data_range": f"{spy_df.index[0]} to {spy_df.index[-1]}",
        "n_samples": len(features),
        "n_features": features.shape[1],
        "n_folds": results["fold"].nunique(),
        "n_predictions": len(results),
        "prediction_accuracy": acc_report,
        "strategy_metrics": strategy_metrics,
        "benchmark_metrics": benchmark_metrics,
        "signal_pct": round(100 * n_signal / n_total, 1),
        "regime_validation": regime_report,
        "day_concentration_pass": day_conc_pass,
        "sharpe_improvement": round(
            strategy_metrics.get("sharpe", 0) - benchmark_metrics.get("sharpe", 0), 3
        ),
    }

    # Save report
    report_path = OUTPUT_DIR / "report.json"
    with open(report_path, "w") as f:
        json.dump(final_report, f, indent=2, default=str)
    log.info(f"\nFull report saved to {report_path}")

    # --- Summary ---
    log.info("\n" + "=" * 70)
    log.info("SUMMARY")
    log.info("=" * 70)
    log.info(f"Ensemble correlation with actual RV: "
             f"{acc_report.get('ensemble', {}).get('correlation', 'N/A')}")
    log.info(f"Strategy Sharpe: {strategy_metrics.get('sharpe', 'N/A')}")
    log.info(f"Benchmark Sharpe: {benchmark_metrics.get('sharpe', 'N/A')}")
    log.info(f"Sharpe improvement: {final_report['sharpe_improvement']}")
    log.info(f"Regime-agnostic pass: {regime_report.get('regime_agnostic_pass', 'N/A')}")
    log.info(f"Signal active {final_report['signal_pct']}% of days")

    verdict = "PROMISING" if final_report["sharpe_improvement"] > 0 else "NO EDGE FOUND"
    log.info(f"\nVERDICT: {verdict}")
    log.info("=" * 70)

    return final_report


if __name__ == "__main__":
    main()
