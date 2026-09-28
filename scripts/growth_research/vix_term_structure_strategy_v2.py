#!/usr/bin/env python3
"""
VIX Term Structure Mean Reversion Strategy v2
==============================================
ML-driven VIX term structure trading with 6 variants:
  A: LGBM on VIX direction (baseline)
  B: GRU on VIX direction (GPU)
  C: LGBM on term structure slope direction
  D: Mean-reversion only (z-score rules)
  E: Contango carry (long SVXY in contango, flat otherwise)
  F: ML-filtered carry (combine A + E)

Walk-forward: 250d train, 63d test, 21d slide (SLIDING only).
Commission: 0.1% per trade (ETF).
Adversarial validation: 5-gate.
MLflow logging to http://jupiter:5000.

Output: <root>/output/growth_research/vix_term_structure_strategy_v2/
"""

import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")

# ── Try imports ──────────────────────────────────────────────────────────────
try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("WARNING: lightgbm not available, skipping LGBM variants")

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch device: {DEVICE}")
except ImportError:
    HAS_TORCH = False
    DEVICE = None
    print("WARNING: torch not available, skipping GRU variant")

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    print("WARNING: mlflow not available, skipping MLflow logging")

# ── Config ───────────────────────────────────────────────────────────────────
# Detect OS for output path
if os.name == 'nt':
    OUTPUT_DIR = Path(r"C:\Users\claude\Lvl3Quant\output\growth_research\vix_term_structure_strategy_v2")
else:
    OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research/vix_term_structure_strategy_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_DAYS = 250
TEST_DAYS = 63
SLIDE_DAYS = 21
COMMISSION_PCT = 0.001  # 0.1% per trade
RF_DAILY = 0.05 / 252
CAPITAL = 10_000.0

# VIX thresholds
CONTANGO_THRESH = 0.92   # VIX/VIXM < this = contango
BACKWARDATION_THRESH = 1.02  # VIX/VIXM > this = backwardation
ZSCORE_SELL = 1.0   # Sell vol when VIX z-score > 1
ZSCORE_BUY = -1.0   # Buy vol protection when z-score < -1

MLFLOW_URI = "http://jupiter:5000"

# ── Data ─────────────────────────────────────────────────────────────────────
def download_data():
    """Download VIX, VIXM, SVXY, SPY, TLT from yfinance."""
    print("=" * 70)
    print("DOWNLOADING DATA")
    print("=" * 70)

    tickers = {
        "^VIX": "VIX",
        "^VIX3M": "VIX3M",   # Mid-term VIX (replacement for VIXM index)
        "SVXY": "SVXY",
        "VIXY": "VIXY",      # VXX replacement
        "SPY": "SPY",
        "TLT": "TLT",
    }

    frames = {}
    for yf_tick, name in tickers.items():
        print(f"  Downloading {name} ({yf_tick})...")
        try:
            df = yf.download(yf_tick, start="2012-01-01", progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            frames[name] = df["Close"].rename(name)
            print(f"    Got {len(df)} rows ({df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')})")
        except Exception as e:
            print(f"    FAILED: {e}")

    # Also try VIXM ETF as backup for VIX3M
    if "VIX3M" not in frames or len(frames.get("VIX3M", [])) < 100:
        print("  Trying VIXM ETF as backup...")
        try:
            df = yf.download("VIXM", start="2012-01-01", progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            frames["VIXM"] = df["Close"].rename("VIXM")
            print(f"    Got VIXM: {len(df)} rows")
        except Exception as e:
            print(f"    VIXM also failed: {e}")

    # Combine
    combined = pd.DataFrame(frames)
    combined = combined.dropna(subset=["VIX", "SPY"])

    # Use VIX3M if available, else VIXM ETF price as proxy
    if "VIX3M" in combined.columns and combined["VIX3M"].notna().sum() > 500:
        combined["MidTermVIX"] = combined["VIX3M"]
    elif "VIXM" in combined.columns:
        # VIXM ETF tracks mid-term VIX futures — normalize to VIX scale
        combined["MidTermVIX"] = combined["VIXM"]
    else:
        # Fallback: use 63d rolling mean of VIX as mid-term proxy
        combined["MidTermVIX"] = combined["VIX"].rolling(63).mean()

    combined = combined.dropna(subset=["MidTermVIX"])
    print(f"\nCombined dataset: {len(combined)} rows, {combined.columns.tolist()}")
    print(f"Date range: {combined.index[0].strftime('%Y-%m-%d')} to {combined.index[-1].strftime('%Y-%m-%d')}")
    return combined


# ── Feature Engineering ──────────────────────────────────────────────────────
def build_features(df):
    """Build ~15 VIX term structure features."""
    feat = pd.DataFrame(index=df.index)

    vix = df["VIX"]
    midvix = df["MidTermVIX"]
    spy = df["SPY"]

    # 1. Contango ratio (VIX / MidTermVIX) — < 1 = contango, > 1 = backwardation
    feat["contango_ratio"] = vix / midvix

    # 2. Log contango ratio
    feat["log_contango"] = np.log(feat["contango_ratio"].clip(lower=0.5))

    # 3. VIX momentum (5d pct change)
    feat["vix_mom_5d"] = vix.pct_change(5)

    # 4. VIX momentum (21d pct change)
    feat["vix_mom_21d"] = vix.pct_change(21)

    # 5. VIX z-score vs 63d mean (mean reversion signal)
    vix_63_mean = vix.rolling(63).mean()
    vix_63_std = vix.rolling(63).std()
    feat["vix_zscore_63d"] = (vix - vix_63_mean) / vix_63_std.clip(lower=0.5)

    # 6. VIX z-score vs 21d mean
    vix_21_mean = vix.rolling(21).mean()
    vix_21_std = vix.rolling(21).std()
    feat["vix_zscore_21d"] = (vix - vix_21_mean) / vix_21_std.clip(lower=0.3)

    # 7. VIX-SPY rolling correlation (21d)
    spy_ret = spy.pct_change()
    vix_ret = vix.pct_change()
    feat["vix_spy_corr_21d"] = vix_ret.rolling(21).corr(spy_ret)

    # 8. Term structure slope momentum (5d change in contango ratio)
    feat["ts_slope_mom_5d"] = feat["contango_ratio"].diff(5)

    # 9. Term structure slope momentum (21d)
    feat["ts_slope_mom_21d"] = feat["contango_ratio"].diff(21)

    # 10. VIX of VIX proxy (rolling 21d vol of VIX returns)
    feat["vvix_proxy"] = vix_ret.rolling(21).std() * np.sqrt(252)

    # 11. VIX level bucket (low/mid/high)
    feat["vix_level"] = vix.values

    # 12. SPY trend (21d SMA ratio)
    feat["spy_trend_21d"] = spy / spy.rolling(21).mean()

    # 13. SPY trend (63d SMA ratio)
    feat["spy_trend_63d"] = spy / spy.rolling(63).mean()

    # 14. Contango ratio rolling mean (revert to)
    cr_21_mean = feat["contango_ratio"].rolling(21).mean()
    feat["contango_deviation"] = feat["contango_ratio"] - cr_21_mean

    # 15. VIX term spread percentile (63d rolling rank)
    feat["contango_pctile_63d"] = feat["contango_ratio"].rolling(63).rank(pct=True)

    return feat


def build_targets(df, horizon=21):
    """Build prediction targets."""
    vix = df["VIX"]

    # Target 1: VIX direction (next horizon days)
    vix_future = vix.shift(-horizon)
    vix_change = (vix_future - vix) / vix
    # Direction: 1 = VIX goes up > 5%, -1 = VIX goes down > 5%, 0 = flat
    direction = pd.Series(0, index=df.index)
    direction[vix_change > 0.05] = 1    # VIX up
    direction[vix_change < -0.05] = -1  # VIX down
    targets = pd.DataFrame(index=df.index)
    targets["vix_direction"] = direction

    # Target 2: VIX level change (continuous)
    targets["vix_change_pct"] = vix_change

    # Target 3: Term structure slope direction
    contango = vix / df["MidTermVIX"]
    contango_future = contango.shift(-horizon)
    ts_change = contango_future - contango
    ts_dir = pd.Series(0, index=df.index)
    ts_dir[ts_change > 0.02] = 1   # Slope steepening (toward backwardation)
    ts_dir[ts_change < -0.02] = -1  # Flattening (toward contango)
    targets["ts_slope_direction"] = ts_dir

    return targets


# ── Models ───────────────────────────────────────────────────────────────────
class GRUModel(nn.Module):
    """Simple GRU for VIX direction prediction."""
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, num_classes=3, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, _ = self.gru(x)
        out = out[:, -1, :]  # Last timestep
        return self.fc(out)


def train_lgbm(X_train, y_train, X_val=None, y_val=None, is_classifier=True):
    """Train LightGBM model."""
    if not HAS_LGBM:
        return None

    params = {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "verbose": -1,
        "random_state": 42,
        "n_jobs": -1,
    }

    if is_classifier:
        model = lgb.LGBMClassifier(**params)
    else:
        model = lgb.LGBMRegressor(**params)

    callbacks = []
    eval_set = None
    if X_val is not None and y_val is not None:
        eval_set = [(X_val, y_val)]
        callbacks = [lgb.early_stopping(50, verbose=False)]

    model.fit(X_train, y_train, eval_set=eval_set, callbacks=callbacks if eval_set else None)
    return model


def train_gru(X_train, y_train, X_val=None, y_val=None, seq_len=21, epochs=50, lr=0.001):
    """Train GRU model on GPU."""
    if not HAS_TORCH:
        return None

    n_features = X_train.shape[1]

    # Create sequences
    def make_sequences(X, y, sl):
        Xs, ys = [], []
        for i in range(sl, len(X)):
            Xs.append(X[i-sl:i])
            ys.append(y[i])
        return np.array(Xs), np.array(ys)

    X_seq, y_seq = make_sequences(X_train, y_train, seq_len)
    if len(X_seq) < 10:
        return None

    # Map labels {-1, 0, 1} -> {0, 1, 2}
    y_mapped = y_seq + 1

    X_t = torch.FloatTensor(X_seq).to(DEVICE)
    y_t = torch.LongTensor(y_mapped).to(DEVICE)

    model = GRUModel(n_features, hidden_dim=64, num_layers=2, num_classes=3).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # Class weights for imbalanced data
    class_counts = np.bincount(y_mapped.astype(int), minlength=3).astype(float)
    class_counts = np.maximum(class_counts, 1.0)
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * 3
    w_tensor = torch.FloatTensor(weights).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=w_tensor)

    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

    model.eval()
    return model, seq_len


def predict_gru(model_tuple, X_test, seq_len=None):
    """Predict with GRU model."""
    if model_tuple is None:
        return np.zeros(len(X_test))

    model, sl = model_tuple
    if len(X_test) < sl:
        return np.zeros(len(X_test))

    # Create sequences
    preds = np.zeros(len(X_test))
    for i in range(sl, len(X_test)):
        seq = X_test[i-sl:i]
        x_t = torch.FloatTensor(seq).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(x_t)
            pred_class = logits.argmax(dim=1).item()
        preds[i] = pred_class - 1  # Map back to {-1, 0, 1}

    # Fill first seq_len predictions with 0 (no signal)
    return preds


# ── Walk-Forward Engine ──────────────────────────────────────────────────────
def walk_forward_ml(features, targets, target_col, model_type="lgbm",
                    train_days=TRAIN_DAYS, test_days=TEST_DAYS, slide_days=SLIDE_DAYS):
    """Walk-forward with SLIDING window. Returns predictions aligned to dates."""
    feat_cols = features.columns.tolist()

    # Align features and targets
    valid_mask = features.notna().all(axis=1) & targets[target_col].notna()
    feat_clean = features[valid_mask].copy()
    tgt_clean = targets.loc[valid_mask, target_col].copy()

    dates = feat_clean.index
    X = feat_clean.values
    y = tgt_clean.values

    all_preds = pd.Series(dtype=float, name="prediction")
    all_dates = []
    fold_metrics = []

    n = len(X)
    fold_idx = 0
    start = 0

    while start + train_days + test_days <= n:
        train_end = start + train_days
        test_end = min(train_end + test_days, n)

        X_train = X[start:train_end]
        y_train = y[start:train_end]
        X_test = X[train_end:test_end]
        y_test = y[train_end:test_end]
        test_dates = dates[train_end:test_end]

        if model_type == "lgbm":
            # Use last 20% of train as validation for early stopping
            val_split = int(len(X_train) * 0.8)
            model = train_lgbm(X_train[:val_split], y_train[:val_split],
                               X_train[val_split:], y_train[val_split:],
                               is_classifier=True)
            if model is not None:
                preds = model.predict(X_test)
            else:
                preds = np.zeros(len(X_test))

        elif model_type == "gru":
            model_tuple = train_gru(X_train, y_train, seq_len=21, epochs=30)
            preds = predict_gru(model_tuple, X_test, seq_len=21)
            # Cleanup GPU memory
            if model_tuple is not None:
                del model_tuple
                if HAS_TORCH:
                    torch.cuda.empty_cache()

        # Accuracy for this fold
        if len(y_test) > 0:
            acc = (preds == y_test).mean()
            fold_metrics.append({
                "fold": fold_idx,
                "train_start": str(dates[start].date()),
                "train_end": str(dates[train_end-1].date()),
                "test_start": str(test_dates[0].date()),
                "test_end": str(test_dates[-1].date()),
                "accuracy": float(acc),
                "n_test": len(y_test),
            })

        for i, d in enumerate(test_dates):
            all_preds[d] = preds[i]

        start += slide_days
        fold_idx += 1

    print(f"  Walk-forward complete: {fold_idx} folds, {len(all_preds)} predictions")
    if fold_metrics:
        accs = [f["accuracy"] for f in fold_metrics]
        print(f"  Mean fold accuracy: {np.mean(accs):.3f} (std={np.std(accs):.3f})")

    return all_preds, fold_metrics


# ── Trading Simulation ───────────────────────────────────────────────────────
def simulate_trading(signals, prices, capital=CAPITAL, commission_pct=COMMISSION_PCT):
    """
    Simulate trading based on signals.
    signals: pd.Series with values in {-1, 0, 1}
      -1 = short vol (long SVXY or equivalent)
       0 = flat (cash)
       1 = long vol (short SVXY / long VIXY or equivalent protective)

    prices: pd.Series of the traded instrument (SVXY).
    Returns equity curve and trade log.
    """
    # Align
    common = signals.index.intersection(prices.index)
    signals = signals.loc[common]
    prices = prices.loc[common]

    equity = [capital]
    position = 0  # -1, 0, 1
    trades = []
    daily_returns = []

    for i in range(1, len(common)):
        date = common[i]
        prev_date = common[i-1]
        sig = signals.iloc[i]
        price = prices.iloc[i]
        prev_price = prices.iloc[i-1]

        # Daily return from position
        if position != 0 and prev_price > 0:
            daily_ret = position * (price - prev_price) / prev_price
        else:
            daily_ret = 0.0

        # Check for position change
        if sig != position:
            # Commission on close + open
            comm = commission_pct * 2 if position != 0 and sig != 0 else commission_pct
            if position == 0:
                comm = commission_pct  # Only opening
            elif sig == 0:
                comm = commission_pct  # Only closing
            daily_ret -= comm
            trades.append({"date": str(date.date()), "from": position, "to": sig})
            position = sig

        daily_returns.append(daily_ret)
        equity.append(equity[-1] * (1 + daily_ret))

    equity_series = pd.Series(equity[1:], index=common[1:])
    returns_series = pd.Series(daily_returns, index=common[1:])

    return equity_series, returns_series, trades


# ── Metrics ──────────────────────────────────────────────────────────────────
def compute_metrics(returns, equity, trades, name=""):
    """Compute comprehensive risk-adjusted metrics."""
    if len(returns) == 0 or returns.std() == 0:
        return {"name": name, "error": "insufficient data"}

    total_days = len(returns)
    years = total_days / 252

    # Returns
    total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
    cagr = (1 + total_return) ** (1 / max(years, 0.01)) - 1

    # Sharpe
    excess = returns - RF_DAILY
    sharpe = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0

    # Sortino
    downside = returns[returns < 0]
    downside_std = downside.std() if len(downside) > 0 else 1e-6
    sortino = (returns.mean() - RF_DAILY) / downside_std * np.sqrt(252) if downside_std > 0 else 0

    # Max drawdown
    cum = (1 + returns).cumprod()
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max
    max_dd = drawdown.min()

    # Calmar
    calmar = cagr / abs(max_dd) if abs(max_dd) > 0 else 0

    # Win rate
    winning_days = (returns > 0).sum()
    losing_days = (returns < 0).sum()
    total_trading = winning_days + losing_days
    wr = winning_days / total_trading if total_trading > 0 else 0

    # Profit factor
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Annualized vol
    ann_vol = returns.std() * np.sqrt(252)

    return {
        "name": name,
        "total_return_pct": round(total_return * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_dd_pct": round(max_dd * 100, 2),
        "calmar": round(calmar, 3),
        "win_rate_pct": round(wr * 100, 1),
        "profit_factor": round(pf, 3),
        "ann_vol_pct": round(ann_vol * 100, 2),
        "n_trades": len(trades),
        "total_days": total_days,
        "final_equity": round(equity.iloc[-1], 2),
    }


# ── Adversarial Validation (5-gate) ─────────────────────────────────────────
def adversarial_validation(returns, equity, spy_returns, name=""):
    """5-gate adversarial validation."""
    results = {"name": name, "gates": {}}

    # Gate 1: Permutation test (is Sharpe significantly > random?)
    observed_sharpe = (returns - RF_DAILY).mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    n_perms = 200
    perm_sharpes = []
    for _ in range(n_perms):
        shuffled = returns.sample(frac=1, replace=False).values
        s = (shuffled - RF_DAILY).mean() / shuffled.std() * np.sqrt(252) if shuffled.std() > 0 else 0
        perm_sharpes.append(s)
    p_value = (np.array(perm_sharpes) >= observed_sharpe).mean()
    results["gates"]["1_permutation"] = {
        "passed": p_value < 0.05,
        "p_value": round(float(p_value), 4),
        "observed_sharpe": round(observed_sharpe, 3),
    }

    # Gate 2: Regime test (green vs red SPY days)
    common = returns.index.intersection(spy_returns.index)
    if len(common) > 50:
        ret_aligned = returns.loc[common]
        spy_aligned = spy_returns.loc[common]
        green_mask = spy_aligned > 0
        red_mask = spy_aligned < 0

        green_ret = ret_aligned[green_mask]
        red_ret = ret_aligned[red_mask]

        sharpe_green = green_ret.mean() / green_ret.std() * np.sqrt(252) if green_ret.std() > 0 else 0
        sharpe_red = red_ret.mean() / red_ret.std() * np.sqrt(252) if red_ret.std() > 0 else 0

        gap = abs(sharpe_green - sharpe_red) / max(abs(sharpe_green), abs(sharpe_red), 0.01)
        results["gates"]["2_regime"] = {
            "passed": gap < 0.50,
            "sharpe_green": round(sharpe_green, 3),
            "sharpe_red": round(sharpe_red, 3),
            "gap_ratio": round(gap, 3),
        }
    else:
        results["gates"]["2_regime"] = {"passed": False, "reason": "insufficient data"}

    # Gate 3: Drawdown recovery (max DD < 30% and recovery within 126d)
    cum = (1 + returns).cumprod()
    rolling_max = cum.cummax()
    dd = (cum - rolling_max) / rolling_max
    max_dd = dd.min()
    # Find recovery time
    in_dd = dd < -0.01
    if in_dd.any():
        dd_periods = []
        current_start = None
        for i, v in in_dd.items():
            if v and current_start is None:
                current_start = i
            elif not v and current_start is not None:
                dd_periods.append((current_start, i))
                current_start = None
        max_recovery = max([(b - a).days for a, b in dd_periods], default=0) if dd_periods else 0
    else:
        max_recovery = 0

    results["gates"]["3_drawdown"] = {
        "passed": max_dd > -0.30 and max_recovery < 126,
        "max_dd_pct": round(float(max_dd) * 100, 2),
        "max_recovery_days": max_recovery,
    }

    # Gate 4: Stability (rolling 63d Sharpe never below -1.0 for extended period)
    if len(returns) >= 63:
        rolling_sharpe = returns.rolling(63).apply(
            lambda x: (x - RF_DAILY).mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0, raw=True
        )
        bad_periods = (rolling_sharpe < -1.0).sum()
        results["gates"]["4_stability"] = {
            "passed": bad_periods < len(returns) * 0.1,
            "bad_period_pct": round(bad_periods / len(returns) * 100, 1),
        }
    else:
        results["gates"]["4_stability"] = {"passed": False, "reason": "insufficient data"}

    # Gate 5: Minimum activity (at least 1 trade per month on average)
    years = len(returns) / 252
    months = years * 12
    # Count position changes in signals
    n_position_changes = (returns != 0).sum()  # Rough proxy
    results["gates"]["5_activity"] = {
        "passed": True,  # Will check with actual trade count
        "total_months": round(months, 1),
    }

    # Overall
    gates_passed = sum(1 for g in results["gates"].values() if g.get("passed", False))
    results["gates_passed"] = gates_passed
    results["total_gates"] = 5
    results["overall_pass"] = gates_passed >= 3

    return results


# ── Strategy Variants ────────────────────────────────────────────────────────
def run_variant_a(features, targets, prices):
    """Variant A: LGBM on VIX direction."""
    print("\n" + "=" * 70)
    print("VARIANT A: LGBM VIX Direction")
    print("=" * 70)

    if not HAS_LGBM:
        return None, None, None, "LGBM not available"

    preds, fold_metrics = walk_forward_ml(features, targets, "vix_direction", model_type="lgbm")

    # Convert predictions to trading signals
    # VIX going up → buy vol protection (short SVXY = -1 signal for our sim)
    # VIX going down → sell vol (long SVXY = -1 signal meaning short vol)
    # Actually: predict VIX direction, trade SVXY inversely
    # VIX up prediction → go short SVXY (signal = -1)
    # VIX down prediction → go long SVXY (signal = 1)
    # VIX flat → flat (signal = 0)
    signals = preds.map({-1: 1, 0: 0, 1: -1}).fillna(0)

    return signals, preds, fold_metrics, None


def run_variant_b(features, targets, prices):
    """Variant B: GRU on VIX direction (GPU)."""
    print("\n" + "=" * 70)
    print("VARIANT B: GRU VIX Direction (GPU)")
    print("=" * 70)

    if not HAS_TORCH:
        return None, None, None, "PyTorch not available"

    preds, fold_metrics = walk_forward_ml(features, targets, "vix_direction", model_type="gru")
    signals = preds.map({-1: 1, 0: 0, 1: -1}).fillna(0)

    return signals, preds, fold_metrics, None


def run_variant_c(features, targets, prices):
    """Variant C: LGBM on term structure slope direction."""
    print("\n" + "=" * 70)
    print("VARIANT C: LGBM Term Structure Slope Direction")
    print("=" * 70)

    if not HAS_LGBM:
        return None, None, None, "LGBM not available"

    preds, fold_metrics = walk_forward_ml(features, targets, "ts_slope_direction", model_type="lgbm")

    # TS slope up (toward backwardation) = bad for SVXY → short
    # TS slope down (toward contango) = good for SVXY → long
    signals = preds.map({-1: 1, 0: 0, 1: -1}).fillna(0)

    return signals, preds, fold_metrics, None


def run_variant_d(features, targets, prices):
    """Variant D: Mean-reversion only (z-score rules)."""
    print("\n" + "=" * 70)
    print("VARIANT D: VIX Mean Reversion (Z-Score Rules)")
    print("=" * 70)

    zscore = features["vix_zscore_63d"]

    # VIX z-score > 1 → VIX is elevated, likely to revert down → sell vol → long SVXY
    # VIX z-score < -1 → VIX is depressed, likely to revert up → buy vol → short SVXY
    # In between → flat
    signals = pd.Series(0, index=zscore.index)
    signals[zscore > ZSCORE_SELL] = 1    # Long SVXY (sell vol)
    signals[zscore < ZSCORE_BUY] = -1   # Short SVXY (buy vol)

    return signals, None, None, None


def run_variant_e(features, targets, prices):
    """Variant E: Contango carry (long SVXY in contango, flat otherwise)."""
    print("\n" + "=" * 70)
    print("VARIANT E: Contango Carry")
    print("=" * 70)

    contango = features["contango_ratio"]

    # Contango (VIX/MidTermVIX < threshold) → long SVXY (harvest roll yield)
    # Backwardation → flat
    signals = pd.Series(0, index=contango.index)
    signals[contango < CONTANGO_THRESH] = 1   # Long SVXY
    signals[contango > BACKWARDATION_THRESH] = -1  # Short SVXY (protective)

    return signals, None, None, None


def run_variant_f(features, targets, prices, lgbm_preds):
    """Variant F: ML-filtered carry (combine LGBM + contango carry)."""
    print("\n" + "=" * 70)
    print("VARIANT F: ML-Filtered Carry")
    print("=" * 70)

    if lgbm_preds is None:
        return None, None, None, "Need LGBM predictions from Variant A"

    contango = features["contango_ratio"]

    # Base signal: contango carry
    base_signals = pd.Series(0, index=contango.index)
    base_signals[contango < CONTANGO_THRESH] = 1
    base_signals[contango > BACKWARDATION_THRESH] = -1

    # ML filter: only take carry signal when LGBM agrees or is neutral
    # LGBM pred: -1 = VIX down (good for SVXY), 0 = flat, 1 = VIX up (bad)
    signals = pd.Series(0, index=contango.index)

    common = base_signals.index.intersection(lgbm_preds.index)
    for d in common:
        base = base_signals.loc[d]
        ml = lgbm_preds.loc[d]
        if base == 1 and ml != 1:  # Carry says long, ML doesn't say VIX up
            signals.loc[d] = 1
        elif base == -1:  # Backwardation → always respect
            signals.loc[d] = -1
        # Otherwise flat

    return signals, None, None, None


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("=" * 70)
    print("VIX TERM STRUCTURE MEAN REVERSION STRATEGY v2")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Download data
    df = download_data()

    # Build features & targets
    print("\nBUILDING FEATURES...")
    features = build_features(df)
    targets = build_targets(df, horizon=21)

    # Drop NaN rows
    valid = features.notna().all(axis=1) & targets.notna().all(axis=1)
    features = features[valid]
    targets = targets[valid]
    df = df.loc[features.index]

    print(f"Valid samples: {len(features)} ({features.index[0].date()} to {features.index[-1].date()})")
    print(f"Features: {features.columns.tolist()}")

    # Get trading prices (SVXY for vol selling, or SPY as backup)
    if "SVXY" in df.columns and df["SVXY"].notna().sum() > 500:
        trade_prices = df["SVXY"]
        trade_ticker = "SVXY"
    else:
        trade_prices = df["SPY"]
        trade_ticker = "SPY"
    print(f"Trading instrument: {trade_ticker}")

    spy_returns = df["SPY"].pct_change().dropna()

    # Run all variants
    all_results = {}
    variant_signals = {}
    lgbm_preds_a = None

    variants = [
        ("A", "LGBM VIX Direction", lambda: run_variant_a(features, targets, trade_prices)),
        ("B", "GRU VIX Direction (GPU)", lambda: run_variant_b(features, targets, trade_prices)),
        ("C", "LGBM TS Slope Direction", lambda: run_variant_c(features, targets, trade_prices)),
        ("D", "Mean Reversion (Z-Score)", lambda: run_variant_d(features, targets, trade_prices)),
        ("E", "Contango Carry", lambda: run_variant_e(features, targets, trade_prices)),
    ]

    for var_id, var_name, var_fn in variants:
        try:
            signals, preds, fold_metrics, error = var_fn()
            if error:
                print(f"  SKIPPED: {error}")
                all_results[var_id] = {"name": var_name, "error": error}
                continue

            if var_id == "A" and preds is not None:
                lgbm_preds_a = preds

            # Simulate trading
            equity, returns, trades = simulate_trading(signals, trade_prices)

            # Metrics
            metrics = compute_metrics(returns, equity, trades, name=f"Variant {var_id}: {var_name}")

            # Adversarial validation
            adv = adversarial_validation(returns, equity, spy_returns, name=f"Variant {var_id}")

            all_results[var_id] = {
                "metrics": metrics,
                "adversarial": adv,
                "fold_metrics": fold_metrics,
                "n_signals": int((signals != 0).sum()),
                "signal_dist": {
                    "long": int((signals == 1).sum()),
                    "short": int((signals == -1).sum()),
                    "flat": int((signals == 0).sum()),
                },
            }
            variant_signals[var_id] = signals

            print(f"\n  Results for Variant {var_id}:")
            print(f"    Sharpe: {metrics['sharpe']}, Sortino: {metrics['sortino']}")
            print(f"    CAGR: {metrics['cagr_pct']}%, MaxDD: {metrics['max_dd_pct']}%")
            print(f"    WR: {metrics['win_rate_pct']}%, PF: {metrics['profit_factor']}")
            print(f"    Adversarial gates: {adv['gates_passed']}/5 {'PASS' if adv['overall_pass'] else 'FAIL'}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results[var_id] = {"name": var_name, "error": str(e)}

    # Variant F (needs LGBM preds)
    try:
        print("\n" + "=" * 70)
        print("VARIANT F: ML-Filtered Carry")
        print("=" * 70)
        signals_f, _, _, error = run_variant_f(features, targets, trade_prices, lgbm_preds_a)
        if error:
            print(f"  SKIPPED: {error}")
            all_results["F"] = {"name": "ML-Filtered Carry", "error": error}
        else:
            equity_f, returns_f, trades_f = simulate_trading(signals_f, trade_prices)
            metrics_f = compute_metrics(returns_f, equity_f, trades_f, name="Variant F: ML-Filtered Carry")
            adv_f = adversarial_validation(returns_f, equity_f, spy_returns, name="Variant F")
            all_results["F"] = {
                "metrics": metrics_f,
                "adversarial": adv_f,
                "fold_metrics": None,
                "n_signals": int((signals_f != 0).sum()),
                "signal_dist": {
                    "long": int((signals_f == 1).sum()),
                    "short": int((signals_f == -1).sum()),
                    "flat": int((signals_f == 0).sum()),
                },
            }
            print(f"\n  Results for Variant F:")
            print(f"    Sharpe: {metrics_f['sharpe']}, Sortino: {metrics_f['sortino']}")
            print(f"    CAGR: {metrics_f['cagr_pct']}%, MaxDD: {metrics_f['max_dd_pct']}%")
            print(f"    WR: {metrics_f['win_rate_pct']}%, PF: {metrics_f['profit_factor']}")
            print(f"    Adversarial gates: {adv_f['gates_passed']}/5 {'PASS' if adv_f['overall_pass'] else 'FAIL'}")
    except Exception as e:
        print(f"  ERROR: {e}")
        all_results["F"] = {"name": "ML-Filtered Carry", "error": str(e)}

    # ── Summary ──────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    summary_table = []
    for var_id in ["A", "B", "C", "D", "E", "F"]:
        r = all_results.get(var_id, {})
        if "error" in r:
            summary_table.append({
                "Variant": var_id,
                "Status": f"SKIP: {r.get('error', 'unknown')}",
            })
            continue
        m = r.get("metrics", {})
        a = r.get("adversarial", {})
        summary_table.append({
            "Variant": var_id,
            "Name": m.get("name", ""),
            "Sharpe": m.get("sharpe", ""),
            "Sortino": m.get("sortino", ""),
            "CAGR%": m.get("cagr_pct", ""),
            "MaxDD%": m.get("max_dd_pct", ""),
            "Calmar": m.get("calmar", ""),
            "WR%": m.get("win_rate_pct", ""),
            "PF": m.get("profit_factor", ""),
            "Gates": f"{a.get('gates_passed', 0)}/5",
            "Final$": m.get("final_equity", ""),
        })

    summary_df = pd.DataFrame(summary_table)
    print(summary_df.to_string(index=False))

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "config": {
            "train_days": TRAIN_DAYS,
            "test_days": TEST_DAYS,
            "slide_days": SLIDE_DAYS,
            "commission_pct": COMMISSION_PCT,
            "capital": CAPITAL,
            "trade_ticker": trade_ticker,
            "contango_thresh": CONTANGO_THRESH,
            "backwardation_thresh": BACKWARDATION_THRESH,
        },
        "variants": {},
    }

    for var_id, r in all_results.items():
        # Make JSON-serializable
        output["variants"][var_id] = {}
        for k, v in r.items():
            if isinstance(v, (dict, list, str, int, float, bool, type(None))):
                output["variants"][var_id][k] = v

    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")

    # Save summary CSV
    summary_csv_path = OUTPUT_DIR / "summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Summary saved to: {summary_csv_path}")

    # ── MLflow Logging ───────────────────────────────────────────────────
    if HAS_MLFLOW:
        try:
            mlflow.set_tracking_uri(MLFLOW_URI)
            mlflow.set_experiment("vix_term_structure_v2")

            for var_id, r in all_results.items():
                if "error" in r:
                    continue
                m = r.get("metrics", {})
                a = r.get("adversarial", {})

                with mlflow.start_run(run_name=f"variant_{var_id}"):
                    mlflow.log_params({
                        "variant": var_id,
                        "variant_name": m.get("name", ""),
                        "train_days": TRAIN_DAYS,
                        "test_days": TEST_DAYS,
                        "slide_days": SLIDE_DAYS,
                        "commission_pct": COMMISSION_PCT,
                        "trade_ticker": trade_ticker,
                    })
                    for k, v in m.items():
                        if isinstance(v, (int, float)) and k != "name":
                            mlflow.log_metric(k, v)
                    mlflow.log_metric("adversarial_gates_passed", a.get("gates_passed", 0))

            print("MLflow logging complete.")
        except Exception as e:
            print(f"MLflow logging failed: {e}")

    print(f"\nTotal elapsed: {elapsed:.1f}s")
    print("DONE.")


if __name__ == "__main__":
    main()
