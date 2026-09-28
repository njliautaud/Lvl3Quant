#!/usr/bin/env python3
"""
Neural Regime Allocator v1
==========================
Meta-learner that allocates across 6 validated strategies based on market regime features.
Uses PyTorch MLP with Sharpe-ratio loss, walk-forward validation, and regime-agnostic checks.

Designed for Neptune (RTX 3090). Self-contained: downloads data, trains, evaluates, logs to MLflow.

Strategies:
  1. Sector ETF Momentum (LightGBM-ranked monthly rebalance)
  2. Alt Trend Following (bonds/commodities/gold SMA200)
  3. SPY Iron Condor Income (non-directional premium selling)
  4. VIX Mean-Reversion (sell call spreads when VIX >30)
  5. LEAPS Momentum (3x leveraged sector momentum)
  6. Cross-Asset Trend Following (CTA-style 18 ETFs)
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import time
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "neural_regime_allocator_v1"

TICKERS = [
    "SPY", "^VIX", "TLT", "GLD", "UNG", "XLE", "XLF", "XLY", "XLK",
    "QQQ", "IWM", "HYG", "LQD", "DBA",
]
SECTOR_ETFS = ["XLE", "XLF", "XLY", "XLK", "QQQ", "IWM"]
TREND_ASSETS = ["TLT", "GLD", "DBA"]
ALL_TREND_ASSETS = ["SPY", "TLT", "GLD", "UNG", "XLE", "XLF", "XLY", "XLK", "QQQ", "IWM", "HYG", "LQD", "DBA", "DBA"]

NUM_STRATEGIES = 6
STRATEGY_NAMES = [
    "ETF Momentum", "Alt Trend", "Iron Condor",
    "VIX Mean-Rev", "LEAPS Momentum", "Cross-Asset Trend"
]

# Walk-forward params (sliding window, per HC #0)
TRAIN_MONTHS = 60
TEST_MONTHS = 1

# Model params
HIDDEN_DIM = 256
NUM_LAYERS = 3
DROPOUT = 0.3
LR = 1e-3
EPOCHS = 80
BATCH_SIZE = 64
WEIGHT_DECAY = 1e-4

# Seed
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------
def download_data(start="2007-01-01", end="2026-07-25"):
    """Download daily OHLCV data from yfinance."""
    import yfinance as yf
    print(f"[DATA] Downloading {len(TICKERS)} tickers from {start} to {end} ...")
    data = {}
    for ticker in TICKERS:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if len(df) < 100:
                print(f"  WARNING: {ticker} only has {len(df)} rows")
            # Flatten multi-level columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            data[ticker] = df
            print(f"  {ticker}: {len(df)} rows ({df.index[0].date()} to {df.index[-1].date()})")
        except Exception as e:
            print(f"  ERROR downloading {ticker}: {e}")
    return data


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def build_features(data):
    """Build 25 market regime features from downloaded data."""
    spy = data.get("SPY")
    vix = data.get("^VIX")
    tlt = data.get("TLT")
    gld = data.get("GLD")
    hyg = data.get("HYG")
    lqd = data.get("LQD")

    if spy is None or vix is None:
        raise ValueError("SPY or VIX data missing - cannot build features")

    # Align all on SPY's index
    idx = spy.index
    features = pd.DataFrame(index=idx)

    # --- VIX features ---
    vix_close = vix["Close"].reindex(idx).ffill()
    features["vix_level"] = vix_close
    # VIX term structure proxy: ratio of 10d realized vol to 30d realized vol
    spy_ret = spy["Close"].pct_change()
    rv10 = spy_ret.rolling(10).std() * np.sqrt(252)
    rv30 = spy_ret.rolling(30).std() * np.sqrt(252)
    features["vix_term_proxy"] = (rv10 / rv30).replace([np.inf, -np.inf], np.nan)
    features["vix_percentile_60d"] = vix_close.rolling(60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )

    # --- Equity momentum ---
    spy_close = spy["Close"]
    features["spy_ret_20d"] = spy_close.pct_change(20)
    features["spy_ret_60d"] = spy_close.pct_change(60)
    features["spy_ret_120d"] = spy_close.pct_change(120)

    # --- Realized vol ---
    features["realized_vol_20d"] = spy_ret.rolling(20).std() * np.sqrt(252)
    features["realized_vol_60d"] = spy_ret.rolling(60).std() * np.sqrt(252)

    # --- Credit spreads (HYG - LQD total return proxy) ---
    if hyg is not None and lqd is not None:
        hyg_ret = hyg["Close"].reindex(idx).ffill().pct_change()
        lqd_ret = lqd["Close"].reindex(idx).ffill().pct_change()
        credit_spread_ret = hyg_ret - lqd_ret
        features["credit_spread_20d"] = credit_spread_ret.rolling(20).sum()
        features["credit_spread_60d"] = credit_spread_ret.rolling(60).sum()
    else:
        features["credit_spread_20d"] = 0.0
        features["credit_spread_60d"] = 0.0

    # --- Cross-asset momentum ---
    for ticker_name in ["TLT", "GLD"]:
        t_data = data.get(ticker_name)
        if t_data is not None:
            t_close = t_data["Close"].reindex(idx).ffill()
            features[f"{ticker_name.lower()}_ret_60d"] = t_close.pct_change(60)
        else:
            features[f"{ticker_name.lower()}_ret_60d"] = 0.0

    # --- Breadth: % of sector ETFs above 200d SMA ---
    breadth_count = pd.Series(0.0, index=idx)
    sector_count = 0
    for s in SECTOR_ETFS:
        s_data = data.get(s)
        if s_data is not None:
            s_close = s_data["Close"].reindex(idx).ffill()
            sma200 = s_close.rolling(200).mean()
            above = (s_close > sma200).astype(float).reindex(idx).fillna(0)
            breadth_count = breadth_count + above
            sector_count += 1
    features["breadth_pct"] = breadth_count / max(sector_count, 1)

    # --- Yield curve proxy (TLT return) ---
    if tlt is not None:
        tlt_close = tlt["Close"].reindex(idx).ffill()
        features["tlt_ret_20d"] = tlt_close.pct_change(20)
        features["tlt_ret_60d"] = tlt_close.pct_change(60)
    else:
        features["tlt_ret_20d"] = 0.0
        features["tlt_ret_60d"] = 0.0

    # --- QQQ momentum ---
    qqq = data.get("QQQ")
    if qqq is not None:
        qqq_close = qqq["Close"].reindex(idx).ffill()
        features["qqq_ret_20d"] = qqq_close.pct_change(20)
        features["qqq_ret_60d"] = qqq_close.pct_change(60)
    else:
        features["qqq_ret_20d"] = 0.0
        features["qqq_ret_60d"] = 0.0

    # --- IWM (small cap) momentum ---
    iwm = data.get("IWM")
    if iwm is not None:
        iwm_close = iwm["Close"].reindex(idx).ffill()
        features["iwm_ret_20d"] = iwm_close.pct_change(20)
    else:
        features["iwm_ret_20d"] = 0.0

    # --- VIX rate of change ---
    features["vix_roc_5d"] = vix_close.pct_change(5)
    features["vix_roc_20d"] = vix_close.pct_change(20)

    # --- SPY distance from 200d SMA ---
    sma200_spy = spy_close.rolling(200).mean()
    features["spy_dist_sma200"] = (spy_close - sma200_spy) / sma200_spy

    # --- Momentum dispersion (std of sector returns) ---
    sector_rets = []
    for s in SECTOR_ETFS:
        s_data = data.get(s)
        if s_data is not None:
            sr = s_data["Close"].reindex(idx).ffill().pct_change(20)
            sector_rets.append(sr)
    if len(sector_rets) > 1:
        sector_ret_df = pd.concat(sector_rets, axis=1)
        features["momentum_dispersion"] = sector_ret_df.std(axis=1)
    else:
        features["momentum_dispersion"] = 0.0

    # Ensure exactly the feature columns we want (drop index-only)
    features = features.dropna()
    print(f"[FEATURES] Built {len(features.columns)} features, {len(features)} valid rows")
    print(f"  Features: {list(features.columns)}")
    return features


# ---------------------------------------------------------------------------
# Strategy return simulation
# ---------------------------------------------------------------------------
def simulate_strategy_returns(data, features_index):
    """Simulate daily returns for each of the 6 strategies."""
    idx = features_index
    spy = data["SPY"]
    spy_close = spy["Close"].reindex(idx).ffill()
    spy_ret = spy_close.pct_change().fillna(0)
    vix_close = data["^VIX"]["Close"].reindex(idx).ffill()

    strat_returns = pd.DataFrame(index=idx)

    # ---- 1. Sector ETF Momentum (monthly top-3 by 6-month return) ----
    sector_prices = {}
    for s in SECTOR_ETFS:
        if s in data:
            sector_prices[s] = data[s]["Close"].reindex(idx).ffill()
    sector_price_df = pd.DataFrame(sector_prices)
    mom_6m = sector_price_df.pct_change(126)  # ~6 months

    # Monthly rebalance: pick top 3 each month
    etf_mom_ret = pd.Series(0.0, index=idx)
    current_picks = []
    for i, date in enumerate(idx):
        # Rebalance on first trading day of month
        if i == 0 or date.month != idx[i - 1].month:
            row = mom_6m.loc[date]
            valid = row.dropna().sort_values(ascending=False)
            current_picks = list(valid.index[:3]) if len(valid) >= 3 else list(valid.index)
        if len(current_picks) > 0 and i > 0:
            daily_rets = []
            for p in current_picks:
                if p in sector_price_df.columns:
                    pr = sector_price_df[p]
                    if i < len(pr):
                        r = pr.iloc[i] / pr.iloc[i - 1] - 1 if pr.iloc[i - 1] != 0 else 0
                        daily_rets.append(r)
            etf_mom_ret.iloc[i] = np.mean(daily_rets) if daily_rets else 0.0
    strat_returns["etf_momentum"] = etf_mom_ret

    # ---- 2. Alt Trend Following (SMA200 on TLT, GLD, DBA) ----
    alt_ret = pd.Series(0.0, index=idx)
    active_count = 0
    for t in TREND_ASSETS:
        if t in data:
            t_close = data[t]["Close"].reindex(idx).ffill()
            sma200 = t_close.rolling(200).mean()
            t_ret = t_close.pct_change().fillna(0)
            signal = (t_close > sma200).astype(float).fillna(0)
            alt_ret = alt_ret + signal * t_ret
            active_count += 1
    if active_count > 0:
        alt_ret = alt_ret / active_count
    strat_returns["alt_trend"] = alt_ret

    # ---- 3. SPY Iron Condor Income ----
    # Constant daily income ~0.04%, minus big loss on VIX spike days (>3% SPY abs move)
    ic_ret = pd.Series(0.0004, index=idx)  # baseline daily income
    big_move = spy_ret.abs() > 0.03
    ic_ret[big_move] = -0.05  # 5% loss on crash/spike days
    strat_returns["iron_condor"] = ic_ret

    # ---- 4. VIX Mean-Reversion ----
    # 0 most days. +2% when VIX crosses above 30 then reverts within 30 days
    vmr_ret = pd.Series(0.0, index=idx)
    vix_arr = vix_close.values
    in_trade = False
    trade_start = 0
    for i in range(1, len(idx)):
        if not in_trade:
            if vix_arr[i] > 30 and (i == 0 or vix_arr[i - 1] <= 30):
                in_trade = True
                trade_start = i
        else:
            days_in = i - trade_start
            if vix_arr[i] < 25:
                # Reversion happened, distribute gain
                gain_per_day = 0.02 / max(days_in, 1)
                for j in range(trade_start, i + 1):
                    if j < len(vmr_ret):
                        vmr_ret.iloc[j] = gain_per_day
                in_trade = False
            elif days_in > 30:
                # Didn't revert in time, small loss
                loss_per_day = -0.005 / max(days_in, 1)
                for j in range(trade_start, i + 1):
                    if j < len(vmr_ret):
                        vmr_ret.iloc[j] = loss_per_day
                in_trade = False
    strat_returns["vix_meanrev"] = vmr_ret

    # ---- 5. LEAPS Momentum (3x leveraged ETF momentum) ----
    # Simplified: 3x the ETF momentum returns with leverage decay
    leaps_ret = 3.0 * etf_mom_ret
    # Leverage decay: subtract daily volatility drag
    vol_drag = 0.5 * (3.0 ** 2 - 3.0) * (etf_mom_ret.rolling(20).std().fillna(0.01)) ** 2
    leaps_ret = leaps_ret - vol_drag
    strat_returns["leaps_momentum"] = leaps_ret

    # ---- 6. Cross-Asset Trend Following ----
    # SMA200 trend on all available assets, equal weight active positions
    cat_ret = pd.Series(0.0, index=idx)
    cat_count = 0
    unique_trend = list(set(ALL_TREND_ASSETS))
    for t in unique_trend:
        if t in data:
            t_close = data[t]["Close"].reindex(idx).ffill()
            sma200 = t_close.rolling(200).mean()
            t_ret = t_close.pct_change().fillna(0)
            signal = (t_close > sma200).astype(float).fillna(0)
            cat_ret = cat_ret + signal * t_ret
            cat_count += 1
    if cat_count > 0:
        cat_ret = cat_ret / cat_count
    strat_returns["cross_asset_trend"] = cat_ret

    strat_returns = strat_returns.fillna(0)
    print(f"[STRATEGIES] Simulated {NUM_STRATEGIES} strategy returns, {len(strat_returns)} rows")
    for col in strat_returns.columns:
        ann_ret = strat_returns[col].mean() * 252
        ann_vol = strat_returns[col].std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        print(f"  {col}: Ann Return={ann_ret:.1%}, Ann Vol={ann_vol:.1%}, Sharpe={sharpe:.2f}")
    return strat_returns


# ---------------------------------------------------------------------------
# Target labels: optimal allocation via trailing Sharpe
# ---------------------------------------------------------------------------
def compute_optimal_allocations(strat_returns, lookback=20):
    """For each day, compute trailing Sharpe of each strategy.
    Optimal allocation = softmax of trailing Sharpes (hindsight label)."""
    trailing_sharpes = pd.DataFrame(index=strat_returns.index, columns=strat_returns.columns)
    for col in strat_returns.columns:
        roll_mean = strat_returns[col].rolling(lookback).mean()
        roll_std = strat_returns[col].rolling(lookback).std()
        trailing_sharpes[col] = (roll_mean / roll_std.replace(0, np.nan)).fillna(0)

    # Softmax of trailing Sharpes = target allocation
    sharpe_arr = trailing_sharpes.values.astype(float)
    # Temperature scaling to make allocations more decisive
    temp = 2.0
    exp_sharpes = np.exp(sharpe_arr * temp - np.max(sharpe_arr * temp, axis=1, keepdims=True))
    allocations = exp_sharpes / exp_sharpes.sum(axis=1, keepdims=True)
    allocations = np.nan_to_num(allocations, nan=1.0 / NUM_STRATEGIES)

    alloc_df = pd.DataFrame(allocations, index=strat_returns.index, columns=strat_returns.columns)
    print(f"[TARGETS] Computed optimal allocations. Mean allocation per strategy:")
    for col in alloc_df.columns:
        print(f"  {col}: {alloc_df[col].mean():.3f}")
    return alloc_df


# ---------------------------------------------------------------------------
# PyTorch Model
# ---------------------------------------------------------------------------
class RegimeAllocatorMLP(nn.Module):
    """3-layer MLP with batch norm and dropout. Outputs softmax portfolio weights."""

    def __init__(self, input_dim, num_strategies=6, hidden_dim=256, dropout=0.3):
        super().__init__()
        layers = []
        in_d = input_dim
        for i in range(NUM_LAYERS):
            layers.append(nn.Linear(in_d, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_d = hidden_dim
        layers.append(nn.Linear(hidden_dim, num_strategies))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.network(x)
        weights = torch.softmax(logits, dim=-1)
        return weights


def sharpe_loss(weights, strategy_returns):
    """Negative portfolio Sharpe ratio as loss.
    weights: (batch, num_strategies)
    strategy_returns: (batch, num_strategies)
    Returns: scalar (negative Sharpe to minimize)
    """
    # Portfolio return per day = sum(w_i * r_i)
    port_ret = (weights * strategy_returns).sum(dim=1)
    mean_ret = port_ret.mean()
    std_ret = port_ret.std() + 1e-8
    sharpe = mean_ret / std_ret
    # Also add a small MSE component vs optimal weights for stability
    return -sharpe


def combined_loss(weights, strategy_returns, target_weights, sharpe_weight=0.7, mse_weight=0.3):
    """Combined loss: Sharpe objective + MSE on target weights for stability."""
    s_loss = sharpe_loss(weights, strategy_returns)
    m_loss = nn.functional.mse_loss(weights, target_weights)
    return sharpe_weight * s_loss + mse_weight * m_loss


# ---------------------------------------------------------------------------
# Walk-forward training
# ---------------------------------------------------------------------------
def walk_forward_train(features_df, strat_returns_df, alloc_df):
    """Sliding walk-forward: 60-month train, 1-month test."""
    feature_cols = features_df.columns.tolist()
    strat_cols = strat_returns_df.columns.tolist()

    # Align all dataframes
    common_idx = features_df.index.intersection(strat_returns_df.index).intersection(alloc_df.index)
    features_df = features_df.loc[common_idx]
    strat_returns_df = strat_returns_df.loc[common_idx]
    alloc_df = alloc_df.loc[common_idx]

    # Group by year-month for walk-forward
    ym = features_df.index.to_period("M")
    unique_months = ym.unique().sort_values()
    print(f"\n[WF] Total months: {len(unique_months)}, Train window: {TRAIN_MONTHS}m, Test: {TEST_MONTHS}m")

    all_oot_results = []
    fold_metrics = []

    total_folds = len(unique_months) - TRAIN_MONTHS - TEST_MONTHS + 1
    if total_folds <= 0:
        raise ValueError(f"Not enough data for walk-forward. Have {len(unique_months)} months, need {TRAIN_MONTHS + TEST_MONTHS}")

    print(f"[WF] Will run {total_folds} folds\n")

    for fold_idx in range(total_folds):
        train_start_month = unique_months[fold_idx]
        train_end_month = unique_months[fold_idx + TRAIN_MONTHS - 1]
        test_month = unique_months[fold_idx + TRAIN_MONTHS]

        train_mask = (ym >= train_start_month) & (ym <= train_end_month)
        test_mask = ym == test_month

        X_train = features_df.loc[train_mask].values.astype(np.float32)
        Y_train_alloc = alloc_df.loc[train_mask].values.astype(np.float32)
        R_train = strat_returns_df.loc[train_mask].values.astype(np.float32)

        X_test = features_df.loc[test_mask].values.astype(np.float32)
        Y_test_alloc = alloc_df.loc[test_mask].values.astype(np.float32)
        R_test = strat_returns_df.loc[test_mask].values.astype(np.float32)
        test_dates = features_df.loc[test_mask].index

        if len(X_train) < 100 or len(X_test) < 5:
            continue

        # Scale features
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Tensors
        X_tr = torch.tensor(X_train_s, dtype=torch.float32).to(DEVICE)
        Y_tr = torch.tensor(Y_train_alloc, dtype=torch.float32).to(DEVICE)
        R_tr = torch.tensor(R_train, dtype=torch.float32).to(DEVICE)

        X_te = torch.tensor(X_test_s, dtype=torch.float32).to(DEVICE)
        R_te_tensor = torch.tensor(R_test, dtype=torch.float32).to(DEVICE)

        # Dataloader
        train_ds = TensorDataset(X_tr, Y_tr, R_tr)
        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

        # Model
        model = RegimeAllocatorMLP(
            input_dim=X_train_s.shape[1],
            num_strategies=NUM_STRATEGIES,
            hidden_dim=HIDDEN_DIM,
            dropout=DROPOUT
        ).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        # Train
        model.train()
        best_loss = float("inf")
        best_state = None
        patience_counter = 0
        for epoch in range(EPOCHS):
            epoch_loss = 0
            n_batches = 0
            for xb, yb, rb in train_dl:
                optimizer.zero_grad()
                w = model(xb)
                loss = combined_loss(w, rb, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= 15:
                break

        # Load best
        if best_state is not None:
            model.load_state_dict(best_state)
            model = model.to(DEVICE)

        # Predict OOT
        model.eval()
        with torch.no_grad():
            pred_weights = model(X_te).cpu().numpy()

        # Compute OOT portfolio returns
        port_ret = (pred_weights * R_test).sum(axis=1)

        for i, dt in enumerate(test_dates):
            all_oot_results.append({
                "date": dt,
                "fold": fold_idx,
                "port_return": port_ret[i],
                **{f"w_{STRATEGY_NAMES[j]}": pred_weights[i, j] for j in range(NUM_STRATEGIES)},
                **{f"r_{STRATEGY_NAMES[j]}": R_test[i, j] for j in range(NUM_STRATEGIES)},
            })

        # Fold-level metrics
        if len(port_ret) > 1:
            fold_sharpe = port_ret.mean() / (port_ret.std() + 1e-8) * np.sqrt(252)
            fold_ret = port_ret.mean() * 252
        else:
            fold_sharpe = 0
            fold_ret = 0

        fold_metrics.append({
            "fold": fold_idx,
            "test_month": str(test_month),
            "sharpe": fold_sharpe,
            "ann_return": fold_ret,
            "n_test_days": len(X_test),
        })

        if fold_idx % 20 == 0 or fold_idx == total_folds - 1:
            print(f"  Fold {fold_idx + 1}/{total_folds} | Test: {test_month} | "
                  f"Sharpe: {fold_sharpe:.2f} | Ann Ret: {fold_ret:.1%} | Days: {len(X_test)}")

    print(f"\n[WF] Completed {len(fold_metrics)} folds, {len(all_oot_results)} OOT days")
    return pd.DataFrame(all_oot_results), pd.DataFrame(fold_metrics)


# ---------------------------------------------------------------------------
# Baseline strategies for comparison
# ---------------------------------------------------------------------------
def compute_baselines(oot_df):
    """Compute baseline allocation methods for comparison."""
    strat_ret_cols = [c for c in oot_df.columns if c.startswith("r_")]
    R = oot_df[strat_ret_cols].values

    results = {}

    # Neural allocator
    results["Neural Allocator"] = oot_df["port_return"].values

    # Equal weight
    results["Equal Weight"] = R.mean(axis=1)

    # Risk parity (inverse vol weighting, rolling 60d)
    rp_returns = np.zeros(len(R))
    lookback = 60
    for i in range(lookback, len(R)):
        window = R[i - lookback:i]
        vols = window.std(axis=0) + 1e-8
        inv_vol = 1.0 / vols
        rp_weights = inv_vol / inv_vol.sum()
        rp_returns[i] = (rp_weights * R[i]).sum()
    results["Risk Parity"] = rp_returns

    # Momentum-weighted (trailing 20d return)
    mw_returns = np.zeros(len(R))
    mom_lookback = 20
    for i in range(mom_lookback, len(R)):
        window = R[i - mom_lookback:i]
        trail_ret = window.sum(axis=0)
        # Softmax of trailing returns
        exp_ret = np.exp(trail_ret * 10 - np.max(trail_ret * 10))
        mw_weights = exp_ret / exp_ret.sum()
        mw_returns[i] = (mw_weights * R[i]).sum()
    results["Momentum Weight"] = mw_returns

    return results


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------
def compute_metrics(returns, name="Strategy"):
    """Compute Sharpe, Sortino, MaxDD, CAGR."""
    r = np.array(returns)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return {"name": name, "sharpe": 0, "sortino": 0, "maxdd": 0, "cagr": 0, "n_days": 0}

    ann_ret = r.mean() * 252
    ann_vol = r.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    downside = r[r < 0]
    down_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else 1e-8
    sortino = ann_ret / down_vol if down_vol > 0 else 0

    cum = (1 + r).cumprod()
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    maxdd = dd.min()

    n_years = len(r) / 252
    total_ret = cum[-1] if len(cum) > 0 else 1.0
    cagr = total_ret ** (1 / n_years) - 1 if n_years > 0 and total_ret > 0 else 0

    return {
        "name": name,
        "sharpe": sharpe,
        "sortino": sortino,
        "maxdd": maxdd,
        "cagr": cagr,
        "n_days": len(r),
    }


def regime_stratify(oot_df, baseline_results, spy_data):
    """Stratify by green/red months using SPY close-to-close."""
    spy_close = spy_data["Close"].reindex(oot_df["date"]).ffill()
    oot_df = oot_df.copy()
    oot_df["spy_ret"] = spy_close.pct_change().values

    oot_df["month"] = oot_df["date"].dt.to_period("M")
    monthly_spy = oot_df.groupby("month")["spy_ret"].sum()

    green_months = set(monthly_spy[monthly_spy > 0].index)
    red_months = set(monthly_spy[monthly_spy <= 0].index)

    green_mask = oot_df["month"].isin(green_months).values
    red_mask = oot_df["month"].isin(red_months).values

    regime_results = {}
    for name, rets in baseline_results.items():
        r_green = rets[green_mask] if green_mask.sum() > 0 else np.array([0])
        r_red = rets[red_mask] if red_mask.sum() > 0 else np.array([0])
        m_green = compute_metrics(r_green, f"{name} (Green)")
        m_red = compute_metrics(r_red, f"{name} (Red)")
        regime_results[name] = {
            "green_sharpe": m_green["sharpe"],
            "red_sharpe": m_red["sharpe"],
            "green_months": len(green_months),
            "red_months": len(red_months),
        }

        # R1 check
        gs = abs(m_green["sharpe"])
        rs = abs(m_red["sharpe"])
        denom = max(gs, rs, 1e-8)
        regime_gap = abs(m_green["sharpe"] - m_red["sharpe"]) / denom
        regime_results[name]["regime_gap"] = regime_gap
        regime_results[name]["r1_pass"] = regime_gap < 0.50

    return regime_results


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------
def permutation_test(features_df, strat_returns_df, alloc_df, n_perms=3):
    """Shuffle features and re-train to check if edge is from signal or artifact."""
    print("\n[VALIDATION] Running permutation test ...")
    perm_sharpes = []
    feature_cols = features_df.columns.tolist()
    common_idx = features_df.index.intersection(strat_returns_df.index).intersection(alloc_df.index)

    for p in range(n_perms):
        print(f"  Permutation {p + 1}/{n_perms} ...")
        # Shuffle features row-wise (destroy temporal structure)
        perm_features = features_df.loc[common_idx].copy()
        for col in perm_features.columns:
            perm_features[col] = np.random.permutation(perm_features[col].values)

        # Quick train on last 60 months only (not full WF for speed)
        ym = perm_features.index.to_period("M")
        unique_months = ym.unique().sort_values()
        if len(unique_months) < TRAIN_MONTHS + 3:
            continue

        # Train on last TRAIN_MONTHS, test on last 3 months
        train_months_set = unique_months[-(TRAIN_MONTHS + 3):-3]
        test_months_set = unique_months[-3:]

        train_mask = ym.isin(train_months_set)
        test_mask = ym.isin(test_months_set)

        X_train = perm_features.loc[train_mask].values.astype(np.float32)
        Y_train = alloc_df.loc[common_idx].loc[train_mask].values.astype(np.float32)
        R_train = strat_returns_df.loc[common_idx].loc[train_mask].values.astype(np.float32)
        X_test = perm_features.loc[test_mask].values.astype(np.float32)
        R_test = strat_returns_df.loc[common_idx].loc[test_mask].values.astype(np.float32)

        if len(X_train) < 50 or len(X_test) < 10:
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = RegimeAllocatorMLP(X_train_s.shape[1], NUM_STRATEGIES, HIDDEN_DIM, DROPOUT).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        X_tr = torch.tensor(X_train_s, dtype=torch.float32).to(DEVICE)
        Y_tr = torch.tensor(Y_train, dtype=torch.float32).to(DEVICE)
        R_tr = torch.tensor(R_train, dtype=torch.float32).to(DEVICE)

        train_ds = TensorDataset(X_tr, Y_tr, R_tr)
        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

        model.train()
        for epoch in range(30):  # Fewer epochs for permutation
            for xb, yb, rb in train_dl:
                optimizer.zero_grad()
                w = model(xb)
                loss = combined_loss(w, rb, yb)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            X_te = torch.tensor(X_test_s, dtype=torch.float32).to(DEVICE)
            pred_w = model(X_te).cpu().numpy()

        port_ret = (pred_w * R_test).sum(axis=1)
        perm_sharpe = port_ret.mean() / (port_ret.std() + 1e-8) * np.sqrt(252)
        perm_sharpes.append(perm_sharpe)
        print(f"    Perm {p + 1} Sharpe: {perm_sharpe:.2f}")

    return perm_sharpes


def subperiod_stability(oot_df):
    """Check first-half vs second-half Sharpe."""
    n = len(oot_df)
    half = n // 2
    r1 = oot_df["port_return"].iloc[:half].values
    r2 = oot_df["port_return"].iloc[half:].values

    m1 = compute_metrics(r1, "First Half")
    m2 = compute_metrics(r2, "Second Half")

    print(f"\n[VALIDATION] Sub-period stability:")
    print(f"  First half:  Sharpe={m1['sharpe']:.2f}, CAGR={m1['cagr']:.1%}")
    print(f"  Second half: Sharpe={m2['sharpe']:.2f}, CAGR={m2['cagr']:.1%}")
    return m1, m2


def outlier_robustness(oot_df):
    """Drop top/bottom 1% return days, recompute metrics."""
    r = oot_df["port_return"].values
    lo = np.percentile(r, 1)
    hi = np.percentile(r, 99)
    trimmed = r[(r >= lo) & (r <= hi)]
    m = compute_metrics(trimmed, "Outlier-Trimmed")
    print(f"\n[VALIDATION] Outlier robustness (drop top/bottom 1%):")
    print(f"  Sharpe={m['sharpe']:.2f}, CAGR={m['cagr']:.1%}, MaxDD={m['maxdd']:.1%}")
    return m


# ---------------------------------------------------------------------------
# MLflow logging
# ---------------------------------------------------------------------------
def log_to_mlflow(all_metrics, fold_metrics_df, regime_results, perm_sharpes,
                  subperiod, outlier_m, oot_df):
    """Log everything to MLflow."""
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)

        with mlflow.start_run(run_name=f"neural_regime_alloc_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            # Log params
            mlflow.log_param("train_months", TRAIN_MONTHS)
            mlflow.log_param("test_months", TEST_MONTHS)
            mlflow.log_param("hidden_dim", HIDDEN_DIM)
            mlflow.log_param("num_layers", NUM_LAYERS)
            mlflow.log_param("dropout", DROPOUT)
            mlflow.log_param("lr", LR)
            mlflow.log_param("epochs", EPOCHS)
            mlflow.log_param("batch_size", BATCH_SIZE)
            mlflow.log_param("device", str(DEVICE))
            mlflow.log_param("num_strategies", NUM_STRATEGIES)
            mlflow.log_param("n_oot_days", len(oot_df))
            mlflow.log_param("n_folds", len(fold_metrics_df))

            # Log per-method metrics
            for name, m in all_metrics.items():
                prefix = name.replace(" ", "_").lower()
                mlflow.log_metric(f"{prefix}_sharpe", m["sharpe"])
                mlflow.log_metric(f"{prefix}_sortino", m["sortino"])
                mlflow.log_metric(f"{prefix}_maxdd", m["maxdd"])
                mlflow.log_metric(f"{prefix}_cagr", m["cagr"])

            # Log regime results for neural allocator
            if "Neural Allocator" in regime_results:
                rr = regime_results["Neural Allocator"]
                mlflow.log_metric("neural_green_sharpe", rr["green_sharpe"])
                mlflow.log_metric("neural_red_sharpe", rr["red_sharpe"])
                mlflow.log_metric("neural_regime_gap", rr["regime_gap"])
                mlflow.log_metric("neural_r1_pass", 1.0 if rr["r1_pass"] else 0.0)

            # Log permutation test
            if perm_sharpes:
                mlflow.log_metric("perm_mean_sharpe", np.mean(perm_sharpes))
                mlflow.log_metric("perm_std_sharpe", np.std(perm_sharpes))

            # Log subperiod
            if subperiod:
                mlflow.log_metric("first_half_sharpe", subperiod[0]["sharpe"])
                mlflow.log_metric("second_half_sharpe", subperiod[1]["sharpe"])

            # Log outlier robustness
            if outlier_m:
                mlflow.log_metric("outlier_trimmed_sharpe", outlier_m["sharpe"])

            # Log per-fold metrics
            for _, row in fold_metrics_df.iterrows():
                mlflow.log_metric("fold_sharpe", row["sharpe"], step=int(row["fold"]))

            print(f"\n[MLFLOW] Logged to experiment '{EXPERIMENT_NAME}'")

    except Exception as e:
        print(f"\n[MLFLOW] WARNING: Failed to log to MLflow: {e}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("=" * 80)
    print("NEURAL REGIME ALLOCATOR v1")
    print(f"Device: {DEVICE}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # 1. Download data
    data = download_data()
    if "SPY" not in data or "^VIX" not in data:
        raise RuntimeError("Critical tickers missing. Aborting.")

    # 2. Build features
    features = build_features(data)

    # 3. Simulate strategy returns
    strat_returns = simulate_strategy_returns(data, features.index)

    # 4. Compute optimal allocations (labels)
    alloc = compute_optimal_allocations(strat_returns)

    # 5. Walk-forward training
    oot_df, fold_metrics_df = walk_forward_train(features, strat_returns, alloc)

    if len(oot_df) < 20:
        raise RuntimeError(f"Only {len(oot_df)} OOT days - insufficient for evaluation")

    # 6. Baselines
    baseline_results = compute_baselines(oot_df)

    # 7. Compute all metrics
    print("\n" + "=" * 80)
    print("OUT-OF-TIME RESULTS COMPARISON")
    print("=" * 80)
    all_metrics = {}
    for name, rets in baseline_results.items():
        m = compute_metrics(rets, name)
        all_metrics[name] = m

    # Print summary table
    print(f"\n{'Method':<25} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} {'CAGR':>8} {'Days':>6}")
    print("-" * 65)
    for name, m in all_metrics.items():
        print(f"{m['name']:<25} {m['sharpe']:>8.2f} {m['sortino']:>8.2f} "
              f"{m['maxdd']:>7.1%} {m['cagr']:>7.1%} {m['n_days']:>6}")

    # 8. Regime stratification
    print("\n" + "=" * 80)
    print("REGIME STRATIFICATION (Green/Red Months by SPY)")
    print("=" * 80)
    regime_results = regime_stratify(oot_df, baseline_results, data["SPY"])
    print(f"\n{'Method':<25} {'Green Sharpe':>12} {'Red Sharpe':>12} {'Gap':>8} {'R1 Pass':>8}")
    print("-" * 70)
    for name, rr in regime_results.items():
        status = "PASS" if rr["r1_pass"] else "FAIL"
        print(f"{name:<25} {rr['green_sharpe']:>12.2f} {rr['red_sharpe']:>12.2f} "
              f"{rr['regime_gap']:>7.2f} {status:>8}")

    # 9. Validation tests
    print("\n" + "=" * 80)
    print("VALIDATION TESTS")
    print("=" * 80)

    perm_sharpes = permutation_test(features, strat_returns, alloc)
    neural_sharpe = all_metrics["Neural Allocator"]["sharpe"]
    if perm_sharpes:
        perm_mean = np.mean(perm_sharpes)
        perm_passed = neural_sharpe > perm_mean + 0.5  # Must beat random by 0.5 Sharpe
        print(f"\n  Permutation test: Neural Sharpe={neural_sharpe:.2f} vs Perm Mean={perm_mean:.2f}")
        print(f"  Result: {'PASS' if perm_passed else 'FAIL (potential artifact)'}")
    else:
        perm_passed = False
        print("  Permutation test: SKIPPED (insufficient data)")

    subperiod = subperiod_stability(oot_df)
    outlier_m = outlier_robustness(oot_df)

    # 10. Average allocation weights
    print("\n" + "=" * 80)
    print("AVERAGE NEURAL ALLOCATION WEIGHTS")
    print("=" * 80)
    w_cols = [c for c in oot_df.columns if c.startswith("w_")]
    for col in w_cols:
        strat_name = col.replace("w_", "")
        print(f"  {strat_name:<25} {oot_df[col].mean():>6.1%} (std: {oot_df[col].std():.1%})")

    # 11. MLflow logging
    log_to_mlflow(all_metrics, fold_metrics_df, regime_results, perm_sharpes,
                  subperiod, outlier_m, oot_df)

    # 12. Final summary
    elapsed = time.time() - t0
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"  Total runtime: {elapsed / 60:.1f} minutes")
    print(f"  OOT days evaluated: {len(oot_df)}")
    print(f"  Folds completed: {len(fold_metrics_df)}")
    print(f"  Neural Allocator Sharpe: {all_metrics['Neural Allocator']['sharpe']:.2f}")
    print(f"  Equal Weight Sharpe: {all_metrics['Equal Weight']['sharpe']:.2f}")
    print(f"  Improvement vs EW: {all_metrics['Neural Allocator']['sharpe'] - all_metrics['Equal Weight']['sharpe']:+.2f}")

    r1_status = regime_results.get("Neural Allocator", {}).get("r1_pass", False)
    print(f"  R1 Regime-Agnostic: {'PASS' if r1_status else 'FAIL'}")
    print(f"  Permutation Test: {'PASS' if perm_passed else 'FAIL'}")

    if subperiod:
        half_ratio = min(abs(subperiod[0]["sharpe"]), abs(subperiod[1]["sharpe"])) / max(
            abs(subperiod[0]["sharpe"]), abs(subperiod[1]["sharpe"]), 1e-8
        )
        print(f"  Sub-period stability ratio: {half_ratio:.2f} (>0.5 = stable)")

    print(f"\n  Script: neural_regime_allocator_v1.py")
    print(f"  MLflow experiment: {EXPERIMENT_NAME}")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        traceback.print_exc()
        sys.exit(1)
