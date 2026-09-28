#!/usr/bin/env python3
"""
BPS Spread Width Optimizer — MLP on Walk-Forward Simulation
=============================================================
Hypothesis: optimal BPS spread width varies by ticker price level, IV,
VIX regime, and momentum.  Currently the system uses fixed $15 for all
tickers.  This script:

  1. Simulates BPS trades at widths {5, 10, 15, 20, 25} for each
     ticker × date in the GA-optimized 20-ticker universe.
  2. Labels each (ticker, date, width) with outcome metrics.
  3. Trains an MLP to predict the best width given market features.
  4. Walk-forward: 120d train, 30d OOS, sliding window.
  5. Reports regime-stratified Sharpe and OOS accuracy.

Designed to run on Neptune (RTX 3090).
MLflow tracking to http://jupiter:5000.

Author: Claude (2026-07-09)
"""

import argparse
import math
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# ── Paths (work on both Jupiter and Neptune) ──
if Path("/home/nick/Lvl3Quant").exists():
    ROOT = Path("/home/nick/Lvl3Quant")
    CACHE = ROOT / "wheel_strategy_v1" / "data" / "cache"
    FEATURE_STORE = ROOT / "data" / "feature_store" / "v2"
else:
    ROOT = Path("/home/jupiter/Lvl3Quant")
    CACHE = ROOT / "wheel_strategy_v1" / "data" / "cache"
    FEATURE_STORE = ROOT / "data" / "feature_store" / "v2"

OUTPUT_DIR = ROOT / "output" / "bps_width_optimizer"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──
GA_UNIVERSE = [
    "NFLX", "NVDA", "PLTR", "WMT", "GM", "PFE", "LLY", "MCD",
    "CL", "T", "SMCI", "OXY", "HOOD", "JNJ", "F", "TGT",
    "VZ", "TSLA", "PANW", "TMUS",
]

WIDTHS = [5.0, 10.0, 15.0, 20.0, 25.0]
DTE_TARGET = 10
RISK_FREE = 0.04
COST_PER_CONTRACT_LEG = 0.65  # $0.65/contract/leg, 2 legs = $1.30/spread
PROFIT_TAKE_PCT = 0.65
PUT_DELTA = 0.30

SECTOR_MAP = {
    "CL": "ConsumerDefensive", "F": "ConsumerCyclical", "GM": "ConsumerCyclical",
    "HOOD": "FinancialServices", "JNJ": "Healthcare", "LLY": "Healthcare",
    "MCD": "ConsumerCyclical", "NFLX": "CommServices", "NVDA": "Technology",
    "OXY": "Energy", "PANW": "Technology", "PFE": "Healthcare",
    "PLTR": "Technology", "SMCI": "Technology", "T": "CommServices",
    "TGT": "ConsumerDefensive", "TMUS": "CommServices", "TSLA": "ConsumerCyclical",
    "VZ": "CommServices", "WMT": "ConsumerDefensive",
}
SECTORS = sorted(set(SECTOR_MAP.values()))
SECTOR_TO_IDX = {s: i for i, s in enumerate(SECTORS)}

# ── Black-Scholes helpers ──
SQRT_2PI = math.sqrt(2 * math.pi)


def _Phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_put_price(S, K, T, sigma, r=0.04):
    """BS put price."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * _Phi(-d2) - S * _Phi(-d1)


def strike_from_delta(S, T, sigma, target_delta, r=0.04):
    """Find put strike for given delta (negative convention: -0.30)."""
    if T <= 0 or sigma <= 0:
        return S
    from scipy.optimize import brentq

    def obj(K):
        if K <= 0:
            return 1.0
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        return _Phi(d1) - 1.0 + target_delta

    try:
        K = brentq(obj, S * 0.5, S * 1.1, xtol=0.01)
        # Round to nearest dollar (option strikes are in $1 increments for most)
        return round(K)
    except Exception:
        return round(S * (1 - sigma * math.sqrt(T) * 0.5))


# ═══════════════════════════════════════════════════════════════
# PHASE 1: SIMULATE BPS AT MULTIPLE WIDTHS
# ═══════════════════════════════════════════════════════════════

def load_data():
    """Load prices, IV, macro, and regime data."""
    print("Loading data...")

    # Prices
    prices = pd.read_parquet(CACHE / "prices_v2.parquet")
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices[prices["ticker"].isin(GA_UNIVERSE)].copy()
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    # 20d momentum
    prices["ret_20d"] = prices.groupby("ticker")["close"].pct_change(20)
    # 60d realized vol
    prices["rv_20d"] = prices.groupby("ticker")["ret"].rolling(20).std().reset_index(0, drop=True) * np.sqrt(252)

    # IV features
    iv = pd.read_parquet(CACHE / "iv_features_real.parquet")
    iv["date"] = pd.to_datetime(iv["date"])
    iv = iv[iv["ticker"].isin(GA_UNIVERSE)][["date", "ticker", "sigma", "iv_rank"]].copy()

    # Macro (VIX, VIX term structure)
    macro = pd.read_parquet(CACHE / "macro.parquet")
    macro["date"] = pd.to_datetime(macro["date"])
    macro = macro[["date", "vix", "vix3m", "vix_ts"]].dropna(subset=["vix"])

    # Regime
    regime = pd.read_parquet(CACHE / "regime_overlay.parquet")
    regime["date"] = pd.to_datetime(regime["date"])

    # SPY for regime classification (green/red day)
    spy = pd.read_parquet(CACHE / "spy_prices.parquet")
    spy["date"] = pd.to_datetime(spy["date"])
    spy = spy.rename(columns={"close": "spy_close"})
    spy["spy_sma50"] = spy["spy_close"].rolling(50).mean()
    spy["spy_ret"] = spy["spy_close"].pct_change()

    print(f"  Prices: {len(prices):,} rows, {prices['ticker'].nunique()} tickers")
    print(f"  IV: {len(iv):,} rows")
    print(f"  Macro: {len(macro):,} rows")
    print(f"  Date range: {prices['date'].min().date()} to {prices['date'].max().date()}")

    return prices, iv, macro, regime, spy


def simulate_bps_trades(prices, iv, macro, spy):
    """
    For each (ticker, date, width), simulate a BPS trade opened on that date
    and held for DTE_TARGET days, with profit_take exit.

    Returns DataFrame with trade features and outcomes per width.
    """
    print("\nSimulating BPS trades across widths...")

    # Merge everything by date
    df = prices.merge(iv, on=["ticker", "date"], how="left", suffixes=("", "_iv"))
    df = df.merge(macro, on="date", how="left")
    df = df.merge(spy[["date", "spy_close", "spy_sma50", "spy_ret"]], on="date", how="left")

    # Forward-fill IV and macro for missing days
    for col in ["sigma", "iv_rank", "vix", "vix3m", "vix_ts"]:
        df[col] = df.groupby("ticker")[col].ffill()

    # Drop rows without critical data
    df = df.dropna(subset=["close", "sigma", "vix"]).copy()

    # Pre-compute IV rank as rolling percentile if missing
    mask = df["iv_rank"].isna()
    if mask.any():
        df.loc[mask, "iv_rank"] = df.groupby("ticker")["sigma"].rank(pct=True)

    # For each trade: we need the close price DTE_TARGET days forward
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Create forward price columns for daily mark-to-market
    for d in range(1, DTE_TARGET + 1):
        df[f"fwd_close_{d}"] = df.groupby("ticker")["close"].shift(-d)

    # Drop rows where we can't see the full horizon
    df = df.dropna(subset=[f"fwd_close_{DTE_TARGET}"]).copy()

    records = []
    n_total = len(df)
    print(f"  Processing {n_total:,} ticker-date combinations...")

    for idx, row in df.iterrows():
        if idx % 10000 == 0:
            print(f"    {idx:,}/{n_total:,} ({100*idx/n_total:.0f}%)")

        S = row["close"]
        sigma = row["sigma"]
        T = DTE_TARGET / 252.0

        if sigma <= 0.05 or S < 5:
            continue

        # Find short strike at target delta
        short_strike = strike_from_delta(S, T, sigma, PUT_DELTA)

        for width in WIDTHS:
            long_strike = short_strike - width

            if long_strike <= 0:
                continue

            # Price short and long puts
            short_put_price = bs_put_price(S, short_strike, T, sigma)
            long_put_price = bs_put_price(S, long_strike, T, sigma)
            net_credit = short_put_price - long_put_price

            if net_credit <= 0:
                continue

            max_loss = (width * 100) - (net_credit * 100)
            max_loss_per_share = width - net_credit

            # Commission cost per spread (2 legs)
            commission = 2 * COST_PER_CONTRACT_LEG

            # Simulate daily evolution - check for profit take
            realized_pnl = None
            exit_day = DTE_TARGET
            exit_type = "expiry"

            for d in range(1, DTE_TARGET + 1):
                fwd_S = row[f"fwd_close_{d}"]
                T_rem = (DTE_TARGET - d) / 252.0

                # Current spread value
                sp = bs_put_price(fwd_S, short_strike, T_rem, sigma)
                lp = bs_put_price(fwd_S, long_strike, T_rem, sigma)
                spread_value = sp - lp

                unrealized = (net_credit - spread_value) * 100

                # Profit take check
                if unrealized >= PROFIT_TAKE_PCT * net_credit * 100:
                    realized_pnl = unrealized - commission
                    exit_day = d
                    exit_type = "profit_take"
                    break

            if realized_pnl is None:
                # Held to expiry
                fwd_S_exp = row[f"fwd_close_{DTE_TARGET}"]
                if fwd_S_exp >= short_strike:
                    # Both OTM - keep full credit
                    realized_pnl = net_credit * 100 - commission
                elif fwd_S_exp >= long_strike:
                    # Short put ITM, long put OTM
                    intrinsic = short_strike - fwd_S_exp
                    realized_pnl = (net_credit - intrinsic) * 100 - commission
                else:
                    # Both ITM - max loss
                    realized_pnl = -max_loss - commission

            # Record
            records.append({
                "ticker": row["ticker"],
                "date": row["date"],
                "width": width,
                "stock_price": S,
                "short_strike": short_strike,
                "sigma": sigma,
                "iv_rank": row["iv_rank"],
                "vix": row["vix"],
                "vix_ts": row.get("vix_ts", np.nan),
                "ret_20d": row.get("ret_20d", np.nan),
                "rv_20d": row.get("rv_20d", np.nan),
                "spy_above_sma50": 1.0 if row.get("spy_close", 0) > row.get("spy_sma50", 0) else 0.0,
                "sector": SECTOR_MAP.get(row["ticker"], "Other"),
                "net_credit": net_credit,
                "max_loss": max_loss,
                "realized_pnl": realized_pnl,
                "exit_type": exit_type,
                "exit_day": exit_day,
                "win": 1 if realized_pnl > 0 else 0,
                "pnl_pct": realized_pnl / max_loss if max_loss > 0 else 0,
            })

    sim_df = pd.DataFrame(records)
    print(f"\n  Total simulated trades: {len(sim_df):,}")
    print(f"  By width: {sim_df.groupby('width')['realized_pnl'].count().to_dict()}")

    return sim_df


def compute_width_stats(sim_df):
    """Compute per-width performance statistics."""
    print("\n" + "=" * 70)
    print("PERFORMANCE BY SPREAD WIDTH")
    print("=" * 70)

    stats = []
    for width in WIDTHS:
        wdf = sim_df[sim_df["width"] == width]
        n = len(wdf)
        wr = wdf["win"].mean()
        avg_pnl = wdf["realized_pnl"].mean()
        total_pnl = wdf["realized_pnl"].sum()
        std_pnl = wdf["realized_pnl"].std()
        sharpe = avg_pnl / std_pnl * np.sqrt(252 / DTE_TARGET) if std_pnl > 0 else 0

        # By regime
        bull = wdf[wdf["spy_above_sma50"] == 1]
        bear = wdf[wdf["spy_above_sma50"] == 0]
        bull_sharpe = (bull["realized_pnl"].mean() / bull["realized_pnl"].std() * np.sqrt(252 / DTE_TARGET)
                       if len(bull) > 30 and bull["realized_pnl"].std() > 0 else np.nan)
        bear_sharpe = (bear["realized_pnl"].mean() / bear["realized_pnl"].std() * np.sqrt(252 / DTE_TARGET)
                       if len(bear) > 30 and bear["realized_pnl"].std() > 0 else np.nan)

        stats.append({
            "width": width, "n": n, "wr": wr, "avg_pnl": avg_pnl,
            "total_pnl": total_pnl, "sharpe": sharpe,
            "bull_sharpe": bull_sharpe, "bear_sharpe": bear_sharpe,
        })

        print(f"\n  Width ${width:.0f}:")
        print(f"    Trades: {n:,}  WR: {wr:.1%}  Avg PnL: ${avg_pnl:.2f}")
        print(f"    Sharpe: {sharpe:.2f}  Bull Sharpe: {bull_sharpe:.2f}  Bear Sharpe: {bear_sharpe:.2f}")

    return pd.DataFrame(stats)


# ═══════════════════════════════════════════════════════════════
# PHASE 2: LABEL OPTIMAL WIDTH PER (TICKER, DATE)
# ═══════════════════════════════════════════════════════════════

def label_optimal_width(sim_df):
    """
    For each (ticker, date), find the width that maximizes risk-adjusted PnL.
    Use pnl_pct (PnL / max_loss) as the metric — makes widths comparable.
    """
    print("\nLabeling optimal width per (ticker, date)...")

    # Pivot: for each (ticker, date), get pnl_pct per width
    pivot = sim_df.pivot_table(
        index=["ticker", "date"],
        columns="width",
        values="pnl_pct",
        aggfunc="first",
    )

    # Best width = highest pnl_pct
    best_width = pivot.idxmax(axis=1)
    best_width.name = "best_width"

    # Map width to class index
    width_to_class = {w: i for i, w in enumerate(WIDTHS)}
    best_class = best_width.map(width_to_class)
    best_class.name = "best_class"

    # Merge features from any width row (features are the same across widths)
    # Take width=15 as reference for features
    features = sim_df[sim_df["width"] == 15.0].set_index(["ticker", "date"])[[
        "stock_price", "sigma", "iv_rank", "vix", "vix_ts",
        "ret_20d", "rv_20d", "spy_above_sma50", "sector",
    ]].copy()

    labeled = features.join(best_width).join(best_class).dropna(subset=["best_width", "best_class"])

    print(f"  Labeled samples: {len(labeled):,}")
    print(f"  Best width distribution:")
    for w in WIDTHS:
        ct = (labeled["best_width"] == w).sum()
        print(f"    ${w:.0f}: {ct:,} ({ct / len(labeled):.1%})")

    return labeled.reset_index()


# ═══════════════════════════════════════════════════════════════
# PHASE 3: MLP MODEL
# ═══════════════════════════════════════════════════════════════

class WidthMLP(nn.Module):
    """MLP predicting optimal BPS spread width class."""

    def __init__(self, n_features, n_classes=5, hidden_dims=(128, 64, 32), dropout=0.3):
        super().__init__()
        layers = []
        in_dim = n_features
        for h in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h
        layers.append(nn.Linear(in_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def prepare_features(labeled_df):
    """
    Build feature matrix X and target y from labeled data.

    Features:
      - log(stock_price)          : price level (normalizes ticker differences)
      - sigma (ATM IV)            : current implied volatility
      - iv_rank                   : IV percentile vs 252d history
      - vix                       : market-wide fear gauge
      - vix_ts                    : VIX term structure (contango/backwardation)
      - ret_20d                   : 20-day momentum
      - rv_20d                    : 20-day realized vol
      - spy_above_sma50           : bull/bear regime binary
      - sector one-hot (6 cats)   : sector encoding
      - width_as_pct_of_price     : $10 on a $50 stock != $10 on a $500 stock
    """
    df = labeled_df.copy()

    # Numeric features
    df["log_price"] = np.log(df["stock_price"].clip(lower=1))

    # Sector one-hot
    for s in SECTORS:
        df[f"sector_{s}"] = (df["sector"] == s).astype(float)

    feature_cols = [
        "log_price", "sigma", "iv_rank", "vix", "vix_ts",
        "ret_20d", "rv_20d", "spy_above_sma50",
    ] + [f"sector_{s}" for s in SECTORS]

    X = df[feature_cols].values.astype(np.float32)
    y = df["best_class"].values.astype(np.int64)

    # Handle NaN — fill with median
    for col_idx in range(X.shape[1]):
        col_data = X[:, col_idx]
        mask = np.isnan(col_data)
        if mask.any():
            median_val = np.nanmedian(col_data)
            X[mask, col_idx] = median_val if not np.isnan(median_val) else 0.0

    return X, y, feature_cols


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)
        correct += (logits.argmax(1) == yb).sum().item()
        total += len(xb)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_true = []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * len(xb)
        preds = logits.argmax(1)
        correct += (preds == yb).sum().item()
        total += len(xb)
        all_preds.extend(preds.cpu().numpy())
        all_true.extend(yb.cpu().numpy())
    return total_loss / total, correct / total, np.array(all_preds), np.array(all_true)


# ═══════════════════════════════════════════════════════════════
# PHASE 4: WALK-FORWARD TRAINING
# ═══════════════════════════════════════════════════════════════

def walk_forward_train(labeled_df, device, mlflow_run=None,
                       train_days=120, oos_days=30, epochs=50, lr=1e-3, batch_size=256):
    """
    Sliding window walk-forward:
      - 120d train, 30d OOS, slide by 30d
      - MLP trained from scratch each fold (no warm start for clean OOS)
    """
    print("\n" + "=" * 70)
    print("WALK-FORWARD TRAINING")
    print(f"  Train: {train_days}d, OOS: {oos_days}d, sliding")
    print("=" * 70)

    labeled_df = labeled_df.sort_values("date").reset_index(drop=True)
    dates = sorted(labeled_df["date"].unique())
    min_date, max_date = dates[0], dates[-1]

    X_all, y_all, feature_cols = prepare_features(labeled_df)
    n_features = X_all.shape[1]
    n_classes = len(WIDTHS)

    # Compute per-feature stats for normalization (use train windows)
    all_dates = labeled_df["date"].values

    oos_records = []
    fold = 0
    train_start = min_date

    while True:
        train_end = train_start + pd.Timedelta(days=train_days)
        oos_start = train_end
        oos_end = oos_start + pd.Timedelta(days=oos_days)

        if oos_end > max_date:
            break

        train_mask = (all_dates >= train_start) & (all_dates < train_end)
        oos_mask = (all_dates >= oos_start) & (all_dates < oos_end)

        n_train = train_mask.sum()
        n_oos = oos_mask.sum()

        if n_train < 100 or n_oos < 20:
            train_start += pd.Timedelta(days=oos_days)
            continue

        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_oos, y_oos = X_all[oos_mask], y_all[oos_mask]

        # Z-score normalize using train stats
        mu = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std[std < 1e-8] = 1.0
        X_train_n = (X_train - mu) / std
        X_oos_n = (X_oos - mu) / std

        # Build loaders
        train_ds = TensorDataset(
            torch.from_numpy(X_train_n), torch.from_numpy(y_train)
        )
        oos_ds = TensorDataset(
            torch.from_numpy(X_oos_n), torch.from_numpy(y_oos)
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=4, pin_memory=True)
        oos_loader = DataLoader(oos_ds, batch_size=batch_size, shuffle=False,
                                num_workers=4, pin_memory=True)

        # Fresh model per fold
        model = WidthMLP(n_features, n_classes).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_oos_acc = 0
        best_state = None

        for ep in range(epochs):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            oos_loss, oos_acc, oos_preds, oos_true = eval_model(model, oos_loader, criterion, device)
            scheduler.step()

            if oos_acc > best_oos_acc:
                best_oos_acc = oos_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Reload best
        if best_state:
            model.load_state_dict(best_state)
            model.to(device)
            _, final_oos_acc, oos_preds, oos_true = eval_model(
                model, oos_loader, criterion, device
            )
        else:
            final_oos_acc = best_oos_acc

        # Log fold results
        fold_str = (f"Fold {fold:3d} | "
                    f"Train: {pd.Timestamp(train_start).date()} to {pd.Timestamp(train_end).date()} "
                    f"({n_train:,}) | "
                    f"OOS: {pd.Timestamp(oos_start).date()} to {pd.Timestamp(oos_end).date()} "
                    f"({n_oos:,}) | "
                    f"OOS Acc: {final_oos_acc:.3f}")
        print(fold_str)

        # Compute OOS PnL improvement: MLP-chosen width vs fixed $15
        oos_dates = labeled_df.loc[oos_mask, "date"].values
        oos_tickers = labeled_df.loc[oos_mask, "ticker"].values

        for i, (pred_cls, true_cls) in enumerate(zip(oos_preds, oos_true)):
            oos_records.append({
                "fold": fold,
                "date": oos_dates[i],
                "ticker": oos_tickers[i],
                "pred_class": pred_cls,
                "true_class": true_cls,
                "pred_width": WIDTHS[pred_cls],
                "true_best_width": WIDTHS[true_cls],
                "correct": int(pred_cls == true_cls),
            })

        if mlflow_run:
            try:
                import mlflow
                mlflow.log_metrics({
                    f"fold_{fold}_oos_acc": final_oos_acc,
                    f"fold_{fold}_n_train": n_train,
                    f"fold_{fold}_n_oos": n_oos,
                }, step=fold)
            except Exception:
                pass

        fold += 1
        train_start += pd.Timedelta(days=oos_days)

    oos_df = pd.DataFrame(oos_records)
    print(f"\n  Total folds: {fold}")
    print(f"  Total OOS predictions: {len(oos_df):,}")

    if len(oos_df) > 0:
        overall_acc = oos_df["correct"].mean()
        print(f"  Overall OOS accuracy: {overall_acc:.3f}")

        # Per-width accuracy
        print("\n  Accuracy by true optimal width:")
        for cls_idx, w in enumerate(WIDTHS):
            mask = oos_df["true_class"] == cls_idx
            if mask.sum() > 0:
                acc = oos_df.loc[mask, "correct"].mean()
                print(f"    ${w:.0f}: {acc:.3f} (n={mask.sum():,})")

    return oos_df, model, feature_cols


# ═══════════════════════════════════════════════════════════════
# PHASE 5: EVALUATE OOS — PNL IMPROVEMENT VS FIXED WIDTH
# ═══════════════════════════════════════════════════════════════

def evaluate_oos_pnl(oos_df, sim_df):
    """
    Compare OOS performance:
      - MLP-selected width vs fixed $15
      - Both using the same sim_df trade outcomes
    """
    print("\n" + "=" * 70)
    print("OOS PNL COMPARISON: MLP vs FIXED $15")
    print("=" * 70)

    if len(oos_df) == 0:
        print("  No OOS data to evaluate.")
        return {}

    # Join OOS predictions with simulation outcomes
    sim_lookup = sim_df.set_index(["ticker", "date", "width"])[
        ["realized_pnl", "pnl_pct", "max_loss", "win"]
    ]

    mlp_pnls = []
    fixed_pnls = []
    oracle_pnls = []

    for _, row in oos_df.iterrows():
        key_mlp = (row["ticker"], row["date"], row["pred_width"])
        key_fixed = (row["ticker"], row["date"], 15.0)
        key_oracle = (row["ticker"], row["date"], row["true_best_width"])

        try:
            mlp_pnls.append(sim_lookup.loc[key_mlp, "realized_pnl"])
        except KeyError:
            mlp_pnls.append(np.nan)

        try:
            fixed_pnls.append(sim_lookup.loc[key_fixed, "realized_pnl"])
        except KeyError:
            fixed_pnls.append(np.nan)

        try:
            oracle_pnls.append(sim_lookup.loc[key_oracle, "realized_pnl"])
        except KeyError:
            oracle_pnls.append(np.nan)

    oos_df = oos_df.copy()
    oos_df["mlp_pnl"] = mlp_pnls
    oos_df["fixed_pnl"] = fixed_pnls
    oos_df["oracle_pnl"] = oracle_pnls
    oos_df = oos_df.dropna(subset=["mlp_pnl", "fixed_pnl"])

    if len(oos_df) == 0:
        print("  No matched OOS trades.")
        return {}

    # Aggregate metrics
    def compute_metrics(pnl_series, label):
        total = pnl_series.sum()
        avg = pnl_series.mean()
        std = pnl_series.std()
        wr = (pnl_series > 0).mean()
        sharpe = avg / std * np.sqrt(252 / DTE_TARGET) if std > 0 else 0
        neg = pnl_series[pnl_series < 0]
        downside_std = neg.std() if len(neg) > 1 else 1e-6
        sortino = avg / downside_std * np.sqrt(252 / DTE_TARGET) if downside_std > 0 else 0
        pf = pnl_series[pnl_series > 0].sum() / abs(pnl_series[pnl_series < 0].sum()) if (pnl_series < 0).any() else float("inf")

        print(f"\n  {label}:")
        print(f"    Total PnL:  ${total:>12,.2f}")
        print(f"    Avg PnL:    ${avg:>12,.2f}")
        print(f"    Win Rate:    {wr:>12.1%}")
        print(f"    Sharpe:      {sharpe:>12.2f}")
        print(f"    Sortino:     {sortino:>12.2f}")
        print(f"    Profit Fac:  {pf:>12.2f}")
        return {"total_pnl": total, "avg_pnl": avg, "wr": wr, "sharpe": sharpe,
                "sortino": sortino, "pf": pf}

    results = {}
    results["fixed_15"] = compute_metrics(oos_df["fixed_pnl"], "FIXED $15 WIDTH")
    results["mlp"] = compute_metrics(oos_df["mlp_pnl"], "MLP-SELECTED WIDTH")
    results["oracle"] = compute_metrics(oos_df["oracle_pnl"], "ORACLE (perfect selection)")

    # Width selection distribution
    print("\n  MLP width selection distribution (OOS):")
    for w in WIDTHS:
        ct = (oos_df["pred_width"] == w).sum()
        print(f"    ${w:.0f}: {ct:,} ({ct / len(oos_df):.1%})")

    # Regime analysis (R1 check)
    print("\n  REGIME CHECK (R1 gate):")
    # Use sim_df to get spy_above_sma50 for OOS dates
    spy_regime = sim_df[sim_df["width"] == 15.0].set_index(["ticker", "date"])["spy_above_sma50"]

    regimes = []
    for _, row in oos_df.iterrows():
        try:
            regimes.append(spy_regime.loc[(row["ticker"], row["date"])])
        except KeyError:
            regimes.append(np.nan)

    oos_df["regime"] = regimes
    oos_df = oos_df.dropna(subset=["regime"])

    for regime_label, regime_val in [("BULL (SPY>SMA50)", 1.0), ("BEAR (SPY<SMA50)", 0.0)]:
        rmask = oos_df["regime"] == regime_val
        if rmask.sum() < 30:
            continue
        rpnl = oos_df.loc[rmask, "mlp_pnl"]
        std_r = rpnl.std()
        sharpe_r = rpnl.mean() / std_r * np.sqrt(252 / DTE_TARGET) if std_r > 0 else 0
        print(f"    {regime_label}: Sharpe {sharpe_r:.2f} (n={rmask.sum():,})")

    return results


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="BPS Width Optimizer")
    parser.add_argument("--skip-sim", action="store_true",
                        help="Skip simulation, load from cached parquet")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-days", type=int, default=120)
    parser.add_argument("--oos-days", type=int, default=30)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # MLflow setup
    mlflow_run = None
    if not args.no_mlflow:
        try:
            import mlflow
            mlflow.set_tracking_uri("http://jupiter:5000")
            mlflow.set_experiment("bps_width_optimizer")
            mlflow_run = mlflow.start_run(run_name=f"width_mlp_{datetime.now():%Y%m%d_%H%M%S}")
            mlflow.log_params({
                "widths": str(WIDTHS),
                "train_days": args.train_days,
                "oos_days": args.oos_days,
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "put_delta": PUT_DELTA,
                "dte_target": DTE_TARGET,
                "n_tickers": len(GA_UNIVERSE),
            })
            print(f"MLflow run started: {mlflow_run.info.run_id}")
        except Exception as e:
            print(f"MLflow setup failed: {e}")
            mlflow_run = None

    sim_path = OUTPUT_DIR / "sim_trades.parquet"
    labeled_path = OUTPUT_DIR / "labeled_data.parquet"

    if args.skip_sim and sim_path.exists():
        print("Loading cached simulation...")
        sim_df = pd.read_parquet(sim_path)
        labeled_df = pd.read_parquet(labeled_path)
    else:
        # Phase 1: Simulate
        prices, iv, macro, regime, spy = load_data()
        sim_df = simulate_bps_trades(prices, iv, macro, spy)
        sim_df.to_parquet(sim_path, index=False)
        print(f"Saved simulation to {sim_path}")

        # Phase 1b: Width stats
        width_stats = compute_width_stats(sim_df)

        # Phase 2: Label
        labeled_df = label_optimal_width(sim_df)
        labeled_df.to_parquet(labeled_path, index=False)
        print(f"Saved labeled data to {labeled_path}")

    # Phase 3+4: Walk-forward MLP training
    oos_df, model, feature_cols = walk_forward_train(
        labeled_df, device, mlflow_run,
        train_days=args.train_days, oos_days=args.oos_days,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
    )

    # Phase 5: Evaluate OOS PnL
    results = evaluate_oos_pnl(oos_df, sim_df)

    # Save outputs
    oos_df.to_parquet(OUTPUT_DIR / "oos_predictions.parquet", index=False)

    # Save model
    if model is not None:
        torch.save(model.state_dict(), OUTPUT_DIR / "width_mlp_final.pt")
        print(f"\nModel saved to {OUTPUT_DIR / 'width_mlp_final.pt'}")

    # Log final metrics to MLflow
    if mlflow_run:
        try:
            import mlflow
            if results.get("mlp"):
                for k, v in results["mlp"].items():
                    if isinstance(v, (int, float)) and not np.isinf(v):
                        mlflow.log_metric(f"oos_mlp_{k}", v)
            if results.get("fixed_15"):
                for k, v in results["fixed_15"].items():
                    if isinstance(v, (int, float)) and not np.isinf(v):
                        mlflow.log_metric(f"oos_fixed15_{k}", v)
            if oos_df is not None and len(oos_df) > 0:
                mlflow.log_metric("oos_accuracy", oos_df["correct"].mean())
            mlflow.end_run()
            print("MLflow run completed.")
        except Exception as e:
            print(f"MLflow finalization error: {e}")

    print("\n" + "=" * 70)
    print("BPS WIDTH OPTIMIZER COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
