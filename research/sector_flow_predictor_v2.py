"""
Sector Rotation Timing Predictor — MLP on untapped sector flow features.

Predicts 21-day forward sector ETF returns using:
  - Sector flow features (dollar volume z-scores, relative strength, macro correlations)
  - Cross-sector flow dispersion
  - Momentum acceleration (not level)
  - Regime features (yield curve, VIX term structure, fed funds)
  - Theme features (AI/semis, energy, financials flow momentum)

Walk-forward: 252d train, 21d OOS, sliding window (HC #0: NEVER expanding).
MLflow logging mandatory.
HC #428 R1: regime-agnostic validation (|Sharpe_green - Sharpe_red| / max <= 0.50).
HC #670: anti-concentration check.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path("/home/nick/Lvl3Quant") if Path("/home/nick/Lvl3Quant").exists() else Path("/home/jupiter/Lvl3Quant")
DATA = ROOT / "data"
FEATURE_STORE = DATA / "feature_store" / "v2"
PRICE_PATH = ROOT / "wheel_strategy_v1" / "data" / "cache" / "prices_v2.parquet"
MACRO_EXTRA_PATH = ROOT / "wheel_strategy_v1" / "data" / "cache" / "macro_extra.parquet"
SECTOR_ROTATION_PATH = DATA / "feature_store" / "sector_rotation" / "daily.parquet"
OUTPUT_DIR = ROOT / "output" / "sector_flow_predictor"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLY", "XLP", "XLU", "XLI", "XLV", "XLB", "XLC", "XLRE"]
BENCHMARK = "SPY"

# Walk-forward config
TRAIN_DAYS = 252
OOS_DAYS = 21
FWD_DAYS = 21  # predict 21-day forward return
MIN_TRAIN_SAMPLES = 200

# Model config
HIDDEN_DIMS = [128, 64, 32]
DROPOUT = 0.3
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 50
BATCH_SIZE = 256
PATIENCE = 8

# Cost
TXN_COST_BPS = 5.0


# ============================================================================
# DATA LOADING
# ============================================================================

def load_prices() -> pd.DataFrame:
    """Load sector ETF + SPY daily prices."""
    df = pd.read_parquet(PRICE_PATH)
    universe = SECTOR_ETFS + [BENCHMARK]
    df = df[df["ticker"].isin(universe)].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


def compute_forward_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute 21-day forward returns for each sector ETF."""
    records = []
    for etf in SECTOR_ETFS:
        sub = prices[prices["ticker"] == etf][["date", "close"]].copy()
        sub = sub.sort_values("date").reset_index(drop=True)
        sub[f"fwd_ret_{FWD_DAYS}d"] = sub["close"].pct_change(FWD_DAYS).shift(-FWD_DAYS)
        sub["ticker"] = etf
        records.append(sub[["ticker", "date", f"fwd_ret_{FWD_DAYS}d"]])
    return pd.concat(records, ignore_index=True)


def compute_spy_regime(prices: pd.DataFrame) -> pd.DataFrame:
    """Classify each day as green/red/flat based on SPY close-to-close."""
    spy = prices[prices["ticker"] == BENCHMARK][["date", "close"]].copy()
    spy = spy.sort_values("date").reset_index(drop=True)
    spy["spy_ret_21d"] = spy["close"].pct_change(21).shift(-21)
    spy["regime"] = "flat"
    spy.loc[spy["spy_ret_21d"] > 0.01, "regime"] = "green"
    spy.loc[spy["spy_ret_21d"] < -0.01, "regime"] = "red"
    return spy[["date", "regime", "spy_ret_21d"]]


def load_sector_flow_features() -> pd.DataFrame:
    """Load sector flow z-score features (dollar volume, RS, macro correlations)."""
    df = pd.read_parquet(FEATURE_STORE / "sector_flow_features.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_flow_features() -> pd.DataFrame:
    """Load sector AUM/return features."""
    df = pd.read_parquet(FEATURE_STORE / "flow_features.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_regime_features() -> pd.DataFrame:
    """Load macro regime features (yield curve, VIX, fed funds)."""
    df = pd.read_parquet(FEATURE_STORE / "regime_features.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_theme_features() -> pd.DataFrame:
    """Load theme basket features (AI/semis, energy, healthcare inflows)."""
    df = pd.read_parquet(FEATURE_STORE / "theme_features.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_factor_features() -> pd.DataFrame:
    """Load factor exposure features (momentum, quality, value, lowvol)."""
    df = pd.read_parquet(FEATURE_STORE / "factor_features.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_sector_rotation_features() -> pd.DataFrame:
    """Load sector rotation features (relative strength, momentum cross, lead-lag)."""
    if SECTOR_ROTATION_PATH.exists():
        df = pd.read_parquet(SECTOR_ROTATION_PATH)
        df["date"] = pd.to_datetime(df["date"])
        df = df.rename(columns={"etf": "ticker"})
        return df
    return pd.DataFrame()


def load_macro_extra() -> pd.DataFrame:
    """Load macro extra features (yield curve, employment, inflation)."""
    if MACRO_EXTRA_PATH.exists():
        df = pd.read_parquet(MACRO_EXTRA_PATH)
        df["date"] = pd.to_datetime(df["date"])
        return df
    return pd.DataFrame()


def compute_cross_sector_features(panel: pd.DataFrame, flow_cols: list[str]) -> pd.DataFrame:
    """Compute cross-sector dispersion and rank features per date (vectorized)."""
    result = pd.DataFrame({"date": panel["date"].unique()})
    for col in flow_cols:
        grp = panel.groupby("date")[col]
        result = result.merge(
            grp.std().rename(f"xs_disp_{col}").reset_index(),
            on="date", how="left",
        )
        result = result.merge(
            (grp.max() - grp.min()).rename(f"xs_range_{col}").reset_index(),
            on="date", how="left",
        )
    return result


def compute_momentum_acceleration(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute momentum acceleration (change in momentum, not level)."""
    records = []
    for etf in SECTOR_ETFS:
        sub = prices[prices["ticker"] == etf][["date", "close"]].copy()
        sub = sub.sort_values("date").reset_index(drop=True)
        sub["mom_20d"] = sub["close"].pct_change(20)
        sub["mom_60d"] = sub["close"].pct_change(60)
        sub["mom_accel_20d"] = sub["mom_20d"].diff(5)  # 5-day change in 20d momentum
        sub["mom_accel_60d"] = sub["mom_60d"].diff(5)
        sub["mom_ratio"] = sub["mom_20d"] / sub["mom_60d"].replace(0, np.nan)
        sub["ticker"] = etf
        records.append(sub[["ticker", "date", "mom_accel_20d", "mom_accel_60d", "mom_ratio"]])
    return pd.concat(records, ignore_index=True)


# ============================================================================
# PANEL CONSTRUCTION
# ============================================================================

def build_panel() -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """
    Build the master panel: merge all feature sources + forward returns.
    Returns (panel, feature_columns, regime_df).
    """
    print("Loading data sources...")
    prices = load_prices()
    print(f"  Prices: {prices.shape}")

    # Forward returns (target)
    fwd = compute_forward_returns(prices)
    print(f"  Forward returns: {fwd.shape}")

    # Regime
    regime_df = compute_spy_regime(prices)

    # Sector flow features -- individual stock level, need to aggregate to sector ETF
    sf = load_sector_flow_features()
    fl_raw = load_flow_features()  # has sector_etf mapping
    # Map individual tickers to sector ETFs via flow_features
    ticker_sector = fl_raw[["ticker", "sector_etf"]].drop_duplicates()
    sf = sf.merge(ticker_sector, on="ticker", how="left")
    sf = sf.dropna(subset=["sector_etf"])
    sf_feature_cols = [c for c in sf.columns if c not in ["ticker", "date", "sector_etf"]]
    # Aggregate: mean z-score per sector per date
    sf_agg = sf.groupby(["sector_etf", "date"])[sf_feature_cols].mean().reset_index()
    sf_agg = sf_agg.rename(columns={"sector_etf": "ticker"})
    print(f"  Sector flow features (aggregated to sector): {sf_agg.shape}")

    # Flow features -- also aggregate to sector ETF level
    fl = fl_raw.copy()
    fl_feat_cols = [c for c in fl.columns if c not in ["ticker", "date", "sector_etf"]]
    fl_agg = fl.groupby(["sector_etf", "date"])[fl_feat_cols].mean().reset_index()
    fl_agg = fl_agg.rename(columns={"sector_etf": "ticker"})
    print(f"  Flow features (aggregated to sector): {fl_agg.shape}")

    # Regime features (date-level, no ticker)
    reg = load_regime_features()
    print(f"  Regime features: {reg.shape}")

    # Theme features -- aggregate to sector ETF level
    theme = load_theme_features()
    theme = theme.merge(ticker_sector, on="ticker", how="left")
    theme = theme.dropna(subset=["sector_etf"])
    theme_feat_cols = [c for c in theme.columns if c not in ["ticker", "date", "sector_etf"]
                       and theme[c].dtype in ("float64", "float32", "int64", "int32")]
    theme_agg = theme.groupby(["sector_etf", "date"])[theme_feat_cols].mean().reset_index()
    theme_agg = theme_agg.rename(columns={"sector_etf": "ticker"})
    print(f"  Theme features (aggregated to sector): {theme_agg.shape}")

    # Factor features -- aggregate to sector ETF level
    factor = load_factor_features()
    factor = factor.merge(ticker_sector, on="ticker", how="left")
    factor = factor.dropna(subset=["sector_etf"])
    factor_feat_raw = [c for c in factor.columns if c not in ["ticker", "date", "sector_etf"]]
    factor_agg = factor.groupby(["sector_etf", "date"])[factor_feat_raw].mean().reset_index()
    factor_agg = factor_agg.rename(columns={"sector_etf": "ticker"})
    print(f"  Factor features (aggregated to sector): {factor_agg.shape}")

    # Sector rotation features
    sr = load_sector_rotation_features()
    if not sr.empty:
        sr = sr[sr["ticker"].isin(SECTOR_ETFS)]
        print(f"  Sector rotation features: {sr.shape}")

    # Momentum acceleration
    mom_accel = compute_momentum_acceleration(prices)
    print(f"  Momentum acceleration: {mom_accel.shape}")

    # Macro extra (date-level)
    macro = load_macro_extra()
    if not macro.empty:
        # Select key macro columns
        macro_cols = ["date", "yc_2s10s", "be_10y", "fed_funds", "umich_sent",
                      "jobless_claims", "unrate", "cap_util"]
        macro_cols = [c for c in macro_cols if c in macro.columns]
        macro = macro[macro_cols]
        print(f"  Macro extra: {macro.shape}")

    # === MERGE ===
    panel = fwd.copy()

    # Merge sector flow features (aggregated to sector ETF level)
    panel = panel.merge(sf_agg, on=["ticker", "date"], how="left")

    # Merge flow features (aggregated to sector ETF level)
    panel = panel.merge(fl_agg, on=["ticker", "date"], how="left")

    # Merge theme features (aggregated to sector ETF level)
    if not theme_agg.empty:
        panel = panel.merge(theme_agg, on=["ticker", "date"], how="left")

    # Merge factor features (aggregated to sector ETF level)
    if not factor_agg.empty:
        panel = panel.merge(factor_agg, on=["ticker", "date"], how="left")

    # Merge sector rotation features
    if not sr.empty:
        sr_cols = [c for c in sr.columns if c not in ["ticker", "date"]]
        panel = panel.merge(sr[["ticker", "date"] + sr_cols], on=["ticker", "date"], how="left")

    # Merge momentum acceleration
    panel = panel.merge(mom_accel, on=["ticker", "date"], how="left")

    # Merge regime features (date-level)
    panel = panel.merge(reg, on="date", how="left")

    # Merge macro extra (date-level)
    if not macro.empty:
        panel = panel.merge(macro, on="date", how="left")

    # Merge regime classification
    panel = panel.merge(regime_df[["date", "regime"]], on="date", how="left")

    # === Cross-sector dispersion features ===
    flow_z_cols = [c for c in sf_feature_cols if "_z" in c]
    if flow_z_cols:
        xs = compute_cross_sector_features(panel, flow_z_cols[:4])  # top 4 to limit dimensionality
        if not xs.empty:
            panel = panel.merge(xs, on="date", how="left")

    # Identify feature columns (everything except ticker, date, target, regime)
    exclude = {"ticker", "date", f"fwd_ret_{FWD_DAYS}d", "regime", "spy_ret_21d", "close", "sector_etf"}
    feature_cols = sorted([c for c in panel.columns if c not in exclude])

    # Drop rows where target is NaN
    panel = panel.dropna(subset=[f"fwd_ret_{FWD_DAYS}d"])

    # Fill NaN features with 0 (z-scores, so 0 = neutral)
    for c in feature_cols:
        panel[c] = panel[c].fillna(0.0)

    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)

    print(f"\nFinal panel: {panel.shape}")
    print(f"Feature columns ({len(feature_cols)}): {feature_cols[:10]}...")
    print(f"Date range: {panel['date'].min()} to {panel['date'].max()}")
    print(f"Unique tickers: {panel['ticker'].nunique()}")
    print(f"Unique dates: {panel['date'].nunique()}")

    return panel, feature_cols, regime_df


# ============================================================================
# MODEL
# ============================================================================

class SectorFlowMLP(nn.Module):
    """MLP for predicting sector forward returns."""

    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float = 0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ============================================================================
# TRAINING
# ============================================================================

def train_one_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    fold_idx: int,
) -> tuple[SectorFlowMLP, float, list[float]]:
    """Train a single fold. Returns (model, best_val_loss, val_losses)."""

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    train_ds = TensorDataset(
        torch.tensor(X_train_s, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val_s, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=8, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True)

    model = SectorFlowMLP(X_train.shape[1], HIDDEN_DIMS, DROPOUT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    val_losses = []

    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0.0
        n_batches = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        train_loss /= max(n_batches, 1)

        # Validate
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += criterion(pred, yb).item()
                n_val += 1
        val_loss /= max(n_val, 1)
        val_losses.append(val_loss)

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    if best_state:
        model.load_state_dict(best_state)

    return model, best_val_loss, val_losses


# ============================================================================
# WALK-FORWARD ENGINE
# ============================================================================

def walk_forward(panel: pd.DataFrame, feature_cols: list[str], device: torch.device):
    """
    Sliding walk-forward: 252d train, 21d OOS.
    Returns all OOS predictions with metadata.
    """
    target_col = f"fwd_ret_{FWD_DAYS}d"
    dates = sorted(panel["date"].unique())
    n_dates = len(dates)

    print(f"\nWalk-forward: {n_dates} unique dates, {TRAIN_DAYS}d train, {OOS_DAYS}d OOS")

    all_oos = []
    fold_metrics = []
    fold_idx = 0

    i = TRAIN_DAYS
    while i + OOS_DAYS <= n_dates:
        train_dates = dates[i - TRAIN_DAYS: i]
        oos_dates = dates[i: i + OOS_DAYS]

        train_mask = panel["date"].isin(train_dates)
        oos_mask = panel["date"].isin(oos_dates)

        train_data = panel[train_mask]
        oos_data = panel[oos_mask]

        if len(train_data) < MIN_TRAIN_SAMPLES or len(oos_data) < 10:
            i += OOS_DAYS
            continue

        X_train = train_data[feature_cols].values
        y_train = train_data[target_col].values
        X_oos = oos_data[feature_cols].values
        y_oos = oos_data[target_col].values

        model, best_loss, _ = train_one_fold(X_train, y_train, X_oos, y_oos, device, fold_idx)

        # Predict OOS
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_oos_s = scaler.transform(X_oos)

        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(X_oos_s, dtype=torch.float32).to(device)).cpu().numpy()

        oos_df = oos_data[["ticker", "date", target_col, "regime"]].copy()
        oos_df["pred"] = preds

        # Per-fold IC
        ic = np.corrcoef(preds, y_oos)[0, 1] if len(preds) > 2 else 0.0

        fold_metrics.append({
            "fold": fold_idx,
            "train_start": str(train_dates[0]),
            "train_end": str(train_dates[-1]),
            "oos_start": str(oos_dates[0]),
            "oos_end": str(oos_dates[-1]),
            "n_train": len(train_data),
            "n_oos": len(oos_data),
            "val_loss": best_loss,
            "oos_ic": ic,
        })

        all_oos.append(oos_df)

        # Save fold weights
        fold_path = OUTPUT_DIR / f"fold_{fold_idx:03d}.pt"
        torch.save(model.state_dict(), fold_path)

        if fold_idx % 10 == 0:
            print(f"  Fold {fold_idx}: OOS IC={ic:.4f}, loss={best_loss:.6f}, "
                  f"OOS {oos_dates[0].strftime('%Y-%m-%d')} to {oos_dates[-1].strftime('%Y-%m-%d')}")

        fold_idx += 1
        i += OOS_DAYS

    return pd.concat(all_oos, ignore_index=True), fold_metrics


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_predictions(oos: pd.DataFrame, fold_metrics: list[dict]) -> dict:
    """Comprehensive evaluation with regime symmetry and anti-concentration checks."""
    target_col = f"fwd_ret_{FWD_DAYS}d"

    # Concat IC
    concat_ic = np.corrcoef(oos["pred"].values, oos[target_col].values)[0, 1]

    # Per-date rank IC (cross-sectional)
    rank_ics = []
    for dt, grp in oos.groupby("date"):
        if len(grp) >= 5:
            from scipy.stats import spearmanr
            rho, _ = spearmanr(grp["pred"].values, grp[target_col].values)
            rank_ics.append(rho)
    mean_rank_ic = np.nanmean(rank_ics) if rank_ics else 0.0

    # === STRATEGY SIMULATION ===
    # At each rebalance date: rank sectors by predicted return, go long top-3, equal weight
    rebal_dates = sorted(oos["date"].unique())[::21]  # every 21 days
    if len(rebal_dates) < 2:
        rebal_dates = sorted(oos["date"].unique())

    strategy_returns = []
    holdings_history = []
    for rd in rebal_dates:
        snap = oos[oos["date"] == rd].dropna(subset=["pred"])
        if len(snap) < 5:
            continue
        top3 = snap.nlargest(3, "pred")["ticker"].tolist()
        avg_ret = snap[snap["ticker"].isin(top3)][target_col].mean()
        # Cost: if holdings changed
        strategy_returns.append(avg_ret - TXN_COST_BPS / 10000)
        holdings_history.append((rd, top3))

    strategy_returns = np.array(strategy_returns)
    if len(strategy_returns) < 2:
        return {"error": "insufficient OOS data"}

    # Annualize (each return is 21-day)
    periods_per_year = 252 / 21
    ann_ret = np.mean(strategy_returns) * periods_per_year
    ann_vol = np.std(strategy_returns) * np.sqrt(periods_per_year)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    # Sortino
    downside = strategy_returns[strategy_returns < 0]
    downside_vol = np.std(downside) * np.sqrt(periods_per_year) if len(downside) > 0 else 1e-6
    sortino = ann_ret / downside_vol

    # Win rate
    wr = np.mean(strategy_returns > 0)

    # Profit factor
    gains = strategy_returns[strategy_returns > 0].sum()
    losses = abs(strategy_returns[strategy_returns < 0].sum())
    pf = gains / losses if losses > 0 else float("inf")

    # === HC #428 R1: REGIME SYMMETRY ===
    regime_sharpes = {}
    for regime_label in ["green", "red", "flat"]:
        regime_dates = set(oos[oos["regime"] == regime_label]["date"].unique())
        regime_rets = []
        for rd, top3 in holdings_history:
            if rd in regime_dates:
                snap = oos[(oos["date"] == rd) & (oos["ticker"].isin(top3))]
                if not snap.empty:
                    regime_rets.append(snap[target_col].mean() - TXN_COST_BPS / 10000)
        if len(regime_rets) >= 3:
            r_arr = np.array(regime_rets)
            r_sharpe = (np.mean(r_arr) * periods_per_year) / (np.std(r_arr) * np.sqrt(periods_per_year) + 1e-8)
            regime_sharpes[regime_label] = r_sharpe

    # Regime asymmetry check
    regime_pass = True
    regime_asym = 0.0
    if "green" in regime_sharpes and "red" in regime_sharpes:
        sg, sr = regime_sharpes["green"], regime_sharpes["red"]
        denom = max(abs(sg), abs(sr), 1e-8)
        regime_asym = abs(sg - sr) / denom
        regime_pass = regime_asym <= 0.50

    # === HC #670: ANTI-CONCENTRATION ===
    from collections import Counter
    all_picks = []
    for _, picks in holdings_history:
        all_picks.extend(picks)
    pick_counts = Counter(all_picks)
    total_picks = len(all_picks)
    max_concentration = max(pick_counts.values()) / total_picks if total_picks > 0 else 0
    unique_sectors_held = len(pick_counts)
    # Check consecutive holds
    max_consecutive = 0
    for etf in SECTOR_ETFS:
        consec = 0
        for _, picks in holdings_history:
            if etf in picks:
                consec += 1
                max_consecutive = max(max_consecutive, consec)
            else:
                consec = 0
    anti_conc_pass = max_concentration <= 0.60 and unique_sectors_held >= 3

    # Per-fold IC stats
    fold_ics = [f["oos_ic"] for f in fold_metrics]

    results = {
        "concat_ic": float(concat_ic),
        "mean_rank_ic": float(mean_rank_ic),
        "mean_fold_ic": float(np.nanmean(fold_ics)),
        "std_fold_ic": float(np.nanstd(fold_ics)),
        "n_folds": len(fold_metrics),
        "n_oos_samples": len(oos),
        "strategy_ann_return": float(ann_ret),
        "strategy_ann_vol": float(ann_vol),
        "strategy_sharpe": float(sharpe),
        "strategy_sortino": float(sortino),
        "strategy_win_rate": float(wr),
        "strategy_profit_factor": float(pf),
        "n_rebalances": len(holdings_history),
        "regime_sharpes": {k: float(v) for k, v in regime_sharpes.items()},
        "regime_asymmetry": float(regime_asym),
        "regime_symmetry_pass": regime_pass,
        "max_sector_concentration": float(max_concentration),
        "unique_sectors_held": unique_sectors_held,
        "max_consecutive_holds": max_consecutive,
        "anti_concentration_pass": anti_conc_pass,
        "top_sectors": dict(pick_counts.most_common(5)),
    }

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlflow-tracking-uri", default="http://jupiter:5000")
    parser.add_argument("--experiment-name", default="sector_flow_predictor")
    parser.add_argument("--dry-run", action="store_true", help="Skip training, just build panel")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    # MLflow setup
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=f"sector_flow_mlp_{datetime.now().strftime('%Y%m%d_%H%M')}"):
        # Log params
        mlflow.log_params({
            "model_type": "MLP",
            "hidden_dims": str(HIDDEN_DIMS),
            "dropout": DROPOUT,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "patience": PATIENCE,
            "train_days": TRAIN_DAYS,
            "oos_days": OOS_DAYS,
            "fwd_days": FWD_DAYS,
            "txn_cost_bps": TXN_COST_BPS,
            "device": str(device),
            "window_type": "sliding",
        })

        # Build panel
        panel, feature_cols, regime_df = build_panel()
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("feature_names", str(feature_cols[:20]))

        if args.dry_run:
            print("\n[DRY RUN] Panel built. Exiting.")
            return

        # Walk-forward training
        oos_df, fold_metrics = walk_forward(panel, feature_cols, device)

        # Save OOS predictions
        pred_path = OUTPUT_DIR / "oos_predictions.parquet"
        oos_df.to_parquet(pred_path, index=False)
        print(f"\nSaved OOS predictions: {pred_path}")

        # Also save as .npz for compatibility
        npz_path = OUTPUT_DIR / "oos_predictions.npz"
        np.savez(
            npz_path,
            tickers=oos_df["ticker"].values,
            dates=oos_df["date"].values.astype(str),
            predictions=oos_df["pred"].values,
            actuals=oos_df[f"fwd_ret_{FWD_DAYS}d"].values,
            regimes=oos_df["regime"].values,
        )

        # Save fold metrics
        fold_path = OUTPUT_DIR / "fold_metrics.json"
        with open(fold_path, "w") as f:
            json.dump(fold_metrics, f, indent=2, default=str)

        # Evaluate
        results = evaluate_predictions(oos_df, fold_metrics)

        # Print results
        print("\n" + "=" * 70)
        print("SECTOR FLOW PREDICTOR — RESULTS")
        print("=" * 70)
        print(f"  Concat IC:           {results['concat_ic']:.4f}")
        print(f"  Mean Rank IC:        {results['mean_rank_ic']:.4f}")
        print(f"  Mean Fold IC:        {results['mean_fold_ic']:.4f} +/- {results['std_fold_ic']:.4f}")
        print(f"  N Folds:             {results['n_folds']}")
        print(f"  N OOS samples:       {results['n_oos_samples']}")
        print()
        print("  --- Strategy (top-3 equal weight, 21d rebalance) ---")
        print(f"  Ann Return:          {results['strategy_ann_return']:.2%}")
        print(f"  Ann Vol:             {results['strategy_ann_vol']:.2%}")
        print(f"  Sharpe:              {results['strategy_sharpe']:.2f}")
        print(f"  Sortino:             {results['strategy_sortino']:.2f}")
        print(f"  Win Rate:            {results['strategy_win_rate']:.1%}")
        print(f"  Profit Factor:       {results['strategy_profit_factor']:.2f}")
        print(f"  N Rebalances:        {results['n_rebalances']}")
        print()
        print("  --- Regime Symmetry (HC #428 R1) ---")
        for regime, s in results["regime_sharpes"].items():
            print(f"    {regime:6s} Sharpe: {s:.2f}")
        print(f"  Regime Asymmetry:    {results['regime_asymmetry']:.3f} (max 0.50)")
        print(f"  Regime Pass:         {'PASS' if results['regime_symmetry_pass'] else 'FAIL'}")
        print()
        print("  --- Anti-Concentration (HC #670) ---")
        print(f"  Max Concentration:   {results['max_sector_concentration']:.1%}")
        print(f"  Unique Sectors:      {results['unique_sectors_held']}")
        print(f"  Max Consec Holds:    {results['max_consecutive_holds']}")
        print(f"  Anti-Conc Pass:      {'PASS' if results['anti_concentration_pass'] else 'FAIL'}")
        print(f"  Top Sectors:         {results['top_sectors']}")

        # Log metrics to MLflow
        mlflow.log_metrics({
            "concat_ic": results["concat_ic"],
            "mean_rank_ic": results["mean_rank_ic"],
            "mean_fold_ic": results["mean_fold_ic"],
            "strategy_sharpe": results["strategy_sharpe"],
            "strategy_sortino": results["strategy_sortino"],
            "strategy_win_rate": results["strategy_win_rate"],
            "strategy_profit_factor": results["strategy_profit_factor"],
            "strategy_ann_return": results["strategy_ann_return"],
            "regime_asymmetry": results["regime_asymmetry"],
            "max_concentration": results["max_sector_concentration"],
            "n_folds": results["n_folds"],
        })

        # Log regime sharpes
        for regime, s in results["regime_sharpes"].items():
            mlflow.log_metric(f"sharpe_{regime}", s)

        # Log artifacts
        mlflow.log_artifact(str(pred_path))
        mlflow.log_artifact(str(fold_path))

        # Save results summary
        results_path = OUTPUT_DIR / "results_summary.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        mlflow.log_artifact(str(results_path))

        # Final verdict
        print("\n" + "=" * 70)
        if results["strategy_sharpe"] > 1.5 and results["regime_symmetry_pass"] and results["anti_concentration_pass"]:
            print("VERDICT: PROMISING — worth integrating with existing rotation strategy")
        elif results["strategy_sharpe"] > 0.5:
            print("VERDICT: MARGINAL — may add value as confluence signal")
        else:
            print("VERDICT: WEAK — flow features alone insufficient for timing")
        print("=" * 70)


if __name__ == "__main__":
    main()
