#!/usr/bin/env python3
"""
ML Deep Ensemble: Neural network for optimal daily allocation across validated strategies.

Trains a small MLP to learn nonlinear interactions between:
  - v4.4 regime (VIX percentile-based UPRO/SPY/GLD switching)
  - VMR (vol mean-reversion timing)
  - ML vol targeting (realized vol -> UPRO sizing)
  - VIX spike predictor (VIX>30 within 5d)
  - Macro-scaled position sizing (credit spreads, gold momentum, etc.)

Walk-forward: 252d sliding train, 21d test, step 21d.
Full adversarial validation per HC #705/#709/#713.
Fixed $100K capital, NO DCA.

Author: Claude (automated research)
"""

import os
import sys
import warnings
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/ml_deep_ensemble")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INITIAL_CAPITAL = 100_000.0
TICKERS = ["SPY", "UPRO", "QQQ", "GLD", "TLT", "SHY", "IEF", "HYG", "UUP"]
VIX_TICKER = "^VIX"

# Walk-forward params
TRAIN_DAYS = 252
TEST_DAYS = 21
STEP_DAYS = 21

# Model params
HIDDEN1 = 128
HIDDEN2 = 64
NUM_CLASSES = 5
DROPOUT = 0.3
LR = 1e-3
EPOCHS = 30
BATCH_SIZE = 64
WEIGHT_DECAY = 1e-4

# Allocation classes (portfolio weights)
ALLOC_LABELS = [
    "100% UPRO",
    "67% UPRO / 33% SPY",
    "50% UPRO / 50% SPY",
    "100% SPY",
    "100% SHY (defensive)",
]
# Corresponding weights: [UPRO_weight, SPY_weight, SHY_weight]
ALLOC_WEIGHTS = np.array([
    [1.00, 0.00, 0.00],  # 100% UPRO
    [0.67, 0.33, 0.00],  # 67/33
    [0.50, 0.50, 0.00],  # 50/50
    [0.00, 1.00, 0.00],  # SPY only
    [0.00, 0.00, 1.00],  # SHY defensive
])

# Adversarial validation params
PERM_SHUFFLES = 100
SUBPERIOD_BLOCKS = 4
OUTLIER_TRIM_PCT = 5

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Data Download ───────────────────────────────────────────────────────────
def download_data():
    """Download all required price data via yfinance."""
    import yfinance as yf

    print("[1/8] Downloading market data (2010-2026)...")
    all_tickers = TICKERS + [VIX_TICKER]
    end_date = dt.date.today().strftime("%Y-%m-%d")

    data = yf.download(all_tickers, start="2010-01-01", end=end_date,
                       auto_adjust=True, progress=False)

    # Handle multi-level columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        closes = data["Close"].copy()
    else:
        closes = data.copy()

    # Rename ^VIX column
    if "^VIX" in closes.columns:
        closes = closes.rename(columns={"^VIX": "VIX"})

    closes = closes.dropna(how="all")
    closes = closes.ffill().bfill()

    print(f"  Data shape: {closes.shape}, range: {closes.index[0].date()} to {closes.index[-1].date()}")
    return closes


# ── Feature Engineering ─────────────────────────────────────────────────────
def compute_features(df):
    """
    Build all strategy signal features.
    Returns DataFrame of features aligned with df index.
    """
    print("[2/8] Computing strategy signal features...")
    feat = pd.DataFrame(index=df.index)

    # ── v4.4 Regime: VIX percentile ──
    vix = df["VIX"]
    feat["vix_pctile_63d"] = vix.rolling(63).apply(
        lambda x: scipy_stats.percentileofscore(x, x.iloc[-1]) / 100.0, raw=False
    )
    # Binary regime indicators
    feat["regime_risk_on"] = (feat["vix_pctile_63d"] < 0.20).astype(float)
    feat["regime_risk_off"] = (feat["vix_pctile_63d"] > 0.80).astype(float)
    feat["regime_cautious"] = 1.0 - feat["regime_risk_on"] - feat["regime_risk_off"]
    feat["vix_level"] = vix / 100.0  # normalized

    # ── VMR: Vol mean-reversion signal ──
    spy_ret = df["SPY"].pct_change()
    vol_10d = spy_ret.rolling(10).std() * np.sqrt(252)
    vol_60d = spy_ret.rolling(60).std() * np.sqrt(252)
    feat["vmr_ratio"] = vol_10d / vol_60d.replace(0, np.nan)
    feat["vmr_signal"] = (feat["vmr_ratio"] < 0.8).astype(float)  # low short-term vol = risk on
    feat["vmr_ratio_clipped"] = feat["vmr_ratio"].clip(0.3, 3.0)

    # ── Vol targeting: realized vol -> position scaling ──
    feat["realized_vol_10d"] = vol_10d
    feat["realized_vol_60d"] = vol_60d
    target_vol = 0.16  # 16% target
    feat["vol_scale_factor"] = (target_vol / vol_10d.replace(0, np.nan)).clip(0.2, 2.0)

    # ── VIX spike predictor features ──
    # Term structure proxy: VIX vs 63d rolling mean of VIX (proxy for 3m VIX)
    vix_3m_proxy = vix.rolling(63).mean()
    feat["vix_term_structure"] = (vix / vix_3m_proxy.replace(0, np.nan)) - 1.0
    feat["vix_term_contango"] = (feat["vix_term_structure"] < 0).astype(float)

    # Credit spread proxy: HYG - IEF return spread
    hyg_ret = df["HYG"].pct_change()
    ief_ret = df["IEF"].pct_change()
    credit_spread_ret = hyg_ret - ief_ret
    feat["credit_spread_20d"] = credit_spread_ret.rolling(20).sum()
    feat["credit_spread_zscore"] = (
        (feat["credit_spread_20d"] - feat["credit_spread_20d"].rolling(252).mean())
        / feat["credit_spread_20d"].rolling(252).std().replace(0, np.nan)
    ).clip(-3, 3)

    # SPY drawdown from recent high
    spy_high_63d = df["SPY"].rolling(63).max()
    feat["spy_drawdown"] = (df["SPY"] / spy_high_63d.replace(0, np.nan)) - 1.0

    # Combined spike risk score
    feat["spike_risk"] = (
        feat["vix_term_structure"].clip(-0.5, 0.5) * 0.3
        + (-feat["credit_spread_zscore"]) * 0.3
        + (-feat["spy_drawdown"]) * 0.4
    )

    # ── Macro features ──
    # Gold momentum (20d)
    gld_ret = df["GLD"].pct_change()
    feat["gold_mom_20d"] = gld_ret.rolling(20).sum()

    # Dollar momentum (20d)
    uup_ret = df["UUP"].pct_change()
    feat["dollar_mom_20d"] = uup_ret.rolling(20).sum()

    # Bond trend: TLT 50d SMA signal
    tlt_sma50 = df["TLT"].rolling(50).mean()
    feat["tlt_above_sma50"] = (df["TLT"] > tlt_sma50).astype(float)
    feat["tlt_sma_dist"] = (df["TLT"] / tlt_sma50.replace(0, np.nan)) - 1.0

    # ── Additional cross-asset features ──
    # SPY momentum
    feat["spy_mom_20d"] = spy_ret.rolling(20).sum()
    feat["spy_mom_5d"] = spy_ret.rolling(5).sum()

    # QQQ relative strength
    qqq_ret = df["QQQ"].pct_change()
    feat["qqq_rel_strength_20d"] = qqq_ret.rolling(20).sum() - spy_ret.rolling(20).sum()

    # VIX rate of change
    feat["vix_roc_5d"] = vix.pct_change(5)

    print(f"  Built {len(feat.columns)} features")
    return feat


# ── Label Construction ──────────────────────────────────────────────────────
def compute_labels(df):
    """
    For each day, determine which allocation would have been best
    over the NEXT 5 trading days (highest Sharpe).
    """
    print("[3/8] Computing optimal allocation labels (5-day forward Sharpe)...")

    upro_ret = df["UPRO"].pct_change()
    spy_ret = df["SPY"].pct_change()
    shy_ret = df["SHY"].pct_change()

    returns_matrix = pd.DataFrame({
        "UPRO": upro_ret,
        "SPY": spy_ret,
        "SHY": shy_ret,
    })

    n = len(df)
    labels = np.full(n, -1, dtype=int)

    for i in range(n - 5):
        fwd_rets = returns_matrix.iloc[i + 1: i + 6]  # next 5 days
        if len(fwd_rets) < 5 or fwd_rets.isnull().any().any():
            continue

        best_sharpe = -np.inf
        best_alloc = 0

        for j, weights in enumerate(ALLOC_WEIGHTS):
            port_ret = (fwd_rets.values * weights).sum(axis=1)
            mean_r = port_ret.mean()
            std_r = port_ret.std()
            sharpe = mean_r / std_r if std_r > 1e-10 else 0.0
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_alloc = j

        labels[i] = best_alloc

    labels_series = pd.Series(labels, index=df.index)
    valid = labels_series >= 0
    print(f"  Valid labels: {valid.sum()} / {n}")

    # Distribution
    for j in range(NUM_CLASSES):
        count = (labels_series[valid] == j).sum()
        pct = count / valid.sum() * 100
        print(f"    Class {j} ({ALLOC_LABELS[j]}): {count} ({pct:.1f}%)")

    return labels_series


# ── Model Definition ────────────────────────────────────────────────────────
class AllocationMLP(nn.Module):
    """Small 2-layer MLP for allocation classification."""

    def __init__(self, input_dim, hidden1=HIDDEN1, hidden2=HIDDEN2,
                 num_classes=NUM_CLASSES, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ── Walk-Forward Engine ─────────────────────────────────────────────────────
def walk_forward(features, labels, df):
    """
    Sliding walk-forward: 252d train, 21d test, step 21d.
    Returns DataFrame of daily predictions with portfolio returns.
    """
    print("[4/8] Running walk-forward validation...")
    print(f"  Train window: {TRAIN_DAYS}d, Test window: {TEST_DAYS}d, Step: {STEP_DAYS}d")

    # Align valid data
    valid_mask = labels >= 0
    feat_cols = features.columns.tolist()
    combined = pd.concat([features, labels.rename("label")], axis=1)
    combined = combined[valid_mask].dropna()

    X_all = combined[feat_cols].values.astype(np.float32)
    y_all = combined["label"].values.astype(np.int64)
    dates = combined.index

    n = len(X_all)
    total_folds = (n - TRAIN_DAYS - TEST_DAYS) // STEP_DAYS + 1
    print(f"  Total valid samples: {n}, estimated folds: {total_folds}")
    sys.stdout.flush()

    # Normalize features: compute rolling stats within train window
    results = []
    fold = 0

    start = 0
    while start + TRAIN_DAYS + TEST_DAYS <= n:
        train_end = start + TRAIN_DAYS
        test_end = min(train_end + TEST_DAYS, n)

        X_train = X_all[start:train_end].copy()
        y_train = y_all[start:train_end].copy()
        X_test = X_all[train_end:test_end].copy()
        y_test = y_all[train_end:test_end].copy()
        test_dates = dates[train_end:test_end]

        # Normalize using train stats
        mu = X_train.mean(axis=0)
        sigma = X_train.std(axis=0)
        sigma[sigma < 1e-8] = 1.0
        X_train = (X_train - mu) / sigma
        X_test = (X_test - mu) / sigma

        # Replace any remaining NaN/inf
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        # Train model
        model = AllocationMLP(input_dim=X_train.shape[1]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        # Class weights for imbalanced labels
        class_counts = np.bincount(y_train, minlength=NUM_CLASSES).astype(np.float32)
        class_counts[class_counts == 0] = 1.0
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum() * NUM_CLASSES
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, device=device)
        )

        train_ds = TensorDataset(
            torch.tensor(X_train, device=device),
            torch.tensor(y_train, device=device),
        )
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

        model.train()
        for epoch in range(EPOCHS):
            for xb, yb in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()

        # Predict on test
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(X_test, device=device))
            preds = logits.argmax(dim=1).cpu().numpy()
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        # Record predictions
        for i in range(len(test_dates)):
            results.append({
                "date": test_dates[i],
                "pred": preds[i],
                "actual": y_test[i],
                "fold": fold,
                "prob_max": probs[i].max(),
            })

        fold += 1
        start += STEP_DAYS

        if fold % 5 == 0:
            print(f"    Fold {fold}/{total_folds}: trained on {dates[start-STEP_DAYS].date()}-{dates[train_end-1].date()}, "
                  f"tested {test_dates[0].date()}-{test_dates[-1].date()}")
            sys.stdout.flush()

    print(f"  Completed {fold} walk-forward folds")
    results_df = pd.DataFrame(results).set_index("date")
    return results_df


# ── Backtesting ─────────────────────────────────────────────────────────────
def backtest(predictions, df):
    """
    Backtest ensemble predictions vs individual strategies.
    Fixed $100K, NO DCA (HC #713).
    """
    print("[5/8] Backtesting ensemble vs individual strategies...")

    upro_ret = df["UPRO"].pct_change()
    spy_ret = df["SPY"].pct_change()
    shy_ret = df["SHY"].pct_change()

    # Align
    common_dates = predictions.index.intersection(upro_ret.dropna().index)
    pred_aligned = predictions.loc[common_dates]

    # Ensemble portfolio return
    ensemble_rets = []
    for date in common_dates:
        alloc_idx = pred_aligned.loc[date, "pred"]
        w = ALLOC_WEIGHTS[alloc_idx]
        r = w[0] * upro_ret.loc[date] + w[1] * spy_ret.loc[date] + w[2] * shy_ret.loc[date]
        ensemble_rets.append(r)

    ensemble_rets = pd.Series(ensemble_rets, index=common_dates, name="Ensemble")

    # Individual strategies
    strategies = {}

    # v4.4 regime: use VIX percentile to pick allocation each day
    vix = df["VIX"]
    vix_pctile = vix.rolling(63).apply(
        lambda x: scipy_stats.percentileofscore(x, x.iloc[-1]) / 100.0, raw=False
    )
    v44_rets = []
    for date in common_dates:
        pctile = vix_pctile.get(date, 0.5)
        if pd.isna(pctile):
            pctile = 0.5
        if pctile < 0.20:  # risk on
            r = upro_ret.loc[date]
        elif pctile > 0.80:  # risk off
            r = 0.5 * shy_ret.loc[date] + 0.5 * spy_ret.loc[date]  # GLD proxy
        else:
            r = spy_ret.loc[date]
        v44_rets.append(r)
    strategies["v4.4 Regime"] = pd.Series(v44_rets, index=common_dates)

    # VMR strategy
    vol_10d = spy_ret.rolling(10).std() * np.sqrt(252)
    vol_60d = spy_ret.rolling(60).std() * np.sqrt(252)
    vmr = vol_10d / vol_60d.replace(0, np.nan)
    vmr_rets = []
    for date in common_dates:
        ratio = vmr.get(date, 1.0)
        if pd.isna(ratio):
            ratio = 1.0
        if ratio < 0.8:
            r = upro_ret.loc[date]  # low vol regime -> risk on
        elif ratio > 1.3:
            r = shy_ret.loc[date]  # high vol regime -> defensive
        else:
            r = spy_ret.loc[date]
        vmr_rets.append(r)
    strategies["VMR"] = pd.Series(vmr_rets, index=common_dates)

    # Vol targeting
    target_vol = 0.16
    vt_rets = []
    for date in common_dates:
        rv = vol_10d.get(date, 0.16)
        if pd.isna(rv) or rv < 0.01:
            rv = 0.16
        scale = min(target_vol / rv, 2.0)
        # Scale between UPRO and SHY
        upro_w = min(scale / 3.0, 1.0)  # UPRO is ~3x leveraged
        r = upro_w * upro_ret.loc[date] + (1 - upro_w) * spy_ret.loc[date]
        vt_rets.append(r)
    strategies["Vol Target"] = pd.Series(vt_rets, index=common_dates)

    # Buy and hold benchmarks
    strategies["Buy & Hold SPY"] = spy_ret.loc[common_dates]
    strategies["Buy & Hold UPRO"] = upro_ret.loc[common_dates]
    strategies["Buy & Hold 60/40 SPY/SHY"] = (
        0.6 * spy_ret.loc[common_dates] + 0.4 * shy_ret.loc[common_dates]
    )

    # Oracle (perfect foresight)
    oracle_rets = []
    for date in common_dates:
        alloc_idx = pred_aligned.loc[date, "actual"]
        w = ALLOC_WEIGHTS[alloc_idx]
        r = w[0] * upro_ret.loc[date] + w[1] * spy_ret.loc[date] + w[2] * shy_ret.loc[date]
        oracle_rets.append(r)
    strategies["Oracle (Perfect)"] = pd.Series(oracle_rets, index=common_dates)

    strategies["ML Ensemble"] = ensemble_rets

    return strategies


def compute_metrics(returns_series, name="Strategy"):
    """Compute risk-adjusted metrics for a return series."""
    rets = returns_series.dropna()
    if len(rets) < 20:
        return None

    ann_ret = rets.mean() * 252
    ann_vol = rets.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 1e-10 else 0.0

    downside = rets[rets < 0].std() * np.sqrt(252)
    sortino = ann_ret / downside if downside > 1e-10 else 0.0

    # Max drawdown
    cum = (1 + rets).cumprod()
    peak = cum.cummax()
    dd = (cum / peak - 1)
    max_dd = dd.min()

    # Profit factor
    gross_profit = rets[rets > 0].sum()
    gross_loss = abs(rets[rets < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 1e-10 else np.inf

    # Win rate (days)
    wr = (rets > 0).mean()

    # CAGR
    n_years = len(rets) / 252
    total_ret = cum.iloc[-1] if len(cum) > 0 else 1.0
    cagr = total_ret ** (1 / n_years) - 1 if n_years > 0 else 0.0

    # Final equity
    final_equity = INITIAL_CAPITAL * total_ret

    return {
        "Strategy": name,
        "CAGR": f"{cagr:.1%}",
        "Ann Vol": f"{ann_vol:.1%}",
        "Sharpe": f"{sharpe:.2f}",
        "Sortino": f"{sortino:.2f}",
        "Max DD": f"{max_dd:.1%}",
        "Profit Factor": f"{pf:.2f}",
        "Win Rate": f"{wr:.1%}",
        "Final Equity": f"${final_equity:,.0f}",
        "N Days": len(rets),
        "_sharpe_raw": sharpe,
        "_returns": rets,
    }


# ── Adversarial Validation (HC #705) ───────────────────────────────────────
def adversarial_validation(ensemble_rets, spy_rets):
    """
    Full adversarial validation per HC #705:
    1. Permutation test (100 shuffles)
    2. Sub-period consistency (4 blocks, CV of Sharpe < 0.5)
    3. Outlier robustness (trim top/bottom 5%)
    4. R1 regime check (HC #709 nuance)
    """
    print("[7/8] Running adversarial validation (HC #705)...")
    results = {}
    rets = ensemble_rets.dropna()
    n = len(rets)

    # ── 1. Permutation test ──
    print("  [A] Permutation test (100 shuffles)...")
    actual_sharpe = rets.mean() / rets.std() * np.sqrt(252)
    perm_sharpes = []
    for _ in range(PERM_SHUFFLES):
        shuffled = rets.sample(frac=1.0, replace=False).values
        s = shuffled.mean() / shuffled.std() * np.sqrt(252)
        perm_sharpes.append(s)
    perm_sharpes = np.array(perm_sharpes)
    p_value = (perm_sharpes >= actual_sharpe).mean()
    results["perm_test_pvalue"] = p_value
    results["perm_test_pass"] = p_value < 0.05
    print(f"    Actual Sharpe: {actual_sharpe:.3f}, p-value: {p_value:.3f} "
          f"({'PASS' if p_value < 0.05 else 'FAIL'})")

    # ── 2. Sub-period consistency ──
    print("  [B] Sub-period consistency (4 blocks)...")
    block_size = n // SUBPERIOD_BLOCKS
    block_sharpes = []
    for b in range(SUBPERIOD_BLOCKS):
        start = b * block_size
        end = start + block_size if b < SUBPERIOD_BLOCKS - 1 else n
        block_rets = rets.iloc[start:end]
        if len(block_rets) > 10:
            bs = block_rets.mean() / block_rets.std() * np.sqrt(252)
            block_sharpes.append(bs)
            print(f"    Block {b+1}: Sharpe={bs:.3f} ({block_rets.index[0].date()} to {block_rets.index[-1].date()})")

    if len(block_sharpes) >= 2:
        cv = np.std(block_sharpes) / abs(np.mean(block_sharpes)) if abs(np.mean(block_sharpes)) > 1e-10 else 999
        results["subperiod_cv"] = cv
        results["subperiod_pass"] = cv < 0.50
        results["block_sharpes"] = block_sharpes
        print(f"    CV of Sharpe: {cv:.3f} ({'PASS' if cv < 0.50 else 'FAIL'}: threshold 0.50)")
    else:
        results["subperiod_pass"] = False
        results["subperiod_cv"] = np.nan

    # ── 3. Outlier robustness ──
    print("  [C] Outlier robustness (trim top/bottom 5%)...")
    lower = np.percentile(rets, OUTLIER_TRIM_PCT)
    upper = np.percentile(rets, 100 - OUTLIER_TRIM_PCT)
    trimmed = rets[(rets >= lower) & (rets <= upper)]
    trimmed_sharpe = trimmed.mean() / trimmed.std() * np.sqrt(252) if trimmed.std() > 1e-10 else 0
    sharpe_drop = 1 - (trimmed_sharpe / actual_sharpe) if actual_sharpe > 1e-10 else 0
    results["trimmed_sharpe"] = trimmed_sharpe
    results["sharpe_drop_pct"] = sharpe_drop
    results["outlier_pass"] = abs(sharpe_drop) < 0.50  # less than 50% drop
    print(f"    Full Sharpe: {actual_sharpe:.3f}, Trimmed Sharpe: {trimmed_sharpe:.3f}, "
          f"Drop: {sharpe_drop:.1%} ({'PASS' if abs(sharpe_drop) < 0.50 else 'FAIL'})")

    # ── 4. R1 Regime check (HC #709 nuance) ──
    print("  [D] R1 regime-agnostic check...")
    spy_daily = spy_rets.reindex(rets.index).dropna()
    common = rets.index.intersection(spy_daily.index)
    if len(common) > 50:
        spy_aligned = spy_daily.loc[common]
        rets_aligned = rets.loc[common]

        green_days = spy_aligned > 0
        red_days = spy_aligned < 0

        green_rets = rets_aligned[green_days]
        red_rets = rets_aligned[red_days]

        sharpe_green = (green_rets.mean() / green_rets.std() * np.sqrt(252)
                        if len(green_rets) > 10 and green_rets.std() > 1e-10 else 0)
        sharpe_red = (red_rets.mean() / red_rets.std() * np.sqrt(252)
                      if len(red_rets) > 10 and red_rets.std() > 1e-10 else 0)

        max_sharpe = max(abs(sharpe_green), abs(sharpe_red))
        regime_gap = abs(sharpe_green - sharpe_red) / max_sharpe if max_sharpe > 1e-10 else 0
        results["sharpe_green"] = sharpe_green
        results["sharpe_red"] = sharpe_red
        results["regime_gap"] = regime_gap
        results["regime_pass"] = regime_gap < 0.50
        print(f"    Sharpe green days: {sharpe_green:.3f}, red days: {sharpe_red:.3f}")
        print(f"    Regime gap: {regime_gap:.3f} ({'PASS' if regime_gap < 0.50 else 'FAIL'}: threshold 0.50)")
        print(f"    Green day count: {green_days.sum()}, Red day count: {red_days.sum()}")
    else:
        results["regime_pass"] = False
        results["regime_gap"] = np.nan

    # ── Overall ──
    all_pass = all([
        results.get("perm_test_pass", False),
        results.get("subperiod_pass", False),
        results.get("outlier_pass", False),
        results.get("regime_pass", False),
    ])
    results["all_pass"] = all_pass

    print(f"\n  ADVERSARIAL VALIDATION: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    return results


# ── Results Reporting ───────────────────────────────────────────────────────
def print_results(strategies, adv_results, predictions):
    """Print comprehensive results table."""
    print("\n" + "=" * 100)
    print("ML DEEP ENSEMBLE: RESULTS SUMMARY")
    print("=" * 100)

    # Compute metrics for all strategies
    metrics_list = []
    for name, rets in strategies.items():
        m = compute_metrics(rets, name)
        if m:
            metrics_list.append(m)

    # Print table
    print(f"\n{'Strategy':<30} {'CAGR':>8} {'Vol':>8} {'Sharpe':>8} {'Sortino':>8} "
          f"{'MaxDD':>8} {'PF':>8} {'WR':>8} {'Final$':>14}")
    print("-" * 110)
    for m in metrics_list:
        print(f"{m['Strategy']:<30} {m['CAGR']:>8} {m['Ann Vol']:>8} {m['Sharpe']:>8} "
              f"{m['Sortino']:>8} {m['Max DD']:>8} {m['Profit Factor']:>8} "
              f"{m['Win Rate']:>8} {m['Final Equity']:>14}")

    # Classification accuracy
    if predictions is not None and len(predictions) > 0:
        accuracy = (predictions["pred"] == predictions["actual"]).mean()
        print(f"\nClassification accuracy: {accuracy:.1%}")

        # Per-class accuracy
        print("\nPer-class accuracy:")
        for j in range(NUM_CLASSES):
            mask = predictions["actual"] == j
            if mask.sum() > 0:
                acc = (predictions.loc[mask, "pred"] == j).mean()
                print(f"  {ALLOC_LABELS[j]}: {acc:.1%} ({mask.sum()} samples)")

    # Adversarial summary
    print(f"\n{'='*60}")
    print("ADVERSARIAL VALIDATION SUMMARY (HC #705)")
    print(f"{'='*60}")
    tests = [
        ("Permutation Test (p<0.05)", "perm_test_pass", f"p={adv_results.get('perm_test_pvalue', 'N/A')}"),
        ("Sub-period Consistency (CV<0.50)", "subperiod_pass", f"CV={adv_results.get('subperiod_cv', 'N/A'):.3f}" if isinstance(adv_results.get('subperiod_cv'), float) else "N/A"),
        ("Outlier Robustness (<50% drop)", "outlier_pass", f"drop={adv_results.get('sharpe_drop_pct', 'N/A'):.1%}" if isinstance(adv_results.get('sharpe_drop_pct'), float) else "N/A"),
        ("Regime Agnostic (gap<0.50)", "regime_pass", f"gap={adv_results.get('regime_gap', 'N/A'):.3f}" if isinstance(adv_results.get('regime_gap'), float) else "N/A"),
    ]
    for name, key, detail in tests:
        passed = adv_results.get(key, False)
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name} -- {detail}")

    overall = adv_results.get("all_pass", False)
    print(f"\n  OVERALL: {'VALIDATED' if overall else 'NOT VALIDATED'}")

    return metrics_list


def save_results(strategies, adv_results, predictions, metrics_list):
    """Save results to output directory."""
    print("\n[8/8] Saving results...")

    # Save predictions
    if predictions is not None:
        predictions.to_csv(OUTPUT_DIR / "walk_forward_predictions.csv")

    # Save equity curves
    equity = pd.DataFrame()
    for name, rets in strategies.items():
        cum = (1 + rets.dropna()).cumprod() * INITIAL_CAPITAL
        equity[name] = cum
    equity.to_csv(OUTPUT_DIR / "equity_curves.csv")

    # Save metrics
    metrics_clean = []
    for m in metrics_list:
        m_copy = {k: v for k, v in m.items() if not k.startswith("_")}
        metrics_clean.append(m_copy)
    pd.DataFrame(metrics_clean).to_csv(OUTPUT_DIR / "strategy_metrics.csv", index=False)

    # Save adversarial results
    adv_save = {k: v for k, v in adv_results.items()
                if not isinstance(v, (pd.Series, list))}
    pd.Series(adv_save).to_csv(OUTPUT_DIR / "adversarial_validation.csv")

    print(f"  Results saved to {OUTPUT_DIR}/")


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("ML DEEP ENSEMBLE: Optimal Daily Allocation via Neural Network")
    print(f"Started: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")
    print(f"Capital: ${INITIAL_CAPITAL:,.0f} (fixed, no DCA per HC #713)")
    print("=" * 80)

    # 1. Download data
    df = download_data()

    # 2. Compute features
    features = compute_features(df)

    # 3. Compute labels
    labels = compute_labels(df)

    # 4. Walk-forward
    predictions = walk_forward(features, labels, df)

    # 5. Backtest
    strategies = backtest(predictions, df)

    # 6. Compute spy returns for adversarial validation
    spy_rets = df["SPY"].pct_change()

    # 7. Adversarial validation
    adv_results = adversarial_validation(strategies["ML Ensemble"], spy_rets)

    # 8. Print and save
    metrics_list = print_results(strategies, adv_results, predictions)
    save_results(strategies, adv_results, predictions, metrics_list)

    print(f"\nCompleted: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return adv_results.get("all_pass", False)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
