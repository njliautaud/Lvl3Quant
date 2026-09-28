#!/usr/bin/env python3
"""
GPU Drawdown Predictor — LSTM + LightGBM for UPRO Crash Avoidance
==================================================================
HC #0: Sliding walk-forward (NO expanding window).
HC #428 R1: Regime-agnostic OOT (all available OOT days, regime-stratified).
HC #705: Adversarial checks (permutation, regime split, sub-period, outlier removal).

THESIS: A neural net that predicts 5-day-ahead drawdowns > 3% in SPY lets us
exit UPRO *before* the crash, cutting MaxDD from -26.8% to ≤25% while keeping
most of the 80% CAGR from the VIX-gated leveraged strategy.

MODELS:
  1. LSTM (PyTorch, GPU) — learns temporal patterns in vol/credit/momentum
  2. LightGBM baseline — tree model on same tabular features (often competitive)

VALIDATION:
  - Sliding walk-forward: 756d train (~36 months), 126d test (~6 months), slide 126d
  - Concat all OOT predictions for final metrics
  - AUC-ROC, precision/recall, F1
  - Backtest: UPRO with drawdown shield vs buy-and-hold UPRO vs VIX-threshold UPRO
  - Adversarial: 100 permutation shuffles, bull/bear regime split, sub-period consistency

Runtime: ~10-20 min on RTX 3090.
"""

import sys, os, json, warnings, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

# ── PATHS ────────────────────────────────────────────────────────
if os.path.exists("/home/nick"):
    ROOT = Path("/home/nick/Lvl3Quant")
else:
    ROOT = Path("/home/jupiter/Lvl3Quant")

OUTPUT = ROOT / "output" / "growth_research"
OUTPUT.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = OUTPUT / "drawdown_predictor_results.json"
CACHE_PATH = OUTPUT / "dd_predictor_cache.parquet"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INIT] Device: {DEVICE}")
print(f"[INIT] Output: {OUTPUT}")
if DEVICE.type == "cuda":
    print(f"[INIT] GPU: {torch.cuda.get_device_name(0)}")

# ── CONFIG ───────────────────────────────────────────────────────
SEQ_LEN = 20            # 20 trading days lookback
PRED_HORIZON = 5         # predict 5-day forward
DD_THRESHOLD = 0.03      # 3% drop = drawdown event
TRAIN_DAYS = 756         # 36 months
TEST_DAYS = 126          # 6 months
STEP_DAYS = 126          # slide 6 months
START_DATE = "2004-01-01"
PERM_ITERS = 100

# LSTM hyperparams
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.3
LR = 1e-3
EPOCHS = 50
BATCH_SIZE = 128
PATIENCE = 8  # early stopping

# ── DATA DOWNLOAD ────────────────────────────────────────────────
def download_data():
    """Download daily data for feature engineering."""
    import yfinance as yf

    if CACHE_PATH.exists():
        print("[DATA] Loading cached data...")
        return pd.read_parquet(CACHE_PATH)

    tickers = {
        "SPY": "SPY",
        "VIX": "^VIX",
        "VIX3M": "^VIX3M",
        "VIX9D": "^VIX9D",
        "TLT": "TLT",
        "HYG": "HYG",
        "GLD": "GLD",
        "UPRO": "UPRO",
        "SHV": "SHV",
    }

    print("[DATA] Downloading price history...")
    dfs = {}
    for name, ticker in tickers.items():
        try:
            df = yf.download(ticker, start=START_DATE, progress=False, auto_adjust=True)
            if len(df) > 100:
                close = df["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                dfs[name] = close
                print(f"  {name} ({ticker}): {len(df)} rows")
            else:
                print(f"  {name} ({ticker}): SKIPPED (too few rows)")
        except Exception as e:
            print(f"  {name} ({ticker}): ERROR — {e}")

    # Merge into single DataFrame
    data = pd.DataFrame(dfs)
    data = data.sort_index().ffill().dropna(subset=["SPY", "VIX"])
    print(f"[DATA] Combined: {len(data)} rows, {data.columns.tolist()}")

    data.to_parquet(CACHE_PATH)
    return data


# ── FEATURE ENGINEERING ──────────────────────────────────────────
def build_features(data):
    """All lookback-only features. No future leakage."""
    print("[FEATURES] Building features...")
    df = pd.DataFrame(index=data.index)

    spy = data["SPY"]
    vix = data["VIX"]

    # --- VIX features ---
    df["vix_level"] = vix
    df["vix_ma5"] = vix.rolling(5).mean()
    df["vix_ma10"] = vix.rolling(10).mean()
    df["vix_ma21"] = vix.rolling(21).mean()
    df["vix_pctile_252"] = vix.rolling(252).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    df["vix_zscore_63"] = (vix - vix.rolling(63).mean()) / vix.rolling(63).std()

    # VIX term structure slope (VIX3M - VIX, or proxy)
    if "VIX3M" in data.columns:
        df["vix_term_slope"] = data["VIX3M"] - vix
        df["vix_term_ratio"] = data["VIX3M"] / vix.clip(lower=1)
    else:
        # Proxy: VIX vs its 21d MA (contango ~ VIX < MA)
        df["vix_term_slope"] = vix.rolling(21).mean() - vix
        df["vix_term_ratio"] = vix.rolling(21).mean() / vix.clip(lower=1)

    # VIX9D for near-term skew
    if "VIX9D" in data.columns:
        df["vix9d_vix_ratio"] = data["VIX9D"] / vix.clip(lower=1)
    else:
        df["vix9d_vix_ratio"] = vix.rolling(5).mean() / vix.clip(lower=1)

    # --- SPY return features ---
    for w in [1, 5, 10, 21]:
        df[f"spy_ret_{w}d"] = spy.pct_change(w)

    # SPY drawdown from 252d high
    spy_hi_252 = spy.rolling(252).max()
    df["spy_dd_from_252hi"] = (spy - spy_hi_252) / spy_hi_252

    # SPY realized vol
    spy_logret = np.log(spy / spy.shift(1))
    for w in [5, 10, 21]:
        df[f"spy_rvol_{w}d"] = spy_logret.rolling(w).std() * np.sqrt(252)

    # VIX / realized vol ratio (put/call proxy)
    df["vix_rvol_ratio"] = vix / (df["spy_rvol_21d"] * 100).clip(lower=1)

    # --- Credit spread proxy ---
    if "HYG" in data.columns and "TLT" in data.columns:
        hyg_ret = data["HYG"].pct_change(5)
        tlt_ret = data["TLT"].pct_change(5)
        df["credit_spread_5d"] = hyg_ret - tlt_ret
        df["credit_spread_21d"] = data["HYG"].pct_change(21) - data["TLT"].pct_change(21)

    # --- Gold momentum (flight to safety) ---
    if "GLD" in data.columns:
        df["gld_ret_5d"] = data["GLD"].pct_change(5)
        df["gld_ret_21d"] = data["GLD"].pct_change(21)
        df["gld_spy_corr_21d"] = (
            data["GLD"].pct_change().rolling(21).corr(spy.pct_change())
        )

    # --- Breadth proxy: % of last 21d SPY was above 200d MA ---
    spy_ma200 = spy.rolling(200).mean()
    above_200 = (spy > spy_ma200).astype(float)
    df["breadth_pct_21d"] = above_200.rolling(21).mean()

    # --- Momentum MAs (binary) ---
    df["spy_above_50ma"] = (spy > spy.rolling(50).mean()).astype(float)
    df["spy_above_100ma"] = (spy > spy.rolling(100).mean()).astype(float)
    df["spy_above_200ma"] = (spy > spy_ma200).astype(float)

    # --- Rate of change in VIX ---
    df["vix_roc_5d"] = vix.pct_change(5)
    df["vix_roc_10d"] = vix.pct_change(10)

    # --- Interaction features ---
    df["vix_x_dd"] = df["vix_level"] * df["spy_dd_from_252hi"].abs()
    df["vix_x_rvol"] = df["vix_level"] * df["spy_rvol_5d"]

    # Drop any rows with NaN (from rolling windows)
    feature_cols = [c for c in df.columns]
    print(f"[FEATURES] {len(feature_cols)} features: {feature_cols}")

    return df, feature_cols


# ── TARGET ───────────────────────────────────────────────────────
def build_target(data):
    """Binary: will SPY drop > DD_THRESHOLD in the next PRED_HORIZON days?"""
    spy = data["SPY"]
    # Forward-looking min price in next PRED_HORIZON days
    fwd_min = spy.shift(-PRED_HORIZON).rolling(PRED_HORIZON).min().shift(-(PRED_HORIZON - 1))

    # More correct: for each day t, look at min(SPY[t+1]...SPY[t+PRED_HORIZON])
    fwd_min = pd.Series(np.nan, index=spy.index)
    for i in range(len(spy) - PRED_HORIZON):
        future_prices = spy.iloc[i+1 : i+1+PRED_HORIZON]
        fwd_min.iloc[i] = future_prices.min()

    drawdown = (fwd_min - spy) / spy
    target = (drawdown < -DD_THRESHOLD).astype(float)

    n_pos = target.sum()
    n_total = target.notna().sum()
    print(f"[TARGET] Drawdown > {DD_THRESHOLD*100:.0f}% in {PRED_HORIZON}d: "
          f"{int(n_pos)}/{int(n_total)} = {n_pos/n_total*100:.1f}% positive rate")
    return target


# ── DATASET ──────────────────────────────────────────────────────
class SequenceDataset(Dataset):
    def __init__(self, X_seq, y):
        self.X = torch.FloatTensor(X_seq)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_sequences(features_arr, target_arr, seq_len):
    """Create (seq_len, n_features) sequences with aligned targets."""
    X, y = [], []
    for i in range(seq_len, len(features_arr)):
        if np.isnan(target_arr[i]) or np.any(np.isnan(features_arr[i-seq_len:i])):
            continue
        X.append(features_arr[i-seq_len:i])
        y.append(target_arr[i])
    return np.array(X), np.array(y)


# ── LSTM MODEL ───────────────────────────────────────────────────
class DrawdownLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
                 dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, _ = self.lstm(x)
        # Take last timestep
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


# ── TRAINING UTILS ───────────────────────────────────────────────
def train_lstm(X_train, y_train, X_val, y_val, input_dim):
    """Train LSTM with early stopping. Returns model + val predictions."""
    # Handle class imbalance with pos_weight
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)]).to(DEVICE)

    model = DrawdownLSTM(input_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    train_ds = SequenceDataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=(DEVICE.type == 'cuda'))

    best_loss = float('inf')
    best_state = None
    patience_ctr = 0

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(yb)
        epoch_loss /= len(train_ds)

        # Validation
        model.eval()
        with torch.no_grad():
            val_X = torch.FloatTensor(X_val).to(DEVICE)
            val_logits = model(val_X)
            val_loss = criterion(val_logits, torch.FloatTensor(y_val).to(DEVICE)).item()

        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break

    # Load best and predict
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_X = torch.FloatTensor(X_val).to(DEVICE)
        val_probs = torch.sigmoid(model(val_X)).cpu().numpy()

    return model, val_probs


def train_lgbm(X_train_flat, y_train, X_val_flat, y_val, feature_names):
    """Train LightGBM on flattened features."""
    import lightgbm as lgb

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale = n_neg / max(n_pos, 1)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 20,
        "scale_pos_weight": scale,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbose": -1,
        "seed": 42,
    }

    dtrain = lgb.Dataset(X_train_flat, label=y_train)
    dval = lgb.Dataset(X_val_flat, label=y_val, reference=dtrain)

    callbacks = [lgb.early_stopping(stopping_rounds=20, verbose=False)]
    model = lgb.train(
        params, dtrain, num_boost_round=500,
        valid_sets=[dval], callbacks=callbacks
    )

    val_probs = model.predict(X_val_flat)
    return model, val_probs


# ── WALK-FORWARD ─────────────────────────────────────────────────
def run_walkforward(features_df, target_series, feature_cols):
    """Sliding walk-forward. Returns concat OOT predictions for both models."""
    print("\n" + "="*70)
    print("[WF] Starting sliding walk-forward validation")
    print(f"     Train={TRAIN_DAYS}d, Test={TEST_DAYS}d, Step={STEP_DAYS}d")
    print("="*70)

    # Align features and target
    valid_idx = features_df.dropna().index.intersection(target_series.dropna().index)
    features_aligned = features_df.loc[valid_idx]
    target_aligned = target_series.loc[valid_idx]

    feat_arr = features_aligned.values.astype(np.float32)
    tgt_arr = target_aligned.values.astype(np.float32)
    dates = features_aligned.index

    # Normalize features per-fold (fit on train only)
    input_dim = len(feature_cols)

    # Collect OOT results
    lstm_oot = []  # (date, y_true, y_prob)
    lgbm_oot = []

    n_total = len(feat_arr)
    fold = 0
    start = 0

    while start + TRAIN_DAYS + TEST_DAYS + SEQ_LEN <= n_total:
        train_end = start + TRAIN_DAYS
        test_end = min(train_end + TEST_DAYS, n_total)

        # Get raw slices
        train_feat = feat_arr[start:train_end].copy()
        test_feat = feat_arr[train_end - SEQ_LEN:test_end].copy()  # need SEQ_LEN extra for sequences
        train_tgt = tgt_arr[start:train_end]
        test_tgt = tgt_arr[train_end:test_end]
        test_dates = dates[train_end:test_end]

        # Normalize (fit on train)
        mu = np.nanmean(train_feat, axis=0)
        std = np.nanstd(train_feat, axis=0) + 1e-8
        train_feat = (train_feat - mu) / std
        test_feat_full = (test_feat - mu) / std

        # Make sequences for LSTM
        X_train_seq, y_train_seq = make_sequences(train_feat, train_tgt, SEQ_LEN)

        # For test: build sequences using the extra SEQ_LEN prefix rows
        test_feat_with_prefix = test_feat_full  # already has SEQ_LEN prefix
        test_tgt_with_prefix = tgt_arr[train_end - SEQ_LEN:test_end]
        X_test_seq, y_test_seq = make_sequences(test_feat_with_prefix, test_tgt_with_prefix, SEQ_LEN)

        if len(X_train_seq) < 50 or len(X_test_seq) < 10:
            start += STEP_DAYS
            continue

        fold += 1
        train_period = f"{dates[start].strftime('%Y-%m')}->{dates[train_end-1].strftime('%Y-%m')}"
        test_period = f"{dates[train_end].strftime('%Y-%m')}->{dates[min(test_end-1, len(dates)-1)].strftime('%Y-%m')}"
        print(f"\n[WF] Fold {fold}: Train {train_period} | Test {test_period} "
              f"(train={len(X_train_seq)}, test={len(X_test_seq)}, "
              f"pos_rate={y_train_seq.mean()*100:.1f}%)")

        # --- LSTM ---
        t0 = time.time()
        try:
            _, lstm_probs = train_lstm(X_train_seq, y_train_seq, X_test_seq, y_test_seq, input_dim)
            lstm_time = time.time() - t0

            # Align dates (sequences start at SEQ_LEN offset into test period)
            n_test = len(y_test_seq)
            fold_test_dates = test_dates[:n_test]
            for i in range(n_test):
                lstm_oot.append((fold_test_dates[i], y_test_seq[i], lstm_probs[i]))

            from sklearn.metrics import roc_auc_score
            try:
                auc = roc_auc_score(y_test_seq, lstm_probs)
            except:
                auc = 0.5
            print(f"       LSTM: AUC={auc:.3f} ({lstm_time:.1f}s)")
        except Exception as e:
            print(f"       LSTM: ERROR — {e}")
            lstm_time = 0

        # --- LightGBM (flattened features — use last day only, not sequence) ---
        t0 = time.time()
        try:
            # For LGBM: use unsequenced features (current day)
            X_train_flat = train_feat[SEQ_LEN:]  # align with sequence targets
            y_train_flat = train_tgt[SEQ_LEN:len(X_train_flat) + SEQ_LEN]

            # Test: features from test period
            X_test_flat = test_feat_full[SEQ_LEN:]
            y_test_flat = test_tgt_with_prefix[SEQ_LEN:]

            # Ensure alignment
            min_train = min(len(X_train_flat), len(y_train_flat))
            X_train_flat = X_train_flat[:min_train]
            y_train_flat = y_train_flat[:min_train]
            min_test = min(len(X_test_flat), len(y_test_flat))
            X_test_flat = X_test_flat[:min_test]
            y_test_flat = y_test_flat[:min_test]

            _, lgbm_probs = train_lgbm(X_train_flat, y_train_flat,
                                        X_test_flat, y_test_flat, feature_cols)
            lgbm_time = time.time() - t0

            fold_test_dates_lgbm = test_dates[:min_test]
            for i in range(min_test):
                lgbm_oot.append((fold_test_dates_lgbm[i], y_test_flat[i], lgbm_probs[i]))

            try:
                auc_lgbm = roc_auc_score(y_test_flat, lgbm_probs)
            except:
                auc_lgbm = 0.5
            print(f"       LGBM: AUC={auc_lgbm:.3f} ({lgbm_time:.1f}s)")
        except Exception as e:
            print(f"       LGBM: ERROR — {e}")

        start += STEP_DAYS

    print(f"\n[WF] Complete: {fold} folds, "
          f"LSTM OOT={len(lstm_oot)}, LGBM OOT={len(lgbm_oot)}")

    return lstm_oot, lgbm_oot


# ── EVALUATION ───────────────────────────────────────────────────
def evaluate_predictions(oot_list, model_name):
    """Compute classification metrics from OOT predictions."""
    from sklearn.metrics import (roc_auc_score, precision_recall_curve,
                                  average_precision_score, f1_score,
                                  precision_score, recall_score)

    if len(oot_list) < 50:
        print(f"[EVAL] {model_name}: Too few OOT predictions ({len(oot_list)})")
        return {}

    dates = [x[0] for x in oot_list]
    y_true = np.array([x[1] for x in oot_list])
    y_prob = np.array([x[2] for x in oot_list])

    results = {"model": model_name, "n_samples": len(y_true)}
    results["positive_rate"] = float(y_true.mean())

    # AUC
    try:
        results["auc_roc"] = float(roc_auc_score(y_true, y_prob))
    except:
        results["auc_roc"] = 0.5

    # Average precision
    try:
        results["avg_precision"] = float(average_precision_score(y_true, y_prob))
    except:
        results["avg_precision"] = 0.0

    # Metrics at various thresholds
    thresholds_to_test = [0.3, 0.4, 0.5, 0.6, 0.7]
    threshold_results = {}
    for thr in thresholds_to_test:
        y_pred = (y_prob >= thr).astype(int)
        n_pred_pos = y_pred.sum()
        if n_pred_pos == 0 or n_pred_pos == len(y_pred):
            continue
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        threshold_results[str(thr)] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "n_alerts": int(n_pred_pos),
            "alert_rate": round(n_pred_pos / len(y_pred), 4),
        }
    results["thresholds"] = threshold_results

    print(f"\n[EVAL] {model_name}: AUC={results['auc_roc']:.3f}, "
          f"AvgPrec={results['avg_precision']:.3f}, "
          f"Pos rate={results['positive_rate']*100:.1f}%")
    for thr, m in threshold_results.items():
        print(f"       thr={thr}: P={m['precision']:.3f} R={m['recall']:.3f} "
              f"F1={m['f1']:.3f} alerts={m['n_alerts']} ({m['alert_rate']*100:.1f}%)")

    return results


# ── BACKTEST ─────────────────────────────────────────────────────
def run_backtest(data, oot_list, model_name):
    """
    Backtest three strategies on UPRO:
      1. Buy-and-hold UPRO
      2. VIX-threshold UPRO (scale down at VIX>20, cash at VIX>30)
      3. Drawdown-aware UPRO (model predicts safe -> UPRO, else -> cash/SHV)
    """
    if len(oot_list) < 50:
        return {}

    dates_oot = [x[0] for x in oot_list]
    y_prob = np.array([x[2] for x in oot_list])

    # Use UPRO if available, else simulate 3x SPY
    if "UPRO" in data.columns:
        upro = data["UPRO"].pct_change()
    else:
        upro = data["SPY"].pct_change() * 3  # approximate

    vix = data["VIX"]

    # Create aligned daily series
    start_date = min(dates_oot)
    end_date = max(dates_oot)
    bt_dates = data.index[(data.index >= start_date) & (data.index <= end_date)]

    # Build signal series (forward-fill model predictions to daily)
    signal = pd.Series(0.0, index=bt_dates)
    prob_series = pd.Series(np.nan, index=bt_dates)
    for d, _, p in oot_list:
        if d in prob_series.index:
            prob_series.loc[d] = p
    prob_series = prob_series.ffill().fillna(0)

    # Threshold for going defensive (optimize later, use 0.5 for now)
    DD_ALERT_THRESHOLD = 0.5

    results = {}

    for strat_name, get_weight in [
        ("buy_hold_upro", lambda d: 1.0),
        ("vix_threshold_upro", lambda d: (
            1.0 if vix.get(d, 15) < 20 else
            0.5 if vix.get(d, 15) < 25 else
            0.25 if vix.get(d, 15) < 30 else
            0.0
        )),
        ("dd_aware_upro", lambda d: (
            0.0 if prob_series.get(d, 0) > DD_ALERT_THRESHOLD else 1.0
        )),
        ("ensemble_upro", lambda d: (
            0.0 if prob_series.get(d, 0) > DD_ALERT_THRESHOLD or vix.get(d, 15) > 30 else
            0.5 if prob_series.get(d, 0) > 0.3 or vix.get(d, 15) > 25 else
            1.0
        )),
    ]:
        equity = [1.0]
        for d in bt_dates:
            if d not in upro.index or pd.isna(upro.loc[d]):
                equity.append(equity[-1])
                continue
            w = get_weight(d)
            daily_ret = upro.loc[d] * w
            equity.append(equity[-1] * (1 + daily_ret))

        equity = np.array(equity[1:])
        if len(equity) < 2:
            continue

        daily_rets = np.diff(equity) / equity[:-1]
        daily_rets = daily_rets[~np.isnan(daily_rets)]

        # Compute metrics
        total_ret = equity[-1] / equity[0] - 1
        years = len(equity) / 252
        cagr = (equity[-1] / equity[0]) ** (1/max(years, 0.1)) - 1

        # Max drawdown
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak
        max_dd = dd.min()

        # Sharpe / Sortino
        ann_ret = np.mean(daily_rets) * 252
        ann_vol = np.std(daily_rets) * np.sqrt(252)
        sharpe = ann_ret / max(ann_vol, 1e-6)

        downside = daily_rets[daily_rets < 0]
        down_vol = np.std(downside) * np.sqrt(252) if len(downside) > 0 else 1e-6
        sortino = ann_ret / max(down_vol, 1e-6)

        results[strat_name] = {
            "total_return_pct": round(total_ret * 100, 1),
            "cagr_pct": round(cagr * 100, 1),
            "max_drawdown_pct": round(max_dd * 100, 1),
            "sharpe": round(sharpe, 2),
            "sortino": round(sortino, 2),
            "ann_vol_pct": round(ann_vol * 100, 1),
            "years": round(years, 1),
        }

        print(f"  {strat_name:25s}: CAGR={cagr*100:6.1f}% MaxDD={max_dd*100:6.1f}% "
              f"Sharpe={sharpe:5.2f} Sortino={sortino:5.2f}")

    return results


# ── ADVERSARIAL CHECKS (HC #705) ────────────────────────────────
def adversarial_checks(oot_list, model_name):
    """HC #705: permutation test, regime split, sub-period, outlier removal."""
    from sklearn.metrics import roc_auc_score

    if len(oot_list) < 100:
        return {"status": "too_few_samples"}

    dates = np.array([x[0] for x in oot_list])
    y_true = np.array([x[1] for x in oot_list])
    y_prob = np.array([x[2] for x in oot_list])

    results = {}

    # 1. Permutation test (100 shuffles)
    print(f"\n[ADVERSARIAL] {model_name}: Permutation test ({PERM_ITERS} shuffles)...")
    try:
        real_auc = roc_auc_score(y_true, y_prob)
    except:
        real_auc = 0.5

    perm_aucs = []
    for i in range(PERM_ITERS):
        shuffled = np.random.permutation(y_true)
        try:
            perm_aucs.append(roc_auc_score(shuffled, y_prob))
        except:
            perm_aucs.append(0.5)
    perm_aucs = np.array(perm_aucs)
    p_value = (perm_aucs >= real_auc).mean()
    results["permutation"] = {
        "real_auc": round(real_auc, 4),
        "perm_mean_auc": round(perm_aucs.mean(), 4),
        "perm_std_auc": round(perm_aucs.std(), 4),
        "p_value": round(p_value, 4),
        "significant": bool(p_value < 0.05),
    }
    print(f"  Real AUC={real_auc:.4f}, Perm mean={perm_aucs.mean():.4f}, p={p_value:.4f}")

    # 2. Regime split — bull years (SPY up) vs bear years (SPY down)
    print(f"[ADVERSARIAL] {model_name}: Regime split...")
    years = np.array([d.year for d in dates])
    unique_years = sorted(set(years))

    # Classify each year's overall market regime
    bull_mask = np.zeros(len(dates), dtype=bool)
    bear_mask = np.zeros(len(dates), dtype=bool)

    for yr in unique_years:
        yr_mask = years == yr
        yr_true = y_true[yr_mask]
        # If drawdown event rate < 10%, call it bull; if > 10%, bear
        if yr_true.mean() < 0.10:
            bull_mask |= yr_mask
        else:
            bear_mask |= yr_mask

    regime_results = {}
    for regime_name, mask in [("bull", bull_mask), ("bear", bear_mask)]:
        if mask.sum() < 30:
            continue
        try:
            auc = roc_auc_score(y_true[mask], y_prob[mask])
        except:
            auc = 0.5
        regime_results[regime_name] = {
            "auc": round(auc, 4),
            "n_samples": int(mask.sum()),
            "pos_rate": round(y_true[mask].mean(), 4),
        }
        print(f"  {regime_name}: AUC={auc:.4f}, n={mask.sum()}, "
              f"pos_rate={y_true[mask].mean()*100:.1f}%")

    if "bull" in regime_results and "bear" in regime_results:
        gap = abs(regime_results["bull"]["auc"] - regime_results["bear"]["auc"])
        max_auc = max(regime_results["bull"]["auc"], regime_results["bear"]["auc"])
        regime_gap_ratio = gap / max(max_auc, 0.5)
        regime_results["gap_ratio"] = round(regime_gap_ratio, 4)
        regime_results["r1_pass"] = bool(regime_gap_ratio < 0.50)
        print(f"  Regime gap ratio: {regime_gap_ratio:.4f} "
              f"({'PASS' if regime_results['r1_pass'] else 'FAIL'} R1 < 0.50)")

    results["regime_split"] = regime_results

    # 3. Sub-period consistency (split OOT in half)
    print(f"[ADVERSARIAL] {model_name}: Sub-period consistency...")
    mid = len(y_true) // 2
    sub_results = {}
    for name, sl in [("first_half", slice(0, mid)), ("second_half", slice(mid, None))]:
        try:
            auc = roc_auc_score(y_true[sl], y_prob[sl])
        except:
            auc = 0.5
        sub_results[name] = {"auc": round(auc, 4), "n": int(len(y_true[sl]))}
        print(f"  {name}: AUC={auc:.4f}, n={len(y_true[sl])}")
    results["sub_period"] = sub_results

    # 4. Outlier removal (remove top/bottom 5% of predictions)
    print(f"[ADVERSARIAL] {model_name}: Outlier removal...")
    p5, p95 = np.percentile(y_prob, [5, 95])
    inlier_mask = (y_prob >= p5) & (y_prob <= p95)
    if inlier_mask.sum() > 50:
        try:
            auc_inlier = roc_auc_score(y_true[inlier_mask], y_prob[inlier_mask])
        except:
            auc_inlier = 0.5
        results["outlier_removal"] = {
            "auc_with_outliers": round(real_auc, 4),
            "auc_without_outliers": round(auc_inlier, 4),
            "n_removed": int((~inlier_mask).sum()),
            "delta": round(auc_inlier - real_auc, 4),
        }
        print(f"  With outliers: AUC={real_auc:.4f}, "
              f"Without: AUC={auc_inlier:.4f}, delta={auc_inlier - real_auc:+.4f}")

    return results


# ── MAIN ─────────────────────────────────────────────────────────
def main():
    t_start = time.time()
    print("="*70)
    print("GPU DRAWDOWN PREDICTOR — LSTM + LightGBM for UPRO Crash Avoidance")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    # 1. Download data
    data = download_data()

    # 2. Build features and target
    features_df, feature_cols = build_features(data)
    target = build_target(data)

    # 3. Walk-forward
    lstm_oot, lgbm_oot = run_walkforward(features_df, target, feature_cols)

    # 4. Evaluate both models
    print("\n" + "="*70)
    print("[RESULTS] Concatenated OOT Evaluation")
    print("="*70)

    all_results = {
        "config": {
            "seq_len": SEQ_LEN,
            "pred_horizon": PRED_HORIZON,
            "dd_threshold": DD_THRESHOLD,
            "train_days": TRAIN_DAYS,
            "test_days": TEST_DAYS,
            "step_days": STEP_DAYS,
            "n_features": len(feature_cols),
            "feature_names": feature_cols,
        },
        "timestamp": datetime.now().isoformat(),
    }

    for name, oot in [("LSTM", lstm_oot), ("LightGBM", lgbm_oot)]:
        eval_results = evaluate_predictions(oot, name)
        all_results[f"{name.lower()}_metrics"] = eval_results

    # 5. Backtest
    print("\n" + "="*70)
    print("[BACKTEST] Strategy Comparison")
    print("="*70)

    for name, oot in [("LSTM", lstm_oot), ("LightGBM", lgbm_oot)]:
        print(f"\n--- {name} Drawdown Shield ---")
        bt_results = run_backtest(data, oot, name)
        all_results[f"{name.lower()}_backtest"] = bt_results

    # 6. Adversarial checks
    print("\n" + "="*70)
    print("[ADVERSARIAL] HC #705 Validation Checks")
    print("="*70)

    for name, oot in [("LSTM", lstm_oot), ("LightGBM", lgbm_oot)]:
        adv_results = adversarial_checks(oot, name)
        all_results[f"{name.lower()}_adversarial"] = adv_results

    # 7. Summary
    elapsed = time.time() - t_start
    all_results["runtime_seconds"] = round(elapsed, 1)

    print("\n" + "="*70)
    print("[SUMMARY]")
    print("="*70)

    for model_key in ["lstm", "lightgbm"]:
        metrics = all_results.get(f"{model_key}_metrics", {})
        bt = all_results.get(f"{model_key}_backtest", {})
        adv = all_results.get(f"{model_key}_adversarial", {})

        print(f"\n{model_key.upper()}:")
        print(f"  AUC-ROC: {metrics.get('auc_roc', 'N/A')}")
        print(f"  Avg Precision: {metrics.get('avg_precision', 'N/A')}")

        if bt:
            bh = bt.get("buy_hold_upro", {})
            vt = bt.get("vix_threshold_upro", {})
            dd = bt.get("dd_aware_upro", {})
            ens = bt.get("ensemble_upro", {})

            print(f"  Backtest:")
            print(f"    Buy-Hold UPRO:      CAGR={bh.get('cagr_pct','?')}% MaxDD={bh.get('max_drawdown_pct','?')}% Sharpe={bh.get('sharpe','?')}")
            print(f"    VIX-Threshold UPRO: CAGR={vt.get('cagr_pct','?')}% MaxDD={vt.get('max_drawdown_pct','?')}% Sharpe={vt.get('sharpe','?')}")
            print(f"    DD-Aware UPRO:      CAGR={dd.get('cagr_pct','?')}% MaxDD={dd.get('max_drawdown_pct','?')}% Sharpe={dd.get('sharpe','?')}")
            print(f"    Ensemble UPRO:      CAGR={ens.get('cagr_pct','?')}% MaxDD={ens.get('max_drawdown_pct','?')}% Sharpe={ens.get('sharpe','?')}")

        perm = adv.get("permutation", {})
        if perm:
            print(f"  Permutation: p={perm.get('p_value','?')} ({'SIGNIFICANT' if perm.get('significant') else 'NOT significant'})")

        regime = adv.get("regime_split", {})
        if "r1_pass" in regime:
            print(f"  Regime R1: gap={regime.get('gap_ratio','?')} ({'PASS' if regime.get('r1_pass') else 'FAIL'})")

    # Save results
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[SAVED] Results: {RESULTS_PATH}")
    print(f"[DONE] Total runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
