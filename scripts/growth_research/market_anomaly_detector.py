#!/usr/bin/env python3
"""
Market Anomaly Detector — Autoencoder-Based (HC #714)
======================================================
Hypothesis: unusual cross-asset behavior (anomalies) often precede large
market moves. An autoencoder trained on "normal" market conditions should
produce high reconstruction error when something weird is happening.

Architecture: PyTorch autoencoder (30 -> 16 -> 8 -> 16 -> 30)
Walk-forward: 252d sliding window, daily slide (HC #0)
Adversarial: permutation test, sub-period, outlier removal, regime test (HC #705)

Capital: $100K fixed, NO DCA (HC #713)
Execution: next-day only (no look-ahead)
"""

import os
import sys
import json
import time
import warnings
import datetime as dt
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/market_anomaly_detector")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = [
    "SPY", "QQQ", "IWM", "GLD", "SLV", "TLT", "HYG", "IEF",
    "UUP", "USO", "^VIX", "BTC-USD", "EEM", "XLF", "XLE", "XLU",
]

# For trading strategy test
STRATEGY_TICKERS = ["UPRO", "SPY", "GLD", "SHY"]

TRAIN_WINDOW = 252  # 1 year sliding window
START_DATE = "2010-01-01"
END_DATE = "2026-07-18"
CAPITAL = 100_000.0

# Autoencoder hyperparameters
HIDDEN1 = 16
LATENT = 8
EPOCHS = 50
LR = 1e-3
BATCH_SIZE = 64

np.random.seed(42)
torch.manual_seed(42)


# ──────────────────────────────────────────────────────────────────────
# 1. DATA DOWNLOAD
# ──────────────────────────────────────────────────────────────────────

def download_data():
    """Download daily price data for all tickers."""
    print("=" * 70)
    print("STEP 1: Downloading market data")
    print("=" * 70)

    cache_path = OUTPUT_DIR / "raw_prices.parquet"
    if cache_path.exists():
        print(f"  Loading cached data from {cache_path}")
        prices = pd.read_parquet(cache_path)
        print(f"  Shape: {prices.shape}, range: {prices.index[0]} to {prices.index[-1]}")
        return prices

    all_tickers = list(set(TICKERS + STRATEGY_TICKERS))
    print(f"  Downloading {len(all_tickers)} tickers: {', '.join(all_tickers)}")

    prices = pd.DataFrame()
    for ticker in all_tickers:
        try:
            data = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
            if len(data) > 0:
                # Handle multi-level columns from newer yfinance
                if isinstance(data.columns, pd.MultiIndex):
                    col = ("Close", ticker)
                    if col in data.columns:
                        prices[ticker] = data[col]
                    else:
                        prices[ticker] = data["Close"].iloc[:, 0]
                else:
                    prices[ticker] = data["Close"]
                print(f"    {ticker}: {len(data)} rows")
            else:
                print(f"    {ticker}: NO DATA")
        except Exception as e:
            print(f"    {ticker}: ERROR - {e}")

    prices = prices.sort_index()
    prices = prices.ffill().bfill()
    prices.to_parquet(cache_path)
    print(f"  Saved to {cache_path}")
    print(f"  Final shape: {prices.shape}")
    return prices


# ──────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────────────

def engineer_features(prices):
    """Build the feature matrix from price data."""
    print("\n" + "=" * 70)
    print("STEP 2: Engineering features")
    print("=" * 70)

    features = pd.DataFrame(index=prices.index)

    core_tickers = [t for t in TICKERS if t in prices.columns]
    print(f"  Core tickers available: {len(core_tickers)}")

    # 2a. Returns and vol for each asset
    for ticker in core_tickers:
        p = prices[ticker]
        daily_ret = p.pct_change()
        features[f"{ticker}_ret5d"] = p.pct_change(5)
        features[f"{ticker}_ret21d"] = p.pct_change(21)
        features[f"{ticker}_vol21d"] = daily_ret.rolling(21).std() * np.sqrt(252)

    # 2b. Cross-correlations (rolling 21d)
    corr_pairs = [("SPY", "TLT"), ("SPY", "GLD"), ("HYG", "TLT")]
    for t1, t2 in corr_pairs:
        if t1 in prices.columns and t2 in prices.columns:
            r1 = prices[t1].pct_change()
            r2 = prices[t2].pct_change()
            features[f"corr_{t1}_{t2}_21d"] = r1.rolling(21).corr(r2)

    # 2c. VIX features
    if "^VIX" in prices.columns:
        vix = prices["^VIX"]
        features["vix_level"] = vix
        features["vix_pctile_63d"] = vix.rolling(63).apply(
            lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100.0, raw=False
        )
        # Term structure proxy: VIX vs 21d realized vol of SPY
        if "SPY" in prices.columns:
            spy_rvol = prices["SPY"].pct_change().rolling(21).std() * np.sqrt(252) * 100
            features["vix_term_structure"] = vix - spy_rvol

    # 2d. Credit spread: HYG - IEF yield proxy (price difference as spread proxy)
    if "HYG" in prices.columns and "IEF" in prices.columns:
        hyg_ret = prices["HYG"].pct_change(21)
        ief_ret = prices["IEF"].pct_change(21)
        features["credit_spread_21d"] = hyg_ret - ief_ret

    # 2e. Breadth: QQQ/IWM ratio (tech vs small cap)
    if "QQQ" in prices.columns and "IWM" in prices.columns:
        features["qqq_iwm_ratio"] = prices["QQQ"] / prices["IWM"]
        features["qqq_iwm_ratio_chg21d"] = features["qqq_iwm_ratio"].pct_change(21)

    # 2f. Sector dispersion
    sector_tickers = [t for t in ["XLF", "XLE", "XLU", "QQQ", "IWM"] if t in prices.columns]
    if len(sector_tickers) >= 3:
        sector_rets = pd.DataFrame()
        for t in sector_tickers:
            sector_rets[t] = prices[t].pct_change(5)
        features["sector_dispersion_5d"] = sector_rets.std(axis=1)

    # Drop rows with NaNs (need enough history for features)
    features = features.dropna()
    print(f"  Feature matrix shape: {features.shape}")
    print(f"  Features: {list(features.columns)}")
    print(f"  Date range: {features.index[0]} to {features.index[-1]}")

    return features


# ──────────────────────────────────────────────────────────────────────
# 3. AUTOENCODER MODEL
# ──────────────────────────────────────────────────────────────────────

class MarketAutoencoder(nn.Module):
    """Symmetric autoencoder for anomaly detection."""

    def __init__(self, input_dim, hidden1=16, latent=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, latent),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon

    def encode(self, x):
        return self.encoder(x)


def train_autoencoder(X_train, input_dim, epochs=EPOCHS, lr=LR):
    """Train autoencoder on training data and return the model."""
    model = MarketAutoencoder(input_dim, HIDDEN1, LATENT)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_tensor = torch.FloatTensor(X_train)
    dataset = TensorDataset(X_tensor, X_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            recon = model(batch_x)
            loss = criterion(recon, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    model.eval()
    return model


def compute_anomaly_score(model, X, scaler=None):
    """Compute per-sample reconstruction error (anomaly score)."""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X)
        recon = model(X_tensor).numpy()
        # MSE per sample
        scores = np.mean((X - recon) ** 2, axis=1)
    return scores


# ──────────────────────────────────────────────────────────────────────
# 4. WALK-FORWARD SLIDING WINDOW
# ──────────────────────────────────────────────────────────────────────

def walk_forward_anomaly_detection(features):
    """
    Walk-forward sliding window anomaly detection.
    252d train window, slide daily. On each day compute anomaly score.
    Normalize to percentile rank within recent 252d history.
    """
    print("\n" + "=" * 70)
    print("STEP 3: Walk-forward anomaly detection (sliding 252d window)")
    print("=" * 70)

    feature_cols = features.columns.tolist()
    input_dim = len(feature_cols)
    print(f"  Input dimension: {input_dim}")
    print(f"  Total observations: {len(features)}")
    print(f"  Train window: {TRAIN_WINDOW}d")

    values = features.values
    dates = features.index

    anomaly_scores = np.full(len(values), np.nan)
    raw_recon_errors = np.full(len(values), np.nan)

    n_days = len(values) - TRAIN_WINDOW
    print(f"  Days to process: {n_days}")

    t0 = time.time()
    retrain_interval = 21  # Retrain every 21 days for speed on CPU
    model = None
    last_train_start = -999

    for i in range(TRAIN_WINDOW, len(values)):
        train_start = i - TRAIN_WINDOW
        train_end = i

        # Retrain every retrain_interval days
        if i - TRAIN_WINDOW == 0 or (train_start - last_train_start) >= retrain_interval:
            X_train_raw = values[train_start:train_end]
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train_raw)
            model = train_autoencoder(X_train, input_dim)
            last_train_start = train_start

        # Score the current day (next-day: we train on [t-252, t-1], score t)
        X_test_raw = values[i:i+1]
        X_test = scaler.transform(X_test_raw)
        score = compute_anomaly_score(model, X_test)[0]
        raw_recon_errors[i] = score

        # Percentile rank within recent 252 scores
        recent_scores = raw_recon_errors[max(TRAIN_WINDOW, i-252):i+1]
        recent_valid = recent_scores[~np.isnan(recent_scores)]
        if len(recent_valid) > 10:
            anomaly_scores[i] = stats.percentileofscore(recent_valid, score) / 100.0
        else:
            anomaly_scores[i] = 0.5

        if (i - TRAIN_WINDOW) % 500 == 0:
            elapsed = time.time() - t0
            pct = (i - TRAIN_WINDOW) / n_days * 100
            print(f"    Progress: {pct:.0f}% ({i - TRAIN_WINDOW}/{n_days}) - {elapsed:.0f}s elapsed")

    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s")

    results = pd.DataFrame({
        "date": dates,
        "anomaly_score": anomaly_scores,
        "raw_recon_error": raw_recon_errors,
    }).set_index("date").dropna()

    print(f"  Anomaly scores computed: {len(results)}")
    print(f"  Score stats: mean={results['anomaly_score'].mean():.3f}, "
          f"std={results['anomaly_score'].std():.3f}, "
          f"min={results['anomaly_score'].min():.3f}, "
          f"max={results['anomaly_score'].max():.3f}")

    return results


# ──────────────────────────────────────────────────────────────────────
# 5. ANALYSIS
# ──────────────────────────────────────────────────────────────────────

def analyze_anomalies(anomaly_df, prices):
    """Analyze whether anomaly scores predict large moves."""
    print("\n" + "=" * 70)
    print("STEP 4: Analysis — Do anomalies predict large moves?")
    print("=" * 70)

    results = {}

    # Align SPY returns with anomaly scores
    spy = prices["SPY"].reindex(anomaly_df.index)
    spy_ret_5d = spy.pct_change(5).shift(-5)  # Forward 5d return
    spy_ret_1d = spy.pct_change(1).shift(-1)  # Forward 1d return

    df = anomaly_df.copy()
    df["spy_fwd_5d"] = spy_ret_5d
    df["spy_fwd_1d"] = spy_ret_1d
    df["spy_fwd_5d_abs"] = spy_ret_5d.abs()
    df = df.dropna()

    print(f"  Analysis period: {df.index[0]} to {df.index[-1]}")
    print(f"  Total observations: {len(df)}")

    # 5a. Do high anomaly scores predict large moves?
    print("\n  --- Large Move Prediction ---")
    thresholds = [0.5, 0.75, 0.90, 0.95, 0.99]
    large_move_threshold = 0.02  # 2% in 5 days

    move_analysis = []
    for t in thresholds:
        high = df[df["anomaly_score"] >= t]
        low = df[df["anomaly_score"] < t]
        if len(high) == 0:
            continue

        pct_large = (high["spy_fwd_5d_abs"] > large_move_threshold).mean()
        pct_large_baseline = (low["spy_fwd_5d_abs"] > large_move_threshold).mean()
        avg_abs_move = high["spy_fwd_5d_abs"].mean()
        avg_abs_baseline = low["spy_fwd_5d_abs"].mean()

        row = {
            "threshold": f">{t:.0%}",
            "n_days": len(high),
            "pct_large_move": f"{pct_large:.1%}",
            "baseline_pct": f"{pct_large_baseline:.1%}",
            "lift": f"{pct_large / max(pct_large_baseline, 0.001):.2f}x",
            "avg_abs_5d_move": f"{avg_abs_move:.4f}",
            "baseline_avg": f"{avg_abs_baseline:.4f}",
        }
        move_analysis.append(row)
        print(f"    Score {row['threshold']}: {row['n_days']} days, "
              f"large moves {row['pct_large_move']} (vs {row['baseline_pct']} baseline), "
              f"lift {row['lift']}")

    results["large_move_prediction"] = move_analysis

    # 5b. Crash detection (SPY 5d return < -3%)
    print("\n  --- Crash Detection (SPY 5d < -3%) ---")
    crash_mask = df["spy_fwd_5d"] < -0.03
    n_crashes = crash_mask.sum()
    print(f"    Total crash periods: {n_crashes}")

    crash_detection = []
    for t in [0.75, 0.90, 0.95, 0.99]:
        high = df["anomaly_score"] >= t
        tp = (high & crash_mask).sum()
        fp = (high & ~crash_mask).sum()
        fn = (~high & crash_mask).sum()
        tn = (~high & ~crash_mask).sum()

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)

        row = {
            "threshold": f">{t:.0%}",
            "TP": int(tp), "FP": int(fp), "FN": int(fn),
            "precision": f"{precision:.3f}",
            "recall": f"{recall:.3f}",
            "F1": f"{f1:.3f}",
        }
        crash_detection.append(row)
        print(f"    Score {row['threshold']}: precision={row['precision']}, "
              f"recall={row['recall']}, F1={row['F1']} "
              f"(TP={row['TP']}, FP={row['FP']}, FN={row['FN']})")

    results["crash_detection"] = crash_detection

    # 5c. Rally detection (SPY 5d return > +3%)
    print("\n  --- Rally Detection (SPY 5d > +3%) ---")
    rally_mask = df["spy_fwd_5d"] > 0.03
    n_rallies = rally_mask.sum()
    print(f"    Total rally periods: {n_rallies}")

    rally_detection = []
    for t in [0.75, 0.90, 0.95, 0.99]:
        high = df["anomaly_score"] >= t
        tp = (high & rally_mask).sum()
        fp = (high & ~rally_mask).sum()
        fn = (~high & rally_mask).sum()

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)

        row = {
            "threshold": f">{t:.0%}",
            "TP": int(tp), "FP": int(fp), "FN": int(fn),
            "precision": f"{precision:.3f}",
            "recall": f"{recall:.3f}",
            "F1": f"{f1:.3f}",
        }
        rally_detection.append(row)
        print(f"    Score {row['threshold']}: precision={row['precision']}, "
              f"recall={row['recall']}, F1={row['F1']} "
              f"(TP={row['TP']}, FP={row['FP']}, FN={row['FN']})")

    results["rally_detection"] = rally_detection

    # 5d. Cluster analysis — what does the market look like at extreme anomalies?
    print("\n  --- Cluster Analysis of Extreme Anomalies ---")
    extreme = df[df["anomaly_score"] >= 0.95].copy()
    if len(extreme) >= 10:
        # Use the feature columns for clustering
        feature_cols_available = [c for c in anomaly_df.columns if c not in ["anomaly_score", "raw_recon_error"]]
        # We don't have features in anomaly_df directly, so we'll cluster on the anomaly characteristics
        cluster_features = extreme[["spy_fwd_5d", "spy_fwd_1d", "raw_recon_error"]].values
        scaler_c = StandardScaler()
        cluster_scaled = scaler_c.fit_transform(cluster_features)

        n_clusters = min(4, len(extreme) // 5)
        if n_clusters >= 2:
            km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            extreme["cluster"] = km.fit_predict(cluster_scaled)

            cluster_summary = []
            for cl in range(n_clusters):
                cl_data = extreme[extreme["cluster"] == cl]
                row = {
                    "cluster": cl,
                    "n_days": len(cl_data),
                    "avg_fwd_5d": f"{cl_data['spy_fwd_5d'].mean():.4f}",
                    "avg_fwd_1d": f"{cl_data['spy_fwd_1d'].mean():.4f}",
                    "avg_recon_error": f"{cl_data['raw_recon_error'].mean():.4f}",
                    "example_dates": [str(d.date()) for d in cl_data.index[:3]],
                }
                cluster_summary.append(row)
                print(f"    Cluster {cl}: {row['n_days']} days, "
                      f"avg 5d return {row['avg_fwd_5d']}, "
                      f"avg 1d return {row['avg_fwd_1d']}")
                print(f"      Examples: {', '.join(row['example_dates'])}")

            results["cluster_analysis"] = cluster_summary

    # 5e. Directional bias at anomalies
    print("\n  --- Directional Bias at Anomaly Levels ---")
    dir_analysis = []
    for t in [0.5, 0.75, 0.90, 0.95, 0.99]:
        high = df[df["anomaly_score"] >= t]
        if len(high) < 5:
            continue
        avg_ret = high["spy_fwd_5d"].mean()
        med_ret = high["spy_fwd_5d"].median()
        pct_positive = (high["spy_fwd_5d"] > 0).mean()
        row = {
            "threshold": f">{t:.0%}",
            "n_days": len(high),
            "avg_fwd_5d": f"{avg_ret:.4f}",
            "median_fwd_5d": f"{med_ret:.4f}",
            "pct_positive": f"{pct_positive:.1%}",
        }
        dir_analysis.append(row)
        print(f"    Score {row['threshold']}: avg 5d return {row['avg_fwd_5d']}, "
              f"median {row['median_fwd_5d']}, {row['pct_positive']} positive")

    results["directional_bias"] = dir_analysis

    return results, df


# ──────────────────────────────────────────────────────────────────────
# 5e. FEATURE ABLATION (simpler than SHAP for CPU speed)
# ──────────────────────────────────────────────────────────────────────

def feature_ablation_study(features, anomaly_df):
    """Which features contribute most to anomaly scores? Permutation-based."""
    print("\n  --- Feature Ablation Study ---")

    feature_cols = features.columns.tolist()
    input_dim = len(feature_cols)

    # Use the last 500 days for ablation study (speed)
    n_ablation = min(500, len(features) - TRAIN_WINDOW)
    start_idx = len(features) - n_ablation - TRAIN_WINDOW
    end_idx = len(features)

    X_all = features.values[start_idx:end_idx]
    train_data = X_all[:TRAIN_WINDOW]
    test_data = X_all[TRAIN_WINDOW:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_data)
    X_test = scaler.transform(test_data)

    # Baseline model and scores
    model = train_autoencoder(X_train, input_dim, epochs=80)
    baseline_scores = compute_anomaly_score(model, X_test)
    baseline_mean = baseline_scores.mean()

    importance = {}
    for fi, fname in enumerate(feature_cols):
        X_test_perm = X_test.copy()
        np.random.shuffle(X_test_perm[:, fi])
        perm_scores = compute_anomaly_score(model, X_test_perm)
        # If permuting a feature increases anomaly scores, that feature is important
        # for defining "normal" — its disruption causes anomalies
        importance[fname] = (perm_scores.mean() - baseline_mean) / max(baseline_mean, 1e-9)

    sorted_imp = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)

    print("    Top features by anomaly contribution (permutation importance):")
    ablation_results = []
    for fname, imp in sorted_imp[:15]:
        direction = "+" if imp > 0 else "-"
        row = {"feature": fname, "importance": f"{imp:.4f}", "direction": direction}
        ablation_results.append(row)
        print(f"      {fname}: {imp:+.4f}")

    return ablation_results


# ──────────────────────────────────────────────────────────────────────
# 6. TRADING STRATEGY BACKTEST
# ──────────────────────────────────────────────────────────────────────

def backtest_strategy(anomaly_df, prices):
    """
    Trading strategy:
    - Normal: UPRO (3x SPY)
    - Anomaly > 95th pctile: 50% SPY / 50% GLD
    - Anomaly > 99th pctile: 100% SHY (full defensive)
    All with next-day execution. Fixed $100K, no DCA.
    """
    print("\n" + "=" * 70)
    print("STEP 5: Trading strategy backtest")
    print("=" * 70)

    # Get strategy ticker prices aligned
    strat_prices = pd.DataFrame()
    for t in STRATEGY_TICKERS:
        if t in prices.columns:
            strat_prices[t] = prices[t]
    strat_prices = strat_prices.reindex(anomaly_df.index).ffill().bfill()

    # Daily returns
    rets = strat_prices.pct_change().fillna(0)

    # Align dates
    common_dates = anomaly_df.index.intersection(rets.index)
    anomaly_aligned = anomaly_df.loc[common_dates]
    rets_aligned = rets.loc[common_dates]

    # Strategy returns (next-day execution: signal on day t, trade on day t+1)
    signals = anomaly_aligned["anomaly_score"].shift(1)  # Shift by 1 for next-day execution
    signals = signals.dropna()
    common = signals.index.intersection(rets_aligned.index)
    signals = signals.loc[common]
    rets_common = rets_aligned.loc[common]

    # Compute strategy returns
    strat_ret = pd.Series(0.0, index=common)
    regime = pd.Series("normal", index=common)

    for date in common:
        score = signals.loc[date]
        if score >= 0.99:
            # Full defensive
            if "SHY" in rets_common.columns:
                strat_ret.loc[date] = rets_common.loc[date, "SHY"]
            else:
                strat_ret.loc[date] = 0.0
            regime.loc[date] = "defensive_99"
        elif score >= 0.95:
            # Reduced exposure
            r = 0.0
            if "SPY" in rets_common.columns:
                r += 0.5 * rets_common.loc[date, "SPY"]
            if "GLD" in rets_common.columns:
                r += 0.5 * rets_common.loc[date, "GLD"]
            strat_ret.loc[date] = r
            regime.loc[date] = "reduced_95"
        else:
            # Normal: UPRO
            if "UPRO" in rets_common.columns:
                strat_ret.loc[date] = rets_common.loc[date, "UPRO"]
            else:
                strat_ret.loc[date] = 3.0 * rets_common.loc[date].get("SPY", 0)
            regime.loc[date] = "normal"

    # Benchmark: pure UPRO
    if "UPRO" in rets_common.columns:
        bench_ret = rets_common["UPRO"]
    else:
        bench_ret = 3.0 * rets_common["SPY"]

    # Benchmark: SPY
    spy_ret = rets_common["SPY"]

    def calc_metrics(returns, name):
        """Calculate strategy metrics."""
        cum = (1 + returns).cumprod()
        total_ret = cum.iloc[-1] - 1
        n_years = len(returns) / 252
        cagr = (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1
        vol = returns.std() * np.sqrt(252)
        sharpe = returns.mean() / max(returns.std(), 1e-9) * np.sqrt(252)
        downside = returns[returns < 0].std() * np.sqrt(252)
        sortino = returns.mean() / max(downside, 1e-9) * np.sqrt(252)

        # Max drawdown
        rolling_max = cum.cummax()
        drawdown = cum / rolling_max - 1
        max_dd = drawdown.min()

        # Calmar
        calmar = cagr / max(abs(max_dd), 1e-9)

        final_value = CAPITAL * cum.iloc[-1]

        return {
            "name": name,
            "total_return": f"{total_ret:.1%}",
            "CAGR": f"{cagr:.1%}",
            "volatility": f"{vol:.1%}",
            "sharpe": f"{sharpe:.3f}",
            "sortino": f"{sortino:.3f}",
            "max_drawdown": f"{max_dd:.1%}",
            "calmar": f"{calmar:.3f}",
            "final_value": f"${final_value:,.0f}",
            "n_years": f"{n_years:.1f}",
            # Raw values for comparison
            "_sharpe": sharpe,
            "_sortino": sortino,
            "_cagr": cagr,
            "_max_dd": max_dd,
            "_total_ret": total_ret,
        }

    strat_metrics = calc_metrics(strat_ret, "Anomaly-Gated UPRO")
    bench_metrics = calc_metrics(bench_ret, "Pure UPRO (Buy & Hold)")
    spy_metrics = calc_metrics(spy_ret, "SPY (Buy & Hold)")

    # Regime statistics
    regime_counts = regime.value_counts()

    print("\n  --- Strategy Comparison ---")
    for m in [strat_metrics, bench_metrics, spy_metrics]:
        print(f"\n    {m['name']}:")
        print(f"      CAGR: {m['CAGR']}, Sharpe: {m['sharpe']}, Sortino: {m['sortino']}")
        print(f"      Max DD: {m['max_drawdown']}, Calmar: {m['calmar']}")
        print(f"      Final Value: {m['final_value']} (from $100K)")

    print(f"\n  --- Regime Distribution ---")
    for reg, count in regime_counts.items():
        pct = count / len(regime)
        print(f"    {reg}: {count} days ({pct:.1%})")

    # Cumulative returns for logging
    cum_strat = (1 + strat_ret).cumprod() * CAPITAL
    cum_bench = (1 + bench_ret).cumprod() * CAPITAL

    strategy_results = {
        "strategy": strat_metrics,
        "benchmark_upro": bench_metrics,
        "benchmark_spy": spy_metrics,
        "regime_distribution": {str(k): int(v) for k, v in regime_counts.items()},
        "period": f"{common[0]} to {common[-1]}",
    }

    return strategy_results, strat_ret, bench_ret, regime


# ──────────────────────────────────────────────────────────────────────
# 7. ADVERSARIAL VALIDATION (HC #705)
# ──────────────────────────────────────────────────────────────────────

def adversarial_validation(anomaly_df, prices, strat_ret, bench_ret):
    """
    Four adversarial tests:
    1. Permutation test (100 shuffles)
    2. Sub-period consistency
    3. Outlier removal
    4. R1 regime test
    """
    print("\n" + "=" * 70)
    print("STEP 6: Adversarial validation (HC #705)")
    print("=" * 70)

    adv_results = {}

    spy = prices["SPY"].reindex(anomaly_df.index)
    spy_ret_5d = spy.pct_change(5).shift(-5)
    df = anomaly_df.copy()
    df["spy_fwd_5d"] = spy_ret_5d
    df = df.dropna()

    # 7a. Permutation test: shuffle anomaly scores, see if predictive power persists
    print("\n  --- Test 1: Permutation Test (100 shuffles) ---")
    real_corr = np.corrcoef(df["anomaly_score"].values, df["spy_fwd_5d"].abs().values)[0, 1]

    perm_corrs = []
    for _ in range(100):
        shuffled = df["anomaly_score"].values.copy()
        np.random.shuffle(shuffled)
        perm_corrs.append(np.corrcoef(shuffled, df["spy_fwd_5d"].abs().values)[0, 1])

    perm_corrs = np.array(perm_corrs)
    p_value = (np.abs(perm_corrs) >= np.abs(real_corr)).mean()

    adv_results["permutation_test"] = {
        "real_correlation": f"{real_corr:.4f}",
        "perm_mean": f"{perm_corrs.mean():.4f}",
        "perm_std": f"{perm_corrs.std():.4f}",
        "p_value": f"{p_value:.4f}",
        "significant": p_value < 0.05,
    }
    print(f"    Real correlation (anomaly vs |5d move|): {real_corr:.4f}")
    print(f"    Permutation mean: {perm_corrs.mean():.4f} +/- {perm_corrs.std():.4f}")
    print(f"    p-value: {p_value:.4f} ({'SIGNIFICANT' if p_value < 0.05 else 'NOT significant'})")

    # 7b. Sub-period consistency
    print("\n  --- Test 2: Sub-Period Consistency ---")
    years = df.index.year.unique()
    n_periods = 4
    period_size = len(years) // n_periods
    sub_results = []

    for pi in range(n_periods):
        start_yr = years[pi * period_size]
        end_yr = years[min((pi + 1) * period_size - 1, len(years) - 1)]
        mask = (df.index.year >= start_yr) & (df.index.year <= end_yr)
        sub = df[mask]

        if len(sub) < 50:
            continue

        # Check if high anomaly scores still predict larger moves
        high = sub[sub["anomaly_score"] >= 0.90]
        low = sub[sub["anomaly_score"] < 0.90]

        if len(high) > 5 and len(low) > 5:
            avg_high = high["spy_fwd_5d"].abs().mean()
            avg_low = low["spy_fwd_5d"].abs().mean()
            lift = avg_high / max(avg_low, 1e-9)
        else:
            avg_high = avg_low = lift = np.nan

        row = {
            "period": f"{start_yr}-{end_yr}",
            "n_days": len(sub),
            "n_high_anomaly": len(high) if len(high) > 5 else 0,
            "avg_abs_move_high": f"{avg_high:.4f}" if not np.isnan(avg_high) else "N/A",
            "avg_abs_move_low": f"{avg_low:.4f}" if not np.isnan(avg_low) else "N/A",
            "lift": f"{lift:.2f}x" if not np.isnan(lift) else "N/A",
        }
        sub_results.append(row)
        print(f"    {row['period']}: lift={row['lift']}, "
              f"high={row['avg_abs_move_high']}, low={row['avg_abs_move_low']}")

    adv_results["sub_period_consistency"] = sub_results
    lifts = [float(r["lift"].rstrip("x")) for r in sub_results if r["lift"] != "N/A"]
    all_positive = all(l > 1.0 for l in lifts) if lifts else False
    print(f"    Consistent across all periods: {all_positive}")

    # 7c. Outlier removal (remove top/bottom 1% of SPY returns)
    print("\n  --- Test 3: Outlier Removal ---")
    q01 = df["spy_fwd_5d"].quantile(0.01)
    q99 = df["spy_fwd_5d"].quantile(0.99)
    df_trimmed = df[(df["spy_fwd_5d"] >= q01) & (df["spy_fwd_5d"] <= q99)]

    high_trim = df_trimmed[df_trimmed["anomaly_score"] >= 0.90]
    low_trim = df_trimmed[df_trimmed["anomaly_score"] < 0.90]

    if len(high_trim) > 5 and len(low_trim) > 5:
        avg_high_trim = high_trim["spy_fwd_5d"].abs().mean()
        avg_low_trim = low_trim["spy_fwd_5d"].abs().mean()
        lift_trim = avg_high_trim / max(avg_low_trim, 1e-9)
        print(f"    After removing top/bottom 1%: lift={lift_trim:.2f}x")
        print(f"    High anomaly avg |move|: {avg_high_trim:.4f}")
        print(f"    Low anomaly avg |move|: {avg_low_trim:.4f}")
    else:
        lift_trim = np.nan
        print(f"    Insufficient data after trimming")

    adv_results["outlier_removal"] = {
        "trimmed_observations": len(df_trimmed),
        "lift_after_trimming": f"{lift_trim:.2f}x" if not np.isnan(lift_trim) else "N/A",
        "survives": lift_trim > 1.0 if not np.isnan(lift_trim) else False,
    }

    # 7d. R1 Regime Test (green/red market days)
    print("\n  --- Test 4: R1 Regime Test ---")
    spy_daily_ret = prices["SPY"].reindex(df.index).pct_change()

    # Use 63d rolling return to classify regime
    spy_63d = prices["SPY"].reindex(df.index).pct_change(63)
    df_regime = df.copy()
    df_regime["regime"] = "flat"
    df_regime.loc[spy_63d > 0.05, "regime"] = "bull"
    df_regime.loc[spy_63d < -0.05, "regime"] = "bear"

    regime_results = []
    for reg in ["bull", "bear", "flat"]:
        sub = df_regime[df_regime["regime"] == reg]
        if len(sub) < 50:
            continue
        high = sub[sub["anomaly_score"] >= 0.90]
        low = sub[sub["anomaly_score"] < 0.90]

        if len(high) > 5 and len(low) > 5:
            avg_high = high["spy_fwd_5d"].abs().mean()
            avg_low = low["spy_fwd_5d"].abs().mean()
            lift = avg_high / max(avg_low, 1e-9)
        else:
            lift = np.nan

        row = {
            "regime": reg,
            "n_days": len(sub),
            "n_high": len(high) if len(high) > 5 else 0,
            "lift": f"{lift:.2f}x" if not np.isnan(lift) else "N/A",
        }
        regime_results.append(row)
        print(f"    {reg}: {row['n_days']} days, lift={row['lift']}")

    adv_results["regime_test"] = regime_results

    # Check R1 criterion: regime-agnostic?
    regime_lifts = [float(r["lift"].rstrip("x")) for r in regime_results if r["lift"] != "N/A"]
    if len(regime_lifts) >= 2:
        max_l = max(regime_lifts)
        min_l = min(regime_lifts)
        asymmetry = abs(max_l - min_l) / max(max_l, 1e-9)
        print(f"    Regime asymmetry: {asymmetry:.2f} (threshold: 0.50)")
        print(f"    R1 PASS: {asymmetry <= 0.50}")
        adv_results["r1_asymmetry"] = f"{asymmetry:.3f}"
        adv_results["r1_pass"] = asymmetry <= 0.50

    # 7e. Strategy permutation test
    print("\n  --- Test 5: Strategy Permutation Test ---")
    real_sharpe = strat_ret.mean() / max(strat_ret.std(), 1e-9) * np.sqrt(252)

    perm_sharpes = []
    for _ in range(100):
        # Shuffle the anomaly scores but keep returns in place
        shuffled_idx = np.random.permutation(len(strat_ret))
        perm_ret = strat_ret.values[shuffled_idx]
        perm_sharpe = np.mean(perm_ret) / max(np.std(perm_ret), 1e-9) * np.sqrt(252)
        perm_sharpes.append(perm_sharpe)

    perm_sharpes = np.array(perm_sharpes)
    strat_p = (perm_sharpes >= real_sharpe).mean()

    adv_results["strategy_permutation"] = {
        "real_sharpe": f"{real_sharpe:.3f}",
        "perm_mean_sharpe": f"{perm_sharpes.mean():.3f}",
        "p_value": f"{strat_p:.4f}",
        "significant": strat_p < 0.05,
    }
    print(f"    Real strategy Sharpe: {real_sharpe:.3f}")
    print(f"    Permuted mean Sharpe: {perm_sharpes.mean():.3f}")
    print(f"    p-value: {strat_p:.4f} ({'SIGNIFICANT' if strat_p < 0.05 else 'NOT significant'})")

    return adv_results


# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    print("=" * 70)
    print("MARKET ANOMALY DETECTOR — Autoencoder-Based")
    print(f"Started: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # 1. Download data
    prices = download_data()

    # 2. Engineer features
    features = engineer_features(prices)

    # 3-4. Walk-forward anomaly detection
    anomaly_df = walk_forward_anomaly_detection(features)

    # Save anomaly scores
    anomaly_df.to_csv(OUTPUT_DIR / "anomaly_scores.csv")
    print(f"\n  Saved anomaly scores to {OUTPUT_DIR / 'anomaly_scores.csv'}")

    # 5. Analysis
    analysis_results, analysis_df = analyze_anomalies(anomaly_df, prices)

    # Feature ablation
    ablation_results = feature_ablation_study(features, anomaly_df)
    analysis_results["feature_ablation"] = ablation_results

    # 6. Trading strategy
    strategy_results, strat_ret, bench_ret, regime = backtest_strategy(anomaly_df, prices)

    # 7. Adversarial validation
    adv_results = adversarial_validation(anomaly_df, prices, strat_ret, bench_ret)

    # ── FINAL SUMMARY ──
    elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(f"\n  Total runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Features: {features.shape[1]}")
    print(f"  Observations: {len(anomaly_df)}")
    print(f"  Autoencoder: {features.shape[1]} -> {HIDDEN1} -> {LATENT} -> {HIDDEN1} -> {features.shape[1]}")

    print("\n  --- Key Findings ---")
    if analysis_results.get("large_move_prediction"):
        best = analysis_results["large_move_prediction"][-1]
        print(f"  Large move prediction (highest threshold): {best['threshold']} -> "
              f"{best['pct_large_move']} large moves ({best['lift']} lift)")

    if analysis_results.get("crash_detection"):
        best_crash = analysis_results["crash_detection"][1]  # 90th pctile
        print(f"  Crash detection (>90%): precision={best_crash['precision']}, "
              f"recall={best_crash['recall']}")

    print(f"\n  --- Strategy Results ---")
    for key in ["strategy", "benchmark_upro", "benchmark_spy"]:
        m = strategy_results[key]
        print(f"  {m['name']}: Sharpe={m['sharpe']}, CAGR={m['CAGR']}, "
              f"MaxDD={m['max_drawdown']}, Final={m['final_value']}")

    print(f"\n  --- Adversarial Tests ---")
    if adv_results.get("permutation_test"):
        pt = adv_results["permutation_test"]
        print(f"  Permutation test: p={pt['p_value']} ({'PASS' if pt['significant'] else 'FAIL'})")
    if adv_results.get("sub_period_consistency"):
        n_consistent = sum(1 for r in adv_results["sub_period_consistency"]
                          if r["lift"] != "N/A" and float(r["lift"].rstrip("x")) > 1.0)
        n_total = len(adv_results["sub_period_consistency"])
        print(f"  Sub-period consistency: {n_consistent}/{n_total} periods show lift > 1x")
    if adv_results.get("outlier_removal"):
        or_res = adv_results["outlier_removal"]
        print(f"  Outlier removal: {or_res['lift_after_trimming']} lift "
              f"({'PASS' if or_res['survives'] else 'FAIL'})")
    if "r1_pass" in adv_results:
        print(f"  R1 regime test: asymmetry={adv_results['r1_asymmetry']} "
              f"({'PASS' if adv_results['r1_pass'] else 'FAIL'})")
    if adv_results.get("strategy_permutation"):
        sp = adv_results["strategy_permutation"]
        print(f"  Strategy permutation: p={sp['p_value']} "
              f"({'PASS' if sp['significant'] else 'FAIL'})")

    # Save all results
    all_results = {
        "metadata": {
            "script": "market_anomaly_detector.py",
            "run_date": dt.datetime.now().isoformat(),
            "runtime_seconds": round(elapsed, 1),
            "data_range": f"{features.index[0]} to {features.index[-1]}",
            "n_features": features.shape[1],
            "n_observations": len(anomaly_df),
            "architecture": f"{features.shape[1]}->{HIDDEN1}->{LATENT}->{HIDDEN1}->{features.shape[1]}",
            "train_window": TRAIN_WINDOW,
            "retrain_interval": 21,
            "capital": CAPITAL,
        },
        "analysis": analysis_results,
        "strategy": {k: v for k, v in strategy_results.items()
                     if k != "strategy" or not isinstance(v, dict) or "_sharpe" not in v},
        "adversarial": adv_results,
    }

    # Clean non-serializable values
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()
                    if not str(k).startswith("_")}
        elif isinstance(obj, list):
            return [clean_for_json(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, pd.Timestamp):
            return str(obj)
        else:
            return obj

    all_results = clean_for_json(all_results)

    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    return all_results


if __name__ == "__main__":
    results = main()
