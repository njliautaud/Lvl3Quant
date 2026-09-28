#!/usr/bin/env python3
"""
ML Mean Reversion Research Script
===================================
Tests whether ML can identify short-term mean reversion opportunities in SPY/QQQ
for income-style returns. Uses sliding 252-day walk-forward validation (HC #0).

Usage:
    python3 -u scripts/growth_research/ml_mean_reversion.py

Output:
    - stdout: progress, metrics, summary
    - output/ml_mean_reversion/results.json
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import stats
import traceback

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
CAPITAL = 100_000       # Fixed capital, NO DCA (HC #713)
TRAIN_WINDOW = 252      # Sliding window size in trading days
MIN_TRAIN_DAYS = 180    # Minimum days to attempt training
RSI_OVERSOLD = 30       # Mean reversion trigger (long signal zone)
RSI_OVERBOUGHT = 70     # Mean reversion trigger (short/cash signal zone)
N_PERMUTATIONS = 100    # Permutation test iterations
OUTLIER_PCTILE = 95     # Remove top 5% of absolute return days

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/ml_mean_reversion")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = ["SPY", "QQQ", "IWM", "VIX", "TLT", "GLD", "HYG"]
START_DATE = "2010-01-01"
END_DATE   = "2026-01-01"

print("=" * 70)
print("ML MEAN REVERSION RESEARCH")
print(f"Capital: ${CAPITAL:,}  |  Walk-forward: {TRAIN_WINDOW}-day SLIDING (HC #0)")
print(f"Period: {START_DATE} → {END_DATE}")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS (with graceful failure)
# ─────────────────────────────────────────────────────────────────────────────
try:
    import yfinance as yf
    print("[OK] yfinance loaded")
except ImportError:
    raise ImportError("yfinance not installed. Run: pip install yfinance")

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
    print("[OK] lightgbm loaded")
except ImportError:
    LGBM_AVAILABLE = False
    print("[WARN] lightgbm not available — will use Logistic Regression only")

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    print("[OK] scikit-learn loaded")
except ImportError:
    raise ImportError("scikit-learn not installed. Run: pip install scikit-learn")

print()

# ─────────────────────────────────────────────────────────────────────────────
# DATA DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────
def download_data(tickers, start, end):
    """Download adjusted close + volume data for all tickers."""
    print(f"Downloading data for: {', '.join(tickers)}")
    raw = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)

    # Normalize to MultiIndex if single ticker returned
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
        volume = raw["Volume"].copy()
    else:
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})
        volume = raw[["Volume"]].rename(columns={"Volume": tickers[0]})

    close = close.dropna(how="all")
    volume = volume.dropna(how="all")

    # VIX has no volume — fill with NaN
    for tk in tickers:
        if tk not in close.columns:
            print(f"  [WARN] {tk} not in data — skipping")

    print(f"  Data shape: {close.shape[0]} days × {close.shape[1]} tickers")
    print(f"  Date range: {close.index[0].date()} → {close.index[-1].date()}")
    return close, volume


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def compute_rsi(series, period):
    """RSI using Wilder's smoothing."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_features(close, volume):
    """Construct feature matrix for SPY (primary instrument)."""
    spy = close["SPY"].copy()
    spy_vol = volume.get("SPY", pd.Series(dtype=float))

    features = pd.DataFrame(index=spy.index)

    # ── RSI (2, 5, 14) ──────────────────────────────────────────────────────
    features["rsi_2"]  = compute_rsi(spy, 2)
    features["rsi_5"]  = compute_rsi(spy, 5)
    features["rsi_14"] = compute_rsi(spy, 14)

    # ── Bollinger Band position ──────────────────────────────────────────────
    sma_20  = spy.rolling(20).mean()
    std_20  = spy.rolling(20).std()
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    features["bb_pos"] = (spy - bb_lower) / (bb_upper - bb_lower + 1e-9)

    # ── Distance from SMAs ──────────────────────────────────────────────────
    for period in [20, 50, 200]:
        sma = spy.rolling(period).mean()
        features[f"dist_sma{period}"] = (spy - sma) / sma

    # ── Volume ratio (current vs 20-day avg) ────────────────────────────────
    if len(spy_vol) > 0:
        vol_20 = spy_vol.rolling(20).mean()
        features["vol_ratio"] = spy_vol / vol_20.replace(0, np.nan)
    else:
        features["vol_ratio"] = np.nan

    # ── VIX level ───────────────────────────────────────────────────────────
    if "VIX" in close.columns:
        features["vix_level"] = close["VIX"].reindex(spy.index)

        # VIX term structure proxy: VIX vs 10-day SMA of VIX (if no VIX3M available)
        vix_sma10 = features["vix_level"].rolling(10).mean()
        features["vix_ts_proxy"] = features["vix_level"] / vix_sma10 - 1
    else:
        features["vix_level"] = np.nan
        features["vix_ts_proxy"] = np.nan

    # ── Prior day returns ────────────────────────────────────────────────────
    ret = spy.pct_change()
    features["ret_1d"] = ret.shift(1)
    features["ret_3d"] = spy.pct_change(3).shift(1)
    features["ret_5d"] = spy.pct_change(5).shift(1)

    # ── Cross-asset features (momentum context) ──────────────────────────────
    for tk in ["QQQ", "IWM", "TLT", "GLD", "HYG"]:
        if tk in close.columns:
            features[f"ret1_{tk.lower()}"] = close[tk].pct_change().shift(1)

    # ── Intraday range proxy (using daily returns as stand-in for close data) ─
    # True intraday range needs OHLC; here we use |ret| as a volatility proxy.
    features["abs_ret_1d"] = features["ret_1d"].abs()

    # ── Gap size (next open vs prior close — approximated by overnight shift) ─
    features["gap_1d"] = ret.shift(1)   # Proxy; real gap needs open prices

    return features


# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE METRICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(returns_series, name="strategy"):
    """Compute full performance suite."""
    r = returns_series.dropna()
    if len(r) < 10:
        return {"name": name, "error": "insufficient data"}

    ann_factor = 252
    total_return = (1 + r).prod() - 1
    n_years = len(r) / ann_factor
    cagr = (1 + total_return) ** (1 / max(n_years, 0.1)) - 1

    excess = r - 0  # risk-free ≈ 0 (conservative)
    sharpe = (excess.mean() / excess.std()) * np.sqrt(ann_factor) if excess.std() > 0 else 0

    downside = r[r < 0]
    sortino_denom = np.sqrt((downside**2).mean()) * np.sqrt(ann_factor) if len(downside) > 0 else 1e-9
    sortino = (r.mean() * ann_factor) / sortino_denom if sortino_denom > 0 else 0

    # Max Drawdown
    cumulative = (1 + r).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_dd = drawdown.min()

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Win Rate and Profit Factor
    wins = r[r > 0]
    losses = r[r < 0]
    win_rate = len(wins) / len(r) if len(r) > 0 else 0
    profit_factor = abs(wins.sum() / losses.sum()) if len(losses) > 0 and losses.sum() != 0 else np.inf

    return {
        "name": name,
        "n_trades": int(len(r)),
        "n_years": round(n_years, 2),
        "total_return_pct": round(total_return * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_dd_pct": round(max_dd * 100, 2),
        "calmar": round(calmar, 3),
        "win_rate_pct": round(win_rate * 100, 2),
        "profit_factor": round(float(profit_factor), 3),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SIMPLE RSI BASELINE (no ML)
# ─────────────────────────────────────────────────────────────────────────────
def rsi_only_strategy(spy_returns, rsi_14, mode="long_only"):
    """
    Simple RSI mean reversion:
      - Long when RSI < 30, hold 1 day
      - Short/cash when RSI > 70, hold 1 day
    """
    signal = pd.Series(0.0, index=spy_returns.index)

    if mode == "long_only":
        signal[rsi_14.shift(1) < RSI_OVERSOLD] = 1.0
    elif mode == "long_short":
        signal[rsi_14.shift(1) < RSI_OVERSOLD] = 1.0
        signal[rsi_14.shift(1) > RSI_OVERBOUGHT] = -1.0

    strat_returns = signal * spy_returns
    return strat_returns


# ─────────────────────────────────────────────────────────────────────────────
# SLIDING WALK-FORWARD ML BACKTEST
# ─────────────────────────────────────────────────────────────────────────────
def sliding_walkforward(features, spy_returns, rsi_14, model_type="lgbm"):
    """
    Sliding 252-day walk-forward.
    Train on [t-252, t-1], predict day t.
    Only trade on days when RSI(14) < 30 or RSI(14) > 70 (signal zones).
    Target: next-day positive return (binary).
    """
    feat_cols = [c for c in features.columns if not features[c].isna().all()]
    df = features[feat_cols].copy()
    df["target"] = (spy_returns > 0).astype(int)   # 1=up, 0=down next day
    df["spy_ret"] = spy_returns
    df["rsi_14"] = rsi_14
    df = df.dropna()

    idx = df.index
    n = len(idx)
    min_idx = TRAIN_WINDOW

    print(f"\n[WF] Walk-forward: {n} total days, window={TRAIN_WINDOW}, model={model_type}")
    print(f"[WF] OOT days: {n - min_idx}")

    all_preds  = []
    all_proba  = []
    all_dates  = []
    all_actual = []
    all_rets   = []

    step = 5  # Predict every 5 days (refit weekly — balance between speed and freshness)

    refit_dates = list(range(min_idx, n, step))
    print(f"[WF] Refit steps: {len(refit_dates)}")

    last_model = None
    last_scaler = None

    for step_idx, t in enumerate(range(min_idx, n)):
        train_start = t - TRAIN_WINDOW
        train_end   = t

        train_rows = df.iloc[train_start:train_end]
        test_row   = df.iloc[t]

        X_train = train_rows[feat_cols].values
        y_train = train_rows["target"].values

        # Only refit model on refit steps
        if t in refit_dates or last_model is None:
            X_train_clean = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

            if len(np.unique(y_train)) < 2:
                last_model = None
                continue

            try:
                if model_type == "lgbm" and LGBM_AVAILABLE:
                    model = lgb.LGBMClassifier(
                        n_estimators=100,
                        learning_rate=0.05,
                        num_leaves=15,
                        min_child_samples=20,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42,
                        verbose=-1,
                        n_jobs=2,
                    )
                    model.fit(X_train_clean, y_train)
                else:
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X_train_clean)
                    model = LogisticRegression(C=0.1, max_iter=500, random_state=42)
                    model.fit(X_scaled, y_train)
                    last_scaler = scaler

                last_model = model
            except Exception as e:
                print(f"  [WARN] Model fit failed at t={t}: {e}")
                last_model = None
                continue

        if last_model is None:
            continue

        x_test = np.nan_to_num(test_row[feat_cols].values.reshape(1, -1),
                               nan=0.0, posinf=0.0, neginf=0.0)

        try:
            if model_type == "lgbm" and LGBM_AVAILABLE:
                pred = last_model.predict(x_test)[0]
                proba = last_model.predict_proba(x_test)[0][1]
            else:
                x_scaled = last_scaler.transform(x_test)
                pred = last_model.predict(x_scaled)[0]
                proba = last_model.predict_proba(x_scaled)[0][1]
        except Exception:
            continue

        all_preds.append(pred)
        all_proba.append(proba)
        all_dates.append(idx[t])
        all_actual.append(test_row["target"])
        all_rets.append(test_row["spy_ret"])

    if not all_dates:
        print("[ERROR] No predictions generated!")
        return pd.Series(dtype=float), pd.Series(dtype=float)

    pred_df = pd.DataFrame({
        "pred": all_preds,
        "proba": all_proba,
        "actual": all_actual,
        "spy_ret": all_rets,
        "rsi_14": rsi_14.reindex(all_dates).values,
    }, index=all_dates)

    print(f"[WF] Predictions: {len(pred_df)} days")
    print(f"[WF] Accuracy (all days): {accuracy_score(pred_df['actual'], pred_df['pred']):.3f}")

    # ── ML Strategy: Only trade in signal zones (RSI < 30 or > 70) ─────────
    # Long: ML predicts up AND RSI oversold
    # Short/cash: ML predicts down AND RSI overbought
    oversold  = pred_df["rsi_14"] < RSI_OVERSOLD
    overbought = pred_df["rsi_14"] > RSI_OVERBOUGHT

    ml_long_signal  = oversold  & (pred_df["proba"] > 0.55)
    ml_short_signal = overbought & (pred_df["proba"] < 0.45)

    ml_signal = pd.Series(0.0, index=pred_df.index)
    ml_signal[ml_long_signal]  = 1.0
    ml_signal[ml_short_signal] = -1.0

    ml_returns = ml_signal * pred_df["spy_ret"]

    # RSI-only baseline (aligned to same OOT period)
    rsi_signal_base = pd.Series(0.0, index=pred_df.index)
    rsi_signal_base[oversold]  = 1.0
    rsi_signal_base[overbought] = -1.0
    rsi_returns_base = rsi_signal_base * pred_df["spy_ret"]

    n_long  = int(ml_long_signal.sum())
    n_short = int(ml_short_signal.sum())
    print(f"[WF] ML long trades: {n_long}  |  ML short trades: {n_short}")

    return ml_returns, rsi_returns_base, pred_df


# ─────────────────────────────────────────────────────────────────────────────
# ADVERSARIAL VALIDATION SUITE
# ─────────────────────────────────────────────────────────────────────────────
def permutation_test(strategy_returns, n_iter=N_PERMUTATIONS, seed=42):
    """Shuffle daily returns and recompute Sharpe. Report p-value."""
    rng = np.random.RandomState(seed)
    r = strategy_returns.dropna().values
    r = r[r != 0]  # Only active trading days

    if len(r) < 10:
        return {"error": "insufficient data", "p_value": np.nan}

    obs_sharpe = r.mean() / (r.std() + 1e-9) * np.sqrt(252)
    null_sharpes = []
    for _ in range(n_iter):
        shuffled = rng.permutation(r)
        null_sharpes.append(shuffled.mean() / (shuffled.std() + 1e-9) * np.sqrt(252))

    p_value = float(np.mean(np.array(null_sharpes) >= obs_sharpe))
    return {
        "observed_sharpe": round(obs_sharpe, 3),
        "null_mean_sharpe": round(float(np.mean(null_sharpes)), 3),
        "null_std_sharpe": round(float(np.std(null_sharpes)), 3),
        "p_value": round(p_value, 4),
        "significant": bool(p_value < 0.05),
    }


def sub_period_consistency(strategy_returns, n_blocks=4):
    """Split into N equal blocks, compute Sharpe per block, report CV."""
    r = strategy_returns.dropna()
    r = r[r != 0]  # Active days only

    if len(r) < n_blocks * 20:
        return {"error": "insufficient data for sub-period split"}

    block_size = len(r) // n_blocks
    sharpes = []
    for i in range(n_blocks):
        block = r.iloc[i * block_size: (i + 1) * block_size]
        if block.std() > 0:
            s = block.mean() / block.std() * np.sqrt(252)
        else:
            s = 0.0
        sharpes.append(round(float(s), 3))

    cv = float(np.std(sharpes) / (abs(np.mean(sharpes)) + 1e-9))
    return {
        "block_sharpes": sharpes,
        "mean_sharpe": round(float(np.mean(sharpes)), 3),
        "cv_sharpe": round(cv, 3),
        "consistent": bool(cv < 1.5 and sum(s > 0 for s in sharpes) >= n_blocks // 2),
    }


def outlier_removal_test(strategy_returns):
    """Remove top 5% absolute return days. Check if Sharpe survives."""
    r = strategy_returns.dropna()
    r_active = r[r != 0]

    if len(r_active) < 20:
        return {"error": "insufficient data"}

    threshold = np.percentile(r_active.abs(), OUTLIER_PCTILE)
    r_filtered = r_active[r_active.abs() <= threshold]

    orig_sharpe = r_active.mean() / r_active.std() * np.sqrt(252) if r_active.std() > 0 else 0
    filt_sharpe = r_filtered.mean() / r_filtered.std() * np.sqrt(252) if r_filtered.std() > 0 else 0

    return {
        "n_original": int(len(r_active)),
        "n_after_filter": int(len(r_filtered)),
        "threshold_abs_ret": round(float(threshold), 5),
        "sharpe_original": round(float(orig_sharpe), 3),
        "sharpe_no_outliers": round(float(filt_sharpe), 3),
        "sharpe_survives": bool(filt_sharpe > 0),
    }


def regime_test(strategy_returns, spy_returns):
    """
    R1 from HC #428: stratify Sharpe by green vs red SPY days.
    Gap |Sharpe_green - Sharpe_red| / max(|Sg|, |Sr|) must be < 0.50.
    """
    r = strategy_returns.dropna()
    spy_aligned = spy_returns.reindex(r.index).fillna(0)

    green_days = spy_aligned > 0
    red_days   = spy_aligned < 0

    def block_sharpe(returns, mask):
        block = returns[mask]
        if len(block) < 5 or block.std() == 0:
            return 0.0
        return float(block.mean() / block.std() * np.sqrt(252))

    sg = block_sharpe(r, green_days)
    sr = block_sharpe(r, red_days)
    sf = block_sharpe(r, ~green_days & ~red_days)

    denom = max(abs(sg), abs(sr), 1e-9)
    gap   = abs(sg - sr) / denom

    return {
        "sharpe_green_days": round(sg, 3),
        "sharpe_red_days":   round(sr, 3),
        "sharpe_flat_days":  round(sf, 3),
        "gap_ratio": round(gap, 3),
        "passes_r1": bool(gap < 0.50),
        "note": "R1 gate: gap < 0.50 required (HC #428)",
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
def main():
    results = {
        "run_date": datetime.now().isoformat(),
        "config": {
            "capital": CAPITAL,
            "train_window": TRAIN_WINDOW,
            "rsi_oversold": RSI_OVERSOLD,
            "rsi_overbought": RSI_OVERBOUGHT,
            "n_permutations": N_PERMUTATIONS,
            "start_date": START_DATE,
            "end_date": END_DATE,
        },
        "models": {},
        "baselines": {},
        "adversarial": {},
        "summary": {},
    }

    # ── 1. Download data ─────────────────────────────────────────────────────
    print("\n[1/6] Downloading market data...")
    try:
        close, volume = download_data(TICKERS, START_DATE, END_DATE)
    except Exception as e:
        print(f"[ERROR] Data download failed: {e}")
        traceback.print_exc()
        return

    if "SPY" not in close.columns:
        print("[ERROR] SPY data unavailable. Cannot proceed.")
        return

    spy_returns = close["SPY"].pct_change().dropna()
    rsi_14 = compute_rsi(close["SPY"], 14)

    # ── 2. Build features ────────────────────────────────────────────────────
    print("\n[2/6] Building features...")
    try:
        features = build_features(close, volume)
        features = features.reindex(spy_returns.index)
        print(f"  Feature matrix: {features.shape[0]} rows × {features.shape[1]} cols")
        print(f"  Features: {', '.join(features.columns.tolist())}")
    except Exception as e:
        print(f"[ERROR] Feature build failed: {e}")
        traceback.print_exc()
        return

    # ── 3. RSI-only baseline ─────────────────────────────────────────────────
    print("\n[3/6] Computing RSI-only baselines...")

    # Buy & Hold
    bh_returns = spy_returns
    bh_metrics = compute_metrics(bh_returns, "buy_and_hold_spy")
    results["baselines"]["buy_and_hold"] = bh_metrics

    # RSI long-only
    rsi_long_only = rsi_only_strategy(spy_returns, rsi_14, mode="long_only")
    rsi_lo_metrics = compute_metrics(rsi_long_only[rsi_long_only != 0], "rsi_long_only")
    results["baselines"]["rsi_long_only"] = rsi_lo_metrics

    # RSI long-short
    rsi_ls = rsi_only_strategy(spy_returns, rsi_14, mode="long_short")
    rsi_ls_metrics = compute_metrics(rsi_ls[rsi_ls != 0], "rsi_long_short")
    results["baselines"]["rsi_long_short"] = rsi_ls_metrics

    for k, v in results["baselines"].items():
        print(f"  {v['name']:30s}  Sharpe={v.get('sharpe', 'N/A'):6.3f}  "
              f"CAGR={v.get('cagr_pct', 'N/A'):6.2f}%  "
              f"MaxDD={v.get('max_dd_pct', 'N/A'):7.2f}%  "
              f"Sortino={v.get('sortino', 'N/A'):6.3f}")

    # ── 4. ML Walk-Forward ───────────────────────────────────────────────────
    model_types = []
    if LGBM_AVAILABLE:
        model_types.append("lgbm")
    model_types.append("logistic")

    all_ml_returns = {}

    print("\n[4/6] Running ML walk-forward (this may take several minutes)...")
    for model_type in model_types:
        print(f"\n  ── Model: {model_type.upper()} ──")
        try:
            ml_rets, rsi_rets_oot, pred_df = sliding_walkforward(
                features, spy_returns, rsi_14, model_type=model_type
            )

            ml_metrics = compute_metrics(ml_rets[ml_rets != 0], f"ml_{model_type}")
            rsi_oot_metrics = compute_metrics(rsi_rets_oot[rsi_rets_oot != 0],
                                               f"rsi_baseline_oot_{model_type}")

            results["models"][model_type] = {
                "ml_strategy": ml_metrics,
                "rsi_baseline_oot": rsi_oot_metrics,
            }
            all_ml_returns[model_type] = ml_rets

            print(f"  ML {model_type:10s}  Sharpe={ml_metrics.get('sharpe', 'N/A'):6.3f}  "
                  f"CAGR={ml_metrics.get('cagr_pct', 'N/A'):6.2f}%  "
                  f"MaxDD={ml_metrics.get('max_dd_pct', 'N/A'):7.2f}%  "
                  f"Sortino={ml_metrics.get('sortino', 'N/A'):6.3f}  "
                  f"WR={ml_metrics.get('win_rate_pct', 'N/A'):5.1f}%  "
                  f"PF={ml_metrics.get('profit_factor', 'N/A'):5.3f}")
            print(f"  RSI-only OOT   Sharpe={rsi_oot_metrics.get('sharpe', 'N/A'):6.3f}  "
                  f"CAGR={rsi_oot_metrics.get('cagr_pct', 'N/A'):6.2f}%")

        except Exception as e:
            print(f"  [ERROR] {model_type} failed: {e}")
            traceback.print_exc()

    # ── 5. Adversarial Validation ────────────────────────────────────────────
    print("\n[5/6] Running adversarial validation suite...")

    # Use best model (prefer lgbm, fallback logistic)
    best_model = "lgbm" if ("lgbm" in all_ml_returns) else ("logistic" if "logistic" in all_ml_returns else None)

    if best_model and best_model in all_ml_returns:
        ml_rets = all_ml_returns[best_model]

        print(f"  Using {best_model.upper()} for adversarial tests")
        print(f"  Active trading days: {int((ml_rets != 0).sum())}")

        # Permutation test
        print(f"  Running permutation test ({N_PERMUTATIONS} iterations)...")
        perm = permutation_test(ml_rets)
        results["adversarial"]["permutation_test"] = perm
        print(f"  Permutation p-value={perm.get('p_value', 'N/A'):.4f}  "
              f"significant={perm.get('significant', 'N/A')}")

        # Sub-period consistency
        sp_result = sub_period_consistency(ml_rets)
        results["adversarial"]["sub_period_consistency"] = sp_result
        cv_val = sp_result.get('cv_sharpe', 'N/A')
        cv_str = f"{cv_val:.3f}" if isinstance(cv_val, (int, float)) else str(cv_val)
        print(f"  Sub-period Sharpes: {sp_result.get('block_sharpes', [])}  "
              f"CV={cv_str}  "
              f"consistent={sp_result.get('consistent', 'N/A')}")

        # Outlier removal
        out_result = outlier_removal_test(ml_rets)
        results["adversarial"]["outlier_removal"] = out_result
        print(f"  Outlier removal: Sharpe {out_result.get('sharpe_original', 'N/A'):.3f} → "
              f"{out_result.get('sharpe_no_outliers', 'N/A'):.3f}  "
              f"survives={out_result.get('sharpe_survives', 'N/A')}")

        # R1 regime test
        regime = regime_test(ml_rets, spy_returns)
        results["adversarial"]["regime_test_r1"] = regime
        print(f"  R1 Regime: Sharpe_green={regime.get('sharpe_green_days', 'N/A'):.3f}  "
              f"Sharpe_red={regime.get('sharpe_red_days', 'N/A'):.3f}  "
              f"gap={regime.get('gap_ratio', 'N/A'):.3f}  "
              f"passes={regime.get('passes_r1', 'N/A')}")
    else:
        print("  [SKIP] No ML returns available for adversarial tests")

    # ── 6. Summary ───────────────────────────────────────────────────────────
    print("\n[6/6] Building summary...")

    # Determine if ML adds value vs RSI-only
    ml_verdict = {}
    for mt in model_types:
        if mt not in results["models"]:
            continue
        ml_sh = results["models"][mt]["ml_strategy"].get("sharpe", 0) or 0
        rsi_sh = results["models"][mt]["rsi_baseline_oot"].get("sharpe", 0) or 0
        ml_adds_value = ml_sh > rsi_sh
        ml_verdict[mt] = {
            "ml_sharpe": ml_sh,
            "rsi_sharpe": rsi_sh,
            "ml_improvement": round(ml_sh - rsi_sh, 3),
            "ml_adds_value": ml_adds_value,
        }

    # Overall viability
    overall_pass = False
    adv = results["adversarial"]
    if adv:
        perm_ok = adv.get("permutation_test", {}).get("significant", False)
        regime_ok = adv.get("regime_test_r1", {}).get("passes_r1", False)
        outlier_ok = adv.get("outlier_removal", {}).get("sharpe_survives", False)
        consistent = adv.get("sub_period_consistency", {}).get("consistent", False)
        overall_pass = perm_ok and regime_ok and outlier_ok
    else:
        perm_ok = regime_ok = outlier_ok = consistent = False

    best_ml_sharpe = 0.0
    if best_model and best_model in results.get("models", {}):
        best_ml_sharpe = results["models"][best_model]["ml_strategy"].get("sharpe", 0) or 0

    results["summary"] = {
        "best_model": best_model,
        "ml_vs_rsi_verdict": ml_verdict,
        "adversarial_gates": {
            "permutation_test_significant_p05": perm_ok,
            "regime_agnostic_r1": regime_ok,
            "outlier_robust": outlier_ok,
            "sub_period_consistent": consistent,
        },
        "overall_viable": overall_pass,
        "buy_and_hold_sharpe": results["baselines"].get("buy_and_hold", {}).get("sharpe", 0),
        "best_ml_sharpe": round(best_ml_sharpe, 3),
        "recommendation": (
            "VIABLE — ML adds edge, passes adversarial gates"
            if overall_pass and best_ml_sharpe > 0.3
            else "MARGINAL — Passes some gates but edge is weak or adversarial tests fail"
            if best_ml_sharpe > 0 and not overall_pass
            else "REJECT — Strategy fails adversarial validation or negative Sharpe"
        ),
    }

    # ── Save results ─────────────────────────────────────────────────────────
    out_path = OUTPUT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[SAVED] Results written to: {out_path}")

    # ── Print SUMMARY ─────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    bh = results["baselines"].get("buy_and_hold", {})
    print(f"  Buy & Hold SPY:  Sharpe={bh.get('sharpe', 0):6.3f}  "
          f"CAGR={bh.get('cagr_pct', 0):5.2f}%  "
          f"MaxDD={bh.get('max_dd_pct', 0):7.2f}%  "
          f"Sortino={bh.get('sortino', 0):6.3f}")

    rsi_lo = results["baselines"].get("rsi_long_only", {})
    print(f"  RSI Long-Only:   Sharpe={rsi_lo.get('sharpe', 0):6.3f}  "
          f"CAGR={rsi_lo.get('cagr_pct', 0):5.2f}%  "
          f"MaxDD={rsi_lo.get('max_dd_pct', 0):7.2f}%  "
          f"Sortino={rsi_lo.get('sortino', 0):6.3f}")

    rsi_ls = results["baselines"].get("rsi_long_short", {})
    print(f"  RSI Long-Short:  Sharpe={rsi_ls.get('sharpe', 0):6.3f}  "
          f"CAGR={rsi_ls.get('cagr_pct', 0):5.2f}%  "
          f"MaxDD={rsi_ls.get('max_dd_pct', 0):7.2f}%  "
          f"Sortino={rsi_ls.get('sortino', 0):6.3f}")

    print()
    for mt in model_types:
        if mt not in results.get("models", {}):
            continue
        ml_m = results["models"][mt]["ml_strategy"]
        print(f"  ML {mt.upper():10s}:  Sharpe={ml_m.get('sharpe', 0):6.3f}  "
              f"CAGR={ml_m.get('cagr_pct', 0):5.2f}%  "
              f"MaxDD={ml_m.get('max_dd_pct', 0):7.2f}%  "
              f"Sortino={ml_m.get('sortino', 0):6.3f}  "
              f"WR={ml_m.get('win_rate_pct', 0):5.1f}%  "
              f"PF={ml_m.get('profit_factor', 0):5.3f}  "
              f"Calmar={ml_m.get('calmar', 0):5.3f}  "
              f"Trades={ml_m.get('n_trades', 0)}")

    print()
    print("  Adversarial Gates:")
    ag = results["summary"].get("adversarial_gates", {})
    for gate, passed in ag.items():
        status = "PASS" if passed else "FAIL"
        print(f"    [{status}] {gate}")

    if ml_verdict:
        print()
        print("  ML vs RSI-only:")
        for mt, v in ml_verdict.items():
            symbol = "+" if v["ml_improvement"] > 0 else ""
            print(f"    {mt.upper():10s}: ML={v['ml_sharpe']:.3f}  RSI={v['rsi_sharpe']:.3f}  "
                  f"delta={symbol}{v['ml_improvement']:.3f}  adds_value={v['ml_adds_value']}")

    print()
    s = results["summary"]
    print(f"  OVERALL:  {s['recommendation']}")
    print(f"  Best ML Sharpe: {s['best_ml_sharpe']:.3f}  |  B&H Sharpe: {s['buy_and_hold_sharpe']:.3f}")
    print("=" * 70)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
