#!/usr/bin/env python3
"""
ML Stock Picker v2 — Walk-Forward LightGBM Individual Stock Selection
=====================================================================
Universe: S&P 500 (top 100 by market cap, hardcoded for reliability)
Features: momentum, reversal, 52wk-high proximity, rel volume, RSI, beta, sector
Target:   next-month top-quintile return (binary)
Model:    LightGBM classifier, 12-month train / 1-month test sliding
Eval:     Sharpe, Sortino, CAGR, MaxDD, WR, regime stratification
Adversarial: permutation test, sub-period stability, outlier robustness, regime gap
"""

import os, sys, json, warnings, time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path("/home/jupiter/Lvl3Quant")
CACHE_DIR = BASE_DIR / "output" / "growth_research" / "stock_picker_v2" / "cache"
OUTPUT_DIR = BASE_DIR / "output" / "growth_research" / "stock_picker_v2"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Universe: Top 100 S&P 500 by market cap ───────────────────────────────
SP500_TOP100 = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "LLY", "AVGO", "JPM",
    "TSLA", "UNH", "XOM", "V", "MA", "PG", "COST", "JNJ", "HD", "ABBV",
    "MRK", "NFLX", "CRM", "BAC", "AMD", "CVX", "KO", "ORCL", "PEP", "TMO",
    "WMT", "ACN", "LIN", "MCD", "ABT", "CSCO", "DHR", "ADBE", "PM", "TXN",
    "QCOM", "ISRG", "INTU", "CMCSA", "NEE", "GE", "AMGN", "PFE", "AMAT", "HON",
    "UNP", "RTX", "BKNG", "LOW", "T", "SPGI", "BLK", "SYK", "COP", "VRTX",
    "GILD", "C", "ADP", "MDLZ", "PANW", "SCHW", "CB", "BSX", "LRCX",
    "DE", "REGN", "SBUX", "ADI", "BMY", "KLAC", "CI", "NOW", "ZTS", "CME",
    "PLD", "SO", "ICE", "DUK", "MO", "EQIX", "SNPS", "CDNS", "SHW", "APD",
    "WM", "MCO", "NOC", "FDX", "ITW", "MSI", "ORLY", "GD", "NSC", "PNC",
]

BENCHMARK = "SPY"

# ── Parameters ─────────────────────────────────────────────────────────────
TRAIN_MONTHS = 12
TOP_N = 10
BACKTEST_START = "2018-01-01"
BACKTEST_END = "2026-06-30"
DATA_START = "2016-06-01"

# ── Sector mapping ────────────────────────────────────────────────────────
SECTOR_MAP = {
    "AAPL": "Tech", "MSFT": "Tech", "NVDA": "Tech", "GOOGL": "Tech", "META": "Tech",
    "AVGO": "Tech", "AMD": "Tech", "CRM": "Tech", "ORCL": "Tech", "ACN": "Tech",
    "CSCO": "Tech", "ADBE": "Tech", "TXN": "Tech", "QCOM": "Tech", "INTU": "Tech",
    "AMAT": "Tech", "PANW": "Tech", "LRCX": "Tech", "ADI": "Tech", "KLAC": "Tech",
    "NOW": "Tech", "SNPS": "Tech", "CDNS": "Tech", "MSI": "Tech",
    "AMZN": "ConsDisc", "TSLA": "ConsDisc", "HD": "ConsDisc", "MCD": "ConsDisc",
    "BKNG": "ConsDisc", "LOW": "ConsDisc", "SBUX": "ConsDisc", "ORLY": "ConsDisc",
    "NFLX": "Comm", "CMCSA": "Comm", "T": "Comm",
    "BRK-B": "Fin", "JPM": "Fin", "BAC": "Fin", "V": "Fin",
    "MA": "Fin", "SPGI": "Fin", "BLK": "Fin", "SCHW": "Fin",
    "C": "Fin", "CB": "Fin", "ICE": "Fin",
    "CME": "Fin", "MCO": "Fin", "PNC": "Fin",
    "LLY": "Health", "UNH": "Health", "JNJ": "Health", "ABBV": "Health", "MRK": "Health",
    "TMO": "Health", "ABT": "Health", "DHR": "Health", "AMGN": "Health", "PFE": "Health",
    "ISRG": "Health", "VRTX": "Health", "GILD": "Health", "BSX": "Health", "REGN": "Health",
    "BMY": "Health", "CI": "Health", "ZTS": "Health", "SYK": "Health",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "PG": "ConsStap", "COST": "ConsStap", "KO": "ConsStap", "PEP": "ConsStap",
    "PM": "ConsStap", "WMT": "ConsStap", "MDLZ": "ConsStap", "MO": "ConsStap",
    "HON": "Ind", "UNP": "Ind", "RTX": "Ind", "GE": "Ind",
    "DE": "Ind", "ADP": "Ind", "NOC": "Ind", "FDX": "Ind",
    "ITW": "Ind", "GD": "Ind", "NSC": "Ind", "WM": "Ind",
    "LIN": "Materials", "SHW": "Materials", "APD": "Materials",
    "NEE": "Util", "SO": "Util", "DUK": "Util",
    "PLD": "RE", "EQIX": "RE",
}


def download_data():
    """Download or load cached price data."""
    cache_file = CACHE_DIR / "price_data.parquet"
    if cache_file.exists():
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        age_days = (datetime.now() - mtime).days
        if age_days < 7:
            print(f"  Loading cached data ({age_days}d old)...")
            return pd.read_parquet(cache_file)

    all_tickers = SP500_TOP100 + [BENCHMARK]
    print(f"  Downloading {len(all_tickers)} tickers from yfinance...")

    all_dfs = []
    batch_size = 50
    for i in range(0, len(all_tickers), batch_size):
        batch = all_tickers[i:i+batch_size]
        print(f"    Batch {i//batch_size + 1}: {batch[0]}...{batch[-1]}")
        try:
            data = yf.download(batch, start=DATA_START, end=BACKTEST_END,
                               auto_adjust=True, progress=False, threads=True)
            if isinstance(data.columns, pd.MultiIndex):
                close = data["Close"]
                volume = data["Volume"]
            else:
                close = data[["Close"]]
                volume = data[["Volume"]]

            for ticker in batch:
                if ticker in close.columns:
                    df = pd.DataFrame({
                        "close": close[ticker],
                        "volume": volume[ticker],
                        "ticker": ticker,
                    })
                    df.index.name = "date"
                    all_dfs.append(df.reset_index())
        except Exception as e:
            print(f"    WARNING: batch download failed: {e}")
        time.sleep(0.3)

    if not all_dfs:
        raise RuntimeError("No data downloaded!")

    combined = pd.concat(all_dfs, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.dropna(subset=["close"])
    combined.to_parquet(cache_file, index=False)
    print(f"  Cached {len(combined)} rows")
    return combined


def compute_rsi(series, period=14):
    """Vectorized RSI computation."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_features_vectorized(price_df):
    """Vectorized monthly feature computation — much faster than per-row loop."""
    print("  Computing features (vectorized)...")
    t0 = time.time()

    # Separate SPY
    spy_daily = price_df[price_df["ticker"] == BENCHMARK].set_index("date")["close"].sort_index()
    spy_daily_ret = spy_daily.pct_change()
    spy_sma200 = spy_daily.rolling(200).mean()

    # Work with stock data only
    stocks = price_df[price_df["ticker"] != BENCHMARK].copy()
    stocks = stocks.sort_values(["ticker", "date"]).set_index("date")

    all_monthly = []

    for ticker in stocks["ticker"].unique():
        tdf = stocks[stocks["ticker"] == ticker][["close", "volume"]].copy()
        if len(tdf) < 300:
            continue

        close = tdf["close"]
        volume = tdf["volume"]
        daily_ret = close.pct_change()

        # Compute daily features first, then resample
        # 12-month momentum
        tdf["mom_12m"] = close / close.shift(252) - 1
        # 3-month momentum
        tdf["mom_3m"] = close / close.shift(63) - 1
        # 1-month reversal
        tdf["rev_1m"] = close / close.shift(21) - 1
        # 52-week high proximity
        tdf["high_52w"] = close.rolling(252).max()
        tdf["proximity_52w"] = close / tdf["high_52w"]
        # Relative volume
        vol_21 = volume.rolling(21).mean()
        vol_63 = volume.rolling(63).mean()
        tdf["rel_volume"] = vol_21 / vol_63.replace(0, np.nan)
        # RSI
        tdf["rsi"] = compute_rsi(close, 14)

        # Beta vs SPY (60-day rolling)
        common_dates = daily_ret.index.intersection(spy_daily_ret.index)
        stock_ret_aligned = daily_ret.reindex(common_dates)
        spy_ret_aligned = spy_daily_ret.reindex(common_dates)
        cov_60 = stock_ret_aligned.rolling(60).cov(spy_ret_aligned)
        var_spy_60 = spy_ret_aligned.rolling(60).var()
        beta_series = cov_60 / var_spy_60.replace(0, np.nan)
        tdf["beta"] = beta_series.reindex(tdf.index)

        # Resample to month-end: take last available value each month
        monthly = tdf[["mom_12m", "mom_3m", "rev_1m", "proximity_52w",
                        "rel_volume", "rsi", "beta"]].resample("ME").last()
        monthly_close = close.resample("ME").last()

        # Next-month return
        monthly["next_month_ret"] = monthly_close.pct_change().shift(-1)

        # SPY regime
        spy_me = spy_daily.resample("ME").last()
        spy_sma_me = spy_sma200.resample("ME").last()
        regime = (spy_me > spy_sma_me).map({True: "bull", False: "bear"})
        monthly["regime"] = regime.reindex(monthly.index)

        monthly["ticker"] = ticker
        monthly["sector"] = SECTOR_MAP.get(ticker, "Other")

        all_monthly.append(monthly)

    features_df = pd.concat(all_monthly).reset_index()
    features_df.rename(columns={"date": "date"}, inplace=True)

    # Filter to backtest period
    features_df = features_df[features_df["date"] >= BACKTEST_START]

    print(f"  Generated {len(features_df)} stock-month obs across "
          f"{features_df['ticker'].nunique()} stocks in {time.time()-t0:.1f}s")
    return features_df


def prepare_features(df, feat_names=None):
    """Prepare feature matrix with sector dummies."""
    base_cols = ["mom_12m", "mom_3m", "rev_1m", "proximity_52w",
                 "rel_volume", "rsi", "beta"]
    sector_dummies = pd.get_dummies(df["sector"], prefix="sect")
    X = pd.concat([df[base_cols].reset_index(drop=True),
                    sector_dummies.reset_index(drop=True)], axis=1)

    if feat_names is not None:
        for col in feat_names:
            if col not in X.columns:
                X[col] = 0
        X = X[feat_names]

    return X


def walk_forward_backtest(features_df):
    """Run walk-forward LightGBM backtest."""
    print("\n=== WALK-FORWARD BACKTEST ===")

    features_df = features_df.sort_values("date")
    months = sorted(features_df["date"].unique())

    print(f"  Total months: {len(months)}")
    print(f"  Date range: {pd.Timestamp(months[0]).strftime('%Y-%m')} to "
          f"{pd.Timestamp(months[-1]).strftime('%Y-%m')}")

    monthly_returns = []
    monthly_picks = []
    regimes = []
    feature_importances = []

    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "verbose": -1,
        "seed": 42,
        "n_jobs": -1,
    }

    # Determine feature names from full dataset
    all_feat_names = prepare_features(features_df).columns.tolist()

    for i in range(TRAIN_MONTHS, len(months) - 1):
        test_month = months[i]
        train_start = months[max(0, i - TRAIN_MONTHS)]

        # Training data: last TRAIN_MONTHS months
        train_mask = (features_df["date"] >= train_start) & (features_df["date"] < test_month)
        train_df = features_df[train_mask].dropna(subset=["next_month_ret"]).copy()

        # Test data
        test_mask = features_df["date"] == test_month
        test_df = features_df[test_mask].copy()

        if len(train_df) < 50 or len(test_df) < 10:
            continue

        # Binary target: top quintile
        threshold = train_df["next_month_ret"].quantile(0.80)
        train_df["target"] = (train_df["next_month_ret"] >= threshold).astype(int)

        X_train = prepare_features(train_df, all_feat_names).fillna(0)
        y_train = train_df["target"].values

        X_test = prepare_features(test_df, all_feat_names).fillna(0)

        # Train
        dtrain = lgb.Dataset(X_train, label=y_train)
        model = lgb.train(lgb_params, dtrain, num_boost_round=200,
                          callbacks=[lgb.log_evaluation(0)])

        # Predict
        preds = model.predict(X_test)
        test_df = test_df.copy()
        test_df["pred_prob"] = preds

        # Pick top N
        top_picks = test_df.nlargest(min(TOP_N, len(test_df)), "pred_prob")
        valid_picks = top_picks.dropna(subset=["next_month_ret"])

        if len(valid_picks) > 0:
            port_ret = valid_picks["next_month_ret"].mean()
            monthly_returns.append(port_ret)

            regime = valid_picks["regime"].iloc[0] if "regime" in valid_picks.columns else "unknown"
            regimes.append(regime)

            monthly_picks.append({
                "date": pd.Timestamp(test_month).strftime("%Y-%m"),
                "picks": valid_picks["ticker"].tolist(),
                "pred_probs": [round(p, 4) for p in valid_picks["pred_prob"].tolist()],
                "actual_rets": [round(r, 4) for r in valid_picks["next_month_ret"].tolist()],
                "port_ret": round(port_ret, 5),
            })

            fi = dict(zip(all_feat_names, model.feature_importance(importance_type="gain")))
            feature_importances.append(fi)

        # Progress
        idx = i - TRAIN_MONTHS
        if idx % 12 == 0:
            cumret = np.prod([1 + r for r in monthly_returns]) - 1 if monthly_returns else 0
            print(f"  Month {idx+1}: {pd.Timestamp(test_month).strftime('%Y-%m')} | "
                  f"ret={port_ret if len(valid_picks)>0 else 0:.3%} | cumret={cumret:.2%}")

    print(f"  Backtest complete: {len(monthly_returns)} months")
    return monthly_returns, monthly_picks, regimes, feature_importances


def compute_metrics(returns, label=""):
    """Compute key performance metrics from monthly returns."""
    returns = np.array(returns)
    n = len(returns)
    if n == 0:
        return {"label": label, "n_months": 0}

    ann = 12
    mean_ret = returns.mean()
    std_ret = returns.std()

    sharpe = mean_ret / std_ret * np.sqrt(ann) if std_ret > 0 else 0
    downside = returns[returns < 0]
    ds_std = downside.std() if len(downside) > 1 else 0.0001
    sortino = mean_ret / ds_std * np.sqrt(ann) if ds_std > 0 else 0

    cumret = np.prod(1 + returns)
    years = n / 12
    cagr = cumret ** (1 / years) - 1 if years > 0 else 0

    cum = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cum)
    max_dd = (cum / running_max - 1).min()

    wr = (returns > 0).mean()
    gp = returns[returns > 0].sum() if (returns > 0).any() else 0
    gl = abs(returns[returns < 0].sum()) if (returns < 0).any() else 0.0001
    pf = gp / gl if gl > 0 else float('inf')

    return {
        "label": label, "n_months": n,
        "CAGR": cagr, "Sharpe": sharpe, "Sortino": sortino,
        "MaxDD": max_dd, "WinRate": wr, "ProfitFactor": pf,
        "MeanMonthly": mean_ret, "StdMonthly": std_ret,
    }


def adversarial_validation(monthly_returns, regimes):
    """Run 4-gate adversarial validation."""
    print("\n=== ADVERSARIAL VALIDATION (4 GATES) ===")
    returns = np.array(monthly_returns)
    regimes = np.array(regimes[:len(returns)])
    results = {}

    # Gate 1: Permutation Test
    print("\n  Gate 1: Permutation Test (100 shuffles)...")
    real_sharpe = compute_metrics(returns)["Sharpe"]
    rng = np.random.RandomState(42)
    perm_sharpes = np.array([
        compute_metrics(rng.permutation(returns))["Sharpe"] for _ in range(100)
    ])
    p_value = (perm_sharpes >= real_sharpe).mean()
    gate1 = p_value < 0.05
    results["gate1_permutation"] = {
        "real_sharpe": round(real_sharpe, 4),
        "perm_mean": round(perm_sharpes.mean(), 4),
        "p_value": round(p_value, 4),
        "pass": gate1,
    }
    print(f"    Real Sharpe: {real_sharpe:.3f} | Perm mean: {perm_sharpes.mean():.3f} | "
          f"p={p_value:.3f} | {'PASS' if gate1 else 'FAIL'}")

    # Gate 2: Sub-period Stability
    print("\n  Gate 2: Sub-period Stability (3 blocks)...")
    n = len(returns)
    bs = n // 3
    block_sharpes = []
    for b in range(3):
        s = b * bs
        e = s + bs if b < 2 else n
        block_sharpes.append(compute_metrics(returns[s:e])["Sharpe"])
    block_sharpes = np.array(block_sharpes)
    cv = block_sharpes.std() / abs(block_sharpes.mean()) if abs(block_sharpes.mean()) > 0.001 else 999
    gate2 = cv < 0.50
    results["gate2_subperiod"] = {
        "block_sharpes": [round(s, 4) for s in block_sharpes],
        "cv": round(cv, 4),
        "pass": gate2,
    }
    print(f"    Blocks: {[f'{s:.3f}' for s in block_sharpes]} | CV: {cv:.3f} | {'PASS' if gate2 else 'FAIL'}")

    # Gate 3: Outlier Robustness
    print("\n  Gate 3: Outlier Robustness (trim 5%)...")
    trimmed = np.sort(returns)
    tn = max(1, int(len(trimmed) * 0.05))
    trimmed = trimmed[tn:-tn]
    trim_sharpe = compute_metrics(trimmed)["Sharpe"]
    ratio = trim_sharpe / real_sharpe if abs(real_sharpe) > 0.001 else 0
    gate3 = ratio > 0.5
    results["gate3_outlier"] = {
        "full_sharpe": round(real_sharpe, 4),
        "trimmed_sharpe": round(trim_sharpe, 4),
        "ratio": round(ratio, 4),
        "pass": gate3,
    }
    print(f"    Full: {real_sharpe:.3f} | Trimmed: {trim_sharpe:.3f} | Ratio: {ratio:.3f} | "
          f"{'PASS' if gate3 else 'FAIL'}")

    # Gate 4: Regime Gap
    print("\n  Gate 4: Regime Gap...")
    bull_ret = returns[regimes == "bull"] if (regimes == "bull").any() else np.array([0])
    bear_ret = returns[regimes == "bear"] if (regimes == "bear").any() else np.array([0])
    bull_sh = compute_metrics(bull_ret)["Sharpe"]
    bear_sh = compute_metrics(bear_ret)["Sharpe"]
    max_abs = max(abs(bull_sh), abs(bear_sh), 0.001)
    gap = abs(bull_sh - bear_sh) / max_abs
    gate4 = gap < 0.50
    results["gate4_regime"] = {
        "bull_sharpe": round(bull_sh, 4), "bear_sharpe": round(bear_sh, 4),
        "n_bull": int((regimes == "bull").sum()), "n_bear": int((regimes == "bear").sum()),
        "gap": round(gap, 4), "pass": gate4,
    }
    print(f"    Bull: {bull_sh:.3f} ({(regimes=='bull').sum()} mo) | "
          f"Bear: {bear_sh:.3f} ({(regimes=='bear').sum()} mo) | "
          f"Gap: {gap:.3f} | {'PASS' if gate4 else 'FAIL'}")

    gates_passed = sum([gate1, gate2, gate3, gate4])
    results["summary"] = {
        "gates_passed": gates_passed, "total": 4,
        "overall": "PASS" if gates_passed == 4 else f"PARTIAL ({gates_passed}/4)",
    }
    return results


def main():
    print("=" * 70)
    print("ML Stock Picker v2 — Walk-Forward LightGBM")
    print("=" * 70)
    t0 = time.time()

    # Step 1: Download
    print("\n[1/5] Downloading price data...")
    price_df = download_data()
    print(f"  {len(price_df)} rows, {price_df['ticker'].nunique()} tickers, "
          f"{price_df['date'].min().strftime('%Y-%m-%d')} to {price_df['date'].max().strftime('%Y-%m-%d')}")

    # Step 2: Features
    print("\n[2/5] Computing features...")
    features_df = compute_features_vectorized(price_df)
    if len(features_df) < 100:
        print("ERROR: Not enough data. Exiting.")
        return
    features_df.to_parquet(OUTPUT_DIR / "features.parquet", index=False)

    # Step 3: Backtest
    print("\n[3/5] Walk-forward backtest...")
    monthly_returns, monthly_picks, regimes, feat_imps = walk_forward_backtest(features_df)
    if not monthly_returns:
        print("ERROR: No months produced.")
        return

    # Step 4: Metrics
    print("\n[4/5] Performance metrics...")
    metrics = compute_metrics(monthly_returns, "ML Stock Picker v2")
    print(f"\n{'='*50}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*50}")
    for k in ["n_months", "CAGR", "Sharpe", "Sortino", "MaxDD", "WinRate", "ProfitFactor", "MeanMonthly"]:
        v = metrics[k]
        if k in ("CAGR", "MaxDD", "WinRate", "MeanMonthly"):
            print(f"  {k:16s}: {v:.2%}")
        elif k == "n_months":
            print(f"  {k:16s}: {v}")
        else:
            print(f"  {k:16s}: {v:.3f}")

    # Feature importance
    if feat_imps:
        avg_fi = pd.DataFrame(feat_imps).mean().sort_values(ascending=False)
        print("\n  Top Features (avg gain):")
        for feat, imp in avg_fi.head(8).items():
            print(f"    {feat:22s}: {imp:.1f}")

    # Step 5: Adversarial
    print("\n[5/5] Adversarial validation...")
    adv = adversarial_validation(monthly_returns, regimes)

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "params": {"train_months": TRAIN_MONTHS, "top_n": TOP_N,
                    "universe": len(SP500_TOP100), "start": BACKTEST_START, "end": BACKTEST_END},
        "metrics": {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in metrics.items()},
        "adversarial": adv,
        "recent_picks": monthly_picks[-6:],
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    pd.DataFrame({
        "date": [p["date"] for p in monthly_picks],
        "return": monthly_returns,
        "regime": regimes[:len(monthly_returns)],
    }).to_csv(OUTPUT_DIR / "monthly_returns.csv", index=False)

    pd.DataFrame(monthly_picks).to_parquet(OUTPUT_DIR / "all_picks.parquet", index=False)

    elapsed = time.time() - t0
    print(f"\n{'='*50}")
    print(f"ADVERSARIAL: {adv['summary']['overall']}")
    print(f"Runtime: {elapsed:.0f}s")
    print(f"Results: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
