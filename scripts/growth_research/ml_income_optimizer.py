#!/usr/bin/env python3
"""
ML Income Optimizer — Premium Selling Timing Model
====================================================
HC #714: ML-driven options income optimization study.

Hypothesis: ML can identify WHEN to sell premium (high IV rank + calm conditions)
and WHEN to sit out (pre-spike conditions), optimizing timing of income strategies.

Rules applied:
- Fixed $100K capital, NO DCA (HC #713)
- Black-Scholes pricing with 25% haircut (HC #713 R2)
- SLIDING 252d windows only (HC #0)
- Adversarial validation (HC #705)

Output: /home/jupiter/Lvl3Quant/output/ml_income_optimizer/
"""

import os
import json
import warnings
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/ml_income_optimizer")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──
INITIAL_CAPITAL = 100_000
ES_TICK_VALUE = 12.50
RISK_FREE_ANNUAL = 0.045  # ~current T-bill yield approximation
TRAIN_WINDOW = 252  # 1 year sliding
FORWARD_WINDOW = 21  # 21 trading days (~1 month)
BS_HAIRCUT = 0.25  # HC #713 R2: 25% haircut on BS prices

# Put selling parameters
PUT_DELTA = 0.30
PUT_DTE = 30
PREMIUM_PCT_APPROX = 0.01  # ~1% of notional for 30-delta put, adjusted by model
PROFIT_TARGET = 0.50  # Close at 50% profit
LOSS_LIMIT = 2.00  # Close at 200% loss
SPY_MULTIPLIER = 100  # Options multiplier


def download_data():
    """Download daily data via yfinance (2010-2026)."""
    tickers = ["SPY", "QQQ", "^VIX", "^VIX3M", "TLT", "HYG", "IEF", "GLD"]
    cache_path = OUTPUT_DIR / "raw_data.parquet"

    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        df.columns = [c.replace("^", "") for c in df.columns]
        print(f"  Loaded cached data: {len(df)} rows, {df.index[0].date()} to {df.index[-1].date()}")
        return df

    print("  Downloading from yfinance...")
    data = {}
    for t in tickers:
        print(f"    Fetching {t}...")
        try:
            d = yf.download(t, start="2010-01-01", end="2026-07-18", progress=False)
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            data[t] = d["Close"].rename(t.replace("^", ""))
        except Exception as e:
            print(f"    ERROR fetching {t}: {e}")

    df = pd.DataFrame(data)
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df.dropna()
    df.to_parquet(cache_path)
    print(f"  Downloaded {len(df)} rows, {df.index[0].date()} to {df.index[-1].date()}")
    return df


def build_features(df):
    """Build feature matrix for premium selling conditions classifier."""
    feat = pd.DataFrame(index=df.index)

    # ── VIX features ──
    feat["vix"] = df["VIX"]
    feat["vix_pct_21d"] = df["VIX"].rolling(21).apply(lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-8))
    feat["vix_pct_63d"] = df["VIX"].rolling(63).apply(lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-8))
    feat["vix_pct_252d"] = df["VIX"].rolling(252).apply(lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-8))

    # VIX term structure slope (VIX3M vs VIX)
    if "VIX3M" in df.columns:
        feat["vix_term_slope"] = (df["VIX3M"] - df["VIX"]) / df["VIX"]
    else:
        feat["vix_term_slope"] = 0.0

    # VIX momentum
    feat["vix_chg_1d"] = df["VIX"].pct_change(1)
    feat["vix_chg_5d"] = df["VIX"].pct_change(5)
    feat["vix_chg_10d"] = df["VIX"].pct_change(10)

    # ── SPY features ──
    spy_ret = df["SPY"].pct_change()
    feat["spy_ret_1d"] = spy_ret
    feat["spy_ret_5d"] = df["SPY"].pct_change(5)
    feat["spy_ret_21d"] = df["SPY"].pct_change(21)

    # Realized vol
    feat["spy_rvol_10d"] = spy_ret.rolling(10).std() * np.sqrt(252) * 100
    feat["spy_rvol_21d"] = spy_ret.rolling(21).std() * np.sqrt(252) * 100

    # Max drawdown trailing 21d
    def rolling_max_dd(prices, window=21):
        dd = pd.Series(index=prices.index, dtype=float)
        for i in range(window, len(prices)):
            w = prices.iloc[i - window : i + 1]
            peak = w.cummax()
            drawdown = (w - peak) / peak
            dd.iloc[i] = drawdown.min()
        return dd

    feat["spy_maxdd_21d"] = rolling_max_dd(df["SPY"], 21)

    # ── IV-RV spread (volatility risk premium) ──
    feat["iv_rv_spread"] = df["VIX"] - feat["spy_rvol_21d"]

    # ── Credit spread: HYG - IEF ──
    hyg_ret = df["HYG"].pct_change(21)
    ief_ret = df["IEF"].pct_change(21)
    feat["credit_spread"] = hyg_ret - ief_ret
    feat["credit_spread_zscore"] = (
        feat["credit_spread"] - feat["credit_spread"].rolling(63).mean()
    ) / (feat["credit_spread"].rolling(63).std() + 1e-8)

    # ── Bonds: TLT trend ──
    feat["tlt_ret_21d"] = df["TLT"].pct_change(21)

    # ── Gold momentum (flight to safety) ──
    feat["gld_ret_21d"] = df["GLD"].pct_change(21)

    # ── Calendar features ──
    feat["month"] = df.index.month
    feat["dow"] = df.index.dayofweek

    # ── QQQ relative strength ──
    feat["qqq_spy_ratio_21d"] = df["QQQ"].pct_change(21) - df["SPY"].pct_change(21)

    return feat


def build_target(df):
    """
    Build proxy target for premium selling conditions.
    GOOD (1): VIX > 20 AND next 21 days SPY doesn't drop > 3%
    BAD (-1): Next 21 days SPY drops > 5%
    NEUTRAL (0): Everything else
    """
    spy = df["SPY"]
    vix = df["VIX"]
    target = pd.Series(0, index=df.index, name="target")

    # Forward-looking max drawdown over next 21 days
    fwd_min = spy.rolling(FORWARD_WINDOW).min().shift(-FORWARD_WINDOW)
    fwd_dd = (fwd_min - spy) / spy

    # GOOD: VIX > 20 AND forward drawdown > -3% (i.e., calm)
    good_mask = (vix > 20) & (fwd_dd > -0.03)
    target[good_mask] = 1

    # BAD: forward drawdown <= -5%
    bad_mask = fwd_dd <= -0.05
    target[bad_mask] = -1

    return target


def bs_put_price(S, K, T, r, sigma):
    """Black-Scholes put price with 25% haircut (HC #713 R2)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    # Apply 25% haircut
    return price * (1 - BS_HAIRCUT)


def bs_put_delta(S, K, T, r, sigma):
    """Black-Scholes put delta."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) - 1


def find_strike_for_delta(S, target_delta, T, r, sigma, tol=0.001):
    """Find strike that gives target put delta via bisection."""
    lo, hi = S * 0.70, S * 1.05
    for _ in range(100):
        mid = (lo + hi) / 2
        d = bs_put_delta(S, mid, T, r, sigma)
        if abs(d - target_delta) < tol:
            return mid
        if d < target_delta:
            lo = mid
        else:
            hi = mid
    return mid


def walk_forward_train(features, target, model_type="lgbm"):
    """
    Walk-forward with SLIDING 252d window.
    Returns predictions aligned to dates.
    """
    valid_mask = features.notna().all(axis=1) & target.notna()
    features = features[valid_mask].copy()
    target = target[valid_mask].copy()

    preds = pd.Series(index=features.index, dtype=float)
    preds[:] = np.nan

    feat_cols = features.columns.tolist()
    n = len(features)

    fold_count = 0
    for i in range(TRAIN_WINDOW, n):
        train_start = i - TRAIN_WINDOW
        train_end = i

        X_train = features.iloc[train_start:train_end][feat_cols].values
        y_train = target.iloc[train_start:train_end].values

        X_test = features.iloc[i:i+1][feat_cols].values

        # Map target: -1 -> 0, 0 -> 1, 1 -> 2 for multiclass
        y_mapped = y_train + 1  # {-1,0,1} -> {0,1,2}

        try:
            if model_type == "lgbm":
                model = lgb.LGBMClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.05,
                    num_leaves=15,
                    min_child_samples=20,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbose=-1,
                    force_col_wise=True,
                )
                model.fit(X_train, y_mapped)
                pred = model.predict(X_test)[0]
                preds.iloc[i] = pred - 1  # back to {-1,0,1}

            elif model_type == "rf":
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=4,
                    min_samples_leaf=20,
                    random_state=42,
                    n_jobs=-1,
                )
                model.fit(X_train, y_mapped)
                pred = model.predict(X_test)[0]
                preds.iloc[i] = pred - 1

            elif model_type == "lr":
                scaler = StandardScaler()
                X_train_sc = scaler.fit_transform(X_train)
                X_test_sc = scaler.transform(X_test)
                model = LogisticRegression(
                    max_iter=500,
                    random_state=42,
                    multi_class="multinomial",
                    C=0.1,
                )
                model.fit(X_train_sc, y_mapped)
                pred = model.predict(X_test_sc)[0]
                preds.iloc[i] = pred - 1

        except Exception:
            preds.iloc[i] = 0  # default to neutral

        fold_count += 1
        if fold_count % 500 == 0:
            print(f"    {model_type}: {fold_count}/{n - TRAIN_WINDOW} folds done")

    # Get feature importance from last model if available
    importances = None
    if model_type == "lgbm" and hasattr(model, "feature_importances_"):
        importances = dict(zip(feat_cols, model.feature_importances_))
    elif model_type == "rf" and hasattr(model, "feature_importances_"):
        importances = dict(zip(feat_cols, model.feature_importances_))

    return preds, importances


def backtest_strategy(df, signals, strategy_name, use_model=True):
    """
    Backtest put-selling income strategy.

    When signal = 1 (GOOD): sell 30-delta put, 30 DTE, collect premium (BS with 25% haircut)
    When signal != 1: sit in T-bills (earn risk-free rate)
    Close at 50% profit or 200% loss or expiration (21 trading days).
    """
    spy = df["SPY"]
    vix = df["VIX"]

    capital = INITIAL_CAPITAL
    daily_equity = []
    trades = []
    position = None  # {entry_date, strike, premium, spy_entry, dte_remaining, notional}

    daily_rf = (1 + RISK_FREE_ANNUAL) ** (1 / 252) - 1

    for i in range(len(spy)):
        date = spy.index[i]

        if i >= len(signals) or pd.isna(signals.iloc[i]):
            daily_equity.append({"date": date, "equity": capital})
            continue

        sig = signals.iloc[i]

        # Manage existing position
        if position is not None:
            position["dte_remaining"] -= 1
            current_spy = spy.iloc[i]
            current_vix = vix.iloc[i] / 100 if i < len(vix) else 0.20

            # Current put value
            T_remaining = max(position["dte_remaining"] / 252, 0.001)
            current_put_value = bs_put_price(
                current_spy, position["strike"], T_remaining,
                RISK_FREE_ANNUAL, current_vix
            )

            # P&L on short put
            pnl_per_contract = (position["premium"] - current_put_value) * SPY_MULTIPLIER

            # Check exit conditions
            close_trade = False
            exit_reason = ""

            if pnl_per_contract >= position["premium"] * SPY_MULTIPLIER * PROFIT_TARGET:
                close_trade = True
                exit_reason = "profit_target"
            elif pnl_per_contract <= -position["premium"] * SPY_MULTIPLIER * LOSS_LIMIT:
                close_trade = True
                exit_reason = "stop_loss"
            elif position["dte_remaining"] <= 0:
                # Expiration: if ITM, assignment loss
                if current_spy < position["strike"]:
                    pnl_per_contract = (position["premium"] - (position["strike"] - current_spy)) * SPY_MULTIPLIER
                else:
                    pnl_per_contract = position["premium"] * SPY_MULTIPLIER
                close_trade = True
                exit_reason = "expiration"

            if close_trade:
                capital += pnl_per_contract * position["num_contracts"]
                trades.append({
                    "entry_date": position["entry_date"],
                    "exit_date": date,
                    "strike": position["strike"],
                    "premium_collected": position["premium"],
                    "pnl_per_contract": pnl_per_contract,
                    "num_contracts": position["num_contracts"],
                    "total_pnl": pnl_per_contract * position["num_contracts"],
                    "exit_reason": exit_reason,
                })
                position = None

        # Open new position if signal is GOOD and no existing position
        if position is None and sig == 1:
            current_spy = spy.iloc[i]
            current_vix = vix.iloc[i] / 100 if i < len(vix) else 0.20

            T = PUT_DTE / 252
            strike = find_strike_for_delta(current_spy, -PUT_DELTA, T, RISK_FREE_ANNUAL, current_vix)

            premium = bs_put_price(current_spy, strike, T, RISK_FREE_ANNUAL, current_vix)

            if premium <= 0:
                daily_equity.append({"date": date, "equity": capital})
                continue

            # Size: max risk = strike * 100 per contract, limit to ~20% of capital
            max_risk_per = strike * SPY_MULTIPLIER
            num_contracts = max(1, int(capital * 0.20 / max_risk_per))

            position = {
                "entry_date": date,
                "strike": round(strike, 2),
                "premium": premium,
                "spy_entry": current_spy,
                "dte_remaining": PUT_DTE,
                "num_contracts": num_contracts,
                "notional": strike * SPY_MULTIPLIER * num_contracts,
            }
        elif position is None:
            # T-bill income when sitting out
            capital *= (1 + daily_rf)

        daily_equity.append({"date": date, "equity": capital})

    eq_df = pd.DataFrame(daily_equity)
    eq_df.set_index("date", inplace=True)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

    return eq_df, trades_df


def compute_metrics(equity_df, label="Strategy"):
    """Compute key performance metrics."""
    eq = equity_df["equity"]
    rets = eq.pct_change().dropna()

    if len(rets) < 2:
        return {"label": label, "error": "insufficient data"}

    years = len(rets) / 252
    total_ret = eq.iloc[-1] / eq.iloc[0] - 1
    cagr = (1 + total_ret) ** (1 / max(years, 0.01)) - 1

    # Drawdown
    peak = eq.cummax()
    dd = (eq - peak) / peak
    max_dd = dd.min()

    # Sharpe
    sharpe = rets.mean() / (rets.std() + 1e-10) * np.sqrt(252)

    # Sortino
    downside = rets[rets < 0]
    sortino = rets.mean() / (downside.std() + 1e-10) * np.sqrt(252)

    # Monthly returns for worst month
    monthly = eq.resample("ME").last().pct_change().dropna()
    worst_month = monthly.min() if len(monthly) > 0 else 0

    return {
        "label": label,
        "total_return_pct": round(total_ret * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "worst_month_pct": round(worst_month * 100, 2) if worst_month != 0 else 0,
        "final_equity": round(eq.iloc[-1], 2),
        "years": round(years, 1),
    }


def always_on_signals(df):
    """Always sell premium (baseline)."""
    return pd.Series(1, index=df.index)


def vix_only_signals(df):
    """Sell only when VIX > 20."""
    return (df["VIX"] > 20).astype(int)


def buy_hold_spy(df):
    """Buy and hold SPY benchmark."""
    spy = df["SPY"]
    eq = INITIAL_CAPITAL * spy / spy.iloc[0]
    eq_df = pd.DataFrame({"equity": eq.values}, index=spy.index)
    return eq_df


def adversarial_validation(features, target, preds, model_name):
    """
    HC #705 adversarial validation:
    1. Permutation test
    2. Sub-period analysis
    3. Outlier sensitivity
    4. R1 regime-agnostic check
    """
    results = {}
    valid_idx = preds.dropna().index
    common_idx = valid_idx.intersection(target.dropna().index)
    p = preds.loc[common_idx]
    t = target.loc[common_idx]

    # 1. Permutation test: accuracy with shuffled labels
    real_acc = accuracy_score(t, p)
    perm_accs = []
    for _ in range(100):
        shuffled = t.sample(frac=1, replace=False).values
        perm_accs.append(accuracy_score(shuffled, p))
    perm_mean = np.mean(perm_accs)
    perm_std = np.std(perm_accs)
    perm_zscore = (real_acc - perm_mean) / (perm_std + 1e-10)
    results["permutation"] = {
        "real_accuracy": round(real_acc, 4),
        "permuted_mean": round(perm_mean, 4),
        "z_score": round(perm_zscore, 2),
        "significant": perm_zscore > 2.0,
    }

    # 2. Sub-period analysis (split into 3 equal periods)
    n = len(p)
    thirds = [slice(0, n // 3), slice(n // 3, 2 * n // 3), slice(2 * n // 3, n)]
    period_accs = []
    for s in thirds:
        acc = accuracy_score(t.iloc[s], p.iloc[s])
        period_accs.append(round(acc, 4))
    results["sub_period"] = {
        "period_accuracies": period_accs,
        "stability": round(np.std(period_accs), 4),
        "stable": np.std(period_accs) < 0.05,
    }

    # 3. Outlier sensitivity: remove top/bottom 5% VIX days
    feat_valid = features.loc[common_idx]
    if "vix" in feat_valid.columns:
        vix_vals = feat_valid["vix"]
        q05, q95 = vix_vals.quantile(0.05), vix_vals.quantile(0.95)
        normal_mask = (vix_vals >= q05) & (vix_vals <= q95)
        if normal_mask.sum() > 100:
            acc_normal = accuracy_score(t[normal_mask], p[normal_mask])
            acc_outlier = accuracy_score(t[~normal_mask], p[~normal_mask]) if (~normal_mask).sum() > 10 else None
            results["outlier_sensitivity"] = {
                "accuracy_normal": round(acc_normal, 4),
                "accuracy_outlier": round(acc_outlier, 4) if acc_outlier else "N/A",
                "robust": abs(acc_normal - (acc_outlier or acc_normal)) < 0.10,
            }

    # 4. R1: Regime-agnostic check
    # Split by SPY 21d return: up regime vs down regime
    if "spy_ret_21d" in feat_valid.columns:
        spy_21 = feat_valid["spy_ret_21d"]
        up_mask = spy_21 > 0
        down_mask = spy_21 <= 0
        if up_mask.sum() > 50 and down_mask.sum() > 50:
            acc_up = accuracy_score(t[up_mask], p[up_mask])
            acc_down = accuracy_score(t[down_mask], p[down_mask])
            regime_gap = abs(acc_up - acc_down) / max(acc_up, acc_down)
            results["regime_agnostic"] = {
                "accuracy_up_regime": round(acc_up, 4),
                "accuracy_down_regime": round(acc_down, 4),
                "regime_gap_ratio": round(regime_gap, 4),
                "pass_r1": regime_gap < 0.50,
            }

    return results


def compute_premium_capture_rate(trades_df):
    """What % of collected premium was kept as profit."""
    if trades_df.empty:
        return 0.0
    total_collected = (trades_df["premium_collected"] * trades_df["num_contracts"] * SPY_MULTIPLIER).sum()
    total_pnl = trades_df["total_pnl"].sum()
    if total_collected == 0:
        return 0.0
    return round(total_pnl / total_collected * 100, 2)


def main():
    print("=" * 70)
    print("ML INCOME OPTIMIZER — Premium Selling Timing Model")
    print("=" * 70)
    print(f"  Capital: ${INITIAL_CAPITAL:,} (fixed, no DCA — HC #713)")
    print(f"  BS pricing with {int(BS_HAIRCUT*100)}% haircut (HC #713 R2)")
    print(f"  Sliding {TRAIN_WINDOW}d window (HC #0)")
    print()

    # ── Step 1: Download data ──
    print("[1/7] Downloading data...")
    df = download_data()

    # ── Step 2: Build features ──
    print("[2/7] Building features...")
    features = build_features(df)
    print(f"  {len(features.columns)} features built")

    # ── Step 3: Build target ──
    print("[3/7] Building target labels...")
    target = build_target(df)
    good_pct = (target == 1).mean() * 100
    bad_pct = (target == -1).mean() * 100
    neutral_pct = (target == 0).mean() * 100
    print(f"  GOOD (sell premium): {good_pct:.1f}%")
    print(f"  BAD (sit out): {bad_pct:.1f}%")
    print(f"  NEUTRAL: {neutral_pct:.1f}%")

    # ── Step 4: Walk-forward training ──
    print("[4/7] Walk-forward training (SLIDING 252d window)...")
    models = {}
    importances = {}
    for name in ["lgbm", "rf", "lr"]:
        print(f"  Training {name.upper()}...")
        preds, imp = walk_forward_train(features, target, model_type=name)
        models[name] = preds
        if imp:
            importances[name] = imp
        valid_idx = preds.dropna().index
        common_idx = valid_idx.intersection(target.dropna().index)
        if len(common_idx) > 0:
            acc = accuracy_score(target.loc[common_idx], preds.loc[common_idx])
            print(f"    {name.upper()} OOT accuracy: {acc:.4f}")

    # ── Step 5: Backtest all strategies ──
    print("[5/7] Backtesting strategies...")
    all_results = {}
    all_trades = {}

    # ML models
    for name, sigs in models.items():
        # Convert to binary: 1 = sell, else = sit out
        binary_sigs = (sigs == 1).astype(int)
        binary_sigs[sigs.isna()] = np.nan
        eq, trades = backtest_strategy(df, binary_sigs, f"ML-{name.upper()}")
        all_results[f"ML-{name.upper()}"] = eq
        all_trades[f"ML-{name.upper()}"] = trades
        print(f"  ML-{name.upper()}: {len(trades)} trades")

    # Baselines
    print("  Running baselines...")
    eq_always, trades_always = backtest_strategy(df, always_on_signals(df), "Always-On")
    all_results["Always-On"] = eq_always
    all_trades["Always-On"] = trades_always
    print(f"  Always-On: {len(trades_always)} trades")

    eq_vix, trades_vix = backtest_strategy(df, vix_only_signals(df), "VIX>20 Only")
    all_results["VIX>20 Only"] = eq_vix
    all_trades["VIX>20 Only"] = trades_vix
    print(f"  VIX>20 Only: {len(trades_vix)} trades")

    eq_bh = buy_hold_spy(df)
    all_results["Buy-Hold SPY"] = eq_bh

    # ── Step 6: Compute metrics ──
    print("[6/7] Computing metrics...")
    metrics_table = []
    for name, eq in all_results.items():
        m = compute_metrics(eq, name)
        if name in all_trades and not all_trades[name].empty:
            m["num_trades"] = len(all_trades[name])
            m["win_rate"] = round((all_trades[name]["total_pnl"] > 0).mean() * 100, 1)
            m["premium_capture_pct"] = compute_premium_capture_rate(all_trades[name])
            m["avg_trade_pnl"] = round(all_trades[name]["total_pnl"].mean(), 2)
        metrics_table.append(m)

    metrics_df = pd.DataFrame(metrics_table)
    print()
    print("=" * 70)
    print("RESULTS — ALL STRATEGIES")
    print("=" * 70)
    print(metrics_df.to_string(index=False))
    print()

    # ── Step 7: Adversarial validation ──
    print("[7/7] Adversarial validation (HC #705)...")
    adv_results = {}
    for name, sigs in models.items():
        print(f"  Validating {name.upper()}...")
        adv = adversarial_validation(features, target, sigs, name)
        adv_results[name] = adv

        for test_name, result in adv.items():
            status = "PASS" if result.get("significant", result.get("stable", result.get("robust", result.get("pass_r1", False)))) else "FAIL"
            print(f"    {test_name}: {status}")

    # ── Feature importance ──
    print()
    print("FEATURE IMPORTANCE (LightGBM, last fold):")
    if "lgbm" in importances and importances["lgbm"]:
        sorted_imp = sorted(importances["lgbm"].items(), key=lambda x: x[1], reverse=True)
        for feat_name, imp_val in sorted_imp[:15]:
            print(f"  {feat_name:25s}: {imp_val}")

    # ── Trade analysis ──
    print()
    print("TRADE ANALYSIS:")
    for name, trades in all_trades.items():
        if trades.empty:
            continue
        print(f"\n  {name}:")
        print(f"    Total trades: {len(trades)}")
        if "exit_reason" in trades.columns:
            reasons = trades["exit_reason"].value_counts()
            for r, c in reasons.items():
                print(f"    Exit {r}: {c} ({c/len(trades)*100:.1f}%)")
        winners = (trades["total_pnl"] > 0).sum()
        losers = (trades["total_pnl"] <= 0).sum()
        print(f"    Winners: {winners} ({winners/len(trades)*100:.1f}%)")
        print(f"    Losers: {losers} ({losers/len(trades)*100:.1f}%)")
        print(f"    Avg P&L per trade: ${trades['total_pnl'].mean():.2f}")
        print(f"    Best trade: ${trades['total_pnl'].max():.2f}")
        print(f"    Worst trade: ${trades['total_pnl'].min():.2f}")

    # ── Save results ──
    print()
    print("Saving results...")

    # Save metrics
    metrics_df.to_csv(OUTPUT_DIR / "metrics_comparison.csv", index=False)

    # Save equity curves
    for name, eq in all_results.items():
        safe_name = name.lower().replace(" ", "_").replace(">", "gt")
        eq.to_parquet(OUTPUT_DIR / f"equity_{safe_name}.parquet")

    # Save trade logs
    for name, trades in all_trades.items():
        if not trades.empty:
            safe_name = name.lower().replace(" ", "_").replace(">", "gt")
            trades.to_csv(OUTPUT_DIR / f"trades_{safe_name}.csv", index=False)

    # Save adversarial results
    with open(OUTPUT_DIR / "adversarial_validation.json", "w") as f:
        json.dump(adv_results, f, indent=2, default=str)

    # Save full report
    report = {
        "timestamp": dt.datetime.now().isoformat(),
        "parameters": {
            "initial_capital": INITIAL_CAPITAL,
            "train_window": TRAIN_WINDOW,
            "forward_window": FORWARD_WINDOW,
            "bs_haircut_pct": BS_HAIRCUT * 100,
            "put_delta": PUT_DELTA,
            "put_dte": PUT_DTE,
            "profit_target_pct": PROFIT_TARGET * 100,
            "loss_limit_pct": LOSS_LIMIT * 100,
        },
        "target_distribution": {
            "good_pct": round(good_pct, 1),
            "bad_pct": round(bad_pct, 1),
            "neutral_pct": round(neutral_pct, 1),
        },
        "metrics": metrics_table,
        "adversarial_validation": adv_results,
        "feature_importance": importances.get("lgbm", {}),
    }
    with open(OUTPUT_DIR / "full_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nAll outputs saved to {OUTPUT_DIR}/")
    print("DONE.")

    return report


if __name__ == "__main__":
    report = main()
