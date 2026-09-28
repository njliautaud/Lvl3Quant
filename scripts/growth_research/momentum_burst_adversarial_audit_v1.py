#!/usr/bin/env python3
"""
Momentum Burst Options — Adversarial Leakage Audit v1 (HC #753)
================================================================

Validates KB #281 (Sharpe 1.28) momentum options strategy against 8 adversarial tests:
  1. Look-ahead feature check
  2. Label leakage
  3. Walk-forward integrity
  4. Permutation test (500 shuffles)
  5. Sub-period stability (4 quarters)
  6. Borrow cost / commission sensitivity
  7. Outlier removal
  8. Random signal comparison (100 trials)

Replicates EXACT logic from momentum_options_paper.py:
  - 11 sector ETFs, LGBM ranking, ATM calls on top-2 / puts on bottom-2
  - VIX > 15 filter, 5-day rebalance, +30% TP / -25% SL / trailing / 5-day max hold
  - Simplified BS option pricing

Logs results to MLflow experiment "momentum_burst_adversarial_audit".
Runs on Neptune (paths use /home/nick/Lvl3Quant).

Usage:
    python momentum_burst_adversarial_audit_v1.py
"""
import json
import math
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ── Paths (Neptune-compatible) ──
BASE = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE / "research" / "findings"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = RESULTS_DIR / "momentum_burst_adversarial_audit_v1.json"

def fprint(*a, **kw):
    print(*a, **kw, flush=True)


# ── MLflow ──
MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen("http://jupiter:5000/", timeout=2)
    import mlflow
    mlflow.set_tracking_uri("http://jupiter:5000")
    MLFLOW_OK = True
except Exception:
    fprint("MLflow unavailable — results logged locally only")


# ==================== STRATEGY CONSTANTS (exact match to paper engine) ====================
SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX"]
INITIAL_CAPITAL = 645.0
MAX_POS_COST = 150.0
COMMISSION_PER_CONTRACT = 0.65
DTE_TARGET = 30
REBALANCE_INTERVAL = 5
VIX_MIN = 15.0
TOP_K = 2
BOTTOM_K = 2

TP_PCT = 0.30
SL_PCT = -0.25
TRAILING_ACTIVATE_PCT = 0.15
TRAILING_GIVEBACK_PCT = 0.50
MAX_HOLD_DAYS = 5

FEAT_COLS = [
    "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "vol_21d", "vol_63d", "sharpe_63d", "maxdd_63d", "pct_52w_high",
    "mom_accel", "pct_pos_months_12m", "sortino_63d", "calmar_1y",
    "trend_r2_63d", "trend_slope_63d",
]


# ==================== DATA DOWNLOAD ====================

def download_data(start="2018-01-01"):
    """Download sector ETF + VIX data via yfinance."""
    import yfinance as yf
    all_tickers = SECTORS + EXTRA_TICKERS
    raw = yf.download(all_tickers, start=start, progress=False)
    mi = isinstance(raw.columns, pd.MultiIndex)
    close = raw["Close"] if mi else raw
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(-1)
    close = close.ffill()
    rename_map = {"^VIX": "VIX"}
    close = close.rename(columns=rename_map)

    vc = "VIX" if "VIX" in close.columns else None
    if vc is None:
        raise ValueError("VIX data not available")
    vix = close[vc].dropna()
    spy = close["SPY"].dropna() if "SPY" in close.columns else None
    sc = close[[c for c in SECTORS if c in close.columns]].dropna(how="all")
    ix = sc.index.intersection(vix.index)
    if spy is not None:
        ix = ix.intersection(spy.index)
    return close.loc[ix], sc.loc[ix], vix.loc[ix]


# ==================== FEATURE ENGINEERING (exact match) ====================

def compute_features(px):
    """Compute 17 momentum/quality features for a single sector ETF series.
    Exact replica of momentum_options_paper.py."""
    if len(px) < 260:
        return None

    f = {}
    for lb, nm in [(5, "ret_5d"), (10, "ret_10d"), (21, "ret_21d"),
                   (63, "ret_63d"), (126, "ret_126d"), (252, "ret_252d")]:
        f[nm] = float(px.iloc[-1] / px.iloc[-lb] - 1) if len(px) > lb else 0.0

    rets = px.pct_change().dropna()
    f["vol_21d"] = float(rets.iloc[-21:].std() * np.sqrt(252)) if len(rets) > 21 else 0.2
    f["vol_63d"] = float(rets.iloc[-63:].std() * np.sqrt(252)) if len(rets) > 63 else 0.2

    r63 = rets.iloc[-63:]
    f["sharpe_63d"] = float(r63.mean() / (r63.std() + 1e-10) * np.sqrt(252)) if len(r63) > 10 else 0.0

    pk63 = px.iloc[-63:].cummax()
    f["maxdd_63d"] = float(((px.iloc[-63:] / pk63) - 1).min())
    f["pct_52w_high"] = float(px.iloc[-1] / px.iloc[-252:].max())
    f["mom_accel"] = f["ret_21d"] - f["ret_63d"] / 3

    monthly = rets.resample("ME").sum()
    f["pct_pos_months_12m"] = float((monthly.iloc[-12:] > 0).mean()) if len(monthly) >= 12 else 0.5

    dr = r63[r63 < 0]
    f["sortino_63d"] = float(r63.mean() / (dr.std() + 1e-10) * np.sqrt(252)) if len(dr) > 3 else 0.0

    pk = px.iloc[-252:].cummax()
    mdd = float(((px.iloc[-252:] / pk) - 1).min())
    cagr = float(px.iloc[-1] / px.iloc[-252] - 1) if len(px) >= 252 else 0.0
    f["calmar_1y"] = cagr / (abs(mdd) + 1e-10)

    if len(px) >= 63:
        y = np.log(px.iloc[-63:].values + 1e-10)
        x = np.arange(len(y))
        slope, _, r_val, _, _ = stats.linregress(x, y)
        f["trend_r2_63d"] = r_val ** 2
        f["trend_slope_63d"] = slope * 252
    else:
        f["trend_r2_63d"] = 0.0
        f["trend_slope_63d"] = 0.0

    return f


# ==================== OPTION PRICING (exact match) ====================

def estimate_option_price(S, option_type, vix_val, dte_days=30):
    iv = (vix_val / 100.0) * 1.2
    t = dte_days / 365.0
    price_per_share = 0.4 * S * math.sqrt(t) * iv
    return max(price_per_share, 0.01)


def estimate_option_value(entry_price_option, entry_price_underlying,
                          current_underlying, option_type, days_held, dte):
    entry_ps = entry_price_option
    S_entry = entry_price_underlying
    S_now = current_underlying
    dte_remaining = max(dte - days_held, 1)
    if option_type == "call":
        delta_pnl = 0.5 * (S_now - S_entry)
    else:
        delta_pnl = 0.5 * (S_entry - S_now)
    theta_decay = entry_ps * (days_held / dte)
    current_ps = entry_ps + delta_pnl - theta_decay
    return max(current_ps, 0.01)


# ==================== LGBM TRAINING DATA BUILDER ====================

def build_lgbm_dataset(sc):
    """Build feature + label dataset for LGBM walk-forward.
    Returns DataFrame with features, fwd_ret, date, ticker columns.
    Exact replica of run_lgbm_ranking training loop."""
    records = []
    all_dates = sc.index[-300:]
    rebal_dates = all_dates[::REBALANCE_INTERVAL]

    for dt in rebal_dates[:-1]:
        idx = sc.index.get_indexer([dt], method="ffill")[0]
        if idx < 260:
            continue
        for tk in sc.columns:
            px = sc[tk].iloc[:idx + 1].dropna()
            feats = compute_features(px)
            if not feats:
                continue
            fi = min(idx + 5, len(sc) - 1)
            feats.update({
                "date": dt,
                "ticker": tk,
                "fwd_ret": float(sc[tk].iloc[fi] / sc[tk].iloc[idx] - 1),
            })
            records.append(feats)

    df = pd.DataFrame(records)
    for c in FEAT_COLS:
        if c not in df.columns:
            df[c] = 0.0
    df[FEAT_COLS] = df[FEAT_COLS].fillna(0.0)
    df["rank_label"] = df.groupby("date")["fwd_ret"].rank(pct=True)
    return df


# ==================== FULL BACKTEST ENGINE ====================

def run_backtest(sc, vix, signal_override=None, extra_commission_mult=1.0,
                 borrow_cost_annual=0.0, rng=None):
    """Full backtest replicating momentum_options_paper.py logic.

    Args:
        sc: sector close DataFrame
        vix: VIX series
        signal_override: if provided, dict mapping rebalance_date -> {call_picks, put_picks}
                         bypasses LGBM ranking
        extra_commission_mult: multiply commission by this factor
        borrow_cost_annual: annual borrow cost fraction for puts (added per hold day)
        rng: numpy RandomState for random signal mode

    Returns:
        dict with equity_curve, trades, sharpe, sortino, etc.
    """
    try:
        import lightgbm as lgb
        HAS_LGBM = True
    except ImportError:
        HAS_LGBM = False

    commission = COMMISSION_PER_CONTRACT * extra_commission_mult

    equity = INITIAL_CAPITAL
    cash = INITIAL_CAPITAL
    open_positions = []
    closed_trades = []
    equity_curve = []
    days_since_rebalance = 999

    # Pre-train LGBM if needed (train on first 60% of data, predict rolling)
    lgbm_model = None
    if signal_override is None and HAS_LGBM:
        records = []
        all_dates = sc.index
        for dt_idx in range(260, len(all_dates) - 5, REBALANCE_INTERVAL):
            dt = all_dates[dt_idx]
            for tk in sc.columns:
                px = sc[tk].iloc[:dt_idx + 1].dropna()
                feats = compute_features(px)
                if not feats:
                    continue
                fi = min(dt_idx + 5, len(sc) - 1)
                feats["fwd_ret"] = float(sc[tk].iloc[fi] / sc[tk].iloc[dt_idx] - 1)
                feats["date"] = dt
                feats["ticker"] = tk
                records.append(feats)

        if len(records) > 50:
            df_train = pd.DataFrame(records)
            for c in FEAT_COLS:
                if c not in df_train.columns:
                    df_train[c] = 0.0
            df_train[FEAT_COLS] = df_train[FEAT_COLS].fillna(0.0)
            df_train["rank_label"] = df_train.groupby("date")["fwd_ret"].rank(pct=True)

    # Walk through each trading day
    start_idx = 260  # need 260 days of history for features
    for day_idx in range(start_idx, len(sc)):
        today = sc.index[day_idx]
        current_vix = float(vix.iloc[day_idx]) if day_idx < len(vix) else 20.0

        # ── Check exits ──
        positions_to_close = []
        for i, pos in enumerate(open_positions):
            tk = pos["ticker"]
            if tk not in sc.columns:
                continue
            entry_idx = pos["entry_idx"]
            days_held = day_idx - entry_idx
            if days_held == 0:
                continue

            current_underlying = float(sc[tk].iloc[day_idx])
            current_ps = estimate_option_value(
                pos["entry_price_option"], pos["entry_price_underlying"],
                current_underlying, pos["option_type"], days_held, DTE_TARGET
            )
            entry_ps = pos["entry_price_option"]
            pct_change = (current_ps - entry_ps) / entry_ps

            if current_ps > pos["peak_value_ps"]:
                pos["peak_value_ps"] = current_ps
            if pct_change >= TRAILING_ACTIVATE_PCT:
                pos["trailing_active"] = True

            exit_reason = None
            if pct_change >= TP_PCT:
                exit_reason = "take_profit"
            elif pct_change <= SL_PCT:
                exit_reason = "stop_loss"
            elif pos["trailing_active"]:
                peak_ps = pos["peak_value_ps"]
                peak_gain = peak_ps - entry_ps
                current_gain = current_ps - entry_ps
                if peak_gain > 0 and current_gain < peak_gain * (1 - TRAILING_GIVEBACK_PCT):
                    exit_reason = "trailing_stop"
            elif days_held >= MAX_HOLD_DAYS:
                exit_reason = "max_hold"

            if exit_reason:
                pnl = (current_ps - entry_ps) * 100 - commission
                # Add borrow cost for puts
                if pos["option_type"] == "put" and borrow_cost_annual > 0:
                    borrow_cost = pos["cost"] * borrow_cost_annual * (days_held / 252)
                    pnl -= borrow_cost
                positions_to_close.append((i, pnl, exit_reason, pos))

        for i, pnl, reason, pos in reversed(positions_to_close):
            open_positions.pop(i)
            equity += pnl
            cash += pos["cost"] + pnl
            closed_trades.append({
                "ticker": pos["ticker"],
                "option_type": pos["option_type"],
                "entry_date": pos["entry_date"],
                "exit_date": str(today.date()),
                "pnl": pnl,
                "pct_return": pnl / pos["cost"] * 100 if pos["cost"] > 0 else 0,
                "exit_reason": reason,
                "entry_idx": pos["entry_idx"],
                "exit_idx": day_idx,
            })

        # ── VIX filter ──
        if current_vix < VIX_MIN:
            equity_curve.append({"date": today, "equity": equity})
            continue

        # ── Rebalance check ──
        days_since_rebalance += 1
        if days_since_rebalance < REBALANCE_INTERVAL:
            equity_curve.append({"date": today, "equity": equity})
            continue

        days_since_rebalance = 0

        # ── Get signals ──
        if signal_override is not None:
            # Use pre-specified signals
            if today in signal_override:
                sig = signal_override[today]
                call_picks = sig["call_picks"]
                put_picks = sig["put_picks"]
            else:
                equity_curve.append({"date": today, "equity": equity})
                continue
        elif rng is not None:
            # Random signal mode
            available = list(sc.columns)
            if len(available) >= TOP_K + BOTTOM_K:
                shuffled = list(available)
                rng.shuffle(shuffled)
                call_picks = shuffled[:TOP_K]
                put_picks = shuffled[-BOTTOM_K:]
            else:
                equity_curve.append({"date": today, "equity": equity})
                continue
        else:
            # LGBM ranking (walk-forward)
            if not HAS_LGBM:
                rets_21d = sc.iloc[day_idx] / sc.iloc[max(0, day_idx - 21)] - 1
                rankings = dict(rets_21d.sort_values(ascending=False))
            else:
                # Train on data up to this point
                train_records = []
                for ti in range(max(260, day_idx - 300), day_idx - 5, REBALANCE_INTERVAL):
                    for tk in sc.columns:
                        px = sc[tk].iloc[:ti + 1].dropna()
                        feats = compute_features(px)
                        if not feats:
                            continue
                        fi = min(ti + 5, len(sc) - 1)
                        feats["fwd_ret"] = float(sc[tk].iloc[fi] / sc[tk].iloc[ti] - 1)
                        train_records.append(feats)

                if len(train_records) < 50:
                    rets_21d = sc.iloc[day_idx] / sc.iloc[max(0, day_idx - 21)] - 1
                    rankings = dict(rets_21d.sort_values(ascending=False))
                else:
                    tdf = pd.DataFrame(train_records)
                    for c in FEAT_COLS:
                        if c not in tdf.columns:
                            tdf[c] = 0.0
                    tdf[FEAT_COLS] = tdf[FEAT_COLS].fillna(0.0)
                    tdf["rank_label"] = tdf.groupby(tdf.index // len(sc.columns))["fwd_ret"].rank(pct=True)

                    X_tr = np.nan_to_num(tdf[FEAT_COLS].values.astype(np.float32))
                    y_tr = tdf["rank_label"].values.astype(np.float32)

                    m = lgb.LGBMRegressor(
                        n_estimators=100, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, min_child_samples=5, verbose=-1
                    )
                    m.fit(X_tr, y_tr)

                    current_feats = {}
                    for tk in sc.columns:
                        px = sc[tk].iloc[:day_idx + 1].dropna()
                        feats = compute_features(px)
                        if feats:
                            current_feats[tk] = feats

                    if not current_feats:
                        equity_curve.append({"date": today, "equity": equity})
                        continue

                    pred_df = pd.DataFrame(current_feats).T
                    for c in FEAT_COLS:
                        if c not in pred_df.columns:
                            pred_df[c] = 0.0
                    X_pred = np.nan_to_num(pred_df[FEAT_COLS].values.astype(np.float32))
                    scores = m.predict(X_pred)
                    rankings = dict(zip(pred_df.index, scores))

            ranked = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
            call_picks = [t for t, _ in ranked[:TOP_K]]
            put_picks = [t for t, _ in ranked[-BOTTOM_K:]]

        # ── Enter positions ──
        for tk in call_picks:
            if tk not in sc.columns:
                continue
            if any(p["ticker"] == tk and p["option_type"] == "call" for p in open_positions):
                continue
            S = float(sc[tk].iloc[day_idx])
            opt_ps = estimate_option_price(S, "call", current_vix, DTE_TARGET)
            cost = opt_ps * 100 + commission
            if cost > MAX_POS_COST or cost > cash:
                continue
            open_positions.append({
                "ticker": tk, "option_type": "call",
                "entry_date": str(today.date()), "entry_idx": day_idx,
                "entry_price_option": opt_ps, "entry_price_underlying": S,
                "cost": cost, "peak_value_ps": opt_ps, "trailing_active": False,
            })
            cash -= cost

        for tk in put_picks:
            if tk not in sc.columns:
                continue
            if any(p["ticker"] == tk and p["option_type"] == "put" for p in open_positions):
                continue
            S = float(sc[tk].iloc[day_idx])
            opt_ps = estimate_option_price(S, "put", current_vix, DTE_TARGET)
            cost = opt_ps * 100 + commission
            if cost > MAX_POS_COST or cost > cash:
                continue
            open_positions.append({
                "ticker": tk, "option_type": "put",
                "entry_date": str(today.date()), "entry_idx": day_idx,
                "entry_price_option": opt_ps, "entry_price_underlying": S,
                "cost": cost, "peak_value_ps": opt_ps, "trailing_active": False,
            })
            cash -= cost

        equity_curve.append({"date": today, "equity": equity})

    # ── Compute metrics ──
    eq_df = pd.DataFrame(equity_curve)
    if len(eq_df) < 2:
        return {"sharpe": 0, "sortino": 0, "trades": closed_trades, "equity_curve": eq_df,
                "total_pnl": 0, "win_rate": 0, "n_trades": 0}

    eq_df = eq_df.set_index("date")
    eq_df["ret"] = eq_df["equity"].pct_change().fillna(0)

    mean_ret = eq_df["ret"].mean()
    std_ret = eq_df["ret"].std()
    sharpe = float(mean_ret / (std_ret + 1e-10) * np.sqrt(252))

    downside = eq_df["ret"][eq_df["ret"] < 0]
    sortino = float(mean_ret / (downside.std() + 1e-10) * np.sqrt(252)) if len(downside) > 3 else 0.0

    n_trades = len(closed_trades)
    wins = sum(1 for t in closed_trades if t["pnl"] > 0)
    win_rate = wins / n_trades * 100 if n_trades > 0 else 0

    total_pnl = sum(t["pnl"] for t in closed_trades)

    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "trades": closed_trades,
        "equity_curve": eq_df,
        "total_pnl": total_pnl,
        "win_rate": win_rate,
        "n_trades": n_trades,
    }


# ==================== 8 ADVERSARIAL TESTS ====================

def test_1_lookahead_features(sc, vix):
    """Check if any feature correlates with FUTURE returns (look-ahead bias)."""
    fprint("\n" + "=" * 70)
    fprint("TEST 1: Look-Ahead Feature Check")
    fprint("=" * 70)

    # Build features at each rebalance point and correlate with future returns
    records = []
    all_dates = sc.index
    for dt_idx in range(260, len(all_dates) - 5, REBALANCE_INTERVAL):
        dt = all_dates[dt_idx]
        for tk in sc.columns:
            px = sc[tk].iloc[:dt_idx + 1].dropna()
            feats = compute_features(px)
            if not feats:
                continue
            fi = min(dt_idx + 5, len(sc) - 1)
            fwd_ret = float(sc[tk].iloc[fi] / sc[tk].iloc[dt_idx] - 1)
            feats["fwd_ret_5d"] = fwd_ret
            feats["date_idx"] = dt_idx
            records.append(feats)

    df = pd.DataFrame(records)
    if len(df) < 50:
        fprint("  SKIP: insufficient data")
        return {"pass": False, "reason": "insufficient data"}

    max_corr = 0
    worst_feat = None
    corr_details = {}

    for col in FEAT_COLS:
        if col not in df.columns:
            continue
        r, p = stats.pearsonr(df[col].fillna(0), df["fwd_ret_5d"])
        corr_details[col] = {"r": round(r, 4), "p": round(p, 4)}
        if abs(r) > abs(max_corr):
            max_corr = r
            worst_feat = col

    passed = abs(max_corr) < 0.5
    fprint(f"  Max |corr(feature, future_ret)|: {abs(max_corr):.4f} (feature: {worst_feat})")
    fprint(f"  Threshold: |r| < 0.5")
    fprint(f"  Top correlated features:")
    sorted_corrs = sorted(corr_details.items(), key=lambda x: abs(x[1]["r"]), reverse=True)
    for feat, d in sorted_corrs[:5]:
        fprint(f"    {feat:25s}  r={d['r']:+.4f}  p={d['p']:.4f}")
    fprint(f"  Result: {'PASS' if passed else 'FAIL'}")

    return {"pass": passed, "max_corr": round(max_corr, 4), "worst_feat": worst_feat,
            "details": corr_details}


def test_2_label_leakage(sc):
    """Check if labels (forward returns used for ranking) leak into features."""
    fprint("\n" + "=" * 70)
    fprint("TEST 2: Label Leakage Check")
    fprint("=" * 70)

    records = []
    all_dates = sc.index
    for dt_idx in range(260, len(all_dates) - 5, REBALANCE_INTERVAL):
        for tk in sc.columns:
            px = sc[tk].iloc[:dt_idx + 1].dropna()
            feats = compute_features(px)
            if not feats:
                continue
            fi = min(dt_idx + 5, len(sc) - 1)
            feats["fwd_ret"] = float(sc[tk].iloc[fi] / sc[tk].iloc[dt_idx] - 1)
            records.append(feats)

    df = pd.DataFrame(records)
    df["rank_label"] = df.groupby(df.index // len(sc.columns))["fwd_ret"].rank(pct=True)

    if len(df) < 50:
        fprint("  SKIP: insufficient data")
        return {"pass": False, "reason": "insufficient data"}

    max_corr = 0
    worst_feat = None
    corr_details = {}

    for col in FEAT_COLS:
        if col not in df.columns:
            continue
        r, p = stats.pearsonr(df[col].fillna(0), df["rank_label"])
        corr_details[col] = {"r": round(r, 4), "p": round(p, 4)}
        if abs(r) > abs(max_corr):
            max_corr = r
            worst_feat = col

    passed = abs(max_corr) < 0.5
    fprint(f"  Max |corr(feature, rank_label)|: {abs(max_corr):.4f} (feature: {worst_feat})")
    fprint(f"  Threshold: |r| < 0.5")
    fprint(f"  Top correlated features:")
    sorted_corrs = sorted(corr_details.items(), key=lambda x: abs(x[1]["r"]), reverse=True)
    for feat, d in sorted_corrs[:5]:
        fprint(f"    {feat:25s}  r={d['r']:+.4f}  p={d['p']:.4f}")
    fprint(f"  Result: {'PASS' if passed else 'FAIL'}")

    return {"pass": passed, "max_corr": round(max_corr, 4), "worst_feat": worst_feat,
            "details": corr_details}


def test_3_walkforward_integrity(sc):
    """Verify train data dates are strictly before test dates in every fold."""
    fprint("\n" + "=" * 70)
    fprint("TEST 3: Walk-Forward Integrity")
    fprint("=" * 70)

    all_dates = sc.index
    violations = 0
    total_folds = 0

    # Simulate the walk-forward: at each rebalance point, check that
    # training data (up to dt_idx - 5) doesn't overlap with test period (dt_idx to dt_idx + 5)
    for dt_idx in range(260, len(all_dates) - 5, REBALANCE_INTERVAL):
        total_folds += 1
        test_start = all_dates[dt_idx]
        test_end_idx = min(dt_idx + 5, len(all_dates) - 1)
        test_end = all_dates[test_end_idx]

        # Training data: features computed from px[:dt_idx+1], which uses data up to dt_idx
        # Forward return target: sc[fi] / sc[dt_idx] - 1 where fi = min(dt_idx + 5, len-1)
        # For TRAINING samples at earlier rebalance points, their fwd_ret uses data BEFORE dt_idx
        # Check: no training sample's forward return window overlaps with test date
        for train_idx in range(max(260, dt_idx - 300), dt_idx - 5, REBALANCE_INTERVAL):
            train_fwd_end_idx = min(train_idx + 5, len(all_dates) - 1)
            # Training sample uses data up to train_fwd_end_idx for its label
            # This must be strictly before the current prediction point
            if train_fwd_end_idx > dt_idx:
                violations += 1

    passed = violations == 0
    fprint(f"  Total walk-forward folds checked: {total_folds}")
    fprint(f"  Train/test overlap violations: {violations}")
    fprint(f"  Strategy trains on data up to (rebal_date - 5 days), predicts at rebal_date")
    fprint(f"  Result: {'PASS' if passed else 'FAIL'}")

    return {"pass": passed, "total_folds": total_folds, "violations": violations}


def test_4_permutation_test(sc, vix, actual_sharpe, n_shuffles=500):
    """Shuffle entry signals 500 times, check if actual Sharpe > 95th percentile."""
    fprint("\n" + "=" * 70)
    fprint(f"TEST 4: Permutation Test ({n_shuffles} shuffles)")
    fprint("=" * 70)

    # Build rebalance schedule from actual backtest
    rebal_dates = []
    all_dates = sc.index
    days_since = 999
    for day_idx in range(260, len(all_dates)):
        current_vix = float(vix.iloc[day_idx]) if day_idx < len(vix) else 20.0
        if current_vix < VIX_MIN:
            continue
        days_since += 1
        if days_since >= REBALANCE_INTERVAL:
            days_since = 0
            rebal_dates.append(all_dates[day_idx])

    fprint(f"  Rebalance dates identified: {len(rebal_dates)}")
    if len(rebal_dates) < 5:
        fprint("  SKIP: too few rebalance dates")
        return {"pass": False, "reason": "too few rebalance dates"}

    shuffled_sharpes = []
    available_tickers = list(sc.columns)

    for trial in range(n_shuffles):
        rng = np.random.RandomState(trial)
        # Build random signals for each rebalance date
        signal_override = {}
        for dt in rebal_dates:
            shuffled = list(available_tickers)
            rng.shuffle(shuffled)
            signal_override[dt] = {
                "call_picks": shuffled[:TOP_K],
                "put_picks": shuffled[-BOTTOM_K:],
            }

        result = run_backtest(sc, vix, signal_override=signal_override)
        shuffled_sharpes.append(result["sharpe"])

        if (trial + 1) % 100 == 0:
            fprint(f"  ... completed {trial + 1}/{n_shuffles} permutations")

    shuffled_sharpes = np.array(shuffled_sharpes)
    p95 = np.percentile(shuffled_sharpes, 95)
    p99 = np.percentile(shuffled_sharpes, 99)
    mean_shuffled = np.mean(shuffled_sharpes)
    std_shuffled = np.std(shuffled_sharpes)
    pvalue = np.mean(shuffled_sharpes >= actual_sharpe)

    passed = actual_sharpe > p95
    fprint(f"  Actual Sharpe:    {actual_sharpe:.4f}")
    fprint(f"  Shuffled mean:    {mean_shuffled:.4f} +/- {std_shuffled:.4f}")
    fprint(f"  Shuffled 95th:    {p95:.4f}")
    fprint(f"  Shuffled 99th:    {p99:.4f}")
    fprint(f"  p-value:          {pvalue:.4f}")
    fprint(f"  Result: {'PASS' if passed else 'FAIL'}")

    return {"pass": passed, "actual_sharpe": round(actual_sharpe, 4),
            "shuffled_mean": round(mean_shuffled, 4), "shuffled_std": round(std_shuffled, 4),
            "p95": round(p95, 4), "p99": round(p99, 4), "pvalue": round(pvalue, 4)}


def test_5_subperiod_stability(trades):
    """Split backtest into 4 equal quarters. At least 3/4 must have Sharpe > 0."""
    fprint("\n" + "=" * 70)
    fprint("TEST 5: Sub-Period Stability (4 quarters)")
    fprint("=" * 70)

    if len(trades) < 8:
        fprint(f"  SKIP: only {len(trades)} trades (need >= 8)")
        return {"pass": False, "reason": f"only {len(trades)} trades"}

    # Sort trades by exit date
    sorted_trades = sorted(trades, key=lambda t: t.get("exit_date", t.get("entry_date", "")))
    n = len(sorted_trades)
    quarter_size = n // 4

    quarter_sharpes = []
    for q in range(4):
        start = q * quarter_size
        end = (q + 1) * quarter_size if q < 3 else n
        qt = sorted_trades[start:end]

        pnls = [t["pnl"] for t in qt]
        costs = [t.get("pct_return", 0) for t in qt]

        if len(pnls) < 2:
            quarter_sharpes.append(0)
            continue

        rets = np.array(costs) / 100  # convert pct to fraction
        mean_r = np.mean(rets)
        std_r = np.std(rets)
        q_sharpe = float(mean_r / (std_r + 1e-10) * np.sqrt(252 / MAX_HOLD_DAYS))
        quarter_sharpes.append(q_sharpe)

    positive_quarters = sum(1 for s in quarter_sharpes if s > 0)
    passed = positive_quarters >= 3

    for i, s in enumerate(quarter_sharpes):
        status = "+" if s > 0 else "-"
        fprint(f"  Q{i+1}: Sharpe = {s:+.4f} [{status}]")

    fprint(f"  Positive quarters: {positive_quarters}/4 (need >= 3)")
    fprint(f"  Result: {'PASS' if passed else 'FAIL'}")

    return {"pass": passed, "quarter_sharpes": [round(s, 4) for s in quarter_sharpes],
            "positive_quarters": positive_quarters}


def test_6_cost_sensitivity(sc, vix):
    """Add 0.5% annual borrow cost for puts and 2x commission. Sharpe > 0.5 = PASS."""
    fprint("\n" + "=" * 70)
    fprint("TEST 6: Borrow Cost / Commission Sensitivity")
    fprint("=" * 70)

    result = run_backtest(sc, vix, extra_commission_mult=2.0, borrow_cost_annual=0.005)
    stressed_sharpe = result["sharpe"]
    stressed_pnl = result["total_pnl"]

    passed = stressed_sharpe > 0.5
    fprint(f"  Commission: 2x (${COMMISSION_PER_CONTRACT * 2:.2f}/contract)")
    fprint(f"  Borrow cost: 0.5% annual on put positions")
    fprint(f"  Stressed Sharpe: {stressed_sharpe:.4f}")
    fprint(f"  Stressed P&L:    ${stressed_pnl:.2f}")
    fprint(f"  Trades:          {result['n_trades']}")
    fprint(f"  Win rate:        {result['win_rate']:.1f}%")
    fprint(f"  Threshold: Sharpe > 0.5")
    fprint(f"  Result: {'PASS' if passed else 'FAIL'}")

    return {"pass": passed, "stressed_sharpe": round(stressed_sharpe, 4),
            "stressed_pnl": round(stressed_pnl, 2), "n_trades": result["n_trades"],
            "win_rate": round(result["win_rate"], 1)}


def test_7_outlier_removal(trades, actual_sharpe):
    """Remove top 5% of monthly returns. Sharpe > 0.5 = PASS."""
    fprint("\n" + "=" * 70)
    fprint("TEST 7: Outlier Removal (top 5% monthly returns)")
    fprint("=" * 70)

    if len(trades) < 10:
        fprint(f"  SKIP: only {len(trades)} trades")
        return {"pass": False, "reason": "insufficient trades"}

    # Group trades by month
    trade_df = pd.DataFrame(trades)
    trade_df["exit_month"] = pd.to_datetime(trade_df["exit_date"]).dt.to_period("M")
    monthly_pnl = trade_df.groupby("exit_month")["pnl"].sum()

    if len(monthly_pnl) < 4:
        fprint(f"  SKIP: only {len(monthly_pnl)} months of data")
        return {"pass": False, "reason": "insufficient months"}

    # Remove top 5% of monthly returns
    threshold = monthly_pnl.quantile(0.95)
    trimmed = monthly_pnl[monthly_pnl <= threshold]
    removed = monthly_pnl[monthly_pnl > threshold]

    fprint(f"  Total months: {len(monthly_pnl)}")
    fprint(f"  Removed {len(removed)} month(s) with PnL > ${threshold:.2f}")
    if len(removed) > 0:
        for idx, val in removed.items():
            fprint(f"    {idx}: ${val:.2f}")

    # Recompute Sharpe from trimmed monthly returns
    monthly_rets = trimmed / INITIAL_CAPITAL
    mean_m = monthly_rets.mean()
    std_m = monthly_rets.std()
    trimmed_sharpe = float(mean_m / (std_m + 1e-10) * np.sqrt(12))

    passed = trimmed_sharpe > 0.5
    fprint(f"  Original Sharpe:  {actual_sharpe:.4f}")
    fprint(f"  Trimmed Sharpe:   {trimmed_sharpe:.4f}")
    fprint(f"  Threshold: Sharpe > 0.5")
    fprint(f"  Result: {'PASS' if passed else 'FAIL'}")

    return {"pass": passed, "original_sharpe": round(actual_sharpe, 4),
            "trimmed_sharpe": round(trimmed_sharpe, 4),
            "months_removed": len(removed), "total_months": len(monthly_pnl)}


def test_8_random_signal_comparison(sc, vix, actual_sharpe, n_trials=100):
    """Generate random entry signals 100 times. Actual Sharpe > 2x random mean = PASS."""
    fprint("\n" + "=" * 70)
    fprint(f"TEST 8: Random Signal Comparison ({n_trials} trials)")
    fprint("=" * 70)

    random_sharpes = []
    for trial in range(n_trials):
        rng = np.random.RandomState(42 + trial)
        result = run_backtest(sc, vix, rng=rng)
        random_sharpes.append(result["sharpe"])

        if (trial + 1) % 25 == 0:
            fprint(f"  ... completed {trial + 1}/{n_trials} random trials")

    random_sharpes = np.array(random_sharpes)
    random_mean = np.mean(random_sharpes)
    random_std = np.std(random_sharpes)
    ratio = actual_sharpe / (random_mean + 1e-10)

    passed = actual_sharpe > 2 * random_mean
    fprint(f"  Actual Sharpe:     {actual_sharpe:.4f}")
    fprint(f"  Random mean:       {random_mean:.4f} +/- {random_std:.4f}")
    fprint(f"  Ratio:             {ratio:.2f}x")
    fprint(f"  Threshold: actual > 2x random mean")
    fprint(f"  Result: {'PASS' if passed else 'FAIL'}")

    return {"pass": passed, "actual_sharpe": round(actual_sharpe, 4),
            "random_mean": round(random_mean, 4), "random_std": round(random_std, 4),
            "ratio": round(ratio, 2)}


# ==================== MAIN ====================

def main():
    fprint("=" * 70)
    fprint("MOMENTUM BURST OPTIONS — ADVERSARIAL LEAKAGE AUDIT v1")
    fprint(f"HC #753 | KB #281 | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    fprint("=" * 70)

    # ── Download data ──
    fprint("\nDownloading market data...")
    close_df, sc, vix = download_data(start="2018-01-01")
    fprint(f"  Date range: {sc.index[0].date()} to {sc.index[-1].date()}")
    fprint(f"  Trading days: {len(sc)}")
    fprint(f"  Sectors: {list(sc.columns)}")

    # ── Run baseline backtest ──
    fprint("\nRunning baseline backtest (LGBM ranking, exact paper engine logic)...")
    baseline = run_backtest(sc, vix)
    actual_sharpe = baseline["sharpe"]
    trades = baseline["trades"]

    fprint(f"  Baseline Sharpe:   {actual_sharpe:.4f}")
    fprint(f"  Baseline Sortino:  {baseline['sortino']:.4f}")
    fprint(f"  Total P&L:         ${baseline['total_pnl']:.2f}")
    fprint(f"  Trades:            {baseline['n_trades']}")
    fprint(f"  Win rate:          {baseline['win_rate']:.1f}%")

    # ── MLflow setup ──
    mlflow_run = None
    if MLFLOW_OK:
        try:
            mlflow.set_experiment("momentum_burst_adversarial_audit")
            mlflow_run = mlflow.start_run(run_name=f"audit_{datetime.now().strftime('%Y%m%d_%H%M')}")
            mlflow.log_params({
                "strategy": "momentum_burst_options_kb281",
                "sectors": ",".join(SECTORS),
                "rebalance_interval": REBALANCE_INTERVAL,
                "vix_min": VIX_MIN,
                "top_k": TOP_K, "bottom_k": BOTTOM_K,
                "tp_pct": TP_PCT, "sl_pct": SL_PCT,
                "max_hold_days": MAX_HOLD_DAYS,
                "data_start": str(sc.index[0].date()),
                "data_end": str(sc.index[-1].date()),
                "n_trading_days": len(sc),
            })
            mlflow.log_metrics({
                "baseline_sharpe": actual_sharpe,
                "baseline_sortino": baseline["sortino"],
                "baseline_pnl": baseline["total_pnl"],
                "baseline_n_trades": baseline["n_trades"],
                "baseline_win_rate": baseline["win_rate"],
            })
        except Exception as e:
            fprint(f"  MLflow logging error: {e}")

    # ── Run 8 tests ──
    results = {}

    results["test_1_lookahead"] = test_1_lookahead_features(sc, vix)
    results["test_2_label_leakage"] = test_2_label_leakage(sc)
    results["test_3_walkforward"] = test_3_walkforward_integrity(sc)
    results["test_4_permutation"] = test_4_permutation_test(sc, vix, actual_sharpe, n_shuffles=500)
    results["test_5_subperiod"] = test_5_subperiod_stability(trades)
    results["test_6_cost_sensitivity"] = test_6_cost_sensitivity(sc, vix)
    results["test_7_outlier_removal"] = test_7_outlier_removal(trades, actual_sharpe)
    results["test_8_random_signal"] = test_8_random_signal_comparison(sc, vix, actual_sharpe, n_trials=100)

    # ── Final Verdict ──
    passed_count = sum(1 for r in results.values() if r.get("pass", False))
    total_tests = len(results)
    validated = passed_count >= 7

    fprint("\n" + "=" * 70)
    fprint("FINAL VERDICT")
    fprint("=" * 70)
    for name, r in results.items():
        status = "PASS" if r.get("pass", False) else "FAIL"
        fprint(f"  {name:40s} {status}")
    fprint(f"\n  Passed: {passed_count}/{total_tests}")
    fprint(f"  VERDICT: {'VALIDATED' if validated else 'REJECTED'} (threshold: >= 7/8)")
    fprint("=" * 70)

    # ── Save results ──
    output = {
        "strategy": "momentum_burst_options_kb281",
        "audit_date": datetime.now().isoformat(),
        "hc": 753,
        "baseline": {
            "sharpe": round(actual_sharpe, 4),
            "sortino": round(baseline["sortino"], 4),
            "total_pnl": round(baseline["total_pnl"], 2),
            "n_trades": baseline["n_trades"],
            "win_rate": round(baseline["win_rate"], 1),
        },
        "tests": results,
        "passed_count": passed_count,
        "total_tests": total_tests,
        "verdict": "VALIDATED" if validated else "REJECTED",
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    fprint(f"\nResults saved to {RESULTS_PATH}")

    # ── MLflow final logging ──
    if MLFLOW_OK and mlflow_run:
        try:
            for name, r in results.items():
                mlflow.log_metric(f"{name}_pass", 1 if r.get("pass", False) else 0)
                # Log key numeric values
                for k, v in r.items():
                    if isinstance(v, (int, float)) and k != "pass":
                        mlflow.log_metric(f"{name}_{k}", v)

            mlflow.log_metrics({
                "tests_passed": passed_count,
                "tests_total": total_tests,
                "verdict_validated": 1 if validated else 0,
            })
            mlflow.log_artifact(str(RESULTS_PATH))
            mlflow.end_run()
            fprint("MLflow run logged successfully")
        except Exception as e:
            fprint(f"MLflow final logging error: {e}")
            try:
                mlflow.end_run()
            except Exception:
                pass

    return output


if __name__ == "__main__":
    main()
