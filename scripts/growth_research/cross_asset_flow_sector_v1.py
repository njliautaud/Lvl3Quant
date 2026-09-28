#!/usr/bin/env python3
"""
Cross-Asset Flow/Positioning Features for Sector LGBM — HC #746
================================================================

Tests whether adding 6 cross-asset flow/positioning features improves
the production LGBM sector ranking model (17 momentum/quality features).

Flow features tested:
  1. cta_pressure      — Net CTA trend alignment across 6 assets
  2. cash_vs_equity    — SHY vs SPY 20d momentum (risk-off/on proxy)
  3. credit_spread_mom — 5d change in HYG-TLT spread
  4. sector_flow_rank  — 20d volume ratio rank across sectors
  5. relative_vol_flow — Sector volume ratio normalized by SPY
  6. gold_equity_ratio — GLD vs SPY 20d momentum

Variants:
  A: Production 17 features (control)
  B: Production 17 + 6 flow features (23 total)
  C: Flow-only (6 features)
  D: Best subset (auto-select top features via importance)

Architecture: Walk-forward LGBM, weekly Friday rebalance, DTE=21,
3% OTM, adaptive width max($3, 3%), $645 capital, $2.60 commission,
15% haircut, hold-to-expiry. 2008-2026. 5-gate adversarial audit.

MLflow experiment: cross_asset_flow_sector_v1
"""

import json
import sys
import time
import platform
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

warnings.filterwarnings("ignore")

_builtin_print = print
def fprint(*args, **kwargs):
    _builtin_print(*args, **kwargs)
    sys.stdout.flush()

# ── Path setup (Jupiter / Neptune / Razer) ──────────────────────────────
_host = platform.node().lower()
if "razer" in _host or platform.system() == "Windows":
    BASE = Path(r"C:\Users\claude\Lvl3Quant")
elif "neptune" in _host:
    BASE = Path("/home/nick/Lvl3Quant")
else:
    BASE = Path("/home/jupiter/Lvl3Quant")

OUTPUT_DIR = BASE / "output" / "growth_research" / "cross_asset_flow_sector_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = OUTPUT_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants (V8/V9 production) ────────────────────────────────────────
SECTORS = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
FLOW_TICKERS = ["SPY", "QQQ", "IWM", "GLD", "TLT", "SHY", "HYG", "LQD"]
ALL_TICKERS = sorted(set(SECTORS + FLOW_TICKERS))

CAP = 645.0
TOP_K = 3
BOT_K = 3
DTE = 21
OTM_PCT = 0.03
SPREAD_WIDTH_PCT = 0.03
SPREAD_WIDTH_MIN = 3.0
COMMISSION_RT = 2.60
HAIRCUT_PCT = 0.15
REBAL_FREQ = "W-FRI"
WF_TRAIN_WEEKS = 52  # ~1 year
COST_WIDTH_MAX = 0.50
RF_ANNUAL = 0.04

# ── MLflow setup ────────────────────────────────────────────────────────
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "cross_asset_flow_sector_v1"
MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen(MLFLOW_URI, timeout=3)
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    MLFLOW_OK = True
    fprint(f"MLflow connected: {MLFLOW_URI}")
except Exception:
    fprint("MLflow unavailable — will still run and save results locally")


# ══════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════

def download_data(start="2006-01-01", end=None):
    """Download all tickers from yfinance. Cache to parquet."""
    cache_path = CACHE_DIR / "all_prices.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        fprint(f"Loaded cached data: {df.shape}")
        return df

    import yfinance as yf
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    fprint(f"Downloading {len(ALL_TICKERS)} tickers from yfinance...")
    data = yf.download(ALL_TICKERS, start=start, end=end, progress=False, group_by="ticker")

    # Flatten multi-index columns
    records = {}
    for tk in ALL_TICKERS:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                tkdata = data[tk].copy()
            else:
                tkdata = data.copy()
            if tkdata.index.tz is not None:
                tkdata.index = tkdata.index.tz_convert(None)
            if len(tkdata.dropna(subset=["Close"])) > 252:
                records[tk] = tkdata[["Open", "High", "Low", "Close", "Volume"]].copy()
        except Exception as e:
            fprint(f"  SKIP {tk}: {e}")

    # Combine into single DataFrame with ticker column
    frames = []
    for tk, df in records.items():
        df = df.copy()
        df["ticker"] = tk
        df.index.name = "date"
        frames.append(df.reset_index())
    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined.to_parquet(cache_path)
    fprint(f"Cached {len(combined):,} rows for {len(records)} tickers")
    return combined


def pivot_prices(data, col="Close"):
    """Pivot long-form data to wide: date x ticker."""
    return data.pivot_table(index="date", columns="ticker", values=col).sort_index()


# ══════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════

# -- Production V6 features (17 total) -----------------------------------

def build_production_features(close_wide, volume_wide, spy_close):
    """Build the production 17 features for all sectors. Returns dict[ticker -> DataFrame]."""
    sector_features = {}
    spy_lr = np.log(spy_close / spy_close.shift(1))

    for tk in SECTORS:
        if tk not in close_wide.columns:
            continue
        c = close_wide[tk].dropna()
        v = volume_wide[tk].dropna() if tk in volume_wide.columns else pd.Series(0, index=c.index)
        lr = np.log(c / c.shift(1))

        # Align indices
        idx = c.index.intersection(spy_close.index)
        c = c.reindex(idx)
        v = v.reindex(idx, fill_value=0)
        sc = spy_close.reindex(idx)
        slr = spy_lr.reindex(idx)

        feat = pd.DataFrame(index=idx)

        # 6 momentum
        feat["ret_5d"] = c.pct_change(5)
        feat["ret_10d"] = c.pct_change(10)
        feat["ret_21d"] = c.pct_change(21)
        feat["ret_63d"] = c.pct_change(63)
        feat["ret_126d"] = c.pct_change(126)
        feat["ret_252d"] = c.pct_change(252)

        # Quality / risk
        feat["sharpe_63d"] = lr.rolling(63).mean() / lr.rolling(63).std()
        feat["pct_52w_high"] = c / c.rolling(252).max()
        feat["mom_accel"] = c.pct_change(63) - c.pct_change(63).shift(63)

        # Consistency
        monthly_ret = c.pct_change(21)
        feat["pct_pos_months_12m"] = monthly_ret.rolling(12).apply(lambda x: (x > 0).mean(), raw=True)

        ds = lr.copy()
        ds[ds > 0] = 0
        feat["sortino_63d"] = lr.rolling(63).mean() / ds.rolling(63).std()

        # Calmar 1y
        roll_max = c.rolling(252).max()
        dd = (c - roll_max) / roll_max
        max_dd_1y = dd.rolling(252).min()
        feat["calmar_1y"] = c.pct_change(252) / (-max_dd_1y + 1e-8)

        # Up capture
        spy_up = slr > 0
        up_lr = lr.where(spy_up, 0)
        up_spy = slr.where(spy_up, 0)
        feat["up_capture"] = up_lr.rolling(63).sum() / (up_spy.rolling(63).sum() + 1e-8)

        # Trend quality
        def _trend_r2_slope(series, window=63):
            r2 = series.rolling(window).apply(
                lambda x: np.corrcoef(np.arange(len(x)), x)[0, 1] ** 2
                if len(x) == window and np.std(x) > 0 else 0, raw=True)
            slope = series.rolling(window).apply(
                lambda x: np.polyfit(np.arange(len(x)), x, 1)[0]
                if len(x) == window else 0, raw=True)
            return r2, slope

        r2, slope = _trend_r2_slope(c, 63)
        feat["trend_r2_63d"] = r2
        feat["trend_slope_63d"] = slope

        # Cross-sectional
        feat["sector_spy_beta_63d"] = lr.rolling(63).cov(slr) / (slr.rolling(63).var() + 1e-10)

        sector_features[tk] = feat

    # Cross-sector dispersion (same across all sectors for a given date)
    all_ret21 = pd.DataFrame({tk: close_wide[tk].pct_change(21) for tk in SECTORS if tk in close_wide.columns})
    dispersion = all_ret21.std(axis=1)

    for tk in sector_features:
        sector_features[tk]["cross_sector_dispersion"] = dispersion.reindex(sector_features[tk].index)

    return sector_features


PRODUCTION_FEATURE_NAMES = [
    "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "sharpe_63d", "pct_52w_high", "mom_accel",
    "pct_pos_months_12m", "sortino_63d", "calmar_1y", "up_capture",
    "trend_r2_63d", "trend_slope_63d",
    "sector_spy_beta_63d", "cross_sector_dispersion",
]
assert len(PRODUCTION_FEATURE_NAMES) == 17


# -- 6 Flow features (new) -----------------------------------------------

def build_flow_features(close_wide, volume_wide):
    """Build the 6 cross-asset flow/positioning features for all sectors.
    Returns dict[ticker -> DataFrame] with 6 columns each."""
    spy = close_wide.get("SPY")
    qqq = close_wide.get("QQQ")
    iwm = close_wide.get("IWM")
    gld = close_wide.get("GLD")
    tlt = close_wide.get("TLT")
    shy = close_wide.get("SHY")
    hyg = close_wide.get("HYG")

    # Common index across flow tickers
    idx = spy.dropna().index
    for s in [qqq, iwm, gld, tlt, shy, hyg]:
        if s is not None:
            idx = idx.intersection(s.dropna().index)

    # ── 1. CTA Pressure ─────────────────────────────────────────────────
    cta_assets = {"SPY": spy, "QQQ": qqq, "IWM": iwm, "GLD": gld, "TLT": tlt, "HYG": hyg}
    cta_scores = pd.DataFrame(index=idx)
    for name, price in cta_assets.items():
        p = price.reindex(idx)
        sma50 = p.rolling(50).mean()
        sma200 = p.rolling(200).mean()
        above_both = (p > sma50) & (sma50 > sma200)
        below_both = (p < sma50) & (sma50 < sma200)
        score = pd.Series(0.0, index=idx)
        score[above_both] = 1.0
        score[below_both] = -1.0
        cta_scores[name] = score
    cta_pressure = cta_scores.mean(axis=1)  # Range [-1, +1]

    # ── 2. Cash vs Equity Ratio ──────────────────────────────────────────
    shy_ret20 = shy.reindex(idx).pct_change(20)
    spy_ret20 = spy.reindex(idx).pct_change(20)
    cash_vs_equity = shy_ret20 - spy_ret20  # positive = risk-off

    # ── 3. Credit Spread Momentum ────────────────────────────────────────
    hyg_tlt_spread = hyg.reindex(idx).pct_change(1) - tlt.reindex(idx).pct_change(1)
    credit_spread_mom = hyg_tlt_spread.rolling(5).sum()  # 5-day change

    # ── 4 & 5. Volume-based sector flow features ────────────────────────
    # SPY volume ratio for normalization
    spy_vol = volume_wide.get("SPY")
    spy_vol_ratio = (spy_vol / spy_vol.rolling(20).mean()).reindex(idx) if spy_vol is not None else pd.Series(1.0, index=idx)

    sector_vol_ratios = {}
    for tk in SECTORS:
        if tk in volume_wide.columns:
            v = volume_wide[tk].reindex(idx)
            vol_ratio = v / v.rolling(20).mean()
            sector_vol_ratios[tk] = vol_ratio

    vol_ratio_df = pd.DataFrame(sector_vol_ratios)

    # ── 6. Gold/Equity Ratio ────────────────────────────────────────────
    gld_ret20 = gld.reindex(idx).pct_change(20)
    gold_equity_ratio = gld_ret20 - spy_ret20  # positive = gold outperforming

    # Build per-sector flow features
    sector_flow_features = {}
    for tk in SECTORS:
        feat = pd.DataFrame(index=idx)
        feat["cta_pressure"] = cta_pressure
        feat["cash_vs_equity_ratio"] = cash_vs_equity
        feat["credit_spread_mom"] = credit_spread_mom

        # Sector-specific flow rank
        if tk in vol_ratio_df.columns:
            # Rank across sectors (1 = highest flow)
            ranks = vol_ratio_df.rank(axis=1, ascending=False, pct=True)
            feat["sector_flow_rank"] = ranks[tk] if tk in ranks.columns else 0.5
        else:
            feat["sector_flow_rank"] = 0.5

        # Relative volume flow (sector vs market)
        if tk in sector_vol_ratios:
            feat["relative_vol_flow"] = sector_vol_ratios[tk] / (spy_vol_ratio + 1e-8)
        else:
            feat["relative_vol_flow"] = 1.0

        feat["gold_equity_ratio"] = gold_equity_ratio

        sector_flow_features[tk] = feat

    return sector_flow_features


FLOW_FEATURE_NAMES = [
    "cta_pressure", "cash_vs_equity_ratio", "credit_spread_mom",
    "sector_flow_rank", "relative_vol_flow", "gold_equity_ratio",
]
assert len(FLOW_FEATURE_NAMES) == 6


# ══════════════════════════════════════════════════════════════════════════
# OPTIONS SPREAD PRICING (Black-Scholes based, production V9 architecture)
# ══════════════════════════════════════════════════════════════════════════

def bs_price(S, K, T, r, sigma, option_type="call"):
    """Black-Scholes option price."""
    if T <= 0 or sigma <= 0:
        return max(0, S - K) if option_type == "call" else max(0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def price_spread(spot, direction, dte, otm_pct, width_pct, width_min, sigma, r=0.04):
    """Price a bull call or bear put spread. Returns (entry_cost, max_profit, width)."""
    T = dte / 365.0
    if direction == "bull":
        K_buy = spot * (1 + otm_pct)
        width = max(width_min, spot * width_pct)
        K_sell = K_buy + width
        entry = bs_price(spot, K_buy, T, r, sigma, "call") - bs_price(spot, K_sell, T, r, sigma, "call")
    else:
        K_buy = spot * (1 - otm_pct)
        width = max(width_min, spot * width_pct)
        K_sell = K_buy - width
        entry = bs_price(spot, K_buy, T, r, sigma, "put") - bs_price(spot, K_sell, T, r, sigma, "put")

    entry = max(entry, 0.01)
    max_profit = width - entry
    return entry, max_profit, width


def simulate_spread_pnl(spot_entry, spot_exit, direction, dte, otm_pct, width_pct, width_min,
                         sigma, haircut_pct, commission):
    """Simulate a spread's P&L at expiry (hold-to-expiry). Returns (pnl, cost, invested)."""
    entry_cost, max_profit, width = price_spread(
        spot_entry, direction, dte, otm_pct, width_pct, width_min, sigma)

    # Cost/width gate
    if entry_cost / width > COST_WIDTH_MAX:
        return 0.0, 0.0, 0.0, True  # skipped

    # Apply haircut
    entry_cost_after_haircut = entry_cost * (1 + haircut_pct)
    invested = entry_cost_after_haircut * 100 + commission  # per contract

    # Payoff at expiry
    if direction == "bull":
        K_buy = spot_entry * (1 + otm_pct)
        K_sell = K_buy + width
        intrinsic_buy = max(0, spot_exit - K_buy)
        intrinsic_sell = max(0, spot_exit - K_sell)
        payoff = (intrinsic_buy - intrinsic_sell) * 100
    else:
        K_buy = spot_entry * (1 - otm_pct)
        K_sell = K_buy - width
        intrinsic_buy = max(0, K_buy - spot_exit)
        intrinsic_sell = max(0, K_sell - spot_exit)
        payoff = (intrinsic_buy - intrinsic_sell) * 100

    pnl = payoff - entry_cost * 100 - commission
    return pnl, entry_cost, invested, False


# ══════════════════════════════════════════════════════════════════════════
# WALK-FORWARD LGBM ENGINE
# ══════════════════════════════════════════════════════════════════════════

def run_walkforward(sector_features, close_wide, volume_wide, feature_names, variant_name):
    """Walk-forward LGBM with weekly Friday rebalance + options spread P&L."""
    import lightgbm as lgb

    fprint(f"\n{'='*60}")
    fprint(f"VARIANT: {variant_name}  ({len(feature_names)} features)")
    fprint(f"{'='*60}")

    # Build panel: date x ticker x features
    panel_frames = []
    spy_close = close_wide["SPY"]
    spy_lr = np.log(spy_close / spy_close.shift(1))

    for tk in SECTORS:
        if tk not in sector_features:
            continue
        feat = sector_features[tk][feature_names].copy()
        c = close_wide[tk]

        # Forward return (21d) as label
        fwd_ret = c.pct_change(21).shift(-21)
        feat["fwd_ret_21d"] = fwd_ret.reindex(feat.index)
        feat["ticker"] = tk
        feat["close"] = c.reindex(feat.index)

        # Realized vol (for BS pricing)
        lr = np.log(c / c.shift(1))
        feat["realized_vol"] = lr.rolling(63).std() * np.sqrt(252)
        feat = feat.reset_index()
        panel_frames.append(feat)

    panel = pd.concat(panel_frames, ignore_index=True)
    panel = panel.dropna(subset=feature_names + ["fwd_ret_21d"])
    panel = panel.sort_values("date").reset_index(drop=True)

    # Weekly Friday rebalance dates
    fridays = panel["date"].drop_duplicates().sort_values()
    fridays = fridays[fridays.dt.dayofweek == 4]  # Friday = 4
    if len(fridays) < WF_TRAIN_WEEKS + 10:
        fprint(f"  Not enough Fridays: {len(fridays)}")
        return None

    fprint(f"  Panel: {len(panel):,} rows, {len(fridays)} Fridays")
    fprint(f"  Date range: {panel['date'].min().date()} to {panel['date'].max().date()}")

    # Walk-forward
    trades = []
    weekly_pnls = []
    feat_imp_accum = np.zeros(len(feature_names))
    n_folds = 0
    n_skipped = 0

    for i in range(WF_TRAIN_WEEKS, len(fridays) - 4):  # Need 4 weeks ahead for DTE=21
        train_end = fridays.iloc[i]
        train_start = fridays.iloc[i - WF_TRAIN_WEEKS]
        test_date = fridays.iloc[i]

        # Find expiry date (~21 calendar days ahead)
        expiry_candidates = fridays[fridays > test_date + pd.Timedelta(days=14)]
        expiry_candidates = expiry_candidates[expiry_candidates <= test_date + pd.Timedelta(days=28)]
        if len(expiry_candidates) == 0:
            continue
        expiry_date = expiry_candidates.iloc[0]

        # Train data
        train_mask = (panel["date"] >= train_start) & (panel["date"] < train_end)
        train = panel[train_mask]

        # Test data (cross-section on test_date)
        test_mask = panel["date"] == test_date
        test = panel[test_mask].copy()

        if len(train) < 50 or len(test) < 5:
            continue

        X_train = train[feature_names].values
        y_train = train["fwd_ret_21d"].values
        X_test = test[feature_names].values

        # Clean
        valid_train = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
        X_train = X_train[valid_train]
        y_train = y_train[valid_train]

        valid_test = np.isfinite(X_test).all(axis=1)
        X_test_clean = X_test[valid_test]
        test_clean = test[valid_test].copy()

        if len(X_train) < 30 or len(X_test_clean) < 3:
            continue

        model = lgb.LGBMRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=5,
            verbose=-1, n_jobs=-1,
        )
        model.fit(X_train, y_train)

        feat_imp_accum += model.feature_importances_.astype(float)
        n_folds += 1

        # Predict and rank
        test_clean["pred"] = model.predict(X_test_clean)
        test_clean = test_clean.sort_values("pred", ascending=False)

        top_k = test_clean.head(TOP_K)
        bot_k = test_clean.tail(BOT_K)

        # Simulate spreads
        week_pnl = 0.0
        n_trades_week = 0

        for _, row in top_k.iterrows():
            tk = row["ticker"]
            spot_entry = row["close"]
            sigma = row["realized_vol"] if np.isfinite(row["realized_vol"]) and row["realized_vol"] > 0 else 0.25

            # Get spot at expiry
            exp_data = panel[(panel["ticker"] == tk) & (panel["date"] >= expiry_date)]
            if len(exp_data) == 0:
                continue
            spot_exit = exp_data.iloc[0]["close"]
            if not np.isfinite(spot_exit) or spot_exit <= 0:
                continue

            pnl, entry_cost, invested, skipped = simulate_spread_pnl(
                spot_entry, spot_exit, "bull", DTE, OTM_PCT,
                SPREAD_WIDTH_PCT, SPREAD_WIDTH_MIN, sigma, HAIRCUT_PCT, COMMISSION_RT)

            if skipped:
                n_skipped += 1
                continue

            trades.append({
                "date": str(test_date.date()),
                "expiry": str(expiry_date.date()),
                "ticker": tk,
                "direction": "bull",
                "spot_entry": round(spot_entry, 2),
                "spot_exit": round(spot_exit, 2),
                "entry_cost": round(entry_cost, 2),
                "pnl": round(pnl, 2),
                "invested": round(invested, 2),
            })
            week_pnl += pnl
            n_trades_week += 1

        for _, row in bot_k.iterrows():
            tk = row["ticker"]
            spot_entry = row["close"]
            sigma = row["realized_vol"] if np.isfinite(row["realized_vol"]) and row["realized_vol"] > 0 else 0.25

            exp_data = panel[(panel["ticker"] == tk) & (panel["date"] >= expiry_date)]
            if len(exp_data) == 0:
                continue
            spot_exit = exp_data.iloc[0]["close"]
            if not np.isfinite(spot_exit) or spot_exit <= 0:
                continue

            pnl, entry_cost, invested, skipped = simulate_spread_pnl(
                spot_entry, spot_exit, "bear", DTE, OTM_PCT,
                SPREAD_WIDTH_PCT, SPREAD_WIDTH_MIN, sigma, HAIRCUT_PCT, COMMISSION_RT)

            if skipped:
                n_skipped += 1
                continue

            trades.append({
                "date": str(test_date.date()),
                "expiry": str(expiry_date.date()),
                "ticker": tk,
                "direction": "bear",
                "spot_entry": round(spot_entry, 2),
                "spot_exit": round(spot_exit, 2),
                "entry_cost": round(entry_cost, 2),
                "pnl": round(pnl, 2),
                "invested": round(invested, 2),
            })
            week_pnl += pnl
            n_trades_week += 1

        if n_trades_week > 0:
            weekly_pnls.append({"date": str(test_date.date()), "pnl": week_pnl, "n_trades": n_trades_week})

    if n_folds == 0 or len(trades) == 0:
        fprint("  No trades generated")
        return None

    # Feature importance (normalized)
    feat_imp = feat_imp_accum / n_folds
    importance = dict(zip(feature_names, (feat_imp / feat_imp.sum() * 100).tolist()))

    fprint(f"  Folds: {n_folds}, Trades: {len(trades)}, Skipped: {n_skipped}")

    return {
        "variant": variant_name,
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "trades": trades,
        "weekly_pnls": weekly_pnls,
        "n_folds": n_folds,
        "n_trades": len(trades),
        "n_skipped": n_skipped,
        "feature_importance": importance,
    }


# ══════════════════════════════════════════════════════════════════════════
# METRICS + ADVERSARIAL AUDIT
# ══════════════════════════════════════════════════════════════════════════

def compute_metrics(trades, cap=CAP):
    """Compute risk-adjusted metrics from trade list."""
    if not trades:
        return {}

    df = pd.DataFrame(trades)
    df["date"] = pd.to_datetime(df["date"])

    # Weekly P&L
    weekly = df.groupby("date")["pnl"].sum().sort_index()

    # Equity curve
    equity = [cap]
    for pnl in weekly.values:
        equity.append(equity[-1] + pnl)
    equity = np.array(equity[1:])

    # Returns
    returns = weekly.values / cap  # simple return on starting capital
    n_weeks = len(returns)

    if n_weeks < 10:
        return {"error": "too few weeks"}

    # Annualize (52 weeks/year)
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    sharpe = mean_ret / std_ret * np.sqrt(52) if std_ret > 0 else 0

    downside = returns[returns < 0]
    downside_std = np.std(downside) if len(downside) > 0 else 1e-8
    sortino = mean_ret / downside_std * np.sqrt(52)

    years = n_weeks / 52
    total_return = (equity[-1] - cap) / cap
    cagr = ((1 + total_return) ** (1 / max(years, 0.1)) - 1) * 100

    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = float(np.min(dd))

    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] <= 0]
    wr = len(wins) / len(df) * 100 if len(df) > 0 else 0
    gross_profit = wins["pnl"].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses["pnl"].sum()) if len(losses) > 0 else 1e-8
    pf = gross_profit / gross_loss if gross_loss > 0 else 999

    calmar = cagr / abs(max_dd * 100) if max_dd != 0 else 0

    return {
        "sharpe": round(float(sharpe), 3),
        "sortino": round(float(sortino), 3),
        "cagr": round(float(cagr), 1),
        "max_dd": round(float(max_dd * 100), 2),
        "wr": round(float(wr), 1),
        "pf": round(float(pf), 2),
        "calmar": round(float(calmar), 2),
        "total_pnl": round(float(df["pnl"].sum()), 2),
        "n_trades": len(df),
        "n_weeks": n_weeks,
        "final_equity": round(float(equity[-1]), 2),
        "avg_pnl_per_trade": round(float(df["pnl"].mean()), 2),
        "avg_weekly_pnl": round(float(weekly.mean()), 2),
    }


def adversarial_audit(trades, name, n_perm=500):
    """5-gate adversarial audit. Returns dict of gate results."""
    if not trades or len(trades) < 20:
        return {"error": "too few trades"}

    df = pd.DataFrame(trades)
    df["date"] = pd.to_datetime(df["date"])
    weekly = df.groupby("date")["pnl"].sum().sort_index()
    returns = weekly.values / CAP

    gates = {}

    # Gate 1: Permutation test (shuffle trade signs)
    fprint(f"  [{name}] Gate 1: Permutation test ({n_perm} shuffles)...")
    real_sharpe = np.mean(returns) / np.std(returns) * np.sqrt(52) if np.std(returns) > 0 else 0
    perm_sharpes = []
    for _ in range(n_perm):
        perm_r = returns * np.random.choice([-1, 1], size=len(returns))
        ps = np.mean(perm_r) / np.std(perm_r) * np.sqrt(52) if np.std(perm_r) > 0 else 0
        perm_sharpes.append(ps)
    perm_p = np.mean(np.array(perm_sharpes) >= real_sharpe)
    gates["G1_permutation"] = {"p_value": round(float(perm_p), 4), "pass": perm_p < 0.05}
    fprint(f"    p={perm_p:.4f} {'PASS' if perm_p < 0.05 else 'FAIL'}")

    # Gate 2: Regime balance (R1) — bull vs bear months
    fprint(f"  [{name}] Gate 2: Regime balance (R1)...")
    monthly_dates = pd.to_datetime(weekly.index)
    # Use SPY SMA200 proxy: positive weekly return = "bull"
    mid = len(returns) // 2
    h1 = returns[:mid]
    h2 = returns[mid:]
    h1_sharpe = np.mean(h1) / np.std(h1) * np.sqrt(52) if np.std(h1) > 0 else 0
    h2_sharpe = np.mean(h2) / np.std(h2) * np.sqrt(52) if np.std(h2) > 0 else 0
    max_s = max(abs(h1_sharpe), abs(h2_sharpe), 1e-8)
    gap = abs(h1_sharpe - h2_sharpe) / max_s
    gates["G2_regime"] = {
        "h1_sharpe": round(float(h1_sharpe), 3),
        "h2_sharpe": round(float(h2_sharpe), 3),
        "gap": round(float(gap), 3),
        "pass": gap < 0.50,
    }
    fprint(f"    H1={h1_sharpe:.3f}, H2={h2_sharpe:.3f}, gap={gap:.3f} {'PASS' if gap < 0.50 else 'FAIL'}")

    # Gate 3: Sub-period positivity (both halves > 0)
    fprint(f"  [{name}] Gate 3: Sub-period positivity...")
    g3_pass = h1_sharpe > 0 and h2_sharpe > 0
    gates["G3_subperiod"] = {"h1_positive": h1_sharpe > 0, "h2_positive": h2_sharpe > 0, "pass": g3_pass}
    fprint(f"    H1>0: {h1_sharpe > 0}, H2>0: {h2_sharpe > 0} {'PASS' if g3_pass else 'FAIL'}")

    # Gate 4: Outlier robustness (remove top 5% returns, still positive Sharpe)
    fprint(f"  [{name}] Gate 4: Outlier robustness...")
    n_remove = max(1, int(len(returns) * 0.05))
    sorted_idx = np.argsort(returns)[::-1]
    trimmed = np.delete(returns, sorted_idx[:n_remove])
    trim_sharpe = np.mean(trimmed) / np.std(trimmed) * np.sqrt(52) if np.std(trimmed) > 0 else 0
    g4_pass = trim_sharpe > 0
    gates["G4_outlier"] = {
        "trimmed_sharpe": round(float(trim_sharpe), 3),
        "n_removed": n_remove,
        "pass": g4_pass,
    }
    fprint(f"    Trimmed Sharpe={trim_sharpe:.3f} (removed {n_remove}) {'PASS' if g4_pass else 'FAIL'}")

    # Gate 5: Day-concentration cap (no single week > 70% of total P&L)
    fprint(f"  [{name}] Gate 5: Day concentration...")
    total_pnl = abs(weekly.sum())
    max_week_pnl = weekly.abs().max()
    conc = max_week_pnl / total_pnl if total_pnl > 0 else 0
    g5_pass = conc < 0.70
    gates["G5_concentration"] = {
        "max_week_pct": round(float(conc * 100), 1),
        "pass": g5_pass,
    }
    fprint(f"    Max week = {conc*100:.1f}% of total {'PASS' if conc < 0.70 else 'FAIL'}")

    n_pass = sum(1 for g in gates.values() if g.get("pass"))
    gates["summary"] = {"passed": n_pass, "total": len(gates) - 1, "all_pass": n_pass == 5}
    fprint(f"  [{name}] GATES: {n_pass}/5")

    return gates


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    fprint("=" * 70)
    fprint("CROSS-ASSET FLOW FEATURES FOR SECTOR LGBM — HC #746")
    fprint("=" * 70)
    fprint(f"Sectors: {len(SECTORS)} | Flow tickers: {len(FLOW_TICKERS)}")
    fprint(f"Options: DTE={DTE}, OTM={OTM_PCT*100}%, width=max(${SPREAD_WIDTH_MIN},{SPREAD_WIDTH_PCT*100}%)")
    fprint(f"Capital: ${CAP}, Commission: ${COMMISSION_RT}, Haircut: {HAIRCUT_PCT*100}%")
    fprint(f"Walk-forward: {WF_TRAIN_WEEKS} weeks train, weekly rebalance, hold-to-expiry")
    fprint(f"Node: {_host}")
    fprint()

    # Download data
    data = download_data(start="2006-01-01")
    close_wide = pivot_prices(data, "Close")
    volume_wide = pivot_prices(data, "Volume")
    spy_close = close_wide["SPY"]

    # Verify all sectors present
    missing = [s for s in SECTORS if s not in close_wide.columns]
    if missing:
        fprint(f"WARNING: Missing sectors: {missing}")

    fprint(f"\nBuilding features...")

    # Build production features
    prod_features = build_production_features(close_wide, volume_wide, spy_close)
    fprint(f"  Production features: 17 per sector")

    # Build flow features
    flow_features = build_flow_features(close_wide, volume_wide)
    fprint(f"  Flow features: 6 per sector")

    # Merge features per sector
    combined_features = {}
    for tk in SECTORS:
        if tk not in prod_features or tk not in flow_features:
            continue
        idx = prod_features[tk].index.intersection(flow_features[tk].index)
        combined = pd.concat([
            prod_features[tk].reindex(idx),
            flow_features[tk].reindex(idx),
        ], axis=1)
        combined_features[tk] = combined

    fprint(f"  Combined features: {17 + 6} per sector, {len(combined_features)} sectors")

    # ── Run 4 variants ──────────────────────────────────────────────────
    all_results = {}

    # Variant A: Production 17 features (control)
    result_a = run_walkforward(prod_features, close_wide, volume_wide,
                                PRODUCTION_FEATURE_NAMES, "A_production_17")
    if result_a:
        result_a["metrics"] = compute_metrics(result_a["trades"])
        result_a["gates"] = adversarial_audit(result_a["trades"], "A_production_17")
        all_results["A"] = result_a

    # Variant B: Production 17 + 6 flow features (23 total)
    result_b = run_walkforward(combined_features, close_wide, volume_wide,
                                PRODUCTION_FEATURE_NAMES + FLOW_FEATURE_NAMES, "B_prod_plus_flow_23")
    if result_b:
        result_b["metrics"] = compute_metrics(result_b["trades"])
        result_b["gates"] = adversarial_audit(result_b["trades"], "B_prod_plus_flow_23")
        all_results["B"] = result_b

    # Variant C: Flow-only (6 features)
    result_c = run_walkforward(flow_features, close_wide, volume_wide,
                                FLOW_FEATURE_NAMES, "C_flow_only_6")
    if result_c:
        result_c["metrics"] = compute_metrics(result_c["trades"])
        result_c["gates"] = adversarial_audit(result_c["trades"], "C_flow_only_6")
        all_results["C"] = result_c

    # Variant D: Best subset (auto-select from B's importance)
    if result_b and result_b.get("feature_importance"):
        imp = result_b["feature_importance"]
        # Select top features that contribute >= 3% importance each
        top_feats = [f for f, v in sorted(imp.items(), key=lambda x: -x[1]) if v >= 3.0]
        if len(top_feats) < 5:
            top_feats = [f for f, _ in sorted(imp.items(), key=lambda x: -x[1])[:10]]
        fprint(f"\n  Variant D auto-selected {len(top_feats)} features: {top_feats}")

        result_d = run_walkforward(combined_features, close_wide, volume_wide,
                                    top_feats, "D_best_subset")
        if result_d:
            result_d["metrics"] = compute_metrics(result_d["trades"])
            result_d["gates"] = adversarial_audit(result_d["trades"], "D_best_subset")
            all_results["D"] = result_d

    # ── Summary ─────────────────────────────────────────────────────────
    fprint(f"\n{'='*70}")
    fprint("VARIANT COMPARISON")
    fprint(f"{'='*70}")
    fprint(f"{'Variant':<28} {'Sharpe':>7} {'Sortino':>8} {'CAGR':>7} {'MaxDD':>7} {'WR':>6} {'PF':>6} {'Gates':>6}")
    fprint("-" * 80)

    for key in ["A", "B", "C", "D"]:
        if key not in all_results:
            continue
        r = all_results[key]
        m = r["metrics"]
        g = r["gates"]
        n_pass = g.get("summary", {}).get("passed", 0)
        fprint(f"{r['variant']:<28} {m.get('sharpe',0):>7.3f} {m.get('sortino',0):>8.3f} "
               f"{m.get('cagr',0):>6.1f}% {m.get('max_dd',0):>6.1f}% "
               f"{m.get('wr',0):>5.1f}% {m.get('pf',0):>6.2f} {n_pass}/5")

    # Feature importance comparison
    fprint(f"\n{'='*70}")
    fprint("FEATURE IMPORTANCE (Variant B — all 23 features)")
    fprint(f"{'='*70}")
    if "B" in all_results and all_results["B"].get("feature_importance"):
        imp = all_results["B"]["feature_importance"]
        for fn, iv in sorted(imp.items(), key=lambda x: -x[1]):
            marker = " [FLOW]" if fn in FLOW_FEATURE_NAMES else ""
            fprint(f"  {fn:<28} {iv:>6.1f}%{marker}")

        flow_imp = sum(imp.get(f, 0) for f in FLOW_FEATURE_NAMES)
        prod_imp = sum(imp.get(f, 0) for f in PRODUCTION_FEATURE_NAMES)
        fprint(f"\n  Flow features total: {flow_imp:.1f}% | Production features total: {prod_imp:.1f}%")

    # Determine winner
    fprint(f"\n{'='*70}")
    fprint("VERDICT")
    fprint(f"{'='*70}")

    best = None
    for key in ["A", "B", "C", "D"]:
        if key not in all_results:
            continue
        r = all_results[key]
        m = r["metrics"]
        g = r["gates"]
        if g.get("summary", {}).get("all_pass", False):
            if best is None or m.get("sharpe", 0) > best["metrics"].get("sharpe", 0):
                best = r

    if best is None:
        # Fall back to highest Sharpe among those passing >= 3 gates
        for key in ["A", "B", "C", "D"]:
            if key not in all_results:
                continue
            r = all_results[key]
            m = r["metrics"]
            if best is None or m.get("sharpe", 0) > best["metrics"].get("sharpe", 0):
                best = r

    if best:
        bm = best["metrics"]
        bg = best["gates"]
        n_pass = bg.get("summary", {}).get("passed", 0)
        fprint(f"  WINNER: {best['variant']}")
        fprint(f"  Sharpe:  {bm.get('sharpe', 0):.3f}")
        fprint(f"  Sortino: {bm.get('sortino', 0):.3f}")
        fprint(f"  CAGR:    {bm.get('cagr', 0):.1f}%")
        fprint(f"  MaxDD:   {bm.get('max_dd', 0):.1f}%")
        fprint(f"  WR:      {bm.get('wr', 0):.1f}%")
        fprint(f"  PF:      {bm.get('pf', 0):.2f}")
        fprint(f"  Gates:   {n_pass}/5")
        fprint(f"  Trades:  {bm.get('n_trades', 0)}")
        fprint(f"  Final:   ${bm.get('final_equity', 0):.2f}")

        # Flow feature impact
        if "A" in all_results and "B" in all_results:
            a_sharpe = all_results["A"]["metrics"].get("sharpe", 0)
            b_sharpe = all_results["B"]["metrics"].get("sharpe", 0)
            delta = b_sharpe - a_sharpe
            pct_change = (delta / abs(a_sharpe) * 100) if a_sharpe != 0 else 0
            fprint(f"\n  Flow feature impact on Sharpe: {delta:+.3f} ({pct_change:+.1f}%)")
            fprint(f"    A (production 17): {a_sharpe:.3f}")
            fprint(f"    B (prod + flow 23): {b_sharpe:.3f}")
            if delta > 0.1:
                fprint(f"  CONCLUSION: Flow features ADD value (+{delta:.3f} Sharpe)")
            elif delta < -0.1:
                fprint(f"  CONCLUSION: Flow features HURT performance ({delta:.3f} Sharpe)")
            else:
                fprint(f"  CONCLUSION: Flow features have MARGINAL impact ({delta:+.3f} Sharpe)")

    # Save results
    output = {
        "experiment": "cross_asset_flow_sector_v1",
        "run_date": str(datetime.now()),
        "node": _host,
        "config": {
            "sectors": SECTORS,
            "flow_tickers": FLOW_TICKERS,
            "dte": DTE,
            "otm_pct": OTM_PCT,
            "spread_width_pct": SPREAD_WIDTH_PCT,
            "spread_width_min": SPREAD_WIDTH_MIN,
            "capital": CAP,
            "commission": COMMISSION_RT,
            "haircut_pct": HAIRCUT_PCT,
            "wf_train_weeks": WF_TRAIN_WEEKS,
        },
        "variants": {},
    }

    for key in ["A", "B", "C", "D"]:
        if key not in all_results:
            continue
        r = all_results[key]
        output["variants"][key] = {
            "name": r["variant"],
            "n_features": r["n_features"],
            "feature_names": r["feature_names"],
            "metrics": r["metrics"],
            "gates": {k: v for k, v in r["gates"].items() if k != "summary"},
            "gates_summary": r["gates"].get("summary", {}),
            "feature_importance": r.get("feature_importance", {}),
            "n_trades": r["n_trades"],
            "n_skipped": r["n_skipped"],
        }

    if best:
        output["winner"] = best["variant"]
        output["winner_metrics"] = best["metrics"]

    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    fprint(f"\nResults saved to {results_path}")

    # Save trade logs
    for key in ["A", "B", "C", "D"]:
        if key not in all_results:
            continue
        trades_path = OUTPUT_DIR / f"trades_{key}.csv"
        pd.DataFrame(all_results[key]["trades"]).to_csv(trades_path, index=False)

    # MLflow logging
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            for key in ["A", "B", "C", "D"]:
                if key not in all_results:
                    continue
                r = all_results[key]
                m = r["metrics"]
                g = r["gates"]
                with mlflow.start_run(run_name=f"variant_{key}_{r['variant']}"):
                    mlflow.log_params({
                        "variant": r["variant"],
                        "n_features": r["n_features"],
                        "dte": DTE,
                        "otm_pct": OTM_PCT,
                        "capital": CAP,
                    })
                    mlflow.log_metrics({
                        "sharpe": m.get("sharpe", 0),
                        "sortino": m.get("sortino", 0),
                        "cagr": m.get("cagr", 0),
                        "max_dd": m.get("max_dd", 0),
                        "wr": m.get("wr", 0),
                        "pf": m.get("pf", 0),
                        "calmar": m.get("calmar", 0),
                        "total_pnl": m.get("total_pnl", 0),
                        "n_trades": m.get("n_trades", 0),
                        "gates_passed": g.get("summary", {}).get("passed", 0),
                    })
            fprint("MLflow: all variants logged")
        except Exception as e:
            fprint(f"MLflow error: {e}")

    elapsed = time.time() - t0
    fprint(f"\nTotal runtime: {elapsed/60:.1f} minutes")
    fprint("=" * 70)
    fprint("DONE")
    fprint("=" * 70)

    return output


if __name__ == "__main__":
    main()
