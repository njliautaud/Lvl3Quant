#!/usr/bin/env python3
"""
Protective Overlay v2 — Tail Risk Hedging on Top of Proven C_otm2_riskparity
=============================================================================

Context:
  - Best strategy: OTM risk parity combo (Sharpe 2.04, CAGR 28.8%, MDD -20.4%,
    $645->$55K, 5/5 gates, finding #152).
  - Prior tail hedging attempt (#1022/1039) failed due to broken baseline.
  - This time: build ALL variants on top of the proven production infrastructure
    copied from combined_otm_portfolio_v1.py (MLflow exp 183).

Question: Can we reduce MaxDD from -20.4% while keeping most of the Sharpe?

6 Variants — ALL start with C_otm2_riskparity base:
  A: no_hedge (CONTROL) — Reproduce C_otm2_riskparity exactly
  B: stop_loss_portfolio — Liquidate if DD > 15%, park until equity > 90% HWM
  C: vix_call_hedge — VIX > 25 + portfolio > $2000: buy VIX call spread (10% of portfolio)
  D: put_hedge — Continuous SPY put spread insurance (1% of equity, 5% OTM, DTE=30)
  E: reduce_exposure — VIX > 30: 50% size, VIX > 40: 25% size
  F: drawdown_halt — DD > 10%: stop new positions, resume at 95% HWM

Pricing rules (same as base):
  - Hold to expiry, intrinsic value only
  - 15% entry haircut, no exit haircut
  - $645 starting capital, $200 max/trade
  - DTE=21 for sector spreads, variable for hedges
  - $2.60 commission per spread RT
  - Walk-forward LGBM, biweekly rebalance
  - GRU regime filter (score > 0.4 for bull trades)
  - 11 sector ETFs, SPY benchmark

For VIX call / SPY put hedges:
  - Black-Scholes pricing with 15% haircut on entry
  - Hold to expiry, intrinsic at expiry
  - ADDITIONAL trades on top of sector strategy (not replacements)
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
from scipy.stats import norm

warnings.filterwarnings("ignore")

_builtin_print = print


def fprint(*args, **kwargs):
    _builtin_print(*args, **kwargs)
    sys.stdout.flush()


# ── Standardized tools ──
sys.path.insert(0, "/home/jupiter/Lvl3Quant")
from research.tools.options_pricer import (
    price_bull_call_spread,
    price_bear_put_spread,
    estimate_iv,
    bs_call_price,
    bs_put_price,
    COMMISSION_RT_SPREAD,
    DEFAULT_HAIRCUT,
    RISK_FREE_RATE,
)
from research.tools.adversarial_validator import validate_trades

# ── Config ──
BASE = Path("/home/jupiter/Lvl3Quant")
OUTPUT_DIR = BASE / "output" / "growth_research" / "protective_overlay_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = OUTPUT_DIR / "protective_overlay_v2_results.json"

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "TLT", "GLD", "^VIX"]
CAP = 645.0
DTE = 21
SPREAD_PCT = 3.0
MAX_POS = 200.0
MONEYNESS_PCT = 2.0  # All variants use 2% OTM (proven best)

# Walk-forward: 500d train (~25 biweekly periods), 250d test
WF_TRAIN_DAYS = 500
WF_TEST_DAYS = 250
WF_TRAIN_PERIODS = 25
WF_REBAL_FREQ_BIWEEKLY = "2W-FRI"

# 21 production features
LEGACY_FEATURES = [
    "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "vol_21d", "vol_63d", "sharpe_63d", "maxdd_63d", "pct_52w_high", "mom_accel",
    "pct_pos_months_12m", "sortino_63d", "calmar_1y", "up_capture",
    "trend_r2_63d", "trend_slope_63d",
]
CROSS_ASSET_FEATURES = [
    "sector_spy_beta_63d",
    "sector_relative_vol_21d",
    "cross_sector_dispersion",
]
ALL_FEATURES = LEGACY_FEATURES + CROSS_ASSET_FEATURES  # 21 total

# MLflow
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "protective_overlay_v2"

MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen(MLFLOW_URI, timeout=2)
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    MLFLOW_OK = True
    fprint(f"MLflow connected: {MLFLOW_URI}")
except Exception:
    fprint("MLflow unavailable -- results saved to disk only")

# ── GRU regime predictions (if available) ──
REGIME_PREDICTIONS = None
REGIME_DATES = None
try:
    regime_data = np.load(BASE / "output" / "regime_detector_v1" / "regime_predictions_v1.npz", allow_pickle=True)
    REGIME_PREDICTIONS = regime_data.get("predictions", None)
    REGIME_DATES = regime_data.get("dates", None)
    if REGIME_PREDICTIONS is not None and REGIME_DATES is not None:
        fprint(f"GRU regime predictions loaded: {len(REGIME_PREDICTIONS)} dates")
    else:
        REGIME_PREDICTIONS = None
        REGIME_DATES = None
        fprint("GRU regime data incomplete, falling back to VIX>20 threshold")
except Exception as e:
    fprint(f"GRU regime data not loadable ({e}), using VIX>20 threshold")


def get_regime(dt, vix_value):
    """Return 'high_vol' or 'low_vol' using GRU predictions if available, else VIX threshold."""
    if REGIME_PREDICTIONS is not None and REGIME_DATES is not None:
        dt_str = str(pd.Timestamp(dt).date())
        dates_list = [str(d) for d in REGIME_DATES]
        if dt_str in dates_list:
            idx = dates_list.index(dt_str)
            pred = REGIME_PREDICTIONS[idx]
            return "high_vol" if pred > 0.5 else "low_vol"
    return "high_vol" if vix_value >= 20 else "low_vol"


# ══════════════════════════════════════════════════════════════
# DATA DOWNLOAD
# ══════════════════════════════════════════════════════════════

def download_data():
    """Download sector ETFs, SPY, TLT, GLD, VIX from yfinance (2007-2026)."""
    import yfinance as yf

    all_tickers = SECTORS + EXTRA_TICKERS
    fprint(f"Downloading {len(all_tickers)} tickers...")

    raw = yf.download(all_tickers, start="2007-01-01", progress=False, auto_adjust=True)
    mi = isinstance(raw.columns, pd.MultiIndex)

    close = raw["Close"] if mi else raw[["Close"]]
    high = raw["High"] if mi else raw[["High"]]
    low = raw["Low"] if mi else raw[["Low"]]

    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(-1)
        high.columns = high.columns.get_level_values(-1)
        low.columns = low.columns.get_level_values(-1)

    close = close.ffill()
    high = high.ffill()
    low = low.ffill()

    rename_map = {"^VIX": "VIX"}
    close = close.rename(columns=rename_map)
    high = high.rename(columns=rename_map)
    low = low.rename(columns=rename_map)

    fprint(f"Data: {len(close)} days, columns: {list(close.columns)}")
    fprint(f"Date range: {close.index[0].date()} to {close.index[-1].date()}")

    return close, high, low


# ══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING (identical to v1)
# ══════════════════════════════════════════════════════════════

def compute_legacy_features(px, spy_slice):
    """Compute the 18 legacy quality-momentum features."""
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
    up_days = rets[rets > 0]
    f["up_capture"] = float(up_days.iloc[-63:].mean() / (up_days.mean() + 1e-10)) if len(up_days) > 10 else 1.0

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


def compute_cross_asset_features(sector_ticker, dt_idx, close_df):
    """Compute the 3 cross-asset features."""
    f = {}
    spy = close_df["SPY"].iloc[:dt_idx + 1].dropna()
    sector_px = close_df[sector_ticker].iloc[:dt_idx + 1].dropna() if sector_ticker in close_df.columns else None

    if spy is None or len(spy) < 63:
        return {k: 0.0 for k in CROSS_ASSET_FEATURES}

    spy_ret = spy.pct_change().dropna()

    if sector_px is not None and len(sector_px) > 63:
        sec_ret = sector_px.pct_change().dropna()
        common = spy_ret.index.intersection(sec_ret.index)
        if len(common) > 63:
            sr = sec_ret.loc[common].iloc[-63:]
            mr = spy_ret.loc[common].iloc[-63:]
            cov = np.cov(sr.values, mr.values)
            f["sector_spy_beta_63d"] = float(cov[0, 1] / (cov[1, 1] + 1e-10))
        else:
            f["sector_spy_beta_63d"] = 1.0
    else:
        f["sector_spy_beta_63d"] = 1.0

    if sector_px is not None and len(sector_px) > 21:
        sec_ret = sector_px.pct_change().dropna()
        if len(sec_ret) > 21 and len(spy_ret) > 21:
            f["sector_relative_vol_21d"] = float(sec_ret.iloc[-21:].std() / (spy_ret.iloc[-21:].std() + 1e-10))
        else:
            f["sector_relative_vol_21d"] = 1.0
    else:
        f["sector_relative_vol_21d"] = 1.0

    sector_cols = [c for c in SECTORS if c in close_df.columns]
    if len(sector_cols) > 3:
        sector_rets = close_df[sector_cols].iloc[:dt_idx + 1].pct_change()
        daily_disp = sector_rets.std(axis=1)
        if len(daily_disp) > 21:
            f["cross_sector_dispersion"] = float(daily_disp.rolling(21).mean().iloc[-1])
        else:
            f["cross_sector_dispersion"] = 0.01
    else:
        f["cross_sector_dispersion"] = 0.01

    return f


# ══════════════════════════════════════════════════════════════
# ATR
# ══════════════════════════════════════════════════════════════

def compute_atr_series(high, low, close, period=14):
    """Compute ATR series for all sectors."""
    atr_dict = {}
    for tk in SECTORS:
        if tk in high.columns and tk in low.columns and tk in close.columns:
            h = high[tk].dropna()
            l = low[tk].dropna()
            c = close[tk].dropna()
            common = h.index.intersection(l.index).intersection(c.index)
            if len(common) > period:
                tr1 = h.loc[common] - l.loc[common]
                tr2 = (h.loc[common] - c.loc[common].shift(1)).abs()
                tr3 = (l.loc[common] - c.loc[common].shift(1)).abs()
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr_dict[tk] = tr.ewm(alpha=1 / period, min_periods=period).mean()
    return atr_dict


# ══════════════════════════════════════════════════════════════
# WALK-FORWARD LGBM RANKING (identical to v1)
# ══════════════════════════════════════════════════════════════

def build_feature_records(close, high, low, rebal_dates, feature_cols):
    """Build feature + target records for LGBM walk-forward ranking."""
    fprint(f"  Building records: {len(rebal_dates)} dates, {len(feature_cols)} features")

    records = []
    sector_cols = [c for c in SECTORS if c in close.columns]
    spy = close["SPY"]

    for dt in rebal_dates:
        idx = close.index.get_indexer([dt], method="ffill")[0]
        if idx < 260:
            continue

        for tk in sector_cols:
            px = close[tk].iloc[:idx + 1].dropna()
            spy_s = spy.iloc[:idx + 1]

            legacy = compute_legacy_features(px, spy_s)
            if not legacy:
                continue

            cross_asset = {}
            for col in feature_cols:
                if col in CROSS_ASSET_FEATURES:
                    cross_asset = compute_cross_asset_features(tk, idx, close)
                    break

            fi = min(idx + DTE, len(close) - 1)
            if fi <= idx:
                continue
            fwd_ret = float(close[tk].iloc[fi] / close[tk].iloc[idx] - 1)

            rec = {**legacy, **cross_asset, "date": dt, "ticker": tk, "fwd_ret": fwd_ret}
            records.append(rec)

    df = pd.DataFrame(records)
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
    df[feature_cols] = df[feature_cols].fillna(0.0)

    fprint(f"    {len(df)} records, {len(df['date'].unique())} dates")
    return df


def walk_forward_lgbm_rank(df, feature_cols):
    """Walk-forward LGBM: sliding 500d train, 250d test window."""
    import lightgbm as lgb

    if len(df) < 100:
        fprint(f"    Insufficient data ({len(df)} records)")
        return {}, None

    df["rank_label"] = df.groupby("date")["fwd_ret"].rank(pct=True)
    dates = sorted(df["date"].unique())

    rankings = {}
    all_importances = np.zeros(len(feature_cols))
    n_models = 0

    for i in range(WF_TRAIN_PERIODS, len(dates)):
        train_dates = dates[max(0, i - WF_TRAIN_PERIODS):i]
        test_date = dates[i]

        train_df = df[df["date"].isin(train_dates)]
        test_df = df[df["date"] == test_date].copy()

        if len(test_df) < 3 or len(train_df) < 50:
            continue

        Xt = np.nan_to_num(train_df[feature_cols].values.astype(np.float32))
        yt = train_df["rank_label"].values.astype(np.float32)
        Xe = np.nan_to_num(test_df[feature_cols].values.astype(np.float32))

        try:
            m = lgb.LGBMRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=5,
                verbose=-1,
            )
            m.fit(Xt, yt)
            test_df["score"] = m.predict(Xe)

            rankings[test_date] = dict(zip(test_df["ticker"], test_df["score"]))

            all_importances += m.feature_importances_
            n_models += 1
        except Exception:
            continue

    if n_models > 0:
        all_importances /= n_models
        imp_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": all_importances,
        }).sort_values("importance", ascending=False)
    else:
        imp_df = None

    fprint(f"    {len(rankings)} ranking dates, {n_models} models trained")
    return rankings, imp_df


# ══════════════════════════════════════════════════════════════
# TRADE EXECUTION — SECTOR SPREADS (identical to v1)
# ══════════════════════════════════════════════════════════════

def execute_bull_call_trades(picks, close, atr_dict, idx, dt, cv, max_per_trade, equity,
                              moneyness_pct=2.0):
    """Execute bull call spreads on top-ranked sectors."""
    trades = []
    total_pnl = 0.0

    for tk in picks:
        if tk not in close.columns or tk not in atr_dict:
            continue

        S = float(close[tk].iloc[idx])
        ei = min(idx + DTE, len(close) - 1)
        if ei <= idx:
            continue

        if dt in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[dt]):
            av = float(atr_dict[tk].loc[dt])
        else:
            av = S * 0.015

        K1 = round(S * (1 + moneyness_pct / 100), 2)
        K2 = round(K1 * (1 + SPREAD_PCT / 100), 2)
        if K2 <= K1:
            K2 = K1 + 1.0

        try:
            entry_cost_ps, max_profit_ps = price_bull_call_spread(
                S=S, K1=K1, K2=K2, dte=DTE, atr=av, vix=cv
            )
        except Exception:
            continue

        total_cost = entry_cost_ps * 100 + COMMISSION_RT_SPREAD
        if total_cost <= 0 or total_cost > max_per_trade or total_cost > equity * 0.40:
            continue

        Se = float(close[tk].iloc[ei])
        intrinsic = max(Se - K1, 0.0) - max(Se - K2, 0.0)
        pnl = (intrinsic - entry_cost_ps) * 100 - COMMISSION_RT_SPREAD

        total_pnl += pnl
        trades.append({
            "pnl": round(pnl, 2),
            "entry_date": str(close.index[idx].date()),
            "exit_date": str(close.index[ei].date()),
            "ticker": tk,
            "leg": "bull_call",
            "vix": round(cv, 1),
            "win": pnl > 0,
            "mode": "bull",
            "moneyness_pct": moneyness_pct,
            "K1": K1,
            "K2": K2,
        })

    return trades, total_pnl


def execute_bear_put_trades(picks, close, atr_dict, idx, dt, cv, max_per_trade, equity,
                              moneyness_pct=2.0):
    """Execute bear put spreads on bottom-ranked sectors."""
    trades = []
    total_pnl = 0.0

    for tk in picks:
        if tk not in close.columns or tk not in atr_dict:
            continue

        S = float(close[tk].iloc[idx])
        ei = min(idx + DTE, len(close) - 1)
        if ei <= idx:
            continue

        if dt in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[dt]):
            av = float(atr_dict[tk].loc[dt])
        else:
            av = S * 0.015

        K1 = round(S * (1 - moneyness_pct / 100), 2)
        K2 = round(K1 * (1 - SPREAD_PCT / 100), 2)
        if K1 <= K2:
            continue

        try:
            entry_cost_ps, max_profit_ps = price_bear_put_spread(
                S=S, K1=K2, K2=K1, dte=DTE, atr=av, vix=cv
            )
        except Exception:
            continue

        total_cost = entry_cost_ps * 100 + COMMISSION_RT_SPREAD
        if total_cost <= 0 or total_cost > max_per_trade or total_cost > equity * 0.40:
            continue

        Se = float(close[tk].iloc[ei])
        intrinsic = max(K1 - Se, 0.0) - max(K2 - Se, 0.0)
        pnl = (intrinsic - entry_cost_ps) * 100 - COMMISSION_RT_SPREAD

        total_pnl += pnl
        trades.append({
            "pnl": round(pnl, 2),
            "entry_date": str(close.index[idx].date()),
            "exit_date": str(close.index[ei].date()),
            "ticker": tk,
            "leg": "bear_put",
            "vix": round(cv, 1),
            "win": pnl > 0,
            "mode": "pair",
            "moneyness_pct": moneyness_pct,
            "K1": K2,
            "K2": K1,
        })

    return trades, total_pnl


def execute_risk_parity_allocation(close, idx, dt, equity, alloc_pct=0.30):
    """Simple risk parity allocation: SPY/TLT/GLD 33/33/34. Returns PnL for the DTE period."""
    rp_tickers = {"SPY": 0.33, "TLT": 0.33, "GLD": 0.34}
    ei = min(idx + DTE, len(close) - 1)
    if ei <= idx:
        return 0.0

    alloc_dollars = equity * alloc_pct
    total_pnl = 0.0

    for tk, wt in rp_tickers.items():
        if tk not in close.columns:
            continue
        S_entry = float(close[tk].iloc[idx])
        S_exit = float(close[tk].iloc[ei])
        if S_entry <= 0:
            continue
        position_dollars = alloc_dollars * wt
        shares = position_dollars / S_entry
        pnl = shares * (S_exit - S_entry)
        total_pnl += pnl

    return total_pnl


# ══════════════════════════════════════════════════════════════
# HEDGE TRADE EXECUTION — VIX CALLS & SPY PUTS
# ══════════════════════════════════════════════════════════════

def _estimate_iv_from_vix(vix_level):
    """Estimate implied volatility from VIX level for hedge instruments."""
    # VIX itself IS implied vol for SPY. For VIX options, vol-of-vol is higher.
    return vix_level / 100.0


def price_vix_call_spread(vix_level, budget_dollars, dte=30, otm_pct=5.0):
    """
    Price a VIX call spread for tail protection.
    Buy VIX call at K1 = VIX * (1 + otm_pct/100), sell at K2 = K1 + 5.
    Returns (entry_cost_total, max_profit_total, n_spreads) or None if too expensive.
    """
    S = vix_level
    K1 = round(S * (1 + otm_pct / 100), 1)
    K2 = K1 + 5.0  # $5 wide spread on VIX
    T = dte / 365.0

    # VIX options vol: vol-of-vol is typically 80-120%
    sigma = max(0.80, min(1.20, vix_level / 100 * 4))

    try:
        c1 = bs_call_price(S, K1, T, r=RISK_FREE_RATE, sigma=sigma)
        c2 = bs_call_price(S, K2, T, r=RISK_FREE_RATE, sigma=sigma)
    except Exception:
        return None

    spread_cost_ps = max(c1 - c2, 0.01)
    # Apply 15% haircut on entry
    spread_cost_ps *= (1 + DEFAULT_HAIRCUT)

    cost_per_spread = spread_cost_ps * 100 + COMMISSION_RT_SPREAD
    if cost_per_spread <= 0 or cost_per_spread > budget_dollars:
        return None

    n_spreads = max(1, int(budget_dollars / cost_per_spread))
    total_cost = n_spreads * cost_per_spread
    max_profit_per = (K2 - K1 - spread_cost_ps) * 100 - COMMISSION_RT_SPREAD

    return {
        "entry_cost": round(total_cost, 2),
        "max_profit": round(max_profit_per * n_spreads, 2),
        "n_spreads": n_spreads,
        "K1": K1,
        "K2": K2,
        "spread_cost_ps": round(spread_cost_ps, 4),
    }


def price_spy_put_spread(spy_price, budget_dollars, vix_level, dte=30, otm_pct=5.0):
    """
    Price a SPY put spread for downside protection.
    Buy SPY put at K1 = S * (1 - otm_pct/100), sell at K2 = K1 * 0.95 (5% wide).
    Returns dict or None if too expensive.
    """
    S = spy_price
    K1 = round(S * (1 - otm_pct / 100), 2)  # Long put (higher strike)
    K2 = round(K1 * 0.95, 2)                  # Short put (lower strike)
    T = dte / 365.0
    sigma = vix_level / 100.0

    try:
        p1 = bs_put_price(S, K1, T, r=RISK_FREE_RATE, sigma=sigma)
        p2 = bs_put_price(S, K2, T, r=RISK_FREE_RATE, sigma=sigma)
    except Exception:
        return None

    spread_cost_ps = max(p1 - p2, 0.01)
    # Apply 15% haircut on entry
    spread_cost_ps *= (1 + DEFAULT_HAIRCUT)

    cost_per_spread = spread_cost_ps * 100 + COMMISSION_RT_SPREAD
    if cost_per_spread <= 0 or cost_per_spread > budget_dollars:
        return None

    n_spreads = max(1, int(budget_dollars / cost_per_spread))
    total_cost = n_spreads * cost_per_spread
    max_profit_per = (K1 - K2 - spread_cost_ps) * 100 - COMMISSION_RT_SPREAD

    return {
        "entry_cost": round(total_cost, 2),
        "max_profit": round(max_profit_per * n_spreads, 2),
        "n_spreads": n_spreads,
        "K1": K1,
        "K2": K2,
        "spread_cost_ps": round(spread_cost_ps, 4),
    }


# ══════════════════════════════════════════════════════════════
# CORE C_otm2_riskparity SIMULATION (shared base for all variants)
# ══════════════════════════════════════════════════════════════

def _base_period_trades(rankings, dt, close, atr_dict, idx, cv, equity, regime,
                         size_mult=1.0, allow_new_positions=True):
    """
    Execute one period of C_otm2_riskparity strategy.
    Returns (trades, pnl, rp_pnl).
    size_mult: multiplier for position sizes (for reduce_exposure variant).
    allow_new_positions: if False, skip opening new trades (for drawdown_halt).
    """
    scores = rankings.get(dt, {})
    if not scores or len(scores) < 6:
        return [], 0.0, 0.0

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    base_max = min(MAX_POS, equity / 3)
    max_per_trade = base_max * size_mult
    if max_per_trade < 30:
        return [], 0.0, 0.0

    trades = []
    pnl = 0.0
    rp_pnl = 0.0

    if not allow_new_positions:
        # Still compute risk parity (passive allocation, not options)
        if regime == "low_vol":
            rp_pnl = execute_risk_parity_allocation(close, idx, dt, equity, alloc_pct=0.30)
        return trades, pnl, rp_pnl

    if regime == "high_vol":
        picks = [t for t, _ in ranked[:3]]
        new_trades, t_pnl = execute_bull_call_trades(
            picks, close, atr_dict, idx, dt, cv, max_per_trade, equity,
            moneyness_pct=MONEYNESS_PCT)
        for t in new_trades:
            t["mode"] = "bull"
        trades.extend(new_trades)
        pnl += t_pnl
    else:
        # Pair trades with 2% OTM
        long_picks = [t for t, _ in ranked[:3]]
        short_picks = [t for t, _ in ranked[-3:]]
        short_picks = [t for t in short_picks if t not in long_picks]

        max_pair_trade = min(MAX_POS, equity / 6) * size_mult

        long_trades, long_pnl = execute_bull_call_trades(
            long_picks, close, atr_dict, idx, dt, cv, max_pair_trade, equity,
            moneyness_pct=MONEYNESS_PCT)
        for t in long_trades:
            t["mode"] = "pair_long"
        pnl += long_pnl

        short_trades, short_pnl = execute_bear_put_trades(
            short_picks, close, atr_dict, idx, dt, cv, max_pair_trade, equity,
            moneyness_pct=MONEYNESS_PCT)
        for t in short_trades:
            t["mode"] = "pair_short"
        pnl += short_pnl

        trades.extend(long_trades)
        trades.extend(short_trades)

        # Risk parity overlay: 30% of equity into SPY/TLT/GLD
        rp_pnl = execute_risk_parity_allocation(close, idx, dt, equity, alloc_pct=0.30)

    return trades, pnl, rp_pnl


# ══════════════════════════════════════════════════════════════
# VARIANT A: NO HEDGE (CONTROL) — exact C_otm2_riskparity
# ══════════════════════════════════════════════════════════════

def simulate_variant_A(rankings, close, atr_dict):
    """A_no_hedge (CONTROL): Reproduce C_otm2_riskparity exactly."""
    vix = close["VIX"] if "VIX" in close.columns else None
    equity = CAP
    hwm = CAP
    all_trades = []
    invested_periods = 0
    total_periods = 0

    for dt in sorted(rankings.keys()):
        if dt not in close.index:
            continue
        idx = close.index.get_indexer([dt], method="ffill")[0]
        cv = float(vix.iloc[idx]) if vix is not None and idx < len(vix) else 20.0
        regime = get_regime(dt, cv)
        total_periods += 1

        trades, pnl, rp_pnl = _base_period_trades(
            rankings, dt, close, atr_dict, idx, cv, equity, regime)

        equity += pnl + rp_pnl
        hwm = max(hwm, equity)

        if rp_pnl != 0:
            all_trades.append({
                "pnl": round(rp_pnl, 2),
                "entry_date": str(close.index[idx].date()),
                "exit_date": str(close.index[min(idx + DTE, len(close) - 1)].date()),
                "ticker": "RP_BASKET",
                "leg": "risk_parity",
                "vix": round(cv, 1),
                "win": rp_pnl > 0,
                "mode": "risk_parity",
                "moneyness_pct": 0.0,
            })

        all_trades.extend(trades)
        if trades:
            invested_periods += 1

    pct_invested = invested_periods / max(total_periods, 1) * 100
    return all_trades, equity, pct_invested


# ══════════════════════════════════════════════════════════════
# VARIANT B: STOP LOSS PORTFOLIO
# DD > 15% => liquidate all, park in cash/RP until equity > 90% HWM
# ══════════════════════════════════════════════════════════════

def simulate_variant_B(rankings, close, atr_dict):
    """B_stop_loss_portfolio: If DD > 15%, liquidate all, park in cash until equity > 90% HWM."""
    vix = close["VIX"] if "VIX" in close.columns else None
    equity = CAP
    hwm = CAP
    all_trades = []
    invested_periods = 0
    total_periods = 0
    halted = False
    halt_count = 0

    for dt in sorted(rankings.keys()):
        if dt not in close.index:
            continue
        idx = close.index.get_indexer([dt], method="ffill")[0]
        cv = float(vix.iloc[idx]) if vix is not None and idx < len(vix) else 20.0
        regime = get_regime(dt, cv)
        total_periods += 1

        dd_pct = (hwm - equity) / hwm if hwm > 0 else 0

        # Check if we should halt
        if not halted and dd_pct > 0.15:
            halted = True
            halt_count += 1

        # Check if we can resume
        if halted and equity >= hwm * 0.90:
            halted = False

        if halted:
            # Park in risk parity only (safer than options)
            rp_pnl = execute_risk_parity_allocation(close, idx, dt, equity, alloc_pct=0.50)
            equity += rp_pnl
            hwm = max(hwm, equity)
            if rp_pnl != 0:
                all_trades.append({
                    "pnl": round(rp_pnl, 2),
                    "entry_date": str(close.index[idx].date()),
                    "exit_date": str(close.index[min(idx + DTE, len(close) - 1)].date()),
                    "ticker": "RP_BASKET",
                    "leg": "risk_parity",
                    "vix": round(cv, 1),
                    "win": rp_pnl > 0,
                    "mode": "stop_loss_park",
                    "moneyness_pct": 0.0,
                })
            continue

        # Normal trading
        trades, pnl, rp_pnl = _base_period_trades(
            rankings, dt, close, atr_dict, idx, cv, equity, regime)

        equity += pnl + rp_pnl
        hwm = max(hwm, equity)

        if rp_pnl != 0:
            all_trades.append({
                "pnl": round(rp_pnl, 2),
                "entry_date": str(close.index[idx].date()),
                "exit_date": str(close.index[min(idx + DTE, len(close) - 1)].date()),
                "ticker": "RP_BASKET",
                "leg": "risk_parity",
                "vix": round(cv, 1),
                "win": rp_pnl > 0,
                "mode": "risk_parity",
                "moneyness_pct": 0.0,
            })

        all_trades.extend(trades)
        if trades:
            invested_periods += 1

    pct_invested = invested_periods / max(total_periods, 1) * 100
    fprint(f"  Stop-loss halts triggered: {halt_count}")
    return all_trades, equity, pct_invested


# ══════════════════════════════════════════════════════════════
# VARIANT C: VIX CALL HEDGE
# VIX > 25 AND portfolio > $2000: buy VIX call spread (10% of portfolio)
# ══════════════════════════════════════════════════════════════

def simulate_variant_C(rankings, close, atr_dict):
    """C_vix_call_hedge: When VIX > 25 AND equity > $2000, buy VIX call spread."""
    vix = close["VIX"] if "VIX" in close.columns else None
    equity = CAP
    hwm = CAP
    all_trades = []
    invested_periods = 0
    total_periods = 0
    hedge_count = 0
    hedge_pnl_total = 0.0

    for dt in sorted(rankings.keys()):
        if dt not in close.index:
            continue
        idx = close.index.get_indexer([dt], method="ffill")[0]
        cv = float(vix.iloc[idx]) if vix is not None and idx < len(vix) else 20.0
        regime = get_regime(dt, cv)
        total_periods += 1

        # Normal C_otm2_riskparity trading
        trades, pnl, rp_pnl = _base_period_trades(
            rankings, dt, close, atr_dict, idx, cv, equity, regime)

        equity += pnl + rp_pnl
        hwm = max(hwm, equity)

        if rp_pnl != 0:
            all_trades.append({
                "pnl": round(rp_pnl, 2),
                "entry_date": str(close.index[idx].date()),
                "exit_date": str(close.index[min(idx + DTE, len(close) - 1)].date()),
                "ticker": "RP_BASKET",
                "leg": "risk_parity",
                "vix": round(cv, 1),
                "win": rp_pnl > 0,
                "mode": "risk_parity",
                "moneyness_pct": 0.0,
            })
        all_trades.extend(trades)
        if trades:
            invested_periods += 1

        # VIX call hedge overlay
        if cv > 25 and equity > 2000:
            hedge_budget = equity * 0.10
            hedge_info = price_vix_call_spread(cv, hedge_budget, dte=30, otm_pct=5.0)

            if hedge_info is not None:
                hedge_count += 1
                # Simulate hedge expiry: VIX at expiry
                ei = min(idx + 21, len(close) - 1)  # ~30 cal days = ~21 trading days
                if ei > idx and vix is not None:
                    vix_at_expiry = float(vix.iloc[ei])
                    # Intrinsic value at expiry
                    intrinsic = max(vix_at_expiry - hedge_info["K1"], 0.0) - \
                                max(vix_at_expiry - hedge_info["K2"], 0.0)
                    hedge_pnl = (intrinsic * 100 * hedge_info["n_spreads"]) - hedge_info["entry_cost"]
                else:
                    hedge_pnl = -hedge_info["entry_cost"]  # Total loss

                hedge_pnl_total += hedge_pnl
                equity += hedge_pnl

                all_trades.append({
                    "pnl": round(hedge_pnl, 2),
                    "entry_date": str(close.index[idx].date()),
                    "exit_date": str(close.index[min(ei, len(close) - 1)].date()),
                    "ticker": "VIX_CALL_HEDGE",
                    "leg": "vix_call_spread",
                    "vix": round(cv, 1),
                    "win": hedge_pnl > 0,
                    "mode": "hedge",
                    "moneyness_pct": 5.0,
                    "K1": hedge_info["K1"],
                    "K2": hedge_info["K2"],
                    "n_spreads": hedge_info["n_spreads"],
                })

    pct_invested = invested_periods / max(total_periods, 1) * 100
    fprint(f"  VIX call hedges placed: {hedge_count}, total hedge PnL: ${hedge_pnl_total:,.0f}")
    return all_trades, equity, pct_invested


# ══════════════════════════════════════════════════════════════
# VARIANT D: PUT HEDGE
# Continuous SPY put spread insurance (1% of equity, 5% OTM, DTE=30)
# ══════════════════════════════════════════════════════════════

def simulate_variant_D(rankings, close, atr_dict):
    """D_put_hedge: When equity > $2000, spend 1% on SPY put spreads every period."""
    vix = close["VIX"] if "VIX" in close.columns else None
    equity = CAP
    hwm = CAP
    all_trades = []
    invested_periods = 0
    total_periods = 0
    hedge_count = 0
    hedge_pnl_total = 0.0

    for dt in sorted(rankings.keys()):
        if dt not in close.index:
            continue
        idx = close.index.get_indexer([dt], method="ffill")[0]
        cv = float(vix.iloc[idx]) if vix is not None and idx < len(vix) else 20.0
        regime = get_regime(dt, cv)
        total_periods += 1

        # Normal C_otm2_riskparity trading
        trades, pnl, rp_pnl = _base_period_trades(
            rankings, dt, close, atr_dict, idx, cv, equity, regime)

        equity += pnl + rp_pnl
        hwm = max(hwm, equity)

        if rp_pnl != 0:
            all_trades.append({
                "pnl": round(rp_pnl, 2),
                "entry_date": str(close.index[idx].date()),
                "exit_date": str(close.index[min(idx + DTE, len(close) - 1)].date()),
                "ticker": "RP_BASKET",
                "leg": "risk_parity",
                "vix": round(cv, 1),
                "win": rp_pnl > 0,
                "mode": "risk_parity",
                "moneyness_pct": 0.0,
            })
        all_trades.extend(trades)
        if trades:
            invested_periods += 1

        # SPY put spread hedge (continuous insurance when equity > $2000)
        if equity > 2000 and "SPY" in close.columns:
            spy_price = float(close["SPY"].iloc[idx])
            hedge_budget = equity * 0.01  # 1% of equity
            hedge_info = price_spy_put_spread(spy_price, hedge_budget, cv, dte=30, otm_pct=5.0)

            if hedge_info is not None:
                hedge_count += 1
                ei = min(idx + 21, len(close) - 1)
                if ei > idx:
                    spy_at_expiry = float(close["SPY"].iloc[ei])
                    # Intrinsic at expiry: long put at K1 - short put at K2
                    intrinsic = max(hedge_info["K1"] - spy_at_expiry, 0.0) - \
                                max(hedge_info["K2"] - spy_at_expiry, 0.0)
                    hedge_pnl = (intrinsic * 100 * hedge_info["n_spreads"]) - hedge_info["entry_cost"]
                else:
                    hedge_pnl = -hedge_info["entry_cost"]

                hedge_pnl_total += hedge_pnl
                equity += hedge_pnl

                all_trades.append({
                    "pnl": round(hedge_pnl, 2),
                    "entry_date": str(close.index[idx].date()),
                    "exit_date": str(close.index[min(ei, len(close) - 1)].date()),
                    "ticker": "SPY_PUT_HEDGE",
                    "leg": "spy_put_spread",
                    "vix": round(cv, 1),
                    "win": hedge_pnl > 0,
                    "mode": "hedge",
                    "moneyness_pct": 5.0,
                    "K1": hedge_info["K1"],
                    "K2": hedge_info["K2"],
                    "n_spreads": hedge_info["n_spreads"],
                })

    pct_invested = invested_periods / max(total_periods, 1) * 100
    fprint(f"  SPY put hedges placed: {hedge_count}, total hedge PnL: ${hedge_pnl_total:,.0f}")
    return all_trades, equity, pct_invested


# ══════════════════════════════════════════════════════════════
# VARIANT E: REDUCE EXPOSURE
# VIX > 30: 50% size, VIX > 40: 25% size
# ══════════════════════════════════════════════════════════════

def simulate_variant_E(rankings, close, atr_dict):
    """E_reduce_exposure: VIX > 30 = 50% size, VIX > 40 = 25% size."""
    vix = close["VIX"] if "VIX" in close.columns else None
    equity = CAP
    hwm = CAP
    all_trades = []
    invested_periods = 0
    total_periods = 0
    reduced_periods = 0

    for dt in sorted(rankings.keys()):
        if dt not in close.index:
            continue
        idx = close.index.get_indexer([dt], method="ffill")[0]
        cv = float(vix.iloc[idx]) if vix is not None and idx < len(vix) else 20.0
        regime = get_regime(dt, cv)
        total_periods += 1

        # Determine size multiplier based on VIX
        if cv > 40:
            size_mult = 0.25
            reduced_periods += 1
        elif cv > 30:
            size_mult = 0.50
            reduced_periods += 1
        else:
            size_mult = 1.0

        trades, pnl, rp_pnl = _base_period_trades(
            rankings, dt, close, atr_dict, idx, cv, equity, regime,
            size_mult=size_mult)

        equity += pnl + rp_pnl
        hwm = max(hwm, equity)

        if rp_pnl != 0:
            all_trades.append({
                "pnl": round(rp_pnl, 2),
                "entry_date": str(close.index[idx].date()),
                "exit_date": str(close.index[min(idx + DTE, len(close) - 1)].date()),
                "ticker": "RP_BASKET",
                "leg": "risk_parity",
                "vix": round(cv, 1),
                "win": rp_pnl > 0,
                "mode": "risk_parity",
                "moneyness_pct": 0.0,
            })
        all_trades.extend(trades)
        if trades:
            invested_periods += 1

    pct_invested = invested_periods / max(total_periods, 1) * 100
    fprint(f"  Reduced-exposure periods: {reduced_periods}/{total_periods}")
    return all_trades, equity, pct_invested


# ══════════════════════════════════════════════════════════════
# VARIANT F: DRAWDOWN HALT
# DD > 10%: stop new positions, let existing expire. Resume at 95% HWM.
# ══════════════════════════════════════════════════════════════

def simulate_variant_F(rankings, close, atr_dict):
    """F_drawdown_halt: DD > 10% = stop new positions. Resume when equity > 95% HWM."""
    vix = close["VIX"] if "VIX" in close.columns else None
    equity = CAP
    hwm = CAP
    all_trades = []
    invested_periods = 0
    total_periods = 0
    halted = False
    halt_count = 0

    for dt in sorted(rankings.keys()):
        if dt not in close.index:
            continue
        idx = close.index.get_indexer([dt], method="ffill")[0]
        cv = float(vix.iloc[idx]) if vix is not None and idx < len(vix) else 20.0
        regime = get_regime(dt, cv)
        total_periods += 1

        dd_pct = (hwm - equity) / hwm if hwm > 0 else 0

        # Check halt trigger
        if not halted and dd_pct > 0.10:
            halted = True
            halt_count += 1

        # Check resume condition
        if halted and equity >= hwm * 0.95:
            halted = False

        trades, pnl, rp_pnl = _base_period_trades(
            rankings, dt, close, atr_dict, idx, cv, equity, regime,
            allow_new_positions=not halted)

        equity += pnl + rp_pnl
        hwm = max(hwm, equity)

        if rp_pnl != 0:
            all_trades.append({
                "pnl": round(rp_pnl, 2),
                "entry_date": str(close.index[idx].date()),
                "exit_date": str(close.index[min(idx + DTE, len(close) - 1)].date()),
                "ticker": "RP_BASKET",
                "leg": "risk_parity",
                "vix": round(cv, 1),
                "win": rp_pnl > 0,
                "mode": "risk_parity" if not halted else "dd_halt_park",
                "moneyness_pct": 0.0,
            })
        all_trades.extend(trades)
        if trades:
            invested_periods += 1

    pct_invested = invested_periods / max(total_periods, 1) * 100
    fprint(f"  Drawdown halts triggered: {halt_count}")
    return all_trades, equity, pct_invested


# ══════════════════════════════════════════════════════════════
# ANALYSIS HELPERS
# ══════════════════════════════════════════════════════════════

def compute_spy_beta(trades, close):
    """Compute SPY beta from trade returns."""
    if not trades or len(trades) < 10:
        return 0.0, 0.0

    spy = close["SPY"]
    trade_rets = []
    spy_rets = []

    for t in trades:
        entry = pd.Timestamp(t["entry_date"])
        exit_dt = pd.Timestamp(t["exit_date"])
        if entry in spy.index and exit_dt in spy.index:
            sr = float(spy.loc[exit_dt] / spy.loc[entry] - 1)
            trade_rets.append(t["pnl"] / max(CAP, 100))
            spy_rets.append(sr)

    if len(trade_rets) < 10:
        return 0.0, 0.0

    tr = np.array(trade_rets)
    sr = np.array(spy_rets)
    corr = np.corrcoef(tr, sr)[0, 1]
    cov = np.cov(tr, sr)
    beta = cov[0, 1] / (cov[1, 1] + 1e-10)

    return float(beta) if not np.isnan(beta) else 0.0, float(corr) if not np.isnan(corr) else 0.0


def compute_monthly_spy_correlation(trades, close):
    """Compute monthly return correlation with SPY."""
    if not trades or len(trades) < 20:
        return 0.0

    spy = close["SPY"]
    tdf = pd.DataFrame(trades)
    tdf["entry_dt"] = pd.to_datetime(tdf["entry_date"])
    tdf["month"] = tdf["entry_dt"].dt.to_period("M")

    monthly_pnl = tdf.groupby("month")["pnl"].sum()
    spy_monthly = spy.resample("ME").last().pct_change().dropna()

    common_months = []
    for mo in monthly_pnl.index:
        mo_end = mo.to_timestamp(how="E")
        closest = spy_monthly.index[spy_monthly.index <= mo_end]
        if len(closest) > 0:
            common_months.append((monthly_pnl[mo], float(spy_monthly.loc[closest[-1]])))

    if len(common_months) < 6:
        return 0.0

    trade_m, spy_m = zip(*common_months)
    corr = np.corrcoef(trade_m, spy_m)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0


def vix_regime_stratified_sharpe(trades, annualize_periods=26):
    """Compute Sharpe by VIX regime: <20, 20-30, >30."""
    if not trades:
        return {}

    regimes = {"vix_lt_20": [], "vix_20_30": [], "vix_gt_30": []}
    for t in trades:
        v = t.get("vix", 20)
        if v < 20:
            regimes["vix_lt_20"].append(t["pnl"])
        elif v <= 30:
            regimes["vix_20_30"].append(t["pnl"])
        else:
            regimes["vix_gt_30"].append(t["pnl"])

    result = {}
    for label, pnls in regimes.items():
        if len(pnls) < 5:
            result[label] = {"n": len(pnls), "sharpe": 0.0, "avg_pnl": 0.0}
            continue
        arr = np.array(pnls)
        avg = arr.mean()
        std = arr.std()
        sharpe = float(avg / (std + 1e-10) * np.sqrt(annualize_periods))
        result[label] = {
            "n": len(pnls),
            "sharpe": round(sharpe, 2),
            "avg_pnl": round(float(avg), 2),
            "total_pnl": round(float(arr.sum()), 2),
        }

    return result


def compute_equity_curve_mdd(trades, initial_capital):
    """Compute max drawdown from equity curve built from trade PnLs."""
    if not trades:
        return 0.0

    equity = initial_capital
    hwm = initial_capital
    max_dd = 0.0

    # Sort trades by entry date
    sorted_trades = sorted(trades, key=lambda t: t["entry_date"])
    for t in sorted_trades:
        equity += t["pnl"]
        hwm = max(hwm, equity)
        dd = (hwm - equity) / hwm if hwm > 0 else 0
        max_dd = max(max_dd, dd)

    return max_dd


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    t0 = datetime.now()
    fprint("=" * 80)
    fprint(f"PROTECTIVE OVERLAY v2 — Tail Risk Hedging on C_otm2_riskparity")
    fprint(f"Run: {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 80)
    fprint(f"Capital: ${CAP:.0f} | DTE: {DTE} | Spread: {SPREAD_PCT:.0f}% | "
           f"Haircut: {DEFAULT_HAIRCUT:.0%} entry only | Comm: ${COMMISSION_RT_SPREAD:.2f}")
    fprint(f"HOLD TO EXPIRY | Intrinsic value only | No exit haircut")
    fprint(f"Moneyness: {MONEYNESS_PCT}% OTM (all variants)")
    fprint(f"Question: Can we reduce MDD from -20.4% while keeping Sharpe ~2.04?")
    fprint()

    # 1. Download data
    close, high, low = download_data()

    # 2. ATR
    atr_dict = compute_atr_series(high, low, close)

    # 3. Rebalance dates (biweekly for all variants)
    rebal_dates_bw = pd.DatetimeIndex(
        close.index.to_series().resample(WF_REBAL_FREQ_BIWEEKLY).last().dropna().values
    )
    fprint(f"Biweekly rebalance dates: {len(rebal_dates_bw)} "
           f"({rebal_dates_bw[0].date()} to {rebal_dates_bw[-1].date()})")

    spy_close = close["SPY"]

    # 4. Build LGBM rankings
    fprint("\n" + "=" * 80)
    fprint("BUILDING LGBM RANKINGS (biweekly)")
    fprint("=" * 80)

    records_bw = build_feature_records(close, high, low, rebal_dates_bw, ALL_FEATURES)
    rankings_bw, imp_df = walk_forward_lgbm_rank(records_bw, ALL_FEATURES)

    if imp_df is not None:
        fprint("\nTop 5 features by importance:")
        for _, row in imp_df.head(5).iterrows():
            fprint(f"    {row['feature']}: {row['importance']:.1f}")

    if not rankings_bw:
        fprint("ERROR: No rankings generated. Exiting.")
        return

    # 5. Define and run all variants
    fprint("\n" + "=" * 80)
    fprint("SIMULATING ALL 6 VARIANTS (A-F)")
    fprint("=" * 80)

    variant_configs = [
        ("A_no_hedge", "CONTROL: Exact C_otm2_riskparity reproduction", simulate_variant_A),
        ("B_stop_loss", "DD>15%: liquidate, park until 90% HWM", simulate_variant_B),
        ("C_vix_call", "VIX>25 + eq>$2K: buy VIX call spread (10%)", simulate_variant_C),
        ("D_put_hedge", "Continuous SPY put spread (1% equity, 5%OTM)", simulate_variant_D),
        ("E_reduce_exp", "VIX>30: 50% size, VIX>40: 25% size", simulate_variant_E),
        ("F_dd_halt", "DD>10%: stop new positions, resume 95% HWM", simulate_variant_F),
    ]

    all_results = {}

    for name, desc, sim_func in variant_configs:
        fprint(f"\n--- {name}: {desc} ---")

        trades, final_eq, pct_invested = sim_func(rankings_bw, close, atr_dict)

        fprint(f"  Trades: {len(trades)}, Final equity: ${final_eq:,.0f}, "
               f"% time invested: {pct_invested:.1f}%")

        if not trades or len(trades) < 10:
            fprint(f"  Only {len(trades) if trades else 0} trades, skipping validation")
            all_results[name] = {
                "description": desc,
                "n_trades": len(trades) if trades else 0,
                "final_equity": round(final_eq, 2),
                "pct_invested": round(pct_invested, 1),
                "error": "insufficient_trades",
            }
            continue

        # 5-gate adversarial validation + random baseline
        result = validate_trades(
            trades, initial_capital=CAP,
            spy_prices=spy_close,
            strategy_name=name,
        )
        result.print_summary()

        # SPY beta and correlation
        beta, spy_corr = compute_spy_beta(trades, close)
        monthly_spy_corr = compute_monthly_spy_correlation(trades, close)
        fprint(f"  SPY beta: {beta:.3f}, trade correlation: {spy_corr:.3f}, "
               f"monthly correlation: {monthly_spy_corr:.3f}")

        # VIX regime stratified Sharpe
        regime_sharpe = vix_regime_stratified_sharpe(trades, annualize_periods=26)
        for label, data in regime_sharpe.items():
            fprint(f"    {label}: {data['n']} trades, Sharpe {data['sharpe']:.2f}, "
                   f"avg PnL ${data['avg_pnl']:.2f}")

        # Leg breakdown
        sector_trades = [t for t in trades if t.get("mode") not in ("risk_parity", "hedge", "stop_loss_park", "dd_halt_park")]
        hedge_trades = [t for t in trades if t.get("mode") == "hedge"]
        rp_trades = [t for t in trades if t.get("mode") in ("risk_parity", "stop_loss_park", "dd_halt_park")]

        sector_pnl = sum(t["pnl"] for t in sector_trades)
        hedge_pnl = sum(t["pnl"] for t in hedge_trades)
        rp_pnl = sum(t["pnl"] for t in rp_trades)

        fprint(f"  Sector trades: {len(sector_trades)}, PnL ${sector_pnl:,.0f}")
        if hedge_trades:
            hedge_wr = sum(1 for t in hedge_trades if t["win"]) / len(hedge_trades) * 100
            fprint(f"  Hedge trades: {len(hedge_trades)}, PnL ${hedge_pnl:,.0f}, WR {hedge_wr:.1f}%")
        if rp_trades:
            fprint(f"  Risk parity: {len(rp_trades)}, PnL ${rp_pnl:,.0f}")

        # Compute equity curve MDD (more accurate than validator MDD for portfolios)
        eq_mdd = compute_equity_curve_mdd(trades, CAP)

        total_return = (final_eq / CAP - 1) * 100

        rd = result.to_dict()
        rd.update({
            "description": desc,
            "pct_invested": round(pct_invested, 1),
            "spy_beta": round(beta, 3),
            "spy_corr": round(spy_corr, 3),
            "monthly_spy_corr": round(monthly_spy_corr, 3),
            "total_return_pct": round(total_return, 1),
            "regime_stratified_sharpe": regime_sharpe,
            "sector_trades": len(sector_trades),
            "hedge_trades": len(hedge_trades),
            "rp_trades": len(rp_trades),
            "sector_pnl": round(sector_pnl, 2),
            "hedge_pnl": round(hedge_pnl, 2),
            "rp_pnl": round(rp_pnl, 2),
            "equity_curve_mdd": round(eq_mdd, 4),
        })

        all_results[name] = rd

    # ── Results summary table ──
    fprint("\n" + "=" * 80)
    fprint("RESULTS SUMMARY — SORTED BY SHARPE (HIGHER IS BETTER)")
    fprint("=" * 80)

    sortable = [(k, v) for k, v in all_results.items() if "error" not in v]
    sortable.sort(key=lambda x: x[1].get("sharpe", 0), reverse=True)

    fprint(f"\n{'Variant':<18} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} {'WR%':>6} "
           f"{'MDD%':>7} {'CAGR%':>7} {'#Trades':>8} {'%Inv':>6} {'Beta':>6} {'Gates':>6}")
    fprint("-" * 110)

    for name, rd in sortable:
        fprint(f"{name:<18} {rd['sharpe']:>7.2f} {rd['sortino']:>8.2f} "
               f"{rd['profit_factor']:>6.2f} {rd['win_rate']*100:>6.1f} "
               f"{rd['max_dd']*100:>7.1f} {rd['cagr']*100:>7.1f} "
               f"{rd['n_trades']:>8} {rd['pct_invested']:>5.1f}% "
               f"{rd['spy_beta']:>6.3f} {rd['gates_passed']}/{rd['gates_total']}")

    errored = [(k, v) for k, v in all_results.items() if "error" in v]
    for name, rd in errored:
        fprint(f"{name:<18} {'--':>7} {'--':>8} {'--':>6} {'--':>6} "
               f"{'--':>7} {'--':>7} {rd['n_trades']:>8} {rd['pct_invested']:>5.1f}% "
               f"{'--':>6} {'--':>6}")

    # ── Key analysis: MDD reduction vs Sharpe cost ──
    fprint("\n" + "=" * 80)
    fprint("KEY ANALYSIS: MDD REDUCTION vs SHARPE COST")
    fprint("=" * 80)

    if "A_no_hedge" in all_results and "error" not in all_results["A_no_hedge"]:
        control = all_results["A_no_hedge"]
        fprint(f"\nControl (A_no_hedge): Sharpe {control['sharpe']:.2f}, "
               f"MDD {control['max_dd']*100:.1f}%")

        fprint(f"\n{'Variant':<18} {'Sharpe':>7} {'MDD%':>7} {'Sharpe_chg':>11} {'MDD_chg':>9} {'Verdict':>20}")
        fprint("-" * 80)

        for name, rd in sortable:
            if name == "A_no_hedge":
                fprint(f"{name:<18} {rd['sharpe']:>7.2f} {rd['max_dd']*100:>7.1f} "
                       f"{'baseline':>11} {'baseline':>9} {'CONTROL':>20}")
                continue

            sharpe_chg = rd['sharpe'] - control['sharpe']
            mdd_chg = (rd['max_dd'] - control['max_dd']) * 100  # negative = improvement

            # Verdict logic
            if mdd_chg < -2 and sharpe_chg > -0.3:
                verdict = "WINNER"
            elif mdd_chg < -1 and sharpe_chg > -0.15:
                verdict = "PROMISING"
            elif mdd_chg < 0 and sharpe_chg > -0.1:
                verdict = "MARGINAL"
            elif sharpe_chg >= 0:
                verdict = "NO MDD HELP"
            else:
                verdict = "NOT WORTH IT"

            fprint(f"{name:<18} {rd['sharpe']:>7.2f} {rd['max_dd']*100:>7.1f} "
                   f"{sharpe_chg:>+11.2f} {mdd_chg:>+8.1f}% {verdict:>20}")

    # ── Best risk-adjusted: Sharpe / MDD ratio ──
    fprint("\n" + "=" * 80)
    fprint("SHARPE/MDD EFFICIENCY (higher = better risk-adjusted)")
    fprint("=" * 80)

    for name, rd in sortable:
        mdd_abs = abs(rd['max_dd']) + 1e-10
        efficiency = rd['sharpe'] / mdd_abs
        fprint(f"  {name:<18} Sharpe/|MDD| = {efficiency:.1f}")

    # ── Save JSON ──
    output = {
        "metadata": {
            "script": "protective_overlay_v2.py",
            "run_date": t0.strftime("%Y-%m-%d %H:%M:%S"),
            "capital": CAP,
            "dte": DTE,
            "spread_pct": SPREAD_PCT,
            "moneyness_pct": MONEYNESS_PCT,
            "haircut": DEFAULT_HAIRCUT,
            "commission": COMMISSION_RT_SPREAD,
            "wf_train_periods": WF_TRAIN_PERIODS,
            "rebal_freq": WF_REBAL_FREQ_BIWEEKLY,
            "data_range": f"{close.index[0].date()} to {close.index[-1].date()}",
            "n_sectors": len(SECTORS),
            "n_features": len(ALL_FEATURES),
            "question": "Can we reduce MDD from -20.4% while keeping Sharpe ~2.04?",
            "base_strategy": "C_otm2_riskparity (finding #152, Sharpe 2.04, MDD -20.4%)",
        },
        "results": all_results,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    fprint(f"\nResults saved to {RESULTS_PATH}")

    # ── MLflow logging ──
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"protective_overlay_v2_{t0.strftime('%Y%m%d_%H%M')}"):
                mlflow.log_params({
                    "capital": CAP,
                    "dte": DTE,
                    "spread_pct": SPREAD_PCT,
                    "moneyness_pct": MONEYNESS_PCT,
                    "haircut": DEFAULT_HAIRCUT,
                    "commission": COMMISSION_RT_SPREAD,
                    "wf_train_periods": WF_TRAIN_PERIODS,
                    "n_variants": len(variant_configs),
                    "base_strategy": "C_otm2_riskparity",
                })

                for name, rd in all_results.items():
                    prefix = name.replace(" ", "_")
                    if "error" not in rd:
                        mlflow.log_metrics({
                            f"{prefix}_sharpe": rd.get("sharpe", 0),
                            f"{prefix}_sortino": rd.get("sortino", 0),
                            f"{prefix}_cagr": rd.get("cagr", 0),
                            f"{prefix}_maxdd": rd.get("max_dd", 0),
                            f"{prefix}_wr": rd.get("win_rate", 0),
                            f"{prefix}_pf": rd.get("profit_factor", 0),
                            f"{prefix}_pct_invested": rd.get("pct_invested", 0),
                            f"{prefix}_spy_beta": rd.get("spy_beta", 0),
                            f"{prefix}_gates": rd.get("gates_passed", 0),
                            f"{prefix}_eq_mdd": rd.get("equity_curve_mdd", 0),
                        })

                        # Log hedge-specific metrics
                        if rd.get("hedge_trades", 0) > 0:
                            mlflow.log_metrics({
                                f"{prefix}_hedge_trades": rd["hedge_trades"],
                                f"{prefix}_hedge_pnl": rd["hedge_pnl"],
                            })

                mlflow.log_artifact(str(RESULTS_PATH))
            fprint("MLflow run logged successfully")
        except Exception as e:
            fprint(f"MLflow logging failed: {e}")

    elapsed = (datetime.now() - t0).total_seconds()
    fprint(f"\nCompleted in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
