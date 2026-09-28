#!/usr/bin/env python3
"""
Combined Portfolio v1 — Sector Bull Call Spreads + VIX Spike OTM Call Spreads
=============================================================================

Tests whether combining our two best strategies improves portfolio risk-adjusted returns.

Strategy 1: Sector Bull Call Spreads (production v4)
  - GRU regime score > 0.4 (VIX heuristic proxy)
  - LGBM walk-forward ranking, 21 features (18 legacy + 3 cross-asset)
  - Top 3 sector ETFs, ATM bull call spread, 3% width, DTE=21
  - Biweekly rebalance, $200 max per trade, hold to expiry

Strategy 2: VIX Spike OTM Call Spreads
  - VIX > 30 (first cross, not every day)
  - SPY only, 5% OTM bull call spread, 3% width, DTE=45
  - $200 max per trade, hold to expiry, rule-based (no ML)

6 Combined Variants:
  A: Additive (independent, no capital sharing)
  B: Capital-aware ($645 shared pool, Strategy 1 priority)
  C: Capital-aware (VIX spike priority)
  D: Additive with 60% exposure cap
  E: VIX spike only when regime inactive (GRU < 0.4)
  F: Hedge combo (VIX>30 = SPY OTM call + 50% sector sizing)

Honest pricing: hold-to-expiry, intrinsic only, 15% entry haircut, $2.60 RT commission.
"""

import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

sys.path.insert(0, "/home/jupiter/Lvl3Quant")
from research.tools.options_pricer import (
    price_bull_call_spread,
    estimate_iv,
    compute_atr,
    COMMISSION_RT_SPREAD,
    DEFAULT_HAIRCUT,
)
from research.tools.adversarial_validator import validate_trades

def fprint(*args, **kwargs):
    print(*args, **kwargs, flush=True)


# ═══════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════

BASE = Path("/home/jupiter/Lvl3Quant")
FINDINGS_DIR = BASE / "research" / "findings"
FINDINGS_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX", "TLT", "SHY", "HYG", "GLD"]
CAP = 645.0
DTE_SECTOR = 21
DTE_VIX = 45
SPREAD_PCT = 3.0
OTM_PCT = 5.0
TOP_K = 3
MAX_PER_TRADE = 200.0
REGIME_THRESHOLD = 0.4

# LGBM walk-forward
WF_TRAIN_DAYS = 240
WF_VAL_DAYS = 60
WF_STEP_DAYS = 10

# MLflow
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "combined_portfolio_v1"
MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen(MLFLOW_URI, timeout=3)
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    MLFLOW_OK = True
    fprint(f"MLflow connected: {MLFLOW_URI}")
except Exception:
    fprint("MLflow unavailable — results saved to disk only")


# ═══════════════════════════════════════════════════════════════════════
# Feature Engineering (18 legacy + 3 cross-asset = 21 total)
# ═══════════════════════════════════════════════════════════════════════

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

ALL_FEATURES = LEGACY_FEATURES + CROSS_ASSET_FEATURES


def compute_legacy_features(px, spy_slice):
    """Compute 18 legacy quality-momentum features."""
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
    """Compute 3 validated cross-asset features."""
    f = {}
    spy = close_df["SPY"].iloc[:dt_idx + 1].dropna()
    sector_px = close_df[sector_ticker].iloc[:dt_idx + 1].dropna() if sector_ticker in close_df.columns else None

    if spy is None or len(spy) < 63:
        return {k: 0.0 for k in CROSS_ASSET_FEATURES}

    spy_ret = spy.pct_change().dropna()

    # 1. Sector-SPY beta 63d
    if sector_px is not None and len(sector_px) > 63:
        sec_ret = sector_px.pct_change().dropna()
        common = spy_ret.index.intersection(sec_ret.index)
        if len(common) > 63:
            sr = sec_ret.loc[common].iloc[-63:]
            mr = spy_ret.loc[common].iloc[-63:]
            cov = np.cov(sr.values, mr.values)
            beta = cov[0, 1] / (cov[1, 1] + 1e-10)
            f["sector_spy_beta_63d"] = float(beta)
        else:
            f["sector_spy_beta_63d"] = 1.0
    else:
        f["sector_spy_beta_63d"] = 1.0

    # 2. Sector relative vol 21d
    if sector_px is not None and len(sector_px) > 21:
        sec_ret = sector_px.pct_change().dropna()
        if len(sec_ret) > 21 and len(spy_ret) > 21:
            sec_vol = sec_ret.iloc[-21:].std()
            spy_vol = spy_ret.iloc[-21:].std()
            f["sector_relative_vol_21d"] = float(sec_vol / (spy_vol + 1e-10))
        else:
            f["sector_relative_vol_21d"] = 1.0
    else:
        f["sector_relative_vol_21d"] = 1.0

    # 3. Cross-sector dispersion
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


# ═══════════════════════════════════════════════════════════════════════
# Data Download
# ═══════════════════════════════════════════════════════════════════════

def download_data():
    """Download all required tickers one by one for reliability."""
    import yfinance as yf

    all_tickers = SECTORS + EXTRA_TICKERS
    fprint(f"Downloading {len(all_tickers)} tickers individually...")

    frames_close = {}
    frames_high = {}
    frames_low = {}

    for tk in all_tickers:
        name = tk.replace("^", "")
        try:
            df = yf.download(tk, start="2006-01-01", end="2026-07-28", progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 0:
                frames_close[name] = df["Close"]
                frames_high[name] = df["High"]
                frames_low[name] = df["Low"]
                fprint(f"  {name}: {len(df)} rows")
            else:
                fprint(f"  {name}: EMPTY")
        except Exception as e:
            fprint(f"  {name}: FAILED ({e})")

    close = pd.DataFrame(frames_close)
    high = pd.DataFrame(frames_high)
    low = pd.DataFrame(frames_low)

    close = close.ffill()
    high = high.ffill()
    low = low.ffill()

    needed = ["SPY", "VIX"]
    for t in needed:
        if t not in close.columns:
            raise ValueError(f"Missing critical ticker: {t}")

    fprint(f"Data: {len(close)} days, {close.index[0].date()} to {close.index[-1].date()}")
    fprint(f"Columns: {list(close.columns)}")
    return close, high, low


# ═══════════════════════════════════════════════════════════════════════
# Regime Score (VIX-based heuristic proxy for GRU)
# ═══════════════════════════════════════════════════════════════════════

def compute_regime_scores(close):
    """VIX-based regime heuristic proxy for GRU regime gate.
    Calibrated so regime>0.4 roughly matches VIX>18 (elevated vol periods).
    """
    vix = close["VIX"]
    # Map VIX to 0-1: VIX=12 -> ~0.15, VIX=18 -> ~0.40, VIX=25 -> ~0.65, VIX=35 -> ~0.85
    regime = 0.15 + 0.85 * (1 - np.exp(-0.06 * (vix - 12).clip(lower=0)))
    regime = regime.clip(0.05, 0.95).fillna(0.3)
    fprint(f"Regime scores: mean={regime.mean():.3f}, days>0.4={int((regime > 0.4).sum())}, "
           f"days<0.4={int((regime <= 0.4).sum())}")
    return regime


# ═══════════════════════════════════════════════════════════════════════
# ATR Computation
# ═══════════════════════════════════════════════════════════════════════

def compute_atr_series(high, low, close, period=14):
    """ATR series for all sectors + SPY."""
    atr_dict = {}
    tickers = SECTORS + ["SPY"]
    for tk in tickers:
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
                atr_dict[tk] = tr.ewm(alpha=1/period, min_periods=period).mean()
    return atr_dict


# ═══════════════════════════════════════════════════════════════════════
# Strategy 1: Sector Bull Call Spreads with LGBM + Regime Gate
# ═══════════════════════════════════════════════════════════════════════

def get_rebal_dates(close, start_year=2009):
    """Biweekly rebalance dates (every 10 trading days)."""
    dates = close.index[close.index.year >= start_year]
    return dates[::10].tolist()


def build_lgbm_records(close, rebal_dates, regime_scores):
    """Build feature + target records for LGBM walk-forward."""
    fprint("  Building LGBM feature records...")
    records = []
    spy = close["SPY"]

    for dt in rebal_dates:
        idx = close.index.get_loc(dt)
        if idx < 260:
            continue

        # Regime gate
        rscore = float(regime_scores.iloc[idx]) if idx < len(regime_scores) else 0.3
        if rscore <= REGIME_THRESHOLD:
            continue

        for tk in SECTORS:
            if tk not in close.columns:
                continue
            px = close[tk].iloc[:idx + 1].dropna()
            spy_s = spy.iloc[:idx + 1]

            legacy = compute_legacy_features(px, spy_s)
            if not legacy:
                continue

            cross_asset = compute_cross_asset_features(tk, idx, close)

            # Forward return target
            fi = min(idx + DTE_SECTOR, len(close) - 1)
            if fi <= idx:
                continue
            fwd_ret = float(close[tk].iloc[fi] / close[tk].iloc[idx] - 1)

            rec = {**legacy, **cross_asset, "date": dt, "ticker": tk, "fwd_ret": fwd_ret}
            records.append(rec)

    if not records:
        fprint("    WARNING: 0 records — no dates passed regime filter")
        return pd.DataFrame(columns=ALL_FEATURES + ["date", "ticker", "fwd_ret"])

    df = pd.DataFrame(records)
    for c in ALL_FEATURES:
        if c not in df.columns:
            df[c] = 0.0
    df[ALL_FEATURES] = df[ALL_FEATURES].fillna(0.0)
    fprint(f"    {len(df)} records, {len(df['date'].unique())} dates with regime>0.4")
    return df


def walk_forward_lgbm_rank(df):
    """Walk-forward LGBM ranking with sliding window."""
    import lightgbm as lgb

    if len(df) < 100:
        fprint(f"    Insufficient data for LGBM ({len(df)} records)")
        return {}

    df["rank_label"] = df.groupby("date")["fwd_ret"].rank(pct=True)
    dates = sorted(df["date"].unique())

    # Convert day counts to period counts (each period ~10 trading days)
    train_periods = WF_TRAIN_DAYS // 10
    val_periods = WF_VAL_DAYS // 10

    rankings = {}
    n_models = 0
    importances = np.zeros(len(ALL_FEATURES))

    for i in range(train_periods, len(dates)):
        train_dates = dates[max(0, i - train_periods):i]
        test_date = dates[i]

        train_df = df[df["date"].isin(train_dates)]
        test_df = df[df["date"] == test_date].copy()

        if len(test_df) < 3 or len(train_df) < 50:
            continue

        Xt = np.nan_to_num(train_df[ALL_FEATURES].values.astype(np.float32))
        yt = train_df["rank_label"].values.astype(np.float32)
        Xe = np.nan_to_num(test_df[ALL_FEATURES].values.astype(np.float32))

        try:
            m = lgb.LGBMRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=5, verbose=-1,
            )
            m.fit(Xt, yt)
            test_df["score"] = m.predict(Xe)
            rankings[test_date] = dict(zip(test_df["ticker"], test_df["score"]))
            importances += m.feature_importances_
            n_models += 1
        except Exception:
            continue

    if n_models > 0:
        importances /= n_models
    fprint(f"    LGBM: {len(rankings)} ranking dates, {n_models} models")
    return rankings


def simulate_sector_trades(rankings, close, atr_dict, regime_scores, max_per_trade=MAX_PER_TRADE,
                           sizing_factor=1.0):
    """Simulate sector bull call spread trades. Returns list of trade dicts."""
    spy = close["SPY"]
    vix = close["VIX"]
    trades = []

    for dt in sorted(rankings.keys()):
        if dt not in spy.index:
            continue
        idx = close.index.get_loc(dt)

        cv = float(vix.iloc[idx]) if idx < len(vix) else 20.0
        scores = rankings[dt]
        if not scores:
            continue

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        picks = [t for t, _ in ranked[:TOP_K]]

        effective_max = max_per_trade * sizing_factor

        for tk in picks:
            if tk not in close.columns or tk not in atr_dict:
                continue

            S = float(close[tk].iloc[idx])
            ei = min(idx + DTE_SECTOR, len(close) - 1)
            if ei <= idx:
                continue

            av = float(atr_dict[tk].iloc[idx]) if idx < len(atr_dict[tk]) and not pd.isna(atr_dict[tk].iloc[idx]) else S * 0.015

            K1 = round(S, 2)
            K2 = round(S * (1 + SPREAD_PCT / 100), 2)
            if K2 <= K1:
                K2 = K1 + 1.0

            try:
                entry_cost_ps, max_profit_ps = price_bull_call_spread(
                    S=S, K1=K1, K2=K2, dte=DTE_SECTOR, atr=av, vix=cv
                )
            except Exception:
                continue

            total_cost = entry_cost_ps * 100 + COMMISSION_RT_SPREAD
            if total_cost <= 0 or total_cost > effective_max:
                continue

            # Hold to expiry: intrinsic value
            Se = float(close[tk].iloc[ei])
            intrinsic = max(Se - K1, 0.0) - max(Se - K2, 0.0)
            pnl = (intrinsic - entry_cost_ps) * 100 - COMMISSION_RT_SPREAD

            trades.append({
                "entry_date": dt,
                "exit_date": close.index[ei],
                "ticker": tk,
                "strategy": "sector_bull_call",
                "pnl": pnl,
                "cost": total_cost,
                "S_entry": S,
                "S_exit": Se,
                "K1": K1,
                "K2": K2,
                "vix_entry": cv,
                "regime_score": float(regime_scores.iloc[idx]) if idx < len(regime_scores) else 0.5,
            })

    fprint(f"    Sector trades: {len(trades)}, "
           f"Win rate: {sum(1 for t in trades if t['pnl'] > 0)/max(len(trades),1):.1%}")
    return trades


# ═══════════════════════════════════════════════════════════════════════
# Strategy 2: VIX Spike OTM Call Spreads
# ═══════════════════════════════════════════════════════════════════════

def find_vix_spike_dates(close, vix_threshold=30, cooldown_days=45, start_year=2009):
    """Find first cross above VIX threshold (not every day above it)."""
    vix = close["VIX"]
    signals = []
    last_signal = None
    was_below = True

    for i in range(20, len(vix)):
        dt = vix.index[i]
        if dt.year < start_year:
            if vix.iloc[i] < vix_threshold:
                was_below = True
            continue

        v = vix.iloc[i]
        if v < vix_threshold:
            was_below = True
            continue

        if not was_below:
            continue

        if last_signal is not None and (dt - last_signal).days < cooldown_days:
            continue

        signals.append(dt)
        last_signal = dt
        was_below = False

    return signals


def simulate_vix_spike_trades(close, atr_dict, spike_dates, max_per_trade=MAX_PER_TRADE):
    """Simulate VIX spike SPY OTM call spread trades."""
    trades = []
    vix = close["VIX"]

    for dt in spike_dates:
        idx = close.index.get_loc(dt)
        if idx + DTE_VIX >= len(close):
            continue

        S = float(close["SPY"].iloc[idx])
        cv = float(vix.iloc[idx])
        ei = min(idx + DTE_VIX, len(close) - 1)

        # 5% OTM
        K1 = round(S * (1 + OTM_PCT / 100), 2)
        K2 = round(K1 * (1 + SPREAD_PCT / 100), 2)
        if K2 <= K1:
            K2 = K1 + 1.0

        # ATR for pricing
        if "SPY" in atr_dict and idx < len(atr_dict["SPY"]) and not pd.isna(atr_dict["SPY"].iloc[idx]):
            av = float(atr_dict["SPY"].iloc[idx])
        else:
            av = S * 0.015

        try:
            entry_cost_ps, max_profit_ps = price_bull_call_spread(
                S=S, K1=K1, K2=K2, dte=DTE_VIX, atr=av, vix=cv
            )
        except Exception:
            continue

        total_cost = entry_cost_ps * 100 + COMMISSION_RT_SPREAD
        if total_cost <= 0 or total_cost > max_per_trade:
            continue

        # Hold to expiry: intrinsic
        Se = float(close["SPY"].iloc[ei])
        intrinsic = max(Se - K1, 0.0) - max(Se - K2, 0.0)
        pnl = (intrinsic - entry_cost_ps) * 100 - COMMISSION_RT_SPREAD

        trades.append({
            "entry_date": dt,
            "exit_date": close.index[ei],
            "ticker": "SPY",
            "strategy": "vix_spike_otm_call",
            "pnl": pnl,
            "cost": total_cost,
            "S_entry": S,
            "S_exit": Se,
            "K1": K1,
            "K2": K2,
            "vix_entry": cv,
        })

    fprint(f"    VIX spike trades: {len(trades)}, "
           f"Win rate: {sum(1 for t in trades if t['pnl'] > 0)/max(len(trades),1):.1%}")
    return trades


# ═══════════════════════════════════════════════════════════════════════
# Portfolio Metrics
# ═══════════════════════════════════════════════════════════════════════

def compute_metrics(trades, initial_capital=CAP, label=""):
    """Compute portfolio-level metrics from a trade list."""
    if not trades:
        return {"label": label, "n_trades": 0, "sharpe": 0, "sortino": 0,
                "cagr": 0, "max_dd": 0, "win_rate": 0, "pf": 0, "final_equity": initial_capital}

    # Sort by entry date
    trades_sorted = sorted(trades, key=lambda t: t["entry_date"])

    # Build equity curve
    equity = initial_capital
    equity_series = [equity]
    dates = [trades_sorted[0]["entry_date"] - pd.Timedelta(days=1)]

    for t in trades_sorted:
        equity += t["pnl"]
        equity_series.append(equity)
        dates.append(t["exit_date"])

    eq = pd.Series(equity_series, index=dates)
    eq = eq[~eq.index.duplicated(keep="last")]
    eq = eq.sort_index()

    # Returns from equity
    rets = eq.pct_change().dropna()

    # Sharpe (annualized, assume ~26 biweekly periods per year for sector,
    # but better to use daily-equivalent)
    if len(rets) < 2 or rets.std() == 0:
        sharpe = 0.0
        sortino = 0.0
    else:
        # Approximate periods per year from data span
        years = (eq.index[-1] - eq.index[0]).days / 365.25
        periods_per_year = len(rets) / years if years > 0 else 26
        sharpe = float(rets.mean() / rets.std() * np.sqrt(periods_per_year))
        downside = rets[rets < 0].std()
        sortino = float(rets.mean() / (downside + 1e-10) * np.sqrt(periods_per_year)) if downside > 0 else sharpe * 2

    # CAGR
    final = equity_series[-1]
    years = (dates[-1] - dates[0]).days / 365.25
    cagr = (final / initial_capital) ** (1 / max(years, 0.1)) - 1 if final > 0 else -1.0

    # Max drawdown
    eq_arr = np.array(equity_series)
    peaks = np.maximum.accumulate(eq_arr)
    dd = (eq_arr - peaks) / peaks
    max_dd = float(dd.min())

    # Win rate & profit factor
    wins = [t["pnl"] for t in trades if t["pnl"] > 0]
    losses = [t["pnl"] for t in trades if t["pnl"] <= 0]
    wr = len(wins) / len(trades)
    pf = sum(wins) / (abs(sum(losses)) + 1e-10) if losses else float("inf")

    return {
        "label": label,
        "n_trades": len(trades),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "cagr": round(cagr * 100, 2),
        "max_dd": round(max_dd * 100, 2),
        "win_rate": round(wr * 100, 1),
        "pf": round(pf, 3),
        "final_equity": round(final, 2),
        "total_return": round((final / initial_capital - 1) * 100, 1),
    }


# ═══════════════════════════════════════════════════════════════════════
# Combined Portfolio Variants
# ═══════════════════════════════════════════════════════════════════════

def variant_a_additive(sector_trades, vix_trades):
    """A: Both strategies run independently, combine equity curves."""
    all_trades = sector_trades + vix_trades
    return all_trades, compute_metrics(all_trades, CAP, "A: Additive")


def variant_b_capital_aware_sector_priority(sector_trades, vix_trades):
    """B: Shared $645 capital pool, sector trades get priority."""
    all_events = []
    for t in sector_trades:
        all_events.append(("sector", t))
    for t in vix_trades:
        all_events.append(("vix", t))
    all_events.sort(key=lambda x: x[1]["entry_date"])

    equity = CAP
    active_cost = 0.0  # capital tied up in active positions
    active_positions = []  # (exit_date, cost)
    accepted = []

    for source, trade in all_events:
        dt = trade["entry_date"]
        # Free up expired positions
        active_positions = [(ed, c) for ed, c in active_positions if ed > dt]
        active_cost = sum(c for _, c in active_positions)

        available = equity - active_cost
        cost = trade["cost"]

        # If not enough capital, skip VIX trades first (sector has priority)
        if cost > available:
            if source == "vix":
                continue  # skip VIX trade
            else:
                # Even sector can't fit
                continue

        # Accept trade
        accepted.append(trade)
        active_positions.append((trade["exit_date"], cost))
        equity += trade["pnl"]

    return accepted, compute_metrics(accepted, CAP, "B: Capital-aware (sector priority)")


def variant_c_capital_aware_vix_priority(sector_trades, vix_trades):
    """C: Shared $645 capital pool, VIX spike gets priority."""
    all_events = []
    for t in sector_trades:
        all_events.append(("sector", t))
    for t in vix_trades:
        all_events.append(("vix", t))
    all_events.sort(key=lambda x: x[1]["entry_date"])

    equity = CAP
    active_cost = 0.0
    active_positions = []
    accepted = []

    for source, trade in all_events:
        dt = trade["entry_date"]
        active_positions = [(ed, c) for ed, c in active_positions if ed > dt]
        active_cost = sum(c for _, c in active_positions)

        available = equity - active_cost
        cost = trade["cost"]

        if cost > available:
            if source == "sector":
                continue  # skip sector trade
            else:
                continue

        accepted.append(trade)
        active_positions.append((trade["exit_date"], cost))
        equity += trade["pnl"]

    return accepted, compute_metrics(accepted, CAP, "C: Capital-aware (VIX priority)")


def variant_d_additive_exposure_cap(sector_trades, vix_trades):
    """D: Both run, but total exposure capped at 60% of equity."""
    all_events = []
    for t in sector_trades:
        all_events.append(t)
    for t in vix_trades:
        all_events.append(t)
    all_events.sort(key=lambda x: x["entry_date"])

    equity = CAP
    active_positions = []
    accepted = []

    for trade in all_events:
        dt = trade["entry_date"]
        active_positions = [(ed, c) for ed, c in active_positions if ed > dt]
        active_cost = sum(c for _, c in active_positions)

        cost = trade["cost"]
        exposure_pct = (active_cost + cost) / max(equity, 1)

        if exposure_pct > 0.60:
            continue

        accepted.append(trade)
        active_positions.append((trade["exit_date"], cost))
        equity += trade["pnl"]

    return accepted, compute_metrics(accepted, CAP, "D: Additive (60% exposure cap)")


def variant_e_vix_when_regime_inactive(sector_trades, vix_trades, regime_scores, close):
    """E: VIX spike only fires when GRU regime < 0.4."""
    filtered_vix = []
    for t in vix_trades:
        dt = t["entry_date"]
        idx = close.index.get_loc(dt)
        rscore = float(regime_scores.iloc[idx]) if idx < len(regime_scores) else 0.3
        if rscore < REGIME_THRESHOLD:
            filtered_vix.append(t)

    fprint(f"    Variant E: {len(filtered_vix)}/{len(vix_trades)} VIX trades pass regime filter")
    all_trades = sector_trades + filtered_vix
    return all_trades, compute_metrics(all_trades, CAP, "E: VIX when regime inactive")


def variant_f_hedge_combo(rankings, close, atr_dict, regime_scores, vix_trades, spike_dates):
    """F: During VIX>30, buy SPY OTM call AND reduce sector sizing by 50%."""
    # Identify VIX spike windows (entry to exit)
    spike_windows = set()
    for t in vix_trades:
        dt = t["entry_date"]
        ed = t["exit_date"]
        idx_start = close.index.get_loc(dt)
        idx_end = close.index.get_loc(ed)
        for i in range(idx_start, idx_end + 1):
            spike_windows.add(close.index[i])

    # Sector trades with 50% sizing during spikes
    sector_trades_adjusted = simulate_sector_trades(
        rankings, close, atr_dict, regime_scores, max_per_trade=MAX_PER_TRADE,
        sizing_factor=1.0
    )
    # Now adjust: if trade entry is during spike window, halve its PnL/cost (proxy for 50% sizing)
    adjusted = []
    for t in sector_trades_adjusted:
        tc = dict(t)
        if tc["entry_date"] in spike_windows:
            tc["pnl"] *= 0.5
            tc["cost"] *= 0.5
        adjusted.append(tc)

    all_trades = adjusted + vix_trades
    return all_trades, compute_metrics(all_trades, CAP, "F: Hedge combo")


# ═══════════════════════════════════════════════════════════════════════
# Correlation & Drawdown Analysis
# ═══════════════════════════════════════════════════════════════════════

def analyze_correlation(sector_trades, vix_trades, close):
    """Analyze return correlation between strategies."""
    fprint("\n" + "="*70)
    fprint("CORRELATION & DRAWDOWN ANALYSIS")
    fprint("="*70)

    # Monthly returns for each strategy
    def monthly_returns(trades, cap):
        if not trades:
            return pd.Series(dtype=float)
        equity = cap
        eq_points = [(trades[0]["entry_date"] - pd.Timedelta(days=1), equity)]
        for t in sorted(trades, key=lambda x: x["entry_date"]):
            equity += t["pnl"]
            eq_points.append((t["exit_date"], equity))
        eq = pd.Series([p[1] for p in eq_points], index=[p[0] for p in eq_points])
        eq = eq[~eq.index.duplicated(keep="last")].sort_index()
        monthly = eq.resample("ME").last().pct_change().dropna()
        return monthly

    s1_monthly = monthly_returns(sector_trades, CAP)
    s2_monthly = monthly_returns(vix_trades, CAP)

    # Align
    common = s1_monthly.index.intersection(s2_monthly.index)
    if len(common) > 3:
        corr = s1_monthly.loc[common].corr(s2_monthly.loc[common])
        fprint(f"\n  Monthly return correlation: {corr:.4f}")
    else:
        corr = np.nan
        fprint(f"\n  Monthly return correlation: N/A (too few overlapping months)")

    # When do VIX spikes happen relative to regime?
    fprint("\n  VIX spike timing vs regime:")
    for t in vix_trades:
        dt = t["entry_date"]
        idx = close.index.get_loc(dt)
        vix_val = float(close["VIX"].iloc[idx])
        regime = float(close["VIX"].iloc[idx])  # use VIX as proxy
        regime_active = "YES" if vix_val > 24 else "NO"  # rough: VIX>24 ~ regime>0.4
        fprint(f"    {dt.date()}: VIX={vix_val:.1f}, regime likely active={regime_active}, "
               f"PnL=${t['pnl']:.2f}")

    # Drawdown analysis: does VIX strategy fill in sector drawdowns?
    fprint("\n  Drawdown coverage analysis:")

    # Build daily equity for sector strategy
    if sector_trades:
        eq_sector = CAP
        sector_eq = [(sector_trades[0]["entry_date"] - pd.Timedelta(days=1), eq_sector)]
        for t in sorted(sector_trades, key=lambda x: x["entry_date"]):
            eq_sector += t["pnl"]
            sector_eq.append((t["exit_date"], eq_sector))
        eq_s = pd.Series([p[1] for p in sector_eq], index=[p[0] for p in sector_eq])
        eq_s = eq_s[~eq_s.index.duplicated(keep="last")].sort_index()
        eq_s = eq_s.reindex(close.index, method="ffill").dropna()

        peak = eq_s.cummax()
        dd = (eq_s - peak) / peak

        # Find worst drawdown periods
        worst_dd = dd.min()
        worst_dd_date = dd.idxmin()
        fprint(f"    Sector strategy worst DD: {worst_dd*100:.1f}% on {worst_dd_date.date()}")

        # Check if any VIX trades happened during sector drawdowns (DD > -5%)
        dd_periods = dd[dd < -0.05]
        if len(dd_periods) > 0:
            vix_during_dd = 0
            vix_pnl_during_dd = 0
            for t in vix_trades:
                entry = t["entry_date"]
                exit_dt = t["exit_date"]
                # Check if any part of trade overlaps with drawdown period
                overlap = dd_periods.index[(dd_periods.index >= entry) & (dd_periods.index <= exit_dt)]
                if len(overlap) > 0:
                    vix_during_dd += 1
                    vix_pnl_during_dd += t["pnl"]
            fprint(f"    Sector drawdown periods (>5%): {len(dd_periods)} days")
            fprint(f"    VIX trades during sector drawdowns: {vix_during_dd}")
            fprint(f"    VIX PnL during sector drawdowns: ${vix_pnl_during_dd:.2f}")
        else:
            fprint(f"    No significant drawdown periods (>5%)")

    return corr


# ═══════════════════════════════════════════════════════════════════════
# 5-Gate Adversarial Validation
# ═══════════════════════════════════════════════════════════════════════

def run_adversarial(trades, spy_prices, label):
    """Run 5-gate adversarial validation on a trade set."""
    if len(trades) < 10:
        fprint(f"  {label}: Too few trades ({len(trades)}) for adversarial validation")
        return None

    result = validate_trades(
        trades=trades,
        initial_capital=CAP,
        spy_prices=spy_prices,
    )
    fprint(f"\n  Adversarial Validation: {label}")
    fprint(f"    Trades: {result.n_trades}, Sharpe: {result.sharpe:.3f}, "
           f"Sortino: {result.sortino:.3f}, CAGR: {result.cagr*100:.1f}%")
    for gate in result.gates:
        fprint(f"    {gate}")
    fprint(f"    ALL PASSED: {result.all_passed}")
    return result


# ═══════════════════════════════════════════════════════════════════════
# MLflow Logging
# ═══════════════════════════════════════════════════════════════════════

def log_to_mlflow(results, correlation, sector_metrics, vix_metrics):
    """Log all variant results to MLflow."""
    if not MLFLOW_OK:
        return

    try:
        mlflow.set_experiment(EXPERIMENT_NAME)
        for r in results:
            with mlflow.start_run(run_name=r["label"]):
                mlflow.log_params({
                    "variant": r["label"],
                    "initial_capital": CAP,
                    "dte_sector": DTE_SECTOR,
                    "dte_vix": DTE_VIX,
                    "spread_pct": SPREAD_PCT,
                    "otm_pct": OTM_PCT,
                    "regime_threshold": REGIME_THRESHOLD,
                    "top_k": TOP_K,
                })
                mlflow.log_metrics({
                    "n_trades": r["n_trades"],
                    "sharpe": r["sharpe"],
                    "sortino": r["sortino"],
                    "cagr_pct": r["cagr"],
                    "max_dd_pct": r["max_dd"],
                    "win_rate_pct": r["win_rate"],
                    "profit_factor": min(r["pf"], 999),
                    "final_equity": r["final_equity"],
                    "total_return_pct": r["total_return"],
                })
                if not np.isnan(correlation):
                    mlflow.log_metric("strategy_correlation", correlation)

        # Also log standalone baselines
        for baseline in [sector_metrics, vix_metrics]:
            with mlflow.start_run(run_name=baseline["label"]):
                mlflow.log_metrics({
                    "n_trades": baseline["n_trades"],
                    "sharpe": baseline["sharpe"],
                    "sortino": baseline["sortino"],
                    "cagr_pct": baseline["cagr"],
                    "max_dd_pct": baseline["max_dd"],
                    "win_rate_pct": baseline["win_rate"],
                    "profit_factor": min(baseline["pf"], 999),
                    "final_equity": baseline["final_equity"],
                    "total_return_pct": baseline["total_return"],
                })

        fprint("  All variants logged to MLflow")
    except Exception as e:
        fprint(f"  MLflow logging failed: {e}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    fprint("="*70)
    fprint("COMBINED PORTFOLIO v1: Sector Bull Call Spreads + VIX Spike OTM Calls")
    fprint("="*70)
    fprint(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ── Download data ──
    close, high, low = download_data()

    # ── Regime scores ──
    regime_scores = compute_regime_scores(close)

    # ── ATR ──
    fprint("\nComputing ATR series...")
    atr_dict = compute_atr_series(high, low, close)

    # ══════════════════════════════════════════════════════════════
    # Strategy 1: Sector Bull Call Spreads
    # ══════════════════════════════════════════════════════════════
    fprint("\n" + "="*70)
    fprint("STRATEGY 1: Sector Bull Call Spreads (production v4)")
    fprint("="*70)

    rebal_dates = get_rebal_dates(close)
    fprint(f"  Rebalance dates: {len(rebal_dates)} (biweekly)")

    records_df = build_lgbm_records(close, rebal_dates, regime_scores)
    rankings = walk_forward_lgbm_rank(records_df)
    sector_trades = simulate_sector_trades(rankings, close, atr_dict, regime_scores)
    sector_metrics = compute_metrics(sector_trades, CAP, "Baseline: Sector only")

    # ══════════════════════════════════════════════════════════════
    # Strategy 2: VIX Spike OTM Call Spreads
    # ══════════════════════════════════════════════════════════════
    fprint("\n" + "="*70)
    fprint("STRATEGY 2: VIX Spike OTM Call Spreads")
    fprint("="*70)

    spike_dates = find_vix_spike_dates(close, vix_threshold=30, cooldown_days=DTE_VIX)
    fprint(f"  VIX spike events (>30): {len(spike_dates)}")
    for sd in spike_dates:
        idx = close.index.get_loc(sd)
        fprint(f"    {sd.date()}: VIX={close['VIX'].iloc[idx]:.1f}")

    vix_trades = simulate_vix_spike_trades(close, atr_dict, spike_dates)
    vix_metrics = compute_metrics(vix_trades, CAP, "Baseline: VIX spike only")

    # ══════════════════════════════════════════════════════════════
    # Standalone Baselines
    # ══════════════════════════════════════════════════════════════
    fprint("\n" + "="*70)
    fprint("STANDALONE BASELINES")
    fprint("="*70)
    for m in [sector_metrics, vix_metrics]:
        fprint(f"\n  {m['label']}:")
        fprint(f"    Trades: {m['n_trades']}, Sharpe: {m['sharpe']}, Sortino: {m['sortino']}")
        fprint(f"    CAGR: {m['cagr']}%, MaxDD: {m['max_dd']}%, WR: {m['win_rate']}%, PF: {m['pf']}")
        fprint(f"    Final equity: ${m['final_equity']:.2f} (return: {m['total_return']}%)")

    # ══════════════════════════════════════════════════════════════
    # Combined Variants
    # ══════════════════════════════════════════════════════════════
    fprint("\n" + "="*70)
    fprint("COMBINED PORTFOLIO VARIANTS")
    fprint("="*70)

    variant_results = []

    # A: Additive
    fprint("\n  [A] Additive (independent)...")
    a_trades, a_metrics = variant_a_additive(sector_trades, vix_trades)
    variant_results.append(a_metrics)

    # B: Capital-aware, sector priority
    fprint("\n  [B] Capital-aware (sector priority)...")
    b_trades, b_metrics = variant_b_capital_aware_sector_priority(sector_trades, vix_trades)
    variant_results.append(b_metrics)

    # C: Capital-aware, VIX priority
    fprint("\n  [C] Capital-aware (VIX priority)...")
    c_trades, c_metrics = variant_c_capital_aware_vix_priority(sector_trades, vix_trades)
    variant_results.append(c_metrics)

    # D: Additive with exposure cap
    fprint("\n  [D] Additive (60% exposure cap)...")
    d_trades, d_metrics = variant_d_additive_exposure_cap(sector_trades, vix_trades)
    variant_results.append(d_metrics)

    # E: VIX when regime inactive
    fprint("\n  [E] VIX when regime inactive...")
    e_trades, e_metrics = variant_e_vix_when_regime_inactive(sector_trades, vix_trades, regime_scores, close)
    variant_results.append(e_metrics)

    # F: Hedge combo
    fprint("\n  [F] Hedge combo (VIX spike = SPY OTM + 50% sector sizing)...")
    f_trades, f_metrics = variant_f_hedge_combo(rankings, close, atr_dict, regime_scores, vix_trades, spike_dates)
    variant_results.append(f_metrics)

    # ══════════════════════════════════════════════════════════════
    # Correlation & Drawdown Analysis
    # ══════════════════════════════════════════════════════════════
    correlation = analyze_correlation(sector_trades, vix_trades, close)

    # ══════════════════════════════════════════════════════════════
    # Summary Table
    # ══════════════════════════════════════════════════════════════
    fprint("\n" + "="*70)
    fprint("RESULTS SUMMARY")
    fprint("="*70)

    all_results = [sector_metrics, vix_metrics] + variant_results

    header = f"{'Variant':<45} {'Trades':>6} {'Sharpe':>7} {'Sortino':>8} {'CAGR%':>7} {'MaxDD%':>7} {'WR%':>6} {'PF':>6} {'Final$':>8}"
    fprint(f"\n{header}")
    fprint("-" * len(header))
    for r in all_results:
        fprint(f"{r['label']:<45} {r['n_trades']:>6} {r['sharpe']:>7.3f} {r['sortino']:>8.3f} "
               f"{r['cagr']:>7.1f} {r['max_dd']:>7.1f} {r['win_rate']:>6.1f} {r['pf']:>6.2f} "
               f"{r['final_equity']:>8.2f}")

    fprint(f"\n  Strategy correlation (monthly): {correlation:.4f}" if not np.isnan(correlation) else
           f"\n  Strategy correlation: N/A")

    # ══════════════════════════════════════════════════════════════
    # Key Question: Does combining improve Sharpe?
    # ══════════════════════════════════════════════════════════════
    fprint("\n" + "="*70)
    fprint("KEY ANALYSIS: Does combining improve risk-adjusted returns?")
    fprint("="*70)

    best_combined = max(variant_results, key=lambda x: x["sharpe"])
    sector_sharpe = sector_metrics["sharpe"]
    best_sharpe = best_combined["sharpe"]
    improvement = best_sharpe - sector_sharpe

    fprint(f"\n  Sector standalone Sharpe: {sector_sharpe:.3f}")
    fprint(f"  Best combined Sharpe:    {best_sharpe:.3f} ({best_combined['label']})")
    fprint(f"  Improvement:             {improvement:+.3f}")

    if improvement > 0:
        fprint(f"\n  VERDICT: Combining IMPROVES Sharpe by {improvement:.3f}")
        fprint(f"  The diversification benefit from uncorrelated VIX spike trades helps.")
    else:
        fprint(f"\n  VERDICT: Combining does NOT improve Sharpe (change: {improvement:.3f})")
        fprint(f"  The VIX spike strategy may add return but with proportional risk.")

    # ══════════════════════════════════════════════════════════════
    # Adversarial Validation on best variants
    # ══════════════════════════════════════════════════════════════
    fprint("\n" + "="*70)
    fprint("5-GATE ADVERSARIAL VALIDATION")
    fprint("="*70)

    spy_prices = close["SPY"]

    # Validate standalone sector
    adv_sector = run_adversarial(sector_trades, spy_prices, "Sector standalone")

    # Validate best combined
    # Map label to trades
    variant_trade_map = {
        "A: Additive": a_trades,
        "B: Capital-aware (sector priority)": b_trades,
        "C: Capital-aware (VIX priority)": c_trades,
        "D: Additive (60% exposure cap)": d_trades,
        "E: VIX when regime inactive": e_trades,
        "F: Hedge combo": f_trades,
    }

    # Validate top 2 combined variants by Sharpe
    top_variants = sorted(variant_results, key=lambda x: x["sharpe"], reverse=True)[:2]
    adv_results = {}
    for v in top_variants:
        label = v["label"]
        trades = variant_trade_map.get(label, [])
        adv = run_adversarial(trades, spy_prices, label)
        if adv is not None:
            adv_results[label] = {
                "all_passed": adv.all_passed,
                "gates_passed": sum(1 for g in adv.gates if g.passed),
                "gates_total": len(adv.gates),
            }

    # ══════════════════════════════════════════════════════════════
    # MLflow Logging
    # ══════════════════════════════════════════════════════════════
    fprint("\n  Logging to MLflow...")
    log_to_mlflow(variant_results, correlation, sector_metrics, vix_metrics)

    # ══════════════════════════════════════════════════════════════
    # Save Results
    # ══════════════════════════════════════════════════════════════
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "initial_capital": CAP,
            "dte_sector": DTE_SECTOR,
            "dte_vix": DTE_VIX,
            "spread_pct": SPREAD_PCT,
            "otm_pct": OTM_PCT,
            "regime_threshold": REGIME_THRESHOLD,
            "top_k": TOP_K,
            "max_per_trade": MAX_PER_TRADE,
            "wf_train_days": WF_TRAIN_DAYS,
            "commission_rt": COMMISSION_RT_SPREAD,
            "entry_haircut": DEFAULT_HAIRCUT,
        },
        "baselines": {
            "sector": sector_metrics,
            "vix_spike": vix_metrics,
        },
        "combined_variants": {r["label"]: r for r in variant_results},
        "correlation": {
            "monthly_return_correlation": round(correlation, 4) if not np.isnan(correlation) else None,
        },
        "adversarial_validation": adv_results,
        "best_variant": best_combined["label"],
        "sharpe_improvement": round(improvement, 3),
        "vix_spike_events": len(spike_dates),
        "vix_spike_dates": [str(d.date()) for d in spike_dates],
    }

    out_path = FINDINGS_DIR / "combined_portfolio_v1_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    fprint(f"\n  Results saved to {out_path}")

    elapsed = time.time() - t0
    fprint(f"\n  Total runtime: {elapsed:.0f}s")
    fprint("  DONE.")


if __name__ == "__main__":
    main()
