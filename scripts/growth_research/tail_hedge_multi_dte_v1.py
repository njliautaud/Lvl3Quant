#!/usr/bin/env python3
"""
Tail Hedge + Multi-DTE Research v1
====================================

Tests 4 risk-management variants on top of the V9.1 sector rotation
options strategy (LGBM ranking, bull/bear spreads on sector ETFs).

Variants:
  A: No hedge baseline — standard V9.1 (3 bull + 3 bear, DTE=28, equal weight)
  B: VIX call overlay — 5% capital to VIX calls when VIX < 16
  C: Multi-DTE ladder — 3 positions at DTE=14, 3 at DTE=28
  D: Correlation gate — reduce to 4 positions when avg sector corr > 0.7

All use: adaptive max($3,3%) width, cost/width < 50%, real chain pricing,
LGBM 17 features, 52-week sliding window walk-forward.

Output: output/growth_research/tail_hedge_multi_dte_v1/
MLflow experiment: tail_hedge_multi_dte_v1
"""

import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# Limit LGBM threads to avoid CPU contention
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

_builtin_print = print
def fprint(*args, **kwargs):
    _builtin_print(*args, **kwargs)
    sys.stdout.flush()

sys.path.insert(0, "/home/jupiter/Lvl3Quant")
from research.tools.options_pricer import (
    price_bull_call_spread, price_bear_put_spread, COMMISSION_RT_SPREAD,
)
from research.tools.adversarial_validator import validate_trades

BASE = Path("/home/jupiter/Lvl3Quant")
CHAINS_DIR = BASE / "wheel_strategy_v1" / "data" / "cache" / "options_real" / "chains"
OUTPUT_DIR = BASE / "output" / "growth_research" / "tail_hedge_multi_dte_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
EXTRA_TICKERS = ["SPY", "^VIX", "^VIX3M", "TLT", "SHY", "HYG", "GLD"]
CAP = 645.0
TOP_K = 3
REBAL_FREQ = "W-FRI"
WF_TRAIN_PERIODS = 52  # 52-week sliding window

COST_WIDTH_MAX = 0.50
DTE_TOLERANCE = 5
STRIKE_TOLERANCE = 0.03
HAIRCUT = 0.15

MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "tail_hedge_multi_dte_v1"

MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen(MLFLOW_URI, timeout=3)
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    MLFLOW_OK = True
    fprint(f"MLflow connected: {MLFLOW_URI}")
except Exception:
    fprint("MLflow unavailable — will save results locally only")

V6_FEATURES = [
    "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "sharpe_63d", "pct_52w_high", "mom_accel",
    "pct_pos_months_12m", "sortino_63d", "calmar_1y", "up_capture",
    "trend_r2_63d", "trend_slope_63d",
    "sector_spy_beta_63d", "cross_sector_dispersion",
]
assert len(V6_FEATURES) == 17


# ══════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════

def download_data():
    import yfinance as yf
    all_tickers = SECTORS + EXTRA_TICKERS
    fprint(f"Downloading {len(all_tickers)} tickers...")
    raw = yf.download(all_tickers, start="2006-01-01", progress=False, auto_adjust=True)
    mi = isinstance(raw.columns, pd.MultiIndex)
    close = raw["Close"] if mi else raw[["Close"]]
    high = raw["High"] if mi else raw[["High"]]
    low = raw["Low"] if mi else raw[["Low"]]
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(-1)
        high.columns = high.columns.get_level_values(-1)
        low.columns = low.columns.get_level_values(-1)
    close = close.ffill(); high = high.ffill(); low = low.ffill()
    rename_map = {"^VIX": "VIX", "^VIX3M": "VIX3M"}
    close = close.rename(columns=rename_map)
    high = high.rename(columns=rename_map)
    low = low.rename(columns=rename_map)
    fprint(f"Data: {len(close)} days, {close.index[0].date()} to {close.index[-1].date()}")
    return close, high, low


def load_all_chains():
    chains = {}
    for tk in SECTORS:
        path = CHAINS_DIR / f"{tk}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        df["expiration"] = pd.to_datetime(df["expiration"])
        for c in ["strike", "bid", "ask", "mid", "vol", "delta"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["dte"] = (df["expiration"] - df["date"]).dt.days
        chains[tk] = df
        fprint(f"  {tk}: {len(df):,} chain rows")
    return chains


def compute_atr_series(high, low, close, period=14):
    atr_dict = {}
    for tk in SECTORS:
        if tk in high.columns and tk in low.columns and tk in close.columns:
            h = high[tk].dropna(); l = low[tk].dropna(); c = close[tk].dropna()
            common = h.index.intersection(l.index).intersection(c.index)
            if len(common) > period:
                tr1 = h.loc[common] - l.loc[common]
                tr2 = (h.loc[common] - c.loc[common].shift(1)).abs()
                tr3 = (l.loc[common] - c.loc[common].shift(1)).abs()
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr_dict[tk] = tr.ewm(alpha=1/period, min_periods=period).mean()
    return atr_dict


# ══════════════════════════════════════════════════════════════
# FEATURES + LGBM RANKING
# ══════════════════════════════════════════════════════════════

def compute_legacy_features(px, spy_slice):
    if len(px) < 260:
        return None
    f = {}
    for lb, nm in [(5, "ret_5d"), (10, "ret_10d"), (21, "ret_21d"),
                   (63, "ret_63d"), (126, "ret_126d"), (252, "ret_252d")]:
        f[nm] = float(px.iloc[-1] / px.iloc[-lb] - 1) if len(px) > lb else 0.0
    rets = px.pct_change().dropna()
    f["sharpe_63d"] = float(rets.iloc[-63:].mean() / (rets.iloc[-63:].std() + 1e-10) * np.sqrt(252)) if len(rets) > 63 else 0.0
    f["pct_52w_high"] = float(px.iloc[-1] / px.iloc[-252:].max())
    f["mom_accel"] = f["ret_21d"] - f["ret_63d"] / 3
    monthly = rets.resample("ME").sum()
    f["pct_pos_months_12m"] = float((monthly.iloc[-12:] > 0).mean()) if len(monthly) >= 12 else 0.5
    dr = rets.iloc[-63:][rets.iloc[-63:] < 0]
    f["sortino_63d"] = float(rets.iloc[-63:].mean() / (dr.std() + 1e-10) * np.sqrt(252)) if len(dr) > 3 else 0.0
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
    f = {}
    spy = close_df["SPY"].iloc[:dt_idx + 1].dropna()
    sector_px = close_df[sector_ticker].iloc[:dt_idx + 1].dropna() if sector_ticker in close_df.columns else None
    if spy is None or len(spy) < 63:
        return {"sector_spy_beta_63d": 1.0, "cross_sector_dispersion": 0.01}
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


def build_feature_records(close, high, low, rebal_dates, dte):
    """Build feature records for LGBM walk-forward ranking."""
    records = []
    sector_cols = [c for c in SECTORS if c in close.columns]
    spy = close["SPY"]
    for dt in rebal_dates:
        idx = close.index.get_indexer([dt], method="ffill")[0]
        if idx < 260:
            continue
        for tk in sector_cols:
            px = close[tk].iloc[:idx + 1].dropna()
            legacy = compute_legacy_features(px, spy.iloc[:idx + 1])
            if not legacy:
                continue
            cross_asset = compute_cross_asset_features(tk, idx, close)
            fi = min(idx + dte, len(close) - 1)
            if fi <= idx:
                continue
            fwd_ret = float(close[tk].iloc[fi] / close[tk].iloc[idx] - 1)
            rec = {**legacy, **cross_asset, "date": dt, "ticker": tk, "fwd_ret": fwd_ret}
            records.append(rec)
    df = pd.DataFrame(records)
    for c in V6_FEATURES:
        if c not in df.columns:
            df[c] = 0.0
    df[V6_FEATURES] = df[V6_FEATURES].fillna(0.0)
    fprint(f"    {len(df)} records, {len(df['date'].unique())} dates")
    return df


def walk_forward_lgbm_rank(df):
    """52-week sliding window LGBM walk-forward ranking."""
    import lightgbm as lgb
    if len(df) < 100:
        return {}
    df["rank_label"] = df.groupby("date")["fwd_ret"].rank(pct=True)
    dates = sorted(df["date"].unique())
    rankings = {}
    for i in range(WF_TRAIN_PERIODS, len(dates)):
        train_dates = dates[max(0, i - WF_TRAIN_PERIODS):i]
        test_date = dates[i]
        train_df = df[df["date"].isin(train_dates)]
        test_df = df[df["date"] == test_date].copy()
        if len(test_df) < 3 or len(train_df) < 50:
            continue
        Xt = np.nan_to_num(train_df[V6_FEATURES].values.astype(np.float32))
        yt = train_df["rank_label"].values.astype(np.float32)
        Xe = np.nan_to_num(test_df[V6_FEATURES].values.astype(np.float32))
        try:
            m = lgb.LGBMRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=5,
                n_jobs=2, verbose=-1,
            )
            m.fit(Xt, yt)
            test_df["score"] = m.predict(Xe)
            rankings[test_date] = dict(zip(test_df["ticker"], test_df["score"]))
        except Exception:
            continue
    fprint(f"    {len(rankings)} ranking dates")
    return rankings


# ══════════════════════════════════════════════════════════════
# CHAIN LOOKUP + TRADE EXECUTION
# ══════════════════════════════════════════════════════════════

def find_chain_spread_price(chain_df, trade_date, direction, K1, K2, dte_target):
    if chain_df is None:
        return None
    day_mask = chain_df["date"] == pd.Timestamp(trade_date)
    chain_day = chain_df[day_mask]
    if chain_day.empty:
        nearby = chain_df[
            (chain_df["date"] >= pd.Timestamp(trade_date) - pd.Timedelta(days=2)) &
            (chain_df["date"] <= pd.Timestamp(trade_date) + pd.Timedelta(days=2))
        ]
        if nearby.empty:
            return None
        nearest_date = min(nearby["date"].unique(),
                          key=lambda x: abs((x - pd.Timestamp(trade_date)).days))
        chain_day = chain_df[chain_df["date"] == nearest_date]

    exps = chain_day[["expiration", "dte"]].drop_duplicates()
    exps["dte_dist"] = (exps["dte"] - dte_target).abs()
    valid_exps = exps[exps["dte_dist"] <= DTE_TOLERANCE]
    if valid_exps.empty:
        return None
    best_exp_row = valid_exps.loc[valid_exps["dte_dist"].idxmin()]
    chain_exp = chain_day[chain_day["expiration"] == best_exp_row["expiration"]]

    opt_type = "c" if direction == "bull" else "p"
    near_target = K1 if direction == "bull" else K2
    far_target = K2 if direction == "bull" else K1

    near_opts = chain_exp[chain_exp["type"] == opt_type].copy()
    if near_opts.empty:
        return None
    near_opts["dist"] = (near_opts["strike"] - near_target).abs()
    near_leg = near_opts.sort_values("dist").iloc[0]
    if near_leg["dist"] / max(near_target, 1) > STRIKE_TOLERANCE:
        return None

    far_opts = chain_exp[chain_exp["type"] == opt_type].copy()
    far_opts["dist"] = (far_opts["strike"] - far_target).abs()
    far_leg = far_opts.sort_values("dist").iloc[0]
    if far_leg["dist"] / max(far_target, 1) > STRIKE_TOLERANCE:
        return None

    near_mid = float(near_leg["mid"]) if not pd.isna(near_leg["mid"]) else \
        (float(near_leg["bid"]) + float(near_leg["ask"])) / 2
    far_mid = float(far_leg["mid"]) if not pd.isna(far_leg["mid"]) else \
        (float(far_leg["bid"]) + float(far_leg["ask"])) / 2

    spread_cost_mid = abs(near_mid - far_mid)
    return {
        "found": True,
        "spread_cost_mid": spread_cost_mid,
        "near_strike": float(near_leg["strike"]),
        "far_strike": float(far_leg["strike"]),
        "actual_dte": int(best_exp_row["dte"]),
    }


def compute_strikes(S, direction):
    if direction == "bull":
        K1 = round(S * 1.03, 2)  # 3% OTM
        pct_w = K1 * 0.03
        w = max(3.0, pct_w)
        K2 = round(K1 + w, 2)
    else:
        K2 = round(S * 0.97, 2)  # 3% OTM
        pct_w = K2 * 0.03
        w = max(3.0, pct_w)
        K1 = round(K2 - w, 2)
    if K2 <= K1:
        K2 = K1 + 0.50
    return K1, K2


def execute_trade(tk, dt, direction, close, atr_dict, vix_val, equity,
                  chains, dte, max_pos):
    if tk not in close.columns or tk not in atr_dict:
        return None
    S = float(close[tk].loc[dt])
    di = close.index.get_loc(dt)
    ei = min(di + dte, len(close) - 1)
    if ei <= di:
        return None

    av = float(atr_dict[tk].loc[dt]) if dt in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[dt]) else S * 0.015
    K1, K2 = compute_strikes(S, direction)

    used_real = False
    entry_cost_ps = None

    chain_df = chains.get(tk)
    if chain_df is not None:
        result = find_chain_spread_price(chain_df, dt, direction, K1, K2, dte)
        if result and result["found"]:
            entry_cost_ps = result["spread_cost_mid"]
            used_real = True
            if direction == "bull":
                K1 = result["near_strike"]
                K2 = result["far_strike"]
            else:
                K1 = result["far_strike"]
                K2 = result["near_strike"]

    if entry_cost_ps is None:
        try:
            if direction == "bull":
                entry_cost_ps, _ = price_bull_call_spread(S=S, K1=K1, K2=K2, dte=dte, atr=av, vix=vix_val)
            else:
                entry_cost_ps, _ = price_bear_put_spread(S=S, K1=K1, K2=K2, dte=dte, atr=av, vix=vix_val)
        except Exception:
            return None

    if entry_cost_ps is None or entry_cost_ps <= 0:
        return None

    spread_width = abs(K2 - K1)
    cwr = entry_cost_ps / max(spread_width, 0.01)
    if cwr > COST_WIDTH_MAX:
        return None

    total_cost = entry_cost_ps * 100 + COMMISSION_RT_SPREAD
    if total_cost <= 0 or total_cost > max_pos or total_cost > equity * 0.40:
        return None

    Se = float(close[tk].iloc[ei])
    if direction == "bull":
        intrinsic = max(Se - K1, 0.0) - max(Se - K2, 0.0)
    else:
        intrinsic = max(K2 - Se, 0.0) - max(K1 - Se, 0.0)

    pnl = (intrinsic - entry_cost_ps) * 100 - COMMISSION_RT_SPREAD

    return {
        "pnl": round(pnl, 2), "entry_cost_ps": round(entry_cost_ps, 4),
        "total_cost": round(total_cost, 2), "used_real_pricing": used_real,
        "K1": K1, "K2": K2, "S_entry": round(S, 2), "S_exit": round(Se, 2),
        "intrinsic": round(intrinsic, 4), "spread_width": round(spread_width, 2),
        "cost_width_ratio": round(cwr, 3), "dte": dte,
    }


# ══════════════════════════════════════════════════════════════
# VIX CALL OVERLAY (Variant B)
# ══════════════════════════════════════════════════════════════

def simulate_vix_call_overlay(close, rebal_dates, equity_curve_dates):
    """
    When VIX < 16, allocate 5% of capital to buy VIX calls (cheap insurance).
    Model: buy 1-month ATM VIX call at VIX level, expires in ~28 days.
    Payoff: max(VIX_exit - VIX_entry, 0) * 100 - premium.
    Premium modeled as ~15% of VIX level (typical for ATM monthly VIX call).
    """
    vix = close["VIX"] if "VIX" in close.columns else None
    if vix is None:
        fprint("  WARNING: VIX data not available for overlay")
        return {}

    overlay_pnl = {}  # date -> pnl from VIX call

    for dt in rebal_dates:
        if dt not in vix.index:
            continue
        cv = float(vix.loc[dt])
        if cv >= 16.0:
            # Don't buy insurance when VIX already elevated
            overlay_pnl[dt] = 0.0
            continue

        # Buy ATM VIX call
        di = close.index.get_loc(dt)
        ei = min(di + 28, len(close) - 1)
        if ei <= di:
            continue

        exit_date = close.index[ei]
        if exit_date not in vix.index:
            continue

        vix_exit = float(vix.iloc[ei])
        premium_per_contract = cv * 0.15  # ~15% of VIX level for ATM call
        alloc = CAP * 0.05  # 5% of capital
        n_contracts = max(1, int(alloc / (premium_per_contract * 100)))
        cost = n_contracts * premium_per_contract * 100 + 1.30  # $0.65/leg * 2 legs

        # VIX call payoff
        intrinsic = max(vix_exit - cv, 0.0)
        proceeds = n_contracts * intrinsic * 100
        pnl = proceeds - cost

        overlay_pnl[dt] = round(pnl, 2)

    return overlay_pnl


# ══════════════════════════════════════════════════════════════
# CORRELATION GATE (Variant D)
# ══════════════════════════════════════════════════════════════

def compute_sector_correlation(close, dt, lookback=21):
    """Compute trailing pairwise sector correlation matrix average."""
    sector_cols = [c for c in SECTORS if c in close.columns]
    if len(sector_cols) < 5:
        return 0.3  # default low correlation

    idx = close.index.get_loc(dt)
    if idx < lookback + 5:
        return 0.3

    sector_rets = close[sector_cols].iloc[idx - lookback:idx + 1].pct_change().dropna()
    if len(sector_rets) < lookback - 3:
        return 0.3

    corr_matrix = sector_rets.corr()
    # Average off-diagonal correlation
    mask = np.ones_like(corr_matrix.values, dtype=bool)
    np.fill_diagonal(mask, False)
    avg_corr = float(corr_matrix.values[mask].mean())
    return avg_corr


# ══════════════════════════════════════════════════════════════
# SIMULATION ENGINE
# ══════════════════════════════════════════════════════════════

def simulate_variant_a(rankings, close, atr_dict, chains):
    """Variant A: No hedge baseline — standard V9.1."""
    vix = close["VIX"] if "VIX" in close.columns else None
    equity = CAP
    trades = []
    real_count = 0; bs_count = 0

    for dt in sorted(rankings.keys()):
        if dt not in close["SPY"].index:
            continue
        cv = float(vix.loc[dt]) if vix is not None and dt in vix.index else 20.0
        scores = rankings[dt]
        if not scores or len(scores) < 6:
            continue

        ranked_desc = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ranked_asc = sorted(scores.items(), key=lambda x: x[1])
        bull_picks = [t for t, _ in ranked_desc[:TOP_K]]
        bear_picks = [t for t, _ in ranked_asc[:TOP_K]]

        max_pos = min(100, equity / 6)
        if max_pos < 30:
            continue

        for direction, picks in [("bull", bull_picks), ("bear", bear_picks)]:
            for tk in picks:
                result = execute_trade(tk, dt, direction, close, atr_dict, cv,
                                       equity, chains, 28, max_pos)
                if result is not None:
                    equity += result["pnl"]
                    real_count += 1 if result["used_real_pricing"] else 0
                    bs_count += 0 if result["used_real_pricing"] else 1
                    di = close.index.get_loc(dt)
                    ei = min(di + 28, len(close) - 1)
                    spy_entry = float(close["SPY"].loc[dt])
                    spy_exit = float(close["SPY"].iloc[ei])
                    trades.append({
                        **result, "entry_date": str(dt.date()),
                        "exit_date": str(close.index[ei].date()),
                        "ticker": tk, "direction": direction,
                        "regime": "bull" if spy_exit >= spy_entry else "bear",
                        "vix": round(cv, 1), "win": result["pnl"] > 0,
                        "variant": "A",
                    })

    return trades, equity, real_count, bs_count


def simulate_variant_b(rankings, close, atr_dict, chains):
    """Variant B: VIX call overlay — same as A + VIX call insurance."""
    # Run base strategy (same as A)
    trades, equity_base, real_count, bs_count = simulate_variant_a(rankings, close, atr_dict, chains)

    # Add VIX call overlay PnL
    rebal_dates = sorted(rankings.keys())
    overlay = simulate_vix_call_overlay(close, rebal_dates, None)

    total_overlay_pnl = 0.0
    overlay_trades = 0
    overlay_wins = 0
    for dt, pnl in overlay.items():
        if pnl != 0:
            total_overlay_pnl += pnl
            overlay_trades += 1
            if pnl > 0:
                overlay_wins += 1

    # Adjust final equity
    equity = equity_base + total_overlay_pnl

    # Tag trades with variant
    for t in trades:
        t["variant"] = "B"
        t["pnl_with_overlay"] = t["pnl"]  # individual trade PnL unchanged

    # Add overlay summary as a synthetic trade entry
    if overlay_trades > 0:
        trades.append({
            "pnl": total_overlay_pnl,
            "ticker": "VIX_OVERLAY",
            "direction": "hedge",
            "entry_date": str(rebal_dates[0].date()) if rebal_dates else "N/A",
            "exit_date": str(rebal_dates[-1].date()) if rebal_dates else "N/A",
            "variant": "B",
            "win": total_overlay_pnl > 0,
            "overlay_trades": overlay_trades,
            "overlay_wins": overlay_wins,
            "regime": "hedge",
            "vix": 0,
            "used_real_pricing": False,
            "total_cost": 0, "entry_cost_ps": 0,
            "K1": 0, "K2": 0, "S_entry": 0, "S_exit": 0,
            "intrinsic": 0, "spread_width": 0, "cost_width_ratio": 0,
            "dte": 28,
        })

    fprint(f"  VIX overlay: {overlay_trades} trades, {overlay_wins} wins, "
           f"PnL ${total_overlay_pnl:,.0f}")
    return trades, equity, real_count, bs_count


def simulate_variant_c(rankings, close, atr_dict, chains):
    """Variant C: Multi-DTE ladder — 3 at DTE=14, 3 at DTE=28."""
    vix = close["VIX"] if "VIX" in close.columns else None
    equity = CAP
    trades = []
    real_count = 0; bs_count = 0

    for dt in sorted(rankings.keys()):
        if dt not in close["SPY"].index:
            continue
        cv = float(vix.loc[dt]) if vix is not None and dt in vix.index else 20.0
        scores = rankings[dt]
        if not scores or len(scores) < 6:
            continue

        ranked_desc = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ranked_asc = sorted(scores.items(), key=lambda x: x[1])

        # Top 3 bull, bottom 3 bear
        bull_picks = [t for t, _ in ranked_desc[:TOP_K]]
        bear_picks = [t for t, _ in ranked_asc[:TOP_K]]

        max_pos = min(100, equity / 6)
        if max_pos < 30:
            continue

        # Assign DTEs: first 2 bull + first 1 bear at DTE=14, rest at DTE=28
        # (stagger for smoother returns)
        assignments = []
        for i, tk in enumerate(bull_picks):
            dte = 14 if i < 2 else 28
            assignments.append(("bull", tk, dte))
        for i, tk in enumerate(bear_picks):
            dte = 14 if i < 1 else 28
            assignments.append(("bear", tk, dte))

        for direction, tk, dte in assignments:
            result = execute_trade(tk, dt, direction, close, atr_dict, cv,
                                   equity, chains, dte, max_pos)
            if result is not None:
                equity += result["pnl"]
                real_count += 1 if result["used_real_pricing"] else 0
                bs_count += 0 if result["used_real_pricing"] else 1
                di = close.index.get_loc(dt)
                ei = min(di + dte, len(close) - 1)
                spy_entry = float(close["SPY"].loc[dt])
                spy_exit = float(close["SPY"].iloc[ei])
                trades.append({
                    **result, "entry_date": str(dt.date()),
                    "exit_date": str(close.index[ei].date()),
                    "ticker": tk, "direction": direction,
                    "regime": "bull" if spy_exit >= spy_entry else "bear",
                    "vix": round(cv, 1), "win": result["pnl"] > 0,
                    "variant": "C",
                })

    return trades, equity, real_count, bs_count


def simulate_variant_d(rankings, close, atr_dict, chains):
    """Variant D: Correlation gate — reduce to 4 positions when corr > 0.7."""
    vix = close["VIX"] if "VIX" in close.columns else None
    equity = CAP
    trades = []
    real_count = 0; bs_count = 0
    crisis_weeks = 0; normal_weeks = 0

    for dt in sorted(rankings.keys()):
        if dt not in close["SPY"].index:
            continue
        cv = float(vix.loc[dt]) if vix is not None and dt in vix.index else 20.0
        scores = rankings[dt]
        if not scores or len(scores) < 6:
            continue

        # Correlation gate
        avg_corr = compute_sector_correlation(close, dt, lookback=21)
        crisis_mode = avg_corr > 0.70

        if crisis_mode:
            crisis_weeks += 1
            n_bull = 2
            n_bear = 2
        else:
            normal_weeks += 1
            n_bull = TOP_K  # 3
            n_bear = TOP_K  # 3

        ranked_desc = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ranked_asc = sorted(scores.items(), key=lambda x: x[1])
        bull_picks = [t for t, _ in ranked_desc[:n_bull]]
        bear_picks = [t for t, _ in ranked_asc[:n_bear]]

        total_positions = n_bull + n_bear
        max_pos = min(100, equity / total_positions)
        if max_pos < 30:
            continue

        for direction, picks in [("bull", bull_picks), ("bear", bear_picks)]:
            for tk in picks:
                result = execute_trade(tk, dt, direction, close, atr_dict, cv,
                                       equity, chains, 28, max_pos)
                if result is not None:
                    equity += result["pnl"]
                    real_count += 1 if result["used_real_pricing"] else 0
                    bs_count += 0 if result["used_real_pricing"] else 1
                    di = close.index.get_loc(dt)
                    ei = min(di + 28, len(close) - 1)
                    spy_entry = float(close["SPY"].loc[dt])
                    spy_exit = float(close["SPY"].iloc[ei])
                    trades.append({
                        **result, "entry_date": str(dt.date()),
                        "exit_date": str(close.index[ei].date()),
                        "ticker": tk, "direction": direction,
                        "regime": "bull" if spy_exit >= spy_entry else "bear",
                        "vix": round(cv, 1), "win": result["pnl"] > 0,
                        "variant": "D", "crisis_mode": crisis_mode,
                        "avg_corr": round(avg_corr, 3),
                    })

    fprint(f"  Correlation gate: {crisis_weeks} crisis weeks, {normal_weeks} normal weeks")
    return trades, equity, real_count, bs_count


# ══════════════════════════════════════════════════════════════
# METRICS + ANALYSIS
# ══════════════════════════════════════════════════════════════

def compute_metrics(trades, label):
    """Compute comprehensive metrics for a set of trades."""
    if not trades or len(trades) < 5:
        return None

    pnls = np.array([t["pnl"] for t in trades])
    n = len(pnls)
    wins = sum(1 for p in pnls if p > 0)
    losses = sum(1 for p in pnls if p <= 0)

    # Equity curve
    equity = [CAP]
    for p in pnls:
        equity.append(equity[-1] + p)
    equity = np.array(equity)

    # Returns
    returns = np.diff(equity) / equity[:-1]
    returns = returns[np.isfinite(returns)]

    # Sharpe (annualized, weekly rebal)
    sharpe = float(np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(52))

    # Sortino
    downside = returns[returns < 0]
    sortino = float(np.mean(returns) / (np.std(downside) + 1e-10) * np.sqrt(52)) if len(downside) > 0 else sharpe

    # Max Drawdown
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = float(dd.min())

    # Calmar
    years = n / 52.0
    total_return = (equity[-1] / equity[0]) - 1
    cagr = (1 + total_return) ** (1 / max(years, 0.1)) - 1
    calmar = float(cagr / (abs(max_dd) + 1e-10))

    # Profit factor
    gross_wins = sum(p for p in pnls if p > 0)
    gross_losses = abs(sum(p for p in pnls if p <= 0))
    pf = float(gross_wins / (gross_losses + 1e-10))

    # Win rate
    wr = wins / n

    return {
        "label": label,
        "n_trades": n,
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "calmar": round(calmar, 3),
        "max_dd": round(max_dd, 4),
        "max_dd_pct": round(max_dd * 100, 2),
        "win_rate": round(wr, 4),
        "profit_factor": round(pf, 3),
        "total_pnl": round(float(sum(pnls)), 2),
        "avg_pnl": round(float(np.mean(pnls)), 2),
        "final_equity": round(float(equity[-1]), 2),
        "cagr": round(cagr, 4),
    }


def compute_tail_protection_ratio(baseline_dd, variant_dd):
    """How much did the variant improve MaxDD over baseline?"""
    if baseline_dd == 0:
        return 0.0
    improvement = (abs(variant_dd) - abs(baseline_dd)) / abs(baseline_dd)
    return round(-improvement, 4)  # positive = variant has less DD


def monte_carlo_ci(trades, n_bootstrap=1000, seed=42):
    if len(trades) < 10:
        return None
    pnls = np.array([t["pnl"] for t in trades])
    rng = np.random.RandomState(seed)
    sharpes = []
    for _ in range(n_bootstrap):
        sample = rng.choice(pnls, size=len(pnls), replace=True)
        eq = np.cumsum(np.concatenate([[CAP], sample]))
        rets = np.diff(eq) / eq[:-1]
        rets = rets[np.isfinite(rets)]
        sh = float(np.mean(rets) / (np.std(rets) + 1e-10) * np.sqrt(52))
        sharpes.append(sh)
    sharpes = np.array(sharpes)
    return {
        "mean": round(float(np.mean(sharpes)), 3),
        "ci_95_low": round(float(np.percentile(sharpes, 2.5)), 3),
        "ci_95_high": round(float(np.percentile(sharpes, 97.5)), 3),
    }


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    t0 = datetime.now()
    fprint("=" * 110)
    fprint(f"TAIL HEDGE + MULTI-DTE RESEARCH v1 — {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 110)
    fprint(f"Capital: ${CAP:.0f} | Commission: ${COMMISSION_RT_SPREAD:.2f} | Haircut: {HAIRCUT:.0%}")
    fprint(f"Sectors: {len(SECTORS)} | LGBM features: {len(V6_FEATURES)} | WF window: {WF_TRAIN_PERIODS} weeks")
    fprint()

    # ── Load data ──
    fprint("Loading chain data...")
    chains = load_all_chains()

    fprint("\nDownloading price data...")
    close, high, low = download_data()

    fprint("\nComputing ATR...")
    atr_dict = compute_atr_series(high, low, close)

    # ── Build LGBM rankings ──
    rebal_dates = pd.DatetimeIndex(
        close.index.to_series().resample(REBAL_FREQ).last().dropna().values
    )
    fprint(f"\nRebalance dates: {len(rebal_dates)} ({rebal_dates[0].date()} to {rebal_dates[-1].date()})")

    fprint(f"\n{'=' * 80}")
    fprint("BUILDING LGBM WALK-FORWARD RANKINGS")
    fprint(f"{'=' * 80}")

    # Build for DTE=28 (used by A, B, D) and DTE=14 (used by C)
    lgbm_rankings = {}
    for dte in [14, 28]:
        fprint(f"\n  DTE={dte}:")
        records = build_feature_records(close, high, low, rebal_dates, dte=dte)
        rankings = walk_forward_lgbm_rank(records)
        lgbm_rankings[dte] = rankings

    # ── Run all 4 variants ──
    VARIANTS = {
        "A": ("No hedge baseline (V9.1)", simulate_variant_a, lgbm_rankings[28]),
        "B": ("VIX call overlay", simulate_variant_b, lgbm_rankings[28]),
        "C": ("Multi-DTE ladder (14+28)", simulate_variant_c, lgbm_rankings[28]),
        "D": ("Correlation gate", simulate_variant_d, lgbm_rankings[28]),
    }

    all_results = {}
    all_trades = {}
    spy_close = close["SPY"]

    for vname, (desc, sim_func, rankings) in VARIANTS.items():
        fprint(f"\n{'~' * 100}")
        fprint(f"VARIANT {vname}: {desc}")
        fprint(f"{'~' * 100}")

        trades, final_eq, real_count, bs_count = sim_func(rankings, close, atr_dict, chains)

        if not trades or len(trades) < 5:
            fprint(f"  Only {len(trades) if trades else 0} trades — skipping")
            continue

        all_trades[vname] = trades
        total = real_count + bs_count
        fprint(f"\n  Trades: {len(trades)} | Real pricing: {real_count} ({real_count/max(total,1)*100:.0f}%) | "
               f"Final equity: ${final_eq:,.0f}")

        # Compute metrics
        metrics = compute_metrics(trades, vname)
        if metrics is None:
            continue

        fprint(f"  Sharpe: {metrics['sharpe']:.2f} | Sortino: {metrics['sortino']:.2f} | "
               f"Calmar: {metrics['calmar']:.2f}")
        fprint(f"  MaxDD: {metrics['max_dd_pct']:.1f}% | WR: {metrics['win_rate']*100:.1f}% | "
               f"PF: {metrics['profit_factor']:.2f}")

        # 5-gate validation
        result = validate_trades(trades, initial_capital=CAP, spy_prices=spy_close, strategy_name=vname)
        result.print_summary()

        rd = result.to_dict()
        mc = monte_carlo_ci(trades)

        # Bull/bear breakdown
        for side in ["bull", "bear"]:
            st = [t for t in trades if t.get("direction") == side]
            if st:
                pnls = [t["pnl"] for t in st]
                wr = sum(1 for p in pnls if p > 0) / len(pnls)
                fprint(f"  {side}: {len(st)} trades, WR {wr:.1%}, PnL ${sum(pnls):,.0f}")

        # Real-only analysis
        real_trades = [t for t in trades if t.get("used_real_pricing")]
        real_metrics = compute_metrics(real_trades, f"{vname}_real") if len(real_trades) >= 5 else None

        all_results[vname] = {
            **metrics, **rd,
            "desc": desc,
            "real_pct": round(real_count / max(total, 1) * 100, 1),
            "real_metrics": real_metrics,
            "monte_carlo": mc,
        }

    if len(all_results) < 2:
        fprint("\nERROR: Not enough variants completed. Aborting.")
        return

    # ── COMPARISON TABLE ──
    fprint(f"\n{'=' * 140}")
    fprint("COMPARISON — ALL VARIANTS")
    fprint(f"{'=' * 140}")
    fprint(f"  {'Variant':<35} {'N':>5} {'Sharpe':>7} {'Sort':>7} {'Calmar':>7} "
           f"{'MDD':>7} {'WR':>6} {'PF':>6} {'Gate':>5} {'Final$':>9}")
    fprint(f"  {'-' * 110}")

    baseline_dd = all_results.get("A", {}).get("max_dd", 0)

    for vn in ["A", "B", "C", "D"]:
        r = all_results.get(vn)
        if not r:
            continue
        fprint(f"  {vn}: {r['desc']:<30} {r['n_trades']:>5} {r['sharpe']:>7.2f} "
               f"{r['sortino']:>7.2f} {r['calmar']:>7.2f} "
               f"{r['max_dd_pct']:>6.1f}% {r['win_rate']*100:>5.1f}% "
               f"{r['profit_factor']:>5.2f} {r['gates_passed']}/{r['gates_total']} "
               f"${r['final_equity']:>8,.0f}")

    # ── TAIL PROTECTION RATIO ──
    fprint(f"\n{'=' * 80}")
    fprint("TAIL PROTECTION RATIO (MaxDD improvement vs Baseline A)")
    fprint(f"{'=' * 80}")
    for vn in ["B", "C", "D"]:
        r = all_results.get(vn)
        if not r:
            continue
        tpr = compute_tail_protection_ratio(baseline_dd, r["max_dd"])
        fprint(f"  {vn}: {r['desc']:<35} TPR={tpr:+.1%} "
               f"(MDD {r['max_dd_pct']:.1f}% vs baseline {baseline_dd*100:.1f}%)")

    # ── REAL-ONLY COMPARISON ──
    fprint(f"\n{'=' * 80}")
    fprint("REAL-ONLY TRADES (chain-priced, most trustworthy)")
    fprint(f"{'=' * 80}")
    fprint(f"  {'Variant':<35} {'N':>5} {'Sharpe':>8} {'WR':>6} {'PnL':>9}")
    fprint(f"  {'-' * 70}")
    for vn in ["A", "B", "C", "D"]:
        r = all_results.get(vn)
        if not r or not r.get("real_metrics"):
            continue
        rm = r["real_metrics"]
        fprint(f"  {vn}: {r['desc']:<30} {rm['n_trades']:>5} {rm['sharpe']:>8.2f} "
               f"{rm['win_rate']*100:>5.1f}% ${rm['total_pnl']:>8.0f}")

    # ── MONTE CARLO ──
    fprint(f"\n{'=' * 80}")
    fprint("MONTE CARLO BOOTSTRAP (1000 resamples)")
    fprint(f"{'=' * 80}")
    fprint(f"  {'Variant':<35} {'Mean':>8} {'95% CI':>22}")
    fprint(f"  {'-' * 70}")
    for vn in ["A", "B", "C", "D"]:
        r = all_results.get(vn)
        if not r or not r.get("monte_carlo"):
            continue
        mc = r["monte_carlo"]
        fprint(f"  {vn}: {r['desc']:<30} {mc['mean']:>8.2f} "
               f"[{mc['ci_95_low']:.2f}, {mc['ci_95_high']:.2f}]")

    # ── VERDICT ──
    fprint(f"\n{'=' * 110}")
    fprint("VERDICT")
    fprint(f"{'=' * 110}")

    best_vn = None
    best_sh = -999
    for vn in ["A", "B", "C", "D"]:
        r = all_results.get(vn)
        if not r:
            continue
        if r.get("gates_passed", 0) >= 4 and r["sharpe"] > best_sh:
            best_sh = r["sharpe"]
            best_vn = vn

    if best_vn:
        r = all_results[best_vn]
        fprint(f"\n  BEST VARIANT: {best_vn} — {r['desc']}")
        fprint(f"  Sharpe {r['sharpe']:.2f} | Sortino {r['sortino']:.2f} | "
               f"Calmar {r['calmar']:.2f} | MDD {r['max_dd_pct']:.1f}% | "
               f"Gates {r['gates_passed']}/{r['gates_total']}")

        if best_vn != "A":
            tpr = compute_tail_protection_ratio(baseline_dd, r["max_dd"])
            fprint(f"  Tail protection ratio vs baseline: {tpr:+.1%}")

    # Per-variant verdicts
    for vn in ["B", "C", "D"]:
        r = all_results.get(vn)
        a = all_results.get("A")
        if not r or not a:
            continue
        sharpe_delta = r["sharpe"] - a["sharpe"]
        tpr = compute_tail_protection_ratio(baseline_dd, r["max_dd"])
        verdict = "IMPROVES" if (sharpe_delta > -0.1 and tpr > 0.05) else "NO IMPROVEMENT"
        fprint(f"\n  {vn} ({r['desc']}): {verdict}")
        fprint(f"    Sharpe delta: {sharpe_delta:+.2f} | TPR: {tpr:+.1%}")

    # ── SAVE ──
    results_file = OUTPUT_DIR / "results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    fprint(f"\nResults saved to {results_file}")

    for vn, trades in all_trades.items():
        tf = OUTPUT_DIR / f"trades_{vn}.json"
        with open(tf, "w") as f:
            json.dump(trades, f, indent=1, default=str)

    # ── MLflow ──
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"tail_hedge_{t0.strftime('%Y%m%d_%H%M')}"):
                for vn, r in all_results.items():
                    mlflow.log_metric(f"{vn}_sharpe", r["sharpe"])
                    mlflow.log_metric(f"{vn}_sortino", r["sortino"])
                    mlflow.log_metric(f"{vn}_calmar", r.get("calmar", 0))
                    mlflow.log_metric(f"{vn}_wr", r["win_rate"])
                    mlflow.log_metric(f"{vn}_pf", r["profit_factor"])
                    mlflow.log_metric(f"{vn}_mdd", r["max_dd"])
                    mlflow.log_metric(f"{vn}_gates", r["gates_passed"])
                    if r.get("real_metrics"):
                        mlflow.log_metric(f"{vn}_real_sharpe", r["real_metrics"]["sharpe"])
                mlflow.log_artifact(str(results_file))
            fprint(f"MLflow logged to experiment '{EXPERIMENT_NAME}'")
        except Exception as e:
            fprint(f"MLflow error: {e}")

    elapsed = (datetime.now() - t0).total_seconds()
    fprint(f"\nCompleted in {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
