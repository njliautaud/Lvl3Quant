#!/usr/bin/env python3
"""
V9 Candidate v1 — Combining All Optimizations
==============================================

V9 combines the best findings from V8 validation:
  - KB #233: Cost/width filter < 50%
  - KB #234: Bull-only has better real-pricing performance (but fails regime gate)
  - KB #235: Adaptive width max($3, 3%) is optimal (5/5 gates, chain-only 3.51)

V9 Config vs V8:
  V8: 3% width, pairs when VIX<20, no cost/width filter
  V9: Adaptive max($3, 3%) width, cost/width<50% filter, pairs when VIX<20

4 Variants to test the full V9 candidate:
  A: V9 core (adaptive width + filter + pairs) — THE CANDIDATE
  B: V9 + bull-only (highest 2026 but may fail regime gate)
  C: V9 + wider OTM (3% instead of 2% — from earlier finding that 3% OTM helps)
  D: V9 + 21-day DTE (instead of 14-day — more time for move but higher entry cost)

Each variant runs with BOTH BS and real pricing for comparison.

Output: output/growth_research/v9_candidate_v1/
MLflow experiment: v9_candidate_v1
"""

import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

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
OUTPUT_DIR = BASE / "output" / "growth_research" / "v9_candidate_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX", "^VIX3M", "TLT", "SHY", "HYG", "GLD"]
CAP = 645.0
TOP_K = 3
REGIME_BULL_THRESHOLD = 0.4
REBAL_FREQ = "W-FRI"
WF_TRAIN_PERIODS = 12

COST_WIDTH_MAX = 0.50
DTE_TOLERANCE = 5
STRIKE_TOLERANCE = 0.03
MIN_BID = 0.05

REGIME_FILE = BASE / "output" / "regime_detector_v1" / "regime_predictions_v1.npz"
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "v9_candidate_v1"

MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen(MLFLOW_URI, timeout=3)
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    MLFLOW_OK = True
    fprint(f"MLflow connected: {MLFLOW_URI}")
except Exception:
    fprint("MLflow unavailable")

V6_FEATURES = [
    "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "sharpe_63d", "pct_52w_high", "mom_accel",
    "pct_pos_months_12m", "sortino_63d", "calmar_1y", "up_capture",
    "trend_r2_63d", "trend_slope_63d",
    "sector_spy_beta_63d", "cross_sector_dispersion",
]
assert len(V6_FEATURES) == 17


# ══════════════════════════════════════════════════════════════
# CHAIN DATA
# ══════════════════════════════════════════════════════════════

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
        fprint(f"  {tk}: {len(df):,} rows")
    return chains


def find_chain_spread_price(chain_df, trade_date, direction, K1, K2, dte_target):
    if chain_df is None:
        return None
    day_mask = chain_df["date"] == pd.Timestamp(trade_date)
    chain_day = chain_df[day_mask]
    if chain_day.empty:
        nearby = chain_df[(chain_df["date"] >= pd.Timestamp(trade_date) - pd.Timedelta(days=2)) &
                          (chain_df["date"] <= pd.Timestamp(trade_date) + pd.Timedelta(days=2))]
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
    }


# ══════════════════════════════════════════════════════════════
# DATA + FEATURES
# ══════════════════════════════════════════════════════════════

def download_data():
    import yfinance as yf
    all_tickers = SECTORS + EXTRA_TICKERS
    fprint(f"Downloading {len(all_tickers)} tickers...")
    raw = yf.download(all_tickers, start="2008-01-01", progress=False, auto_adjust=True)
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


def load_regime_predictions():
    if not REGIME_FILE.exists():
        return None
    data = np.load(REGIME_FILE, allow_pickle=True)
    dates = pd.to_datetime(data["dates"])
    scores = data["regime_scores"]
    regime_series = pd.Series(scores, index=dates, name="regime_score")
    regime_series = regime_series[~regime_series.index.duplicated(keep="last")]
    return regime_series


def get_regime_score_at(regime_series, dt):
    if regime_series is None:
        return 0.5
    if dt in regime_series.index:
        return float(regime_series.loc[dt])
    nearest = regime_series.index[regime_series.index.get_indexer([dt], method="ffill")]
    if len(nearest) > 0:
        return float(regime_series.loc[nearest[0]])
    return 0.5


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
# LGBM RANKING
# ══════════════════════════════════════════════════════════════

def build_feature_records(close, high, low, rebal_dates, regime_series, dte):
    records = []
    sector_cols = [c for c in SECTORS if c in close.columns]
    spy = close["SPY"]
    for dt in rebal_dates:
        idx = close.index.get_indexer([dt], method="ffill")[0]
        if idx < 260:
            continue
        rscore = get_regime_score_at(regime_series, dt)
        if rscore <= REGIME_BULL_THRESHOLD:
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
                subsample=0.8, colsample_bytree=0.8, min_child_samples=5, verbose=-1,
            )
            m.fit(Xt, yt)
            test_df["score"] = m.predict(Xe)
            rankings[test_date] = dict(zip(test_df["ticker"], test_df["score"]))
        except Exception:
            continue
    fprint(f"    {len(rankings)} ranking dates")
    return rankings


# ══════════════════════════════════════════════════════════════
# STRIKES + EXECUTION
# ══════════════════════════════════════════════════════════════

def compute_strikes(S, direction, otm_pct, width_mode, width_value):
    if direction == "bull":
        K1 = round(S * (1 + otm_pct), 2)
        if width_mode == "adaptive":
            pct_w = K1 * 0.03  # 3% of strike
            w = max(width_value, pct_w)
            K2 = round(K1 + w, 2)
        else:
            K2 = round(K1 * (1 + width_value / 100), 2)
    else:
        K2 = round(S * (1 - otm_pct), 2)
        if width_mode == "adaptive":
            pct_w = K2 * 0.03
            w = max(width_value, pct_w)
            K1 = round(K2 - w, 2)
        else:
            K1 = round(K2 * (1 - width_value / 100), 2)
    if K2 <= K1:
        K2 = K1 + 0.50
    return K1, K2


def execute_trade(tk, dt, direction, close, atr_dict, vix_val, equity,
                  chains, use_real, max_pos, dte, otm_pct, width_mode, width_value,
                  cost_filter):
    if tk not in close.columns or tk not in atr_dict:
        return None
    S = float(close[tk].loc[dt])
    di = close.index.get_loc(dt)
    ei = min(di + dte, len(close) - 1)
    if ei <= di:
        return None

    av = float(atr_dict[tk].loc[dt]) if dt in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[dt]) else S * 0.015
    K1, K2 = compute_strikes(S, direction, otm_pct, width_mode, width_value)

    used_real = False
    entry_cost_ps = None

    if use_real:
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
    if cost_filter and cwr > COST_WIDTH_MAX:
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
        "cost_width_ratio": round(cwr, 3),
    }


# ══════════════════════════════════════════════════════════════
# VARIANTS
# ══════════════════════════════════════════════════════════════

VARIANTS = {
    "V8_baseline_bs": {
        "dte": 14, "otm_pct": 0.02, "width_mode": "pct", "width_value": 3.0,
        "bear_mode": "pairs", "vix_threshold": 20.0, "cost_filter": False,
        "use_real": False, "desc": "V8 baseline (3% width, BS pricing)",
    },
    "V8_baseline_real": {
        "dte": 14, "otm_pct": 0.02, "width_mode": "pct", "width_value": 3.0,
        "bear_mode": "pairs", "vix_threshold": 20.0, "cost_filter": True,
        "use_real": True, "desc": "V8 + real + filter (current prod)",
    },
    "V9_core": {
        "dte": 14, "otm_pct": 0.02, "width_mode": "adaptive", "width_value": 3.0,
        "bear_mode": "pairs", "vix_threshold": 20.0, "cost_filter": True,
        "use_real": True, "desc": "V9 CORE: adaptive max($3,3%) + real + filter",
    },
    "V9_bull_only": {
        "dte": 14, "otm_pct": 0.02, "width_mode": "adaptive", "width_value": 3.0,
        "bear_mode": "never", "vix_threshold": 0.0, "cost_filter": True,
        "use_real": True, "desc": "V9 + bull-only",
    },
    "V9_3pct_otm": {
        "dte": 14, "otm_pct": 0.03, "width_mode": "adaptive", "width_value": 3.0,
        "bear_mode": "pairs", "vix_threshold": 20.0, "cost_filter": True,
        "use_real": True, "desc": "V9 + 3% OTM (further out)",
    },
    "V9_21d_dte": {
        "dte": 21, "otm_pct": 0.02, "width_mode": "adaptive", "width_value": 3.0,
        "bear_mode": "pairs", "vix_threshold": 20.0, "cost_filter": True,
        "use_real": True, "desc": "V9 + 21-day DTE",
    },
    "V9_5dollar_width": {
        "dte": 14, "otm_pct": 0.02, "width_mode": "adaptive", "width_value": 5.0,
        "bear_mode": "pairs", "vix_threshold": 20.0, "cost_filter": True,
        "use_real": True, "desc": "V9 + $5 min width (wider spreads)",
    },
}


def simulate_variant(vcfg, rankings, close, atr_dict, chains):
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None

    equity = CAP
    trades = []
    real_count = 0
    bs_count = 0

    for dt in sorted(rankings.keys()):
        if dt not in spy.index:
            continue
        cv = float(vix.loc[dt]) if vix is not None and dt in vix.index else 20.0
        scores = rankings[dt]
        if not scores:
            continue

        if vcfg["bear_mode"] == "never":
            trade_mode = "bull_only"
        elif cv < vcfg["vix_threshold"]:
            trade_mode = "pairs"
        else:
            trade_mode = "bull_only"

        ranked_desc = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ranked_asc = sorted(scores.items(), key=lambda x: x[1])
        bull_picks = [t for t, _ in ranked_desc[:TOP_K]]
        bear_picks = [t for t, _ in ranked_asc[:TOP_K]] if trade_mode == "pairs" else []

        max_pos = min(100, equity / 6) if trade_mode == "pairs" else min(200, equity / 3)
        if max_pos < 30:
            continue

        for direction, picks in [("bull", bull_picks), ("bear", bear_picks)]:
            for tk in picks:
                result = execute_trade(
                    tk, dt, direction, close, atr_dict, cv, equity, chains,
                    vcfg["use_real"], max_pos, vcfg["dte"], vcfg["otm_pct"],
                    vcfg["width_mode"], vcfg["width_value"], vcfg["cost_filter"]
                )
                if result is not None:
                    equity += result["pnl"]
                    real_count += 1 if result["used_real_pricing"] else 0
                    bs_count += 0 if result["used_real_pricing"] else 1
                    di = close.index.get_loc(dt)
                    ei = min(di + vcfg["dte"], len(close) - 1)
                    sv = float(spy.loc[dt])
                    se = float(spy.iloc[ei]) if ei < len(spy) else sv
                    trades.append({
                        **result, "entry_date": str(dt.date()),
                        "exit_date": str(close.index[ei].date()),
                        "ticker": tk, "regime": "bull" if se >= sv else "bear",
                        "direction": direction, "vix": round(cv, 1),
                        "win": result["pnl"] > 0, "trade_mode": trade_mode,
                    })

    return trades, equity, real_count, bs_count


def chain_only_analysis(trades):
    ct = [t for t in trades if t["entry_date"] >= "2019"]
    if not ct:
        return None
    pnls = [t["pnl"] for t in ct]
    return {
        "trades": len(ct), "wr": sum(1 for p in pnls if p > 0) / len(pnls),
        "sharpe": float(np.mean(pnls) / (np.std(pnls) + 1e-10) * np.sqrt(52)),
        "total_pnl": sum(pnls),
        "pct_real": sum(1 for t in ct if t["used_real_pricing"]) / len(ct) * 100,
    }


def year_2026_analysis(trades):
    t26 = [t for t in trades if t["entry_date"][:4] == "2026"]
    if not t26:
        return None
    pnls = [t["pnl"] for t in t26]
    return {
        "trades": len(t26), "wr": sum(1 for p in pnls if p > 0) / len(pnls),
        "sharpe": float(np.mean(pnls) / (np.std(pnls) + 1e-10) * np.sqrt(52)),
        "total_pnl": sum(pnls),
    }


# ══════════════════════════════════════════════════════════════
# RANDOM BASELINE TEST
# ══════════════════════════════════════════════════════════════

def random_baseline_test(vcfg, rankings, close, atr_dict, chains, n_trials=5):
    """Run strategy with random sector rankings to measure structural edge."""
    fprint(f"\n  RANDOM BASELINE TEST ({n_trials} trials)...")
    random_sharpes = []
    for trial in range(n_trials):
        np.random.seed(42 + trial)
        random_rankings = {}
        for dt, scores in rankings.items():
            tickers = list(scores.keys())
            random_scores = {tk: np.random.uniform(0, 1) for tk in tickers}
            random_rankings[dt] = random_scores

        trades, final_eq, _, _ = simulate_variant(vcfg, random_rankings, close, atr_dict, chains)
        if trades:
            pnls = [t["pnl"] for t in trades]
            sh = float(np.mean(pnls) / (np.std(pnls) + 1e-10) * np.sqrt(52))
            random_sharpes.append(sh)
            fprint(f"    Trial {trial}: Sharpe {sh:.2f}, ${CAP:.0f}->${final_eq:,.0f}")

    if random_sharpes:
        mean_random = np.mean(random_sharpes)
        fprint(f"    Random mean Sharpe: {mean_random:.2f}")
        return mean_random
    return None


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    t0 = datetime.now()
    fprint("=" * 100)
    fprint(f"V9 CANDIDATE BACKTEST v1 — {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 100)
    fprint(f"Testing V9 upgrade: adaptive width + cost/width filter + real pricing")
    fprint(f"Capital: ${CAP:.0f} | Commission: ${COMMISSION_RT_SPREAD:.2f}")
    fprint(f"\n{len(VARIANTS)} variants:")
    for vn, vc in VARIANTS.items():
        fprint(f"  {vn}: {vc['desc']}")
    fprint()

    fprint("Loading chains...")
    chains = load_all_chains()
    close, high, low = download_data()
    regime_series = load_regime_predictions()
    atr_dict = compute_atr_series(high, low, close)

    # Build rankings with DTE=14 (shared for most variants)
    fprint(f"\n{'=' * 80}")
    fprint("BUILDING LGBM RANKINGS (DTE=14)")
    fprint(f"{'=' * 80}")
    rebal_dates = pd.DatetimeIndex(
        close.index.to_series().resample(REBAL_FREQ).last().dropna().values
    )
    records_14 = build_feature_records(close, high, low, rebal_dates, regime_series, dte=14)
    rankings_14 = walk_forward_lgbm_rank(records_14)

    # Build rankings with DTE=21 for the 21d variant
    fprint(f"\n{'=' * 80}")
    fprint("BUILDING LGBM RANKINGS (DTE=21)")
    fprint(f"{'=' * 80}")
    records_21 = build_feature_records(close, high, low, rebal_dates, regime_series, dte=21)
    rankings_21 = walk_forward_lgbm_rank(records_21)

    if not rankings_14:
        fprint("ERROR: No rankings. Aborting.")
        return

    fprint(f"\n{'=' * 100}")
    fprint("SIMULATING ALL VARIANTS")
    fprint(f"{'=' * 100}")

    all_results = {}
    all_trades = {}
    spy_close = close["SPY"]

    for vname, vcfg in VARIANTS.items():
        fprint(f"\n{'~' * 90}")
        fprint(f"VARIANT {vname}: {vcfg['desc']}")
        fprint(f"{'~' * 90}")

        rankings = rankings_21 if vcfg["dte"] == 21 else rankings_14

        trades, final_eq, real_count, bs_count = simulate_variant(
            vcfg, rankings, close, atr_dict, chains
        )
        if not trades or len(trades) < 10:
            fprint(f"  Only {len(trades) if trades else 0} trades, skipping")
            continue

        all_trades[vname] = trades
        total = real_count + bs_count
        fprint(f"\n  Trades: {len(trades)} | Real: {real_count} ({real_count/total*100:.0f}%) | "
               f"Final: ${final_eq:,.0f}")

        result = validate_trades(trades, initial_capital=CAP, spy_prices=spy_close, strategy_name=vname)
        result.print_summary()

        # Side breakdown
        for side in ["bull", "bear"]:
            st = [t for t in trades if t["direction"] == side]
            if st:
                pnls = [t["pnl"] for t in st]
                wr = sum(1 for p in pnls if p > 0) / len(pnls)
                fprint(f"  {side}: {len(st)} trades, WR {wr:.1%}, PnL ${sum(pnls):,.0f}")

        rd = result.to_dict()
        co = chain_only_analysis(trades)
        y26 = year_2026_analysis(trades)
        all_results[vname] = {
            **rd, "chain_only_2019": co, "year_2026": y26,
            "real_pct": round(real_count / max(total, 1) * 100, 1),
        }

    # Random baseline on V9 core
    fprint(f"\n{'=' * 80}")
    fprint("STRUCTURAL EDGE TEST (V9 core)")
    fprint(f"{'=' * 80}")
    v9_cfg = VARIANTS["V9_core"]
    random_mean = random_baseline_test(v9_cfg, rankings_14, close, atr_dict, chains)

    # ── COMPARISON ──
    fprint(f"\n{'=' * 140}")
    fprint("COMPARISON — ALL VARIANTS (FULL PERIOD)")
    fprint(f"{'=' * 140}")
    fprint(f"  {'Variant':<25} {'N':>5} {'Sharpe':>7} {'Sort':>7} {'WR':>6} "
           f"{'PF':>6} {'MDD':>7} {'Gate':>5} {'Final$':>9}")
    fprint(f"  {'-' * 85}")
    for vn in VARIANTS:
        r = all_results.get(vn)
        if not r:
            continue
        fprint(f"  {vn:<25} {r['n_trades']:>5} {r['sharpe']:>7.2f} {r['sortino']:>7.2f} "
               f"{r['win_rate']*100:>5.1f}% {r['profit_factor']:>5.2f} "
               f"{r['max_dd']*100:>6.1f}% {r['gates_passed']}/{r['gates_total']} "
               f"${r['final_equity']:>8,.0f}")

    fprint(f"\n{'=' * 100}")
    fprint("CHAIN-ONLY 2019+")
    fprint(f"{'=' * 100}")
    fprint(f"  {'Variant':<25} {'N':>5} {'Sharpe':>8} {'WR':>6} {'PnL':>9} {'%R':>5}")
    fprint(f"  {'-' * 60}")
    for vn in VARIANTS:
        r = all_results.get(vn)
        if not r or not r.get("chain_only_2019"):
            continue
        co = r["chain_only_2019"]
        fprint(f"  {vn:<25} {co['trades']:>5} {co['sharpe']:>8.2f} {co['wr']*100:>5.1f}% "
               f"${co['total_pnl']:>8.0f} {co['pct_real']:>4.0f}%")

    fprint(f"\n{'=' * 100}")
    fprint("2026 PERFORMANCE")
    fprint(f"{'=' * 100}")
    fprint(f"  {'Variant':<25} {'N':>5} {'Sharpe':>8} {'WR':>6} {'PnL':>9}")
    fprint(f"  {'-' * 55}")
    for vn in VARIANTS:
        r = all_results.get(vn)
        if not r or not r.get("year_2026"):
            continue
        y = r["year_2026"]
        fprint(f"  {vn:<25} {y['trades']:>5} {y['sharpe']:>8.2f} {y['wr']*100:>5.1f}% "
               f"${y['total_pnl']:>8.0f}")

    # ── VERDICT ──
    fprint(f"\n{'=' * 100}")
    fprint("VERDICT")
    fprint(f"{'=' * 100}")
    v8_bs = all_results.get("V8_baseline_bs")
    v8_real = all_results.get("V8_baseline_real")
    v9_core = all_results.get("V9_core")
    if v8_bs and v9_core:
        fprint(f"\n  V8 BS baseline Sharpe:     {v8_bs['sharpe']:.2f}")
    if v8_real and v9_core:
        v8_co = v8_real.get("chain_only_2019", {})
        v9_co = v9_core.get("chain_only_2019", {})
        fprint(f"  V8 real+filter chain Sharpe: {v8_co.get('sharpe', 0):.2f}")
        fprint(f"  V9 core chain Sharpe:        {v9_co.get('sharpe', 0):.2f}")
        if v9_co.get("sharpe", 0) > v8_co.get("sharpe", 0):
            fprint(f"\n  ✅ V9 IMPROVES on V8: chain-only {v9_co['sharpe']:.2f} > {v8_co['sharpe']:.2f}")
        else:
            fprint(f"\n  ❌ V9 does NOT improve: {v9_co.get('sharpe', 0):.2f} <= {v8_co.get('sharpe', 0):.2f}")
    if random_mean and v9_core:
        ml_ratio = v9_core["sharpe"] / max(random_mean, 0.01)
        fprint(f"\n  V9 ML Sharpe: {v9_core['sharpe']:.2f} vs Random: {random_mean:.2f}")
        fprint(f"  ML alpha ratio: {ml_ratio:.2f}x")

    # ── Save ──
    results_file = OUTPUT_DIR / "results.json"
    with open(results_file, "w") as f:
        serializable = {}
        for k, v in all_results.items():
            sv = {}
            for k2, v2 in v.items():
                if isinstance(v2, (np.floating, np.integer)):
                    sv[k2] = float(v2)
                elif isinstance(v2, dict):
                    sv[k2] = {k3: float(v3) if isinstance(v3, (np.floating, np.integer)) else v3
                              for k3, v3 in v2.items()}
                else:
                    sv[k2] = v2
            serializable[k] = sv
        json.dump(serializable, f, indent=2, default=str)
    fprint(f"\nResults saved to {results_file}")

    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"v9_candidate_{t0.strftime('%Y%m%d_%H%M')}"):
                for vn, r in all_results.items():
                    p = vn.replace(" ", "_")
                    mlflow.log_metric(f"{p}_sharpe", r["sharpe"])
                    mlflow.log_metric(f"{p}_wr", r["win_rate"])
                    mlflow.log_metric(f"{p}_pf", r["profit_factor"])
                    mlflow.log_metric(f"{p}_gates", r["gates_passed"])
                    co = r.get("chain_only_2019")
                    if co:
                        mlflow.log_metric(f"{p}_chain_sharpe", co["sharpe"])
                if random_mean:
                    mlflow.log_metric("random_mean_sharpe", random_mean)
                mlflow.log_artifact(str(results_file))
            fprint(f"MLflow logged to '{EXPERIMENT_NAME}'")
        except Exception as e:
            fprint(f"MLflow failed: {e}")

    elapsed = (datetime.now() - t0).total_seconds()
    fprint(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    fprint(f"\n{'=' * 100}")
    fprint("DONE — V9 Candidate Backtest v1")
    fprint(f"{'=' * 100}")


if __name__ == "__main__":
    main()
