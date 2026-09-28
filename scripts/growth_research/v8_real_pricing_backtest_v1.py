#!/usr/bin/env python3
"""
V8 Real-Pricing Backtest v1 — Definitive Strategy Performance With Actual Options Prices
=========================================================================================

THE MOST IMPORTANT EXPERIMENT: Re-runs the V8 production strategy using REAL options
chain prices from Dolt database instead of Black-Scholes approximations.

Previous findings:
  - V8 with BS pricing: Sharpe 3.23, 5/5 gates
  - Options validation: BS+haircut ≈ mid-price (median error -13.4%)
  - Bull call spreads: real entry 2.3× more expensive (multiplier 0.44×)
  - Bear put spreads: real entry 0.9× cheaper (multiplier 1.11×)
  - Estimated realistic Sharpe: ~2.10 (0.65× multiplier)

This script gives the DEFINITIVE answer by pricing EVERY trade with real chain data
where available, falling back to BS only when chain data is missing.

4 Variants (all using V8 config: weekly, 2% OTM, pairs, DTE=14):
  A: V8 with BS pricing (baseline, reproduces known Sharpe ~3.23)
  B: V8 with real mid-price (limit order scenario)
  C: V8 with real ask/bid (market order worst case)
  D: V8 with real mid + 50% haircut (conservative limit order)

Dolt chain data covers 2019-2026 for all 11 sector ETFs.
Before 2019, all variants use BS pricing (same trades, same results).

Output: output/growth_research/v8_real_pricing_backtest_v1/
MLflow experiment: v8_real_pricing_backtest_v1
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


# ── Standardized tools ──
sys.path.insert(0, "/home/jupiter/Lvl3Quant")
from research.tools.options_pricer import (
    bs_call_price,
    bs_put_price,
    price_bull_call_spread,
    price_bear_put_spread,
    estimate_iv,
    compute_atr,
    COMMISSION_RT_SPREAD,
    DEFAULT_HAIRCUT,
    RISK_FREE_RATE,
)
from research.tools.adversarial_validator import validate_trades

# ── Config ──
BASE = Path("/home/jupiter/Lvl3Quant")
CHAINS_DIR = BASE / "wheel_strategy_v1" / "data" / "cache" / "options_real" / "chains"
OUTPUT_DIR = BASE / "output" / "growth_research" / "v8_real_pricing_backtest_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX", "^VIX3M", "TLT", "SHY", "HYG", "GLD"]
CAP = 645.0
SPREAD_PCT = 3.0
TOP_K = 3
REGIME_BULL_THRESHOLD = 0.4
REGIME_BEAR_THRESHOLD = 0.2

# V8 config
DTE = 14
OTM_PCT = 0.02
REBAL_FREQ = "W-FRI"
WF_TRAIN_PERIODS = 12

# Chain matching
DTE_TOLERANCE = 5  # accept DTE within +/- 5 days of target
STRIKE_TOLERANCE = 0.03  # accept strikes within 3% of target
MIN_BID = 0.05  # minimum bid for liquidity filter

REGIME_FILE = BASE / "output" / "regime_detector_v1" / "regime_predictions_v1.npz"

# MLflow
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "v8_real_pricing_backtest_v1"

MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen(MLFLOW_URI, timeout=3)
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    MLFLOW_OK = True
    fprint(f"MLflow connected: {MLFLOW_URI}")
except Exception:
    fprint("MLflow unavailable -- results saved to disk only")


# ══════════════════════════════════════════════════════════════
# CHAIN DATA LOADING
# ══════════════════════════════════════════════════════════════

def load_all_chains():
    """Load all sector ETF chain data from Dolt parquets."""
    chains = {}
    for tk in SECTORS:
        path = CHAINS_DIR / f"{tk}.parquet"
        if not path.exists():
            fprint(f"  {tk}: chain parquet not found")
            continue
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        df["expiration"] = pd.to_datetime(df["expiration"])
        for c in ["strike", "bid", "ask", "mid", "vol", "delta"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        # Compute DTE
        df["dte"] = (df["expiration"] - df["date"]).dt.days
        chains[tk] = df
        fprint(f"  {tk}: {len(df):,} chain rows, "
               f"{df['date'].min().date()} to {df['date'].max().date()}")
    return chains


def find_chain_spread_price(chain_df, trade_date, ticker, direction, K1, K2, dte_target):
    """
    Look up real spread price from Dolt chain data.

    Args:
        chain_df: full chain DataFrame for this ticker
        trade_date: datetime of trade entry
        ticker: sector ETF ticker
        direction: 'bull' or 'bear'
        K1: lower strike
        K2: upper strike
        dte_target: target DTE

    Returns:
        dict with keys:
          - found: bool
          - spread_cost_market: ask-bid spread cost (worst case)
          - spread_cost_mid: mid-mid spread cost (limit order)
          - leg1_bid, leg1_ask, leg1_mid: near leg quotes
          - leg2_bid, leg2_ask, leg2_mid: far leg quotes
          - real_iv_near, real_iv_far: actual implied vols
          - actual_dte: DTE of matched expiration
        Or None if no match.
    """
    if chain_df is None:
        return None

    # Filter to trade date
    day_mask = chain_df["date"] == pd.Timestamp(trade_date)
    chain_day = chain_df[day_mask]
    if chain_day.empty:
        # Try nearest trading day within 2 days
        nearby = chain_df[(chain_df["date"] >= pd.Timestamp(trade_date) - pd.Timedelta(days=2)) &
                          (chain_df["date"] <= pd.Timestamp(trade_date) + pd.Timedelta(days=2))]
        if nearby.empty:
            return None
        nearest_date = nearby["date"].unique()
        nearest_date = min(nearest_date, key=lambda x: abs((x - pd.Timestamp(trade_date)).days))
        chain_day = chain_df[chain_df["date"] == nearest_date]

    # Find best expiration
    exps = chain_day[["expiration", "dte"]].drop_duplicates()
    exps["dte_dist"] = (exps["dte"] - dte_target).abs()
    valid_exps = exps[exps["dte_dist"] <= DTE_TOLERANCE]
    if valid_exps.empty:
        return None
    best_exp_row = valid_exps.loc[valid_exps["dte_dist"].idxmin()]
    best_exp = best_exp_row["expiration"]
    actual_dte = int(best_exp_row["dte"])

    chain_exp = chain_day[chain_day["expiration"] == best_exp]

    if direction == "bull":
        # Bull call spread: buy call at K1, sell call at K2
        opt_type = "c"
        near_target = K1
        far_target = K2
    else:
        # Bear put spread: buy put at K2, sell put at K1
        opt_type = "p"
        near_target = K2  # buy this one
        far_target = K1  # sell this one

    # Find near leg (the one we BUY)
    near_opts = chain_exp[chain_exp["type"] == opt_type].copy()
    if near_opts.empty:
        return None
    near_opts["dist"] = (near_opts["strike"] - near_target).abs()
    near_opts = near_opts.sort_values("dist")
    near_leg = near_opts.iloc[0]
    if near_leg["dist"] / max(near_target, 1) > STRIKE_TOLERANCE:
        return None

    # Find far leg (the one we SELL)
    far_opts = chain_exp[chain_exp["type"] == opt_type].copy()
    far_opts["dist"] = (far_opts["strike"] - far_target).abs()
    far_opts = far_opts.sort_values("dist")
    far_leg = far_opts.iloc[0]
    if far_leg["dist"] / max(far_target, 1) > STRIKE_TOLERANCE:
        return None

    # Extract prices
    near_bid = float(near_leg["bid"]) if not pd.isna(near_leg["bid"]) else 0
    near_ask = float(near_leg["ask"]) if not pd.isna(near_leg["ask"]) else 0
    near_mid = float(near_leg["mid"]) if not pd.isna(near_leg["mid"]) else (near_bid + near_ask) / 2
    far_bid = float(far_leg["bid"]) if not pd.isna(far_leg["bid"]) else 0
    far_ask = float(far_leg["ask"]) if not pd.isna(far_leg["ask"]) else 0
    far_mid = float(far_leg["mid"]) if not pd.isna(far_leg["mid"]) else (far_bid + far_ask) / 2

    near_iv = float(near_leg["vol"]) if "vol" in near_leg.index and not pd.isna(near_leg["vol"]) else None
    far_iv = float(far_leg["vol"]) if "vol" in far_leg.index and not pd.isna(far_leg["vol"]) else None

    # Spread cost = what we pay to enter
    # We BUY near leg (pay ask), SELL far leg (receive bid)
    if direction == "bull":
        # Buy K1 call at ask, sell K2 call at bid
        spread_cost_market = near_ask - far_bid  # worst case
        spread_cost_mid = near_mid - far_mid     # limit order
    else:
        # Buy K2 put at ask, sell K1 put at bid
        spread_cost_market = near_ask - far_bid   # worst case
        spread_cost_mid = near_mid - far_mid      # limit order

    # Sanity checks
    if spread_cost_market < 0:
        spread_cost_market = abs(spread_cost_market)  # credit spread edge case
    if spread_cost_mid < 0:
        spread_cost_mid = abs(spread_cost_mid)

    # Liquidity check
    fillable = near_bid >= MIN_BID and far_bid >= MIN_BID

    return {
        "found": True,
        "fillable": fillable,
        "spread_cost_market": spread_cost_market,
        "spread_cost_mid": spread_cost_mid,
        "near_bid": near_bid,
        "near_ask": near_ask,
        "near_mid": near_mid,
        "far_bid": far_bid,
        "far_ask": far_ask,
        "far_mid": far_mid,
        "real_iv_near": near_iv,
        "real_iv_far": far_iv,
        "actual_dte": actual_dte,
        "near_strike": float(near_leg["strike"]),
        "far_strike": float(far_leg["strike"]),
    }


# ══════════════════════════════════════════════════════════════
# V6 FEATURES (17 features, same as V8 production)
# ══════════════════════════════════════════════════════════════

V6_FEATURES = [
    "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "sharpe_63d", "pct_52w_high", "mom_accel",
    "pct_pos_months_12m", "sortino_63d", "calmar_1y", "up_capture",
    "trend_r2_63d", "trend_slope_63d",
    "sector_spy_beta_63d",
    "cross_sector_dispersion",
]

assert len(V6_FEATURES) == 17


# ══════════════════════════════════════════════════════════════
# DATA DOWNLOAD + FEATURE ENGINEERING
# (Reused from production_v8_candidate_v1.py)
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
    close = close.ffill()
    high = high.ffill()
    low = low.ffill()
    rename_map = {"^VIX": "VIX", "^VIX3M": "VIX3M"}
    close = close.rename(columns=rename_map)
    high = high.rename(columns=rename_map)
    low = low.rename(columns=rename_map)
    fprint(f"Data: {len(close)} days, {close.index[0].date()} to {close.index[-1].date()}")
    return close, high, low


def load_regime_predictions():
    if not REGIME_FILE.exists():
        fprint("WARNING: Regime file not found, using VIX proxy")
        return None
    data = np.load(REGIME_FILE, allow_pickle=True)
    dates = pd.to_datetime(data["dates"])
    scores = data["regime_scores"]
    regime_series = pd.Series(scores, index=dates, name="regime_score")
    regime_series = regime_series[~regime_series.index.duplicated(keep="last")]
    fprint(f"Regime predictions: {len(regime_series)} days")
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
            beta = cov[0, 1] / (cov[1, 1] + 1e-10)
            f["sector_spy_beta_63d"] = float(beta)
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


# ══════════════════════════════════════════════════════════════
# WALK-FORWARD LGBM RANKING
# ══════════════════════════════════════════════════════════════

def build_feature_records(close, high, low, rebal_dates, regime_series):
    fprint(f"  Building records: {len(rebal_dates)} dates, {len(V6_FEATURES)} features, DTE={DTE}")
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
            spy_s = spy.iloc[:idx + 1]
            legacy = compute_legacy_features(px, spy_s)
            if not legacy:
                continue
            cross_asset = compute_cross_asset_features(tk, idx, close)
            fi = min(idx + DTE, len(close) - 1)
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
        fprint(f"    Insufficient data ({len(df)} records)")
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
# STRIKE COMPUTATION
# ══════════════════════════════════════════════════════════════

def compute_strikes(S, direction):
    if direction == "bull":
        K1 = round(S * (1 + OTM_PCT), 2)
        K2 = round(K1 * (1 + SPREAD_PCT / 100), 2)
    else:
        K2 = round(S * (1 - OTM_PCT), 2)
        K1 = round(K2 * (1 - SPREAD_PCT / 100), 2)
    if K2 <= K1:
        K2 = K1 + 0.50
    return K1, K2


# ══════════════════════════════════════════════════════════════
# TRADE EXECUTION (MULTI-PRICING)
# ══════════════════════════════════════════════════════════════

PRICING_MODES = {
    "A_bs_baseline": "bs",           # BS + 15% haircut (known Sharpe ~3.23)
    "B_real_mid": "real_mid",        # Real mid-mid (limit order scenario)
    "C_real_market": "real_market",   # Real ask-bid (market order worst case)
    "D_real_mid_haircut": "real_mid_haircut",  # Real mid + 50% haircut (conservative)
}


def execute_trade(tk, dt, direction, close, atr_dict, vix_val, equity,
                  chains, pricing_mode, max_pos):
    """
    Execute a single spread trade with specified pricing mode.

    Returns dict with pnl and pricing details, or None if trade fails.
    """
    if tk not in close.columns or tk not in atr_dict:
        return None

    S = float(close[tk].loc[dt])
    di = close.index.get_loc(dt)
    ei = min(di + DTE, len(close) - 1)
    if ei <= di:
        return None

    # ATR for BS fallback
    if dt in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[dt]):
        av = float(atr_dict[tk].loc[dt])
    else:
        av = S * 0.015

    K1, K2 = compute_strikes(S, direction)

    # ── Determine entry cost based on pricing mode ──
    used_real = False
    entry_cost_ps = None  # per-share entry cost

    if pricing_mode != "bs":
        # Try to look up real chain price
        chain_df = chains.get(tk)
        if chain_df is not None:
            chain_result = find_chain_spread_price(
                chain_df, dt, tk, direction, K1, K2, DTE
            )
            if chain_result is not None and chain_result["found"]:
                if pricing_mode == "real_mid":
                    entry_cost_ps = chain_result["spread_cost_mid"]
                elif pricing_mode == "real_market":
                    entry_cost_ps = chain_result["spread_cost_market"]
                elif pricing_mode == "real_mid_haircut":
                    entry_cost_ps = chain_result["spread_cost_mid"] * 1.50  # 50% haircut
                used_real = True

                # Use actual strikes from chain for more accurate expiry calc
                if direction == "bull":
                    K1 = chain_result["near_strike"]
                    K2 = chain_result["far_strike"]
                else:
                    K1 = chain_result["far_strike"]
                    K2 = chain_result["near_strike"]

    # Fall back to BS if no real price available
    if entry_cost_ps is None:
        try:
            if direction == "bull":
                entry_cost_ps, _ = price_bull_call_spread(
                    S=S, K1=K1, K2=K2, dte=DTE, atr=av, vix=vix_val
                )
            else:
                entry_cost_ps, _ = price_bear_put_spread(
                    S=S, K1=K1, K2=K2, dte=DTE, atr=av, vix=vix_val
                )
        except Exception:
            return None

    if entry_cost_ps is None or entry_cost_ps <= 0:
        return None

    total_cost = entry_cost_ps * 100 + COMMISSION_RT_SPREAD

    if total_cost <= 0 or total_cost > max_pos or total_cost > equity * 0.40:
        return None

    # ── HOLD TO EXPIRY: intrinsic value ──
    Se = float(close[tk].iloc[ei])

    if direction == "bull":
        intrinsic = max(Se - K1, 0.0) - max(Se - K2, 0.0)
    else:
        intrinsic = max(K2 - Se, 0.0) - max(K1 - Se, 0.0)

    pnl = (intrinsic - entry_cost_ps) * 100 - COMMISSION_RT_SPREAD

    return {
        "pnl": round(pnl, 2),
        "entry_cost_ps": round(entry_cost_ps, 4),
        "total_cost": round(total_cost, 2),
        "used_real_pricing": used_real,
        "K1": K1,
        "K2": K2,
        "S_entry": round(S, 2),
        "S_exit": round(Se, 2),
        "intrinsic": round(intrinsic, 4),
    }


# ══════════════════════════════════════════════════════════════
# TRADE SIMULATION
# ══════════════════════════════════════════════════════════════

def simulate_variant(name, pricing_mode, rankings, close, atr_dict, chains):
    """Simulate all trades for one pricing variant."""
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

        # VIX pairs logic
        if cv < 20.0:
            trade_mode = "pairs"
        else:
            trade_mode = "bull_only"

        ranked_desc = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ranked_asc = sorted(scores.items(), key=lambda x: x[1])

        bull_picks = [t for t, _ in ranked_desc[:TOP_K]]
        bear_picks = [t for t, _ in ranked_asc[:TOP_K]] if trade_mode == "pairs" else []

        if trade_mode == "pairs":
            max_pos = min(100, equity / 6)
        else:
            max_pos = min(200, equity / 3)

        if max_pos < 30:
            continue

        # Execute bull leg
        for tk in bull_picks:
            result = execute_trade(
                tk, dt, "bull", close, atr_dict, cv, equity,
                chains, pricing_mode, max_pos
            )
            if result is not None:
                equity += result["pnl"]
                if result["used_real_pricing"]:
                    real_count += 1
                else:
                    bs_count += 1

                di = close.index.get_loc(dt)
                ei = min(di + DTE, len(close) - 1)
                sv = float(spy.loc[dt])
                se = float(spy.iloc[ei]) if ei < len(spy) else sv

                trades.append({
                    **result,
                    "entry_date": str(dt.date()),
                    "exit_date": str(close.index[ei].date()),
                    "ticker": tk,
                    "regime": "bull" if se >= sv else "bear",
                    "direction": "bull",
                    "vix": round(cv, 1),
                    "win": result["pnl"] > 0,
                    "trade_mode": trade_mode,
                })

        # Execute bear leg
        for tk in bear_picks:
            result = execute_trade(
                tk, dt, "bear", close, atr_dict, cv, equity,
                chains, pricing_mode, max_pos
            )
            if result is not None:
                equity += result["pnl"]
                if result["used_real_pricing"]:
                    real_count += 1
                else:
                    bs_count += 1

                di = close.index.get_loc(dt)
                ei = min(di + DTE, len(close) - 1)
                sv = float(spy.loc[dt])
                se = float(spy.iloc[ei]) if ei < len(spy) else sv

                trades.append({
                    **result,
                    "entry_date": str(dt.date()),
                    "exit_date": str(close.index[ei].date()),
                    "ticker": tk,
                    "regime": "bull" if se >= sv else "bear",
                    "direction": "bear",
                    "vix": round(cv, 1),
                    "win": result["pnl"] > 0,
                    "trade_mode": trade_mode,
                })

    return trades, equity, real_count, bs_count


# ══════════════════════════════════════════════════════════════
# ANALYSIS
# ══════════════════════════════════════════════════════════════

def analyze_real_vs_bs(trades):
    """Compare trades using real pricing vs BS fallback."""
    real_trades = [t for t in trades if t["used_real_pricing"]]
    bs_trades = [t for t in trades if not t["used_real_pricing"]]

    fprint(f"\n  REAL vs BS PRICING BREAKDOWN:")
    fprint(f"  {'Source':<15} {'Trades':>7} {'WR':>7} {'AvgPnL':>9} {'TotalPnL':>10}")
    fprint(f"  {'-'*50}")

    for label, subset in [("Real chain", real_trades), ("BS fallback", bs_trades)]:
        if not subset:
            fprint(f"  {label:<15} {'(none)':>7}")
            continue
        pnls = [t["pnl"] for t in subset]
        wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
        avg = np.mean(pnls)
        tot = sum(pnls)
        fprint(f"  {label:<15} {len(subset):>7} {wr:>6.1f}% ${avg:>8.2f} ${tot:>9.0f}")

    # Time period breakdown
    if real_trades:
        real_dates = [t["entry_date"] for t in real_trades]
        fprint(f"\n  Real pricing date range: {min(real_dates)} to {max(real_dates)}")
        fprint(f"  Real pricing coverage: {len(real_trades)}/{len(trades)} trades "
               f"({len(real_trades)/len(trades)*100:.0f}%)")

    # Entry cost comparison (real-priced trades only)
    if real_trades:
        entry_costs = [t["entry_cost_ps"] for t in real_trades]
        fprint(f"\n  Real entry cost stats (per share):")
        fprint(f"    Mean: ${np.mean(entry_costs):.4f}")
        fprint(f"    Median: ${np.median(entry_costs):.4f}")
        fprint(f"    P10: ${np.percentile(entry_costs, 10):.4f}")
        fprint(f"    P90: ${np.percentile(entry_costs, 90):.4f}")


def regime_stratification(trades):
    fprint(f"\n  REGIME STRATIFICATION:")
    fprint(f"  {'Regime':<10} {'Trades':>7} {'WR':>7} {'AvgPnL':>9} {'TotalPnL':>10} {'Sharpe':>8}")
    fprint(f"  {'-'*55}")
    for regime in ["bull", "bear"]:
        rt = [t for t in trades if t["regime"] == regime]
        if not rt:
            continue
        pnls = [t["pnl"] for t in rt]
        wr = sum(1 for p in pnls if p > 0) / len(pnls)
        avg = np.mean(pnls)
        tot = sum(pnls)
        sh = float(np.mean(pnls) / (np.std(pnls) + 1e-10) * np.sqrt(52))
        fprint(f"  {regime:<10} {len(rt):>7} {wr:>6.1%} ${avg:>8.2f} ${tot:>9.0f} {sh:>8.2f}")

    bull_trades = [t for t in trades if t["regime"] == "bull"]
    bear_trades = [t for t in trades if t["regime"] == "bear"]
    if bull_trades and bear_trades:
        bull_sh = float(np.mean([t["pnl"] for t in bull_trades]) /
                        (np.std([t["pnl"] for t in bull_trades]) + 1e-10) * np.sqrt(52))
        bear_sh = float(np.mean([t["pnl"] for t in bear_trades]) /
                        (np.std([t["pnl"] for t in bear_trades]) + 1e-10) * np.sqrt(52))
        imbalance = abs(bull_sh - bear_sh) / max(abs(bull_sh), abs(bear_sh), 0.01)
        status = "FAIL" if imbalance > 0.50 else "PASS"
        fprint(f"\n  Regime imbalance: {imbalance:.2f} [{status}]")


def pnl_by_side(trades):
    fprint(f"\n  PnL BY SIDE:")
    fprint(f"  {'Side':<10} {'Trades':>7} {'WR':>7} {'AvgPnL':>9} {'TotalPnL':>10}")
    fprint(f"  {'-'*45}")
    for side in ["bull", "bear"]:
        st = [t for t in trades if t["direction"] == side]
        if not st:
            continue
        pnls = [t["pnl"] for t in st]
        wr = sum(1 for p in pnls if p > 0) / len(pnls)
        avg = np.mean(pnls)
        tot = sum(pnls)
        fprint(f"  {side:<10} {len(st):>7} {wr:>6.1%} ${avg:>8.2f} ${tot:>9.0f}")


def yearly_breakdown(trades):
    fprint(f"\n  YEARLY BREAKDOWN:")
    fprint(f"  {'Year':<6} {'Trades':>7} {'WR':>7} {'PnL':>10} {'Sharpe':>8} {'%Real':>7}")
    fprint(f"  {'-'*50}")
    by_year = {}
    for t in trades:
        yr = t["entry_date"][:4]
        by_year.setdefault(yr, []).append(t)
    for yr in sorted(by_year.keys()):
        yt = by_year[yr]
        pnls = [t["pnl"] for t in yt]
        wr = sum(1 for p in pnls if p > 0) / len(pnls)
        tot = sum(pnls)
        sh = float(np.mean(pnls) / (np.std(pnls) + 1e-10) * np.sqrt(52))
        pct_real = sum(1 for t in yt if t["used_real_pricing"]) / len(yt) * 100
        fprint(f"  {yr:<6} {len(yt):>7} {wr:>6.1%} ${tot:>9.0f} {sh:>8.2f} {pct_real:>6.0f}%")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    t0 = datetime.now()
    fprint("=" * 100)
    fprint(f"V8 REAL-PRICING BACKTEST v1 — {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 100)
    fprint(f"V8 Config: DTE={DTE} | OTM={OTM_PCT*100:.0f}% | Spread={SPREAD_PCT:.0f}% | "
           f"Rebal={REBAL_FREQ} | Pairs VIX<20 | 17 features")
    fprint(f"Capital: ${CAP:.0f} | Commission: ${COMMISSION_RT_SPREAD:.2f} | Hold to expiry")
    fprint(f"\n4 pricing variants:")
    fprint(f"  A: BS + 15% haircut (baseline, should reproduce Sharpe ~3.23)")
    fprint(f"  B: Real mid-price from Dolt chains (limit order)")
    fprint(f"  C: Real ask/bid from Dolt chains (market order worst case)")
    fprint(f"  D: Real mid + 50% haircut (conservative limit order)")
    fprint(f"\nDolt chain data: 2019-2026. Before 2019: all use BS (identical results).")
    fprint()

    # 1. Load chain data
    fprint("Loading Dolt chain data...")
    chains = load_all_chains()
    fprint(f"  {len(chains)} sectors with chain data\n")

    # 2. Download price data
    close, high, low = download_data()

    # 3. Load regime
    regime_series = load_regime_predictions()

    # 4. ATR
    atr_dict = compute_atr_series(high, low, close)

    # 5. Build LGBM rankings (shared across all variants)
    fprint(f"\n{'=' * 80}")
    fprint("BUILDING LGBM RANKINGS")
    fprint(f"{'=' * 80}")

    rebal_dates = pd.DatetimeIndex(
        close.index.to_series().resample(REBAL_FREQ).last().dropna().values
    )
    fprint(f"  Rebalance dates: {len(rebal_dates)} ({rebal_dates[0].date()} to {rebal_dates[-1].date()})")

    records = build_feature_records(close, high, low, rebal_dates, regime_series)
    rankings = walk_forward_lgbm_rank(records)

    if not rankings:
        fprint("ERROR: No rankings generated. Aborting.")
        return

    # 6. Simulate all 4 variants
    fprint(f"\n{'=' * 100}")
    fprint("SIMULATING ALL PRICING VARIANTS")
    fprint(f"{'=' * 100}")

    all_results = {}
    all_trades = {}

    spy_close = close["SPY"]

    for vname, pmode in PRICING_MODES.items():
        fprint(f"\n{'~' * 90}")
        fprint(f"VARIANT {vname} (pricing: {pmode})")
        fprint(f"{'~' * 90}")

        trades, final_eq, real_count, bs_count = simulate_variant(
            vname, pmode, rankings, close, atr_dict, chains
        )

        if not trades or len(trades) < 10:
            fprint(f"  Only {len(trades) if trades else 0} trades, skipping")
            continue

        all_trades[vname] = trades
        fprint(f"\n  Total trades: {len(trades)}")
        fprint(f"  Real-priced: {real_count} ({real_count/(real_count+bs_count)*100:.0f}%)")
        fprint(f"  BS-fallback: {bs_count} ({bs_count/(real_count+bs_count)*100:.0f}%)")
        fprint(f"  Final equity: ${final_eq:,.0f} (from ${CAP:.0f})")

        # 5-gate validation
        fprint(f"\n  5-GATE ADVERSARIAL VALIDATION:")
        result = validate_trades(
            trades, initial_capital=CAP,
            spy_prices=spy_close,
            strategy_name=vname,
        )
        result.print_summary()

        # Analysis
        analyze_real_vs_bs(trades)
        regime_stratification(trades)
        pnl_by_side(trades)
        yearly_breakdown(trades)

        rd = result.to_dict()
        all_results[vname] = {
            **rd,
            "pricing_mode": pmode,
            "real_priced_trades": real_count,
            "bs_fallback_trades": bs_count,
            "real_pct": round(real_count / max(real_count + bs_count, 1) * 100, 1),
        }

    # ── COMPARISON TABLE ──
    fprint(f"\n{'=' * 120}")
    fprint("COMPARISON — ALL PRICING VARIANTS")
    fprint(f"{'=' * 120}")
    fprint(f"{'Variant':<25} {'Trades':>6} {'Sharpe':>7} {'Sortino':>8} {'WR':>6} "
           f"{'PF':>6} {'MaxDD':>7} {'Gates':>6} {'Final$':>9} {'%Real':>6}")
    fprint("-" * 100)

    for vname in PRICING_MODES.keys():
        r = all_results.get(vname)
        if not r:
            fprint(f"  {vname:<25} -- NO DATA --")
            continue
        fprint(f"  {vname:<25} {r['n_trades']:>5} {r['sharpe']:>7.2f} {r['sortino']:>8.2f} "
               f"{r['win_rate']*100:>5.1f}% {r['profit_factor']:>5.2f} "
               f"{r['max_dd']*100:>6.1f}% {r['gates_passed']}/{r['gates_total']} "
               f"${r['final_equity']:>8,.0f} {r['real_pct']:>5.0f}%")

    # ── KEY FINDING ──
    fprint(f"\n{'=' * 100}")
    fprint("KEY FINDING: REAL vs BS PRICING IMPACT")
    fprint(f"{'=' * 100}")

    bs_result = all_results.get("A_bs_baseline")
    mid_result = all_results.get("B_real_mid")
    market_result = all_results.get("C_real_market")

    if bs_result and mid_result:
        bs_sh = bs_result["sharpe"]
        mid_sh = mid_result["sharpe"]
        ratio = mid_sh / max(bs_sh, 0.01)
        fprint(f"\n  BS baseline Sharpe:     {bs_sh:.2f}")
        fprint(f"  Real mid-price Sharpe:  {mid_sh:.2f}  ({ratio:.2f}x)")
        if market_result:
            mkt_sh = market_result["sharpe"]
            mkt_ratio = mkt_sh / max(bs_sh, 0.01)
            fprint(f"  Real market Sharpe:     {mkt_sh:.2f}  ({mkt_ratio:.2f}x)")
        fprint(f"\n  INTERPRETATION:")
        if mid_sh >= 2.0:
            fprint(f"    Strategy is VIABLE with limit orders (Sharpe {mid_sh:.2f} >= 2.0)")
        elif mid_sh >= 1.5:
            fprint(f"    Strategy is MARGINAL (Sharpe {mid_sh:.2f}), needs optimization")
        else:
            fprint(f"    Strategy FAILS with real pricing (Sharpe {mid_sh:.2f} < 1.5)")

        fprint(f"\n  BS overestimates Sharpe by {(1 - ratio)*100:.0f}%")
        fprint(f"  Previous estimate: 0.65x multiplier (from validation study)")
        fprint(f"  Actual result:     {ratio:.2f}x multiplier")

    # ── CHAIN-ONLY ANALYSIS (2019+ only) ──
    fprint(f"\n{'=' * 100}")
    fprint("CHAIN-ONLY PERIOD ANALYSIS (2019+ where real data exists)")
    fprint(f"{'=' * 100}")

    for vname in PRICING_MODES.keys():
        trades = all_trades.get(vname, [])
        chain_trades = [t for t in trades if t["entry_date"] >= "2019-01-01"]
        if not chain_trades or len(chain_trades) < 10:
            continue
        pnls = [t["pnl"] for t in chain_trades]
        wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
        sh = float(np.mean(pnls) / (np.std(pnls) + 1e-10) * np.sqrt(52))
        eq = CAP + sum(pnls)
        real_pct = sum(1 for t in chain_trades if t["used_real_pricing"]) / len(chain_trades) * 100
        fprint(f"  {vname:<25} {len(chain_trades):>5} trades | Sharpe {sh:.2f} | "
               f"WR {wr:.1f}% | ${eq:,.0f} final | {real_pct:.0f}% real-priced")

    # ── Save results ──
    results_data = {
        "experiment": EXPERIMENT_NAME,
        "timestamp": t0.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "capital": CAP,
            "dte": DTE,
            "otm_pct": OTM_PCT,
            "spread_pct": SPREAD_PCT,
            "rebal_freq": REBAL_FREQ,
            "n_features": len(V6_FEATURES),
            "commission": COMMISSION_RT_SPREAD,
            "hold_to_expiry": True,
            "lgbm_n_estimators": 100,
            "lgbm_max_depth": 4,
            "lgbm_lr": 0.05,
            "wf_train_periods": WF_TRAIN_PERIODS,
            "chain_data_source": "Dolt options database (2019-2026)",
        },
        "variants": all_results,
    }

    results_path = OUTPUT_DIR / "v8_real_pricing_v1_results.json"
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2, default=str)
    fprint(f"\nResults saved to {results_path}")

    for vname, trades in all_trades.items():
        tpath = OUTPUT_DIR / f"{vname}_trades.json"
        with open(tpath, "w") as f:
            json.dump(trades, f, indent=2, default=str)

    # MLflow
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"real_pricing_{t0.strftime('%Y%m%d_%H%M')}"):
                for vname, r in all_results.items():
                    prefix = vname.split("_")[0]
                    for k in ["sharpe", "sortino", "win_rate", "profit_factor",
                              "max_dd", "n_trades", "gates_passed", "final_equity"]:
                        if k in r:
                            mlflow.log_metric(f"{prefix}_{k}", r[k])
                    mlflow.log_metric(f"{prefix}_real_pct", r.get("real_pct", 0))

                mlflow.log_params({
                    "capital": CAP,
                    "dte": DTE,
                    "otm_pct": OTM_PCT,
                    "spread_pct": SPREAD_PCT,
                    "rebal_freq": REBAL_FREQ,
                    "commission": COMMISSION_RT_SPREAD,
                    "n_sectors": len(SECTORS),
                    "chain_coverage": "2019-2026",
                })

                mlflow.log_artifact(str(results_path))
            fprint(f"MLflow run logged to experiment '{EXPERIMENT_NAME}'")
        except Exception as e:
            fprint(f"MLflow logging failed: {e}")

    elapsed = (datetime.now() - t0).total_seconds()
    fprint(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f}m)")

    fprint(f"\n{'=' * 100}")
    fprint("DONE — V8 Real-Pricing Backtest v1")
    fprint(f"{'=' * 100}")


if __name__ == "__main__":
    main()
