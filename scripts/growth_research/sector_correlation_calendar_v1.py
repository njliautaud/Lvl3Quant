#!/usr/bin/env python3
"""
Sector Correlation & Calendar Effects — V9.1 Options Framework
===============================================================

Tests whether filtering trades by correlation regime, calendar timing,
or VIX-adaptive position count can improve risk-adjusted returns.

Variants:
  A: Baseline V9.1 — 6 positions (top 3 + bottom 3), DTE=28, weekly rebalance
  B: Correlation gate — Skip weeks when avg sector pairwise corr (21d) > 0.7
  C: Calendar effect — Only enter in first week of each month
  D: VIX-reduced positions — 4 positions (top 2 + bottom 2) when VIX > 25

All use V9.1 config: DTE=28, 3% OTM, adaptive width max($3, 3%),
cost/width < 50%, real chain pricing, $645 capital, $2.60 commission, 15% haircut.

LGBM walk-forward: sliding 52-week train, predict next period, 2008-2026.
Universe: 11 sector ETFs.

Output: output/growth_research/sector_correlation_calendar_v1/
MLflow experiment: sector_correlation_calendar_v1
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

# Detect environment — Jupiter vs Neptune
_JUPITER_BASE = Path("/home/jupiter/Lvl3Quant")
_NEPTUNE_BASE = Path("/home/nick/Lvl3Quant")

if _NEPTUNE_BASE.exists():
    BASE = _NEPTUNE_BASE
    fprint(f"Running on Neptune: {BASE}")
else:
    BASE = _JUPITER_BASE
    fprint(f"Running on Jupiter: {BASE}")

sys.path.insert(0, str(BASE))
from research.tools.options_pricer import (
    price_bull_call_spread, price_bear_put_spread, COMMISSION_RT_SPREAD,
)
from research.tools.adversarial_validator import validate_trades

CHAINS_DIR = BASE / "wheel_strategy_v1" / "data" / "cache" / "options_real" / "chains"
OUTPUT_DIR = BASE / "output" / "growth_research" / "sector_correlation_calendar_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX", "^VIX3M", "TLT", "SHY", "HYG", "GLD"]
CAP = 645.0
TOP_K = 3
DTE = 28
REGIME_BULL_THRESHOLD = 0.4
WF_TRAIN_PERIODS = 52  # 52-week sliding window

COST_WIDTH_MAX = 0.50
DTE_TOLERANCE = 5
STRIKE_TOLERANCE = 0.03

# Variant-specific constants
CORR_THRESHOLD = 0.70       # Variant B: skip if avg pairwise corr > this
CORR_LOOKBACK = 21           # Variant B: trailing days for correlation
VIX_HIGH_THRESHOLD = 25.0    # Variant D: reduce positions when VIX > this
VIX_REDUCED_K = 2            # Variant D: top/bottom K in high-VIX

REGIME_FILE = BASE / "output" / "regime_detector_v1" / "regime_predictions_v1.npz"
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "sector_correlation_calendar_v1"

MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen(MLFLOW_URI, timeout=3)
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    MLFLOW_OK = True
    fprint(f"MLflow connected: {MLFLOW_URI}")
except Exception:
    fprint("MLflow unavailable — will skip logging")

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
# CORRELATION COMPUTATION (Variant B)
# ══════════════════════════════════════════════════════════════

def compute_sector_correlation_series(close):
    """Compute trailing average pairwise correlation of sector ETFs.

    Returns a Series indexed by date with the mean pairwise correlation
    over the trailing CORR_LOOKBACK days.
    """
    sector_cols = [c for c in SECTORS if c in close.columns]
    if len(sector_cols) < 4:
        fprint("  WARNING: fewer than 4 sectors available for correlation")
        return pd.Series(dtype=float)

    sector_rets = close[sector_cols].pct_change().dropna()

    # Rolling pairwise correlation — compute mean of upper triangle
    corr_means = []
    corr_dates = []

    for i in range(CORR_LOOKBACK, len(sector_rets)):
        window = sector_rets.iloc[i - CORR_LOOKBACK:i]
        corr_matrix = window.corr()
        # Extract upper triangle (excluding diagonal)
        n = len(sector_cols)
        upper_vals = []
        for r in range(n):
            for c in range(r + 1, n):
                val = corr_matrix.iloc[r, c]
                if not pd.isna(val):
                    upper_vals.append(val)
        if upper_vals:
            corr_means.append(np.mean(upper_vals))
            corr_dates.append(sector_rets.index[i])

    corr_series = pd.Series(corr_means, index=pd.DatetimeIndex(corr_dates), name="avg_pairwise_corr")
    fprint(f"  Correlation series: {len(corr_series)} dates, "
           f"mean={corr_series.mean():.3f}, std={corr_series.std():.3f}")
    fprint(f"  % above {CORR_THRESHOLD}: {(corr_series > CORR_THRESHOLD).mean()*100:.1f}%")
    return corr_series


# ══════════════════════════════════════════════════════════════
# LGBM WALK-FORWARD RANKING
# ══════════════════════════════════════════════════════════════

def build_feature_records(close, high, low, rebal_dates, regime_series):
    """Build feature records for all rebalance dates."""
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
    """Walk-forward LGBM ranking with 52-week sliding window."""
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
            preds = m.predict(Xe)
            test_df["score"] = preds
            rankings[test_date] = dict(zip(test_df["ticker"], test_df["score"]))
        except Exception:
            continue
    fprint(f"    {len(rankings)} ranking dates")
    return rankings


# ══════════════════════════════════════════════════════════════
# STRIKES + EXECUTION
# ══════════════════════════════════════════════════════════════

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
                  chains, max_pos):
    """Execute a single spread trade, return result dict or None."""
    if tk not in close.columns or tk not in atr_dict:
        return None
    S = float(close[tk].loc[dt])
    di = close.index.get_loc(dt)
    ei = min(di + DTE, len(close) - 1)
    if ei <= di:
        return None

    av = float(atr_dict[tk].loc[dt]) if dt in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[dt]) else S * 0.015
    K1, K2 = compute_strikes(S, direction)

    used_real = False
    entry_cost_ps = None

    chain_df = chains.get(tk)
    if chain_df is not None:
        result = find_chain_spread_price(chain_df, dt, direction, K1, K2, DTE)
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
                entry_cost_ps, _ = price_bull_call_spread(S=S, K1=K1, K2=K2, dte=DTE, atr=av, vix=vix_val)
            else:
                entry_cost_ps, _ = price_bear_put_spread(S=S, K1=K1, K2=K2, dte=DTE, atr=av, vix=vix_val)
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
        "cost_width_ratio": round(cwr, 3),
    }


# ══════════════════════════════════════════════════════════════
# REBALANCE DATE GENERATION
# ══════════════════════════════════════════════════════════════

def get_weekly_fridays(close):
    """Every Friday — baseline."""
    return pd.DatetimeIndex(
        close.index.to_series().resample("W-FRI").last().dropna().values
    )


def get_first_week_of_month_fridays(close):
    """First Friday of each month only — Variant C calendar effect."""
    weekly = get_weekly_fridays(close)
    monthly = []
    seen_months = set()
    for dt in weekly:
        key = (dt.year, dt.month)
        if key not in seen_months:
            seen_months.add(key)
            monthly.append(dt)
    return pd.DatetimeIndex(monthly)


def filter_by_correlation(rebal_dates, corr_series, threshold=CORR_THRESHOLD):
    """Filter out rebalance dates where avg sector correlation > threshold (Variant B).

    Returns filtered DatetimeIndex and count of skipped dates.
    """
    if corr_series is None or corr_series.empty:
        return rebal_dates, 0

    kept = []
    skipped = 0
    for dt in rebal_dates:
        # Find most recent correlation value
        if dt in corr_series.index:
            corr_val = corr_series.loc[dt]
        else:
            # Find nearest prior date
            prior = corr_series.index[corr_series.index <= dt]
            if len(prior) == 0:
                kept.append(dt)
                continue
            corr_val = corr_series.loc[prior[-1]]

        if corr_val > threshold:
            skipped += 1
        else:
            kept.append(dt)

    return pd.DatetimeIndex(kept), skipped


def get_adaptive_top_k(vix_val, default_k=TOP_K):
    """Variant D: reduce positions in high-VIX environments."""
    if vix_val > VIX_HIGH_THRESHOLD:
        return VIX_REDUCED_K
    return default_k


# ══════════════════════════════════════════════════════════════
# SIMULATION ENGINE
# ══════════════════════════════════════════════════════════════

def simulate_variant(rankings, close, atr_dict, chains,
                     rebal_dates, variant_name="", adaptive_k=False):
    """Run backtest simulation for a variant.

    Args:
        rankings: dict {date: {ticker: score}}
        close: price DataFrame
        atr_dict: ATR series per ticker
        chains: chain data dict
        rebal_dates: DatetimeIndex of rebalance dates
        variant_name: for logging
        adaptive_k: if True, use VIX-adaptive position count (Variant D)
    """
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None

    equity = CAP
    trades = []
    real_count = 0
    bs_count = 0
    turnover_events = 0
    skipped_corr = 0
    reduced_pos_count = 0  # Track how often we reduced positions

    for dt in sorted(rebal_dates):
        if dt not in spy.index:
            continue
        # Find closest ranking date
        ranking_date = None
        for rd in sorted(rankings.keys()):
            if rd <= dt:
                ranking_date = rd
        if ranking_date is None:
            continue

        scores = rankings[ranking_date]
        if not scores or len(scores) < 6:
            continue

        cv = float(vix.loc[dt]) if vix is not None and dt in vix.index else 20.0

        # Determine position count for this rebalance
        if adaptive_k:
            k = get_adaptive_top_k(cv)
            if k < TOP_K:
                reduced_pos_count += 1
        else:
            k = TOP_K

        ranked_desc = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ranked_asc = sorted(scores.items(), key=lambda x: x[1])
        bull_picks = [t for t, _ in ranked_desc[:k]]
        bear_picks = [t for t, _ in ranked_asc[:k]]

        turnover_events += 1
        n_positions = len(bull_picks) + len(bear_picks)

        for direction, picks in [("bull", bull_picks), ("bear", bear_picks)]:
            for tk in picks:
                max_pos = min(CAP / n_positions * 2, equity * 0.40)
                if max_pos < 20:
                    continue

                result = execute_trade(
                    tk, dt, direction, close, atr_dict, cv, equity,
                    chains, max_pos
                )
                if result is not None:
                    equity += result["pnl"]
                    real_count += 1 if result["used_real_pricing"] else 0
                    bs_count += 0 if result["used_real_pricing"] else 1
                    di = close.index.get_loc(dt)
                    ei = min(di + DTE, len(close) - 1)
                    sv = float(spy.loc[dt])
                    se = float(spy.iloc[ei]) if ei < len(spy) else sv
                    trades.append({
                        **result, "entry_date": str(dt.date()),
                        "exit_date": str(close.index[ei].date()),
                        "ticker": tk, "regime": "bull" if se >= sv else "bear",
                        "direction": direction, "vix": round(cv, 1),
                        "win": result["pnl"] > 0,
                        "n_positions": n_positions,
                    })

    return trades, equity, real_count, bs_count, turnover_events, reduced_pos_count


def monte_carlo_ci(trades, n_bootstrap=1000, seed=42):
    if len(trades) < 10:
        return None
    pnls = np.array([t["pnl"] for t in trades])
    rng = np.random.RandomState(seed)
    sharpes = []
    for _ in range(n_bootstrap):
        sample = rng.choice(pnls, size=len(pnls), replace=True)
        sh = float(np.mean(sample) / (np.std(sample) + 1e-10) * np.sqrt(52))
        sharpes.append(sh)
    sharpes = np.array(sharpes)
    return {
        "mean": float(np.mean(sharpes)),
        "ci_95_low": float(np.percentile(sharpes, 2.5)),
        "ci_95_high": float(np.percentile(sharpes, 97.5)),
    }


def compute_turnover_rate(trades, turnover_events, total_weeks):
    """Compute annualized turnover rate."""
    if total_weeks <= 0:
        return 0.0
    rebal_per_year = turnover_events / (total_weeks / 52)
    trades_per_rebal = len(trades) / max(turnover_events, 1)
    return round(rebal_per_year * trades_per_rebal, 1)


def compute_pct_time_in_market(trades, close):
    """Compute percentage of trading days with at least one open position."""
    if not trades:
        return 0.0
    trade_days = set()
    for t in trades:
        entry = pd.Timestamp(t["entry_date"])
        exit_dt = pd.Timestamp(t["exit_date"])
        mask = (close.index >= entry) & (close.index <= exit_dt)
        for d in close.index[mask]:
            trade_days.add(d)
    total_days = len(close.index)
    return len(trade_days) / max(total_days, 1)


def compute_calmar_ratio(trades, initial_capital=CAP):
    """Compute Calmar ratio = CAGR / MaxDD."""
    if not trades:
        return 0.0
    equity = [initial_capital]
    for t in trades:
        equity.append(equity[-1] + t["pnl"])
    equity = np.array(equity)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = abs(dd.min())
    if max_dd < 1e-10:
        return 0.0

    # Approximate CAGR from first to last trade date
    dates_str = [t["entry_date"] for t in trades]
    first_date = pd.Timestamp(min(dates_str))
    last_date = pd.Timestamp(max(dates_str))
    years = (last_date - first_date).days / 365.25
    if years < 0.5:
        return 0.0
    total_ret = equity[-1] / equity[0]
    cagr = total_ret ** (1 / years) - 1
    return cagr / max_dd


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    t0 = datetime.now()
    fprint("=" * 100)
    fprint(f"SECTOR CORRELATION & CALENDAR EFFECTS v1 — {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 100)
    fprint(f"Capital: ${CAP:.0f} | DTE: {DTE} | Commission: ${COMMISSION_RT_SPREAD:.2f}")
    fprint(f"Universe: {len(SECTORS)} sector ETFs | Top/Bottom K: {TOP_K}")
    fprint(f"Walk-forward: {WF_TRAIN_PERIODS}-week sliding window")
    fprint(f"Correlation threshold: {CORR_THRESHOLD} | Corr lookback: {CORR_LOOKBACK}d")
    fprint(f"VIX high threshold: {VIX_HIGH_THRESHOLD} | Reduced K: {VIX_REDUCED_K}")
    fprint()

    fprint("Loading chains...")
    chains = load_all_chains()

    fprint("\nDownloading price data...")
    close, high, low = download_data()
    regime_series = load_regime_predictions()
    atr_dict = compute_atr_series(high, low, close)

    # Generate rebalance schedules
    weekly_dates = get_weekly_fridays(close)
    monthly_first_week = get_first_week_of_month_fridays(close)

    fprint(f"\nRebalance schedules:")
    fprint(f"  Weekly:         {len(weekly_dates)} dates")
    fprint(f"  Monthly (1st):  {len(monthly_first_week)} dates")

    # Compute sector correlation series for Variant B
    fprint(f"\nComputing sector pairwise correlation series...")
    corr_series = compute_sector_correlation_series(close)

    # Filter weekly dates by correlation for Variant B
    corr_filtered_dates, n_corr_skipped = filter_by_correlation(weekly_dates, corr_series)
    fprint(f"  Correlation-filtered: {len(corr_filtered_dates)} dates "
           f"(skipped {n_corr_skipped} high-correlation weeks)")

    # Build features and LGBM rankings (always on weekly cadence for training)
    fprint(f"\n{'=' * 80}")
    fprint("BUILDING LGBM RANKINGS (52-week walk-forward)")
    fprint(f"{'=' * 80}")
    records = build_feature_records(close, high, low, weekly_dates, regime_series)
    rankings = walk_forward_lgbm_rank(records)

    if len(rankings) < 20:
        fprint(f"ERROR: Only {len(rankings)} ranking dates — insufficient data")
        return

    spy_close = close["SPY"]
    total_weeks = len(weekly_dates)

    # Define variants
    VARIANTS = {
        "A_baseline_weekly": {
            "rebal_dates": weekly_dates,
            "adaptive_k": False,
            "desc": "Baseline V9.1 — 6 positions, weekly rebalance",
        },
        "B_correlation_gate": {
            "rebal_dates": corr_filtered_dates,
            "adaptive_k": False,
            "desc": f"Correlation gate — skip weeks with avg corr > {CORR_THRESHOLD}",
        },
        "C_calendar_first_week": {
            "rebal_dates": monthly_first_week,
            "adaptive_k": False,
            "desc": "Calendar effect — first week of month only",
        },
        "D_vix_reduced_positions": {
            "rebal_dates": weekly_dates,
            "adaptive_k": True,
            "desc": f"VIX-reduced — 4 positions when VIX > {VIX_HIGH_THRESHOLD}",
        },
    }

    fprint(f"\n{len(VARIANTS)} variants to test:")
    for vn, vc in VARIANTS.items():
        fprint(f"  {vn}: {vc['desc']} ({len(vc['rebal_dates'])} rebalance dates)")

    # Simulate all variants
    all_results = {}
    all_trades = {}

    for vname, vcfg in VARIANTS.items():
        fprint(f"\n{'~' * 90}")
        fprint(f"VARIANT {vname}: {vcfg['desc']}")
        fprint(f"{'~' * 90}")

        trades, final_eq, real_count, bs_count, turnover_events, reduced_pos_count = simulate_variant(
            rankings, close, atr_dict, chains,
            vcfg["rebal_dates"], vname, vcfg["adaptive_k"],
        )

        if not trades or len(trades) < 5:
            fprint(f"  Only {len(trades) if trades else 0} trades, skipping")
            continue

        all_trades[vname] = trades
        total = real_count + bs_count
        turnover = compute_turnover_rate(trades, turnover_events, total_weeks)
        pct_in_market = compute_pct_time_in_market(trades, close)
        calmar = compute_calmar_ratio(trades)

        fprint(f"\n  Trades: {len(trades)} | Real: {real_count} ({real_count/max(total,1)*100:.0f}%) | "
               f"Final: ${final_eq:,.0f} | Turnover: {turnover:.0f} trades/yr | "
               f"Rebalances: {turnover_events} | % in market: {pct_in_market*100:.1f}%")
        if vcfg["adaptive_k"]:
            fprint(f"  Reduced-position weeks (VIX > {VIX_HIGH_THRESHOLD}): {reduced_pos_count}")

        result = validate_trades(trades, initial_capital=CAP, spy_prices=spy_close, strategy_name=vname)
        result.print_summary()

        # Side analysis
        for side in ["bull", "bear"]:
            st = [t for t in trades if t["direction"] == side]
            if st:
                pnls = [t["pnl"] for t in st]
                wr = sum(1 for p in pnls if p > 0) / len(pnls)
                fprint(f"  {side}: {len(st)} trades, WR {wr:.1%}, PnL ${sum(pnls):,.0f}")

        # Real-only analysis
        real_trades = [t for t in trades if t.get("used_real_pricing")]
        real_analysis = None
        if len(real_trades) >= 10:
            real_pnls = [t["pnl"] for t in real_trades]
            real_analysis = {
                "n_trades": len(real_trades),
                "sharpe": float(np.mean(real_pnls) / (np.std(real_pnls) + 1e-10) * np.sqrt(52)),
                "wr": sum(1 for p in real_pnls if p > 0) / len(real_pnls),
                "total_pnl": sum(real_pnls),
            }

        mc = monte_carlo_ci(trades)

        rd = result.to_dict()
        all_results[vname] = {
            **rd, "final_equity": round(final_eq, 2),
            "turnover_rate": turnover,
            "rebalance_events": turnover_events,
            "real_pct": round(real_count / max(total, 1) * 100, 1),
            "real_only": real_analysis,
            "monte_carlo": mc,
            "calmar_ratio": round(calmar, 3),
            "pct_time_in_market": round(pct_in_market * 100, 1),
            "reduced_pos_weeks": reduced_pos_count if vcfg["adaptive_k"] else 0,
        }

    # ── COMPARISON TABLE ──
    fprint(f"\n{'=' * 160}")
    fprint("COMPARISON — ALL VARIANTS")
    fprint(f"{'=' * 160}")
    fprint(f"  {'Variant':<30} {'N':>5} {'Sharpe':>7} {'Sort':>7} {'Calmar':>7} {'WR':>6} "
           f"{'PF':>6} {'MDD':>7} {'Gate':>5} {'Final$':>9} {'Turn':>6} {'%Mkt':>6} {'%Real':>6}")
    fprint(f"  {'-' * 130}")
    for vn in VARIANTS:
        r = all_results.get(vn)
        if not r:
            continue
        fprint(f"  {vn:<30} {r['n_trades']:>5} {r['sharpe']:>7.2f} {r['sortino']:>7.2f} "
               f"{r['calmar_ratio']:>7.2f} "
               f"{r['win_rate']*100:>5.1f}% {r['profit_factor']:>5.2f} "
               f"{r['max_dd']*100:>6.1f}% {r['gates_passed']}/{r['gates_total']} "
               f"${r['final_equity']:>8,.0f} {r['turnover_rate']:>5.0f} "
               f"{r['pct_time_in_market']:>5.1f}% {r['real_pct']:>5.1f}%")

    # Real-only comparison
    fprint(f"\n{'=' * 100}")
    fprint("REAL-PRICED TRADES ONLY")
    fprint(f"{'=' * 100}")
    fprint(f"  {'Variant':<30} {'N':>5} {'Sharpe':>8} {'WR':>6} {'PnL':>9}")
    fprint(f"  {'-' * 65}")
    for vn in VARIANTS:
        r = all_results.get(vn)
        if not r or not r.get("real_only"):
            continue
        ro = r["real_only"]
        fprint(f"  {vn:<30} {ro['n_trades']:>5} {ro['sharpe']:>8.2f} {ro['wr']*100:>5.1f}% "
               f"${ro['total_pnl']:>8.0f}")

    # Monte Carlo
    fprint(f"\n{'=' * 100}")
    fprint("MONTE CARLO BOOTSTRAP (1000 resamples)")
    fprint(f"{'=' * 100}")
    fprint(f"  {'Variant':<30} {'Mean':>8} {'95% CI':>20}")
    fprint(f"  {'-' * 65}")
    for vn in VARIANTS:
        r = all_results.get(vn)
        if not r or not r.get("monte_carlo"):
            continue
        mc = r["monte_carlo"]
        fprint(f"  {vn:<30} {mc['mean']:>8.2f} [{mc['ci_95_low']:.2f}, {mc['ci_95_high']:.2f}]")

    # ── CORRELATION ANALYSIS ──
    fprint(f"\n{'=' * 100}")
    fprint("CORRELATION GATE ANALYSIS (Variant B)")
    fprint(f"{'=' * 100}")
    if not corr_series.empty:
        fprint(f"  Avg pairwise correlation: mean={corr_series.mean():.3f}, "
               f"median={corr_series.median():.3f}")
        fprint(f"  % weeks with corr > {CORR_THRESHOLD}: {(corr_series > CORR_THRESHOLD).mean()*100:.1f}%")
        fprint(f"  Weeks skipped by gate: {n_corr_skipped} / {len(weekly_dates)} "
               f"({n_corr_skipped/max(len(weekly_dates),1)*100:.1f}%)")
        # Correlation by year
        yearly_corr = corr_series.resample("YE").mean()
        fprint(f"\n  Yearly avg correlation:")
        for yr_dt, val in yearly_corr.items():
            fprint(f"    {yr_dt.year}: {val:.3f}")

    # ── VIX ANALYSIS ──
    fprint(f"\n{'=' * 100}")
    fprint(f"VIX-ADAPTIVE POSITION ANALYSIS (Variant D, threshold={VIX_HIGH_THRESHOLD})")
    fprint(f"{'=' * 100}")
    r_d = all_results.get("D_vix_reduced_positions")
    if r_d:
        fprint(f"  Weeks with reduced positions (VIX > {VIX_HIGH_THRESHOLD}): {r_d['reduced_pos_weeks']}")
        d_trades = all_trades.get("D_vix_reduced_positions", [])
        if d_trades:
            normal = [t for t in d_trades if t["n_positions"] == 6]
            reduced = [t for t in d_trades if t["n_positions"] == 4]
            if normal:
                n_pnls = [t["pnl"] for t in normal]
                fprint(f"  Normal (6 pos): {len(normal)} trades, "
                       f"WR {sum(1 for p in n_pnls if p>0)/len(n_pnls)*100:.1f}%, "
                       f"avg PnL ${np.mean(n_pnls):.2f}")
            if reduced:
                r_pnls = [t["pnl"] for t in reduced]
                fprint(f"  Reduced (4 pos): {len(reduced)} trades, "
                       f"WR {sum(1 for p in r_pnls if p>0)/len(r_pnls)*100:.1f}%, "
                       f"avg PnL ${np.mean(r_pnls):.2f}")

    # ── 5-GATE VALIDATION SUMMARY ──
    fprint(f"\n{'=' * 100}")
    fprint("5-GATE VALIDATION SUMMARY")
    fprint(f"{'=' * 100}")
    fprint(f"  Gates: Sharpe>0.5, Regime gap<50%, MDD>-35%, PF>1.2, + sub-period/perm")
    fprint()
    for vn in VARIANTS:
        r = all_results.get(vn)
        if not r:
            continue
        status = "PASS" if r["all_passed"] else "FAIL"
        fprint(f"  [{status}] {vn}: {r['gates_passed']}/{r['gates_total']} gates")
        if r.get("gates"):
            for g in r["gates"]:
                gs = "PASS" if g["passed"] else "FAIL"
                fprint(f"         [{gs}] {g['name']}: {g['metric_name']}={g['metric_value']:.4f}")

    # ── VERDICT ──
    fprint(f"\n{'=' * 100}")
    fprint("VERDICT")
    fprint(f"{'=' * 100}")

    # Compare vs baseline (A)
    baseline = all_results.get("A_baseline_weekly")
    if baseline:
        fprint(f"\n  Baseline (A — weekly, 6 positions): Sharpe {baseline['sharpe']:.2f}, "
               f"PF {baseline['profit_factor']:.2f}, MDD {baseline['max_dd']*100:.1f}%, "
               f"Calmar {baseline['calmar_ratio']:.2f}")
        for vn in ["B_correlation_gate", "C_calendar_first_week", "D_vix_reduced_positions"]:
            r = all_results.get(vn)
            if not r:
                continue
            sh_diff = r["sharpe"] - baseline["sharpe"]
            pf_diff = r["profit_factor"] - baseline["profit_factor"]
            mdd_diff = (r["max_dd"] - baseline["max_dd"]) * 100
            calmar_diff = r["calmar_ratio"] - baseline["calmar_ratio"]
            n_diff = r["n_trades"] - baseline["n_trades"]
            fprint(f"\n  {vn}:")
            fprint(f"    Sharpe: {r['sharpe']:.2f} ({sh_diff:+.2f} vs baseline)")
            fprint(f"    PF:     {r['profit_factor']:.2f} ({pf_diff:+.2f})")
            fprint(f"    MDD:    {r['max_dd']*100:.1f}% ({mdd_diff:+.1f}pp)")
            fprint(f"    Calmar: {r['calmar_ratio']:.2f} ({calmar_diff:+.2f})")
            fprint(f"    Trades: {r['n_trades']} ({n_diff:+d} vs baseline)")
            fprint(f"    % in market: {r['pct_time_in_market']:.1f}%")
            if sh_diff > 0.1:
                fprint(f"    -> IMPROVEMENT: +{sh_diff:.2f} Sharpe")
            elif sh_diff < -0.1:
                fprint(f"    -> DEGRADATION: {sh_diff:.2f} Sharpe")
            else:
                fprint(f"    -> NEUTRAL: within 0.1 Sharpe of baseline")

    # Best overall
    best_vn = None
    best_sh = -999
    for vn in VARIANTS:
        r = all_results.get(vn)
        if not r:
            continue
        if r["sharpe"] > best_sh:
            best_sh = r["sharpe"]
            best_vn = vn

    if best_vn:
        r = all_results[best_vn]
        fprint(f"\n  BEST VARIANT: {best_vn}")
        fprint(f"    Sharpe {r['sharpe']:.2f}, Sortino {r['sortino']:.2f}, "
               f"Calmar {r['calmar_ratio']:.2f}, "
               f"PF {r['profit_factor']:.2f}, WR {r['win_rate']*100:.1f}%, "
               f"MDD {r['max_dd']*100:.1f}%")
        fprint(f"    {r['n_trades']} trades, {r['pct_time_in_market']:.1f}% time in market")
        fprint(f"    Gates: {r['gates_passed']}/{r['gates_total']}")
        if r["all_passed"]:
            fprint(f"    All gates PASSED — candidate for production upgrade")
        else:
            fprint(f"    Not all gates passed — needs investigation")

    # ── Save ──
    results_file = OUTPUT_DIR / "results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    fprint(f"\nResults saved to {results_file}")

    for vn, trades in all_trades.items():
        tf = OUTPUT_DIR / f"trades_{vn}.json"
        with open(tf, "w") as f:
            json.dump(trades, f, indent=1, default=str)

    # Save correlation series for future reference
    if not corr_series.empty:
        corr_df = corr_series.reset_index()
        corr_df.columns = ["date", "avg_pairwise_corr"]
        corr_file = OUTPUT_DIR / "sector_correlation_series.csv"
        corr_df.to_csv(corr_file, index=False)
        fprint(f"Correlation series saved to {corr_file}")

    # MLflow logging
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"corr_calendar_{t0.strftime('%Y%m%d_%H%M')}"):
                mlflow.log_param("capital", CAP)
                mlflow.log_param("dte", DTE)
                mlflow.log_param("top_k", TOP_K)
                mlflow.log_param("wf_train_weeks", WF_TRAIN_PERIODS)
                mlflow.log_param("n_sectors", len(SECTORS))
                mlflow.log_param("corr_threshold", CORR_THRESHOLD)
                mlflow.log_param("corr_lookback", CORR_LOOKBACK)
                mlflow.log_param("vix_high_threshold", VIX_HIGH_THRESHOLD)
                mlflow.log_param("vix_reduced_k", VIX_REDUCED_K)
                for vn, r in all_results.items():
                    mlflow.log_metric(f"{vn}_sharpe", r["sharpe"])
                    mlflow.log_metric(f"{vn}_sortino", r["sortino"])
                    mlflow.log_metric(f"{vn}_calmar", r["calmar_ratio"])
                    mlflow.log_metric(f"{vn}_wr", r["win_rate"])
                    mlflow.log_metric(f"{vn}_pf", r["profit_factor"])
                    mlflow.log_metric(f"{vn}_mdd", r["max_dd"])
                    mlflow.log_metric(f"{vn}_gates", r["gates_passed"])
                    mlflow.log_metric(f"{vn}_turnover", r["turnover_rate"])
                    mlflow.log_metric(f"{vn}_pct_in_market", r["pct_time_in_market"])
                    if r.get("real_only"):
                        mlflow.log_metric(f"{vn}_real_sharpe", r["real_only"].get("sharpe", 0))
                mlflow.log_artifact(str(results_file))
            fprint(f"MLflow logged to experiment '{EXPERIMENT_NAME}'")
        except Exception as e:
            fprint(f"MLflow error: {e}")

    elapsed = (datetime.now() - t0).total_seconds()
    fprint(f"\nCompleted in {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
