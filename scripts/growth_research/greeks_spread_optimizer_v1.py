#!/usr/bin/env python3
"""
Greeks-Based Spread Optimizer v1 — Delta-Targeted Strike Selection
===================================================================

Tests whether selecting option spreads by Greeks targeting (rather than
fixed % OTM) improves sector rotation strategy performance.

Problem: V8 uses fixed 2% OTM which creates pathologically narrow spreads
on low-priced ETFs (XLF $23-28 → only $0.46-$0.56 width). Real mid-pricing
drops Sharpe from 3.24 (BS) to 2.45 (0.76x).

5 Variants:
  A: Delta-targeted — long leg delta=0.30, short leg delta=0.15 (bull call).
     For bear put: long leg delta=-0.30, short leg delta=-0.15.
  B: Theta-optimized — maximize |theta|/premium per leg. Income-focused.
  C: IV-relative OTM — deeper OTM (3%) when sector IV>median, tighter (1.5%) when low.
  D: Min-width filter — 2% OTM but require width >= $3 AND cost < 50% of max value. Reject bad trades.
  E: Delta + min-width — Variant A with Variant D filters applied.

All use real mid-price from Dolt chain data, LGBM walk-forward ranking,
weekly rebalance, $645 capital, commission $2.60/spread.

Output: output/growth_research/greeks_spread_optimizer_v1/
MLflow experiment: greeks_spread_optimizer_v1
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


# ── Path detection (Jupiter vs Neptune/Razer) ──
POSSIBLE_ROOTS = [
    Path("/home/jupiter/Lvl3Quant"),
    Path("/home/nick/Lvl3Quant"),
    Path("C:/Users/claude/Lvl3Quant"),
]
BASE = None
for p in POSSIBLE_ROOTS:
    if p.exists():
        BASE = p
        break
if BASE is None:
    BASE = Path("/home/jupiter/Lvl3Quant")
    fprint(f"WARNING: No known root found, defaulting to {BASE}")

sys.path.insert(0, str(BASE))

try:
    from research.tools.options_pricer import (
        bs_call_price,
        bs_put_price,
        price_bull_call_spread,
        price_bear_put_spread,
        COMMISSION_RT_SPREAD,
    )
    HAVE_PRICER = True
except ImportError:
    HAVE_PRICER = False
    COMMISSION_RT_SPREAD = 2.60
    fprint("WARNING: options_pricer not available, using BS fallback disabled")

try:
    from research.tools.adversarial_validator import validate_trades
    HAVE_VALIDATOR = True
except ImportError:
    HAVE_VALIDATOR = False

# ── Config ──
CHAINS_DIR = BASE / "wheel_strategy_v1" / "data" / "cache" / "options_real" / "chains"
OUTPUT_DIR = BASE / "output" / "growth_research" / "greeks_spread_optimizer_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX", "^VIX3M", "TLT", "SHY", "HYG", "GLD"]
CAP = 645.0
TOP_K = 3
REGIME_BULL_THRESHOLD = 0.4

DTE = 14
OTM_PCT = 0.02
REBAL_FREQ = "W-FRI"
WF_TRAIN_PERIODS = 12  # ~60 weeks lookback for LGBM

# Chain matching
DTE_TOLERANCE = 5
STRIKE_TOLERANCE = 0.05  # slightly wider for delta-targeted picks
MIN_BID = 0.03

# Min-width filter (Variant D/E)
MIN_SPREAD_WIDTH = 3.0   # $3 absolute minimum
MAX_COST_WIDTH_RATIO = 0.50  # entry cost < 50% of max spread value

REGIME_FILE = BASE / "output" / "regime_detector_v1" / "regime_predictions_v1.npz"

# MLflow
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "greeks_spread_optimizer_v1"

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
# V6/V8 FEATURES (17 features)
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
# DATA LOADING
# ══════════════════════════════════════════════════════════════

def download_data():
    """Download price data via yfinance."""
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


def load_all_chains():
    """Load all sector ETF chain data from Dolt parquets."""
    chains = {}
    for tk in SECTORS:
        path = CHAINS_DIR / f"{tk}.parquet"
        if not path.exists():
            fprint(f"  {tk}: chain parquet not found at {path}")
            continue
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        df["expiration"] = pd.to_datetime(df["expiration"])
        for c in ["strike", "bid", "ask", "mid", "vol", "delta", "gamma", "theta", "vega"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["dte"] = (df["expiration"] - df["date"]).dt.days
        chains[tk] = df
        fprint(f"  {tk}: {len(df):>7,} rows, "
               f"{df['date'].min().date()} to {df['date'].max().date()}")
    return chains


def load_regime_predictions():
    """Load regime predictions or fall back to VIX proxy."""
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


# ══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING (same as V8 production)
# ══════════════════════════════════════════════════════════════

def compute_legacy_features(px, spy_slice):
    """Compute 17 V6/V8 features for a single sector."""
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
    n_folds = len(dates) - WF_TRAIN_PERIODS
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
        fold_num = i - WF_TRAIN_PERIODS + 1
        if fold_num % 10 == 0:
            fprint(f"    Fold {fold_num}/{n_folds} done")
    fprint(f"    {len(rankings)} ranking dates")
    return rankings


# ══════════════════════════════════════════════════════════════
# CHAIN-BASED STRIKE SELECTION (THE CORE OF THIS SCRIPT)
# ══════════════════════════════════════════════════════════════

def _get_chain_day(chain_df, trade_date, dte_target):
    """Get chain data for a specific date and expiration near dte_target."""
    if chain_df is None:
        return None

    day_mask = chain_df["date"] == pd.Timestamp(trade_date)
    chain_day = chain_df[day_mask]
    if chain_day.empty:
        # Try nearest trading day within 2 days
        nearby = chain_df[
            (chain_df["date"] >= pd.Timestamp(trade_date) - pd.Timedelta(days=2)) &
            (chain_df["date"] <= pd.Timestamp(trade_date) + pd.Timedelta(days=2))
        ]
        if nearby.empty:
            return None
        nearest_date = min(nearby["date"].unique(),
                          key=lambda x: abs((x - pd.Timestamp(trade_date)).days))
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

    chain_exp = chain_day[chain_day["expiration"] == best_exp].copy()
    chain_exp["_actual_dte"] = actual_dte
    return chain_exp


def _select_strikes_v8_baseline(chain_exp, S, direction):
    """V8 baseline: fixed 2% OTM, 3% spread width."""
    if direction == "bull":
        K1 = round(S * (1 + OTM_PCT), 2)
        K2 = round(K1 * 1.03, 2)
        opt_type = "c"
    else:
        K2 = round(S * (1 - OTM_PCT), 2)
        K1 = round(K2 * 0.97, 2)
        opt_type = "p"
    return _match_strikes_from_chain(chain_exp, K1, K2, opt_type, direction)


def _select_strikes_delta_targeted(chain_exp, S, direction):
    """
    Variant A: Delta-targeted entries.
    Bull call: long leg delta~0.30, short leg delta~0.15.
    Bear put: long leg delta~-0.30, short leg delta~-0.15.
    """
    if direction == "bull":
        opt_type = "c"
        calls = chain_exp[chain_exp["type"] == "c"].copy()
        if calls.empty:
            return None
        # Long leg: closest to delta=0.30
        calls["d_dist_long"] = (calls["delta"] - 0.30).abs()
        long_leg = calls.loc[calls["d_dist_long"].idxmin()]
        # Short leg: closest to delta=0.15 and strike > long leg strike
        further_otm = calls[calls["strike"] > long_leg["strike"]].copy()
        if further_otm.empty:
            # If no strikes further OTM, take closest to delta=0.15 overall
            calls["d_dist_short"] = (calls["delta"] - 0.15).abs()
            short_leg = calls.loc[calls["d_dist_short"].idxmin()]
            if short_leg["strike"] <= long_leg["strike"]:
                return None
        else:
            further_otm["d_dist_short"] = (further_otm["delta"] - 0.15).abs()
            short_leg = further_otm.loc[further_otm["d_dist_short"].idxmin()]
        K1, K2 = float(long_leg["strike"]), float(short_leg["strike"])
    else:
        opt_type = "p"
        puts = chain_exp[chain_exp["type"] == "p"].copy()
        if puts.empty:
            return None
        # Long leg: closest to delta=-0.30
        puts["d_dist_long"] = (puts["delta"] - (-0.30)).abs()
        long_leg = puts.loc[puts["d_dist_long"].idxmin()]
        # Short leg: closest to delta=-0.15 and strike < long leg strike
        further_otm = puts[puts["strike"] < long_leg["strike"]].copy()
        if further_otm.empty:
            puts["d_dist_short"] = (puts["delta"] - (-0.15)).abs()
            short_leg = puts.loc[puts["d_dist_short"].idxmin()]
            if short_leg["strike"] >= long_leg["strike"]:
                return None
        else:
            further_otm["d_dist_short"] = (further_otm["delta"] - (-0.15)).abs()
            short_leg = further_otm.loc[further_otm["d_dist_short"].idxmin()]
        # Bear put spread: buy put at higher strike, sell put at lower strike
        K1, K2 = float(short_leg["strike"]), float(long_leg["strike"])

    return _build_spread_result(chain_exp, K1, K2, opt_type, direction)


def _select_strikes_theta_optimized(chain_exp, S, direction):
    """
    Variant B: Theta-optimized — maximize |theta|/premium ratio.
    Select strikes that give best theta decay per dollar of premium paid.
    """
    if direction == "bull":
        opt_type = "c"
        opts = chain_exp[chain_exp["type"] == "c"].copy()
    else:
        opt_type = "p"
        opts = chain_exp[chain_exp["type"] == "p"].copy()

    if opts.empty or len(opts) < 2:
        return None

    # Filter to OTM options with reasonable prices
    if direction == "bull":
        otm = opts[opts["strike"] > S].copy()
    else:
        otm = opts[opts["strike"] < S].copy()

    if len(otm) < 2:
        return None

    # For each option, compute |theta| / mid ratio (theta per dollar)
    otm = otm[otm["mid"] > 0.01].copy()
    if len(otm) < 2:
        return None
    otm["theta_ratio"] = otm["theta"].abs() / (otm["mid"] + 1e-10)

    # Long leg: best theta_ratio (we want the decay to work FOR us on exit,
    # so pick the option with highest absolute theta for its price)
    # Actually for a debit spread we want: high theta on short leg (decay helps us)
    # and reasonable theta on long leg. So maximize short_leg theta_ratio.
    if direction == "bull":
        otm_sorted = otm.sort_values("strike")
    else:
        otm_sorted = otm.sort_values("strike", ascending=False)

    if len(otm_sorted) < 2:
        return None

    # Try pairs: long leg is closer to ATM, short leg is further OTM
    best_score = -np.inf
    best_pair = None

    # Limit search to reasonable range (top 10 by theta_ratio for short leg)
    candidates = otm_sorted.head(15)
    for i in range(len(candidates)):
        for j in range(i + 1, min(len(candidates), i + 6)):
            if direction == "bull":
                long_row = candidates.iloc[i]
                short_row = candidates.iloc[j]
                if short_row["strike"] <= long_row["strike"]:
                    continue
            else:
                long_row = candidates.iloc[i]
                short_row = candidates.iloc[j]
                if short_row["strike"] >= long_row["strike"]:
                    continue

            # Score: theta of short leg (positive = decays in our favor) / spread cost
            spread_cost = long_row["mid"] - short_row["mid"]
            if spread_cost <= 0.01:
                continue
            # Short leg theta decay benefit per dollar of spread cost
            score = abs(float(short_row["theta"])) / spread_cost
            if score > best_score:
                best_score = score
                if direction == "bull":
                    best_pair = (float(long_row["strike"]), float(short_row["strike"]))
                else:
                    best_pair = (float(short_row["strike"]), float(long_row["strike"]))

    if best_pair is None:
        return None

    K1, K2 = best_pair
    return _build_spread_result(chain_exp, K1, K2, opt_type, direction)


def _select_strikes_iv_relative(chain_exp, S, direction, sector_iv_median):
    """
    Variant C: IV-relative OTM.
    When sector IV > median: go deeper OTM (3%).
    When sector IV < median: go tighter (1.5%).
    """
    # Get current sector IV from the chain
    if direction == "bull":
        opts = chain_exp[chain_exp["type"] == "c"]
    else:
        opts = chain_exp[chain_exp["type"] == "p"]

    if opts.empty:
        return None

    # Current IV = median IV of near-ATM options
    near_atm = opts[(opts["strike"] - S).abs() / S < 0.05]
    if near_atm.empty:
        near_atm = opts
    current_iv = float(near_atm["vol"].median()) if "vol" in near_atm.columns else 0.25

    # Adapt OTM based on IV
    if sector_iv_median > 0 and current_iv > sector_iv_median:
        otm_pct = 0.03  # deeper OTM when high IV
    else:
        otm_pct = 0.015  # tighter when low IV

    if direction == "bull":
        K1 = round(S * (1 + otm_pct), 2)
        K2 = round(K1 * 1.03, 2)
        opt_type = "c"
    else:
        K2 = round(S * (1 - otm_pct), 2)
        K1 = round(K2 * 0.97, 2)
        opt_type = "p"

    return _match_strikes_from_chain(chain_exp, K1, K2, opt_type, direction)


def _select_strikes_min_width(chain_exp, S, direction):
    """
    Variant D: Same as V8 (2% OTM) but require:
    - Spread width >= $3 absolute
    - Entry cost < 50% of max spread value
    Reject trades that fail.
    """
    result = _select_strikes_v8_baseline(chain_exp, S, direction)
    if result is None:
        return None

    # Apply min-width filter
    spread_width = abs(result["K2"] - result["K1"])
    if spread_width < MIN_SPREAD_WIDTH:
        result["rejected"] = True
        result["reject_reason"] = f"width ${spread_width:.2f} < ${MIN_SPREAD_WIDTH:.2f}"
        return result

    # Cost/width filter
    max_spread_value = spread_width * 100  # per contract
    entry_cost = result["spread_cost_mid"] * 100
    if entry_cost > MAX_COST_WIDTH_RATIO * max_spread_value:
        ratio = entry_cost / max_spread_value
        result["rejected"] = True
        result["reject_reason"] = f"cost/width {ratio:.1%} > {MAX_COST_WIDTH_RATIO:.0%}"
        return result

    result["rejected"] = False
    return result


def _select_strikes_delta_min_width(chain_exp, S, direction):
    """
    Variant E: Delta-targeted (Variant A) + min-width filters (Variant D).
    """
    result = _select_strikes_delta_targeted(chain_exp, S, direction)
    if result is None:
        return None

    # Apply min-width filter
    spread_width = abs(result["K2"] - result["K1"])
    if spread_width < MIN_SPREAD_WIDTH:
        result["rejected"] = True
        result["reject_reason"] = f"width ${spread_width:.2f} < ${MIN_SPREAD_WIDTH:.2f}"
        return result

    # Cost/width filter
    max_spread_value = spread_width * 100
    entry_cost = result["spread_cost_mid"] * 100
    if entry_cost > MAX_COST_WIDTH_RATIO * max_spread_value:
        ratio = entry_cost / max_spread_value
        result["rejected"] = True
        result["reject_reason"] = f"cost/width {ratio:.1%} > {MAX_COST_WIDTH_RATIO:.0%}"
        return result

    result["rejected"] = False
    return result


def _match_strikes_from_chain(chain_exp, K1_target, K2_target, opt_type, direction):
    """Match target strikes to actual chain strikes, return spread result."""
    opts = chain_exp[chain_exp["type"] == opt_type].copy()
    if opts.empty:
        return None

    # Find nearest strikes
    opts["dist_k1"] = (opts["strike"] - K1_target).abs()
    opts["dist_k2"] = (opts["strike"] - K2_target).abs()

    k1_row = opts.loc[opts["dist_k1"].idxmin()]
    k2_row = opts.loc[opts["dist_k2"].idxmin()]

    K1 = float(k1_row["strike"])
    K2 = float(k2_row["strike"])

    if K2 <= K1:
        return None

    return _build_spread_result(chain_exp, K1, K2, opt_type, direction)


def _build_spread_result(chain_exp, K1, K2, opt_type, direction):
    """Build spread pricing result from chain data."""
    opts = chain_exp[chain_exp["type"] == opt_type]

    # Find exact strike matches
    near_matches = opts[opts["strike"] == K1]
    far_matches = opts[opts["strike"] == K2]

    if near_matches.empty or far_matches.empty:
        # Try closest match
        near_matches = opts.iloc[(opts["strike"] - K1).abs().argsort()[:1]]
        far_matches = opts.iloc[(opts["strike"] - K2).abs().argsort()[:1]]
        K1 = float(near_matches.iloc[0]["strike"])
        K2 = float(far_matches.iloc[0]["strike"])
        if K2 <= K1:
            return None

    near_leg = near_matches.iloc[0]
    far_leg = far_matches.iloc[0]

    near_bid = float(near_leg["bid"]) if not pd.isna(near_leg["bid"]) else 0
    near_ask = float(near_leg["ask"]) if not pd.isna(near_leg["ask"]) else 0
    near_mid = float(near_leg["mid"]) if not pd.isna(near_leg["mid"]) else (near_bid + near_ask) / 2
    far_bid = float(far_leg["bid"]) if not pd.isna(far_leg["bid"]) else 0
    far_ask = float(far_leg["ask"]) if not pd.isna(far_leg["ask"]) else 0
    far_mid = float(far_leg["mid"]) if not pd.isna(far_leg["mid"]) else (far_bid + far_ask) / 2

    near_delta = float(near_leg["delta"]) if not pd.isna(near_leg["delta"]) else np.nan
    far_delta = float(far_leg["delta"]) if not pd.isna(far_leg["delta"]) else np.nan
    near_theta = float(near_leg["theta"]) if "theta" in near_leg.index and not pd.isna(near_leg["theta"]) else np.nan
    far_theta = float(far_leg["theta"]) if "theta" in far_leg.index and not pd.isna(far_leg["theta"]) else np.nan
    near_iv = float(near_leg["vol"]) if "vol" in near_leg.index and not pd.isna(near_leg["vol"]) else np.nan
    far_iv = float(far_leg["vol"]) if "vol" in far_leg.index and not pd.isna(far_leg["vol"]) else np.nan

    actual_dte = int(chain_exp["_actual_dte"].iloc[0]) if "_actual_dte" in chain_exp.columns else DTE

    if direction == "bull":
        # Bull call: buy call at K1 (near), sell call at K2 (far)
        spread_cost_mid = near_mid - far_mid
        spread_cost_market = near_ask - far_bid
    else:
        # Bear put: buy put at K2 (near=higher strike), sell put at K1 (far=lower strike)
        # In our convention K1 < K2: buy K2 put, sell K1 put
        near_leg_buy = far_matches.iloc[0]  # K2, higher strike
        far_leg_sell = near_matches.iloc[0]  # K1, lower strike

        buy_mid = float(near_leg_buy["mid"]) if not pd.isna(near_leg_buy["mid"]) else 0
        sell_mid = float(far_leg_sell["mid"]) if not pd.isna(far_leg_sell["mid"]) else 0
        buy_ask = float(near_leg_buy["ask"]) if not pd.isna(near_leg_buy["ask"]) else 0
        sell_bid = float(far_leg_sell["bid"]) if not pd.isna(far_leg_sell["bid"]) else 0

        spread_cost_mid = buy_mid - sell_mid
        spread_cost_market = buy_ask - sell_bid

    if spread_cost_mid < 0:
        spread_cost_mid = abs(spread_cost_mid)
    if spread_cost_market < 0:
        spread_cost_market = abs(spread_cost_market)

    return {
        "found": True,
        "rejected": False,
        "spread_cost_mid": spread_cost_mid,
        "spread_cost_market": spread_cost_market,
        "K1": K1,
        "K2": K2,
        "near_delta": near_delta,
        "far_delta": far_delta,
        "near_theta": near_theta,
        "far_theta": far_theta,
        "near_iv": near_iv,
        "far_iv": far_iv,
        "actual_dte": actual_dte,
        "spread_width": K2 - K1,
    }


# ══════════════════════════════════════════════════════════════
# IV MEDIAN TRACKER (for Variant C)
# ══════════════════════════════════════════════════════════════

def compute_sector_iv_medians(chains):
    """Compute rolling median IV for each sector from chain history."""
    iv_medians = {}
    for tk, df in chains.items():
        # Get near-ATM IV per date
        grouped = df.groupby("date").apply(
            lambda g: g.loc[(g["strike"] - g["strike"].median()).abs().nsmallest(5).index, "vol"].median()
            if len(g) > 0 else np.nan,
            include_groups=False,
        )
        if len(grouped) > 0:
            # Use expanding median (all history up to each point)
            iv_medians[tk] = grouped.expanding().median()
    return iv_medians


# ══════════════════════════════════════════════════════════════
# TRADE EXECUTION
# ══════════════════════════════════════════════════════════════

VARIANT_NAMES = {
    "A_delta_targeted": "Delta-targeted (d=0.30/0.15)",
    "B_theta_optimized": "Theta-optimized (max |theta|/prem)",
    "C_iv_relative_otm": "IV-relative OTM (1.5-3%)",
    "D_min_width_filter": "Min-width filter ($3, <50%)",
    "E_delta_min_width": "Delta + min-width combined",
}


def execute_trade_variant(variant, tk, dt, direction, S, close, chains,
                          equity, max_pos, sector_iv_medians):
    """
    Execute a single spread trade for a given variant.
    Returns trade result dict or None.
    """
    chain_df = chains.get(tk)
    if chain_df is None:
        return None

    chain_exp = _get_chain_day(chain_df, dt, DTE)
    if chain_exp is None:
        return None

    # Select strikes based on variant
    if variant == "A_delta_targeted":
        result = _select_strikes_delta_targeted(chain_exp, S, direction)
    elif variant == "B_theta_optimized":
        result = _select_strikes_theta_optimized(chain_exp, S, direction)
    elif variant == "C_iv_relative_otm":
        iv_med = 0.25  # default
        if tk in sector_iv_medians:
            iv_series = sector_iv_medians[tk]
            if dt in iv_series.index:
                iv_med = float(iv_series.loc[dt])
            else:
                mask = iv_series.index <= pd.Timestamp(dt)
                if mask.any():
                    iv_med = float(iv_series.loc[mask].iloc[-1])
        result = _select_strikes_iv_relative(chain_exp, S, direction, iv_med)
    elif variant == "D_min_width_filter":
        result = _select_strikes_min_width(chain_exp, S, direction)
    elif variant == "E_delta_min_width":
        result = _select_strikes_delta_min_width(chain_exp, S, direction)
    else:
        return None

    if result is None:
        return None

    # Check if rejected by filter (Variant D/E)
    if result.get("rejected", False):
        return {"rejected": True, "reject_reason": result.get("reject_reason", "unknown")}

    entry_cost_ps = result["spread_cost_mid"]
    if entry_cost_ps <= 0:
        return None

    K1, K2 = result["K1"], result["K2"]
    total_cost = entry_cost_ps * 100 + COMMISSION_RT_SPREAD

    if total_cost <= 0 or total_cost > max_pos or total_cost > equity * 0.40:
        return None

    # Hold to expiry: intrinsic value
    di = close.index.get_loc(dt)
    actual_dte = result.get("actual_dte", DTE)
    ei = min(di + actual_dte, len(close) - 1)
    if ei <= di:
        return None
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
        "K1": K1,
        "K2": K2,
        "S_entry": round(S, 2),
        "S_exit": round(Se, 2),
        "intrinsic": round(intrinsic, 4),
        "spread_width": round(K2 - K1, 2),
        "cost_width_ratio": round(entry_cost_ps / (K2 - K1 + 1e-10), 4),
        "near_delta": result.get("near_delta", np.nan),
        "far_delta": result.get("far_delta", np.nan),
        "near_theta": result.get("near_theta", np.nan),
        "far_theta": result.get("far_theta", np.nan),
        "rejected": False,
    }


# ══════════════════════════════════════════════════════════════
# SIMULATION
# ══════════════════════════════════════════════════════════════

def simulate_variant(variant_name, rankings, close, chains, sector_iv_medians):
    """Simulate all trades for one variant."""
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None

    equity = CAP
    trades = []
    rejected_count = 0
    no_chain_count = 0

    for dt in sorted(rankings.keys()):
        if dt not in spy.index:
            continue

        cv = float(vix.loc[dt]) if vix is not None and dt in vix.index else 20.0
        scores = rankings[dt]
        if not scores:
            continue

        # VIX pairs logic (same as V8)
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

        # Execute trades
        for tk, dir_label in [(t, "bull") for t in bull_picks] + [(t, "bear") for t in bear_picks]:
            if tk not in close.columns:
                continue
            S = float(close[tk].loc[dt])
            result = execute_trade_variant(
                variant_name, tk, dt, dir_label, S, close, chains,
                equity, max_pos, sector_iv_medians
            )
            if result is None:
                no_chain_count += 1
                continue

            if result.get("rejected", False):
                rejected_count += 1
                continue

            equity += result["pnl"]
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
                "direction": dir_label,
                "vix": round(cv, 1),
                "win": result["pnl"] > 0,
                "trade_mode": trade_mode,
            })

    return trades, equity, rejected_count, no_chain_count


# ══════════════════════════════════════════════════════════════
# ANALYSIS & METRICS
# ══════════════════════════════════════════════════════════════

def compute_metrics(trades, initial_capital=CAP):
    """Compute key performance metrics from trade list."""
    if not trades:
        return {}

    pnls = np.array([t["pnl"] for t in trades])
    n = len(pnls)
    total_pnl = float(pnls.sum())
    final_equity = initial_capital + total_pnl
    total_return = total_pnl / initial_capital

    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    wr = len(wins) / n if n > 0 else 0

    # Sharpe (annualized, weekly trades)
    sharpe = float(pnls.mean() / (pnls.std() + 1e-10) * np.sqrt(52))

    # Sortino
    downside = pnls[pnls < 0]
    sortino = float(pnls.mean() / (downside.std() + 1e-10) * np.sqrt(52)) if len(downside) > 0 else sharpe

    # Profit factor
    gross_profit = float(wins.sum()) if len(wins) > 0 else 0
    gross_loss = float(abs(losses.sum())) if len(losses) > 0 else 1e-10
    pf = gross_profit / gross_loss

    # Max drawdown
    equity_curve = initial_capital + np.cumsum(pnls)
    peak = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - peak) / peak
    max_dd = float(drawdowns.min())

    # Calmar
    # Estimate CAGR from total return over years
    dates = sorted(set(t["entry_date"] for t in trades))
    if len(dates) >= 2:
        first_date = pd.Timestamp(dates[0])
        last_date = pd.Timestamp(dates[-1])
        years = (last_date - first_date).days / 365.25
        if years > 0:
            cagr = (final_equity / initial_capital) ** (1 / years) - 1
        else:
            cagr = 0
    else:
        cagr = 0
        years = 0
    calmar = cagr / (abs(max_dd) + 1e-10)

    # Average entry cost vs spread width
    entry_costs = [t["entry_cost_ps"] for t in trades if "entry_cost_ps" in t]
    widths = [t["spread_width"] for t in trades if "spread_width" in t]
    avg_cost = float(np.mean(entry_costs)) if entry_costs else 0
    avg_width = float(np.mean(widths)) if widths else 0
    avg_cost_width = float(np.mean([c / w for c, w in zip(entry_costs, widths) if w > 0])) if widths else 0

    return {
        "n_trades": n,
        "total_pnl": round(total_pnl, 2),
        "final_equity": round(final_equity, 2),
        "total_return": round(total_return * 100, 1),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "profit_factor": round(pf, 3),
        "win_rate": round(wr * 100, 1),
        "max_drawdown": round(max_dd * 100, 1),
        "calmar": round(calmar, 3),
        "cagr": round(cagr * 100, 1),
        "avg_entry_cost": round(avg_cost, 4),
        "avg_spread_width": round(avg_width, 2),
        "avg_cost_width_ratio": round(avg_cost_width * 100, 1),
        "years": round(years, 1),
    }


def per_sector_breakdown(trades):
    """Return per-sector performance breakdown."""
    by_sector = {}
    for t in trades:
        tk = t["ticker"]
        by_sector.setdefault(tk, []).append(t)

    rows = []
    for tk in sorted(by_sector.keys()):
        st = by_sector[tk]
        pnls = [t["pnl"] for t in st]
        wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
        widths = [t.get("spread_width", 0) for t in st]
        costs = [t.get("cost_width_ratio", 0) for t in st]
        rows.append({
            "sector": tk,
            "trades": len(st),
            "wr": round(wr, 1),
            "total_pnl": round(sum(pnls), 2),
            "avg_width": round(np.mean(widths), 2) if widths else 0,
            "avg_cost_ratio": round(np.mean(costs) * 100, 1) if costs else 0,
        })
    return rows


def print_summary(variant_name, metrics, rejected, no_chain, sector_rows):
    """Print detailed summary for one variant."""
    desc = VARIANT_NAMES.get(variant_name, variant_name)
    fprint(f"\n{'='*80}")
    fprint(f"  VARIANT: {variant_name} — {desc}")
    fprint(f"{'='*80}")

    if not metrics:
        fprint("  No trades executed.")
        return

    fprint(f"  Trades: {metrics['n_trades']} | Rejected: {rejected} | No-chain: {no_chain}")
    fprint(f"  Total PnL: ${metrics['total_pnl']:,.0f} | Final equity: ${metrics['final_equity']:,.0f} "
           f"| Return: {metrics['total_return']}%")
    fprint(f"  Sharpe: {metrics['sharpe']:.3f} | Sortino: {metrics['sortino']:.3f} | "
           f"PF: {metrics['profit_factor']:.2f} | WR: {metrics['win_rate']:.1f}%")
    fprint(f"  MaxDD: {metrics['max_drawdown']:.1f}% | Calmar: {metrics['calmar']:.2f} | "
           f"CAGR: {metrics['cagr']:.1f}%")
    fprint(f"  Avg entry cost: ${metrics['avg_entry_cost']:.4f}/sh | "
           f"Avg width: ${metrics['avg_spread_width']:.2f} | "
           f"Avg cost/width: {metrics['avg_cost_width_ratio']:.1f}%")

    if sector_rows:
        fprint(f"\n  {'Sector':<6} {'Trades':>7} {'WR':>7} {'PnL':>10} {'AvgW':>7} {'C/W%':>6}")
        fprint(f"  {'-'*45}")
        for r in sector_rows:
            fprint(f"  {r['sector']:<6} {r['trades']:>7} {r['wr']:>6.1f}% ${r['total_pnl']:>8.0f} "
                   f"${r['avg_width']:>5.2f} {r['avg_cost_ratio']:>5.1f}%")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    t0 = datetime.now()
    fprint("=" * 100)
    fprint(f"GREEKS-BASED SPREAD OPTIMIZER v1 — {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 100)
    fprint(f"Config: DTE={DTE} | Rebal={REBAL_FREQ} | Capital=${CAP:.0f} | "
           f"Commission=${COMMISSION_RT_SPREAD:.2f} | Hold to expiry")
    fprint(f"\n5 variants:")
    for k, v in VARIANT_NAMES.items():
        fprint(f"  {k}: {v}")
    fprint()

    # 1. Load chain data
    fprint("Loading Dolt chain data...")
    chains = load_all_chains()
    if not chains:
        fprint("FATAL: No chain data found. Check CHAINS_DIR path.")
        return
    fprint(f"  {len(chains)} sectors loaded\n")

    # 2. Download price data
    close, high, low = download_data()

    # 3. Regime predictions
    regime_series = load_regime_predictions()

    # 4. Compute ATR and IV medians
    fprint("\nComputing ATR series...")
    atr_dict = compute_atr_series(high, low, close)
    fprint(f"  {len(atr_dict)} sectors with ATR")

    fprint("Computing sector IV medians for Variant C...")
    sector_iv_medians = compute_sector_iv_medians(chains)
    fprint(f"  {len(sector_iv_medians)} sectors with IV history")

    # 5. Build rebalance dates (weekly Fridays, chain data period only)
    rebal_dates = close.resample(REBAL_FREQ).last().index
    # Filter to chain data period (2019+)
    chain_start = pd.Timestamp("2019-02-01")
    rebal_dates_all = rebal_dates[rebal_dates >= chain_start]
    fprint(f"\nRebalance dates: {len(rebal_dates_all)} (from {rebal_dates_all[0].date()})")

    # 6. Build features and LGBM rankings (shared across all variants)
    fprint("\n── LGBM Walk-Forward Ranking ──")
    # Use ALL rebal dates for feature building (need lookback before chain period)
    feat_df = build_feature_records(close, high, low, rebal_dates, regime_series)
    rankings = walk_forward_lgbm_rank(feat_df)

    # Filter rankings to chain data period only
    rankings_chain = {dt: r for dt, r in rankings.items() if dt >= chain_start}
    fprint(f"  Rankings in chain period: {len(rankings_chain)}")

    if not rankings_chain:
        fprint("FATAL: No rankings in chain data period.")
        return

    # 7. Run all 5 variants
    all_results = {}
    variants = list(VARIANT_NAMES.keys())

    for vi, variant in enumerate(variants, 1):
        fprint(f"\n── Simulating Variant {vi}/5: {variant} ──")
        t_v = time.time()

        trades, final_eq, rejected, no_chain = simulate_variant(
            variant, rankings_chain, close, chains, sector_iv_medians
        )

        metrics = compute_metrics(trades)
        sector_rows = per_sector_breakdown(trades)
        print_summary(variant, metrics, rejected, no_chain, sector_rows)

        all_results[variant] = {
            "metrics": metrics,
            "trades": trades,
            "rejected": rejected,
            "no_chain": no_chain,
            "sector_breakdown": sector_rows,
        }

        elapsed = time.time() - t_v
        fprint(f"  [{variant}] done in {elapsed:.1f}s")

    # 8. Comparison table
    fprint(f"\n\n{'='*100}")
    fprint("FINAL COMPARISON — ALL VARIANTS")
    fprint(f"{'='*100}")
    fprint(f"  {'Variant':<25} {'Trades':>7} {'Rej':>5} {'Sharpe':>8} {'Sortino':>8} "
           f"{'PF':>6} {'WR':>6} {'MaxDD':>7} {'Calmar':>7} {'PnL':>10} {'C/W%':>6}")
    fprint(f"  {'-'*100}")

    for v in variants:
        r = all_results[v]
        m = r["metrics"]
        if not m:
            fprint(f"  {v:<25} {'(no trades)':>7}")
            continue
        fprint(f"  {v:<25} {m['n_trades']:>7} {r['rejected']:>5} {m['sharpe']:>8.3f} "
               f"{m['sortino']:>8.3f} {m['profit_factor']:>6.2f} {m['win_rate']:>5.1f}% "
               f"{m['max_drawdown']:>6.1f}% {m['calmar']:>7.2f} ${m['total_pnl']:>9,.0f} "
               f"{m['avg_cost_width_ratio']:>5.1f}%")

    # 9. Best variant
    best_v = max(variants, key=lambda v: all_results[v]["metrics"].get("sharpe", -999))
    best_m = all_results[best_v]["metrics"]
    fprint(f"\n  BEST VARIANT: {best_v} (Sharpe {best_m['sharpe']:.3f})")

    baseline_sharpe = 2.45  # V8 real-mid known Sharpe
    improvement = (best_m['sharpe'] - baseline_sharpe) / baseline_sharpe * 100
    fprint(f"  vs V8 real-mid baseline (Sharpe 2.45): {improvement:+.1f}% improvement")

    # 10. Save results
    summary = {
        "run_timestamp": t0.isoformat(),
        "config": {
            "dte": DTE, "rebal": REBAL_FREQ, "capital": CAP,
            "commission": COMMISSION_RT_SPREAD, "min_spread_width": MIN_SPREAD_WIDTH,
            "max_cost_width_ratio": MAX_COST_WIDTH_RATIO,
        },
        "variants": {},
    }
    for v in variants:
        r = all_results[v]
        summary["variants"][v] = {
            "metrics": r["metrics"],
            "rejected": r["rejected"],
            "no_chain": r["no_chain"],
            "sector_breakdown": r["sector_breakdown"],
        }
    summary["best_variant"] = best_v
    summary["best_sharpe"] = best_m["sharpe"]

    summary_path = OUTPUT_DIR / "greeks_optimizer_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    fprint(f"\nSummary saved: {summary_path}")

    # Save all trades per variant
    for v in variants:
        trades_path = OUTPUT_DIR / f"trades_{v}.json"
        with open(trades_path, "w") as f:
            json.dump(all_results[v]["trades"], f, indent=2, default=str)

    # 11. MLflow logging
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"greeks_opt_v1_{t0.strftime('%Y%m%d_%H%M')}"):
                mlflow.log_param("dte", DTE)
                mlflow.log_param("rebal_freq", REBAL_FREQ)
                mlflow.log_param("capital", CAP)
                mlflow.log_param("commission", COMMISSION_RT_SPREAD)
                mlflow.log_param("n_variants", len(variants))
                mlflow.log_param("best_variant", best_v)

                for v in variants:
                    m = all_results[v]["metrics"]
                    if m:
                        mlflow.log_metric(f"{v}_sharpe", m["sharpe"])
                        mlflow.log_metric(f"{v}_sortino", m["sortino"])
                        mlflow.log_metric(f"{v}_pf", m["profit_factor"])
                        mlflow.log_metric(f"{v}_wr", m["win_rate"])
                        mlflow.log_metric(f"{v}_maxdd", m["max_drawdown"])
                        mlflow.log_metric(f"{v}_calmar", m["calmar"])
                        mlflow.log_metric(f"{v}_pnl", m["total_pnl"])
                        mlflow.log_metric(f"{v}_trades", m["n_trades"])
                        mlflow.log_metric(f"{v}_cost_width", m["avg_cost_width_ratio"])

                mlflow.log_artifact(str(summary_path))
            fprint("MLflow run logged successfully")
        except Exception as e:
            fprint(f"MLflow logging failed: {e}")

    elapsed_total = (datetime.now() - t0).total_seconds()
    fprint(f"\n{'='*100}")
    fprint(f"COMPLETE — Total time: {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")
    fprint(f"{'='*100}")


if __name__ == "__main__":
    main()
